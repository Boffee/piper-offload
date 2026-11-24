"""CPU component for selected model parameters and buffers.

Holds selected parameters and buffers in CPU memory and
bulk-copies them to the activation device. This is the component used
by :class:`ModelOffloader` for both non-streamed names and whole-model
capture. It is a composable activate/deactivate lifecycle piece, not a
top-level model runtime.

Application compatibility is covered by the
`models <https://github.com/Boffee/piper-offload/blob/main/docs/models.md>`_ and
`streaming <https://github.com/Boffee/piper-offload/blob/main/docs/streaming.md>`_
guides. :class:`ModelOffloader` owns the coordinated model lifecycle and
exclusive activation contract.

Class-specific caveats
----------------------
- Binding mutates the wrapped ``model`` — parameter
  registry entries (``module._parameters[leaf]``) are replaced with frozen
  Parameters wrapping CPU storage, and registered buffers are replaced with host
  copies.
- Buffer mutations during forward (RNN/SSM state, KV cache,
  stateful BatchNorm running stats) are *discarded* on
  :meth:`release` or :meth:`deactivate`. Suitable for inference of stateless
  modules; not suitable for models that need persistent buffer state across
  calls.
- **Caller owns lifecycle correctness.** Calling :meth:`activate`
  twice without an intervening :meth:`deactivate` raises before registry
  movement or GPU allocation. Host construction optimizes peak host memory
  by letting :class:`HostParam` repoint plain ``Parameter.data`` at captured
  host backing as each host parameter is created; if construction or activation
  raises after that point, retrying the same model/component is unsupported —
  drop references and rebuild from a fresh model instance.
- There is no ``close()``. Host memory is freed when the caller
  drops the component AND model references; Python's refcount-based
  GC reclaims the host tensors immediately. The component releases
  what it owns (its internal name tracking); the user's model is the
  user's concern.
- Tied weights share one :class:`HostParam` and one active target, including
  distinct parameter objects that the tensor adapter identifies as aliases.
"""

import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Self

import torch
from torch import nn

from ._devices import canonical_device
from ._working_set import WorkingSetState
from .host_module import (
    HostModuleInstance,
    HostModuleLoadPlan,
    HostModuleStore,
    ParameterOverride,
)
from .pin_manager import TransferLease, host_pin_manager
from .target_lease import CudaTargetLease


@dataclass(frozen=True, slots=True)
class HostComponentStore:
    """Reusable host backing storage for :class:`HostComponent`.

    Component-level wrapper over the internal name-keyed module store.
    Build once from a prototype module, then bind it to concrete
    compatible modules with :meth:`bind`.
    """

    _module_store: HostModuleStore

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        include_param_names: Iterable[str] | None = None,
        include_buffer_names: Iterable[str] | None = None,
    ) -> Self:
        """Create a reusable store that owns captured pageable CPU backing."""
        return cls(
            HostModuleStore.from_module(
                model,
                include_param_names=include_param_names,
                include_buffer_names=include_buffer_names,
            )
        )

    @property
    def param_names(self) -> frozenset[str]:
        """Host parameter names in this store."""
        return frozenset(self._module_store.params)

    @property
    def buffer_names(self) -> frozenset[str]:
        """Host buffer names in this store."""
        return frozenset(self._module_store.buffers)

    @property
    def cache_bytes(self) -> int:
        """Total host-backing bytes held by this store."""
        return self._module_store.cache_bytes

    def _validate_bind(self, model: nn.Module) -> None:
        self._module_store._validate_bind(model)

    def _bind_validated(self, model: nn.Module) -> HostComponent:
        return HostComponent(self._module_store._install(model))

    def bind(self, model: nn.Module) -> HostComponent:
        """Bind this store's host backing bytes to ``model``."""
        self._validate_bind(model)
        return self._bind_validated(model)


class HostComponent:
    """CPU component with bulk device transfer.

    Instances are created by binding a :class:`HostComponentStore` to a
    compatible model. Tensor adapters capture the CPU storage for each
    managed parameter, sharing it across tied weights.
    :meth:`activate` starts a device session and eagerly calls
    :meth:`acquire`. Acquisition allocates GPU tensors for each unique host
    parameter and installs that active storage into the managed model registry
    entries. :meth:`release` restores CPU storage and frees
    the CUDA working set without ending the session; :meth:`deactivate` also
    ends the session.

    Buffer-only modules (only registered buffers, no params)
    are valid — common for sibling tables like RoPE/positional
    embeddings managed via :class:`ModelOffloader`'s non-block
    composition. Empty selections are valid no-op components; the
    top-level :class:`ModelOffloader` still rejects configurations
    with no components to manage.

    Stores are constructed with :meth:`HostComponentStore.from_module`.
    Managed tensors may start on CPU or CUDA; store construction clones
    them directly into CPU storage.
    """

    def __init__(self, instance: HostModuleInstance) -> None:
        self._instance = instance
        self._param_names = frozenset(instance.params)
        self._buffer_names = frozenset(instance.buffers)
        self._active_device: torch.device | None = None
        self._load_plan: HostModuleLoadPlan | None = None
        self._lease: CudaTargetLease | None = None
        self._transfer = TransferLease()
        self._use_hook: torch.utils.hooks.RemovableHandle | None = None
        self._working_set_state = WorkingSetState.RELEASED

    # ------------------------------------------------------------------
    # Component API
    # ------------------------------------------------------------------

    @property
    def param_names(self) -> frozenset[str]:
        """Host parameter names managed by this instance."""
        return self._param_names

    @property
    def buffer_names(self) -> frozenset[str]:
        """Host buffer names managed by this instance."""
        return self._buffer_names

    @property
    def failed(self) -> bool:
        """Whether a prior session left its working set unreleased.

        Deactivation ends the session even when releasing fails, keeping
        whatever it could not release. A later :meth:`release` or
        :meth:`deactivate` retries.
        """
        return self._active_device is None and self._working_set_state is not WorkingSetState.RELEASED

    def activate(
        self,
        device: torch.device,
        *,
        parameter_overrides: Mapping[str, ParameterOverride] | None = None,
        **kwargs: object,
    ) -> None:
        """Activate the managed tensors on ``device``.

        CUDA activation bulk-DMAs host weights to GPU: per-tensor
        ``.to()`` (non-blocking), then a single ``cuda.synchronize`` to
        make the writes visible. Tied parameter
        names all receive the same GPU Parameter. CPU activation repoints
        registry entries back to the CPU Parameters and performs no
        device copy.

        Calling activate() twice without an intervening deactivate()
        raises before any registry movement or GPU allocation.

        **Activation failure semantics:** failed CUDA acquisition releases its
        partial working set and restores host storage. The caller's cleanup
        path remains :meth:`deactivate` followed by dropping the component.
        """
        del kwargs  # streaming-only policy; bulk-host activation ignores it
        if self._active_device is not None:
            raise RuntimeError(
                "HostComponent.activate() called while already active "
                f"on {self._active_device}. Deactivate first, or check "
                "for a leaked context manager."
            )
        if self.failed:
            # A new session would silently reuse the old target and transfer.
            raise RuntimeError(
                "HostComponent cannot activate after its prior CUDA session "
                "failed to finish releasing; deactivate() retries that "
                "release. Recreate the CUDA worker only if the retry fails too."
            )
        active_device = canonical_device(device)
        if active_device.type == "cpu":
            if parameter_overrides:
                raise ValueError(
                    "Parameter overrides require CUDA activation."
                )
            self._instance.install_host()
            self._active_device = active_device
            return
        if active_device.type == "cuda":
            self._load_plan = self._instance.resolve_load_plan(
                parameter_overrides,
            )
            self._active_device = active_device
            self.acquire()
            return
        raise ValueError(
            "HostComponent.activate() supports CUDA or CPU; "
            f"got {active_device}."
        )

    def acquire(self) -> None:
        """Acquire this active session's CUDA working set.

        Acquisition is idempotent, also after an acquire or release that an
        exception stopped partway. CPU sessions already use their host
        backing and therefore have no separate working set.
        """
        active_device = self._active_device
        if active_device is None:
            raise RuntimeError(
                "HostComponent.acquire() requires an active session; "
                "call activate() first."
            )
        if active_device.type == "cpu" or self._working_set_state is WorkingSetState.ACQUIRED:
            return
        if self._working_set_state is WorkingSetState.CHANGING:
            self.release()

        plan = self._load_plan
        if plan is None:
            raise RuntimeError("HostComponent CUDA session has no load plan.")
        current_stream = torch.cuda.current_stream(active_device)
        self._working_set_state = WorkingSetState.CHANGING
        # A one-time upload: lease the sources pageable, for the upload only.
        self._transfer.start(host_pin_manager, plan.storage_tensors(), active_device)
        try:
            lease = CudaTargetLease.allocate(plan, active_device)
            self._lease = lease
            lease.stage(
                plan,
                current_stream,
                non_blocking=True,
            )
            self._instance.install_target(lease.acquire(current_stream))
            self._transfer.finish()
            self._install_use_hook()
        except BaseException:
            self.release()
            raise
        self._working_set_state = WorkingSetState.ACQUIRED

    def _install_use_hook(self) -> None:
        """Track the CUDA stream that actually executes the bound module."""
        if self._use_hook is not None:
            raise RuntimeError("HostComponent CUDA use hook is already installed.")
        component_ref = weakref.ref(self)

        def record_stream(_module: nn.Module, _args: tuple[object, ...]) -> None:
            component = component_ref()
            if component is None:
                return
            device = component._active_device
            if device is not None and device.type == "cuda":
                component.record_stream(torch.cuda.current_stream(device))

        self._use_hook = self._instance.module.register_forward_pre_hook(
            record_stream,
            prepend=True,
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Record a CUDA stream that may still be using the active target."""
        lease = self._lease
        if lease is not None:
            lease.record_stream(stream)

    def _remove_use_hook(self) -> None:
        hook = self._use_hook
        self._use_hook = None
        if hook is not None:
            hook.remove()

    def release(self) -> None:
        """Idempotently release this session's CUDA working set.

        Registry entries return to CPU storage, and the CUDA target lease is
        closed. The activation session remains active so :meth:`acquire` can
        prepare another traversal. Target retirement completes recorded CUDA
        work, so release is safe to call immediately after a forward.

        Each step must finish before the next releases anything: a failed
        synchronization keeps the transfer lease, a failed host restore or
        target close keeps the target, and a later call retries.
        """
        if self._working_set_state is WorkingSetState.RELEASED:
            return
        self._working_set_state = WorkingSetState.CHANGING
        # A partial upload may still be in flight; its lease
        # closes only once the device has synchronized.
        self._transfer.finish()
        lease = self._lease
        if lease is not None:
            self._remove_use_hook()
            self._instance.install_host()
            lease.close()
            self._lease = None
        self._working_set_state = WorkingSetState.RELEASED

    def deactivate(self) -> None:
        """Release working storage and end the activation session.

        Idempotent and safe to call before :meth:`activate` or multiple times.
        Drop the component and model references afterward to release host
        memory.
        """
        try:
            self.release()
        finally:
            self._active_device = None
            self._load_plan = None


__all__ = [
    "HostComponent",
    "HostComponentStore",
]
