"""CPU component for selected model parameters and buffers.

Holds selected parameters and buffers in CPU memory and
bulk-copies them to the activation device. This is the component used
by :class:`ModelOffloader` for both non-streamed names and whole-model
capture. It is a composable activate/deactivate lifecycle piece, not a
top-level model runtime.

Cross-cutting compatibility caveats (the narrow ``torch.compile`` scope,
DDP/FSDP wrap-before requirement, single-thread contract) live in the
:mod:`~piper_offload` package docstring.

Class-specific caveats
----------------------
- Binding mutates the wrapped ``model`` — frozen parameter
  registry entries (``module._parameters[leaf]``) are replaced with Parameters
  wrapping CPU storage, trainable parameter ``.data`` points at
  CPU storage while preserving the user's Parameter objects, and
  registered buffers are replaced with host copies.
- Registry replacement (rather than ``param.data`` swap) is required for
  correctness with quanto ``WeightQBytesTensor``: assigning
  ``param.data = new_quanto_tensor`` is a no-op for the inner ``_data``
  / ``_scale`` storages, so the model would silently keep referencing
  the original quanto wrapper.
- Buffer mutations during forward (RNN/SSM state, KV cache,
  training-mode BatchNorm running stats) are *discarded* on
  :meth:`release` or :meth:`deactivate`. Suitable for inference of stateless
  modules; not suitable for models that need persistent buffer state across
  calls.
- Trainable parameter updates on CUDA must run inside
  :meth:`optimizer_step`. Without that boundary, release or deactivation
  restores older CPU bytes and discards active GPU updates.
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
- Tied weights *are* deduplicated. Two parameter names whose values
  share underlying storage — whether the standard ``tie_weights()``
  pattern (one ``Parameter`` under multiple names) or the rarer case
  of distinct quanto wrappers around shared inner ``_data`` — share a
  single :class:`HostParam` and a single target storage on
  activation, preserving the tying invariant on GPU.
"""

import contextlib
import weakref
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass
from typing import Self

import torch
from torch import nn

from ._devices import canonical_device
from .host_module import (
    HostModuleInstance,
    HostModuleLoadPlan,
    HostModuleStore,
    ParameterOverride,
)
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

    @property
    def has_trainables(self) -> bool:
        """Whether any host parameter is trainable."""
        return self._module_store.has_trainables

    def bind(self, model: nn.Module) -> HostComponent:
        """Bind this store's host backing bytes to ``model``."""
        return HostComponent(self._module_store.bind(model))


class HostComponent:
    """CPU component with bulk device transfer.

    Instances are created by binding a :class:`HostComponentStore` to a
    compatible model. Every managed parameter is backed by CPU
    storage (handling quanto decomposition and tied-weight dedup).
    :meth:`activate` starts a device session and eagerly calls
    :meth:`acquire`. Acquisition allocates GPU tensors for each unique host
    parameter and installs that active storage into the managed model registry
    entries. Frozen parameters use registry replacement; trainable parameters
    preserve the user's Parameter objects and swap only ``.data`` so optimizer
    state remains valid. :meth:`release` restores CPU storage and frees
    the CUDA working set without ending the session; :meth:`deactivate` also
    ends the session.

    If trainable params are active on CUDA, run ``optimizer.step()``
    inside :meth:`optimizer_step` so updated GPU bytes are copied back
    into the CPU cache before the next deactivate/reactivate
    cycle.

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
        self._has_trainables = instance.has_trainables
        self._active_device: torch.device | None = None
        self._load_plan: HostModuleLoadPlan | None = None
        self._lease: CudaTargetLease | None = None
        self._use_hook: torch.utils.hooks.RemovableHandle | None = None
        self._optimizer_step_active: bool = False

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
        make the writes visible, then realigns any retained trainable
        ``.grad`` to the GPU so the next backward accumulates on-device
        (mirrors :meth:`deactivate` moving grad to CPU). Tied parameter
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

        Acquisition is idempotent. CPU sessions already use their host backing
        and therefore have no separate working set.
        """
        active_device = self._active_device
        if active_device is None:
            raise RuntimeError(
                "HostComponent.acquire() requires an active session; "
                "call activate() first."
            )
        if active_device.type == "cpu" or self._lease is not None:
            return

        plan = self._load_plan
        if plan is None:
            raise RuntimeError("HostComponent CUDA session has no load plan.")
        current_stream = torch.cuda.current_stream(active_device)
        lease = CudaTargetLease.allocate(plan, active_device)
        self._lease = lease
        try:
            lease.stage(
                plan,
                current_stream,
                non_blocking=True,
            )
            self._instance.install_target(lease.acquire(current_stream))
            torch.cuda.synchronize(active_device)
            # Realign trainable grads with their now-GPU data so the next
            # backward accumulates on-device. A no-op unless a prior CPU
            # optimizer step left a retained CPU grad (set_to_none=False).
            self._instance.move_trainable_grads_to(active_device)
            self._install_use_hook()
        except BaseException:
            self.release()
            raise

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

        Registry entries and trainable gradients return to CPU storage,
        and the CUDA target lease is closed. The activation session remains
        active so :meth:`acquire` can prepare another traversal.
        Target retirement completes recorded CUDA work, so release is safe to
        call immediately after a forward.

        Trainable ``.grad`` follows ``.data`` to CPU here (grads
        otherwise linger wherever ``AccumulateGrad`` left them, i.e. on the
        GPU, retaining device memory and stranding the gradient off-host).
        This gives a uniform released resting state — ``.data`` and
        ``.grad`` both on CPU — so a context-free CPU
        ``optimizer.step()`` works the same for host and streamed
        trainables."""
        lease = self._lease
        if lease is None:
            return
        self._remove_use_hook()
        try:
            self._instance.install_host()
            self._instance.move_trainable_grads_to(torch.device("cpu"))
        finally:
            try:
                lease.close()
            finally:
                self._lease = None

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

    @contextlib.contextmanager
    def optimizer_step(self) -> Generator[None]:
        """Optimizer-step boundary for managed trainable parameters.

        On CUDA activation, the model's trainable ``.data`` points at
        active GPU target storage. Wrap ``optimizer.step()`` in this
        context so updated trainable bytes are copied back into host
        CPU storage before the model is deactivated or reactivated.

        On CPU activation, or when inactive, this is a guarded no-op
        because trainable data is already resident in CPU storage.

        This context does not move ``param.grad``; grad placement is owned
        by the activate/deactivate cycle, which keeps grad on the same
        device as ``.data`` (GPU while active, CPU once deactivated).
        Use this context to step on the GPU; to step on the CPU instead,
        call ``optimizer.step()`` while deactivated.
        """
        if self._optimizer_step_active:
            raise RuntimeError(
                "HostComponent.optimizer_step() does not support reentrant entry."
            )

        self._optimizer_step_active = True
        try:
            active_device = self._active_device
            lease = self._lease
            if (
                self._has_trainables
                and active_device is not None
                and active_device.type == "cuda"
            ):
                if lease is None:
                    raise RuntimeError(
                        "HostComponent optimizer-step state is inconsistent: "
                        "CUDA active without an active target."
                    )
                try:
                    yield
                finally:
                    self._instance.copy_trainables_from_target(
                        lease.target,
                        non_blocking=True,
                    )
                    torch.cuda.synchronize(active_device)
            else:
                yield
        finally:
            self._optimizer_step_active = False

__all__ = [
    "HostComponent",
    "HostComponentStore",
]
