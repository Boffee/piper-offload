"""Internal CPU component for selected model parameters and buffers.

Holds selected parameters and buffers in CPU memory and bulk-copies them to the
activation device. :class:`~piper_offload.model_offloader.ModelOffloader`
composes :class:`HostComponent` instances internally for resident and
transient model state.
"""

import weakref
from collections.abc import Mapping

import torch
from torch import nn

from ._devices import canonical_device
from ._working_set import WorkingSetState
from .host_module import (
    HostModuleInstance,
    HostModuleLoadPlan,
    ParameterOverride,
)
from .pin_manager import TransferLease, host_pin_manager
from .target_lease import CudaTargetLease


class HostComponent:
    """Internal CPU component with bulk device transfer.

    Manages one bound :class:`HostModuleInstance` covering a disjoint set of
    managed parameter and buffer names. :meth:`activate` starts a device session
    and eagerly calls :meth:`acquire`. :meth:`release` restores CPU storage and
    frees the CUDA working set without ending the session; :meth:`deactivate`
    also ends the session.
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


__all__ = ["HostComponent"]
