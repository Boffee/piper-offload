"""CUDA target storage lifetime and stream synchronization."""

from collections.abc import Iterable

import torch

from .host_module import HostModuleLoadPlan, HostModuleTarget
from .stream_lease import StreamLease


class CudaTargetLease:
    """Stream-ordered GPU module target lease composed over :class:`StreamLease`.

    Owns an allocated :class:`HostModuleTarget` on CUDA until :meth:`close`
    completes synchronizing all pending stage, consumer, and allocation work.
    """

    def __init__(
        self,
        target: HostModuleTarget,
        allocation_stream: torch.cuda.Stream,
    ) -> None:
        self._stream_lease = StreamLease(allocation_stream)
        self._target: HostModuleTarget | None = target

    @classmethod
    def allocate(
        cls,
        plan: HostModuleLoadPlan,
        device: torch.device,
        *,
        allocation_stream: torch.cuda.Stream | None = None,
        buffer_names: Iterable[str] | None = None,
    ) -> CudaTargetLease:
        """Allocate a reusable CUDA target from the requested stream's pool."""
        stream = (
            torch.cuda.default_stream(device)
            if allocation_stream is None
            else allocation_stream
        )
        with torch.cuda.stream(stream):
            target = plan.allocate_target(device, buffer_names=buffer_names)
        return cls(target, stream)

    @property
    def stream_lease(self) -> StreamLease:
        """Return the underlying :class:`StreamLease` ordering CUDA streams."""
        return self._stream_lease

    @property
    def allocation_stream(self) -> torch.cuda.Stream:
        """Return the stream on which the backing target was allocated."""
        return self._stream_lease.allocation_stream

    @property
    def is_staged(self) -> bool:
        """Return whether the target holds a valid staged copy."""
        return self._stream_lease.is_staged

    @property
    def is_acquired(self) -> bool:
        """Return whether a consumer currently holds the target."""
        return self._stream_lease.is_acquired

    @property
    def target(self) -> HostModuleTarget:
        """Return the underlying CUDA target or raise once closed."""
        if self._target is None:
            raise RuntimeError("CudaTargetLease is closed")
        return self._target

    def stage(
        self,
        plan: HostModuleLoadPlan,
        stream: torch.cuda.Stream,
        *,
        non_blocking: bool = True,
    ) -> None:
        """Stage ``plan`` onto the leased target on ``stream``."""
        target = self.target
        self._stream_lease.stage_with(
            lambda: plan.copy_to_target(target, non_blocking=non_blocking),
            stream,
        )

    def acquire(self, stream: torch.cuda.Stream) -> HostModuleTarget:
        """Order ``stream`` after the staged copy and return the target."""
        target = self.target
        self._stream_lease.acquire(stream, exclusive=True)
        try:
            self.record_stream(stream)
        except BaseException:
            self._stream_lease.release(stream)
            raise
        return target

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Record an additional consumer stream during an active acquisition."""
        self._stream_lease.record_stream(stream)

    def release(self) -> None:
        """End the acquired state and snapshot stream completion for later reuse."""
        self._stream_lease.release()

    def close(self) -> None:
        """Synchronize all target streams and release the CUDA target reference."""
        if self._target is None:
            return
        self._stream_lease.synchronize_and_reset()
        self._target = None


__all__ = ["CudaTargetLease"]
