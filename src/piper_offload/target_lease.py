"""Stream-aware ownership for reusable CUDA module targets."""

from collections.abc import Iterable
from typing import cast

import torch

from .pinned_module import PinnedModuleInstance, PinnedModuleTarget


class _CudaTargetLease:
    """Own one target plus the ordering needed to copy, use, and reuse it.

    The caller chooses the allocation-owner stream: short-lived staged
    components use the default allocator pool, while block lookahead targets
    stay with their copy stream. A ready event orders consumption after copies;
    recorded streams order later refills. These stream handoffs keep
    storage in its original allocator pool and work for opaque adapter state
    without inspecting its physical tensors. Closing the lease synchronizes all
    recorded work before dropping the target so its storage cannot be
    immediately reused while CUDA still accesses it.
    """

    def __init__(
        self,
        target: PinnedModuleTarget,
        allocation_stream: torch.cuda.Stream,
    ) -> None:
        self._target: PinnedModuleTarget | None = target
        self._allocation_stream = allocation_stream
        self._ready_event: torch.cuda.Event | None = None
        self._staged = False
        self._acquired = False
        self._recorded_streams: set[torch.cuda.Stream] = set()

    @classmethod
    def allocate(
        cls,
        instance: PinnedModuleInstance,
        device: torch.device,
        *,
        allocation_stream: torch.cuda.Stream | None = None,
        param_names: Iterable[str] | None = None,
        buffer_names: Iterable[str] | None = None,
    ) -> _CudaTargetLease:
        """Allocate a target from the requested CUDA stream's pool."""
        if allocation_stream is None:
            allocation_stream = torch.cuda.default_stream(device)
        with torch.cuda.stream(allocation_stream):
            target = instance.allocate_target(
                device,
                param_names=param_names,
                buffer_names=buffer_names,
            )
        return cls(target, allocation_stream)

    @property
    def target(self) -> PinnedModuleTarget:
        target = self._target
        if target is None:
            raise RuntimeError("CUDA target lease is closed")
        return target

    def stage(
        self,
        instance: PinnedModuleInstance,
        stream: torch.cuda.Stream,
        *,
        run_post_copy_hooks: bool = False,
        non_blocking: bool = True,
    ) -> None:
        """Refill this target on ``stream`` without changing a module."""
        if self._acquired:
            raise RuntimeError("cannot stage a CUDA target while it is in use")
        target = self.target
        prior = self._ready_event
        self._staged = False
        with torch.cuda.stream(stream):
            if prior is not None:
                stream.wait_event(prior)
            for recorded in self._recorded_streams:
                if recorded != stream:
                    stream.wait_stream(recorded)
            self._recorded_streams.clear()
            try:
                instance.copy_to_target(
                    target,
                    run_post_copy_hooks=run_post_copy_hooks,
                    non_blocking=non_blocking,
                )
                self._staged = True
            finally:
                # A hook or later tensor copy can fail after earlier async
                # copies were enqueued. Preserve a completion marker so
                # cleanup still hands the allocation back safely.
                self._ready_event = cast(
                    torch.cuda.Event,
                    stream.record_event(),
                )

    def acquire(self, stream: torch.cuda.Stream) -> PinnedModuleTarget:
        """Order ``stream`` after staging and mark it as the consumer."""
        if self._acquired:
            raise RuntimeError("CUDA target is already in use")
        event = self._ready_event
        if not self._staged or event is None:
            raise RuntimeError("CUDA target must be staged before use")
        stream.wait_event(event)
        self._staged = False
        self._acquired = True
        self.record_stream(stream)
        return self.target

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Record a stream that may read or write the acquired target."""
        if not self._acquired:
            raise RuntimeError("CUDA target must be acquired before use")
        self._recorded_streams.add(stream)

    def release(self) -> None:
        """End the acquired state while retaining its stream dependencies."""
        self._acquired = False

    def close(self) -> None:
        """Release target storage after all recorded CUDA work completes."""
        if self._target is None:
            return
        self.release()
        try:
            if self._ready_event is not None:
                self._ready_event.synchronize()
            streams = set(self._recorded_streams)
            streams.add(self._allocation_stream)
            for stream in streams:
                stream.synchronize()
        finally:
            self._target = None
            self._ready_event = None
            self._staged = False
            self._recorded_streams.clear()
