"""Stream-aware ownership for reusable CUDA module targets."""

from typing import cast

import torch

from .pinned_module import PinnedModuleInstance, PinnedModuleTarget


class _CudaTargetLease:
    """Own one target plus the ordering needed to copy, use, and reuse it.

    The caller chooses the allocation-owner stream: short-lived staged
    components use the default allocator pool, while block lookahead targets
    stay with their copy stream. A ready event orders consumption after copies;
    recorded consumer streams order later refills, and the allocation stream
    waits for all remaining target work before storage is dropped. These stream
    handoffs keep storage in its original allocator pool and work for opaque
    adapter state without inspecting its physical tensors.
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
        self._consumer_streams: set[torch.cuda.Stream] = set()
        self._lifetime_streams: set[torch.cuda.Stream] = set()

    @classmethod
    def allocate(
        cls,
        instance: PinnedModuleInstance,
        device: torch.device,
        *,
        allocation_stream: torch.cuda.Stream | None = None,
    ) -> _CudaTargetLease:
        """Allocate a target from the requested CUDA stream's pool."""
        if allocation_stream is None:
            allocation_stream = torch.cuda.default_stream(device)
        with torch.cuda.stream(allocation_stream):
            target = instance.allocate_target(device)
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
            for consumer in self._consumer_streams:
                if consumer != stream:
                    stream.wait_stream(consumer)
            self._consumer_streams.clear()
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
        self.mark_used(stream)
        return self.target

    def mark_used(self, stream: torch.cuda.Stream) -> None:
        """Record a stream that may read or write the acquired target."""
        if not self._acquired:
            raise RuntimeError("CUDA target must be acquired before use")
        self._consumer_streams.add(stream)

    def track_lifetime_stream(self, stream: torch.cuda.Stream) -> None:
        """Protect externally ordered target work until the lease closes."""
        if stream != self._allocation_stream:
            self._lifetime_streams.add(stream)

    def release(self) -> None:
        """End the acquired state while retaining its stream dependencies."""
        self._acquired = False

    def close(self) -> None:
        """Release target storage without a device- or host-wide sync."""
        if self._target is None:
            return
        self.release()
        if self._ready_event is not None:
            self._allocation_stream.wait_event(self._ready_event)
        for tracked in self._lifetime_streams | self._consumer_streams:
            if tracked != self._allocation_stream:
                self._allocation_stream.wait_stream(tracked)
        self._target = None
        self._ready_event = None
        self._staged = False
        self._consumer_streams.clear()
        self._lifetime_streams.clear()
