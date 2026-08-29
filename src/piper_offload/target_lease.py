"""Stream-aware ownership for reusable CUDA module targets."""

import torch

from .pinned_module import PinnedModuleInstance, PinnedModuleTarget


class _CudaTargetLease:
    """Own one target plus the ordering needed to copy, use, and reuse it.

    The caller chooses the allocation-owner stream: short-lived staged
    components use the default allocator pool, while block lookahead targets
    stay with their copy stream. Copies and consumers may run on other streams;
    a completion event orders target reuse, and the allocation stream waits
    for the target's last operation before it is dropped. This event handoff
    keeps storage in its original allocator pool and works for opaque adapter
    state without inspecting its physical tensors.
    """

    def __init__(
        self,
        target: PinnedModuleTarget,
        allocation_stream: torch.cuda.Stream,
    ) -> None:
        self._target: PinnedModuleTarget | None = target
        self._allocation_stream = allocation_stream
        self._event: torch.cuda.Event | None = None
        self._staged = False
        self._consumer_stream: torch.cuda.Stream | None = None
        self._tracked_streams: set[torch.cuda.Stream] = set()

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

    @property
    def device(self) -> torch.device:
        return self._allocation_stream.device

    def stage(
        self,
        instance: PinnedModuleInstance,
        stream: torch.cuda.Stream,
        *,
        run_post_copy_hooks: bool = False,
        non_blocking: bool = True,
    ) -> None:
        """Refill this target on ``stream`` without changing a module."""
        if self._consumer_stream is not None:
            raise RuntimeError("cannot stage a CUDA target while it is in use")
        target = self.target
        prior = self._event
        self._staged = False
        with torch.cuda.stream(stream):
            if prior is not None:
                stream.wait_event(prior)
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
                self._event = stream.record_event()

    def acquire(self, stream: torch.cuda.Stream) -> PinnedModuleTarget:
        """Order ``stream`` after staging and mark it as the consumer."""
        if self._consumer_stream is not None:
            raise RuntimeError("CUDA target is already in use")
        event = self._event
        if not self._staged or event is None:
            raise RuntimeError("CUDA target must be staged before use")
        stream.wait_event(event)
        self._staged = False
        self._consumer_stream = stream
        return self.target

    def track_stream(self, stream: torch.cuda.Stream) -> None:
        """Protect target work outside the stage/acquire/release cycle."""
        if stream != self._allocation_stream:
            self._tracked_streams.add(stream)

    def release(self, stream: torch.cuda.Stream | None = None) -> None:
        """Record completion of the current consumer, if any."""
        if self._consumer_stream is None:
            return
        consumer = stream or self._consumer_stream
        self._event = consumer.record_event()
        self._consumer_stream = None

    def close(self, stream: torch.cuda.Stream | None = None) -> None:
        """Release target storage without a device- or host-wide sync."""
        if self._target is None:
            return
        self.release(stream)
        if self._event is not None:
            self._allocation_stream.wait_event(self._event)
        for tracked in self._tracked_streams:
            self._allocation_stream.wait_stream(tracked)
        self._target = None
        self._event = None
        self._staged = False
        self._tracked_streams.clear()
