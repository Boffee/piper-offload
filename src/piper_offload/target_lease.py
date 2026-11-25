"""Stream-aware ownership for reusable CUDA module targets."""

from collections.abc import Iterable
from typing import cast

import torch

from .host_module import HostModuleLoadPlan, HostModuleTarget


class CudaTargetLease:
    """Own one target plus the ordering needed to copy, use, and reuse it.

    The caller chooses the allocation-owner stream: short-lived staged
    components use the default allocator pool, while block lookahead targets
    stay with their copy stream. A ready event orders consumption after copies;
    release events recorded synchronously at :meth:`release` order later
    refills after prior consumption without waiting for subsequent work on
    those streams. These stream handoffs keep storage in its original allocator
    pool and work for opaque adapter state without inspecting its physical
    tensors. Closing the lease synchronizes all recorded work before dropping
    the target so its storage cannot be immediately reused while CUDA still
    accesses it.
    """

    def __init__(
        self,
        target: HostModuleTarget,
        allocation_stream: torch.cuda.Stream,
    ) -> None:
        self._target: HostModuleTarget | None = target
        self._allocation_stream = allocation_stream
        self._ready_event: torch.cuda.Event | None = None
        self._staged = False
        self._acquired = False
        self._recorded_streams: set[torch.cuda.Stream] = set()
        self._release_events: dict[torch.cuda.Stream, torch.cuda.Event] = {}
        self._waited_streams: set[torch.cuda.Stream] = set()

    @property
    def is_staged(self) -> bool:
        """Whether this lease currently holds completed staged weights."""
        return self._staged

    @property
    def is_acquired(self) -> bool:
        """Whether this lease is currently acquired by a consumer."""
        return self._acquired

    @classmethod
    def allocate(
        cls,
        plan: HostModuleLoadPlan,
        device: torch.device,
        *,
        allocation_stream: torch.cuda.Stream | None = None,
        buffer_names: Iterable[str] | None = None,
    ) -> CudaTargetLease:
        """Allocate a target from the requested CUDA stream's pool."""
        if allocation_stream is None:
            allocation_stream = torch.cuda.default_stream(device)
        with torch.cuda.stream(allocation_stream):
            target = plan.allocate_target(
                device,
                buffer_names=buffer_names,
            )
        return cls(target, allocation_stream)

    @property
    def target(self) -> HostModuleTarget:
        target = self._target
        if target is None:
            raise RuntimeError("CUDA target lease is closed")
        return target

    def stage(
        self,
        plan: HostModuleLoadPlan,
        stream: torch.cuda.Stream,
        *,
        non_blocking: bool = True,
    ) -> None:
        """Refill this target on ``stream`` without changing a module."""
        if self._acquired:
            raise RuntimeError("cannot stage a CUDA target while it is in use")
        target = self.target
        prior = self._ready_event
        self._staged = False
        self._waited_streams.clear()
        copied = False
        with torch.cuda.stream(stream):
            if prior is not None:
                stream.wait_event(prior)
            elif stream != self._allocation_stream:
                # A fresh target can reuse cached memory with pending work on
                # its allocator stream. Order the first upload after that work.
                stream.wait_stream(self._allocation_stream)
            for recorded, event in tuple(self._release_events.items()):
                if recorded != stream:
                    stream.wait_event(event)
            for recorded in tuple(self._recorded_streams):
                if recorded != stream:
                    stream.wait_stream(recorded)
            try:
                plan.copy_to_target(
                    target,
                    non_blocking=non_blocking,
                )
                copied = True
            finally:
                # An update or later tensor copy can fail after earlier async
                # copies were enqueued. Record completion before retiring prior
                # consumer dependencies so cleanup still synchronizes all work.
                try:
                    self._ready_event = cast(torch.cuda.Event, stream.record_event())
                except BaseException:
                    self._recorded_streams.add(stream)
                    raise
                self._staged = copied
                self._release_events.clear()
                self._recorded_streams.clear()

    def acquire(self, stream: torch.cuda.Stream) -> HostModuleTarget:
        """Order ``stream`` after staging and mark it as the consumer."""
        if self._acquired:
            raise RuntimeError("CUDA target is already in use")
        event = self._ready_event
        if not self._staged or event is None:
            raise RuntimeError("CUDA target must be staged before use")
        if stream not in self._waited_streams:
            stream.wait_event(event)
            self._waited_streams.add(stream)
        self._acquired = True
        self.record_stream(stream)
        return self.target

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Order ``stream`` after staging and record it as a consumer."""
        if not self._acquired:
            raise RuntimeError("CUDA target must be acquired before use")
        event = self._ready_event
        if event is not None and stream not in self._waited_streams:
            stream.wait_event(event)
            self._waited_streams.add(stream)
        self._recorded_streams.add(stream)

    def release(self) -> None:
        """End the acquired state and snapshot stream completion for later reuse."""
        if not self._acquired:
            return
        for recorded in tuple(self._recorded_streams):
            self._release_events[recorded] = cast(
                torch.cuda.Event,
                recorded.record_event(),
            )
            self._recorded_streams.discard(recorded)
        self._acquired = False

    def close(self) -> None:
        """Release target storage after all recorded CUDA work completes.

        If a synchronization fails or is interrupted, the lease keeps its
        target and stream dependencies, so its owner still holds the storage
        and a later ``close()`` synchronizes again.
        """
        if self._target is None:
            return
        self._acquired = False
        if self._ready_event is not None:
            self._ready_event.synchronize()
        for event in self._release_events.values():
            event.synchronize()
        streams = set(self._recorded_streams)
        streams.add(self._allocation_stream)
        for stream in streams:
            stream.synchronize()
        self._target = None
        self._ready_event = None
        self._staged = False
        self._recorded_streams.clear()
        self._release_events.clear()
        self._waited_streams.clear()


__all__ = ["CudaTargetLease"]
