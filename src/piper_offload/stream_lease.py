"""Stream-ordered CUDA buffer lifetime and reader/writer synchronization."""

import contextlib
from collections.abc import Callable, Iterator
from typing import cast

import torch


def _record_event(stream: torch.cuda.Stream) -> torch.cuda.Event:
    # PyTorch stubs type Stream.record_event() as ``Event | torch._C.Event``.
    return cast(torch.cuda.Event, stream.record_event())


class StreamLease:
    """Coordinate asynchronous writer/reader stream ordering on CUDA memory.

    Invariants:
    - At most one writer may populate leased storage at a time via
      :meth:`write` or :meth:`stage_with`, ordered after the allocation stream,
      prior ready event, and all prior consumers (``_release_events`` and
      ``_recorded_streams``).
    - Concurrent read-only consumers may hold overlapping acquisitions
      (``exclusive=False``) once ``_staged`` is true, whereas exclusive users
      require sole access (``exclusive=True``).
    - ``synchronize_and_reset()`` leaves state intact on interruption so a
      subsequent call retries safely without leaking unsynchronized work.
    """

    def __init__(self, allocation_stream: torch.cuda.Stream) -> None:
        self._allocation_stream = allocation_stream
        self._ready_event: torch.cuda.Event | None = None
        self._staged = False
        self._writing = False
        self._acquired_count = 0
        self._exclusive = False
        self._stream_readers: dict[torch.cuda.Stream, int] = {}
        self._recorded_streams: set[torch.cuda.Stream] = set()
        self._release_events: dict[torch.cuda.Stream, torch.cuda.Event] = {}
        self._waited_streams: set[torch.cuda.Stream] = set()

    @property
    def allocation_stream(self) -> torch.cuda.Stream:
        """Return the stream on which the backing storage was allocated."""
        return self._allocation_stream

    @property
    def is_staged(self) -> bool:
        """Return whether the latest write or stage completed without error."""
        return self._staged

    @property
    def is_acquired(self) -> bool:
        """Return whether at least one consumer holds this lease."""
        return self._acquired_count > 0

    @property
    def reader_count(self) -> int:
        """Return the number of active acquisitions on this lease."""
        return self._acquired_count

    def _wait_prior_dependencies(self, stream: torch.cuda.Stream) -> None:
        if self._ready_event is not None:
            stream.wait_event(self._ready_event)
        elif stream != self._allocation_stream:
            stream.wait_stream(self._allocation_stream)
        for recorded, event in self._release_events.items():
            if recorded != stream:
                stream.wait_event(event)
        for recorded_stream in self._recorded_streams:
            if recorded_stream != stream:
                stream.wait_stream(recorded_stream)

    def _wait_ready(self, stream: torch.cuda.Stream) -> None:
        if self._ready_event is not None and stream not in self._waited_streams:
            stream.wait_event(self._ready_event)
            self._waited_streams.add(stream)

    def _record_ready_event(self, stream: torch.cuda.Stream) -> None:
        try:
            event = _record_event(stream)
        except BaseException:
            self._recorded_streams.add(stream)
            raise
        self._ready_event = event
        self._release_events.clear()
        self._recorded_streams.clear()

    @contextlib.contextmanager
    def write(self, stream: torch.cuda.Stream) -> Iterator[None]:
        """Order ``stream`` after prior users for a single exclusive write."""
        if self._writing:
            raise RuntimeError("StreamLease cannot be written while a write is active")
        if self._acquired_count > 0:
            raise RuntimeError("StreamLease cannot be written while it is in use")
        try:
            self._staged = False
            self._writing = True
            self._waited_streams.clear()
            self._wait_prior_dependencies(stream)
            self._recorded_streams.add(stream)
            try:
                with torch.cuda.stream(stream):
                    yield
            except BaseException:
                self._record_ready_event(stream)
                raise
            self._record_ready_event(stream)
            self._staged = True
        finally:
            self._writing = False

    def stage_with(
        self,
        copy_fn: Callable[[], None],
        stream: torch.cuda.Stream,
    ) -> None:
        """Order ``stream`` after prior users and run ``copy_fn`` to populate storage."""
        with self.write(stream):
            copy_fn()

    def acquire(
        self,
        stream: torch.cuda.Stream,
        *,
        exclusive: bool = False,
    ) -> None:
        """Order ``stream`` after the ready event and register a lease holder."""
        if self._writing:
            raise RuntimeError("StreamLease cannot be acquired while a write is active")
        if not self._staged:
            raise RuntimeError("StreamLease must be staged before acquire()")
        if self._exclusive or (exclusive and self._acquired_count > 0):
            raise RuntimeError("StreamLease is already acquired")
        prev_count = self._acquired_count
        prev_exclusive = self._exclusive
        prev_stream_count = 0 if prev_count == 0 else self._stream_readers.get(stream, 0)
        if prev_count == 0:
            self._stream_readers.clear()
            self._exclusive = False
        try:
            self._acquired_count = prev_count + 1
            if exclusive:
                self._exclusive = True
            self._stream_readers[stream] = prev_stream_count + 1
            self.record_stream(stream)
        except BaseException:
            self._acquired_count = prev_count
            self._exclusive = prev_exclusive
            if prev_stream_count == 0:
                self._stream_readers.pop(stream, None)
            else:
                self._stream_readers[stream] = prev_stream_count
            raise

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Record an additional consumer stream during an active acquisition."""
        if self._acquired_count == 0:
            raise RuntimeError("StreamLease must be acquired before record_stream()")
        self._wait_ready(stream)
        self._recorded_streams.add(stream)

    def release(self, stream: torch.cuda.Stream | None = None) -> None:
        """Snapshot completion events for ``stream`` (or all streams when omitted)."""
        if self._acquired_count == 0:
            return
        if stream is not None:
            if stream not in self._stream_readers:
                raise ValueError("Stream has not acquired this StreamLease")
            stream_count = self._stream_readers.get(stream, 0)
            if self._acquired_count > 1:
                if stream_count > 1:
                    self._stream_readers[stream] = stream_count - 1
                    self._acquired_count -= 1
                    return
                if stream in self._recorded_streams:
                    self._release_events[stream] = _record_event(stream)
                    self._recorded_streams.remove(stream)
                self._stream_readers.pop(stream, None)
                self._acquired_count -= 1
                return

        while self._recorded_streams:
            recorded_stream = next(iter(self._recorded_streams))
            self._release_events[recorded_stream] = _record_event(recorded_stream)
            self._recorded_streams.remove(recorded_stream)
        self._exclusive = False
        self._stream_readers.clear()
        self._acquired_count = 0

    def synchronize_and_reset(self) -> None:
        """Synchronize every pending ready, consumer, and allocation stream."""
        if self._writing:
            raise RuntimeError(
                "StreamLease cannot be synchronized or reset while a write is active"
            )
        if self._ready_event is not None:
            self._ready_event.synchronize()
            self._ready_event = None
        for stream, event in list(self._release_events.items()):
            event.synchronize()
            del self._release_events[stream]
        while self._recorded_streams:
            recorded_stream = next(iter(self._recorded_streams))
            recorded_stream.synchronize()
            self._recorded_streams.remove(recorded_stream)
        self._allocation_stream.synchronize()
        self._acquired_count = 0
        self._exclusive = False
        self._stream_readers.clear()
        self._staged = False
        self._writing = False
        self._waited_streams.clear()


__all__ = ["StreamLease"]
