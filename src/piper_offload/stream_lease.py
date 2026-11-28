"""Stream-ordered CUDA buffer lifetime and reader/writer synchronization."""

import contextlib
from collections.abc import Callable, Iterator
from typing import cast

import torch


def _record_event(stream: torch.cuda.Stream) -> torch.cuda.Event:
    # PyTorch stubs type Stream.record_event() as ``Event | torch._C.Event``.
    return cast(torch.cuda.Event, stream.record_event())


class StreamLease:
    """Coordinate asynchronous H2D, compute, and D2H ordering on CUDA memory.

    Invariants:
    - At most one writer may stage into (or produce onto) the leased memory at a
      time, only after all prior consumers (``_release_events``,
      ``_recorded_streams``) and any pending D2H copy (``_d2h_event``) are
      ordered ahead of the write stream.
    - Concurrent read-only consumers may hold overlapping acquisitions
      (``exclusive=False``) once ``_staged`` is true, whereas exclusive users
      require sole access (``exclusive=True``).
    - GPU-born writes via ``produce()`` or ``produce_with()`` become readable
      once completed, may spill once to host via ``spill_to_host()``, and must
      complete ``wait_d2h()`` / ``poll_d2h()`` or be explicitly discarded via
      ``discard_dirty()`` before the buffer can be overwritten or freed.
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
        self._dirty = False
        self._d2h_event: torch.cuda.Event | None = None
        self._d2h_fallback_stream: torch.cuda.Stream | None = None
        self._d2h_succeeded = False

    @property
    def allocation_stream(self) -> torch.cuda.Stream:
        """Return the stream on which the backing storage was allocated."""
        return self._allocation_stream

    @property
    def is_staged(self) -> bool:
        """Return whether the latest H2D stage or GPU-born write succeeded."""
        return self._staged

    @property
    def is_acquired(self) -> bool:
        """Return whether at least one consumer holds this lease."""
        return self._acquired_count > 0

    @property
    def reader_count(self) -> int:
        """Return the number of active acquisitions on this lease."""
        return self._acquired_count

    @property
    def is_dirty(self) -> bool:
        """Return whether GPU-born contents have not yet finished spilling/discarding."""
        return self._dirty

    @property
    def d2h_in_flight(self) -> bool:
        """Return whether a write-once D2H spill is pending completion."""
        return self._d2h_event is not None or self._d2h_fallback_stream is not None

    def _wait_prior_dependencies(self, stream: torch.cuda.Stream) -> None:
        if self._ready_event is not None:
            stream.wait_event(self._ready_event)
        elif stream != self._allocation_stream:
            stream.wait_stream(self._allocation_stream)
        if self._d2h_event is not None:
            stream.wait_event(self._d2h_event)
        if self._d2h_fallback_stream is not None and self._d2h_fallback_stream != stream:
            stream.wait_stream(self._d2h_fallback_stream)
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

    def _record_ready_event(
        self,
        stream: torch.cuda.Stream,
        *,
        clear_d2h: bool = False,
    ) -> None:
        try:
            event = _record_event(stream)
        except BaseException:
            self._recorded_streams.add(stream)
            raise
        self._ready_event = event
        if clear_d2h:
            self._d2h_event = None
            self._d2h_fallback_stream = None
        self._release_events.clear()
        self._recorded_streams.clear()

    @contextlib.contextmanager
    def _write_ctx(
        self,
        stream: torch.cuda.Stream,
        *,
        mark_dirty: bool,
    ) -> Iterator[None]:
        action = "produced" if mark_dirty else "staged"
        if self._writing:
            raise RuntimeError(f"StreamLease cannot be {action} while a write is active")
        if self._acquired_count > 0:
            raise RuntimeError(f"StreamLease cannot be {action} while it is in use")
        if self._dirty:
            raise RuntimeError(
                "Dirty GPU-produced StreamLease must be spilled or discarded before restaging"
            )
        if self.d2h_in_flight:
            raise RuntimeError(
                f"StreamLease cannot be {action} while a D2H spill is pending"
            )
        try:
            self._staged = False
            self._writing = True
            self._waited_streams.clear()
            self._d2h_succeeded = False
            self._wait_prior_dependencies(stream)
            self._recorded_streams.add(stream)
            try:
                with torch.cuda.stream(stream):
                    yield
            except BaseException:
                self._record_ready_event(stream, clear_d2h=True)
                raise
            self._record_ready_event(stream, clear_d2h=True)
            if mark_dirty:
                self._dirty = True
            self._staged = True
        finally:
            self._writing = False

    @contextlib.contextmanager
    def produce(self, stream: torch.cuda.Stream) -> Iterator[None]:
        """Order ``stream`` after prior users for a batched GPU-born write, then mark dirty."""
        with self._write_ctx(stream, mark_dirty=True):
            yield

    def stage_with(
        self,
        copy_fn: Callable[[], None],
        stream: torch.cuda.Stream,
    ) -> None:
        """Order ``stream`` after prior users and run ``copy_fn`` to populate storage."""
        with self._write_ctx(stream, mark_dirty=False):
            copy_fn()

    def produce_with(
        self,
        write_fn: Callable[[], None],
        stream: torch.cuda.Stream,
    ) -> None:
        """Order ``stream`` after prior users, run GPU-born ``write_fn``, and mark dirty."""
        with self.produce(stream):
            write_fn()

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
        if exclusive and self.d2h_in_flight:
            raise RuntimeError(
                "StreamLease cannot be exclusively acquired while a D2H spill is in flight"
            )
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
            if self._stream_readers:
                if stream not in self._stream_readers:
                    raise ValueError("Stream has not acquired this StreamLease")
            elif stream not in self._recorded_streams and stream not in self._release_events:
                raise ValueError(
                    "Stream has not acquired or been recorded on this StreamLease"
                )
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

    def spill_to_host(
        self,
        copy_fn: Callable[[], None],
        d2h_stream: torch.cuda.Stream,
    ) -> None:
        """Launch a write-once asynchronous D2H copy on ``d2h_stream``."""
        if self._writing:
            raise RuntimeError("StreamLease cannot spill while a write is active")
        if not self._dirty:
            raise RuntimeError("StreamLease has no dirty GPU data to spill")
        if self.d2h_in_flight:
            raise RuntimeError("StreamLease D2H spill has already been started")
        if self._exclusive:
            raise RuntimeError("StreamLease cannot spill while exclusively acquired")
        self._d2h_succeeded = False
        if self._ready_event is not None:
            d2h_stream.wait_event(self._ready_event)
        elif d2h_stream != self._allocation_stream:
            d2h_stream.wait_stream(self._allocation_stream)
        self._d2h_fallback_stream = d2h_stream
        try:
            with torch.cuda.stream(d2h_stream):
                copy_fn()
        except BaseException:
            try:
                ev = _record_event(d2h_stream)
                self._d2h_event = ev
                self._d2h_fallback_stream = None
            except BaseException:
                self._d2h_fallback_stream = d2h_stream
            raise
        try:
            ev = _record_event(d2h_stream)
            self._d2h_event = ev
            self._d2h_fallback_stream = None
        except BaseException:
            self._d2h_fallback_stream = d2h_stream
            raise
        self._d2h_succeeded = True

    def _finish_d2h_flags(self) -> None:
        if self._d2h_succeeded:
            self._dirty = False
            self._d2h_succeeded = False

    def poll_d2h(self) -> bool:
        """Non-blockingly finalize completed D2H work; return ``True`` when idle."""
        if self._d2h_event is not None:
            if not self._d2h_event.query():
                return False
            self._d2h_event = None
        if self._d2h_fallback_stream is not None:
            if not self._d2h_fallback_stream.query():
                return False
            self._d2h_fallback_stream = None
        self._finish_d2h_flags()
        return True

    def wait_d2h(self) -> None:
        """Wait for any pending D2H spill to finish and clear dirty only on success."""
        if self._d2h_event is not None:
            self._d2h_event.synchronize()
            self._d2h_event = None
        if self._d2h_fallback_stream is not None:
            self._d2h_fallback_stream.synchronize()
            self._d2h_fallback_stream = None
        self._finish_d2h_flags()

    def discard_dirty(self) -> None:
        """Drop dirty GPU-born contents cleanly without leaking in-flight D2H writes."""
        if self._writing:
            raise RuntimeError("StreamLease cannot be discarded while a write is active")
        if self._acquired_count > 0:
            raise RuntimeError("StreamLease cannot be discarded while it is in use")
        self.wait_d2h()
        self._dirty = False
        self._d2h_succeeded = False
        self._staged = False

    def synchronize_and_reset(self) -> None:
        """Synchronize every pending D2H, ready, consumer, and allocation stream."""
        if self._writing:
            raise RuntimeError("StreamLease cannot be synchronized or reset while a write is active")
        self.wait_d2h()
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
        self._dirty = False
        self._d2h_succeeded = False
        self._waited_streams.clear()


__all__ = ["StreamLease"]
