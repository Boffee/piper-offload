"""One CPU allocation: its source, an optional pinned copy, and the reads on it.

CPU tensors keep their source storage: a private file mapping for mmap
weights, or anonymous memory for parameters constructed in RAM. ``pin`` makes
the bytes DMA-ready: a source that may be pinned in place is registered where
it is; a private file mapping never is, because a writable page lock forces
copy-on-write across the checkpoint, so it is copied once into an owned
anonymous allocation that is registered instead. ``unpin`` releases the
registration and keeps the copy; ``evict`` frees the copy too. ``copy_to``
reads the copy when there is one, else the source.

The backing protects itself: every asynchronous CUDA copy records a
completion event, and the backing refuses to unpin or evict until every
recorded event has passed. A lease from the manager additionally keeps a
backing from being unpinned or evicted across a whole activation. Disposing
the last owner waits for in-flight copies, unregisters, and frees the
allocation.

Parameters, buffers, and views sharing a storage share one backing. Backings
do not reference their manager; the manager indexes them weakly and applies
budget policy through the primitives below.
"""

import contextlib
import logging
import mmap
import sys
import threading
import time
import weakref
from collections.abc import Callable, Generator
from typing import Protocol, Self

import torch

from ._host_registration import HostRegistrationBackend

logger = logging.getLogger(__name__)


class CopyEvent(Protocol):
    """The event recorded after an asynchronous copy, shaped like ``torch.cuda.Event``."""

    def query(self) -> bool: ...

    def synchronize(self) -> None: ...


class EventStream(Protocol):
    """Records an event for the work enqueued so far, shaped like ``torch.cuda.Stream``."""

    def record_event(self) -> CopyEvent: ...

    def synchronize(self) -> None: ...


def _close_lease(backings: tuple[HostBacking, ...], on_close: Callable[[], None] | None) -> None:
    try:
        for backing in backings:
            backing.release()
    finally:
        if on_close is not None:
            on_close()


class HostLease:
    """Protect a batch of backings until close.

    Created by ``HostMemoryManager.acquire``. While open, the leased backings
    are neither unpinned nor evicted. Dropping the lease closes it. Close is
    idempotent, and a closed lease retains nothing.
    """

    def __init__(self, backings: tuple[HostBacking, ...], on_close: Callable[[], None] | None = None) -> None:
        self._backings = backings
        self._finalizer = weakref.finalize(self, _close_lease, backings, on_close)
        self._finalizer.atexit = False

    @property
    def backings(self) -> tuple[HostBacking, ...]:
        """The protected backings; unavailable once closed."""
        if self.closed:
            raise RuntimeError("Host lease is closed")
        return self._backings

    @property
    def pinned(self) -> bool:
        """Whether every leased backing is registered by its manager."""
        return all(backing.pinned for backing in self.backings)

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def close(self) -> None:
        try:
            self._finalizer()
        finally:
            self._backings = ()

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("Host lease is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class HostBacking:
    """A source storage, an optional owned copy of it, and the registration covering one of them.

    The registration, when present, always covers the copy if there is one,
    else the source. Source tensor shape, stride, dtype and storage offset
    determine each resolved view; the copy holds the entire source
    allocation's bytes, including portions outside an individual tensor view.
    Non-trainable captures promise immutable bytes, which is what makes a copy
    valid.
    """

    __slots__ = (
        "__weakref__",
        "_active",
        "_backend",
        "_copy",
        "_in_flight",
        "_lock",
        "_pin_in_place",
        "_pinned",
        "_released_at",
        "_source",
    )

    def __init__(
        self,
        storage: torch.UntypedStorage,
        backend: HostRegistrationBackend,
        *,
        pin_in_place: bool | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._source = storage
        # Memory PyTorch did not allocate is assumed to be a private file mapping.
        self._pin_in_place = storage.resizable() if pin_in_place is None else pin_in_place
        self._copy: torch.UntypedStorage | None = None
        self._pinned = False
        self._backend = backend
        self._active = 0
        self._in_flight: list[CopyEvent] = []
        self._released_at = 0

    def __del__(self) -> None:
        # Native registrations must never outlive their storage. Runtime
        # calls are skipped during interpreter exit. The backend always issues
        # the native unregister; it only raises when that call itself fails,
        # which means an unregistered pointer or a dead context, and neither
        # leaves pages the driver could still touch, so the storage is freed.
        if sys.is_finalizing():
            return
        for event in getattr(self, "_in_flight", ()):
            event.synchronize()
        if getattr(self, "_pinned", False):
            try:
                self._backend.unregister(self._selected().data_ptr())
            except Exception as error:
                logger.warning("Host unregistration failed during cleanup: %s", str(error))

    # Facts.

    @property
    def storage(self) -> torch.UntypedStorage:
        """The source storage this backing was captured from."""
        return self._source

    @property
    def pin_in_place(self) -> bool:
        """Whether ``pin`` registers the source where it is, rather than a copy of it.

        False for a private file mapping, whose pages the kernel would copy
        into RAM the moment they were locked writable. Memory PyTorch did not
        allocate is assumed to be one unless the owner said otherwise at
        capture; a shared mapping, for example, has no copy-on-write.
        """
        return self._pin_in_place

    # State.

    @property
    def pinned(self) -> bool:
        """Whether the copy, or else the source, is registered."""
        with self._lock:
            return self._pinned

    @property
    def copy_bytes(self) -> int:
        """Bytes held by the owned copy, or zero."""
        with self._lock:
            return 0 if self._copy is None else self._copy.nbytes()

    @property
    def needs_copy(self) -> bool:
        """Whether ``pin`` would have to allocate a copy: a private file mapping with none yet."""
        with self._lock:
            return not self._pin_in_place and self._copy is None

    @property
    def leases(self) -> int:
        """Open leases."""
        with self._lock:
            return self._active

    @property
    def in_flight(self) -> int:
        """Asynchronous copies enqueued but not yet complete."""
        with self._lock:
            self._discard_completed()
            return len(self._in_flight)

    @property
    def idle(self) -> bool:
        """No open lease and no unfinished copy, so nothing can be reading this backing."""
        with self._lock:
            self._discard_completed()
            return self._active == 0 and not self._in_flight

    @property
    def released_at(self) -> int:
        """Monotonic time the last lease closed, ordering idle backings for eviction."""
        return self._released_at

    @property
    def span(self) -> tuple[int, int]:
        """Pointer and size of the storage reads use: the copy if any, else the source."""
        with self._lock:
            storage = self._selected()
            return storage.data_ptr(), storage.nbytes()

    @property
    def page_bytes(self) -> int:
        """Bytes of the OS pages ``pin`` locks: the selected storage's pages, or the aligned copy's."""
        page = mmap.PAGESIZE
        with self._lock:
            pointer, size = self.span
            if size == 0:
                return 0
            if self.needs_copy:
                return -(-size // page) * page
            return ((pointer + size - 1) // page - pointer // page + 1) * page

    # Reads.

    def copy_to(self, destination: torch.Tensor, source: torch.Tensor, *, non_blocking: bool) -> None:
        """Copy the source view's bytes to ``destination`` from the copy if any, else the source.

        A CUDA destination is written on its current stream: asynchronous DMA
        when the storage is pinned, otherwise the driver's synchronous pageable
        copy. A completion event recorded on that stream keeps the backing from
        being unpinned or evicted until the copy has finished, so no caller
        needs a lease to copy safely. CUDA graph capture is rejected while a
        copy is selected, because a captured node reads the pointer on every
        replay and the copy can be evicted. Other accelerators cannot report
        completion, so their copies are made synchronous.
        """
        if source.device.type != "cpu":
            raise ValueError("Host copies require a CPU source")
        stream: EventStream | None = None
        if destination.device.type == "cuda":
            with torch.cuda.device(destination.device):
                if torch.cuda.is_current_stream_capturing():
                    if self.copy_bytes:
                        raise RuntimeError("Copies from an evictable host copy cannot be captured in CUDA graphs")
                else:
                    stream = torch.cuda.current_stream(destination.device)
        elif destination.device.type != "cpu":
            non_blocking = False
        with self._read(source, stream) as view:
            _transfer(destination, view, non_blocking=non_blocking)

    @contextlib.contextmanager
    def _read(self, source: torch.Tensor, stream: EventStream | None = None) -> Generator[torch.Tensor]:
        """Yield the source's view on the selected storage, busy until any marker on ``stream`` passes."""
        if source.untyped_storage()._cdata != self._source._cdata:
            raise ValueError("Source does not belong to this host backing")
        with self._lock:
            if self._copy is None:
                view = source
            else:
                view = torch.empty(0, dtype=source.dtype, device="cpu").set_(
                    self._copy,
                    source.storage_offset(),
                    source.shape,
                    source.stride(),
                )
            self._active += 1
        event: CopyEvent | None = None
        try:
            yield view
        finally:
            try:
                if stream is not None:
                    try:
                        event = stream.record_event()
                    except BaseException:
                        # Even a failing copy can have queued partial work.
                        # Without a marker, wait for it before releasing.
                        stream.synchronize()
                        raise
            finally:
                with self._lock:
                    if event is not None:
                        # Prune here too: steady-state reuse never queries
                        # idle, so this is what keeps the list bounded.
                        self._discard_completed()
                        self._in_flight.append(event)
                self.release()

    # Primitives for the owning manager. They are budget-unaware: pin and
    # unpin through HostMemoryManager, which accounts for them.

    def hold(self) -> None:
        """Open a lease. Pair with ``release``; ``HostLease`` does this for a batch."""
        with self._lock:
            self._active += 1

    def release(self) -> None:
        with self._lock:
            self._active -= 1
            if self._active == 0:
                self._released_at = time.monotonic_ns()

    def pin(self) -> bool:
        """Make the bytes DMA-ready, returning False when the runtime refuses capacity.

        A source that may be pinned in place, or an existing copy, is
        registered where it is. Otherwise the source bytes are copied once
        into an owned page-aligned anonymous allocation that is registered
        instead (see ``needs_copy``). Budget is the caller's concern.
        """
        with self._lock:
            if self._pinned:
                return True
            if self.needs_copy:
                copy = _anonymous_storage(self._source.nbytes())
                _bytes(copy).copy_(_bytes(self._source))
                if not self._backend.register(copy.data_ptr(), copy.nbytes()):
                    return False
                self._copy = copy
                self._pinned = True
                return True
            if not self._backend.register(*self.span):
                return False
            self._pinned = True
            return True

    def unpin(self) -> bool:
        """Release the registration, keeping the copy. False while busy or when the runtime refuses.

        Busy means a lease is open or an asynchronous copy has not completed.
        A refused unregistration keeps the storage and its budget charge so the
        registration can be retried; nothing registered is ever freed.
        """
        with self._lock:
            if not self.idle:
                return False
            if not self._pinned:
                return True
            try:
                self._backend.unregister(self._selected().data_ptr())
            except Exception as error:
                # Tracebacks in buffered logs can retain storage after a later retry.
                logger.warning("Host unregistration failed; retaining storage and budget charge: %s", str(error))
                return False
            self._pinned = False
            return True

    def evict(self) -> bool:
        """Unpin and free the copy, so reads fall back to the source. False when unpin fails."""
        with self._lock:
            if not self.unpin():
                return False
            self._copy = None
            return True

    def _selected(self) -> torch.UntypedStorage:
        return self._copy if self._copy is not None else self._source

    def _discard_completed(self) -> None:
        self._in_flight[:] = [event for event in self._in_flight if not event.query()]


def _anonymous_storage(nbytes: int) -> torch.UntypedStorage:
    """An owned, page-aligned anonymous allocation released with its last reference.

    Page alignment keeps two copies from sharing an OS page, which the
    runtime would refuse to register twice.
    """
    if nbytes == 0:
        return torch.UntypedStorage(0)
    return torch.frombuffer(mmap.mmap(-1, nbytes), dtype=torch.uint8).untyped_storage()


def _bytes(storage: torch.UntypedStorage) -> torch.Tensor:
    return torch.empty(0, dtype=torch.uint8, device="cpu").set_(storage, 0, (storage.nbytes(),), (1,))


def _transfer(destination: torch.Tensor, view: torch.Tensor, *, non_blocking: bool) -> None:
    """The single copy primitive; tests observe the resolved view here."""
    destination.copy_(view, non_blocking=non_blocking)


__all__ = ["CopyEvent", "EventStream", "HostBacking", "HostLease"]
