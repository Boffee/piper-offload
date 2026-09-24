"""Windows checkpoint copies: offered retention, reclamation, and commitment pressure."""

import contextlib
import logging
import os
import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor

from . import _copy_memory as shared
from ._copy_memory import Copy, CopyLoad, CopyOwner, ReleaseBatch
from ._host_memory_windows import Readers, VirtualRegion, commit_exhausted, reading_below_offers
from .checkpoint import FileSlice

logger = logging.getLogger(__name__)

# Page-table work stops scaling before the core count: six reclaim workers
# took 1.25 s for 18 GiB under pressure, against 3.5 s on sixteen.
_RECLAIM_WORKERS = 6
_COMMIT_WATCH_SECONDS = 0.5


class WindowsCopy(Copy):
    region: VirtualRegion

    def __init__(self, size: int) -> None:
        super().__init__(size, VirtualRegion(size))

    def offer(self) -> None:
        """Drop the storage and offer the pages to Windows; ``BufferError`` if a view of them is still alive."""
        self.storage = None
        self.region.offer()

    def reclaim(self) -> list[tuple[int, int]]:
        """Take the offered pages back and expose them again, returning the spans that need refilling."""
        discarded = self.region.reclaim()
        self._expose()
        intact = not discarded
        if intact:
            # Here rather than in the pass before registration, so that one
            # worker's touch overlaps another's reclaim; a copy the fill has
            # to rebuild is touched after it, once it holds its bytes.
            self.touch()
        return discarded


def _read_range_below_offers(
    read_at: Callable[[memoryview, int], int], view: memoryview, offset: int, start: int, stop: int,
) -> None:
    """``_read_range``, bringing the file into the cache below offered copies.

    The slice of the copy faults in first, at the thread's own memory
    priority, so the lowest one the read runs at (:func:`reading_below_offers`)
    reaches only the file cache. At it, the copy's own pages would be what
    Windows trims first: under pressure, registering 18 GiB of refilled copies
    took 11.0 s that way, against 3.2 s.
    """
    shared._fault_in(view, start, stop)
    with reading_below_offers():
        shared._read_range(read_at, view, offset, start, stop)


def _offer_copy(copy: Copy) -> str | None:
    """Offer one copy's pages: None, or why they could not be offered, as text that holds no traceback."""
    try:
        assert isinstance(copy, WindowsCopy)
        copy.offer()
    except (BufferError, OSError) as error:
        return str(error)
    return None


class _OfferBatch(ReleaseBatch):
    """The copies an eviction is offering, each started as its unregistration succeeds.

    Unregistration is serial whatever the caller does, because the runtime
    serializes it: 6.9 GiB of copies leave the runtime at 28.7 GiB/s on one
    thread and 26.1 on sixteen. Offering, which does scale with threads, runs
    beside that serial stream rather than after it, so an eviction costs what
    the slower of the two does: 6.9 GiB took 0.52 s as two passes and 0.37 s
    this way.

    Copies are added under the manager's lock and waited for without it: a
    worker that collects garbage may run a finalizer that takes the lock, so
    the lock may be held while they run but never while they are joined.
    Until then the copies are neither pinned nor in the tier, and no other
    thread can reclaim, free, or read them.
    """

    __slots__ = ("_futures", "_pool")

    def __init__(self, memory: Memory) -> None:
        self._memory = memory
        self._futures: list[tuple[CopyOwner, Future[str | None]]] = []
        self._pool: ThreadPoolExecutor | None = None

    def add(self, registration: CopyOwner) -> None:
        """Start offering a copy whose unregistration has just succeeded; called with the lock held."""
        assert registration.copy is not None
        if self._pool is None:
            workers = os.process_cpu_count() or 1
            self._pool = ThreadPoolExecutor(workers, thread_name_prefix="piper-offload-offer")
        self._futures.append((registration, self._pool.submit(_offer_copy, registration.copy)))

    def take(self) -> list[tuple[CopyOwner, str | None]]:
        """Empty the batch and wait for its offers, returning each copy and why it could not be offered."""
        futures, self._futures = self._futures, []
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown()
        return [(registration, future.result()) for registration, future in futures]

    def __enter__(self) -> None:
        self._memory._offering = self

    def __exit__(self, *_exc: object) -> None:
        self._memory._offering = None

    def finish(self) -> None:
        self._memory._offer_together(self)


def _reclaim_copy(item: CopyLoad) -> None:
    """Take one offered copy back: intact, discarded in part and left to the fill, or rebuilt where Windows refuses.

    A rebuilt copy keeps ``missing`` as it was reserved, the whole copy.
    """
    try:
        assert isinstance(item.copy, WindowsCopy)
        item.missing = item.copy.reclaim()
    except OSError as error:
        # Tracebacks in buffered logs would hold the region this frees.
        logger.warning("Could not reclaim an offered pinned copy; rebuilding it: %s", str(error))
        item.copy.free()
        try:
            item.copy = WindowsCopy(item.copy.size)
        except (OSError, MemoryError) as failure:
            item.error = OSError(f"could not allocate its replacement: {failure}")


def _reclaimed(pending: list[CopyLoad]) -> Generator[CopyLoad]:
    """Reclaim the offered copies in ``pending``, yielding each in order as soon as it is back.

    Intact pages need no fill. Discarded ones (``ERROR_BUSY``) are undefined
    and are refilled in full. A copy that cannot be reclaimed is freed and
    rebuilt in a new allocation of the size already reserved, which fills like
    any other. Runs without the manager's lock, as the fill does, and on a
    small pool: reclaiming is page-table work, 233 ms/GiB on one thread and
    102 ms/GiB on four, with nothing left beyond that.

    The caller registers each intact copy as it arrives, so the runtime's
    serial registration runs beside the reclaims rather than after them:
    reacquiring 6.9 GiB of intact copies took 1.16 to 1.20 s as two passes and
    0.93 to 0.97 s this way. It must not hold the lock while it waits for
    the next copy, since a worker's finalizer may need it. Closing the
    generator cancels the reclaims not yet started, whose copies stay
    offered, and waits for the rest, so the caller may then free every copy.
    """
    offered = [item for item in pending if item.offered]
    if len(offered) < 2:
        for item in offered:
            _reclaim_copy(item)
            yield item
        return
    pool = ThreadPoolExecutor(min(_RECLAIM_WORKERS, len(offered)), thread_name_prefix="piper-offload-reclaim")
    try:
        futures = [pool.submit(_reclaim_copy, item) for item in offered]
        for item, future in zip(offered, futures, strict=True):
            future.result()
            yield item
    finally:
        pool.shutdown(cancel_futures=True)


def _watch_commit(memory_ref: weakref.ReferenceType[Memory]) -> None:
    """Free offered copies once commitment is exhausted; return once its tier is empty or it is gone."""
    while True:
        commit_exhausted(_COMMIT_WATCH_SECONDS)  # returns as soon as it is, at the latest after the interval
        memory = memory_ref()
        if memory is None or not memory._keep_commit():
            return
        del memory


class Memory(shared.Memory):
    """Own the offered tier, sharing the manager's lock but never retaining the manager.

    A reserved copy leaves the tier before the manager makes room for it.
    A released copy enters only after successful unregistration. Batch workers
    own copies until joined, so neither retirement nor commitment pressure can
    free memory while an offer is running.
    """

    readers = Readers

    def __init__(self, max_offered_bytes: int, lock: threading.RLock, occupied: Callable[[int], bool]) -> None:
        super().__init__(max_offered_bytes, lock, occupied)
        self._offered: OrderedDict[int, CopyOwner] = OrderedDict()
        self.offered_bytes = 0
        self._offering: _OfferBatch | None = None
        self._watching_commit = False

    def new_copy(self, size: int) -> Copy:
        return WindowsCopy(size)

    def reserve(self, _pointer: int, size: int, source: FileSlice, fit: Callable[[int], bool]) -> CopyLoad | None:
        pointer = _pointer
        offered = self._offered.get(pointer)
        if offered is None:
            return super().reserve(pointer, size, source, fit)
        copy = self._take_offered(pointer)
        # Out of the tier before making room, so the evictions cannot free it.
        if not fit(copy.size):
            # It still fits: this acquisition's new offers are not admitted yet.
            self._offered[pointer] = offered
            self.offered_bytes += copy.size
            return None
        return CopyLoad(source=source, copy=copy, missing=[(0, copy.size)], offered=True)

    def allocate(self, size: int) -> Copy | None:
        # The new copy's commitment may be what Windows cannot find; the tier's is the process's to give.
        for released in self._relieve_commit():
            released.free()
        try:
            copy = self.new_copy(size)
        except (OSError, MemoryError) as error:
            copy = self._allocate_after_freeing_offered(size) if self._offered else None
            if copy is None:
                logger.warning("Could not allocate a pinned copy; the checkpoint storage stays pageable: %s", error)
                return None
        return copy

    def _allocate_after_freeing_offered(self, size: int) -> Copy | None:
        """Free offered copies, least recent first, until ``size`` bytes of their commitment return, and retry.

        Offered pages stay committed, so a failed allocation is the commit
        pressure that has to free them.
        """
        for copy in self._evict_offered(self.offered_bytes - size):
            copy.free()
        try:
            return self.new_copy(size)
        except (OSError, MemoryError):
            return None

    def _offer_together(self, batch: _OfferBatch) -> None:
        """Wait for the offers a locked section started, then put each copy in the tier or free it.

        A copy is freed if Windows refused it, or if while it was being
        offered its owner was dropped, its storage got another copy, or the
        budget shrank below it.
        """
        taken = batch.take()
        if not taken:
            return
        released: list[Copy] = []
        with self._lock:
            for registration, failure in taken:
                released.extend(self._settle_offer(registration, failure))
        shared._free_copies(released)

    def _settle_offer(self, registration: CopyOwner, failure: str | None) -> list[Copy]:
        """Admit a completed offer and return the copies its caller must free, including any rejected offer."""
        copy = registration.copy
        assert copy is not None
        if failure is not None:
            logger.warning("Could not offer an evicted pinned copy; freeing it: %s", failure)
            return [copy]
        pointer = registration.pointer
        rebuilt = self._occupied(pointer) or pointer in self._offered
        if registration.retired or rebuilt or copy.size > self.max_offered_bytes:
            return [copy]
        released = self._evict_offered(self.max_offered_bytes - copy.size)
        self._offered[pointer] = registration
        self.offered_bytes += copy.size
        released.extend(self._relieve_commit())
        if self._offered and not self._watching_commit:
            self._watching_commit = True
            threading.Thread(
                target=_watch_commit, args=(weakref.ref(self),), name="piper-offload-commit", daemon=True,
            ).start()
        return released

    def _relieve_commit(self) -> list[Copy]:
        """Detach offered copies for freeing if Windows reports its commitment exhausted."""
        if self._offered and commit_exhausted():
            logger.warning("Windows cannot commit more memory; freeing %d offered copies", len(self._offered))
            return self._evict_offered(0)
        return []

    def _keep_commit(self) -> bool:
        """The watcher's turn: relieve exhausted commitment; False once the tier is empty, which ends the watch."""
        with self._lock:
            released = self._relieve_commit()
            watching = self._watching_commit = bool(self._offered)
        shared._free_copies(released)
        return watching

    def _evict_offered(self, target: int) -> list[Copy]:
        """Detach least recently offered copies down to ``target`` bytes; the caller frees their regions."""
        released: list[Copy] = []
        while self._offered and self.offered_bytes > target:
            released.append(self._take_offered(next(iter(self._offered))))
        return released

    def _take_offered(self, pointer: int) -> Copy:
        """Remove a copy and its charge from the tier, transferring ownership to the caller."""
        copy = self._offered.pop(pointer).copy
        assert copy is not None
        self.offered_bytes -= copy.size
        return copy

    def release_batch(self) -> ReleaseBatch:
        return _OfferBatch(self)

    def release(self, owner: CopyOwner, *, demote: bool) -> list[Copy]:
        copy = owner.copy
        assert copy is not None
        if demote and not owner.retired and copy.size <= self.max_offered_bytes:
            if self._offering is not None:
                self._offering.add(owner)
                return []
            return self._settle_offer(owner, _offer_copy(copy))
        return [copy]

    def retire(self, _owner: CopyOwner) -> None:
        if self._offered.get(_owner.pointer) is _owner:
            self._take_offered(_owner.pointer).free()

    def resize(self, value: int) -> list[Copy]:
        self.max_offered_bytes = value
        return self._evict_offered(value)

    def clear(self) -> list[Copy]:
        return self._evict_offered(0)

    def prepare(self, copies: list[CopyLoad]) -> Generator[None]:
        # Snapshot the fill policy while reservation still holds the lock.
        return self._prepare(copies, below_offers=self.max_offered_bytes > 0)

    def _prepare(self, copies: list[CopyLoad], *, below_offers: bool) -> Generator[None]:
        """Reclaim all requested copies before any fill, overlapping intact copies with registration.

        Closing joins all reclaim workers before the manager may free any
        pending copy. Readiness is published only after its worker is joined.
        """
        with contextlib.closing(_reclaimed(copies)) as reclaimed:
            for item in reclaimed:
                item.offered = False
                yield
        unfilled = [item for item in copies if item.missing and item.error is None]
        # Apply low-priority file reads from the first load, before anything
        # is offered, so its page cache cannot outrank copies offered later.
        read = _read_range_below_offers if below_offers else shared._read_range
        yield from self._filled(unfilled, read=read)
