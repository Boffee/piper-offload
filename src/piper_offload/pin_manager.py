"""Budgeted host registrations with active leases and an idle LRU.

Use the process-wide ``host_pin_manager`` for application registrations. Its
budget defaults to half of the memory available to the process, physical RAM
or the container's cgroup limit when that is lower, rounded down to whole OS
pages. Set
it to zero to disable registration or ``None`` to remove the budget and
register up to the CUDA/HIP runtime's capacity. Construction
and configuration perform no CUDA initialization. Isolated ``PinManager``
instances can use an injected backend for testing.

Every host transfer runs under a lease, so nothing is read or written while
it could be evicted, and only transfers that repeat ask the lease to pin:
streaming, rolling, and the relay register their storage; a resident or host
upload and the optimizer copy-back lease it pageable.

Registration uses whole storage byte ranges. Storage that records a
checkpoint file slice (:func:`file_slice`) is never pinned in place: the
mapping stays read-only page cache, and pinning it means allocating an owned
page-aligned copy, filling it from the file with parallel reads, and
registering that. Evicting the copy unregisters and frees it, so the RAM
returns; on Windows an optional offered tier (``max_offered_bytes``) instead
offers an evicted copy's pages to the OS, which may discard them, and a later
pin reclaims them and refills only what was discarded. Transfers reach the
copy through ``transfer_``, which resolves each
source tensor at copy time; an asynchronous transfer of pinned storage
outside a lease raises, since eviction is safe only because every such
reader holds one, while a synchronous one completes under the manager's
lock. Other storage, anonymous memory and mappings without provenance,
registers where it is, as before.
Piper never writes into a file mapping: ``HostParam`` copies trainable
parameters out at capture and ``merge_adapter`` copies its targets out before
merging.

Budget accounting counts the union of OS pages, including pages shared by
separate storages. Registrations retain storage until unregistration
succeeds, but refer to their owners weakly while idle so a dropped resource
can release memory. Storage must not be resized or independently
registered while managed here.
"""

import contextlib
import ctypes
import enum
import functools
import io
import logging
import mmap
import operator
import os
import sys
import threading
import weakref
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import torch

from ._host_memory import VirtualRegion, commit_exhausted, reading_below_offers
from ._host_registration import HostRegistrationBackend, HostRegistrationRefusedError, RuntimeHostRegistration
from .checkpoint import FileSlice, file_slice

logger = logging.getLogger(__name__)


class _MemoryStatus(ctypes.Structure):
    # Windows MEMORYSTATUSEX; fixed-width types also allow testing on POSIX.
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("load", ctypes.c_uint32),
        ("total_physical", ctypes.c_uint64),
        ("available_physical", ctypes.c_uint64),
        ("total_page_file", ctypes.c_uint64),
        ("available_page_file", ctypes.c_uint64),
        ("total_virtual", ctypes.c_uint64),
        ("available_virtual", ctypes.c_uint64),
        ("available_extended_virtual", ctypes.c_uint64),
    ]


_PROC_CGROUP = "/proc/self/cgroup"
_CGROUP_MOUNT = "/sys/fs/cgroup"


def _physical_memory() -> int:
    if sys.platform == "win32":
        status = _MemoryStatus()
        status.length = ctypes.sizeof(status)
        query = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
        query.argtypes = (ctypes.POINTER(_MemoryStatus),)
        query.restype = ctypes.c_int
        if not query(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return status.total_physical
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def _cgroup_memory_limit() -> int | None:
    """The tightest memory limit on the process's cgroup or any ancestor, v2 or v1; None if unlimited.

    A container or a systemd service can be far below the host's RAM, and its
    limit lives at the process's own cgroup path, not at the hierarchy root.
    The hierarchy is assumed to be mounted at the standard ``/sys/fs/cgroup``;
    set ``max_pinned_bytes`` explicitly on a host that mounts it elsewhere.
    """
    try:
        with open(_PROC_CGROUP, encoding="ascii") as membership:
            lines = membership.read().splitlines()
    except OSError:
        return None
    limits: list[int] = []
    for line in lines:
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hierarchy == "0":
            base, name = Path(_CGROUP_MOUNT), "memory.max"
        elif "memory" in controllers.split(","):
            base, name = Path(_CGROUP_MOUNT, "memory"), "memory.limit_in_bytes"
        else:
            continue
        directory = base.joinpath(*path.strip("/").split("/")) if path.strip("/") else base
        while True:
            try:
                text = (directory / name).read_text(encoding="ascii").strip()
            except OSError:
                text = ""
            if text.isdigit():
                limits.append(int(text))
            if directory == base or base not in directory.parents:
                break
            directory = directory.parent
    return min(limits, default=None)


def _default_pin_budget() -> int:
    """Half of the memory available to the process, rounded down to OS pages, without touching CUDA."""
    try:
        total = _physical_memory()
        if total <= 0:
            raise ValueError("physical RAM is unavailable")
        limit = _cgroup_memory_limit()
        if limit is not None:
            total = min(total, limit)
        return total // (2 * mmap.PAGESIZE) * mmap.PAGESIZE
    except (AttributeError, OSError, ValueError) as error:
        logger.warning("Cannot determine physical RAM; the default host pin budget is zero: %s", error)
        return 0


_DEFAULT_PIN_BUDGET = _default_pin_budget()


@dataclass(frozen=True, slots=True)
class PinStats:
    """Registration counts and the union of reserved OS pages.

    ``copy_bytes`` is the part of ``pinned_bytes`` held by owned copies of
    checkpoint storage, filling or pinned; the rest is storage pinned in
    place. ``offered_bytes`` counts the evicted copies offered to Windows,
    whole pages that stay committed but are neither pinned nor in the working
    set, under their own budget, ``max_offered_bytes``.
    """

    max_pinned_bytes: int | None
    pinned_bytes: int
    registrations: int
    idle_registrations: int
    active_leases: int
    registration_failures: int
    unregistration_failures: int
    copy_bytes: int
    max_offered_bytes: int
    offered_bytes: int


def _page_rounded(nbytes: int) -> int:
    return -(-nbytes // mmap.PAGESIZE) * mmap.PAGESIZE


def _new_region(size: int) -> mmap.mmap | VirtualRegion:
    """A copy's memory: a ``VirtualAlloc`` region on Windows, which can be offered; an anonymous map elsewhere."""
    if sys.platform == "win32":
        return VirtualRegion(size)
    return mmap.mmap(-1, size)


class _Copy:
    """An owned page-aligned region holding one checkpoint storage's bytes.

    Page alignment keeps two copies from sharing an OS page, which the
    runtime would refuse to register twice. ``size`` is the registered and
    reserved extent, whole pages.
    """

    __slots__ = ("region", "size", "storage")

    def __init__(self, size: int) -> None:
        assert size % mmap.PAGESIZE == 0
        self.size = size
        self.region = _new_region(size)
        # None once freed or offered.
        self.storage: torch.UntypedStorage | None = None
        self._expose()

    def _expose(self) -> None:
        # Built over a memoryview, so the storage and every view of it hold an
        # export of the region, which cannot close or be offered under them.
        self.storage = torch.frombuffer(memoryview(self.region), dtype=torch.uint8).untyped_storage()

    @property
    def pointer(self) -> int:
        assert self.storage is not None
        return self.storage.data_ptr()

    @property
    def offerable(self) -> bool:
        return isinstance(self.region, VirtualRegion)

    def offer(self) -> None:
        """Drop the storage and offer the pages to Windows; ``BufferError`` if a view of them is still alive."""
        assert isinstance(self.region, VirtualRegion)
        self.storage = None
        self.region.offer()

    def reclaim(self) -> list[tuple[int, int]]:
        """Take the offered pages back and expose them again, returning the spans that need refilling."""
        assert isinstance(self.region, VirtualRegion)
        discarded = self.region.reclaim()
        self._expose()
        intact = not discarded
        if intact:
            # Here rather than in the pass before registration, so that one
            # worker's touch overlaps another's reclaim; a copy the fill has
            # to rebuild is touched after it, once it holds its bytes.
            self.touch()
        return discarded

    def touch(self) -> None:
        """Write one byte of every page back over itself, which brings the pages into the working set.

        Pages a copy holds can be outside the working set whether they were
        just reclaimed, which leaves them present but not in it, or filled
        long enough ago that memory pressure has trimmed them again.
        ``cudaHostRegister`` faults such pages in one at a time, at 8 GiB/s,
        where this write brings them back at 42 GiB/s and leaves registration
        running at 55. It leaves the bytes as they were.
        """
        with memoryview(self.region) as view:
            _fault_in(view, 0, self.size)

    def view(self, tensor: torch.Tensor) -> torch.Tensor:
        """``tensor``'s geometry over the copy, including its lazy conjugation and negation."""
        assert self.storage is not None
        view = torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
            self.storage, tensor.storage_offset(), tensor.shape, tensor.stride(),
        )
        if tensor.is_conj():
            view = view.conj()
        if tensor.is_neg():
            view = torch._neg_view(view)
        return view

    def free(self) -> None:
        self.storage = None
        try:
            self.region.close()
        except BufferError:
            # A transfer view outlived its lease; the region returns with it.
            logger.warning("An evicted pinned copy is still referenced; its memory returns when the reference dies")


_FILL_SLICE = 64 * 2**20
# Reclaiming pages is work behind the kernel's address-space lock, so it stops
# scaling well before the core count and then reverses: 6.9 GiB of copies come
# back in 2.73 s on one thread, 0.97 s on four, 0.85 s on five, 0.94 s on
# eight and 1.57 s on sixteen, and under memory pressure an 18 GiB checkpoint's
# copies take 1.25 s on six against 3.5 s on sixteen. This is why the number is
# small where the fill uses one thread per core.
_RECLAIM_WORKERS = 6
# How long the thread watching commitment for the offered tier waits on the
# kernel's condition before it checks whether the tier still holds copies.
_COMMIT_WATCH_SECONDS = 0.5


def _preadv(fd: int, buffer: memoryview, offset: int) -> int:
    return os.preadv(fd, [buffer], offset)


class _Readers:
    """Positional reads of the checkpoints a fill needs, one way per platform.

    ``preadv`` reads at an offset without moving the descriptor's position, so
    every worker shares the reader's own descriptor. A Windows handle has one
    position and the kernel serializes concurrent reads on it, so a worker
    reads through a handle of its own, opened from the file's path; the
    handles close when the fill is over. Provenance names an open file on
    disk, which is what ``MappedCheckpoint`` holds.
    """

    def __init__(self) -> None:
        self._local = threading.local()
        self._opened: list[io.FileIO] = []
        self._lock = threading.Lock()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self._opened:
            handle.close()

    def read_at(self, file: io.BufferedReader) -> Callable[[memoryview, int], int]:
        """A positional read of ``file`` that any worker may call."""
        if hasattr(os, "preadv"):
            return functools.partial(_preadv, file.fileno())
        return functools.partial(self._read_alone, file)

    def _read_alone(self, file: io.BufferedReader, buffer: memoryview, offset: int) -> int:
        handles: dict[int, io.FileIO] = getattr(self._local, "handles", None) or {}
        self._local.handles = handles
        handle = handles.get(id(file))
        if handle is None:
            handle = open(file.name, "rb", buffering=0)  # noqa: SIM115 - closed with the fill's readers
            handles[id(file)] = handle
            with self._lock:
                self._opened.append(handle)
        handle.seek(offset)
        return handle.readinto(buffer)


def _read_range(
    read_at: Callable[[memoryview, int], int], view: memoryview, offset: int, start: int, stop: int,
) -> None:
    """Fill ``view[start:stop]`` from ``offset + start`` in the file, looping over short reads."""
    done = start
    while done < stop:
        count = read_at(view[done:stop], offset + done)
        if count <= 0:
            raise OSError(f"checkpoint ended at byte {offset + done}")
        done += count


def _fault_in(view: memoryview, start: int, stop: int) -> None:
    """Write one byte of every page of ``view[start:stop]`` back over itself, faulting the pages in."""
    with view[start:stop] as window:
        torch.frombuffer(window, dtype=torch.uint8)[:: mmap.PAGESIZE].bitwise_or_(0)


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
    _fault_in(view, start, stop)
    with reading_below_offers():
        _read_range(read_at, view, offset, start, stop)


@dataclass(eq=False, slots=True)
class _InPlace:
    """In-place storage reserved under the budget for one request, registered with the rest of its acquisition."""

    pointer: int
    request: _Request


@dataclass(eq=False, slots=True)
class _PendingCopy:
    """A copy reserved for one request, filled and registered with the rest of its acquisition.

    ``offered`` marks a copy whose reclaim worker has not yet been joined;
    every reclaim finishes before anything is filled. ``missing`` is what the
    fill has to read of the copy, as offsets and lengths: all of a new or rebuilt one, the spans
    Windows discarded of a reclaimed one, and nothing of one it kept intact
    or whose fill completed successfully.
    """

    pointer: int
    request: _Request
    source: FileSlice
    copy: _Copy
    missing: list[tuple[int, int]]
    offered: bool = False
    error: OSError | None = None

    def spans(self) -> list[tuple[int, int]]:
        """``missing`` within the storage's bytes, as start and stop offsets."""
        length = self.source.length
        return [(start, min(start + covered, length)) for start, covered in self.missing if start < length]


type _Pending = _InPlace | _PendingCopy


def _fill_copies(pending: list[_PendingCopy], *, below_offers: bool = False) -> None:
    """Fill every pending copy from its file; a read failure marks that copy's ``error``.

    Positional reads over fixed-size slices of every copy run together on
    one thread per logical core, hyperthreads included: the fill is bound
    by page faults on the fresh regions, which overlap across threads, and
    a checkpoint's tensors are mostly smaller than one slice. Each worker
    reads through the reader that suits the platform (:class:`_Readers`),
    bringing the file into the cache below offered copies if
    ``below_offers``. Runs without the manager's lock: the workers run Python
    code, so a garbage collection on one of them may run a finalizer that
    needs that lock.
    """
    workers = os.process_cpu_count() or 1
    # At the normal memory priority the file cache outranks offered pages, so
    # while copies can be offered the reads bring the file in below them. With
    # 21 GiB of RAM available, reloading the 18 GiB H3 checkpoint after the
    # 6.9 GiB Hunyuan 3D one kept 12.4 to 12.7 GiB of its offered copies and
    # took 13.1 to 13.7 s this way, against 4.1 GiB and 28.2 s at the normal
    # priority. That holds from the first load, before anything is offered: at
    # the normal priority it left the file cached above the copies offered
    # later, and at full RAM every switch between the two then found its
    # offered copies discarded, 10.1 to 10.4 s against 0.87 to 0.95 s. Reading
    # around the cache kept as much, 12.3 to 12.7 GiB in 12.8 to 13.6 s, but
    # read a file the cache already held from disk: a first pin of H3 took
    # 35.7 s that way against 2.3 to 2.4 s.
    read = _read_range_below_offers if below_offers else _read_range
    # The pool closes before the views, so a failure waits for the queued
    # slices to finish, and the readers close after all of them.
    with (
        _Readers() as readers,
        contextlib.ExitStack() as views,
        ThreadPoolExecutor(workers, thread_name_prefix="piper-offload-fill") as pool,
    ):
        futures: list[tuple[_PendingCopy, Future[None]]] = []
        for item in pending:
            view = views.enter_context(memoryview(item.copy.region)[: item.source.length])
            read_at = readers.read_at(item.source.file)
            for first, last in item.spans():
                for start in range(first, last, _FILL_SLICE):
                    stop = min(start + _FILL_SLICE, last)
                    futures.append((item, pool.submit(read, read_at, view, item.source.offset, start, stop)))
        for item, future in futures:
            try:
                future.result()
            except OSError as error:
                item.error = _detached(error)
    for item in pending:
        if item.error is None:
            item.missing.clear()


def _on_workers(action: Callable[[_Copy], None], copies: list[_Copy], workers: int, name: str) -> None:
    """Run ``action`` on every copy on up to ``workers`` threads named ``name``, or inline for one copy.

    Runs without the manager's lock: the workers run Python code, so a
    garbage collection on one of them may run a finalizer that needs it.
    """
    if len(copies) < 2:
        for copy in copies:
            action(copy)
        return
    with ThreadPoolExecutor(min(workers, len(copies)), thread_name_prefix=name) as pool:
        for future in [pool.submit(action, copy) for copy in copies]:
            future.result()


def _touch_copies(pending: list[_PendingCopy]) -> None:
    """Bring the copies a fill has just written into the working set, on one thread per core.

    A long fill leaves the copies it wrote first outside the working set by
    the time the last one is read, and pages outside it are where
    ``cudaHostRegister`` faults them in one at a time at 8 GiB/s: writing a
    byte to each brings them back at 42 GiB/s and leaves registration running
    at 55, so the pair costs 60 ms/GiB against the 130 that registering them
    cold does.
    """
    ready = [item.copy for item in pending if item.error is None]
    _on_workers(operator.methodcaller("touch"), ready, os.process_cpu_count() or 1, "piper-offload-touch")


def _offer_copy(copy: _Copy) -> str | None:
    """Offer one copy's pages: None, or why they could not be offered, as text that holds no traceback."""
    try:
        copy.offer()
    except (BufferError, OSError) as error:
        return str(error)
    return None


class _OfferBatch:
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

    def __init__(self) -> None:
        self._futures: list[tuple[_Registration, Future[str | None]]] = []
        self._pool: ThreadPoolExecutor | None = None

    def add(self, registration: _Registration) -> None:
        """Start offering a copy whose unregistration has just succeeded; called with the lock held."""
        assert registration.copy is not None
        if self._pool is None:
            workers = os.process_cpu_count() or 1
            self._pool = ThreadPoolExecutor(workers, thread_name_prefix="piper-offload-offer")
        self._futures.append((registration, self._pool.submit(_offer_copy, registration.copy)))

    def take(self) -> list[tuple[_Registration, str | None]]:
        """Empty the batch and wait for its offers, returning each copy and why it could not be offered."""
        futures, self._futures = self._futures, []
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown()
        return [(registration, future.result()) for registration, future in futures]


def _reclaim_copy(item: _PendingCopy) -> None:
    """Take one offered copy back: intact, discarded in part and left to the fill, or rebuilt where Windows refuses.

    A rebuilt copy keeps ``missing`` as it was reserved, the whole copy.
    """
    try:
        item.missing = item.copy.reclaim()
    except OSError as error:
        # Tracebacks in buffered logs would hold the region this frees.
        logger.warning("Could not reclaim an offered pinned copy; rebuilding it: %s", str(error))
        item.copy.free()
        try:
            item.copy = _Copy(item.copy.size)
        except (OSError, MemoryError) as failure:
            item.error = OSError(f"could not allocate its replacement: {failure}")


def _reclaimed(pending: list[_PendingCopy]) -> Generator[_PendingCopy]:
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


def _watch_commit(manager_ref: weakref.ReferenceType[PinManager]) -> None:
    """Free a manager's offered copies once commitment is exhausted; return once its tier is empty or it is gone."""
    while True:
        commit_exhausted(_COMMIT_WATCH_SECONDS)  # returns as soon as it is, at the latest after the interval
        manager = manager_ref()
        if manager is None or not manager._keep_commit():
            return
        del manager


def _detached(error: OSError) -> OSError:
    """The error without its traceback, whose frames would hold a slice of the copy and keep it mapped."""
    return error.with_traceback(None)


class _Refusal(enum.Enum):
    STORAGE = "storage"  # this storage stays pageable; a later one in the acquisition may register
    CAPACITY = "capacity"  # the runtime's capacity is exhausted; stop registering


@dataclass(eq=False)
class _Registration:
    pointer: int
    size: int
    storage: torch.UntypedStorage
    copy: _Copy | None = None
    owners: dict[int, weakref.ReferenceType[torch.Tensor]] = field(default_factory=dict)
    leases: int = 0
    retired: bool = False

    @property
    def registered(self) -> int:
        """The pointer the backend registered: the copy's, or the storage's own."""
        return self.pointer if self.copy is None else self.copy.pointer


@dataclass(slots=True)
class _Request:
    storage: torch.UntypedStorage
    tensors: list[torch.Tensor]


@dataclass(slots=True)
class _Pageable:
    size: int
    leases: int = 0


@dataclass(slots=True)
class _LeaseState:
    registrations: tuple[_Registration, ...]
    pageable: tuple[int, ...]
    tensors: tuple[torch.Tensor, ...]


class PinLease:
    """Hold storage, pinned or pageable, until closed.

    ``registered_bytes`` and ``pageable_bytes`` count unique requested storage
    bytes, without page rounding. The holder must keep the lease open until no
    asynchronous operation can read or write its host tensors. CUDA ordering
    belongs to the runtime that enqueues those operations; the pin manager
    does not synchronize accelerator streams. Dropping the lease closes it, so
    whoever runs asynchronous work must retain it through completion.
    """

    def __init__(
        self,
        manager: PinManager,
        key: int,
        registered_bytes: int,
        pageable_bytes: int,
    ) -> None:
        self.registered_bytes = registered_bytes
        self.pageable_bytes = pageable_bytes
        self._finalizer = weakref.finalize(self, manager._close_lease, key)
        self._finalizer.atexit = False

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def close(self) -> None:
        """Close the lease, releasing what it holds; idempotent."""
        self._finalizer()

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("Pin lease is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class TransferLease:
    """A pageable lease over one transfer, closed only once its device has synchronized.

    ``start`` leases the transfer's storage without registering it, and
    ``finish`` synchronizes the device the transfer used before closing the
    lease. A synchronization that fails leaves the lease open, so the storage
    stays held until a later ``finish`` succeeds, and no new transfer can
    start on this token until then.
    """

    __slots__ = ("_device", "_lease")

    def __init__(self) -> None:
        self._lease: PinLease | None = None
        self._device: torch.device | None = None

    @property
    def open(self) -> bool:
        return self._lease is not None

    def start(self, manager: PinManager, tensors: Iterable[torch.Tensor], device: torch.device) -> None:
        if self._lease is not None:
            raise RuntimeError(
                "A host transfer is still unfinished; release the component so its "
                "synchronization can be retried before starting another."
            )
        self._lease = manager.acquire(tensors, pin=False)
        self._device = device

    def finish(self) -> None:
        """Synchronize the transfer's device, then close the lease; idempotent."""
        if self._lease is None:
            return
        torch.cuda.synchronize(self._device)
        self.close()

    def close(self) -> None:
        """Close once the owner has observed the transfer complete; idempotent.

        A runtime that synchronizes the transfer's own stream reaches that
        point without a second, device-wide synchronization.
        """
        if self._lease is None:
            return
        self._lease.close()
        self._lease = None
        self._device = None


# Unmapping is page-table teardown, and the kernel's address-space lock lets
# only a few threads through at once: freeing 32 GiB of copies measures 1.79 s
# on one thread, 0.64 s on four, and no better beyond eight. This is why the
# number is small where ``_fill_copies`` uses one thread per core.
_FREE_WORKERS = 4


def _free_copies(copies: list[_Copy]) -> None:
    """Free owned regions together, returning their memory to the OS."""
    _on_workers(operator.methodcaller("free"), copies, _FREE_WORKERS, "piper-offload-free")


class PinManager:
    """Own registrations under a page-rounded budget.

    A finite ``max_pinned_bytes`` bounds pinned pages in this process,
    including the owned copies made for checkpoint storage; the default is
    half of physical RAM at import. ``None`` treats the CUDA/HIP runtime's
    capacity as the budget, evicting unrelated idle registrations when the
    runtime refuses a new registration.

    Acquire accepts the plain CPU tensors returned by ``storage_tensors()``.
    Tensor views share one whole-storage registration. Separate storages
    may share OS pages, which are reserved once. Distinct overlapping byte
    ranges (for example separate ``frombuffer`` wrappers) are
    rejected before registration; registering only part of a copy's range can
    make the CUDA/HIP copy invalid. Use views of one storage for such aliases.

    All metadata and backend operations are serialized by a reentrant lock.
    An active lease holds its pageable storage as well. Idle registrations
    retain storage, but no model or tensor wrappers. Losing an owner retires
    its registration as soon as the leases holding it close. Pageable storage
    remains pageable until every lease holding it closes.

    Deactivation keeps copies pinned in the idle LRU, and evicting one frees
    it. With a positive ``max_offered_bytes`` on Windows, an eviction that
    makes room, for the pin budget or after a runtime capacity refusal,
    offers the copy's pages to the OS instead, once its unregistration has
    succeeded; see ``max_offered_bytes``. Elsewhere copies are anonymous
    mappings, which are always freed.
    """

    def __init__(
        self,
        max_pinned_bytes: int | None = _DEFAULT_PIN_BUDGET,
        *,
        max_offered_bytes: int = 0,
        backend: HostRegistrationBackend | None = None,
    ) -> None:
        if max_pinned_bytes is not None and max_pinned_bytes < 0:
            raise ValueError("max_pinned_bytes must be >= 0")
        if max_offered_bytes < 0:
            raise ValueError("max_offered_bytes must be >= 0")
        self._max_pinned_bytes = max_pinned_bytes
        self._max_offered_bytes = max_offered_bytes
        self._backend = backend if backend is not None else RuntimeHostRegistration()
        self._lock = threading.RLock()
        self._registrations: dict[int, _Registration] = {}
        self._pageable: dict[int, _Pageable] = {}
        # Storage reserved by an acquisition that has not registered it yet;
        # an acquisition of the same storage waits until it registers or is evicted.
        self._pending: dict[int, _Pending] = {}
        self._pending_changed = threading.Condition(self._lock)
        # All pinned, pending, and leased pageable ranges.
        self._starts: list[int] = []
        self._idle: OrderedDict[int, None] = OrderedDict()
        # Disjoint byte ranges can share only their boundary pages. Counting
        # only those endpoints avoids one dictionary item per page of a large model.
        self._boundary_pages: dict[int, int] = {}
        self._pinned_bytes = 0
        self._registration_failures = 0
        self._unregistration_failures = 0
        self._leases: dict[int, _LeaseState] = {}
        self._next_lease = 0
        # Unregistered copies whose pages are offered to Windows, keyed like
        # registrations and least recently offered first; only the manager
        # holds their regions, and only a reclaim makes them readable again.
        self._offered: OrderedDict[int, _Registration] = OrderedDict()
        self._offered_bytes = 0
        # The copies evictions offer while an acquisition or a budget change
        # holds the lock, which are offered as they come and joined once it is released.
        self._offering: _OfferBatch | None = None
        # Whether a thread is watching commitment while the tier holds copies.
        self._watching_commit = False

    @property
    def max_pinned_bytes(self) -> int | None:
        with self._lock:
            return self._max_pinned_bytes

    @max_pinned_bytes.setter
    def max_pinned_bytes(self, value: int | None) -> None:
        """Set the budget, or ``None`` to register up to the runtime's capacity.

        Runtime capacity failures still evict unrelated idle registrations
        before falling back to pageable storage. A finite budget evicts idle
        registrations down to it now, and what active leases hold above it
        as they release. Failed unregistrations stay reserved and can be
        retried with ``clear()`` or later budget pressure.
        """
        if value is not None and value < 0:
            raise ValueError("max_pinned_bytes must be >= 0")
        offers = _OfferBatch()
        try:
            with self._lock, self._offers_deferred(offers):
                self._max_pinned_bytes = value
                self._fit(lambda: 0)
        finally:
            self._offer_together(offers)

    @property
    def max_offered_bytes(self) -> int:
        with self._lock:
            return self._max_offered_bytes

    @max_offered_bytes.setter
    def max_offered_bytes(self, value: int) -> None:
        """Set the budget for evicted copies offered to Windows; zero, the default, frees every evicted copy.

        An offered copy is unregistered, and Windows may discard its pages
        under memory pressure, but they stay committed, so this budget bounds
        commitment rather than RAM and is separate from ``max_pinned_bytes``.
        A copy is offered when an eviction makes room for the pin budget, a
        budget reduction, or a runtime capacity retry, after its unregistration
        succeeds, if it fits this budget and nothing outside the manager still
        views it; the least recently offered copies are freed to make room.
        Otherwise it is freed, as it is on ``clear()`` and when its owner is
        dropped. A pinning acquisition of the same storage reclaims the copy,
        skipping the fill if Windows kept its pages and refilling them in full
        if not, and registers it before any transfer can read it.

        Offered copies keep their commitment, which is what allocations
        elsewhere run short of once Windows can no longer grow its paging
        files. So when Windows reports that condition, every offered copy is
        freed: a thread waits for it while the tier holds copies, and the tier
        checks it before taking a copy, which it then frees instead, and
        before a new copy is allocated. A failed allocation of a new copy
        frees offered copies, least recent first, and retries once. Lowering
        the budget frees offered copies down to it now. The copies an acquisition
        or a budget change evicts are offered together once it has released
        the lock, and are counted in ``offered_bytes`` from then on. On other
        platforms the budget has no effect.
        """
        if value < 0:
            raise ValueError("max_offered_bytes must be >= 0")
        with self._lock:
            self._max_offered_bytes = value
            released = self._evict_offered(value)
        _free_copies(released)

    @property
    def stats(self) -> PinStats:
        with self._lock:
            return PinStats(
                self._max_pinned_bytes,
                self._pinned_bytes,
                len(self._registrations),
                len(self._idle),
                len(self._leases),
                self._registration_failures,
                self._unregistration_failures,
                sum(entry.copy.size for entry in self._registrations.values() if entry.copy is not None)
                + sum(item.copy.size for item in self._pending.values() if isinstance(item, _PendingCopy)),
                self._max_offered_bytes,
                self._offered_bytes,
            )

    def acquire(self, tensors: Iterable[torch.Tensor], *, pin: bool = True) -> PinLease:
        """Lease whole storages for a transfer, registering them only if ``pin``.

        A lease holds its storage until it closes: a registration it holds is
        not evicted, and pageable storage it holds is not registered. With
        ``pin`` the lease also registers what the budget allows, for storage
        that repeats every step; without it nothing new is registered, for a
        transfer that runs once, so leasing never changes which storage is
        pinned.

        A pinning acquisition waits while another has pending storage, and a
        pageable one waits only for its own storage, so pinning acquisitions
        run in turn as they did under the lock while pageable leases and
        transfers proceed. All input validation happens before registration
        or eviction. Existing registrations anywhere in the request are held
        before new storage is reserved, so reserving cannot evict what this
        same lease will use. New storage is reserved under the budget in
        request order, with a copy allocated for checkpoint storage, or taken
        from the offered tier when the storage has one there; the copies are
        reclaimed and fill with the lock released, the first alone and the
        rest together once it has registered, and a reclaimed copy whose pages
        Windows kept intact is not read and registers as soon as preceding
        requests are settled, while later copies can still be reclaiming.
        Registration always follows request order. A runtime capacity failure
        evicts unrelated idle registrations and retries; if capacity remains unavailable, later
        storage in this acquisition skips registration. A rejected range alone
        stays pageable, without evicting other registrations or skipping later
        storage.
        """
        requests = self._requests(tensors)
        held: dict[int, _Registration] = {}
        created: list[_Registration] = []
        pending: list[_Pending] = []
        offers = _OfferBatch()
        try:
            with self._lock:
                while self._pending and (pin or not self._pending.keys().isdisjoint(requests)):
                    self._pending_changed.wait()
                self._validate_ranges(requests)
                for pointer, request in requests.items():
                    registration = self._registrations.get(pointer)
                    if registration is not None:
                        self._hold(registration, request, held)
                if pin:
                    with self._offers_deferred(offers):
                        self._reserve(requests, held, pending)
                copies = [item for item in pending if isinstance(item, _PendingCopy)]
                # Copies the tier can take fill below the ones it holds (_read_range_below_offers).
                below_offers = self._max_offered_bytes > 0 and any(item.copy.offerable for item in copies)
            # What was evicted to make room is offered before anything is read.
            self._offer_together(offers)
            exhausted = bool(copies) and self._reclaim_and_fill(copies, pending, held, created, below_offers)
            with self._lock:
                if not exhausted:
                    self._register_pending(pending, held, created)
                self._discard_pending(pending)
                return self._open_lease(requests, held)
        except BaseException:
            with self._lock:
                self._discard_pending(pending)
                for registration in created:
                    registration.retired = True
                self._release(tuple(held.values()))
            self._offer_together(offers)
            raise

    def _reclaim_and_fill(
        self,
        copies: list[_PendingCopy],
        pending: list[_Pending],
        held: dict[int, _Registration],
        created: list[_Registration],
        below_offers: bool,
    ) -> bool:
        """Reclaim and fill an acquisition's copies with the lock released; whether the runtime's capacity ran out.

        Copies Windows returned intact register as each comes back, once
        preceding requests are settled, because registering locks their pages
        where the fill of the rest would otherwise push them out of the working set
        again: reacquiring an 18 GiB checkpoint whose fill was 6 GiB took
        19.4 s with those pages registered after the fill and 14.1 s before
        it, the registration itself 5.3 s against 0.5. It also finds a runtime
        out of capacity before any read. Otherwise the first copy fills and
        registers alone, for the same early answer, and the rest fill together.
        """
        with contextlib.closing(_reclaimed(copies)) as reclaimed:
            for item in reclaimed:
                # Publish readiness only after this copy's worker has been joined.
                item.offered = False
                with self._lock:
                    if self._register_pending(pending, held, created):
                        return True  # the rest can only be evicted, so their reclaims are cancelled
        unfilled = [item for item in copies if item.missing and item.error is None]
        if unfilled:
            _fill_copies(unfilled[:1], below_offers=below_offers)
            _touch_copies(unfilled[:1])
            with self._lock:
                if self._register_pending(pending, held, created):
                    return True
            if len(unfilled) > 1:
                _fill_copies(unfilled[1:], below_offers=below_offers)
                _touch_copies(unfilled[1:])
        return False

    def _open_lease(self, requests: dict[int, _Request], held: dict[int, _Registration]) -> PinLease:
        key = self._next_lease
        self._next_lease += 1
        pageable_pointers = tuple(pointer for pointer in requests if pointer not in held)
        for pointer in pageable_pointers:
            pageable = self._pageable.get(pointer)
            if pageable is None:
                pageable = _Pageable(requests[pointer].storage.nbytes())
                self._pageable[pointer] = pageable
                self._starts.insert(bisect_left(self._starts, pointer), pointer)
            pageable.leases += 1
        self._leases[key] = _LeaseState(
            tuple(held.values()),
            pageable_pointers,
            tuple(tensor for request in requests.values() for tensor in request.tensors),
        )
        _live_managers.add(self)
        registered = sum(registration.size for registration in held.values())
        total = sum(request.storage.nbytes() for request in requests.values())
        return PinLease(self, key, registered, total - registered)

    def transfer(self, destination: torch.Tensor, source: torch.Tensor, *, non_blocking: bool) -> None:
        """Copy ``source`` into ``destination``, reading ``source``'s pinned copy when one exists.

        A synchronous transfer has completed when this returns and runs under
        the manager's lock, so nothing can evict its source meanwhile. An
        asynchronous one, a non-blocking copy to a CUDA device, may still be
        in flight afterwards, so it must run under a lease that holds the
        source, and raises otherwise: eviction is safe only because every
        such reader holds one. Pageable storage copies as is.
        """
        if source.device.type != "cpu":
            destination.copy_(source, non_blocking=non_blocking)
            return
        asynchronous = non_blocking and destination.device.type == "cuda"
        with self._lock:
            registration = self._registrations.get(source.untyped_storage().data_ptr())
            view = source
            if registration is not None:
                if asynchronous and registration.leases == 0:
                    raise RuntimeError(
                        "Asynchronous transfer of pinned host storage outside a lease; acquire one "
                        "over the tensors first (pin=False for a one-time transfer) and keep it "
                        "until the transfer has completed."
                    )
                if registration.copy is not None:
                    view = registration.copy.view(source)
            if not asynchronous:
                destination.copy_(view, non_blocking=non_blocking)
                return
        # The lease keeps the registration, so the copy itself needs no lock.
        destination.copy_(view, non_blocking=True)

    def clear(self) -> None:
        """Unregister idle registrations and free their copies, and free every offered copy.

        What active leases hold stays. A failed unregistration retains its
        storage and budget reservation; cleanup errors propagate so callers can
        retry without losing ownership of pinned memory.

        The copies are unregistered under the lock and freed together once it is
        released, so an acquisition racing this call can see the budget before
        the memory is back.
        """
        released: list[_Copy] = []
        with self._lock:
            failed = 0
            for pointer in tuple(self._idle):
                registration = self._registrations.get(pointer)
                if registration is not None and not self._unregister(registration, released):
                    failed += 1
            released.extend(self._evict_offered(0))
        _free_copies(released)
        if failed:
            raise RuntimeError(f"Could not release {failed} host registration(s); storage remains retained")

    @staticmethod
    def _requests(tensors: Iterable[torch.Tensor]) -> dict[int, _Request]:
        requests: dict[int, _Request] = {}
        seen: set[int] = set()
        for tensor in tensors:
            if type(tensor) is not torch.Tensor:
                raise TypeError("PinManager requires plain CPU storage tensors")
            if tensor.device.type != "cpu" or tensor.layout is not torch.strided:
                raise ValueError("PinManager requires strided CPU storage tensors")
            if id(tensor) in seen or tensor.numel() == 0:
                continue
            seen.add(id(tensor))
            storage = tensor.untyped_storage()
            pointer = storage.data_ptr()
            request = requests.get(pointer)
            if request is None:
                requests[pointer] = _Request(storage, [tensor])
            elif request.storage.nbytes() != storage.nbytes():
                raise ValueError("Overlapping host storage ranges must use views of one storage")
            else:
                request.tensors.append(tensor)
        return requests

    def _validate_ranges(self, requests: dict[int, _Request]) -> None:
        prior_end = 0
        for pointer in sorted(requests):
            end = pointer + requests[pointer].storage.nbytes()
            index = bisect_left(self._starts, pointer)
            neighbors = self._starts[max(0, index - 1):index + 1]
            if pointer < prior_end:
                raise ValueError("Overlapping host storage ranges must use views of one storage")
            for other in neighbors:
                other_end = other + self._range_size(other)
                if pointer < other_end and other < end and (pointer != other or end != other_end):
                    raise ValueError("Overlapping host storage ranges must use views of one storage")
            prior_end = end

    def _range_size(self, pointer: int) -> int:
        """The extent of a range in ``_starts``: pinned, leased pageable, or pending."""
        registration = self._registrations.get(pointer)
        if registration is not None:
            return registration.size
        pageable = self._pageable.get(pointer)
        if pageable is not None:
            return pageable.size
        return self._pending[pointer].request.storage.nbytes()

    @staticmethod
    def _boundaries(pointer: int, size: int) -> tuple[int, ...]:
        first = pointer // mmap.PAGESIZE
        last = (pointer + size - 1) // mmap.PAGESIZE
        return (first,) if first == last else (first, last)

    def _page_reservation(self, pointer: int, size: int) -> int:
        if size == 0:
            return 0
        pages = (pointer + size - 1) // mmap.PAGESIZE - pointer // mmap.PAGESIZE + 1
        shared = sum(page in self._boundary_pages for page in self._boundaries(pointer, size))
        return (pages - shared) * mmap.PAGESIZE

    def _reserve(self, requests: dict[int, _Request], held: dict[int, _Registration], pending: list[_Pending]) -> None:
        """Reserve every unheld storage the budget allows, in request order, into ``pending``.

        In-place storage reserves its pages; checkpoint storage gets a
        reserved copy. Each stays in ``self._pending``, and its range in
        ``self._starts``, until it registers or is evicted.
        """
        for pointer, request in requests.items():
            if pointer in held or pointer in self._pageable:
                continue
            source = file_slice(request.storage)
            if source is None:
                item: _Pending | None = self._reserve_in_place(pointer, request)
            else:
                item = self._reserve_copy(pointer, request, source)
            if item is not None:
                pending.append(item)
                self._pending[pointer] = item
                self._starts.insert(bisect_left(self._starts, pointer), pointer)

    def _reserve_in_place(self, pointer: int, request: _Request) -> _InPlace | None:
        size = request.storage.nbytes()
        if not self._fit(lambda: self._page_reservation(pointer, size)):
            return None
        self._reserve_range(pointer, size)
        return _InPlace(pointer, request)

    def _reserve_copy(self, pointer: int, request: _Request, source: FileSlice) -> _PendingCopy | None:
        """Reserve a copy under the budget: the storage's offered copy if it has one, otherwise a new allocation."""
        offered = self._offered.get(pointer)
        if offered is None:
            copy = self._allocate_copy(request.storage.nbytes())
            return None if copy is None else _PendingCopy(pointer, request, source, copy, [(0, copy.size)])
        copy = self._take_offered(pointer)
        size = copy.size
        # Out of the tier before making room, so the evictions cannot free it.
        if not self._fit(lambda: size):
            # It stays offered, now the most recent. It still fits: the
            # evictions that made room went to this acquisition's offers.
            self._offered[pointer] = offered
            self._offered_bytes += size
            return None
        self._pinned_bytes += size
        return _PendingCopy(pointer, request, source, copy, [(0, size)], offered=True)

    def _allocate_copy(self, nbytes: int) -> _Copy | None:
        """Allocate and reserve a copy under the budget, or None if it does not fit or cannot be allocated."""
        size = _page_rounded(nbytes)
        if not self._fit(lambda: size):
            return None
        # The new copy's commitment may be what Windows cannot find; the tier's is the process's to give.
        for released in self._relieve_commit():
            released.free()
        try:
            copy = _Copy(size)
        except (OSError, MemoryError) as error:
            copy = self._allocate_after_freeing_offered(size) if self._offered else None
            if copy is None:
                logger.warning("Could not allocate a pinned copy; the checkpoint storage stays pageable: %s", error)
                return None
        # Reserved at allocation, before the fill, so nothing else is reserved
        # from the same budget while the read is in progress.
        self._pinned_bytes += size
        return copy

    def _allocate_after_freeing_offered(self, size: int) -> _Copy | None:
        """Free offered copies, least recent first, until ``size`` bytes of their commitment return, and retry.

        Offered pages stay committed, so a failed allocation is the commit
        pressure that has to free them.
        """
        for copy in self._evict_offered(self._offered_bytes - size):
            copy.free()
        try:
            return _Copy(size)
        except (OSError, MemoryError):
            return None

    @contextlib.contextmanager
    def _offers_deferred(self, batch: _OfferBatch) -> Iterator[None]:
        """Send the copies evictions offer to ``batch``, which offers them and which only its owner waits for."""
        self._offering = batch
        try:
            yield
        finally:
            self._offering = None

    def _offer_together(self, batch: _OfferBatch) -> None:
        """Wait for the offers a locked section started, then put each copy in the tier or free it.

        A copy is freed if Windows refused it, or if while it was being
        offered its owner was dropped, its storage got another copy, or the
        budget shrank below it.
        """
        taken = batch.take()
        if not taken:
            return
        released: list[_Copy] = []
        with self._lock:
            for registration, failure in taken:
                released.extend(self._settle_offer(registration, failure))
        _free_copies(released)

    def _settle_offer(self, registration: _Registration, failure: str | None) -> list[_Copy]:
        """Admit a completed offer and return the copies its caller must free, including any rejected offer."""
        copy = registration.copy
        assert copy is not None
        if failure is not None:
            logger.warning("Could not offer an evicted pinned copy; freeing it: %s", failure)
            return [copy]
        pointer = registration.pointer
        rebuilt = pointer in self._registrations or pointer in self._pending or pointer in self._offered
        if registration.retired or rebuilt or copy.size > self._max_offered_bytes:
            return [copy]
        released = self._evict_offered(self._max_offered_bytes - copy.size)
        self._offered[pointer] = registration
        self._offered_bytes += copy.size
        released.extend(self._relieve_commit())
        if self._offered and not self._watching_commit:
            self._watching_commit = True
            threading.Thread(
                target=_watch_commit, args=(weakref.ref(self),), name="piper-offload-commit", daemon=True,
            ).start()
        return released

    def _relieve_commit(self) -> list[_Copy]:
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
        _free_copies(released)
        return watching

    def _evict_offered(self, target: int) -> list[_Copy]:
        """Detach least recently offered copies down to ``target`` bytes; the caller frees their regions."""
        released: list[_Copy] = []
        while self._offered and self._offered_bytes > target:
            released.append(self._take_offered(next(iter(self._offered))))
        return released

    def _take_offered(self, pointer: int) -> _Copy:
        """Remove a copy and its charge from the tier, transferring ownership to the caller."""
        copy = self._offered.pop(pointer).copy
        assert copy is not None
        self._offered_bytes -= copy.size
        return copy

    def _register_pending(
        self,
        pending: list[_Pending],
        held: dict[int, _Registration],
        created: list[_Registration],
    ) -> bool:
        """Register the ready prefix in request order; return whether runtime capacity is exhausted.

        Refused storage is evicted. Capacity exhaustion stops the pass; the
        caller joins workers before discarding the remaining pending storage.
        Only the acquisition thread marks a copy no longer offered after
        joining its reclaim worker, so registration never races a worker.
        """
        exhausted = False
        while pending:
            item = pending[0]
            if isinstance(item, _PendingCopy) and (
                item.offered or (item.missing and item.error is None)
            ):
                break
            try:
                outcome = self._register_pending_item(item)
            except HostRegistrationRefusedError:
                outcome = _Refusal.STORAGE
            if isinstance(outcome, _Registration):
                _live_managers.add(self)
                self._registrations[item.pointer] = outcome
                created.append(outcome)
                self._hold(outcome, item.request, held)
            else:
                self._evict_pending(item)
                exhausted = outcome is _Refusal.CAPACITY
            del self._pending[item.pointer]
            del pending[0]
            if exhausted:
                break
        self._pending_changed.notify_all()
        return exhausted

    def _discard_pending(self, pending: list[_Pending]) -> None:
        """Release remaining reservations once all acquisition workers have stopped."""
        for item in pending:
            self._evict_pending(item)
            del self._pending[item.pointer]
        pending.clear()
        self._pending_changed.notify_all()

    def _register_pending_item(self, item: _Pending) -> _Registration | _Refusal:
        """Register one pending storage with the runtime; a copy whose fill failed leaves its storage pageable."""
        size = item.request.storage.nbytes()
        if isinstance(item, _InPlace):
            if not self._register(item.pointer, size):
                return _Refusal.CAPACITY
            return _Registration(item.pointer, size, item.request.storage)
        if item.error is not None:
            logger.warning("Could not fill a pinned copy from the checkpoint; it stays pageable: %s", item.error)
            return _Refusal.STORAGE
        if not self._register(item.copy.pointer, item.copy.size):
            return _Refusal.CAPACITY
        return _Registration(item.pointer, size, item.request.storage, item.copy)

    def _evict_pending(self, item: _Pending) -> None:
        """Take back pending storage's reservation and range, freeing its copy."""
        self._starts.pop(bisect_left(self._starts, item.pointer))
        if isinstance(item, _InPlace):
            self._unreserve_range(item.pointer, item.request.storage.nbytes())
        else:
            self._pinned_bytes -= item.copy.size
            item.copy.free()

    def _register(self, pointer: int, size: int) -> bool:
        """Register with the runtime, evicting idle registrations in LRU batches while it refuses capacity."""
        registered = self._try_register(pointer, size)
        while not registered and self._evict_idle_for_runtime_retry(size):
            registered = self._try_register(pointer, size)
        return registered

    def _reserve_range(self, pointer: int, size: int) -> None:
        self._pinned_bytes += self._page_reservation(pointer, size)
        for page in self._boundaries(pointer, size):
            self._boundary_pages[page] = self._boundary_pages.get(page, 0) + 1

    def _unreserve_range(self, pointer: int, size: int) -> None:
        for page in self._boundaries(pointer, size):
            count = self._boundary_pages[page] - 1
            if count:
                self._boundary_pages[page] = count
            else:
                del self._boundary_pages[page]
        self._pinned_bytes -= self._page_reservation(pointer, size)

    def _try_register(self, pointer: int, size: int) -> bool:
        try:
            registered = self._backend.register(pointer, size)
        except Exception:
            self._registration_failures += 1
            raise
        if not registered:
            self._registration_failures += 1
        return registered

    def _evict_idle_for_runtime_retry(self, size: int) -> bool:
        """Evict an LRU batch before retrying a runtime capacity failure."""
        target = max(mmap.PAGESIZE, _page_rounded(size))
        before = self._pinned_bytes
        self._evict_idle(lambda: before - self._pinned_bytes >= target)
        return self._pinned_bytes < before

    def _fit(self, needed: Callable[[], int]) -> bool:
        """Evict idle registrations in LRU order until ``needed()`` more bytes fit under a finite budget.

        The reservation is re-evaluated after each eviction: an in-place range's
        boundary page stops being shared once its neighbour is gone.
        """
        budget = self._max_pinned_bytes
        if budget is None:
            return True
        if needed() > budget:
            return False

        def fits() -> bool:
            return self._pinned_bytes + needed() <= budget

        self._evict_idle(fits)
        return fits()

    def _evict_idle(self, until: Callable[[], bool]) -> None:
        """Unregister idle registrations, least recently released first, until ``until()`` holds.

        These evictions make room, so their copies go to the offered tier when it admits them.
        """
        for candidate in tuple(self._idle):
            if until():
                return
            registration = self._registrations.get(candidate)
            if registration is not None:
                self._unregister(registration, demote=True)

    def _hold(self, registration: _Registration, request: _Request, held: dict[int, _Registration]) -> None:
        registration.leases += 1
        held[registration.pointer] = registration
        self._idle.pop(registration.pointer, None)
        manager_ref, registration_ref = weakref.ref(self), weakref.ref(registration)

        def owner_gone(_ref: weakref.ReferenceType[torch.Tensor]) -> None:
            manager, registration = manager_ref(), registration_ref()
            if manager is not None and registration is not None:
                with manager._lock:
                    registration.retired = True
                    if registration.leases == 0 and manager._registrations.get(registration.pointer) is registration:
                        manager._unregister(registration)
                    elif manager._offered.get(registration.pointer) is registration:
                        manager._take_offered(registration.pointer).free()

        for tensor in request.tensors:
            if id(tensor) not in registration.owners:
                registration.owners[id(tensor)] = weakref.ref(tensor, owner_gone)

    def _unregister(
        self, registration: _Registration, released: list[_Copy] | None = None, *, demote: bool = False,
    ) -> bool:
        """Unregister an idle registration and drop it, freeing its copy, or offering it if ``demote``.

        A failed unregistration keeps the registration, its storage, and its
        reservation for a retry, and returns False.
        """
        assert registration.leases == 0
        try:
            self._backend.unregister(registration.registered)
        except Exception as error:
            self._unregistration_failures += 1
            # Tracebacks in buffered logs can retain storage after a later retry.
            logger.warning("Host unregistration failed; retaining storage and budget reservation: %s", str(error))
            return False
        del self._registrations[registration.pointer]
        self._starts.pop(bisect_left(self._starts, registration.pointer))
        self._idle.pop(registration.pointer, None)
        copy = registration.copy
        if copy is None:
            self._unreserve_range(registration.pointer, registration.size)
        else:
            self._pinned_bytes -= copy.size
            copies = [copy]
            if demote and not registration.retired and copy.offerable and copy.size <= self._max_offered_bytes:
                if self._offering is not None:
                    self._offering.add(registration)
                    copies = []
                else:
                    copies = self._settle_offer(registration, _offer_copy(copy))
            if released is None:
                for copy in copies:
                    copy.free()
            else:
                # Freed by the caller once it has released the lock.
                released.extend(copies)
        self._drop_lifetime_root_if_empty()
        return True

    def _release(self, registrations: tuple[_Registration, ...]) -> None:
        for registration in registrations:
            registration.leases -= 1
            if registration.leases == 0:
                self._idle[registration.pointer] = None
                if registration.retired:
                    self._unregister(registration)
        self._fit(lambda: 0)

    def _close_lease(self, key: int) -> None:
        with self._lock:
            state = self._leases.get(key)
            if state is None:
                return
            self._release(state.registrations)
            for pointer in state.pageable:
                pageable = self._pageable[pointer]
                pageable.leases -= 1
                if pageable.leases == 0:
                    del self._pageable[pointer]
                    self._starts.pop(bisect_left(self._starts, pointer))
            del self._leases[key]
            self._drop_lifetime_root_if_empty()

    def _drop_lifetime_root_if_empty(self) -> None:
        if not self._registrations and not self._leases:
            _live_managers.discard(self)


# Registrations must outlive Python references to a manager. This root
# retains managers with live registrations/leases, but their idle tensor owners
# remain weak. Dropping the last owner retires its registration and releases
# the root. Failed cleanup keeps storage alive rather than freeing pinned bytes.
_live_managers: set[PinManager] = set()

host_pin_manager = PinManager()


def transfer_(destination: torch.Tensor, source: torch.Tensor, *, non_blocking: bool) -> None:
    """Copy ``source`` into ``destination`` through the process-wide manager.

    Every host-to-device copy of a physical tensor goes through here, so a
    checkpoint storage pinned through an owned copy is read from that copy
    while the module's own tensor keeps pointing at the resting mapping. An
    asynchronous copy of pinned storage must run under a lease kept until the
    copy has completed, and raises otherwise; a synchronous copy needs none.
    """
    host_pin_manager.transfer(destination, source, non_blocking=non_blocking)


__all__ = ["PinLease", "PinManager", "PinStats", "TransferLease", "host_pin_manager", "transfer_"]
