"""Budgeted host registrations with active leases and an idle LRU.

Use the process-wide ``host_pin_manager`` for application registrations. Its
budget defaults to half of physical RAM, rounded down to whole OS pages. Set
it to zero to disable registration or ``None`` to remove the application cap.
Construction and configuration perform no CUDA initialization. Isolated
``PinManager`` instances can use an injected backend for testing.

A registration whose last lease closes stays registered in an idle LRU so a
reactivation costs nothing. Idle entries leave only under pressure: budget
admission, a native capacity refusal, an available-memory headroom check for
admissions that consume memory, ``trim()``, ``ensure_available()`` ahead of
a known pageable allocation, or the source tensor being dropped. Releasing a
pin is not the same as freeing memory: unregistering anonymous storage a
model still owns frees nothing, and results report the two separately.

Native registration uses whole storage byte ranges, including private file
mappings of checkpoints. Locking such a mapping for writing makes the kernel
copy every page into private memory, and unregistering does not undo that. On
Linux the manager discards those pages when it unregisters an immutable
mapping (``MADV_DONTNEED``), so the mapping refaults from the file, then warms
the page cache (``MADV_WILLNEED``) so the next registration copies from RAM.
Piper never writes into a file mapping: ``HostParam`` copies trainable
parameters out at capture and ``merge_adapter`` copies its targets out before
merging, so a mapping's private pages never hold anything the file does not.
Windows cannot discard the pages
of a view it did not create, so private pages there stay until the mapping is
released, within the budget.

Budget accounting counts the union of OS pages, including pages shared by
separate allocations. Registrations retain storage until unregistration
succeeds, but track their source tensors weakly while idle so discarded
resources can release memory. Storage must not be resized or independently
registered while managed here.
"""

import ctypes
import functools
import logging
import mmap
import os
import sys
import threading
import weakref
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Self

import torch

from ._host_registration import HostRegistrationBackend, RuntimeHostRegistration

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


def _default_pin_budget() -> int:
    """Use half of physical RAM as the cap without touching CUDA."""
    try:
        if sys.platform == "win32":
            status = _MemoryStatus()
            status.length = ctypes.sizeof(status)
            query = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
            query.argtypes = (ctypes.POINTER(_MemoryStatus),)
            query.restype = ctypes.c_int
            if not query(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx failed")
            total = status.total_physical
        else:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        if total <= 0:
            raise ValueError("physical RAM is unavailable")
        return total // (2 * mmap.PAGESIZE) * mmap.PAGESIZE
    except (AttributeError, OSError, ValueError) as error:
        logger.warning("Cannot determine physical RAM; default host pin budget is zero: %s", error)
        return 0


_DEFAULT_PIN_BUDGET = _default_pin_budget()


def available_memory_bytes() -> int | None:
    """Memory the OS could hand out now without swapping, or ``None`` when unknown."""
    try:
        if sys.platform == "win32":
            status = _MemoryStatus()
            status.length = ctypes.sizeof(status)
            query = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
            query.argtypes = (ctypes.POINTER(_MemoryStatus),)
            query.restype = ctypes.c_int
            return int(status.available_physical) if query(ctypes.byref(status)) else None
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _default_headroom() -> int:
    """Keep a sixteenth of RAM, at least 1 GiB, for the page cache and everything else."""
    total = _DEFAULT_PIN_BUDGET * 2
    return max(2**30, total // 16) // mmap.PAGESIZE * mmap.PAGESIZE


@dataclass(frozen=True, slots=True)
class PinStats:
    """Registration counts and the union of charged OS pages.

    ``reserved_bytes`` are pending admissions (see :meth:`PinManager.reserve`)
    that count against the budget before their registration exists.
    """

    max_pinned_bytes: int | None
    pinned_bytes: int
    registrations: int
    idle_registrations: int
    active_leases: int
    registration_failures: int
    unregistration_failures: int
    reserved_bytes: int = 0
    headroom_bytes: int = 0


@dataclass(frozen=True, slots=True)
class TrimResult:
    """What an eviction pass did.

    ``released_bytes`` is pin charge given back to the budget. ``freed_bytes``
    is memory the manager itself returned to the OS by evicting, which today
    is the discarded interior pages of a private file mapping; unregistering
    anonymous storage a model still owns frees nothing. ``failures`` counts
    entries whose native unregistration failed and stay charged.
    """

    released_bytes: int
    freed_bytes: int
    failures: int


class Reservation:
    """Budget held for a copy that is not registered yet.

    Obtained from :meth:`PinManager.reserve`. The bytes count against the pin
    budget until the reservation is consumed by ``acquire(reservation=...)``
    or released. Use as a context manager to release on error.
    """

    def __init__(self, manager: PinManager, size: int) -> None:
        self.size = size
        self._manager: PinManager | None = manager

    @property
    def active(self) -> bool:
        return self._manager is not None

    def release(self) -> None:
        manager, self._manager = self._manager, None
        if manager is not None:
            with manager._lock:
                manager._reserved_bytes -= self.size

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


@dataclass(eq=False)
class _Registration:
    pointer: int
    size: int
    storage: torch.UntypedStorage
    # A private file mapping: its copied pages return to the file on
    # unregistration.
    discard: bool = False
    owners: dict[int, weakref.ReferenceType[torch.Tensor]] = field(default_factory=dict)
    leases: int = 0
    retired: bool = False


@dataclass(slots=True)
class _Request:
    storage: torch.UntypedStorage
    tensors: list[torch.Tensor]


@dataclass(frozen=True, slots=True)
class _Mapping:
    start: int
    end: int


def _private_file_mappings() -> tuple[_Mapping, ...]:
    """Private file mappings of this process in address order; empty off Linux.

    Only the kernel knows whether storage is a private file mapping.
    ``storage.resizable()`` cannot tell one from ``frombuffer`` over anonymous
    memory, which ``MADV_DONTNEED`` would zero. ``/dev/zero`` is skipped for
    the same reason: it is anonymous memory with an inode. Files on tmpfs,
    such as a checkpoint under ``/dev/shm``, refault like any other file.
    """
    if not sys.platform.startswith("linux"):
        return ()
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            lines = maps.readlines()
    except OSError:
        return ()
    mappings: list[_Mapping] = []
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 5 or fields[1][3:4] != "p" or fields[4] == "0":
            continue
        if len(fields) == 6 and fields[5].rstrip() == "/dev/zero":
            continue
        start, end = fields[0].split("-", maxsplit=1)
        mappings.append(_Mapping(int(start, 16), int(end, 16)))
    return tuple(mappings)


class _PrivateFileMappings:
    """The process's private file mappings, read on first use.

    Relay collectives re-acquire an already registered buffer per operation;
    the memory map is read only when a new registration needs it.
    """

    @functools.cached_property
    def ranges(self) -> tuple[_Mapping, ...]:
        return _private_file_mappings()

    def cover(self, pointer: int, size: int) -> bool:
        ranges = self.ranges
        index = bisect_right(ranges, pointer, key=lambda mapping: mapping.start) - 1
        return index >= 0 and pointer + size <= ranges[index].end


_MADV_WILLNEED = 3
_MADV_DONTNEED = 4


@functools.cache
def _madvise() -> Callable[[int, int, int], int]:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    libc.madvise.restype = ctypes.c_int
    return libc.madvise


def _interior_pages(pointer: int, size: int) -> tuple[int, int]:
    """The whole pages inside a range, as (start, length).

    A page shared with a neighbouring registration may still be locked for
    the device; discarding it would split the host and device views of that
    page. Such a boundary page keeps its private copy instead.
    """
    start = -(-pointer // mmap.PAGESIZE) * mmap.PAGESIZE
    end = (pointer + size) // mmap.PAGESIZE * mmap.PAGESIZE
    return start, max(0, end - start)


def _discard_and_warm(pointer: int, size: int, *, warm: bool) -> None:
    """Return a private file mapping's copied pages to the file, then read the file back into cache."""
    start, length = _interior_pages(pointer, size)
    if length == 0:
        return
    try:
        madvise = _madvise()
        if madvise(start, length, _MADV_DONTNEED):
            logger.warning("MADV_DONTNEED failed with errno %d; private pages stay resident", ctypes.get_errno())
        elif warm:
            madvise(start, length, _MADV_WILLNEED)
    except Exception as error:
        logger.warning("Discarding private pages failed: %s", str(error))


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
    """Protect registrations and source tensors until explicitly released.

    ``registered_bytes`` and ``pageable_bytes`` count unique requested storage
    bytes, without page rounding. The owner must keep the lease open until no
    asynchronous operation can read or write its host tensors. CUDA ordering belongs to
    the runtime that enqueues those operations; the pin manager does not track
    or synchronize accelerator streams. Dropping the token also releases its
    protection, so asynchronous owners must retain it through completion.
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
        """Release registration and source protection, idempotently."""
        self._finalizer()

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("Pin lease is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class PinManager:
    """Own registrations under a page-rounded budget.

    A finite ``max_pinned_bytes`` bounds registered pages in this process.
    The default is half of physical RAM at import, rounded down to OS pages.
    Explicit ``None`` removes the application cap; native capacity failures
    still reclaim unrelated idle registrations and fall back to pageable memory.

    Acquire accepts the plain CPU tensors returned by ``storage_tensors()``.
    Tensor views share one whole-storage registration. Separate allocations
    may share OS pages, which are charged once. Distinct overlapping byte
    ranges (for example separate ``frombuffer`` wrappers) are
    rejected before registration; registering only part of a copy's range can
    make the CUDA/HIP copy invalid. Use views of one storage for such aliases.

    All metadata and backend operations are serialized by a reentrant lock.
    Active leases also retain pageable sources. Idle entries retain storage,
    but no model or tensor wrappers. Losing a source tensor retires its
    registration as soon as active leases have finished with it.
    Pageable allocations remain pageable until all their active leases close.
    """

    def __init__(
        self,
        max_pinned_bytes: int | None = _DEFAULT_PIN_BUDGET,
        *,
        headroom_bytes: int | None = None,
        backend: HostRegistrationBackend | None = None,
        available_memory: Callable[[], int | None] = available_memory_bytes,
    ) -> None:
        if max_pinned_bytes is not None and max_pinned_bytes < 0:
            raise ValueError("max_pinned_bytes must be >= 0")
        if headroom_bytes is not None and headroom_bytes < 0:
            raise ValueError("headroom_bytes must be >= 0")
        self._max_pinned_bytes = max_pinned_bytes
        # Available memory that an admission consuming memory must leave behind.
        self._headroom_bytes = _default_headroom() if headroom_bytes is None else headroom_bytes
        self._available_memory = available_memory
        self._backend = backend if backend is not None else RuntimeHostRegistration()
        self._lock = threading.RLock()
        self._entries: dict[int, _Registration] = {}
        self._pageable: dict[int, _Pageable] = {}
        # All registered ranges and actively leased pageable ranges.
        self._starts: list[int] = []
        self._idle: OrderedDict[int, None] = OrderedDict()
        # Disjoint byte ranges can share only their boundary pages. Tracking
        # those endpoints avoids one Python entry per page of a large model.
        self._boundary_pages: dict[int, int] = {}
        self._pinned_bytes = 0
        self._reserved_bytes = 0
        self._freed_bytes = 0
        self._registration_failures = 0
        self._unregistration_failures = 0
        self._leases: dict[int, _LeaseState] = {}
        self._next_lease = 0

    @property
    def max_pinned_bytes(self) -> int | None:
        with self._lock:
            return self._max_pinned_bytes

    @max_pinned_bytes.setter
    def max_pinned_bytes(self, value: int | None) -> None:
        """Set the budget or enable opportunistic native-capacity discovery.

        ``None`` removes the application byte limit. Native capacity failures
        still reclaim unrelated idle registrations before falling back to
        pageable storage. For a finite limit, releases trim active excess back
        to budget. Failed unregistrations stay charged and can be retried with
        ``clear()`` or later admission pressure.
        """
        if value is not None and value < 0:
            raise ValueError("max_pinned_bytes must be >= 0")
        with self._lock:
            self._max_pinned_bytes = value
            self._make_room(0, 0)

    @property
    def headroom_bytes(self) -> int:
        with self._lock:
            return self._headroom_bytes

    @headroom_bytes.setter
    def headroom_bytes(self, value: int) -> None:
        if value < 0:
            raise ValueError("headroom_bytes must be >= 0")
        with self._lock:
            self._headroom_bytes = value

    @property
    def stats(self) -> PinStats:
        with self._lock:
            return PinStats(
                self._max_pinned_bytes,
                self._pinned_bytes,
                len(self._entries),
                len(self._idle),
                len(self._leases),
                self._registration_failures,
                self._unregistration_failures,
                self._reserved_bytes,
                self._headroom_bytes,
            )

    def acquire(self, tensors: Iterable[torch.Tensor], *, reservation: Reservation | None = None) -> PinLease:
        """Lease whole allocations, leaving capacity failures pageable.

        ``reservation`` hands back budget held by :meth:`reserve` for these
        allocations, so a copy admitted before it was filled is not charged
        twice. The reservation is consumed whether or not registration
        succeeds; a refused registration simply returns its bytes.

        Storage inside a private file mapping returns to the file when it is
        unregistered (Linux). The host must not write into such storage
        while it is managed here; anonymous and shared storage is never
        discarded.

        All input validation happens before registration or eviction. Existing
        registrations anywhere in the request are protected before admitting
        new ones, avoiding eviction of backing this same lease will use. A
        native capacity failure reclaims unrelated idle registrations and
        retries; if capacity remains unavailable, later allocations in this
        acquisition skip registration.
        """
        requests = self._requests(tensors)
        held: dict[int, _Registration] = {}
        created: list[_Registration] = []
        mappings = _PrivateFileMappings()
        with self._lock:
            self._validate_ranges(requests)
            if reservation is not None:
                reservation.release()
            try:
                for pointer, request in requests.items():
                    entry = self._entries.get(pointer)
                    if entry is not None:
                        self._hold(entry, request, held)
                for pointer, request in requests.items():
                    if pointer in held or pointer in self._pageable:
                        continue
                    size = request.storage.nbytes()
                    # Registering a private file mapping in place makes the
                    # kernel copy every page, so it consumes memory.
                    discard = mappings.cover(pointer, size)
                    if not self._make_room(pointer, size, consumes_memory=discard):
                        continue
                    registered = self._try_register(pointer, size)
                    while not registered and self._reclaim_idle_for_native_retry(
                        pointer,
                        size,
                    ):
                        registered = self._try_register(pointer, size)
                    if not registered:
                        # Native capacity is still unavailable after reclaiming
                        # every lower-priority idle registration that can help.
                        # Avoid one failed runtime call per remaining tensor.
                        break
                    entry = _Registration(pointer, size, request.storage, discard=discard)
                    _live_managers.add(self)
                    self._entries[pointer] = entry
                    self._starts.insert(bisect_left(self._starts, pointer), pointer)
                    self._pinned_bytes += self._page_charge(pointer, size)
                    for page in self._boundaries(pointer, size):
                        self._boundary_pages[page] = self._boundary_pages.get(page, 0) + 1
                    created.append(entry)
                    self._hold(entry, request, held)
            except BaseException:
                for entry in created:
                    entry.retired = True
                self._release(tuple(held.values()))
                raise

            return self._open_lease(requests, held)

    def _open_lease(self, requests: dict[int, _Request], held: dict[int, _Registration]) -> PinLease:
        key = self._next_lease
        self._next_lease += 1
        pageable = tuple(pointer for pointer in requests if pointer not in held)
        for pointer in pageable:
            allocation = self._pageable.get(pointer)
            if allocation is None:
                allocation = _Pageable(requests[pointer].storage.nbytes())
                self._pageable[pointer] = allocation
                self._starts.insert(bisect_left(self._starts, pointer), pointer)
            allocation.leases += 1
        self._leases[key] = _LeaseState(
            tuple(held.values()),
            pageable,
            tuple(tensor for request in requests.values() for tensor in request.tensors),
        )
        _live_managers.add(self)
        registered = sum(entry.size for entry in held.values())
        total = sum(request.storage.nbytes() for request in requests.values())
        return PinLease(self, key, registered, total - registered)

    def clear(self) -> None:
        """Unregister every idle entry.

        Live leases remain protected. A failed unregistration retains its
        storage and budget charge; cleanup errors propagate so callers can
        retry without losing ownership of registered memory.
        """
        result = self.trim()
        if result.failures:
            raise RuntimeError(
                f"Could not release {result.failures} host registration(s); storage remains retained",
            )

    def trim(self, size: int | None = None) -> TrimResult:
        """Evict idle entries, least recently released first, until ``size`` bytes of pin charge are released.

        ``None`` evicts every idle entry. Live leases are never touched, so
        the result may fall short; a failed unregistration counts as a
        failure and keeps its charge. Freed memory is reported separately
        from released charge, since unregistering anonymous storage a model
        still owns frees nothing.
        """
        with self._lock:
            released = freed = failures = 0
            freed_before = self._freed_bytes
            for pointer in tuple(self._idle):
                if size is not None and released >= size:
                    break
                entry = self._entries.get(pointer)
                if entry is None:
                    continue
                charged_before = self._pinned_bytes
                if self._unregister(entry):
                    released += charged_before - self._pinned_bytes
                else:
                    failures += 1
            freed = self._freed_bytes - freed_before
            return TrimResult(released, freed, failures)

    def reserve(self, size: int) -> Reservation | None:
        """Hold ``size`` bytes of budget for a copy that is about to be filled.

        Admission is the same as for a registration: idle entries are
        evicted to fit the budget, and available memory must keep the
        headroom. Returns ``None`` when the bytes cannot be admitted, in
        which case the caller keeps the source pageable. Pass the result to
        :meth:`acquire` once the copy is registered, or release it on error.
        """
        if size < 0:
            raise ValueError("size must be >= 0")
        with self._lock:
            if not self._make_room(0, size, consumes_memory=True):
                return None
            self._reserved_bytes += size
            return Reservation(self, size)

    def ensure_available(self, size: int) -> int:
        """Make room for a known pageable allocation of ``size`` bytes, returning the shortfall.

        Idle entries are evicted, least recently released first, until the OS
        reports ``size`` plus the headroom available or nothing idle remains.
        Only evictions that return memory help; unregistering anonymous
        storage a model owns changes nothing, and the loop stops when no
        idle entry is left. Returns zero when satisfied, or the bytes still
        missing. Returns zero without evicting when availability is unknown.
        """
        if size < 0:
            raise ValueError("size must be >= 0")
        with self._lock:
            shortfall = self._headroom_shortfall(size)
            for pointer in tuple(self._idle):
                if shortfall <= 0:
                    break
                entry = self._entries.get(pointer)
                if entry is not None:
                    self._unregister(entry)
                shortfall = self._headroom_shortfall(size)
            return max(0, shortfall)

    def _headroom_shortfall(self, size: int) -> int:
        """Bytes by which ``size`` plus the headroom exceeds available memory; zero when unknown."""
        available = self._available_memory()
        if available is None:
            return 0
        return size + self._headroom_bytes - available

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
                allocation = self._entries.get(other) or self._pageable[other]
                other_end = other + allocation.size
                if pointer < other_end and other < end and (pointer != other or end != other_end):
                    raise ValueError("Overlapping host storage ranges must use views of one storage")
            prior_end = end

    @staticmethod
    def _boundaries(pointer: int, size: int) -> tuple[int, ...]:
        first = pointer // mmap.PAGESIZE
        last = (pointer + size - 1) // mmap.PAGESIZE
        return (first,) if first == last else (first, last)

    def _page_charge(self, pointer: int, size: int) -> int:
        if size == 0:
            return 0
        pages = (pointer + size - 1) // mmap.PAGESIZE - pointer // mmap.PAGESIZE + 1
        shared = sum(page in self._boundary_pages for page in self._boundaries(pointer, size))
        return (pages - shared) * mmap.PAGESIZE

    def _try_register(self, pointer: int, size: int) -> bool:
        try:
            registered = self._backend.register(pointer, size)
        except Exception:
            self._registration_failures += 1
            raise
        if not registered:
            self._registration_failures += 1
        return registered

    def _reclaim_idle_for_native_retry(self, pointer: int, size: int) -> bool:
        """Evict an LRU batch before retrying a native-capacity failure."""
        target = max(mmap.PAGESIZE, self._page_charge(pointer, size))
        before = self._pinned_bytes
        for candidate in tuple(self._idle):
            entry = self._entries.get(candidate)
            if entry is not None:
                self._unregister(entry)
            if before - self._pinned_bytes >= target:
                break
        return self._pinned_bytes < before

    def _make_room(self, pointer: int, size: int, *, consumes_memory: bool = False) -> bool:
        """Evict idle entries until ``size`` fits the budget and, if it consumes memory, the headroom."""
        limit = self._max_pinned_bytes
        if limit is not None and size:
            pages = (pointer + size - 1) // mmap.PAGESIZE - pointer // mmap.PAGESIZE + 1
            if pages * mmap.PAGESIZE > limit:
                return False
        if self._fits(pointer, size, consumes_memory):
            return True
        for candidate in tuple(self._idle):
            entry = self._entries.get(candidate)
            if entry is not None:
                self._unregister(entry)
            if self._fits(pointer, size, consumes_memory):
                return True
        return False

    def _fits(self, pointer: int, size: int, consumes_memory: bool) -> bool:
        limit = self._max_pinned_bytes
        if limit is not None and self._pinned_bytes + self._reserved_bytes + self._page_charge(pointer, size) > limit:
            return False
        return not consumes_memory or self._headroom_shortfall(size) <= 0

    def _hold(self, entry: _Registration, request: _Request, held: dict[int, _Registration]) -> None:
        entry.leases += 1
        held[entry.pointer] = entry
        self._idle.pop(entry.pointer, None)
        manager_ref, entry_ref = weakref.ref(self), weakref.ref(entry)

        def owner_gone(_ref: weakref.ReferenceType[torch.Tensor]) -> None:
            manager, registration = manager_ref(), entry_ref()
            if manager is not None and registration is not None:
                with manager._lock:
                    registration.retired = True
                    if registration.leases == 0 and manager._entries.get(registration.pointer) is registration:
                        manager._unregister(registration)

        for tensor in request.tensors:
            if id(tensor) not in entry.owners:
                entry.owners[id(tensor)] = weakref.ref(tensor, owner_gone)

    def _unregister(self, entry: _Registration) -> bool:
        assert entry.leases == 0
        try:
            self._backend.unregister(entry.pointer)
        except Exception as error:
            self._unregistration_failures += 1
            # Tracebacks in buffered logs can retain storage after a later retry.
            logger.warning("Host unregistration failed; retaining storage and budget charge: %s", str(error))
            return False
        del self._entries[entry.pointer]
        self._starts.pop(bisect_left(self._starts, entry.pointer))
        self._idle.pop(entry.pointer, None)
        for page in self._boundaries(entry.pointer, entry.size):
            count = self._boundary_pages[page] - 1
            if count:
                self._boundary_pages[page] = count
            else:
                del self._boundary_pages[page]
        self._pinned_bytes -= self._page_charge(entry.pointer, entry.size)
        self._drop_lifetime_root_if_empty()
        if entry.discard:
            # A retired mapping is on its way out; do not read its file back.
            _discard_and_warm(entry.pointer, entry.size, warm=not entry.retired)
            self._freed_bytes += _interior_pages(entry.pointer, entry.size)[1]
        return True

    def _release(self, entries: tuple[_Registration, ...]) -> None:
        for entry in entries:
            entry.leases -= 1
            if entry.leases == 0:
                self._idle[entry.pointer] = None
                if entry.retired:
                    self._unregister(entry)
        self._make_room(0, 0)

    def _close_lease(self, key: int) -> None:
        with self._lock:
            state = self._leases.get(key)
            if state is None:
                return
            self._release(state.registrations)
            for pointer in state.pageable:
                allocation = self._pageable[pointer]
                allocation.leases -= 1
                if allocation.leases == 0:
                    del self._pageable[pointer]
                    self._starts.pop(bisect_left(self._starts, pointer))
            del self._leases[key]
            self._drop_lifetime_root_if_empty()

    def _drop_lifetime_root_if_empty(self) -> None:
        if not self._entries and not self._leases:
            _live_managers.discard(self)


# Native registrations must outlive Python references to a manager. This root
# retains managers with live registrations/leases, but their idle tensor owners
# remain weak. Discarding the last source retires its registration and releases
# the root. Failed cleanup keeps storage alive rather than freeing pinned bytes.
_live_managers: set[PinManager] = set()

host_pin_manager = PinManager()

__all__ = [
    "PinLease",
    "PinManager",
    "PinStats",
    "Reservation",
    "TrimResult",
    "available_memory_bytes",
    "host_pin_manager",
]
