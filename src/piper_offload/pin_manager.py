"""Budgeted host registrations with active leases and an idle LRU.

Use the process-wide ``host_pin_manager`` for application registrations. Its
budget defaults to half of the memory available to the process, physical RAM
or the container's cgroup limit when that is lower, rounded down to whole OS
pages. Set
it to zero to disable registration or ``None`` to remove the application cap
and register opportunistically up to native CUDA/HIP capacity. Construction
and configuration perform no CUDA initialization. Isolated ``PinManager``
instances can use an injected backend for testing.

Every host transfer runs under a lease, so nothing is read or written while
it could be evicted, and only transfers that repeat ask the lease to pin:
streaming, rolling, and the relay register their storage; a resident or host
upload and the optimizer copy-back lease it pageable.

Native registration uses whole storage byte ranges. Storage that records a
checkpoint file slice (:func:`file_slice`) is never registered in place: the
mapping stays read-only page cache, and pinning it means allocating an owned
page-aligned copy, filling it from the file with positional reads, and
registering that. Evicting the copy unregisters and frees it, so the RAM
returns. Transfers reach the copy through ``transfer_``, which resolves each
source tensor at copy time; an asynchronous transfer of pinned storage
outside a lease raises, since eviction is safe only because every such
reader holds one, while a synchronous one completes under the manager's
lock. Other storage, anonymous memory and mappings without provenance,
registers where it is, as before.
Piper never writes into a file mapping: ``HostParam`` copies trainable
parameters out at capture and ``merge_adapter`` copies its targets out before
merging.

Budget accounting counts the union of OS pages, including pages shared by
separate allocations. Registrations retain storage until unregistration
succeeds, but track their source tensors weakly while idle so discarded
resources can release memory. Storage must not be resized or independently
registered while managed here.
"""

import contextlib
import ctypes
import enum
import functools
import io
import logging
import mmap
import os
import sys
import threading
import weakref
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import torch

from ._host_registration import HostRegistrationBackend, RuntimeHostRegistration
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
    checkpoint storage, filling or registered; the rest is storage
    registered in place.
    """

    max_pinned_bytes: int | None
    pinned_bytes: int
    registrations: int
    idle_registrations: int
    active_leases: int
    registration_failures: int
    unregistration_failures: int
    copy_bytes: int


def _page_rounded(nbytes: int) -> int:
    return -(-nbytes // mmap.PAGESIZE) * mmap.PAGESIZE


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
        self.region = mmap.mmap(-1, size)
        # None once freed; the region cannot close while a storage exports it.
        self.storage: torch.UntypedStorage | None = (
            torch.frombuffer(self.region, dtype=torch.uint8).untyped_storage()
        )

    @property
    def pointer(self) -> int:
        assert self.storage is not None
        return self.storage.data_ptr()

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
# The seek-and-read fallback moves each file's shared position.
_fill_lock = threading.Lock()


def _preadv(fd: int, buffer: memoryview, offset: int) -> int:
    return os.preadv(fd, [buffer], offset)


def _read_serially(file: io.BufferedReader, buffer: memoryview, offset: int) -> int:
    file.seek(offset)
    return file.readinto(buffer)


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


@dataclass(eq=False, slots=True)
class _InPlace:
    """In-place storage reserved under the budget for one request, registered with the rest of its acquisition."""

    pointer: int
    request: _Request


@dataclass(eq=False, slots=True)
class _PendingCopy:
    """A copy allocated and reserved for one request, filled and registered with the rest of its acquisition."""

    pointer: int
    request: _Request
    source: FileSlice
    copy: _Copy
    error: OSError | None = None


type _Pending = _InPlace | _PendingCopy


def _fill_copies(pending: list[_PendingCopy]) -> None:
    """Fill every pending copy from its file; a read failure marks that copy's ``error``.

    Positional reads over fixed-size slices of every copy run together on
    one thread per logical core, hyperthreads included: the fill is bound
    by page faults on the fresh regions, which overlap across threads, and
    a checkpoint's tensors are mostly smaller than one slice. Without
    positional reads the copies fill one at a time under the lock. Runs
    without the manager's lock: the workers run Python code, so a garbage
    collection on one of them may run a finalizer that needs that lock.
    """
    if not hasattr(os, "preadv"):
        with _fill_lock:
            for item in pending:
                read_at = functools.partial(_read_serially, item.source.file)
                try:
                    with memoryview(item.copy.region)[: item.source.length] as view:
                        _read_range(read_at, view, item.source.offset, 0, item.source.length)
                except OSError as error:
                    item.error = _detached(error)
        return
    workers = os.process_cpu_count() or 1
    # The pool closes before the views: a failure waits for the queued slices to finish.
    with contextlib.ExitStack() as views, ThreadPoolExecutor(workers, thread_name_prefix="piper-offload-fill") as pool:
        futures: list[tuple[_PendingCopy, Future[None]]] = []
        for item in pending:
            length = item.source.length
            view = views.enter_context(memoryview(item.copy.region)[:length])
            read_at = functools.partial(_preadv, item.source.file.fileno())
            for start in range(0, length, _FILL_SLICE):
                stop = min(start + _FILL_SLICE, length)
                futures.append((item, pool.submit(_read_range, read_at, view, item.source.offset, start, stop)))
        for item, future in futures:
            try:
                future.result()
            except OSError as error:
                item.error = _detached(error)


def _detached(error: OSError) -> OSError:
    """The error without its traceback, whose frames would hold a slice of the copy and keep it mapped."""
    return error.with_traceback(None)


class _Refusal(enum.Enum):
    BUDGET = "budget"  # this storage stays pageable; a later one in the acquisition may register
    CAPACITY = "capacity"  # native capacity is exhausted; stop registering


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


class TransferLease:
    """A pageable lease over one transfer, closed only once its device has synchronized.

    ``start`` leases the transfer's sources without registering them, and
    ``finish`` synchronizes the device the transfer used before closing the
    lease. A synchronization that fails leaves the lease open, so the sources
    stay protected until a later ``finish`` succeeds, and no new transfer can
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


class PinManager:
    """Own registrations under a page-rounded budget.

    A finite ``max_pinned_bytes`` bounds registered pages in this process,
    including the owned copies made for checkpoint storage; the default is
    half of physical RAM at import. ``None`` treats native CUDA/HIP capacity
    as the limit, reclaiming unrelated idle registrations when the runtime
    refuses a new allocation.

    Acquire accepts the plain CPU tensors returned by ``storage_tensors()``.
    Tensor views share one whole-storage registration. Separate allocations
    may share OS pages, which are reserved once. Distinct overlapping byte
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
        backend: HostRegistrationBackend | None = None,
    ) -> None:
        if max_pinned_bytes is not None and max_pinned_bytes < 0:
            raise ValueError("max_pinned_bytes must be >= 0")
        self._max_pinned_bytes = max_pinned_bytes
        self._backend = backend if backend is not None else RuntimeHostRegistration()
        self._lock = threading.RLock()
        self._entries: dict[int, _Registration] = {}
        self._pageable: dict[int, _Pageable] = {}
        # Storage reserved by an acquisition that has not registered it yet;
        # an acquisition of the same storage waits until it settles.
        self._pending: dict[int, _Pending] = {}
        self._settled = threading.Condition(self._lock)
        # All registered ranges and actively leased pageable ranges.
        self._starts: list[int] = []
        self._idle: OrderedDict[int, None] = OrderedDict()
        # Disjoint byte ranges can share only their boundary pages. Tracking
        # those endpoints avoids one Python entry per page of a large model.
        self._boundary_pages: dict[int, int] = {}
        self._pinned_bytes = 0
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
        to budget. Failed unregistrations stay reserved and can be retried with
        ``clear()`` or later admission pressure.
        """
        if value is not None and value < 0:
            raise ValueError("max_pinned_bytes must be >= 0")
        with self._lock:
            self._max_pinned_bytes = value
            self._make_room(lambda: 0)

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
                sum(entry.copy.size for entry in self._entries.values() if entry.copy is not None)
                + sum(item.copy.size for item in self._pending.values() if isinstance(item, _PendingCopy)),
            )

    def acquire(self, tensors: Iterable[torch.Tensor], *, pin: bool = True) -> PinLease:
        """Lease whole allocations for a transfer, registering them only if ``pin``.

        A lease protects its sources until it closes: registrations they
        already have are held out of eviction, and everything else is tracked
        as pageable. With ``pin`` the lease also registers what the budget
        allows, for storage that repeats every step; without it nothing new
        is registered, for a transfer that runs once, so leasing never
        changes which storage is pinned.

        A pinning acquisition waits while another has reserved storage it has
        not yet registered, and a pageable one waits only for its own storage,
        so pinning settles in turn as it did under the lock while pageable
        leases and transfers proceed. All input validation happens before
        registration or eviction. Existing registrations anywhere in the
        request are protected before admitting new ones, avoiding eviction of
        backing this same lease will use. New storage is reserved under the
        budget in request order, with a copy allocated for checkpoint
        storage; the copies fill with the lock released, the first alone and
        the rest together once it has registered; then everything registers
        in request order. A native capacity failure reclaims unrelated idle
        registrations and retries; if capacity remains unavailable, later
        storage in this acquisition skips registration.
        """
        requests = self._requests(tensors)
        held: dict[int, _Registration] = {}
        created: list[_Registration] = []
        pending: list[_Pending] = []
        try:
            with self._lock:
                while self._pending and (pin or not self._pending.keys().isdisjoint(requests)):
                    self._settled.wait()
                self._validate_ranges(requests)
                for pointer, request in requests.items():
                    entry = self._entries.get(pointer)
                    if entry is not None:
                        self._hold(entry, request, held)
                if pin:
                    self._reserve(requests, held, pending)
                copies = [item for item in pending if isinstance(item, _PendingCopy)]
            if copies:
                # The first copy fills and registers alone, so a runtime out
                # of capacity is found before the rest is read.
                _fill_copies(copies[:1])
                with self._lock:
                    self._register_pending(pending, held, created, through=copies[0])
                if pending and len(copies) > 1:
                    _fill_copies(copies[1:])
            with self._lock:
                self._register_pending(pending, held, created)
                return self._open_lease(requests, held)
        except BaseException:
            with self._lock:
                self._abandon(pending)
                for entry in created:
                    entry.retired = True
                self._release(tuple(held.values()))
            raise

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

    def transfer(self, destination: torch.Tensor, source: torch.Tensor, *, non_blocking: bool) -> None:
        """Copy ``source`` into ``destination``, reading ``source``'s pinned copy when one exists.

        A synchronous transfer has completed when this returns and runs under
        the manager's lock, so nothing can evict its source meanwhile. An
        asynchronous one, a non-blocking copy to a CUDA device, may still be
        in flight afterwards, so it must run under a lease that holds the
        source, and raises otherwise: eviction is safe only because every
        such reader holds one. Unregistered storage copies as is.
        """
        if source.device.type != "cpu":
            destination.copy_(source, non_blocking=non_blocking)
            return
        asynchronous = non_blocking and destination.device.type == "cuda"
        with self._lock:
            entry = self._entries.get(source.untyped_storage().data_ptr())
            view = source
            if entry is not None:
                if asynchronous and entry.leases == 0:
                    raise RuntimeError(
                        "Asynchronous transfer of pinned host storage outside a lease; acquire one "
                        "over the tensors first (pin=False for a one-time transfer) and keep it "
                        "until the transfer has completed."
                    )
                if entry.copy is not None:
                    view = entry.copy.view(source)
            if not asynchronous:
                destination.copy_(view, non_blocking=non_blocking)
                return
        # The lease keeps the entry, so the copy itself needs no lock.
        destination.copy_(view, non_blocking=True)

    def clear(self) -> None:
        """Unregister idle entries and free their copies.

        Live leases remain protected. A failed unregistration retains its
        storage and budget reservation; cleanup errors propagate so callers can
        retry without losing ownership of registered memory.
        """
        with self._lock:
            failed = 0
            for pointer in tuple(self._idle):
                entry = self._entries.get(pointer)
                if entry is not None and not self._unregister(entry):
                    failed += 1
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
        """The extent of a range in ``_starts``: registered, leased pageable, or reserved but pending."""
        entry = self._entries.get(pointer)
        if entry is not None:
            return entry.size
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

        In-place storage reserves its pages; checkpoint storage gets an
        allocated, reserved copy. Each stays in ``self._pending``, and its
        range in ``self._starts``, until it registers or is discarded.
        """
        for pointer, request in requests.items():
            if pointer in held or pointer in self._pageable:
                continue
            source = file_slice(request.storage)
            if source is None:
                item: _Pending | None = self._reserve_in_place(pointer, request)
            else:
                copy = self._allocate_copy(request.storage.nbytes())
                item = None if copy is None else _PendingCopy(pointer, request, source, copy)
            if item is not None:
                pending.append(item)
                self._pending[pointer] = item
                self._starts.insert(bisect_left(self._starts, pointer), pointer)

    def _reserve_in_place(self, pointer: int, request: _Request) -> _InPlace | None:
        size = request.storage.nbytes()
        if not self._make_room(lambda: self._page_reservation(pointer, size)):
            return None
        self._reserve_range(pointer, size)
        return _InPlace(pointer, request)

    def _allocate_copy(self, nbytes: int) -> _Copy | None:
        """Allocate and reserve a copy under the budget, or None if it does not fit or cannot be allocated."""
        size = _page_rounded(nbytes)
        if not self._make_room(lambda: size):
            return None
        try:
            copy = _Copy(size)
        except (OSError, MemoryError) as error:
            logger.warning("Could not allocate a pinned copy; the checkpoint storage stays pageable: %s", error)
            return None
        # Reserved at allocation, before the fill, so nothing else is admitted
        # into the same budget while the read is in progress.
        self._pinned_bytes += size
        return copy

    def _register_pending(
        self,
        pending: list[_Pending],
        held: dict[int, _Registration],
        created: list[_Registration],
        through: _Pending | None = None,
    ) -> None:
        """Register the pending storage in request order, consuming ``pending`` through ``through`` or entirely.

        Refused storage is unreserved, and its copy freed; once native
        capacity is exhausted the rest is discarded the same way, avoiding
        one failed runtime call per remaining storage.
        """
        exhausted = False
        while pending:
            item = pending[0]
            registration = _Refusal.CAPACITY if exhausted else self._register_pending_item(item)
            if isinstance(registration, _Registration):
                self._publish(registration, item.request, held, created)
            else:
                self._discard(item)
                exhausted = registration is _Refusal.CAPACITY
            del self._pending[item.pointer]
            del pending[0]
            if item is through and not exhausted:
                break
        self._settled.notify_all()

    def _register_pending_item(self, item: _Pending) -> _Registration | _Refusal:
        """Register one pending storage natively; a copy whose fill failed leaves its storage pageable."""
        size = item.request.storage.nbytes()
        if isinstance(item, _InPlace):
            if not self._register(item.pointer, size):
                return _Refusal.CAPACITY
            return _Registration(item.pointer, size, item.request.storage)
        if item.error is not None:
            logger.warning("Could not fill a pinned copy from the checkpoint; it stays pageable: %s", item.error)
            return _Refusal.BUDGET
        if not self._register(item.copy.pointer, item.copy.size):
            return _Refusal.CAPACITY
        return _Registration(item.pointer, size, item.request.storage, item.copy)

    def _publish(
        self, entry: _Registration, request: _Request, held: dict[int, _Registration], created: list[_Registration],
    ) -> None:
        _live_managers.add(self)
        self._entries[entry.pointer] = entry
        created.append(entry)
        self._hold(entry, request, held)

    def _discard(self, item: _Pending) -> None:
        """Take back pending storage's reservation and range, freeing its copy."""
        self._starts.pop(bisect_left(self._starts, item.pointer))
        if isinstance(item, _InPlace):
            self._unreserve_range(item.pointer, item.request.storage.nbytes())
        else:
            self._pinned_bytes -= item.copy.size
            item.copy.free()

    def _abandon(self, pending: list[_Pending]) -> None:
        """Discard everything still pending after a failure."""
        for item in pending:
            self._discard(item)
            del self._pending[item.pointer]
        self._settled.notify_all()

    def _register(self, pointer: int, size: int) -> bool:
        """Register natively, reclaiming idle LRU batches while the runtime refuses capacity."""
        registered = self._try_register(pointer, size)
        while not registered and self._reclaim_idle_for_native_retry(size):
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

    def _reclaim_idle_for_native_retry(self, size: int) -> bool:
        """Evict an LRU batch before retrying a native-capacity failure."""
        target = max(mmap.PAGESIZE, _page_rounded(size))
        before = self._pinned_bytes
        self._evict_idle(lambda: before - self._pinned_bytes >= target)
        return self._pinned_bytes < before

    def _make_room(self, needed: Callable[[], int]) -> bool:
        """Fit ``needed()`` more bytes under a finite limit, evicting idle entries in LRU order.

        The reservation is re-evaluated after each eviction: an in-place range's
        boundary page stops being shared once its neighbour is gone.
        """
        limit = self._max_pinned_bytes
        if limit is None:
            return True
        if needed() > limit:
            return False

        def fits() -> bool:
            return self._pinned_bytes + needed() <= limit

        self._evict_idle(fits)
        return fits()

    def _evict_idle(self, until: Callable[[], bool]) -> None:
        """Unregister idle entries, least recently released first, until ``until()`` holds."""
        for candidate in tuple(self._idle):
            if until():
                return
            entry = self._entries.get(candidate)
            if entry is not None:
                self._unregister(entry)

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
            self._backend.unregister(entry.registered)
        except Exception as error:
            self._unregistration_failures += 1
            # Tracebacks in buffered logs can retain storage after a later retry.
            logger.warning("Host unregistration failed; retaining storage and budget reservation: %s", str(error))
            return False
        del self._entries[entry.pointer]
        self._starts.pop(bisect_left(self._starts, entry.pointer))
        self._idle.pop(entry.pointer, None)
        if entry.copy is None:
            self._unreserve_range(entry.pointer, entry.size)
        else:
            self._pinned_bytes -= entry.copy.size
            entry.copy.free()
        self._drop_lifetime_root_if_empty()
        return True

    def _release(self, entries: tuple[_Registration, ...]) -> None:
        for entry in entries:
            entry.leases -= 1
            if entry.leases == 0:
                self._idle[entry.pointer] = None
                if entry.retired:
                    self._unregister(entry)
        self._make_room(lambda: 0)

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
