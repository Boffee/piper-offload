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
import enum
import logging
import mmap
import threading
import weakref
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from typing import Self

import torch

from . import _host_memory as host_memory
from ._copy_memory import Copy, CopyLoad, _free_copies
from ._host_registration import HostRegistrationBackend, HostRegistrationRefusedError, RuntimeHostRegistration
from .checkpoint import FileSlice, file_slice

logger = logging.getLogger(__name__)


def _default_pin_budget() -> int:
    """Half of the memory available to the process, rounded down to OS pages, without touching CUDA."""
    try:
        total = host_memory.available_memory()
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


@dataclass(eq=False, slots=True)
class _InPlace:
    """In-place storage reserved under the budget for one request, registered with the rest of its acquisition."""

    pointer: int
    request: _Request


@dataclass(eq=False, slots=True)
class _PendingCopy:
    pointer: int
    request: _Request
    load: CopyLoad


type _Pending = _InPlace | _PendingCopy


class _Refusal(enum.Enum):
    STORAGE = "storage"  # this storage stays pageable; a later one in the acquisition may register
    CAPACITY = "capacity"  # the runtime's capacity is exhausted; stop registering


@dataclass(eq=False)
class _Registration:
    pointer: int
    size: int
    storage: torch.UntypedStorage
    copy: Copy | None = None
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
        # Capture the dictionaries rather than self: the memory component's
        # commitment watcher must not retain the manager.
        registrations, pending = self._registrations, self._pending
        self._memory = host_memory.Memory(
            max_offered_bytes, self._lock, lambda pointer: pointer in registrations or pointer in pending,
        )

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
        releases = self._memory.release_batch()
        try:
            with self._lock, releases:
                self._max_pinned_bytes = value
                self._fit(lambda: 0)
        finally:
            releases.finish()

    @property
    def max_offered_bytes(self) -> int:
        with self._lock:
            return self._memory.max_offered_bytes

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
            released = self._memory.resize(value)
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
                + sum(item.load.copy.size for item in self._pending.values() if isinstance(item, _PendingCopy)),
                self._memory.max_offered_bytes,
                self._memory.offered_bytes,
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
        Windows kept intact is not read but registers as soon as it is back,
        while the rest are still being reclaimed; everything else registers in
        request order. A runtime capacity failure evicts unrelated idle
        registrations and retries; if capacity remains unavailable, the
        storage this acquisition has not registered yet skips registration. A
        rejected range alone stays pageable, without evicting other
        registrations or skipping later storage.
        """
        requests = self._requests(tensors)
        held: dict[int, _Registration] = {}
        created: list[_Registration] = []
        pending: list[_Pending] = []
        releases = self._memory.release_batch()
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
                    with releases:
                        self._reserve(requests, held, pending)
                prepared = self._memory.prepare([item.load for item in pending if isinstance(item, _PendingCopy)])
            # What was evicted to make room is offered before anything is read.
            releases.finish()
            exhausted = self._prepare_copies(prepared, pending, held, created)
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
            releases.finish()
            raise

    def _prepare_copies(
        self,
        prepared: Generator[CopyLoad | None],
        pending: list[_Pending],
        held: dict[int, _Registration],
        created: list[_Registration],
    ) -> bool:
        """Register copies as memory prepares them, off the lock; whether the runtime's capacity ran out.

        A ready copy that preparation hands back registers at once, ahead of
        earlier storage still waiting for its fill, because registering locks
        its pages where that fill would otherwise push them out of the
        working set: reacquiring an 18 GiB checkpoint whose fill was 6 GiB
        took 19.4 s with the intact copies registered after the fill and
        14.1 s before it, the registration itself 5.3 s against 0.5.
        """
        waiting = {item.load: item for item in pending if isinstance(item, _PendingCopy)}
        with contextlib.closing(prepared):
            for load in prepared:
                early = waiting[load] if load is not None and load.ready else None
                with self._lock:
                    if self._register_pending(pending, held, created, early):
                        return True
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
        released: list[Copy] = []
        with self._lock:
            failed = 0
            for pointer in tuple(self._idle):
                registration = self._registrations.get(pointer)
                if registration is not None and not self._unregister(registration, released):
                    failed += 1
            released.extend(self._memory.clear())
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
        """Take a copy from memory after making room under the pinned budget."""
        load = self._memory.reserve(
            pointer, _page_rounded(request.storage.nbytes()), source, lambda size: self._fit(lambda: size),
        )
        if load is None:
            return None
        self._pinned_bytes += load.copy.size
        return _PendingCopy(pointer, request, load)

    def _register_pending(
        self,
        pending: list[_Pending],
        held: dict[int, _Registration],
        created: list[_Registration],
        early: _Pending | None = None,
    ) -> bool:
        """Register ``early``, then the ready prefix in request order; return whether runtime capacity is exhausted.

        Refused storage is evicted. Capacity exhaustion stops the pass; the
        caller joins workers before discarding the remaining pending storage.
        Memory preparation publishes readiness after joining the worker, so
        registration never races a worker.
        """
        exhausted = early is not None and self._settle_pending(early, pending, held, created)
        while not exhausted and pending:
            item = pending[0]
            if isinstance(item, _PendingCopy) and not item.load.ready:
                break
            exhausted = self._settle_pending(item, pending, held, created)
        self._pending_changed.notify_all()
        return exhausted

    def _settle_pending(
        self,
        item: _Pending,
        pending: list[_Pending],
        held: dict[int, _Registration],
        created: list[_Registration],
    ) -> bool:
        """Register or evict one pending storage and take it out of ``pending``; whether runtime capacity ran out."""
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
        del self._pending[item.pointer]
        pending.remove(item)
        return outcome is _Refusal.CAPACITY

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
        if item.load.error is not None:
            logger.warning("Could not fill a pinned copy from the checkpoint; it stays pageable: %s", item.load.error)
            return _Refusal.STORAGE
        if not self._register(item.load.copy.pointer, item.load.copy.size):
            return _Refusal.CAPACITY
        return _Registration(item.pointer, size, item.request.storage, item.load.copy)

    def _evict_pending(self, item: _Pending) -> None:
        """Take back pending storage's reservation and range, freeing its copy."""
        self._starts.pop(bisect_left(self._starts, item.pointer))
        if isinstance(item, _InPlace):
            self._unreserve_range(item.pointer, item.request.storage.nbytes())
        else:
            self._pinned_bytes -= item.load.copy.size
            item.load.copy.free()

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
                    else:
                        manager._memory.retire(registration)

        for tensor in request.tensors:
            if id(tensor) not in registration.owners:
                registration.owners[id(tensor)] = weakref.ref(tensor, owner_gone)

    def _unregister(
        self, registration: _Registration, released: list[Copy] | None = None, *, demote: bool = False,
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
            copies = self._memory.release(registration, demote=demote)
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
