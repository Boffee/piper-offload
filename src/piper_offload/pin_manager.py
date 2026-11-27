"""Budgeted host registrations, Tier 2a/2b slab arenas, Direct I/O, and bounce staging.

Use the process-wide ``host_pin_manager`` for application registrations. Its
default budgets are derived from physical RAM (and on Linux the tightest cgroup
memory limit), rounded down to whole OS pages: 70% pinned / 80% total host
memory on Linux, and 45% pinned / 75% total host memory on Windows WDDM. Set
``max_pinned_bytes`` to zero to disable registration or ``None`` to remove the
budget cap and register up to the CUDA/HIP runtime's capacity. Set
``max_host_bytes`` to bound combined Tier 2a (pinned) + Tier 2b (private
pageable) slab reservations, or ``None`` for no private host ceiling.
Construction and configuration perform no CUDA initialization. Isolated
``PinManager`` instances can use an injected backend for testing.

Every host transfer runs under a lease, so nothing is read or written while
it could be evicted, and only transfers that repeat ask the lease to pin:
streaming/budgeted unit offloading and the relay register their storage; a
one-shot resident upload leases it pageable.

Storage without checkpoint file provenance registers in place using whole
storage byte ranges (preserving alias shared pages and in-place mutation
for relay buffers). Storage that records a checkpoint file slice
(:func:`file_slice`) is never pinned in place: the mapping stays read-only
page cache, and pinning allocates an owned span inside a pre-pinned Tier 2a
``HostSlabArena`` filled via ``DirectFileReader`` (``O_DIRECT`` /
``FILE_FLAG_NO_BUFFERING`` with buffered fallback). Evicting an idle Tier 2a
copy under pin pressure demotes it to a Tier 2b private pageable
``HostSlabArena`` when ``max_host_bytes`` allows, and a later pinning
acquisition promotes it back in RAM without re-reading disk. Dropping the last
tensor owner retires the entry and releases empty slabs once active leases
close.

Transfers go through :func:`transfer_`, which reuses pre-resolved views on
open leases without acquiring the manager lock when pinned, or streams
Tier 2b / uncached checkpoint / pageable CUDA copies through a lazily
initialized ``PinnedBounceRing`` when eligible (direct ``Disk -> Bounce Ring
-> GPU`` leaves zero extra bytes in host RAM). An asynchronous transfer of
pinned storage outside a lease raises, while a synchronous transfer completes
cleanly.
"""

from __future__ import annotations

import contextlib
import logging
import mmap
import threading
import weakref
from bisect import bisect_left
from collections import OrderedDict, deque
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Self

import torch

from . import _host_memory as host_memory
from . import _transfer_priority as transfer_priority
from ._direct_io import (
    DEFAULT_COALESCE_THRESHOLD,
    DirectFileReader,
    DirectReadRequest,
    align_span,
)
from ._host_registration import (
    HostRegistrationBackend,
    HostRegistrationRefusedError,
    RuntimeHostRegistration,
)
from .checkpoint import FileSlice, file_slice
from .host_arena import (
    _MIN_BOUNCE_SLOT_BYTES,
    DEFAULT_BOUNCE_SLOT_BYTES,
    DEFAULT_BOUNCE_SLOTS,
    DEFAULT_MIN_BOUNCE_BYTES,
    DEFAULT_SLAB_BYTES,
    ArenaAllocation,
    HostSlabArena,
    PinnedBounceRing,
    _align_down,
    _align_up,
)

logger = logging.getLogger(__name__)

_DEFAULT_PIN_BUDGET = host_memory.default_pin_budget()
_DEFAULT_HOST_BUDGET = host_memory.default_host_budget()


def _validate_non_negative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_optional_non_negative_int(value: int | None, name: str) -> None:
    if value is not None:
        _validate_non_negative_int(value, name)


def _page_rounded(nbytes: int) -> int:
    return -(-nbytes // mmap.PAGESIZE) * mmap.PAGESIZE


@dataclass(frozen=True, slots=True)
class PinStats:
    """Registration counts and host memory accounting across Tier 2a, Tier 2b, and bounce ring.

    ``pinned_bytes`` equals in-place pinned OS pages plus ``copy_bytes``
    (Tier 2a pinned slab reservation) plus ``bounce_bytes`` (active/retained
    ``PinnedBounceRing`` reservation). ``private_copy_bytes`` counts Tier 2b
    private pageable slab reservations under the combined ``max_host_bytes``
    ceiling.
    """

    max_pinned_bytes: int | None
    pinned_bytes: int
    registrations: int
    idle_registrations: int
    active_leases: int
    registration_failures: int
    unregistration_failures: int
    copy_bytes: int
    max_host_bytes: int | None
    private_copy_bytes: int
    bounce_bytes: int


@dataclass(slots=True, weakref_slot=True)
class _OwnerTracker:
    """Shared retirement bit across Tier 2a <-> Tier 2b promotions and demotions."""

    retired: bool = False


@dataclass(eq=False)
class _Registration:
    pointer: int
    size: int
    storage: torch.UntypedStorage
    copy: ArenaAllocation | None = None
    owners: dict[int, weakref.ReferenceType[torch.Tensor]] = field(default_factory=dict)
    tracker: _OwnerTracker = field(default_factory=_OwnerTracker)
    leases: int = 0

    @property
    def retired(self) -> bool:
        return self.tracker.retired

    @retired.setter
    def retired(self, value: bool) -> None:
        self.tracker.retired = value

    @property
    def registered(self) -> int:
        """The pointer registered with the backend: the copy's, or the storage's own."""
        return self.pointer if self.copy is None else self.copy.pointer


@dataclass(eq=False)
class _PrivateCopy:
    """Warm Tier 2b private pageable copy of a demoted checkpoint storage."""

    pointer: int
    size: int
    storage: torch.UntypedStorage
    alloc: ArenaAllocation
    owners: dict[int, weakref.ReferenceType[torch.Tensor]]
    tracker: _OwnerTracker
    leases: int = 0

    @property
    def retired(self) -> bool:
        return self.tracker.retired


@dataclass(slots=True)
class _Request:
    storage: torch.UntypedStorage
    tensors: list[torch.Tensor]


@dataclass(slots=True)
class _Pageable:
    size: int
    leases: int = 0


@dataclass(slots=True)
class _Candidate:
    pointer: int
    request: _Request
    size: int
    source: FileSlice | None
    sector: int = 0
    head_pad: int = 0
    aligned_size: int = 0
    use_direct_aligned: bool = False
    private_source: _PrivateCopy | None = None


@dataclass(frozen=True, slots=True)
class _CandidateRun:
    candidates: Sequence[_Candidate]
    start: int
    end: int


@dataclass(slots=True)
class _PreparedCopy:
    pointer: int
    request: _Request
    source: FileSlice
    alloc: ArenaAllocation
    head_pad: int
    use_direct_aligned: bool
    private_source: _PrivateCopy | None = None
    filled: bool = False
    promoted_owners: dict[int, weakref.ReferenceType[torch.Tensor]] | None = None
    promoted_tracker: _OwnerTracker | None = None
    error: str | None = None


class _Tier2aBackendProxy:
    """Backend wrapper tracking ``HostRegistrationRefusedError`` for Tier 2a per-range classification."""

    __slots__ = ("_backend", "_lock", "refused_count")

    def __init__(self, backend: HostRegistrationBackend) -> None:
        self._backend = backend
        self._lock = threading.Lock()
        self.refused_count = 0

    def register(self, pointer: int, size: int) -> bool:
        try:
            return self._backend.register(pointer, size)
        except HostRegistrationRefusedError:
            with self._lock:
                self.refused_count += 1
            raise

    def unregister(self, pointer: int) -> None:
        self._backend.unregister(pointer)


@dataclass(slots=True)
class _ManagerResources:
    """GC-safe owner holding native resources without a back-reference to ``PinManager``."""

    reader: DirectFileReader
    tier2a: HostSlabArena
    tier2b: HostSlabArena
    bounce_ring: PinnedBounceRing | None = None
    creating_ring: PinnedBounceRing | None = None
    closing_rings: list[PinnedBounceRing] = field(default_factory=list)
    retaining_rings: list[PinnedBounceRing] = field(default_factory=list)


def _finalize_manager_resources(res: _ManagerResources) -> None:
    with contextlib.suppress(Exception):
        res.reader.close()
    rings: list[PinnedBounceRing] = [
        *res.retaining_rings,
        *res.closing_rings,
    ]
    res.retaining_rings.clear()
    res.closing_rings.clear()
    if res.creating_ring is not None:
        rings.append(res.creating_ring)
        res.creating_ring = None
    if res.bounce_ring is not None:
        rings.append(res.bounce_ring)
        res.bounce_ring = None
    for ring in rings:
        with contextlib.suppress(Exception):
            ring.close()
    with contextlib.suppress(Exception):
        res.tier2b.close()
    with contextlib.suppress(Exception):
        res.tier2a.close()


class _SortedPointers:
    """Sorted set of unique storage start addresses with O(log N) neighbor queries and bounded buckets."""

    __slots__ = ("_buckets", "_maxes")
    _LOAD = 256

    def __init__(self) -> None:
        self._buckets: list[list[int]] = []
        self._maxes: list[int] = []

    def add(self, pointer: int) -> None:
        if not self._buckets:
            self._buckets.append([pointer])
            self._maxes.append(pointer)
            return
        bucket_idx = bisect_left(self._maxes, pointer)
        if bucket_idx == len(self._buckets):
            bucket_idx -= 1
            bucket = self._buckets[bucket_idx]
            bucket.append(pointer)
            self._maxes[bucket_idx] = pointer
        else:
            bucket = self._buckets[bucket_idx]
            pos = bisect_left(bucket, pointer)
            if pos < len(bucket) and bucket[pos] == pointer:
                return
            bucket.insert(pos, pointer)
        if len(bucket) > self._LOAD * 2:
            mid = len(bucket) // 2
            right = bucket[mid:]
            del bucket[mid:]
            self._maxes[bucket_idx] = bucket[-1]
            self._buckets.insert(bucket_idx + 1, right)
            self._maxes.insert(bucket_idx + 1, right[-1])

    def remove(self, pointer: int) -> None:
        if not self._buckets:
            raise KeyError(pointer)
        bucket_idx = bisect_left(self._maxes, pointer)
        if bucket_idx == len(self._buckets):
            raise KeyError(pointer)
        bucket = self._buckets[bucket_idx]
        pos = bisect_left(bucket, pointer)
        if pos == len(bucket) or bucket[pos] != pointer:
            raise KeyError(pointer)
        bucket.pop(pos)
        if not bucket:
            del self._buckets[bucket_idx]
            del self._maxes[bucket_idx]
            return
        self._maxes[bucket_idx] = bucket[-1]
        if len(bucket) < self._LOAD // 2 and len(self._buckets) > 1:
            if bucket_idx + 1 < len(self._buckets) and len(bucket) + len(self._buckets[bucket_idx + 1]) <= self._LOAD:
                right = self._buckets.pop(bucket_idx + 1)
                del self._maxes[bucket_idx + 1]
                bucket.extend(right)
                self._maxes[bucket_idx] = bucket[-1]
            elif bucket_idx > 0 and len(self._buckets[bucket_idx - 1]) + len(bucket) <= self._LOAD:
                left = self._buckets[bucket_idx - 1]
                left.extend(bucket)
                del self._buckets[bucket_idx]
                del self._maxes[bucket_idx]
                self._maxes[bucket_idx - 1] = left[-1]

    def neighbors(self, pointer: int) -> tuple[int, ...]:
        if not self._buckets:
            return ()
        bucket_idx = bisect_left(self._maxes, pointer)
        if bucket_idx == len(self._buckets):
            return (self._maxes[-1],)
        bucket = self._buckets[bucket_idx]
        pos = bisect_left(bucket, pointer)
        current = bucket[pos]
        if pos > 0:
            return (bucket[pos - 1], current)
        if bucket_idx > 0:
            return (self._maxes[bucket_idx - 1], current)
        return (current,)


@dataclass(slots=True)
class _LeaseState:
    registrations: deque[_Registration]
    private_copies: deque[_PrivateCopy]
    pageable: deque[int]
    tensors: tuple[torch.Tensor, ...]
    view_ids: deque[int]
    retired_registrations: deque[_Registration] = field(default_factory=deque)


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
        self._manager: PinManager | None = manager
        self._key = key
        self._closed = False
        self._finalizer = weakref.finalize(self, manager._close_lease, key)
        self._finalizer.atexit = False

    @property
    def closed(self) -> bool:
        return self._closed or not self._finalizer.alive

    def close(self) -> None:
        """Close the lease, releasing what it holds; idempotent and retry-safe."""
        if self._closed:
            return
        manager = self._manager
        if manager is not None:
            manager._close_lease(self._key)
        self._closed = True
        self._finalizer.detach()
        self._manager = None

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
    """Own pinned registrations, Tier 2a/2b host slab arenas, Direct I/O reader, and bounce ring.

    A finite ``max_pinned_bytes`` bounds pinned pages in this process across
    in-place registrations, Tier 2a checkpoint slabs, and the lazy
    ``PinnedBounceRing``; ``None`` removes the ceiling and evicts idle
    registrations when the CUDA/HIP runtime refuses capacity. A finite
    ``max_host_bytes`` bounds total managed host RAM (``pinned_bytes +
    private_copy_bytes``) for Tier 2b demotion retention.

    Strict lock ordering: ``self._lock`` protects only metadata transitions and
    never wraps ``HostSlabArena``, ``ArenaAllocation.release()``,
    ``DirectFileReader``, or ``PinnedBounceRing`` operations.
    """

    @staticmethod
    def _validate_init_args(
        max_pinned_bytes: int | None,
        max_host_bytes: int | None,
        slab_bytes: int,
        bounce_bytes: int | None,
        bounce_slot_bytes: int,
        min_bounce_bytes: int,
    ) -> None:
        _validate_optional_non_negative_int(max_pinned_bytes, "max_pinned_bytes")
        _validate_optional_non_negative_int(max_host_bytes, "max_host_bytes")
        _validate_optional_non_negative_int(bounce_bytes, "bounce_bytes")
        _validate_non_negative_int(min_bounce_bytes, "min_bounce_bytes")
        for value, name in ((slab_bytes, "slab_bytes"), (bounce_slot_bytes, "bounce_slot_bytes")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if value % mmap.PAGESIZE != 0:
                raise ValueError(f"{name} must be a multiple of {mmap.PAGESIZE}")

    def __init__(
        self,
        max_pinned_bytes: int | None = _DEFAULT_PIN_BUDGET,
        *,
        max_host_bytes: int | None = _DEFAULT_HOST_BUDGET,
        slab_bytes: int = DEFAULT_SLAB_BYTES,
        bounce_bytes: int | None = None,
        bounce_slot_bytes: int = DEFAULT_BOUNCE_SLOT_BYTES,
        min_bounce_bytes: int = DEFAULT_MIN_BOUNCE_BYTES,
        backend: HostRegistrationBackend | None = None,
    ) -> None:
        self._validate_init_args(
            max_pinned_bytes, max_host_bytes, slab_bytes, bounce_bytes, bounce_slot_bytes, min_bounce_bytes
        )

        self._max_pinned_bytes = max_pinned_bytes
        self._max_host_bytes = max_host_bytes
        self._slab_bytes = slab_bytes
        self._bounce_bytes_config = bounce_bytes
        self._bounce_slot_bytes = bounce_slot_bytes
        self._min_bounce_bytes = min_bounce_bytes
        self._backend = backend if backend is not None else RuntimeHostRegistration()

        self._lock = threading.RLock()
        self._registrations: dict[int, _Registration] = {}
        self._private_copies: OrderedDict[int, _PrivateCopy] = OrderedDict()
        self._pageable: dict[int, _Pageable] = {}
        self._pending: dict[int, _Request] = {}
        self._pending_changed = threading.Condition(self._lock)
        self._starts = _SortedPointers()
        self._idle: OrderedDict[int, None] = OrderedDict()
        # Only in-place ranges share sub-page endpoints; Tier 2a slabs are whole pages.
        self._boundary_pages: dict[int, int] = {}
        self._in_place_pinned_bytes = 0
        self._registration_failures = 0
        self._unregistration_failures = 0
        self._leases: dict[int, _LeaseState] = {}
        # Maps id(source_tensor) -> (resolved_view, refcount, is_pinned) for lock-free async fast paths.
        self._leased_views: dict[int, tuple[torch.Tensor, int, bool]] = {}
        self._next_lease = 0

        self._bounce_users = 0
        self._bounce_creating = False
        self._bounce_closing = False
        self._bounce_publishing = False
        self._pin_reserving = 0
        self._lease_close_seq = 0
        self._bounce_cv = threading.Condition(self._lock)
        self._pin_budget_epoch = 0
        self._host_budget_epoch = 0
        self._retired_bounce_reg_failures = 0
        self._retired_bounce_unreg_failures = 0

        self._tier2a_backend = _Tier2aBackendProxy(self._backend)
        self._resources = _ManagerResources(
            reader=DirectFileReader(),
            tier2a=HostSlabArena(
                pinned=True,
                max_bytes=max_pinned_bytes,
                slab_bytes=slab_bytes,
                backend=self._tier2a_backend,
            ),
            tier2b=HostSlabArena(
                pinned=False,
                max_bytes=max_host_bytes,
                slab_bytes=slab_bytes,
            ),
        )
        self._reader = self._resources.reader
        self._tier2a = self._resources.tier2a
        self._tier2b = self._resources.tier2b
        self._finalizer: weakref.finalize[[_ManagerResources], Any] = weakref.finalize(
            self, _finalize_manager_resources, self._resources
        )
        self._finalizer.atexit = False

    def _desired_bounce_bytes(self, pin_budget: int | None) -> int | None:
        """Compute target bounce-ring bytes under ``pin_budget``, ``0`` when disabled, or ``None`` if unbounded."""
        if pin_budget == 0 or self._bounce_bytes_config == 0:
            return 0
        if self._bounce_bytes_config is not None:
            target = (
                self._bounce_bytes_config
                if pin_budget is None
                else min(self._bounce_bytes_config, pin_budget)
            )
            aligned = _align_down(target, mmap.PAGESIZE)
            return 0 if aligned < _MIN_BOUNCE_SLOT_BYTES else aligned
        if pin_budget is None:
            return None
        if pin_budget < 4 * self._min_bounce_bytes:
            return 0
        quarter = _align_down(pin_budget // 4, mmap.PAGESIZE)
        target = min(DEFAULT_BOUNCE_SLOTS * self._bounce_slot_bytes, quarter)
        return 0 if target < _MIN_BOUNCE_SLOT_BYTES else target

    def _active_bounce_bytes_off_lock(self) -> tuple[int, int, int]:
        """Return ``(reserved, reg_fails, unreg_fails)`` across live, creating, closing, and retained rings."""
        with self._lock:
            rings: list[PinnedBounceRing] = [
                *self._resources.retaining_rings,
                *self._resources.closing_rings,
            ]
            if self._resources.creating_ring is not None:
                rings.append(self._resources.creating_ring)
            if self._resources.bounce_ring is not None:
                rings.append(self._resources.bounce_ring)
            retired_reg = self._retired_bounce_reg_failures
            retired_unreg = self._retired_bounce_unreg_failures
        reserved = 0
        reg_fails = retired_reg
        unreg_fails = retired_unreg
        for ring in rings:
            reserved += ring.reserved_bytes
            reg_fails += ring.registration_failures
            unreg_fails += ring.unregistration_failures
        return reserved, reg_fails, unreg_fails

    @property
    def max_pinned_bytes(self) -> int | None:
        with self._lock:
            return self._max_pinned_bytes

    @max_pinned_bytes.setter
    def max_pinned_bytes(self, value: int | None) -> None:
        """Set the pin budget, or ``None`` to register up to the runtime's capacity."""
        _validate_optional_non_negative_int(value, "max_pinned_bytes")
        with self._lock:
            self._max_pinned_bytes = value
            self._pin_budget_epoch += 1
        self._enforce_pin_budget()

    @property
    def max_host_bytes(self) -> int | None:
        with self._lock:
            return self._max_host_bytes

    @max_host_bytes.setter
    def max_host_bytes(self, value: int | None) -> None:
        """Set the combined Tier 2a + Tier 2b host ceiling, or ``None`` for unbounded Tier 2b."""
        _validate_optional_non_negative_int(value, "max_host_bytes")
        with self._lock:
            self._max_host_bytes = value
            self._host_budget_epoch += 1
        self._enforce_host_budget()

    @property
    def stats(self) -> PinStats:
        with self._lock:
            max_pinned = self._max_pinned_bytes
            max_host = self._max_host_bytes
            in_place = self._in_place_pinned_bytes
            registrations = len(self._registrations)
            idle = len(self._idle)
            active_leases = len(self._leases)
            in_place_reg_fails = self._registration_failures
            in_place_unreg_fails = self._unregistration_failures

        tier2a_stats = self._tier2a.stats
        tier2b_stats = self._tier2b.stats
        bounce_reserved, bounce_reg_fails, bounce_unreg_fails = self._active_bounce_bytes_off_lock()
        copy_bytes = tier2a_stats.reserved_bytes
        private_bytes = tier2b_stats.reserved_bytes
        return PinStats(
            max_pinned_bytes=max_pinned,
            pinned_bytes=in_place + copy_bytes + bounce_reserved,
            registrations=registrations,
            idle_registrations=idle,
            active_leases=active_leases,
            registration_failures=in_place_reg_fails + tier2a_stats.registration_failures + bounce_reg_fails,
            unregistration_failures=(
                in_place_unreg_fails + tier2a_stats.unregistration_failures + bounce_unreg_fails
            ),
            copy_bytes=copy_bytes,
            max_host_bytes=max_host,
            private_copy_bytes=private_bytes,
            bounce_bytes=bounce_reserved,
        )

    def acquire(self, tensors: Iterable[torch.Tensor], *, pin: bool = True) -> PinLease:
        """Lease whole storages for a transfer, registering them only if ``pin``."""
        requests = self._requests(tensors)
        held: dict[int, _Registration] = {}
        held_private: dict[int, _PrivateCopy] = {}
        created: list[_Registration] = []
        pageable: list[int] = []
        pending_keys: list[int] = []
        candidate_list: list[_Candidate] = []
        prepared_copies: list[_PreparedCopy] = []
        deferred_2a: list[ArenaAllocation] = []
        deferred_2b: list[ArenaAllocation] = []
        start_pin_epoch = 0

        try:
            early_ready = False
            with self._lock:
                start_pin_epoch = self._pin_budget_epoch
                early_lease, early_ready = self._prepare_acquire_locked(
                    requests,
                    pin=pin,
                    held=held,
                    held_private=held_private,
                    pageable=pageable,
                    candidate_list=candidate_list,
                    pending_keys=pending_keys,
                )
                if early_lease is not None:
                    return early_lease
            if early_ready:
                return self._finish_lease_off_lock(requests, held, held_private)

            effective_ceiling = self._plan_candidates(candidate_list, held)
            self._allocate_and_reserve_candidates(
                candidate_list,
                effective_ceiling,
                held,
                created,
                prepared_copies,
                pending_keys,
            )
            self._fill_prepared_copies(prepared_copies)
            lease, pin_epoch_changed = self._commit_acquire(
                requests,
                prepared_copies,
                held,
                held_private,
                created,
                pageable,
                pending_keys,
                deferred_2a,
                deferred_2b,
                start_pin_epoch,
            )
        except BaseException:
            self._rollback_acquire(
                prepared_copies,
                candidate_list,
                pending_keys,
                created,
                held,
                held_private,
                pageable,
                deferred_2a,
                deferred_2b,
                start_pin_epoch,
            )
            raise

        try:
            if deferred_2a or deferred_2b:
                self._drop_detached_allocations(
                    deferred_2a,
                    deferred_2b,
                    trim_2a=bool(deferred_2a),
                    trim_2b=bool(deferred_2b),
                )
            if pin_epoch_changed:
                self._enforce_pin_budget()
            else:
                self._enforce_host_budget()
        except BaseException:
            try:
                if deferred_2a or deferred_2b:
                    self._drop_detached_allocations(
                        deferred_2a,
                        deferred_2b,
                        trim_2a=bool(deferred_2a),
                        trim_2b=bool(deferred_2b),
                    )
            finally:
                lease.close()
            raise
        return lease

    @staticmethod
    def _needs_off_lock_views(
        held: dict[int, _Registration],
        held_private: dict[int, _PrivateCopy],
    ) -> bool:
        return bool(held_private) or any(reg.copy is not None for reg in held.values())

    def _finish_lease_off_lock(
        self,
        requests: dict[int, _Request],
        held: dict[int, _Registration],
        held_private: dict[int, _PrivateCopy],
    ) -> PinLease:
        built_views: dict[int, torch.Tensor] = {}
        try:
            self._build_lease_views_off_lock(requests, held, held_private, built_views)
            with self._lock:
                return self._commit_lease_locked(requests, held, held_private, built_views)
        except BaseException:
            built_views.clear()
            raise

    def _maybe_commit_early_locked(
        self,
        requests: dict[int, _Request],
        held: dict[int, _Registration],
        held_private: dict[int, _PrivateCopy],
    ) -> tuple[PinLease | None, bool]:
        if self._needs_off_lock_views(held, held_private):
            return None, True
        return self._commit_lease_locked(requests, held, held_private, {}), True

    def _prepare_acquire_locked(
        self,
        requests: dict[int, _Request],
        *,
        pin: bool,
        held: dict[int, _Registration],
        held_private: dict[int, _PrivateCopy],
        pageable: list[int],
        candidate_list: list[_Candidate],
        pending_keys: list[int],
    ) -> tuple[PinLease | None, bool]:
        if self._try_hold_registered_locked(requests, held):
            return self._maybe_commit_early_locked(requests, held, held_private)
        while self._pending and (pin or not self._pending.keys().isdisjoint(requests)):
            self._pending_changed.wait()
        self._validate_ranges_locked(requests)
        for pointer, request in requests.items():
            registration = self._registrations.get(pointer)
            if registration is not None:
                self._hold_locked(registration, request, held)
            else:
                self._hold_pageable_locked(pointer, request)
                pageable.append(pointer)

        if not pin or self._max_pinned_bytes == 0:
            self._hold_warm_private_copies_locked(pageable, requests, held_private)
            return self._maybe_commit_early_locked(requests, held, held_private)

        for pointer, request in requests.items():
            if pointer in held or self._pageable[pointer].leases > 1:
                continue
            candidate_list.append(
                _Candidate(
                    pointer=pointer,
                    request=request,
                    size=request.storage.nbytes(),
                    source=file_slice(request.storage),
                )
            )
        if not candidate_list:
            self._hold_warm_private_copies_locked(pageable, requests, held_private)
            return self._maybe_commit_early_locked(requests, held, held_private)

        for candidate in candidate_list:
            self._pending[candidate.pointer] = candidate.request
            pending_keys.append(candidate.pointer)
        return None, False

    def _commit_acquire(
        self,
        requests: dict[int, _Request],
        prepared_copies: list[_PreparedCopy],
        held: dict[int, _Registration],
        held_private: dict[int, _PrivateCopy],
        created: list[_Registration],
        pageable: list[int],
        pending_keys: list[int],
        deferred_2a: list[ArenaAllocation],
        deferred_2b: list[ArenaAllocation],
        start_pin_epoch: int,
    ) -> tuple[PinLease, bool]:
        with self._lock:
            self._commit_prepared_copies_locked(
                prepared_copies,
                held,
                created,
                deferred_2a,
                deferred_2b,
            )
            unheld_pageable = [p for p in pageable if p not in held]
            self._hold_warm_private_copies_locked(unheld_pageable, requests, held_private)

        built_views: dict[int, torch.Tensor] = {}
        try:
            self._build_lease_views_off_lock(requests, held, held_private, built_views)
            with self._lock:
                lease = self._commit_lease_locked(
                    requests,
                    held,
                    held_private,
                    built_views,
                    pending_keys=pending_keys,
                )
                return lease, self._pin_budget_epoch != start_pin_epoch
        except BaseException:
            built_views.clear()
            raise

    def _plan_candidates(
        self,
        candidates: list[_Candidate],
        held: dict[int, _Registration],
    ) -> int | None:
        """Compute alignment metadata, pin warm Tier 2b promotion sources, and determine headroom."""
        with self._lock:
            budget = self._max_pinned_bytes
            held_in_place = sum(
                self._page_reservation(reg.pointer, reg.size)
                for reg in held.values()
                if reg.copy is None
            )
            held_copy_approx = sum(
                _page_rounded(reg.size) for reg in held.values() if reg.copy is not None
            )
            for candidate in candidates:
                if candidate.source is None:
                    continue
                priv = self._private_copies.get(candidate.pointer)
                if priv is not None and priv.size == candidate.size and not priv.retired:
                    priv.leases += 1
                    self._private_copies.move_to_end(candidate.pointer)
                    candidate.private_source = priv
                    candidate.aligned_size = _page_rounded(candidate.size)
            ring_active = self._resources.bounce_ring is not None

        for candidate in candidates:
            if candidate.source is None or candidate.private_source is not None:
                continue
            sector = self._reader.sector_size(candidate.source.file)
            candidate.sector = sector
            span = align_span(candidate.source.offset, candidate.source.length, sector_size=sector)
            candidate.use_direct_aligned = span.head_pad == 0 or (
                span.aligned_length > DEFAULT_COALESCE_THRESHOLD
                and (budget is None or span.aligned_length <= budget)
            )
            candidate.head_pad = span.head_pad if candidate.use_direct_aligned else 0
            sub_align = max(sector, mmap.PAGESIZE)
            candidate.aligned_size = _align_up(candidate.head_pad + candidate.source.length, sub_align)

        if budget is None:
            return None
        if ring_active:
            return budget
        auto_bounce = self._desired_bounce_bytes(budget)
        if not auto_bounce:
            return budget

        total_demand = held_in_place + held_copy_approx + sum(
            _page_rounded(c.size) if c.source is None else c.aligned_size for c in candidates
        )
        # Reserve bounce headroom only when the pinning request would overflow the pin budget.
        effective = max(0, budget - auto_bounce) if total_demand > budget else budget
        for candidate in candidates:
            if (
                candidate.source is not None
                and candidate.use_direct_aligned
                and candidate.head_pad > 0
                and candidate.aligned_size > effective
            ):
                rounded = _page_rounded(candidate.source.length)
                if rounded <= effective:
                    candidate.use_direct_aligned = False
                    candidate.head_pad = 0
                    candidate.aligned_size = rounded
        return effective

    def _release_exhausted_candidates(
        self,
        unvisited: Sequence[_Candidate],
        prepared_copies: list[_PreparedCopy],
        pending_keys: list[int],
    ) -> None:
        if not unvisited:
            return
        self._release_private_candidate_holds(unvisited)
        prepared_pointers = {item.pointer for item in prepared_copies}
        with self._lock:
            removed = False
            for skipped in unvisited:
                if skipped.pointer in prepared_pointers:
                    continue
                if self._pending.pop(skipped.pointer, None) is not None:
                    removed = True
                if skipped.pointer in pending_keys:
                    pending_keys.remove(skipped.pointer)
            if removed:
                self._pending_changed.notify_all()

    @staticmethod
    def _apply_host_ceiling(
        pin_ceiling: int | None,
        host_budget: int | None,
        private_bytes: int,
    ) -> tuple[int | None, bool]:
        """Clamp ``pin_ceiling`` to remaining combined host headroom when Tier 2b holds pages."""
        if host_budget is None or host_budget == 0 or private_bytes == 0:
            return pin_ceiling, False
        host_pin_cap = max(0, host_budget - private_bytes)
        if pin_ceiling is None or host_pin_cap < pin_ceiling:
            return host_pin_cap, True
        return pin_ceiling, False

    def _reserve_host_headroom_for_pinned(
        self,
        host_budget: int | None,
        required_pinned_total: int,
    ) -> bool:
        """Evict idle Tier 2b copies to make room for ``required_pinned_total`` under ``host_budget``."""
        if host_budget is None or host_budget == 0:
            return False
        target_2b = max(0, host_budget - required_pinned_total)
        before_2b = self._tier2b.stats.reserved_bytes
        if before_2b <= target_2b:
            return False
        self._evict_private_to_fit(target_2b)
        return self._tier2b.stats.reserved_bytes < before_2b

    def _effective_candidate_ceiling_locked(
        self, effective_ceiling: int | None
    ) -> tuple[int, int, bool, int | None, int | None, int]:
        """Return ``(pin_epoch, host_epoch, budget_open, active_ceiling, host_budget, in_place)``."""
        cur_epoch = self._pin_budget_epoch
        cur_host_epoch = self._host_budget_epoch
        cur_budget = self._max_pinned_bytes
        host_budget = self._max_host_bytes
        in_place = self._in_place_pinned_bytes
        if cur_budget is None:
            active_ceiling = None
        elif effective_ceiling is None:
            active_ceiling = cur_budget
        else:
            active_ceiling = min(effective_ceiling, cur_budget)
        return cur_epoch, cur_host_epoch, cur_budget != 0, active_ceiling, host_budget, in_place

    def _allocate_and_reserve_candidates(
        self,
        candidates: list[_Candidate],
        effective_ceiling: int | None,
        held: dict[int, _Registration],
        created: list[_Registration],
        prepared_copies: list[_PreparedCopy],
        pending_keys: list[int],
    ) -> None:
        batched_hint = True
        held_set = set(held)
        run_end = 0
        for idx, candidate in enumerate(candidates):
            if candidate.source is None:
                exhausted = self._acquire_in_place_candidate(
                    candidate, effective_ceiling, held, created, held_set
                )
            else:
                if idx >= run_end:
                    run_end = idx + 1
                    while run_end < len(candidates) and candidates[run_end].source is not None:
                        run_end += 1
                run_tail = _CandidateRun(candidates, idx + 1, run_end if batched_hint else idx + 1)
                exhausted, used_fallback = self._acquire_tier2a_candidate(
                    candidate,
                    effective_ceiling,
                    run_tail,
                    prepared_copies,
                    held_set,
                )
                if used_fallback:
                    batched_hint = False

            # Pop already-settled / skipped non-copy candidates so waiters proceed promptly.
            if candidate.source is None or (
                not prepared_copies or prepared_copies[-1].pointer != candidate.pointer
            ):
                self._release_private_candidate_holds([candidate])
                with self._lock:
                    if self._pending.pop(candidate.pointer, None) is not None:
                        self._pending_changed.notify_all()
                if candidate.pointer in pending_keys:
                    pending_keys.remove(candidate.pointer)

            if exhausted:
                self._release_exhausted_candidates(candidates[idx + 1 :], prepared_copies, pending_keys)
                break

    def _commit_in_place_locked(
        self,
        candidate: _Candidate,
        held: dict[int, _Registration],
        created: list[_Registration],
        protected: set[int],
        *,
        retired: bool = False,
    ) -> None:
        pointer = candidate.pointer
        reg = _Registration(pointer, candidate.size, candidate.request.storage)
        reg.retired = retired
        self._register_owners_locked(pointer, candidate.request, reg.owners, reg.tracker)
        self._registrations[pointer] = reg
        created.append(reg)
        reg.leases += 1
        held[pointer] = reg
        self._idle.pop(pointer, None)
        protected.add(pointer)
        del self._pageable[pointer]
        _live_managers.add(self)

    def _enter_in_place_reserve_locked(
        self,
        pointer: int,
        size: int,
        expected_epoch: int,
        expected_host_epoch: int,
    ) -> str:
        if (
            self._pin_budget_epoch != expected_epoch
            or self._host_budget_epoch != expected_host_epoch
            or self._bounce_publishing
        ):
            return "retry"
        if self._max_pinned_bytes == 0:
            return "disabled"
        self._reserve_range_locked(pointer, size)
        self._pin_reserving += 1
        return "ok"

    def _attempt_in_place_registration(
        self,
        candidate: _Candidate,
        cur_epoch: int,
        held: dict[int, _Registration],
        created: list[_Registration],
        protected: set[int],
    ) -> str:
        pointer = candidate.pointer
        size = candidate.size
        reserving = True
        try:
            if not self._register_in_place_with_retry(pointer, size, protected=protected):
                return "exhausted"
            with self._lock:
                if self._max_pinned_bytes != 0 and self._pin_budget_epoch == cur_epoch:
                    self._commit_in_place_locked(candidate, held, created, protected)
                    reserving = False
                    self._pin_reserving -= 1
                    self._bounce_cv.notify_all()
                    return "committed"
                action = "disabled" if self._max_pinned_bytes == 0 else "retry"
                try:
                    self._backend.unregister(pointer)
                except BaseException as error:
                    # Cancellation failed: retain the still-registered pointer, storage, and budget charge.
                    self._commit_in_place_locked(candidate, held, created, protected, retired=True)
                    reserving = False
                    self._pin_reserving -= 1
                    self._bounce_cv.notify_all()
                    if isinstance(error, Exception):
                        self._unregistration_failures += 1
                        logger.warning(
                            "Host unregistration failed while cancelling an in-place registration; "
                            "retaining storage and budget reservation: %s",
                            str(error),
                        )
                        return "committed"
                    raise
                self._unreserve_range_locked(pointer, size)
                reserving = False
                self._pin_reserving -= 1
                self._bounce_cv.notify_all()
                self._drop_lifetime_root_if_empty_locked()
                return action
        except HostRegistrationRefusedError:
            return "disabled"
        finally:
            if reserving:
                with self._lock:
                    self._pin_reserving -= 1
                    self._bounce_cv.notify_all()
                    self._unreserve_range_locked(pointer, size)

    def _acquire_in_place_candidate(
        self,
        candidate: _Candidate,
        effective_ceiling: int | None,
        held: dict[int, _Registration],
        created: list[_Registration],
        protected: set[int],
    ) -> bool:
        """Reserve and register one in-place range in request order; return whether runtime capacity ran out."""
        pointer = candidate.pointer
        size = candidate.size
        while True:
            with self._lock:
                while self._bounce_publishing:
                    self._bounce_cv.wait()
                (
                    cur_epoch,
                    cur_host_epoch,
                    budget_open,
                    active_ceiling,
                    host_budget,
                    in_place,
                ) = self._effective_candidate_ceiling_locked(effective_ceiling)
                needed = self._page_reservation(pointer, size)
            if not budget_open or (active_ceiling is not None and needed > active_ceiling):
                return False

            bounce_bytes, _, _ = self._active_bounce_bytes_off_lock()
            private_bytes = self._tier2b.stats.reserved_bytes
            combined_ceiling, host_capped = self._apply_host_ceiling(
                active_ceiling, host_budget, private_bytes
            )
            if combined_ceiling is not None and needed > combined_ceiling:
                if host_capped and self._reserve_host_headroom_for_pinned(host_budget, needed):
                    continue
                return False

            tier2a_bytes = self._tier2a.stats.reserved_bytes
            if combined_ceiling is not None and in_place + tier2a_bytes + bounce_bytes + needed > combined_ceiling:
                required = in_place + tier2a_bytes + bounce_bytes + needed
                if host_capped and self._reserve_host_headroom_for_pinned(host_budget, required):
                    continue
                target_total = max(0, combined_ceiling - needed)
                if not self._evict_idle_for_budget(target_total, protected=protected, close_bounce_last=True):
                    return False
                continue

            with self._lock:
                gate = self._enter_in_place_reserve_locked(pointer, size, cur_epoch, cur_host_epoch)
            if gate == "retry":
                continue
            if gate == "disabled":
                return False

            outcome = self._attempt_in_place_registration(candidate, cur_epoch, held, created, protected)
            if outcome != "retry":
                return outcome == "exhausted"

    def _register_in_place_with_retry(
        self, pointer: int, size: int, *, protected: set[int]
    ) -> bool:
        target = max(mmap.PAGESIZE, _page_rounded(size))
        while True:
            try:
                registered = self._backend.register(pointer, size)
            except HostRegistrationRefusedError:
                with self._lock:
                    self._registration_failures += 1
                raise
            except Exception:
                with self._lock:
                    self._registration_failures += 1
                raise
            if registered:
                return True
            with self._lock:
                self._registration_failures += 1
            if not self._evict_idle_for_runtime_retry(target, protected=protected):
                return False

    def _plan_tier2a_slab_hint(
        self,
        candidate: _Candidate,
        run_tail: _CandidateRun,
        max_2a_bytes: int | None,
    ) -> int:
        # Must not be called while holding self._lock (Lock order: PinManager._lock before arena locks).
        sub_align = max(candidate.sector, mmap.PAGESIZE)
        following = (
            (run_tail.candidates[j].aligned_size, max(run_tail.candidates[j].sector, mmap.PAGESIZE))
            for j in range(run_tail.start, run_tail.end)
        )
        return self._tier2a.plan_slab_bytes(
            candidate.aligned_size,
            sub_align,
            following,
            max_bytes=max_2a_bytes,
        )

    def _finish_prepared_tier2a_copy(
        self,
        candidate: _Candidate,
        alloc: ArenaAllocation,
        prepared_copies: list[_PreparedCopy],
    ) -> None:
        assert candidate.source is not None
        priv = candidate.private_source
        candidate.private_source = None
        item = _PreparedCopy(
            pointer=candidate.pointer,
            request=candidate.request,
            source=candidate.source,
            alloc=alloc,
            head_pad=candidate.head_pad,
            use_direct_aligned=candidate.use_direct_aligned,
            private_source=priv,
        )
        prepared_copies.append(item)
        if priv is None:
            return
        item.alloc.as_uint8_tensor().copy_(priv.alloc.as_uint8_tensor())
        retired_2b: list[ArenaAllocation] = []
        try:
            with self._lock:
                item.private_source = None
                item.filled = True
                priv.leases -= 1
                if self._private_copies.get(candidate.pointer) is priv and priv.leases == 0:
                    del self._private_copies[candidate.pointer]
                    retired_2b.append(priv.alloc)
                    if not priv.tracker.retired:
                        item.promoted_owners = priv.owners
                        item.promoted_tracker = priv.tracker
                self._drop_lifetime_root_if_empty_locked()
        finally:
            if retired_2b:
                try:
                    self._drop_detached_allocations(allocs_2b=retired_2b, trim_2b=True)
                finally:
                    with self._lock:
                        self._drop_lifetime_root_if_empty_locked()

    def _reconcile_tier2a_host_epoch(
        self,
        alloc: ArenaAllocation,
        cur_host_budget: int | None,
        cur_in_place: int,
    ) -> bool:
        """Keep ``alloc`` only if concurrent Tier 2b growth still fits under ``cur_host_budget``."""
        if cur_host_budget is None or cur_host_budget == 0:
            return True
        cur_tier2b = self._tier2b.stats.reserved_bytes
        if cur_tier2b == 0:
            return True
        cur_pinned = (
            cur_in_place + self._tier2a.stats.reserved_bytes + self._active_bounce_bytes_off_lock()[0]
        )
        if cur_pinned + cur_tier2b <= cur_host_budget:
            return True
        self._evict_private_to_fit(max(0, cur_host_budget - cur_pinned))
        if (
            self._tier2b.stats.reserved_bytes == 0
            or cur_pinned + self._tier2b.stats.reserved_bytes <= cur_host_budget
        ):
            return True
        try:
            alloc.release()
        finally:
            self._tier2a.trim_empty(0)
        self._enforce_host_budget()
        return False

    def _finalize_tier2a_span_alloc(
        self,
        alloc: ArenaAllocation | None,
        *,
        disabled: bool,
        epoch_changed: bool,
        host_epoch_changed: bool,
        cur_host_budget: int | None,
        cur_in_place: int,
        reg_failed: bool,
        refused: bool,
    ) -> tuple[ArenaAllocation | None, bool, bool, str]:
        try:
            if (disabled or epoch_changed) and alloc is not None:
                released_alloc = alloc
                alloc = None
                try:
                    released_alloc.release()
                finally:
                    self._tier2a.trim_empty(0)
            elif alloc is not None and host_epoch_changed:
                checked = alloc
                alloc = None
                if self._reconcile_tier2a_host_epoch(checked, cur_host_budget, cur_in_place):
                    alloc = checked
                else:
                    return None, reg_failed, refused, "retry"
            if epoch_changed:
                self._enforce_pin_budget()
            if disabled or epoch_changed:
                return None, reg_failed, refused, ("disabled" if disabled else "retry")
            return alloc, reg_failed, refused, "ok"
        except BaseException:
            if alloc is not None:
                leaked = alloc
                alloc = None
                try:
                    leaked.release()
                finally:
                    self._tier2a.trim_empty(0)
            raise

    def _try_alloc_tier2a_span(
        self,
        candidate: _Candidate,
        avail_2a: int | None,
        slab_hint_bytes: int,
        cur_epoch: int,
        cur_host_epoch: int,
    ) -> tuple[ArenaAllocation | None, bool, bool, str]:
        with self._lock:
            if (
                self._pin_budget_epoch != cur_epoch
                or self._host_budget_epoch != cur_host_epoch
                or self._bounce_publishing
            ):
                return None, False, False, "retry"
            if self._max_pinned_bytes == 0:
                return None, False, False, "disabled"
            self._pin_reserving += 1

        alloc: ArenaAllocation | None = None
        fails_before = self._tier2a.stats.registration_failures
        refused_before = self._tier2a_backend.refused_count
        try:
            try:
                self._tier2a.max_bytes = avail_2a
                alloc = self._tier2a.allocate(
                    candidate.source.length if candidate.source is not None else candidate.size,
                    head_pad=candidate.head_pad,
                    alignment=max(candidate.sector, mmap.PAGESIZE),
                    slab_hint_bytes=max(candidate.aligned_size, slab_hint_bytes),
                )
                fails_after = self._tier2a.stats.registration_failures
                refused_after = self._tier2a_backend.refused_count
            finally:
                with self._lock:
                    self._pin_reserving -= 1
                    self._bounce_cv.notify_all()
                    disabled = self._max_pinned_bytes == 0
                    epoch_changed = self._pin_budget_epoch != cur_epoch
                    host_epoch_changed = self._host_budget_epoch != cur_host_epoch
                    cur_host_budget = self._max_host_bytes
                    cur_in_place = self._in_place_pinned_bytes
                    if alloc is not None and not disabled and not epoch_changed:
                        self._host_budget_epoch += 1

            taken_alloc = alloc
            alloc = None
            return self._finalize_tier2a_span_alloc(
                taken_alloc,
                disabled=disabled,
                epoch_changed=epoch_changed,
                host_epoch_changed=host_epoch_changed,
                cur_host_budget=cur_host_budget,
                cur_in_place=cur_in_place,
                reg_failed=fails_after > fails_before,
                refused=refused_after > refused_before,
            )
        except BaseException:
            if alloc is not None:
                leaked = alloc
                alloc = None
                try:
                    leaked.release()
                finally:
                    self._tier2a.trim_empty(0)
            raise

    def _evict_for_tier2a_headroom(
        self,
        candidate: _Candidate,
        *,
        host_budget: int | None,
        host_capped: bool,
        required_pinned: int,
        evict_target_pinned: int | None,
        protected: set[int],
    ) -> bool:
        if host_capped and self._reserve_host_headroom_for_pinned(host_budget, required_pinned):
            return True
        if evict_target_pinned is not None and self._evict_idle_for_budget(
            evict_target_pinned, protected=protected, close_bounce_last=True
        ):
            return True
        if host_capped and candidate.private_source is not None:
            self._release_private_candidate_holds([candidate])
            return self._reserve_host_headroom_for_pinned(host_budget, required_pinned)
        return False

    def _prepare_tier2a_headroom(
        self,
        candidate: _Candidate,
        run_tail: _CandidateRun,
        *,
        budget_open: bool,
        active_ceiling: int | None,
        host_budget: int | None,
        pinned_2b_floor: int,
        in_place: int,
        protected: set[int],
    ) -> tuple[str, int | None, int | None, int, bool]:
        if not budget_open or (active_ceiling is not None and candidate.aligned_size > active_ceiling):
            return "stop", None, None, 0, False
        bounce_bytes, _, _ = self._active_bounce_bytes_off_lock()
        private_bytes = self._tier2b.stats.reserved_bytes
        combined_ceiling, host_capped = self._apply_host_ceiling(
            active_ceiling, host_budget, private_bytes
        )
        post_evict_ceiling, _ = self._apply_host_ceiling(
            active_ceiling, host_budget, pinned_2b_floor
        )
        post_evict_avail_2a = (
            None
            if post_evict_ceiling is None
            else max(0, post_evict_ceiling - in_place - bounce_bytes)
        )
        avail_2a = (
            None if combined_ceiling is None else max(0, combined_ceiling - in_place - bounce_bytes)
        )
        if combined_ceiling is not None and candidate.aligned_size > combined_ceiling:
            action = "retry" if self._evict_for_tier2a_headroom(
                candidate,
                host_budget=host_budget,
                host_capped=host_capped,
                required_pinned=candidate.aligned_size,
                evict_target_pinned=None,
                protected=protected,
            ) else "stop"
            return action, avail_2a, post_evict_avail_2a, bounce_bytes, host_capped
        if host_capped and avail_2a is not None and self._tier2a.stats.reserved_bytes == 0:
            growth = self._plan_tier2a_slab_hint(candidate, run_tail, post_evict_avail_2a)
            if avail_2a < growth and self._reserve_host_headroom_for_pinned(
                host_budget, in_place + bounce_bytes + growth
            ):
                return "retry", avail_2a, post_evict_avail_2a, bounce_bytes, host_capped
        if avail_2a is not None and avail_2a < candidate.aligned_size:
            assert combined_ceiling is not None
            action = "retry" if self._evict_for_tier2a_headroom(
                candidate,
                host_budget=host_budget,
                host_capped=host_capped,
                required_pinned=in_place + bounce_bytes + candidate.aligned_size,
                evict_target_pinned=max(0, combined_ceiling - candidate.aligned_size),
                protected=protected,
            ) else "stop"
            return action, avail_2a, post_evict_avail_2a, bounce_bytes, host_capped
        return "ready", avail_2a, post_evict_avail_2a, bounce_bytes, host_capped

    def _evict_for_tier2a_miss(
        self,
        candidate: _Candidate,
        run_tail: _CandidateRun,
        *,
        avail_2a: int | None,
        post_evict_avail_2a: int | None,
        host_budget: int | None,
        host_capped: bool,
        in_place: int,
        bounce_bytes: int,
        protected: set[int],
    ) -> bool:
        tier2a_reserved = 0
        if host_capped:
            tier2a_reserved = self._tier2a.stats.reserved_bytes
            growth = self._plan_tier2a_slab_hint(candidate, run_tail, post_evict_avail_2a)
            planned_2a = tier2a_reserved + growth
            if post_evict_avail_2a is not None:
                planned_2a = min(planned_2a, post_evict_avail_2a)
            if self._reserve_host_headroom_for_pinned(host_budget, in_place + bounce_bytes + planned_2a):
                return True
        if self._evict_one_for_tier2a(avail_2a, protected=protected):
            return True
        if host_capped and candidate.private_source is not None:
            self._release_private_candidate_holds([candidate])
            return self._reserve_host_headroom_for_pinned(
                host_budget, in_place + bounce_bytes + tier2a_reserved + candidate.aligned_size
            )
        return False

    def _acquire_tier2a_candidate(
        self,
        candidate: _Candidate,
        effective_ceiling: int | None,
        run_tail: _CandidateRun,
        prepared_copies: list[_PreparedCopy],
        protected: set[int],
    ) -> tuple[bool, bool]:
        """Allocate a Tier 2a span for one checkpoint candidate; return ``(capacity_exhausted, used_fallback)``."""
        assert candidate.source is not None
        if effective_ceiling is not None and candidate.aligned_size > effective_ceiling:
            return False, False

        used_fallback = False
        while True:
            with self._lock:
                while self._bounce_publishing:
                    self._bounce_cv.wait()
                (
                    cur_epoch,
                    cur_host_epoch,
                    budget_open,
                    active_ceiling,
                    host_budget,
                    in_place,
                ) = self._effective_candidate_ceiling_locked(effective_ceiling)
                pinned_2b_floor = (
                    sum(_page_rounded(priv.size) for priv in self._private_copies.values() if priv.leases > 0)
                    if host_budget is not None and host_budget > 0 and self._private_copies
                    else 0
                )
            state, avail_2a, post_evict_avail_2a, bounce_bytes, host_capped = (
                self._prepare_tier2a_headroom(
                    candidate,
                    run_tail,
                    budget_open=budget_open,
                    active_ceiling=active_ceiling,
                    host_budget=host_budget,
                    pinned_2b_floor=pinned_2b_floor,
                    in_place=in_place,
                    protected=protected,
                )
            )
            if state == "retry":
                continue
            if state == "stop":
                break

            slab_hint_bytes = self._plan_tier2a_slab_hint(candidate, run_tail, avail_2a)
            alloc, reg_failed, refused, status = self._try_alloc_tier2a_span(
                candidate, avail_2a, slab_hint_bytes, cur_epoch, cur_host_epoch
            )
            if reg_failed:
                used_fallback = True
            if status == "retry":
                continue
            if alloc is not None:
                self._finish_prepared_tier2a_copy(candidate, alloc, prepared_copies)
                return False, used_fallback
            if status == "disabled" or refused:
                break
            if reg_failed:
                if self._evict_idle_for_runtime_retry(
                    max(mmap.PAGESIZE, candidate.aligned_size), protected=protected
                ):
                    continue
                return True, used_fallback
            if not self._evict_for_tier2a_miss(
                candidate,
                run_tail,
                avail_2a=avail_2a,
                post_evict_avail_2a=post_evict_avail_2a,
                host_budget=host_budget,
                host_capped=host_capped,
                in_place=in_place,
                bounce_bytes=bounce_bytes,
                protected=protected,
            ):
                break
        return False, used_fallback

    def _fill_prepared_copies(self, prepared_copies: list[_PreparedCopy]) -> None:
        if not prepared_copies:
            return
        cold_items = [item for item in prepared_copies if not item.filled]
        if cold_items:
            self._read_cold_copies(cold_items)

    def _read_cold_copies(self, cold_items: list[_PreparedCopy]) -> None:
        views: list[memoryview] = []
        requests: list[DirectReadRequest] = []
        try:
            for item in cold_items:
                mv = item.alloc.aligned_view() if item.use_direct_aligned else item.alloc.payload_view()
                views.append(mv)
                requests.append(
                    DirectReadRequest(
                        source=item.source,
                        destination=mv,
                        head_pad=item.head_pad if item.use_direct_aligned else 0,
                        aligned_target=item.use_direct_aligned,
                    )
                )
            try:
                self._reader.read_into(requests)
            except (OSError, ValueError) as batch_error:
                self._retry_failed_read_batch(cold_items, requests, batch_error)
        finally:
            for mv in views:
                mv.release()
            views.clear()

    def _retry_failed_read_batch(
        self,
        cold_items: list[_PreparedCopy],
        requests: list[DirectReadRequest],
        batch_error: Exception,
    ) -> None:
        if len(cold_items) == 1:
            cold_items[0].error = str(batch_error)
            return
        for item, req in zip(cold_items, requests, strict=True):
            try:
                self._reader.read_into([req])
            except (OSError, ValueError) as item_error:
                item.error = str(item_error)

    def _commit_prepared_copies_locked(
        self,
        prepared_copies: list[_PreparedCopy],
        held: dict[int, _Registration],
        created: list[_Registration],
        deferred_2a: list[ArenaAllocation],
        deferred_2b: list[ArenaAllocation],
    ) -> None:
        while prepared_copies:
            item = prepared_copies.pop(0)
            pointer = item.pointer
            priv = item.private_source
            item.private_source = None

            if item.error is not None:
                logger.warning("Could not fill a pinned copy from the checkpoint; it stays pageable: %s", item.error)
                deferred_2a.append(item.alloc)
                if priv is not None:
                    self._release_private_copies_locked(deque([priv]), deferred_2b)
                continue

            if priv is not None:
                priv.leases -= 1
                promoted_out = False
                if self._private_copies.get(pointer) is priv and priv.leases == 0:
                    del self._private_copies[pointer]
                    deferred_2b.append(priv.alloc)
                    promoted_out = True
                if promoted_out and not priv.tracker.retired:
                    owners = priv.owners
                    tracker = priv.tracker
                else:
                    owners = {}
                    tracker = _OwnerTracker()
            elif item.promoted_tracker is not None and not item.promoted_tracker.retired:
                owners = item.promoted_owners if item.promoted_owners is not None else {}
                tracker = item.promoted_tracker
            else:
                owners = {}
                tracker = _OwnerTracker()
            item.promoted_owners = None
            item.promoted_tracker = None

            reg = _Registration(
                pointer=pointer,
                size=item.request.storage.nbytes(),
                storage=item.request.storage,
                copy=item.alloc,
                owners=owners,
                tracker=tracker,
            )
            self._registrations[pointer] = reg
            created.append(reg)
            self._hold_locked(reg, item.request, held)
            del self._pageable[pointer]
            _live_managers.add(self)

    def _rollback_acquire(
        self,
        prepared_copies: list[_PreparedCopy],
        candidates: list[_Candidate],
        pending_keys: list[int],
        created: list[_Registration],
        held: dict[int, _Registration],
        held_private: dict[int, _PrivateCopy],
        pageable: list[int],
        deferred_2a: list[ArenaAllocation],
        deferred_2b: list[ArenaAllocation],
        start_pin_epoch: int,
    ) -> None:
        retired_2a: list[ArenaAllocation] = []
        retired_2b: list[ArenaAllocation] = []
        priv_from_prepared: deque[_PrivateCopy] = deque()
        pin_epoch_changed = False
        try:
            with self._lock:
                for item in prepared_copies:
                    retired_2a.append(item.alloc)
                    if item.private_source is not None:
                        priv_from_prepared.append(item.private_source)
                        item.private_source = None
                prepared_copies.clear()
                if priv_from_prepared:
                    self._release_private_copies_locked(priv_from_prepared, retired_2b)
                self._release_private_candidate_holds_locked(candidates, retired_2b)
                self._release_pageable_locked(deque(p for p in pageable if p not in held))
                self._release_private_copies_locked(deque(held_private.values()), retired_2b)
                retired_regs = self._release_registrations_locked(deque(held.values()))
                already_retired = {id(reg) for reg in retired_regs}
                for reg in created:
                    if reg.leases == 0:
                        reg.retired = True
                        if id(reg) not in already_retired:
                            retired_regs.append(reg)
                            already_retired.add(id(reg))
                self._pop_retired_registrations_locked(retired_regs, retired_2a)
                for key in pending_keys:
                    self._pending.pop(key, None)
                if pending_keys:
                    pending_keys.clear()
                    self._pending_changed.notify_all()
                pin_epoch_changed = self._pin_budget_epoch != start_pin_epoch
                self._drop_lifetime_root_if_empty_locked()
        finally:
            retired_2a.extend(deferred_2a)
            deferred_2a.clear()
            retired_2b.extend(deferred_2b)
            deferred_2b.clear()
            if retired_2a or retired_2b:
                self._drop_detached_allocations(
                    retired_2a,
                    retired_2b,
                    trim_2a=bool(retired_2a),
                    trim_2b=bool(retired_2b),
                )
        if pin_epoch_changed:
            self._enforce_pin_budget()
        else:
            self._enforce_host_budget()
        with self._lock:
            self._drop_lifetime_root_if_empty_locked()

    def transfer(self, destination: torch.Tensor, source: torch.Tensor, *, non_blocking: bool) -> None:
        """Copy ``source`` into ``destination``, routing Tier 2a, Tier 2b, or direct FileSlice cleanly.

        Asynchronous transfers whose exact source tensor is held by an open
        lease reuse its pre-resolved view lock-free when pinned. Transfers from
        Tier 2b private copies, uncached ``FileSlice`` tensors, or plain
        pageable CPU memory to CUDA pipeline through the lazy
        ``PinnedBounceRing`` when eligible (leaving zero extra bytes in host
        RAM for direct ``Disk -> Bounce Ring -> GPU`` copies). Asynchronous
        transfers of pinned storage outside an active lease raise, while a
        synchronous transfer completes cleanly.
        """
        if source.device.type != "cpu" or source.numel() == 0:
            destination.copy_(source, non_blocking=non_blocking)
            return

        to_cuda = destination.device.type == "cuda"
        asynchronous = non_blocking and to_cuda
        if asynchronous:
            entry = self._leased_views.get(id(source))
            if entry is not None:
                self._transfer_leased_async(destination, source, entry[0], is_pinned=entry[2])
                return

        self._transfer_locked_fallback(destination, source, to_cuda=to_cuda, non_blocking=non_blocking)

    def _transfer_leased_async(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        view: torch.Tensor,
        *,
        is_pinned: bool,
    ) -> None:
        if is_pinned:
            transfer_priority.copy_host_to_device(destination, view)
            return
        if not self._transfer_unpinned_cuda(
            destination,
            view,
            from_private=(view is not source),
            non_blocking=True,
        ):
            transfer_priority.copy_host_to_device(destination, view)

    def _acquire_transfer_hold_locked(
        self,
        source: torch.Tensor,
        *,
        asynchronous: bool,
    ) -> tuple[
        torch.Tensor | None,
        ArenaAllocation | None,
        bool,
        bool,
        _Registration | None,
        _PrivateCopy | None,
        int,
        int,
        int,
    ]:
        pin_epoch = self._pin_budget_epoch
        host_epoch = self._host_budget_epoch
        close_seq = self._lease_close_seq
        ptr = source.untyped_storage().data_ptr()

        entry = self._leased_views.get(id(source))
        if entry is not None:
            view, _count, is_pinned = entry
            held_reg: _Registration | None = None
            held_priv: _PrivateCopy | None = None
            from_private = not is_pinned and view is not source
            if is_pinned:
                reg = self._registrations.get(ptr)
                if reg is not None:
                    reg.leases += 1
                    self._idle.pop(ptr, None)
                    held_reg = reg
            elif from_private:
                priv = self._private_copies.get(ptr)
                if priv is not None:
                    priv.leases += 1
                    self._private_copies.move_to_end(ptr)
                    held_priv = priv
            return view, None, is_pinned, from_private, held_reg, held_priv, pin_epoch, host_epoch, close_seq

        registration = self._registrations.get(ptr)
        if registration is not None:
            if asynchronous and registration.leases == 0:
                raise RuntimeError(
                    "Asynchronous transfer of pinned host storage outside a lease; acquire one "
                    "over the tensors first (pin=False for a one-time transfer) and keep it "
                    "until the transfer has completed."
                )
            registration.leases += 1
            self._idle.pop(ptr, None)
            if registration.copy is None:
                return source, None, True, False, registration, None, pin_epoch, host_epoch, close_seq
            return None, registration.copy, True, False, registration, None, pin_epoch, host_epoch, close_seq

        priv = self._private_copies.get(ptr)
        if priv is not None and priv.size == source.untyped_storage().nbytes() and not priv.retired:
            priv.leases += 1
            self._private_copies.move_to_end(ptr)
            return None, priv.alloc, False, True, None, priv, pin_epoch, host_epoch, close_seq
        return source, None, False, False, None, None, pin_epoch, host_epoch, close_seq

    def _release_transient_transfer_hold(
        self,
        held_reg: _Registration | None,
        held_priv: _PrivateCopy | None,
        *,
        pin_epoch: int,
        host_epoch: int,
        close_seq: int,
    ) -> None:
        retired_2a: list[ArenaAllocation] = []
        retired_2b: list[ArenaAllocation] = []
        enforce_pin = False
        enforce_host = False
        had_retired = False
        try:
            with self._lock:
                if held_reg is not None:
                    retired_regs = self._release_registrations_locked(deque([held_reg]))
                    self._pop_retired_registrations_locked(retired_regs, retired_2a)
                if held_priv is not None:
                    self._release_private_copies_locked(deque([held_priv]), retired_2b)
                lease_closed = self._lease_close_seq != close_seq
                enforce_pin = self._max_pinned_bytes is not None and (
                    self._pin_budget_epoch != pin_epoch or (lease_closed and held_reg is not None)
                )
                enforce_host = self._max_host_bytes is not None and (
                    self._host_budget_epoch != host_epoch or (lease_closed and held_priv is not None)
                )
                self._drop_lifetime_root_if_empty_locked()
        finally:
            had_retired = bool(retired_2a or retired_2b)
            if had_retired:
                self._drop_detached_allocations(
                    retired_2a,
                    retired_2b,
                    trim_2a=bool(retired_2a),
                    trim_2b=bool(retired_2b),
                )
        if enforce_pin:
            self._enforce_pin_budget()
        elif enforce_host:
            self._enforce_host_budget()
        if had_retired:
            with self._lock:
                self._drop_lifetime_root_if_empty_locked()

    def _transfer_locked_fallback(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        *,
        to_cuda: bool,
        non_blocking: bool,
    ) -> None:
        asynchronous = non_blocking and to_cuda
        view: torch.Tensor | None = None
        source_alloc: ArenaAllocation | None = None
        held_reg: _Registration | None = None
        held_priv: _PrivateCopy | None = None
        pin_epoch = 0
        host_epoch = 0
        close_seq = 0
        try:
            with self._lock:
                (
                    view,
                    source_alloc,
                    is_pinned_source,
                    from_private,
                    held_reg,
                    held_priv,
                    pin_epoch,
                    host_epoch,
                    close_seq,
                ) = self._acquire_transfer_hold_locked(source, asynchronous=asynchronous)
            if view is None:
                view = source if source_alloc is None else source_alloc.tensor_view(source)

            if is_pinned_source:
                if to_cuda and asynchronous:
                    transfer_priority.copy_host_to_device(destination, view)
                else:
                    destination.copy_(view, non_blocking=non_blocking)
                return

            if to_cuda and self._transfer_unpinned_cuda(
                destination,
                view,
                from_private=from_private,
                non_blocking=non_blocking,
            ):
                return
            if asynchronous:
                transfer_priority.copy_host_to_device(destination, view)
                return
            destination.copy_(view, non_blocking=non_blocking)
        finally:
            view = None
            source_alloc = None
            if held_reg is not None or held_priv is not None:
                self._release_transient_transfer_hold(
                    held_reg,
                    held_priv,
                    pin_epoch=pin_epoch,
                    host_epoch=host_epoch,
                    close_seq=close_seq,
                )

    def clear(self) -> None:
        """Release all idle registrations, Tier 2a/2b slabs, and the bounce ring.

        Active leases stay valid. Failed unregistrations retain their storage
        and budget charges so callers can retry cleanly.
        """
        failed = 0
        to_release_2a: list[ArenaAllocation] = []
        to_release_2b: list[ArenaAllocation] = []
        try:
            with self._lock:
                for pointer in tuple(self._idle):
                    registration = self._registrations.get(pointer)
                    if registration is None:
                        self._idle.pop(pointer, None)
                        continue
                    if registration.copy is None:
                        if not self._unregister_in_place_locked(registration):
                            failed += 1
                    else:
                        del self._registrations[pointer]
                        self._starts.remove(pointer)
                        self._idle.pop(pointer, None)
                        to_release_2a.append(registration.copy)

                for pointer in tuple(self._private_copies):
                    priv = self._private_copies[pointer]
                    if priv.leases == 0:
                        del self._private_copies[pointer]
                        to_release_2b.append(priv.alloc)
        except BaseException:
            if to_release_2a or to_release_2b:
                self._drop_detached_allocations(
                    to_release_2a,
                    to_release_2b,
                    trim_2a=bool(to_release_2a),
                    trim_2b=bool(to_release_2b),
                )
            with self._lock:
                self._drop_lifetime_root_if_empty_locked()
            raise

        try:
            self._release_allocations(to_release_2a)
        finally:
            self._release_allocations(to_release_2b)
        try:
            failed += self._tier2a.trim_empty(0)
        finally:
            self._tier2b.trim_empty(0)
        _closed_rings, failed_rings = self._close_bounce_rings(wait_for_users=True)
        failed += failed_rings

        with self._lock:
            self._drop_lifetime_root_if_empty_locked()

        if failed:
            raise RuntimeError(f"Could not release {failed} host registration(s); storage remains retained")

    # ------------------------------------------------------------------
    # Validation and boundary-page accounting
    # ------------------------------------------------------------------

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

    def _validate_ranges_locked(self, requests: dict[int, _Request]) -> None:
        prior_end = 0
        for pointer in sorted(requests):
            end = pointer + requests[pointer].storage.nbytes()
            neighbors = self._starts.neighbors(pointer)
            if pointer < prior_end:
                raise ValueError("Overlapping host storage ranges must use views of one storage")
            for other in neighbors:
                other_end = other + self._range_size_locked(other)
                if pointer < other_end and other < end and (pointer != other or end != other_end):
                    raise ValueError("Overlapping host storage ranges must use views of one storage")
            prior_end = end

    def _range_size_locked(self, pointer: int) -> int:
        registration = self._registrations.get(pointer)
        if registration is not None:
            return registration.size
        return self._pageable[pointer].size

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

    def _reserve_range_locked(self, pointer: int, size: int) -> None:
        self._in_place_pinned_bytes += self._page_reservation(pointer, size)
        self._host_budget_epoch += 1
        for page in self._boundaries(pointer, size):
            self._boundary_pages[page] = self._boundary_pages.get(page, 0) + 1

    def _unreserve_range_locked(self, pointer: int, size: int) -> None:
        for page in self._boundaries(pointer, size):
            count = self._boundary_pages[page] - 1
            if count:
                self._boundary_pages[page] = count
            else:
                del self._boundary_pages[page]
        self._in_place_pinned_bytes -= self._page_reservation(pointer, size)

    # ------------------------------------------------------------------
    # Leases and weakref owner tracking
    # ------------------------------------------------------------------

    def _try_hold_registered_locked(
        self, requests: dict[int, _Request], held: dict[int, _Registration]
    ) -> bool:
        if self._pending and not self._pending.keys().isdisjoint(requests):
            return False
        pairs: list[tuple[_Registration, _Request]] = []
        for pointer, request in requests.items():
            registration = self._registrations.get(pointer)
            if registration is None or registration.size != request.storage.nbytes():
                return False
            pairs.append((registration, request))
        for registration, request in pairs:
            self._hold_locked(registration, request, held)
        return True

    def _make_owner_gone_callback(
        self, pointer: int, tracker: _OwnerTracker
    ) -> Callable[[weakref.ReferenceType[torch.Tensor]], None]:
        manager_ref = weakref.ref(self)
        tracker_ref = weakref.ref(tracker)

        def owner_gone(_ref: weakref.ReferenceType[torch.Tensor]) -> None:
            manager, live_tracker = manager_ref(), tracker_ref()
            if manager is None or live_tracker is None:
                return
            retired_2a: list[ArenaAllocation] = []
            retired_2b: list[ArenaAllocation] = []
            try:
                with manager._lock:
                    live_tracker.retired = True
                    reg = manager._registrations.get(pointer)
                    if reg is not None and reg.tracker is live_tracker and reg.leases == 0:
                        if reg.copy is None:
                            manager._unregister_in_place_locked(reg)
                        else:
                            del manager._registrations[pointer]
                            manager._starts.remove(pointer)
                            manager._idle.pop(pointer, None)
                            retired_2a.append(reg.copy)
                    priv = manager._private_copies.get(pointer)
                    if priv is not None and priv.tracker is live_tracker and priv.leases == 0:
                        del manager._private_copies[pointer]
                        retired_2b.append(priv.alloc)
                    manager._drop_lifetime_root_if_empty_locked()
            finally:
                # Release spans and reclaim empty slabs strictly off manager._lock.
                if retired_2a or retired_2b:
                    try:
                        manager._drop_detached_allocations(
                            retired_2a,
                            retired_2b,
                            trim_2a=bool(retired_2a),
                            trim_2b=bool(retired_2b),
                        )
                    finally:
                        with manager._lock:
                            manager._drop_lifetime_root_if_empty_locked()

        return owner_gone

    def _register_owners_locked(
        self,
        pointer: int,
        request: _Request,
        owners: dict[int, weakref.ReferenceType[torch.Tensor]],
        tracker: _OwnerTracker,
    ) -> None:
        if all(id(tensor) in owners for tensor in request.tensors):
            return
        callback = self._make_owner_gone_callback(pointer, tracker)
        for tensor in request.tensors:
            tid = id(tensor)
            if tid not in owners:
                owners[tid] = weakref.ref(tensor, callback)

    def _hold_locked(
        self, registration: _Registration, request: _Request, held: dict[int, _Registration]
    ) -> None:
        registration.leases += 1
        held[registration.pointer] = registration
        self._idle.pop(registration.pointer, None)
        self._register_owners_locked(
            registration.pointer, request, registration.owners, registration.tracker
        )

    def _hold_warm_private_copies_locked(
        self,
        pageable_pointers: Sequence[int],
        requests: dict[int, _Request],
        held_private: dict[int, _PrivateCopy],
    ) -> None:
        for pointer in pageable_pointers:
            priv = self._private_copies.get(pointer)
            request = requests[pointer]
            if priv is not None and priv.size == request.storage.nbytes() and not priv.retired:
                priv.leases += 1
                self._private_copies.move_to_end(pointer)
                held_private[pointer] = priv
                self._register_owners_locked(pointer, request, priv.owners, priv.tracker)

    def _hold_pageable_locked(self, pointer: int, request: _Request) -> None:
        pageable = self._pageable.get(pointer)
        if pageable is None:
            pageable = _Pageable(request.storage.nbytes())
            self._pageable[pointer] = pageable
            self._starts.add(pointer)
        pageable.leases += 1

    def _release_views_locked(self, view_ids: deque[int]) -> None:
        while view_ids:
            tid = view_ids.popleft()
            entry = self._leased_views.get(tid)
            if entry is None:
                continue
            view, count, is_pinned = entry
            if count > 1:
                self._leased_views[tid] = (view, count - 1, is_pinned)
            else:
                del self._leased_views[tid]

    @staticmethod
    def _build_lease_views_off_lock(
        requests: dict[int, _Request],
        held: dict[int, _Registration],
        held_private: dict[int, _PrivateCopy],
        built: dict[int, torch.Tensor],
    ) -> None:
        view: torch.Tensor | None = None
        try:
            for pointer, request in requests.items():
                reg = held.get(pointer)
                copy_alloc = (
                    reg.copy
                    if reg is not None
                    else (held_private[pointer].alloc if pointer in held_private else None)
                )
                if copy_alloc is None:
                    continue
                for tensor in request.tensors:
                    tid = id(tensor)
                    if tid not in built:
                        view = copy_alloc.tensor_view(tensor)
                        built[tid] = view
                        view = None
        except BaseException:
            view = None
            raise

    def _commit_lease_locked(
        self,
        requests: dict[int, _Request],
        held: dict[int, _Registration],
        held_private: dict[int, _PrivateCopy],
        built_views: dict[int, torch.Tensor],
        *,
        pending_keys: list[int] | None = None,
    ) -> PinLease:
        key = self._next_lease
        self._next_lease += 1
        pageable_pointers = () if len(held) == len(requests) else tuple(
            pointer for pointer in requests if pointer not in held
        )
        registered = sum(registration.size for registration in held.values())
        pageable_bytes = (
            sum(requests[pointer].storage.nbytes() for pointer in pageable_pointers)
            if pageable_pointers
            else 0
        )
        prepared: list[
            tuple[int, tuple[torch.Tensor, int, bool] | None, tuple[torch.Tensor, int, bool]]
        ] = []
        view_ids: deque[int] = deque()
        for pointer, request in requests.items():
            is_pinned = pointer in held
            for tensor in request.tensors:
                tid = id(tensor)
                prev = self._leased_views.get(tid)
                if prev is not None:
                    nxt = (prev[0], prev[1] + 1, prev[2])
                else:
                    view = built_views.get(tid, tensor)
                    nxt = (view, 1, is_pinned)
                prepared.append((tid, prev, nxt))
                view_ids.append(tid)
        state = _LeaseState(
            deque(held.values()),
            deque(held_private.values()),
            deque(pageable_pointers),
            tuple(tensor for request in requests.values() for tensor in request.tensors),
            view_ids,
        )
        lease = PinLease(self, key, registered, pageable_bytes)
        for tid, _prev, committed_entry in prepared:
            self._leased_views[tid] = committed_entry
        built_views.clear()
        self._leases[key] = state
        _live_managers.add(self)
        if pending_keys:
            for pkey in pending_keys:
                self._pending.pop(pkey, None)
            pending_keys.clear()
            self._pending_changed.notify_all()
        return lease

    def _release_registrations_locked(self, registrations: deque[_Registration]) -> list[_Registration]:
        retired: list[_Registration] = []
        while registrations:
            registration = registrations.popleft()
            registration.leases -= 1
            if registration.leases == 0:
                self._idle[registration.pointer] = None
                if registration.retired:
                    retired.append(registration)
        return retired

    def _pop_one_retired_locked(
        self, registration: _Registration, retired_2a: list[ArenaAllocation]
    ) -> None:
        if registration.leases == 0 and self._registrations.get(registration.pointer) is registration:
            if registration.copy is None:
                self._unregister_in_place_locked(registration)
            else:
                del self._registrations[registration.pointer]
                self._starts.remove(registration.pointer)
                self._idle.pop(registration.pointer, None)
                retired_2a.append(registration.copy)

    def _pop_retired_registrations_locked(
        self, retired: deque[_Registration] | Sequence[_Registration], retired_2a: list[ArenaAllocation]
    ) -> None:
        if isinstance(retired, deque):
            while retired:
                self._pop_one_retired_locked(retired[0], retired_2a)
                retired.popleft()
            return
        for registration in retired:
            self._pop_one_retired_locked(registration, retired_2a)

    def _release_pageable_locked(self, pointers: deque[int]) -> None:
        while pointers:
            pointer = pointers.popleft()
            pageable = self._pageable[pointer]
            pageable.leases -= 1
            if pageable.leases == 0:
                del self._pageable[pointer]
                self._starts.remove(pointer)

    def _release_private_copies_locked(
        self, privates: deque[_PrivateCopy], retired_2b: list[ArenaAllocation]
    ) -> None:
        while privates:
            priv = privates.popleft()
            priv.leases -= 1
            if priv.leases == 0 and self._private_copies.get(priv.pointer) is priv:
                self._private_copies.move_to_end(priv.pointer)
                if priv.retired:
                    del self._private_copies[priv.pointer]
                    retired_2b.append(priv.alloc)

    def _release_private_candidate_holds_locked(
        self,
        candidates: Sequence[_Candidate],
        retired_2b: list[ArenaAllocation],
    ) -> None:
        privates: deque[_PrivateCopy] = deque()
        for candidate in candidates:
            if candidate.private_source is not None:
                privates.append(candidate.private_source)
                candidate.private_source = None
        if privates:
            self._release_private_copies_locked(privates, retired_2b)

    def _release_private_candidate_holds(self, candidates: Sequence[_Candidate]) -> None:
        if not any(candidate.private_source is not None for candidate in candidates):
            return
        retired_2b: list[ArenaAllocation] = []
        with self._lock:
            self._release_private_candidate_holds_locked(candidates, retired_2b)
            self._drop_lifetime_root_if_empty_locked()
        self._drop_detached_allocations(allocs_2b=retired_2b, trim_2b=bool(retired_2b))

    def _close_lease(self, key: int) -> None:
        retired_2a: list[ArenaAllocation] = []
        retired_2b: list[ArenaAllocation] = []
        budget: int | None = None
        host_budget: int | None = None
        closed = False
        try:
            with self._lock:
                state = self._leases.get(key)
                if state is None:
                    return
                try:
                    self._release_views_locked(state.view_ids)
                    self._release_pageable_locked(state.pageable)
                    self._release_private_copies_locked(state.private_copies, retired_2b)
                    state.retired_registrations.extend(
                        self._release_registrations_locked(state.registrations)
                    )
                    self._pop_retired_registrations_locked(state.retired_registrations, retired_2a)
                    del self._leases[key]
                    self._lease_close_seq += 1
                    budget = self._max_pinned_bytes
                    host_budget = self._max_host_bytes
                    closed = True
                finally:
                    self._drop_lifetime_root_if_empty_locked()
        finally:
            if retired_2a or retired_2b:
                self._drop_detached_allocations(
                    retired_2a,
                    retired_2b,
                    trim_2a=bool(retired_2a),
                    trim_2b=bool(retired_2b),
                )
                with self._lock:
                    self._drop_lifetime_root_if_empty_locked()

        if closed:
            if budget is not None:
                self._enforce_pin_budget()
            elif host_budget is not None:
                self._enforce_host_budget()
            with self._lock:
                self._drop_lifetime_root_if_empty_locked()

    def _drop_lifetime_root_if_empty_locked(self) -> None:
        if (
            not self._registrations
            and not self._leases
            and not self._private_copies
            and not self._resources.retaining_rings
            and not self._resources.closing_rings
            and self._resources.creating_ring is None
        ):
            _live_managers.discard(self)

    # ------------------------------------------------------------------
    # Eviction, Tier 2a -> Tier 2b demotion, and cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def _release_allocations(allocs: list[ArenaAllocation]) -> None:
        while allocs:
            allocs.pop().release()

    def _drop_detached_allocations(
        self,
        allocs_2a: list[ArenaAllocation] | None = None,
        allocs_2b: list[ArenaAllocation] | None = None,
        *,
        trim_2a: bool = False,
        trim_2b: bool = False,
    ) -> None:
        try:
            if allocs_2a:
                self._release_allocations(allocs_2a)
        finally:
            if allocs_2b:
                self._release_allocations(allocs_2b)
        try:
            if trim_2a:
                self._tier2a.trim_empty(0)
        finally:
            if trim_2b:
                self._tier2b.trim_empty(0)

    def _unregister_in_place_locked(self, registration: _Registration) -> bool:
        assert registration.leases == 0
        assert registration.copy is None
        try:
            self._backend.unregister(registration.pointer)
        except Exception as error:
            self._unregistration_failures += 1
            logger.warning(
                "Host unregistration failed; retaining storage and budget reservation: %s",
                str(error),
            )
            return False
        del self._registrations[registration.pointer]
        self._starts.remove(registration.pointer)
        self._idle.pop(registration.pointer, None)
        self._unreserve_range_locked(registration.pointer, registration.size)
        self._drop_lifetime_root_if_empty_locked()
        return True

    def _allocate_demotion_slot(
        self,
        reg: _Registration,
        alloc: ArenaAllocation,
        host_epoch: int,
        host_budget: int | None,
        in_place: int,
    ) -> ArenaAllocation | None:
        needed_2b = _page_rounded(reg.size)
        if host_budget is not None:
            bounce_bytes, _, _ = self._active_bounce_bytes_off_lock()
            avail_2b = max(0, host_budget - (in_place + self._tier2a.stats.reserved_bytes + bounce_bytes))
            if needed_2b > avail_2b or not self._evict_private_to_fit(max(0, avail_2b - needed_2b)):
                return None
            new_limit: int | None = avail_2b
        else:
            new_limit = None

        with self._lock:
            if self._host_budget_epoch != host_epoch:
                return None
        self._tier2b.max_bytes = new_limit
        with self._lock:
            if self._host_budget_epoch != host_epoch:
                return None

        priv_alloc = self._tier2b.allocate(
            reg.size, head_pad=0, alignment=mmap.PAGESIZE, slab_hint_bytes=needed_2b
        )
        if priv_alloc is None:
            return None
        try:
            priv_alloc.as_uint8_tensor().copy_(alloc.as_uint8_tensor())
            return priv_alloc
        except BaseException:
            priv_alloc.release()
            self._tier2b.trim_empty(0)
            raise

    def _demote_tier2a_copy(self, reg: _Registration) -> None:
        """Demote one popped Tier 2a copy to Tier 2b if ``max_host_bytes`` allows, then release its 2a span."""
        alloc = reg.copy
        assert alloc is not None
        host_epoch: int | None = None
        priv_alloc: ArenaAllocation | None = None
        try:
            if reg.retired:
                return
            with self._lock:
                host_epoch = self._host_budget_epoch
                host_budget = self._max_host_bytes
                in_place = self._in_place_pinned_bytes
            if host_budget == 0:
                return

            priv_alloc = self._allocate_demotion_slot(reg, alloc, host_epoch, host_budget, in_place)
            if priv_alloc is None:
                return

            with self._lock:
                if (
                    not reg.retired
                    and self._host_budget_epoch == host_epoch
                    and self._max_host_bytes != 0
                    and reg.pointer not in self._registrations
                    and reg.pointer not in self._private_copies
                ):
                    self._private_copies[reg.pointer] = _PrivateCopy(
                        pointer=reg.pointer,
                        size=reg.size,
                        storage=reg.storage,
                        alloc=priv_alloc,
                        owners=reg.owners,
                        tracker=reg.tracker,
                    )
                    self._private_copies.move_to_end(reg.pointer)
                    self._host_budget_epoch += 1
                    host_epoch = self._host_budget_epoch
                    _live_managers.add(self)
                    priv_alloc = None
        finally:
            try:
                if priv_alloc is not None:
                    unclaimed = priv_alloc
                    priv_alloc = None
                    try:
                        unclaimed.release()
                    finally:
                        self._tier2b.trim_empty(0)
            finally:
                alloc.release()
            if host_epoch is not None:
                with self._lock:
                    epoch_changed = self._host_budget_epoch != host_epoch
                if epoch_changed:
                    self._enforce_host_budget()

    def _evict_private_to_fit(self, target_2b_reserved: int) -> bool:
        """Evict LRU idle Tier 2b entries and trim empty Tier 2b slabs down to ``target_2b_reserved``."""
        while True:
            self._tier2b.trim_empty(target_2b_reserved)
            if self._tier2b.stats.reserved_bytes <= target_2b_reserved:
                return True
            popped: _PrivateCopy | None = None
            with self._lock:
                for ptr, priv in self._private_copies.items():
                    if priv.leases == 0:
                        popped = self._private_copies.pop(ptr)
                        break
                self._drop_lifetime_root_if_empty_locked()
            if popped is None:
                return False
            popped.alloc.release()

    def _enforce_host_budget(self) -> None:
        while True:
            with self._lock:
                epoch = self._host_budget_epoch
                host_budget = self._max_host_bytes
                in_place = self._in_place_pinned_bytes
            if host_budget is None:
                self._tier2b.max_bytes = None
            else:
                tier2a_bytes = self._tier2a.stats.reserved_bytes
                bounce_bytes, _, _ = self._active_bounce_bytes_off_lock()
                avail_2b = max(0, host_budget - (in_place + tier2a_bytes + bounce_bytes))
                self._tier2b.max_bytes = avail_2b
                self._evict_private_to_fit(avail_2b)
                with self._lock:
                    if self._host_budget_epoch != epoch:
                        continue
                    cur_in_place = self._in_place_pinned_bytes
                cur_tier2a = self._tier2a.stats.reserved_bytes
                cur_bounce, _, _ = self._active_bounce_bytes_off_lock()
                cur_avail = max(0, host_budget - (cur_in_place + cur_tier2a + cur_bounce))
                if cur_avail != avail_2b:
                    self._tier2b.max_bytes = cur_avail
                    self._evict_private_to_fit(cur_avail)
            with self._lock:
                if self._host_budget_epoch == epoch:
                    return

    def _pop_next_idle_locked(
        self, *, protected: set[int], unreleasable: set[int]
    ) -> tuple[bool, _Registration | None]:
        """Evict the oldest un-protected idle registration, returning ``(did_unreg_in_place, popped_tier2a)``."""
        for ptr in tuple(self._idle):
            if ptr in protected or ptr in unreleasable:
                continue
            reg = self._registrations.get(ptr)
            if reg is None:
                self._idle.pop(ptr, None)
                continue
            if reg.copy is None:
                if self._unregister_in_place_locked(reg):
                    return True, None
                unreleasable.add(ptr)
                continue
            del self._registrations[ptr]
            self._starts.remove(ptr)
            self._idle.pop(ptr, None)
            return False, reg
        return False, None

    def _evict_idle_for_budget(
        self,
        target_pinned: int,
        *,
        protected: set[int],
        close_bounce_last: bool,
    ) -> bool:
        """Evict idle registrations (and trim empty Tier 2a slabs) until ``pinned_bytes <= target_pinned``."""
        unreleasable: set[int] = set()
        progress = False
        while True:
            bounce_bytes, _, _ = self._active_bounce_bytes_off_lock()
            in_place = self._in_place_pinned_bytes_atomic()
            target_2a = max(0, target_pinned - in_place - bounce_bytes)
            before_2a = self._tier2a.stats.reserved_bytes
            self._tier2a.trim_empty(target_2a)
            after_2a = self._tier2a.stats.reserved_bytes
            if after_2a < before_2a:
                progress = True
            if in_place + after_2a + bounce_bytes <= target_pinned:
                return True

            with self._lock:
                unreg_in_place, popped_2a = self._pop_next_idle_locked(
                    protected=protected, unreleasable=unreleasable
                )
            if unreg_in_place:
                progress = True
                continue
            if popped_2a is not None:
                progress = True
                self._demote_tier2a_copy(popped_2a)
                continue
            if close_bounce_last and self._close_idle_bounce_ring_if_present():
                progress = True
                continue
            return progress

    def _evict_one_for_tier2a(self, avail_2a: int | None, *, protected: set[int]) -> bool:
        """Release one idle Tier 2a span for in-slab reuse, trim empty slabs, or evict an idle in-place entry."""
        unreleasable: set[int] = set()
        with self._lock:
            unreg_in_place, popped_2a = self._pop_next_idle_locked(
                protected=protected, unreleasable=unreleasable
            )
        if unreg_in_place:
            return True
        if popped_2a is not None:
            # Release the span without prematurely unregistering its warm pre-pinned slab.
            self._demote_tier2a_copy(popped_2a)
            return True
        if avail_2a is not None:
            before = self._tier2a.stats.reserved_bytes
            self._tier2a.trim_empty(avail_2a)
            if self._tier2a.stats.reserved_bytes < before:
                return True
        return False

    def _evict_idle_for_runtime_retry(self, target_bytes: int, *, protected: set[int]) -> bool:
        """Evict an LRU batch and unregister empty Tier 2a slabs before retrying runtime capacity."""
        unreleasable: set[int] = set()
        before_total = (
            self._in_place_pinned_bytes_atomic()
            + self._tier2a.stats.reserved_bytes
            + self._active_bounce_bytes_off_lock()[0]
        )

        while True:
            self._tier2a.trim_empty(0)
            cur_total = (
                self._in_place_pinned_bytes_atomic()
                + self._tier2a.stats.reserved_bytes
                + self._active_bounce_bytes_off_lock()[0]
            )
            if before_total - cur_total >= target_bytes:
                return True

            with self._lock:
                unreg_in_place, popped_2a = self._pop_next_idle_locked(
                    protected=protected, unreleasable=unreleasable
                )
            if unreg_in_place:
                continue
            if popped_2a is not None:
                self._demote_tier2a_copy(popped_2a)
                continue
            if self._close_idle_bounce_ring_if_present():
                continue
            self._tier2a.trim_empty(0)
            cur_total = (
                self._in_place_pinned_bytes_atomic()
                + self._tier2a.stats.reserved_bytes
                + self._active_bounce_bytes_off_lock()[0]
            )
            return cur_total < before_total

    def _enforce_pin_budget(self) -> None:
        while True:
            with self._lock:
                epoch = self._pin_budget_epoch
                budget = self._max_pinned_bytes
                in_place = self._in_place_pinned_bytes
            if budget is None:
                self._tier2a.max_bytes = None
            else:
                desired_bounce = self._desired_bounce_bytes(budget) or 0
                if budget == 0 or self._active_bounce_bytes_off_lock()[0] > desired_bounce:
                    self._close_idle_bounce_ring_if_present()
                bounce_bytes, _, _ = self._active_bounce_bytes_off_lock()
                if in_place + self._tier2a.stats.reserved_bytes + bounce_bytes > budget:
                    self._evict_idle_for_budget(budget, protected=set(), close_bounce_last=True)
                with self._lock:
                    if self._pin_budget_epoch != epoch:
                        continue
                    cur_in_place = self._in_place_pinned_bytes
                self._tier2a.max_bytes = max(
                    0, budget - cur_in_place - self._active_bounce_bytes_off_lock()[0]
                )
            with self._lock:
                if self._pin_budget_epoch == epoch:
                    break
        self._enforce_host_budget()

    def _in_place_pinned_bytes_atomic(self) -> int:
        with self._lock:
            return self._in_place_pinned_bytes

    def _close_idle_bounce_ring_if_present(self) -> bool:
        with self._lock:
            if self._bounce_creating or self._bounce_closing:
                return False
            has_idle_active = self._resources.bounce_ring is not None and self._bounce_users == 0
            has_retained = bool(self._resources.retaining_rings)
            if not has_idle_active and not has_retained:
                return False
        closed, _failed = self._close_bounce_rings(wait_for_users=False)
        return closed > 0

    # ------------------------------------------------------------------
    # Bounce ring lifecycle and CUDA unpinned streaming
    # ------------------------------------------------------------------

    @staticmethod
    def _is_standard_strided(tensor: object) -> bool:
        return isinstance(tensor, torch.Tensor) and type(tensor) in (
            torch.Tensor,
            torch.nn.Parameter,
        ) and tensor.layout is torch.strided

    def _is_bounce_transfer_eligible(
        self, destination: torch.Tensor, source: torch.Tensor
    ) -> bool:
        return (
            self._is_standard_strided(destination)
            and self._is_standard_strided(source)
            and not destination.requires_grad
            and not source.requires_grad
            and destination.device.type == "cuda"
            and source.device.type == "cpu"
            and destination.shape == source.shape
            and destination.dtype is source.dtype
            and destination.is_contiguous()
            and source.is_contiguous()
            and not destination.is_conj()
            and not destination.is_neg()
            and not source.is_conj()
            and not source.is_neg()
            and source.nbytes > 0
            and source.nbytes >= self._min_bounce_bytes
        )

    def _clear_bounce_creating(self) -> None:
        with self._lock:
            if self._bounce_creating:
                self._bounce_creating = False
                self._bounce_cv.notify_all()

    def _release_bounce_user(self) -> None:
        ring_to_check: PinnedBounceRing | None = None
        budget: int | None = None
        bounce_cfg: int | None = None
        in_place = 0
        with self._lock:
            self._bounce_users -= 1
            if self._bounce_users == 0:
                self._bounce_cv.notify_all()
                ring_to_check = self._resources.bounce_ring
                budget = self._max_pinned_bytes
                bounce_cfg = self._bounce_bytes_config
                in_place = self._in_place_pinned_bytes
            self._drop_lifetime_root_if_empty_locked()
        if ring_to_check is None:
            return
        if bounce_cfg == 0:
            self._close_idle_bounce_ring_if_present()
        if budget is None:
            return
        if budget == 0:
            self._enforce_pin_budget()
            return
        total_bounce = self._active_bounce_bytes_off_lock()[0]
        desired = self._desired_bounce_bytes(budget) or 0
        if (
            total_bounce > desired
            or in_place + self._tier2a.stats.reserved_bytes + total_bounce > budget
        ):
            self._enforce_pin_budget()

    @contextmanager
    def _use_bounce_ring(self) -> Generator[PinnedBounceRing | None]:
        created_by_self = False
        while True:
            should_create = False
            with self._lock:
                while self._bounce_creating or self._bounce_closing:
                    self._bounce_cv.wait()
                if self._max_pinned_bytes == 0 or self._bounce_bytes_config == 0:
                    yield None
                    return
                ring = self._resources.bounce_ring
                if ring is not None:
                    self._bounce_users += 1
                    break
                if self._desired_bounce_bytes(self._max_pinned_bytes) == 0:
                    yield None
                    return
                self._bounce_creating = True
                should_create = True

            if should_create:
                ring = self._create_and_publish_bounce_ring()
                if ring is None:
                    yield None
                    return
                created_by_self = True
                break

        try:
            if created_by_self:
                self._enforce_pin_budget()
            yield ring
        finally:
            self._release_bounce_user()

    def _cleanup_failed_ring(self, ring: PinnedBounceRing, *, track_creating: bool) -> None:
        with self._lock:
            if track_creating and self._resources.creating_ring is ring:
                self._resources.creating_ring = None
            self._resources.closing_rings.append(ring)
        reg_fails = 0
        unreg_fails = 0
        retained = False
        try:
            ring.close()
            reg_fails = ring.registration_failures
            unreg_fails = ring.unregistration_failures
        except BaseException as error:
            retained = True
            if not isinstance(error, Exception):
                raise
            reg_fails = ring.registration_failures
            unreg_fails = ring.unregistration_failures
            logger.warning(
                "Failed to release PinnedBounceRing registration; retaining ring: %s",
                str(error),
            )
        finally:
            with self._lock:
                with contextlib.suppress(ValueError):
                    self._resources.closing_rings.remove(ring)
                if retained:
                    self._resources.retaining_rings.append(ring)
                    _live_managers.add(self)
                else:
                    self._retired_bounce_reg_failures += reg_fails
                    self._retired_bounce_unreg_failures += unreg_fails
                self._drop_lifetime_root_if_empty_locked()

    def _compute_bounce_target_bytes(self) -> tuple[bool, int | None]:
        for can_evict in (True, False):
            existing_bounce = self._active_bounce_bytes_off_lock()[0]
            private_bytes = self._tier2b.stats.reserved_bytes
            with self._lock:
                budget = self._max_pinned_bytes
                host_budget = self._max_host_bytes
                in_place = self._in_place_pinned_bytes
            desired = self._desired_bounce_bytes(budget)
            desired_new = (
                None
                if desired is None
                else _align_down(max(0, desired - existing_bounce), mmap.PAGESIZE)
            )
            if desired == 0 or (desired_new is not None and desired_new < _MIN_BOUNCE_SLOT_BYTES):
                return False, None
            combined_budget, host_capped = self._apply_host_ceiling(budget, host_budget, private_bytes)
            if combined_budget is None:
                return True, desired_new
            want = DEFAULT_BOUNCE_SLOTS * self._bounce_slot_bytes if desired_new is None else desired_new
            tier2a_reserved = self._tier2a.stats.reserved_bytes
            needed_pinned = in_place + tier2a_reserved + existing_bounce + want
            if can_evict and needed_pinned > combined_budget:
                if host_capped and self._reserve_host_headroom_for_pinned(host_budget, needed_pinned):
                    combined_budget, _ = self._apply_host_ceiling(
                        budget, host_budget, self._tier2b.stats.reserved_bytes
                    )
                if combined_budget is not None and needed_pinned > combined_budget:
                    self._evict_idle_for_budget(
                        max(0, combined_budget - want), protected=set(), close_bounce_last=False
                    )
                continue
            headroom = _align_down(
                max(0, combined_budget - in_place - tier2a_reserved - existing_bounce),
                mmap.PAGESIZE,
            )
            target_bytes = min(want, headroom)
            return (True, target_bytes) if target_bytes >= _MIN_BOUNCE_SLOT_BYTES else (False, None)
        return False, None

    def _create_bounce_ring(self, *, track_creating: bool = False) -> PinnedBounceRing | None:
        ok_to_create, target_bytes = self._compute_bounce_target_bytes()
        if not ok_to_create:
            return None

        ring = PinnedBounceRing(
            slot_bytes=self._bounce_slot_bytes,
            num_slots=DEFAULT_BOUNCE_SLOTS,
            min_bounce_bytes=self._min_bounce_bytes,
            max_bytes=target_bytes,
            backend=self._backend,
        )
        if track_creating:
            with self._lock:
                self._resources.creating_ring = ring
                self._host_budget_epoch += 1
        try:
            ok = ring._ensure_initialized()
            if not ok and ring.registration_failures > 0:
                geom = ring._compute_geometry()
                retry_target = geom[0] * geom[1] if geom is not None else _MIN_BOUNCE_SLOT_BYTES
                if self._evict_idle_for_runtime_retry(retry_target, protected=set()):
                    ok = ring._ensure_initialized()
        except BaseException:
            self._cleanup_failed_ring(ring, track_creating=track_creating)
            raise

        if not ok:
            self._cleanup_failed_ring(ring, track_creating=track_creating)
            return None
        return ring

    def _publish_bounce_ring_locked(self, ring: PinnedBounceRing) -> PinnedBounceRing:
        self._resources.creating_ring = None
        self._resources.bounce_ring = ring
        self._bounce_users += 1
        self._bounce_creating = False
        self._pin_budget_epoch += 1
        self._host_budget_epoch += 1
        _live_managers.add(self)
        self._bounce_cv.notify_all()
        return ring

    def _release_bounce_publishing(self) -> None:
        with self._lock:
            if self._bounce_publishing:
                self._bounce_publishing = False
                self._bounce_cv.notify_all()

    def _try_evict_before_bounce_publish(self, cur_budget: int) -> bool:
        self._release_bounce_publishing()
        return self._evict_idle_for_budget(cur_budget, protected=set(), close_bounce_last=False)

    def _await_bounce_publish_gate_locked(
        self, ring: PinnedBounceRing
    ) -> tuple[str, tuple[int, int, int | None, int | None, int] | None]:
        while self._pin_reserving > 0:
            self._bounce_cv.wait()
        cur_epoch = self._pin_budget_epoch
        cur_host_epoch = self._host_budget_epoch
        cur_budget = self._max_pinned_bytes
        cur_host_budget = self._max_host_bytes
        if cur_budget == 0 or self._bounce_bytes_config == 0:
            return "abort", None
        if cur_budget is None and (
            cur_host_budget is None or cur_host_budget == 0 or not self._private_copies
        ):
            self._publish_bounce_ring_locked(ring)
            return "published", None
        self._bounce_publishing = True
        return "check", (cur_epoch, cur_host_epoch, cur_budget, cur_host_budget, self._in_place_pinned_bytes)

    def _publish_loop(self, ring: PinnedBounceRing, touched_limit: list[bool]) -> bool:
        ring_bytes = ring.reserved_bytes
        while True:
            with self._lock:
                status, gate = self._await_bounce_publish_gate_locked(ring)
            if status == "abort":
                return False
            if status == "published":
                return True
            assert gate is not None
            cur_epoch, cur_host_epoch, cur_budget, cur_host_budget, cur_in_place = gate

            try:
                if cur_budget is not None and ring_bytes > (self._desired_bounce_bytes(cur_budget) or 0):
                    return False
                total_bounce, _, _ = self._active_bounce_bytes_off_lock()
                private_bytes = self._tier2b.stats.reserved_bytes
                combined_budget, host_capped = self._apply_host_ceiling(
                    cur_budget, cur_host_budget, private_bytes
                )
                required = cur_in_place + self._tier2a.stats.reserved_bytes + total_bounce
                if combined_budget is not None and required > combined_budget:
                    if host_capped:
                        self._release_bounce_publishing()
                        if self._reserve_host_headroom_for_pinned(cur_host_budget, required):
                            continue
                    if self._try_evict_before_bounce_publish(combined_budget):
                        continue
                    return False

                if combined_budget is not None:
                    self._tier2a.max_bytes = max(0, combined_budget - cur_in_place - total_bounce)
                    touched_limit[0] = True
                after_tier2a = self._tier2a.stats.reserved_bytes
                with self._lock:
                    self._bounce_publishing = False
                    if (
                        self._pin_budget_epoch == cur_epoch
                        and self._host_budget_epoch == cur_host_epoch
                        and self._in_place_pinned_bytes == cur_in_place
                        and (
                            combined_budget is None
                            or self._in_place_pinned_bytes + after_tier2a + total_bounce <= combined_budget
                        )
                    ):
                        self._publish_bounce_ring_locked(ring)
                        return True
                    self._bounce_cv.notify_all()
            finally:
                self._release_bounce_publishing()

    def _create_and_publish_bounce_ring(self) -> PinnedBounceRing | None:
        ring: PinnedBounceRing | None = None
        published = False
        touched_limit = [False]
        try:
            ring = self._create_bounce_ring(track_creating=True)
            if ring is None:
                return None
            published = self._publish_loop(ring, touched_limit)
            return ring if published else None
        finally:
            if not published:
                if ring is not None:
                    self._cleanup_failed_ring(ring, track_creating=True)
                self._clear_bounce_creating()
                if touched_limit[0]:
                    self._enforce_pin_budget()

    def _close_bounce_rings(self, *, wait_for_users: bool) -> tuple[int, int]:
        with self._lock:
            while self._bounce_creating or self._bounce_closing or (wait_for_users and self._bounce_users > 0):
                self._bounce_cv.wait()
            rings: list[PinnedBounceRing] = list(self._resources.retaining_rings)
            self._resources.retaining_rings.clear()
            if self._resources.bounce_ring is not None and self._bounce_users == 0:
                rings.append(self._resources.bounce_ring)
                self._resources.bounce_ring = None
            if not rings:
                return 0, 0
            self._resources.closing_rings.extend(rings)
            self._bounce_closing = True

        closed = 0
        failed = 0
        try:
            for ring in rings:
                try:
                    ring.close()
                except Exception as error:
                    failed += 1
                    logger.warning(
                        "Failed to release PinnedBounceRing registration; retaining ring: %s",
                        str(error),
                    )
                    with self._lock:
                        with contextlib.suppress(ValueError):
                            self._resources.closing_rings.remove(ring)
                        self._resources.retaining_rings.append(ring)
                        _live_managers.add(self)
                else:
                    reg_fails = ring.registration_failures
                    unreg_fails = ring.unregistration_failures
                    closed += 1
                    with self._lock:
                        with contextlib.suppress(ValueError):
                            self._resources.closing_rings.remove(ring)
                        self._retired_bounce_reg_failures += reg_fails
                        self._retired_bounce_unreg_failures += unreg_fails
        finally:
            with self._lock:
                if self._resources.closing_rings:
                    self._resources.retaining_rings.extend(self._resources.closing_rings)
                    self._resources.closing_rings.clear()
                    _live_managers.add(self)
                self._bounce_closing = False
                self._bounce_cv.notify_all()
                self._drop_lifetime_root_if_empty_locked()
        return closed, failed

    def _transfer_unpinned_cuda(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        *,
        from_private: bool,
        non_blocking: bool,
    ) -> bool:
        if not self._is_bounce_transfer_eligible(destination, source):
            return False
        source_file = None if from_private else file_slice(source)
        with self._use_bounce_ring() as ring:
            if ring is None:
                return False
            used = False
            if source_file is not None and not getattr(source_file.file, "closed", False):
                view_offset_bytes = int(source.storage_offset()) * source.element_size()
                if view_offset_bytes + source.nbytes <= source_file.length:
                    used = ring.stream_file_slice_to_device(
                        destination,
                        source_file,
                        reader=self._reader,
                        view_offset_bytes=view_offset_bytes,
                        view_nbytes=source.nbytes,
                    )
            if not used:
                used = ring.copy_pageable_to_device(destination, source)
            if used and not non_blocking:
                torch.cuda.current_stream(destination.device).synchronize()
            return used

# Registrations must outlive Python references to a manager. This root
# retains managers with live registrations, private copies, retained bounce
# rings, or leases; idle owners remain weak so dropping the last owner
# retires its entry and releases empty slabs.
_live_managers: set[PinManager] = set()

host_pin_manager = PinManager()


def transfer_(destination: torch.Tensor, source: torch.Tensor, *, non_blocking: bool) -> None:
    """Copy ``source`` into ``destination`` through the process-wide manager.

    Every host-to-device copy of a physical tensor goes through here, so a
    checkpoint storage pinned through a Tier 2a slab is read from that copy
    while the module's own tensor keeps pointing at the resting mapping, and
    Tier 2b / uncached checkpoint tensors stream through the lazy
    ``PinnedBounceRing`` when eligible. An asynchronous copy of pinned storage
    must run under a lease kept until completion, and raises otherwise; a
    synchronous copy needs none.
    """
    host_pin_manager.transfer(destination, source, non_blocking=non_blocking)


__all__ = ["PinLease", "PinManager", "PinStats", "TransferLease", "host_pin_manager", "transfer_"]
