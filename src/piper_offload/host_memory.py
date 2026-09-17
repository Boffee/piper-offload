"""Registry of host backings and the pin budget applied to them.

Share one HostMemoryManager across captures to share backing handles and a
pin budget. Host parameters and buffers retain their backings; the manager's
weak index does not keep unused weights alive, and backings do not reference
the manager. Construction and configuration do not initialize CUDA. The
default pin budget is None (native capacity); zero disables registration.

Each backing pins itself (see ``HostBacking.pin``); the manager decides only
whether there is room. A private file mapping is pinned through an owned
copy, and only under a finite budget, so the default unbounded budget never
duplicates a checkpoint into RAM. The budget charges each registration's OS
pages, rounded per allocation. A registration lives as long as its backing:
the last owner's disposal unregisters before the storage is freed. Storage
must not be resized or independently registered while managed here.
"""

import mmap
import threading
import weakref
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from ._host_backing import HostBacking, HostLease
from ._host_registration import HostRegistrationBackend, RuntimeHostRegistration


@dataclass(frozen=True, slots=True)
class HostMemoryStats:
    """Budget, page-rounded pinned bytes, and backing counts."""

    max_pinned_bytes: int | None
    pinned_bytes: int
    copy_bytes: int
    backings: int
    registrations: int
    idle_registrations: int
    active_backings: int


class HostMemoryManager:
    """Share backing handles and register them under one pin budget.

    A finite ``max_pinned_bytes`` bounds pages registered by this manager.
    The default, ``None``, treats native CUDA/HIP capacity as the limit,
    reclaiming unrelated idle registrations when the runtime refuses a new
    allocation.

    ``capture`` returns one handle per storage and records once whether its
    source may be pinned in place. Memory PyTorch did not allocate itself
    (not ``resizable()``) is assumed to be a private file mapping; owners of
    shared or anonymous mappings pass ``pin_in_place=True``. Distinct
    storages must not overlap in memory; use views of one storage for
    aliases. ``acquire`` leases handles: it protects them first, then registers the
    eligible unpinned ones when the budget and runtime allow.

    Registration and backing metadata are read under one reentrant lock;
    budget state derives from the live handles rather than separate counters.
    A backing stays pageable while another lease or an unfinished copy may be
    reading it. Under
    pin pressure idle backings are unpinned least recently released first,
    keeping their copies; ``clear`` evicts idle copies as well.
    """

    def __init__(
        self,
        max_pinned_bytes: int | None = None,
        *,
        backend: HostRegistrationBackend | None = None,
    ) -> None:
        if max_pinned_bytes is not None and max_pinned_bytes < 0:
            raise ValueError("max_pinned_bytes must be >= 0")
        self._max_pinned_bytes = max_pinned_bytes
        self._backend = backend if backend is not None else RuntimeHostRegistration()
        self._lock = threading.RLock()
        self._backings: weakref.WeakValueDictionary[int, HostBacking] = weakref.WeakValueDictionary()

    def capture(
        self, tensors: Iterable[torch.Tensor], *, pin_in_place: bool | None = None,
    ) -> dict[int, HostBacking]:
        """Share backing handles within this manager without retaining owners.

        ``pin_in_place`` overrides the default classification, for example
        ``True`` for a shared mapping the caller owns, which has no
        copy-on-write; it must agree with any existing handle. Host
        parameters and buffers keep handles alive. Keeping a manager alive
        alone does not retain captured weights.
        """
        result: dict[int, HostBacking] = {}
        with self._lock:
            for tensor in tensors:
                if tensor.device.type != "cpu" or tensor.layout is not torch.strided:
                    raise ValueError("Host backings require strided CPU tensors")
                storage = tensor.untyped_storage()
                key = storage._cdata
                backing = self._backings.get(key)
                if backing is not None:
                    if pin_in_place is not None and backing.pin_in_place != pin_in_place:
                        raise ValueError("Host backing was already captured with a different pin_in_place")
                else:
                    backing = HostBacking(storage, self._backend, pin_in_place=pin_in_place)
                    self._backings[key] = backing
                result[key] = backing
        return result

    @property
    def max_pinned_bytes(self) -> int | None:
        with self._lock:
            return self._max_pinned_bytes

    @max_pinned_bytes.setter
    def max_pinned_bytes(self, value: int | None) -> None:
        """Set the budget or enable opportunistic native-capacity discovery.

        ``None`` removes the application byte limit. Native capacity failures
        still reclaim unrelated idle registrations before falling back to
        pageable storage. For a finite limit, releases trim idle excess back
        to budget. Failed unregistrations stay charged and can be retried with
        ``clear()`` or later admission pressure.
        """
        if value is not None and value < 0:
            raise ValueError("max_pinned_bytes must be >= 0")
        with self._lock:
            self._max_pinned_bytes = value
            self._make_room(0)

    @property
    def stats(self) -> HostMemoryStats:
        with self._lock:
            live = self._live()
            pinned = [backing for backing in live if backing.pinned]
            return HostMemoryStats(
                self._max_pinned_bytes,
                sum(backing.page_bytes for backing in pinned),
                sum(backing.copy_bytes for backing in live),
                len(live),
                len(pinned),
                sum(backing.idle for backing in pinned),
                sum(backing.leases > 0 for backing in live),
            )

    def acquire(self, backings: Iterable[HostBacking]) -> HostLease:
        """Lease backings, registering what the budget allows.

        Every requested backing is protected before any registration, so
        admitting one cannot evict another in the same request. A native
        capacity failure reclaims unrelated idle registrations and retries; if
        capacity remains unavailable, later backings in this request skip
        registration. Budget misses and backings with other active readers
        stay pageable, as do private file mappings under an unbounded budget,
        since they would need a copy.
        """
        requested = self._requested(backings)
        with self._lock:
            for backing in requested:
                backing.hold()
            registered: list[HostBacking] = []
            try:
                for backing in requested:
                    if backing.pinned or backing.storage.nbytes() == 0 or backing.leases > 1 or backing.in_flight:
                        continue
                    if backing.needs_copy and self._max_pinned_bytes is None:
                        # Copies duplicate the mapping into RAM; only a finite
                        # budget bounds that.
                        continue
                    if not self._make_room(backing.page_bytes):
                        continue
                    admitted = backing.pin()
                    while not admitted and self._evict_idle(max(mmap.PAGESIZE, backing.page_bytes)):
                        # Native capacity refused: reclaim an LRU batch and retry.
                        admitted = backing.pin()
                    if not admitted:
                        # Native capacity is still unavailable after reclaiming
                        # every idle registration that can help. Avoid one
                        # failed runtime call per remaining backing.
                        break
                    registered.append(backing)
            except BaseException:
                for backing in requested:
                    backing.release()
                for backing in registered:
                    backing.unpin()
                raise
            return HostLease(requested, self._trim)

    def clear(self) -> None:
        """Unregister idle backings and free their copies.

        Live leases remain protected. A failed unregistration retains its
        storage and budget charge; cleanup errors propagate so callers can
        retry without losing ownership of registered memory.
        """
        with self._lock:
            failed = sum(
                not backing.evict()
                for backing in self._live()
                if backing.idle and (backing.pinned or backing.copy_bytes)
            )
            if failed:
                raise RuntimeError(f"Could not release {failed} host registration(s); storage remains retained")

    def _trim(self) -> None:
        """Evict idle excess after a release under a finite budget."""
        with self._lock:
            self._make_room(0)

    def _live(self) -> list[HostBacking]:
        return list(self._backings.values())

    def _idle_pinned(self) -> list[HostBacking]:
        """Idle registrations, least recently released first."""
        candidates = [backing for backing in self._live() if backing.pinned and backing.idle]
        candidates.sort(key=lambda backing: backing.released_at)
        return candidates

    def _pinned_bytes(self) -> int:
        return sum(backing.page_bytes for backing in self._live() if backing.pinned)

    def _requested(self, backings: Iterable[HostBacking]) -> tuple[HostBacking, ...]:
        requested: dict[int, HostBacking] = {}
        for backing in backings:
            if not isinstance(backing, HostBacking):
                raise TypeError("HostMemoryManager.acquire requires HostBacking handles from capture()")
            if self._backings.get(backing.storage._cdata) is not backing:
                raise ValueError("Host backing was not captured by this manager")
            requested.setdefault(id(backing), backing)
        return tuple(requested.values())

    def _evict_idle(self, target: int) -> int:
        """Unregister idle backings, least recently released first, until ``target`` bytes are freed."""
        freed = 0
        for candidate in self._idle_pinned():
            if freed >= target:
                break
            charge = candidate.page_bytes
            if candidate.unpin():
                freed += charge
        return freed

    def _make_room(self, charge: int) -> bool:
        """Fit ``charge`` more pinned bytes under a finite budget, evicting idle ones if needed."""
        limit = self._max_pinned_bytes
        if limit is None:
            return True
        if charge > limit:
            return False
        excess = self._pinned_bytes() + charge - limit
        return excess <= 0 or self._evict_idle(excess) >= excess


__all__ = ["HostMemoryManager", "HostMemoryStats"]
