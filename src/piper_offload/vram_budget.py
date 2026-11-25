"""Coordinator-grade shared VRAM budget across caches, pools, and arenas."""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

from .offload_config import validate_vram_budget


class VramBudgetExhaustedError(RuntimeError):
    """Raised when a VRAM reservation exceeds capacity even after evictions."""


def _validate_nonneg_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}.")
    return value


def _validate_priority(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"priority must be an integer; got {value!r}.")
    return value


@dataclass
class _TenantRecord:
    tenant_id: int
    name: str
    priority: int
    evict_cb: Callable[[int], int] | None
    usage_fn: Callable[[], int]
    min_bytes: int
    yield_to_peers: bool = False
    active: bool = True

    def current_bytes(self) -> int:
        if not self.active:
            return 0
        return _validate_nonneg_int(self.usage_fn(), name="usage_fn()")


class VramReservation:
    """Explicit capacity hold in a :class:`VramBudget`."""

    def __init__(
        self,
        budget: VramBudget,
        reservation_id: int,
        name: str,
        nbytes: int,
    ) -> None:
        self._budget = budget
        self._reservation_id = reservation_id
        self.name = name
        self._nbytes = nbytes
        self._released = False

    @property
    def nbytes(self) -> int:
        """Return currently held bytes (``0`` once released)."""
        return 0 if self._released else self._nbytes

    @property
    def is_active(self) -> bool:
        """Return whether this reservation is still held against the budget."""
        return not self._released

    def resize(self, new_bytes: int) -> None:
        """Grow or shrink the reserved byte count cleanly (may raise :class:`VramBudgetExhaustedError`)."""
        self._budget._resize_reservation(self, new_bytes)

    def release(self) -> None:
        """Release reserved bytes back to the budget (idempotent)."""
        self._budget._release_reservation(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class BudgetTenantHandle:
    """Handle representing one registered consumer inside a :class:`VramBudget`."""

    def __init__(self, budget: VramBudget, record: _TenantRecord) -> None:
        self._budget = budget
        self._record = record

    @property
    def tenant_id(self) -> int:
        return self._record.tenant_id

    @property
    def name(self) -> str:
        return self._record.name

    @property
    def priority(self) -> int:
        """Eviction order rank: higher values yield (evict) first."""
        return self._record.priority

    @property
    def min_bytes(self) -> int:
        return self._record.min_bytes

    @property
    def yield_to_peers(self) -> bool:
        """Whether this tenant yields idle bytes to equal-priority peers."""
        return self._record.yield_to_peers

    @property
    def is_active(self) -> bool:
        return self._record.active

    @property
    def allocated_bytes(self) -> int:
        return self._budget._tenant_allocated_bytes(self._record)

    def reclaimable_headroom_bytes(self) -> int | None:
        """Return extra bytes reachable including evictable lower-retention tenants."""
        return self._budget.reclaimable_headroom_for(self)

    def ceiling_bytes(self) -> int | None:
        """Return this tenant's current total cap given other active usage."""
        return self._budget.ceiling_for(self)

    def request_bytes(self, nbytes: int) -> int | None:
        """Evict strictly lower-retention tenants if needed, then return new ceiling."""
        return self._budget.request_bytes(self, nbytes)

    def unregister(self) -> None:
        """Detach this tenant from the budget (idempotent)."""
        self._budget.unregister_tenant(self)


class VramBudget:
    """Thread-safe shared per-GPU VRAM budget coordinating caches, pools, and arenas.

    Each ``VramBudget`` instance tracks one target GPU's VRAM capacity.
    Eviction ordering convention:
    ``priority`` ranks retention value numerically — higher numbers (for
    example ``priority=10`` for re-stageable weight slots) are evicted **before**
    lower numbers (for example ``priority=0`` for live KV blocks). Equal-priority
    peers only yield to each other when registered with ``yield_to_peers=True``.

    Eviction callbacks run strictly outside ``self._lock`` so callbacks can
    safely close CUDA leases or query other state without lock inversion.
    """

    def __init__(
        self,
        limit_bytes: int | None,
        *,
        headroom_bytes: int = 0,
    ) -> None:
        self._limit_bytes = validate_vram_budget(limit_bytes)
        self._headroom_bytes = _validate_nonneg_int(
            headroom_bytes, name="headroom_bytes"
        )
        self._lock = threading.Lock()
        self._next_tenant_id = 1
        self._next_reservation_id = 1
        self._tenants: dict[int, _TenantRecord] = {}
        self._reservations: dict[int, VramReservation] = {}

    def _effective_limit_unlocked(self) -> int | None:
        if self._limit_bytes is None:
            return None
        return max(0, self._limit_bytes - self._headroom_bytes)

    def _reserved_bytes_unlocked(self) -> int:
        return sum(res._nbytes for res in self._reservations.values() if not res._released)

    def _tenant_bytes_unlocked(self, *, exclude_id: int | None = None) -> int:
        return sum(
            record.current_bytes()
            for tid, record in self._tenants.items()
            if tid != exclude_id and record.active
        )

    def _used_bytes_unlocked(self) -> int:
        return self._reserved_bytes_unlocked() + self._tenant_bytes_unlocked()

    @property
    def limit_bytes(self) -> int | None:
        """Configured hard VRAM limit before subtracting headroom, or ``None``."""
        with self._lock:
            return self._limit_bytes

    @property
    def headroom_bytes(self) -> int:
        """Reserved driver/workspace safety headroom subtracted from ``limit_bytes``."""
        with self._lock:
            return self._headroom_bytes

    @property
    def effective_limit_bytes(self) -> int | None:
        """Net budget ceiling after headroom, or ``None`` if unbounded."""
        with self._lock:
            return self._effective_limit_unlocked()

    @property
    def reserved_bytes(self) -> int:
        """Total active bytes held via :class:`VramReservation` handles."""
        with self._lock:
            return self._reserved_bytes_unlocked()

    @property
    def tenant_bytes(self) -> int:
        """Total bytes reported across registered tenants."""
        with self._lock:
            return self._tenant_bytes_unlocked()

    @property
    def used_bytes(self) -> int:
        """Sum of active reservation and tenant bytes."""
        with self._lock:
            return self._used_bytes_unlocked()

    @property
    def available_bytes(self) -> int | None:
        """Unclaimed bytes remaining under the effective ceiling, or ``None``."""
        with self._lock:
            limit = self._effective_limit_unlocked()
            if limit is None:
                return None
            return max(0, limit - self._used_bytes_unlocked())

    def _extract_tenant_id(self, tenant: BudgetTenantHandle | int) -> int:
        if isinstance(tenant, BudgetTenantHandle):
            if tenant._budget is not self:
                raise ValueError(
                    "BudgetTenantHandle does not belong to this VramBudget."
                )
            return tenant.tenant_id
        if isinstance(tenant, bool) or not isinstance(tenant, int):
            raise TypeError(
                f"tenant must be a BudgetTenantHandle or int; got {tenant!r}."
            )
        return tenant

    def _resolve_tenant_unlocked(self, tenant: BudgetTenantHandle | int) -> _TenantRecord:
        tid = self._extract_tenant_id(tenant)
        record = self._tenants.get(tid)
        if record is None or not record.active:
            raise KeyError(f"Tenant {tid} is not actively registered in VramBudget.")
        return record

    def _tenant_allocated_bytes(self, record: _TenantRecord) -> int:
        with self._lock:
            return record.current_bytes()

    def register_tenant(
        self,
        name: str,
        *,
        usage_fn: Callable[[], int],
        priority: int = 0,
        evict_cb: Callable[[int], int] | None = None,
        min_bytes: int = 0,
        yield_to_peers: bool = False,
    ) -> BudgetTenantHandle:
        """Register an adaptive consumer backed by live ``usage_fn``.

        Higher ``priority`` values are evicted **before** lower values. Pass
        ``yield_to_peers=True`` to allow equal-priority siblings (such as
        multiple :class:`ModelOffloader` runtimes sharing one GPU) to evict idle
        bytes from this tenant. ``usage_fn`` is the single source of truth for
        tracked tenant bytes and must not re-enter ``self``.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Tenant name must be a non-empty string.")
        if not callable(usage_fn):
            raise TypeError("usage_fn must be callable.")
        if evict_cb is not None and not callable(evict_cb):
            raise TypeError("evict_cb must be callable or None.")
        if not isinstance(yield_to_peers, bool):
            raise ValueError(
                f"yield_to_peers must be a bool; got {yield_to_peers!r}."
            )
        prio = _validate_priority(priority)
        min_b = _validate_nonneg_int(min_bytes, name="min_bytes")
        with self._lock:
            tid = self._next_tenant_id
            self._next_tenant_id += 1
            record = _TenantRecord(
                tenant_id=tid,
                name=name,
                priority=prio,
                evict_cb=evict_cb,
                usage_fn=usage_fn,
                min_bytes=min_b,
                yield_to_peers=yield_to_peers,
            )
            self._tenants[tid] = record
        return BudgetTenantHandle(self, record)

    def unregister_tenant(self, tenant: BudgetTenantHandle | int) -> None:
        """Remove a registered tenant (idempotent)."""
        tid = self._extract_tenant_id(tenant)
        with self._lock:
            record = self._tenants.pop(tid, None)
            if record is not None:
                record.active = False

    @staticmethod
    def _can_yield_to(
        record: _TenantRecord,
        *,
        requester_priority: int | None = None,
        exclude_id: int | None = None,
    ) -> bool:
        if not record.active or record.tenant_id == exclude_id:
            return False
        if record.evict_cb is None:
            return False
        if requester_priority is None:
            return True
        if record.priority > requester_priority:
            return True
        return record.priority == requester_priority and record.yield_to_peers

    def reclaimable_headroom_for(self, tenant: BudgetTenantHandle | int) -> int | None:
        """Return additional headroom reachable by ``tenant`` (including evictable peers)."""
        with self._lock:
            target = self._resolve_tenant_unlocked(tenant)
            limit = self._effective_limit_unlocked()
            if limit is None:
                return None
            used = self._used_bytes_unlocked()
            evictable = sum(
                max(0, record.current_bytes() - record.min_bytes)
                for record in self._tenants.values()
                if self._can_yield_to(
                    record,
                    requester_priority=target.priority,
                    exclude_id=target.tenant_id,
                )
            )
            return max(0, (limit - used) + evictable)

    def ceiling_for(self, tenant: BudgetTenantHandle | int) -> int | None:
        """Return max total bytes ``tenant`` may hold without overflowing budget."""
        with self._lock:
            target = self._resolve_tenant_unlocked(tenant)
            limit = self._effective_limit_unlocked()
            if limit is None:
                return None
            other_used = self._reserved_bytes_unlocked() + self._tenant_bytes_unlocked(
                exclude_id=target.tenant_id
            )
            return max(0, limit - other_used)

    def _evict_candidates_unlocked(
        self,
        *,
        requester_priority: int | None = None,
        exclude_id: int | None = None,
    ) -> list[tuple[_TenantRecord, int]]:
        candidates: list[tuple[_TenantRecord, int]] = []
        for record in self._tenants.values():
            if not self._can_yield_to(
                record,
                requester_priority=requester_priority,
                exclude_id=exclude_id,
            ):
                continue
            reducible = max(0, record.current_bytes() - record.min_bytes)
            if reducible > 0:
                candidates.append((record, reducible))
        candidates.sort(key=lambda item: (-item[0].priority, item[0].tenant_id))
        return candidates

    def _evict_step(
        self,
        deficit: int,
        *,
        requester_priority: int | None = None,
        exclude_id: int | None = None,
    ) -> int:
        with self._lock:
            candidates = self._evict_candidates_unlocked(
                requester_priority=requester_priority,
                exclude_id=exclude_id,
            )
        if not candidates:
            return 0
        freed_total = 0
        remaining = deficit
        for record, _reducible in candidates:
            if remaining <= 0:
                break
            with self._lock:
                cb = record.evict_cb
                if cb is None or not record.active:
                    continue
                before = record.current_bytes()
                ask = min(remaining, max(0, before - record.min_bytes))
            if ask <= 0:
                continue
            _validate_nonneg_int(cb(ask), name="evict_cb()")
            with self._lock:
                actual = (
                    max(0, before - record.current_bytes())
                    if record.active
                    else before
                )
            if actual > 0:
                freed_total += actual
                remaining = max(0, remaining - actual)
        return freed_total

    def request_bytes(
        self,
        tenant: BudgetTenantHandle | int,
        nbytes: int,
    ) -> int | None:
        """Make room for an additional ``nbytes`` by evicting yielding/lower-retention peers."""
        needed = _validate_nonneg_int(nbytes, name="nbytes")
        while True:
            with self._lock:
                target = self._resolve_tenant_unlocked(tenant)
                limit = self._effective_limit_unlocked()
                if limit is None:
                    return None
                used = self._used_bytes_unlocked()
                deficit = used + needed - limit
                if deficit <= 0:
                    return self._ceiling_for_unlocked(target)
                target_id = target.tenant_id
                target_priority = target.priority
            freed = self._evict_step(
                deficit,
                requester_priority=target_priority,
                exclude_id=target_id,
            )
            if freed <= 0:
                return self.ceiling_for(target_id)

    def _ceiling_for_unlocked(self, target: _TenantRecord) -> int | None:
        limit = self._effective_limit_unlocked()
        if limit is None:
            return None
        other_used = self._reserved_bytes_unlocked() + self._tenant_bytes_unlocked(
            exclude_id=target.tenant_id
        )
        return max(0, limit - other_used)

    def _available_unlocked(self) -> int | None:
        limit = self._effective_limit_unlocked()
        if limit is None:
            return None
        return max(0, limit - self._used_bytes_unlocked())

    def _create_reservation_unlocked(self, name: str, requested: int) -> VramReservation:
        rid = self._next_reservation_id
        self._next_reservation_id += 1
        res = VramReservation(self, rid, name, requested)
        self._reservations[rid] = res
        return res

    def reserve(self, nbytes: int, *, name: str = "reservation") -> VramReservation:
        """Reserve ``nbytes``, evicting idle/adaptable tenants if needed.

        Raises :class:`VramBudgetExhaustedError` if capacity remains insufficient
        after evicting all eligible tenants.
        """
        requested = _validate_nonneg_int(nbytes, name="nbytes")
        if not isinstance(name, str) or not name:
            raise ValueError("Reservation name must be a non-empty string.")
        while True:
            with self._lock:
                limit = self._effective_limit_unlocked()
                used = self._used_bytes_unlocked()
                if limit is None or used + requested <= limit:
                    return self._create_reservation_unlocked(name, requested)
                deficit = used + requested - limit
            if self._evict_step(deficit) <= 0:
                with self._lock:
                    limit = self._effective_limit_unlocked()
                    used = self._used_bytes_unlocked()
                    if limit is None or used + requested <= limit:
                        return self._create_reservation_unlocked(name, requested)
                    avail = max(0, limit - used)
                raise VramBudgetExhaustedError(
                    f"Cannot reserve {requested} bytes for {name!r}; only {avail} bytes available."
                )

    def _resize_reservation(self, res: VramReservation, new_bytes: int) -> None:
        target_bytes = _validate_nonneg_int(new_bytes, name="new_bytes")
        while True:
            with self._lock:
                if res._released:
                    raise RuntimeError("Cannot resize a released VramReservation.")
                delta = target_bytes - res._nbytes
                if delta <= 0:
                    res._nbytes = target_bytes
                    return
                limit = self._effective_limit_unlocked()
                used = self._used_bytes_unlocked()
                if limit is None or used + delta <= limit:
                    res._nbytes = target_bytes
                    return
                deficit = used + delta - limit
            if self._evict_step(deficit) <= 0:
                with self._lock:
                    if res._released:
                        raise RuntimeError("Cannot resize a released VramReservation.")
                    delta = target_bytes - res._nbytes
                    limit = self._effective_limit_unlocked()
                    used = self._used_bytes_unlocked()
                    if delta <= 0 or limit is None or used + delta <= limit:
                        res._nbytes = target_bytes
                        return
                    avail = max(0, limit - used)
                raise VramBudgetExhaustedError(
                    f"Cannot grow reservation {res.name!r} by {delta} bytes; "
                    f"only {avail} bytes available."
                )

    def _release_reservation(self, res: VramReservation) -> None:
        with self._lock:
            if res._released:
                return
            res._released = True
            res._nbytes = 0
            self._reservations.pop(res._reservation_id, None)

    def set_limit(
        self,
        limit_bytes: int | None,
        *,
        headroom_bytes: int | None = None,
        evict: bool = True,
    ) -> None:
        """Update the budget ceiling and optionally evict tenants to meet it."""
        validated_limit = validate_vram_budget(limit_bytes)
        validated_headroom = (
            None
            if headroom_bytes is None
            else _validate_nonneg_int(headroom_bytes, name="headroom_bytes")
        )
        with self._lock:
            self._limit_bytes = validated_limit
            if validated_headroom is not None:
                self._headroom_bytes = validated_headroom
        if not evict:
            return
        while True:
            with self._lock:
                eff = self._effective_limit_unlocked()
                if eff is None:
                    return
                deficit = self._used_bytes_unlocked() - eff
                if deficit <= 0:
                    return
            if self._evict_step(deficit) <= 0:
                return


__all__ = [
    "BudgetTenantHandle",
    "VramBudget",
    "VramBudgetExhaustedError",
    "VramReservation",
]
