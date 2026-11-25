"""Layer 1 Elastic VRAM Pool: capacity-bounded GPU target slot allocator."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from .host_module import HostModuleLoadPlan, HostModuleTarget
from .module_names import group_names
from .offload_config import validate_vram_budget
from .target_lease import CudaTargetLease


class _CapacityClaim(Protocol):
    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class UnitSignature:
    """Layout signature identifying allocation-compatible unit load plans.

    Two plans with equal ``UnitSignature`` allocate identically laid-out GPU
    parameter and buffer targets, allowing safe zero-allocation slot recycling.
    """

    params: tuple[tuple[tuple[str, ...], tuple[object, object]], ...]
    buffers: tuple[tuple[tuple[str, ...], tuple[object, ...]], ...]


def plan_target_signature(plan: HostModuleLoadPlan) -> UnitSignature:
    """Compute the structural GPU allocation signature for ``plan``."""
    param_groups = tuple(
        sorted(
            (
                (tuple(sorted(names)), plan.loads[names[0]].source.target_layout)
                for names in group_names(
                    plan.loads.keys(),
                    lambda name: id(plan.instance.params[name]),
                )
            ),
            key=lambda entry: entry[0],
        )
    )
    buffer_groups = tuple(
        sorted(
            (
                (tuple(sorted(names)), plan.instance.buffers[names[0]].target_layout)
                for names in group_names(
                    plan.instance.buffers.keys(),
                    lambda name: id(plan.instance.buffers[name]),
                )
            ),
            key=lambda entry: entry[0],
        )
    )
    return UnitSignature(params=param_groups, buffers=buffer_groups)


def estimate_plan_gpu_bytes(plan: HostModuleLoadPlan) -> int:
    """Estimate unique GPU backing bytes required to stage ``plan``."""
    total = 0
    seen_params: set[int] = set()
    for name, load in plan.loads.items():
        key = id(plan.instance.params[name])
        if key in seen_params:
            continue
        seen_params.add(key)
        total += load.source.gpu_target_bytes

    seen_buffers: set[int] = set()
    for buf in plan.instance.buffers.values():
        key = id(buf)
        if key in seen_buffers:
            continue
        seen_buffers.add(key)
        total += buf.cache_bytes
    return total


def target_gpu_bytes(target: HostModuleTarget) -> int:
    """Measure unique GPU backing bytes held by an allocated ``target``."""
    total = 0
    seen_params: set[int] = set()
    for param_target in target.param_targets.values():
        key = id(param_target)
        if key in seen_params:
            continue
        seen_params.add(key)
        total += param_target._backing.gpu_target_bytes

    seen_buffers: set[int] = set()
    for buffer_target in target.buffer_targets.values():
        key = id(buffer_target)
        if key in seen_buffers:
            continue
        seen_buffers.add(key)
        total += buffer_target.tensor.nbytes
    return total


@dataclass(slots=True)
class VramSlot:
    """One live GPU target allocation managed by :class:`ElasticVramPool`."""

    lease: CudaTargetLease
    signature: UnitSignature
    nbytes: int
    staged_unit_idx: int | None = None
    last_used_seq: int = 0
    checked_out: bool = False


class ElasticVramPool:
    """Manages reusable CUDA target slots under a configurable VRAM policy.

    - ``vram_budget_bytes is None``: strict 2-slot streaming mode (one active
      slot + one lookahead slot), with zero-allocation same-signature reuse and
      no idle caching beyond 2 slots.
    - ``vram_budget_bytes >= 0``: elastic budget mode retaining staged units in
      VRAM across forwards and evicting furthest-future-use idle slots when a
      new allocation or budget shrink exceeds ``vram_budget_bytes``.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        vram_budget_bytes: int | None,
        budget_fn: Callable[[], int | None] | None = None,
        claim_bytes_fn: Callable[[int], _CapacityClaim | None] | None = None,
    ) -> None:
        self._device = device
        self._budget_bytes = validate_vram_budget(vram_budget_bytes)
        self._budget_fn = budget_fn
        self._claim_bytes_fn = claim_bytes_fn
        self._lock = threading.Lock()
        self._slots: list[VramSlot] = []
        self._seq = 0

    def set_budget_callbacks(
        self,
        *,
        budget_fn: Callable[[], int | None] | None,
        claim_bytes_fn: Callable[[int], _CapacityClaim | None] | None,
    ) -> None:
        """Bind dynamic budget ceiling/claim callbacks prior to pool use."""
        with self._lock:
            self._budget_fn = budget_fn
            self._claim_bytes_fn = claim_bytes_fn

    def _dynamic_budget_bytes(self) -> int | None:
        dynamic = None if self._budget_fn is None else self._budget_fn()
        if dynamic is not None:
            dynamic = validate_vram_budget(dynamic)
        return dynamic

    def _combine_budget(self, dynamic: int | None) -> int | None:
        if self._budget_bytes is None:
            return dynamic
        if dynamic is None:
            return self._budget_bytes
        return min(self._budget_bytes, dynamic)

    @property
    def budget_bytes(self) -> int | None:
        return self._combine_budget(self._dynamic_budget_bytes())

    @property
    def used_bytes(self) -> int:
        # Read without ``self._lock`` so VramBudget usage queries never invert lock order.
        return sum(slot.nbytes for slot in tuple(self._slots))

    def _used_bytes_locked(self) -> int:
        return sum(slot.nbytes for slot in self._slots)

    def _checked_out_bytes_locked(self) -> int:
        return sum(slot.nbytes for slot in self._slots if slot.checked_out)

    def _checkout_slot_locked(self, slot: VramSlot) -> VramSlot:
        slot.staged_unit_idx = None
        slot.checked_out = True
        slot.last_used_seq = self._next_seq()
        return slot

    @property
    def live_slot_count(self) -> int:
        with self._lock:
            return len(self._slots)

    def cached_unit_indices(self) -> set[int]:
        """Unit indices currently staged and parked idle in the pool."""
        with self._lock:
            return {
                slot.staged_unit_idx
                for slot in self._slots
                if not slot.checked_out
                and slot.staged_unit_idx is not None
                and slot.lease.is_staged
            }

    def has_cached_unit(self, unit_idx: int) -> bool:
        """Whether ``unit_idx`` is currently staged and available for cache hit."""
        with self._lock:
            return any(
                not slot.checked_out
                and slot.staged_unit_idx == unit_idx
                and slot.lease.is_staged
                for slot in self._slots
            )

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _idle_slots(self) -> list[VramSlot]:
        return [slot for slot in self._slots if not slot.checked_out]

    @staticmethod
    def _pick_victim(
        candidates: list[VramSlot],
        distance_fn: Callable[[int], float],
    ) -> VramSlot:
        if len(candidates) == 1:
            return candidates[0]
        unstaged = [
            slot
            for slot in candidates
            if slot.staged_unit_idx is None or not slot.lease.is_staged
        ]
        if unstaged:
            return min(unstaged, key=lambda slot: slot.last_used_seq)

        def _staged_score(slot: VramSlot) -> tuple[float, int]:
            assert slot.staged_unit_idx is not None
            return (distance_fn(slot.staged_unit_idx), -slot.last_used_seq)

        return max(candidates, key=_staged_score)

    def _close_slot(self, slot: VramSlot) -> None:
        # Leave slot tracked until lease.close() succeeds so synchronization
        # failures can be retried cleanly without leaking CUDA targets.
        slot.lease.close()
        self._slots.remove(slot)

    def _trim_to_budget_locked(
        self,
        distance_fn: Callable[[int], float],
        *,
        budget: int | None,
        exclude: VramSlot | None = None,
    ) -> None:
        if budget is None:
            while len(self._slots) > 2:
                candidates = [s for s in self._idle_slots() if s is not exclude]
                if not candidates:
                    break
                self._close_slot(self._pick_victim(candidates, distance_fn))
            return

        while self._used_bytes_locked() > budget:
            candidates = [s for s in self._idle_slots() if s is not exclude]
            if not candidates:
                break
            self._close_slot(self._pick_victim(candidates, distance_fn))

    def _close_slots_locked(self, slots: list[VramSlot]) -> None:
        first_error: BaseException | None = None
        for slot in tuple(slots):
            try:
                self._close_slot(slot)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def evict_idle_bytes(
        self,
        needed_bytes: int,
        *,
        distance_fn: Callable[[int], float],
    ) -> int:
        """Close idle cached slots via ``_pick_victim`` until ``needed_bytes`` is freed."""
        if (
            isinstance(needed_bytes, bool)
            or not isinstance(needed_bytes, int)
            or needed_bytes < 0
        ):
            raise ValueError(
                f"needed_bytes must be a non-negative integer; got {needed_bytes!r}."
            )
        if needed_bytes == 0:
            return 0
        freed = 0
        with self._lock:
            while freed < needed_bytes:
                idle = self._idle_slots()
                if not idle:
                    break
                victim = self._pick_victim(idle, distance_fn)
                victim_bytes = victim.nbytes
                self._close_slot(victim)
                freed += victim_bytes
        return freed

    def close_idle(self) -> None:
        """Close all idle cached slots immediately; retry-safe on close errors."""
        with self._lock:
            self._close_slots_locked(self._idle_slots())

    def _allocate_slot(
        self,
        plan: HostModuleLoadPlan,
        sig: UnitSignature,
        stream: torch.cuda.Stream,
    ) -> VramSlot:
        lease = CudaTargetLease.allocate(
            plan,
            self._device,
            allocation_stream=stream,
        )
        slot = VramSlot(
            lease=lease,
            signature=sig,
            nbytes=target_gpu_bytes(lease.target),
            staged_unit_idx=None,
            last_used_seq=self._next_seq(),
            checked_out=True,
        )
        self._slots.append(slot)
        return slot

    def _checkout_streaming_locked(
        self,
        sig: UnitSignature,
        plan: HostModuleLoadPlan,
        stream: torch.cuda.Stream,
        *,
        distance_fn: Callable[[int], float],
        is_lookahead: bool,
        budget: int | None,
    ) -> VramSlot | None:
        matching_idle = [s for s in self._idle_slots() if s.signature == sig]
        if matching_idle:
            slot = self._pick_victim(matching_idle, distance_fn)
            self._trim_to_budget_locked(distance_fn, budget=budget, exclude=slot)
            slot.staged_unit_idx = None
            slot.checked_out = True
            slot.last_used_seq = self._next_seq()
            return slot

        while len(self._slots) >= 2:
            idle = self._idle_slots()
            if not idle:
                if is_lookahead:
                    return None
                raise RuntimeError(
                    "Streaming VRAM pool ceiling (2 slots) reached with no idle slot."
                )
            self._close_slot(self._pick_victim(idle, distance_fn))

        return self._allocate_slot(plan, sig, stream)

    def _checkout_budgeted_locked(
        self,
        sig: UnitSignature,
        plan: HostModuleLoadPlan,
        stream: torch.cuda.Stream,
        *,
        distance_fn: Callable[[int], float],
        is_lookahead: bool,
        budget: int,
        needed_bytes: int,
    ) -> VramSlot | None:
        self._trim_to_budget_locked(distance_fn, budget=budget)

        if is_lookahead:
            checked_out_bytes = sum(s.nbytes for s in self._slots if s.checked_out)
            if checked_out_bytes + needed_bytes > budget:
                return None

        unstaged_matching = [
            s
            for s in self._idle_slots()
            if s.signature == sig and (s.staged_unit_idx is None or not s.lease.is_staged)
        ]
        if unstaged_matching and self._used_bytes_locked() <= budget:
            slot = unstaged_matching[0]
            slot.staged_unit_idx = None
            slot.checked_out = True
            slot.last_used_seq = self._next_seq()
            return slot

        while self._used_bytes_locked() + needed_bytes > budget:
            idle = self._idle_slots()
            if not idle:
                break
            victim = self._pick_victim(idle, distance_fn)
            if victim.signature == sig and self._used_bytes_locked() <= budget:
                victim.staged_unit_idx = None
                victim.checked_out = True
                victim.last_used_seq = self._next_seq()
                return victim
            self._close_slot(victim)

        if is_lookahead and self._used_bytes_locked() + needed_bytes > budget:
            return None

        return self._allocate_slot(plan, sig, stream)

    def _find_cached_hit_locked(
        self,
        unit_idx: int,
        distance_fn: Callable[[int], float],
        *,
        budget: int | None,
    ) -> VramSlot | None:
        for slot in self._slots:
            if (
                not slot.checked_out
                and slot.staged_unit_idx == unit_idx
                and slot.lease.is_staged
            ):
                self._trim_to_budget_locked(distance_fn, budget=budget, exclude=slot)
                slot.checked_out = True
                slot.last_used_seq = self._next_seq()
                return slot
        return None

    def _reclaim_for_budget_locked(
        self,
        sig: UnitSignature,
        *,
        needed_bytes: int,
        target_budget: int,
        ceiling: int | None,
        distance_fn: Callable[[int], float],
    ) -> VramSlot | None:
        while self._used_bytes_locked() + needed_bytes > target_budget:
            idle = self._idle_slots()
            if not idle:
                break
            victim = self._pick_victim(idle, distance_fn)
            if (
                victim.signature == sig
                and self._used_bytes_locked() <= target_budget
                and (ceiling is None or self._used_bytes_locked() <= ceiling)
            ):
                return self._checkout_slot_locked(victim)
            self._close_slot(victim)
        return None

    def _try_streaming_reuse_locked(
        self,
        sig: UnitSignature,
        *,
        distance_fn: Callable[[int], float],
        is_lookahead: bool,
    ) -> tuple[VramSlot | None, bool, bool]:
        matching_idle = [s for s in self._idle_slots() if s.signature == sig]
        if matching_idle:
            slot = self._pick_victim(matching_idle, distance_fn)
            self._trim_to_budget_locked(distance_fn, budget=None, exclude=slot)
            return self._checkout_slot_locked(slot), False, True
        while len(self._slots) >= 2:
            idle = self._idle_slots()
            if not idle:
                if is_lookahead:
                    return None, False, True
                raise RuntimeError(
                    "Streaming VRAM pool ceiling (2 slots) reached with no idle slot."
                )
            self._close_slot(self._pick_victim(idle, distance_fn))
        return None, False, False

    def _try_budgeted_reuse_locked(
        self,
        sig: UnitSignature,
        *,
        distance_fn: Callable[[int], float],
        is_lookahead: bool,
        needed_bytes: int,
        budget: int,
    ) -> tuple[VramSlot | None, bool, bool]:
        self._trim_to_budget_locked(distance_fn, budget=budget)
        unstaged = [
            s
            for s in self._idle_slots()
            if s.signature == sig and (s.staged_unit_idx is None or not s.lease.is_staged)
        ]
        if unstaged and self._used_bytes_locked() <= budget:
            return self._checkout_slot_locked(unstaged[0]), False, True
        if self._budget_bytes is not None:
            if (
                is_lookahead
                and self._checked_out_bytes_locked() + needed_bytes > self._budget_bytes
            ):
                return None, False, True
            reused = self._reclaim_for_budget_locked(
                sig,
                needed_bytes=needed_bytes,
                target_budget=self._budget_bytes,
                ceiling=budget,
                distance_fn=distance_fn,
            )
            if reused is not None:
                return reused, False, True
            if is_lookahead and self._used_bytes_locked() + needed_bytes > self._budget_bytes:
                return None, False, True
        return None, False, False

    def _try_local_reuse_locked(
        self,
        unit_idx: int,
        sig: UnitSignature,
        *,
        distance_fn: Callable[[int], float],
        is_lookahead: bool,
        needed_bytes: int,
        budget: int | None,
    ) -> tuple[VramSlot | None, bool, bool]:
        hit = self._find_cached_hit_locked(unit_idx, distance_fn, budget=budget)
        if hit is not None:
            return hit, True, True
        if budget is None:
            return self._try_streaming_reuse_locked(
                sig,
                distance_fn=distance_fn,
                is_lookahead=is_lookahead,
            )
        return self._try_budgeted_reuse_locked(
            sig,
            distance_fn=distance_fn,
            is_lookahead=is_lookahead,
            needed_bytes=needed_bytes,
            budget=budget,
        )

    def _allocate_under_claim_locked(
        self,
        unit_idx: int,
        sig: UnitSignature,
        plan: HostModuleLoadPlan,
        stream: torch.cuda.Stream,
        *,
        distance_fn: Callable[[int], float],
        is_lookahead: bool,
        needed_bytes: int,
        was_streaming: bool,
    ) -> tuple[VramSlot | None, bool]:
        hit = self._find_cached_hit_locked(unit_idx, distance_fn, budget=self._budget_bytes)
        if hit is not None:
            return hit, True
        if self._budget_bytes is not None:
            self._trim_to_budget_locked(distance_fn, budget=self._budget_bytes)
            if is_lookahead and self._checked_out_bytes_locked() + needed_bytes > self._budget_bytes:
                return None, False
            reused = self._reclaim_for_budget_locked(
                sig,
                needed_bytes=needed_bytes,
                target_budget=self._budget_bytes,
                ceiling=self._budget_bytes,
                distance_fn=distance_fn,
            )
            if reused is not None:
                return reused, False
            if is_lookahead and self._used_bytes_locked() + needed_bytes > self._budget_bytes:
                return None, False
        elif was_streaming:
            while len(self._slots) >= 2:
                idle = self._idle_slots()
                if not idle:
                    if is_lookahead:
                        return None, False
                    break
                self._close_slot(self._pick_victim(idle, distance_fn))
        return self._allocate_slot(plan, sig, stream), False

    def _reclaim_own_slots_on_denial_locked(
        self,
        unit_idx: int,
        sig: UnitSignature,
        *,
        distance_fn: Callable[[int], float],
        is_lookahead: bool,
        needed_bytes: int,
        eff_budget: int | None,
    ) -> tuple[VramSlot | None, bool, int, bool]:
        before_used = self._used_bytes_locked()
        hit = self._find_cached_hit_locked(unit_idx, distance_fn, budget=eff_budget)
        if hit is not None:
            return hit, True, 0, False
        if eff_budget is not None:
            self._trim_to_budget_locked(distance_fn, budget=eff_budget)
            if is_lookahead and self._checked_out_bytes_locked() + needed_bytes > eff_budget:
                return None, False, 0, True
        while before_used - self._used_bytes_locked() < needed_bytes:
            idle = self._idle_slots()
            if not idle:
                break
            victim = self._pick_victim(idle, distance_fn)
            if victim.signature == sig and (
                eff_budget is None or self._used_bytes_locked() <= eff_budget
            ):
                return self._checkout_slot_locked(victim), False, 0, False
            self._close_slot(victim)
        return None, False, max(0, before_used - self._used_bytes_locked()), False

    def _checkout_with_claim(
        self,
        unit_idx: int,
        sig: UnitSignature,
        plan: HostModuleLoadPlan,
        stream: torch.cuda.Stream,
        *,
        distance_fn: Callable[[int], float],
        is_lookahead: bool,
        needed_bytes: int,
        claim_fn: Callable[[int], _CapacityClaim | None],
    ) -> tuple[VramSlot | None, bool]:
        budget = self.budget_bytes
        with self._lock:
            slot, hit, handled = self._try_local_reuse_locked(
                unit_idx,
                sig,
                distance_fn=distance_fn,
                is_lookahead=is_lookahead,
                needed_bytes=needed_bytes,
                budget=budget,
            )
        if handled:
            return slot, hit

        was_streaming = budget is None
        while True:
            claim = claim_fn(needed_bytes)
            if claim is not None:
                try:
                    with self._lock:
                        return self._allocate_under_claim_locked(
                            unit_idx,
                            sig,
                            plan,
                            stream,
                            distance_fn=distance_fn,
                            is_lookahead=is_lookahead,
                            needed_bytes=needed_bytes,
                            was_streaming=was_streaming,
                        )
                finally:
                    claim.release()

            eff_budget = self.budget_bytes
            with self._lock:
                slot, hit, freed_own, exhausted = self._reclaim_own_slots_on_denial_locked(
                    unit_idx,
                    sig,
                    distance_fn=distance_fn,
                    is_lookahead=is_lookahead,
                    needed_bytes=needed_bytes,
                    eff_budget=eff_budget,
                )
            if slot is not None:
                return slot, hit
            if exhausted:
                return None, False
            if freed_own <= 0:
                break

        if is_lookahead:
            return None, False
        with self._lock:
            return self._allocate_slot(plan, sig, stream), False

    def checkout(
        self,
        unit_idx: int,
        plan: HostModuleLoadPlan,
        stream: torch.cuda.Stream,
        *,
        distance_fn: Callable[[int], float],
        is_lookahead: bool,
    ) -> tuple[VramSlot | None, bool]:
        """Acquire a pool slot for ``unit_idx``; returns ``(slot, cache_hit)``.

        When ``is_lookahead=True`` and neither budget nor streaming ceiling can
        accommodate another concurrent slot, returns ``(None, False)`` without
        evicting active slots.
        """
        sig = plan_target_signature(plan)
        needed_bytes = estimate_plan_gpu_bytes(plan)
        claim_fn = self._claim_bytes_fn
        if claim_fn is not None:
            return self._checkout_with_claim(
                unit_idx,
                sig,
                plan,
                stream,
                distance_fn=distance_fn,
                is_lookahead=is_lookahead,
                needed_bytes=needed_bytes,
                claim_fn=claim_fn,
            )

        budget = self.budget_bytes
        with self._lock:
            hit = self._find_cached_hit_locked(unit_idx, distance_fn, budget=budget)
            if hit is not None:
                return hit, True
            if budget is None:
                return (
                    self._checkout_streaming_locked(
                        sig,
                        plan,
                        stream,
                        distance_fn=distance_fn,
                        is_lookahead=is_lookahead,
                        budget=budget,
                    ),
                    False,
                )
            return (
                self._checkout_budgeted_locked(
                    sig,
                    plan,
                    stream,
                    distance_fn=distance_fn,
                    is_lookahead=is_lookahead,
                    budget=budget,
                    needed_bytes=needed_bytes,
                ),
                False,
            )

    def return_slot(
        self,
        slot: VramSlot,
        *,
        staged_unit_idx: int | None,
        distance_fn: Callable[[int], float],
    ) -> None:
        """Return a checked-out slot to idle and trim to budget/ceiling."""
        with self._lock:
            if slot not in self._slots:
                raise RuntimeError("Returned VramSlot does not belong to this ElasticVramPool.")
            slot.checked_out = False
            slot.staged_unit_idx = staged_unit_idx if slot.lease.is_staged else None
            slot.last_used_seq = self._next_seq()
        budget = self.budget_bytes
        with self._lock:
            self._trim_to_budget_locked(distance_fn, budget=budget)

    def admit_preloaded_slot(
        self,
        unit_idx: int,
        lease: CudaTargetLease,
        sig: UnitSignature,
    ) -> None:
        """Register an already-staged resident unit slot into the idle pool."""
        nbytes = target_gpu_bytes(lease.target)
        with self._lock:
            slot = VramSlot(
                lease=lease,
                signature=sig,
                nbytes=nbytes,
                staged_unit_idx=unit_idx,
                last_used_seq=self._next_seq(),
                checked_out=False,
            )
            self._slots.append(slot)

    def close_all(self) -> None:
        """Close every live slot; keeps slots whose ``close()`` raises for retry."""
        with self._lock:
            self._close_slots_locked(self._slots)


__all__ = [
    "ElasticVramPool",
    "UnitSignature",
    "VramSlot",
    "estimate_plan_gpu_bytes",
    "plan_target_signature",
    "target_gpu_bytes",
]
