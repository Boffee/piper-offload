"""Shared internal CUDA lifetime, validation, and wave-lease helpers for Layer 2 slab pools."""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import torch

from . import pin_manager as _pin_manager
from .device_transfer import DeviceTransferContext
from .dual_arena import SlotSchema, TypedSidePool, Uint8DualArena
from .pin_manager import PinLease, PinManager
from .stream_lease import StreamLease, _record_event
from .vram_budget import VramBudget


class _SlotLike(Protocol):
    """Minimal slot contract shared by :class:`_WaveLease` and teardown helpers."""

    slot_id: int
    active_readers: int
    lease: StreamLease | None
    lease_lock: threading.Lock


@dataclass(frozen=True, slots=True)
class _PoolViews:
    arena_bytes: torch.Tensor
    arena_views: MappingProxyType[str, torch.Tensor]
    arena_flat_views: dict[str, torch.Tensor]
    side_views: MappingProxyType[str, torch.Tensor]
    slot_offsets: MappingProxyType[str, torch.Tensor]


@dataclass(eq=False, slots=True)
class _PendingWaveRelease:
    slot: _SlotLike
    release_stream: torch.cuda.Stream
    end_streams: tuple[torch.cuda.Stream, ...] = ()


@dataclass(slots=True)
class _WaveSlotClaim:
    slot: _SlotLike
    acquired: bool = False
    end_streams: tuple[torch.cuda.Stream, ...] = ()
    released: bool = False


def _merge_error(current: BaseException | None, incoming: BaseException | None) -> BaseException | None:
    if incoming is None:
        return current
    if current is None or (isinstance(current, Exception) and not isinstance(incoming, Exception)):
        return incoming
    return current


def _run_until_complete(fn: Callable[[], None]) -> tuple[bool, BaseException | None]:
    """Run ``fn()``, retrying non-``Exception`` interrupts; return ``(completed, err)``."""
    interrupt: BaseException | None = None
    while True:
        try:
            fn()
            return True, interrupt
        except Exception as exc:
            return False, (interrupt if interrupt is not None else exc)
        except BaseException as intr:
            if interrupt is None:
                interrupt = intr


def _sync_interrupt_safe(fn: Callable[[], None]) -> None:
    """Run ``fn()``, retrying if interrupted by a non-``Exception`` ``BaseException``."""
    completed, err = _run_until_complete(fn)
    if not completed or err is not None:
        assert err is not None
        raise err


def _synchronize_pending_leases(
    pending: deque[tuple[torch.cuda.Event | None, PinLease]],
    copy_stream: torch.cuda.Stream | None,
) -> None:
    """Wait for every in-flight DMA event before closing its :class:`PinLease`."""
    interrupt: BaseException | None = None
    while pending:
        event, lease = pending[0]
        target_fn = (
            event.synchronize
            if event is not None
            else (copy_stream.synchronize if copy_stream is not None else None)
        )
        if target_fn is not None:
            completed, err = _run_until_complete(target_fn)
            if not completed:
                assert err is not None
                if interrupt is not None and isinstance(err, Exception):
                    raise interrupt from err
                raise err
            interrupt = _merge_error(interrupt, err)
        lease.close()
        pending.popleft()
    if interrupt is not None:
        raise interrupt


def _validate_optional_positive_int(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None; got {value!r}.")
    return value


def _validate_slab_pool_base_args(
    pool_name: str,
    arena: Uint8DualArena,
    schema: SlotSchema,
    *,
    max_slots: int | None,
    budget: VramBudget | None,
    pin_manager: PinManager | None,
) -> tuple[int | None, PinManager, VramBudget | None]:
    if not isinstance(arena, Uint8DualArena):
        raise TypeError(f"arena must be a Uint8DualArena; got {type(arena).__name__}.")
    if not arena.is_open:
        raise RuntimeError(f"Cannot attach {pool_name} to a closed Uint8DualArena.")
    if not isinstance(schema, SlotSchema):
        raise TypeError(f"schema must be a SlotSchema; got {type(schema).__name__}.")
    if arena.alignment < schema.field_alignment:
        raise ValueError(
            f"Arena alignment ({arena.alignment}) must be >= "
            f"schema.field_alignment ({schema.field_alignment})."
        )
    resolved_max = _validate_optional_positive_int(max_slots, name="max_slots")
    if budget is not None and not isinstance(budget, VramBudget):
        raise TypeError(f"budget must be a VramBudget or None; got {type(budget).__name__}.")
    effective_budget = arena.budget if budget is None else budget
    resolved_pin = _pin_manager.host_pin_manager if pin_manager is None else pin_manager
    if not isinstance(resolved_pin, PinManager):
        raise TypeError(
            f"pin_manager must be a PinManager or None; got {type(resolved_pin).__name__}."
        )
    return resolved_max, resolved_pin, effective_budget


def _setup_transfer_context(
    pool_name: str,
    device: torch.device,
    transfer_context: DeviceTransferContext | None,
) -> tuple[DeviceTransferContext | None, bool, bool]:
    if device.type != "cuda":
        if transfer_context is not None:
            raise ValueError(f"CPU {pool_name} does not accept a CUDA DeviceTransferContext.")
        return None, False, False
    if transfer_context is None:
        return DeviceTransferContext(device), True, False
    if not isinstance(transfer_context, DeviceTransferContext):
        raise TypeError(
            f"transfer_context must be a DeviceTransferContext or None; "
            f"got {type(transfer_context).__name__}."
        )
    if transfer_context.device != device:
        raise ValueError(
            f"transfer_context device ({transfer_context.device}) must match arena device ({device})."
        )
    transfer_context._acquire_borrow()
    return transfer_context, False, True


def _build_side_pools(
    schema: SlotSchema,
    max_slots: int,
    device: torch.device,
    budget: VramBudget | None,
    *,
    reservation_prefix: str,
) -> dict[str, TypedSidePool]:
    pools: dict[str, TypedSidePool] = {}
    try:
        with torch.inference_mode(False):
            for spec in schema.side_fields:
                pools[spec.name] = TypedSidePool(
                    max_slots,
                    spec.shape,
                    dtype=spec.dtype,
                    device=device,
                    alignment_bytes=spec.alignment_bytes,
                    budget=budget,
                    reservation_name=f"{reservation_prefix}{spec.name}",
                )
    except BaseException:
        for pool in pools.values():
            with contextlib.suppress(BaseException):
                pool.close()
        raise
    return pools


def _build_pool_views_common(
    arena: Uint8DualArena,
    schema: SlotSchema,
    *,
    max_slots: int,
    slot_stride: int,
    base_offset: int,
    pool_view_fn: Callable[[tuple[int, ...], int], torch.Tensor],
    side_pools: Mapping[str, TypedSidePool],
    device: torch.device,
) -> _PoolViews:
    with torch.inference_mode(False):
        arena_bytes = arena.raw_uint8_view()
        arena_views = {
            af.name: pool_view_fn(af.shape, af.offset)
            for af in schema.arena_fields
        }
        arena_flat_views = {
            af.name: pool_view_fn((af.nbytes,), af.offset)
            for af in schema.arena_fields
        }
        side_views = {name: sp.tensor for name, sp in side_pools.items()}
        slot_idx_vec = torch.arange(max_slots, dtype=torch.int64, device=device)
        slot_offsets = {
            af.name: base_offset + slot_idx_vec * slot_stride + af.offset
            for af in schema.arena_fields
        }
        if device.type == "cuda":
            alloc_ev = getattr(arena, "_alloc_event", None)
            if alloc_ev is not None:
                _sync_interrupt_safe(alloc_ev.synchronize)
            event = _record_event(torch.cuda.current_stream(device))
            _sync_interrupt_safe(event.synchronize)
    return _PoolViews(
        arena_bytes=arena_bytes,
        arena_views=MappingProxyType(arena_views),
        arena_flat_views=arena_flat_views,
        side_views=MappingProxyType(side_views),
        slot_offsets=MappingProxyType(slot_offsets),
    )


def _make_device_int_table(mapping: Sequence[int], *, device: torch.device) -> torch.Tensor:
    with torch.inference_mode(False):
        if device.type == "cuda":
            try:
                return torch.tensor(mapping, dtype=torch.int32, pin_memory=True).to(
                    device, non_blocking=True
                )
            except RuntimeError:
                return torch.tensor(mapping, dtype=torch.int32).to(device, non_blocking=False)
        return torch.tensor(mapping, dtype=torch.int32, device=device)


def _release_slot_stream_lease(
    slot: _SlotLike,
    release_stream: torch.cuda.Stream,
    end_streams: Sequence[torch.cuda.Stream],
) -> tuple[bool, BaseException | None]:
    with slot.lease_lock:
        slot_lease = slot.lease
        if slot_lease is None:
            return True, None
        first_err: BaseException | None = None
        for end_stream in end_streams:
            if end_stream == release_stream:
                continue
            rec_ok, rec_err = _run_until_complete(
                lambda sl=slot_lease, es=end_stream: sl.record_stream(es)
            )
            first_err = _merge_error(first_err, rec_err)
            if not rec_ok:
                return False, first_err
        rel_ok, rel_err = _run_until_complete(
            lambda sl=slot_lease, rs=release_stream: sl.release(rs)
        )
        return rel_ok, _merge_error(first_err, rel_err)


def _merge_end_stream(
    existing: tuple[torch.cuda.Stream, ...],
    end_stream: torch.cuda.Stream | None,
    release_stream: torch.cuda.Stream,
) -> tuple[torch.cuda.Stream, ...]:
    if end_stream is None or end_stream == release_stream or end_stream in existing:
        return existing
    return (*existing, end_stream)


class _WaveLease:
    """Single RAII owner pairing ``slot.active_readers`` and ``StreamLease`` readers."""

    def __init__(
        self,
        device: torch.device,
        lock: threading.Lock,
        tracker: _WaveReleaseTracker,
    ) -> None:
        self._device = device
        self._lock = lock
        self._tracker = tracker
        self._claims: list[_WaveSlotClaim] = []
        self._stream: torch.cuda.Stream | None = None

    def add_unlocked(self, slot: _SlotLike) -> None:
        """Increment ``slot.active_readers`` under ``lock`` and track ownership."""
        slot.active_readers += 1
        self._claims.append(_WaveSlotClaim(slot=slot))

    def adopt_unlocked(self, slot: _SlotLike) -> None:
        """Adopt an already-incremented reader slot under ``lock``."""
        self._claims.append(_WaveSlotClaim(slot=slot))

    def acquire(self) -> None:
        """Acquire shared ``StreamLease`` readers on the active CUDA compute stream."""
        if self._device.type != "cuda":
            for claim in self._claims:
                claim.acquired = True
            return
        stream = torch.cuda.current_stream(self._device)
        self._stream = stream
        for claim in self._claims:
            if claim.acquired:
                continue
            slot = claim.slot
            assert slot.lease is not None
            with slot.lease_lock:
                slot.lease.acquire(stream, exclusive=False)
                claim.acquired = True

    def release(self) -> None:
        """Idempotently release ``StreamLease``s and retain failed slots for retry."""
        if not self._claims:
            return
        end_stream = torch.cuda.current_stream(self._device) if self._device.type == "cuda" else None
        release_stream = self._stream
        first_err: BaseException | None = None
        try:
            for claim in self._claims:
                if claim.released:
                    continue
                slot = claim.slot
                if (
                    self._device.type != "cuda"
                    or slot.lease is None
                    or release_stream is None
                    or not claim.acquired
                ):
                    claim.released = True
                    continue
                extras = _merge_end_stream(claim.end_streams, end_stream, release_stream)
                claim.end_streams = extras
                released, err = _release_slot_stream_lease(slot, release_stream, extras)
                first_err = _merge_error(first_err, err)
                if released:
                    claim.released = True
        finally:
            released_claims = [c for c in self._claims if c.released]
            remaining = [c for c in self._claims if not c.released]
            if released_claims:
                with self._lock:
                    for claim in released_claims:
                        claim.slot.active_readers = max(0, claim.slot.active_readers - 1)
            self._claims = remaining
            if not self._claims:
                self._stream = None
        if first_err is not None:
            raise first_err

    def handoff_to_pool(self) -> None:
        """Transfer any still-unreleased slots to the pool before exiting a wave session."""
        if not self._claims:
            return
        release_stream = self._stream
        with self._lock:
            for claim in self._claims:
                if claim.released:
                    continue
                slot = claim.slot
                if (
                    self._device.type == "cuda"
                    and slot.lease is not None
                    and release_stream is not None
                    and claim.acquired
                ):
                    self._tracker.pending.append(
                        _PendingWaveRelease(
                            slot=slot,
                            release_stream=release_stream,
                            end_streams=claim.end_streams,
                        )
                    )
                else:
                    slot.active_readers = max(0, slot.active_readers - 1)
            self._claims.clear()
            self._stream = None


def _synchronize_slots_for_close(
    slots: Sequence[_SlotLike],
    pending_by_slot: dict[int, list[_PendingWaveRelease]],
    lock: threading.Lock,
) -> None:
    """Synchronize all pending wave consumer streams and reset every slot's ``StreamLease``."""
    for slot in slots:
        slot_items = pending_by_slot.get(slot.slot_id)
        if slot_items:
            for item in slot_items:
                for end_stream in item.end_streams:
                    _sync_interrupt_safe(end_stream.synchronize)
        if slot.lease is not None:
            lease = slot.lease
            with slot.lease_lock:
                _sync_interrupt_safe(lease.synchronize_and_reset)
        if slot_items:
            del pending_by_slot[slot.slot_id]
            with lock:
                slot.active_readers = max(0, slot.active_readers - len(slot_items))
    for slot_id, slot_items in list(pending_by_slot.items()):
        for item in slot_items:
            for end_stream in item.end_streams:
                _sync_interrupt_safe(end_stream.synchronize)
            _sync_interrupt_safe(item.release_stream.synchronize)
        del pending_by_slot[slot_id]


class _WaveReleaseTracker:
    """Own deferred ``StreamLease`` wave releases, non-blocking reap, and close sync."""

    def __init__(
        self,
        device: torch.device,
        lock: threading.Lock,
        cond: threading.Condition,
    ) -> None:
        self._device = device
        self._lock = lock
        self._cond = cond
        self.pending: deque[_PendingWaveRelease] = deque()
        self.inflight_reaps = 0

    def new_lease(self) -> _WaveLease:
        return _WaveLease(self._device, self._lock, self)

    def enqueue_unlocked(self, slot: _SlotLike, release_stream: torch.cuda.Stream) -> None:
        slot.active_readers += 1
        self.pending.append(_PendingWaveRelease(slot=slot, release_stream=release_stream))

    def pending_counts_unlocked(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for item in self.pending:
            counts[item.slot.slot_id] = counts.get(item.slot.slot_id, 0) + 1
        return counts

    def drain_by_slot_unlocked(
        self,
        out: dict[int, list[_PendingWaveRelease]] | None = None,
    ) -> dict[int, list[_PendingWaveRelease]]:
        by_slot = {} if out is None else out
        while self.pending:
            item = self.pending.popleft()
            by_slot.setdefault(item.slot.slot_id, []).append(item)
        return by_slot

    def restore_unlocked(self, by_slot: Mapping[int, list[_PendingWaveRelease]]) -> None:
        for slot_items in by_slot.values():
            for item in slot_items:
                if not any(existing is item for existing in self.pending):
                    self.pending.append(item)

    def reap(self, is_stopped_unlocked: Callable[[], bool]) -> None:
        if self._device.type != "cuda":
            return
        with self._lock:
            if not self.pending or is_stopped_unlocked():
                return
            batch = list(self.pending)
            self.pending.clear()
            self.inflight_reaps += 1
        released_slots: list[_SlotLike] = []
        still_pending: list[_PendingWaveRelease] = []
        interrupt: BaseException | None = None
        try:
            for idx, item in enumerate(batch):
                try:
                    released, err = _release_slot_stream_lease(
                        item.slot, item.release_stream, item.end_streams
                    )
                except BaseException as exc:
                    interrupt = _merge_error(interrupt, exc)
                    still_pending.extend(batch[idx:])
                    break
                if released:
                    released_slots.append(item.slot)
                    if err is not None and not isinstance(err, Exception):
                        interrupt = _merge_error(interrupt, err)
                        still_pending.extend(batch[idx + 1 :])
                        break
                elif err is not None and not isinstance(err, Exception):
                    interrupt = _merge_error(interrupt, err)
                    still_pending.extend(batch[idx:])
                    break
                else:
                    still_pending.append(item)
        finally:
            with self._lock:
                for slot in released_slots:
                    slot.active_readers = max(0, slot.active_readers - 1)
                if still_pending:
                    self.pending.extendleft(reversed(still_pending))
                self.inflight_reaps = max(0, self.inflight_reaps - 1)
                self._cond.notify_all()
        if interrupt is not None:
            raise interrupt

    def synchronize_slots_for_close(
        self,
        slots: Sequence[_SlotLike],
        by_slot: dict[int, list[_PendingWaveRelease]],
    ) -> None:
        _synchronize_slots_for_close(slots, by_slot, self._lock)

