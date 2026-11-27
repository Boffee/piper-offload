"""Layer 2 global MoE/slab residency manager over shared Layer 1 hardware substrate.

``ExpertSlabPool`` caches read-only keyed items (such as MoE experts across
layers) inside ascending fixed-size drawers of a :class:`Uint8DualArena` plus
dedicated :class:`TypedSidePool` allocations described by one format-agnostic
:class:`SlotSchema`. Variable-size items fit via fixed-drawer padding: each
field transfers only ``src.nbytes <= field.nbytes`` into an aligned drawer.

It never imports ``piper_kernels`` or embeds tensor-format math: callers consume
yielded :class:`ExpertWave` descriptors (``expert_to_slot``, strided
``arena_views``, 1-D ``arena_bytes`` + ``slot_offsets``, and strided
``side_views`` with aligned ``stride(0)`` and contiguous per-slot inner
dimensions) and pass ``accumulate=wave.index > 0`` to their kernel.
"""

from __future__ import annotations

import contextlib
import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Generator, Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Self

import torch

from . import pin_manager as _pin_manager
from .device_transfer import DeviceTransferContext
from .dual_arena import (
    ArenaCapacityError,
    LowSlotLayout,
    SlotSchema,
    TypedSidePool,
    Uint8DualArena,
)
from .pin_manager import PinLease, PinManager
from .stream_lease import StreamLease, _record_event
from .tensor_adapters import capture_host_tensor
from .vram_budget import VramBudget

_INTEGER_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_HALVE_TABLE = bytes(i >> 1 for i in range(256))


class _SlotPhase(Enum):
    """Single source of truth for one physical slot's lifecycle state."""

    EMPTY = "empty"
    STAGING = "staging"
    READY = "ready"
    RELOCATING = "relocating"


@dataclass(frozen=True, slots=True)
class ExpertWave:
    """One ready wave of active expert weights for a forward pass.

    Attributes:
        index: Zero-based wave index within the current :meth:`ExpertSlabPool.waves`
            session (pass ``accumulate=wave.index > 0`` to downstream kernels).
        group_id: Group key supplied to :meth:`ExpertSlabPool.waves`.
        expert_ids: Active expert indices covered by this wave (or all group indices
            ``0..num_experts-1`` on the zero-sync all-resident fast path).
        expert_to_slot: ``int32`` tensor of shape ``[num_experts]`` mapping each
            expert to its slot index, or ``-1`` when handled by another wave / unused.
        active_slots_hint: Upper bound on active slots in this wave.
        is_hit_wave: ``True`` when every weight in this wave was already cached on GPU.
        arena_views: Read-only mapping of arena field name -> strided ``uint8`` view
            ``[max_slots, *shape]``. Tail bytes beyond ``src.nbytes`` are uninitialized.
        side_views: Read-only mapping of side field name -> strided ``[max_slots, *shape]``
            view with aligned ``stride(0)`` and contiguous per-slot inner dimensions.
        arena_bytes: Shared 1-D ``uint8`` view over the backing :class:`Uint8DualArena`.
        slot_offsets: Read-only mapping of arena field name -> ``int64[max_slots]``
            byte offsets inside :attr:`arena_bytes`.
    """

    index: int
    group_id: Hashable
    expert_ids: tuple[int, ...]
    expert_to_slot: torch.Tensor
    active_slots_hint: int
    is_hit_wave: bool
    arena_views: Mapping[str, torch.Tensor]
    side_views: Mapping[str, torch.Tensor]
    arena_bytes: torch.Tensor
    slot_offsets: Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class ExpertPoolStats:
    """Point-in-time capacity and residency counters for :class:`ExpertSlabPool`.

    Note: zero-sync fast-path waves increment ``fast_path_waves`` and
    ``waves_executed`` without updating W-TinyLFU frequencies or ``hits``.
    """

    max_slots: int
    allocated_slots: int
    resident_experts: int
    transient_slots: int
    hits: int
    misses: int
    evictions: int
    relocations: int
    waves_executed: int
    fast_path_waves: int


@dataclass(frozen=True, slots=True)
class _PoolViews:
    arena_bytes: torch.Tensor
    arena_views: MappingProxyType[str, torch.Tensor]
    arena_flat_views: dict[str, torch.Tensor]
    side_views: MappingProxyType[str, torch.Tensor]
    slot_offsets: MappingProxyType[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class _ExpertHostEntry:
    arena_tensors: tuple[tuple[str, torch.Tensor], ...]
    side_tensors: tuple[tuple[str, torch.Tensor], ...]
    all_host_tensors: tuple[torch.Tensor, ...]


@dataclass(slots=True)
class _SlotState:
    slot_id: int
    key: tuple[int, int] | None = None
    phase: _SlotPhase = _SlotPhase.EMPTY
    active_readers: int = 0
    lease: StreamLease | None = None
    lease_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True, slots=True)
class _PendingWaveRelease:
    slot: _SlotState
    release_stream: torch.cuda.Stream
    end_streams: tuple[torch.cuda.Stream, ...] = ()


@dataclass(frozen=True, slots=True)
class _PreparedMiss:
    slot: _SlotState
    key: tuple[int, int]
    entry: _ExpertHostEntry
    admit_resident: bool
    needs_stage: bool = True


@dataclass(frozen=True, slots=True)
class _MissRequest:
    group_id: Hashable
    group_ord: int
    entries: tuple[_ExpertHostEntry, ...]
    bypass: bool
    record_stats: bool
    recorded_eids: set[int] | None = None


class _CountMinSketch:
    """Compact 4-bit W-TinyLFU frequency estimator with periodic aging."""

    _HASH_SEEDS = (
        0x9E3779B97F4A7C15,
        0xBF58476D1CE4E5B9,
        0x94D049BB133111EB,
        0xD6E8FEB86659FD93,
    )
    _MAX_COUNT = 15

    def __init__(self, max_slots: int) -> None:
        target = max(64, 4 * max(1, max_slots))
        width = 1 << (target - 1).bit_length()
        self._width = min(65536, max(64, width))
        self._mask = self._width - 1
        self._rows = tuple(bytearray(self._width) for _ in self._HASH_SEEDS)
        self._samples = 0
        self._reset_threshold = max(64, 10 * max(1, max_slots))

    def _indices(self, group_ord: int, expert_id: int) -> tuple[int, int, int, int]:
        base = ((group_ord * 0x9E3779B97F4A7C15) ^ (expert_id + 1)) & 0xFFFFFFFFFFFFFFFF
        out: list[int] = []
        for seed in self._HASH_SEEDS:
            h = (base * seed) & 0xFFFFFFFFFFFFFFFF
            h ^= h >> 30
            h = (h * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
            out.append(((h >> 32) ^ h) & self._mask)
        return out[0], out[1], out[2], out[3]

    def increment(self, group_ord: int, expert_id: int) -> int:
        idxs = self._indices(group_ord, expert_id)
        min_val = self._MAX_COUNT
        for row, idx in zip(self._rows, idxs, strict=True):
            val = row[idx]
            if val < self._MAX_COUNT:
                val += 1
                row[idx] = val
            min_val = min(min_val, val)
        self._samples += 1
        if self._samples >= self._reset_threshold:
            self._halve()
        return min_val

    def _halve(self) -> None:
        for row in self._rows:
            row[:] = row.translate(_HALVE_TABLE)
        self._samples >>= 1

    def estimate(self, group_ord: int, expert_id: int) -> int:
        idxs = self._indices(group_ord, expert_id)
        return min(row[idx] for row, idx in zip(self._rows, idxs, strict=True))


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
    """Wait for every in-flight H2D DMA event before closing its ``PinLease``."""
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


def _release_slot_stream_lease(
    slot: _SlotState,
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
    """Single RAII owner pairing ``_SlotState.active_readers`` and ``StreamLease`` readers."""

    def __init__(self, pool: ExpertSlabPool) -> None:
        self._pool = pool
        self._slots: list[_SlotState] = []
        self._acquired_slots: set[int] = set()
        self._end_streams: dict[int, tuple[torch.cuda.Stream, ...]] = {}
        self._stream: torch.cuda.Stream | None = None

    def add_unlocked(self, slot: _SlotState) -> None:
        """Increment ``slot.active_readers`` under ``pool._lock`` and track ownership."""
        slot.active_readers += 1
        self._slots.append(slot)

    def adopt_unlocked(self, slot: _SlotState) -> None:
        """Adopt an already-incremented reader slot under ``pool._lock``."""
        self._slots.append(slot)

    def acquire(self) -> None:
        """Acquire shared ``StreamLease`` readers on the active CUDA compute stream."""
        if self._pool._device.type != "cuda":
            self._acquired_slots = {s.slot_id for s in self._slots}
            return
        stream = torch.cuda.current_stream(self._pool._device)
        self._stream = stream
        for slot in self._slots:
            assert slot.lease is not None
            with slot.lease_lock:
                slot.lease.acquire(stream, exclusive=False)
            self._acquired_slots.add(slot.slot_id)

    def release(self) -> None:
        """Idempotently release ``StreamLease``s and retain failed slots for retry."""
        if not self._slots:
            return
        end_stream = (
            torch.cuda.current_stream(self._pool._device)
            if self._pool._device.type == "cuda"
            else None
        )
        release_stream = self._stream
        remaining_slots: list[_SlotState] = []
        remaining_acquired: set[int] = set()
        remaining_ends: dict[int, tuple[torch.cuda.Stream, ...]] = {}
        first_err: BaseException | None = None
        for slot in self._slots:
            if (
                self._pool._device.type == "cuda"
                and slot.lease is not None
                and release_stream is not None
                and slot.slot_id in self._acquired_slots
            ):
                extras = _merge_end_stream(
                    self._end_streams.get(slot.slot_id, ()),
                    end_stream,
                    release_stream,
                )
                released, err = _release_slot_stream_lease(slot, release_stream, extras)
                first_err = _merge_error(first_err, err)
                if released:
                    with self._pool._lock:
                        slot.active_readers = max(0, slot.active_readers - 1)
                else:
                    remaining_slots.append(slot)
                    remaining_acquired.add(slot.slot_id)
                    if extras:
                        remaining_ends[slot.slot_id] = extras
            else:
                with self._pool._lock:
                    slot.active_readers = max(0, slot.active_readers - 1)
        self._slots = remaining_slots
        self._acquired_slots = remaining_acquired
        self._end_streams = remaining_ends
        if not self._slots:
            self._stream = None
        if first_err is not None:
            raise first_err

    def handoff_to_pool(self) -> None:
        """Transfer any still-unreleased slots to the pool before exiting a wave session."""
        if not self._slots:
            return
        slots = self._slots
        acquired = self._acquired_slots
        end_streams = self._end_streams
        release_stream = self._stream
        with self._pool._lock:
            for slot in slots:
                if (
                    self._pool._device.type == "cuda"
                    and slot.lease is not None
                    and release_stream is not None
                    and slot.slot_id in acquired
                ):
                    self._pool._pending_wave_releases.append(
                        _PendingWaveRelease(
                            slot=slot,
                            release_stream=release_stream,
                            end_streams=end_streams.get(slot.slot_id, ()),
                        )
                    )
                else:
                    slot.active_readers = max(0, slot.active_readers - 1)
        self._slots = []
        self._acquired_slots = set()
        self._end_streams = {}
        self._stream = None


def _validate_optional_positive_int(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None; got {value!r}.")
    return value


def _validate_pool_init_args(
    arena: Uint8DualArena,
    schema: SlotSchema,
    *,
    max_slots: int | None,
    budget: VramBudget | None,
    pin_manager: PinManager | None,
    default_bypass_wave_slots: int | None,
) -> tuple[int | None, int | None, PinManager, VramBudget | None]:
    if not isinstance(arena, Uint8DualArena):
        raise TypeError(f"arena must be a Uint8DualArena; got {type(arena).__name__}.")
    if not arena.is_open:
        raise RuntimeError("Cannot attach ExpertSlabPool to a closed Uint8DualArena.")
    if not isinstance(schema, SlotSchema):
        raise TypeError(f"schema must be a SlotSchema; got {type(schema).__name__}.")
    if arena.alignment < schema.field_alignment:
        raise ValueError(
            f"Arena alignment ({arena.alignment}) must be >= "
            f"schema.field_alignment ({schema.field_alignment})."
        )
    resolved_max = _validate_optional_positive_int(max_slots, name="max_slots")
    resolved_bypass = _validate_optional_positive_int(
        default_bypass_wave_slots, name="default_bypass_wave_slots"
    )
    if budget is not None and not isinstance(budget, VramBudget):
        raise TypeError(f"budget must be a VramBudget or None; got {type(budget).__name__}.")
    effective_budget = arena.budget if budget is None else budget
    resolved_pin = _pin_manager.host_pin_manager if pin_manager is None else pin_manager
    if not isinstance(resolved_pin, PinManager):
        raise TypeError(
            f"pin_manager must be a PinManager or None; got {type(resolved_pin).__name__}."
        )
    return resolved_max, resolved_bypass, resolved_pin, effective_budget


def _configure_or_validate_low_layout(
    arena: Uint8DualArena,
    schema: SlotSchema,
    max_slots: int | None,
) -> tuple[LowSlotLayout, bool]:
    existing = arena.low_slot_layout
    if existing is None:
        return arena.configure_low_slots(schema.drawer_bytes, max_slots=max_slots), True
    if arena.active_low_slots:
        raise RuntimeError(
            "Cannot attach ExpertSlabPool to a Uint8DualArena with active low slots."
        )
    if existing.slot_bytes < schema.drawer_bytes:
        raise ValueError(
            f"Arena low slot_bytes ({existing.slot_bytes}) is smaller than "
            f"schema drawer_bytes ({schema.drawer_bytes})."
        )
    if max_slots is not None and existing.max_slots != max_slots:
        raise ValueError(
            f"Arena low max_slots ({existing.max_slots}) does not match "
            f"requested max_slots={max_slots}."
        )
    return existing, False


def _build_side_pools(
    schema: SlotSchema,
    max_slots: int,
    device: torch.device,
    budget: VramBudget | None,
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
                    reservation_name=f"expert_side_{spec.name}",
                )
    except BaseException:
        for pool in pools.values():
            with contextlib.suppress(BaseException):
                pool.close()
        raise
    return pools


def _setup_transfer_context(
    device: torch.device,
    transfer_context: DeviceTransferContext | None,
) -> tuple[DeviceTransferContext | None, bool, bool]:
    if device.type != "cuda":
        if transfer_context is not None:
            raise ValueError("CPU ExpertSlabPool does not accept a CUDA DeviceTransferContext.")
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


def _validate_host_tensor_base(tensor: object, *, expert_idx: int, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or type(tensor) not in (
        torch.Tensor,
        torch.nn.Parameter,
    ):
        raise ValueError(
            f"Expert {expert_idx} field {name!r} must be a plain torch.Tensor "
            f"(or frozen nn.Parameter); got {type(tensor).__name__}."
        )
    if tensor.layout is not torch.strided:
        raise ValueError(f"Expert {expert_idx} field {name!r} must be a strided Tensor.")
    if tensor.device.type != "cpu":
        raise ValueError(f"Expert {expert_idx} field {name!r} must live on CPU; got {tensor.device}.")
    if tensor.requires_grad:
        raise ValueError(
            f"Expert {expert_idx} field {name!r} must be frozen (requires_grad=False) before capture."
        )
    return tensor


def _validate_and_capture_expert(
    schema: SlotSchema,
    expert_mapping: Mapping[str, torch.Tensor],
    *,
    expert_idx: int,
) -> _ExpertHostEntry:
    if not isinstance(expert_mapping, Mapping):
        raise TypeError(f"Expert {expert_idx} must be a mapping of field name -> Tensor.")
    expected_keys = set(schema.arena_field_names) | set(schema.side_field_names)
    actual_keys = set(expert_mapping.keys())
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"Expert {expert_idx} fields mismatch: missing={missing}, extra={extra}.")

    arena_tensors: list[tuple[str, torch.Tensor]] = []
    side_tensors: list[tuple[str, torch.Tensor]] = []
    all_host: list[torch.Tensor] = []

    for af in schema.arena_fields:
        tensor = _validate_host_tensor_base(expert_mapping[af.name], expert_idx=expert_idx, name=af.name)
        if tensor.dtype is not torch.uint8:
            raise ValueError(
                f"Expert {expert_idx} arena field {af.name!r} must be torch.uint8; got {tensor.dtype}."
            )
        if tensor.nbytes == 0 or tensor.nbytes > af.nbytes:
            raise ValueError(
                f"Expert {expert_idx} arena field {af.name!r} size {tensor.nbytes} "
                f"must lie in [1, {af.nbytes}]."
            )
        captured = capture_host_tensor(tensor, memory_format=torch.contiguous_format)
        flat = captured if captured.ndim == 1 else captured.reshape(-1)
        arena_tensors.append((af.name, flat))
        all_host.append(flat)

    for sf in schema.side_fields:
        tensor = _validate_host_tensor_base(expert_mapping[sf.name], expert_idx=expert_idx, name=sf.name)
        if tensor.dtype is not sf.dtype or tuple(tensor.shape) != sf.shape:
            raise ValueError(
                f"Expert {expert_idx} side field {sf.name!r} expected "
                f"{sf.dtype} {sf.shape}, got {tensor.dtype} {tuple(tensor.shape)}."
            )
        captured = capture_host_tensor(tensor, memory_format=torch.contiguous_format)
        side_tensors.append((sf.name, captured))
        all_host.append(captured)

    return _ExpertHostEntry(
        arena_tensors=tuple(arena_tensors),
        side_tensors=tuple(side_tensors),
        all_host_tensors=tuple(all_host),
    )


def _build_pool_views(
    arena: Uint8DualArena,
    schema: SlotSchema,
    layout: LowSlotLayout,
    side_pools: Mapping[str, TypedSidePool],
    device: torch.device,
) -> _PoolViews:
    with torch.inference_mode(False):
        arena_bytes = arena.raw_uint8_view()
        arena_views = {
            af.name: arena.low_pool_view(af.shape, field_offset=af.offset)
            for af in schema.arena_fields
        }
        arena_flat_views = {
            af.name: arena.low_pool_view((af.nbytes,), field_offset=af.offset)
            for af in schema.arena_fields
        }
        side_views = {name: sp.tensor for name, sp in side_pools.items()}
        slot_idx_vec = torch.arange(layout.max_slots, dtype=torch.int64, device=device)
        slot_offsets = {
            af.name: slot_idx_vec * layout.slot_stride + af.offset
            for af in schema.arena_fields
        }
        if device.type == "cuda":
            event = _record_event(torch.cuda.current_stream(device))
            _sync_interrupt_safe(event.synchronize)
    return _PoolViews(
        arena_bytes=arena_bytes,
        arena_views=MappingProxyType(arena_views),
        arena_flat_views=arena_flat_views,
        side_views=MappingProxyType(side_views),
        slot_offsets=MappingProxyType(slot_offsets),
    )


class ExpertSlabPool:
    """Global multi-group fixed-drawer cache and wave streamer in ``Uint8DualArena``.

    Manages read-only keyed blobs on the ascending low end of a
    :class:`Uint8DualArena` alongside dedicated :class:`TypedSidePool`s. One
    pool can serve multiple layers via distinct ``group_id``s, supports variable
    per-expert byte sizes via fixed-drawer padding, runs hit-first double-buffered
    waves with W-TinyLFU + ``bypass`` transient slots, skips active-expert
    readbacks when an entire group is cached and uncontended, and cooperates with
    peer high-end allocations via frontier relocation and eviction.

    Internal Lock Ordering:
        ``_close_lock`` -> ``_lock`` -> ``arena._lock``.
        ``slot.lease_lock`` is disjoint from ``_lock`` and ordered strictly
        before ``arena._lock`` (never acquire ``_lock`` while holding
        ``slot.lease_lock``; never acquire two slots' ``lease_lock``s at once).
        All CUDA syncs, blocking I/O, and external reclaimer calls run outside
        ``_lock``.

    Caller Contracts:
        - Host expert tensors passed to :meth:`register_group` must be frozen
          (``requires_grad=False``) CPU plain tensors and stay immutable afterward.
        - All :class:`ExpertWave` views are strictly read-only; GPU work using a
          wave must be enqueued on the stream active when fetching the wave before
          advancing the iterator or exiting ``with pool.waves(...)``.
        - Close this pool (or exit its ``with`` block) outside active ``waves()``
          sessions and before closing the backing ``Uint8DualArena``.
    """

    def __init__(
        self,
        arena: Uint8DualArena,
        schema: SlotSchema,
        *,
        max_slots: int | None = None,
        budget: VramBudget | None = None,
        transfer_context: DeviceTransferContext | None = None,
        pin_manager: PinManager | None = None,
        default_bypass_wave_slots: int | None = None,
    ) -> None:
        """Attach a fixed-drawer slab pool to the low end of ``arena``.

        Args:
            arena: Backing :class:`Uint8DualArena` owning raw ``uint8`` device storage.
            schema: :class:`SlotSchema` describing arena drawers and typed side fields.
            max_slots: Optional slot limit <= what ``arena`` can hold at the configured stride.
            budget: Optional :class:`VramBudget` for typed side pools; defaults to
                ``arena.budget`` when omitted.
            transfer_context: Optional CUDA :class:`DeviceTransferContext` shared across
                components on ``arena.device``.
            pin_manager: Optional :class:`PinManager`; defaults to ``host_pin_manager``.
            default_bypass_wave_slots: Optional default wave cap when ``bypass=True``
                and ``waves(..., max_wave_slots=None)``.
        """
        resolved_max, resolved_bypass, resolved_pin, effective_budget = (
            _validate_pool_init_args(
                arena,
                schema,
                max_slots=max_slots,
                budget=budget,
                pin_manager=pin_manager,
                default_bypass_wave_slots=default_bypass_wave_slots,
            )
        )
        device = arena.device
        self._lock = threading.Lock()
        self._op_cond = threading.Condition(self._lock)
        self._close_lock = threading.Lock()
        self._arena = arena
        self._schema = schema
        self._device = device
        self._pin_manager = resolved_pin
        self._default_bypass_wave_slots = resolved_bypass
        self._layout: LowSlotLayout | None = None
        self._max_slots = 0
        self._configured_fresh_layout = False
        self._transfer_context: DeviceTransferContext | None = None
        self._owns_transfer_context = False
        self._borrowed_transfer_context = False
        self._side_pools: dict[str, TypedSidePool] = {}
        self._views: _PoolViews | None = None
        self._init_tracking_state()
        self._reclaim_cb = self._reclaim_low_frontier

        # Claim low-end ownership first so two pools cannot race on arena layout.
        arena.set_low_reclaimer(self._reclaim_cb)
        try:
            layout, configured_fresh = _configure_or_validate_low_layout(
                arena, schema, resolved_max
            )
            self._layout = layout
            self._max_slots = layout.max_slots
            self._configured_fresh_layout = configured_fresh
            self._sketch = _CountMinSketch(layout.max_slots)

            ctx, owns_ctx, borrowed_ctx = _setup_transfer_context(device, transfer_context)
            self._transfer_context = ctx
            self._owns_transfer_context = owns_ctx
            self._borrowed_transfer_context = borrowed_ctx

            self._side_pools = _build_side_pools(schema, layout.max_slots, device, effective_budget)
            self._views = _build_pool_views(arena, schema, layout, self._side_pools, device)
        except BaseException as exc:
            cleanup_err = self._teardown_resources()
            merged = _merge_error(exc, cleanup_err)
            if merged is not None and merged is not exc:
                raise merged from exc
            raise

    def _init_tracking_state(self) -> None:
        self._groups: dict[Hashable, tuple[int, tuple[_ExpertHostEntry, ...]]] = {}
        self._next_group_ord = 0
        self._total_registered_experts = 0

        self._slots: dict[int, _SlotState] = {}
        self._resident: OrderedDict[tuple[int, int], int] = OrderedDict()
        self._group_slot_maps: dict[int, dict[int, int]] = {}
        self._all_resident_tables: dict[int, tuple[torch.Tensor, torch.cuda.Event | None]] = {}
        self._sketch = _CountMinSketch(1)

        self._inflight_arena_allocs = 0
        self._inflight_arena_frees = 0
        self._inflight_wave_release_reaps = 0
        self._active_wave_sessions = 0
        self._suppress_stats_depth = 0
        self._pending_pin_leases: deque[tuple[torch.cuda.Event | None, PinLease]] = deque()
        self._pending_wave_releases: deque[_PendingWaveRelease] = deque()
        self._closing = False
        self._closed = False

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._relocations = 0
        self._waves_executed = 0
        self._fast_path_waves = 0

    @property
    def schema(self) -> SlotSchema:
        """Schema describing arena and side fields per expert slot."""
        return self._schema

    @property
    def layout(self) -> LowSlotLayout:
        """Ascending low-end fixed-slot geometry on the backing arena."""
        assert self._layout is not None
        return self._layout

    @property
    def max_slots(self) -> int:
        """Maximum addressable slots owned by this pool."""
        return self._max_slots

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_open(self) -> bool:
        with self._lock:
            return not self._closed and not self._closing and self._arena.is_open

    def _require_open_unlocked(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("ExpertSlabPool is closed.")
        if not self._arena.is_open:
            raise RuntimeError("Backing Uint8DualArena is closed.")

    def _require_views(self) -> _PoolViews:
        views = self._views
        if views is None:
            raise RuntimeError("ExpertSlabPool is closed.")
        return views

    def _lookup_slot_id_unlocked(self, key: tuple[int, int]) -> int | None:
        group_map = self._group_slot_maps.get(key[0])
        if group_map is None:
            return None
        return group_map.get(key[1])

    def _unbind_slot_unlocked(
        self,
        slot: _SlotState,
        *,
        count_eviction: bool,
        next_phase: _SlotPhase = _SlotPhase.EMPTY,
    ) -> None:
        old_key = slot.key
        slot.key = None
        slot.phase = next_phase
        if old_key is None:
            return
        group_ord, expert_id = old_key
        group_map = self._group_slot_maps.get(group_ord)
        if group_map is not None and group_map.get(expert_id) == slot.slot_id:
            del group_map[expert_id]
            self._all_resident_tables.pop(group_ord, None)
        if self._resident.get(old_key) == slot.slot_id:
            del self._resident[old_key]
            if count_eviction:
                self._evictions += 1

    def _bind_slot_unlocked(
        self,
        slot: _SlotState,
        key: tuple[int, int],
        *,
        admit_resident: bool,
    ) -> None:
        group_ord, expert_id = key
        prev_id = self._lookup_slot_id_unlocked(key)
        if prev_id is not None and prev_id != slot.slot_id:
            prev_slot = self._slots.get(prev_id)
            if prev_slot is not None and prev_slot.key == key:
                prev_slot.key = None
        slot.key = key
        slot.phase = _SlotPhase.READY
        self._group_slot_maps[group_ord][expert_id] = slot.slot_id
        self._all_resident_tables.pop(group_ord, None)
        if admit_resident:
            self._resident[key] = slot.slot_id
            self._resident.move_to_end(key)
        else:
            self._resident.pop(key, None)

    def _is_transient_unlocked(self, slot: _SlotState) -> bool:
        return (
            slot.phase is _SlotPhase.READY
            and slot.key is not None
            and self._resident.get(slot.key) != slot.slot_id
        )

    def register_group(
        self,
        group_id: Hashable,
        experts: Iterable[Mapping[str, torch.Tensor]],
    ) -> None:
        """Register an immutable sequence of frozen host expert tensors under ``group_id``."""
        with self._lock:
            self._require_open_unlocked()
            if group_id in self._groups:
                raise ValueError(f"Group {group_id!r} is already registered.")
        captured = tuple(
            _validate_and_capture_expert(self._schema, item, expert_idx=idx)
            for idx, item in enumerate(experts)
        )
        if not captured:
            raise ValueError("register_group() requires at least one expert.")
        with self._lock:
            self._require_open_unlocked()
            if group_id in self._groups:
                raise ValueError(f"Group {group_id!r} is already registered.")
            group_ord = self._next_group_ord
            self._next_group_ord += 1
            self._groups[group_id] = (group_ord, captured)
            self._group_slot_maps[group_ord] = {}
            self._total_registered_experts += len(captured)

    @contextlib.contextmanager
    def no_record_stats(self) -> Iterator[None]:
        """Temporarily suppress W-TinyLFU / LRU residency updates (re-entrant)."""
        with self._lock:
            self._require_open_unlocked()
            self._suppress_stats_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._suppress_stats_depth = max(0, self._suppress_stats_depth - 1)

    def commit_accepted(self, keys: Iterable[tuple[Hashable, int]]) -> None:
        """Atomically credit W-TinyLFU / LRU for accepted ``(group_id, expert_id)`` keys."""
        key_list = list(keys)
        resolved: list[tuple[int, int]] = []
        with self._lock:
            self._require_open_unlocked()
            for group_id, expert_id in key_list:
                resolved.append(self._resolve_expert_key_unlocked(group_id, expert_id))
            for key in resolved:
                freq = self._sketch.increment(*key)
                if key in self._resident:
                    self._resident.move_to_end(key)
                else:
                    slot_id = self._lookup_slot_id_unlocked(key)
                    if slot_id is not None and freq >= 2:
                        self._promote_transient_unlocked(key, slot_id)

    def _resolve_expert_key_unlocked(self, group_id: Hashable, expert_id: int) -> tuple[int, int]:
        info = self._groups.get(group_id)
        if info is None:
            raise KeyError(f"Unknown expert group {group_id!r}.")
        if isinstance(expert_id, bool) or not isinstance(expert_id, int):
            raise TypeError(f"expert_id must be an int; got {expert_id!r}.")
        group_ord, entries = info
        if expert_id < 0 or expert_id >= len(entries):
            raise IndexError(
                f"expert_id {expert_id} out of range [0, {len(entries)}) for group {group_id!r}."
            )
        return group_ord, expert_id

    def _promote_transient_unlocked(self, key: tuple[int, int], slot_id: int) -> None:
        slot = self._slots.get(slot_id)
        if slot is None or slot.key != key or not self._is_transient_unlocked(slot):
            return
        self._resident[key] = slot_id
        self._resident.move_to_end(key)

    def _reap_completed_pin_leases(self) -> None:
        to_close: list[PinLease] = []
        with self._lock:
            while self._pending_pin_leases:
                event, _lease = self._pending_pin_leases[0]
                if event is None or not event.query():
                    break
                self._pending_pin_leases.popleft()
                to_close.append(_lease)
        first_err: BaseException | None = None
        for lease in to_close:
            try:
                lease.close()
            except BaseException as exc:
                if first_err is None:
                    first_err = exc
        if first_err is not None:
            raise first_err

    def _reap_pending_wave_releases(self) -> None:
        if self._device.type != "cuda":
            return
        with self._lock:
            if not self._pending_wave_releases or self._closed or self._closing:
                return
            batch = list(self._pending_wave_releases)
            self._pending_wave_releases.clear()
            self._inflight_wave_release_reaps += 1

        released_slots: list[_SlotState] = []
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
                    self._pending_wave_releases.extendleft(reversed(still_pending))
                self._inflight_wave_release_reaps = max(
                    0, self._inflight_wave_release_reaps - 1
                )
                self._op_cond.notify_all()
        if interrupt is not None:
            raise interrupt

    def stats(self) -> ExpertPoolStats:
        """Return a snapshot of slot usage, hit/miss, eviction, and wave statistics.

        On CUDA, reaps any finished :class:`PinLease` instances and retries any
        deferred :class:`StreamLease` releases before reporting.
        """
        if self._device.type == "cuda":
            self._reap_pending_wave_releases()
            self._reap_completed_pin_leases()
        with self._lock:
            transient_slots = sum(1 for s in self._slots.values() if self._is_transient_unlocked(s))
            return ExpertPoolStats(
                max_slots=self._max_slots,
                allocated_slots=len(self._slots),
                resident_experts=len(self._resident),
                transient_slots=transient_slots,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                relocations=self._relocations,
                waves_executed=self._waves_executed,
                fast_path_waves=self._fast_path_waves,
            )

    def _ensure_copy_stream(self) -> torch.cuda.Stream:
        assert self._transfer_context is not None
        return self._transfer_context.ensure_stream()

    def _pick_resident_victim_unlocked(
        self,
        *,
        max_slot_id: int | None = None,
        max_freq: int | None = None,
    ) -> tuple[tuple[int, int], _SlotState, int] | None:
        best: tuple[int, tuple[int, int], _SlotState] | None = None
        for v_key, v_slot_id in self._resident.items():
            if max_slot_id is not None and v_slot_id > max_slot_id:
                continue
            slot = self._slots.get(v_slot_id)
            if slot is None or slot.active_readers > 0 or slot.phase is not _SlotPhase.READY:
                continue
            v_freq = self._sketch.estimate(*v_key)
            if max_freq is not None and v_freq >= max_freq:
                continue
            if best is None or v_freq < best[0]:
                best = (v_freq, v_key, slot)
                if v_freq == 0:
                    break
        if best is None:
            return None
        return best[1], best[2], best[0]

    def _find_reusable_slot_unlocked(
        self,
        *,
        want_transient: bool,
        max_slot_id: int | None = None,
    ) -> _SlotState | None:
        best_id: int | None = None
        for sid, slot in self._slots.items():
            if max_slot_id is not None and sid > max_slot_id:
                continue
            if slot.active_readers > 0:
                continue
            matches = (
                self._is_transient_unlocked(slot)
                if want_transient
                else slot.phase is _SlotPhase.EMPTY
            )
            if matches and (best_id is None or sid < best_id):
                best_id = sid
        return None if best_id is None else self._slots[best_id]

    def _claim_reusable_slot_unlocked(
        self,
        *,
        next_phase: _SlotPhase,
        max_slot_id: int | None = None,
    ) -> _SlotState | None:
        slot = self._find_reusable_slot_unlocked(want_transient=False, max_slot_id=max_slot_id)
        if slot is None:
            slot = self._find_reusable_slot_unlocked(want_transient=True, max_slot_id=max_slot_id)
            if slot is not None:
                self._unbind_slot_unlocked(slot, count_eviction=False, next_phase=next_phase)
                return slot
        if slot is None:
            return None
        slot.phase = next_phase
        return slot

    def _claim_transient_for_miss_unlocked(
        self,
        key: tuple[int, int],
        *,
        bypass: bool,
    ) -> tuple[_SlotState, bool] | None:
        trans = self._find_reusable_slot_unlocked(want_transient=True)
        if trans is None:
            return None
        self._unbind_slot_unlocked(trans, count_eviction=False, next_phase=_SlotPhase.STAGING)
        if bypass or not self._resident:
            return trans, not bypass
        min_res_freq = min(self._sketch.estimate(*res_key) for res_key in self._resident)
        return trans, (self._sketch.estimate(*key) >= min_res_freq)

    def _make_slot_state(self, slot_id: int, *, phase: _SlotPhase) -> _SlotState:
        alloc_stream = torch.cuda.current_stream(self._device) if self._device.type == "cuda" else None
        return _SlotState(
            slot_id=slot_id,
            phase=phase,
            lease=None if alloc_stream is None else StreamLease(alloc_stream),
        )

    def _free_slot_to_arena(self, slot: _SlotState, *, sync_lease: bool = True) -> None:
        if sync_lease and slot.lease is not None:
            lease = slot.lease
            with slot.lease_lock:
                _sync_interrupt_safe(lease.synchronize_and_reset)
        popped = False
        with self._lock:
            if slot.slot_id in self._slots:
                del self._slots[slot.slot_id]
                self._inflight_arena_frees += 1
                popped = True
        if not popped:
            return
        try:
            if self._arena.is_open:
                _sync_interrupt_safe(lambda: self._arena.free_low_slot(slot.slot_id))
        finally:
            with self._lock:
                self._inflight_arena_frees = max(0, self._inflight_arena_frees - 1)
                self._op_cond.notify_all()

    def _alloc_new_slot_from_arena(
        self,
        *,
        allow_reclaim: bool,
        max_slot_id: int | None = None,
        phase: _SlotPhase,
    ) -> _SlotState | None:
        if max_slot_id is not None and max_slot_id < 0:
            return None
        try:
            allocated_id = self._arena.alloc_low_slot(
                allow_reclaim=allow_reclaim, max_slot_id=max_slot_id
            )
        except ArenaCapacityError:
            return None

        try:
            new_slot = self._make_slot_state(allocated_id, phase=phase)
        except BaseException:
            if self._arena.is_open:
                _sync_interrupt_safe(lambda: self._arena.free_low_slot(allocated_id))
            raise
        with self._lock:
            if self._closed or self._closing or not self._arena.is_open:
                self._inflight_arena_frees += 1
                release_new = True
            else:
                self._slots[allocated_id] = new_slot
                release_new = False
        if release_new:
            try:
                if new_slot.lease is not None:
                    _sync_interrupt_safe(new_slot.lease.synchronize_and_reset)
                if self._arena.is_open:
                    _sync_interrupt_safe(lambda: self._arena.free_low_slot(allocated_id))
            finally:
                with self._lock:
                    self._inflight_arena_frees = max(0, self._inflight_arena_frees - 1)
                    self._op_cond.notify_all()
            raise RuntimeError("ExpertSlabPool was closed during slot allocation.")
        return new_slot

    def _try_alloc_miss_slot(self, *, allow_reclaim: bool) -> _SlotState | None:
        with self._lock:
            self._require_open_unlocked()
            if len(self._slots) + self._inflight_arena_allocs >= self._max_slots:
                return None
            self._inflight_arena_allocs += 1
        try:
            return self._alloc_new_slot_from_arena(
                allow_reclaim=allow_reclaim, phase=_SlotPhase.STAGING
            )
        finally:
            with self._lock:
                self._inflight_arena_allocs = max(0, self._inflight_arena_allocs - 1)

    def _evict_resident_for_miss_unlocked(
        self,
        key: tuple[int, int],
        *,
        bypass: bool,
        force_resident_evict: bool,
    ) -> tuple[_SlotState, bool] | None:
        empty = self._find_reusable_slot_unlocked(want_transient=False)
        if empty is not None:
            empty.phase = _SlotPhase.STAGING
            return empty, not bypass
        claimed = self._claim_transient_for_miss_unlocked(key, bypass=bypass)
        if claimed is not None:
            return claimed
        victim = self._pick_resident_victim_unlocked()
        if victim is None:
            return None
        _v_key, v_slot, v_freq = victim
        cand_freq = self._sketch.estimate(*key)
        admit = (not bypass) and (cand_freq >= v_freq)
        if not force_resident_evict and not admit:
            return None
        self._unbind_slot_unlocked(v_slot, count_eviction=True, next_phase=_SlotPhase.STAGING)
        return v_slot, admit

    def _obtain_slot_for_miss(
        self,
        key: tuple[int, int],
        *,
        bypass: bool,
        can_evict_resident: bool,
        require_for_execution: bool,
        force_resident_evict: bool,
    ) -> tuple[_SlotState, bool] | None:
        found: tuple[_SlotState, bool] | None = None
        can_alloc_more = False
        with self._lock:
            self._require_open_unlocked()
            empty = self._find_reusable_slot_unlocked(want_transient=False)
            if empty is not None:
                empty.phase = _SlotPhase.STAGING
                found = (empty, not bypass)
            elif bypass:
                found = self._claim_transient_for_miss_unlocked(key, bypass=True)
            can_alloc_more = found is None and (
                len(self._slots) + self._inflight_arena_allocs < self._max_slots
            )

        # Let decode grow into uncontested free slots before reusing transient slots or reclaiming peers.
        if found is None and can_alloc_more:
            allocated = self._try_alloc_miss_slot(allow_reclaim=False)
            if allocated is not None:
                found = (allocated, not bypass)

        if found is None:
            can_reclaim_alloc = False
            with self._lock:
                self._require_open_unlocked()
                if not bypass:
                    found = self._claim_transient_for_miss_unlocked(key, bypass=False)
                can_reclaim_alloc = (
                    found is None
                    and require_for_execution
                    and ((not bypass) or (len(self._slots) == 0))
                    and (len(self._slots) + self._inflight_arena_allocs < self._max_slots)
                )
            if can_reclaim_alloc:
                allocated = self._try_alloc_miss_slot(allow_reclaim=True)
                if allocated is not None:
                    found = (allocated, not bypass)

        if found is None and can_evict_resident:
            with self._lock:
                self._require_open_unlocked()
                found = self._evict_resident_for_miss_unlocked(
                    key, bypass=bypass, force_resident_evict=force_resident_evict
                )
        return found

    def _copy_entry_fields(
        self,
        slot_id: int,
        entry: _ExpertHostEntry,
        *,
        non_blocking: bool,
    ) -> None:
        views = self._require_views()
        for field_name, src in entry.arena_tensors:
            dst = views.arena_flat_views[field_name][slot_id, : src.nbytes]
            self._pin_manager.transfer(dst, src, non_blocking=non_blocking)
        for field_name, src in entry.side_tensors:
            dst = self._side_pools[field_name].slot(slot_id)
            self._pin_manager.transfer(dst, src, non_blocking=non_blocking)

    def _stage_batch(self, batch: Sequence[_PreparedMiss], *, bypass: bool) -> None:
        stage_items = [item for item in batch if item.needs_stage]
        if not stage_items:
            return
        if self._device.type != "cuda":
            for item in stage_items:
                self._copy_entry_fields(item.slot.slot_id, item.entry, non_blocking=False)
            return

        self._reap_completed_pin_leases()
        host_tensors = [t for item in stage_items for t in item.entry.all_host_tensors]
        pin_lease = self._pin_manager.acquire(host_tensors, pin=not bypass)
        copy_stream = self._ensure_copy_stream()
        try:
            for item in stage_items:
                assert item.slot.lease is not None
                slot_id = item.slot.slot_id
                entry = item.entry
                with item.slot.lease_lock:
                    item.slot.lease.stage_with(
                        lambda sid=slot_id, ent=entry: self._copy_entry_fields(
                            sid, ent, non_blocking=True
                        ),
                        copy_stream,
                    )
            batch_event = _record_event(copy_stream)
            with self._lock:
                self._pending_pin_leases.append((batch_event, pin_lease))
        except BaseException as exc:
            self._finalize_pin_lease_on_error(copy_stream, pin_lease, exc)

    def _finalize_pin_lease_on_error(
        self,
        copy_stream: torch.cuda.Stream,
        pin_lease: PinLease,
        exc: BaseException,
    ) -> None:
        completed, sync_err = _run_until_complete(copy_stream.synchronize)
        if completed:
            pin_lease.close()
        else:
            with self._lock:
                self._pending_pin_leases.append((None, pin_lease))
        if sync_err is not None and not isinstance(sync_err, Exception):
            raise sync_err from exc
        raise exc

    def _commit_staged_batch_unlocked(
        self,
        batch: Sequence[_PreparedMiss],
        *,
        wave_lease: _WaveLease | None,
    ) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for item in batch:
            slot = item.slot
            if not item.needs_stage:
                if wave_lease is not None:
                    wave_lease.adopt_unlocked(slot)
                else:
                    slot.active_readers = max(0, slot.active_readers - 1)
                pairs.append((item.key[1], slot.slot_id))
                continue

            existing_id = self._lookup_slot_id_unlocked(item.key)
            existing = None if existing_id is None else self._slots.get(existing_id)
            if (
                existing is not None
                and existing.slot_id != slot.slot_id
                and existing.phase is _SlotPhase.READY
            ):
                slot.key = None
                slot.phase = _SlotPhase.EMPTY
                chosen = existing
            else:
                self._bind_slot_unlocked(slot, item.key, admit_resident=item.admit_resident)
                chosen = slot
            if wave_lease is not None:
                wave_lease.add_unlocked(chosen)
            pairs.append((item.key[1], chosen.slot_id))
        return pairs

    def _rollback_claimed_batch(self, items: Sequence[_PreparedMiss]) -> None:
        if not items:
            return
        with self._lock:
            for item in items:
                if item.needs_stage:
                    if item.slot.phase is _SlotPhase.STAGING:
                        item.slot.key = None
                        item.slot.phase = _SlotPhase.EMPTY
                else:
                    item.slot.active_readers = max(0, item.slot.active_readers - 1)

    def _make_device_int_table(self, mapping: Sequence[int]) -> torch.Tensor:
        with torch.inference_mode(False):
            if self._device.type == "cuda":
                return torch.tensor(mapping, dtype=torch.int32, pin_memory=True).to(
                    self._device, non_blocking=True
                )
            return torch.tensor(mapping, dtype=torch.int32, device=self._device)

    def _build_wave_object(
        self,
        *,
        index: int,
        group_id: Hashable,
        num_experts: int,
        expert_slot_pairs: Sequence[tuple[int, int]],
        is_hit_wave: bool,
        full_table: torch.Tensor | None = None,
    ) -> ExpertWave:
        views = self._require_views()
        if full_table is None:
            mapping = [-1] * num_experts
            for eid, sid in expert_slot_pairs:
                mapping[eid] = sid
            expert_to_slot = self._make_device_int_table(mapping)
        else:
            expert_to_slot = full_table
        return ExpertWave(
            index=index,
            group_id=group_id,
            expert_ids=tuple(eid for eid, _sid in expert_slot_pairs),
            expert_to_slot=expert_to_slot,
            active_slots_hint=len(expert_slot_pairs),
            is_hit_wave=is_hit_wave,
            arena_views=views.arena_views,
            side_views=views.side_views,
            arena_bytes=views.arena_bytes,
            slot_offsets=views.slot_offsets,
        )

    def _prepare_miss_batch(
        self,
        req: _MissRequest,
        remaining_misses: deque[int],
        out_batch: list[_PreparedMiss],
        *,
        wave_cap: int,
        require_at_least_one: bool,
        allow_evict_resident: bool,
        hold_existing_ready: bool = True,
    ) -> None:
        claiming: list[_PreparedMiss] = []
        claimed_eids: list[int] = []
        try:
            while remaining_misses and len(claiming) < wave_cap:
                eid = remaining_misses[0]
                key = (req.group_ord, eid)
                matched, held_slot = self._try_claim_existing_ready(
                    key,
                    hold_reader=hold_existing_ready,
                    record_stats=req.record_stats,
                    recorded_eids=req.recorded_eids,
                )
                if matched:
                    remaining_misses.popleft()
                    if held_slot is not None:
                        claimed_eids.append(eid)
                        claiming.append(
                            _PreparedMiss(
                                slot=held_slot,
                                key=key,
                                entry=req.entries[eid],
                                admit_resident=not req.bypass,
                                needs_stage=False,
                            )
                        )
                    continue
                if req.record_stats and (req.recorded_eids is None or eid not in req.recorded_eids):
                    with self._lock:
                        self._sketch.increment(req.group_ord, eid)
                    if req.recorded_eids is not None:
                        req.recorded_eids.add(eid)
                got = self._obtain_slot_for_miss(
                    key,
                    bypass=req.bypass,
                    can_evict_resident=allow_evict_resident,
                    require_for_execution=require_at_least_one,
                    force_resident_evict=allow_evict_resident
                    and require_at_least_one
                    and (req.bypass or not claiming),
                )
                if got is None:
                    if not require_at_least_one and not hold_existing_ready:
                        # Non-blocking prefetch: skip un-admitted cold expert and inspect remaining IDs.
                        remaining_misses.popleft()
                        continue
                    if not claiming and require_at_least_one:
                        raise ArenaCapacityError(
                            f"No unleased expert slot available for miss {(req.group_id, eid)}."
                        )
                    break
                slot, admit = got
                remaining_misses.popleft()
                claimed_eids.append(eid)
                claiming.append(
                    _PreparedMiss(slot=slot, key=key, entry=req.entries[eid], admit_resident=admit)
                )
            self._stage_batch(claiming, bypass=req.bypass)
            out_batch.extend(claiming)
        except BaseException:
            self._rollback_claimed_batch(claiming)
            for eid in reversed(claimed_eids):
                remaining_misses.appendleft(eid)
            raise

    def _try_claim_existing_ready(
        self,
        key: tuple[int, int],
        *,
        hold_reader: bool,
        record_stats: bool,
        recorded_eids: set[int] | None,
    ) -> tuple[bool, _SlotState | None]:
        with self._lock:
            self._require_open_unlocked()
            existing_id = self._lookup_slot_id_unlocked(key)
            existing = None if existing_id is None else self._slots.get(existing_id)
            if existing is None or existing.phase is not _SlotPhase.READY:
                return False, None
            if record_stats and (recorded_eids is None or key[1] not in recorded_eids):
                self._sketch.increment(*key)
                if recorded_eids is not None:
                    recorded_eids.add(key[1])
            if not hold_reader:
                return True, None
            existing.active_readers += 1
            return True, existing

    def _build_device_int_table(
        self, mapping: Sequence[int]
    ) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        table = self._make_device_int_table(mapping)
        if self._device.type == "cuda":
            return table, _record_event(torch.cuda.current_stream(self._device))
        return table, None

    def _snapshot_fast_path_unlocked(
        self,
        group_id: Hashable,
        *,
        effective_record: bool,
    ) -> tuple[int, int, list[_SlotState], tuple[torch.Tensor, torch.cuda.Event | None] | None] | None:
        group_ord, entries = self._groups[group_id]
        num_experts = len(entries)
        group_map = self._group_slot_maps[group_ord]
        if len(group_map) != num_experts:
            return None
        if effective_record and self._total_registered_experts > self._max_slots:
            return None
        slots = [self._slots[group_map[eid]] for eid in range(num_experts)]
        if any(s.phase is not _SlotPhase.READY for s in slots):
            return None
        return group_ord, num_experts, slots, self._all_resident_tables.get(group_ord)

    def _try_fast_path_wave(
        self,
        group_id: Hashable,
        wave_lease: _WaveLease,
        *,
        effective_record: bool,
    ) -> ExpertWave | None:
        with self._lock:
            self._require_open_unlocked()
            snap = self._snapshot_fast_path_unlocked(group_id, effective_record=effective_record)
            if snap is None:
                return None
            group_ord, num_experts, slots, table_entry = snap
            expected_ids = [s.slot_id for s in slots] if table_entry is None else None
            if table_entry is not None:
                self._activate_fast_path_slots_unlocked(
                    slots, wave_lease, effective_record=effective_record
                )

        if table_entry is None:
            assert expected_ids is not None
            built = self._build_device_int_table(expected_ids)
            with self._lock:
                self._require_open_unlocked()
                snap2 = self._snapshot_fast_path_unlocked(
                    group_id, effective_record=effective_record
                )
                if snap2 is None:
                    return None
                _, num_experts, slots, table_entry = snap2
                if [s.slot_id for s in slots] != expected_ids:
                    return None
                if table_entry is None:
                    self._all_resident_tables[group_ord] = built
                    table_entry = built
                self._activate_fast_path_slots_unlocked(
                    slots, wave_lease, effective_record=effective_record
                )

        assert table_entry is not None
        wave_lease.acquire()
        table, ready_event = table_entry
        if self._device.type == "cuda":
            cur_stream = torch.cuda.current_stream(self._device)
            if ready_event is not None:
                if ready_event.query():
                    with self._lock:
                        cached = self._all_resident_tables.get(group_ord)
                        if cached is not None and cached[0] is table:
                            self._all_resident_tables[group_ord] = (table, None)
                else:
                    cur_stream.wait_event(ready_event)
            table.record_stream(cur_stream)
        pairs = tuple((eid, slots[eid].slot_id) for eid in range(num_experts))
        return self._build_wave_object(
            index=0,
            group_id=group_id,
            num_experts=num_experts,
            expert_slot_pairs=pairs,
            is_hit_wave=True,
            full_table=table,
        )

    def _activate_fast_path_slots_unlocked(
        self,
        slots: Sequence[_SlotState],
        wave_lease: _WaveLease,
        *,
        effective_record: bool,
    ) -> None:
        for s in slots:
            wave_lease.add_unlocked(s)
            if effective_record and s.key is not None:
                if s.key in self._resident:
                    self._resident.move_to_end(s.key)
                else:
                    self._promote_transient_unlocked(s.key, s.slot_id)
        self._fast_path_waves += 1
        self._waves_executed += 1

    def _resolve_active_ids(
        self,
        expert_ids: torch.Tensor | Sequence[int] | Iterable[int],
        *,
        num_experts: int,
    ) -> tuple[int, ...]:
        if isinstance(expert_ids, torch.Tensor):
            _validate_expert_id_tensor(expert_ids)
            if expert_ids.numel() == 0:
                return ()
            raw_ids = torch.unique(expert_ids.reshape(-1).to(torch.int64)).cpu().tolist()
            return tuple(eid for eid in raw_ids if 0 <= eid < num_experts)

        seen: dict[int, None] = {}
        for eid in expert_ids:
            if isinstance(eid, bool) or not isinstance(eid, int):
                raise TypeError(f"expert_id must be an int; got {eid!r}.")
            if 0 <= eid < num_experts:
                seen[eid] = None
        return tuple(seen.keys())

    def prefetch(
        self,
        group_id: Hashable,
        expert_ids: torch.Tensor | Sequence[int] | Iterable[int],
        *,
        bypass: bool = False,
    ) -> int:
        """Best-effort asynchronous prefetch of missing ``expert_ids`` into unleased slots.

        Never reclaims high-end arena peer space or displaces hotter residents when
        ``bypass=False``.
        """
        if self._device.type == "cuda":
            self._reap_pending_wave_releases()
        with self._lock:
            self._require_open_unlocked()
            info = self._groups.get(group_id)
            if info is None:
                raise KeyError(f"Unknown expert group {group_id!r}.")
            group_ord, entries = info
        active_ids = self._resolve_active_ids(expert_ids, num_experts=len(entries))
        if not active_ids:
            return 0

        misses: deque[int] = deque()
        with self._lock:
            self._require_open_unlocked()
            for eid in active_ids:
                if self._lookup_slot_id_unlocked((group_ord, eid)) is None:
                    misses.append(eid)
        if not misses:
            return 0

        req = _MissRequest(
            group_id=group_id,
            group_ord=group_ord,
            entries=entries,
            bypass=bypass,
            record_stats=False,
        )
        staged: list[_PreparedMiss] = []
        try:
            self._prepare_miss_batch(
                req,
                misses,
                staged,
                wave_cap=len(misses),
                require_at_least_one=False,
                allow_evict_resident=not bypass,
                hold_existing_ready=False,
            )
        finally:
            if staged:
                with self._lock:
                    self._commit_staged_batch_unlocked(staged, wave_lease=None)
        return len(staged)

    @contextlib.contextmanager
    def waves(
        self,
        group_id: Hashable,
        expert_ids: torch.Tensor | Sequence[int] | Iterable[int],
        *,
        bypass: bool = False,
        record_stats: bool | None = None,
        max_wave_slots: int | None = None,
    ) -> Iterator[Iterator[ExpertWave]]:
        """Yield ready :class:`ExpertWave` batches overlapping misses behind active hits.

        Kernels reading a yielded wave must launch on the CUDA stream active when
        advancing the iterator, before requesting the next wave or exiting the
        context manager. Out-of-range integer expert IDs (``< 0`` or
        ``>= num_experts``) are ignored as routing sentinels.
        """
        resolved_wave_cap = _validate_optional_positive_int(max_wave_slots, name="max_wave_slots")
        if self._device.type == "cuda":
            self._reap_pending_wave_releases()
        with self._lock:
            self._require_open_unlocked()
            info = self._groups.get(group_id)
            if info is None:
                raise KeyError(f"Unknown expert group {group_id!r}.")
            group_ord, entries = info
            effective_record = (
                (not bypass if record_stats is None else bool(record_stats))
                and self._suppress_stats_depth == 0
            )
        has_work, pre_resolved = self._preflight_wave_inputs(expert_ids, num_experts=len(entries))
        if not has_work:
            yield iter(())
            return

        with self._lock:
            self._require_open_unlocked()
            self._active_wave_sessions += 1

        gen = self._run_waves_generator(
            group_id=group_id,
            group_ord=group_ord,
            entries=entries,
            expert_ids=expert_ids,
            pre_resolved=pre_resolved,
            bypass=bypass,
            effective_record=effective_record,
            max_wave_slots=resolved_wave_cap,
        )
        try:
            yield gen
        finally:
            try:
                try:
                    gen.close()
                finally:
                    if self._device.type == "cuda":
                        self._reap_pending_wave_releases()
                        self._reap_completed_pin_leases()
            finally:
                with self._lock:
                    self._active_wave_sessions = max(0, self._active_wave_sessions - 1)

    def _partition_hits_and_misses(
        self,
        group_ord: int,
        active_ids: tuple[int, ...],
        wave_lease: _WaveLease,
        *,
        effective_record: bool,
    ) -> tuple[list[tuple[int, int]], deque[int]]:
        hit_pairs: list[tuple[int, int]] = []
        miss_queue: deque[int] = deque()
        with self._lock:
            self._require_open_unlocked()
            for eid in active_ids:
                key = (group_ord, eid)
                slot_id = self._lookup_slot_id_unlocked(key)
                slot = None if slot_id is None else self._slots.get(slot_id)
                if slot is not None and slot.phase is _SlotPhase.READY:
                    wave_lease.add_unlocked(slot)
                    hit_pairs.append((eid, slot.slot_id))
                    self._hits += 1
                    if effective_record:
                        freq = self._sketch.increment(*key)
                        if key in self._resident:
                            self._resident.move_to_end(key)
                        elif freq >= 2:
                            self._promote_transient_unlocked(key, slot.slot_id)
                else:
                    miss_queue.append(eid)
                    self._misses += 1
            if hit_pairs:
                self._waves_executed += 1
        return hit_pairs, miss_queue

    def _resolve_miss_wave_cap(
        self,
        miss_count: int,
        *,
        bypass: bool,
        max_wave_slots: int | None,
    ) -> int:
        if max_wave_slots is not None:
            return max_wave_slots
        if bypass:
            return self._default_bypass_wave_slots or max(
                1, min(miss_count, max(1, self._max_slots // 4))
            )
        return max(1, miss_count)

    def _preflight_wave_inputs(
        self,
        expert_ids: torch.Tensor | Sequence[int] | Iterable[int],
        *,
        num_experts: int,
    ) -> tuple[bool, tuple[int, ...] | None]:
        if isinstance(expert_ids, torch.Tensor):
            _validate_expert_id_tensor(expert_ids)
            return (expert_ids.numel() > 0), None
        resolved = self._resolve_active_ids(expert_ids, num_experts=num_experts)
        return bool(resolved), resolved

    def _run_waves_generator(
        self,
        *,
        group_id: Hashable,
        group_ord: int,
        entries: tuple[_ExpertHostEntry, ...],
        expert_ids: torch.Tensor | Sequence[int] | Iterable[int],
        pre_resolved: tuple[int, ...] | None,
        bypass: bool,
        effective_record: bool,
        max_wave_slots: int | None,
    ) -> Generator[ExpertWave]:
        held_lease = _WaveLease(self)
        staged_batch: list[_PreparedMiss] = []
        try:
            fast_wave = self._try_fast_path_wave(
                group_id, held_lease, effective_record=effective_record
            )
            if fast_wave is not None:
                yield fast_wave
                held_lease.release()
                return

            active_ids = (
                pre_resolved
                if pre_resolved is not None
                else self._resolve_active_ids(expert_ids, num_experts=len(entries))
            )
            if not active_ids:
                return

            num_experts = len(entries)
            hit_pairs, miss_queue = self._partition_hits_and_misses(
                group_ord, active_ids, held_lease, effective_record=effective_record
            )
            wave_idx = 0
            miss_req = _MissRequest(
                group_id=group_id,
                group_ord=group_ord,
                entries=entries,
                bypass=bypass,
                record_stats=effective_record,
                recorded_eids=set(),
            )
            wave_cap = self._resolve_miss_wave_cap(
                len(miss_queue), bypass=bypass, max_wave_slots=max_wave_slots
            )

            if hit_pairs:
                held_lease.acquire()
                if miss_queue:
                    self._prepare_miss_batch(
                        miss_req, miss_queue, staged_batch,
                        wave_cap=wave_cap, require_at_least_one=False, allow_evict_resident=False,
                    )
                yield self._build_wave_object(
                    index=wave_idx, group_id=group_id, num_experts=num_experts,
                    expert_slot_pairs=hit_pairs, is_hit_wave=True,
                )
                held_lease.release()
                wave_idx += 1

            while staged_batch or miss_queue:
                if not staged_batch:
                    self._prepare_miss_batch(
                        miss_req, miss_queue, staged_batch,
                        wave_cap=wave_cap, require_at_least_one=True, allow_evict_resident=True,
                    )
                with self._lock:
                    pairs = self._commit_staged_batch_unlocked(
                        staged_batch, wave_lease=held_lease
                    )
                    staged_batch.clear()
                    self._waves_executed += 1
                held_lease.acquire()
                if miss_queue:
                    self._prepare_miss_batch(
                        miss_req, miss_queue, staged_batch,
                        wave_cap=wave_cap, require_at_least_one=False, allow_evict_resident=False,
                    )
                yield self._build_wave_object(
                    index=wave_idx, group_id=group_id, num_experts=num_experts,
                    expert_slot_pairs=pairs, is_hit_wave=False,
                )
                held_lease.release()
                wave_idx += 1
        finally:
            try:
                held_lease.release()
            finally:
                held_lease.handoff_to_pool()
                if staged_batch:
                    with self._lock:
                        self._commit_staged_batch_unlocked(staged_batch, wave_lease=None)
                        staged_batch.clear()

    def _copy_slot_d2d(self, src_id: int, dst_id: int, *, non_blocking: bool) -> None:
        self._arena.low_slot_tensor(dst_id).copy_(
            self._arena.low_slot_tensor(src_id),
            non_blocking=non_blocking,
        )
        for pool in self._side_pools.values():
            pool.slot(dst_id).copy_(pool.slot(src_id), non_blocking=non_blocking)

    def _find_or_alloc_relocate_target(
        self,
        key: tuple[int, int],
        *,
        max_allowed_slot: int,
    ) -> tuple[_SlotState, bool] | None:
        if max_allowed_slot < 0:
            return None
        result: tuple[_SlotState, bool] | None = None
        with self._lock:
            if self._closed or self._closing:
                return None
            k_freq = self._sketch.estimate(*key)
            dst = self._claim_reusable_slot_unlocked(
                next_phase=_SlotPhase.RELOCATING, max_slot_id=max_allowed_slot
            )
            if dst is not None:
                result = (dst, False)

        if result is None:
            allocated = self._alloc_new_slot_from_arena(
                allow_reclaim=False, max_slot_id=max_allowed_slot, phase=_SlotPhase.RELOCATING
            )
            if allocated is not None:
                result = (allocated, True)

        if result is None:
            with self._lock:
                if not self._closed and not self._closing:
                    reused = self._claim_reusable_slot_unlocked(
                        next_phase=_SlotPhase.RELOCATING, max_slot_id=max_allowed_slot
                    )
                    if reused is not None:
                        result = (reused, False)
                    else:
                        victim = self._pick_resident_victim_unlocked(
                            max_slot_id=max_allowed_slot, max_freq=k_freq
                        )
                        if victim is not None:
                            _victim_key, victim_slot, _ = victim
                            self._unbind_slot_unlocked(
                                victim_slot, count_eviction=True, next_phase=_SlotPhase.RELOCATING
                            )
                            result = (victim_slot, False)
        return result

    def _queue_slot_d2d(self, src: _SlotState, dst: _SlotState) -> None:
        if self._device.type != "cuda":
            self._copy_slot_d2d(src.slot_id, dst.slot_id, non_blocking=False)
            return
        src_lease = src.lease
        dst_lease = dst.lease
        assert src_lease is not None
        assert dst_lease is not None
        copy_stream = self._ensure_copy_stream()
        with src.lease_lock:
            src_lease.acquire(copy_stream, exclusive=False)
        stage_err: BaseException | None = None
        try:
            with dst.lease_lock:
                dst_lease.stage_with(
                    lambda: self._copy_slot_d2d(src.slot_id, dst.slot_id, non_blocking=True),
                    copy_stream,
                )
        except BaseException as exc:
            stage_err = exc
        finally:
            released, rel_err = _release_slot_stream_lease(src, copy_stream, ())
            if not released:
                with self._lock:
                    src.active_readers += 1
                    self._pending_wave_releases.append(
                        _PendingWaveRelease(slot=src, release_stream=copy_stream)
                    )
        merged = _merge_error(stage_err, rel_err)
        if merged is not None:
            if stage_err is not None and merged is not stage_err:
                raise merged from stage_err
            raise merged

    def _relocate_slot_to_lower(
        self,
        src: _SlotState,
        key: tuple[int, int],
        *,
        max_allowed_slot: int,
    ) -> bool:
        target = self._find_or_alloc_relocate_target(key, max_allowed_slot=max_allowed_slot)
        if target is None:
            return False
        dst, dst_fresh = target
        try:
            self._queue_slot_d2d(src, dst)
        except BaseException:
            if dst_fresh:
                try:
                    self._free_slot_to_arena(dst, sync_lease=True)
                except BaseException:
                    with self._lock:
                        if dst.slot_id in self._slots:
                            dst.phase = _SlotPhase.EMPTY
                    raise
            else:
                with self._lock:
                    dst.phase = _SlotPhase.EMPTY
            raise

        with self._lock:
            key_still_owned = self._lookup_slot_id_unlocked(key) == src.slot_id
            self._unbind_slot_unlocked(src, count_eviction=False, next_phase=_SlotPhase.RELOCATING)
            if key_still_owned:
                self._bind_slot_unlocked(dst, key, admit_resident=True)
                self._relocations += 1
            else:
                dst.key = None
                dst.phase = _SlotPhase.EMPTY
        self._free_slot_to_arena(src, sync_lease=True)
        return True

    def _evict_frontier_slot(self, src: _SlotState) -> None:
        with self._lock:
            self._unbind_slot_unlocked(src, count_eviction=True, next_phase=_SlotPhase.RELOCATING)
        self._free_slot_to_arena(src, sync_lease=True)

    def _reclaim_low_frontier(self, target_low_frontier: int) -> None:
        self._reap_pending_wave_releases()
        layout = self._layout
        if layout is None:
            return
        max_allowed_slot = (target_low_frontier // layout.slot_stride) - 1
        while True:
            candidate: _SlotState | None = None
            candidate_key: tuple[int, int] | None = None
            candidate_is_resident = False
            with self._lock:
                while (
                    self._inflight_arena_frees > 0
                    and not self._closed
                    and not self._closing
                ):
                    if not self._slots or max(self._slots) <= max_allowed_slot:
                        self._op_cond.wait(timeout=0.05)
                        continue
                    break
                if self._closed or self._closing or not self._slots:
                    return
                top_id = max(self._slots)
                if top_id <= max_allowed_slot:
                    return
                slot = self._slots[top_id]
                if slot.active_readers > 0 or slot.phase in (
                    _SlotPhase.STAGING,
                    _SlotPhase.RELOCATING,
                ):
                    return
                slot.phase = _SlotPhase.RELOCATING
                candidate = slot
                candidate_key = slot.key
                candidate_is_resident = (
                    candidate_key is not None and candidate_key in self._resident
                )

            assert candidate is not None
            try:
                if candidate_is_resident and candidate_key is not None and self._relocate_slot_to_lower(
                    candidate, candidate_key, max_allowed_slot=max_allowed_slot
                ):
                    continue
                self._evict_frontier_slot(candidate)
            except BaseException:
                with self._lock:
                    if candidate.slot_id in self._slots:
                        if (
                            candidate.key is not None
                            and self._lookup_slot_id_unlocked(candidate.key) == candidate.slot_id
                        ):
                            candidate.phase = _SlotPhase.READY
                        else:
                            candidate.key = None
                            candidate.phase = _SlotPhase.EMPTY
                raise

    def _teardown_resources(self, *, slots: Sequence[_SlotState] = ()) -> BaseException | None:
        first_err: BaseException | None = None

        def _capture(fn: Callable[[], None]) -> None:
            nonlocal first_err
            try:
                fn()
            except BaseException as exc:
                first_err = _merge_error(first_err, exc)

        for slot in slots:
            _capture(lambda s=slot: self._free_slot_to_arena(s, sync_lease=False))
        if self._arena.is_open:
            reset_layout = self._layout if self._configured_fresh_layout else None
            _capture(
                lambda: self._arena._release_low_owner(
                    self._reclaim_cb, reset_expected=reset_layout
                )
            )
        for name, side_pool in list(self._side_pools.items()):
            def _close_side(pool_name: str = name, sp: TypedSidePool = side_pool) -> None:
                sp.close()
                self._side_pools.pop(pool_name, None)

            _capture(_close_side)
        if self._transfer_context is not None:
            ctx = self._transfer_context

            def _release_ctx() -> None:
                if self._borrowed_transfer_context:
                    ctx._release_borrow()
                    self._borrowed_transfer_context = False
                elif self._owns_transfer_context:
                    ctx.close()
                    self._owns_transfer_context = False
                self._transfer_context = None

            _capture(_release_ctx)

        with self._lock:
            self._views = None
            self._groups.clear()
            self._total_registered_experts = 0
            self._slots.clear()
            self._resident.clear()
            self._group_slot_maps.clear()
            self._all_resident_tables.clear()
            self._pending_wave_releases.clear()
            if first_err is None:
                self._closing = False
                self._closed = True
        return first_err

    def _synchronize_slots_for_close(
        self,
        slots: Sequence[_SlotState],
        pending_by_slot: dict[int, list[_PendingWaveRelease]],
    ) -> None:
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
                with self._lock:
                    slot.active_readers = max(0, slot.active_readers - len(slot_items))
        for slot_id, slot_items in list(pending_by_slot.items()):
            for item in slot_items:
                for end_stream in item.end_streams:
                    _sync_interrupt_safe(end_stream.synchronize)
                _sync_interrupt_safe(item.release_stream.synchronize)
            del pending_by_slot[slot_id]

    def close(self) -> None:
        """Synchronize pending CUDA/pin state, free arena low slots, and release all views."""
        with self._close_lock:
            with self._lock:
                if self._closed:
                    return
                pending_counts: dict[int, int] = {}
                for item in self._pending_wave_releases:
                    pending_counts[item.slot.slot_id] = (
                        pending_counts.get(item.slot.slot_id, 0) + 1
                    )
                if (
                    self._active_wave_sessions > 0
                    or self._inflight_arena_allocs > 0
                    or self._inflight_arena_frees > 0
                    or self._inflight_wave_release_reaps > 0
                    or any(
                        s.active_readers > pending_counts.get(s.slot_id, 0)
                        or s.phase in (_SlotPhase.STAGING, _SlotPhase.RELOCATING)
                        for s in self._slots.values()
                    )
                ):
                    raise RuntimeError(
                        "Cannot close ExpertSlabPool while wave, staging, or relocation work is active."
                    )
                self._closing = True
                slots = list(self._slots.values())
                pending_by_slot: dict[int, list[_PendingWaveRelease]] = {}
                while self._pending_wave_releases:
                    item = self._pending_wave_releases.popleft()
                    pending_by_slot.setdefault(item.slot.slot_id, []).append(item)
                pending = deque(self._pending_pin_leases)
                self._pending_pin_leases.clear()
                copy_stream = (
                    None
                    if self._transfer_context is None
                    else self._transfer_context.copy_stream
                )

            try:
                self._synchronize_slots_for_close(slots, pending_by_slot)
                _synchronize_pending_leases(pending, copy_stream)
            except BaseException:
                # Keep _closing=True so half-reset READY slots cannot be re-entered; retryable via close().
                with self._lock:
                    for slot_items in pending_by_slot.values():
                        self._pending_wave_releases.extend(slot_items)
                    if pending:
                        self._pending_pin_leases.extendleft(reversed(pending))
                raise

            err = self._teardown_resources(slots=slots)
            if err is not None:
                raise err

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _validate_expert_id_tensor(expert_ids: torch.Tensor) -> None:
    if expert_ids.layout is not torch.strided:
        raise TypeError("expert_ids tensor must use strided layout.")
    if expert_ids.device.type not in ("cpu", "cuda"):
        raise ValueError(
            f"expert_ids tensor must live on cpu or cuda; got {expert_ids.device}."
        )
    if expert_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"expert_ids tensor must have integer dtype; got {expert_ids.dtype}.")


__all__ = [
    "ExpertPoolStats",
    "ExpertSlabPool",
    "ExpertWave",
]
