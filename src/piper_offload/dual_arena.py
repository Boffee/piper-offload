"""Dual-ended single-dtype byte arena and typed side pools."""

import heapq
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Self, cast

import torch

from ._devices import canonical_device
from .vram_budget import VramBudget, VramReservation

_MAX_RECLAIM_RACE_RETRIES = 8


class ArenaCapacityError(RuntimeError):
    """Raised when an allocation cannot fit in the remaining arena gap."""


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return value


def _validate_nonneg_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}.")
    return value


def _record_cuda_alloc_event(
    device: torch.device, reservation: VramReservation | None
) -> torch.cuda.Event | None:
    if device.type != "cuda":
        return None
    try:
        ev = cast(torch.cuda.Event, torch.cuda.Event())
        ev.record(torch.cuda.current_stream(device))
        return ev
    except BaseException:
        if reservation is not None:
            reservation.release()
        raise


def _validate_alignment(
    value: object,
    *,
    name: str,
    min_alignment: int = 16,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < min_alignment:
        raise ValueError(
            f"{name} must be a power of two >= {min_alignment}; got {value!r}."
        )
    if (value & (value - 1)) != 0:
        raise ValueError(
            f"{name} must be a power of two >= {min_alignment}; got {value}."
        )
    return value


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for dim in reversed(shape):
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"All dimensions must be positive ints; got {shape!r}.")
        strides.append(stride)
        stride *= dim
    strides.reverse()
    return tuple(strides)


def _element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _aligned_side_stride0(
    inner_numel: int,
    dtype: torch.dtype,
    alignment_bytes: int,
) -> int:
    """Return element-count ``stride(0) >= inner_numel`` aligned to ``alignment_bytes``."""
    elem_bytes = _element_size(dtype)
    align_elems = max(1, math.lcm(alignment_bytes, elem_bytes) // elem_bytes)
    return _align_up(inner_numel, align_elems)


def _allocate_aligned_storage(
    numel: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    alignment_bytes: int,
    budget: VramBudget | None,
    reservation_name: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int, VramReservation | None]:
    """Allocate normal-mode storage with guaranteed ``alignment_bytes`` base alignment."""
    elem_bytes = _element_size(dtype)
    if alignment_bytes % elem_bytes != 0:
        raise ValueError(
            f"alignment_bytes ({alignment_bytes}) must be divisible by "
            f"element size ({elem_bytes}) for {dtype}."
        )
    native_align = 512 if device.type == "cuda" else 64
    pad_elems = (
        0 if alignment_bytes <= native_align else (alignment_bytes // elem_bytes) - 1
    )
    total_elems = numel + pad_elems
    allocated_bytes = total_elems * elem_bytes
    if budget is not None and not isinstance(budget, VramBudget):
        raise TypeError(f"budget must be a VramBudget or None; got {type(budget).__name__}.")
    reservation = (
        budget.reserve(allocated_bytes, name=reservation_name)
        if budget is not None
        else None
    )
    try:
        with torch.inference_mode(False):
            raw = torch.empty((total_elems,), dtype=dtype, device=device)
        rem = raw.data_ptr() % alignment_bytes
        if rem != 0 and pad_elems == 0:
            needed_pad = (alignment_bytes // elem_bytes) - 1
            extra_bytes = needed_pad * elem_bytes
            if reservation is not None:
                reservation.resize(allocated_bytes + extra_bytes)
            total_elems += needed_pad
            allocated_bytes += extra_bytes
            with torch.inference_mode(False):
                raw = torch.empty((total_elems,), dtype=dtype, device=device)
            rem = raw.data_ptr() % alignment_bytes
        if rem % elem_bytes != 0:
            raise RuntimeError(
                f"Allocator returned pointer {raw.data_ptr()} incompatible with element size {elem_bytes}."
            )
        base_offset = 0 if rem == 0 else (alignment_bytes - rem) // elem_bytes
        view = raw[base_offset : base_offset + numel]
        if view.data_ptr() % alignment_bytes != 0:
            raise RuntimeError(
                f"Failed to align {device} storage pointer {view.data_ptr()} to {alignment_bytes} bytes."
            )
    except BaseException:
        if reservation is not None:
            reservation.release()
        raise
    return raw, view, base_offset, allocated_bytes, reservation


def _validate_slot_id(slot_id: object, *, max_slots: int) -> int:
    if isinstance(slot_id, bool) or not isinstance(slot_id, int):
        raise TypeError(f"slot_id must be an int; got {slot_id!r}.")
    if slot_id < 0 or slot_id >= max_slots:
        raise IndexError(f"slot_id {slot_id} out of range [0, {max_slots}).")
    return slot_id


@dataclass(frozen=True, slots=True)
class ArenaFieldSpec:
    """Layout descriptor for one raw ``uint8`` field inside a :class:`SlotSchema` drawer."""

    name: str
    nbytes: int
    offset: int
    aligned_bytes: int
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SideFieldSpec:
    """Layout descriptor for one typed non-``uint8`` field stored in a :class:`TypedSidePool`."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    alignment_bytes: int

    @property
    def nbytes(self) -> int:
        """Logical unpadded payload size in bytes for one slot of this field."""
        return math.prod(self.shape) * _element_size(self.dtype)


def _parse_arena_fields(
    arena_fields: Mapping[str, int | tuple[int, ...]],
    alignment: int,
) -> tuple[tuple[ArenaFieldSpec, ...], dict[str, ArenaFieldSpec], int]:
    if not isinstance(arena_fields, Mapping) or not arena_fields:
        raise ValueError("SlotSchema requires at least one named arena field.")
    parsed: list[ArenaFieldSpec] = []
    by_name: dict[str, ArenaFieldSpec] = {}
    offset = 0
    for name, raw_spec in arena_fields.items():
        if not isinstance(name, str) or not name:
            raise ValueError("SlotSchema arena field names must be non-empty strings.")
        if isinstance(raw_spec, bool):
            raise ValueError(f"Arena field {name!r} byte count cannot be a bool.")
        if isinstance(raw_spec, int):
            nbytes = _validate_positive_int(raw_spec, name=f"arena_fields[{name!r}]")
            shape: tuple[int, ...] = (nbytes,)
        elif isinstance(raw_spec, tuple):
            if not raw_spec:
                raise ValueError(f"Arena field {name!r} shape tuple must be non-empty.")
            _contiguous_strides(raw_spec)
            shape = raw_spec
            nbytes = math.prod(shape)
        else:
            raise ValueError(
                f"Arena field {name!r} must be a positive int or shape tuple; got {raw_spec!r}."
            )
        aligned = _align_up(nbytes, alignment)
        spec = ArenaFieldSpec(
            name=name,
            nbytes=nbytes,
            offset=offset,
            aligned_bytes=aligned,
            shape=shape,
        )
        parsed.append(spec)
        by_name[name] = spec
        offset += aligned
    return tuple(parsed), by_name, offset


def _parse_side_fields(
    side_fields: Mapping[str, tuple[torch.dtype, tuple[int, ...]]] | None,
    arena_names: Mapping[str, ArenaFieldSpec],
    alignment: int,
) -> tuple[tuple[SideFieldSpec, ...], dict[str, SideFieldSpec], int]:
    if side_fields is not None and not isinstance(side_fields, Mapping):
        raise TypeError("side_fields must be a mapping or None.")
    if not side_fields:
        return (), {}, 0
    parsed: list[SideFieldSpec] = []
    by_name: dict[str, SideFieldSpec] = {}
    bytes_per_slot = 0
    for name, raw_side in side_fields.items():
        if not isinstance(name, str) or not name:
            raise ValueError("SlotSchema side field names must be non-empty strings.")
        if name in arena_names:
            raise ValueError(
                f"Duplicate field name {name!r} across arena and side fields."
            )
        if not isinstance(raw_side, tuple) or len(raw_side) != 2:
            raise ValueError(
                f"Side field {name!r} must be a (dtype, shape) tuple; got {raw_side!r}."
            )
        dtype, raw_shape = raw_side
        if not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"Side field {name!r} dtype must be a torch.dtype; got {dtype!r}."
            )
        if not isinstance(raw_shape, tuple) or not raw_shape:
            raise ValueError(
                f"Side field {name!r} shape must be a non-empty tuple; got {raw_shape!r}."
            )
        _contiguous_strides(raw_shape)
        spec = SideFieldSpec(
            name=name,
            dtype=dtype,
            shape=raw_shape,
            alignment_bytes=alignment,
        )
        stride0_elems = _aligned_side_stride0(math.prod(raw_shape), dtype, alignment)
        bytes_per_slot += stride0_elems * _element_size(dtype)
        parsed.append(spec)
        by_name[name] = spec
    return tuple(parsed), by_name, bytes_per_slot


class SlotSchema:
    """Generic drawer schema pairing raw ``uint8`` arena fields with typed side fields.

    Both low-end (for example ``ExpertSlabPool``) and high-end descending
    arena consumers describe per-slot layouts through ``SlotSchema`` without
    hardcoding format-specific field names or coupling ``piper-offload`` to
    kernel math.
    """

    def __init__(
        self,
        arena_fields: Mapping[str, int | tuple[int, ...]],
        side_fields: Mapping[str, tuple[torch.dtype, tuple[int, ...]]] | None = None,
        *,
        field_alignment: int = 256,
        side_alignment: int = 16,
    ) -> None:
        """Validate field specs and compute aligned drawer and side-pool sizes.

        Args:
            arena_fields: Non-empty mapping of field name -> byte count or ``uint8``
                inner shape tuple packed inside each aligned arena drawer.
            side_fields: Optional mapping of field name -> ``(dtype, shape)`` backed
                by dedicated :class:`TypedSidePool` allocations outside the arena.
            field_alignment: Power-of-two byte alignment >= 16 applied to each
                arena field offset inside a drawer.
            side_alignment: Power-of-two byte alignment >= 16 applied to each side
                pool's base pointer and ``stride(0)``.
        """
        f_align = _validate_alignment(
            field_alignment, name="field_alignment", min_alignment=16
        )
        s_align = _validate_alignment(
            side_alignment, name="side_alignment", min_alignment=16
        )
        parsed_arena, arena_by_name, drawer_bytes = _parse_arena_fields(
            arena_fields, f_align
        )
        parsed_side, side_by_name, side_bytes_per_slot = _parse_side_fields(
            side_fields, arena_by_name, s_align
        )

        self._arena_fields = parsed_arena
        self._side_fields = parsed_side
        self._arena_by_name = arena_by_name
        self._side_by_name = side_by_name
        self._field_alignment = f_align
        self._side_alignment = s_align
        self._drawer_bytes = drawer_bytes
        self._side_bytes_per_slot = side_bytes_per_slot

    @property
    def arena_fields(self) -> tuple[ArenaFieldSpec, ...]:
        """Ordered :class:`ArenaFieldSpec` descriptors packed into one drawer."""
        return self._arena_fields

    @property
    def side_fields(self) -> tuple[SideFieldSpec, ...]:
        """Ordered :class:`SideFieldSpec` descriptors backed by :class:`TypedSidePool`."""
        return self._side_fields

    @property
    def arena_field_names(self) -> tuple[str, ...]:
        """Names of all raw ``uint8`` arena fields."""
        return tuple(spec.name for spec in self._arena_fields)

    @property
    def side_field_names(self) -> tuple[str, ...]:
        """Names of all typed side fields."""
        return tuple(spec.name for spec in self._side_fields)

    @property
    def field_alignment(self) -> int:
        """Byte alignment applied to every arena field inside each drawer."""
        return self._field_alignment

    @property
    def side_alignment(self) -> int:
        """Byte alignment requested for every typed side pool."""
        return self._side_alignment

    @property
    def drawer_bytes(self) -> int:
        """Total aligned byte footprint of one ``uint8`` arena drawer."""
        return self._drawer_bytes

    @property
    def side_bytes_per_slot(self) -> int:
        """Total aligned side-pool bytes required per slot across all side fields."""
        return self._side_bytes_per_slot

    def arena_field(self, name: str) -> ArenaFieldSpec:
        """Look up the :class:`ArenaFieldSpec` for ``name``."""
        return self._arena_by_name[name]

    def side_field(self, name: str) -> SideFieldSpec:
        """Look up the :class:`SideFieldSpec` for ``name``."""
        return self._side_by_name[name]


@dataclass(frozen=True, slots=True)
class LowSlotLayout:
    """Layout descriptor for ascending fixed-size slots inside a byte arena."""

    slot_bytes: int
    slot_stride: int
    max_slots: int
    alignment: int

    def slot_offset(self, slot_id: int) -> int:
        """Return the aligned starting byte offset for ``slot_id``."""
        idx = _validate_slot_id(slot_id, max_slots=self.max_slots)
        return idx * self.slot_stride

    def slot_byte_range(self, slot_id: int) -> tuple[int, int]:
        """Return ``[start, end)`` byte bounds owned by ``slot_id``."""
        start = self.slot_offset(slot_id)
        return start, start + self.slot_stride


@dataclass(frozen=True, slots=True)
class HighSlotLayout:
    """Layout descriptor for descending fixed-size slots inside a byte arena."""

    slot_bytes: int
    slot_stride: int
    max_slots: int
    base_offset: int
    alignment: int

    def slot_offset(self, slot_id: int) -> int:
        """Return the aligned starting byte offset for ``slot_id``."""
        idx = _validate_slot_id(slot_id, max_slots=self.max_slots)
        return self.base_offset + idx * self.slot_stride

    def slot_byte_range(self, slot_id: int) -> tuple[int, int]:
        """Return ``[start, end)`` byte bounds owned by ``slot_id``."""
        start = self.slot_offset(slot_id)
        return start, start + self.slot_stride


def _resolve_fixed_slot_counts(
    capacity_bytes: int,
    alignment: int,
    slot_bytes: int,
    max_slots: int | None,
    *,
    end_label: str,
) -> tuple[int, int, int, int]:
    sb = _validate_positive_int(slot_bytes, name="slot_bytes")
    slot_stride = _align_up(sb, alignment)
    aligned_cap = (capacity_bytes // alignment) * alignment
    fit_slots = aligned_cap // slot_stride
    if fit_slots <= 0:
        raise ArenaCapacityError(
            f"Arena capacity {capacity_bytes} cannot hold one aligned "
            f"{slot_stride}-byte {end_label} slot."
        )
    if max_slots is not None:
        resolved_max = _validate_positive_int(max_slots, name="max_slots")
        if resolved_max > fit_slots:
            raise ArenaCapacityError(
                f"Requested {end_label} max_slots={resolved_max} "
                f"({resolved_max * slot_stride} bytes) exceeds arena capacity {aligned_cap}."
            )
    else:
        resolved_max = fit_slots
    return sb, slot_stride, aligned_cap, resolved_max


def _compute_low_slot_layout(
    capacity_bytes: int,
    alignment: int,
    slot_bytes: int,
    *,
    max_slots: int | None = None,
) -> LowSlotLayout:
    sb, slot_stride, _aligned_cap, resolved_max = _resolve_fixed_slot_counts(
        capacity_bytes,
        alignment,
        slot_bytes,
        max_slots,
        end_label="low",
    )
    return LowSlotLayout(
        slot_bytes=sb,
        slot_stride=slot_stride,
        max_slots=resolved_max,
        alignment=alignment,
    )


def _compute_high_slot_layout(
    capacity_bytes: int,
    alignment: int,
    slot_bytes: int,
    *,
    max_slots: int | None = None,
) -> HighSlotLayout:
    sb, slot_stride, aligned_top, resolved_max = _resolve_fixed_slot_counts(
        capacity_bytes,
        alignment,
        slot_bytes,
        max_slots,
        end_label="high",
    )
    base_offset = aligned_top - resolved_max * slot_stride
    return HighSlotLayout(
        slot_bytes=sb,
        slot_stride=slot_stride,
        max_slots=resolved_max,
        base_offset=base_offset,
        alignment=alignment,
    )


@dataclass(slots=True)
class _SlotTracker:
    """O(log N) free-slot tracker with O(1) frontier tracking and ``free_epoch``."""

    allocated: set[int] = field(default_factory=set)
    next_rank: int = 0
    holes: list[int] = field(default_factory=list)
    free_epoch: int = 0

    def _trim_holes(self) -> None:
        while self.holes and (
            self.holes[0] >= self.next_rank or self.holes[0] in self.allocated
        ):
            heapq.heappop(self.holes)

    def peek_lowest_free(
        self,
        max_slots: int,
        *,
        max_rank: int | None = None,
    ) -> int | None:
        self._trim_holes()
        if self.holes:
            cand = self.holes[0]
        elif self.next_rank < max_slots:
            cand = self.next_rank
        else:
            return None
        if max_rank is not None and cand > max_rank:
            return None
        return cand

    def mark_allocated(self, rank: int) -> None:
        if self.holes and self.holes[0] == rank:
            heapq.heappop(self.holes)
        if rank >= self.next_rank:
            self.next_rank = rank + 1
        self.allocated.add(rank)

    def mark_freed(self, rank: int) -> bool:
        if rank not in self.allocated:
            return False
        self.allocated.remove(rank)
        self.free_epoch += 1
        if rank == self.next_rank - 1:
            self.next_rank -= 1
            while self.next_rank > 0 and (self.next_rank - 1) not in self.allocated:
                self.next_rank -= 1
        else:
            heapq.heappush(self.holes, rank)
        return True

    def clear(self) -> None:
        self.allocated.clear()
        self.next_rank = 0
        self.holes.clear()


@dataclass(slots=True)
class _ArenaEnd[LayoutT: (LowSlotLayout, HighSlotLayout)]:
    label: Literal["low", "high"]
    layout: LayoutT | None = None
    tracker: _SlotTracker = field(default_factory=_SlotTracker)
    reclaimer: Callable[[int], None] | None = None
    owner: object | None = None


class Uint8DualArena:
    """Single-dtype ``torch.uint8`` arena shared between low-up and high-down users.

    All tensor views returned by this class preserve ``torch.uint8`` dtype so a
    single ``torch.compile(fullgraph=True)`` graph can read low-end weight
    slots and write high-end KV slots inside the same underlying storage
    without triggering PyTorch AOTAutograd's mixed-dtype mutation assertion.

    The low end supports ascending fixed-size slots (:meth:`configure_low_slots` /
    :meth:`alloc_low_slot`). The high end supports descending fixed-size slots
    allocated from ``max_slots - 1`` down so both ends pack toward each other
    across one frontier. Optional peer reclaim callbacks
    (:meth:`set_low_reclaimer`, :meth:`set_high_reclaimer`) let fixed-slot
    allocations ask the opposite end to release frontier space outside
    ``Uint8DualArena``'s internal lock. Callers own CUDA stream synchronization
    and live view lifetimes before invoking :meth:`close`.
    """

    def __init__(
        self,
        capacity_bytes: int,
        *,
        device: torch.device | str = "cpu",
        slot_bytes: int | None = None,
        high_max_slots: int | None = None,
        low_slot_bytes: int | None = None,
        low_max_slots: int | None = None,
        alignment: int = 256,
        budget: VramBudget | None = None,
        reservation_name: str = "uint8_dual_arena",
    ) -> None:
        """Allocate aligned single-``uint8`` arena storage and optional slot geometries.

        Args:
            capacity_bytes: Logical arena byte size (must be >= ``alignment``).
            device: Target ``cpu`` or ``cuda`` device.
            slot_bytes: Optional descending high-end slot payload size in bytes.
            high_max_slots: Optional cap on descending high-end fixed slots (requires
                ``slot_bytes``).
            low_slot_bytes: Optional ascending low-end fixed drawer size in bytes.
            low_max_slots: Optional cap on ascending low-end fixed slots (requires
                ``low_slot_bytes``).
            alignment: Power-of-two byte alignment >= 16 for base storage and slot strides.
            budget: Optional :class:`VramBudget` reserving the arena's allocated bytes.
            reservation_name: Diagnostic label attached to ``budget`` reservations.
        """
        cap = _validate_positive_int(capacity_bytes, name="capacity_bytes")
        align = _validate_alignment(alignment, name="alignment", min_alignment=16)
        if cap < align:
            raise ValueError(
                f"capacity_bytes ({cap}) must be >= alignment ({align})."
            )
        resolved_device = canonical_device(device)
        if resolved_device.type not in ("cpu", "cuda"):
            raise ValueError(
                f"Uint8DualArena supports cpu or cuda; got {resolved_device}."
            )
        if not isinstance(reservation_name, str) or not reservation_name:
            raise ValueError("reservation_name must be a non-empty string.")
        if high_max_slots is not None and slot_bytes is None:
            raise ValueError("high_max_slots requires slot_bytes.")
        if low_max_slots is not None and low_slot_bytes is None:
            raise ValueError("low_max_slots requires low_slot_bytes.")
        initial_high_layout = (
            _compute_high_slot_layout(cap, align, slot_bytes, max_slots=high_max_slots)
            if slot_bytes is not None
            else None
        )
        initial_low_layout = (
            _compute_low_slot_layout(cap, align, low_slot_bytes, max_slots=low_max_slots)
            if low_slot_bytes is not None
            else None
        )

        raw, storage, base_offset, allocated_bytes, reservation = (
            _allocate_aligned_storage(
                cap,
                dtype=torch.uint8,
                device=resolved_device,
                alignment_bytes=align,
                budget=budget,
                reservation_name=reservation_name,
            )
        )

        alloc_event = _record_cuda_alloc_event(resolved_device, reservation)

        self._lock = threading.Lock()
        self._capacity_bytes = cap
        self._allocated_bytes = allocated_bytes
        self._device = resolved_device
        self._alignment = align
        self._budget: VramBudget | None = budget
        self._reservation: VramReservation | None = reservation
        self._alloc_event: torch.cuda.Event | None = alloc_event
        self._low_end: _ArenaEnd[LowSlotLayout] = _ArenaEnd(
            label="low", layout=initial_low_layout
        )
        self._high_end: _ArenaEnd[HighSlotLayout] = _ArenaEnd(
            label="high", layout=initial_high_layout
        )
        self._raw_storage: torch.Tensor | None = raw
        self._storage: torch.Tensor | None = storage
        self._base_offset = base_offset

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._storage is not None

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    @property
    def allocated_bytes(self) -> int:
        """Total backing bytes allocated (including any alignment padding)."""
        return self._allocated_bytes

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def alignment(self) -> int:
        return self._alignment

    @property
    def budget(self) -> VramBudget | None:
        """Optional :class:`VramBudget` coordinating this arena's reservation."""
        return self._budget

    @property
    def slot_layout(self) -> HighSlotLayout | None:
        """Return configured descending high-slot geometry, or ``None`` if unconfigured."""
        with self._lock:
            return self._high_end.layout

    @property
    def high_slot_layout(self) -> HighSlotLayout | None:
        """Return configured descending high-slot geometry, or ``None`` if unconfigured."""
        return self.slot_layout

    @property
    def low_slot_layout(self) -> LowSlotLayout | None:
        """Return configured ascending low-slot geometry, or ``None`` if unconfigured."""
        with self._lock:
            return self._low_end.layout

    def _low_frontier_unlocked(self) -> int:
        layout = self._low_end.layout
        next_rank = self._low_end.tracker.next_rank
        if layout is None or next_rank == 0:
            return 0
        return next_rank * layout.slot_stride

    def _high_frontier_unlocked(self) -> int:
        layout = self._high_end.layout
        next_rank = self._high_end.tracker.next_rank
        if layout is None or next_rank == 0:
            return self._capacity_bytes
        return layout.base_offset + (layout.max_slots - next_rank) * layout.slot_stride

    @property
    def low_frontier(self) -> int:
        """Lowest byte above which all low-end allocations currently sit below."""
        with self._lock:
            return self._low_frontier_unlocked()

    @property
    def high_frontier(self) -> int:
        """Lowest byte currently occupied by an active high-end slot."""
        with self._lock:
            return self._high_frontier_unlocked()

    @property
    def free_gap_bytes(self) -> int:
        """Contiguous gap between ``low_frontier`` and ``high_frontier``."""
        with self._lock:
            if self._storage is None:
                return 0
            return max(0, self._high_frontier_unlocked() - self._low_frontier_unlocked())

    @property
    def active_low_bytes(self) -> int:
        with self._lock:
            layout = self._low_end.layout
            if layout is None:
                return 0
            return len(self._low_end.tracker.allocated) * layout.slot_stride

    @property
    def active_low_slots(self) -> tuple[int, ...]:
        """Return currently allocated low slot IDs sorted ascending."""
        with self._lock:
            return tuple(sorted(self._low_end.tracker.allocated))

    @property
    def available_low_slots(self) -> int:
        """Return count of unallocated low slots still clear of ``high_frontier``."""
        with self._lock:
            layout = self._low_end.layout
            if self._storage is None or layout is None:
                return 0
            high_frontier = self._high_frontier_unlocked()
            fit = min(layout.max_slots, high_frontier // layout.slot_stride)
            return max(0, fit - len(self._low_end.tracker.allocated))

    @property
    def active_high_slots(self) -> tuple[int, ...]:
        """Return currently allocated slot IDs sorted descending."""
        with self._lock:
            layout = self._high_end.layout
            if layout is None:
                return ()
            max_slots = layout.max_slots
            return tuple(max_slots - 1 - r for r in sorted(self._high_end.tracker.allocated))

    @property
    def available_high_slots(self) -> int:
        """Return count of unallocated high slots still clear of ``low_frontier``."""
        with self._lock:
            layout = self._high_end.layout
            if self._storage is None or layout is None:
                return 0
            low_frontier = self._low_frontier_unlocked()
            if low_frontier <= layout.base_offset:
                min_fit = 0
            else:
                min_fit = (
                    low_frontier - layout.base_offset + layout.slot_stride - 1
                ) // layout.slot_stride
            fit = max(0, layout.max_slots - min_fit)
            return max(0, fit - len(self._high_end.tracker.allocated))

    def _require_storage_unlocked(self) -> torch.Tensor:
        if self._storage is None:
            raise RuntimeError("Uint8DualArena is closed.")
        return self._storage

    def _configure_end_slots[LayoutT: (LowSlotLayout, HighSlotLayout)](
        self,
        end: _ArenaEnd[LayoutT],
        slot_bytes: int,
        max_slots: int | None,
        compute_fn: Callable[..., LayoutT],
    ) -> LayoutT:
        layout = compute_fn(
            self._capacity_bytes,
            self._alignment,
            slot_bytes,
            max_slots=max_slots,
        )
        with self._lock:
            self._require_storage_unlocked()
            if end.owner is not None:
                raise RuntimeError(
                    f"Cannot reconfigure {end.label} slot layout while a {end.label} owner is attached."
                )
            if end.tracker.allocated:
                raise RuntimeError(
                    f"Cannot reconfigure {end.label} slot layout while {end.label} slots are active."
                )
            end.layout = layout
            return layout

    def _claim_end_owner[LayoutT: (LowSlotLayout, HighSlotLayout)](
        self,
        end: _ArenaEnd[LayoutT],
        callback: Callable[[int], None],
        slot_bytes: int,
        max_slots: int | None,
        compute_fn: Callable[..., LayoutT],
    ) -> tuple[LayoutT, bool]:
        if not callable(callback):
            raise TypeError(f"{end.label}_reclaimer must be callable.")
        with self._lock:
            self._require_storage_unlocked()
            if end.owner is not None or end.reclaimer is not None:
                raise RuntimeError(f"A {end.label} frontier reclaimer is already registered.")
            existing = end.layout
            if existing is None:
                layout = compute_fn(
                    self._capacity_bytes,
                    self._alignment,
                    slot_bytes,
                    max_slots=max_slots,
                )
                end.layout = layout
                end.owner = callback
                end.reclaimer = callback
                return layout, True
            if end.tracker.allocated:
                raise RuntimeError(
                    f"Cannot claim {end.label} owner while arena has active {end.label} slots."
                )
            if existing.slot_bytes < slot_bytes:
                raise ValueError(
                    f"Arena {end.label} slot_bytes ({existing.slot_bytes}) is smaller than "
                    f"schema drawer_bytes ({slot_bytes})."
                )
            if max_slots is not None and existing.max_slots != max_slots:
                raise ValueError(
                    f"Arena {end.label} max_slots ({existing.max_slots}) does not match "
                    f"requested max_slots={max_slots}."
                )
            end.owner = callback
            end.reclaimer = callback
            return existing, False

    def _release_end_owner[LayoutT: (LowSlotLayout, HighSlotLayout)](
        self,
        end: _ArenaEnd[LayoutT],
        callback: Callable[[int], None],
        *,
        reset_expected: LayoutT | None = None,
    ) -> None:
        with self._lock:
            if end.owner == callback:
                end.owner = None
            if end.reclaimer == callback:
                end.reclaimer = None
            if (
                self._storage is not None
                and reset_expected is not None
                and end.layout is reset_expected
                and not end.tracker.allocated
                and end.owner is None
                and end.reclaimer is None
            ):
                end.layout = None

    def _set_end_reclaimer[LayoutT: (LowSlotLayout, HighSlotLayout)](
        self,
        end: _ArenaEnd[LayoutT],
        callback: Callable[[int], None] | None,
    ) -> None:
        if callback is not None and not callable(callback):
            raise TypeError(f"{end.label}_reclaimer must be callable or None.")
        with self._lock:
            if callback is not None:
                self._require_storage_unlocked()
            if end.owner is not None:
                raise RuntimeError(
                    f"Cannot modify {end.label} frontier reclaimer while a {end.label} owner is attached."
                )
            if callback is not None and end.reclaimer is not None and end.reclaimer != callback:
                raise RuntimeError(f"A {end.label} frontier reclaimer is already registered.")
            end.reclaimer = callback

    def _clear_end_reclaimer[LayoutT: (LowSlotLayout, HighSlotLayout)](
        self,
        end: _ArenaEnd[LayoutT],
        callback: Callable[[int], None],
    ) -> None:
        with self._lock:
            if end.owner is not None:
                if end.owner == callback:
                    raise RuntimeError(
                        f"Cannot clear {end.label} frontier reclaimer while a {end.label} owner is attached."
                    )
                return
            if end.reclaimer == callback:
                end.reclaimer = None

    def configure_low_slots(
        self,
        slot_bytes: int,
        *,
        max_slots: int | None = None,
    ) -> LowSlotLayout:
        """Configure or replace ascending fixed-size low slots when the low end is empty."""
        return self._configure_end_slots(
            self._low_end, slot_bytes, max_slots, _compute_low_slot_layout
        )

    def _claim_low_owner(
        self,
        callback: Callable[[int], None],
        slot_bytes: int,
        max_slots: int | None,
    ) -> tuple[LowSlotLayout, bool]:
        return self._claim_end_owner(
            self._low_end, callback, slot_bytes, max_slots, _compute_low_slot_layout
        )

    def _release_low_owner(
        self,
        callback: Callable[[int], None],
        *,
        reset_expected: LowSlotLayout | None = None,
    ) -> None:
        self._release_end_owner(self._low_end, callback, reset_expected=reset_expected)

    def configure_high_slots(
        self,
        slot_bytes: int,
        *,
        max_slots: int | None = None,
    ) -> HighSlotLayout:
        """Configure or replace the descending high-end slot geometry when empty."""
        return self._configure_end_slots(
            self._high_end, slot_bytes, max_slots, _compute_high_slot_layout
        )

    def _claim_high_owner(
        self,
        callback: Callable[[int], None],
        slot_bytes: int,
        max_slots: int | None,
    ) -> tuple[HighSlotLayout, bool]:
        return self._claim_end_owner(
            self._high_end, callback, slot_bytes, max_slots, _compute_high_slot_layout
        )

    def _release_high_owner(
        self,
        callback: Callable[[int], None],
        *,
        reset_expected: HighSlotLayout | None = None,
    ) -> None:
        self._release_end_owner(self._high_end, callback, reset_expected=reset_expected)

    def set_low_reclaimer(self, callback: Callable[[int], None] | None) -> None:
        """Register or replace the low-frontier reclaimer invoked outside ``_lock``.

        :meth:`alloc_high_slot` calls ``callback(target_low)`` when high slots
        need ``low_frontier <= target_low``. Callbacks must pass
        ``allow_reclaim=False`` when allocating relocation slots on their own end
        to avoid mutual recursion; non-progress stops reclaim with
        :class:`ArenaCapacityError`. Passing ``None`` clears when no owner is
        attached; use :meth:`clear_low_reclaimer` for identity-checked teardown.
        """
        self._set_end_reclaimer(self._low_end, callback)

    def clear_low_reclaimer(self, callback: Callable[[int], None]) -> None:
        """Clear ``callback`` if it is currently registered as the low frontier reclaimer."""
        self._clear_end_reclaimer(self._low_end, callback)

    def set_high_reclaimer(self, callback: Callable[[int], None] | None) -> None:
        """Register or replace the high-frontier reclaimer invoked outside ``_lock``.

        :meth:`alloc_low_slot` calls ``callback(needed_high)`` when low slots
        need ``high_frontier >= needed_high``. Callbacks must pass
        ``allow_reclaim=False`` when allocating relocation slots on their own end
        to avoid mutual recursion; non-progress stops reclaim with
        :class:`ArenaCapacityError`. Passing ``None`` clears when no owner is
        attached; use :meth:`clear_high_reclaimer` for identity-checked teardown.
        """
        self._set_end_reclaimer(self._high_end, callback)

    def clear_high_reclaimer(self, callback: Callable[[int], None]) -> None:
        """Clear ``callback`` if it is currently registered as the high frontier reclaimer."""
        self._clear_end_reclaimer(self._high_end, callback)

    def raw_uint8_view(self) -> torch.Tensor:
        """Return the underlying 1-D aligned ``torch.uint8`` arena view."""
        with self._lock:
            return self._require_storage_unlocked()

    def alloc_low_slot(
        self,
        *,
        allow_reclaim: bool = True,
        max_slot_id: int | None = None,
    ) -> int:
        """Allocate the lowest free low slot index whose range stays below ``high_frontier``."""
        if max_slot_id is not None and (
            isinstance(max_slot_id, bool) or not isinstance(max_slot_id, int)
        ):
            raise ValueError(f"max_slot_id must be an int or None; got {max_slot_id!r}.")
        prev_high: int | None = None
        prev_free_epoch: int | None = None
        race_retries = 0
        while True:
            reclaimer: Callable[[int], None] | None = None
            needed_high = 0
            with self._lock:
                self._require_storage_unlocked()
                layout = self._low_end.layout
                if layout is None:
                    raise RuntimeError("Low slots are not configured on this Uint8DualArena.")
                high_frontier = self._high_frontier_unlocked()
                max_rank = (
                    None
                    if max_slot_id is None
                    else min(layout.max_slots - 1, max_slot_id)
                )
                first_free = self._low_end.tracker.peek_lowest_free(
                    layout.max_slots, max_rank=max_rank
                )
                if (
                    first_free is not None
                    and (first_free + 1) * layout.slot_stride <= high_frontier
                ):
                    self._low_end.tracker.mark_allocated(first_free)
                    return first_free
                if (
                    not allow_reclaim
                    or first_free is None
                    or self._high_end.reclaimer is None
                ):
                    raise ArenaCapacityError(
                        f"No low slot fits below high_frontier={high_frontier} "
                        f"(slot_stride={layout.slot_stride}, max_slots={layout.max_slots})."
                    )
                needed_high = (first_free + 1) * layout.slot_stride
                if needed_high > self._capacity_bytes:
                    raise ArenaCapacityError(
                        f"No low slot fits below high_frontier={high_frontier} "
                        f"(slot_stride={layout.slot_stride}, max_slots={layout.max_slots})."
                    )
                cur_free_epoch = self._high_end.tracker.free_epoch
                if prev_high is not None:
                    if high_frontier > prev_high:
                        race_retries = 0
                    elif cur_free_epoch != prev_free_epoch:
                        race_retries += 1
                        if race_retries > _MAX_RECLAIM_RACE_RETRIES:
                            raise ArenaCapacityError(
                                f"No low slot fits below high_frontier={high_frontier} "
                                f"(slot_stride={layout.slot_stride}, max_slots={layout.max_slots})."
                            )
                    else:
                        raise ArenaCapacityError(
                            f"No low slot fits below high_frontier={high_frontier} "
                            f"(slot_stride={layout.slot_stride}, max_slots={layout.max_slots})."
                        )
                prev_high = high_frontier
                prev_free_epoch = cur_free_epoch
                reclaimer = self._high_end.reclaimer
            reclaimer(needed_high)

    def free_low_slot(self, slot_id: int) -> None:
        """Return fixed low ``slot_id`` to the arena (idempotent if already free)."""
        with self._lock:
            layout = self._low_end.layout
            if layout is None:
                raise RuntimeError("Low slots are not configured on this Uint8DualArena.")
            layout.slot_offset(slot_id)
            self._low_end.tracker.mark_freed(slot_id)

    def alloc_high_slot(self, *, allow_reclaim: bool = True) -> int:
        """Allocate the highest free slot index whose range stays above ``low_frontier``."""
        prev_low: int | None = None
        prev_free_epoch: int | None = None
        race_retries = 0
        while True:
            reclaimer: Callable[[int], None] | None = None
            target_low = 0
            with self._lock:
                self._require_storage_unlocked()
                layout = self._high_end.layout
                if layout is None:
                    raise RuntimeError("High slots are not configured on this Uint8DualArena.")
                low_frontier = self._low_frontier_unlocked()
                first_rank = self._high_end.tracker.peek_lowest_free(layout.max_slots)
                if first_rank is not None:
                    slot_id = layout.max_slots - 1 - first_rank
                    start, _end = layout.slot_byte_range(slot_id)
                    if start >= low_frontier:
                        self._high_end.tracker.mark_allocated(first_rank)
                        return slot_id
                if (
                    not allow_reclaim
                    or first_rank is None
                    or self._low_end.reclaimer is None
                ):
                    raise ArenaCapacityError(
                        f"No high slot fits above low_frontier={low_frontier} "
                        f"(slot_stride={layout.slot_stride}, max_slots={layout.max_slots})."
                    )
                target_low = layout.slot_offset(layout.max_slots - 1 - first_rank)
                if target_low < 0:
                    raise ArenaCapacityError(
                        f"No high slot fits above low_frontier={low_frontier} "
                        f"(slot_stride={layout.slot_stride}, max_slots={layout.max_slots})."
                    )
                cur_free_epoch = self._low_end.tracker.free_epoch
                if prev_low is not None:
                    if low_frontier < prev_low:
                        race_retries = 0
                    elif cur_free_epoch != prev_free_epoch:
                        race_retries += 1
                        if race_retries > _MAX_RECLAIM_RACE_RETRIES:
                            raise ArenaCapacityError(
                                f"No high slot fits above low_frontier={low_frontier} "
                                f"(slot_stride={layout.slot_stride}, max_slots={layout.max_slots})."
                            )
                    else:
                        raise ArenaCapacityError(
                            f"No high slot fits above low_frontier={low_frontier} "
                            f"(slot_stride={layout.slot_stride}, max_slots={layout.max_slots})."
                        )
                prev_low = low_frontier
                prev_free_epoch = cur_free_epoch
                reclaimer = self._low_end.reclaimer
            reclaimer(target_low)

    def free_high_slot(self, slot_id: int) -> None:
        """Return ``slot_id`` to the high-end pool (idempotent if already free)."""
        with self._lock:
            layout = self._high_end.layout
            if layout is None:
                raise RuntimeError("High slots are not configured on this Uint8DualArena.")
            layout.slot_offset(slot_id)
            self._high_end.tracker.mark_freed(layout.max_slots - 1 - slot_id)

    def _slot_tensor[LayoutT: (LowSlotLayout, HighSlotLayout)](
        self, end: _ArenaEnd[LayoutT], slot_id: int
    ) -> torch.Tensor:
        with self._lock:
            layout = end.layout
            if layout is None:
                raise RuntimeError(
                    f"{end.label.capitalize()} slots are not configured on this Uint8DualArena."
                )
            storage = self._require_storage_unlocked()
            start = layout.slot_offset(slot_id)
            return storage[start : start + layout.slot_bytes]

    def low_slot_tensor(self, slot_id: int) -> torch.Tensor:
        """Return a 1D ``torch.uint8`` view of length ``slot_bytes`` for low ``slot_id``."""
        return self._slot_tensor(self._low_end, slot_id)

    def high_slot_tensor(self, slot_id: int) -> torch.Tensor:
        """Return a 1D ``torch.uint8`` view of length ``slot_bytes`` for ``slot_id``."""
        return self._slot_tensor(self._high_end, slot_id)

    def _strided_slot_pool_view_unlocked(
        self,
        *,
        max_slots: int,
        slot_bytes: int,
        slot_stride: int,
        base_offset: int,
        inner_shape: tuple[int, ...],
        field_offset: int,
        end_label: str,
    ) -> torch.Tensor:
        shape = tuple(inner_shape)
        inner_strides = _contiguous_strides(shape)
        inner_numel = math.prod(shape)
        off = _validate_nonneg_int(field_offset, name="field_offset")
        if off % 16 != 0:
            raise ValueError(
                f"field_offset ({off}) must be a multiple of 16 bytes for alignment."
            )
        self._require_storage_unlocked()
        raw = self._raw_storage
        assert raw is not None
        if off + inner_numel > slot_bytes:
            raise ValueError(
                f"field_offset ({off}) + inner_shape {shape} ({inner_numel} bytes) "
                f"exceeds {end_label} slot_bytes={slot_bytes}."
            )
        return raw.as_strided(
            size=(max_slots, *shape),
            stride=(slot_stride, *inner_strides),
            storage_offset=self._base_offset + base_offset + off,
        )

    def _pool_view[LayoutT: (LowSlotLayout, HighSlotLayout)](
        self,
        end: _ArenaEnd[LayoutT],
        inner_shape: tuple[int, ...],
        *,
        field_offset: int = 0,
    ) -> torch.Tensor:
        with self._lock:
            layout = end.layout
            if layout is None:
                raise RuntimeError(
                    f"{end.label.capitalize()} slots are not configured on this Uint8DualArena."
                )
            base_offset = 0 if isinstance(layout, LowSlotLayout) else layout.base_offset
            return self._strided_slot_pool_view_unlocked(
                max_slots=layout.max_slots,
                slot_bytes=layout.slot_bytes,
                slot_stride=layout.slot_stride,
                base_offset=base_offset,
                inner_shape=inner_shape,
                field_offset=field_offset,
                end_label=end.label,
            )

    def low_pool_view(
        self,
        inner_shape: tuple[int, ...],
        *,
        field_offset: int = 0,
    ) -> torch.Tensor:
        """Return a positive-stride ``uint8`` tensor ``[max_slots, *inner_shape]`` on low slots."""
        return self._pool_view(self._low_end, inner_shape, field_offset=field_offset)

    def high_pool_view(
        self,
        inner_shape: tuple[int, ...],
        *,
        field_offset: int = 0,
    ) -> torch.Tensor:
        """Return a positive-stride ``uint8`` tensor ``[max_slots, *inner_shape]`` on high slots."""
        return self._pool_view(self._high_end, inner_shape, field_offset=field_offset)

    def close(self) -> None:
        """Release storage and reservation (caller must finish CUDA work and drop views first)."""
        with self._lock:
            self._low_end.tracker.clear()
            self._high_end.tracker.clear()
            self._low_end.reclaimer = None
            self._high_end.reclaimer = None
            self._low_end.owner = None
            self._high_end.owner = None
            self._alloc_event = None
            self._raw_storage = None
            self._storage = None
            self._base_offset = 0
            reservation, self._reservation = self._reservation, None
        if reservation is not None:
            reservation.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class TypedSidePool:
    """Dedicated single-dtype strided pool outside the shared ``uint8`` arena.

    Holds tiny non-``uint8`` tensors (such as ``float32`` KV scales/centers/means
    or ``bfloat16``/``float16``/``float8_e4m3fn`` buffers) with positive
    16-byte-aligned (or greater power-of-two) ``stride(0)`` and base pointer so
    AOTAutograd never sees mixed-dtype alias mutations against
    ``Uint8DualArena`` storage.
    """

    def __init__(
        self,
        num_slots: int,
        inner_shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
        stride0: int | None = None,
        alignment_bytes: int = 16,
        budget: VramBudget | None = None,
        reservation_name: str = "typed_side_pool",
    ) -> None:
        slots = _validate_positive_int(num_slots, name="num_slots")
        align_b = _validate_alignment(
            alignment_bytes,
            name="alignment_bytes",
            min_alignment=16,
        )
        resolved_device = canonical_device(device)
        if resolved_device.type not in ("cpu", "cuda"):
            raise ValueError(
                f"TypedSidePool supports cpu or cuda; got {resolved_device}."
            )
        if not isinstance(reservation_name, str) or not reservation_name:
            raise ValueError("reservation_name must be a non-empty string.")
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"dtype must be a torch.dtype; got {dtype!r}.")
        shape = tuple(inner_shape)
        inner_strides = _contiguous_strides(shape)
        inner_numel = math.prod(shape)
        elem_bytes = _element_size(dtype)

        if stride0 is None:
            resolved_stride0 = _aligned_side_stride0(inner_numel, dtype, align_b)
        else:
            resolved_stride0 = _validate_positive_int(stride0, name="stride0")
            if resolved_stride0 < inner_numel:
                raise ValueError(
                    f"stride0 ({resolved_stride0}) must be >= inner_numel ({inner_numel})."
                )
            if (resolved_stride0 * elem_bytes) % align_b != 0:
                raise ValueError(
                    f"stride0 * elem_bytes ({resolved_stride0 * elem_bytes}) must be "
                    f"a multiple of {align_b} bytes."
                )

        nbytes = slots * resolved_stride0 * elem_bytes
        raw_flat, flat, base_offset, allocated_bytes, reservation = (
            _allocate_aligned_storage(
                slots * resolved_stride0,
                dtype=dtype,
                device=resolved_device,
                alignment_bytes=align_b,
                budget=budget,
                reservation_name=reservation_name,
            )
        )

        self._lock = threading.Lock()
        self._num_slots = slots
        self._inner_shape = shape
        self._dtype = dtype
        self._device = resolved_device
        self._stride0 = resolved_stride0
        self._nbytes = nbytes
        self._allocated_bytes = allocated_bytes
        self._reservation: VramReservation | None = reservation
        self._raw_flat: torch.Tensor | None = raw_flat
        self._flat: torch.Tensor | None = flat
        self._base_offset = base_offset
        try:
            self._view: torch.Tensor | None = raw_flat.as_strided(
                size=(slots, *shape),
                stride=(resolved_stride0, *inner_strides),
                storage_offset=base_offset,
            )
        except BaseException:
            self.close()
            raise

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def inner_shape(self) -> tuple[int, ...]:
        return self._inner_shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def stride0(self) -> int:
        return self._stride0

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def allocated_bytes(self) -> int:
        """Total backing bytes allocated (including any alignment padding)."""
        return self._allocated_bytes

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._view is not None

    @property
    def tensor(self) -> torch.Tensor:
        with self._lock:
            if self._view is None:
                raise RuntimeError("TypedSidePool is closed.")
            return self._view

    def slot(self, slot_id: int) -> torch.Tensor:
        """Return the ``inner_shape`` view for ``slot_id``."""
        if isinstance(slot_id, bool) or not isinstance(slot_id, int):
            raise TypeError(f"slot_id must be an int; got {slot_id!r}.")
        if slot_id < 0 or slot_id >= self._num_slots:
            raise IndexError(
                f"slot_id {slot_id} out of range [0, {self._num_slots})."
            )
        return self.tensor[slot_id]

    def close(self) -> None:
        with self._lock:
            self._view = None
            self._flat = None
            self._raw_flat = None
            self._base_offset = 0
            reservation, self._reservation = self._reservation, None
        if reservation is not None:
            reservation.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "ArenaCapacityError",
    "ArenaFieldSpec",
    "HighSlotLayout",
    "LowSlotLayout",
    "SideFieldSpec",
    "SlotSchema",
    "TypedSidePool",
    "Uint8DualArena",
]
