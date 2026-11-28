"""Dual-ended single-dtype byte arena and typed side pools."""

import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Self, cast

import torch

from ._devices import canonical_device
from .vram_budget import VramBudget, VramReservation


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

    Both low-end (for example ``ExpertSlabPool``) and high-end (for example
    fixed-slot KV caches) arena consumers describe per-slot layouts through
    ``SlotSchema`` without hardcoding format-specific field names or coupling
    ``piper-offload`` to kernel math.
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


class ArenaSlice:
    """Variable-size ``torch.uint8`` allocation from the low end of :class:`Uint8DualArena`."""

    def __init__(
        self,
        arena: Uint8DualArena,
        *,
        offset: int,
        size: int,
        aligned_size: int,
        tag: str,
    ) -> None:
        self._arena = arena
        self._offset = offset
        self._size = size
        self._aligned_size = aligned_size
        self._tag = tag
        self._released = False

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def size(self) -> int:
        return self._size

    @property
    def aligned_size(self) -> int:
        return self._aligned_size

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def is_active(self) -> bool:
        return self._arena._is_slice_active(self)

    @property
    def tensor(self) -> torch.Tensor:
        """Return a 1D ``torch.uint8`` view of length ``size`` into the arena."""
        return self._arena._slice_low(self)

    def as_uint8(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Reshape the backing ``uint8`` slice cleanly without dtype reinterpretation."""
        dims = tuple(shape)
        for dim in dims:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
                raise ValueError(
                    f"ArenaSlice shape dimensions must be non-negative ints; got {dims!r}."
                )
        expected = math.prod(dims)
        if expected > self._size:
            raise ValueError(
                f"Requested shape {dims} ({expected} bytes) exceeds slice size {self._size}."
            )
        return self.tensor[:expected].view(dims)

    def release(self) -> None:
        """Return this slice to the arena free list (idempotent)."""
        self._arena.free_low(self)


class Uint8DualArena:
    """Single-dtype ``torch.uint8`` arena shared between low-up and high-down users.

    All tensor views returned by this class preserve ``torch.uint8`` dtype so a
    single ``torch.compile(fullgraph=True)`` graph can read low-end weight
    slices and mutate high-end KV slots inside the same underlying storage
    without triggering PyTorch AOTAutograd's mixed-dtype mutation assertion.

    The low end supports either variable-size slices (:meth:`alloc_low`) or
    symmetric ascending fixed-size slots (:meth:`configure_low_slots` /
    :meth:`alloc_low_slot`). The high end supports descending fixed-size slots
    allocated from ``max_slots - 1`` down so both ends pack toward each other
    across one frontier. Optional peer reclaim callbacks
    (:meth:`set_low_reclaimer`, :meth:`set_high_reclaimer`) let fixed-slot
    allocations (:meth:`alloc_low_slot` and :meth:`alloc_high_slot`; variable
    :meth:`alloc_low` does not invoke peer reclaim) ask the opposite end to
    release frontier space outside ``Uint8DualArena``'s internal lock. Callers
    own CUDA stream synchronization and live view lifetimes before invoking
    :meth:`close`.
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
        self._low_cursor = 0
        self._low_free: list[tuple[int, int]] = []
        self._active_low: dict[int, ArenaSlice] = {}
        self._low_layout: LowSlotLayout | None = initial_low_layout
        self._allocated_low_slots: set[int] = set()
        self._high_layout: HighSlotLayout | None = initial_high_layout
        self._allocated_high_slots: set[int] = set()
        self._low_reclaimer: Callable[[int], None] | None = None
        self._high_reclaimer: Callable[[int], None] | None = None
        self._low_owner: object | None = None
        self._high_owner: object | None = None
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
        with self._lock:
            return self._high_layout

    @property
    def low_slot_layout(self) -> LowSlotLayout | None:
        """Return configured ascending low-slot geometry, or ``None`` if using variable slices."""
        with self._lock:
            return self._low_layout

    def _low_frontier_unlocked(self) -> int:
        if self._low_layout is not None:
            if not self._allocated_low_slots:
                return 0
            return (max(self._allocated_low_slots) + 1) * self._low_layout.slot_stride
        return self._low_cursor

    def _high_frontier_unlocked(self) -> int:
        if not self._allocated_high_slots:
            return self._capacity_bytes
        assert self._high_layout is not None
        min_slot = min(self._allocated_high_slots)
        return self._high_layout.slot_offset(min_slot)

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
            if self._low_layout is not None:
                return len(self._allocated_low_slots) * self._low_layout.slot_stride
            return sum(slc.aligned_size for slc in self._active_low.values())

    @property
    def active_low_slots(self) -> tuple[int, ...]:
        """Return currently allocated low slot IDs sorted ascending."""
        with self._lock:
            return tuple(sorted(self._allocated_low_slots))

    @property
    def available_low_slots(self) -> int:
        """Return count of unallocated low slots still clear of ``high_frontier``."""
        with self._lock:
            if self._storage is None or self._low_layout is None:
                return 0
            layout = self._low_layout
            high_frontier = self._high_frontier_unlocked()
            return sum(
                1
                for slot_id in range(layout.max_slots)
                if slot_id not in self._allocated_low_slots
                and (slot_id + 1) * layout.slot_stride <= high_frontier
            )

    @property
    def active_high_slots(self) -> tuple[int, ...]:
        """Return currently allocated slot IDs sorted descending."""
        with self._lock:
            return tuple(sorted(self._allocated_high_slots, reverse=True))

    @property
    def available_high_slots(self) -> int:
        """Return count of unallocated high slots still clear of ``low_frontier``."""
        with self._lock:
            if self._storage is None or self._high_layout is None:
                return 0
            low_frontier = self._low_frontier_unlocked()
            return sum(
                1
                for slot_id in range(self._high_layout.max_slots)
                if slot_id not in self._allocated_high_slots
                and self._high_layout.slot_offset(slot_id) >= low_frontier
            )

    def _require_storage_unlocked(self) -> torch.Tensor:
        if self._storage is None:
            raise RuntimeError("Uint8DualArena is closed.")
        return self._storage

    def _is_slice_active(self, arena_slice: ArenaSlice) -> bool:
        if arena_slice._arena is not self:
            return False
        with self._lock:
            return not arena_slice._released and self._storage is not None

    def _slice_low(self, arena_slice: ArenaSlice) -> torch.Tensor:
        if arena_slice._arena is not self:
            raise ValueError("ArenaSlice does not belong to this Uint8DualArena.")
        with self._lock:
            if arena_slice._released:
                raise RuntimeError("ArenaSlice has been released.")
            storage = self._require_storage_unlocked()
            return storage[arena_slice._offset : arena_slice._offset + arena_slice._size]

    def configure_low_slots(
        self,
        slot_bytes: int,
        *,
        max_slots: int | None = None,
    ) -> LowSlotLayout:
        """Configure or replace ascending fixed-size low slots when the low end is empty."""
        layout = _compute_low_slot_layout(
            self._capacity_bytes,
            self._alignment,
            slot_bytes,
            max_slots=max_slots,
        )
        with self._lock:
            self._require_storage_unlocked()
            if self._low_owner is not None:
                raise RuntimeError(
                    "Cannot reconfigure low slot layout while a low owner is attached."
                )
            if self._active_low or self._allocated_low_slots:
                raise RuntimeError(
                    "Cannot reconfigure low slot layout while low allocations are active."
                )
            self._low_layout = layout
            return layout

    def _claim_low_owner(
        self,
        callback: Callable[[int], None],
        slot_bytes: int,
        max_slots: int | None,
    ) -> tuple[LowSlotLayout, bool]:
        if not callable(callback):
            raise TypeError("low_reclaimer must be callable.")
        with self._lock:
            self._require_storage_unlocked()
            if self._low_owner is not None or self._low_reclaimer is not None:
                raise RuntimeError("A low frontier reclaimer is already registered.")
            existing = self._low_layout
            if existing is None:
                if self._active_low:
                    raise RuntimeError(
                        "Cannot configure fixed low slots while low allocations are active."
                    )
                layout = _compute_low_slot_layout(
                    self._capacity_bytes,
                    self._alignment,
                    slot_bytes,
                    max_slots=max_slots,
                )
                self._low_layout = layout
                self._low_owner = callback
                self._low_reclaimer = callback
                return layout, True
            if self._allocated_low_slots:
                raise RuntimeError("Cannot claim low owner while arena has active low slots.")
            if existing.slot_bytes < slot_bytes:
                raise ValueError(
                    f"Arena low slot_bytes ({existing.slot_bytes}) is smaller than "
                    f"schema drawer_bytes ({slot_bytes})."
                )
            if max_slots is not None and existing.max_slots != max_slots:
                raise ValueError(
                    f"Arena low max_slots ({existing.max_slots}) does not match "
                    f"requested max_slots={max_slots}."
                )
            self._low_owner = callback
            self._low_reclaimer = callback
            return existing, False

    def _release_low_owner(
        self,
        callback: Callable[[int], None],
        *,
        reset_expected: LowSlotLayout | None = None,
    ) -> None:
        with self._lock:
            if self._low_owner == callback:
                self._low_owner = None
            if self._low_reclaimer == callback:
                self._low_reclaimer = None
            if (
                self._storage is not None
                and reset_expected is not None
                and self._low_layout is reset_expected
                and not self._active_low
                and not self._allocated_low_slots
                and self._low_owner is None
                and self._low_reclaimer is None
            ):
                self._low_layout = None

    def configure_high_slots(
        self,
        slot_bytes: int,
        *,
        max_slots: int | None = None,
    ) -> HighSlotLayout:
        """Configure or replace the descending high-end slot geometry when empty."""
        layout = _compute_high_slot_layout(
            self._capacity_bytes,
            self._alignment,
            slot_bytes,
            max_slots=max_slots,
        )
        with self._lock:
            self._require_storage_unlocked()
            if self._high_owner is not None:
                raise RuntimeError(
                    "Cannot reconfigure high slot layout while a high owner is attached."
                )
            if self._allocated_high_slots:
                raise RuntimeError(
                    "Cannot reconfigure high slot layout while high slots are allocated."
                )
            self._high_layout = layout
            return layout

    def _claim_high_owner(
        self,
        callback: Callable[[int], None],
        slot_bytes: int,
        max_slots: int | None,
    ) -> tuple[HighSlotLayout, bool]:
        if not callable(callback):
            raise TypeError("high_reclaimer must be callable.")
        with self._lock:
            self._require_storage_unlocked()
            if self._high_owner is not None or self._high_reclaimer is not None:
                raise RuntimeError("A high frontier reclaimer is already registered.")
            existing = self._high_layout
            if existing is None:
                layout = _compute_high_slot_layout(
                    self._capacity_bytes,
                    self._alignment,
                    slot_bytes,
                    max_slots=max_slots,
                )
                self._high_layout = layout
                self._high_owner = callback
                self._high_reclaimer = callback
                return layout, True
            if self._allocated_high_slots:
                raise RuntimeError("Cannot claim high owner while arena has active high slots.")
            if existing.slot_bytes < slot_bytes:
                raise ValueError(
                    f"Arena high slot_bytes ({existing.slot_bytes}) is smaller than "
                    f"schema drawer_bytes ({slot_bytes})."
                )
            if max_slots is not None and existing.max_slots != max_slots:
                raise ValueError(
                    f"Arena high max_slots ({existing.max_slots}) does not match "
                    f"requested max_slots={max_slots}."
                )
            self._high_owner = callback
            self._high_reclaimer = callback
            return existing, False

    def _release_high_owner(
        self,
        callback: Callable[[int], None],
        *,
        reset_expected: HighSlotLayout | None = None,
    ) -> None:
        with self._lock:
            if self._high_owner == callback:
                self._high_owner = None
            if self._high_reclaimer == callback:
                self._high_reclaimer = None
            if (
                self._storage is not None
                and reset_expected is not None
                and self._high_layout is reset_expected
                and not self._allocated_high_slots
                and self._high_owner is None
                and self._high_reclaimer is None
            ):
                self._high_layout = None

    def set_low_reclaimer(self, callback: Callable[[int], None] | None) -> None:
        """Register or replace the low-frontier reclaimer invoked outside ``_lock``.

        :meth:`alloc_high_slot` calls ``callback(target_low)`` when high slots
        need ``low_frontier <= target_low``. Callbacks must pass
        ``allow_reclaim=False`` when allocating relocation slots on their own end
        to avoid mutual recursion; non-progress stops reclaim with
        :class:`ArenaCapacityError`. Passing ``None`` clears unconditionally; use
        :meth:`clear_low_reclaimer` for identity-checked teardown.
        """
        if callback is not None and not callable(callback):
            raise TypeError("low_reclaimer must be callable or None.")
        with self._lock:
            if callback is not None:
                self._require_storage_unlocked()
            if (
                callback is not None
                and self._low_reclaimer is not None
                and self._low_reclaimer != callback
            ):
                raise RuntimeError("A low frontier reclaimer is already registered.")
            self._low_reclaimer = callback

    def clear_low_reclaimer(self, callback: Callable[[int], None]) -> None:
        """Clear ``callback`` if it is currently registered as the low frontier reclaimer."""
        with self._lock:
            if self._low_reclaimer == callback:
                self._low_reclaimer = None

    def set_high_reclaimer(self, callback: Callable[[int], None] | None) -> None:
        """Register or replace the high-frontier reclaimer invoked outside ``_lock``.

        :meth:`alloc_low_slot` calls ``callback(needed_high)`` when low slots
        need ``high_frontier >= needed_high``. Callbacks must pass
        ``allow_reclaim=False`` when allocating relocation slots on their own end
        to avoid mutual recursion; non-progress stops reclaim with
        :class:`ArenaCapacityError`. Passing ``None`` clears unconditionally; use
        :meth:`clear_high_reclaimer` for identity-checked teardown.
        """
        if callback is not None and not callable(callback):
            raise TypeError("high_reclaimer must be callable or None.")
        with self._lock:
            if callback is not None:
                self._require_storage_unlocked()
            if (
                callback is not None
                and self._high_reclaimer is not None
                and self._high_reclaimer != callback
            ):
                raise RuntimeError("A high frontier reclaimer is already registered.")
            self._high_reclaimer = callback

    def clear_high_reclaimer(self, callback: Callable[[int], None]) -> None:
        """Clear ``callback`` if it is currently registered as the high frontier reclaimer."""
        with self._lock:
            if self._high_reclaimer == callback:
                self._high_reclaimer = None

    def raw_uint8_view(self) -> torch.Tensor:
        """Return the underlying 1-D aligned ``torch.uint8`` arena view."""
        with self._lock:
            return self._require_storage_unlocked()

    def alloc_low(self, nbytes: int, *, tag: str = "") -> ArenaSlice:
        """Allocate ``nbytes`` (aligned up) from the low end with free-list reuse."""
        req = _validate_positive_int(nbytes, name="nbytes")
        aligned = _align_up(req, self._alignment)
        with self._lock:
            self._require_storage_unlocked()
            if self._low_layout is not None:
                raise RuntimeError(
                    "Cannot use variable alloc_low() when fixed low slots are configured."
                )
            for idx, (hole_off, hole_size) in enumerate(self._low_free):
                if hole_size >= aligned:
                    if hole_size == aligned:
                        del self._low_free[idx]
                    else:
                        self._low_free[idx] = (hole_off + aligned, hole_size - aligned)
                    slc = ArenaSlice(
                        self, offset=hole_off, size=req, aligned_size=aligned, tag=tag
                    )
                    self._active_low[hole_off] = slc
                    return slc

            high_frontier = self._high_frontier_unlocked()
            if self._low_cursor + aligned > high_frontier:
                gap = max(0, high_frontier - self._low_cursor)
                raise ArenaCapacityError(
                    f"Low allocation of {aligned} bytes exceeds free gap {gap} "
                    f"(low_frontier={self._low_cursor}, high_frontier={high_frontier})."
                )
            offset = self._low_cursor
            self._low_cursor += aligned
            slc = ArenaSlice(self, offset=offset, size=req, aligned_size=aligned, tag=tag)
            self._active_low[offset] = slc
            return slc

    def free_low(self, arena_slice: ArenaSlice) -> None:
        """Release ``arena_slice`` back to the low end and coalesce free space."""
        if arena_slice._arena is not self:
            raise ValueError("ArenaSlice does not belong to this Uint8DualArena.")
        with self._lock:
            if arena_slice._released:
                return
            arena_slice._released = True
            self._active_low.pop(arena_slice.offset, None)

            free_list = self._low_free
            free_list.append((arena_slice.offset, arena_slice.aligned_size))
            free_list.sort(key=lambda item: item[0])

            merged: list[tuple[int, int]] = []
            for off, size in free_list:
                if merged and merged[-1][0] + merged[-1][1] == off:
                    merged[-1] = (merged[-1][0], merged[-1][1] + size)
                else:
                    merged.append((off, size))
            if merged and merged[-1][0] + merged[-1][1] == self._low_cursor:
                self._low_cursor = merged[-1][0]
                merged.pop()
            self._low_free = merged

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
        while True:
            reclaimer: Callable[[int], None] | None = None
            needed_high = 0
            with self._lock:
                self._require_storage_unlocked()
                if self._low_layout is None:
                    raise RuntimeError("Low slots are not configured on this Uint8DualArena.")
                layout = self._low_layout
                high_frontier = self._high_frontier_unlocked()
                upper = (
                    layout.max_slots
                    if max_slot_id is None
                    else max(0, min(layout.max_slots, max_slot_id + 1))
                )
                first_free: int | None = None
                for slot_id in range(upper):
                    if slot_id in self._allocated_low_slots:
                        continue
                    if first_free is None:
                        first_free = slot_id
                    if (slot_id + 1) * layout.slot_stride <= high_frontier:
                        self._allocated_low_slots.add(slot_id)
                        return slot_id
                    break
                if (
                    not allow_reclaim
                    or first_free is None
                    or self._high_reclaimer is None
                ):
                    raise ArenaCapacityError(
                        f"No low slot fits below high_frontier={high_frontier} "
                        f"(slot_stride={layout.slot_stride}, max_slots={layout.max_slots})."
                    )
                needed_high = (first_free + 1) * layout.slot_stride
                if needed_high > self._capacity_bytes or (
                    prev_high is not None and high_frontier <= prev_high
                ):
                    raise ArenaCapacityError(
                        f"No low slot fits below high_frontier={high_frontier} "
                        f"(slot_stride={layout.slot_stride}, max_slots={layout.max_slots})."
                    )
                prev_high = high_frontier
                reclaimer = self._high_reclaimer
            reclaimer(needed_high)

    def free_low_slot(self, slot_id: int) -> None:
        """Return fixed low ``slot_id`` to the arena (idempotent if already free)."""
        with self._lock:
            if self._low_layout is None:
                raise RuntimeError("Low slots are not configured on this Uint8DualArena.")
            self._low_layout.slot_offset(slot_id)
            self._allocated_low_slots.discard(slot_id)

    def low_slot_tensor(self, slot_id: int) -> torch.Tensor:
        """Return a 1D ``torch.uint8`` view of length ``slot_bytes`` for low ``slot_id``."""
        with self._lock:
            if self._low_layout is None:
                raise RuntimeError("Low slots are not configured on this Uint8DualArena.")
            storage = self._require_storage_unlocked()
            start = self._low_layout.slot_offset(slot_id)
            return storage[start : start + self._low_layout.slot_bytes]

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

    def low_pool_view(
        self,
        inner_shape: tuple[int, ...],
        *,
        field_offset: int = 0,
    ) -> torch.Tensor:
        """Return a positive-stride ``uint8`` tensor ``[max_slots, *inner_shape]`` on low slots."""
        with self._lock:
            if self._low_layout is None:
                raise RuntimeError("Low slots are not configured on this Uint8DualArena.")
            layout = self._low_layout
            return self._strided_slot_pool_view_unlocked(
                max_slots=layout.max_slots,
                slot_bytes=layout.slot_bytes,
                slot_stride=layout.slot_stride,
                base_offset=0,
                inner_shape=inner_shape,
                field_offset=field_offset,
                end_label="low",
            )

    def alloc_high_slot(
        self,
        *,
        allow_reclaim: bool = True,
        min_slot_id: int | None = None,
    ) -> int:
        """Allocate the highest free slot index whose range stays above ``low_frontier``."""
        if min_slot_id is not None and (
            isinstance(min_slot_id, bool) or not isinstance(min_slot_id, int) or min_slot_id < 0
        ):
            raise ValueError(
                f"min_slot_id must be a non-negative int or None; got {min_slot_id!r}."
            )
        prev_low: int | None = None
        while True:
            reclaimer: Callable[[int], None] | None = None
            target_low = 0
            with self._lock:
                self._require_storage_unlocked()
                if self._high_layout is None:
                    raise RuntimeError("High slots are not configured on this Uint8DualArena.")
                layout = self._high_layout
                low_frontier = self._low_frontier_unlocked()
                lower = 0 if min_slot_id is None else min(layout.max_slots, min_slot_id)
                first_free: int | None = None
                for slot_id in range(layout.max_slots - 1, lower - 1, -1):
                    if slot_id in self._allocated_high_slots:
                        continue
                    if first_free is None:
                        first_free = slot_id
                    start, _end = layout.slot_byte_range(slot_id)
                    if start < low_frontier:
                        break
                    self._allocated_high_slots.add(slot_id)
                    return slot_id
                if (
                    not allow_reclaim
                    or first_free is None
                    or self._low_reclaimer is None
                ):
                    raise ArenaCapacityError(
                        f"No high slot fits above low_frontier={low_frontier} "
                        f"(slot_stride={layout.slot_stride}, max_slots={layout.max_slots})."
                    )
                target_low = layout.slot_offset(first_free)
                if target_low < 0 or (prev_low is not None and low_frontier >= prev_low):
                    raise ArenaCapacityError(
                        f"No high slot fits above low_frontier={low_frontier} "
                        f"(slot_stride={layout.slot_stride}, max_slots={layout.max_slots})."
                    )
                prev_low = low_frontier
                reclaimer = self._low_reclaimer
            reclaimer(target_low)

    def free_high_slot(self, slot_id: int) -> None:
        """Return ``slot_id`` to the high-end pool (idempotent if already free)."""
        with self._lock:
            if self._high_layout is None:
                raise RuntimeError("High slots are not configured on this Uint8DualArena.")
            self._high_layout.slot_offset(slot_id)
            self._allocated_high_slots.discard(slot_id)

    def high_slot_tensor(self, slot_id: int) -> torch.Tensor:
        """Return a 1D ``torch.uint8`` view of length ``slot_bytes`` for ``slot_id``."""
        with self._lock:
            if self._high_layout is None:
                raise RuntimeError("High slots are not configured on this Uint8DualArena.")
            storage = self._require_storage_unlocked()
            start = self._high_layout.slot_offset(slot_id)
            return storage[start : start + self._high_layout.slot_bytes]

    def high_pool_view(
        self,
        inner_shape: tuple[int, ...],
        *,
        field_offset: int = 0,
    ) -> torch.Tensor:
        """Return a positive-stride ``uint8`` tensor ``[max_slots, *inner_shape]`` on high slots."""
        with self._lock:
            if self._high_layout is None:
                raise RuntimeError("High slots are not configured on this Uint8DualArena.")
            layout = self._high_layout
            return self._strided_slot_pool_view_unlocked(
                max_slots=layout.max_slots,
                slot_bytes=layout.slot_bytes,
                slot_stride=layout.slot_stride,
                base_offset=layout.base_offset,
                inner_shape=inner_shape,
                field_offset=field_offset,
                end_label="high",
            )

    def interleaved_kv_views(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        chunk_tokens: int = 1024,
        tail_subblock: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-copy ``uint8`` ``(k, v)`` views compatible with ``piper-kernels``.

        ``k`` has shape ``[max_slots, num_kv_heads, chunk_tokens, head_dim]`` and
        ``v`` has shape ``[max_slots, num_kv_heads, chunk_tokens // tail_subblock,
        head_dim, tail_subblock]``. Both carry ``stride(0) == slot_stride > 0``,
        16-byte aligned ``storage_offset()``, and ``torch.uint8`` dtype so
        mutations stay AOTAutograd-safe alongside low-end ``uint8`` slices.
        """
        h_kv = _validate_positive_int(num_kv_heads, name="num_kv_heads")
        d = _validate_positive_int(head_dim, name="head_dim")
        c_tok = _validate_positive_int(chunk_tokens, name="chunk_tokens")
        sub = _validate_positive_int(tail_subblock, name="tail_subblock")
        if c_tok % sub != 0:
            raise ValueError(
                f"chunk_tokens ({c_tok}) must be divisible by tail_subblock ({sub})."
            )
        half_bytes = h_kv * c_tok * d
        if half_bytes % 16 != 0:
            raise ValueError(
                "Each K/V half byte size (num_kv_heads * chunk_tokens * head_dim) "
                f"must be a multiple of 16 for 16-byte storage_offset alignment; "
                f"got {half_bytes}."
            )
        k_inner_shape = (h_kv, c_tok, d)
        v_inner_shape = (h_kv, c_tok // sub, d, sub)
        k_inner_strides = _contiguous_strides(k_inner_shape)
        v_inner_strides = _contiguous_strides(v_inner_shape)
        total_bytes = 2 * half_bytes
        with self._lock:
            if self._high_layout is None:
                raise RuntimeError("High slots are not configured on this Uint8DualArena.")
            self._require_storage_unlocked()
            raw = self._raw_storage
            assert raw is not None
            layout = self._high_layout
            if total_bytes > layout.slot_bytes:
                raise ValueError(
                    f"interleaved KV bytes ({total_bytes}) exceeds slot_bytes={layout.slot_bytes}."
                )
            base = self._base_offset + layout.base_offset
            k = raw.as_strided(
                size=(layout.max_slots, *k_inner_shape),
                stride=(layout.slot_stride, *k_inner_strides),
                storage_offset=base,
            )
            v = raw.as_strided(
                size=(layout.max_slots, *v_inner_shape),
                stride=(layout.slot_stride, *v_inner_strides),
                storage_offset=base + half_bytes,
            )
            return k, v

    def _overlapping_low_unlocked(
        self,
        start_byte: int,
        end_byte: int,
    ) -> tuple[ArenaSlice, ...]:
        hits = [
            slc
            for slc in self._active_low.values()
            if slc.offset < end_byte and (slc.offset + slc.aligned_size) > start_byte
        ]
        hits.sort(key=lambda slc: slc.offset)
        return tuple(hits)

    def overlapping_low_slices(self, slot_id: int) -> tuple[ArenaSlice, ...]:
        """Return active variable :meth:`alloc_low` slices overlapping high ``slot_id``."""
        with self._lock:
            if self._high_layout is None:
                raise RuntimeError("High slots are not configured on this Uint8DualArena.")
            slot_start, slot_end = self._high_layout.slot_byte_range(slot_id)
            return self._overlapping_low_unlocked(slot_start, slot_end)

    def blocking_low_slices(self, slot_id: int) -> tuple[ArenaSlice, ...]:
        """Return active variable :meth:`alloc_low` slices pinning ``low_frontier`` above ``slot_id``.

        Unlike :meth:`overlapping_low_slices`, this inspects variable slices up
        to the variable low cursor whose eviction is required to shrink
        ``low_frontier`` below high ``slot_id``'s start byte.
        """
        with self._lock:
            if self._high_layout is None:
                raise RuntimeError("High slots are not configured on this Uint8DualArena.")
            slot_start, _slot_end = self._high_layout.slot_byte_range(slot_id)
            if self._low_cursor <= slot_start:
                return ()
            return self._overlapping_low_unlocked(slot_start, self._low_cursor)

    def overlapping_low_slices_in_range(
        self,
        start_byte: int,
        end_byte: int,
    ) -> tuple[ArenaSlice, ...]:
        """Return active variable :meth:`alloc_low` slices intersecting ``[start_byte, end_byte)``."""
        start = _validate_nonneg_int(start_byte, name="start_byte")
        end = _validate_nonneg_int(end_byte, name="end_byte")
        if end < start:
            raise ValueError(f"end_byte ({end}) must be >= start_byte ({start}).")
        if start == end:
            return ()
        with self._lock:
            return self._overlapping_low_unlocked(start, end)

    def close(self) -> None:
        """Release storage and reservation (caller must finish CUDA work and drop views first)."""
        with self._lock:
            for slc in list(self._active_low.values()):
                slc._released = True
            self._active_low.clear()
            self._low_free.clear()
            self._low_cursor = 0
            self._allocated_low_slots.clear()
            self._allocated_high_slots.clear()
            self._alloc_event = None
            self._low_reclaimer = None
            self._high_reclaimer = None
            self._low_owner = None
            self._high_owner = None
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

    @classmethod
    def for_kv_metadata(
        cls,
        num_slots: int,
        *,
        key_shape: tuple[int, ...],
        value_shape: tuple[int, ...] | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        alignment_bytes: int = 16,
        budget: VramBudget | None = None,
        reservation_name: str = "kv_metadata_pool",
    ) -> KvMetadataPool:
        """Allocate one aligned flat buffer for the 4 KV chunk stat tensors."""
        return KvMetadataPool(
            num_slots,
            key_shape=key_shape,
            value_shape=value_shape,
            dtype=dtype,
            device=device,
            alignment_bytes=alignment_bytes,
            budget=budget,
            reservation_name=reservation_name,
        )

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


class StructuredSidePool:
    """Dedicated single-dtype strided pool packing arbitrary named fields per slot.

    Each named field maintains contiguous inner dimensions, strictly positive
    ``stride(0)`` across slots, and ``alignment_bytes`` alignment on both
    ``stride(0)`` and every field view's pointer / relative ``storage_offset()``.
    """

    def __init__(
        self,
        num_slots: int,
        fields: Mapping[str, tuple[int, ...]],
        *,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
        alignment_bytes: int = 16,
        budget: VramBudget | None = None,
        reservation_name: str = "structured_side_pool",
    ) -> None:
        if not fields:
            raise ValueError("StructuredSidePool requires at least one named field.")
        slots = _validate_positive_int(num_slots, name="num_slots")
        align_b = _validate_alignment(
            alignment_bytes,
            name="alignment_bytes",
            min_alignment=16,
        )
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"dtype must be a torch.dtype; got {dtype!r}.")

        offset = 0
        field_specs: list[tuple[str, tuple[int, ...], tuple[int, ...], int]] = []
        for name, raw_shape in fields.items():
            if not isinstance(name, str) or not name:
                raise ValueError("StructuredSidePool field names must be non-empty strings.")
            shape = tuple(raw_shape)
            strides = _contiguous_strides(shape)
            numel = math.prod(shape)
            field_stride = _aligned_side_stride0(numel, dtype, align_b)
            field_specs.append((name, shape, strides, offset))
            offset += field_stride
        slot_stride = offset

        self._lock = threading.Lock()
        self._pool = TypedSidePool(
            slots,
            (slot_stride,),
            dtype=dtype,
            device=device,
            stride0=slot_stride,
            alignment_bytes=align_b,
            budget=budget,
            reservation_name=reservation_name,
        )
        self._field_names = tuple(name for name, _, _, _ in field_specs)
        try:
            raw_flat = self._pool._raw_flat
            assert raw_flat is not None
            base_offset = self._pool._base_offset
            self._fields: dict[str, torch.Tensor] | None = {
                name: raw_flat.as_strided(
                    size=(slots, *shape),
                    stride=(slot_stride, *strides),
                    storage_offset=base_offset + f_offset,
                )
                for name, shape, strides, f_offset in field_specs
            }
        except BaseException:
            self._pool.close()
            raise

    @property
    def pool(self) -> TypedSidePool:
        return self._pool

    @property
    def field_names(self) -> tuple[str, ...]:
        return self._field_names

    @property
    def num_slots(self) -> int:
        return self._pool.num_slots

    @property
    def dtype(self) -> torch.dtype:
        return self._pool.dtype

    @property
    def device(self) -> torch.device:
        return self._pool.device

    @property
    def stride0(self) -> int:
        return self._pool.stride0

    @property
    def nbytes(self) -> int:
        return self._pool.nbytes

    @property
    def allocated_bytes(self) -> int:
        return self._pool.allocated_bytes

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._fields is not None and self._pool.is_open

    def field(self, name: str) -> torch.Tensor:
        """Return the ``[num_slots, *field_shape]`` strided view for ``name``."""
        with self._lock:
            if self._fields is None or not self._pool.is_open:
                raise RuntimeError("StructuredSidePool is closed.")
            return self._fields[name]

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.field(name)

    def close(self) -> None:
        with self._lock:
            self._fields = None
        self._pool.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class KvMetadataPool:
    """Thin :class:`StructuredSidePool` view bundle for ``(k_scale, k_center, v_scale, v_mean)``.

    Supports both equal and heterogeneous ``key_shape`` / ``value_shape``
    (e.g., ``key_shape=(H_kv, D)`` and ``value_shape=(H_kv, 1024)``) while
    preserving positive ``stride(0)`` and ``alignment_bytes`` alignment on both
    ``stride(0)`` and every view's base pointer.
    """

    def __init__(
        self,
        num_slots: int,
        *,
        key_shape: tuple[int, ...],
        value_shape: tuple[int, ...] | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        alignment_bytes: int = 16,
        budget: VramBudget | None = None,
        reservation_name: str = "kv_metadata_pool",
    ) -> None:
        resolved_key = tuple(key_shape)
        resolved_val = tuple(resolved_key if value_shape is None else value_shape)
        self._structured = StructuredSidePool(
            num_slots,
            {
                "key_scale": resolved_key,
                "key_center": resolved_key,
                "value_scale": resolved_val,
                "value_mean": resolved_val,
            },
            dtype=dtype,
            device=device,
            alignment_bytes=alignment_bytes,
            budget=budget,
            reservation_name=reservation_name,
        )
        self._key_shape = resolved_key
        self._value_shape = resolved_val

    @property
    def pool(self) -> TypedSidePool:
        return self._structured.pool

    @property
    def structured_pool(self) -> StructuredSidePool:
        return self._structured

    @property
    def num_slots(self) -> int:
        return self._structured.num_slots

    @property
    def key_shape(self) -> tuple[int, ...]:
        return self._key_shape

    @property
    def value_shape(self) -> tuple[int, ...]:
        return self._value_shape

    @property
    def dtype(self) -> torch.dtype:
        return self._structured.dtype

    @property
    def device(self) -> torch.device:
        return self._structured.device

    @property
    def stride0(self) -> int:
        return self._structured.stride0

    @property
    def nbytes(self) -> int:
        return self._structured.nbytes

    @property
    def allocated_bytes(self) -> int:
        return self._structured.allocated_bytes

    @property
    def is_open(self) -> bool:
        return self._structured.is_open

    def _get_field(self, name: str) -> torch.Tensor:
        if not self._structured.is_open:
            raise RuntimeError("KvMetadataPool is closed.")
        return self._structured.field(name)

    @property
    def key_scale(self) -> torch.Tensor:
        return self._get_field("key_scale")

    @property
    def key_center(self) -> torch.Tensor:
        return self._get_field("key_center")

    @property
    def value_scale(self) -> torch.Tensor:
        return self._get_field("value_scale")

    @property
    def value_mean(self) -> torch.Tensor:
        return self._get_field("value_mean")

    def close(self) -> None:
        self._structured.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "ArenaCapacityError",
    "ArenaFieldSpec",
    "ArenaSlice",
    "HighSlotLayout",
    "KvMetadataPool",
    "LowSlotLayout",
    "SideFieldSpec",
    "SlotSchema",
    "StructuredSidePool",
    "TypedSidePool",
    "Uint8DualArena",
]
