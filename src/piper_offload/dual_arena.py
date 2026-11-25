"""Dual-ended single-dtype byte arena and typed side pools."""

import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

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
    elem_bytes = torch.empty((), dtype=dtype).element_size()
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
        if isinstance(slot_id, bool) or not isinstance(slot_id, int):
            raise TypeError(f"slot_id must be an int; got {slot_id!r}.")
        if slot_id < 0 or slot_id >= self.max_slots:
            raise IndexError(
                f"slot_id {slot_id} out of range [0, {self.max_slots})."
            )
        return self.base_offset + slot_id * self.slot_stride

    def slot_byte_range(self, slot_id: int) -> tuple[int, int]:
        """Return ``[start, end)`` byte bounds owned by ``slot_id``."""
        start = self.slot_offset(slot_id)
        return start, start + self.slot_stride


def _compute_high_slot_layout(
    capacity_bytes: int,
    alignment: int,
    slot_bytes: int,
) -> HighSlotLayout:
    sb = _validate_positive_int(slot_bytes, name="slot_bytes")
    slot_stride = _align_up(sb, alignment)
    aligned_top = (capacity_bytes // alignment) * alignment
    max_slots = aligned_top // slot_stride
    if max_slots <= 0:
        raise ArenaCapacityError(
            f"Arena capacity {capacity_bytes} cannot hold one aligned "
            f"{slot_stride}-byte high slot."
        )
    base_offset = aligned_top - max_slots * slot_stride
    return HighSlotLayout(
        slot_bytes=sb,
        slot_stride=slot_stride,
        max_slots=max_slots,
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

    High-end slots are addressed by non-overlapping indices
    ``[0 .. max_slots - 1]`` with strictly positive, ``alignment``-aligned
    ``stride(0)`` and allocated **descending** from ``max_slots - 1`` down so
    the two ends pack toward each other cleanly. Callers own CUDA stream
    synchronization and live view lifetimes before invoking :meth:`close`.
    """

    def __init__(
        self,
        capacity_bytes: int,
        *,
        device: torch.device | str = "cpu",
        slot_bytes: int | None = None,
        alignment: int = 256,
        budget: VramBudget | None = None,
        reservation_name: str = "uint8_dual_arena",
    ) -> None:
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
        initial_layout = (
            _compute_high_slot_layout(cap, align, slot_bytes)
            if slot_bytes is not None
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

        self._lock = threading.Lock()
        self._capacity_bytes = cap
        self._allocated_bytes = allocated_bytes
        self._device = resolved_device
        self._alignment = align
        self._reservation: VramReservation | None = reservation
        self._low_cursor = 0
        self._low_free: list[tuple[int, int]] = []
        self._active_low: dict[int, ArenaSlice] = {}
        self._high_layout: HighSlotLayout | None = initial_layout
        self._allocated_high_slots: set[int] = set()
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
    def slot_layout(self) -> HighSlotLayout | None:
        with self._lock:
            return self._high_layout

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
            return self._low_cursor

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
            return max(0, self._high_frontier_unlocked() - self._low_cursor)

    @property
    def active_low_bytes(self) -> int:
        with self._lock:
            return sum(slc.aligned_size for slc in self._active_low.values())

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
            return sum(
                1
                for slot_id in range(self._high_layout.max_slots)
                if slot_id not in self._allocated_high_slots
                and self._high_layout.slot_offset(slot_id) >= self._low_cursor
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

    def configure_high_slots(self, slot_bytes: int) -> HighSlotLayout:
        """Configure or replace the descending high-end slot geometry when empty."""
        layout = _compute_high_slot_layout(
            self._capacity_bytes,
            self._alignment,
            slot_bytes,
        )
        with self._lock:
            self._require_storage_unlocked()
            if self._allocated_high_slots:
                raise RuntimeError(
                    "Cannot reconfigure high slot layout while high slots are allocated."
                )
            self._high_layout = layout
            return layout

    def alloc_low(self, nbytes: int, *, tag: str = "") -> ArenaSlice:
        """Allocate ``nbytes`` (aligned up) from the low end with free-list reuse."""
        req = _validate_positive_int(nbytes, name="nbytes")
        aligned = _align_up(req, self._alignment)
        with self._lock:
            self._require_storage_unlocked()
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

    def alloc_high_slot(self) -> int:
        """Allocate the highest free slot index whose range stays above ``low_frontier``."""
        with self._lock:
            self._require_storage_unlocked()
            if self._high_layout is None:
                raise RuntimeError("High slots are not configured on this Uint8DualArena.")
            layout = self._high_layout
            for slot_id in range(layout.max_slots - 1, -1, -1):
                if slot_id in self._allocated_high_slots:
                    continue
                start, _end = layout.slot_byte_range(slot_id)
                if start < self._low_cursor:
                    break
                self._allocated_high_slots.add(slot_id)
                return slot_id
            raise ArenaCapacityError(
                f"No high slot fits above low_frontier={self._low_cursor} "
                f"(slot_stride={layout.slot_stride}, max_slots={layout.max_slots})."
            )

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

    def high_pool_view(self, inner_shape: tuple[int, ...]) -> torch.Tensor:
        """Return a positive-stride ``uint8`` tensor ``[max_slots, *inner_shape]``."""
        shape = tuple(inner_shape)
        inner_strides = _contiguous_strides(shape)
        inner_numel = math.prod(shape)
        with self._lock:
            if self._high_layout is None:
                raise RuntimeError("High slots are not configured on this Uint8DualArena.")
            self._require_storage_unlocked()
            raw = self._raw_storage
            assert raw is not None
            layout = self._high_layout
            if inner_numel > layout.slot_bytes:
                raise ValueError(
                    f"inner_shape {shape} ({inner_numel} bytes) exceeds "
                    f"slot_bytes={layout.slot_bytes}."
                )
            return raw.as_strided(
                size=(layout.max_slots, *shape),
                stride=(layout.slot_stride, *inner_strides),
                storage_offset=self._base_offset + layout.base_offset,
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
        """Return active low-end slices overlapping high slot ``slot_id``'s bytes."""
        with self._lock:
            if self._high_layout is None:
                raise RuntimeError("High slots are not configured on this Uint8DualArena.")
            slot_start, slot_end = self._high_layout.slot_byte_range(slot_id)
            return self._overlapping_low_unlocked(slot_start, slot_end)

    def blocking_low_slices(self, slot_id: int) -> tuple[ArenaSlice, ...]:
        """Return active low slices pinning ``low_frontier`` above high slot ``slot_id``.

        Unlike :meth:`overlapping_low_slices`, this includes slices at higher
        offsets up to ``low_frontier`` whose eviction is required for
        ``low_frontier`` to shrink below ``slot_id``'s start byte.
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
        """Return active low-end slices intersecting ``[start_byte, end_byte)``."""
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
            self._allocated_high_slots.clear()
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
        elem_bytes = torch.empty((), dtype=dtype).element_size()
        align_elems = max(1, math.lcm(align_b, elem_bytes) // elem_bytes)

        if stride0 is None:
            resolved_stride0 = _align_up(inner_numel, align_elems)
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
        elem_bytes = torch.empty((), dtype=dtype).element_size()
        align_elems = max(1, math.lcm(align_b, elem_bytes) // elem_bytes)

        offset = 0
        field_specs: list[tuple[str, tuple[int, ...], tuple[int, ...], int]] = []
        for name, raw_shape in fields.items():
            if not isinstance(name, str) or not name:
                raise ValueError("StructuredSidePool field names must be non-empty strings.")
            shape = tuple(raw_shape)
            strides = _contiguous_strides(shape)
            numel = math.prod(shape)
            field_stride = _align_up(numel, align_elems)
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
    "ArenaSlice",
    "HighSlotLayout",
    "KvMetadataPool",
    "StructuredSidePool",
    "TypedSidePool",
    "Uint8DualArena",
]
