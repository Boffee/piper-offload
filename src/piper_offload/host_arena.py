"""Tier 2a pinned slab arena, Tier 2b private overflow arena, and pinned bounce ring.

This Layer 1 memory substrate owns page-aligned OS slabs allocated once (128 MiB
by default) and sub-allocated via a coalescing free list so hot allocations,
evictions, and Direct I/O fills incur zero ``mmap``/``VirtualAlloc`` teardown or
per-tensor ``cudaHostRegister`` syscalls:

- :class:`HostSlabArena` backs both **Tier 2a** (``pinned=True``, single
  registration per OS slab) and **Tier 2b** (``pinned=False``, private pageable
  overflow arena beyond Windows WDDM pin limits or cgroup/pin budgets).
- :class:`PinnedBounceRing` owns a pre-pinned circular staging ring for
  pipelining ``Private RAM -> Pinned Bounce Ring -> VRAM`` and direct
  ``Disk (DirectFileReader) -> Pinned Bounce Ring -> VRAM`` streams without
  dynamically registering source storage.
"""

from __future__ import annotations

import contextlib
import ctypes
import io
import logging
import mmap
import threading
import weakref
from bisect import bisect_left
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Self, cast

import torch

from . import _host_memory as host_memory
from . import _transfer_priority as transfer_priority
from ._direct_io import (
    DEFAULT_SECTOR_SIZE,
    AlignedSpan,
    DirectFileReader,
    align_span,
)
from ._host_memory import AlignedRegion
from ._host_registration import (
    HostRegistrationBackend,
    HostRegistrationError,
    HostRegistrationRefusedError,
    RuntimeHostRegistration,
)
from .checkpoint import FileSlice

logger = logging.getLogger(__name__)

DEFAULT_SLAB_BYTES = 128 * 1024 * 1024
DEFAULT_BOUNCE_SLOT_BYTES = 16 * 1024 * 1024
DEFAULT_BOUNCE_SLOTS = 8
DEFAULT_MIN_BOUNCE_BYTES = 256 * 1024
_MIN_BOUNCE_SLOT_BYTES = max(mmap.PAGESIZE, 2 * DEFAULT_SECTOR_SIZE)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _align_down(value: int, alignment: int) -> int:
    return (value // alignment) * alignment


def _validate_int(value: int, name: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        bound = "non-negative" if minimum == 0 else ("positive" if minimum == 1 else f">= {minimum}")
        raise ValueError(f"{name} must be a {bound} integer, got {value!r}.")


def _validate_power_of_two(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f"{name} must be a positive power of two, got {value!r}.")


def _region_pointer_and_size(region: AlignedRegion) -> tuple[int, int]:
    with memoryview(region) as mv:
        nbytes = mv.nbytes
        if mv.ndim != 1 or mv.itemsize != 1 or not mv.c_contiguous or mv.readonly or nbytes == 0:
            return 0, 0
        return ctypes.addressof(ctypes.c_char.from_buffer(mv)), nbytes


def _as_flat_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """Expose a contiguous strided tensor's payload as a 1-D ``uint8`` view."""
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        tensor.untyped_storage(),
        tensor.storage_offset() * tensor.element_size(),
        (tensor.nbytes,),
        (1,),
    )


class _RegStatus(Enum):
    REGISTERED = "registered"
    UNPINNED = "unpinned"
    OS_FAILED = "os_failed"
    CAPACITY = "capacity"
    REFUSED = "refused"


_RETAINED_REGIONS: dict[int, AlignedRegion] = {}
_RETAINED_LOCK = threading.Lock()


def _retain_region(region: AlignedRegion) -> None:
    with _RETAINED_LOCK:
        _RETAINED_REGIONS[id(region)] = region


@dataclass(slots=True)
class _RegState:
    region: AlignedRegion | None
    pointer: int
    backend: HostRegistrationBackend | None
    registered: bool
    lock: threading.Lock


def _close_region_gracefully(region: AlignedRegion, context_label: str) -> None:
    try:
        region.close()
    except BufferError:
        logger.warning(
            "%s still has surviving buffer references; its memory returns when they die.",
            context_label,
        )


def _finalize_reg_state(state: _RegState, context_label: str) -> None:
    with state.lock:
        region = state.region
        if region is None:
            return
        if state.registered and state.backend is not None:
            try:
                state.backend.unregister(state.pointer)
            except BaseException as error:
                _retain_region(region)
                state.region = None
                if isinstance(error, Exception):
                    logger.warning(
                        "GC cleanup failed to unregister %s; retaining mapping: %s",
                        context_label,
                        str(error),
                    )
                    return
                raise
            state.registered = False
        state.region = None
    _close_region_gracefully(region, context_label)


class _RegisteredRegion:
    """Single owner of OS host allocation, optional CUDA host registration, and GC teardown."""

    def __init__(
        self,
        region: AlignedRegion,
        pointer: int,
        backend: HostRegistrationBackend | None,
        *,
        registered: bool,
        context_label: str,
    ) -> None:
        state = _RegState(
            region=region,
            pointer=pointer,
            backend=backend,
            registered=registered,
            lock=threading.Lock(),
        )
        self._state = state
        self._context_label = context_label
        fin: weakref.finalize[[_RegState, str], Any] = weakref.finalize(
            self, _finalize_reg_state, state, context_label
        )
        fin.atexit = False
        self._finalizer = fin

    @property
    def region(self) -> AlignedRegion:
        region = self._state.region
        if region is None:
            raise RuntimeError(f"{self._context_label} has already been closed.")
        return region

    @property
    def pointer(self) -> int:
        return self._state.pointer

    @property
    def registered(self) -> bool:
        return self._state.registered

    @classmethod
    def allocate(
        cls,
        size: int,
        *,
        allocator: Callable[[int], AlignedRegion],
        pinned: bool,
        backend: HostRegistrationBackend | None,
        context_label: str,
    ) -> tuple[_RegisteredRegion | None, _RegStatus]:
        try:
            region = allocator(size)
        except (OSError, MemoryError) as error:
            logger.warning("Could not allocate %d-byte %s: %s", size, context_label, str(error))
            return None, _RegStatus.OS_FAILED

        try:
            pointer, nbytes = _region_pointer_and_size(region)
            if pointer == 0 or nbytes < size or pointer % mmap.PAGESIZE != 0:
                raise ValueError(
                    f"Allocator returned invalid region for {context_label}: "
                    f"ptr={pointer}, size={nbytes} (expected >= {size}, {mmap.PAGESIZE}-byte aligned)."
                )
            if not pinned:
                return (
                    cls(region, pointer, None, registered=False, context_label=context_label),
                    _RegStatus.UNPINNED,
                )
            assert backend is not None
            ok = backend.register(pointer, size)
        except HostRegistrationRefusedError:
            with contextlib.suppress(BufferError, OSError):
                region.close()
            return None, _RegStatus.REFUSED
        except BaseException:
            with contextlib.suppress(BufferError, OSError):
                region.close()
            raise

        if not ok:
            with contextlib.suppress(BufferError, OSError):
                region.close()
            return None, _RegStatus.CAPACITY

        try:
            return (
                cls(region, pointer, backend, registered=True, context_label=context_label),
                _RegStatus.REGISTERED,
            )
        except BaseException:
            unregistered = False
            try:
                backend.unregister(pointer)
                unregistered = True
            except BaseException:
                _retain_region(region)
            if unregistered:
                with contextlib.suppress(BufferError, OSError):
                    region.close()
            raise

    def unregister_only(self, *, suppress_errors: bool = True) -> bool:
        with self._state.lock:
            if not self._state.registered or self._state.backend is None:
                return True
            try:
                self._state.backend.unregister(self._state.pointer)
            except Exception as error:
                if not suppress_errors:
                    raise
                logger.warning(
                    "Unregistration failed for %s; retaining mapping: %s",
                    self._context_label,
                    str(error),
                )
                return False
            self._state.registered = False
            return True

    def close_region(self) -> None:
        with self._state.lock:
            region = self._state.region
            self._state.region = None
        self._finalizer.detach()
        if region is not None:
            _close_region_gracefully(region, self._context_label)

    def release_or_retain(self, *, keep_region: bool = False) -> bool:
        if keep_region:
            with self._state.lock:
                if self._state.region is not None:
                    _retain_region(self._state.region)
                    self._state.region = None
            self._finalizer.detach()
            return False
        if not self.unregister_only(suppress_errors=True):
            with self._state.lock:
                if self._state.region is not None:
                    _retain_region(self._state.region)
                    self._state.region = None
            self._finalizer.detach()
            return False
        self.close_region()
        return True


@dataclass(frozen=True, slots=True)
class HostArenaStats:
    """Accounting snapshot for a :class:`HostSlabArena`."""

    pinned: bool
    max_bytes: int | None
    reserved_bytes: int
    used_bytes: int
    slab_count: int
    empty_slab_count: int
    live_allocations: int
    registration_failures: int
    unregistration_failures: int


class _Slab:
    """One OS-allocated (and optionally runtime-registered) slab with best-fit coalescing spans."""

    __slots__ = (
        "backing",
        "free_spans",
        "live_allocations",
        "pointer",
        "size",
        "slab_id",
        "used_bytes",
    )

    def __init__(self, slab_id: int, size: int, backing: _RegisteredRegion) -> None:
        self.slab_id = slab_id
        self.size = size
        self.backing = backing
        self.pointer = backing.pointer
        self.free_spans: list[tuple[int, int]] = [(0, size)]
        self.used_bytes = 0
        self.live_allocations = 0

    def _alloc(self, aligned_size: int, sub_align: int) -> int | None:
        best_idx = -1
        best_waste = -1
        best_pad = 0
        for idx, (span_off, span_len) in enumerate(self.free_spans):
            pad = (-(self.pointer + span_off)) % sub_align
            if span_len < pad + aligned_size:
                continue
            waste = span_len - aligned_size
            if best_idx < 0 or waste < best_waste:
                best_idx = idx
                best_waste = waste
                best_pad = pad
                if waste == 0:
                    break
        if best_idx < 0:
            return None

        span_off, span_len = self.free_spans.pop(best_idx)
        alloc_off = span_off + best_pad
        insert_pos = best_idx
        if best_pad > 0:
            self.free_spans.insert(insert_pos, (span_off, best_pad))
            insert_pos += 1
        tail = (span_off + span_len) - (alloc_off + aligned_size)
        if tail > 0:
            self.free_spans.insert(insert_pos, (alloc_off + aligned_size, tail))

        self.used_bytes += aligned_size
        self.live_allocations += 1
        return alloc_off

    def _free(self, offset: int, aligned_size: int) -> None:
        self.used_bytes -= aligned_size
        self.live_allocations -= 1
        idx = bisect_left(self.free_spans, (offset, 0))
        new_off = offset
        new_len = aligned_size

        if idx > 0:
            prev_off, prev_len = self.free_spans[idx - 1]
            if prev_off + prev_len == new_off:
                new_off = prev_off
                new_len += prev_len
                self.free_spans.pop(idx - 1)
                idx -= 1
        if idx < len(self.free_spans):
            next_off, next_len = self.free_spans[idx]
            if new_off + new_len == next_off:
                new_len += next_len
                self.free_spans.pop(idx)

        self.free_spans.insert(idx, (new_off, new_len))


class _SpanLease:
    """Reference-counted span lease tying slab span return to handle release AND live exports."""

    __slots__ = (
        "_arena",
        "_exports",
        "_lock",
        "_released",
        "_returned",
        "offset",
        "size",
        "slab",
    )

    def __init__(self, arena: HostSlabArena, slab: _Slab, offset: int, size: int) -> None:
        self._arena: HostSlabArena | None = arena
        self.slab: _Slab | None = slab
        self.offset = offset
        self.size = size
        self._lock = threading.RLock()
        self._released = False
        self._exports = 0
        self._returned = False

    def acquire_region_for_export(self) -> AlignedRegion:
        with self._lock:
            if self._released or self.slab is None:
                raise RuntimeError("ArenaAllocation has been released.")
            region = self.slab.backing.region
            self._exports += 1
            return region

    def _pop_return_target_locked(self) -> tuple[HostSlabArena, _Slab] | None:
        if not (self._released and self._exports == 0 and not self._returned):
            return None
        self._returned = True
        arena, slab = self._arena, self.slab
        self._arena = None
        self.slab = None
        if arena is None or slab is None:
            return None
        return arena, slab

    def release_export(self) -> None:
        with self._lock:
            self._exports -= 1
            target = self._pop_return_target_locked()
        if target is not None:
            target[0]._return_slab_span(target[1], self.offset, self.size)

    def mark_released_locked(self) -> tuple[HostSlabArena, _Slab] | None:
        if self._released:
            return None
        self._released = True
        return self._pop_return_target_locked()

    def release_handle(self) -> None:
        with self._lock:
            target = self.mark_released_locked()
        if target is not None:
            target[0]._return_slab_span(target[1], self.offset, self.size)


_ACTIVE_SPAN_EXPORTS: dict[int, _SpanLease] = {}
_SPAN_EXPORTS_LOCK = threading.Lock()


class _SpanView:
    """PEP 688 Buffer object ensuring exported views/tensors keep ``_SpanLease`` alive."""

    __slots__ = ("_end", "_lease", "_start")

    def __init__(self, lease: _SpanLease, start: int, end: int) -> None:
        self._lease = lease
        self._start = start
        self._end = end

    def __buffer__(self, flags: int, /) -> memoryview:
        lease = self._lease
        region = lease.acquire_region_for_export()
        try:
            inner = memoryview(region)[self._start : self._end]
        except BaseException:
            lease.release_export()
            raise
        with _SPAN_EXPORTS_LOCK:
            _ACTIVE_SPAN_EXPORTS[id(inner)] = lease
        return inner

    def __release_buffer__(self, view: memoryview, /) -> None:
        try:
            view.release()
        finally:
            with _SPAN_EXPORTS_LOCK:
                lease = _ACTIVE_SPAN_EXPORTS.pop(id(view), None)
            if lease is not None:
                lease.release_export()


def _finalize_lease(lease: _SpanLease) -> None:
    lease.release_handle()


class ArenaAllocation:
    """Sub-allocated byte slice inside a :class:`HostSlabArena` slab."""

    __slots__ = (
        "__weakref__",
        "_finalizer",
        "_lease",
        "_payload_storage",
        "data_length",
        "head_pad",
        "offset",
        "size",
    )

    def __init__(
        self,
        arena: HostSlabArena,
        slab: _Slab,
        offset: int,
        size: int,
        head_pad: int,
        data_length: int,
    ) -> None:
        self._lease = _SpanLease(arena, slab, offset, size)
        self.offset = offset
        self.size = size
        self.head_pad = head_pad
        self.data_length = data_length
        self._payload_storage: torch.UntypedStorage | None = None
        finalizer: weakref.finalize[[_SpanLease], Any] = weakref.finalize(self, _finalize_lease, self._lease)
        finalizer.atexit = False
        self._finalizer = finalizer

    @property
    def is_pinned(self) -> bool:
        with self._lease._lock:
            if self._lease._released or self._lease.slab is None:
                return False
            return self._lease.slab.backing.registered

    @property
    def released(self) -> bool:
        with self._lease._lock:
            return self._lease._released

    @property
    def aligned_pointer(self) -> int:
        with self._lease._lock:
            if self._lease._released or self._lease.slab is None:
                raise RuntimeError("ArenaAllocation has been released.")
            return self._lease.slab.pointer + self.offset

    @property
    def pointer(self) -> int:
        with self._lease._lock:
            if self._lease._released or self._lease.slab is None:
                raise RuntimeError("ArenaAllocation has been released.")
            return self._lease.slab.pointer + self.offset + self.head_pad

    def _require_live(self) -> None:
        with self._lease._lock:
            if self._lease._released:
                raise RuntimeError("ArenaAllocation has been released.")

    def aligned_view(self) -> memoryview:
        """Writable ``memoryview`` keeping this span leased until released."""
        return memoryview(_SpanView(self._lease, self.offset, self.offset + self.size))

    def payload_view(self) -> memoryview:
        """Writable ``memoryview`` of the payload at ``offset + head_pad``."""
        start = self.offset + self.head_pad
        return memoryview(_SpanView(self._lease, start, start + self.data_length))

    def payload_storage(self) -> torch.UntypedStorage:
        """Return a ``torch.UntypedStorage`` aliasing this payload and holding the span lease."""
        with self._lease._lock:
            if self._lease._released:
                raise RuntimeError("ArenaAllocation has been released.")
            cached = self._payload_storage
        if cached is not None:
            return cached

        start = self.offset + self.head_pad
        mv = memoryview(_SpanView(self._lease, start, start + self.data_length))
        new_storage: torch.UntypedStorage | None = None
        discard_storage: torch.UntypedStorage | None = None
        try:
            # Retain mv in torch.frombuffer so both _SpanLease and the underlying region stay exported.
            new_storage = torch.frombuffer(mv, dtype=torch.uint8).untyped_storage()
        except BaseException:
            mv.release()
            raise

        with self._lease._lock:
            chosen: torch.UntypedStorage | None = None
            if self._lease._released:
                discard_storage = new_storage
            elif self._payload_storage is not None:
                discard_storage = new_storage
                chosen = self._payload_storage
            else:
                self._payload_storage = new_storage
                chosen = new_storage
        # Drop any unused UntypedStorage off-lock so __release_buffer__ never runs inside _lease._lock.
        del discard_storage, new_storage, mv
        if chosen is None:
            raise RuntimeError("ArenaAllocation has been released.")
        return chosen

    def as_uint8_tensor(self) -> torch.Tensor:
        """Return a 1-D ``torch.uint8`` CPU tensor over this allocation's payload."""
        storage = self.payload_storage()
        return torch.empty(0, dtype=torch.uint8, device="cpu").set_(
            storage, 0, (self.data_length,), (1,)
        )

    def _validate_template_extent(self, template: torch.Tensor) -> None:
        if template.layout is not torch.strided:
            raise ValueError(f"ArenaAllocation.tensor_view requires a strided tensor, got {template.layout}.")
        elem_size = template.element_size()
        storage_offset = template.storage_offset()
        if storage_offset < 0:
            raise ValueError(f"Template tensor has negative storage_offset {storage_offset}.")
        offset_bytes = storage_offset * elem_size
        if template.numel() == 0:
            required = offset_bytes
        else:
            max_elem = 0
            for dim_size, dim_stride in zip(template.shape, template.stride(), strict=True):
                if dim_size > 1:
                    if dim_stride < 0:
                        raise ValueError("Negative strides are not supported in ArenaAllocation.tensor_view.")
                    max_elem += (dim_size - 1) * dim_stride
            required = offset_bytes + (max_elem + 1) * elem_size
        if required > self.data_length:
            raise ValueError(
                f"Template tensor requires {required} bytes, exceeding allocation payload {self.data_length}."
            )

    def tensor_view(self, template: torch.Tensor) -> torch.Tensor:
        """Build a CPU tensor matching ``template``'s dtype, shape, stride, and offset."""
        self._validate_template_extent(template)
        storage = self.payload_storage()
        view = torch.empty(0, dtype=template.dtype, device="cpu").set_(
            storage,
            template.storage_offset(),
            template.shape,
            template.stride(),
        )
        if template.is_conj():
            view = view.conj()
        if template.is_neg():
            view = torch._neg_view(view)
        return view

    def release(self) -> None:
        """Surrender this handle; span returns immediately or once live views/tensors die."""
        with self._lease._lock:
            stale_storage = self._payload_storage
            self._payload_storage = None
            target = self._lease.mark_released_locked()
        del stale_storage
        if target is not None:
            target[0]._return_slab_span(target[1], self.offset, self.size)
        self._finalizer.detach()

    def __enter__(self) -> Self:
        self._require_live()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class HostSlabArena:
    """Chunked Tier 2a (pinned) or Tier 2b (private pageable) host memory arena."""

    def __init__(
        self,
        *,
        pinned: bool,
        max_bytes: int | None = None,
        slab_bytes: int = DEFAULT_SLAB_BYTES,
        backend: HostRegistrationBackend | None = None,
        allocator: Callable[[int], AlignedRegion] | None = None,
    ) -> None:
        if max_bytes is not None:
            _validate_int(max_bytes, "max_bytes", minimum=0)
        _validate_int(slab_bytes, "slab_bytes", minimum=1)
        if slab_bytes % mmap.PAGESIZE != 0:
            raise ValueError(f"slab_bytes must be a multiple of {mmap.PAGESIZE}, got {slab_bytes!r}.")

        self._pinned = pinned
        self._max_bytes = max_bytes
        self._slab_bytes = slab_bytes
        self._backend: HostRegistrationBackend | None = (
            None if not pinned else (RuntimeHostRegistration() if backend is None else backend)
        )
        self._allocator: Callable[[int], AlignedRegion] = (
            host_memory.allocate_aligned_region if allocator is None else allocator
        )

        self._lock = threading.Lock()
        self._slab_cv = threading.Condition(self._lock)
        self._owner_tid = 0
        self._pending_returns: deque[tuple[_Slab, int, int]] = deque()
        self._slabs: list[_Slab] = []
        self._next_slab_id = 0
        self._reserved_bytes = 0
        self._used_bytes = 0
        self._live_allocations = 0
        self._registration_failures = 0
        self._unregistration_failures = 0
        self._creating_slab = False
        self._reclaiming_slabs = 0
        self._reclaiming_bytes = 0
        self._closed = False

    def _mark_reclaiming_locked(self, slabs: list[_Slab]) -> None:
        self._reclaiming_slabs += len(slabs)
        for slab in slabs:
            self._reclaiming_bytes += slab.size

    def _drain_pending_returns_locked(self) -> list[_Slab]:
        popped: list[_Slab] = []
        projected_reserved = self._reserved_bytes - self._reclaiming_bytes
        while self._pending_returns:
            slab, offset, aligned_size = self._pending_returns.popleft()
            slab._free(offset, aligned_size)
            self._used_bytes -= aligned_size
            self._live_allocations -= 1
            if (
                slab.live_allocations == 0
                and slab in self._slabs
                and (
                    self._closed
                    or (self._max_bytes is not None and projected_reserved > self._max_bytes)
                )
            ):
                self._slabs.remove(slab)
                popped.append(slab)
                projected_reserved -= slab.size
        if popped:
            self._mark_reclaiming_locked(popped)
        return popped

    def _wait_cv_locked(self) -> None:
        self._owner_tid = 0
        try:
            self._slab_cv.wait()
        finally:
            self._owner_tid = threading.get_ident()

    @contextmanager
    def _owned_lock(self) -> Iterator[None]:
        with self._lock:
            self._owner_tid = threading.get_ident()
            try:
                yield
            finally:
                self._owner_tid = 0

    @contextmanager
    def _drain_lock(self, popped: list[_Slab]) -> Iterator[None]:
        with self._owned_lock():
            popped.extend(self._drain_pending_returns_locked())
            try:
                yield
            finally:
                popped.extend(self._drain_pending_returns_locked())

    @contextmanager
    def _state_lock(self) -> Iterator[None]:
        popped: list[_Slab] = []
        try:
            with self._drain_lock(popped):
                yield
        finally:
            if popped:
                self._try_reclaim_popped_slabs(popped)

    @property
    def pinned(self) -> bool:
        return self._pinned

    @property
    def max_bytes(self) -> int | None:
        with self._state_lock():
            return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, value: int | None) -> None:
        if value is not None:
            _validate_int(value, "max_bytes", minimum=0)
        with self._state_lock():
            self._max_bytes = value
        if value is not None:
            self._reclaim_empty_slabs(value)

    @property
    def stats(self) -> HostArenaStats:
        with self._state_lock():
            empty_slabs = 0
            for slab in self._slabs:
                if slab.live_allocations == 0:
                    empty_slabs += 1
            return HostArenaStats(
                pinned=self._pinned,
                max_bytes=self._max_bytes,
                reserved_bytes=self._reserved_bytes,
                used_bytes=self._used_bytes,
                slab_count=len(self._slabs),
                empty_slab_count=empty_slabs,
                live_allocations=self._live_allocations,
                registration_failures=self._registration_failures,
                unregistration_failures=self._unregistration_failures,
            )

    def _try_reclaim_popped_slabs(self, candidates: list[_Slab]) -> int:
        """Unregister and close empty slabs already removed from ``_slabs`` (off-lock)."""
        if not candidates:
            return 0
        reclaimed: list[_Slab] = []
        failed: list[_Slab] = []
        try:
            for slab in candidates:
                if slab.backing.unregister_only(suppress_errors=True):
                    reclaimed.append(slab)
                    slab.backing.close_region()
                else:
                    failed.append(slab)
        finally:
            reclaimed_set = {id(s) for s in reclaimed}
            restore = [s for s in candidates if id(s) not in reclaimed_set]
            reclaimed_bytes = sum(s.size for s in reclaimed)
            total_candidate_bytes = sum(s.size for s in candidates)
            with self._owned_lock():
                self._reserved_bytes -= reclaimed_bytes
                if failed:
                    self._unregistration_failures += len(failed)
                if restore:
                    self._slabs[:0] = restore
                self._reclaiming_slabs -= len(candidates)
                self._reclaiming_bytes -= total_candidate_bytes
                self._slab_cv.notify_all()
        return len(failed)

    def _reclaim_empty_slabs(self, target_bytes: int) -> int:
        candidates: list[_Slab] = []
        try:
            with self._drain_lock(candidates):
                extra: list[_Slab] = []
                remaining: list[_Slab] = []
                projected_reserved = self._reserved_bytes - self._reclaiming_bytes
                for slab in self._slabs:
                    if slab.live_allocations == 0 and projected_reserved > target_bytes:
                        extra.append(slab)
                        projected_reserved -= slab.size
                    else:
                        remaining.append(slab)
                if extra:
                    self._slabs = remaining
                    self._mark_reclaiming_locked(extra)
                    candidates.extend(extra)
        finally:
            if candidates:
                to_reclaim = list(candidates)
                candidates.clear()
                return_count = self._try_reclaim_popped_slabs(to_reclaim)
            else:
                return_count = 0
        return return_count

    def _find_and_alloc_locked(self, aligned_size: int, sub_align: int) -> tuple[_Slab, int] | None:
        for slab in self._slabs:
            offset = slab._alloc(aligned_size, sub_align)
            if offset is not None:
                self._used_bytes += aligned_size
                self._live_allocations += 1
                return slab, offset
        return None

    def _try_backing_once(self, size: int) -> tuple[_RegisteredRegion | None, _RegStatus]:
        try:
            backing, status = _RegisteredRegion.allocate(
                size,
                allocator=self._allocator,
                pinned=self._pinned,
                backend=self._backend,
                context_label="host arena slab",
            )
        except HostRegistrationError:
            with self._owned_lock():
                self._registration_failures += 1
            raise
        if status in (_RegStatus.CAPACITY, _RegStatus.REFUSED):
            with self._owned_lock():
                self._registration_failures += 1
        return backing, status

    def _allocate_slab_backing(self, candidate_size: int, min_slab_size: int) -> tuple[int, _RegisteredRegion] | None:
        backing, status = self._try_backing_once(candidate_size)
        if backing is not None:
            return candidate_size, backing
        if status is _RegStatus.REFUSED:
            return None

        has_empty = False
        with self._owned_lock():
            for slab in self._slabs:
                if slab.live_allocations == 0:
                    has_empty = True
                    break
        if has_empty:
            self._reclaim_empty_slabs(0)
            backing, status = self._try_backing_once(candidate_size)
            if backing is not None:
                return candidate_size, backing
            if status is _RegStatus.REFUSED:
                return None

        if candidate_size > min_slab_size:
            backing, _ = self._try_backing_once(min_slab_size)
            if backing is not None:
                return min_slab_size, backing
        return None

    @staticmethod
    def _validate_alloc_args(
        data_length: int,
        head_pad: int,
        alignment: int,
        slab_hint_bytes: int | None,
    ) -> tuple[int, int]:
        _validate_int(data_length, "data_length", minimum=1)
        _validate_int(head_pad, "head_pad", minimum=0)
        _validate_power_of_two(alignment, "alignment")
        if slab_hint_bytes is not None:
            _validate_int(slab_hint_bytes, "slab_hint_bytes", minimum=1)
        sub_align = max(alignment, mmap.PAGESIZE)
        return sub_align, _align_up(head_pad + data_length, sub_align)

    def _compute_new_slab_size_locked(
        self,
        min_slab_size: int,
        slab_hint_bytes: int | None,
        popped_for_trim: list[_Slab],
    ) -> int | None:
        if self._max_bytes is not None:
            if min_slab_size > self._max_bytes:
                return None
            if self._reserved_bytes + min_slab_size > self._max_bytes:
                headroom_target = self._max_bytes - min_slab_size
                projected = self._reserved_bytes
                remaining: list[_Slab] = []
                for slab in self._slabs:
                    if slab.live_allocations == 0 and projected > headroom_target:
                        popped_for_trim.append(slab)
                        projected -= slab.size
                    else:
                        remaining.append(slab)
                if projected + min_slab_size > self._max_bytes:
                    popped_for_trim.clear()
                    return None
                self._slabs = remaining

        desired = (
            self._slab_bytes
            if slab_hint_bytes is None
            else min(self._slab_bytes, _align_up(slab_hint_bytes, mmap.PAGESIZE))
        )
        candidate_size = max(min_slab_size, desired)
        if self._max_bytes is not None:
            reclaimed_bytes = sum(s.size for s in popped_for_trim)
            headroom = _align_down(self._max_bytes - self._reserved_bytes + reclaimed_bytes, mmap.PAGESIZE)
            candidate_size = min(candidate_size, headroom)
            if candidate_size < min_slab_size:
                return None
        return candidate_size

    def _commit_new_slab_locked(
        self,
        final_size: int,
        backing: _RegisteredRegion,
        reserved_delta: int,
        aligned_size: int,
        sub_align: int,
        trailing_drained: list[_Slab],
    ) -> tuple[_Slab, int] | None:
        with self._drain_lock(trailing_drained):
            if self._closed:
                raise RuntimeError("HostSlabArena is closed.")
            projected_reserved = (self._reserved_bytes - reserved_delta) + final_size
            if self._max_bytes is not None and (
                projected_reserved > self._max_bytes or self._used_bytes + aligned_size > self._max_bytes
            ):
                return None
            slab = _Slab(self._next_slab_id, final_size, backing)
            offset = slab._alloc(aligned_size, sub_align)
            if offset is None:
                raise RuntimeError("Newly created slab could not satisfy alignment.")
            self._reserved_bytes = projected_reserved
            self._next_slab_id += 1
            self._slabs.append(slab)
            self._used_bytes += aligned_size
            self._live_allocations += 1
            return slab, offset

    def _release_create_slab_state(
        self,
        uncommitted_backing: _RegisteredRegion | None,
        reserved_delta: int,
        committed: tuple[_Slab, int] | None,
        trailing_drained: list[_Slab],
        aligned_size: int,
    ) -> None:
        unreg_failed = 0
        try:
            if (
                uncommitted_backing is not None
                and not uncommitted_backing.release_or_retain()
                and uncommitted_backing.registered
            ):
                unreg_failed = 1
        finally:
            with self._drain_lock(trailing_drained):
                if reserved_delta:
                    self._reserved_bytes -= reserved_delta
                if unreg_failed:
                    self._unregistration_failures += unreg_failed
                self._creating_slab = False
                self._slab_cv.notify_all()
        if trailing_drained:
            to_reclaim = list(trailing_drained)
            trailing_drained.clear()
            try:
                self._try_reclaim_popped_slabs(to_reclaim)
            except BaseException:
                if committed is not None:
                    self._return_slab_span(committed[0], committed[1], aligned_size)
                raise

    def _create_slab_for_allocation(
        self,
        candidate_size: int,
        min_slab_size: int,
        aligned_size: int,
        sub_align: int,
        head_pad: int,
        data_length: int,
        initial_drained: list[_Slab],
    ) -> ArenaAllocation | None:
        uncommitted_backing: _RegisteredRegion | None = None
        committed: tuple[_Slab, int] | None = None
        reserved_delta = candidate_size
        trailing_drained: list[_Slab] = []
        try:
            if initial_drained:
                to_reclaim = list(initial_drained)
                initial_drained.clear()
                self._try_reclaim_popped_slabs(to_reclaim)
            result = self._allocate_slab_backing(candidate_size, min_slab_size)
            if result is None:
                return None
            final_size, uncommitted_backing = result
            committed = self._commit_new_slab_locked(
                final_size,
                uncommitted_backing,
                reserved_delta,
                aligned_size,
                sub_align,
                trailing_drained,
            )
            if committed is None:
                return None
            uncommitted_backing = None
            reserved_delta = 0
        finally:
            self._release_create_slab_state(
                uncommitted_backing, reserved_delta, committed, trailing_drained, aligned_size
            )

        assert committed is not None
        return self._wrap_allocation(
            committed[0], committed[1], aligned_size, head_pad, data_length
        )

    def _wrap_allocation(
        self,
        slab: _Slab,
        offset: int,
        aligned_size: int,
        head_pad: int,
        data_length: int,
    ) -> ArenaAllocation:
        try:
            return ArenaAllocation(self, slab, offset, aligned_size, head_pad, data_length)
        except BaseException:
            self._return_slab_span(slab, offset, aligned_size)
            raise

    def _finish_fast_alloc(
        self,
        fast_alloc: tuple[_Slab, int],
        aligned_size: int,
        head_pad: int,
        data_length: int,
        drained: list[_Slab],
    ) -> ArenaAllocation:
        alloc = self._wrap_allocation(
            fast_alloc[0], fast_alloc[1], aligned_size, head_pad, data_length
        )
        if drained:
            try:
                self._try_reclaim_popped_slabs(drained)
            except BaseException:
                alloc.release()
                raise
        return alloc

    def _abort_creating_slab(self, reserved_delta: int) -> None:
        trailing: list[_Slab] = []
        with self._drain_lock(trailing):
            self._reserved_bytes -= reserved_delta
            self._creating_slab = False
            self._slab_cv.notify_all()
        if trailing:
            self._try_reclaim_popped_slabs(trailing)

    def _try_alloc_step_locked(
        self,
        aligned_size: int,
        sub_align: int,
        min_slab_size: int,
        slab_hint_bytes: int | None,
        drained: list[_Slab],
        popped_for_trim: list[_Slab],
    ) -> tuple[tuple[_Slab, int] | None, int | None, bool, bool]:
        if self._closed:
            raise RuntimeError("HostSlabArena is closed.")
        if self._max_bytes is not None and self._used_bytes + aligned_size > self._max_bytes:
            return None, None, False, True
        fast_alloc = self._find_and_alloc_locked(aligned_size, sub_align)
        if fast_alloc is not None:
            return fast_alloc, None, False, False
        if not self._creating_slab:
            candidate_size = self._compute_new_slab_size_locked(
                min_slab_size, slab_hint_bytes, popped_for_trim
            )
            if popped_for_trim:
                self._mark_reclaiming_locked(popped_for_trim)
                return None, candidate_size, False, False
            if candidate_size is not None:
                self._reserved_bytes += candidate_size
                self._creating_slab = True
                return None, candidate_size, True, False
            if self._reclaiming_slabs <= len(drained):
                return None, None, False, not drained
        self._wait_cv_locked()
        return None, None, False, False

    def allocate(
        self,
        data_length: int,
        *,
        head_pad: int = 0,
        alignment: int = DEFAULT_SECTOR_SIZE,
        slab_hint_bytes: int | None = None,
    ) -> ArenaAllocation | None:
        """Allocate a byte span of ``head_pad + data_length`` inside the arena."""
        sub_align, aligned_size = self._validate_alloc_args(
            data_length, head_pad, alignment, slab_hint_bytes
        )
        min_slab_size = _align_up(aligned_size + max(0, sub_align - mmap.PAGESIZE), mmap.PAGESIZE)

        while True:
            drained: list[_Slab] = []
            popped_for_trim: list[_Slab] = []
            candidate_size: int | None = None
            create_slab = False
            try:
                with self._drain_lock(drained):
                    fast_alloc, candidate_size, create_slab, stop = self._try_alloc_step_locked(
                        aligned_size, sub_align, min_slab_size, slab_hint_bytes, drained, popped_for_trim
                    )
            except BaseException:
                if create_slab and candidate_size is not None:
                    self._abort_creating_slab(candidate_size)
                if drained:
                    self._try_reclaim_popped_slabs(drained)
                raise

            if fast_alloc is not None:
                return self._finish_fast_alloc(
                    fast_alloc, aligned_size, head_pad, data_length, drained
                )
            if create_slab:
                assert candidate_size is not None
                return self._create_slab_for_allocation(
                    candidate_size,
                    min_slab_size,
                    aligned_size,
                    sub_align,
                    head_pad,
                    data_length,
                    drained,
                )
            if popped_for_trim:
                drained.extend(popped_for_trim)
            reclaim_failed = self._try_reclaim_popped_slabs(drained) if drained else 0
            if stop or (popped_for_trim and reclaim_failed > 0):
                return None

    def _return_slab_span(self, slab: _Slab, offset: int, aligned_size: int) -> None:
        self._pending_returns.append((slab, offset, aligned_size))
        if self._owner_tid == threading.get_ident():
            return
        with self._state_lock():
            pass

    def trim_empty(self, target_bytes: int = 0) -> int:
        """Unregister and free empty slabs down to ``target_bytes``; return failed unregistration count."""
        _validate_int(target_bytes, "target_bytes", minimum=0)
        return self._reclaim_empty_slabs(target_bytes)

    def clear(self) -> None:
        """Unregister and free all empty slabs while keeping the arena open."""
        failed = self.trim_empty(0)
        if failed:
            raise RuntimeError(
                f"Could not release {failed} host arena slab(s); storage remains retained."
            )

    def close(self) -> None:
        """Wait for in-flight slab creations and trims, release empty slabs, and auto-reclaim live ones."""
        candidates: list[_Slab] = []
        failed = 0
        try:
            with self._drain_lock(candidates):
                self._closed = True
                self._slab_cv.notify_all()
                while self._creating_slab or self._reclaiming_slabs > len(candidates):
                    self._wait_cv_locked()
                    candidates.extend(self._drain_pending_returns_locked())
                extra = [slab for slab in self._slabs if slab.live_allocations == 0]
                if extra:
                    self._slabs = [slab for slab in self._slabs if slab.live_allocations > 0]
                    self._mark_reclaiming_locked(extra)
                    candidates.extend(extra)
        finally:
            if candidates:
                to_reclaim = list(candidates)
                candidates.clear()
                failed = self._try_reclaim_popped_slabs(to_reclaim)
        if failed:
            raise RuntimeError(
                f"Could not release {failed} host arena slab(s); storage remains retained."
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class _RingState(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class _RingGeometry:
    slot_bytes: int
    num_slots: int
    pointer: int


@dataclass(slots=True)
class _BounceSlot:
    index: int
    offset: int
    size: int
    uint8_view: torch.Tensor
    event: torch.cuda.Event | None = None
    poisoned: bool = False


@dataclass(slots=True)
class _RingResources:
    backing: _RegisteredRegion | None = None
    storage: torch.UntypedStorage | None = None
    pool: ThreadPoolExecutor | None = None
    slots: list[_BounceSlot] | None = None
    free: deque[_BounceSlot] = field(default_factory=deque)
    poisoned: bool = False


def _finalize_ring_resources(res: _RingResources) -> None:
    if res.pool is not None:
        try:
            res.pool.shutdown(wait=True)
        except BaseException:
            res.poisoned = True
        res.pool = None
    res.free.clear()
    if res.slots:
        while res.slots:
            slot = res.slots.pop()
            if slot.event is not None:
                try:
                    slot.event.synchronize()
                except BaseException:
                    res.poisoned = True
                slot.event = None
            del slot
    res.storage = None
    if res.backing is not None:
        res.backing.release_or_retain(keep_region=res.poisoned)
        res.backing = None


class PinnedBounceRing:
    """Fixed circular ring of pre-pinned staging slots for CPU->GPU and Disk->GPU streaming."""

    def __init__(
        self,
        *,
        slot_bytes: int = DEFAULT_BOUNCE_SLOT_BYTES,
        num_slots: int = DEFAULT_BOUNCE_SLOTS,
        min_bounce_bytes: int = DEFAULT_MIN_BOUNCE_BYTES,
        max_bytes: int | None = None,
        backend: HostRegistrationBackend | None = None,
        allocator: Callable[[int], AlignedRegion] | None = None,
    ) -> None:
        _validate_int(slot_bytes, "slot_bytes", minimum=1)
        if slot_bytes % mmap.PAGESIZE != 0:
            raise ValueError(f"slot_bytes must be a multiple of {mmap.PAGESIZE}, got {slot_bytes!r}.")
        _validate_int(num_slots, "num_slots", minimum=1)
        _validate_int(min_bounce_bytes, "min_bounce_bytes", minimum=0)
        if max_bytes is not None:
            _validate_int(max_bytes, "max_bytes", minimum=0)

        self._requested_slot_bytes = slot_bytes
        self._requested_num_slots = num_slots
        self._min_bounce_bytes = min_bounce_bytes
        self._max_bytes = max_bytes
        self._backend: HostRegistrationBackend = (
            RuntimeHostRegistration() if backend is None else backend
        )
        self._allocator: Callable[[int], AlignedRegion] = (
            host_memory.allocate_aligned_region if allocator is None else allocator
        )

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._state = _RingState.UNINITIALIZED
        self._permanently_disabled = False
        self._closing_active = False
        self._geom: _RingGeometry | None = None
        self._res = _RingResources()
        self._finalizer: weakref.finalize[[_RingResources], Any] | None = None
        self._checked_out = 0
        self._registration_failures = 0
        self._unregistration_failures = 0

    @property
    def is_initialized(self) -> bool:
        with self._lock:
            return self._state is _RingState.READY

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            if self._state not in (_RingState.READY, _RingState.CLOSING) or self._geom is None:
                return 0
            return self._geom.num_slots * self._geom.slot_bytes

    @property
    def slot_bytes(self) -> int:
        with self._lock:
            return self._geom.slot_bytes if self._geom is not None else self._requested_slot_bytes

    @property
    def num_slots(self) -> int:
        with self._lock:
            return self._geom.num_slots if self._geom is not None else self._requested_num_slots

    @property
    def registration_failures(self) -> int:
        with self._lock:
            return self._registration_failures

    @property
    def unregistration_failures(self) -> int:
        with self._lock:
            return self._unregistration_failures

    def _compute_geometry(self) -> tuple[int, int] | None:
        eff_slots = self._requested_num_slots
        eff_slot_bytes = self._requested_slot_bytes
        if self._max_bytes is None:
            return eff_slots, eff_slot_bytes
        if self._max_bytes < _MIN_BOUNCE_SLOT_BYTES:
            return None
        fit_slots = self._max_bytes // eff_slot_bytes
        min_preferred_slots = min(eff_slots, 2)
        if fit_slots >= min_preferred_slots:
            return min(eff_slots, fit_slots), eff_slot_bytes
        if min_preferred_slots >= 2 and self._max_bytes >= 2 * _MIN_BOUNCE_SLOT_BYTES:
            return 2, _align_down(self._max_bytes // 2, mmap.PAGESIZE)
        single_slot_bytes = min(eff_slot_bytes, _align_down(self._max_bytes, mmap.PAGESIZE))
        if single_slot_bytes < _MIN_BOUNCE_SLOT_BYTES:
            return None
        return 1, single_slot_bytes

    def _init_backing_off_lock(
        self,
    ) -> tuple[_RingGeometry, _RegisteredRegion, torch.UntypedStorage, list[_BounceSlot]] | None:
        geometry = self._compute_geometry()
        if geometry is None:
            with self._lock:
                self._permanently_disabled = True
            return None
        eff_slots, eff_slot_bytes = geometry
        total_bytes = eff_slots * eff_slot_bytes

        try:
            backing, status = _RegisteredRegion.allocate(
                total_bytes,
                allocator=self._allocator,
                pinned=True,
                backend=self._backend,
                context_label="pinned bounce ring",
            )
        except HostRegistrationError:
            with self._lock:
                self._registration_failures += 1
            raise
        if backing is None:
            with self._lock:
                if status in (_RegStatus.CAPACITY, _RegStatus.REFUSED):
                    self._registration_failures += 1
                if status is _RegStatus.REFUSED:
                    self._permanently_disabled = True
            return None

        mv: memoryview | None = None
        storage: torch.UntypedStorage | None = None
        slots: list[_BounceSlot] = []
        try:
            mv = memoryview(backing.region)
            with torch.inference_mode(False):
                storage = torch.frombuffer(mv, dtype=torch.uint8).untyped_storage()
                slots = [
                    _BounceSlot(
                        index=idx,
                        offset=idx * eff_slot_bytes,
                        size=eff_slot_bytes,
                        uint8_view=torch.empty(0, dtype=torch.uint8).set_(
                            storage, idx * eff_slot_bytes, (eff_slot_bytes,), (1,)
                        ),
                    )
                    for idx in range(eff_slots)
                ]
            geom = _RingGeometry(slot_bytes=eff_slot_bytes, num_slots=eff_slots, pointer=backing.pointer)
            return geom, backing, storage, slots
        except BaseException:
            slots.clear()
            storage = None
            if mv is not None:
                mv.release()
            if not backing.release_or_retain() and backing.registered:
                with self._lock:
                    self._unregistration_failures += 1
            raise

    def _acquire_init_slot(self) -> tuple[bool, _RingGeometry | None]:
        while True:
            with self._lock:
                if self._state in (_RingState.CLOSING, _RingState.CLOSED) or self._permanently_disabled:
                    return False, None
                if self._state is _RingState.READY:
                    return False, self._geom
                if self._state is _RingState.INITIALIZING:
                    self._cv.wait()
                    continue
                self._state = _RingState.INITIALIZING
                return True, None

    def _rollback_uncommitted_init(
        self,
        uncommitted_backing: _RegisteredRegion | None,
        fin: weakref.finalize[[_RingResources], Any] | None,
    ) -> None:
        if fin is not None:
            fin.detach()
        unreg_failed = False
        try:
            if (
                uncommitted_backing is not None
                and not uncommitted_backing.release_or_retain()
                and uncommitted_backing.registered
            ):
                unreg_failed = True
        finally:
            with self._lock:
                if unreg_failed:
                    self._unregistration_failures += 1
                if self._state is _RingState.INITIALIZING:
                    self._state = _RingState.UNINITIALIZED
                self._cv.notify_all()

    def _ensure_geometry(self) -> _RingGeometry | None:
        acquired, existing_geom = self._acquire_init_slot()
        if not acquired:
            return existing_geom

        committed = False
        built_res: tuple[_RingGeometry, _RegisteredRegion, torch.UntypedStorage, list[_BounceSlot]] | None = None
        uncommitted_backing: _RegisteredRegion | None = None
        storage: torch.UntypedStorage | None = None
        slots: list[_BounceSlot] | None = None
        free_deque: deque[_BounceSlot] | None = None
        fin: weakref.finalize[[_RingResources], Any] | None = None
        try:
            built_res = self._init_backing_off_lock()
            if built_res is None:
                return None
            geom, uncommitted_backing, storage, slots = built_res
            built_res = None
            free_deque = deque(slots)
            fin = weakref.finalize(self, _finalize_ring_resources, self._res)
            fin.atexit = False
            with self._lock:
                self._res.backing = uncommitted_backing
                self._res.storage = storage
                self._res.slots = slots
                self._res.free = free_deque
                self._geom = geom
                self._finalizer = fin
                self._state = _RingState.READY
                self._cv.notify_all()
                committed = True
                uncommitted_backing = None
                return geom
        finally:
            if not committed:
                if free_deque is not None:
                    free_deque.clear()
                    free_deque = None
                if slots is not None:
                    slots.clear()
                    slots = None
                storage = None
                built_res = None
                self._rollback_uncommitted_init(uncommitted_backing, fin)

    def _ensure_initialized(self) -> bool:
        return self._ensure_geometry() is not None

    def _checkout_slot(self, *, blocking: bool) -> _BounceSlot | None:
        with self._lock:
            while self._state is _RingState.READY and not self._res.free:
                if not blocking:
                    return None
                self._cv.wait()
            if self._state is not _RingState.READY:
                if not blocking:
                    return None
                raise RuntimeError("PinnedBounceRing is closing or closed.")
            slot = self._res.free.popleft()
            self._checked_out += 1
        if slot.event is not None:
            try:
                slot.event.synchronize()
            except BaseException:
                self._checkin_slot(slot)
                raise
            slot.event = None
        return slot

    def _checkin_slot(self, slot: _BounceSlot) -> None:
        with self._lock:
            self._checked_out -= 1
            if slot.poisoned:
                self._res.poisoned = True
                if self._state is _RingState.READY:
                    self._state = _RingState.CLOSING
            else:
                self._res.free.append(slot)
            self._cv.notify_all()

    def _get_pool(self, max_workers: int) -> ThreadPoolExecutor:
        with self._lock:
            if self._state is not _RingState.READY:
                raise RuntimeError("PinnedBounceRing is closing or closed.")
            if self._res.pool is None:
                self._res.pool = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="piper-offload-bounce",
                )
            return self._res.pool

    @staticmethod
    def _is_standard_strided(tensor: torch.Tensor) -> bool:
        return type(tensor) in (torch.Tensor, torch.nn.Parameter) and tensor.layout is torch.strided

    def _copy_slot_to_cuda(
        self,
        slot: _BounceSlot,
        dst_slice: torch.Tensor,
        src_start: int,
        src_end: int,
        device: torch.device,
        target_stream: torch.cuda.Stream,
        *,
        sync_immediately: bool,
    ) -> None:
        copy_enqueued = False
        try:
            with torch.cuda.device(device), torch.cuda.stream(target_stream):
                dst_slice.copy_(slot.uint8_view[src_start:src_end], non_blocking=True)
                copy_enqueued = True
                if sync_immediately:
                    target_stream.synchronize()
                else:
                    slot.event = cast(torch.cuda.Event, target_stream.record_event())
        except BaseException:
            if copy_enqueued and slot.event is None:
                try:
                    target_stream.synchronize()
                except BaseException:
                    slot.poisoned = True
                    raise
            raise

    def copy_pageable_to_device(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> bool:
        """Pipeline pageable CPU ``source`` through the bounce ring onto CUDA ``destination``."""
        if (
            not self._is_standard_strided(source)
            or not self._is_standard_strided(destination)
            or source.requires_grad
            or destination.requires_grad
            or source.device.type != "cpu"
            or destination.device.type != "cuda"
            or destination.shape != source.shape
            or destination.dtype is not source.dtype
            or not destination.is_contiguous()
            or not source.is_contiguous()
            or source.is_conj()
            or source.is_neg()
            or destination.is_conj()
            or destination.is_neg()
        ):
            return False
        if stream is not None and stream.device != destination.device:
            raise ValueError(
                f"CUDA stream device {stream.device} does not match destination device {destination.device}."
            )
        nbytes = source.nbytes
        if nbytes == 0 or nbytes < self._min_bounce_bytes:
            return False
        geom = self._ensure_geometry()
        if geom is None:
            return False

        src_u8 = _as_flat_uint8(source)
        dst_u8 = _as_flat_uint8(destination)
        target_stream = torch.cuda.current_stream(destination.device) if stream is None else stream
        yield_chunk = transfer_priority.background_yield_chunk_bytes(destination.device)
        step = geom.slot_bytes if yield_chunk is None else min(geom.slot_bytes, yield_chunk)

        for pos in range(0, nbytes, step):
            chunk = min(step, nbytes - pos)
            if yield_chunk is not None:
                transfer_priority.wait_for_foreground_if_background(destination.device)
            slot = self._checkout_slot(blocking=True)
            assert slot is not None
            try:
                slot.uint8_view[:chunk].copy_(src_u8[pos : pos + chunk], non_blocking=False)
                self._copy_slot_to_cuda(
                    slot,
                    dst_u8[pos : pos + chunk],
                    0,
                    chunk,
                    destination.device,
                    target_stream,
                    sync_immediately=(yield_chunk is not None),
                )
            finally:
                self._checkin_slot(slot)
        return True

    @staticmethod
    def _build_file_chunks(
        source: FileSlice,
        view_offset_bytes: int,
        nbytes: int,
        max_slot_bytes: int,
        sector_size: int,
    ) -> list[tuple[int, int, AlignedSpan]]:
        max_slot_aligned = _align_down(max_slot_bytes, sector_size)
        chunks: list[tuple[int, int, AlignedSpan]] = []
        dst_pos = 0
        while dst_pos < nbytes:
            file_off = source.offset + view_offset_bytes + dst_pos
            aligned_start = (file_off // sector_size) * sector_size
            max_end = aligned_start + max_slot_aligned
            chunk_len = min(nbytes - dst_pos, max_end - file_off)
            span = align_span(file_off, chunk_len, sector_size=sector_size)
            chunks.append((dst_pos, chunk_len, span))
            dst_pos += chunk_len
        return chunks

    def _read_into_slot(
        self,
        slot: _BounceSlot,
        reader: DirectFileReader,
        file_obj: io.BufferedReader,
        span: AlignedSpan,
    ) -> int:
        region = self._res.backing.region if self._res.backing is not None else None
        if region is None:
            raise RuntimeError("PinnedBounceRing is closing or closed.")
        with memoryview(region)[slot.offset : slot.offset + slot.size] as slot_mv:
            return reader.read_aligned(
                file_obj,
                span.aligned_offset,
                span.aligned_length,
                span.head_pad + span.data_length,
                slot_mv,
            )

    def _validate_stream_request(
        self,
        destination: torch.Tensor,
        source: FileSlice,
        stream: torch.cuda.Stream | None,
        view_offset_bytes: int,
        view_nbytes: int | None,
    ) -> int:
        if not self._is_standard_strided(destination) or destination.requires_grad:
            raise ValueError(
                "stream_file_slice_to_device requires a standard strided non-autograd tensor."
            )
        if destination.device.type != "cuda":
            raise ValueError(
                f"stream_file_slice_to_device requires a CUDA destination, got {destination.device}."
            )
        if stream is not None and stream.device != destination.device:
            raise ValueError(
                f"CUDA stream device {stream.device} does not match destination device {destination.device}."
            )
        if not destination.is_contiguous() or destination.is_conj() or destination.is_neg():
            raise ValueError("stream_file_slice_to_device requires a contiguous, non-conj/non-neg tensor.")
        _validate_int(view_offset_bytes, "view_offset_bytes", minimum=0)
        nbytes = source.length - view_offset_bytes if view_nbytes is None else view_nbytes
        if view_nbytes is not None:
            _validate_int(view_nbytes, "view_nbytes", minimum=0)
        if nbytes < 0 or view_offset_bytes + nbytes > source.length:
            raise ValueError(
                f"Requested byte span [{view_offset_bytes}, {view_offset_bytes + nbytes}) exceeds "
                f"FileSlice length {source.length}."
            )
        if nbytes != destination.nbytes:
            raise ValueError(
                f"Requested byte count {nbytes} does not match destination nbytes {destination.nbytes}."
            )
        return nbytes

    def _submit_prefetch_slot(
        self,
        reader: DirectFileReader,
        file_obj: io.BufferedReader,
        chunk: tuple[int, int, AlignedSpan],
        pool: ThreadPoolExecutor,
        *,
        blocking: bool,
    ) -> tuple[int, int, AlignedSpan, _BounceSlot, Future[int]] | None:
        slot = self._checkout_slot(blocking=blocking)
        if slot is None:
            return None
        dst_pos, chunk_len, span = chunk
        try:
            future = pool.submit(self._read_into_slot, slot, reader, file_obj, span)
        except BaseException:
            self._checkin_slot(slot)
            raise
        return dst_pos, chunk_len, span, slot, future

    def _drain_and_checkin_slot(self, slot: _BounceSlot, future: Future[int]) -> BaseException | None:
        interrupt: BaseException | None = None
        future.cancel()
        try:
            while not future.done():
                try:
                    future.result()
                except Exception:
                    pass
                except BaseException as exc:
                    if interrupt is None:
                        interrupt = exc
        finally:
            self._checkin_slot(slot)
        return interrupt

    def _fill_prefetch_window(
        self,
        inflight: deque[tuple[int, int, AlignedSpan, _BounceSlot, Future[int]]],
        chunks: list[tuple[int, int, AlignedSpan]],
        next_idx: int,
        limit: int,
        reader: DirectFileReader,
        file_obj: io.BufferedReader,
        pool: ThreadPoolExecutor,
        *,
        allow_first_blocking: bool = False,
    ) -> int:
        while next_idx < len(chunks) and len(inflight) < limit:
            blocking = allow_first_blocking and len(inflight) == 0
            submitted = self._submit_prefetch_slot(
                reader, file_obj, chunks[next_idx], pool, blocking=blocking
            )
            if submitted is None:
                break
            inflight.append(submitted)
            next_idx += 1
        return next_idx

    def _drain_inflight_queue(
        self,
        inflight: deque[tuple[int, int, AlignedSpan, _BounceSlot, Future[int]]],
        pending_interrupt: BaseException | None,
    ) -> BaseException | None:
        while inflight:
            _, _, _, pending_slot, pending_future = inflight.popleft()
            interrupt = self._drain_and_checkin_slot(pending_slot, pending_future)
            if interrupt is not None and pending_interrupt is None:
                pending_interrupt = interrupt
        return pending_interrupt

    def _run_pipelined_chunks(
        self,
        chunks: list[tuple[int, int, AlignedSpan]],
        dst_u8: torch.Tensor,
        source: FileSlice,
        reader: DirectFileReader,
        device: torch.device,
        target_stream: torch.cuda.Stream,
        prefetch_depth: int,
    ) -> None:
        pool = self._get_pool(prefetch_depth)
        inflight: deque[tuple[int, int, AlignedSpan, _BounceSlot, Future[int]]] = deque()
        next_idx = 0
        pending_interrupt: BaseException | None = None
        active_exc = False
        try:
            next_idx = self._fill_prefetch_window(
                inflight,
                chunks,
                next_idx,
                prefetch_depth + 1,
                reader,
                source.file,
                pool,
                allow_first_blocking=True,
            )
            while inflight:
                dst_pos, chunk_len, span, slot, future = inflight.popleft()
                try:
                    future.result()
                    next_idx = self._fill_prefetch_window(
                        inflight, chunks, next_idx, prefetch_depth, reader, source.file, pool
                    )
                    self._copy_slot_to_cuda(
                        slot,
                        dst_u8[dst_pos : dst_pos + chunk_len],
                        span.head_pad,
                        span.head_pad + span.data_length,
                        device,
                        target_stream,
                        sync_immediately=False,
                    )
                finally:
                    interrupt = self._drain_and_checkin_slot(slot, future)
                    if interrupt is not None and pending_interrupt is None:
                        pending_interrupt = interrupt

                if pending_interrupt is not None:
                    break
                if next_idx < len(chunks) and not inflight:
                    next_idx = self._fill_prefetch_window(
                        inflight, chunks, next_idx, 1, reader, source.file, pool, allow_first_blocking=True
                    )
        except BaseException:
            active_exc = True
            raise
        finally:
            pending_interrupt = self._drain_inflight_queue(inflight, pending_interrupt)
            if not active_exc and pending_interrupt is not None:
                raise pending_interrupt

    def _run_serial_chunks(
        self,
        chunks: list[tuple[int, int, AlignedSpan]],
        dst_u8: torch.Tensor,
        source: FileSlice,
        reader: DirectFileReader,
        device: torch.device,
        target_stream: torch.cuda.Stream,
        *,
        yielding: bool,
    ) -> None:
        for dst_pos, chunk_len, span in chunks:
            if yielding:
                transfer_priority.wait_for_foreground_if_background(device)
            slot = self._checkout_slot(blocking=True)
            assert slot is not None
            try:
                self._read_into_slot(slot, reader, source.file, span)
                self._copy_slot_to_cuda(
                    slot,
                    dst_u8[dst_pos : dst_pos + chunk_len],
                    span.head_pad,
                    span.head_pad + span.data_length,
                    device,
                    target_stream,
                    sync_immediately=yielding,
                )
            finally:
                self._checkin_slot(slot)

    def stream_file_slice_to_device(
        self,
        destination: torch.Tensor,
        source: FileSlice,
        *,
        reader: DirectFileReader,
        stream: torch.cuda.Stream | None = None,
        view_offset_bytes: int = 0,
        view_nbytes: int | None = None,
    ) -> bool:
        """Stream checkpoint bytes from ``source`` to CUDA ``destination`` via the bounce ring."""
        nbytes = self._validate_stream_request(destination, source, stream, view_offset_bytes, view_nbytes)
        if nbytes == 0:
            return True
        geom = self._ensure_geometry()
        if geom is None:
            return False
        sector = reader.sector_size(source.file)
        if geom.slot_bytes < 2 * sector or (geom.pointer % sector) != 0 or (geom.slot_bytes % sector) != 0:
            return False

        yield_chunk = transfer_priority.background_yield_chunk_bytes(destination.device)
        max_chunk = geom.slot_bytes if yield_chunk is None else max(2 * sector, min(geom.slot_bytes, yield_chunk))
        chunks = self._build_file_chunks(source, view_offset_bytes, nbytes, max_chunk, sector)
        dst_u8 = _as_flat_uint8(destination)
        target_stream = torch.cuda.current_stream(destination.device) if stream is None else stream
        if yield_chunk is not None or len(chunks) == 1 or geom.num_slots < 2:
            self._run_serial_chunks(
                chunks,
                dst_u8,
                source,
                reader,
                destination.device,
                target_stream,
                yielding=(yield_chunk is not None),
            )
            return True

        prefetch_depth = min(2, geom.num_slots - 1)
        self._run_pipelined_chunks(
            chunks,
            dst_u8,
            source,
            reader,
            destination.device,
            target_stream,
            prefetch_depth,
        )
        return True

    @staticmethod
    def _shutdown_ring_pool(pool: ThreadPoolExecutor | None) -> BaseException | None:
        if pool is None:
            return None
        interrupt: BaseException | None = None
        while True:
            try:
                pool.shutdown(wait=True)
                return interrupt
            except KeyboardInterrupt as exc:
                if interrupt is None:
                    interrupt = exc
            except BaseException as exc:
                return exc if interrupt is None else interrupt

    @staticmethod
    def _sync_ring_slots(slots: list[_BounceSlot], *, poisoned: bool) -> None:
        for slot in slots:
            if slot.event is not None:
                if not poisoned:
                    slot.event.synchronize()
                slot.event = None
        slots.clear()

    def _teardown_ring_resources(
        self,
        pool: ThreadPoolExecutor | None,
        slots: list[_BounceSlot],
        backing: _RegisteredRegion | None,
        *,
        poisoned: bool,
    ) -> None:
        interrupt = self._shutdown_ring_pool(pool)
        if interrupt is not None and not isinstance(interrupt, KeyboardInterrupt):
            poisoned = True
        self._sync_ring_slots(slots, poisoned=poisoned)

        if backing is not None and not poisoned:
            try:
                backing.unregister_only(suppress_errors=False)
            except Exception:
                with self._lock:
                    self._unregistration_failures += 1
                raise

        with self._lock:
            finalizer, self._finalizer = self._finalizer, None
            if finalizer is not None:
                finalizer.detach()
            if self._res.slots is not None:
                self._res.slots.clear()
                self._res.slots = None
            self._res.free.clear()
            self._res.storage = None
            self._res.backing = None
            self._geom = None
            self._state = _RingState.CLOSED
            self._cv.notify_all()

        if backing is not None:
            backing.release_or_retain(keep_region=poisoned)
        if interrupt is not None:
            raise interrupt

    def close(self) -> None:
        """Wait for active transfers, sync events, unregister, and free the bounce ring."""
        with self._lock:
            while self._state is _RingState.INITIALIZING or self._closing_active:
                self._cv.wait()
            if self._state is _RingState.CLOSED:
                return
            if self._state is _RingState.UNINITIALIZED:
                self._state = _RingState.CLOSED
                self._cv.notify_all()
                return
            self._state = _RingState.CLOSING
            self._closing_active = True
            self._cv.notify_all()

        try:
            with self._lock:
                while self._checked_out > 0:
                    self._cv.wait()
                pool, self._res.pool = self._res.pool, None
                slots = list(self._res.slots or ())
                poisoned = self._res.poisoned
                backing = self._res.backing
            self._teardown_ring_resources(pool, slots, backing, poisoned=poisoned)
        finally:
            with self._lock:
                self._closing_active = False
                self._cv.notify_all()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "DEFAULT_BOUNCE_SLOTS",
    "DEFAULT_BOUNCE_SLOT_BYTES",
    "DEFAULT_MIN_BOUNCE_BYTES",
    "DEFAULT_SLAB_BYTES",
    "ArenaAllocation",
    "HostArenaStats",
    "HostSlabArena",
    "PinnedBounceRing",
]
