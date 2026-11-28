"""Layer 2 global frozen-KV slab manager over shared Layer 1 hardware substrate.

``KvChunkPool`` owns fixed-size quantized or plain KV/state chunks allocated on
the descending high end of a :class:`Uint8DualArena` alongside dedicated
:class:`TypedSidePool` allocations described by a format-agnostic
:class:`SlotSchema`.

It never imports ``piper_kernels`` or embeds attention math: callers bind
``PiperKVChunkPool`` (or another kernel descriptor) once over the persistent
``arena_views`` and ``side_views`` properties, freeze new immutable chunks via
:meth:`produce`, read prefixes or full contexts (including ``>> VRAM`` streaming
with ``allow_split=True``) via :meth:`waves`, and drop retired or compacted
chunks via :meth:`discard`. Frozen chunks spill at most once to host RAM
(``PinManager`` Tier 2a pinned slabs with Tier 2b synchronous fallback on CUDA,
never disk) and never drop silently across ranks under host RAM exhaustion.
"""

from __future__ import annotations

import contextlib
import math
import mmap
import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Generator, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Self

import torch

from ._slab_pool_common import (
    _build_pool_views_common,
    _build_side_pools,
    _make_device_int_table,
    _merge_error,
    _PendingWaveRelease,
    _PoolViews,
    _release_slot_stream_lease,
    _run_until_complete,
    _setup_transfer_context,
    _sync_interrupt_safe,
    _validate_optional_positive_int,
    _validate_slab_pool_base_args,
    _WaveLease,
    _WaveReleaseTracker,
)
from .device_transfer import DeviceTransferContext
from .dual_arena import (
    ArenaCapacityError,
    HighSlotLayout,
    SlotSchema,
    TypedSidePool,
    Uint8DualArena,
    _align_up,
    _element_size,
)
from .host_arena import ArenaAllocation
from .pin_manager import PinManager
from .stream_lease import StreamLease, _record_event
from .vram_budget import VramBudget


class KvChunkLocation(Enum):
    """Current physical location of one tracked frozen chunk."""

    GPU_DIRTY = "gpu_dirty"
    GPU_SPILLING = "gpu_spilling"
    GPU_CLEAN = "gpu_clean"
    HOST_ONLY = "host_only"


class _SlotPhase(Enum):
    EMPTY = "empty"
    PRODUCING = "producing"
    STAGING = "staging"
    READY = "ready"
    RELOCATING = "relocating"
    RETIRING = "retiring"


class _SpillState(Enum):
    NONE = "none"
    ALLOCATING = "allocating"
    IN_FLIGHT = "in_flight"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class KvBatch:
    """Batch descriptor yielded by :meth:`KvChunkPool.produce` and :meth:`KvChunkPool.waves`.

    Attributes:
        index: Zero-based batch/wave index (``0`` for ``produce`` or single-wave reads).
        keys: Request-ordered tuple of chunk keys and ``None`` sentinels.
        slot_ids: ``int32`` device tensor of mapped high arena slots (``-1`` for ``None``).
        active_slots_hint: Count of non-sentinel active chunk slots in this batch.
        is_hit_wave: ``True`` when the wave serves only already-resident GPU hits.
        arena_views: Mapping from field name to invariant ``[max_slots, *shape]`` uint8 views.
        side_views: Mapping from field name to invariant ``[max_slots, *shape]`` typed views.
        arena_bytes: Underlying 1-D ``uint8`` device storage tensor.
        slot_offsets: Mapping from field name to ``int64[max_slots]`` arena byte offsets.
    """

    index: int
    keys: tuple[Hashable | None, ...]
    slot_ids: torch.Tensor
    active_slots_hint: int
    is_hit_wave: bool
    arena_views: Mapping[str, torch.Tensor]
    side_views: Mapping[str, torch.Tensor]
    arena_bytes: torch.Tensor
    slot_offsets: Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class KvPoolStats:
    """Point-in-time VRAM, host-RAM, and transfer counters for :class:`KvChunkPool`.

    Attributes:
        max_slots: Configured capacity bound in high arena slots.
        allocated_gpu_slots: Currently allocated high arena slots tracked by this pool.
        resident_gpu_chunks: Tracked chunks currently residing in GPU slots.
        dirty_chunks: GPU-resident chunks without an immutable host spill backing.
        clean_chunks: GPU-resident chunks that also retain an immutable host spill backing.
        host_only_chunks: Chunks residing only in host slabs (``slot is None``).
        in_flight_d2h_spills: Resident chunks with asynchronous D2H spills still running.
        host_bytes: Reserved host slab span bytes (including active and pending-retire slabs).
        hits: Cumulative key accesses already ready on GPU when partitioned in ``waves()``.
        misses: Cumulative key accesses requiring host-to-GPU staging in ``waves()``.
        reloads: Cumulative host-to-GPU chunk reloads committed.
        spills: Cumulative write-once D2H spills committed.
        evictions: Cumulative GPU slot evictions releasing a chunk back to host-only residency.
        relocations: Cumulative peer-frontier D2D slot relocations.
        discards: Cumulative chunks detached via ``discard()``.
        waves_executed: Cumulative batches yielded across ``waves()`` with ``record_stats=True``.
    """

    max_slots: int
    allocated_gpu_slots: int
    resident_gpu_chunks: int
    dirty_chunks: int
    clean_chunks: int
    host_only_chunks: int
    in_flight_d2h_spills: int
    host_bytes: int
    hits: int
    misses: int
    reloads: int
    spills: int
    evictions: int
    relocations: int
    discards: int
    waves_executed: int


@dataclass(slots=True)
class _KvSlotState:
    slot_id: int
    key: Hashable | None = None
    phase: _SlotPhase = _SlotPhase.EMPTY
    active_readers: int = 0
    lease: StreamLease | None = None
    lease_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(eq=False, slots=True)
class _PendingStreamToken:
    stream: torch.cuda.Stream


@dataclass(slots=True)
class _ChunkRecord:
    key: Hashable
    slot: _KvSlotState | None = None
    host_alloc: ArenaAllocation | None = None
    spill_state: _SpillState = _SpillState.NONE
    spill_attempt_id: int = 0
    staging_waves: int = 0
    host_readers: int = 0
    pending_h2d_events: list[torch.cuda.Event] = field(default_factory=list)
    pending_h2d_streams: list[_PendingStreamToken] = field(default_factory=list)


@dataclass(eq=False, slots=True)
class _PendingHostRetire:
    slot: _KvSlotState | None
    host_alloc: ArenaAllocation | None
    h2d_events: tuple[torch.cuda.Event, ...] = ()
    fallback_streams: tuple[torch.cuda.Stream, ...] = ()


@dataclass(slots=True)
class _ReloadBatchBuild:
    claimed: list[tuple[_ChunkRecord, _KvSlotState, ArenaAllocation]] = field(default_factory=list)
    prebound: list[tuple[_ChunkRecord, _KvSlotState]] = field(default_factory=list)
    acquired_slots: list[_KvSlotState] = field(default_factory=list)
    taken_recs: list[_ChunkRecord] = field(default_factory=list)
    pinned_recs: list[_ChunkRecord] = field(default_factory=list)
    copy_stream: torch.cuda.Stream | None = None
    stage_started: bool = False
    stage_event: torch.cuda.Event | None = None
    h2d_tracked: bool = False


@dataclass(slots=True)
class _WaveLoopState:
    held_lease: _WaveLease
    next_lease: _WaveLease
    pending_commit: list[tuple[_ChunkRecord, _KvSlotState, ArenaAllocation]] = field(default_factory=list)
    next_staged: list[tuple[_ChunkRecord, _KvSlotState, ArenaAllocation]] = field(default_factory=list)
    next_prebound: list[tuple[_ChunkRecord, _KvSlotState]] = field(default_factory=list)
    unyielded_miss_claims: dict[int, _ChunkRecord] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _HostChunkLayout:
    drawer_bytes: int
    side_fields: tuple[tuple[str, torch.dtype, tuple[int, ...], int, int], ...]
    total_bytes: int


type _DetachedDiscardItem = tuple[
    _KvSlotState | None,
    ArenaAllocation | None,
    bool,
    tuple[torch.cuda.Event, ...],
    tuple[_PendingStreamToken, ...],
]


def _build_host_chunk_layout(schema: SlotSchema) -> _HostChunkLayout:
    offset = _align_up(schema.drawer_bytes, 16)
    side_entries: list[tuple[str, torch.dtype, tuple[int, ...], int, int]] = []
    for sf in schema.side_fields:
        nbytes = math.prod(sf.shape) * _element_size(sf.dtype)
        side_entries.append((sf.name, sf.dtype, sf.shape, offset, nbytes))
        offset = _align_up(offset + nbytes, 16)
    return _HostChunkLayout(
        drawer_bytes=schema.drawer_bytes,
        side_fields=tuple(side_entries),
        total_bytes=max(16, offset),
    )


def _validate_spill_watermark(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"spill_watermark must be a float in (0, 1] or None; got {value!r}.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0 or resolved > 1.0:
        raise ValueError(f"spill_watermark must lie in (0, 1]; got {value!r}.")
    return resolved


def _validate_chunk_keys(keys: Sequence[Hashable | None], *, allow_none: bool) -> tuple[Hashable | None, ...]:
    if isinstance(keys, (str, bytes)):
        raise TypeError("keys must be a sequence of chunk keys, not a string or bytes.")
    resolved: list[Hashable | None] = []
    for key in keys:
        if key is None:
            if not allow_none:
                raise ValueError("None sentinel keys are only supported in waves().")
            resolved.append(None)
            continue
        hash(key)
        resolved.append(key)
    return tuple(resolved)


def _synchronize_slot_before_free(slot: _KvSlotState) -> None:
    if slot.lease is None:
        return
    lease = slot.lease
    with slot.lease_lock:
        _sync_interrupt_safe(lease.synchronize_and_reset)


def _query_pending_events_and_streams(
    events: Sequence[torch.cuda.Event],
    streams: Sequence[_PendingStreamToken],
) -> tuple[tuple[torch.cuda.Event, ...], tuple[torch.cuda.Stream, ...]]:
    pending_events = tuple(ev for ev in events if not ev.query())
    pending_streams = tuple(tok.stream for tok in streams if not tok.stream.query())
    return pending_events, pending_streams


class KvChunkPool:
    """Global shape-class manager for frozen immutable KV/state chunks on high arena slots.

    Manages opaque-keyed chunks on the descending high end of a
    :class:`Uint8DualArena` alongside :class:`TypedSidePool` metadata storage,
    exposing invariant top-level ``arena_views`` and ``side_views`` suitable for
    wrapping once with :class:`piper_kernels.PiperKVChunkPool` under
    ``torch.compile(fullgraph=True)``.

    Internal Lock Ordering:
        ``_close_lock`` -> ``_lock`` -> ``arena._lock``.
        ``slot.lease_lock`` is disjoint from ``_lock`` and ordered strictly
        before ``arena._lock`` (never acquire ``_lock`` while holding
        ``slot.lease_lock``; never acquire two slots' ``lease_lock``s at once).
        ``PinManager.allocate_host_slab``, ``ArenaAllocation.release``, CUDA
        synchronizations, and external reclaimer calls always run outside ``_lock``.
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
        spill_watermark: float | None = 0.8,
        max_host_bytes: int | None = None,
    ) -> None:
        """Attach a descending KV/state chunk pool to the high end of ``arena``.

        Args:
            arena: Backing :class:`Uint8DualArena` owning raw ``uint8`` device storage.
            schema: :class:`SlotSchema` describing arena drawers and typed side fields.
            max_slots: Optional slot limit <= what ``arena`` can hold at the configured stride.
            budget: Optional :class:`VramBudget` for typed side pools; defaults to
                ``arena.budget`` when omitted.
            transfer_context: Optional CUDA :class:`DeviceTransferContext` shared across
                components on ``arena.device``.
            pin_manager: Optional :class:`PinManager`; defaults to ``host_pin_manager``.
            spill_watermark: Fraction of ``max_slots`` in ``(0, 1]`` triggering background
                write-once host spills after ``produce()``; ``None`` disables auto-spills.
            max_host_bytes: Optional ceiling (measured in page-rounded host slab span
                bytes) for host RAM retained by this pool; must be positive or ``None``.
        """
        resolved_max, resolved_pin, effective_budget = _validate_slab_pool_base_args(
            "KvChunkPool",
            arena,
            schema,
            max_slots=max_slots,
            budget=budget,
            pin_manager=pin_manager,
        )
        resolved_wm = _validate_spill_watermark(spill_watermark)
        resolved_host_max = _validate_optional_positive_int(max_host_bytes, name="max_host_bytes")
        device = arena.device
        self._lock = threading.Lock()
        self._op_cond = threading.Condition(self._lock)
        self._close_lock = threading.Lock()
        self._arena = arena
        self._schema = schema
        self._device = device
        self._pin_manager = resolved_pin
        self._spill_watermark = resolved_wm
        self._max_host_bytes = resolved_host_max
        self._host_layout = _build_host_chunk_layout(schema)
        self._host_span_bytes = _align_up(
            self._host_layout.total_bytes, max(256, mmap.PAGESIZE)
        )
        self._layout: HighSlotLayout | None = None
        self._max_slots = 0
        self._configured_fresh_layout = False
        self._transfer_context: DeviceTransferContext | None = None
        self._owns_transfer_context = False
        self._borrowed_transfer_context = False
        self._side_pools: dict[str, TypedSidePool] = {}
        self._views: _PoolViews | None = None
        self._init_tracking_state()
        self._reclaim_cb = self._reclaim_high_frontier

        try:
            layout, configured_fresh = arena._claim_high_owner(
                self._reclaim_cb, schema.drawer_bytes, resolved_max
            )
            self._layout = layout
            self._max_slots = layout.max_slots
            self._configured_fresh_layout = configured_fresh

            ctx, owns_ctx, borrowed_ctx = _setup_transfer_context(
                "KvChunkPool", device, transfer_context
            )
            self._transfer_context = ctx
            self._owns_transfer_context = owns_ctx
            self._borrowed_transfer_context = borrowed_ctx

            self._side_pools = _build_side_pools(
                schema,
                layout.max_slots,
                device,
                effective_budget,
                reservation_prefix="kv_side_",
            )
            self._views = _build_pool_views_common(
                arena,
                schema,
                max_slots=layout.max_slots,
                slot_stride=layout.slot_stride,
                base_offset=layout.base_offset,
                pool_view_fn=lambda shape, off: arena.high_pool_view(shape, field_offset=off),
                side_pools=self._side_pools,
                device=device,
            )
        except BaseException as exc:
            cleanup_err = self._teardown_resources()
            merged = _merge_error(exc, cleanup_err)
            if merged is not None and merged is not exc:
                raise merged from exc
            raise

    def _init_tracking_state(self) -> None:
        self._chunks: OrderedDict[Hashable, _ChunkRecord] = OrderedDict()
        self._producing_keys: set[Hashable] = set()
        self._slots: dict[int, _KvSlotState] = {}
        self._pending_host_retires: deque[_PendingHostRetire] = deque()
        self._wave_releases = _WaveReleaseTracker(self._device, self._lock, self._op_cond)

        self._inflight_arena_allocs = 0
        self._inflight_arena_frees = 0
        self._inflight_host_allocs = 0
        self._active_spill_ops = 0
        self._next_spill_attempt_id = 0
        self._live_host_allocs = 0
        self._inflight_host_retire_ops = 0
        self._active_wave_sessions = 0
        self._active_produce_sessions = 0
        self._closing = False
        self._closed = False

        self._hits = 0
        self._misses = 0
        self._reloads = 0
        self._spills = 0
        self._evictions = 0
        self._relocations = 0
        self._discards = 0
        self._waves_executed = 0

    @property
    def schema(self) -> SlotSchema:
        """Configured :class:`SlotSchema` describing arena drawers and side fields."""
        return self._schema

    @property
    def layout(self) -> HighSlotLayout:
        """Descending :class:`HighSlotLayout` registered with the backing arena."""
        assert self._layout is not None
        return self._layout

    @property
    def max_slots(self) -> int:
        """Configured capacity bound in high arena slots."""
        return self._max_slots

    @property
    def device(self) -> torch.device:
        """Torch device hosting arena drawers and side pools."""
        return self._device

    @property
    def is_open(self) -> bool:
        """Whether the pool and its backing arena remain open."""
        with self._lock:
            return not self._closed and not self._closing and self._arena.is_open

    @property
    def arena_views(self) -> Mapping[str, torch.Tensor]:
        """Invariant strided ``uint8`` views ``[max_slots, *shape]`` on high arena slots."""
        return self._require_views().arena_views

    @property
    def side_views(self) -> Mapping[str, torch.Tensor]:
        """Invariant strided typed views ``[max_slots, *shape]`` for metadata fields."""
        return self._require_views().side_views

    @property
    def arena_bytes(self) -> torch.Tensor:
        """Shared 1-D ``torch.uint8`` view over the backing :class:`Uint8DualArena`."""
        return self._require_views().arena_bytes

    @property
    def slot_offsets(self) -> Mapping[str, torch.Tensor]:
        """Invariant ``int64[max_slots]`` byte offsets inside :attr:`arena_bytes`."""
        return self._require_views().slot_offsets

    def _require_views(self) -> _PoolViews:
        views = self._views
        if self._closed or self._closing or views is None:
            raise RuntimeError("KvChunkPool is closed.")
        return views

    def _require_open_unlocked(self) -> None:
        if self._closed or self._closing or not self._arena.is_open:
            raise RuntimeError("KvChunkPool is closed.")

    def _ensure_copy_stream(self) -> torch.cuda.Stream:
        assert self._transfer_context is not None
        return self._transfer_context.ensure_stream()

    def _release_tracked_host_alloc(self, alloc: ArenaAllocation | None) -> None:
        if alloc is None or alloc.released:
            return
        try:
            alloc.release()
        finally:
            if alloc.released:
                with self._lock:
                    self._live_host_allocs = max(0, self._live_host_allocs - 1)
                    self._op_cond.notify_all()

    def _release_or_defer_host_allocs(self, allocs: Sequence[ArenaAllocation]) -> None:
        try:
            for alloc in allocs:
                self._release_tracked_host_alloc(alloc)
        except BaseException:
            rem = [
                _PendingHostRetire(slot=None, host_alloc=ha)
                for ha in allocs
                if not ha.released
            ]
            if rem:
                with self._lock:
                    self._pending_host_retires.extendleft(reversed(rem))
                    self._op_cond.notify_all()
            raise

    def _poll_spilling_chunks(self) -> None:
        if self._device.type != "cuda":
            return
        with self._lock:
            if self._closed or self._closing:
                return
            candidates = [
                (rec, slot)
                for slot in self._slots.values()
                if slot.key is not None
                and (rec := self._chunks.get(slot.key)) is not None
                and rec.slot is slot
                and rec.spill_state is _SpillState.IN_FLIGHT
                and rec.host_alloc is not None
            ]
        finished: list[_ChunkRecord] = []
        for rec, slot in candidates:
            if slot.lease is None:
                finished.append(rec)
                continue
            with slot.lease_lock:
                if slot.lease.poll_d2h():
                    finished.append(rec)
        if finished:
            with self._lock:
                for rec in finished:
                    if rec.host_alloc is not None and rec.spill_state is _SpillState.IN_FLIGHT:
                        rec.spill_state = _SpillState.COMPLETE
                self._op_cond.notify_all()

    @staticmethod
    def _poll_host_retire_item(item: _PendingHostRetire) -> bool:
        if item.slot is not None and item.slot.lease is not None:
            with item.slot.lease_lock:
                if not item.slot.lease.poll_d2h():
                    return False
                item.slot.lease.discard_dirty()
        return all(ev.query() for ev in item.h2d_events) and all(
            st.query() for st in item.fallback_streams
        )

    def _commit_polled_retires_unlocked(
        self,
        done: Sequence[_PendingHostRetire],
        still_pending: Sequence[_PendingHostRetire],
        *,
        queue_done_hosts: bool,
    ) -> list[ArenaAllocation]:
        hosts_to_free: list[ArenaAllocation] = []
        for item in done:
            slot = item.slot
            if (
                slot is not None
                and not self._closed
                and not self._closing
                and self._slots.get(slot.slot_id) is slot
                and slot.phase is _SlotPhase.RETIRING
            ):
                slot.phase = _SlotPhase.EMPTY
            if item.host_alloc is not None and not item.host_alloc.released:
                if queue_done_hosts:
                    self._pending_host_retires.appendleft(
                        _PendingHostRetire(slot=None, host_alloc=item.host_alloc)
                    )
                else:
                    hosts_to_free.append(item.host_alloc)
        if still_pending:
            self._pending_host_retires.extendleft(reversed(still_pending))
        self._op_cond.notify_all()
        return hosts_to_free

    def _reap_pending_host_retires(self) -> None:
        with self._lock:
            if not self._pending_host_retires or self._closed or self._closing:
                return
            batch = list(self._pending_host_retires)
            self._pending_host_retires.clear()
            self._inflight_host_retire_ops += 1
        done: list[_PendingHostRetire] = []
        still_pending: list[_PendingHostRetire] = []
        hosts_to_free: list[ArenaAllocation] = []
        try:
            for idx, item in enumerate(batch):
                try:
                    if self._poll_host_retire_item(item):
                        done.append(item)
                    else:
                        still_pending.append(item)
                except BaseException:
                    still_pending.extend(batch[idx:])
                    with self._lock:
                        self._commit_polled_retires_unlocked(
                            done, still_pending, queue_done_hosts=True
                        )
                    raise
            with self._lock:
                hosts_to_free = self._commit_polled_retires_unlocked(
                    done, still_pending, queue_done_hosts=False
                )
            if hosts_to_free:
                self._release_or_defer_host_allocs(hosts_to_free)
        finally:
            with self._lock:
                self._inflight_host_retire_ops = max(0, self._inflight_host_retire_ops - 1)
                self._op_cond.notify_all()

    def _reap_all_background(self) -> None:
        self._wave_releases.reap(lambda: self._closed or self._closing)
        self._reap_pending_host_retires()
        self._poll_spilling_chunks()

    def _chunk_location_unlocked(self, rec: _ChunkRecord) -> KvChunkLocation:
        if rec.slot is None:
            return KvChunkLocation.HOST_ONLY
        if rec.host_alloc is None:
            return KvChunkLocation.GPU_DIRTY
        if rec.spill_state is _SpillState.IN_FLIGHT:
            return KvChunkLocation.GPU_SPILLING
        return KvChunkLocation.GPU_CLEAN

    def contains(self, key: Hashable) -> bool:
        """Return whether ``key`` is currently tracked on GPU or in host RAM."""
        hash(key)
        with self._lock:
            self._require_open_unlocked()
            return key in self._chunks

    def __contains__(self, key: object) -> bool:
        try:
            hash(key)
        except TypeError:
            return False
        with self._lock:
            if self._closed or self._closing:
                return False
            return key in self._chunks

    def location(self, key: Hashable) -> KvChunkLocation | None:
        """Return the current :class:`KvChunkLocation` for ``key``, or ``None`` if absent."""
        hash(key)
        self._reap_all_background()
        with self._lock:
            self._require_open_unlocked()
            rec = self._chunks.get(key)
            if rec is None:
                return None
            return self._chunk_location_unlocked(rec)

    def stats(self) -> KvPoolStats:
        """Return point-in-time VRAM, host-RAM, and transfer statistics."""
        self._reap_all_background()
        with self._lock:
            dirty = 0
            clean = 0
            spilling = 0
            host_only = 0
            resident_gpu = 0
            for rec in self._chunks.values():
                loc = self._chunk_location_unlocked(rec)
                if loc is KvChunkLocation.HOST_ONLY:
                    host_only += 1
                else:
                    resident_gpu += 1
                    if loc is KvChunkLocation.GPU_DIRTY:
                        dirty += 1
                    elif loc is KvChunkLocation.GPU_SPILLING:
                        spilling += 1
                    else:
                        clean += 1
            return KvPoolStats(
                max_slots=self._max_slots,
                allocated_gpu_slots=len(self._slots),
                resident_gpu_chunks=resident_gpu,
                dirty_chunks=dirty,
                clean_chunks=clean,
                host_only_chunks=host_only,
                in_flight_d2h_spills=spilling,
                host_bytes=self._live_host_allocs * self._host_span_bytes,
                hits=self._hits,
                misses=self._misses,
                reloads=self._reloads,
                spills=self._spills,
                evictions=self._evictions,
                relocations=self._relocations,
                discards=self._discards,
                waves_executed=self._waves_executed,
            )

    def _host_views_for_alloc(
        self, alloc: ArenaAllocation
    ) -> tuple[torch.Tensor, tuple[tuple[str, torch.Tensor], ...]]:
        raw = alloc.as_uint8_tensor()
        drawer = raw[: self._host_layout.drawer_bytes]
        sides = tuple(
            (name, raw[off : off + nbytes].view(dtype).reshape(shape))
            for name, dtype, shape, off, nbytes in self._host_layout.side_fields
        )
        return drawer, sides

    def _copy_slot_d2h(self, slot_id: int, host_alloc: ArenaAllocation, *, non_blocking: bool) -> None:
        gpu_drawer = self._arena.high_slot_tensor(slot_id)[: self._host_layout.drawer_bytes]
        host_drawer, host_sides = self._host_views_for_alloc(host_alloc)
        host_drawer.copy_(gpu_drawer, non_blocking=non_blocking)
        for name, host_side in host_sides:
            host_side.copy_(self._side_pools[name].slot(slot_id), non_blocking=non_blocking)

    def _copy_host_to_slot(
        self, slot_id: int, host_alloc: ArenaAllocation, *, non_blocking: bool
    ) -> None:
        gpu_drawer = self._arena.high_slot_tensor(slot_id)[: self._host_layout.drawer_bytes]
        host_drawer, host_sides = self._host_views_for_alloc(host_alloc)
        gpu_drawer.copy_(host_drawer, non_blocking=non_blocking)
        for name, host_side in host_sides:
            self._side_pools[name].slot(slot_id).copy_(host_side, non_blocking=non_blocking)

    def _copy_slot_d2d(self, src_id: int, dst_id: int, *, non_blocking: bool) -> None:
        self._arena.high_slot_tensor(dst_id)[: self._host_layout.drawer_bytes].copy_(
            self._arena.high_slot_tensor(src_id)[: self._host_layout.drawer_bytes],
            non_blocking=non_blocking,
        )
        for side_pool in self._side_pools.values():
            side_pool.slot(dst_id).copy_(side_pool.slot(src_id), non_blocking=non_blocking)

    def _make_slot_state(self, slot_id: int, *, phase: _SlotPhase) -> _KvSlotState:
        alloc_stream = torch.cuda.current_stream(self._device) if self._device.type == "cuda" else None
        return _KvSlotState(
            slot_id=slot_id,
            phase=phase,
            lease=None if alloc_stream is None else StreamLease(alloc_stream),
        )

    def _free_slot_to_arena(self, slot: _KvSlotState, *, sync_lease: bool = True) -> None:
        if sync_lease:
            _synchronize_slot_before_free(slot)
        popped = False
        with self._lock:
            if self._slots.get(slot.slot_id) is slot:
                del self._slots[slot.slot_id]
                self._inflight_arena_frees += 1
                popped = True
        if not popped:
            return
        freed = False
        err: BaseException | None = None
        try:
            if self._arena.is_open:
                freed, err = _run_until_complete(lambda: self._arena.free_high_slot(slot.slot_id))
            else:
                freed = True
        finally:
            with self._lock:
                if not freed and slot.slot_id not in self._slots:
                    slot.key = None
                    slot.phase = _SlotPhase.EMPTY
                    self._slots[slot.slot_id] = slot
                self._inflight_arena_frees = max(0, self._inflight_arena_frees - 1)
                self._op_cond.notify_all()
        if err is not None:
            raise err

    def _free_slot_or_retire(self, slot: _KvSlotState, *, sync_lease: bool) -> None:
        try:
            self._free_slot_to_arena(slot, sync_lease=sync_lease)
        except BaseException:
            with self._lock:
                if self._slots.get(slot.slot_id) is slot and slot.phase is not _SlotPhase.EMPTY:
                    slot.key = None
                    slot.phase = _SlotPhase.RETIRING
                    self._pending_host_retires.append(_PendingHostRetire(slot=slot, host_alloc=None))
                    self._op_cond.notify_all()
            raise

    def _alloc_new_slot_from_arena(
        self,
        *,
        allow_reclaim: bool,
        min_slot_id: int | None = None,
        phase: _SlotPhase,
    ) -> _KvSlotState | None:
        if self._max_slots <= 0:
            return None
        try:
            allocated_id = self._arena.alloc_high_slot(
                allow_reclaim=allow_reclaim, min_slot_id=min_slot_id
            )
        except ArenaCapacityError:
            return None
        inserted = False
        try:
            new_slot = self._make_slot_state(allocated_id, phase=phase)
            with self._lock:
                if not (self._closed or self._closing or not self._arena.is_open):
                    self._slots[allocated_id] = new_slot
                    inserted = True
            if not inserted:
                raise RuntimeError("KvChunkPool was closed during slot allocation.")
        except BaseException:
            if not inserted and self._arena.is_open:
                _sync_interrupt_safe(lambda: self._arena.free_high_slot(allocated_id))
            raise
        return new_slot

    def _try_alloc_from_arena(
        self,
        *,
        allow_reclaim: bool,
        min_slot_id: int | None = None,
        phase: _SlotPhase,
    ) -> _KvSlotState | None:
        with self._lock:
            self._require_open_unlocked()
            if len(self._slots) + self._inflight_arena_allocs >= self._max_slots:
                return None
            self._inflight_arena_allocs += 1
        try:
            return self._alloc_new_slot_from_arena(
                allow_reclaim=allow_reclaim,
                min_slot_id=min_slot_id,
                phase=phase,
            )
        finally:
            with self._lock:
                self._inflight_arena_allocs = max(0, self._inflight_arena_allocs - 1)
                self._op_cond.notify_all()

    def _find_empty_slot_unlocked(self, *, next_phase: _SlotPhase) -> _KvSlotState | None:
        best_id: int | None = None
        for sid, slot in self._slots.items():
            if (
                slot.phase is _SlotPhase.EMPTY
                and slot.active_readers == 0
                and (best_id is None or sid > best_id)
            ):
                best_id = sid
        if best_id is None:
            return None
        slot = self._slots[best_id]
        slot.phase = next_phase
        return slot

    def _evict_clean_victim_unlocked(
        self,
        *,
        protected: set[Hashable],
        next_phase: _SlotPhase,
        prefer_higher_than: int | None = None,
    ) -> _KvSlotState | None:
        for key, rec in self._chunks.items():
            if key in protected or rec.slot is None:
                continue
            if rec.host_alloc is None or rec.spill_state is not _SpillState.COMPLETE:
                continue
            slot = rec.slot
            if slot.phase is not _SlotPhase.READY or slot.active_readers > 0:
                continue
            if prefer_higher_than is not None and slot.slot_id <= prefer_higher_than:
                continue
            rec.slot = None
            slot.key = None
            slot.phase = next_phase
            self._evictions += 1
            self._op_cond.notify_all()
            return slot
        return None

    def _evict_spilling_victim(
        self,
        *,
        protected: set[Hashable],
        next_phase: _SlotPhase,
    ) -> _KvSlotState | None:
        chosen: tuple[_ChunkRecord, _KvSlotState] | None = None
        with self._lock:
            for key, rec in self._chunks.items():
                if key in protected or rec.slot is None:
                    continue
                if rec.host_alloc is None or rec.spill_state is not _SpillState.IN_FLIGHT:
                    continue
                slot = rec.slot
                if slot.phase is _SlotPhase.READY and slot.active_readers == 0:
                    slot.phase = _SlotPhase.RELOCATING
                    chosen = (rec, slot)
                    break
        if chosen is None:
            return None
        rec, slot = chosen
        try:
            if slot.lease is not None:
                lease = slot.lease
                with slot.lease_lock:
                    _sync_interrupt_safe(lease.wait_d2h)
            with self._lock:
                if rec.host_alloc is not None:
                    rec.spill_state = _SpillState.COMPLETE
                if self._chunks.get(rec.key) is rec and rec.slot is slot:
                    rec.slot = None
                    self._evictions += 1
                slot.key = None
                slot.phase = next_phase
                self._op_cond.notify_all()
                return slot
        except BaseException:
            with self._lock:
                if slot.slot_id in self._slots and slot.phase is _SlotPhase.RELOCATING:
                    if self._chunks.get(rec.key) is rec and rec.slot is slot:
                        slot.phase = _SlotPhase.READY
                    else:
                        slot.key = None
                        slot.phase = _SlotPhase.EMPTY
                self._op_cond.notify_all()
            raise

    def _host_chunks_used_unlocked(self) -> int:
        return self._live_host_allocs + self._inflight_host_allocs

    def _can_reserve_host_budget_unlocked(self) -> bool:
        if self._max_host_bytes is None:
            return True
        used_bytes = (self._host_chunks_used_unlocked() + 1) * self._host_span_bytes
        return used_bytes <= self._max_host_bytes

    def _begin_chunk_spill_unlocked(self, rec: _ChunkRecord) -> int:
        self._next_spill_attempt_id += 1
        attempt_id = self._next_spill_attempt_id
        rec.spill_attempt_id = attempt_id
        rec.spill_state = _SpillState.ALLOCATING
        return attempt_id

    @staticmethod
    def _clear_allocating_spill_unlocked(rec: _ChunkRecord, attempt_id: int) -> None:
        if (
            rec.spill_attempt_id == attempt_id
            and rec.host_alloc is None
            and rec.spill_state is _SpillState.ALLOCATING
        ):
            rec.spill_state = _SpillState.NONE
            rec.spill_attempt_id = 0

    def _select_spill_candidate_unlocked(
        self,
        target_key: Hashable,
    ) -> tuple[_ChunkRecord, _KvSlotState, int] | None:
        rec = self._chunks.get(target_key)
        if (
            rec is not None
            and rec.slot is not None
            and rec.slot.phase is _SlotPhase.READY
            and rec.host_alloc is None
            and rec.spill_state is _SpillState.NONE
        ):
            return rec, rec.slot, self._begin_chunk_spill_unlocked(rec)
        return None

    def _allocate_host_slab_with_fallback(self, *, batch_hint: int) -> ArenaAllocation | None:
        if self._device.type != "cuda":
            return self._pin_manager.allocate_host_slab(
                self._host_layout.total_bytes,
                pinned=False,
                alignment=256,
                batch_hint=batch_hint,
            )
        alloc = self._pin_manager.allocate_host_slab(
            self._host_layout.total_bytes,
            pinned=True,
            alignment=256,
            batch_hint=batch_hint,
        )
        if alloc is not None:
            return alloc
        return self._pin_manager.allocate_host_slab(
            self._host_layout.total_bytes,
            pinned=False,
            alignment=256,
            batch_hint=batch_hint,
        )

    def _publish_spill_alloc_unlocked(
        self,
        rec: _ChunkRecord,
        slot: _KvSlotState,
        host_alloc: ArenaAllocation,
        *,
        spill_state: _SpillState,
        attempt_id: int = 0,
    ) -> bool:
        if (
            self._closed
            or self._closing
            or self._chunks.get(rec.key) is not rec
            or rec.slot is not slot
            or rec.host_alloc is not None
            or attempt_id not in (0, rec.spill_attempt_id)
        ):
            return False
        rec.host_alloc = host_alloc
        rec.spill_state = spill_state
        rec.spill_attempt_id = 0
        self._spills += 1
        self._op_cond.notify_all()
        return True

    def _defer_rejected_host_alloc(
        self,
        host_alloc: ArenaAllocation,
        copy_stream: torch.cuda.Stream,
        *,
        rec: _ChunkRecord | None = None,
        attempt_id: int = 0,
    ) -> None:
        with self._lock:
            if not any(it.host_alloc is host_alloc for it in self._pending_host_retires):
                self._pending_host_retires.append(
                    _PendingHostRetire(
                        slot=None, host_alloc=host_alloc, fallback_streams=(copy_stream,)
                    )
                )
            if rec is not None:
                self._clear_allocating_spill_unlocked(rec, attempt_id)
            self._op_cond.notify_all()

    def _launch_cuda_d2h_spill(
        self,
        rec: _ChunkRecord,
        slot: _KvSlotState,
        host_alloc: ArenaAllocation,
        *,
        attempt_id: int,
    ) -> _ChunkRecord | None:
        lease = slot.lease
        assert lease is not None
        non_blocking = host_alloc.is_pinned
        copy_stream: torch.cuda.Stream | None = None
        try:
            copy_stream = self._ensure_copy_stream()
            with slot.lease_lock:
                if lease.d2h_in_flight:
                    _sync_interrupt_safe(lease.wait_d2h)
                lease.spill_to_host(
                    lambda sid=slot.slot_id, ha=host_alloc, nb=non_blocking: self._copy_slot_d2h(
                        sid, ha, non_blocking=nb
                    ),
                    copy_stream,
                )
                if not non_blocking:
                    _sync_interrupt_safe(lease.wait_d2h)
                in_flight = lease.d2h_in_flight
        except BaseException as exc:
            if copy_stream is not None:
                with slot.lease_lock:
                    in_flight = lease.d2h_in_flight
                if in_flight:
                    try:
                        with slot.lease_lock:
                            _sync_interrupt_safe(lease.wait_d2h)
                    except BaseException as sync_exc:
                        self._defer_rejected_host_alloc(
                            host_alloc, copy_stream, rec=rec, attempt_id=attempt_id
                        )
                        raise sync_exc from exc
            self._release_or_defer_host_allocs([host_alloc])
            with self._lock:
                self._clear_allocating_spill_unlocked(rec, attempt_id)
                self._op_cond.notify_all()
            raise

        assert copy_stream is not None
        target_state = _SpillState.IN_FLIGHT if in_flight else _SpillState.COMPLETE
        with self._lock:
            published = self._publish_spill_alloc_unlocked(
                rec, slot, host_alloc, spill_state=target_state, attempt_id=attempt_id
            )
        if not published:
            if in_flight:
                self._defer_rejected_host_alloc(host_alloc, copy_stream)
            else:
                self._release_or_defer_host_allocs([host_alloc])
            return None
        return rec

    def _clamp_spill_batch_hint_unlocked(self, requested: int) -> int:
        hint = max(1, requested)
        if self._max_host_bytes is None:
            return hint
        cap_chunks = self._max_host_bytes // self._host_span_bytes
        remaining = max(1, cap_chunks - self._host_chunks_used_unlocked())
        return min(hint, remaining)

    def _execute_reserved_spill(
        self,
        rec: _ChunkRecord,
        slot: _KvSlotState,
        *,
        attempt_id: int,
        batch_hint: int,
    ) -> _ChunkRecord | None:
        """Execute an already-reserved spill attempt (budget + active counters held by caller)."""
        host_alloc: ArenaAllocation | None = None
        alloc_reserved = True
        try:
            host_alloc = self._allocate_host_slab_with_fallback(batch_hint=batch_hint)
            with self._lock:
                self._inflight_host_allocs = max(0, self._inflight_host_allocs - 1)
                alloc_reserved = False
                if host_alloc is None:
                    return None
                self._live_host_allocs += 1
                valid = (
                    not self._closed
                    and not self._closing
                    and self._chunks.get(rec.key) is rec
                    and rec.slot is slot
                )
            if not valid:
                self._release_or_defer_host_allocs([host_alloc])
                return None
            if self._device.type != "cuda":
                try:
                    self._copy_slot_d2h(slot.slot_id, host_alloc, non_blocking=False)
                    with self._lock:
                        published = self._publish_spill_alloc_unlocked(
                            rec, slot, host_alloc, spill_state=_SpillState.COMPLETE, attempt_id=attempt_id
                        )
                except BaseException:
                    self._release_or_defer_host_allocs([host_alloc])
                    raise
                if not published:
                    self._release_or_defer_host_allocs([host_alloc])
                    return None
                return rec
            return self._launch_cuda_d2h_spill(rec, slot, host_alloc, attempt_id=attempt_id)
        except BaseException:
            if alloc_reserved and host_alloc is not None and not host_alloc.released:
                try:
                    host_alloc.release()
                finally:
                    if not host_alloc.released:
                        with self._lock:
                            self._live_host_allocs += 1
                            self._pending_host_retires.append(
                                _PendingHostRetire(slot=None, host_alloc=host_alloc)
                            )
                            self._op_cond.notify_all()
            raise
        finally:
            with self._lock:
                if alloc_reserved:
                    self._inflight_host_allocs = max(0, self._inflight_host_allocs - 1)
                self._active_spill_ops = max(0, self._active_spill_ops - 1)
                self._clear_allocating_spill_unlocked(rec, attempt_id)
                self._op_cond.notify_all()

    def _spill_one_chunk(
        self,
        target_key: Hashable,
        *,
        batch_hint: int = 1,
    ) -> _ChunkRecord | None:
        chosen: tuple[_ChunkRecord, _KvSlotState, int, int] | None = None
        for attempt in range(2):
            need_reap = False
            with self._lock:
                self._require_open_unlocked()
                if not self._can_reserve_host_budget_unlocked():
                    need_reap = bool(self._pending_host_retires)
                else:
                    cand = self._select_spill_candidate_unlocked(target_key)
                    if cand is None:
                        return None
                    rec, slot, attempt_id = cand
                    effective_hint = self._clamp_spill_batch_hint_unlocked(batch_hint)
                    self._inflight_host_allocs += 1
                    self._active_spill_ops += 1
                    chosen = (rec, slot, attempt_id, effective_hint)
            if chosen is not None:
                break
            if need_reap and attempt == 0:
                self._reap_pending_host_retires()
                continue
            return None
        assert chosen is not None
        rec, slot, attempt_id, effective_hint = chosen
        return self._execute_reserved_spill(
            rec, slot, attempt_id=attempt_id, batch_hint=effective_hint
        )

    def _resolve_spill_candidates(self, keys: Sequence[Hashable] | None) -> list[Hashable]:
        if keys is not None:
            resolved = _validate_chunk_keys(keys, allow_none=False)
            with self._lock:
                self._require_open_unlocked()
                candidates = list(dict.fromkeys(k for k in resolved if k is not None))
                for key in candidates:
                    if key not in self._chunks:
                        raise KeyError(f"Unknown chunk key {key!r}.")
                return candidates
        with self._lock:
            self._require_open_unlocked()
            return [
                key
                for key, rec in self._chunks.items()
                if rec.slot is not None
                and rec.slot.phase is _SlotPhase.READY
                and rec.host_alloc is None
                and rec.spill_state is _SpillState.NONE
            ]

    def spill(
        self,
        keys: Sequence[Hashable] | None = None,
        *,
        max_chunks: int | None = None,
    ) -> int:
        """Best-effort write-once async D2H spill of ``GPU_DIRTY`` chunks to host RAM.

        Stops cleanly and returns the count spilled if ``PinManager`` or
        ``max_host_bytes`` cannot accommodate more slabs; never silently drops chunks.
        """
        self._reap_all_background()
        limit = _validate_optional_positive_int(max_chunks, name="max_chunks")
        candidates = self._resolve_spill_candidates(keys)
        spilled = 0
        for idx, key in enumerate(candidates):
            if limit is not None and spilled >= limit:
                break
            remaining_count = len(candidates) - idx
            hint = remaining_count if limit is None else min(remaining_count, limit - spilled)
            rec = self._spill_one_chunk(key, batch_hint=max(1, hint))
            if rec is None:
                with self._lock:
                    cur = self._chunks.get(key)
                    host_full = (
                        cur is not None
                        and cur.slot is not None
                        and cur.slot.phase is _SlotPhase.READY
                        and cur.host_alloc is None
                        and cur.spill_state is _SpillState.NONE
                    )
                if host_full:
                    break
                continue
            spilled += 1
        return spilled

    def _maybe_watermark_spill(self) -> None:
        wm = self._spill_watermark
        if wm is None or self._max_slots <= 0:
            return
        threshold = max(1, math.floor(self._max_slots * wm))
        with self._lock:
            if self._closed or self._closing or len(self._slots) < threshold:
                return
            dirty_count = 0
            for slot in self._slots.values():
                if slot.key is None:
                    continue
                rec = self._chunks.get(slot.key)
                if (
                    rec is not None
                    and rec.slot is slot
                    and rec.host_alloc is None
                    and rec.spill_state is _SpillState.NONE
                ):
                    dirty_count += 1
            excess = dirty_count - max(0, threshold - 1)
        if excess > 0:
            self.spill(max_chunks=excess)

    def _inline_spill_and_evict_dirty_victim(
        self,
        *,
        protected: set[Hashable],
        next_phase: _SlotPhase,
    ) -> _KvSlotState | None:
        while True:
            with self._lock:
                victim_key: Hashable | None = None
                for key, rec in self._chunks.items():
                    if key in protected or rec.slot is None:
                        continue
                    slot = rec.slot
                    if (
                        slot.phase is _SlotPhase.READY
                        and slot.active_readers == 0
                        and rec.host_alloc is None
                        and rec.spill_state is _SpillState.NONE
                    ):
                        victim_key = key
                        break
            if victim_key is None:
                return None
            spilled_rec = self._spill_one_chunk(victim_key)
            if spilled_rec is None:
                with self._lock:
                    cur = self._chunks.get(victim_key)
                    host_full = (
                        cur is not None
                        and cur.slot is not None
                        and cur.slot.phase is _SlotPhase.READY
                        and cur.host_alloc is None
                        and cur.spill_state is _SpillState.NONE
                    )
                if host_full:
                    return None
                continue
            with self._lock:
                slot = spilled_rec.slot
                if slot is None or self._chunks.get(spilled_rec.key) is not spilled_rec:
                    continue
            if self._device.type == "cuda" and slot.lease is not None:
                with slot.lease_lock:
                    if slot.lease.d2h_in_flight:
                        _sync_interrupt_safe(slot.lease.wait_d2h)
            with self._lock:
                if (
                    self._chunks.get(spilled_rec.key) is not spilled_rec
                    or spilled_rec.host_alloc is None
                    or spilled_rec.slot is not slot
                ):
                    continue
                spilled_rec.spill_state = _SpillState.COMPLETE
                if slot.phase is not _SlotPhase.READY or slot.active_readers > 0:
                    continue
                spilled_rec.slot = None
                slot.key = None
                slot.phase = next_phase
                self._evictions += 1
                self._op_cond.notify_all()
                return slot

    def _try_obtain_slot_once(
        self,
        *,
        protected: set[Hashable],
        next_phase: _SlotPhase,
        allow_reclaim_peer: bool,
        allow_spill_dirty: bool,
        allow_wait_d2h: bool,
    ) -> _KvSlotState | None:
        self._poll_spilling_chunks()
        with self._lock:
            self._require_open_unlocked()
            slot = self._find_empty_slot_unlocked(next_phase=next_phase)
        if slot is None:
            slot = self._try_alloc_from_arena(allow_reclaim=False, phase=next_phase)
        if slot is None:
            with self._lock:
                self._require_open_unlocked()
                slot = self._evict_clean_victim_unlocked(
                    protected=protected, next_phase=next_phase
                )
        if slot is None and allow_wait_d2h:
            slot = self._evict_spilling_victim(protected=protected, next_phase=next_phase)
        if slot is None and allow_reclaim_peer:
            slot = self._try_alloc_from_arena(allow_reclaim=True, phase=next_phase)
        if slot is None and allow_spill_dirty:
            slot = self._inline_spill_and_evict_dirty_victim(
                protected=protected, next_phase=next_phase
            )
        return slot

    def _has_obtainable_slot_unlocked(
        self, protected: set[Hashable], *, allow_wait_d2h: bool
    ) -> bool:
        if any(
            slot.phase is _SlotPhase.EMPTY and slot.active_readers == 0
            for slot in self._slots.values()
        ):
            return True
        return any(
            key not in protected
            and rec.slot is not None
            and rec.slot.phase is _SlotPhase.READY
            and rec.slot.active_readers == 0
            and rec.host_alloc is not None
            and (
                rec.spill_state is _SpillState.COMPLETE
                or (allow_wait_d2h and rec.spill_state is _SpillState.IN_FLIGHT)
            )
            for key, rec in self._chunks.items()
        )

    def _obtain_free_or_evicted_slot(
        self,
        *,
        protected: set[Hashable],
        next_phase: _SlotPhase,
        allow_reclaim_peer: bool,
        allow_spill_dirty: bool,
        allow_wait_d2h: bool = True,
    ) -> _KvSlotState | None:
        while True:
            slot = self._try_obtain_slot_once(
                protected=protected,
                next_phase=next_phase,
                allow_reclaim_peer=allow_reclaim_peer,
                allow_spill_dirty=allow_spill_dirty,
                allow_wait_d2h=allow_wait_d2h,
            )
            if slot is not None or not allow_wait_d2h:
                return slot
            with self._lock:
                self._require_open_unlocked()
                if self._has_obtainable_slot_unlocked(protected, allow_wait_d2h=allow_wait_d2h):
                    continue
                has_transient_wait = (
                    self._inflight_arena_frees > 0
                    or any(s.phase is _SlotPhase.RELOCATING for s in self._slots.values())
                    or any(
                        key not in protected
                        and rec.slot is not None
                        and rec.slot.phase is _SlotPhase.READY
                        and rec.slot.active_readers == 0
                        and rec.spill_state is _SpillState.ALLOCATING
                        for key, rec in self._chunks.items()
                    )
                )
                if not has_transient_wait:
                    return None
                self._op_cond.wait(timeout=0.05)

    def _rollback_unused_slots(self, slots: Sequence[_KvSlotState], *, free_back: bool) -> None:
        if not slots:
            return
        to_free: list[_KvSlotState] = []
        with self._lock:
            for slot in slots:
                slot.key = None
                if free_back:
                    to_free.append(slot)
                else:
                    slot.phase = _SlotPhase.EMPTY
            self._op_cond.notify_all()
        first_err: BaseException | None = None
        for slot in to_free:
            try:
                self._free_slot_or_retire(slot, sync_lease=True)
            except BaseException as exc:
                first_err = _merge_error(first_err, exc)
        if first_err is not None:
            raise first_err

    @contextlib.contextmanager
    def _enter_slot_produces(
        self, slots: Sequence[_KvSlotState], stream: torch.cuda.Stream
    ) -> Iterator[None]:
        entered: list[tuple[_KvSlotState, contextlib.AbstractContextManager[None]]] = []
        try:
            for slot in slots:
                assert slot.lease is not None
                with slot.lease_lock:
                    cm = slot.lease.produce(stream)
                    cm.__enter__()
                    entered.append((slot, cm))
            yield
        except BaseException as exc:
            first_exit_err: BaseException | None = exc
            for slot, cm in reversed(entered):
                try:
                    with slot.lease_lock:
                        cm.__exit__(type(exc), exc, exc.__traceback__)
                except BaseException as exit_exc:
                    first_exit_err = _merge_error(first_exit_err, exit_exc)
            assert first_exit_err is not None
            if first_exit_err is not exc:
                raise first_exit_err from exc
            raise
        else:
            exit_err: BaseException | None = None
            for slot, cm in reversed(entered):
                try:
                    with slot.lease_lock:
                        cm.__exit__(None, None, None)
                except BaseException as exc:
                    exit_err = _merge_error(exit_err, exc)
            if exit_err is not None:
                raise exit_err

    def _build_batch(
        self,
        *,
        index: int,
        keys: tuple[Hashable | None, ...],
        mapping: Sequence[int],
        active_slots_hint: int,
        is_hit_wave: bool,
    ) -> KvBatch:
        views = self._require_views()
        return KvBatch(
            index=index,
            keys=keys,
            slot_ids=_make_device_int_table(mapping, device=self._device),
            active_slots_hint=active_slots_hint,
            is_hit_wave=is_hit_wave,
            arena_views=views.arena_views,
            side_views=views.side_views,
            arena_bytes=views.arena_bytes,
            slot_offsets=views.slot_offsets,
        )

    def _claim_produce_slots(
        self,
        unique_keys: tuple[Hashable, ...],
        claimed_slots: list[_KvSlotState],
    ) -> None:
        for key in unique_keys:
            slot = self._obtain_free_or_evicted_slot(
                protected=set(),
                next_phase=_SlotPhase.PRODUCING,
                allow_reclaim_peer=True,
                allow_spill_dirty=True,
            )
            if slot is None:
                raise ArenaCapacityError(
                    f"No unleased high GPU slot or host spill capacity available to produce {key!r}."
                )
            slot.key = key
            claimed_slots.append(slot)

    def _commit_produced_slots_unlocked(
        self,
        unique_keys: tuple[Hashable, ...],
        claimed_slots: list[_KvSlotState],
    ) -> None:
        self._require_open_unlocked()
        for key, slot in zip(unique_keys, claimed_slots, strict=True):
            self._chunks[key] = _ChunkRecord(
                key=key, slot=slot, host_alloc=None, spill_state=_SpillState.NONE
            )
            self._chunks.move_to_end(key)
            slot.key = key
            slot.phase = _SlotPhase.READY
            self._producing_keys.discard(key)
        self._op_cond.notify_all()

    @contextlib.contextmanager
    def produce(self, keys: Sequence[Hashable]) -> Iterator[KvBatch]:
        """Allocate high GPU slots for a new immutable batch of chunk keys and commit on exit."""
        raw_keys = _validate_chunk_keys(keys, allow_none=False)
        if not raw_keys:
            raise ValueError("produce() requires at least one chunk key.")
        unique_keys: tuple[Hashable, ...] = tuple(k for k in raw_keys if k is not None)
        if len(set(unique_keys)) != len(unique_keys):
            raise ValueError("produce() keys must be distinct within one batch.")
        if len(unique_keys) > self._max_slots:
            raise ArenaCapacityError(
                f"produce() batch of {len(unique_keys)} exceeds max_slots={self._max_slots}."
            )
        self._reap_all_background()
        self._require_views()

        claimed_slots: list[_KvSlotState] = []
        committed = False
        session_entered = False
        try:
            with self._lock:
                self._require_open_unlocked()
                for key in unique_keys:
                    if key in self._chunks or key in self._producing_keys:
                        raise ValueError(f"Chunk key {key!r} is already tracked or being produced.")
                self._producing_keys.update(unique_keys)
                self._active_produce_sessions += 1
                session_entered = True
            self._claim_produce_slots(unique_keys, claimed_slots)
            batch = self._build_batch(
                index=0,
                keys=unique_keys,
                mapping=[s.slot_id for s in claimed_slots],
                active_slots_hint=len(unique_keys),
                is_hit_wave=False,
            )
            if self._device.type == "cuda":
                cur_stream = torch.cuda.current_stream(self._device)
                with self._enter_slot_produces(claimed_slots, cur_stream):
                    yield batch
            else:
                yield batch

            with self._lock:
                self._commit_produced_slots_unlocked(unique_keys, claimed_slots)
                committed = True
            self._maybe_watermark_spill()
        finally:
            if session_entered:
                try:
                    if not committed:
                        with self._lock:
                            for key in unique_keys:
                                self._producing_keys.discard(key)
                        self._rollback_unused_slots(claimed_slots, free_back=True)
                finally:
                    with self._lock:
                        self._active_produce_sessions = max(0, self._active_produce_sessions - 1)
                        self._op_cond.notify_all()

    def _stage_reload_batch(self, build: _ReloadBatchBuild) -> None:
        items = build.claimed
        if not items:
            return
        if self._device.type != "cuda":
            for _rec, slot, host_alloc in items:
                self._copy_host_to_slot(slot.slot_id, host_alloc, non_blocking=False)
            build.h2d_tracked = True
            return
        copy_stream = self._ensure_copy_stream()
        build.copy_stream = copy_stream
        try:
            for _rec, slot, host_alloc in items:
                assert slot.lease is not None
                with slot.lease_lock:
                    build.stage_started = True
                    slot.lease.stage_with(
                        lambda sid=slot.slot_id, ha=host_alloc: self._copy_host_to_slot(
                            sid, ha, non_blocking=ha.is_pinned
                        ),
                        copy_stream,
                    )
            build.stage_event = _record_event(copy_stream)
        except BaseException as exc:
            if build.stage_started:
                try:
                    _sync_interrupt_safe(copy_stream.synchronize)
                    build.h2d_tracked = True
                except BaseException as sync_exc:
                    with self._lock:
                        for rec, _slot, _ha in items:
                            rec.pending_h2d_streams.append(_PendingStreamToken(copy_stream))
                        build.h2d_tracked = True
                    raise sync_exc from exc
            raise

    def _finish_claim_reload_slot_unlocked(
        self,
        rec: _ChunkRecord,
        host_alloc: ArenaAllocation,
        slot: _KvSlotState | None,
        remaining: deque[_ChunkRecord],
        build: _ReloadBatchBuild,
        *,
        require_one: bool,
        target_lease: _WaveLease | None,
    ) -> bool:
        ready_slot = (
            rec.slot
            if self._chunks.get(rec.key) is rec
            and rec.slot is not None
            and rec.slot.phase is _SlotPhase.READY
            else None
        )
        if ready_slot is not None:
            rec.host_readers = max(0, rec.host_readers - 1)
            if slot is not None:
                slot.key = None
                slot.phase = _SlotPhase.EMPTY
            self._op_cond.notify_all()
            if remaining and remaining[0] is rec:
                remaining.popleft()
                build.taken_recs.append(rec)
            if target_lease is not None:
                target_lease.add_unlocked(ready_slot)
            build.prebound.append((rec, ready_slot))
            return True
        if slot is None:
            rec.host_readers = max(0, rec.host_readers - 1)
            self._op_cond.notify_all()
            if require_one:
                raise ArenaCapacityError(
                    f"No high GPU slot available to reload chunk {rec.key!r}."
                )
            return False
        slot.key = rec.key
        remaining.popleft()
        build.taken_recs.append(rec)
        build.acquired_slots.append(slot)
        build.claimed.append((rec, slot, host_alloc))
        build.pinned_recs.append(rec)
        return True

    def _claim_reload_step(
        self,
        remaining: deque[_ChunkRecord],
        build: _ReloadBatchBuild,
        *,
        protected: set[Hashable],
        require_at_least_one: bool,
        require_all: bool,
        allow_reclaim_peer: bool,
        allow_spill_dirty: bool,
        allow_wait_d2h: bool,
        target_lease: _WaveLease | None,
    ) -> bool:
        require_one = require_all or (
            require_at_least_one and not build.claimed and not build.prebound
        )
        with self._lock:
            self._require_open_unlocked()
            rec = remaining[0]
            if self._chunks.get(rec.key) is not rec:
                raise KeyError(f"Unknown chunk key {rec.key!r}.")
            if rec.slot is not None:
                if rec.slot.phase is _SlotPhase.READY:
                    remaining.popleft()
                    build.taken_recs.append(rec)
                    if target_lease is not None:
                        target_lease.add_unlocked(rec.slot)
                    build.prebound.append((rec, rec.slot))
                    return True
                if (
                    rec.slot.phase is _SlotPhase.RELOCATING
                    and require_one
                    and not self._closed
                    and not self._closing
                ):
                    self._op_cond.wait(timeout=0.05)
                    return True
                if require_one:
                    raise ArenaCapacityError(
                        f"Chunk key {rec.key!r} is temporarily busy in phase {rec.slot.phase.value}."
                    )
                return False
            host_alloc = rec.host_alloc
            if host_alloc is None:
                raise RuntimeError(f"HOST_ONLY chunk {rec.key!r} is missing host backing.")
            rec.host_readers += 1

        slot: _KvSlotState | None = None
        try:
            slot = self._obtain_free_or_evicted_slot(
                protected=protected,
                next_phase=_SlotPhase.STAGING,
                allow_reclaim_peer=allow_reclaim_peer,
                allow_spill_dirty=allow_spill_dirty,
                allow_wait_d2h=allow_wait_d2h,
            )
        except BaseException:
            with self._lock:
                rec.host_readers = max(0, rec.host_readers - 1)
                self._op_cond.notify_all()
            raise

        with self._lock:
            return self._finish_claim_reload_slot_unlocked(
                rec,
                host_alloc,
                slot,
                remaining,
                build,
                require_one=require_one,
                target_lease=target_lease,
            )

    def _prune_record_h2d_off_lock(self, recs: Sequence[_ChunkRecord]) -> None:
        with self._lock:
            snapshots = [
                (rec, tuple(rec.pending_h2d_events), tuple(rec.pending_h2d_streams))
                for rec in recs
                if rec.pending_h2d_events or rec.pending_h2d_streams
            ]
        if not snapshots:
            return
        done_by_rec: list[tuple[_ChunkRecord, set[int], set[int]]] = []
        for rec, events, streams in snapshots:
            done_evs = {id(ev) for ev in events if ev.query()}
            done_sts = {id(tok) for tok in streams if tok.stream.query()}
            if done_evs or done_sts:
                done_by_rec.append((rec, done_evs, done_sts))
        if not done_by_rec:
            return
        with self._lock:
            for rec, done_evs, done_sts in done_by_rec:
                if done_evs and rec.pending_h2d_events:
                    rec.pending_h2d_events = [
                        ev for ev in rec.pending_h2d_events if id(ev) not in done_evs
                    ]
                if done_sts and rec.pending_h2d_streams:
                    rec.pending_h2d_streams = [
                        tok for tok in rec.pending_h2d_streams if id(tok) not in done_sts
                    ]

    def _abort_reload_batch(
        self,
        build: _ReloadBatchBuild,
        remaining: deque[_ChunkRecord],
    ) -> None:
        if (
            self._device.type == "cuda"
            and build.stage_started
            and not build.h2d_tracked
            and build.pinned_recs
        ):
            if build.stage_event is not None:
                with self._lock:
                    for rec in build.pinned_recs:
                        if build.stage_event not in rec.pending_h2d_events:
                            rec.pending_h2d_events.append(build.stage_event)
            elif build.copy_stream is not None:
                with self._lock:
                    for rec in build.pinned_recs:
                        rec.pending_h2d_streams.append(_PendingStreamToken(build.copy_stream))
            build.h2d_tracked = True
        if build.pinned_recs:
            with self._lock:
                for rec in build.pinned_recs:
                    rec.host_readers = max(0, rec.host_readers - 1)
                build.pinned_recs.clear()
                self._op_cond.notify_all()
        build.claimed.clear()
        self._rollback_unused_slots(build.acquired_slots, free_back=False)
        for rec in reversed(build.taken_recs):
            remaining.appendleft(rec)

    def _prepare_reload_batch(
        self,
        remaining: deque[_ChunkRecord],
        *,
        wave_cap: int,
        protected: set[Hashable],
        require_at_least_one: bool,
        require_all: bool = False,
        allow_reclaim_peer: bool,
        allow_spill_dirty: bool,
        allow_wait_d2h: bool = True,
        target_lease: _WaveLease | None = None,
    ) -> tuple[
        list[tuple[_ChunkRecord, _KvSlotState, ArenaAllocation]],
        list[tuple[_ChunkRecord, _KvSlotState]],
    ]:
        build = _ReloadBatchBuild()
        try:
            while remaining and len(build.claimed) < wave_cap:
                if not self._claim_reload_step(
                    remaining,
                    build,
                    protected=protected,
                    require_at_least_one=require_at_least_one,
                    require_all=require_all,
                    allow_reclaim_peer=allow_reclaim_peer,
                    allow_spill_dirty=allow_spill_dirty,
                    allow_wait_d2h=allow_wait_d2h,
                    target_lease=target_lease,
                ):
                    break
            if self._device.type == "cuda" and build.pinned_recs:
                self._prune_record_h2d_off_lock(build.pinned_recs)
            self._stage_reload_batch(build)
            if build.pinned_recs:
                with self._lock:
                    if build.stage_event is not None:
                        for rec in build.pinned_recs:
                            if build.stage_event not in rec.pending_h2d_events:
                                rec.pending_h2d_events.append(build.stage_event)
                    build.h2d_tracked = True
                    for rec in build.pinned_recs:
                        rec.host_readers = max(0, rec.host_readers - 1)
                    build.pinned_recs.clear()
                    self._op_cond.notify_all()
            return build.claimed, build.prebound
        except BaseException:
            self._abort_reload_batch(build, remaining)
            raise

    def _commit_one_reloaded_item_unlocked(
        self,
        rec: _ChunkRecord,
        slot: _KvSlotState,
        *,
        wave_lease: _WaveLease | None,
        record_stats: bool,
    ) -> tuple[Hashable, _KvSlotState] | None:
        while (
            self._chunks.get(rec.key) is rec
            and rec.slot is not None
            and rec.slot.phase is _SlotPhase.RELOCATING
            and not self._closed
            and not self._closing
        ):
            self._op_cond.wait(timeout=0.05)
        if self._chunks.get(rec.key) is not rec:
            slot.key = None
            slot.phase = _SlotPhase.EMPTY
            return None
        if rec.slot is slot:
            if slot.phase is _SlotPhase.READY:
                if wave_lease is not None:
                    wave_lease.add_unlocked(slot)
                return rec.key, slot
            return None
        if rec.slot is not None:
            slot.key = None
            slot.phase = _SlotPhase.EMPTY
            if rec.slot.phase is _SlotPhase.READY:
                if wave_lease is not None:
                    wave_lease.add_unlocked(rec.slot)
                return rec.key, rec.slot
            return None
        slot.key = rec.key
        slot.phase = _SlotPhase.READY
        rec.slot = slot
        if rec.host_alloc is not None:
            rec.spill_state = _SpillState.COMPLETE
        if record_stats:
            self._reloads += 1
            self._chunks.move_to_end(rec.key)
        if wave_lease is not None:
            wave_lease.add_unlocked(slot)
        return rec.key, slot

    def _commit_reloaded_batch(
        self,
        staged: list[tuple[_ChunkRecord, _KvSlotState, ArenaAllocation]],
        *,
        wave_lease: _WaveLease | None,
        record_stats: bool,
        owned_claims: dict[int, _ChunkRecord] | None = None,
    ) -> list[tuple[Hashable, _KvSlotState]]:
        bound: list[tuple[Hashable, _KvSlotState]] = []
        with self._lock:
            try:
                for rec, slot, _ha in staged:
                    pair = self._commit_one_reloaded_item_unlocked(
                        rec, slot, wave_lease=wave_lease, record_stats=record_stats
                    )
                    if pair is not None:
                        bound.append(pair)
                    if owned_claims is not None and owned_claims.pop(id(rec), None) is not None:
                        rec.staging_waves = max(0, rec.staging_waves - 1)
            finally:
                for _rec, slot, _ha in staged:
                    if slot.phase is _SlotPhase.STAGING:
                        slot.key = None
                        slot.phase = _SlotPhase.EMPTY
                staged.clear()
                self._op_cond.notify_all()
        return bound

    def _release_wave_claims(
        self,
        claims: dict[int, _ChunkRecord],
        recs: Sequence[_ChunkRecord] | None = None,
    ) -> None:
        if not claims:
            return
        with self._lock:
            targets = list(claims.values()) if recs is None else recs
            for rec in targets:
                if claims.pop(id(rec), None) is not None:
                    rec.staging_waves = max(0, rec.staging_waves - 1)
            self._op_cond.notify_all()

    def _partition_wave_keys_unlocked(
        self,
        keys: tuple[Hashable | None, ...],
        wave_lease: _WaveLease,
        *,
        owned_claims: dict[int, _ChunkRecord],
        allow_split: bool,
        record_stats: bool,
    ) -> tuple[list[Hashable], list[tuple[Hashable, _KvSlotState]], deque[_ChunkRecord]]:
        unique = list(dict.fromkeys(k for k in keys if k is not None))
        for key in unique:
            if key not in self._chunks:
                raise KeyError(f"Unknown chunk key {key!r}.")
        if not allow_split and len(unique) > self._max_slots:
            raise ArenaCapacityError(
                f"allow_split=False requires {len(unique)} slots; max_slots={self._max_slots}."
            )
        hits: list[tuple[Hashable, _KvSlotState]] = []
        misses: deque[_ChunkRecord] = deque()
        for key in unique:
            rec = self._chunks[key]
            if rec.slot is not None and rec.slot.phase is _SlotPhase.READY:
                wave_lease.add_unlocked(rec.slot)
                hits.append((key, rec.slot))
                if record_stats:
                    self._hits += 1
                    self._chunks.move_to_end(key)
            else:
                rec.staging_waves += 1
                owned_claims[id(rec)] = rec
                misses.append(rec)
                if record_stats:
                    self._misses += 1
        return unique, hits, misses

    def _resolve_miss_wave_cap(self, miss_count: int, max_wave_slots: int | None) -> int:
        if max_wave_slots is not None:
            return min(miss_count, max_wave_slots)
        return max(1, min(miss_count, max(1, self._max_slots // 2)))

    def _wave_mapping(
        self,
        keys: tuple[Hashable | None, ...],
        active_pairs: Sequence[tuple[Hashable, _KvSlotState]],
    ) -> list[int]:
        pair_map = {key: slot.slot_id for key, slot in active_pairs}
        return [pair_map.get(key, -1) if key is not None else -1 for key in keys]

    def _run_single_wave_no_split(
        self,
        *,
        keys: tuple[Hashable | None, ...],
        unique_keys: list[Hashable],
        hits: list[tuple[Hashable, _KvSlotState]],
        misses: deque[_ChunkRecord],
        state: _WaveLoopState,
        record_stats: bool,
    ) -> Generator[KvBatch]:
        staged, prebound = self._prepare_reload_batch(
            misses,
            wave_cap=len(misses),
            protected=set(unique_keys),
            require_at_least_one=True,
            require_all=True,
            allow_reclaim_peer=True,
            allow_spill_dirty=True,
            allow_wait_d2h=True,
            target_lease=state.held_lease,
        )
        state.pending_commit = staged
        if misses:
            raise ArenaCapacityError(
                f"Could not reload all {len(unique_keys)} chunks into single wave ({len(misses)} remaining)."
            )
        reloaded_pairs = self._commit_reloaded_batch(
            state.pending_commit,
            wave_lease=state.held_lease,
            record_stats=record_stats,
            owned_claims=state.unyielded_miss_claims,
        )
        self._release_wave_claims(state.unyielded_miss_claims, [rec for rec, _slot in prebound])
        all_pairs = [*hits, *((rec.key, slot) for rec, slot in prebound), *reloaded_pairs]
        if not all_pairs:
            return
        state.held_lease.acquire()
        batch = self._build_batch(
            index=0,
            keys=keys,
            mapping=self._wave_mapping(keys, all_pairs),
            active_slots_hint=len(all_pairs),
            is_hit_wave=False,
        )
        if record_stats:
            with self._lock:
                self._waves_executed += 1
        yield batch
        state.held_lease.release()

    def _cleanup_wave_leases(
        self,
        state: _WaveLoopState,
        *,
        record_stats: bool,
    ) -> None:
        try:
            try:
                state.held_lease.release()
            finally:
                state.held_lease.handoff_to_pool()
        finally:
            try:
                try:
                    state.next_lease.release()
                finally:
                    state.next_lease.handoff_to_pool()
            finally:
                if state.pending_commit is state.next_staged:
                    leftover = list(state.pending_commit)
                else:
                    leftover = [*state.pending_commit, *state.next_staged]
                state.pending_commit.clear()
                state.next_staged.clear()
                try:
                    if leftover:
                        self._commit_reloaded_batch(
                            leftover,
                            wave_lease=None,
                            record_stats=record_stats,
                            owned_claims=state.unyielded_miss_claims,
                        )
                finally:
                    self._release_wave_claims(state.unyielded_miss_claims)

    def _run_split_miss_waves(
        self,
        *,
        keys: tuple[Hashable | None, ...],
        misses: deque[_ChunkRecord],
        wave_cap: int,
        wave_idx: int,
        state: _WaveLoopState,
        record_stats: bool,
    ) -> Generator[KvBatch]:
        while state.next_staged or state.next_prebound or misses:
            if not state.next_staged and not state.next_prebound:
                state.next_staged, state.next_prebound = self._prepare_reload_batch(
                    misses,
                    wave_cap=wave_cap,
                    protected={rec.key for rec in misses},
                    require_at_least_one=True,
                    allow_reclaim_peer=True,
                    allow_spill_dirty=True,
                    allow_wait_d2h=True,
                    target_lease=state.next_lease,
                )
            state.held_lease = state.next_lease
            state.next_lease = self._wave_releases.new_lease()
            state.pending_commit = state.next_staged
            state.next_staged = []
            cur_prebound = state.next_prebound
            state.next_prebound = []
            reloaded = self._commit_reloaded_batch(
                state.pending_commit,
                wave_lease=state.held_lease,
                record_stats=record_stats,
                owned_claims=state.unyielded_miss_claims,
            )
            self._release_wave_claims(
                state.unyielded_miss_claims, [rec for rec, _slot in cur_prebound]
            )
            pairs = [*((rec.key, slot) for rec, slot in cur_prebound), *reloaded]
            if not pairs:
                continue
            state.held_lease.acquire()
            if misses:
                state.next_staged, state.next_prebound = self._prepare_reload_batch(
                    misses,
                    wave_cap=wave_cap,
                    protected={rec.key for rec in misses},
                    require_at_least_one=False,
                    allow_reclaim_peer=False,
                    allow_spill_dirty=False,
                    allow_wait_d2h=False,
                    target_lease=state.next_lease,
                )
            batch = self._build_batch(
                index=wave_idx,
                keys=keys,
                mapping=self._wave_mapping(keys, pairs),
                active_slots_hint=len(pairs),
                is_hit_wave=False,
            )
            if record_stats:
                with self._lock:
                    self._waves_executed += 1
            yield batch
            state.held_lease.release()
            wave_idx += 1

    def _run_waves_generator(
        self,
        *,
        keys: tuple[Hashable | None, ...],
        max_wave_slots: int | None,
        allow_split: bool,
        record_stats: bool,
    ) -> Generator[KvBatch]:
        state = _WaveLoopState(
            held_lease=self._wave_releases.new_lease(),
            next_lease=self._wave_releases.new_lease(),
        )
        try:
            with self._lock:
                self._require_open_unlocked()
                unique_keys, hits, misses = self._partition_wave_keys_unlocked(
                    keys,
                    state.held_lease,
                    owned_claims=state.unyielded_miss_claims,
                    allow_split=allow_split,
                    record_stats=record_stats,
                )
            if not allow_split and misses:
                yield from self._run_single_wave_no_split(
                    keys=keys,
                    unique_keys=unique_keys,
                    hits=hits,
                    misses=misses,
                    state=state,
                    record_stats=record_stats,
                )
                return

            wave_idx = 0
            wave_cap = self._resolve_miss_wave_cap(len(misses), max_wave_slots)
            if hits:
                state.held_lease.acquire()
                if misses:
                    state.next_staged, state.next_prebound = self._prepare_reload_batch(
                        misses,
                        wave_cap=wave_cap,
                        protected={rec.key for rec in misses},
                        require_at_least_one=False,
                        allow_reclaim_peer=False,
                        allow_spill_dirty=False,
                        allow_wait_d2h=False,
                        target_lease=state.next_lease,
                    )
                batch = self._build_batch(
                    index=wave_idx,
                    keys=keys,
                    mapping=self._wave_mapping(keys, hits),
                    active_slots_hint=len(hits),
                    is_hit_wave=True,
                )
                if record_stats:
                    with self._lock:
                        self._waves_executed += 1
                yield batch
                state.held_lease.release()
                wave_idx += 1

            yield from self._run_split_miss_waves(
                keys=keys,
                misses=misses,
                wave_cap=wave_cap,
                wave_idx=wave_idx,
                state=state,
                record_stats=record_stats,
            )
        finally:
            self._cleanup_wave_leases(state, record_stats=record_stats)

    @contextlib.contextmanager
    def waves(
        self,
        keys: Sequence[Hashable | None],
        *,
        max_wave_slots: int | None = None,
        allow_split: bool = True,
        record_stats: bool = True,
    ) -> Iterator[Iterator[KvBatch]]:
        """Yield ready :class:`KvBatch` objects with request-ordered ``slot_ids``.

        ``None`` elements map to ``-1`` in ``slot_ids`` (ignored by downstream kernels).
        """
        resolved_keys = _validate_chunk_keys(keys, allow_none=True)
        resolved_cap = _validate_optional_positive_int(max_wave_slots, name="max_wave_slots")
        has_work = any(k is not None for k in resolved_keys)
        self._reap_all_background()
        entered = False
        gen: Generator[KvBatch] | None = None
        try:
            with self._lock:
                self._require_open_unlocked()
                if has_work:
                    for key in resolved_keys:
                        if key is not None and key not in self._chunks:
                            raise KeyError(f"Unknown chunk key {key!r}.")
                    self._active_wave_sessions += 1
                    entered = True
            if not has_work:
                yield iter(())
                return
            gen = self._run_waves_generator(
                keys=resolved_keys,
                max_wave_slots=resolved_cap,
                allow_split=allow_split,
                record_stats=bool(record_stats),
            )
            yield gen
        finally:
            if entered:
                try:
                    try:
                        if gen is not None:
                            gen.close()
                    finally:
                        self._reap_all_background()
                finally:
                    with self._lock:
                        self._active_wave_sessions = max(0, self._active_wave_sessions - 1)
                        self._op_cond.notify_all()

    def prefetch(self, keys: Sequence[Hashable]) -> int:
        """Best-effort async H2D reload of ``HOST_ONLY`` keys into free/clean slots."""
        resolved = _validate_chunk_keys(keys, allow_none=False)
        self._reap_all_background()
        with self._lock:
            self._require_open_unlocked()
            misses: deque[_ChunkRecord] = deque()
            seen = set[Hashable]()
            for key in resolved:
                assert key is not None
                if key not in self._chunks:
                    raise KeyError(f"Unknown chunk key {key!r}.")
                if key in seen:
                    continue
                seen.add(key)
                rec = self._chunks[key]
                if rec.slot is None:
                    misses.append(rec)
        if not misses:
            return 0
        staged: list[tuple[_ChunkRecord, _KvSlotState, ArenaAllocation]] = []
        committed = False
        try:
            staged, _prebound = self._prepare_reload_batch(
                misses,
                wave_cap=len(misses),
                protected=seen,
                require_at_least_one=False,
                allow_reclaim_peer=False,
                allow_spill_dirty=False,
                allow_wait_d2h=False,
                target_lease=None,
            )
            if staged:
                res = len(self._commit_reloaded_batch(staged, wave_lease=None, record_stats=False))
                committed = True
                return res
            committed = True
            return 0
        finally:
            if not committed and staged:
                self._rollback_unused_slots([slot for _rec, slot, _ha in staged], free_back=False)

    def _resolve_evict_targets(self, keys: Sequence[Hashable] | None) -> list[Hashable]:
        if keys is not None:
            resolved = _validate_chunk_keys(keys, allow_none=False)
            with self._lock:
                self._require_open_unlocked()
                targets = list(dict.fromkeys(k for k in resolved if k is not None))
                for key in targets:
                    if key not in self._chunks:
                        raise KeyError(f"Unknown chunk key {key!r}.")
                return targets
        with self._lock:
            self._require_open_unlocked()
            return [key for key, rec in self._chunks.items() if rec.slot is not None]

    def _finish_evict_unlocked(self, rec: _ChunkRecord, expected_slot: _KvSlotState) -> bool:
        if (
            self._closed
            or self._closing
            or self._chunks.get(rec.key) is not rec
            or rec.slot is not expected_slot
            or rec.host_alloc is None
            or rec.spill_state is not _SpillState.COMPLETE
            or expected_slot.phase is not _SlotPhase.READY
            or expected_slot.active_readers > 0
        ):
            return False
        rec.slot = None
        expected_slot.key = None
        expected_slot.phase = _SlotPhase.EMPTY
        self._evictions += 1
        self._op_cond.notify_all()
        return True

    def _await_allocating_spill_unlocked(self, key: Hashable) -> _ChunkRecord | None:
        while True:
            self._require_open_unlocked()
            rec = self._chunks.get(key)
            if rec is None or rec.spill_state is not _SpillState.ALLOCATING:
                return rec
            self._op_cond.wait(timeout=0.05)

    def _check_evict_ready_unlocked(
        self,
        key: Hashable,
        *,
        spilled_none_means_full: bool = False,
        expected_rec: _ChunkRecord | None = None,
    ) -> tuple[tuple[bool, bool] | None, _ChunkRecord | None, _KvSlotState | None]:
        rec = self._await_allocating_spill_unlocked(key)
        if expected_rec is not None and rec is not expected_rec:
            return (False, False), None, None
        if (
            spilled_none_means_full
            and rec is not None
            and rec.slot is not None
            and rec.host_alloc is None
            and rec.spill_state is _SpillState.NONE
        ):
            return (False, True), None, None
        if rec is None or rec.slot is None:
            return (False, False), None, None
        slot = rec.slot
        if slot.phase is not _SlotPhase.READY or slot.active_readers > 0:
            return (False, False), None, None
        if rec.spill_state is _SpillState.COMPLETE:
            return (self._finish_evict_unlocked(rec, slot), False), None, None
        return None, rec, slot

    def _evict_single_chunk(self, key: Hashable, *, batch_hint: int) -> tuple[bool, bool]:
        with self._lock:
            done, orig_rec, slot = self._check_evict_ready_unlocked(key)
            if done is not None or orig_rec is None or slot is None:
                return done if done is not None else (False, False)
            is_dirty = orig_rec.spill_state is _SpillState.NONE

        if is_dirty:
            spilled_rec = self._spill_one_chunk(key, batch_hint=batch_hint)
            with self._lock:
                done, rec, slot = self._check_evict_ready_unlocked(
                    key,
                    spilled_none_means_full=spilled_rec is None,
                    expected_rec=orig_rec,
                )
                if done is not None or rec is None or slot is None:
                    return done if done is not None else (False, False)

        if self._device.type == "cuda" and slot.lease is not None:
            with slot.lease_lock:
                if slot.lease.d2h_in_flight:
                    _sync_interrupt_safe(slot.lease.wait_d2h)
        with self._lock:
            if (
                self._chunks.get(key) is orig_rec
                and orig_rec.slot is slot
                and orig_rec.host_alloc is not None
                and orig_rec.spill_state is _SpillState.IN_FLIGHT
            ):
                orig_rec.spill_state = _SpillState.COMPLETE
            return self._finish_evict_unlocked(orig_rec, slot), False

    def evict_to_host(
        self,
        keys: Sequence[Hashable] | None = None,
        *,
        max_chunks: int | None = None,
    ) -> int:
        """Evict idle GPU chunks off VRAM while keeping their write-once host RAM copy."""
        self._reap_all_background()
        limit = _validate_optional_positive_int(max_chunks, name="max_chunks")
        targets = self._resolve_evict_targets(keys)
        evicted = 0
        for idx, key in enumerate(targets):
            if limit is not None and evicted >= limit:
                break
            remaining_count = len(targets) - idx
            hint = remaining_count if limit is None else min(remaining_count, limit - evicted)
            did_evict, host_full = self._evict_single_chunk(key, batch_hint=max(1, hint))
            if did_evict:
                evicted += 1
            elif host_full:
                break
        return evicted

    def _wait_discard_ready_unlocked(self, unique: Sequence[Hashable]) -> None:
        while True:
            self._require_open_unlocked()
            waited = False
            for key in unique:
                if key in self._producing_keys:
                    raise RuntimeError(f"Cannot discard chunk {key!r} while produce() is active.")
                rec = self._chunks.get(key)
                if rec is None:
                    continue
                if rec.staging_waves > 0:
                    raise RuntimeError(
                        f"Cannot discard chunk {key!r} while readers or transfers are active."
                    )
                if (
                    (rec.spill_state is _SpillState.ALLOCATING or rec.host_readers > 0)
                    and not self._closed
                    and not self._closing
                ):
                    self._op_cond.wait(timeout=0.05)
                    waited = True
                    break
                slot = rec.slot
                if slot is not None and (
                    slot.active_readers > 0 or slot.phase is not _SlotPhase.READY
                ):
                    raise RuntimeError(
                        f"Cannot discard chunk {key!r} while readers or transfers are active."
                    )
            if not waited:
                return

    def _classify_discard_item(
        self,
        slot: _KvSlotState | None,
        host_alloc: ArenaAllocation | None,
        was_spilling: bool,
        h2d_events: tuple[torch.cuda.Event, ...],
        h2d_streams: tuple[_PendingStreamToken, ...],
    ) -> tuple[_PendingHostRetire | None, _KvSlotState | None, ArenaAllocation | None]:
        d2h_pending = False
        if slot is not None and slot.lease is not None:
            with slot.lease_lock:
                if was_spilling or slot.lease.d2h_in_flight:
                    d2h_pending = not slot.lease.poll_d2h()
        pending_evs: tuple[torch.cuda.Event, ...] = ()
        pending_sts: tuple[torch.cuda.Stream, ...] = ()
        if self._device.type == "cuda" and host_alloc is not None:
            pending_evs, pending_sts = _query_pending_events_and_streams(h2d_events, h2d_streams)

        if d2h_pending:
            return (
                _PendingHostRetire(
                    slot=slot,
                    host_alloc=host_alloc,
                    h2d_events=pending_evs,
                    fallback_streams=pending_sts,
                ),
                None,
                None,
            )
        if slot is not None and slot.lease is not None:
            with slot.lease_lock:
                slot.lease.discard_dirty()
        if host_alloc is not None and (pending_evs or pending_sts):
            return (
                _PendingHostRetire(
                    slot=None,
                    host_alloc=host_alloc,
                    h2d_events=pending_evs,
                    fallback_streams=pending_sts,
                ),
                slot,
                None,
            )
        return None, slot, host_alloc

    def _queue_discard_fallback_unlocked(
        self,
        unclassified: Sequence[_DetachedDiscardItem],
        deferred: Sequence[_PendingHostRetire],
        ready_slots: Sequence[_KvSlotState],
        free_hosts: Sequence[ArenaAllocation],
    ) -> None:
        for slot, host_alloc, _ws, h2d_evs, h2d_toks in unclassified:
            if slot is not None or host_alloc is not None:
                self._pending_host_retires.append(
                    _PendingHostRetire(
                        slot=slot,
                        host_alloc=host_alloc,
                        h2d_events=h2d_evs,
                        fallback_streams=tuple(tok.stream for tok in h2d_toks),
                    )
                )
        if deferred:
            self._pending_host_retires.extend(deferred)
        for slot in ready_slots:
            if not self._closed and not self._closing and self._slots.get(slot.slot_id) is slot:
                slot.phase = _SlotPhase.EMPTY
        for ha in free_hosts:
            if not ha.released:
                self._pending_host_retires.append(_PendingHostRetire(slot=None, host_alloc=ha))
        self._op_cond.notify_all()

    def _detach_discard_keys_unlocked(
        self, unique: list[Hashable]
    ) -> list[_DetachedDiscardItem]:
        detached: list[_DetachedDiscardItem] = []
        self._wait_discard_ready_unlocked(unique)
        for key in unique:
            rec = self._chunks.pop(key, None)
            if rec is None:
                continue
            if rec.slot is not None:
                rec.slot.key = None
                rec.slot.phase = _SlotPhase.RETIRING
            detached.append(
                (
                    rec.slot,
                    rec.host_alloc,
                    rec.spill_state is _SpillState.IN_FLIGHT,
                    tuple(rec.pending_h2d_events),
                    tuple(rec.pending_h2d_streams),
                )
            )
            self._discards += 1
        if detached:
            self._inflight_host_retire_ops += 1
        return detached

    def discard(self, keys: Sequence[Hashable]) -> int:
        """Retire retired/compacted chunk keys cleanly from GPU and host RAM.

        Raises ``RuntimeError`` atomically if any key currently has active wave
        readers or staging/relocation transfers. Chunks with in-flight D2H spills
        or H2D reloads retire asynchronously without blocking compute.
        """
        resolved = _validate_chunk_keys(keys, allow_none=False)
        unique = list(dict.fromkeys(k for k in resolved if k is not None))
        if not unique:
            return 0
        self._reap_all_background()
        with self._lock:
            detached = self._detach_discard_keys_unlocked(unique)
        if not detached:
            return 0

        deferred: list[_PendingHostRetire] = []
        ready_slots: list[_KvSlotState] = []
        free_hosts: list[ArenaAllocation] = []
        try:
            for idx, item in enumerate(detached):
                try:
                    d_item, r_slot, f_host = self._classify_discard_item(*item)
                except BaseException:
                    with self._lock:
                        self._queue_discard_fallback_unlocked(
                            detached[idx:], deferred, ready_slots, free_hosts
                        )
                    raise
                if d_item is not None:
                    deferred.append(d_item)
                if r_slot is not None:
                    ready_slots.append(r_slot)
                if f_host is not None:
                    free_hosts.append(f_host)
            with self._lock:
                if deferred:
                    self._pending_host_retires.extend(deferred)
                for slot in ready_slots:
                    if not self._closed and not self._closing and self._slots.get(slot.slot_id) is slot:
                        slot.phase = _SlotPhase.EMPTY
                self._op_cond.notify_all()
            if free_hosts:
                self._release_or_defer_host_allocs(free_hosts)
            return len(detached)
        finally:
            with self._lock:
                self._inflight_host_retire_ops = max(0, self._inflight_host_retire_ops - 1)
                self._op_cond.notify_all()

    def _find_empty_relocate_slot_unlocked(self, edge_slot_id: int) -> _KvSlotState | None:
        self._require_open_unlocked()
        for sid in sorted(self._slots, reverse=True):
            if sid <= edge_slot_id:
                break
            slot = self._slots[sid]
            if slot.phase is _SlotPhase.EMPTY and slot.active_readers == 0:
                slot.phase = _SlotPhase.RELOCATING
                return slot
        return None

    def _find_or_alloc_relocate_target(
        self,
        edge_slot_id: int,
        *,
        allow_evict_clean: bool,
    ) -> tuple[_KvSlotState, bool] | None:
        with self._lock:
            hit = self._find_empty_relocate_slot_unlocked(edge_slot_id)
            if hit is not None:
                return hit, False

        allocated = self._try_alloc_from_arena(
            allow_reclaim=False,
            min_slot_id=edge_slot_id + 1,
            phase=_SlotPhase.RELOCATING,
        )
        if allocated is not None:
            if allocated.slot_id > edge_slot_id:
                return allocated, True
            self._free_slot_to_arena(allocated, sync_lease=False)

        with self._lock:
            hit = self._find_empty_relocate_slot_unlocked(edge_slot_id)
            if hit is not None:
                return hit, False
            if allow_evict_clean:
                evicted = self._evict_clean_victim_unlocked(
                    protected=set(),
                    next_phase=_SlotPhase.RELOCATING,
                    prefer_higher_than=edge_slot_id,
                )
                if evicted is not None:
                    return evicted, False
        return None

    def _queue_slot_d2d(
        self,
        src: _KvSlotState,
        dst: _KvSlotState,
        *,
        is_dirty: bool,
    ) -> None:
        if self._device.type != "cuda":
            self._copy_slot_d2d(src.slot_id, dst.slot_id, non_blocking=False)
            return
        src_lease = src.lease
        dst_lease = dst.lease
        assert src_lease is not None
        assert dst_lease is not None
        copy_stream: torch.cuda.Stream | None = None
        acquired = False
        stage_err: BaseException | None = None
        rel_err: BaseException | None = None
        try:
            copy_stream = self._ensure_copy_stream()
            with src.lease_lock:
                src_lease.acquire(copy_stream, exclusive=False)
                acquired = True
            with dst.lease_lock:
                if is_dirty:
                    dst_lease.produce_with(
                        lambda: self._copy_slot_d2d(src.slot_id, dst.slot_id, non_blocking=True),
                        copy_stream,
                    )
                else:
                    dst_lease.stage_with(
                        lambda: self._copy_slot_d2d(src.slot_id, dst.slot_id, non_blocking=True),
                        copy_stream,
                    )
        except BaseException as exc:
            stage_err = exc
        finally:
            if acquired and copy_stream is not None:
                released, rel_err = _release_slot_stream_lease(src, copy_stream, ())
                if not released:
                    with self._lock:
                        self._wave_releases.enqueue_unlocked(src, copy_stream)
        merged = _merge_error(stage_err, rel_err)
        if merged is not None:
            if stage_err is not None and merged is not stage_err:
                raise merged from stage_err
            raise merged

    def _reset_or_retire_dst_slot(self, dst: _KvSlotState, *, dst_fresh: bool) -> None:
        interrupt: BaseException | None = None
        if dst_fresh:
            try:
                self._free_slot_to_arena(dst, sync_lease=True)
                return
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    interrupt = exc
        elif dst.lease is not None:
            try:
                with dst.lease_lock:
                    _sync_interrupt_safe(dst.lease.synchronize_and_reset)
                with self._lock:
                    if self._slots.get(dst.slot_id) is dst:
                        dst.key = None
                        dst.phase = _SlotPhase.EMPTY
                    self._op_cond.notify_all()
                return
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    interrupt = exc
        else:
            with self._lock:
                if self._slots.get(dst.slot_id) is dst:
                    dst.key = None
                    dst.phase = _SlotPhase.EMPTY
                self._op_cond.notify_all()
            return

        with self._lock:
            if self._slots.get(dst.slot_id) is dst:
                dst.key = None
                dst.phase = _SlotPhase.RETIRING
                self._pending_host_retires.append(_PendingHostRetire(slot=dst, host_alloc=None))
            self._op_cond.notify_all()
        if interrupt is not None:
            raise interrupt

    def _relocate_edge_slot_to_higher(
        self,
        src: _KvSlotState,
        rec: _ChunkRecord,
    ) -> bool:
        target = self._find_or_alloc_relocate_target(
            src.slot_id, allow_evict_clean=(rec.host_alloc is None)
        )
        if target is None:
            return False
        dst, dst_fresh = target
        committed = False
        try:
            self._queue_slot_d2d(src, dst, is_dirty=(rec.host_alloc is None))
            with self._lock:
                still_valid = self._chunks.get(rec.key) is rec and rec.slot is src
                if still_valid:
                    dst.key = rec.key
                    dst.phase = _SlotPhase.READY
                    rec.slot = dst
                    self._relocations += 1
                else:
                    dst.key = None
                committed = True
                src.key = None
                src.phase = _SlotPhase.RELOCATING
                self._op_cond.notify_all()
        except BaseException:
            if not committed:
                self._reset_or_retire_dst_slot(dst, dst_fresh=dst_fresh)
            raise
        try:
            if not still_valid:
                self._reset_or_retire_dst_slot(dst, dst_fresh=dst_fresh)
        finally:
            self._free_slot_or_retire(src, sync_lease=True)
        return True

    def _evict_or_spill_edge_slot(self, src: _KvSlotState, rec: _ChunkRecord) -> bool:
        if rec.spill_state is _SpillState.NONE:
            with self._lock:
                self._require_open_unlocked()
                if not self._can_reserve_host_budget_unlocked():
                    src.phase = _SlotPhase.READY
                    self._op_cond.notify_all()
                    return False
                self._inflight_host_allocs += 1
                self._active_spill_ops += 1
                attempt_id = self._begin_chunk_spill_unlocked(rec)
            spilled = self._execute_reserved_spill(rec, src, attempt_id=attempt_id, batch_hint=1)
            if spilled is None:
                with self._lock:
                    if self._slots.get(src.slot_id) is src:
                        src.phase = _SlotPhase.READY
                    self._op_cond.notify_all()
                return False

        if self._device.type == "cuda" and src.lease is not None:
            with src.lease_lock:
                if src.lease.d2h_in_flight:
                    _sync_interrupt_safe(src.lease.wait_d2h)
        with self._lock:
            if rec.host_alloc is not None:
                rec.spill_state = _SpillState.COMPLETE
            if self._chunks.get(rec.key) is rec and rec.slot is src:
                rec.slot = None
                self._evictions += 1
            src.key = None
            src.phase = _SlotPhase.RELOCATING
            self._op_cond.notify_all()
        self._free_slot_or_retire(src, sync_lease=True)
        return True

    def _should_wait_frontier_unlocked(self, layout: HighSlotLayout, needed_high: int) -> bool:
        if not self._slots:
            return self._inflight_arena_frees > 0
        bottom_id = min(self._slots)
        if layout.slot_offset(bottom_id) >= needed_high:
            return self._inflight_arena_frees > 0
        slot = self._slots[bottom_id]
        wait = slot.phase is _SlotPhase.RELOCATING or (
            slot.phase is _SlotPhase.RETIRING and self._inflight_host_retire_ops > 0
        )
        if not wait and slot.phase is _SlotPhase.READY and slot.active_readers == 0 and slot.key is not None:
            rec = self._chunks.get(slot.key)
            wait = rec is not None and rec.spill_state is _SpillState.ALLOCATING
        return wait

    def _next_frontier_reclaim_candidate(
        self,
        layout: HighSlotLayout,
        needed_high: int,
    ) -> tuple[str, _KvSlotState, _ChunkRecord | None, _PendingHostRetire | None] | None:
        result: tuple[str, _KvSlotState, _ChunkRecord | None, _PendingHostRetire | None] | None = None
        with self._lock:
            while not self._closed and not self._closing:
                if not self._should_wait_frontier_unlocked(layout, needed_high):
                    break
                self._op_cond.wait(timeout=0.05)
            if self._closed or self._closing or not self._slots:
                return None
            bottom_id = min(self._slots)
            slot = self._slots[bottom_id]
            if layout.slot_offset(bottom_id) >= needed_high or slot.active_readers > 0:
                return None
            if slot.phase is _SlotPhase.EMPTY:
                slot.phase = _SlotPhase.RELOCATING
                result = ("free_empty", slot, None, None)
            elif slot.phase is _SlotPhase.RETIRING:
                for idx, item in enumerate(self._pending_host_retires):
                    if item.slot is slot:
                        del self._pending_host_retires[idx]
                        slot.phase = _SlotPhase.RELOCATING
                        result = ("free_retiring", slot, None, item)
                        break
            elif slot.phase is _SlotPhase.READY:
                rec = self._chunks.get(slot.key) if slot.key is not None else None
                if rec is None or rec.spill_state is not _SpillState.ALLOCATING:
                    slot.phase = _SlotPhase.RELOCATING
                    result = ("relocate_or_evict", slot, rec, None)
        return result

    def _restore_frontier_ready_edge_unlocked(
        self, src: _KvSlotState, rec: _ChunkRecord | None
    ) -> None:
        if rec is not None and (
            src.slot_id in self._slots
            and self._chunks.get(rec.key) is rec
            and rec.slot is src
        ):
            src.phase = _SlotPhase.READY
        elif src.slot_id in self._slots and src.phase is _SlotPhase.RELOCATING:
            src.key = None
            src.phase = _SlotPhase.RETIRING if rec is not None else _SlotPhase.EMPTY
            if rec is not None:
                self._pending_host_retires.append(_PendingHostRetire(slot=src, host_alloc=None))
        self._op_cond.notify_all()

    def _reclaim_high_frontier(self, needed_high: int) -> None:
        self._reap_all_background()
        layout = self._layout
        if layout is None:
            return
        while True:
            step = self._next_frontier_reclaim_candidate(layout, needed_high)
            if step is None:
                return
            kind, src, rec, retire = step
            if kind == "free_empty" or (kind == "relocate_or_evict" and rec is None):
                try:
                    self._free_slot_to_arena(src, sync_lease=True)
                except BaseException:
                    with self._lock:
                        if src.slot_id in self._slots and src.phase is _SlotPhase.RELOCATING:
                            src.key = None
                            src.phase = _SlotPhase.EMPTY
                        self._op_cond.notify_all()
                    raise
            elif kind == "free_retiring":
                assert retire is not None
                retires = deque([retire])
                try:
                    self._synchronize_host_retires(retires)
                except BaseException:
                    with self._lock:
                        self._pending_host_retires.extendleft(reversed(retires))
                        if self._slots.get(src.slot_id) is src:
                            src.phase = _SlotPhase.RETIRING
                        self._op_cond.notify_all()
                    raise
                self._free_slot_to_arena(src, sync_lease=False)
            else:
                assert rec is not None
                try:
                    relocated = rec.spill_state is not _SpillState.IN_FLIGHT and self._relocate_edge_slot_to_higher(
                        src, rec
                    )
                    if not relocated and not self._evict_or_spill_edge_slot(src, rec):
                        return
                except BaseException:
                    with self._lock:
                        self._restore_frontier_ready_edge_unlocked(src, rec)
                    raise

    def _synchronize_host_retires(self, retires: deque[_PendingHostRetire]) -> None:
        while retires:
            item = retires[0]
            if item.slot is not None and item.slot.lease is not None:
                with item.slot.lease_lock:
                    _sync_interrupt_safe(item.slot.lease.synchronize_and_reset)
            for ev in item.h2d_events:
                _sync_interrupt_safe(ev.synchronize)
            for st in item.fallback_streams:
                _sync_interrupt_safe(st.synchronize)
            self._release_tracked_host_alloc(item.host_alloc)
            retires.popleft()

    def _teardown_resources(self, *, slots: Sequence[_KvSlotState] = ()) -> BaseException | None:
        first_err: BaseException | None = None

        def _capture(fn: Callable[[], None]) -> None:
            nonlocal first_err
            try:
                fn()
            except BaseException as exc:
                first_err = _merge_error(first_err, exc)

        for slot in slots:
            _capture(lambda s=slot: self._free_slot_to_arena(s, sync_lease=False))
        with self._lock:
            remaining_slots = bool(self._slots)
        if self._arena.is_open and not remaining_slots:
            reset_layout = self._layout if self._configured_fresh_layout else None
            _capture(
                lambda: self._arena._release_high_owner(
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
            if first_err is None:
                self._views = None
                self._chunks.clear()
                self._producing_keys.clear()
                self._slots.clear()
                self._pending_host_retires.clear()
                self._wave_releases.pending.clear()
                self._closing = False
                self._closed = True
        return first_err

    def _validate_idle_for_close_unlocked(self) -> None:
        pending_counts = self._wave_releases.pending_counts_unlocked()
        busy_slot = any(
            s.active_readers > pending_counts.get(s.slot_id, 0)
            or s.phase in (_SlotPhase.PRODUCING, _SlotPhase.STAGING, _SlotPhase.RELOCATING)
            for s in self._slots.values()
        )
        busy_host = any(
            rec.host_readers > 0
            or rec.staging_waves > 0
            or rec.spill_state is _SpillState.ALLOCATING
            for rec in self._chunks.values()
        )
        if (
            self._active_wave_sessions > 0
            or self._active_produce_sessions > 0
            or self._active_spill_ops > 0
            or self._inflight_arena_allocs > 0
            or self._inflight_arena_frees > 0
            or self._inflight_host_allocs > 0
            or self._inflight_host_retire_ops > 0
            or self._wave_releases.inflight_reaps > 0
            or busy_slot
            or busy_host
        ):
            raise RuntimeError(
                "Cannot close KvChunkPool while produce, wave, reload, or relocation work is active."
            )

    def _release_all_chunk_host_allocs(self) -> None:
        with self._lock:
            recs = [rec for rec in self._chunks.values() if rec.host_alloc is not None]
        for rec in recs:
            ha = rec.host_alloc
            if ha is None:
                continue
            self._release_tracked_host_alloc(ha)
            with self._lock:
                rec.host_alloc = None
                rec.spill_state = _SpillState.NONE
                rec.pending_h2d_events.clear()
                rec.pending_h2d_streams.clear()

    def close(self) -> None:
        """Synchronize pending CUDA/D2H/H2D state, free host slabs and high arena slots."""
        with self._close_lock:
            slots: list[_KvSlotState] = []
            pending_by_slot: dict[int, list[_PendingWaveRelease]] = {}
            retires: deque[_PendingHostRetire] = deque()
            h2d_events: list[torch.cuda.Event] = []
            h2d_streams: list[torch.cuda.Stream] = []
            entered_close = False
            try:
                with self._lock:
                    if self._closed:
                        return
                    self._validate_idle_for_close_unlocked()
                    slots.extend(self._slots.values())
                    for rec in self._chunks.values():
                        h2d_events.extend(rec.pending_h2d_events)
                        h2d_streams.extend(tok.stream for tok in rec.pending_h2d_streams)
                    self._closing = True
                    entered_close = True
                    self._wave_releases.drain_by_slot_unlocked(out=pending_by_slot)
                    retires.extend(self._pending_host_retires)
                    self._pending_host_retires.clear()

                self._wave_releases.synchronize_slots_for_close(slots, pending_by_slot)
                self._synchronize_host_retires(retires)
                for ev in h2d_events:
                    _sync_interrupt_safe(ev.synchronize)
                for st in h2d_streams:
                    _sync_interrupt_safe(st.synchronize)
            except BaseException:
                if entered_close:
                    with self._lock:
                        self._wave_releases.restore_unlocked(pending_by_slot)
                        while retires:
                            item = retires.pop()
                            if not any(existing is item for existing in self._pending_host_retires):
                                self._pending_host_retires.appendleft(item)
                raise

            self._release_all_chunk_host_allocs()
            err = self._teardown_resources(slots=slots)
            if err is not None:
                raise err

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "KvBatch",
    "KvChunkLocation",
    "KvChunkPool",
    "KvPoolStats",
]
