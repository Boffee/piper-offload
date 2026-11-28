"""High-end VRAM KV-chunk slot manager inside a shared ``Uint8DualArena``."""

from __future__ import annotations

import contextlib
import enum
import threading
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

import torch

from ._slab_pool_common import (
    _build_pool_views_common,
    _build_side_pools,
    _make_device_int_table,
    _merge_error,
    _PendingWaveRelease,
    _PoolViews,
    _sync_interrupt_safe,
    _validate_slab_pool_base_args,
    _WaveLease,
    _WaveReleaseTracker,
)
from .dual_arena import ArenaCapacityError, HighSlotLayout, SlotSchema, TypedSidePool, Uint8DualArena
from .stream_lease import StreamLease
from .vram_budget import VramBudget


class _SlotPhase(enum.Enum):
    EMPTY = "empty"
    PRODUCING = "producing"
    READY = "ready"
    RECLAIMING = "reclaiming"


@dataclass(eq=False, slots=True)
class _KvSlot:
    slot_id: int
    phase: _SlotPhase = _SlotPhase.EMPTY
    key: Hashable | None = None
    active_readers: int = 0
    lease: StreamLease | None = None
    lease_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True, slots=True)
class KvBatch:
    """Persistent slot views and indices bound to one ``produce()`` or ``read()`` call.

    Attributes:
        keys: Validated sequence of chunk keys (or ``None`` sentinels in ``read()``).
        slot_ids: ``int32`` tensor of length ``len(keys)`` on the pool's device;
            ``None`` sentinels map to ``-1`` while duplicate keys map to the same slot.
        active_slots: Count of unique non-``None`` VRAM slots bound in this batch.
        arena_views: Read/write ``[max_slots, *shape]`` ``uint8`` views into ``Uint8DualArena``.
        side_views: Typed ``[max_slots, *shape]`` views backed by ``TypedSidePool``s.
        arena_bytes: Underlying 1-D ``torch.uint8`` arena tensor.
        slot_offsets: ``int64[max_slots]`` arena byte offsets keyed by arena field name.
    """

    keys: tuple[Hashable | None, ...]
    slot_ids: torch.Tensor
    active_slots: int
    arena_views: Mapping[str, torch.Tensor]
    side_views: Mapping[str, torch.Tensor]
    arena_bytes: torch.Tensor
    slot_offsets: Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class KvPoolStats:
    """Point-in-time residency and lifetime counters for a :class:`KvChunkPool`.

    Attributes:
        max_slots: Maximum number of descending high-end slots configured on this pool.
        allocated_gpu_slots: Arena high slots currently held by this pool (ready + empty + active).
        resident_chunks: Number of ``READY`` chunks addressable by key.
        empty_slots: Held high slots currently ``EMPTY`` (available for reuse or bottom reclaim).
        produced: Cumulative number of chunks committed cleanly via ``produce()``.
        discards: Cumulative number of chunks retired via ``discard()``.
    """

    max_slots: int
    allocated_gpu_slots: int
    resident_chunks: int
    empty_slots: int
    produced: int
    discards: int


def _validate_non_none_keys(keys: Sequence[Hashable]) -> tuple[Hashable, ...]:
    if isinstance(keys, (str, bytes, bytearray)) or not isinstance(keys, Sequence):
        raise TypeError("keys must be a sequence of hashable chunk keys, not a string or bytes.")
    resolved: list[Hashable] = []
    for key in keys:
        if key is None:
            raise ValueError("None sentinel keys are only supported in read().")
        hash(key)
        resolved.append(key)
    return tuple(resolved)


def _validate_read_keys(keys: Sequence[Hashable | None]) -> tuple[Hashable | None, ...]:
    if isinstance(keys, (str, bytes, bytearray)) or not isinstance(keys, Sequence):
        raise TypeError("keys must be a sequence of chunk keys, not a string or bytes.")
    resolved: list[Hashable | None] = []
    for key in keys:
        if key is not None:
            hash(key)
        resolved.append(key)
    return tuple(resolved)


def _push_locked_write(
    stack: contextlib.ExitStack,
    slot: _KvSlot,
    stream: torch.cuda.Stream,
) -> None:
    lease = slot.lease
    assert lease is not None
    cm = lease.write(stream)
    active = False

    def _locked_exit(
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        if not active:
            return None
        with slot.lease_lock:
            return cm.__exit__(exc_type, exc_val, exc_tb)

    stack.push(_locked_exit)
    with slot.lease_lock:
        active = True
        try:
            cm.__enter__()
        except BaseException:
            active = False
            raise


class KvChunkPool:
    """VRAM-only keyed KV-chunk pool occupying the high end of ``Uint8DualArena``.

    Allocates descending fixed drawers from ``Uint8DualArena`` paired with
    ``TypedSidePool`` side buffers for out-of-arena metadata. ``produce()``
    allocates high slots for new GPU-born KV chunks, ``read()`` binds existing
    chunks (with prefix sharing and ``None -> -1`` sentinels) in a single
    :class:`KvBatch`, and ``discard()`` frees chunks immediately without host
    spilling or live-chunk D2D relocation. Exceeding VRAM capacity raises
    :class:`ArenaCapacityError`.

    Internal Lock Ordering:
        ``_close_lock`` -> ``_lock`` -> ``arena._lock``.
        ``slot.lease_lock`` is disjoint from ``_lock`` and ordered strictly
        before ``arena._lock`` (never acquire ``_lock`` while holding
        ``slot.lease_lock``, and never hold two slots' ``lease_lock``s at once).
        All CUDA syncs, arena alloc/free calls, and external reclaimer calls
        run outside ``_lock``.

    Caller Contracts:
        - Launch kernels reading or writing a yielded :class:`KvBatch` on the
          CUDA stream active when entering ``produce()`` or ``read()``, before
          exiting its ``with`` block.
        - Close this pool outside active ``produce()``/``read()`` sessions and
          before closing the backing ``Uint8DualArena``.
    """

    def __init__(
        self,
        arena: Uint8DualArena,
        schema: SlotSchema,
        *,
        max_slots: int | None = None,
        budget: VramBudget | None = None,
    ) -> None:
        """Bind high-end drawers of ``arena`` according to ``schema``."""
        resolved_max, effective_budget = _validate_slab_pool_base_args(
            "KvChunkPool", arena, schema, max_slots=max_slots, budget=budget
        )
        device = arena.device
        self._arena = arena
        self._schema = schema
        self._device = device
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._op_cond = threading.Condition(self._lock)
        self._wave_releases = _WaveReleaseTracker(device, self._lock, self._op_cond)
        self._side_pools: dict[str, TypedSidePool] = {}
        self._views: _PoolViews | None = None
        self._slots: dict[int, _KvSlot] = {}
        self._key_to_slot: dict[Hashable, _KvSlot] = {}
        self._producing_keys: set[Hashable] = set()
        self._active_sessions = 0
        self._inflight_arena_allocs = 0
        self._inflight_arena_frees = 0
        self._produced_count = 0
        self._discard_count = 0
        self._closing = False
        self._closed = False
        self._reclaim_cb: Callable[[int], None] = self._reclaim_high_frontier
        self._layout: HighSlotLayout | None = None
        self._max_slots = 0
        self._configured_fresh_layout = False
        self._owner_claimed = False

        try:
            layout, fresh_layout = arena._claim_high_owner(
                self._reclaim_cb, schema.drawer_bytes, resolved_max
            )
            self._layout = layout
            self._max_slots = layout.max_slots
            self._configured_fresh_layout = fresh_layout
            self._owner_claimed = True
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
            assert merged is not None
            if merged is not exc:
                raise merged from exc
            raise

    @property
    def schema(self) -> SlotSchema:
        """Return the :class:`SlotSchema` describing arena and side fields."""
        return self._schema

    @property
    def layout(self) -> HighSlotLayout:
        """Return the active descending :class:`HighSlotLayout` in the arena."""
        assert self._layout is not None
        return self._layout

    @property
    def max_slots(self) -> int:
        """Return the maximum number of high-end GPU slots owned by this pool."""
        return self._max_slots

    @property
    def device(self) -> torch.device:
        """Return the canonical device of the backing arena and side pools."""
        return self._device

    @property
    def is_open(self) -> bool:
        """Return ``True`` while the pool and its backing arena remain open."""
        with self._lock:
            return not self._closed and not self._closing and self._arena.is_open

    def _is_stopped_unlocked(self) -> bool:
        return self._closed or self._closing or not self._arena.is_open

    def _require_open_unlocked(self) -> None:
        if self._is_stopped_unlocked():
            raise RuntimeError("KvChunkPool is closed.")

    def _require_views(self) -> _PoolViews:
        with self._lock:
            self._require_open_unlocked()
            assert self._views is not None
            return self._views

    @property
    def arena_views(self) -> Mapping[str, torch.Tensor]:
        """Return persistent strided ``[max_slots, *shape]`` ``uint8`` field views."""
        return self._require_views().arena_views

    @property
    def side_views(self) -> Mapping[str, torch.Tensor]:
        """Return persistent strided ``[max_slots, *shape]`` typed side-pool views."""
        return self._require_views().side_views

    @property
    def arena_bytes(self) -> torch.Tensor:
        """Return the underlying 1-D ``torch.uint8`` arena tensor."""
        return self._require_views().arena_bytes

    @property
    def slot_offsets(self) -> Mapping[str, torch.Tensor]:
        """Return ``int64[max_slots]`` arena byte offsets per arena field."""
        return self._require_views().slot_offsets

    def contains(self, key: object) -> bool:
        """Return ``True`` if ``key`` currently resides in a ``READY`` VRAM slot."""
        if key is None:
            return False
        try:
            hash(key)
        except TypeError:
            return False
        with self._lock:
            return key in self._key_to_slot

    def __contains__(self, key: object) -> bool:
        return self.contains(key)

    def stats(self) -> KvPoolStats:
        """Return a snapshot of slot residency and lifetime counters."""
        with self._lock:
            resident = len(self._key_to_slot)
            allocated = len(self._slots)
            empty = sum(1 for s in self._slots.values() if s.phase is _SlotPhase.EMPTY)
            return KvPoolStats(
                max_slots=self._max_slots,
                allocated_gpu_slots=allocated,
                resident_chunks=resident,
                empty_slots=empty,
                produced=self._produced_count,
                discards=self._discard_count,
            )

    def _free_slot_to_arena(self, slot: _KvSlot, *, sync_lease: bool) -> None:
        if sync_lease and slot.lease is not None:
            lease = slot.lease
            with slot.lease_lock:
                _sync_interrupt_safe(lease.synchronize_and_reset)
        popped = False
        with self._lock:
            if self._slots.get(slot.slot_id) is slot:
                del self._slots[slot.slot_id]
                self._inflight_arena_frees += 1
                popped = True
        if not popped:
            return
        try:
            if self._arena.is_open:
                _sync_interrupt_safe(lambda: self._arena.free_high_slot(slot.slot_id))
        finally:
            with self._lock:
                self._inflight_arena_frees = max(0, self._inflight_arena_frees - 1)
                self._op_cond.notify_all()

    def _pick_reusable_empty_unlocked(
        self,
        key: Hashable,
        claimed: list[tuple[Hashable, _KvSlot, bool]],
    ) -> _KvSlot | None:
        # Prefer top-most EMPTY slots (highest slot_id) so lower free slots sit at the bottom frontier.
        candidates = [
            sid
            for sid, slot in self._slots.items()
            if slot.phase is _SlotPhase.EMPTY and slot.active_readers == 0
        ]
        if not candidates:
            return None
        slot = self._slots[max(candidates)]
        slot.phase = _SlotPhase.PRODUCING
        slot.key = key
        claimed.append((key, slot, False))
        return slot

    def _alloc_high_slot_for_produce(
        self,
        key: Hashable,
        claimed: list[tuple[Hashable, _KvSlot, bool]],
    ) -> _KvSlot | None:
        with self._lock:
            self._require_open_unlocked()
            if len(self._slots) + self._inflight_arena_allocs >= self._max_slots:
                return None
            self._inflight_arena_allocs += 1
        try:
            slot_id = self._arena.alloc_high_slot(allow_reclaim=True)
            alloc_stream = (
                torch.cuda.current_stream(self._device) if self._device.type == "cuda" else None
            )
            new_slot = _KvSlot(
                slot_id=slot_id,
                phase=_SlotPhase.PRODUCING,
                key=key,
                lease=None if alloc_stream is None else StreamLease(alloc_stream),
            )
            inserted = False
            try:
                with self._lock:
                    if not self._is_stopped_unlocked():
                        self._slots[slot_id] = new_slot
                        claimed.append((key, new_slot, True))
                        inserted = True
                if not inserted:
                    raise RuntimeError("KvChunkPool was closed during slot allocation.")
            except BaseException:
                if not inserted and self._arena.is_open:
                    _sync_interrupt_safe(lambda: self._arena.free_high_slot(slot_id))
                raise
            return new_slot
        finally:
            with self._lock:
                self._inflight_arena_allocs = max(0, self._inflight_arena_allocs - 1)
                self._op_cond.notify_all()

    def _claim_slot_for_produce(
        self,
        key: Hashable,
        claimed: list[tuple[Hashable, _KvSlot, bool]],
    ) -> _KvSlot:
        while True:
            with self._lock:
                self._require_open_unlocked()
                reused = self._pick_reusable_empty_unlocked(key, claimed)
                if reused is not None:
                    return reused
            try:
                allocated = self._alloc_high_slot_for_produce(key, claimed)
            except ArenaCapacityError:
                allocated = None
            if allocated is not None:
                return allocated
            with self._lock:
                self._require_open_unlocked()
                has_pending_frees = (
                    self._inflight_arena_frees > 0
                    or self._wave_releases.inflight_reaps > 0
                    or any(s.phase is _SlotPhase.RECLAIMING for s in self._slots.values())
                )
                if not has_pending_frees:
                    raise ArenaCapacityError(
                        f"KvChunkPool has no available VRAM slots for chunk {key!r}."
                    )
                self._op_cond.wait(timeout=0.05)

    def _preflight_produce_unlocked(self, keys: tuple[Hashable, ...]) -> _PoolViews:
        self._require_open_unlocked()
        assert self._views is not None
        if not keys:
            return self._views
        for key in keys:
            if key in self._key_to_slot:
                raise ValueError(f"Chunk key {key!r} is already resident in KvChunkPool.")
            if key in self._producing_keys:
                raise ValueError(f"Chunk key {key!r} is already being produced.")
        needed = len(self._key_to_slot) + len(self._producing_keys) + len(keys)
        if needed > self._max_slots:
            raise ArenaCapacityError(
                f"KvChunkPool batch requires {needed} slots "
                f"(resident={len(self._key_to_slot)}, producing={len(self._producing_keys)}, "
                f"requested={len(keys)}), exceeding max_slots={self._max_slots}."
            )
        return self._views

    def _rollback_produce_acquire(
        self,
        keys: tuple[Hashable, ...],
        claimed: Sequence[tuple[Hashable, _KvSlot, bool]],
    ) -> BaseException | None:
        rollback_err: BaseException | None = None
        key_set = set(keys)
        with self._lock:
            for _k, slot, newly_allocated in claimed:
                if not newly_allocated and self._slots.get(slot.slot_id) is slot:
                    slot.key = None
                    slot.phase = _SlotPhase.EMPTY
        for _k, slot, newly_allocated in claimed:
            if newly_allocated:
                try:
                    self._free_slot_to_arena(slot, sync_lease=False)
                except BaseException as exc:
                    rollback_err = _merge_error(rollback_err, exc)
        with self._lock:
            for slot in self._slots.values():
                if slot.phase is _SlotPhase.PRODUCING and slot.key in key_set:
                    slot.key = None
                    slot.phase = _SlotPhase.EMPTY
            self._producing_keys.difference_update(key_set)
            self._op_cond.notify_all()
        return rollback_err

    def _commit_produce_unlocked(
        self, claimed: Sequence[tuple[Hashable, _KvSlot, bool]]
    ) -> None:
        for key, slot, _n in claimed:
            self._producing_keys.discard(key)
            slot.key = key
            slot.phase = _SlotPhase.READY
            self._key_to_slot[key] = slot
        self._produced_count += len(claimed)

    def _rollback_produce_write_unlocked(
        self,
        keys: tuple[Hashable, ...],
        claimed: Sequence[tuple[Hashable, _KvSlot, bool]],
    ) -> None:
        self._producing_keys.difference_update(keys)
        for key, slot, _n in claimed:
            if self._key_to_slot.get(key) is slot:
                del self._key_to_slot[key]
            if self._slots.get(slot.slot_id) is slot:
                slot.key = None
                slot.phase = _SlotPhase.EMPTY
        self._op_cond.notify_all()

    def _make_batch(
        self,
        views: _PoolViews,
        keys: tuple[Hashable | None, ...],
        slot_id_list: Sequence[int],
        active_slots: int,
    ) -> KvBatch:
        if slot_id_list:
            slot_ids = _make_device_int_table(slot_id_list, device=self._device)
        else:
            with torch.inference_mode(False):
                slot_ids = torch.empty((0,), dtype=torch.int32, device=self._device)
        return KvBatch(
            keys=keys,
            slot_ids=slot_ids,
            active_slots=active_slots,
            arena_views=views.arena_views,
            side_views=views.side_views,
            arena_bytes=views.arena_bytes,
            slot_offsets=views.slot_offsets,
        )

    def _open_write_stack(
        self,
        stack: contextlib.ExitStack,
        claimed: Sequence[tuple[Hashable, _KvSlot, bool]],
    ) -> torch.cuda.Stream | None:
        if self._device.type != "cuda":
            return None
        cur_stream = torch.cuda.current_stream(self._device)
        for _k, slot, _n in claimed:
            _push_locked_write(stack, slot, cur_stream)
        return cur_stream

    def _wait_stream_on_exit(self, cur_stream: torch.cuda.Stream | None) -> None:
        if cur_stream is not None:
            end_stream = torch.cuda.current_stream(self._device)
            if end_stream != cur_stream:
                cur_stream.wait_stream(end_stream)

    def _handle_produce_abort(
        self,
        exc: BaseException,
        resolved: tuple[Hashable, ...],
        claimed: Sequence[tuple[Hashable, _KvSlot, bool]],
        *,
        entered_write: bool,
    ) -> BaseException:
        if not entered_write:
            rollback_err = self._rollback_produce_acquire(resolved, claimed)
            merged = _merge_error(exc, rollback_err)
            assert merged is not None
            return merged
        with self._lock:
            self._rollback_produce_write_unlocked(resolved, claimed)
        return exc

    @contextlib.contextmanager
    def produce(self, keys: Sequence[Hashable]) -> Iterator[KvBatch]:
        """Reserve VRAM high slots for new GPU-born chunks and commit them on clean exit.

        Empty ``keys`` yields an empty ``KvBatch(active_slots=0)``. Reuses
        top-most ``EMPTY`` slots before allocating descending high slots from
        ``Uint8DualArena`` (possibly reclaiming low-end peer slots). Kernels
        populating a yielded ``KvBatch`` must launch on the CUDA stream active
        when entering ``produce()`` before exiting its ``with`` block; keys commit
        atomically only on clean exit.

        Raises:
            TypeError: If ``keys`` is a string/bytes, non-sequence, or contains unhashable keys.
            ValueError: If ``keys`` contains ``None``, duplicates, or resident/producing keys.
            ArenaCapacityError: If ``resident + producing + len(keys) > max_slots`` or
                the shared arena lacks room for a required new high slot (any slots newly
                allocated for earlier keys in the batch are returned to the arena).
            RuntimeError: If this pool or its backing arena is closed.
        """
        resolved = _validate_non_none_keys(keys)
        if len(set(resolved)) != len(resolved):
            raise ValueError("produce() keys must not contain duplicates.")
        self._wave_releases.reap(self._is_stopped_unlocked)
        claimed: list[tuple[Hashable, _KvSlot, bool]] = []
        session_active = False
        entered_write = False
        committed = False
        try:
            with self._lock:
                views = self._preflight_produce_unlocked(resolved)
                if resolved:
                    self._producing_keys.update(resolved)
                self._active_sessions += 1
                session_active = True

            if not resolved:
                yield self._make_batch(views, (), (), 0)
                committed = True
                return

            for key in resolved:
                self._claim_slot_for_produce(key, claimed)
            batch = self._make_batch(
                views, resolved, [slot.slot_id for _k, slot, _n in claimed], len(claimed)
            )
            entered_write = True
            with contextlib.ExitStack() as stack:
                cur_stream = self._open_write_stack(stack, claimed)
                try:
                    yield batch
                finally:
                    self._wait_stream_on_exit(cur_stream)

            with self._lock:
                self._commit_produce_unlocked(claimed)
                committed = True
        except BaseException as exc:
            if session_active and not committed:
                merged = self._handle_produce_abort(
                    exc, resolved, claimed, entered_write=entered_write
                )
                if merged is not exc:
                    raise merged from exc
            raise
        finally:
            if session_active:
                with self._lock:
                    self._active_sessions = max(0, self._active_sessions - 1)
                    self._op_cond.notify_all()

    def _resolve_read_slots_unlocked(
        self, resolved: tuple[Hashable | None, ...]
    ) -> tuple[_PoolViews, list[int], int, _WaveLease | None]:
        self._require_open_unlocked()
        assert self._views is not None
        views = self._views
        slot_id_list: list[int] = []
        unique_slots: list[_KvSlot] = []
        seen_slots: set[int] = set()
        for key in resolved:
            if key is None:
                slot_id_list.append(-1)
                continue
            if key in self._producing_keys:
                raise KeyError(f"Chunk key {key!r} is currently being produced.")
            slot = self._key_to_slot.get(key)
            if slot is None or slot.phase is not _SlotPhase.READY or slot.key != key:
                raise KeyError(f"Chunk key {key!r} is not resident in KvChunkPool.")
            slot_id_list.append(slot.slot_id)
            if slot.slot_id not in seen_slots:
                seen_slots.add(slot.slot_id)
                unique_slots.append(slot)
        lease = self._wave_releases.new_lease() if unique_slots else None
        if lease is not None:
            for slot in unique_slots:
                lease.add_unlocked(slot)
        self._active_sessions += 1
        return views, slot_id_list, len(unique_slots), lease

    @contextlib.contextmanager
    def read(self, keys: Sequence[Hashable | None]) -> Iterator[KvBatch]:
        """Bind resident chunks in one :class:`KvBatch` for the active CUDA stream.

        Duplicate non-``None`` keys are deduplicated onto the same VRAM slot
        (counted once in ``active_slots``); ``None`` sentinels map to ``-1``
        in ``batch.slot_ids``. On CUDA, consumer streams must be active on
        entry or switch before exit so ``StreamLease`` orders future writes and
        reclaims after all readers.

        Raises:
            TypeError: If ``keys`` is a string/bytes, non-sequence, or contains unhashable keys.
            KeyError: Atomically if any non-``None`` key is missing or still being produced.
            RuntimeError: If this pool or its backing arena is closed.
        """
        resolved = _validate_read_keys(keys)
        self._wave_releases.reap(self._is_stopped_unlocked)
        lease: _WaveLease | None = None
        session_active = False
        caller_err: BaseException | None = None
        release_err: BaseException | None = None
        try:
            with self._lock:
                views, slot_id_list, active_slots, lease = self._resolve_read_slots_unlocked(
                    resolved
                )
                session_active = True
            if lease is not None:
                lease.acquire()
            yield self._make_batch(views, resolved, slot_id_list, active_slots)
        except BaseException as exc:
            caller_err = exc
            raise
        finally:
            if lease is not None:
                try:
                    lease.release()
                except BaseException as exc:
                    release_err = exc
                finally:
                    lease.handoff_to_pool()
            if session_active:
                with self._lock:
                    self._active_sessions = max(0, self._active_sessions - 1)
                    self._op_cond.notify_all()
            if caller_err is None and release_err is not None:
                raise release_err
            if (
                caller_err is not None
                and release_err is not None
                and isinstance(caller_err, Exception)
                and not isinstance(release_err, Exception)
            ):
                raise release_err from caller_err

    def discard(self, keys: Sequence[Hashable]) -> int:
        """Retire resident chunks immediately (non-blocking) for reuse or frontier reclaim.

        Unknown or duplicate keys are ignored; returns the number of distinct
        resident chunks transitioned to ``EMPTY``. Makes zero CUDA calls.

        Raises:
            TypeError: If ``keys`` is a string/bytes, non-sequence, or contains unhashable keys.
            ValueError: If ``keys`` contains ``None``.
            RuntimeError: Atomically if any key is currently being produced, has active
                readers, or if the pool is closed.
        """
        unique = list(dict.fromkeys(_validate_non_none_keys(keys)))
        self._wave_releases.reap(self._is_stopped_unlocked)
        with self._lock:
            self._require_open_unlocked()
            to_discard: list[tuple[Hashable, _KvSlot]] = []
            for key in unique:
                if key in self._producing_keys:
                    raise RuntimeError(f"Cannot discard chunk {key!r} while it is being produced.")
                slot = self._key_to_slot.get(key)
                if slot is None:
                    continue
                if slot.phase is not _SlotPhase.READY or slot.key != key or slot.active_readers > 0:
                    raise RuntimeError(
                        f"Cannot discard chunk {key!r} while in phase {slot.phase.name} "
                        f"with {slot.active_readers} active reader(s)."
                    )
                to_discard.append((key, slot))
            for key, slot in to_discard:
                self._key_to_slot.pop(key, None)
                slot.key = None
                slot.phase = _SlotPhase.EMPTY
            self._discard_count += len(to_discard)
            self._op_cond.notify_all()
            return len(to_discard)

    def _pick_reclaim_candidate_unlocked(self, needed_high_frontier: int) -> _KvSlot | None:
        layout = self.layout
        while True:
            while (
                self._inflight_arena_frees > 0
                and not self._is_stopped_unlocked()
                and (
                    not self._slots
                    or layout.slot_offset(min(self._slots)) >= needed_high_frontier
                )
            ):
                self._op_cond.wait(timeout=0.05)
            if self._is_stopped_unlocked() or not self._slots:
                return None
            bottom_id = min(self._slots)
            if layout.slot_offset(bottom_id) >= needed_high_frontier:
                return None
            slot = self._slots[bottom_id]
            if slot.phase is _SlotPhase.RECLAIMING:
                self._op_cond.wait(timeout=0.05)
                continue
            if slot.phase is not _SlotPhase.EMPTY or slot.active_readers > 0:
                return None
            slot.phase = _SlotPhase.RECLAIMING
            return slot

    def _reclaim_high_frontier(self, needed_high_frontier: int) -> None:
        self._wave_releases.reap(self._is_stopped_unlocked)
        while True:
            with self._lock:
                candidate = self._pick_reclaim_candidate_unlocked(needed_high_frontier)
            if candidate is None:
                return
            try:
                self._free_slot_to_arena(candidate, sync_lease=True)
            except BaseException:
                with self._lock:
                    if candidate.slot_id in self._slots and candidate.phase is _SlotPhase.RECLAIMING:
                        candidate.phase = _SlotPhase.EMPTY
                    self._op_cond.notify_all()
                raise

    def _teardown_resources(self, *, slots: Sequence[_KvSlot] = ()) -> BaseException | None:
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
        if self._owner_claimed and self._arena.is_open and not remaining_slots:
            reset_layout = self._layout if self._configured_fresh_layout else None

            def _release_owner() -> None:
                self._arena._release_high_owner(self._reclaim_cb, reset_expected=reset_layout)
                self._owner_claimed = False

            _capture(_release_owner)
        for name, side_pool in list(self._side_pools.items()):
            def _close_side(pool_name: str = name, sp: TypedSidePool = side_pool) -> None:
                sp.close()
                self._side_pools.pop(pool_name, None)

            _capture(_close_side)

        with self._lock:
            if first_err is None:
                self._views = None
                self._slots.clear()
                self._key_to_slot.clear()
                self._producing_keys.clear()
                self._wave_releases.pending.clear()
                self._closing = False
                self._closed = True
        return first_err

    def close(self) -> None:
        """Synchronize pending CUDA leases, free arena high slots, and release all views."""
        with self._close_lock:
            slots: list[_KvSlot] = []
            pending_by_slot: dict[int, list[_PendingWaveRelease]] = {}
            entered_close = False
            try:
                with self._lock:
                    if self._closed:
                        return
                    pending_counts = self._wave_releases.pending_counts_unlocked()
                    if (
                        self._active_sessions > 0
                        or self._inflight_arena_allocs > 0
                        or self._inflight_arena_frees > 0
                        or self._wave_releases.inflight_reaps > 0
                        or self._producing_keys
                        or any(
                            s.active_readers > pending_counts.get(s.slot_id, 0)
                            or s.phase in (_SlotPhase.PRODUCING, _SlotPhase.RECLAIMING)
                            for s in self._slots.values()
                        )
                    ):
                        raise RuntimeError(
                            "Cannot close KvChunkPool while produce, read, or reclaim work is active."
                        )
                    slots.extend(self._slots.values())
                    self._closing = True
                    entered_close = True
                    self._wave_releases.drain_by_slot_unlocked(out=pending_by_slot)

                self._wave_releases.synchronize_slots_for_close(slots, pending_by_slot)
            except BaseException:
                if entered_close:
                    with self._lock:
                        self._wave_releases.restore_unlocked(pending_by_slot)
                raise

            err = self._teardown_resources(slots=slots)
            if err is not None:
                raise err

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "KvBatch",
    "KvChunkPool",
    "KvPoolStats",
]
