"""Tests for :class:`KvChunkPool` over high :class:`Uint8DualArena` slots and :class:`TypedSidePool`."""

from __future__ import annotations

import ast
import contextlib
import mmap
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import torch

from piper_offload import (
    ArenaCapacityError,
    DeviceTransferContext,
    ExpertSlabPool,
    KvBatch,
    KvChunkLocation,
    KvChunkPool,
    KvPoolStats,
    PinManager,
    SlotSchema,
    Uint8DualArena,
    VramBudget,
)
import piper_offload.kv_chunk_pool as kv_pool_mod
from piper_offload._slab_pool_common import _WaveReleaseTracker
from piper_offload.dual_arena import HighSlotLayout
from piper_offload.host_arena import ArenaAllocation
from piper_offload.kv_chunk_pool import (
    _SlotPhase,
    _SpillState,
    _WaveLoopState,
    _build_host_chunk_layout,
)
from piper_offload.stream_lease import StreamLease

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _supports_piper_kv_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from piper_kernels import (
            PiperKVChunkPool,
            freeze_kv_chunks_,
            kv_chunk_storage_specs,
            merge_attention_lse,
            piper_attention_decode,
        )

        _ = (
            PiperKVChunkPool,
            freeze_kv_chunks_,
            kv_chunk_storage_specs,
            merge_attention_lse,
            piper_attention_decode,
        )
    except ImportError:
        return False
    return True


PIPER_KV_CUDA = pytest.mark.skipif(
    not _supports_piper_kv_cuda(),
    reason="CUDA and piper_kernels KV APIs required",
)


class _SimpleSlabBackend:
    def __init__(self) -> None:
        self.registered: dict[int, int] = {}

    def register(self, pointer: int, size: int) -> bool:
        self.registered[pointer] = size
        return True

    def unregister(self, pointer: int) -> None:
        self.registered.pop(pointer, None)


def _kv_schema(*, k_bytes: int = 256, v_bytes: int = 256, scale_len: int = 8) -> SlotSchema:
    return SlotSchema(
        {"k": k_bytes, "v": v_bytes},
        {"scale": (torch.float32, (scale_len,))},
        field_alignment=256,
        side_alignment=16,
    )


def _expert_schema(*, up_bytes: int = 256, down_bytes: int = 256, scale_len: int = 8) -> SlotSchema:
    return SlotSchema(
        {"up": up_bytes, "down": down_bytes},
        {"scale": (torch.float32, (scale_len,))},
        field_alignment=256,
        side_alignment=16,
    )


def _make_chunk_data(
    seed: int, *, k_bytes: int = 256, v_bytes: int = 256, scale_len: int = 8
) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        "k": torch.randint(0, 256, (k_bytes,), generator=gen, dtype=torch.uint8),
        "v": torch.randint(0, 256, (v_bytes,), generator=gen, dtype=torch.uint8),
        "scale": torch.randn((scale_len,), generator=gen, dtype=torch.float32),
    }


def _make_expert_data(
    seed: int, *, up_bytes: int = 256, down_bytes: int = 256, scale_len: int = 8
) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        "up": torch.randint(0, 256, (up_bytes,), generator=gen, dtype=torch.uint8),
        "down": torch.randint(0, 256, (down_bytes,), generator=gen, dtype=torch.uint8),
        "scale": torch.randn((scale_len,), generator=gen, dtype=torch.float32),
    }


def _write_kv_batch(batch: KvBatch, payloads: dict[Any, dict[str, torch.Tensor]]) -> None:
    dev = batch.slot_ids.device
    for idx, key in enumerate(batch.keys):
        assert key is not None
        slot = int(batch.slot_ids[idx].item())
        assert slot >= 0
        src = payloads[key]
        for fname in ("k", "v"):
            batch.arena_views[fname][slot].copy_(src[fname].to(dev), non_blocking=(dev.type == "cuda"))
        batch.side_views["scale"][slot].copy_(src["scale"].to(dev), non_blocking=(dev.type == "cuda"))


def _verify_wave_batch(
    batch: KvBatch,
    payloads: dict[Any, dict[str, torch.Tensor]],
    *,
    expected_keys: Sequence[Any] | None = None,
) -> list[int]:
    if expected_keys is not None:
        assert batch.keys == tuple(expected_keys)
    assert batch.slot_ids.dtype is torch.int32
    assert tuple(batch.slot_ids.shape) == (len(batch.keys),)
    present_indices: list[int] = []
    for idx, key in enumerate(batch.keys):
        slot = int(batch.slot_ids[idx].item())
        if key is None or slot == -1:
            assert slot == -1
            continue
        present_indices.append(idx)
        src = payloads[key]
        for fname in ("k", "v"):
            expected = src[fname].reshape(-1).cpu()
            torch.testing.assert_close(
                batch.arena_views[fname][slot].reshape(-1).cpu(), expected, rtol=0, atol=0
            )
            offset = int(batch.slot_offsets[fname][slot].item())
            torch.testing.assert_close(
                batch.arena_bytes[offset : offset + expected.numel()].cpu(),
                expected,
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            batch.side_views["scale"][slot].cpu(), src["scale"].cpu(), rtol=0, atol=0
        )
    return present_indices


def _produce_chunks(
    pool: KvChunkPool,
    keys: Sequence[Any],
    payloads: dict[Any, dict[str, torch.Tensor]],
) -> None:
    with pool.produce(keys) as batch:
        _write_kv_batch(batch, payloads)


def _produce_batched(
    pool: KvChunkPool,
    keys: Sequence[Any],
    payloads: dict[Any, dict[str, torch.Tensor]],
    *,
    batch_size: int | None = None,
) -> None:
    step = pool.max_slots if batch_size is None else batch_size
    for start in range(0, len(keys), step):
        _produce_chunks(pool, keys[start : start + step], payloads)


# ---------------------------------------------------------------------------
# Validation, Persistent Views & Architecture Boundary
# ---------------------------------------------------------------------------


def test_kv_chunk_pool_validation_and_layout_alignment() -> None:
    schema = _kv_schema()
    with Uint8DualArena(4 * schema.drawer_bytes, device="cpu", alignment=256) as arena:
        for bad_kw, match in (
            ({"max_slots": 0}, "positive"),
            ({"max_slots": True}, "positive"),
            ({"spill_watermark": 0.0}, r"\(0, 1\]"),
            ({"spill_watermark": 1.5}, r"\(0, 1\]"),
            ({"spill_watermark": True}, "float"),
            ({"max_host_bytes": 0}, "positive"),
        ):
            with pytest.raises(ValueError, match=match):
                KvChunkPool(arena, schema, **bad_kw)  # type: ignore[arg-type]

        small_schema = SlotSchema({"k": 64}, {"s": (torch.float32, (2,))})
        arena.configure_high_slots(small_schema.drawer_bytes)
        slot = arena.alloc_high_slot()
        with pytest.raises(RuntimeError, match="active high slots"):
            KvChunkPool(arena, small_schema)
        arena.free_high_slot(slot)
        with pytest.raises(ValueError, match="smaller than"):
            KvChunkPool(arena, schema)

    with Uint8DualArena(2048, device="cpu", alignment=128) as bad_align:
        with pytest.raises(ValueError, match="Arena alignment"):
            KvChunkPool(bad_align, schema)

    with Uint8DualArena(4 * schema.drawer_bytes, device="cpu", alignment=256) as ok_arena:
        with KvChunkPool(ok_arena, schema, max_slots=4, spill_watermark=None) as pool:
            for bad_fn, match in (
                (lambda: _produce_chunks(pool, [], {}), "at least one"),
                (lambda: _produce_chunks(pool, "abc", {}), "sequence"),  # type: ignore[arg-type]
                (lambda: _produce_chunks(pool, ["a", None], {}), "only supported in waves"),  # type: ignore[arg-type]
                (lambda: _produce_chunks(pool, ["a", "a"], {}), "distinct"),
            ):
                with pytest.raises((TypeError, ValueError), match=match):
                    bad_fn()
            with pytest.raises(KeyError, match="Unknown"):
                with pool.waves(["missing"]):
                    pass


def test_kv_chunk_pool_persistent_views_and_zero_piper_kernels_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for rel_path in ("src/piper_offload/kv_chunk_pool.py", "src/piper_offload/_slab_pool_common.py"):
        tree = ast.parse((repo_root / rel_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all("piper_kernels" not in alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert "piper_kernels" not in node.module

    schema = _kv_schema()
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    try:
        with (
            Uint8DualArena(4 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=4, pin_manager=pm, spill_watermark=None) as pool,
        ):
            assert isinstance(pool.layout, HighSlotLayout)
            a_views = pool.arena_views
            s_views = pool.side_views
            a_bytes = pool.arena_bytes
            s_offs = pool.slot_offsets
            assert pool.schema is schema
            assert pool.max_slots == 4
            assert pool.device == torch.device("cpu")
            assert isinstance(pool.stats(), KvPoolStats)

            payloads = {i: _make_chunk_data(i) for i in range(6)}
            with pool.produce([0, 1]) as pbatch:
                for k in ("k", "v"):
                    assert pbatch.arena_views[k] is a_views[k]
                    assert pbatch.slot_offsets[k] is s_offs[k]
                assert pbatch.side_views["scale"] is s_views["scale"]
                assert pbatch.arena_bytes is a_bytes
                _write_kv_batch(pbatch, payloads)

            with pool.waves([0, 1]) as waves:
                wbatch = next(waves)
                assert wbatch.arena_views["k"] is a_views["k"]
                assert wbatch.side_views["scale"] is s_views["scale"]
                assert wbatch.arena_bytes is a_bytes
                _verify_wave_batch(wbatch, payloads)
        assert not pool.is_open
        with pytest.raises(RuntimeError, match="closed"):
            _ = pool.arena_views
    finally:
        pm.clear()

# ---------------------------------------------------------------------------
# produce() Lifecycle, Exception Rollback & Stream Ordering
# ---------------------------------------------------------------------------


def test_produce_batched_rollback_and_duplicate_key_rejection() -> None:
    schema = _kv_schema()
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {f"c{i}": _make_chunk_data(10 + i) for i in range(5)}
    try:
        with (
            Uint8DualArena(4 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=4, pin_manager=pm, spill_watermark=None) as pool,
        ):
            _produce_chunks(pool, ["c0"], payloads)
            assert pool.contains("c0") and ("c0" in pool) and ("missing" not in pool)
            assert [[] not in pool][0]  # unhashable key in __contains__ returns False
            assert pool.location("c0") is KvChunkLocation.GPU_DIRTY

            with pytest.raises(ValueError, match="already tracked"):
                _produce_chunks(pool, ["c0"], payloads)

            with pytest.raises(RuntimeError, match="abort_produce"):
                with pool.produce(["c1", "c2"]) as batch:
                    _write_kv_batch(batch, payloads)
                    raise RuntimeError("abort_produce")

            assert not pool.contains("c1")
            assert not pool.contains("c2")
            assert pool.stats().resident_gpu_chunks == 1
            assert len(arena.active_high_slots) == 1

            _produce_chunks(pool, ["c1", "c2"], payloads)
            assert pool.stats().resident_gpu_chunks == 3
            with pool.waves(["c0", "c1", "c2"]) as waves:
                wave_list = list(waves)
                assert len(wave_list) == 1
                _verify_wave_batch(wave_list[0], payloads)
    finally:
        pm.clear()

@CUDA
def test_cuda_produce_and_wave_stream_ordering() -> None:
    device = torch.device("cuda")
    schema = _kv_schema()
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {f"c{i}": _make_chunk_data(20 + i) for i in range(4)}
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)
    try:
        with (
            Uint8DualArena(4 * schema.drawer_bytes, device=device, alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=4, pin_manager=pm, spill_watermark=None) as pool,
        ):
            with torch.cuda.stream(stream_a):
                _produce_chunks(pool, ["c0", "c1"], payloads)
            with torch.cuda.stream(stream_b):
                with pool.waves(["c0", "c1"]) as waves:
                    batch = next(waves)
                    _verify_wave_batch(batch, payloads)
    finally:
        pm.clear()

# ---------------------------------------------------------------------------
# waves(): Empty/None, Prefix Branching, >> VRAM Streaming & Compaction
# ---------------------------------------------------------------------------


def test_waves_empty_none_prefix_sharing_and_branching() -> None:
    schema = _kv_schema()
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {k: _make_chunk_data(30 + idx) for idx, k in enumerate(("p0", "p1", "a0", "b0"))}
    try:
        with (
            Uint8DualArena(4 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=4, pin_manager=pm, spill_watermark=None) as pool,
        ):
            with pool.waves([]) as empty_waves:
                assert list(empty_waves) == []
            with pool.waves([None, None]) as none_waves:
                assert list(none_waves) == []

            _produce_chunks(pool, ["p0", "p1", "a0", "b0"], payloads)
            with pool.waves(["p0", "p1", None, "p0", "a0"]) as branch_a:
                batches_a = list(branch_a)
                assert len(batches_a) == 1
                b_a = batches_a[0]
                assert b_a.is_hit_wave is True
                assert int(b_a.slot_ids[0].item()) == int(b_a.slot_ids[3].item()) >= 0
                assert int(b_a.slot_ids[2].item()) == -1
                _verify_wave_batch(b_a, payloads)
            lru_before = list(pool._chunks.keys())
            with pool.waves(["p0", "p1", "b0"], record_stats=False) as branch_b:
                batches_b = list(branch_b)
                assert len(batches_b) == 1
                _verify_wave_batch(batches_b[0], payloads)
            assert list(pool._chunks.keys()) == lru_before
            st = pool.stats()
            assert st.resident_gpu_chunks == 4
            assert st.hits == 3  # branch_b used record_stats=False
            assert st.waves_executed == 1
    finally:
        pm.clear()

def test_waves_streaming_beyond_vram_and_compaction_modes() -> None:
    schema = _kv_schema()
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    keys = [f"k{i}" for i in range(8)]
    payloads = {k: _make_chunk_data(50 + i) for i, k in enumerate(keys)}
    try:
        with (
            Uint8DualArena(3 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=3, pin_manager=pm, spill_watermark=None) as pool,
        ):
            with pytest.raises(ArenaCapacityError):
                _produce_chunks(pool, keys[:4], payloads)
            _produce_batched(pool, keys, payloads)
            assert pool.stats().spills >= 5
            assert pool.stats().resident_gpu_chunks == 3

            seen_indices: list[int] = []
            hit_waves = 0
            miss_waves = 0
            req: list[Any] = [*keys, None]
            with pool.waves(req, max_wave_slots=2, allow_split=True) as waves:
                for batch in waves:
                    assert int(batch.slot_ids[-1].item()) == -1
                    active_sids = [int(s) for s in batch.slot_ids[:-1].tolist() if int(s) >= 0]
                    if batch.is_hit_wave:
                        hit_waves += 1
                        assert len(set(active_sids)) == 3
                    else:
                        miss_waves += 1
                        assert 1 <= len(set(active_sids)) <= 2
                    seen_indices.extend(_verify_wave_batch(batch, payloads, expected_keys=req))
            assert (hit_waves, miss_waves) == (1, 3)
            assert sorted(seen_indices) == list(range(8))

            # allow_split=False compacts all <= max_slots chunks into 1 wave after evicting them.
            pool.evict_to_host()
            assert pool.stats().resident_gpu_chunks == 0
            with pool.waves(["k0", "k4", "k7", "k0", None], allow_split=False) as compact_waves:
                batches = list(compact_waves)
                assert len(batches) == 1
                assert int(batches[0].slot_ids[-1].item()) == -1
                assert int(batches[0].slot_ids[0].item()) == int(batches[0].slot_ids[3].item()) >= 0
                _verify_wave_batch(batches[0], payloads)

            with pytest.raises(ArenaCapacityError):
                with pool.waves(["k0", "k1", "k2", "k3"], allow_split=False) as over_waves:
                    _ = list(over_waves)
    finally:
        pm.clear()

@CUDA
def test_cuda_waves_beyond_vram_streaming_exact_values() -> None:
    device = torch.device("cuda")
    schema = _kv_schema()
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    keys = [f"k{i}" for i in range(7)]
    payloads = {k: _make_chunk_data(70 + i) for i, k in enumerate(keys)}
    try:
        with (
            Uint8DualArena(3 * schema.drawer_bytes, device=device, alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=3, pin_manager=pm, spill_watermark=None) as pool,
        ):
            _produce_batched(pool, keys, payloads)
            seen: list[int] = []
            hit_waves = 0
            miss_waves = 0
            with pool.waves(keys, max_wave_slots=2, allow_split=True) as waves:
                for batch in waves:
                    active_sids = {int(s) for s in batch.slot_ids.tolist() if int(s) >= 0}
                    if batch.is_hit_wave:
                        hit_waves += 1
                        assert len(active_sids) == 3
                    else:
                        miss_waves += 1
                        assert 1 <= len(active_sids) <= 2
                    seen.extend(_verify_wave_batch(batch, payloads, expected_keys=keys))
            assert (hit_waves, miss_waves) == (1, 2)
            assert sorted(seen) == list(range(7))
    finally:
        pm.clear()

# ---------------------------------------------------------------------------
# Write-Once Spill, Watermark, Evict & Prefetch
# ---------------------------------------------------------------------------


def test_write_once_spill_evict_prefetch_and_watermark() -> None:
    schema = _kv_schema()
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {f"c{i}": _make_chunk_data(90 + i) for i in range(5)}
    try:
        with (
            Uint8DualArena(4 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=4, pin_manager=pm, spill_watermark=0.75) as pool,
        ):
            _produce_chunks(pool, ["c0", "c1", "c2", "c3"], payloads)
            st = pool.stats()
            assert st.spills >= 2
            assert st.clean_chunks >= 2

            # evict_to_host must evict BOTH dirty and clean chunks off CPU arena cleanly.
            evicted = pool.evict_to_host(["c0", "c1", "c2", "c3"])
            assert evicted == 4
            assert pool.stats().resident_gpu_chunks == 0
            assert all(pool.location(f"c{i}") is KvChunkLocation.HOST_ONLY for i in range(4))

            spills_before_reload = pool.stats().spills
            assert pool.prefetch(["c0", "c1"]) == 2
            assert pool.location("c0") is KvChunkLocation.GPU_CLEAN
            assert pool.evict_to_host(["c0", "c1"]) == 2
            assert pool.stats().spills == spills_before_reload  # write-once: no second spill!
    finally:
        pm.clear()

@CUDA
def test_cuda_write_once_async_spill_evict_and_prefetch() -> None:
    device = torch.device("cuda")
    schema = _kv_schema()
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {f"c{i}": _make_chunk_data(110 + i) for i in range(4)}
    try:
        with (
            Uint8DualArena(4 * schema.drawer_bytes, device=device, alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=4, pin_manager=pm, spill_watermark=None) as pool,
        ):
            _produce_chunks(pool, ["c0", "c1"], payloads)
            assert pool.spill(["c0", "c1"]) == 2
            assert pool.evict_to_host(["c0"]) == 1
            assert pool.location("c0") is KvChunkLocation.HOST_ONLY

            spills_before = pool.stats().spills
            assert pool.prefetch(["c0"]) == 1
            assert pool.location("c0") is KvChunkLocation.GPU_CLEAN
            assert pool.evict_to_host(["c0"]) == 1
            assert pool.stats().spills == spills_before
            with pool.waves(["c0", "c1"]) as waves:
                _verify_wave_batch(next(waves), payloads)
    finally:
        pm.clear()

# ---------------------------------------------------------------------------
# discard(), Atomic Refusal & Async D2H / H2D Retire
# ---------------------------------------------------------------------------


def test_discard_across_locations_and_atomic_busy_refusal() -> None:
    schema = _kv_schema()
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {f"c{i}": _make_chunk_data(130 + i) for i in range(4)}
    try:
        with (
            Uint8DualArena(4 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=4, pin_manager=pm, spill_watermark=None) as pool,
        ):
            _produce_chunks(pool, ["c0", "c1", "c2", "c3"], payloads)
            pool.spill(["c1", "c2"])
            pool.evict_to_host(["c2"])

            with pool.waves(["c0"]) as waves:
                _ = next(waves)
                with pytest.raises(RuntimeError, match="readers or transfers"):
                    pool.discard(["c1", "c0"])
            assert pool.contains("c1") and pool.contains("c0")

            dropped = pool.discard(["c0", "c1", "c2", "unknown"])
            assert dropped == 3
            assert not pool.contains("c0") and not pool.contains("c2")
            assert pool.contains("c3")
            assert pool.stats().discards == 3
    finally:
        pm.clear()

@CUDA
def test_cuda_discard_async_d2h_and_h2d_retire_and_close_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    schema = _kv_schema()
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {"d0": _make_chunk_data(150), "h0": _make_chunk_data(151)}
    fake_poll = False
    fake_h2d_pending = False
    orig_poll_d2h = StreamLease.poll_d2h
    orig_ev_query = torch.cuda.Event.query

    def _paused_poll_d2h(self_lease: StreamLease) -> bool:
        return False if fake_poll else orig_poll_d2h(self_lease)

    def _paused_ev_query(self_ev: torch.cuda.Event) -> bool:
        return False if fake_h2d_pending else orig_ev_query(self_ev)

    try:
        with Uint8DualArena(4 * schema.drawer_bytes, device=device, alignment=256) as arena:
            pool = KvChunkPool(arena, schema, max_slots=4, pin_manager=pm, spill_watermark=None)
            _produce_chunks(pool, ["d0", "h0"], payloads)
            pool.spill(["h0"])
            pool.evict_to_host(["h0"])

            # Pause D2H poll so discard(["d0"]) queues an async _PendingHostRetire + RETIRING slot.
            fake_poll = True
            monkeypatch.setattr(StreamLease, "poll_d2h", _paused_poll_d2h)
            pool.spill(["d0"])
            assert pool.discard(["d0"]) == 1
            assert len(pool._pending_host_retires) == 1
            fake_poll = False

            # Pause H2D Event.query so discard(["h0"]) deterministically queues an async H2D retire.
            pool.prefetch(["h0"])
            fake_h2d_pending = True
            monkeypatch.setattr(torch.cuda.Event, "query", _paused_ev_query)
            assert pool.discard(["h0"]) == 1
            assert any(
                item.slot is None and item.host_alloc is not None and bool(item.h2d_events)
                for item in pool._pending_host_retires
            )
            fake_h2d_pending = False
            pool.close()
            assert not pool.is_open
        assert pm.stats.active_leases == 0
    finally:
        pm.clear()

# ---------------------------------------------------------------------------
# TP-Safe No-Silent-Host-Drop Under Host RAM Exhaustion
# ---------------------------------------------------------------------------


def test_tp_safe_no_silent_host_drop_under_pin_and_pool_limits() -> None:
    page = mmap.PAGESIZE
    schema = SlotSchema({"k": page, "v": 256}, {"scale": (torch.float32, (8,))})
    pm = PinManager(2 * page, max_host_bytes=4 * page, backend=_SimpleSlabBackend())
    payloads = {
        f"c{i}": _make_chunk_data(170 + i, k_bytes=page, v_bytes=256) for i in range(6)
    }
    try:
        # Pool-level max_host_bytes cap stops spill() cleanly without silent drops.
        with (
            Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(
                arena,
                schema,
                max_slots=2,
                pin_manager=pm,
                spill_watermark=None,
                max_host_bytes=pool_cap_bytes(schema, 2),
            ) as pool,
        ):
            _produce_chunks(pool, ["c0", "c1"], payloads)
            _produce_chunks(pool, ["c2"], payloads)  # spills c0 (1)
            _produce_chunks(pool, ["c3"], payloads)  # spills c1 (2 -> host cap reached)
            with pytest.raises(ArenaCapacityError):
                _produce_chunks(pool, ["c4"], payloads)
            assert all(pool.contains(f"c{i}") for i in range(4))
            assert pool.evict_to_host(["c2", "c3"]) == 0
            assert pool.discard(["c0"]) == 1
            assert pool.spill(["c2"]) == 1
    finally:
        pm.clear()

def pool_cap_bytes(schema: SlotSchema, count: int) -> int:
    page = mmap.PAGESIZE
    raw = _build_host_chunk_layout(schema).total_bytes
    return count * (((raw + page - 1) // page) * page)


def test_tp_safe_exhaustion_via_pin_manager_ceiling() -> None:
    page = mmap.PAGESIZE
    schema = SlotSchema({"k": 512, "v": 512}, {"scale": (torch.float32, (8,))})
    pm = PinManager(2 * page, max_host_bytes=2 * page, backend=_SimpleSlabBackend())
    payloads = {f"p{i}": _make_chunk_data(190 + i, k_bytes=512, v_bytes=512) for i in range(5)}
    try:
        with (
            Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None) as pool,
        ):
            _produce_chunks(pool, ["p0", "p1"], payloads)
            assert pool.spill() == 2
            assert pool.evict_to_host() == 2
            _produce_chunks(pool, ["p2", "p3"], payloads)
            assert pool.spill(["p2"]) == 0  # PinManager host ceiling reached cleanly
            with pytest.raises(ArenaCapacityError):
                _produce_chunks(pool, ["p4"], payloads)
            assert all(pool.contains(f"p{i}") for i in range(4))
            assert pool.location("p2") is KvChunkLocation.GPU_DIRTY
            assert pool.discard(["p0"]) == 1
            assert pool.spill(["p2"]) == 1
    finally:
        pm.clear()

# ---------------------------------------------------------------------------
# CUDA Release / record_stream Recovery & Retryable close()
# ---------------------------------------------------------------------------


@CUDA
def test_cuda_release_and_record_stream_failure_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    schema = _kv_schema()
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {f"c{i}": _make_chunk_data(210 + i) for i in range(2)}
    alt_stream = torch.cuda.Stream(device=device)
    orig_release = StreamLease.release
    orig_record_stream = StreamLease.record_stream
    fail_release = False
    fail_record = False

    def _flaky_release(self_lease: StreamLease, stream: torch.cuda.Stream | None = None) -> None:
        if fail_release:
            raise RuntimeError("transient kv release failure")
        orig_release(self_lease, stream)

    def _flaky_record(self_lease: StreamLease, stream: torch.cuda.Stream) -> None:
        if fail_record and stream == alt_stream:
            raise RuntimeError("kv record_stream failure")
        orig_record_stream(self_lease, stream)

    try:
        with (
            Uint8DualArena(2 * schema.drawer_bytes, device=device, alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=1, pin_manager=pm, spill_watermark=None) as pool,
        ):
            _produce_chunks(pool, ["c0"], payloads)
            monkeypatch.setattr(StreamLease, "release", _flaky_release)
            fail_release = True
            with pytest.raises(RuntimeError, match="transient kv release failure"):
                with pool.waves(["c0"]) as waves:
                    _ = next(waves)
            assert pool._slots[0].active_readers > 0
            fail_release = False
            _ = pool.stats()
            assert pool._slots[0].active_readers == 0

            monkeypatch.setattr(StreamLease, "record_stream", _flaky_record)
            orig_stream = torch.cuda.current_stream(device)
            fail_record = True
            try:
                with pytest.raises(RuntimeError, match="kv record_stream failure"):
                    with pool.waves(["c0"]) as waves:
                        _ = next(waves)
                        torch.cuda.set_stream(alt_stream)
            finally:
                torch.cuda.set_stream(orig_stream)
                fail_record = False
            assert pool._slots[0].active_readers > 0
            _ = pool.stats()
            assert pool._slots[0].active_readers == 0
    finally:
        pm.clear()

@CUDA
def test_cuda_retryable_close_preserves_host_allocs_until_sync_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    schema = _kv_schema()
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {f"c{i}": _make_chunk_data(230 + i) for i in range(2)}
    orig_sync_close = _WaveReleaseTracker.synchronize_slots_for_close
    sync_calls = 0

    def _fail_first_close_sync(self_tracker: _WaveReleaseTracker, slots, by_slot) -> None:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            raise RuntimeError("close sync failed")
        orig_sync_close(self_tracker, slots, by_slot)

    try:
        with Uint8DualArena(2 * schema.drawer_bytes, device=device, alignment=256) as arena:
            pool = KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None)
            _produce_chunks(pool, ["c0", "c1"], payloads)
            pool.spill(["c0", "c1"])
            host_allocs = [rec.host_alloc for rec in pool._chunks.values() if rec.host_alloc is not None]
            assert len(host_allocs) == 2

            with monkeypatch.context() as patch:
                patch.setattr(_WaveReleaseTracker, "synchronize_slots_for_close", _fail_first_close_sync)
                with pytest.raises(RuntimeError, match="close sync failed"):
                    pool.close()
            assert all(not alloc.released for alloc in host_allocs)
            assert pool._closing and not pool._closed

            pool.close()
            assert all(alloc.released for alloc in host_allocs)
            assert pool._closed
            assert arena.active_high_slots == ()
    finally:
        pm.clear()

# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_multithreaded_waves_prefetch_and_concurrent_discard() -> None:
    schema = _kv_schema()
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    shared = [f"p{i}" for i in range(4)]
    ephemeral = [f"e{i}" for i in range(8)]
    payloads = {k: _make_chunk_data(250 + idx) for idx, k in enumerate([*shared, *ephemeral])}
    try:
        with (
            Uint8DualArena(3 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=3, pin_manager=pm, spill_watermark=0.8) as pool,
        ):
            _produce_batched(pool, shared, payloads)
            _produce_batched(pool, ephemeral, payloads)
            barrier = threading.Barrier(3)

            def reader_worker(offset: int) -> None:
                try:
                    barrier.wait(timeout=5.0)
                    for step in range(10):
                        req = [
                            shared[(step + offset) % 2],
                            shared[2 + ((step + offset) % 2)],
                            None,
                        ]
                        with pool.waves(req, allow_split=True) as waves:
                            seen: list[int] = []
                            for batch in waves:
                                seen.extend(_verify_wave_batch(batch, payloads, expected_keys=req))
                            assert sorted(seen) == [0, 1]
                        time.sleep(0)
                except BaseException:
                    barrier.abort()
                    raise

            def manager_worker() -> None:
                try:
                    barrier.wait(timeout=5.0)
                    for key in ephemeral:
                        pool.prefetch([shared[0]])
                        pool.discard([key])
                        time.sleep(0)
                except BaseException:
                    barrier.abort()
                    raise

            with ThreadPoolExecutor(max_workers=3) as ex:
                futs = [
                    ex.submit(reader_worker, 0),
                    ex.submit(reader_worker, 1),
                    ex.submit(manager_worker),
                ]
                for fut in futs:
                    fut.result(timeout=15.0)
            assert all(not pool.contains(k) for k in ephemeral)
            assert all(pool.contains(k) for k in shared)
    finally:
        pm.clear()

# ---------------------------------------------------------------------------
# Dual-End ExpertSlabPool + KvChunkPool Integration
# ---------------------------------------------------------------------------


def _run_dual_end_coordination_scenario(device: torch.device) -> None:
    expert_schema = _expert_schema()
    kv_schema = _kv_schema()
    budget = VramBudget(32 * 1024 * 1024)
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    experts = [_make_expert_data(300 + i) for i in range(4)]
    kv_payloads = {f"kv{i}": _make_chunk_data(400 + i) for i in range(4)}
    ctx_mgr: Any = DeviceTransferContext(device) if device.type == "cuda" else contextlib.nullcontext()
    try:
        with (
            ctx_mgr as ctx,
            Uint8DualArena(4 * 512, device=device, budget=budget, alignment=256) as arena,
            ExpertSlabPool(
                arena,
                expert_schema,
                budget=budget,
                transfer_context=ctx,
                pin_manager=pm,
            ) as exp_pool,
            KvChunkPool(
                arena,
                kv_schema,
                budget=budget,
                transfer_context=ctx,
                pin_manager=pm,
                spill_watermark=None,
            ) as kv_pool,
        ):
            exp_pool.register_group("moe", experts)
            # 1) Produce GPU_DIRTY KV chunks in slots 3 and 2.
            _produce_chunks(kv_pool, ["kv0", "kv1"], kv_payloads)
            assert sorted(arena.active_high_slots) == [2, 3]

            # 2) Consume low slots 0, 1, 2 -> forces KV slot 2 to spill & free its edge slot!
            with exp_pool.waves("moe", [0, 1, 2]) as exp_waves:
                exp_list = list(exp_waves)
                assert exp_list
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                assert kv_pool.contains("kv0") and kv_pool.contains("kv1")
                assert len(arena.active_high_slots) <= 1
                assert kv_pool.stats().spills >= 1

            # 3) Stream all KV chunks back via allow_split=True while peer experts sit on low slots.
            with kv_pool.waves(["kv0", "kv1"], max_wave_slots=1, allow_split=True) as kv_waves:
                seen: list[int] = []
                for batch in kv_waves:
                    seen.extend(_verify_wave_batch(batch, kv_payloads, expected_keys=["kv0", "kv1"]))
                assert sorted(seen) == [0, 1]

            # 4) Re-produce kv2 into slot 3, then exp_pool reclaims empty/clean bottom slots on pressure.
            kv_pool.evict_to_host()
            _produce_chunks(kv_pool, ["kv2"], kv_payloads)
            assert kv_pool.stats().resident_gpu_chunks == 1
            with exp_pool.waves("moe", [0, 1, 2]) as exp_waves2:
                _ = list(exp_waves2)
            assert len(arena.active_high_slots) == 1
            assert kv_pool.location("kv2") is KvChunkLocation.GPU_DIRTY
        assert budget.reserved_bytes == 0
        if device.type == "cuda":
            assert pm.stats.active_leases == 0
    finally:
        pm.clear()

def test_dual_end_expert_and_kv_pool_coordination_cpu() -> None:
    _run_dual_end_coordination_scenario(torch.device("cpu"))


@CUDA
def test_cuda_dual_end_expert_and_kv_pool_coordination() -> None:
    _run_dual_end_coordination_scenario(torch.device("cuda"))


def test_dual_end_dirty_kv_edge_relocates_to_higher_free_slot() -> None:
    expert_schema = _expert_schema()
    kv_schema = _kv_schema()
    pm = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    experts = [_make_expert_data(500 + i) for i in range(4)]
    kv_payloads = {f"kv{i}": _make_chunk_data(600 + i) for i in range(3)}
    try:
        with (
            Uint8DualArena(4 * 512, device="cpu", alignment=256) as arena,
            ExpertSlabPool(arena, expert_schema, pin_manager=pm) as exp_pool,
            KvChunkPool(arena, kv_schema, pin_manager=pm, spill_watermark=None) as kv_pool,
        ):
            exp_pool.register_group("moe", experts)
            # Occupy high slots 3, 2, then discard slot 3 -> leaves EMPTY slot 3 and GPU_DIRTY "kv1" at edge slot 2!
            _produce_chunks(kv_pool, ["kv0", "kv1"], kv_payloads)
            assert sorted(arena.active_high_slots) == [2, 3]
            kv_pool.discard(["kv0"])
            assert kv_pool.stats().resident_gpu_chunks == 1
            assert kv_pool.location("kv1") is KvChunkLocation.GPU_DIRTY

            # Expert wants 3 slots [0, 1, 2] -> edge slot 2 relocates up to empty slot 3 (zero spills!).
            with exp_pool.waves("moe", [0, 1, 2]) as waves:
                _ = list(waves)
            assert arena.active_high_slots == (3,)
            assert kv_pool.location("kv1") is KvChunkLocation.GPU_DIRTY
            assert kv_pool.stats().relocations == 1
            assert kv_pool.stats().spills == 0
            with kv_pool.waves(["kv1"]) as kv_waves:
                _verify_wave_batch(next(kv_waves), kv_payloads)
    finally:
        pm.clear()

# ---------------------------------------------------------------------------
# Real piper_kernels E2E CUDA + torch.compile(fullgraph=True)
# ---------------------------------------------------------------------------


def _build_piper_kv_schema(kv_heads: int, head_dim: int, device: torch.device) -> SlotSchema:
    from piper_kernels import kv_chunk_storage_specs

    specs = kv_chunk_storage_specs(1, kv_heads, head_dim, device=device, kv_dtype=torch.uint8)
    return SlotSchema(
        {
            "key_int8": specs["key_int8"][0][1:],
            "value_int8": specs["value_int8"][0][1:],
        },
        {
            name: (specs[name][1], specs[name][0][1:])
            for name in ("key_scale", "key_center", "value_scale", "value_mean")
        },
        field_alignment=256,
        side_alignment=16,
    )


def _stream_offloaded_attention_decode(
    kv_pool: KvChunkPool,
    kernel_pool: Any,
    query: torch.Tensor,
    chunk_keys: Sequence[Any],
    *,
    tail_key: torch.Tensor | None = None,
    tail_value: torch.Tensor | None = None,
    tail_len: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from piper_kernels import merge_attention_lse, piper_attention_decode

    outs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    with kv_pool.waves(chunk_keys, max_wave_slots=2, allow_split=True) as waves:
        for batch in waves:
            n_chunks = torch.tensor([len(chunk_keys)], dtype=torch.int32, device=query.device)
            if batch.index == 0 and tail_key is not None:
                out, lse = piper_attention_decode(
                    query,
                    kernel_pool,
                    batch.slot_ids.unsqueeze(0),
                    n_chunks,
                    tail_key=tail_key,
                    tail_value=tail_value,
                    tail_len=tail_len,
                    return_lse=True,
                    active_chunks_hint=max(1, batch.active_slots_hint),
                )
            else:
                out, lse = piper_attention_decode(
                    query,
                    kernel_pool,
                    batch.slot_ids.unsqueeze(0),
                    n_chunks,
                    return_lse=True,
                    active_chunks_hint=max(1, batch.active_slots_hint),
                )
            outs.append(out)
            lses.append(lse)
    if len(chunk_keys) > kv_pool.max_slots:
        assert len(outs) >= 2
    return merge_attention_lse(outs, lses)


@PIPER_KV_CUDA
def test_cuda_piper_kernels_kv_pool_streaming_attention_e2e() -> None:
    from piper_kernels import PiperKVChunkPool, freeze_kv_chunks_, piper_attention_decode

    device = torch.device("cuda")
    kv_heads, q_heads, head_dim = 2, 4, 64
    num_chunks, max_slots = 5, 3
    schema = _build_piper_kv_schema(kv_heads, head_dim, device)
    pm = PinManager(max_pinned_bytes=128 * 1024 * 1024, max_host_bytes=128 * 1024 * 1024)
    gen = torch.Generator(device="cpu").manual_seed(700)
    raw_k = torch.randn((num_chunks, kv_heads, 1024, head_dim), generator=gen, dtype=torch.float16).to(device)
    raw_v = torch.randn((num_chunks, kv_heads, 1024, head_dim), generator=gen, dtype=torch.float16).to(device)
    query = torch.randn((1, q_heads, 1, head_dim), generator=gen, dtype=torch.float16).to(device)
    tail_k = torch.randn((1, kv_heads, 16, head_dim), generator=gen, dtype=torch.float16).to(device)
    tail_v = torch.randn((1, kv_heads, 16, head_dim), generator=gen, dtype=torch.float16).to(device)
    tail_len = torch.tensor([16], dtype=torch.int32, device=device)

    ref_pool = PiperKVChunkPool.empty(num_chunks, kv_heads, head_dim, device=device, kv_dtype=torch.uint8)
    ref_slots = torch.arange(num_chunks, dtype=torch.int32, device=device)
    freeze_kv_chunks_(raw_k, raw_v, ref_pool, ref_slots)
    ref_out, ref_lse = piper_attention_decode(
        query,
        ref_pool,
        ref_slots.unsqueeze(0),
        torch.tensor([num_chunks], dtype=torch.int32, device=device),
        tail_key=tail_k,
        tail_value=tail_v,
        tail_len=tail_len,
        return_lse=True,
    )

    try:
        with (
            Uint8DualArena(max_slots * schema.drawer_bytes, device=device, alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=max_slots, pin_manager=pm, spill_watermark=None) as kv_pool,
        ):
            kernel_pool = PiperKVChunkPool.from_tensors(**kv_pool.arena_views, **kv_pool.side_views)
            chunk_keys = [f"k{i}" for i in range(num_chunks)]
            with kv_pool.produce(chunk_keys[:3]) as b0:
                freeze_kv_chunks_(raw_k[:3], raw_v[:3], kernel_pool, b0.slot_ids)
            with kv_pool.produce(chunk_keys[3:]) as b1:
                freeze_kv_chunks_(raw_k[3:], raw_v[3:], kernel_pool, b1.slot_ids)
            kv_pool.evict_to_host(["k0", "k1"])

            got_out, got_lse = _stream_offloaded_attention_decode(
                kv_pool,
                kernel_pool,
                query,
                chunk_keys,
                tail_key=tail_k,
                tail_value=tail_v,
                tail_len=tail_len,
            )
            torch.cuda.synchronize(device)
            torch.testing.assert_close(got_out, ref_out, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(got_lse, ref_lse, rtol=1e-2, atol=1e-2)
    finally:
        pm.clear()

@PIPER_KV_CUDA
def test_cuda_piper_kernels_dual_end_torch_compile_fullgraph(
    request: pytest.FixtureRequest,
) -> None:
    from piper_kernels import PiperKVChunkPool, freeze_kv_chunks_, piper_attention_decode

    torch.compiler.reset()
    request.addfinalizer(torch.compiler.reset)
    device = torch.device("cuda")
    kv_heads, q_heads, head_dim = 2, 4, 64
    kv_schema = _build_piper_kv_schema(kv_heads, head_dim, device)
    exp_schema = _expert_schema(up_bytes=256, down_bytes=256, scale_len=8)
    pm = PinManager(max_pinned_bytes=128 * 1024 * 1024, max_host_bytes=128 * 1024 * 1024)
    gen = torch.Generator(device="cpu").manual_seed(800)
    raw_k = torch.randn((2, kv_heads, 1024, head_dim), generator=gen, dtype=torch.float16).to(device)
    raw_v = torch.randn((2, kv_heads, 1024, head_dim), generator=gen, dtype=torch.float16).to(device)
    query = torch.randn((1, q_heads, 1, head_dim), generator=gen, dtype=torch.float16).to(device)
    experts = [_make_expert_data(850 + i, up_bytes=256, down_bytes=256) for i in range(2)]

    try:
        with (
            DeviceTransferContext(device) as ctx,
            Uint8DualArena(4 * kv_schema.drawer_bytes, device=device, alignment=256) as arena,
            ExpertSlabPool(arena, exp_schema, max_slots=2, transfer_context=ctx, pin_manager=pm) as exp_pool,
            KvChunkPool(
                arena,
                kv_schema,
                max_slots=3,
                transfer_context=ctx,
                pin_manager=pm,
                spill_watermark=None,
            ) as kv_pool,
        ):
            exp_pool.register_group("moe", experts)
            kernel_pool = PiperKVChunkPool.from_tensors(**kv_pool.arena_views, **kv_pool.side_views)
            n_chunks = torch.tensor([2], dtype=torch.int32, device=device)

            def compiled_decode_step(slot_ids: torch.Tensor) -> torch.Tensor:
                out = piper_attention_decode(
                    query, kernel_pool, slot_ids.unsqueeze(0), n_chunks, active_chunks_hint=2
                )
                return out[0] if isinstance(out, tuple) else out

            inductor = torch._dynamo.lookup_backend("inductor")
            compile_calls = 0

            def counting_backend(gm: Any, inputs: Any) -> Any:
                nonlocal compile_calls
                compile_calls += 1
                return inductor(gm, inputs)

            compiled_fn = torch.compile(compiled_decode_step, fullgraph=True, backend=counting_backend)

            with kv_pool.produce(["c0", "c1"]) as pb:
                freeze_kv_chunks_(raw_k, raw_v, kernel_pool, pb.slot_ids)
            with kv_pool.waves(["c0", "c1"]) as w1:
                b1 = next(w1)
                out1 = compiled_fn(b1.slot_ids)
                ref_pre = compiled_decode_step(b1.slot_ids).clone()
            assert compile_calls == 1
            torch.testing.assert_close(out1, ref_pre, rtol=1e-2, atol=1e-2)

            # Force spill/reload + low-end expert wave sharing the same Uint8DualArena, then rerun compiled graph!
            kv_pool.evict_to_host(["c0", "c1"])
            with exp_pool.waves("moe", [0, 1]) as ew:
                _ = list(ew)
            with kv_pool.waves(["c0", "c1"], allow_split=False) as w2:
                b2 = next(w2)
                out2 = compiled_fn(b2.slot_ids)
                ref2 = compiled_decode_step(b2.slot_ids)
            assert compile_calls == 1
            torch.testing.assert_close(out2, ref_pre, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(out2, ref2, rtol=1e-2, atol=1e-2)
    finally:
        pm.clear()

# ---------------------------------------------------------------------------
# Second-Pass Holistically Fixed Concurrency & Failure-Path Regressions
# ---------------------------------------------------------------------------


def test_paused_spill_concurrent_evict_and_produce_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = mmap.PAGESIZE
    schema = _kv_schema(k_bytes=256, v_bytes=256)
    pm = PinManager(16 * page, max_host_bytes=16 * page, backend=_SimpleSlabBackend())
    payloads = {"a": _make_chunk_data(301), "b": _make_chunk_data(302)}
    entered, proceed = threading.Barrier(2), threading.Barrier(2)
    orig_alloc = pm.allocate_host_slab
    evict_res: list[int] = []
    t_spill: threading.Thread | None = None

    def gated_slab(*args: object, **kwargs: object) -> Any:
        if t_spill is not None and threading.current_thread() is t_spill:
            entered.wait(timeout=5.0)
            proceed.wait(timeout=5.0)
        return orig_alloc(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pm, "allocate_host_slab", gated_slab)
    try:
        with (
            Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None) as pool,
        ):
            _produce_chunks(pool, ["a"], payloads)
            t_spill = threading.Thread(target=lambda: pool.spill(["a"]))
            t_evict = threading.Thread(target=lambda: evict_res.append(pool.evict_to_host(["a"])))
            t_spill.start()
            entered.wait(timeout=5.0)
            assert pool._chunks["a"].spill_state is _SpillState.ALLOCATING
            t_evict.start()
            _produce_chunks(pool, ["b"], payloads)
            proceed.wait(timeout=5.0)
            t_spill.join(timeout=5.0)
            t_evict.join(timeout=5.0)
            assert not t_spill.is_alive() and not t_evict.is_alive()
            assert evict_res == [1]
            assert pool.location("a") is KvChunkLocation.HOST_ONLY
            assert pool.location("b") is KvChunkLocation.GPU_DIRTY
            with pool.waves(["a", "b"], allow_split=False) as waves:
                _verify_wave_batch(next(waves), payloads)
    finally:
        entered.abort()
        proceed.abort()
        pm.clear()


def test_paused_evict_stale_completion_after_discard_and_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    old_payload = {"a": _make_chunk_data(310)}
    new_payload = {"a": _make_chunk_data(311)}
    entered, proceed = threading.Barrier(2), threading.Barrier(2)
    try:
        with (
            Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None) as pool,
        ):
            _produce_chunks(pool, ["a"], old_payload)
            orig_spill_one = pool._spill_one_chunk
            pause_once = True

            def paused_spill_one(*args: object, **kwargs: object) -> Any:
                nonlocal pause_once
                res = orig_spill_one(*args, **kwargs)  # type: ignore[arg-type]
                if pause_once:
                    pause_once = False
                    entered.wait(timeout=5.0)
                    proceed.wait(timeout=5.0)
                return res

            monkeypatch.setattr(pool, "_spill_one_chunk", paused_spill_one)
            evict_out: list[int] = []
            t_evict = threading.Thread(target=lambda: evict_out.append(pool.evict_to_host(["a"])))
            t_evict.start()
            entered.wait(timeout=5.0)
            assert pool.discard(["a"]) == 1
            _produce_chunks(pool, ["a"], new_payload)
            assert pool.spill(["a"]) == 1
            assert pool.location("a") is KvChunkLocation.GPU_CLEAN
            proceed.wait(timeout=5.0)
            t_evict.join(timeout=5.0)
            assert not t_evict.is_alive()
            assert evict_out == [0]
            assert pool.location("a") is KvChunkLocation.GPU_CLEAN
            with pool.waves(["a"]) as waves:
                _verify_wave_batch(next(waves), new_payload)
    finally:
        entered.abort()
        proceed.abort()
        pm.clear()


def test_lookahead_staging_claim_blocks_discard_and_record_identity_commit() -> None:
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {k: _make_chunk_data(320 + i) for i, k in enumerate(("k0", "k1", "k2"))}
    replacement = {"k0": _make_chunk_data(399)}
    try:
        with (
            Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None) as pool,
        ):
            _produce_batched(pool, ["k0", "k1", "k2"], payloads)
            assert pool.location("k0") is KvChunkLocation.HOST_ONLY
            with pool.waves(["k1", "k0"], max_wave_slots=1, allow_split=True) as waves:
                first = next(waves)
                assert first.is_hit_wave
                with pytest.raises(RuntimeError, match="active"):
                    pool.discard(["k0"])
                second = next(waves)
                _verify_wave_batch(second, payloads)

            # Record-identity commit guard rejects stale staged batch for replaced key.
            pool.evict_to_host(["k0"])
            stale_rec = pool._chunks["k0"]
            assert stale_rec.host_alloc is not None
            staging_slot = pool._obtain_free_or_evicted_slot(
                protected={"k0"}, next_phase=_SlotPhase.STAGING, allow_reclaim_peer=False, allow_spill_dirty=False
            )
            assert staging_slot is not None
            pool.discard(["k0"])
            _produce_chunks(pool, ["k0"], replacement)
            bound = pool._commit_reloaded_batch(
                [(stale_rec, staging_slot, stale_rec.host_alloc)],
                wave_lease=None,
                record_stats=False,
            )
            assert bound == []
            assert staging_slot.phase is _SlotPhase.EMPTY
            with pool.waves(["k0"]) as w_check:
                _verify_wave_batch(next(w_check), replacement)
    finally:
        pm.clear()

@CUDA
def test_cuda_out_of_order_h2d_events_and_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {k: _make_chunk_data(330 + i) for i, k in enumerate(("c0", "c1"))}
    orig_query = torch.cuda.Event.query
    try:
        with Uint8DualArena(2 * schema.drawer_bytes, device=device, alignment=256) as arena:
            pool = KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None)
            _produce_chunks(pool, ["c0", "c1"], payloads)
            pool.spill(["c0"])
            pool.evict_to_host(["c0"])
            hold_pending = True
            monkeypatch.setattr(
                torch.cuda.Event,
                "query",
                lambda self_ev: False if hold_pending else orig_query(self_ev),
            )
            pool.prefetch(["c0"])
            pool.evict_to_host(["c0"])
            pool.prefetch(["c0"])
            rec = pool._chunks["c0"]
            assert len(rec.pending_h2d_events) == 2
            ev1, ev2 = rec.pending_h2d_events[0], rec.pending_h2d_events[1]

            def ordered_query(self_ev: torch.cuda.Event) -> bool:
                if self_ev is ev1:
                    return False
                if self_ev is ev2:
                    return True
                return orig_query(self_ev)

            monkeypatch.setattr(torch.cuda.Event, "query", ordered_query)
            assert pool.discard(["c0"]) == 1
            assert pool._live_host_allocs == 1
            assert len(pool._pending_host_retires) == 1
            assert pool._pending_host_retires[0].h2d_events == (ev1,)
            monkeypatch.setattr(torch.cuda.Event, "query", orig_query)
            pool.close()
            assert arena.active_high_slots == ()
    finally:
        pm.clear()

@CUDA
def test_cuda_h2d_record_event_and_discard_query_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {"a": _make_chunk_data(340)}
    orig_record_event = kv_pool_mod._record_event
    orig_query = torch.cuda.Event.query
    try:
        with Uint8DualArena(2 * schema.drawer_bytes, device=device, alignment=256) as arena:
            pool = KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None)
            _produce_chunks(pool, ["a"], payloads)
            pool.spill(["a"])
            pool.evict_to_host(["a"])

            # 1) H2D _record_event failure syncs stream, retains host_alloc, and allows retry.
            monkeypatch.setattr(
                kv_pool_mod, "_record_event", lambda _s: (_ for _ in ()).throw(RuntimeError("rec fail"))
            )
            with pytest.raises(RuntimeError, match="rec fail"):
                pool.prefetch(["a"])
            monkeypatch.setattr(kv_pool_mod, "_record_event", orig_record_event)
            assert pool._chunks["a"].host_readers == 0
            assert pool._live_host_allocs == 1
            assert all(s.phase is _SlotPhase.EMPTY for s in pool._slots.values() if s.key is None)
            assert pool.prefetch(["a"]) == 1
            with pool.waves(["a"]) as w_a:
                _verify_wave_batch(next(w_a), payloads)

            # 2) Discard Event.query failure does not leak or double-publish slot into EMPTY and retiring.
            pool.evict_to_host(["a"])
            pool.prefetch(["a"])
            monkeypatch.setattr(
                torch.cuda.Event, "query", lambda _e: (_ for _ in ()).throw(RuntimeError("q fail"))
            )
            with pytest.raises(RuntimeError, match="q fail"):
                pool.discard(["a"])
            monkeypatch.setattr(torch.cuda.Event, "query", orig_query)
            assert pool._live_host_allocs == 1
            assert len(pool._pending_host_retires) == 1
            retire_slot = pool._pending_host_retires[0].slot
            assert retire_slot is not None and retire_slot.phase is _SlotPhase.RETIRING
            pool.close()
            assert arena.active_high_slots == ()
    finally:
        pm.clear()

@CUDA
def test_cuda_partial_d2h_wait_failure_defers_host_slab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {"b": _make_chunk_data(341)}
    orig_wait_d2h = StreamLease.wait_d2h
    try:
        with Uint8DualArena(2 * schema.drawer_bytes, device=device, alignment=256) as arena:
            pool = KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None)
            _produce_chunks(pool, ["b"], payloads)
            scale_pool = pool._side_pools["scale"]
            orig_scale_slot = scale_pool.slot
            monkeypatch.setattr(
                scale_pool, "slot", lambda _sid: (_ for _ in ()).throw(RuntimeError("side d2h"))
            )
            monkeypatch.setattr(
                StreamLease, "wait_d2h", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("wait"))
            )
            with pytest.raises(RuntimeError, match="wait"):
                pool.spill(["b"])
            monkeypatch.setattr(scale_pool, "slot", orig_scale_slot)
            monkeypatch.setattr(StreamLease, "wait_d2h", orig_wait_d2h)
            assert pool._chunks["b"].spill_state is _SpillState.NONE
            assert pool._live_host_allocs == 1
            assert len(pool._pending_host_retires) == 1
            pending_ha = pool._pending_host_retires[0].host_alloc
            assert pending_ha is not None and not pending_ha.released
            assert pool.spill(["b"]) == 1
            pool.close()
            assert arena.active_high_slots == ()
    finally:
        pm.clear()

@CUDA
def test_cuda_dirty_relocation_failure_retires_reused_dst_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {"k0": _make_chunk_data(360), "k1": _make_chunk_data(361)}
    orig_sync_reset = StreamLease.synchronize_and_reset
    try:
        with Uint8DualArena(3 * schema.drawer_bytes, device=device, alignment=256) as arena:
            pool = KvChunkPool(arena, schema, max_slots=3, pin_manager=pm, spill_watermark=None)
            _produce_chunks(pool, ["k0", "k1"], payloads)
            sid0 = pool._chunks["k0"].slot.slot_id  # type: ignore[union-attr]
            sid1 = pool._chunks["k1"].slot.slot_id  # type: ignore[union-attr]
            edge_slot = pool._slots[min(sid0, sid1)]
            dst_slot = pool._slots[max(sid0, sid1)]
            edge_rec = pool._chunks[edge_slot.key]
            assert edge_rec.key is not None and dst_slot.key is not None
            pool.discard([dst_slot.key])
            assert dst_slot.phase is _SlotPhase.EMPTY

            monkeypatch.setattr(
                KvChunkPool, "_copy_slot_d2d", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("d2d"))
            )
            monkeypatch.setattr(
                StreamLease,
                "synchronize_and_reset",
                lambda self_l: (
                    (_ for _ in ()).throw(RuntimeError("sync fail"))
                    if self_l is dst_slot.lease
                    else orig_sync_reset(self_l)
                ),
            )
            edge_slot.phase = _SlotPhase.RELOCATING
            with pytest.raises(RuntimeError, match="d2d"):
                pool._relocate_edge_slot_to_higher(edge_slot, edge_rec)
            edge_slot.phase = _SlotPhase.READY
            assert dst_slot.phase is _SlotPhase.RETIRING
            assert any(item.slot is dst_slot for item in pool._pending_host_retires)

            monkeypatch.setattr(StreamLease, "synchronize_and_reset", orig_sync_reset)
            pool.close()
            assert arena.active_high_slots == ()
    finally:
        pm.clear()

@CUDA
def test_cuda_max_host_bytes_accounts_pending_retires(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {"c0": _make_chunk_data(370), "c1": _make_chunk_data(371)}
    fake_poll = False
    orig_poll = StreamLease.poll_d2h
    monkeypatch.setattr(StreamLease, "poll_d2h", lambda s: False if fake_poll else orig_poll(s))
    try:
        with (
            Uint8DualArena(2 * schema.drawer_bytes, device=device, alignment=256) as arena,
            KvChunkPool(
                arena,
                schema,
                max_slots=2,
                pin_manager=pm,
                spill_watermark=None,
                max_host_bytes=pool_cap_bytes(schema, 1),
            ) as pool,
        ):
            _produce_chunks(pool, ["c0", "c1"], payloads)
            fake_poll = True
            assert pool.spill(["c0"]) == 1
            assert pool.discard(["c0"]) == 1
            assert pool.spill(["c1"]) == 0
            fake_poll = False
            assert pool.spill(["c1"]) == 1
    finally:
        pm.clear()

def test_allow_split_false_mixed_stats_relocating_and_max_chunks_skips() -> None:
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payloads = {k: _make_chunk_data(380 + i) for i, k in enumerate(("a", "b", "c"))}
    try:
        with (
            Uint8DualArena(3 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=3, pin_manager=pm, spill_watermark=None) as pool,
        ):
            _produce_chunks(pool, ["a", "b", "c"], payloads)
            pool.evict_to_host(["b"])
            with pool.waves(["c", "b"], allow_split=False, record_stats=False) as w1:
                batches = list(w1)
                assert len(batches) == 1
                _verify_wave_batch(batches[0], payloads)
            st = pool.stats()
            assert st.waves_executed == 0 and st.hits == 0 and st.misses == 0 and st.reloads == 0

            # Early break releases wave lease and yields all-hit resident keys together despite cap=1.
            with pool.waves(["a", "b", "c"], max_wave_slots=1, allow_split=True) as w_break:
                hit_batch = next(w_break)
                assert hit_batch.is_hit_wave
                assert sum(int(s) >= 0 for s in hit_batch.slot_ids.tolist()) == 3
                break_test_slot = pool._chunks["a"].slot
                assert break_test_slot is not None
            assert break_test_slot.active_readers == 0
            assert pool.stats().waves_executed == 1

            pool.evict_to_host(["b"])
            slot_a = pool._chunks["a"].slot
            assert slot_a is not None
            slot_a.phase = _SlotPhase.RELOCATING
            assert pool.spill(["a", "c"], max_chunks=1) == 1
            assert pool.location("c") is KvChunkLocation.GPU_CLEAN
            assert pool.spill(["a"], max_chunks=1) == 0

            def finish_reloc() -> None:
                time.sleep(0.05)
                with pool._lock:
                    slot_a.phase = _SlotPhase.READY
                    pool._op_cond.notify_all()

            t = threading.Thread(target=finish_reloc)
            t.start()
            with pool.waves(["a", "b"], allow_split=False) as w2:
                _verify_wave_batch(next(w2), payloads)
            t.join(timeout=5.0)
            assert pool.stats().waves_executed == 2

            # spill/evict with max_chunks=1 skip leading GPU_CLEAN / HOST_ONLY keys.
            assert pool.spill(["c", "a"], max_chunks=1) == 1
            assert pool.location("a") is KvChunkLocation.GPU_CLEAN
            pool.evict_to_host(["a"])
            assert pool.location("a") is KvChunkLocation.HOST_ONLY
            assert pool.evict_to_host(["a", "c"], max_chunks=1) == 1
            assert pool.location("c") is KvChunkLocation.HOST_ONLY
    finally:
        pm.clear()

@CUDA
def test_cuda_tier2b_host_fallback() -> None:
    device = torch.device("cuda")
    kv_schema = _kv_schema()
    pm_2b = PinManager(max_pinned_bytes=0, max_host_bytes=4 * 1024 * 1024)
    payloads = {"t0": _make_chunk_data(390)}
    try:
        with (
            Uint8DualArena(kv_schema.drawer_bytes, device=device, alignment=256) as a2b,
            KvChunkPool(a2b, kv_schema, max_slots=1, pin_manager=pm_2b, spill_watermark=None) as p2b,
        ):
            _produce_chunks(p2b, ["t0"], payloads)
            assert p2b.spill(["t0"]) == 1
            ha = p2b._chunks["t0"].host_alloc
            assert ha is not None and not ha.is_pinned
            assert p2b.evict_to_host(["t0"]) == 1
            with p2b.waves(["t0"]) as w:
                _verify_wave_batch(next(w), payloads)
    finally:
        pm_2b.clear()


@CUDA
def test_cuda_dual_end_multithreaded_frontier_stress() -> None:
    device = torch.device("cuda")
    kv_schema = _kv_schema()
    exp_schema = _expert_schema()
    pm_stress = PinManager(max_pinned_bytes=16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    experts = [_make_expert_data(400 + i) for i in range(4)]
    kv_keys = [f"s{i}" for i in range(8)]
    kv_data = {k: _make_chunk_data(410 + i) for i, k in enumerate(kv_keys)}
    errors: list[BaseException] = []
    try:
        # 5 slots shared across 2 low slots + 4 high slots (total demand 6 > 5) forces frontier reclaim.
        with (
            DeviceTransferContext(device) as ctx,
            Uint8DualArena(5 * kv_schema.drawer_bytes, device=device, alignment=256) as arena,
            ExpertSlabPool(arena, exp_schema, max_slots=2, transfer_context=ctx, pin_manager=pm_stress) as ep,
            KvChunkPool(arena, kv_schema, max_slots=4, transfer_context=ctx, pin_manager=pm_stress) as kp,
        ):
            ep.register_group("g", experts)
            _produce_batched(kp, kv_keys[:4], kv_data)

            def run_exp() -> None:
                try:
                    for step in range(8):
                        with ep.waves("g", [step % 4, (step + 1) % 4]) as ew:
                            for _ in ew:
                                pass
                except BaseException as exc:
                    errors.append(exc)

            def run_kv() -> None:
                try:
                    _produce_batched(kp, kv_keys[4:], kv_data, batch_size=2)
                    with kp.waves(kv_keys, max_wave_slots=2, record_stats=False) as kw:
                        for b in kw:
                            _verify_wave_batch(b, kv_data)
                    kp.discard(kv_keys[:2])
                except BaseException as exc:
                    errors.append(exc)

            te, tk = threading.Thread(target=run_exp), threading.Thread(target=run_kv)
            te.start()
            tk.start()
            te.join(timeout=15.0)
            tk.join(timeout=15.0)
            assert not te.is_alive() and not tk.is_alive()
            assert not errors
            st = kp.stats()
            assert st.relocations + st.evictions > 0
            assert all(kp.contains(k) for k in kv_keys[2:])
    finally:
        pm_stress.clear()


def test_build_batch_failure_does_not_increment_waves_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    try:
        with (
            Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None) as pool,
        ):
            _produce_chunks(pool, ["c0"], {"c0": _make_chunk_data(490)})
            monkeypatch.setattr(
                pool, "_build_batch", lambda **_kw: (_ for _ in ()).throw(RuntimeError("build fail"))
            )
            with pytest.raises(RuntimeError, match="build fail"):
                with pool.waves(["c0"]) as w:
                    _ = next(w)
            assert pool.stats().waves_executed == 0
    finally:
        pm.clear()

def _assert_close_blocked_during_host_release(
    pool: KvChunkPool,
    monkeypatch: pytest.MonkeyPatch,
    action: Any,
) -> None:
    entered, proceed = threading.Barrier(2), threading.Barrier(2)
    orig_release = pool._release_or_defer_host_allocs

    def paused_release(allocs: Any) -> None:
        entered.wait(timeout=5.0)
        proceed.wait(timeout=5.0)
        orig_release(allocs)

    monkeypatch.setattr(pool, "_release_or_defer_host_allocs", paused_release)
    worker = threading.Thread(target=action)
    try:
        worker.start()
        entered.wait(timeout=5.0)
        with pytest.raises(RuntimeError, match="Cannot close"):
            pool.close()
        proceed.wait(timeout=5.0)
        worker.join(timeout=5.0)
        assert not worker.is_alive()
    finally:
        entered.abort()
        proceed.abort()
        if worker.is_alive():
            worker.join(timeout=2.0)
        monkeypatch.setattr(pool, "_release_or_defer_host_allocs", orig_release)


def test_close_blocked_while_discard_and_reap_release_host_allocs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    try:
        with Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena:
            pool = KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None)
            _produce_chunks(pool, ["a", "b"], {"a": _make_chunk_data(501), "b": _make_chunk_data(502)})
            assert pool.spill(["a", "b"]) == 2
            _assert_close_blocked_during_host_release(pool, monkeypatch, lambda: pool.discard(["a"]))
            orig_rel = pool._release_tracked_host_alloc
            monkeypatch.setattr(
                pool, "_release_tracked_host_alloc", lambda _ha: (_ for _ in ()).throw(RuntimeError("rel"))
            )
            with pytest.raises(RuntimeError, match="rel"):
                pool.discard(["b"])
            monkeypatch.setattr(pool, "_release_tracked_host_alloc", orig_rel)
            _assert_close_blocked_during_host_release(
                pool, monkeypatch, pool._reap_pending_host_retires
            )
            assert pool.stats().host_bytes == 0
            pool.close()
    finally:
        pm.clear()

@CUDA
def test_cuda_reclaim_ready_edge_post_commit_free_failure_retires_dirty_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    try:
        with Uint8DualArena(2 * schema.drawer_bytes, device=device, alignment=256) as arena:
            pool = KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None)
            _produce_chunks(pool, ["a", "b"], {"a": _make_chunk_data(540), "b": _make_chunk_data(541)})
            pool.discard(["a"])
            bottom_sid = min(pool._slots)
            orig_free = pool._free_slot_to_arena

            def fail_src_free(s: Any, *, sync_lease: bool = True) -> None:
                if s.slot_id == bottom_sid:
                    raise RuntimeError("src free fail")
                orig_free(s, sync_lease=sync_lease)

            monkeypatch.setattr(pool, "_free_slot_to_arena", fail_src_free)
            with pytest.raises(RuntimeError, match="src free fail"):
                pool._reclaim_high_frontier(pool.layout.slot_offset(bottom_sid) + 1)
            assert pool._slots[bottom_sid].phase is _SlotPhase.RETIRING
            assert len(pool._pending_host_retires) == 1
            assert pool.location("b") is KvChunkLocation.GPU_DIRTY
            monkeypatch.setattr(pool, "_free_slot_to_arena", orig_free)
            with pool.produce(["c"]):
                pass
            pool.close()
    finally:
        pm.clear()

def test_release_tracked_host_alloc_decrements_live_count_when_release_fails_after_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    try:
        with Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena:
            pool = KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None)
            _produce_chunks(pool, ["a"], {"a": _make_chunk_data(550)})
            assert pool.spill(["a"]) == 1
            ha = pool._chunks["a"].host_alloc
            assert ha is not None
            orig_release = ArenaAllocation.release

            def fail_after_mark(self_alloc: ArenaAllocation) -> None:
                orig_release(self_alloc)
                if self_alloc is ha:
                    raise RuntimeError("span return err")

            monkeypatch.setattr(ArenaAllocation, "release", fail_after_mark)
            with pytest.raises(RuntimeError, match="span return err"):
                pool.discard(["a"])
            assert ha.released
            assert pool.stats().host_bytes == 0
            assert not pool._pending_host_retires
            pool.close()
    finally:
        pm.clear()

def test_commit_reloaded_batch_tolerates_already_committed_slot_and_aliased_pending() -> None:
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    try:
        with Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena:
            pool = KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None)
            _produce_chunks(pool, ["a"], {"a": _make_chunk_data(560)})
            assert pool.evict_to_host(["a"]) == 1
            rec = pool._chunks["a"]
            ha = rec.host_alloc
            assert ha is not None
            with pool._lock:
                slot = pool._find_empty_slot_unlocked(next_phase=_SlotPhase.STAGING)
            assert slot is not None
            pool._copy_host_to_slot(slot.slot_id, ha, non_blocking=False)
            shared = [(rec, slot, ha)]
            state = _WaveLoopState(
                held_lease=pool._wave_releases.new_lease(),
                next_lease=pool._wave_releases.new_lease(),
                pending_commit=shared,
                next_staged=shared,
            )
            pool._cleanup_wave_leases(state, record_stats=True)
            assert rec.slot is slot and slot.phase is _SlotPhase.READY and slot.key == "a"
            assert pool.stats().reloads == 1
            assert pool._commit_reloaded_batch([(rec, slot, ha)], wave_lease=None, record_stats=True) == [
                ("a", slot)
            ]
            assert rec.slot is slot and slot.phase is _SlotPhase.READY
            with pool.waves(["a"]) as w:
                assert len(list(w)) == 1
            pool.close()
    finally:
        pm.clear()

def _abort_wait_discard_once(
    pool: KvChunkPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    orig_wait_ready = pool._wait_discard_ready_unlocked

    def _abort(keys_list: Sequence[Any]) -> None:
        monkeypatch.setattr(pool, "_wait_discard_ready_unlocked", orig_wait_ready)
        raise RuntimeError("abort wait")

    monkeypatch.setattr(pool, "_wait_discard_ready_unlocked", _abort)
    with pytest.raises(RuntimeError, match="abort wait"):
        pool.discard(["a"])


def test_concurrent_discard_waiting_on_host_readers_does_not_corrupt_inflight_retire_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    b_in_rel, b_rel_go, a_waiting = threading.Event(), threading.Event(), threading.Event()
    try:
        with Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena:
            pool = KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None)
            _produce_chunks(pool, ["a", "b"], {"a": _make_chunk_data(600), "b": _make_chunk_data(601)})
            assert pool.evict_to_host(["a", "b"]) == 2
            rec_a, ha_b = pool._chunks["a"], pool._chunks["b"].host_alloc
            assert ha_b is not None
            rec_a.host_readers = 1
            orig_release = ArenaAllocation.release

            def gated_release(self_alloc: ArenaAllocation) -> None:
                if self_alloc is ha_b:
                    b_in_rel.set()
                    assert b_rel_go.wait(timeout=5.0)
                orig_release(self_alloc)

            monkeypatch.setattr(ArenaAllocation, "release", gated_release)
            t_b = threading.Thread(target=lambda: pool.discard(["b"]))
            t_b.start()
            assert b_in_rel.wait(timeout=5.0)
            assert pool._inflight_host_retire_ops == 1
            _abort_wait_discard_once(pool, monkeypatch)
            assert pool._inflight_host_retire_ops == 1

            orig_cond_wait = pool._op_cond.wait
            monkeypatch.setattr(
                pool._op_cond,
                "wait",
                lambda timeout=None: (a_waiting.set(), orig_cond_wait(timeout=timeout))[1],
            )
            t_a = threading.Thread(target=lambda: pool.discard(["a"]))
            t_a.start()
            assert a_waiting.wait(timeout=5.0)
            with pool._lock:
                rec_a.host_readers = 0
                pool._op_cond.notify_all()
            t_a.join(timeout=5.0)
            assert pool._inflight_host_retire_ops == 1
            b_rel_go.set()
            t_b.join(timeout=5.0)
            assert pool._inflight_host_retire_ops == 0 and pool.stats().host_bytes == 0
            assert pool.location("a") is None and pool.location("b") is None
            pool.close()
    finally:
        b_rel_go.set()
        pm.clear()


def test_kv_chunk_pool_prefers_uncontested_arena_slot_over_evicting_clean_resident() -> None:
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    try:
        with (
            Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None) as pool,
        ):
            _produce_chunks(pool, ["a"], {"a": _make_chunk_data(606)})
            assert pool.spill(["a"]) == 1
            assert pool.location("a") is KvChunkLocation.GPU_CLEAN
            assert len(arena.active_high_slots) == 1
            _produce_chunks(pool, ["b"], {"b": _make_chunk_data(607)})
            assert len(arena.active_high_slots) == 2
            assert pool.location("a") is KvChunkLocation.GPU_CLEAN
            assert pool.location("b") is KvChunkLocation.GPU_DIRTY
    finally:
        pm.clear()

def test_kv_chunk_pool_spill_clamps_batch_hint_to_max_host_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    hints_seen: list[int] = []
    orig_alloc_slab = pm.allocate_host_slab

    def spy_alloc(*args: Any, **kwargs: Any) -> Any:
        hints_seen.append(int(kwargs.get("batch_hint", 1)))
        return orig_alloc_slab(*args, **kwargs)

    monkeypatch.setattr(pm, "allocate_host_slab", spy_alloc)
    try:
        with (
            Uint8DualArena(4 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(
                arena,
                schema,
                max_slots=4,
                pin_manager=pm,
                spill_watermark=None,
                max_host_bytes=pool_cap_bytes(schema, 2),
            ) as pool,
        ):
            payloads = {f"c{i}": _make_chunk_data(610 + i) for i in range(4)}
            _produce_chunks(pool, list(payloads), payloads)
            assert pool.spill(list(payloads)) == 2
            assert hints_seen and max(hints_seen) <= 2
            hints_seen.clear()
            assert pool.discard(["c0"]) == 1
            assert pool.spill(["c2", "c3"]) == 1
            assert hints_seen and max(hints_seen) == 1
    finally:
        pm.clear()

@CUDA
def test_cuda_wave_async_reader_survives_wave_exit_and_frontier_reclaim_reuse() -> None:
    device = torch.device("cuda")
    kv_schema = _kv_schema()
    exp_schema = _expert_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    kv_payloads = {"kv_bot": _make_chunk_data(620), "kv_top": _make_chunk_data(621)}
    expert = _make_expert_data(622)
    reader_stream = torch.cuda.Stream(device=device)
    dst_k = torch.empty_like(kv_payloads["kv_bot"]["k"], device=device)
    try:
        with (
            Uint8DualArena(2 * kv_schema.drawer_bytes, device=device, alignment=256) as arena,
            ExpertSlabPool(arena, exp_schema, pin_manager=pm) as exp_pool,
            KvChunkPool(arena, kv_schema, pin_manager=pm, spill_watermark=None) as kv_pool,
        ):
            _produce_chunks(kv_pool, ["kv_top", "kv_bot"], kv_payloads)
            kv_bot_slot = kv_pool._chunks["kv_bot"].slot
            assert kv_bot_slot is not None and kv_bot_slot.slot_id == 0
            assert kv_pool.spill(["kv_bot"]) == 1
            with torch.cuda.stream(reader_stream), kv_pool.waves(["kv_bot"]) as w:
                batch = next(w)
                sid = int(batch.slot_ids[0].item())
                k_view = batch.arena_views["k"][sid]
                torch.cuda._sleep(30_000_000)
                dst_k.copy_(k_view)
            exp_pool.register_group("g", [expert])
            with exp_pool.waves("g", [0]) as ew:
                _ = next(ew)
            assert kv_pool.location("kv_bot") is KvChunkLocation.HOST_ONLY
            reader_stream.synchronize()
            torch.testing.assert_close(dst_k.cpu(), kv_payloads["kv_bot"]["k"])
    finally:
        pm.clear()

def _assert_stale_attempt_cannot_publish_or_clear(
    pool: KvChunkPool,
    pm: PinManager,
    rec_a: Any,
    slot_a: Any,
    stale_attempt: int,
    retry_attempt: int,
) -> None:
    dummy = pm.allocate_host_slab(pool._host_span_bytes, pinned=False)
    assert dummy is not None
    try:
        with pool._lock:
            assert rec_a.spill_state is _SpillState.ALLOCATING
            assert rec_a.spill_attempt_id == retry_attempt
            assert not pool._publish_spill_alloc_unlocked(
                rec_a,
                slot_a,
                dummy,
                spill_state=_SpillState.COMPLETE,
                attempt_id=stale_attempt,
            )
            pool._clear_allocating_spill_unlocked(rec_a, retry_attempt)
    finally:
        dummy.release()


def test_concurrent_spill_failure_cannot_clear_retry_claim_or_publish_unwritten_slab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    payload = _make_chunk_data(630)
    entered, proceed = threading.Barrier(2), threading.Barrier(2)
    try:
        with (
            Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None) as pool,
        ):
            _produce_chunks(pool, ["a"], {"a": payload})
            rec_a = pool._chunks["a"]
            slot_a = rec_a.slot
            assert slot_a is not None
            orig_rel = pool._release_or_defer_host_allocs
            attempt1_err: list[BaseException] = []

            def failing_copy(slot_id: int, host_alloc: ArenaAllocation, *, non_blocking: bool) -> None:
                _ = slot_id, non_blocking
                assert pool.spill(["a"]) == 0
                with pool._lock:
                    assert not pool._publish_spill_alloc_unlocked(
                        rec_a, slot_a, host_alloc,
                        spill_state=_SpillState.COMPLETE, attempt_id=rec_a.spill_attempt_id + 1,
                    )
                raise RuntimeError("boom-copy")

            def paused_rel(allocs: Sequence[ArenaAllocation]) -> None:
                orig_rel(allocs)
                entered.wait(timeout=5.0)
                proceed.wait(timeout=5.0)

            monkeypatch.setattr(pool, "_copy_slot_d2h", failing_copy)
            monkeypatch.setattr(pool, "_release_or_defer_host_allocs", paused_rel)

            def _spill_with_err() -> None:
                with contextlib.suppress(RuntimeError):
                    pool.spill(["a"])
                    attempt1_err.append(AssertionError("expected boom-copy"))

            t1 = threading.Thread(target=_spill_with_err)
            t1.start()
            entered.wait(timeout=5.0)
            monkeypatch.undo()
            with pool._lock:
                stale_attempt = rec_a.spill_attempt_id
                retry_attempt = pool._begin_chunk_spill_unlocked(rec_a)
            assert 0 < stale_attempt < retry_attempt
            proceed.wait(timeout=5.0)
            t1.join(timeout=5.0)
            assert not attempt1_err
            _assert_stale_attempt_cannot_publish_or_clear(
                pool, pm, rec_a, slot_a, stale_attempt, retry_attempt
            )
            assert pool.spill(["a"]) == 1
            assert pool.evict_to_host(["a"]) == 1
            with pool.waves(["a"]) as w:
                _verify_wave_batch(next(w), {"a": payload})
    finally:
        entered.abort()
        proceed.abort()
        monkeypatch.undo()
        pm.clear()


def _verify_expert_reclaim_waits_for_relocating(
    pm: PinManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    from piper_offload.expert_slab_pool import _SlotPhase as _ExpSlotPhase

    schema = _kv_schema()
    exp_schema = _expert_schema()
    with (
        Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
        ExpertSlabPool(arena, exp_schema, max_slots=2, pin_manager=pm) as exp_pool,
        KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None) as kv_pool,
    ):
        exp_pool.register_group("g", [_make_expert_data(660), _make_expert_data(661)])
        with exp_pool.waves("g", [0, 1]) as ew:
            _ = next(ew)
        with exp_pool._lock:
            exp_pool._slots[1].phase = _ExpSlotPhase.RELOCATING
        relocating_wait_entered = threading.Event()
        orig_wait = exp_pool._op_cond.wait

        def spy_wait(timeout: float | None = None) -> bool:
            if exp_pool._slots[1].phase is _ExpSlotPhase.RELOCATING:
                relocating_wait_entered.set()
            return orig_wait(timeout=timeout)

        monkeypatch.setattr(exp_pool._op_cond, "wait", spy_wait)
        kv_err: list[BaseException] = []

        def run_kv_prod() -> None:
            try:
                _produce_chunks(kv_pool, ["k0"], {"k0": _make_chunk_data(662)})
            except BaseException as exc:
                kv_err.append(exc)

        t_kv = threading.Thread(target=run_kv_prod)
        t_kv.start()
        assert relocating_wait_entered.wait(timeout=5.0)
        assert t_kv.is_alive()
        with exp_pool._lock:
            exp_pool._slots[1].phase = _ExpSlotPhase.READY
            exp_pool._op_cond.notify_all()
        t_kv.join(timeout=5.0)
        monkeypatch.undo()
        assert not kv_err and kv_pool.location("k0") is KvChunkLocation.GPU_DIRTY


def _verify_kv_obtain_waits_for_allocating_spill(
    pm: PinManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = _kv_schema()
    entered, proceed = threading.Barrier(2), threading.Barrier(2)
    alloc_wait_entered = threading.Event()
    try:
        with (
            Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None) as kv_pool,
        ):
            _produce_chunks(kv_pool, ["k0", "k1"], {"k0": _make_chunk_data(663), "k1": _make_chunk_data(664)})
            orig_copy = kv_pool._copy_slot_d2h
            orig_wait = kv_pool._op_cond.wait
            prod_holder: list[threading.Thread] = []

            def paused_copy(slot_id: int, host_alloc: ArenaAllocation, *, non_blocking: bool) -> None:
                if kv_pool._chunks["k0"].spill_state is _SpillState.ALLOCATING:
                    entered.wait(timeout=5.0)
                    proceed.wait(timeout=5.0)
                orig_copy(slot_id, host_alloc, non_blocking=non_blocking)

            def spy_wait(timeout: float | None = None) -> bool:
                if (
                    prod_holder
                    and threading.current_thread() is prod_holder[0]
                    and kv_pool._chunks["k0"].spill_state is _SpillState.ALLOCATING
                ):
                    alloc_wait_entered.set()
                return orig_wait(timeout=timeout)

            monkeypatch.setattr(kv_pool, "_copy_slot_d2h", paused_copy)
            monkeypatch.setattr(kv_pool._op_cond, "wait", spy_wait)
            t_spill = threading.Thread(target=lambda: kv_pool.spill(["k0"]))
            t_spill.start()
            entered.wait(timeout=5.0)
            prod_err: list[BaseException] = []

            def run_prod() -> None:
                try:
                    _produce_chunks(kv_pool, ["k2"], {"k2": _make_chunk_data(665)})
                except BaseException as exc:
                    prod_err.append(exc)

            with kv_pool.waves(["k1"]) as wk:
                _ = next(wk)
                t_prod = threading.Thread(target=run_prod)
                prod_holder.append(t_prod)
                t_prod.start()
                assert alloc_wait_entered.wait(timeout=5.0)
                assert t_prod.is_alive()
                proceed.wait(timeout=5.0)
                t_spill.join(timeout=5.0)
                t_prod.join(timeout=5.0)
            assert not prod_err
            assert kv_pool.location("k0") is KvChunkLocation.HOST_ONLY
            assert kv_pool.location("k2") is KvChunkLocation.GPU_DIRTY
    finally:
        entered.abort()
        proceed.abort()
        monkeypatch.undo()


def test_expert_reclaim_waits_for_relocating_and_kv_obtain_waits_for_allocating_spill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    try:
        _verify_expert_reclaim_waits_for_relocating(pm, monkeypatch)
        _verify_kv_obtain_waits_for_allocating_spill(pm, monkeypatch)
    finally:
        pm.clear()


def test_waves_empty_and_none_do_not_hold_pool_lock() -> None:
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    try:
        with (
            Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
            KvChunkPool(arena, schema, max_slots=1, pin_manager=pm, spill_watermark=None) as pool,
        ):
            _produce_chunks(pool, ["a"], {"a": _make_chunk_data(720)})
            for empty_keys in ([], [None, None]):
                with pool.waves(empty_keys) as wave_it:
                    assert not pool._lock.locked()
                    assert pool.stats().resident_gpu_chunks == 1
                    assert pool.contains("a")
                    with pool.waves([]) as nested:
                        assert list(nested) == []
                    assert list(wave_it) == []

            assert pool.evict_to_host(["a"]) == 1
            _produce_chunks(pool, ["b"], {"b": _make_chunk_data(721)})
            before = pool.stats()
            with pytest.raises(ArenaCapacityError, match="allow_split=False"):
                with pool.waves(["a", "b"], allow_split=False) as w:
                    _ = next(w)
            after = pool.stats()
            assert (after.hits, after.misses, after.waves_executed) == (
                before.hits,
                before.misses,
                before.waves_executed,
            )
            assert pool._chunks["a"].staging_waves == 0
    finally:
        pm.clear()


def test_alloc_new_slot_frees_arena_slot_when_make_slot_state_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kv_schema = _kv_schema()
    exp_schema = _expert_schema()
    with Uint8DualArena(2 * kv_schema.drawer_bytes, device="cpu", alignment=256) as arena:
        kv_pool = KvChunkPool(arena, kv_schema, max_slots=1, spill_watermark=None)
        exp_pool = ExpertSlabPool(arena, exp_schema, max_slots=1)
        try:
            monkeypatch.setattr(
                kv_pool,
                "_make_slot_state",
                lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom-kv-slot")),
            )
            with pytest.raises(RuntimeError, match="boom-kv-slot"):
                kv_pool._alloc_new_slot_from_arena(allow_reclaim=False, phase=_SlotPhase.EMPTY)
            assert arena.active_high_slots == ()
            assert kv_pool._slots == {}

            from piper_offload.expert_slab_pool import _SlotPhase as _ExpSlotPhase

            monkeypatch.setattr(
                exp_pool,
                "_make_slot_state",
                lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom-exp-slot")),
            )
            with pytest.raises(RuntimeError, match="boom-exp-slot"):
                exp_pool._alloc_new_slot_from_arena(allow_reclaim=False, phase=_ExpSlotPhase.EMPTY)
            assert arena.active_low_slots == ()
            assert exp_pool._slots == {}
        finally:
            monkeypatch.undo()
            kv_pool.close()
            exp_pool.close()


def test_free_slot_to_arena_interrupt_retry_and_error_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _kv_schema()
    pm = PinManager(16 * 1024 * 1024, max_host_bytes=16 * 1024 * 1024)
    try:
        with Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena:
            pool = KvChunkPool(arena, schema, max_slots=2, pin_manager=pm, spill_watermark=None)
            _produce_chunks(pool, ["a", "b"], {"a": _make_chunk_data(730), "b": _make_chunk_data(731)})
            assert pool.discard(["a"]) == 1
            slot_a = next(s for s in pool._slots.values() if s.phase is _SlotPhase.EMPTY)

            orig_free = arena.free_high_slot
            free_calls = 0

            def interrupt_once_then_free(slot_id: int) -> None:
                nonlocal free_calls
                free_calls += 1
                if free_calls == 1:
                    raise KeyboardInterrupt("free-high-interrupt")
                orig_free(slot_id)

            monkeypatch.setattr(arena, "free_high_slot", interrupt_once_then_free)
            with pytest.raises(KeyboardInterrupt, match="free-high-interrupt"):
                pool._free_slot_to_arena(slot_a, sync_lease=False)
            monkeypatch.undo()
            assert free_calls == 2
            assert slot_a.slot_id not in pool._slots
            assert slot_a.slot_id not in arena.active_high_slots
            assert pool._inflight_arena_frees == 0

            slot_b = pool._chunks["b"].slot
            assert slot_b is not None
            assert pool.discard(["b"]) == 1
            monkeypatch.setattr(
                arena,
                "free_high_slot",
                lambda _sid: (_ for _ in ()).throw(RuntimeError("transient-free-err")),
            )
            with pytest.raises(RuntimeError, match="transient-free-err"):
                pool._free_slot_to_arena(slot_b, sync_lease=False)
            monkeypatch.undo()
            assert pool._slots.get(slot_b.slot_id) is slot_b
            assert slot_b.phase is _SlotPhase.EMPTY
            assert slot_b.slot_id in arena.active_high_slots
            pool.close()
            assert arena.active_high_slots == ()
    finally:
        monkeypatch.undo()
        pm.clear()

