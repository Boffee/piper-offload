"""Contracts for VRAM-only KvChunkPool, dual-end frontier sharing, and piper_kernels E2E."""

from __future__ import annotations

import ast
import contextlib
import importlib
import math
import pathlib
import threading
from collections.abc import Sequence
from typing import Any

import pytest
import torch

from piper_offload import (
    ArenaCapacityError,
    DeviceTransferContext,
    ExpertSlabPool,
    KvBatch,
    KvChunkPool,
    KvPoolStats,
    SlotSchema,
    Uint8DualArena,
    VramBudget,
    VramBudgetExhaustedError,
)
from piper_offload import kv_chunk_pool as kv_chunk_pool_mod
from piper_offload import stream_lease as stream_lease_mod
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
            piper_attention_decode,
        )

        _ = (
            PiperKVChunkPool,
            freeze_kv_chunks_,
            kv_chunk_storage_specs,
            piper_attention_decode,
        )
    except ImportError:
        return False
    return True


PIPER_KV_CUDA = pytest.mark.skipif(
    not _supports_piper_kv_cuda(),
    reason="CUDA and piper_kernels KV APIs required",
)


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
            batch.arena_views[fname][slot].copy_(
                src[fname].to(dev), non_blocking=(dev.type == "cuda")
            )
        batch.side_views["scale"][slot].copy_(
            src["scale"].to(dev), non_blocking=(dev.type == "cuda")
        )


def _verify_batch(
    batch: KvBatch,
    payloads: dict[Any, dict[str, torch.Tensor]],
    *,
    expected_keys: Sequence[Any] | None = None,
) -> None:
    if expected_keys is not None:
        assert batch.keys == tuple(expected_keys)
    assert batch.slot_ids.dtype is torch.int32
    assert tuple(batch.slot_ids.shape) == (len(batch.keys),)
    for idx, key in enumerate(batch.keys):
        slot = int(batch.slot_ids[idx].item())
        if key is None:
            assert slot == -1
            continue
        assert slot >= 0
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


def test_init_validation_owner_guard_and_layout_reset() -> None:
    schema = _kv_schema()
    with pytest.raises(TypeError, match="Uint8DualArena"):
        KvChunkPool(object(), schema)  # type: ignore[arg-type]
    closed_arena = Uint8DualArena(1024, device="cpu", alignment=256)
    closed_arena.close()
    with pytest.raises(RuntimeError, match="closed Uint8DualArena"):
        KvChunkPool(closed_arena, schema)
    with Uint8DualArena(2048, device="cpu", alignment=128) as small_align:
        with pytest.raises(ValueError, match="Arena alignment"):
            KvChunkPool(small_align, schema)
    with Uint8DualArena(2048, device="cpu", alignment=256) as arena:
        with pytest.raises(TypeError, match="SlotSchema"):
            KvChunkPool(arena, object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="max_slots"):
            KvChunkPool(arena, schema, max_slots=0)
        with pytest.raises(TypeError, match="VramBudget"):
            KvChunkPool(arena, schema, budget="bad")  # type: ignore[arg-type]
        with KvChunkPool(arena, schema, max_slots=2) as pool:
            assert arena.high_slot_layout is not None and arena.high_slot_layout.max_slots == 2
            with pytest.raises(RuntimeError, match="already registered"):
                KvChunkPool(arena, schema, max_slots=1)
        assert arena.high_slot_layout is None

    with Uint8DualArena(2048, device="cpu", slot_bytes=512, high_max_slots=3, alignment=256) as a2:
        with KvChunkPool(a2, schema) as pool2:
            assert pool2.max_slots == 3
        assert a2.high_slot_layout is not None and a2.high_slot_layout.max_slots == 3


def test_vram_budget_side_pool_accounting_and_init_rollback() -> None:
    schema = SlotSchema({"k": 256, "v": 256}, {"s": (torch.float32, (8,))})
    budget = VramBudget(8192)
    with Uint8DualArena(4 * 512, device="cpu", alignment=256, budget=budget) as arena:
        arena_reserved = budget.reserved_bytes
        with KvChunkPool(arena, schema, max_slots=2) as pool:
            assert budget.reserved_bytes > arena_reserved
            assert pool.is_open
        assert budget.reserved_bytes == arena_reserved

    two_side_schema = SlotSchema(
        {"k": 256, "v": 256},
        {"s1": (torch.float32, (8,)), "s2": (torch.float32, (32,))},
    )
    tight_budget = VramBudget(96)
    with Uint8DualArena(4 * 512, device="cpu", alignment=256) as arena:
        with pytest.raises(VramBudgetExhaustedError):
            KvChunkPool(arena, two_side_schema, max_slots=2, budget=tight_budget)
        assert tight_budget.reserved_bytes == 0
        assert arena.high_slot_layout is None


def test_persistent_views_and_zero_forbidden_imports() -> None:
    mod = importlib.import_module("piper_offload.kv_chunk_pool")
    src_path = pathlib.Path(mod.__file__ or "")
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for forbidden in ("pin_manager", "host_arena", "device_transfer", "piper_kernels"):
        assert not any(forbidden in name for name in imported), f"Forbidden import: {imported}"

    schema = _kv_schema()
    payloads = {f"c{i}": _make_chunk_data(10 + i) for i in range(2)}
    with (
        Uint8DualArena(4 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
        KvChunkPool(arena, schema, max_slots=3) as pool,
    ):
        av0, sv0, ab0, so0 = pool.arena_views, pool.side_views, pool.arena_bytes, pool.slot_offsets
        with pool.produce(["c0"]) as b0:
            assert b0.arena_views["k"] is av0["k"]
            assert b0.side_views["scale"] is sv0["scale"]
            assert b0.arena_bytes is ab0
            assert b0.slot_offsets["k"] is so0["k"]
            _write_kv_batch(b0, payloads)
        with pool.read(["c0"]) as b1:
            assert b1.arena_views["v"] is av0["v"]
            assert b1.slot_offsets["v"] is so0["v"]
            _verify_batch(b1, payloads, expected_keys=["c0"])
    with pytest.raises(RuntimeError, match="closed"):
        _ = pool.arena_views


def test_produce_validation_rollback_and_capacity_error() -> None:
    schema = _kv_schema()
    payloads = {f"c{i}": _make_chunk_data(20 + i) for i in range(4)}
    with (
        Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
        KvChunkPool(arena, schema, max_slots=2) as pool,
    ):
        with pool.produce(()) as empty_b:
            assert empty_b.active_slots == 0 and empty_b.slot_ids.numel() == 0

        with pytest.raises(TypeError, match="sequence of hashable"):
            with pool.produce("abc"):  # type: ignore[arg-type]
                pass
        with pytest.raises(ValueError, match="None sentinel"):
            with pool.produce([None]):  # type: ignore[list-item]
                pass
        with pytest.raises(ValueError, match="duplicates"):
            with pool.produce(["c0", "c0"]):
                pass

        with pytest.raises(RuntimeError, match="boom"):
            with pool.produce(["c0"]) as _b:
                assert "c0" not in pool
                raise RuntimeError("boom")
        assert "c0" not in pool and pool.stats().resident_chunks == 0

        with pool.produce(["c0"]) as b0:
            _write_kv_batch(b0, payloads)
        assert "c0" in pool

        with pytest.raises(ValueError, match="already resident"):
            with pool.produce(["c0"]):
                pass

        # Mid-batch capacity exhaustion rolls back newly allocated slot for "c1".
        before_slots = arena.active_high_slots
        with pytest.raises(ArenaCapacityError):
            with pool.produce(["c1", "c2"]):
                pass
        assert arena.active_high_slots == before_slots
        assert "c1" not in pool

        with pool.produce(["c1"]) as b1:
            _write_kv_batch(b1, payloads)
        assert pool.stats() == KvPoolStats(2, 2, 2, 0, 2, 0)
        with pytest.raises(ArenaCapacityError):
            with pool.produce(["c2"]):
                pass


def test_read_prefix_sharing_sentinels_errors_and_concurrent_readers() -> None:
    schema = _kv_schema()
    payloads = {f"c{i}": _make_chunk_data(40 + i) for i in range(2)}
    with (
        Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
        KvChunkPool(arena, schema, max_slots=2) as pool,
    ):
        with pool.read(()) as eb:
            assert eb.active_slots == 0 and eb.slot_ids.numel() == 0
        with pool.read([None, None]) as nb:
            assert nb.active_slots == 0 and nb.slot_ids.tolist() == [-1, -1]
        with pytest.raises(KeyError, match="not resident"):
            with pool.read(["c0", "missing"]):
                pass
        assert all(s.active_readers == 0 for s in pool._slots.values())

        with pool.produce(["c0", "c1"]) as b:
            _write_kv_batch(b, payloads)

        with pool.read(["c0", None, "c1", "c0"]) as rb:
            assert rb.active_slots == 2
            sids = rb.slot_ids.tolist()
            assert sids[0] == sids[3] and sids[1] == -1 and sids[0] != sids[2]
            _verify_batch(rb, payloads, expected_keys=["c0", None, "c1", "c0"])

        entered = threading.Barrier(2)
        leaving = threading.Barrier(2)
        errs: list[BaseException] = []

        def _reader() -> None:
            try:
                with pool.read(["c0"]):
                    entered.wait(timeout=2.0)
                    assert pool._key_to_slot["c0"].active_readers == 2
                    leaving.wait(timeout=2.0)
            except BaseException as exc:
                errs.append(exc)

        t = threading.Thread(target=_reader)
        t.start()
        try:
            with pool.read(["c0"]):
                entered.wait(timeout=2.0)
                assert pool._key_to_slot["c0"].active_readers == 2
                leaving.wait(timeout=2.0)
        finally:
            t.join(timeout=2.0)
        assert not errs
        assert pool._key_to_slot["c0"].active_readers == 0


def test_read_sees_producing_key_error() -> None:
    schema = _kv_schema()
    with (
        Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
        KvChunkPool(arena, schema, max_slots=2) as pool,
    ):
        with pool.produce(["c0"]):
            with pytest.raises(KeyError, match="being produced"):
                with pool.read(["c0"]):
                    pass


def test_discard_atomic_refusal_top_down_packing_and_reuse() -> None:
    schema = _kv_schema()
    payloads = {f"c{i}": _make_chunk_data(60 + i) for i in range(4)}
    with (
        Uint8DualArena(3 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
        KvChunkPool(arena, schema, max_slots=3) as pool,
    ):
        with pool.produce(["c0", "c1", "c2"]) as b:
            _write_kv_batch(b, payloads)
            assert b.slot_ids.tolist() == [2, 1, 0]

        with pool.read(["c1"]):
            with pytest.raises(RuntimeError, match="active reader"):
                pool.discard(["c0", "c1"])
        assert "c0" in pool and "c1" in pool

        # Discard bottom (c2 -> slot 0) and top (c0 -> slot 2); producing c3 reuses TOP slot 2!
        assert pool.discard(["c2", "c0", "c2", "missing"]) == 2
        with pool.produce(["c3"]) as b3:
            assert b3.slot_ids.tolist() == [2]
            _write_kv_batch(b3, payloads)
        assert pool.stats() == KvPoolStats(3, 3, 2, 1, 4, 2)


def test_discard_refuses_while_producing() -> None:
    schema = _kv_schema()
    with (
        Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena,
        KvChunkPool(arena, schema, max_slots=2) as pool,
    ):
        with pool.produce(["c0"]):
            with pytest.raises(RuntimeError, match="being produced"):
                pool.discard(["c0"])


@CUDA
def test_cuda_produce_read_nonblocking_discard_and_reuse_stream_safety() -> None:
    device = torch.device("cuda")
    schema = _kv_schema(k_bytes=512, v_bytes=512, scale_len=16)
    p0 = _make_chunk_data(100, k_bytes=512, v_bytes=512, scale_len=16)
    p1 = _make_chunk_data(101, k_bytes=512, v_bytes=512, scale_len=16)
    stream_write0 = torch.cuda.Stream(device=device)
    stream_read = torch.cuda.Stream(device=device)
    stream_write1 = torch.cuda.Stream(device=device)

    with (
        Uint8DualArena(schema.drawer_bytes, device=device, alignment=256) as arena,
        KvChunkPool(arena, schema, max_slots=1) as pool,
    ):
        with torch.cuda.stream(stream_write0):
            with pool.produce(["c0"]) as b0:
                _write_kv_batch(b0, {"c0": p0})

        with torch.cuda.stream(stream_read):
            with pool.read(["c0"]) as rb:
                torch.cuda._sleep(2_000_000)
                snap_k = rb.arena_views["k"][rb.slot_ids[0]].clone()
                snap_s = rb.side_views["scale"][rb.slot_ids[0]].clone()

        # Non-blocking discard after read exited (stream_read still running on GPU!), then overwrite on stream_write1.
        assert pool.discard(["c0"]) == 1
        with torch.cuda.stream(stream_write1):
            with pool.produce(["c1"]) as b1:
                _write_kv_batch(b1, {"c1": p1})

        with pool.read(["c1"]) as rb1:
            _verify_batch(rb1, {"c1": p1}, expected_keys=["c1"])
        torch.cuda.synchronize(device)
        torch.testing.assert_close(snap_k.cpu(), p0["k"], rtol=0, atol=0)
        torch.testing.assert_close(snap_s.cpu(), p0["scale"], rtol=0, atol=0)


@CUDA
def test_cuda_read_release_failure_handoff_and_retryable_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    schema = _kv_schema()
    p0 = _make_chunk_data(120)
    with (
        Uint8DualArena(schema.drawer_bytes, device=device, alignment=256) as arena,
        KvChunkPool(arena, schema, max_slots=1) as pool,
    ):
        with pool.produce(["c0"]) as b0:
            _write_kv_batch(b0, {"c0": p0})

        orig_record = stream_lease_mod._record_event
        calls = 0

        def _flaky_record(stream: torch.cuda.Stream) -> torch.cuda.Event:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected event failure")
            return orig_record(stream)

        monkeypatch.setattr(stream_lease_mod, "_record_event", _flaky_record)
        with pytest.raises(RuntimeError, match="injected event failure"):
            with pool.read(["c0"]):
                pass
        monkeypatch.undo()
        assert len(pool._wave_releases.pending) == 1
        assert pool.discard(["c0"]) == 1

        # Simulate interrupted close -> retryable close succeeds without leaking views/slots.
        with pool.produce(["c1"]) as b1:
            _write_kv_batch(b1, {"c1": p0})
        sync_orig = pool._wave_releases.synchronize_slots_for_close
        hit = False

        def _interrupt_once(*args: Any, **kwargs: Any) -> None:
            nonlocal hit
            if not hit:
                hit = True
                raise KeyboardInterrupt("interrupted close")
            sync_orig(*args, **kwargs)

        monkeypatch.setattr(pool._wave_releases, "synchronize_slots_for_close", _interrupt_once)
        with pytest.raises(KeyboardInterrupt, match="interrupted close"):
            pool.close()
        monkeypatch.undo()
        pool.close()
        assert not pool.is_open


def test_close_refused_while_sessions_active() -> None:
    schema = _kv_schema()
    payload = {"c0": _make_chunk_data(130)}
    with Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena:
        pool = KvChunkPool(arena, schema, max_slots=2)
        with pool.produce(["c0"]) as b:
            _write_kv_batch(b, payload)
            with pytest.raises(RuntimeError, match="active"):
                pool.close()
        with pool.read(["c0"]):
            with pytest.raises(RuntimeError, match="active"):
                pool.close()
        pool.close()


def test_dual_end_expert_and_kv_frontier_vram_only_coordination() -> None:
    exp_schema = _expert_schema()
    kv_schema = _kv_schema()
    experts = [_make_expert_data(200 + i) for i in range(4)]
    kv_payloads = {f"c{i}": _make_chunk_data(300 + i) for i in range(4)}
    total_bytes = 4 * kv_schema.drawer_bytes

    with (
        Uint8DualArena(total_bytes, device="cpu", alignment=256) as arena,
        ExpertSlabPool(arena, exp_schema, max_slots=4) as exp_pool,
        KvChunkPool(arena, kv_schema, max_slots=4) as kv_pool,
    ):
        exp_pool.register_group("moe", experts)
        with exp_pool.waves("moe", [0, 1, 2, 3]) as w:
            assert len(list(w)) >= 1
        assert arena.low_frontier == 4 * exp_schema.drawer_bytes

        # Producing 2 KV chunks pushes ExpertSlabPool's low frontier back down to 2 slots.
        with kv_pool.produce(["c0", "c1"]) as b0:
            _write_kv_batch(b0, kv_payloads)
        assert len(arena.active_high_slots) == 2
        assert arena.low_frontier <= kv_pool.layout.slot_offset(min(arena.active_high_slots))

        # Produce c2 into slot 1, then discard bottom c2 -> EMPTY bottom slot is reclaimed cleanly!
        with kv_pool.produce(["c2"]) as b1:
            _write_kv_batch(b1, kv_payloads)
        assert kv_pool.discard(["c2"]) == 1
        with exp_pool.waves("moe", [0, 1, 2]) as w2:
            waves2 = list(w2)
            assert sum(int((wv.expert_to_slot >= 0).sum().item()) for wv in waves2) >= 3
        assert len(arena.active_high_slots) == 2

        # Live READY bottom chunk (c1 in slot 2) blocks further ExpertSlabPool growth (no move or spill).
        assert arena.low_frontier == 2 * exp_schema.drawer_bytes
        with pytest.raises(ArenaCapacityError):
            arena.alloc_low_slot(allow_reclaim=True)
        with kv_pool.read(["c0", "c1"]) as rb:
            _verify_batch(rb, kv_payloads, expected_keys=["c0", "c1"])


@CUDA
def test_cuda_multithreaded_dual_end_expert_and_kv_pressure() -> None:
    device = torch.device("cuda")
    exp_schema = _expert_schema()
    kv_schema = _kv_schema()
    experts = [_make_expert_data(400 + i) for i in range(6)]
    total_bytes = 5 * kv_schema.drawer_bytes
    errs: list[BaseException] = []

    with (
        DeviceTransferContext(device) as ctx,
        Uint8DualArena(total_bytes, device=device, alignment=256) as arena,
        ExpertSlabPool(arena, exp_schema, max_slots=4, transfer_context=ctx) as exp_pool,
        KvChunkPool(arena, kv_schema, max_slots=3) as kv_pool,
    ):
        exp_pool.register_group("moe", experts)

        def _expert_worker() -> None:
            try:
                for step in range(8):
                    req = [step % 6, (step + 1) % 6, (step + 2) % 6]
                    with exp_pool.waves("moe", req) as waves:
                        seen = 0
                        for wv in waves:
                            mask = wv.expert_to_slot >= 0
                            seen += int(mask.sum().item())
                        assert seen == len(set(req))
            except BaseException as exc:
                errs.append(exc)

        def _kv_worker() -> None:
            try:
                for step in range(8):
                    keys = [f"t-{step}-0", f"t-{step}-1"]
                    data = {k: _make_chunk_data(500 + step * 2 + i) for i, k in enumerate(keys)}
                    with kv_pool.produce(keys) as pb:
                        _write_kv_batch(pb, data)
                    with kv_pool.read(keys) as rb:
                        _verify_batch(rb, data, expected_keys=keys)
                    kv_pool.discard(keys)
            except BaseException as exc:
                errs.append(exc)

        t1 = threading.Thread(target=_expert_worker)
        t2 = threading.Thread(target=_kv_worker)
        t1.start()
        t2.start()
        t1.join(timeout=15.0)
        t2.join(timeout=15.0)
        torch.cuda.synchronize(device)
        assert not errs


@PIPER_KV_CUDA
def test_cuda_piper_kernels_kv_pool_attention_e2e() -> None:
    from piper_kernels import PiperKVChunkPool, freeze_kv_chunks_, piper_attention_decode

    device = torch.device("cuda")
    kv_heads, q_heads, head_dim = 2, 4, 64
    num_chunks, max_slots = 4, 5
    schema = _build_piper_kv_schema(kv_heads, head_dim, device)
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

    with (
        Uint8DualArena(max_slots * schema.drawer_bytes, device=device, alignment=256) as arena,
        KvChunkPool(arena, schema, max_slots=max_slots) as kv_pool,
    ):
        kernel_pool = PiperKVChunkPool.from_tensors(**kv_pool.arena_views, **kv_pool.side_views)
        keys = [f"k{i}" for i in range(num_chunks)]
        with kv_pool.produce(keys[:2]) as b0:
            freeze_kv_chunks_(raw_k[:2], raw_v[:2], kernel_pool, b0.slot_ids)
        with kv_pool.produce(keys[2:]) as b1:
            freeze_kv_chunks_(raw_k[2:], raw_v[2:], kernel_pool, b1.slot_ids)
        with kv_pool.read(keys) as rb:
            got_out, got_lse = piper_attention_decode(
                query,
                kernel_pool,
                rb.slot_ids.unsqueeze(0),
                torch.tensor([num_chunks], dtype=torch.int32, device=device),
                tail_key=tail_k,
                tail_value=tail_v,
                tail_len=tail_len,
                return_lse=True,
                active_chunks_hint=max(1, rb.active_slots),
            )
        torch.cuda.synchronize(device)
        torch.testing.assert_close(got_out, ref_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(got_lse, ref_lse, rtol=1e-2, atol=1e-2)


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
    exp_schema = _expert_schema()
    gen = torch.Generator(device="cpu").manual_seed(800)
    raw_k = torch.randn((3, kv_heads, 1024, head_dim), generator=gen, dtype=torch.float16).to(device)
    raw_v = torch.randn((3, kv_heads, 1024, head_dim), generator=gen, dtype=torch.float16).to(device)
    query = torch.randn((1, q_heads, 1, head_dim), generator=gen, dtype=torch.float16).to(device)
    experts = [_make_expert_data(850 + i) for i in range(2)]
    n_chunks = torch.tensor([2], dtype=torch.int32, device=device)

    with (
        DeviceTransferContext(device) as ctx,
        Uint8DualArena(4 * kv_schema.drawer_bytes, device=device, alignment=256) as arena,
        ExpertSlabPool(arena, exp_schema, max_slots=2, transfer_context=ctx) as exp_pool,
        KvChunkPool(arena, kv_schema, max_slots=3) as kv_pool,
    ):
        exp_pool.register_group("moe", experts)
        kernel_pool = PiperKVChunkPool.from_tensors(**kv_pool.arena_views, **kv_pool.side_views)

        def decode_step(slot_ids: torch.Tensor) -> torch.Tensor:
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

        compiled_fn = torch.compile(decode_step, fullgraph=True, backend=counting_backend)
        with kv_pool.produce(["c0", "c1"]) as pb0:
            freeze_kv_chunks_(raw_k[:2], raw_v[:2], kernel_pool, pb0.slot_ids)
        with kv_pool.read(["c0", "c1"]) as rb1:
            out1 = compiled_fn(rb1.slot_ids)
            ref1 = decode_step(rb1.slot_ids).clone()
        assert compile_calls == 1
        torch.testing.assert_close(out1, ref1, rtol=1e-2, atol=1e-2)

        kv_pool.discard(["c0"])
        with exp_pool.waves("moe", [0, 1]) as ew:
            assert len(list(ew)) == 1
        with kv_pool.produce(["c2"]) as pb1:
            freeze_kv_chunks_(raw_k[2:], raw_v[2:], kernel_pool, pb1.slot_ids)
        with kv_pool.read(["c1", "c2"]) as rb2:
            out2 = compiled_fn(rb2.slot_ids)
            ref2 = decode_step(rb2.slot_ids).clone()
        assert compile_calls == 1
        torch.testing.assert_close(out2, ref2, rtol=1e-2, atol=1e-2)
        assert math.isfinite(float(out2.abs().max().item()))


def test_produce_preflight_rejects_oversized_batch_without_evicting_peer() -> None:
    exp_schema = _expert_schema()
    kv_schema = _kv_schema()
    experts = [_make_expert_data(900 + i) for i in range(2)]
    payload = {"c0": _make_chunk_data(910)}
    with (
        Uint8DualArena(4 * kv_schema.drawer_bytes, device="cpu", alignment=256) as arena,
        ExpertSlabPool(arena, exp_schema, max_slots=3) as exp_pool,
        KvChunkPool(arena, kv_schema, max_slots=3) as kv_pool,
    ):
        exp_pool.register_group("moe", experts)
        with exp_pool.waves("moe", [0, 1]) as waves:
            assert len(list(waves)) == 1
        low_frontier_before = arena.low_frontier
        with kv_pool.produce(["c0"]) as b0:
            _write_kv_batch(b0, payload)

        with pytest.raises(ArenaCapacityError, match="exceeding max_slots=3"):
            with kv_pool.produce(["c1", "c2", "c3"]):
                pass
        assert arena.low_frontier == low_frontier_before
        assert exp_pool.stats().resident_experts == 2
        assert kv_pool._active_sessions == 0


def test_produce_rollback_propagates_interrupt_and_clears_producing_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _kv_schema()
    with Uint8DualArena(
        2 * schema.drawer_bytes,
        device="cpu",
        low_slot_bytes=schema.drawer_bytes,
        alignment=256,
    ) as arena:
        with KvChunkPool(arena, schema, max_slots=2) as pool:
            arena.alloc_low_slot()  # Consume low slot 0 so only 1 high slot remains.
            monkeypatch.setattr(
                pool,
                "_free_slot_to_arena",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt("free intr")),
            )
            with pytest.raises(KeyboardInterrupt, match="free intr"):
                with pool.produce(["c0", "c1"]):
                    pass
            monkeypatch.undo()
            assert not pool._producing_keys
            assert pool._active_sessions == 0
            assert all(s.phase is kv_chunk_pool_mod._SlotPhase.EMPTY for s in pool._slots.values())


def test_produce_interrupted_commit_cleans_partial_keys_and_active_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _kv_schema()
    p0 = _make_chunk_data(920)
    p_other = _make_chunk_data(921)

    with Uint8DualArena(2 * schema.drawer_bytes, device="cpu", alignment=256) as arena:
        with KvChunkPool(arena, schema, max_slots=2) as pool:
            def _partial_commit(claimed: Any) -> None:
                key0, slot0, _n = claimed[0]
                pool._producing_keys.discard(key0)
                slot0.key = key0
                slot0.phase = kv_chunk_pool_mod._SlotPhase.READY
                pool._key_to_slot[key0] = slot0
                raise KeyboardInterrupt("commit intr")

            monkeypatch.setattr(pool, "_commit_produce_unlocked", _partial_commit)
            with pytest.raises(KeyboardInterrupt, match="commit intr"):
                with pool.produce(["c0", "c1"]) as b0:
                    _write_kv_batch(b0, {"c0": p0, "c1": p0})
            monkeypatch.undo()

            assert "c0" not in pool and "c1" not in pool
            assert not pool._key_to_slot and not pool._producing_keys
            assert pool._active_sessions == 0
            with pool.produce(["other"]) as b1:
                _write_kv_batch(b1, {"other": p_other})
            with pytest.raises(KeyError, match="not resident"):
                with pool.read(["c0"]):
                    pass


def test_push_locked_write_clears_writing_flag_on_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeEvent:
        def synchronize(self) -> None:
            return None

    class _FakeStream:
        def wait_event(self, _ev: Any) -> None:
            return None

        def wait_stream(self, _st: Any) -> None:
            return None

        def record_event(self) -> _FakeEvent:
            return _FakeEvent()

        def synchronize(self) -> None:
            return None

    fake_stream: Any = _FakeStream()
    monkeypatch.setattr(torch.cuda, "stream", lambda _s: contextlib.nullcontext())
    lease = StreamLease(fake_stream)
    slot = kv_chunk_pool_mod._KvSlot(slot_id=0, lease=lease)
    with contextlib.ExitStack() as stack:
        monkeypatch.setattr(
            stack,
            "push",
            lambda _cb: (_ for _ in ()).throw(KeyboardInterrupt("push intr")),
        )
        with pytest.raises(KeyboardInterrupt, match="push intr"):
            kv_chunk_pool_mod._push_locked_write(stack, slot, fake_stream)
    assert not lease._writing
    with lease.write(fake_stream):
        pass
    assert lease.is_staged
    lease.synchronize_and_reset()


def test_produce_waits_for_inflight_arena_frees_before_failing() -> None:
    schema = _kv_schema()
    p = _make_chunk_data(930)
    with Uint8DualArena(schema.drawer_bytes, device="cpu", alignment=256) as arena:
        with KvChunkPool(arena, schema, max_slots=1) as pool:
            arena.alloc_high_slot()
            with pool._lock:
                pool._inflight_arena_frees = 1

            def _finish_free() -> None:
                threading.Event().wait(0.03)
                arena.free_high_slot(0)
                with pool._lock:
                    pool._inflight_arena_frees = 0
                    pool._op_cond.notify_all()

            t = threading.Thread(target=_finish_free)
            t.start()
            try:
                with pool.produce(["c0"]) as b0:
                    _write_kv_batch(b0, {"c0": p})
            finally:
                t.join(timeout=2.0)
            with pool.read(["c0"]) as rb:
                _verify_batch(rb, {"c0": p}, expected_keys=["c0"])


def test_cpu_read_release_failure_precedence_and_close_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = _kv_schema()
    p = {"c0": _make_chunk_data(940)}
    with Uint8DualArena(schema.drawer_bytes, device="cpu", alignment=256) as arena:
        pool = KvChunkPool(arena, schema, max_slots=1)
        with pool.produce(["c0"]) as b0:
            _write_kv_batch(b0, p)

        def _raise_rt(_self: Any) -> None:
            raise RuntimeError("release boom")

        monkeypatch.setattr(kv_chunk_pool_mod._WaveLease, "release", _raise_rt)
        with pytest.raises(RuntimeError, match="release boom"):
            with pool.read(["c0"]):
                pass
        with pytest.raises(ValueError, match="caller err"):
            with pool.read(["c0"]):
                raise ValueError("caller err")
        monkeypatch.setattr(
            kv_chunk_pool_mod._WaveLease,
            "release",
            lambda _self: (_ for _ in ()).throw(KeyboardInterrupt("release intr")),
        )
        with pytest.raises(KeyboardInterrupt, match="release intr"):
            with pool.read(["c0"]):
                raise ValueError("caller err")
        monkeypatch.undo()
        assert pool._slots[0].active_readers == 0 and pool._active_sessions == 0

        monkeypatch.setattr(
            pool._wave_releases,
            "synchronize_slots_for_close",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt("close intr")),
        )
        with pytest.raises(KeyboardInterrupt, match="close intr"):
            pool.close()
        monkeypatch.undo()
        assert pool._closing and not pool._closed
        pool.close()
        assert pool._closed


@CUDA
def test_cuda_produce_exception_orders_future_reuse_and_reclaim() -> None:
    device = torch.device("cuda")
    kv_schema = _kv_schema(k_bytes=512, v_bytes=512, scale_len=16)
    exp_schema = _expert_schema(up_bytes=512, down_bytes=512, scale_len=16)
    p1 = _make_chunk_data(950, k_bytes=512, v_bytes=512, scale_len=16)
    experts = [_make_expert_data(960, up_bytes=512, down_bytes=512, scale_len=16)]
    s_abort = torch.cuda.Stream(device=device)
    s_reuse = torch.cuda.Stream(device=device)

    with (
        DeviceTransferContext(device) as ctx,
        Uint8DualArena(kv_schema.drawer_bytes, device=device, alignment=256) as arena,
        ExpertSlabPool(arena, exp_schema, max_slots=1, transfer_context=ctx) as exp_pool,
        KvChunkPool(arena, kv_schema, max_slots=1) as kv_pool,
    ):
        exp_pool.register_group("moe", experts)
        with pytest.raises(RuntimeError, match="abort produce"):
            with torch.cuda.stream(s_abort):
                with kv_pool.produce(["aborted"]) as b0:
                    torch.cuda._sleep(2_000_000)
                    b0.arena_views["k"][b0.slot_ids[0]].fill_(0xAA)
                    raise RuntimeError("abort produce")
        assert "aborted" not in kv_pool
        assert kv_pool._slots[0].lease is not None and not kv_pool._slots[0].lease._writing

        with torch.cuda.stream(s_reuse):
            with kv_pool.produce(["c1"]) as b1:
                _write_kv_batch(b1, {"c1": p1})
        with kv_pool.read(["c1"]) as rb:
            _verify_batch(rb, {"c1": p1}, expected_keys=["c1"])
        assert kv_pool.discard(["c1"]) == 1
        with exp_pool.waves("moe", [0]) as ew:
            assert len(list(ew)) == 1
        torch.cuda.synchronize(device)
