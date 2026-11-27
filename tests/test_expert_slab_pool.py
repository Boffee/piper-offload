"""Tests for :class:`ExpertSlabPool` over :class:`Uint8DualArena` and :class:`SlotSchema`."""

from __future__ import annotations

import ast
import contextlib
import ctypes
import gc
import json
import struct
import threading
import time
import weakref
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

import piper_offload.checkpoint as checkpoint_module
from piper_offload import (
    ArenaCapacityError,
    DeviceTransferContext,
    ExpertPoolStats,
    ExpertSlabPool,
    ExpertWave,
    MappedCheckpoint,
    PinManager,
    SlotSchema,
    Uint8DualArena,
    VramBudget,
    VramBudgetExhaustedError,
)
from piper_offload.dual_arena import HighSlotLayout
from piper_offload.stream_lease import StreamLease

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_TAGS = {dtype: tag for tag, dtype in checkpoint_module._DTYPES.items()}


def _supports_trellis_moe_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from piper_kernels.moe import MoEWeights, moe_ffn, prepare_moe_input

        _ = (MoEWeights, moe_ffn, prepare_moe_input)
    except ImportError:
        return False
    return True


TRELLIS_MOE_CUDA = pytest.mark.skipif(
    not _supports_trellis_moe_cuda(),
    reason="CUDA and piper_kernels.moe required",
)


class _TensorSubclass(torch.Tensor):
    pass


def _make_synthetic_expert(
    seed: int,
    *,
    up_bytes: int,
    down_bytes: int,
    scale_len: int = 8,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        "up": torch.randint(0, 256, (up_bytes,), generator=gen, dtype=torch.uint8),
        "down": torch.randint(0, 256, (down_bytes,), generator=gen, dtype=torch.uint8),
        "scale": torch.randn((scale_len,), generator=gen, dtype=dtype),
    }


def _assert_wave_contents(
    wave: ExpertWave,
    experts: list[dict[str, torch.Tensor]],
) -> None:
    num_experts = len(experts)
    assert tuple(wave.expert_to_slot.shape) == (num_experts,)
    active_set = set(wave.expert_ids)
    for eid in range(num_experts):
        slot = int(wave.expert_to_slot[eid].item())
        if eid not in active_set:
            if not wave.is_hit_wave or wave.active_slots_hint < num_experts:
                assert slot == -1
            continue
        assert slot >= 0
        src = experts[eid]
        for fname in ("up", "down"):
            expected = src[fname].reshape(-1).cpu()
            offset = int(wave.slot_offsets[fname][slot].item())
            actual_1d = wave.arena_bytes[offset : offset + expected.numel()].cpu()
            torch.testing.assert_close(actual_1d, expected, rtol=0, atol=0)
            actual_view = wave.arena_views[fname][slot].reshape(-1)[: expected.numel()].cpu()
            torch.testing.assert_close(actual_view, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            wave.side_views["scale"][slot].cpu(), src["scale"].cpu(), rtol=0, atol=0
        )


def _write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header: dict[str, object] = {}
    data = bytearray()
    for name, tensor in tensors.items():
        flat = tensor.detach().cpu().reshape(-1).contiguous().view(torch.uint8)
        raw = ctypes.string_at(flat.data_ptr(), flat.nbytes)
        header[name] = {
            "dtype": _TAGS[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    encoded = json.dumps(header).encode()
    encoded += b" " * (-(8 + len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(data))


def _build_synthetic_trellis_experts(
    *,
    hidden_size: int,
    intermediate_size: int,
    block_size: int = 16,
) -> list[dict[str, torch.Tensor]]:
    def _make_matrix(
        out_f: int,
        in_f: int,
        rw_lo: int,
        rw_hi: int,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(seed)
        blocks_per_row = in_f // block_size
        half = out_f // 2
        rwidths = torch.cat([
            torch.full((half,), rw_lo, dtype=torch.uint8),
            torch.full((out_f - half,), rw_hi, dtype=torch.uint8),
        ]).contiguous()
        qdata = torch.randint(
            0, 256, (int(rwidths.sum().item()) * blocks_per_row,), generator=gen, dtype=torch.uint8
        )
        scale_choices = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.bfloat16)
        bscales = scale_choices[torch.randint(0, 4, (out_f, blocks_per_row), generator=gen)].to(
            torch.float8_e4m3fn
        )
        scale = (torch.rand((out_f,), generator=gen, dtype=torch.float32) * 0.5 + 0.25).contiguous()
        return qdata, bscales, scale, rwidths

    up_out = 2 * intermediate_size
    expert_dicts: list[dict[str, torch.Tensor]] = []
    for idx, (rw_lo, rw_hi) in enumerate(((4, 6), (6, 8), (4, 6), (6, 8))):
        uq, ubs, us, urw = _make_matrix(up_out, hidden_size, rw_lo, rw_hi, idx)
        dq, dbs, ds, drw = _make_matrix(hidden_size, intermediate_size, rw_lo, rw_hi, idx + 16)
        expert_dicts.append({
            "up_gate_qdata": uq,
            "down_qdata": dq,
            "up_gate_block_scales": ubs,
            "down_block_scales": dbs,
            "up_gate_scale": us,
            "down_scale": ds,
            "up_gate_row_widths": urw,
            "down_row_widths": drw,
        })
    return expert_dicts


def test_layer1_and_expert_slab_pool_never_import_piper_kernels() -> None:
    src_root = Path(__file__).resolve().parent.parent / "src" / "piper_offload"
    for filename in ("dual_arena.py", "expert_slab_pool.py"):
        tree = ast.parse((src_root / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("piper_kernels") for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("piper_kernels")


def test_init_budget_rollback_and_arena_inheritance() -> None:
    schema2 = SlotSchema(
        {"up": 256},
        {"s1": (torch.float32, (4,)), "s2": (torch.float32, (4,))},
    )
    # s1 consumes 64B (fits in 96B); s2 consumes 64B (exhausts 96B -> rolls back s1 & layout).
    tiny_budget = VramBudget(96)
    with Uint8DualArena(4096, device="cpu", alignment=256) as arena:
        with pytest.raises(VramBudgetExhaustedError):
            ExpertSlabPool(arena, schema2, max_slots=4, budget=tiny_budget)
        assert tiny_budget.reserved_bytes == 0
        assert tiny_budget.used_bytes == 0
        assert arena.low_slot_layout is None
        arena.set_low_reclaimer(lambda _t: None)
        arena.set_low_reclaimer(None)

    with pytest.raises(RuntimeError, match="closed Uint8DualArena"):
        ExpertSlabPool(arena, schema2, max_slots=2)

    # Auto-inherit arena's VramBudget when budget=None.
    shared_budget = VramBudget(16384)
    with Uint8DualArena(4096, device="cpu", alignment=256, budget=shared_budget) as budgeted_arena:
        arena_reserved = shared_budget.reserved_bytes
        with ExpertSlabPool(budgeted_arena, schema2, max_slots=4):
            assert shared_budget.reserved_bytes >= arena_reserved + 4 * schema2.side_bytes_per_slot
        assert shared_budget.reserved_bytes == arena_reserved
    assert shared_budget.reserved_bytes == 0


def test_init_alignment_ownership_and_clean_teardown() -> None:
    schema = SlotSchema({"up": 256, "down": 128}, {"scale": (torch.float32, (4,))}, field_alignment=256)
    with Uint8DualArena(4096, device="cpu", alignment=16) as low_align:
        with pytest.raises(ValueError, match="Arena alignment"):
            ExpertSlabPool(low_align, schema)

    # Preconfigured low-layout validation error paths.
    with Uint8DualArena(4096, device="cpu", alignment=256, low_slot_bytes=256) as small_low:
        with pytest.raises(ValueError, match="smaller than"):
            ExpertSlabPool(small_low, schema)
        small_low.configure_low_slots(512, max_slots=4)
        sid = small_low.alloc_low_slot()
        with pytest.raises(RuntimeError, match="active low slots"):
            ExpertSlabPool(small_low, schema)
        small_low.free_low_slot(sid)
        with pytest.raises(ValueError, match="does not match"):
            ExpertSlabPool(small_low, schema, max_slots=2)

    # Concurrent / duplicate attach fails immediately without wiping first pool's layout.
    with Uint8DualArena(4096, device="cpu", alignment=256) as arena, ExpertSlabPool(arena, schema, max_slots=4) as pool:
        assert arena.low_slot_layout is not None
        with pytest.raises(RuntimeError, match="already registered"):
            ExpertSlabPool(arena, schema, max_slots=4)
        experts = [_make_synthetic_expert(0, up_bytes=200, down_bytes=100, scale_len=4)]
        pool.register_group("g0", experts)
        with pool.waves("g0", [0]) as waves:
            w = next(waves)
            arena_ref = weakref.ref(w.arena_bytes)
            side_ref = weakref.ref(w.side_views["scale"])
            del w
    # Clean close drops strong view references and resets auto-configured layout.
    gc.collect()
    assert arena_ref() is None
    assert side_ref() is None
    assert not arena.is_open

    # Explicit preconfigured layout is preserved across close(), while auto-configured resets.
    with Uint8DualArena(4096, device="cpu", alignment=256) as reusable:
        with ExpertSlabPool(reusable, schema, max_slots=4):
            assert reusable.low_slot_layout is not None
        assert reusable.low_slot_layout is None
        with ExpertSlabPool(reusable, SlotSchema({"x": 128}), max_slots=2):
            assert reusable.low_slot_layout is not None
            assert reusable.low_slot_layout.max_slots == 2
        reusable.configure_low_slots(512, max_slots=4)
        with ExpertSlabPool(reusable, schema, max_slots=4):
            pass
        assert reusable.low_slot_layout is not None
        assert reusable.low_slot_layout.slot_bytes == 512


def test_failed_close_keeps_closing_and_is_retryable() -> None:
    schema = SlotSchema({"up": 128, "down": 128}, {"scale": (torch.float32, (4,))})
    with Uint8DualArena(4096, device="cpu", alignment=256) as retry_arena:
        retry_pool = ExpertSlabPool(retry_arena, schema, max_slots=2)
        retry_pool.register_group("g", [_make_synthetic_expert(0, up_bytes=128, down_bytes=128, scale_len=4)])
        with retry_pool.waves("g", [0]) as waves:
            _ = next(waves)
        real_close = retry_pool._side_pools["scale"].close
        calls = 0

        def _flaky_close() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient close error")
            real_close()

        retry_pool._side_pools["scale"].close = _flaky_close  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="transient close error"):
            retry_pool.close()
        assert not retry_pool.is_open
        with pytest.raises(RuntimeError, match="closed"), retry_pool.waves("g", [0]):
            pass
        retry_pool.close()
        assert retry_arena.low_slot_layout is None


def test_register_group_validation_frozen_param_and_padding() -> None:
    schema = SlotSchema({"up": 256, "down": (2, 64)}, {"scale": (torch.float32, (4,))})
    with Uint8DualArena(4096, device="cpu", alignment=256) as arena, ExpertSlabPool(arena, schema, max_slots=4) as pool:
        with pytest.raises(ValueError, match="at least one expert"):
            pool.register_group("empty", [])
        with pytest.raises(ValueError, match="at least one expert"):
            pool.register_group("empty_gen", (e for e in ()))
        with pytest.raises(ValueError, match="fields mismatch"):
            pool.register_group("bad_keys", [{"up": torch.zeros(64, dtype=torch.uint8)}])
        with pytest.raises(ValueError, match="must be torch.uint8"):
            pool.register_group(
                "bad_dtype",
                [{
                    "up": torch.zeros(64, dtype=torch.int8),
                    "down": torch.zeros(64, dtype=torch.uint8),
                    "scale": torch.zeros(4),
                }],
            )
        with pytest.raises(ValueError, match="size 257 must lie in"):
            pool.register_group("too_big", [_make_synthetic_expert(0, up_bytes=257, down_bytes=128, scale_len=4)])
        with pytest.raises(ValueError, match="requires_grad=False"):
            pool.register_group(
                "grad_side",
                [{
                    "up": torch.zeros(64, dtype=torch.uint8),
                    "down": torch.zeros(64, dtype=torch.uint8),
                    "scale": torch.ones(4, dtype=torch.float32, requires_grad=True),
                }],
            )
        sub_tensor = _TensorSubclass(torch.zeros(64, dtype=torch.uint8))
        with pytest.raises(ValueError, match="plain torch.Tensor"):
            pool.register_group(
                "sub",
                [{"up": sub_tensor, "down": torch.zeros(64, dtype=torch.uint8), "scale": torch.ones(4)}],
            )
        # Partial-batch failure does not leave group registered.
        bad_second = [
            _make_synthetic_expert(0, up_bytes=64, down_bytes=64, scale_len=4),
            {
                "up": torch.zeros(64, dtype=torch.uint8),
                "down": torch.zeros(64, dtype=torch.uint8),
                "scale": torch.ones(4, dtype=torch.float32, requires_grad=True),
            },
        ]
        with pytest.raises(ValueError, match="requires_grad=False"):
            pool.register_group("atomic_group", bad_second)
        pool.register_group("atomic_group", [_make_synthetic_expert(0, up_bytes=64, down_bytes=64, scale_len=4)])

        # Frozen nn.Parameter + non-contiguous arena/side fields + variable-size dumb padding.
        non_contig_up = torch.arange(384, dtype=torch.uint8)[::2]
        non_contig_scale = torch.arange(8, dtype=torch.float32)[::2]
        experts = [
            {
                "up": non_contig_up,
                "down": _make_synthetic_expert(1, up_bytes=64, down_bytes=96, scale_len=4)["down"],
                "scale": torch.nn.Parameter(non_contig_scale, requires_grad=False),
            },
            _make_synthetic_expert(2, up_bytes=256, down_bytes=128, scale_len=4),
            _make_synthetic_expert(3, up_bytes=128, down_bytes=64, scale_len=4),
        ]
        pool.register_group("g0", experts)
        with pytest.raises(ValueError, match="already registered"):
            pool.register_group("g0", experts)

        with pool.waves("g0", [0, 1, 2]) as waves:
            collected = list(waves)
            assert len(collected) == 1
            _assert_wave_contents(collected[0], experts)
            assert collected[0].arena_views["up"].stride(0) == pool.layout.slot_stride


def test_hit_first_waves_tiny_pool_and_multigroup_sharing() -> None:
    schema = SlotSchema({"up": 256, "down": 128}, {"scale": (torch.float32, (8,))})
    with Uint8DualArena(4096, device="cpu", alignment=256) as arena, ExpertSlabPool(arena, schema, max_slots=2) as pool:
        l0 = [_make_synthetic_expert(i, up_bytes=200, down_bytes=100) for i in range(4)]
        l1 = [_make_synthetic_expert(10 + i, up_bytes=240, down_bytes=110) for i in range(4)]
        pool.register_group("layer0", l0)
        pool.register_group("layer1", l1)

        # Prefetching [0, 1] keeps their frequency at 0 so cold [2, 3] pack into 1 full miss wave.
        assert pool.prefetch("layer0", [0, 1]) == 2
        with pool.waves("layer0", [0, 1, 2, 3]) as waves:
            seen: list[tuple[int, bool, tuple[int, ...]]] = []
            for w in waves:
                seen.append((w.index, w.is_hit_wave, w.expert_ids))
                _assert_wave_contents(w, l0)
            assert seen == [(0, True, (0, 1)), (1, False, (2, 3))]

        # Now warm [0, 1] twice cleanly and verify W-TinyLFU protects hot [1] from cold [2, 3].
        for _ in range(2):
            with pool.waves("layer0", [0, 1]) as waves:
                _ = list(waves)
        with pool.waves("layer0", [0, 1, 2, 3]) as waves:
            seen_protected: list[tuple[int, bool, tuple[int, ...]]] = []
            for w in waves:
                seen_protected.append((w.index, w.is_hit_wave, w.expert_ids))
                _assert_wave_contents(w, l0)
            assert seen_protected == [(0, True, (0, 1)), (1, False, (2,)), (2, False, (3,))]
        misses_before = pool.stats().misses
        with pool.waves("layer0", [1]) as waves:
            assert next(waves).is_hit_wave
        assert pool.stats().misses == misses_before

        # Disjoint group_id owns distinct entries even with matching integer IDs.
        layer1_eids: list[int] = []
        with pool.waves("layer1", [0, 1]) as waves:
            for w_l1 in waves:
                assert w_l1.group_id == "layer1"
                layer1_eids.extend(w_l1.expert_ids)
                _assert_wave_contents(w_l1, l1)
        assert layer1_eids == [0, 1]


def test_overlap_staging_and_early_break_exception_cleanup() -> None:
    schema = SlotSchema({"up": 128, "down": 128}, {"scale": (torch.float32, (4,))})
    experts = [_make_synthetic_expert(i, up_bytes=128, down_bytes=128, scale_len=4) for i in range(4)]

    # max_slots=3, [0] resident -> Wave 0 holds slot 0 and lookahead pre-stages E1 into spare slot 1!
    with Uint8DualArena(4096, device="cpu", alignment=256) as arena, ExpertSlabPool(arena, schema, max_slots=3) as pool:
        pool.register_group("moe", experts)
        with pool.waves("moe", [0]) as waves:
            _ = list(waves)
        assert pool.stats().allocated_slots == 1

        with pool.waves("moe", [0, 1, 2], max_wave_slots=1) as waves:
            w0 = next(waves)
            assert w0.is_hit_wave
            _assert_wave_contents(w0, experts)
            # While Wave 0 is still active, E1 has already claimed and staged slot 1.
            assert pool.stats().allocated_slots == 2
            with pytest.raises(RuntimeError, match="Cannot close ExpertSlabPool"):
                pool.close()
            # Early break commits pre-staged E1 with active_readers == 0.
        misses_before = pool.stats().misses
        with pool.waves("moe", [1]) as waves:
            w1 = next(waves)
            assert w1.is_hit_wave
            _assert_wave_contents(w1, experts)
        assert pool.stats().misses == misses_before

        # Mid-wave exception during overlap staging also commits E3 cleanly without leaking slots.
        with (
            pytest.raises(ValueError, match="boom"),
            pool.waves("moe", [0, 3], max_wave_slots=1) as waves,
        ):
            _ = next(waves)
            assert pool.stats().allocated_slots == 3
            raise ValueError("boom")
        misses_before_3 = pool.stats().misses
        with pool.waves("moe", [3]) as waves:
            w3 = next(waves)
            assert w3.is_hit_wave
            _assert_wave_contents(w3, experts)
        assert pool.stats().misses == misses_before_3


def test_wtinylfu_multi_miss_bypass_speculative_and_stats() -> None:
    schema = SlotSchema({"up": 128, "down": 128}, {"scale": (torch.float32, (4,))})
    experts = [_make_synthetic_expert(i, up_bytes=128, down_bytes=128, scale_len=4) for i in range(16)]

    # Decode W-TinyLFU scan resistance: 4-miss cold decode wave borrows at most 1 resident slot.
    with Uint8DualArena(4096, device="cpu", alignment=256) as a4, ExpertSlabPool(a4, schema, max_slots=4) as p4:
        p4.register_group("moe", experts)
        for _ in range(5):
            with p4.waves("moe", [0, 1, 2, 3]) as waves:
                _ = list(waves)
        with p4.waves("moe", [4, 5, 6, 7]) as waves:
            cold_count = 0
            for w in waves:
                cold_count += 1
                _assert_wave_contents(w, experts)
            assert cold_count == 4
        st4 = p4.stats()
        assert st4.resident_experts == 3
        assert st4.transient_slots == 1
        assert st4.evictions == 1
        assert st4.waves_executed > 0

        # Nested no_record_stats isolates draft frequency and blocks transient->resident promotion.
        with p4.no_record_stats():
            with p4.no_record_stats():
                for _ in range(4):
                    with p4.waves("moe", [7]) as waves:
                        assert next(waves).is_hit_wave
            assert p4.stats().transient_slots == 1
            assert p4.stats().resident_experts == 3

        # Invalid commit_accepted batch is atomic (earlier valid key receives zero credit).
        with pytest.raises(KeyError):
            p4.commit_accepted([("moe", 7), ("missing", 0)])
        assert p4.stats().resident_experts == 3

        # Accepting draft key 7 twice promotes transient -> resident and protects it vs cold 8.
        p4.commit_accepted([("moe", 7), ("moe", 7)])
        assert p4.stats().resident_experts == 4
        assert p4.stats().transient_slots == 0
        assert p4.prefetch("moe", [8], bypass=False) == 0  # hotter residents are protected

    # Prefill bypass=True defaults to max_slots // 4 (=2 in an 8-slot pool), keeping >= 75% hot.
    reclaim_calls: list[int] = []
    with (
        Uint8DualArena(4096, device="cpu", slot_bytes=256, alignment=256) as a8,
        ExpertSlabPool(a8, schema, max_slots=8) as p8,
    ):
        a8.set_high_reclaimer(reclaim_calls.append)
        p8.register_group("moe", experts)
        for _ in range(3):
            with p8.waves("moe", list(range(8))) as waves:
                _ = list(waves)
        assert p8.prefetch("moe", [8], bypass=False) == 0
        assert reclaim_calls == []
        with p8.waves("moe", [8, 9, 10, 11], bypass=True) as waves:
            bypass_lens: list[int] = []
            for w in waves:
                bypass_lens.append(len(w.expert_ids))
                _assert_wave_contents(w, experts)
            assert bypass_lens == [2, 2]
        assert p8.stats().resident_experts == 6
        assert p8.stats().transient_slots == 2
        with p8.waves("moe", [12, 13, 14, 15], bypass=True, max_wave_slots=4) as waves:
            assert [len(w.expert_ids) for w in waves] == [4]
        assert p8.stats().resident_experts == 4
        assert p8.stats().transient_slots == 4
        st8 = p8.stats()
        assert st8 == ExpertPoolStats(
            max_slots=8,
            allocated_slots=8,
            resident_experts=st8.resident_experts,
            transient_slots=st8.transient_slots,
            hits=st8.hits,
            misses=st8.misses,
            evictions=st8.evictions,
            relocations=0,
            waves_executed=st8.waves_executed,
            fast_path_waves=st8.fast_path_waves,
        )


def test_all_resident_zero_sync_fast_path_and_empty_inputs() -> None:
    schema = SlotSchema({"up": 128, "down": 128}, {"scale": (torch.float32, (4,))})
    experts = [_make_synthetic_expert(i, up_bytes=128, down_bytes=128, scale_len=4) for i in range(4)]
    with Uint8DualArena(4096, device="cpu", alignment=256) as arena, ExpertSlabPool(arena, schema, max_slots=4) as pool:
        pool.register_group("small", experts)
        for empty in ([], torch.empty((0, 2), dtype=torch.int32)):
            with pool.waves("small", empty) as waves:
                assert list(waves) == []
        with pool.waves("small", [0, 1, 2, 3]) as waves:
            _ = list(waves)

        before_fast = pool.stats().fast_path_waves
        with pool.waves("small", torch.tensor([[2, -1], [0, 3]], dtype=torch.int32)) as waves:
            fast_list = list(waves)
            assert len(fast_list) == 1
            assert fast_list[0].is_hit_wave
            assert fast_list[0].active_slots_hint == 4
            assert fast_list[0].expert_ids == (0, 1, 2, 3)
            _assert_wave_contents(fast_list[0], experts)
        assert pool.stats().fast_path_waves == before_fast + 1
        # Non-empty tensor of all -1 sentinels still executes fast path with zero CPU readback.
        with pool.waves("small", torch.tensor([-1, -1], dtype=torch.int32)) as waves:
            sentinel_fast = list(waves)
            assert len(sentinel_fast) == 1
            assert sentinel_fast[0].active_slots_hint == 4

        # Meta / non-integer tensors and bad scalar IDs fail cleanly before waves run.
        meta_ids = torch.empty(2, dtype=torch.int32, device="meta")
        with pytest.raises(ValueError, match="cpu or cuda"), pool.waves("small", meta_ids) as _:
            pass
        with pool.waves("small", iter(())) as waves:
            assert list(waves) == []
        with pool.waves("small", [-1, 99]) as waves:
            assert list(waves) == []
        with pytest.raises(TypeError, match="must be an int"), pool.waves("small", ["bad"]) as waves:  # type: ignore[list-item]
            _ = list(waves)

    # Bypass followed by fast-path decode wave promotes all transient slots to resident.
    with (
        Uint8DualArena(4096, device="cpu", alignment=256) as arena2,
        ExpertSlabPool(arena2, schema, max_slots=4) as pool2,
    ):
        pool2.register_group("small", experts)
        with pool2.waves("small", [0, 1, 2, 3], bypass=True, max_wave_slots=4) as waves:
            _ = list(waves)
        assert pool2.stats().transient_slots == 4
        with pool2.waves("small", torch.tensor([[0, 1]], dtype=torch.int32)) as waves:
            assert next(waves).is_hit_wave
        assert pool2.stats().resident_experts == 4
        assert pool2.stats().transient_slots == 0


def test_frontier_relocation_eviction_and_bidirectional_reclaim() -> None:
    schema = SlotSchema({"up": 256, "down": 128}, {"scale": (torch.float32, (4,))})
    experts = [_make_synthetic_expert(i, up_bytes=200, down_bytes=100, scale_len=4) for i in range(5)]
    with (
        Uint8DualArena(2048, device="cpu", slot_bytes=512, alignment=256) as arena,
        ExpertSlabPool(arena, schema, max_slots=4) as pool,
    ):
        assert isinstance(arena.slot_layout, HighSlotLayout)
        pool.register_group("moe", experts)
        with pool.waves("moe", [0, 1, 2, 3]) as waves:
            _ = list(waves)
        for _ in range(4):
            with pool.waves("moe", [1, 2, 3]) as waves:
                _ = list(waves)

        # Active lease on top slot 3 blocks frontier reclaim; freeing it lets E3 relocate into slot 0.
        with pool.waves("moe", [3]) as waves:
            _w = next(waves)
            with pytest.raises(ArenaCapacityError):
                arena.alloc_high_slot()

        assert arena.alloc_high_slot() == 3
        assert pool.stats().relocations >= 1
        assert pool.stats().allocated_slots == 3
        misses_before = pool.stats().misses
        with pool.waves("moe", [3]) as waves:
            w = list(waves)
            assert w[0].is_hit_wave
            _assert_wave_contents(w[0], experts)
        assert pool.stats().misses == misses_before

        # Full reclaim to target_low == 0 evicts every resident slot cleanly (counting evictions).
        evictions_before = pool.stats().evictions
        assert arena.alloc_high_slot() == 2
        assert arena.alloc_high_slot() == 1
        assert arena.alloc_high_slot() == 0
        assert pool.stats().allocated_slots == 0
        assert pool.stats().evictions == evictions_before + 3

        # Bidirectional high_reclaimer: bypass=True does NOT reclaim once len(_slots) > 0; bypass=False does.
        high_reclaims: list[int] = []

        def _high_reclaim(needed_high: int) -> None:
            high_reclaims.append(needed_high)
            for sid in sorted(arena.active_high_slots):
                arena.free_high_slot(sid)
                if arena.high_frontier >= needed_high:
                    break

        arena.set_high_reclaimer(_high_reclaim)
        with pool.waves("moe", [0, 1], bypass=True, max_wave_slots=1) as waves:
            assert [w.expert_ids for w in waves] == [(0,), (1,)]
        assert high_reclaims == [512]  # first miss reclaimed hi0; E1 reused transient slot 0
        with pool.waves("moe", [2, 3], bypass=False) as waves:
            assert [w.expert_ids for w in waves] == [(2, 3)]
        assert high_reclaims == [512, 1024]  # decode packed [2, 3] by reclaiming hi1


def test_nonfull_warmup_bypass_options_and_multigroup_fast_path() -> None:
    schema = SlotSchema({"up": 128, "down": 128}, {"scale": (torch.float32, (4,))})
    experts = [_make_synthetic_expert(i, up_bytes=128, down_bytes=128, scale_len=4) for i in range(8)]

    # Non-full decode grows into unused arena slots instead of clobbering transient or reclaiming high peers.
    high_reclaims: list[int] = []
    with (
        Uint8DualArena(4096, device="cpu", slot_bytes=512, alignment=256) as arena,
        ExpertSlabPool(arena, schema, max_slots=6, default_bypass_wave_slots=3) as pool,
    ):
        arena.set_high_reclaimer(high_reclaims.append)
        pool.register_group("g0", experts)
        for _ in range(4):
            with pool.waves("g0", [0]) as waves:
                _ = list(waves)
        with pool.waves("g0", [1], bypass=True) as waves:
            _ = list(waves)
        assert pool.stats().resident_experts == 1
        assert pool.stats().transient_slots == 1

        with pool.waves("g0", [2, 3], bypass=False) as waves:
            for w in waves:
                _assert_wave_contents(w, experts)
        st = pool.stats()
        assert st.allocated_slots == 4
        assert st.resident_experts == 3
        assert st.transient_slots == 1
        assert high_reclaims == []

        # default_bypass_wave_slots=3 caps bypass=[4, 5, 6, 7] into [3, 1].
        with pool.waves("g0", [4, 5, 6, 7], bypass=True) as waves:
            assert [len(w.expert_ids) for w in waves] == [3, 1]
        # Explicit record_stats overrides + prefetch(bypass=True) (4 was displaced by 7).
        assert pool.prefetch("g0", [4], bypass=True) == 1
        with pool.waves("g0", [7], bypass=True, record_stats=True) as waves:
            _ = list(waves)
        with pool.waves("g0", [7], bypass=True, record_stats=True) as waves:
            assert next(waves).is_hit_wave

    # prefetch(bypass=False) skips cold lower-ID misses and admits hotter higher-ID experts;
    # multi-group total_registered_experts > max_slots gates fast path when recording stats.
    with Uint8DualArena(4096, device="cpu", alignment=256) as a2, ExpertSlabPool(a2, schema, max_slots=4) as p2:
        p2.register_group("g0", experts[:4])
        p2.register_group("g1", experts[4:])  # total=8 > max_slots=4
        for _ in range(3):
            with p2.waves("g0", [0, 1]) as waves:
                _ = list(waves)
        with p2.waves("g0", [2, 3]) as waves:
            _ = list(waves)
        for _ in range(4):
            with p2.waves("g0", [0, 1, 2, 3]) as waves:
                assert next(waves).is_hit_wave
        assert p2.stats().fast_path_waves == 0
        with p2.waves("g0", torch.tensor([0, 1], dtype=torch.int32), record_stats=False) as waves:
            assert next(waves).is_hit_wave
        assert p2.stats().fast_path_waves == 1

        with Uint8DualArena(2048, device="cpu", alignment=256) as a3, ExpertSlabPool(a3, schema, max_slots=2) as p3:
            p3.register_group("moe", experts[:4])
            for _ in range(2):
                with p3.waves("moe", [0, 1]) as waves:
                    _ = list(waves)
            for _ in range(4):
                p3.commit_accepted([("moe", 3)])
            assert p3.prefetch("moe", [2, 3], bypass=False) == 1
            with p3.waves("moe", [3]) as waves:
                assert next(waves).is_hit_wave


def _free_high_slots_until(arena: Uint8DualArena, kv_lock: threading.Lock, needed_high: int) -> None:
    with kv_lock:
        for sid in sorted(arena.active_high_slots):
            arena.free_high_slot(sid)
            if arena.high_frontier >= needed_high:
                break


def _run_kv_churn(
    arena: Uint8DualArena,
    kv_lock: threading.Lock,
    barrier: threading.Barrier,
    stop_kv: threading.Event,
    errors: list[BaseException],
) -> None:
    try:
        barrier.wait()
        while not stop_kv.is_set():
            with kv_lock:
                try:
                    sid = arena.alloc_high_slot()
                    arena.free_high_slot(sid)
                except ArenaCapacityError:
                    pass
            time.sleep(0)
    except BaseException as exc:
        errors.append(exc)


def test_multithreaded_waves_frontier_reclaim_and_close_safety() -> None:
    schema = SlotSchema({"up": 128, "down": 128}, {"scale": (torch.float32, (4,))})
    experts = [_make_synthetic_expert(i, up_bytes=128, down_bytes=128, scale_len=4) for i in range(6)]
    with (
        Uint8DualArena(2048, device="cpu", slot_bytes=512, alignment=256) as arena,
        ExpertSlabPool(arena, schema, max_slots=4) as pool,
    ):
        pool.register_group("moe", experts)
        # Pre-warm all 4 slots ([0] cold, [1..3] hot) so concurrent high alloc triggers real relocations.
        with pool.waves("moe", [0, 1, 2, 3]) as waves:
            for w in waves:
                _assert_wave_contents(w, experts)
        for _ in range(3):
            with pool.waves("moe", [1, 2, 3]) as waves:
                _ = list(waves)

        kv_lock = threading.Lock()
        arena.set_high_reclaimer(lambda needed_high: _free_high_slots_until(arena, kv_lock, needed_high))
        barrier = threading.Barrier(3, timeout=5.0)
        stop_kv = threading.Event()
        errors: list[BaseException] = []

        def _wave_worker(eid_lists: list[list[int]]) -> None:
            try:
                barrier.wait()
                for idx in range(25):
                    ids = eid_lists[idx % len(eid_lists)]
                    with pool.waves("moe", ids, max_wave_slots=1) as waves:
                        for w in waves:
                            _assert_wave_contents(w, experts)
            except BaseException as exc:
                errors.append(exc)

        wave_threads = [
            threading.Thread(target=_wave_worker, args=([[0, 1], [2, 3], [4, 5]],)),
            threading.Thread(target=_wave_worker, args=([[1, 4], [0, 5], [2, 3]],)),
        ]
        kv_thread = threading.Thread(target=_run_kv_churn, args=(arena, kv_lock, barrier, stop_kv, errors))
        for t in (*wave_threads, kv_thread):
            t.start()
        for t in wave_threads:
            t.join(timeout=15.0)
            assert not t.is_alive()
        stop_kv.set()
        kv_thread.join(timeout=5.0)
        assert not kv_thread.is_alive()
        assert errors == []
        assert pool.stats().relocations >= 1


def test_concurrent_close_serialization() -> None:
    schema = SlotSchema({"up": 128, "down": 128}, {"scale": (torch.float32, (4,))})
    experts = [_make_synthetic_expert(i, up_bytes=128, down_bytes=128, scale_len=4) for i in range(4)]
    with (
        Uint8DualArena(2048, device="cpu", slot_bytes=512, alignment=256) as arena,
        ExpertSlabPool(arena, schema, max_slots=4) as pool,
    ):
        pool.register_group("moe", experts)
        with pool.waves("moe", [0, 1]) as waves:
            _ = list(waves)

        close_barrier = threading.Barrier(2, timeout=5.0)
        close_errors: list[BaseException] = []

        def _closer() -> None:
            try:
                close_barrier.wait()
                pool.close()
            except BaseException as exc:
                close_errors.append(exc)

        t1, t2 = threading.Thread(target=_closer), threading.Thread(target=_closer)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        assert not t1.is_alive() and not t2.is_alive()
        assert close_errors == []


def test_mapped_checkpoint_host_inputs_cpu_and_cleanup(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "experts.safetensors"
    up0 = torch.randint(0, 256, (192,), dtype=torch.uint8)
    down0 = torch.randint(0, 256, (128,), dtype=torch.uint8)
    scale0 = torch.randn((4,), dtype=torch.float32)
    _write_safetensors(ckpt_path, {"e0.up": up0, "e0.down": down0, "e0.scale": scale0})

    schema = SlotSchema({"up": 256, "down": 128}, {"scale": (torch.float32, (4,))})
    with MappedCheckpoint(ckpt_path) as ckpt:
        e0_mapped = {
            "up": ckpt.get_tensor("e0.up"),
            "down": ckpt.get_tensor("e0.down"),
            "scale": ckpt.get_tensor("e0.scale"),
        }
        assert not e0_mapped["up"].untyped_storage().resizable()
        with (
            Uint8DualArena(2048, device="cpu", alignment=256) as arena,
            ExpertSlabPool(arena, schema, max_slots=2) as pool,
        ):
            pool.register_group("mapped", [e0_mapped])
            with pool.waves("mapped", [0]) as waves:
                _assert_wave_contents(next(waves), [{"up": up0, "down": down0, "scale": scale0}])


@CUDA
def test_cuda_rejects_device_mismatch_on_cpu_pool() -> None:
    schema = SlotSchema({"up": 128}, {"scale": (torch.float32, (4,))})
    with Uint8DualArena(2048, device="cpu", alignment=256) as arena, DeviceTransferContext("cuda") as cuda_ctx:
        with pytest.raises(ValueError, match="CPU ExpertSlabPool"):
            ExpertSlabPool(arena, schema, max_slots=2, transfer_context=cuda_ctx)
        assert cuda_ctx.borrower_count == 0

    # Borrowed CUDA transfer_context and budget roll back cleanly when side-pool creation fails.
    tiny_budget = VramBudget(16)
    with Uint8DualArena(2048, device="cuda", alignment=256) as cu_arena, DeviceTransferContext("cuda") as cuda_ctx:
        with pytest.raises(VramBudgetExhaustedError):
            ExpertSlabPool(cu_arena, schema, max_slots=2, budget=tiny_budget, transfer_context=cuda_ctx)
        assert cuda_ctx.borrower_count == 0
        assert tiny_budget.reserved_bytes == 0
        assert cu_arena.low_slot_layout is None


@CUDA
def test_cuda_mapped_checkpoint_bypass_and_decode_pinning(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "cuda_experts.safetensors"
    t0 = _make_synthetic_expert(0, up_bytes=200, down_bytes=128, scale_len=4)
    t1 = _make_synthetic_expert(1, up_bytes=256, down_bytes=128, scale_len=4)
    _write_safetensors(
        ckpt_path,
        {
            "e0.up": t0["up"], "e0.down": t0["down"], "e0.scale": t0["scale"],
            "e1.up": t1["up"], "e1.down": t1["down"], "e1.scale": t1["scale"],
        },
    )
    schema = SlotSchema({"up": 256, "down": 128}, {"scale": (torch.float32, (4,))})
    pin_mgr = PinManager(max_pinned_bytes=64 * 1024 * 1024, max_host_bytes=64 * 1024 * 1024)
    try:
        with MappedCheckpoint(ckpt_path) as ckpt:
            experts = [
                {
                    "up": ckpt.get_tensor("e0.up"),
                    "down": ckpt.get_tensor("e0.down"),
                    "scale": ckpt.get_tensor("e0.scale"),
                },
                {
                    "up": ckpt.get_tensor("e1.up"),
                    "down": ckpt.get_tensor("e1.down"),
                    "scale": ckpt.get_tensor("e1.scale"),
                },
            ]
            with (
                DeviceTransferContext("cuda") as ctx,
                Uint8DualArena(4096, device="cuda", alignment=256) as arena,
                ExpertSlabPool(arena, schema, max_slots=2, transfer_context=ctx, pin_manager=pin_mgr) as pool,
            ):
                pool.register_group("mapped", experts)
                out_scale = torch.empty((2, 4), dtype=torch.float32, device="cuda")
                work_stream = torch.cuda.Stream()
                with torch.cuda.stream(work_stream), pool.waves("mapped", [0], bypass=True) as waves:
                    w = next(waves)
                    out_scale[0].copy_(w.side_views["scale"].index_select(0, w.expert_to_slot[0:1])[0])
                assert pin_mgr.stats.registrations == 0
                with torch.cuda.stream(work_stream), pool.waves("mapped", [1], bypass=False) as waves:
                    w = next(waves)
                    out_scale[1].copy_(w.side_views["scale"].index_select(0, w.expert_to_slot[1:2])[0])
                assert pin_mgr.stats.registrations > 0
                work_stream.synchronize()
                torch.testing.assert_close(out_scale[0].cpu(), t0["scale"])
                torch.testing.assert_close(out_scale[1].cpu(), t1["scale"])
            assert pin_mgr.stats.active_leases == 0
    finally:
        pin_mgr.clear()


@contextlib.contextmanager
def _strict_no_cuda_sync() -> Iterator[None]:
    prev_sync_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev_sync_mode)


def _gather_wave_to_gpu(wave: ExpertWave, gpu_scales: torch.Tensor, gpu_up_bytes: torch.Tensor) -> None:
    for eid in wave.expert_ids:
        slot_idx = wave.expert_to_slot[eid : eid + 1]
        gpu_scales[eid].copy_(wave.side_views["scale"].index_select(0, slot_idx)[0])
        gpu_up_bytes[eid].copy_(wave.arena_views["up"].index_select(0, slot_idx)[0])


def _assert_gpu_gathered_experts(
    gpu_scales: torch.Tensor,
    gpu_up_bytes: torch.Tensor,
    experts: list[dict[str, torch.Tensor]],
) -> None:
    for eid, expert in enumerate(experts):
        torch.testing.assert_close(gpu_scales[eid].cpu(), expert["scale"], rtol=0, atol=0)
        up_len = expert["up"].numel()
        torch.testing.assert_close(gpu_up_bytes[eid, :up_len].cpu(), expert["up"], rtol=0, atol=0)


@CUDA
def test_cuda_non_default_stream_prefetch_overlap_and_d2d() -> None:
    device = torch.device("cuda")
    pin_mgr = PinManager(max_pinned_bytes=64 * 1024 * 1024, max_host_bytes=64 * 1024 * 1024)
    budget = VramBudget(16 * 1024 * 1024)
    schema = SlotSchema({"up": 512, "down": 256}, {"scale": (torch.float32, (16,))})
    experts = [
        _make_synthetic_expert(i, up_bytes=400 if i % 2 == 0 else 512, down_bytes=256, scale_len=16)
        for i in range(6)
    ]
    work_stream = torch.cuda.Stream(device=device)
    gpu_scales = torch.empty((6, 16), dtype=torch.float32, device=device)
    gpu_up_bytes = torch.zeros((6, 512), dtype=torch.uint8, device=device)

    try:
        with (
            DeviceTransferContext(device) as ctx,
            Uint8DualArena(3 * 1024, device=device, slot_bytes=768, alignment=256, budget=budget) as arena,
            ExpertSlabPool(arena, schema, max_slots=4, transfer_context=ctx, pin_manager=pin_mgr) as pool,
        ):
            pool.register_group("moe", experts)
            # Prefetch [0], verify overlap + early break commits async pre-staged E1, then run overlap waves.
            assert pool.prefetch("moe", [0], bypass=False) == 1
            assert pin_mgr.stats.registrations > 0
            with (
                torch.cuda.stream(work_stream),
                pool.waves("moe", [0, 1, 2], max_wave_slots=1) as waves,
            ):
                _ = next(waves)
                assert pool.stats().allocated_slots == 2
            assert pool.stats().resident_experts == 2

            with _strict_no_cuda_sync():
                with torch.cuda.stream(work_stream), pool.waves("moe", [0, 1, 2, 3], max_wave_slots=1) as waves:
                    for w in waves:
                        _gather_wave_to_gpu(w, gpu_scales, gpu_up_bytes)

            # Heat expert 3 above expert 0, trigger CUDA D2D relocation on slot 3, and read with zero host sync.
            pool.commit_accepted([("moe", 3), ("moe", 3)])
            assert arena.alloc_high_slot() == 3
            assert pool.stats().relocations >= 1
            with _strict_no_cuda_sync():
                with torch.cuda.stream(work_stream), pool.waves("moe", [3]) as waves:
                    w3 = next(waves)
                    assert w3.is_hit_wave
                    _gather_wave_to_gpu(w3, gpu_scales, gpu_up_bytes)

                # Prefill bypass=True adds 0 new registrations for unseen [4, 5].
                regs_before = pin_mgr.stats.registrations
                with torch.cuda.stream(work_stream), pool.waves("moe", [4, 5], bypass=True, max_wave_slots=1) as waves:
                    for w in waves:
                        _gather_wave_to_gpu(w, gpu_scales, gpu_up_bytes)
            assert pin_mgr.stats.registrations == regs_before

            work_stream.synchronize()
            torch.cuda.synchronize(device)
            _assert_gpu_gathered_experts(gpu_scales, gpu_up_bytes, experts)
            assert pool.stats().allocated_slots == 3
            assert pin_mgr.stats.active_leases == 0
        assert budget.reserved_bytes == 0
    finally:
        pin_mgr.clear()


@CUDA
def test_cuda_zero_sync_all_resident_fast_path() -> None:
    device = torch.device("cuda")
    pin_mgr = PinManager(max_pinned_bytes=64 * 1024 * 1024, max_host_bytes=64 * 1024 * 1024)
    budget = VramBudget(16 * 1024 * 1024)
    schema = SlotSchema({"up": 512, "down": 256}, {"scale": (torch.float32, (16,))})
    experts = [_make_synthetic_expert(i, up_bytes=512, down_bytes=256, scale_len=16) for i in range(4)]
    work_stream, other_stream = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    gpu_scales = torch.empty((4, 16), dtype=torch.float32, device=device)
    gpu_up_bytes = torch.zeros((4, 512), dtype=torch.uint8, device=device)
    other_scales = torch.empty((4, 16), dtype=torch.float32, device=device)
    other_up_bytes = torch.zeros((4, 512), dtype=torch.uint8, device=device)
    gpu_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32, device=device)

    try:
        with (
            DeviceTransferContext(device) as ctx,
            Uint8DualArena(4 * 1024, device=device, slot_bytes=768, alignment=256, budget=budget) as arena,
            ExpertSlabPool(arena, schema, max_slots=4, transfer_context=ctx, pin_manager=pin_mgr) as pool,
        ):
            pool.register_group("moe", experts)
            with pool.waves("moe", [0, 1, 2, 3]) as waves:
                _ = list(waves)

            with _strict_no_cuda_sync():
                for stream, s_buf, u_buf in (
                    (work_stream, gpu_scales, gpu_up_bytes),
                    (other_stream, other_scales, other_up_bytes),
                    (work_stream, gpu_scales, gpu_up_bytes),
                ):
                    with torch.cuda.stream(stream), pool.waves("moe", gpu_ids) as waves:
                        for w in waves:
                            assert w.is_hit_wave
                            _gather_wave_to_gpu(w, s_buf, u_buf)

            torch.cuda.synchronize(device)
            assert pool.stats().fast_path_waves == 3
            _assert_gpu_gathered_experts(gpu_scales, gpu_up_bytes, experts)
            _assert_gpu_gathered_experts(other_scales, other_up_bytes, experts)
    finally:
        pin_mgr.clear()


@TRELLIS_MOE_CUDA
def test_cuda_variable_byte_trellis_moe_ffn_end_to_end() -> None:
    from piper_kernels.moe import MoEWeights, moe_ffn, prepare_moe_input

    torch.manual_seed(42)
    device = torch.device("cuda")
    hidden_size, intermediate_size, group_size, block_size = 128, 64, 64, 16
    expert_dicts = _build_synthetic_trellis_experts(
        hidden_size=hidden_size, intermediate_size=intermediate_size, block_size=block_size
    )
    max_up_bytes = max(d["up_gate_qdata"].nbytes for d in expert_dicts)
    max_down_bytes = max(d["down_qdata"].nbytes for d in expert_dicts)
    assert expert_dicts[0]["up_gate_qdata"].nbytes < max_up_bytes
    assert expert_dicts[0]["down_qdata"].nbytes < max_down_bytes

    def _pad_stack(key: str, max_bytes: int) -> torch.Tensor:
        buf = torch.zeros((len(expert_dicts), max_bytes), dtype=torch.uint8, device=device)
        for i, item in enumerate(expert_dicts):
            src = item[key]
            buf[i, : src.numel()].copy_(src.to(device))
        return buf

    full_weights = MoEWeights.from_quantized(
        up_gate_qdata=_pad_stack("up_gate_qdata", max_up_bytes),
        up_gate_scale=torch.stack([d["up_gate_scale"] for d in expert_dicts]).to(device),
        down_qdata=_pad_stack("down_qdata", max_down_bytes),
        down_scale=torch.stack([d["down_scale"] for d in expert_dicts]).to(device),
        logical_dtype=torch.bfloat16,
        group_size=group_size,
        up_gate_block_scales=torch.stack([d["up_gate_block_scales"] for d in expert_dicts]).to(device),
        down_block_scales=torch.stack([d["down_block_scales"] for d in expert_dicts]).to(device),
        up_gate_row_widths=torch.stack([d["up_gate_row_widths"] for d in expert_dicts]).to(device),
        down_row_widths=torch.stack([d["down_row_widths"] for d in expert_dicts]).to(device),
        up_gate_block_size=block_size,
        down_block_size=block_size,
    )

    x = torch.randn((3, hidden_size), dtype=torch.bfloat16, device=device)
    topk_ids = torch.tensor([[0, 2], [1, 3], [0, 3]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3]], dtype=torch.float32, device=device)
    expected = moe_ffn(x, full_weights, topk_ids, topk_weights)

    schema = SlotSchema(
        {"up_gate_qdata": max_up_bytes, "down_qdata": max_down_bytes},
        {
            "up_gate_block_scales": (torch.float8_e4m3fn, tuple(expert_dicts[0]["up_gate_block_scales"].shape)),
            "down_block_scales": (torch.float8_e4m3fn, tuple(expert_dicts[0]["down_block_scales"].shape)),
            "up_gate_scale": (torch.float32, tuple(expert_dicts[0]["up_gate_scale"].shape)),
            "down_scale": (torch.float32, tuple(expert_dicts[0]["down_scale"].shape)),
            "up_gate_row_widths": (torch.uint8, tuple(expert_dicts[0]["up_gate_row_widths"].shape)),
            "down_row_widths": (torch.uint8, tuple(expert_dicts[0]["down_row_widths"].shape)),
        },
    )
    pin_mgr = PinManager(max_pinned_bytes=64 * 1024 * 1024, max_host_bytes=64 * 1024 * 1024)
    try:
        with (
            Uint8DualArena(4 * schema.drawer_bytes, device=device, alignment=256) as arena,
            ExpertSlabPool(arena, schema, max_slots=2, pin_manager=pin_mgr) as pool,
        ):
            pool.register_group("layer0", expert_dicts)
            assert pool.prefetch("layer0", [0, 1]) == 2

            streamed_out = torch.empty_like(x)
            wave_log: list[tuple[int, bool, tuple[int, ...]]] = []
            with pool.waves("layer0", topk_ids) as waves:
                prepared: Any = None
                for wave in waves:
                    wave_log.append((wave.index, wave.is_hit_wave, wave.expert_ids))
                    slot_weights = MoEWeights.from_quantized(
                        up_gate_qdata=wave.arena_bytes,
                        up_gate_scale=wave.side_views["up_gate_scale"],
                        down_qdata=wave.arena_bytes,
                        down_scale=wave.side_views["down_scale"],
                        logical_dtype=torch.bfloat16,
                        group_size=group_size,
                        up_gate_block_scales=wave.side_views["up_gate_block_scales"],
                        down_block_scales=wave.side_views["down_block_scales"],
                        up_gate_row_widths=wave.side_views["up_gate_row_widths"],
                        down_row_widths=wave.side_views["down_row_widths"],
                        up_gate_slot_offsets=wave.slot_offsets["up_gate_qdata"],
                        down_slot_offsets=wave.slot_offsets["down_qdata"],
                        up_gate_block_size=block_size,
                        down_block_size=block_size,
                    )
                    if prepared is None:
                        prepared = prepare_moe_input(x, slot_weights)
                    moe_ffn(
                        x, slot_weights, topk_ids, topk_weights,
                        expert_to_slot=wave.expert_to_slot, out=streamed_out,
                        accumulate=wave.index > 0, prepared_input=prepared,
                        active_slots_hint=wave.active_slots_hint,
                    )
        torch.cuda.synchronize(device)
        assert wave_log == [(0, True, (0, 1)), (1, False, (2, 3))]
        torch.testing.assert_close(streamed_out, expected, rtol=3e-2, atol=3e-2)
        wave_split_out = torch.empty_like(x)
        prep = prepare_moe_input(x, full_weights)
        moe_ffn(
            x,
            full_weights,
            topk_ids,
            topk_weights,
            expert_to_slot=torch.tensor([0, 1, -1, -1], dtype=torch.int32, device=device),
            out=wave_split_out,
            accumulate=False,
            prepared_input=prep,
            active_slots_hint=2,
        )
        moe_ffn(
            x,
            full_weights,
            topk_ids,
            topk_weights,
            expert_to_slot=torch.tensor([-1, -1, 2, 3], dtype=torch.int32, device=device),
            out=wave_split_out,
            accumulate=True,
            prepared_input=prep,
            active_slots_hint=2,
        )
        torch.cuda.synchronize(device)
        torch.testing.assert_close(streamed_out, wave_split_out, rtol=0, atol=0)
    finally:
        pin_mgr.clear()


@CUDA
def test_cuda_expert_slab_pool_shares_arena_under_torch_compile_fullgraph(
    request: pytest.FixtureRequest,
) -> None:
    torch.compiler.reset()
    request.addfinalizer(torch.compiler.reset)
    device = torch.device("cuda")
    pin_mgr = PinManager(max_pinned_bytes=64 * 1024 * 1024, max_host_bytes=64 * 1024 * 1024)
    schema = SlotSchema({"up": 256, "down": 128}, {"scale": (torch.float32, (8,))})
    try:
        with (
            Uint8DualArena(4096, device=device, slot_bytes=512, alignment=256) as arena,
            ExpertSlabPool(arena, schema, max_slots=4, pin_manager=pin_mgr) as pool,
        ):
            experts = [_make_synthetic_expert(i, up_bytes=256, down_bytes=128) for i in range(2)]
            pool.register_group("moe", experts)
            kv_pool = arena.high_pool_view((512,))
            kv_slot = arena.alloc_high_slot()
            kv_pool[kv_slot].zero_()

            with pool.waves("moe", [0, 1]) as waves:
                w = next(waves)
                s0 = int(w.expert_to_slot[0].item())

                def compiled_step(up_pool: torch.Tensor, scales: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
                    kv[kv_slot, :16].fill_(7)
                    return up_pool[s0].to(torch.float32).sum() + scales[s0].sum()

                fn = torch.compile(compiled_step, fullgraph=True)
                out1 = fn(w.arena_views["up"], w.side_views["scale"], kv_pool)
                out2 = fn(w.arena_views["up"], w.side_views["scale"], kv_pool)
                torch.cuda.synchronize(device)

                expected = experts[0]["up"].to(device, torch.float32).sum() + experts[0]["scale"].to(device).sum()
                torch.testing.assert_close(out1, expected)
                torch.testing.assert_close(out2, expected)
                assert int(kv_pool[kv_slot, :16].min().item()) == 7
    finally:
        pin_mgr.clear()


@CUDA
def test_cuda_wave_release_failure_retries_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    pin_mgr = PinManager(max_pinned_bytes=64 * 1024 * 1024, max_host_bytes=64 * 1024 * 1024)
    budget = VramBudget(16 * 1024 * 1024)
    schema = SlotSchema({"up": 256, "down": 128}, {"scale": (torch.float32, (8,))})
    experts = [_make_synthetic_expert(i, up_bytes=256, down_bytes=128) for i in range(2)]
    orig_release = StreamLease.release
    fail_mode = "once"
    calls = 0

    def _flaky_release(self_lease: StreamLease, stream: torch.cuda.Stream | None = None) -> None:
        nonlocal calls
        calls += 1
        if fail_mode == "always" or (fail_mode == "once" and calls == 1):
            raise RuntimeError("transient release failure")
        orig_release(self_lease, stream)

    try:
        with (
            Uint8DualArena(1024, device=device, slot_bytes=512, alignment=256, budget=budget) as arena,
            ExpertSlabPool(arena, schema, max_slots=1, pin_manager=pin_mgr) as pool,
        ):
            pool.register_group("moe", experts)
            monkeypatch.setattr(StreamLease, "release", _flaky_release)

            # Transient error during loop-end release must recover on the generator finally retry.
            with pytest.raises(RuntimeError, match="transient release failure"):
                with pool.waves("moe", [0]) as waves:
                    _ = list(waves)
            assert pool._slots[0].active_readers == 0
            with pool.waves("moe", [1]) as waves:
                _ = list(waves)

            # Escaping error hands ownership to the pool, protects slot 0, and self-heals when cleared.
            fail_mode = "always"
            with pytest.raises(RuntimeError, match="transient release failure"):
                with pool.waves("moe", [1]) as waves:
                    _ = next(waves)
            assert pool._slots[0].active_readers > 0
            with pytest.raises(ArenaCapacityError):
                with pool.waves("moe", [0]) as waves:
                    _ = list(waves)

            fail_mode = "none"
            _ = pool.stats()
            assert pool._slots[0].active_readers == 0
            with pool.waves("moe", [0]) as waves:
                _ = list(waves)
        assert budget.reserved_bytes == 0
        assert pin_mgr.stats.active_leases == 0
    finally:
        pin_mgr.clear()


@CUDA
def test_cuda_persistent_release_failure_closes_via_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    pin_mgr = PinManager(max_pinned_bytes=64 * 1024 * 1024, max_host_bytes=64 * 1024 * 1024)
    budget = VramBudget(16 * 1024 * 1024)
    schema = SlotSchema({"up": 256, "down": 128}, {"scale": (torch.float32, (8,))})
    experts = [_make_synthetic_expert(i, up_bytes=256, down_bytes=128) for i in range(2)]
    alt_stream = torch.cuda.Stream(device=device)
    orig_sync_reset = StreamLease.synchronize_and_reset
    sync_calls = 0

    def _always_fail_release(_self_lease: StreamLease, _stream: torch.cuda.Stream | None = None) -> None:
        raise RuntimeError("permanent release failure")

    def _fail_first_sync(self_lease: StreamLease) -> None:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            raise RuntimeError("sync failed")
        orig_sync_reset(self_lease)

    try:
        with Uint8DualArena(1024, device=device, slot_bytes=512, alignment=256, budget=budget) as arena:
            pool = ExpertSlabPool(arena, schema, max_slots=2, pin_manager=pin_mgr)
            pool.register_group("moe", experts)
            monkeypatch.setattr(StreamLease, "release", _always_fail_release)
            orig_stream = torch.cuda.current_stream(device)
            try:
                with pytest.raises(RuntimeError, match="permanent release failure"):
                    with pool.waves("moe", [0]) as waves:
                        _ = next(waves)
                        torch.cuda.set_stream(alt_stream)
            finally:
                torch.cuda.set_stream(orig_stream)

            assert pool._slots[0].active_readers > 0
            with monkeypatch.context() as patch:
                patch.setattr(StreamLease, "synchronize_and_reset", _fail_first_sync)
                with pytest.raises(RuntimeError, match="sync failed"):
                    pool.close()
            assert pool._closing
            assert not pool._closed
            assert not pool.is_open
            pool.close()
            assert pool._closed
            assert not pool.is_open
            assert arena.low_frontier == 0
        assert budget.reserved_bytes == 0
        assert pin_mgr.stats.active_leases == 0
    finally:
        pin_mgr.clear()


@CUDA
def test_cuda_record_stream_failure_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    pin_mgr = PinManager(max_pinned_bytes=64 * 1024 * 1024, max_host_bytes=64 * 1024 * 1024)
    budget = VramBudget(16 * 1024 * 1024)
    schema = SlotSchema({"up": 256, "down": 128}, {"scale": (torch.float32, (8,))})
    experts = [_make_synthetic_expert(i, up_bytes=256, down_bytes=128) for i in range(2)]
    alt_stream = torch.cuda.Stream(device=device)
    orig_record_stream = StreamLease.record_stream
    fail_record = False

    def _maybe_fail_record(self_lease: StreamLease, stream: torch.cuda.Stream) -> None:
        if fail_record and stream == alt_stream:
            raise RuntimeError("record_stream failure")
        orig_record_stream(self_lease, stream)

    try:
        with (
            Uint8DualArena(1024, device=device, slot_bytes=512, alignment=256, budget=budget) as arena,
            ExpertSlabPool(arena, schema, max_slots=2, pin_manager=pin_mgr) as pool,
        ):
            pool.register_group("moe", experts)
            with pool.waves("moe", [0, 1]) as waves:
                _ = list(waves)
            monkeypatch.setattr(StreamLease, "record_stream", _maybe_fail_record)

            # record_stream failure on stream switch must retain slot protection rather than releasing unsafely.
            orig_stream = torch.cuda.current_stream(device)
            fail_record = True
            try:
                with pytest.raises(RuntimeError, match="record_stream failure"):
                    with pool.waves("moe", [0]) as waves:
                        _ = next(waves)
                        torch.cuda.set_stream(alt_stream)
            finally:
                torch.cuda.set_stream(orig_stream)
                fail_record = False
            assert pool._slots[0].active_readers > 0
            assert pool._slots[0].lease is not None and pool._slots[0].lease.is_acquired
            _ = pool.stats()
            assert pool._slots[0].active_readers == 0
        assert budget.reserved_bytes == 0
    finally:
        pin_mgr.clear()


@CUDA
def test_cuda_d2d_release_failure_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    pin_mgr = PinManager(max_pinned_bytes=64 * 1024 * 1024, max_host_bytes=64 * 1024 * 1024)
    budget = VramBudget(16 * 1024 * 1024)
    schema = SlotSchema({"up": 256, "down": 128}, {"scale": (torch.float32, (8,))})
    experts = [_make_synthetic_expert(i, up_bytes=256, down_bytes=128) for i in range(4)]
    orig_release = StreamLease.release
    fail_d2d = False

    def _maybe_fail_release(self_lease: StreamLease, stream: torch.cuda.Stream | None = None) -> None:
        if fail_d2d:
            raise RuntimeError("d2d release failure")
        orig_release(self_lease, stream)

    try:
        with (
            Uint8DualArena(2048, device=device, slot_bytes=512, alignment=256, budget=budget) as arena,
            ExpertSlabPool(arena, schema, max_slots=4, pin_manager=pin_mgr) as pool,
        ):
            pool.register_group("moe", experts)
            with pool.waves("moe", [0, 1, 2, 3]) as waves:
                _ = list(waves)
            monkeypatch.setattr(StreamLease, "release", _maybe_fail_release)

            # Sibling _queue_slot_d2d release failure keeps src slot protected and recovers cleanly on reap.
            pool.commit_accepted([("moe", 3), ("moe", 3)])
            fail_d2d = True
            with pytest.raises(RuntimeError, match="d2d release failure"):
                arena.alloc_high_slot()
            src_slot = pool._slots[3]
            assert src_slot.active_readers > 0
            assert src_slot.lease is not None and src_slot.lease.is_acquired
            fail_d2d = False
            _ = pool.stats()
            assert src_slot.active_readers == 0
            assert not src_slot.lease.is_acquired
        assert budget.reserved_bytes == 0
    finally:
        pin_mgr.clear()
