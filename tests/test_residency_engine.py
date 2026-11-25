"""Tests for Layer 1 ElasticVramPool, Layer 2 ResidencyEngine, and auto_offload_paths."""

from __future__ import annotations

import concurrent.futures
import copy
import random
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, cast

import pytest
import torch
from torch import nn

from piper_offload import (
    Adapter,
    BlockCompileConfig,
    ModelCache,
    ModelOffloader,
    ModelSpec,
    ModelStore,
    OffloadConfig,
    TrellisParameter,
    auto_offload_paths,
)
from piper_offload.host_module import HostModuleLoadPlan, HostModuleStore, HostModuleTarget
from piper_offload.host_param import HostParam
from piper_offload.offload_config import _estimate_tensor_bytes
from piper_offload.residency_engine import _ExecutionPredictor, _wait_for_prefetch
from piper_offload.target_lease import CudaTargetLease
from piper_offload.vram_pool import (
    ElasticVramPool,
    UnitSignature,
    VramSlot,
    estimate_plan_gpu_bytes,
)
from tests._gguf_helpers import GGUFParameter, packed_q4_0
from tests.conftest import (
    Interrupt,
    activated_model,
    offloaded_units,
    residency_engine,
    resident_instance,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _install_delayed_lookahead_hooks(
    monkeypatch: pytest.MonkeyPatch,
    allow_worker_finish: threading.Event,
) -> tuple[threading.Event, list[bool], list[bool]]:
    worker_started = threading.Event()
    stage_finished = [False]
    close_saw_finished: list[bool] = []
    original_stage = CudaTargetLease.stage
    original_close = CudaTargetLease.close
    caller_tid = threading.get_ident()

    def delayed_worker_stage(
        lease: CudaTargetLease,
        plan: Any,
        stream: torch.cuda.Stream,
        *,
        non_blocking: bool = True,
    ) -> None:
        if threading.get_ident() != caller_tid:
            worker_started.set()
            allow_worker_finish.wait(timeout=5)
            try:
                original_stage(lease, plan, stream, non_blocking=non_blocking)
            finally:
                stage_finished[0] = True
            return
        original_stage(lease, plan, stream, non_blocking=non_blocking)

    def recording_close(lease: CudaTargetLease) -> None:
        close_saw_finished.append(stage_finished[0])
        original_close(lease)

    monkeypatch.setattr(CudaTargetLease, "stage", delayed_worker_stage)
    monkeypatch.setattr(CudaTargetLease, "close", recording_close)
    return worker_started, stage_finished, close_saw_finished


def _make_seq_blocks_model(dim: int = 32) -> nn.Module:
    class SeqBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, dim, bias=False))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.net(x))

    class SeqBlocksModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([SeqBlock() for _ in range(3)])
            self.requires_grad_(False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for blk in self.blocks:
                x = blk(x)
            return x

    return SeqBlocksModel()


def _make_uniform_model(num_blocks: int = 4, dim: int = 64) -> nn.Module:
    class UniformModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [nn.Linear(dim, dim, bias=False) for _ in range(num_blocks)]
            )
            self.requires_grad_(False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = torch.relu(block(x))
            return x

    return UniformModel()


# ---------------------------------------------------------------------------
# Predictor unit tests
# ---------------------------------------------------------------------------


class TestExecutionPredictor:
    def test_sequential_forward_and_wraparound(self) -> None:
        pred = _ExecutionPredictor(4)
        for idx in (0, 1, 2, 3):
            pred.record(idx)
        assert pred.next_after(3) == 0

        pred.record(0)
        assert pred.next_after(0) == 1
        pred.record(1)
        assert pred.next_after(1) == 2

    def test_reverse_step_zero_stride_fallback(self) -> None:
        pred = _ExecutionPredictor(4)
        pred.record(3)
        assert pred.next_after(3) == 0
        pred.record(2)
        assert pred.next_after(2) == 1
        pred.record(1)
        assert pred.next_after(1) == 0
        pred.record(0)
        assert pred.next_after(0) == 3

    def test_suffix_pattern_matching_up_to_k4(self) -> None:
        # Sparse non-contiguous cycle + repeated looped pattern.
        pred = _ExecutionPredictor(6)
        trace = [0, 2, 4, 1, 3, 0, 2]
        for idx in trace:
            pred.record(idx)
        assert pred.next_after(2) == 4
        pred.record(4)
        assert pred.next_after(4) == 1

        looped = _ExecutionPredictor(3)
        for idx in (0, 0, 1, 1, 2, 2, 0, 0):
            looped.record(idx)
        # k<=4 suffix match predicts repeating transitions or progression.
        assert looped.next_after(0) == 1

    def test_distance_horizon_ranks_upcoming_units(self) -> None:
        pred = _ExecutionPredictor(4)
        for idx in (0, 1, 2, 3, 0):
            pred.record(idx)
        d1 = pred.distance(1)
        d2 = pred.distance(2)
        d3 = pred.distance(3)
        assert d1 < d2 < d3

    def test_indexed_predictor_matches_reference_and_reuses_forecast_in_steady_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def ref_predict_next(history: list[int], tail: list[int], num_units: int) -> int:
            hist_len = len(history)
            max_k = min(4, len(tail), hist_len - 1)
            for k in range(max_k, 0, -1):
                pattern = tail[-k:]
                for start in range(hist_len - 1 - k, -1, -1):
                    if history[start : start + k] == pattern:
                        return history[start + k]
            curr = tail[-1]
            prev = tail[-2] if len(tail) >= 2 else None
            if prev is not None and prev - curr == 1:
                return (curr - 1) % num_units
            return (curr + 1) % num_units

        def ref_distances(history: list[int], num_units: int, max_hist: int) -> dict[int, int]:
            horizon = max(2 * num_units, len(history), 32)
            window = list(history) if history else [num_units - 1]
            distances: dict[int, int] = {}
            for step in range(1, horizon + 1):
                nxt = ref_predict_next(history, window, num_units)
                if nxt not in distances:
                    distances[nxt] = step
                window.append(nxt)
                if len(window) > max_hist:
                    del window[: len(window) - max_hist]
            return distances

        rng = random.Random(42)
        cases: list[tuple[int, list[int]]] = [
            (6, list(range(6)) * 20),
            (6, list(reversed(range(6))) * 20),
            (10, [9, 8] + [0, 1, 2] * 100),  # > max_history (256) with evicted acyclic prefix
            (5, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 30),
            (6, list(range(6)) * 45 + list(reversed(range(6))) * 20),
            (5, [rng.randrange(5) for _ in range(300)]),
        ]
        for num_units, trace in cases:
            pred = _ExecutionPredictor(num_units)
            ref_hist: list[int] = []
            max_hist = max(256, 8 * num_units)
            for idx in range(num_units):
                expected_nxt = ref_predict_next(ref_hist, [idx], num_units)
                assert pred.next_after(idx) == (None if expected_nxt == idx else expected_nxt)
            for step, unit_idx in enumerate(trace):
                pred.record(unit_idx)
                ref_hist.append(unit_idx)
                if len(ref_hist) > max_hist:
                    del ref_hist[: len(ref_hist) - max_hist]
                expected_nxt = ref_predict_next(
                    ref_hist,
                    ref_hist if ref_hist[-1] == unit_idx else [unit_idx],
                    num_units,
                )
                assert pred.next_after(unit_idx) == (None if expected_nxt == unit_idx else expected_nxt), (
                    f"next_after mismatch at step {step}"
                )
                # Exercise both cache priming and steady-state shifts across all units.
                if step % 3 == 0 or step > max_hist - 5:
                    exp_dists = ref_distances(ref_hist, num_units, max_hist)
                    for q in range(num_units):
                        exp_d = float("inf") if q not in exp_dists else float(exp_dists[q])
                        assert pred.distance(q) == exp_d, f"distance mismatch at step {step}, q={q}"

        # Steady-state sequential 128-unit pass (both before and after history eviction)
        # must shift the cached forecast without invalidating or rebuilding it.
        large = _ExecutionPredictor(128)
        for _ in range(10):  # 1280 steps > max_history (1024)
            for idx in range(128):
                large.record(idx)
        _ = large.distance(0)
        invalidations = 0
        orig_invalidate = large._invalidate_forecast

        def count_invalidate() -> None:
            nonlocal invalidations
            invalidations += 1
            orig_invalidate()

        monkeypatch.setattr(large, "_invalidate_forecast", count_invalidate)
        for idx in range(128):
            large.record(idx)
            assert large.next_after(idx) == (idx + 1) % 128
            assert large.distance((idx + 2) % 128) == 2.0
        assert invalidations == 0
        assert large._forecast is not None


# ---------------------------------------------------------------------------
# Layer 1 ElasticVramPool CPU unit tests
# ---------------------------------------------------------------------------


def _fake_pool_helpers(monkeypatch: pytest.MonkeyPatch, target_bytes_fn: Any) -> dict[int, int]:
    alloc_count = {"n": 0}

    class FakeLease:
        def __init__(self, nbytes: int) -> None:
            self.nbytes = nbytes
            self.is_staged = False
            self.closed = False
            self.close_failures = 0
            self.target = cast(HostModuleTarget, object())

        def stage(self, *_args: object, **_kwargs: object) -> None:
            self.is_staged = True

        def close(self) -> None:
            if self.close_failures:
                self.close_failures -= 1
                raise Interrupt("injected lease close failure")
            self.closed = True

    def fake_allocate(
        plan: Any,
        _device: torch.device,
        *,
        allocation_stream: object | None = None,
    ) -> Any:
        del allocation_stream
        alloc_count["n"] += 1
        return FakeLease(plan.nbytes)

    monkeypatch.setattr(CudaTargetLease, "allocate", fake_allocate)
    monkeypatch.setattr("piper_offload.vram_pool.estimate_plan_gpu_bytes", lambda plan: plan.nbytes)
    monkeypatch.setattr("piper_offload.vram_pool.target_gpu_bytes", target_bytes_fn)
    monkeypatch.setattr("piper_offload.vram_pool.plan_target_signature", lambda plan: plan.sig)
    return alloc_count


class _FakePlan:
    def __init__(self, nbytes: int, sig: object) -> None:
        self.nbytes = nbytes
        self.sig = cast(UnitSignature, sig)


class TestElasticVramPool:
    def test_budget_smaller_than_unit_and_equal_unit_recycle(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        allocs = _fake_pool_helpers(monkeypatch, lambda _target: 0)

        class SizedLease:
            def __init__(self, nbytes: int) -> None:
                self.nbytes = nbytes
                self.is_staged = True
                self.closed = False
                self.target = cast(HostModuleTarget, nbytes)

            def close(self) -> None:
                self.closed = True

        monkeypatch.setattr(
            CudaTargetLease,
            "allocate",
            lambda plan, _d, *, allocation_stream=None: (
                allocs.__setitem__("n", allocs["n"] + 1) or SizedLease(plan.nbytes)
            ),
        )
        monkeypatch.setattr("piper_offload.vram_pool.target_gpu_bytes", lambda target: int(cast(Any, target)))

        stream = cast(torch.cuda.Stream, object())
        # 0 < budget < unit_bytes: lookahead declines, sync runs, return_slot trims.
        tiny_pool = ElasticVramPool(torch.device("cuda"), vram_budget_bytes=64)
        p0 = cast(HostModuleLoadPlan, _FakePlan(128, ("u", 128)))
        p1 = cast(HostModuleLoadPlan, _FakePlan(128, ("u", 128)))
        assert tiny_pool.checkout(0, p0, stream, distance_fn=lambda _: 1.0, is_lookahead=True) == (None, False)
        slot, hit = tiny_pool.checkout(0, p0, stream, distance_fn=lambda _: 1.0, is_lookahead=False)
        assert not hit and slot is not None and slot.nbytes == 128
        tiny_pool.return_slot(slot, staged_unit_idx=0, distance_fn=lambda _: 1.0)
        assert tiny_pool.live_slot_count == 0

        # budget == unit_bytes: lookahead declines without eviction, same-sig slot recycles.
        exact_pool = ElasticVramPool(torch.device("cuda"), vram_budget_bytes=128)
        allocs["n"] = 0
        s0, _ = exact_pool.checkout(0, p0, stream, distance_fn=lambda _: 1.0, is_lookahead=False)
        assert s0 is not None
        assert exact_pool.checkout(1, p1, stream, distance_fn=lambda _: 1.0, is_lookahead=True) == (None, False)
        exact_pool.return_slot(s0, staged_unit_idx=0, distance_fn=lambda _: 5.0)
        s1, hit1 = exact_pool.checkout(1, p1, stream, distance_fn=lambda _: 1.0, is_lookahead=False)
        assert not hit1 and s1 is s0 and allocs["n"] == 1
        exact_pool.return_slot(s1, staged_unit_idx=1, distance_fn=lambda _: 1.0)
        exact_pool.close_all()

    def test_lookahead_bailout_preserves_idle_cache_and_evicts_multi_victims_by_priority(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_pool_helpers(monkeypatch, lambda _t: 0)

        class SizedLease:
            def __init__(self, nbytes: int) -> None:
                self.is_staged = True
                self.closed = False
                self.target = cast(HostModuleTarget, nbytes)

            def close(self) -> None:
                self.closed = True

        monkeypatch.setattr(
            CudaTargetLease,
            "allocate",
            lambda plan, _d, *, allocation_stream=None: SizedLease(plan.nbytes),
        )
        monkeypatch.setattr("piper_offload.vram_pool.target_gpu_bytes", lambda target: int(cast(Any, target)))

        stream = cast(torch.cuda.Stream, object())
        pool = ElasticVramPool(torch.device("cuda"), vram_budget_bytes=256)
        small_a = cast(HostModuleLoadPlan, _FakePlan(64, ("s", 64)))
        small_b = cast(HostModuleLoadPlan, _FakePlan(64, ("b", 64)))
        med = cast(HostModuleLoadPlan, _FakePlan(128, ("m", 128)))
        big = cast(HostModuleLoadPlan, _FakePlan(192, ("l", 192)))

        sa, _ = pool.checkout(0, small_a, stream, distance_fn=lambda _: 1.0, is_lookahead=False)
        assert sa is not None
        pool.return_slot(sa, staged_unit_idx=0, distance_fn=lambda _: 1.0)
        sb, _ = pool.checkout(1, small_b, stream, distance_fn=lambda _: 1.0, is_lookahead=False)
        assert sb is not None
        pool.return_slot(sb, staged_unit_idx=1, distance_fn=lambda _: 1.0)
        sm, _ = pool.checkout(2, med, stream, distance_fn=lambda _: 1.0, is_lookahead=False)
        assert sm is not None

        # checked_out=128 + lookahead=192 > 256: must bail out without evicting unit 0 or 1.
        assert pool.checkout(3, big, stream, distance_fn=lambda _: 1.0, is_lookahead=True) == (None, False)
        assert pool.cached_unit_indices() == {0, 1}

        # Returning med and asking synchronously for big (192) evicts both 64B slots or med.
        # Unstaged slots are picked ahead of staged slots; tie distance uses LRU (-seq).
        pool.return_slot(sm, staged_unit_idx=2, distance_fn=lambda _: 1.0)
        cast(Any, sa.lease).is_staged = False
        distances = {0: 1.0, 1: 2.0, 2: 2.0}
        # Victim order: sa first (unstaged despite dist=1.0), then sb (older seq than sm at dist=2.0).
        sbig, _ = pool.checkout(3, big, stream, distance_fn=lambda i: distances[i], is_lookahead=False)
        assert sbig is not None
        assert pool.cached_unit_indices() == set()
        assert cast(Any, sa.lease).closed
        assert cast(Any, sb.lease).closed
        pool.return_slot(sbig, staged_unit_idx=3, distance_fn=lambda _: 1.0)
        pool.close_all()

    def test_cache_hit_trim_failure_does_not_orphan_and_close_retries(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_pool_helpers(monkeypatch, lambda _t: 0)

        class RetryLease:
            def __init__(self, fail_close: bool = False) -> None:
                self.is_staged = True
                self.fail_close = fail_close
                self.close_calls = 0
                self.target = cast(HostModuleTarget, 64)

            def close(self) -> None:
                self.close_calls += 1
                if self.fail_close:
                    self.fail_close = False
                    raise Interrupt("close failed once")

        pool = ElasticVramPool(torch.device("cuda"), vram_budget_bytes=128)
        stream = cast(torch.cuda.Stream, object())
        sig = cast(UnitSignature, ("u", 64))
        l0, l1, l2 = RetryLease(), RetryLease(fail_close=True), RetryLease()
        pool._slots = [
            VramSlot(lease=cast(CudaTargetLease, l0), signature=sig, nbytes=64, staged_unit_idx=0, last_used_seq=1),
            VramSlot(lease=cast(CudaTargetLease, l1), signature=sig, nbytes=64, staged_unit_idx=1, last_used_seq=2),
            VramSlot(lease=cast(CudaTargetLease, l2), signature=sig, nbytes=64, staged_unit_idx=2, last_used_seq=3),
        ]

        # Excess total (192 > 128): hit on slot 0 trims excess slot 1 which raises.
        # Slot 0 must NOT stay checked_out=True!
        with pytest.raises(Interrupt, match="close failed once"):
            pool.checkout(
                0,
                cast(HostModuleLoadPlan, _FakePlan(64, sig)),
                stream,
                distance_fn=lambda i: {0: 1.0, 1: 10.0, 2: 2.0}[i],
                is_lookahead=False,
            )
        assert not pool._slots[0].checked_out

        # close_idle() retry with another middle-slot failure still closes remaining slots cleanly.
        l1.fail_close = True
        with pytest.raises(Interrupt, match="close failed once"):
            pool.close_idle()
        assert pool.live_slot_count == 1
        assert pool._slots[0].lease is l1
        pool.close_all()
        assert pool.live_slot_count == 0
        assert l1.close_calls == 3

    def test_pick_victim_skips_distance_fn_for_single_or_unstaged_candidates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_pool_helpers(monkeypatch, lambda target: int(cast(Any, target)))

        class StagedLease:
            def __init__(self, nbytes: int, *, is_staged: bool = True) -> None:
                self.is_staged = is_staged
                self.closed = False
                self.target = cast(HostModuleTarget, nbytes)

            def close(self) -> None:
                self.closed = True

        monkeypatch.setattr(
            CudaTargetLease,
            "allocate",
            lambda plan, _d, *, allocation_stream=None: StagedLease(plan.nbytes),
        )

        def fail_distance(_unit_idx: int) -> float:
            raise AssertionError("distance_fn must not be called")

        stream = cast(torch.cuda.Stream, object())
        sig_a = cast(UnitSignature, ("a", 64))
        sig_b = cast(UnitSignature, ("b", 64))
        plan_a = cast(HostModuleLoadPlan, _FakePlan(64, sig_a))
        plan_b = cast(HostModuleLoadPlan, _FakePlan(64, sig_b))

        # 2-slot streaming with one checked-out slot has only 1 idle candidate on every reuse/evict.
        pool = ElasticVramPool(torch.device("cuda"), vram_budget_bytes=None)
        s0, _ = pool.checkout(0, plan_a, stream, distance_fn=fail_distance, is_lookahead=False)
        assert s0 is not None
        pool.return_slot(s0, staged_unit_idx=0, distance_fn=fail_distance)
        s1, _ = pool.checkout(1, plan_b, stream, distance_fn=fail_distance, is_lookahead=False)
        assert s1 is not None
        # Evicting sole mismatched idle slot s0 never calls fail_distance.
        s2, hit2 = pool.checkout(2, plan_b, stream, distance_fn=fail_distance, is_lookahead=True)
        assert s2 is not None and not hit2 and s2 is not s0 and cast(Any, s0.lease).closed
        pool.return_slot(s1, staged_unit_idx=1, distance_fn=fail_distance)
        # Reusing sole matching idle slot s1 (while s2 is checked out) never calls fail_distance.
        s3, hit3 = pool.checkout(3, plan_b, stream, distance_fn=fail_distance, is_lookahead=False)
        assert not hit3 and s3 is s1 and not cast(Any, s1.lease).closed
        pool.return_slot(s2, staged_unit_idx=2, distance_fn=fail_distance)
        pool.return_slot(s3, staged_unit_idx=3, distance_fn=fail_distance)
        pool.close_all()

        # Multiple candidates where any slot is unstaged skip distance_fn and pick LRU unstaged.
        s_unstaged_new = VramSlot(
            lease=cast(CudaTargetLease, StagedLease(64, is_staged=False)),
            signature=sig_a,
            nbytes=64,
            staged_unit_idx=0,
            last_used_seq=5,
        )
        s_unstaged_old = VramSlot(
            lease=cast(CudaTargetLease, StagedLease(64, is_staged=False)),
            signature=sig_a,
            nbytes=64,
            staged_unit_idx=None,
            last_used_seq=2,
        )
        s_staged_old = VramSlot(
            lease=cast(CudaTargetLease, StagedLease(64, is_staged=True)),
            signature=sig_a,
            nbytes=64,
            staged_unit_idx=1,
            last_used_seq=1,
        )
        assert ElasticVramPool._pick_victim(
            [s_staged_old, s_unstaged_new, s_unstaged_old],
            fail_distance,
        ) is s_unstaged_old


# ---------------------------------------------------------------------------
# Auto offload discovery tests
# ---------------------------------------------------------------------------


class TestAutoOffloadPaths:
    def test_discovers_outermost_module_list_and_large_standalone_modules(self) -> None:
        class InnerBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(64, 64, bias=False)])
                self.norm = nn.LayerNorm(64)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(64, 64, bias=False)  # 16 KiB
                self.blocks = nn.ModuleList([InnerBlock(), InnerBlock()])
                self.tiny = nn.Linear(4, 4, bias=False)  # 64 B
                self.head = nn.Linear(64, 64, bias=False)  # 16 KiB

        m = Model().requires_grad_(False)
        paths = auto_offload_paths(m, min_unit_bytes=4096)
        assert paths == ("embed", "blocks", "head")

    def test_skips_cross_boundary_ties_and_allows_shared_buffers(self) -> None:
        class Block(nn.Module):
            def __init__(self, rope: torch.Tensor) -> None:
                super().__init__()
                self.proj = nn.Linear(32, 32, bias=False)
                self.register_buffer("rope", rope)

        rope_table = torch.randn(32)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(32, 32, bias=False)
                self.head = nn.Linear(32, 32, bias=False)
                self.head.weight = self.embed.weight
                self.blocks = nn.ModuleList([Block(rope_table), Block(rope_table)])

        m = Model().requires_grad_(False)
        paths = auto_offload_paths(m, min_unit_bytes=1024)
        # Tied embed/head are skipped; blocks sharing read-only RoPE are kept.
        assert paths == ("blocks",)

    def test_handles_sequential_and_root_module_list(self) -> None:
        class SeqModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Linear(32, 32, bias=False),
                    nn.ReLU(),
                    nn.Linear(32, 32, bias=False),
                )

        seq = SeqModel().requires_grad_(False)
        assert auto_offload_paths(seq, min_unit_bytes=1024) == ("stem.0", "stem.2")
        assert auto_offload_paths(seq, min_unit_bytes=6000) == ("stem",)

        root_list = nn.ModuleList([nn.Linear(32, 32, bias=False)]).requires_grad_(False)
        root_paths = auto_offload_paths(root_list, min_unit_bytes=0)
        assert root_paths == ("0",)
        offloader = ModelOffloader.from_module(root_list, OffloadConfig(offload_paths=root_paths))
        offloader.deactivate()

    def test_estimates_meta_and_quantized_bytes(self) -> None:
        with torch.device("meta"):
            m = nn.Module()
            m.big = nn.Linear(128, 128, bias=False)
            m.small = nn.Linear(4, 4, bias=False)
        m.requires_grad_(False)
        assert auto_offload_paths(m, min_unit_bytes=16 * 1024) == ("big",)

        try:
            torchao_quant = pytest.importorskip("torchao.quantization")
        except pytest.skip.Exception:
            return
        qm = nn.Module()
        qm.proj = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
        qm.eval().requires_grad_(False)
        torchao_quant.quantize_(qm.proj, torchao_quant.Float8WeightOnlyConfig())
        assert auto_offload_paths(qm, min_unit_bytes=1024) == ("proj",)


# ---------------------------------------------------------------------------
# ElasticVramPool + ResidencyEngine CUDA tests
# ---------------------------------------------------------------------------


class TestResidencyEngineCuda:
    @CUDA
    def test_strict_2_slot_streaming_with_heterogeneous_units(self) -> None:
        class HeteroModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.Linear(16, 32, bias=False),
                        nn.Linear(32, 24, bias=False),
                        nn.Linear(24, 16, bias=True),
                        nn.Linear(16, 8, bias=False),
                    ]
                )
                self.blocks[1].to(torch.bfloat16)
                self.requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for idx, block in enumerate(self.blocks):
                    in_dtype = torch.bfloat16 if idx == 1 else torch.float32
                    x = block(x.to(in_dtype)).to(torch.float32)
                return x

        torch.manual_seed(0)
        model = HeteroModel()
        reference = copy.deepcopy(model).cuda()
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=None),
        )
        engine = residency_engine(offloader)
        observed_slots: list[int] = []
        handles = [
            blk.register_forward_pre_hook(
                lambda _m, _a: observed_slots.append(
                    engine.pool.live_slot_count if engine.pool is not None else 0
                )
            )
            for blk in model.blocks
        ]
        try:
            x = torch.randn(3, 16, device="cuda")
            with activated_model(offloader, "cuda"), torch.inference_mode():
                actual = model(x)
                expected = reference(x)
                torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            assert max(observed_slots) <= 2
        finally:
            for handle in handles:
                handle.remove()
            offloader.deactivate()

    @CUDA
    def test_partial_vram_belady_min_eviction_achieves_wraparound_hits(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 4 identical 64x64 float32 blocks (16 KiB each = 64 KiB total).
        # With a 3-block budget (48 KiB), FIFO/LRU suffers 4 misses/pass across
        # a 4-block wraparound loop; Belady MIN keeps 2 blocks warm per pass so
        # subsequent passes stage at most 2 blocks with zero extra allocations.
        torch.manual_seed(1)
        model = _make_uniform_model(num_blocks=4, dim=64)
        reference = copy.deepcopy(model).cuda()
        block_bytes = cast(nn.Linear, model.blocks[0]).weight.nbytes
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=3 * block_bytes),
        )
        engine = residency_engine(offloader)
        stage_calls = 0
        alloc_calls = 0
        original_stage = CudaTargetLease.stage
        original_allocate = CudaTargetLease.allocate

        def count_stage(
            lease: CudaTargetLease,
            plan: Any,
            stream: torch.cuda.Stream,
            *,
            non_blocking: bool,
        ) -> None:
            nonlocal stage_calls
            stage_calls += 1
            original_stage(lease, plan, stream, non_blocking=non_blocking)

        def count_allocate(*args: Any, **kwargs: Any) -> CudaTargetLease:
            nonlocal alloc_calls
            alloc_calls += 1
            return original_allocate(*args, **kwargs)

        monkeypatch.setattr(CudaTargetLease, "stage", count_stage)
        monkeypatch.setattr(CudaTargetLease, "allocate", count_allocate)
        try:
            x = torch.randn(2, 64, device="cuda")
            with activated_model(offloader, "cuda"), torch.inference_mode():
                # Pass 1 warms up predictor and cache (at most 3 allocated slots).
                actual = model(x)
                torch.cuda.synchronize()
                assert stage_calls <= 4
                assert alloc_calls <= 3

                stage_calls = 0
                alloc_calls = 0
                actual = model(x)
                torch.cuda.synchronize()
                # Belady MIN retains upcoming blocks across the wraparound boundary.
                assert stage_calls <= 2
                assert alloc_calls == 0
                assert engine.pool is not None
                assert engine.pool.used_bytes <= 3 * block_bytes
                assert len(engine.pool.cached_unit_indices()) >= 1
                expected = reference(x)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        finally:
            offloader.deactivate()

    @CUDA
    def test_all_resident_fast_path_skips_pinning_and_lookahead_thread(self) -> None:
        torch.manual_seed(2)
        model = _make_uniform_model(num_blocks=4, dim=32)
        reference = copy.deepcopy(model).cuda()
        total_bytes = sum(t.nbytes for t in model.state_dict().values())
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=total_bytes),
        )
        engine = residency_engine(offloader)
        try:
            x = torch.randn(2, 32, device="cuda")
            with activated_model(offloader, "cuda"), torch.inference_mode():
                assert engine._pin_lease is None
                assert engine._executor is None
                assert engine.pool is not None
                assert engine.pool.cached_unit_indices() == {0, 1, 2, 3}
                for unit in offloaded_units(offloader):
                    for tensor in unit.params["weight"].storage_tensors():
                        assert not tensor.is_pinned()
                actual = model(x)
                expected = reference(x)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        finally:
            offloader.deactivate()

    @CUDA
    def test_zero_budget_releases_immediately_on_exit(self) -> None:
        torch.manual_seed(2)
        model = _make_uniform_model(num_blocks=2, dim=16)
        reference = copy.deepcopy(model).cuda()
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=0),
        )
        engine = residency_engine(offloader)
        try:
            x = torch.randn(2, 16, device="cuda")
            with activated_model(offloader, "cuda"), torch.inference_mode():
                actual = model.blocks[0](x)
                assert engine._active is None
                assert engine._lookahead is None
                assert engine.pool is not None
                assert engine.pool.live_slot_count == 0
                assert cast(nn.Linear, model.blocks[0]).weight.device.type == "cpu"
                torch.testing.assert_close(actual, reference.blocks[0](x), rtol=1e-5, atol=1e-5)
        finally:
            offloader.deactivate()

    @CUDA
    def test_lazy_active_release_and_looped_reuse_stage_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class LoopAndBufferModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.shared = nn.Linear(16, 16, bias=False)
                self.shared.register_buffer("table", torch.randn(16))
                self.requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for _ in range(4):
                    x = self.shared(x) * self.shared.table
                return x

        torch.manual_seed(3)
        model = LoopAndBufferModel()
        reference = copy.deepcopy(model).cuda()
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("shared",), vram_budget_bytes=None),
        )
        stage_calls = 0
        original_stage = CudaTargetLease.stage

        def count_stage(
            lease: CudaTargetLease,
            plan: Any,
            stream: torch.cuda.Stream,
            *,
            non_blocking: bool,
        ) -> None:
            nonlocal stage_calls
            stage_calls += 1
            original_stage(lease, plan, stream, non_blocking=non_blocking)

        monkeypatch.setattr(CudaTargetLease, "stage", count_stage)
        try:
            x = torch.randn(2, 16, device="cuda")
            with activated_model(offloader, "cuda"), torch.inference_mode():
                actual = model(x)
                expected = reference(x)
            assert stage_calls == 1
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        finally:
            offloader.deactivate()

    @CUDA
    def test_merged_lora_caches_merged_slots_across_passes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(4)
        model = _make_uniform_model(num_blocks=3, dim=16)
        lora = Adapter.from_state_dict(
            {
                f"blocks.{i}.{name}": torch.randn(*shape)
                for i in range(3)
                for name, shape in (("lora_A.weight", (2, 16)), ("lora_B.weight", (16, 2)))
            }
        )
        reference = copy.deepcopy(model)
        total_bytes = sum(t.nbytes for t in model.state_dict().values())
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=total_bytes),
        )
        # Also run a streaming unmerged reference on CPU/CUDA via routed mode to check exact parity.
        ref_offloader = ModelOffloader.from_module(
            reference,
            OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=None),
        )
        stage_calls = 0
        original_stage = CudaTargetLease.stage

        def count_stage(
            lease: CudaTargetLease,
            plan: Any,
            stream: torch.cuda.Stream,
            *,
            non_blocking: bool,
        ) -> None:
            nonlocal stage_calls
            stage_calls += 1
            original_stage(lease, plan, stream, non_blocking=non_blocking)

        try:
            x = torch.randn(2, 16, device="cuda")
            with activated_model(ref_offloader, "cuda", adapters=[lora], adapter_mode="routed"), torch.inference_mode():
                expected = reference(x).clone()

            monkeypatch.setattr(CudaTargetLease, "stage", count_stage)
            with activated_model(offloader, "cuda", adapters=[lora], adapter_mode="merge"), torch.inference_mode():
                assert stage_calls == 3  # preloaded once at activation with merged LoRA
                stage_calls = 0
                actual_1 = model(x)
                actual_2 = model(x)
                assert stage_calls == 0  # hit cached merged slots on both passes
            torch.testing.assert_close(actual_1, expected, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(actual_2, expected, rtol=1e-4, atol=1e-4)
        finally:
            ref_offloader.deactivate()
            offloader.deactivate()

    @CUDA
    def test_mixed_standalone_and_module_list_async_overlap(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class MixedModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(16, 16, bias=False)
                self.blocks = nn.ModuleList([nn.Linear(16, 16, bias=False) for _ in range(3)])
                self.head = nn.Linear(16, 8, bias=False)
                self.requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.relu(self.embed(x))
                for blk in self.blocks:
                    x = torch.relu(blk(x))
                return self.head(x)

        torch.manual_seed(5)
        model = MixedModel()
        reference = copy.deepcopy(model).cuda()
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("embed", "blocks", "head")),
        )
        assert resident_instance(offloader) is None
        engine = residency_engine(offloader)
        assert engine.unit_paths == ("embed", "blocks.0", "blocks.1", "blocks.2", "head")
        lookahead_targets: list[int] = []
        observed_slots: list[int] = []
        original_submit = engine._submit_lookahead

        def spy_submit(next_idx: int) -> None:
            lookahead_targets.append(next_idx)
            original_submit(next_idx)
            if engine.pool is not None:
                observed_slots.append(engine.pool.live_slot_count)

        monkeypatch.setattr(engine, "_submit_lookahead", spy_submit)
        try:
            x = torch.randn(2, 16, device="cuda")
            with activated_model(offloader, "cuda"), torch.inference_mode():
                actual = model(x)
                torch.cuda.synchronize()
                expected = reference(x)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            assert lookahead_targets and max(observed_slots) <= 2
        finally:
            offloader.deactivate()

    @CUDA
    def test_per_run_vram_budget_override_on_activate_and_model_cache_use(self) -> None:
        torch.manual_seed(6)
        model = _make_uniform_model(num_blocks=4, dim=32)
        reference = copy.deepcopy(model).cuda()
        block_bytes = cast(nn.Linear, model.blocks[0]).weight.nbytes
        total_bytes = sum(t.nbytes for t in model.state_dict().values())
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=total_bytes),
        )
        engine = residency_engine(offloader)
        x = torch.randn(2, 32, device="cuda")
        expected = reference(x)
        try:
            # Invalid override fails before acquiring or altering default budget.
            for bad in (-1, True, "128"):
                with pytest.raises((TypeError, ValueError)):
                    offloader.activate("cuda", vram_budget_bytes=cast(Any, bad))
            assert offloader.active_device is None and offloader.vram_budget_bytes == total_bytes

            # Per-run override to streaming (None) triggers pinned 2-slot mode.
            with activated_model(offloader, "cuda", vram_budget_bytes=None), torch.inference_mode():
                assert offloader.vram_budget_bytes is None
                torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-5)
                assert engine.pool is not None and engine.pool.live_slot_count <= 2
                assert engine._pin_lease is not None

            # Per-run override to 0 frees immediately.
            with activated_model(offloader, "cuda", vram_budget_bytes=0), torch.inference_mode():
                assert offloader.vram_budget_bytes == 0
                _ = model.blocks[0](x)
                assert engine.pool is not None and engine.pool.live_slot_count == 0

            # Omitting override resets back to OffloadConfig default (all-resident, unpinned).
            with activated_model(offloader, "cuda"), torch.inference_mode():
                assert offloader.vram_budget_bytes == total_bytes
                assert engine._pin_lease is None and engine._executor is None
                torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-5)
        finally:
            offloader.deactivate()

        cache = ModelCache()
        spec = ModelSpec(
            key="budget-override",
            factory=lambda: _make_uniform_model(num_blocks=3, dim=16).eval().requires_grad_(False),
            config=OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=None),
        )
        cx = torch.randn(2, 16, device="cuda")
        with cache.use(spec, device="cuda") as cached_m, torch.inference_mode():
            out_stream = cached_m(cx).clone()
        b_bytes = 16 * 16 * 4
        with cache.use(spec, device="cuda", vram_budget_bytes=2 * b_bytes) as cached_m, torch.inference_mode():
            out_cached = cached_m(cx).clone()
        torch.testing.assert_close(out_cached, out_stream, rtol=1e-5, atol=1e-5)
        del cached_m
        cache.clear()

    @CUDA
    def test_merged_lora_partial_cache_and_cross_unit_slot_isolation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(11)
        model = _make_uniform_model(num_blocks=4, dim=16)
        lora = Adapter.from_state_dict(
            {
                "blocks.0.lora_A.weight": torch.randn(2, 16),
                "blocks.0.lora_B.weight": torch.randn(16, 2),
            }
        )
        ref = copy.deepcopy(model)
        ref_off = ModelOffloader.from_module(ref, OffloadConfig(offload_paths=("blocks",)))
        block_bytes = cast(nn.Linear, model.blocks[0]).weight.nbytes
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=2 * block_bytes),
        )
        stage_calls = 0
        alloc_calls = 0
        orig_stage = CudaTargetLease.stage
        orig_alloc = CudaTargetLease.allocate

        def spy_stage(lease: CudaTargetLease, plan: Any, stream: torch.cuda.Stream, *, non_blocking: bool) -> None:
            nonlocal stage_calls
            stage_calls += 1
            orig_stage(lease, plan, stream, non_blocking=non_blocking)

        def spy_alloc(*args: Any, **kwargs: Any) -> CudaTargetLease:
            nonlocal alloc_calls
            alloc_calls += 1
            return orig_alloc(*args, **kwargs)

        try:
            x = torch.randn(2, 16, device="cuda")
            with activated_model(ref_off, "cuda", adapters=[lora], adapter_mode="routed"), torch.inference_mode():
                expected = ref(x).clone()
            ref_off.deactivate()

            monkeypatch.setattr(CudaTargetLease, "stage", spy_stage)
            monkeypatch.setattr(CudaTargetLease, "allocate", spy_alloc)

            # Partial cache (2 blocks of 4) and 1-block budget reusing merged<->unmerged slot:
            for budget in (2 * block_bytes, block_bytes, None):
                stage_calls = 0
                alloc_calls = 0
                with activated_model(
                    offloader, "cuda", adapters=[lora], adapter_mode="merge", vram_budget_bytes=budget
                ), torch.inference_mode():
                    actual1 = model(x)
                    actual2 = model(x)
                    torch.cuda.synchronize()
                torch.testing.assert_close(actual1, expected, rtol=1e-4, atol=1e-4)
                torch.testing.assert_close(actual2, expected, rtol=1e-4, atol=1e-4)
                if budget == block_bytes:
                    assert alloc_calls == 1 and stage_calls == 8
        finally:
            ref_off.deactivate()
            offloader.deactivate()

    @CUDA
    @pytest.mark.parametrize("use_budget", [False, True])
    def test_per_unit_block_compile_under_streaming_and_budget(
        self,
        use_budget: bool,
    ) -> None:
        torch.manual_seed(7)
        model = _make_uniform_model(num_blocks=3, dim=16)
        reference = copy.deepcopy(model).cuda()
        total_bytes = sum(t.nbytes for t in model.state_dict().values())
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(
                offload_paths=("blocks",),
                vram_budget_bytes=total_bytes if use_budget else None,
                block_compile=BlockCompileConfig(fullgraph=True),
            ),
        )
        try:
            x = torch.randn(2, 16, device="cuda")
            with activated_model(offloader, "cuda"), torch.inference_mode():
                actual = model(x)
                expected = reference(x)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            # Restored original forwards cleanly after deactivate().
            for blk in model.blocks:
                assert "forward" not in blk.__dict__
        finally:
            offloader.deactivate()

    def test_wait_for_prefetch_waits_for_worker_on_interrupt(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        started = threading.Event()
        release = threading.Event()
        worker_finished = False

        def work() -> None:
            nonlocal worker_finished
            started.set()
            release.wait(timeout=5)
            worker_finished = True

        original_result = Future.result
        original_wait = concurrent.futures.wait
        wait_calls = 0

        def interrupting_result(self: Future[object], timeout: float | None = None) -> object:
            if not started.wait(timeout=5):
                raise AssertionError("worker did not start")
            raise Interrupt("interrupted while waiting")

        def retrying_wait(
            fs: Any,
            timeout: float | None = None,
            return_when: str = concurrent.futures.ALL_COMPLETED,
        ) -> Any:
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                raise Interrupt("interrupted inside concurrent.futures.wait")
            release.set()
            return original_wait(fs, timeout=timeout, return_when=return_when)

        monkeypatch.setattr(Future, "result", interrupting_result)
        monkeypatch.setattr("piper_offload.residency_engine.wait", retrying_wait)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(work)
            with pytest.raises(Interrupt, match="interrupted while waiting"):
                _wait_for_prefetch(future)
            assert worker_finished
            assert future.done()
            assert wait_calls == 2

    @CUDA
    def test_interrupted_wait_in_before_unit_forward_waits_for_worker_and_recovers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(8)
        model = _make_uniform_model(num_blocks=3, dim=32)
        reference = copy.deepcopy(model).cuda()
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=None),
        )
        engine = residency_engine(offloader)
        allow_worker_finish = threading.Event()
        worker_started, stage_finished, _ = _install_delayed_lookahead_hooks(
            monkeypatch,
            allow_worker_finish,
        )

        try:
            x = torch.randn(2, 32, device="cuda")
            with activated_model(offloader, "cuda"), torch.inference_mode():
                expected = reference(x)
                _ = model.blocks[0](x)
                assert worker_started.wait(timeout=5)
                lookahead = engine._lookahead
                assert lookahead is not None
                lookahead_future = lookahead.future

                def interrupt_result(timeout: float | None = None) -> object:
                    del timeout
                    threading.Timer(0.02, allow_worker_finish.set).start()
                    raise Interrupt("interrupted lookahead in forward")

                with pytest.MonkeyPatch.context() as inner:
                    inner.setattr(lookahead_future, "result", interrupt_result)
                    with pytest.raises(Interrupt, match="interrupted lookahead in forward"):
                        _ = model.blocks[1](x)

                assert stage_finished[0]
                assert lookahead_future.done()
                assert engine.pool is not None and engine.pool.live_slot_count <= 2
                torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-5)
        finally:
            allow_worker_finish.set()
            offloader.deactivate()

    @CUDA
    def test_sync_stage_and_acquire_failures_do_not_orphan_slots(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(12)
        model = _make_uniform_model(num_blocks=3, dim=32)
        reference = copy.deepcopy(model).cuda()
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=None),
        )
        engine = residency_engine(offloader)
        orig_stage = CudaTargetLease.stage
        orig_acquire = CudaTargetLease.acquire
        caller_tid = threading.get_ident()
        fail_stage_once = [True]
        fail_acquire_once = [False]

        def failing_stage(
            lease: CudaTargetLease,
            plan: Any,
            stream: torch.cuda.Stream,
            *,
            non_blocking: bool = True,
        ) -> None:
            if threading.get_ident() == caller_tid and fail_stage_once[0]:
                fail_stage_once[0] = False
                raise Interrupt("injected sync stage failure")
            orig_stage(lease, plan, stream, non_blocking=non_blocking)

        def failing_acquire(lease: CudaTargetLease, stream: torch.cuda.Stream) -> Any:
            if fail_acquire_once[0]:
                fail_acquire_once[0] = False
                raise Interrupt("injected acquire failure")
            return orig_acquire(lease, stream)

        monkeypatch.setattr(CudaTargetLease, "stage", failing_stage)
        monkeypatch.setattr(CudaTargetLease, "acquire", failing_acquire)
        try:
            x = torch.randn(2, 32, device="cuda")
            with activated_model(offloader, "cuda"), torch.inference_mode():
                expected = reference(x)
                with pytest.raises(Interrupt, match="injected sync stage failure"):
                    _ = model.blocks[0](x)
                assert engine._active is None
                assert engine.pool is not None and engine.pool.live_slot_count == 1
                assert not any(s.checked_out for s in engine.pool._slots)

                fail_acquire_once[0] = True
                with pytest.raises(Interrupt, match="injected acquire failure"):
                    _ = model.blocks[1](x)
                torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-5)
                assert engine.pool.live_slot_count <= 2
        finally:
            offloader.deactivate()

    @CUDA
    def test_unit_forward_exception_and_interrupt_do_not_strand_active_unit(self) -> None:
        torch.manual_seed(9)
        model = _make_seq_blocks_model(dim=32)
        reference = copy.deepcopy(model).cuda()
        first_net: Any = model.blocks[0].net  # type: ignore[union-attr]
        block_bytes = cast(nn.Linear, first_net[0]).weight.nbytes
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=0),
        )
        engine = residency_engine(offloader)
        fail_with: list[type[BaseException] | None] = [None]

        def fail_hook(_module: nn.Module, _args: tuple[object, ...]) -> None:
            if fail_with[0] is not None:
                raise fail_with[0]("boom inside unit")

        handle = cast(nn.Module, first_net[0]).register_forward_pre_hook(fail_hook)
        try:
            x = torch.randn(2, 32, device="cuda")
            with activated_model(offloader, "cuda"), torch.inference_mode():
                assert engine.pool is not None
                expected = reference(x)

                fail_with[0] = RuntimeError
                with pytest.raises(RuntimeError, match="boom inside unit"):
                    _ = model.blocks[0](x)
                assert engine._active is None and engine.pool.live_slot_count == 0

                fail_with[0] = Interrupt
                with pytest.raises(Interrupt, match="boom inside unit"):
                    _ = model.blocks[0](x)
                fail_with[0] = None
                _ = model.blocks[0](x)
                assert engine._active is None and engine.pool.live_slot_count == 0

            with activated_model(offloader, "cuda", vram_budget_bytes=2 * block_bytes), torch.inference_mode():
                fail_with[0] = Interrupt
                with pytest.raises(Interrupt, match="boom inside unit"):
                    _ = model.blocks[0](x)
                fail_with[0] = None
                torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-5)
            assert engine._active is None and not offloader.failed
        finally:
            handle.remove()
            offloader.deactivate()

    def test_packed_adapters_report_exact_gpu_target_bytes(self) -> None:
        # GGUF Q4_0 (64, 64) -> host 2304 B, GPU ConvRot INT8 = 64*64 + 64*4 = 4352 B.
        packed = packed_q4_0(64, 64, generator=torch.Generator().manual_seed(0))
        gguf_param = GGUFParameter(packed, requires_grad=False, quant_type=2)
        gguf_host = HostParam(gguf_param)
        assert gguf_host.cache_bytes == 2304
        assert gguf_host.gpu_target_bytes == 64 * 64 + 64 * 4

        # Trellis 3-bpw (64, 256) with optional act_per_tensor_scale.
        rows, cols = 64, 256
        gen = torch.Generator().manual_seed(1)
        trellis_no_act = TrellisParameter(
            torch.randint(0, 256, (rows, 1, 96), dtype=torch.uint8, generator=gen),
            torch.randint(0x20, 0x40, (rows, 1), dtype=torch.uint8, generator=gen).view(
                torch.float8_e4m3fn
            ),
            torch.rand(rows, 1, generator=gen) * 1e-3 + 1e-4,
            block_size=256,
            group_size=256,
        )
        trellis_with_act = TrellisParameter(
            trellis_no_act.bitstreams.clone(),
            trellis_no_act.block_scales.clone(),
            trellis_no_act.scale.clone(),
            block_size=256,
            group_size=256,
            act_per_tensor_scale=torch.tensor(0.5),
        )
        t_host_no_act = HostParam(trellis_no_act)
        t_host_with_act = HostParam(trellis_with_act)
        assert t_host_no_act.gpu_target_bytes == rows * cols + rows * 4
        assert t_host_no_act.gpu_target_bytes > t_host_no_act.cache_bytes
        assert t_host_with_act.gpu_target_bytes == rows * cols + rows * 4 + 4

        class PackedBlock(nn.Module):
            def __init__(self, weight: nn.Parameter) -> None:
                super().__init__()
                self.weight = weight

        class PackedModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [PackedBlock(gguf_param), PackedBlock(trellis_with_act)]
                )
                self.requires_grad_(False)

        model = PackedModel()
        plan0 = HostModuleStore.from_module(model.blocks[0]).bind(model.blocks[0]).resolve_load_plan()
        plan1 = HostModuleStore.from_module(model.blocks[1]).bind(model.blocks[1]).resolve_load_plan()
        expected_b0 = rows * 64 + rows * 4
        expected_b1 = rows * cols + rows * 4 + 4
        assert estimate_plan_gpu_bytes(plan0) == expected_b0
        assert estimate_plan_gpu_bytes(plan1) == expected_b1

        if torch.cuda.is_available():
            offloader = ModelOffloader.from_module(
                model,
                OffloadConfig(
                    offload_paths=("blocks",),
                    vram_budget_bytes=expected_b0 + expected_b1,
                ),
            )
            engine = residency_engine(offloader)
            try:
                offloader.activate("cuda")
                assert engine.pool is not None
                assert engine.pool.used_bytes == expected_b0 + expected_b1
                slot_bytes = sorted(s.nbytes for s in engine.pool._slots)
                assert slot_bytes == sorted((expected_b0, expected_b1))
            finally:
                offloader.deactivate()

    @CUDA
    def test_lazy_active_unit_reuse_across_streams_waits_for_staging(self) -> None:
        torch.manual_seed(10)
        model = _make_uniform_model(num_blocks=3, dim=32)
        reference = copy.deepcopy(model).cuda()
        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=None),
        )
        engine = residency_engine(offloader)
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        try:
            x = torch.randn(2, 32, device="cuda")
            expected = reference(x)
            with activated_model(offloader, "cuda"), torch.inference_mode():
                with torch.cuda.stream(stream_a):
                    _ = model(x)
                # Last block stays lazily active; re-entering on stream_b must wait on _ready_event.
                active = engine._active
                assert active is not None and active.unit_idx == 2
                with torch.cuda.stream(stream_b):
                    y = model.blocks[2](x)
                    assert stream_b in active.slot.lease._waited_streams
                torch.cuda.synchronize()
                torch.testing.assert_close(y, reference.blocks[2](x), rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-5)
        finally:
            offloader.deactivate()

    def test_auto_offload_paths_estimates_packed_bytes_for_gguf_trellis_and_projected_shard(
        self,
    ) -> None:
        from piper_offload.float8_adapter import Float8Adapter, _Float8Meta
        from piper_offload.torchao_structured_adapter import TorchaoHost, _ProjectedTorchaoShard

        # 1. GGUF Q4_0 (64, 64): packed = 2304 B (vs dense BF16 = 8192 B)
        gguf_param = GGUFParameter(
            packed_q4_0(64, 64, generator=torch.Generator().manual_seed(2)),
            requires_grad=False,
            quant_type=2,
        )
        assert _estimate_tensor_bytes(gguf_param) == 2304

        # 2. Trellis 3-bpw (64, 256): packed = 6144 + 64 + 256 = 6464 B (vs dense BF16 = 32768 B)
        gen = torch.Generator().manual_seed(3)
        trellis_param = TrellisParameter(
            torch.randint(0, 256, (64, 1, 96), dtype=torch.uint8, generator=gen),
            torch.randint(0x20, 0x40, (64, 1), dtype=torch.uint8, generator=gen).view(
                torch.float8_e4m3fn
            ),
            torch.rand(64, 1, generator=gen) * 1e-3 + 1e-4,
            block_size=256,
            group_size=256,
        )
        assert _estimate_tensor_bytes(trellis_param) == 64 * 96 + 64 + 64 * 4

        # 3. _ProjectedTorchaoShard: float8 qdata (4096 B) + float32 scale (256 B) = 4352 B
        #    (vs logical meta BF16 = 8192 B).
        proj_shard = _ProjectedTorchaoShard(
            Float8Adapter,
            TorchaoHost(
                storage=(
                    torch.empty((64, 64), dtype=torch.float8_e4m3fn),
                    torch.empty((64, 1), dtype=torch.float32),
                ),
                meta=_Float8Meta(
                    block_size=(1, 64),
                    mm_config=None,
                    act_quant_kwargs=None,
                    kernel_preference=None,
                    dtype=torch.bfloat16,
                ),
                logical_shape=(64, 64),
                target_shapes=((64, 64), (64, 1)),
            ),
            torch.empty((64, 64), dtype=torch.bfloat16, device="meta"),
            ("proj",),
        )
        proj_param = nn.Parameter(proj_shard, requires_grad=False)
        assert _estimate_tensor_bytes(proj_param) == 64 * 64 + 64 * 4

        class Holder(nn.Module):
            def __init__(self, p: nn.Parameter) -> None:
                super().__init__()
                self.weight = p

        class MixedPackedModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gguf = Holder(gguf_param)
                self.proj = Holder(proj_param)
                self.trellis = Holder(trellis_param)
                self.requires_grad_(False)

        m = MixedPackedModel()
        assert auto_offload_paths(m, min_unit_bytes=8000) == ()
        assert auto_offload_paths(m, min_unit_bytes=6000) == ("trellis",)
        assert auto_offload_paths(m, min_unit_bytes=4000) == ("proj", "trellis")
        assert auto_offload_paths(m, min_unit_bytes=2000) == ("gguf", "proj", "trellis")

    def test_auto_offload_paths_skips_invalid_module_lists_and_non_callable_containers(
        self,
    ) -> None:
        # 1. Tied weights across ModuleList elements must NOT fall back to ("blocks",).
        tied = nn.Module()
        b0, b1 = nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False)
        b1.weight = b0.weight
        tied.blocks = nn.ModuleList([b0, b1])
        tied.requires_grad_(False)
        assert auto_offload_paths(tied, min_unit_bytes=1) == ()

        # Wrapper around a tied ModuleList stays valid as a standalone unit.
        outer = nn.Module()
        outer.stage = tied
        outer.requires_grad_(False)
        assert auto_offload_paths(outer, min_unit_bytes=1) == ("stage",)

        # 2. ModuleList with direct param and empty elements returns ().
        direct = nn.Module()
        direct.blocks = nn.ModuleList([nn.ReLU(), nn.ReLU()])
        direct.blocks.register_parameter(
            "scale", nn.Parameter(torch.randn(32, 32), requires_grad=False)
        )
        assert auto_offload_paths(direct, min_unit_bytes=1) == ()

        # 3. ModuleDict / ParameterList / ParameterDict and nested ModuleList elements.
        containers = nn.Module()
        containers.mdict = nn.ModuleDict(
            {"a": nn.Linear(16, 16, bias=False), "b": nn.Linear(16, 16, bias=False)}
        )
        containers.plist = nn.ParameterList([nn.Parameter(torch.randn(32, 32), requires_grad=False)])
        containers.pdict = nn.ParameterDict({"w": nn.Parameter(torch.randn(32, 32), requires_grad=False)})
        containers.nested = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(32, 32, bias=False)]),
                nn.ModuleList([nn.Linear(32, 32, bias=False)]),
            ]
        )
        containers.requires_grad_(False)
        assert auto_offload_paths(containers, min_unit_bytes=1500) == ("nested.0", "nested.1")

    def test_shared_stateless_submodule_and_same_list_alias_work_end_to_end(self) -> None:
        class ActBlock(nn.Module):
            def __init__(self, act: nn.Module) -> None:
                super().__init__()
                self.lin = nn.Linear(16, 16, bias=False)
                self.act = act

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.act(self.lin(x))

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                shared_act = nn.SiLU()
                self.blocks = nn.ModuleList([ActBlock(shared_act), ActBlock(shared_act)])
                loop_blk = ActBlock(shared_act)
                self.looped = nn.ModuleList([loop_blk, loop_blk])
                self.requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for b in self.blocks:
                    x = b(x)
                for b in self.looped:
                    x = b(x)
                return x

        model = Net()
        paths = auto_offload_paths(model, min_unit_bytes=1)
        assert paths == ("blocks", "looped")
        offloader = ModelOffloader.from_module(model, OffloadConfig(offload_paths=paths))
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            x = torch.randn(2, 16, device=device)
            with activated_model(offloader, device), torch.inference_mode():
                assert torch.isfinite(model(x)).all()
        finally:
            offloader.deactivate()

    def test_auto_offload_paths_tolerates_custom_subclass_and_sparse_buffer(self) -> None:
        from piper_offload import register_adapter

        class CustomPacked(torch.Tensor):
            @staticmethod
            def __new__(cls, packed: torch.Tensor) -> CustomPacked:
                t = torch.Tensor._make_wrapper_subclass(
                    cls, packed.shape, dtype=torch.bfloat16, requires_grad=False
                )
                t._packed = packed
                return t

            @classmethod
            def __torch_dispatch__(
                cls,
                func: Any,
                types: Any,
                args: Any = (),
                kwargs: Any = None,
            ) -> Any:
                if func == torch.ops.aten.detach.default:
                    return cls(args[0]._packed)
                return NotImplemented

        class CustomAdapter:
            @staticmethod
            def matches(t: torch.Tensor) -> bool:
                return isinstance(t, CustomPacked)

            @staticmethod
            def tensor_id(t: torch.Tensor) -> tuple[object, ...]:
                return ("custom", cast(Any, t)._packed.untyped_storage().data_ptr())

            @staticmethod
            def layout_signature(t: torch.Tensor) -> tuple[object, ...]:
                return (tuple(t.shape),)

            @staticmethod
            def capture_host(t: torch.Tensor) -> torch.Tensor:
                return cast(Any, t)._packed

            @staticmethod
            def storage_tensors(state: torch.Tensor) -> tuple[torch.Tensor, ...]:
                return (state,)

            @staticmethod
            def cpu_param(state: torch.Tensor) -> nn.Parameter:
                return nn.Parameter(CustomPacked(state), requires_grad=False)

            @staticmethod
            def alloc_gpu(state: torch.Tensor, device: torch.device) -> torch.Tensor:
                return torch.empty_like(state, device=device)

            @staticmethod
            def gpu_param(_h: torch.Tensor, gpu_state: torch.Tensor) -> nn.Parameter:
                return nn.Parameter(gpu_state, requires_grad=False)

            @staticmethod
            def copy_to_gpu(src: torch.Tensor, dst: torch.Tensor, *, non_blocking: bool = False) -> None:
                dst.copy_(src, non_blocking=non_blocking)

            @staticmethod
            def compute_dtype(_t: torch.Tensor) -> torch.dtype:
                return torch.bfloat16

            @staticmethod
            def cache_bytes(state: torch.Tensor) -> int:
                return state.nbytes

        remove = register_adapter(cast(Any, CustomAdapter))
        try:
            m = nn.Module()
            m.custom = nn.Module()
            m.custom.w = nn.Parameter(CustomPacked(torch.zeros((32, 32), dtype=torch.uint8)), requires_grad=False)
            sparse = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (4, 4))
            m.custom.register_buffer("sp", sparse)
            assert auto_offload_paths(m, min_unit_bytes=1) == ("custom",)
        finally:
            remove()

    def test_heterogeneous_store_bind_and_atomic_compile_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class HeteroBind(nn.Module):
            def __init__(self, *, meta: bool = False) -> None:
                super().__init__()
                dev = "meta" if meta else "cpu"
                self.embed = nn.Linear(8, 12, bias=False, device=dev)
                self.blocks = nn.ModuleList(
                    [nn.Linear(12, 16, bias=True, device=dev), nn.Linear(16, 4, bias=False, device=dev)]
                )
                self.requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.relu(self.embed(x))
                for b in self.blocks:
                    x = b(x)
                return x

        torch.manual_seed(13)
        proto = HeteroBind(meta=False)
        x = torch.randn(2, 8)
        expected = proto(x)
        store = ModelStore.from_module(proto)
        shell = HeteroBind(meta=True)

        # Compile failure during bind leaves meta shell completely untouched.
        monkeypatch.setattr(torch, "compile", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("compile boom")))
        with pytest.raises(RuntimeError, match="compile boom"):
            store.bind(shell, OffloadConfig(offload_paths=("blocks",), block_compile=BlockCompileConfig()))
        assert all(p.is_meta for p in shell.parameters())
        monkeypatch.undo()

        bound = store.bind(shell, OffloadConfig(offload_paths=("blocks",)))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with activated_model(bound, device), torch.inference_mode():
            torch.testing.assert_close(shell(x.to(device)).cpu(), expected)
        bound.deactivate()
