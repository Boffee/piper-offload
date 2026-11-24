"""Tests for concurrent in-process multi-rank execution via ParallelExecutor."""

from __future__ import annotations

import contextlib
import copy
import gc
import json
import mmap
import os
import subprocess
import sys
import threading
import time
import weakref
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._functional_collectives import AsyncCollectiveTensor, all_reduce as funcol_all_reduce
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, SequenceParallel, parallelize_module

from piper_offload import (
    ModelStore,
    OffloadConfig,
    _parallel_collectives,
    _sequential_runtime,
    _transfer_priority,
    parallel as parallel_mod,
)
from piper_offload._mapped_file import MappedFile
from piper_offload.parallel import ParallelExecutor
from piper_offload.pin_manager import host_pin_manager
from piper_offload.sequential import SequentialExecutor
from tests.conftest import activated_model

pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
TWO_GPUS = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA/HIP GPUs required")
_HAS_TWO_GPU_P2P = (
    torch.cuda.device_count() >= 2
    and torch.cuda.can_device_access_peer(0, 1)
    and torch.cuda.can_device_access_peer(1, 0)
)
TWO_GPUS_P2P = pytest.mark.skipif(not _HAS_TWO_GPU_P2P, reason="Two P2P-capable CUDA/HIP GPUs required")


class _TwoLayerMLP(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2, bias=False)
        self.fc2 = nn.Linear(dim * 2, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc2(torch.relu(self.fc1(x))))


class _StreamableModel(nn.Module):
    def __init__(self, width: int = 8, num_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(num_blocks)])
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _parallelize_mlp(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    return parallelize_module(
        model,
        mesh,
        {
            "fc1": ColwiseParallel(),
            "fc2": RowwiseParallel(output_layouts=Shard(1)),
            "norm": SequenceParallel(use_local_output=True),
        },
    )


def _assert_reductions_and_broadcast(rank: int, world_size: int) -> None:
    expected_sum = sum(r + 1 for r in range(world_size))
    fp32 = torch.full((96,), float(rank + 1), dtype=torch.float32)
    bf16 = torch.full((96,), float(rank + 1), dtype=torch.bfloat16)
    fp16 = torch.full((96,), float(rank + 1), dtype=torch.float16)
    int64 = torch.full((32,), rank + 1, dtype=torch.int64)
    dist.all_reduce(fp32)
    dist.all_reduce(bf16)
    dist.all_reduce(fp16)
    dist.all_reduce(int64)
    torch.testing.assert_close(fp32, torch.full((96,), float(expected_sum)))
    torch.testing.assert_close(bf16.float(), torch.full((96,), float(expected_sum)))
    torch.testing.assert_close(fp16.float(), torch.full((96,), float(expected_sum)))
    assert torch.equal(int64, torch.full((32,), expected_sum, dtype=torch.int64))

    pg = dist.distributed_c10d._get_default_group()
    c0 = torch.full((40,), float(rank + 1))
    c1 = torch.full((40,), float(2 * (rank + 1)))
    work = pg.allreduce_coalesced([c0, c1])
    assert work.wait()
    assert work.is_completed() and work.is_success()
    assert len(work.result()) == 2
    torch.testing.assert_close(c0, torch.full((40,), float(expected_sum)))
    torch.testing.assert_close(c1, torch.full((40,), float(2 * expected_sum)))

    last_root = world_size - 1
    b0 = torch.full((80,), 11.0 if rank == 0 else -1.0)
    dist.broadcast(b0, src=0)
    torch.testing.assert_close(b0, torch.full((80,), 11.0))
    b1 = torch.full((80,), 23.0 if rank == last_root else -1.0)
    dist.broadcast(b1, src=last_root)
    torch.testing.assert_close(b1, torch.full((80,), 23.0))

    sc_list = [torch.full((64,), float(10 + r)) for r in range(world_size)] if rank == last_root else None
    sc_out = sc_list[last_root] if (rank == last_root and sc_list is not None) else torch.empty(64)
    dist.scatter(sc_out, sc_list, src=last_root)
    torch.testing.assert_close(sc_out, torch.full((64,), float(10 + rank)))


def _assert_gather_and_reduce_scatter(rank: int, world_size: int) -> None:
    expected_sum = sum(r + 1 for r in range(world_size))
    pg = dist.distributed_c10d._get_default_group()

    ag_src = torch.arange(48, dtype=torch.float32) + rank * 100
    ag_outs = [torch.empty(48, dtype=torch.float32) for _ in range(world_size)]
    ag_outs[rank].copy_(ag_src)
    dist.all_gather(ag_outs, ag_outs[rank])
    for r in range(world_size):
        torch.testing.assert_close(ag_outs[r], torch.arange(48, dtype=torch.float32) + r * 100)

    ag_single_out = torch.empty(world_size * 48, dtype=torch.float32)
    dist.all_gather_into_tensor(ag_single_out, ag_src)
    torch.testing.assert_close(ag_single_out, torch.cat(ag_outs))

    base_out = torch.empty(world_size, 48, dtype=torch.float32)
    pg._allgather_base(base_out, ag_src).wait()
    torch.testing.assert_close(base_out.reshape(-1), ag_single_out)

    c_ag_out = torch.empty(world_size * 32, dtype=torch.float32)
    c_ag_in = torch.full((32,), float(rank + 5))
    pg.all_gather_single_coalesced([c_ag_out], [c_ag_in]).wait()
    expected_c_ag = torch.cat([torch.full((32,), float(r + 5)) for r in range(world_size)])
    torch.testing.assert_close(c_ag_out, expected_c_ag)

    rs_inputs = [torch.full((48,), float(rank + 1 + k)) for k in range(world_size)]
    rs_out = rs_inputs[rank]
    dist.reduce_scatter(rs_out, rs_inputs)
    torch.testing.assert_close(rs_out, torch.full((48,), float(expected_sum + world_size * rank)))

    rs_tensor_in = torch.cat([torch.full((40,), float(k + 1)) for k in range(world_size)])
    rs_tensor_out = torch.empty(40)
    dist.reduce_scatter_tensor(rs_tensor_out, rs_tensor_in)
    torch.testing.assert_close(rs_tensor_out, torch.full((40,), float(world_size * (rank + 1))))

    rs_base_out = torch.empty(40)
    pg._reduce_scatter_base(rs_base_out, rs_tensor_in).wait()
    torch.testing.assert_close(rs_base_out, rs_tensor_out)

    rs_c_out = torch.empty(20)
    rs_c_in = torch.cat([torch.full((20,), float(rank + 2)) for _ in range(world_size)])
    pg.reduce_scatter_tensor_coalesced([rs_c_out], [rs_c_in]).wait()
    torch.testing.assert_close(rs_c_out, torch.full((20,), float(sum(r + 2 for r in range(world_size)))))


def _assert_alltoall_variants(rank: int, world_size: int) -> None:
    a2a_in = [torch.full((36,), float(rank * 10 + dst)) for dst in range(world_size)]
    a2a_out = [torch.empty(36) for _ in range(world_size)]
    a2a_out[rank] = a2a_in[rank]
    dist.all_to_all(a2a_out, a2a_in)
    for src in range(world_size):
        torch.testing.assert_close(a2a_out[src], torch.full((36,), float(src * 10 + rank)))

    in_splits = [0 if (dst == 0 and world_size > 1) else (dst + 1) * 2 for dst in range(world_size)]
    out_splits = [0 if (rank == 0 and world_size > 1) else (rank + 1) * 2 for _ in range(world_size)]
    a2a_s_in = torch.cat([torch.full((sz, 5), float(rank * 10 + d)) for d, sz in enumerate(in_splits)], dim=0)
    a2a_s_out = torch.empty((sum(out_splits), 5))
    dist.all_to_all_single(a2a_s_out, a2a_s_in, out_splits, in_splits)
    offset = 0
    for src, sz in enumerate(out_splits):
        if sz > 0:
            actual = a2a_s_out[offset : offset + sz]
            torch.testing.assert_close(actual, torch.full((sz, 5), float(src * 10 + rank)))
        offset += sz


def _run_simple_allreduce(rank: int) -> float:
    t = torch.full((4,), float(rank + 1))
    dist.all_reduce(t)
    return float(t[0].item())


def test_constructor_validation_and_properties() -> None:
    with pytest.raises(ValueError, match="empty"):
        ParallelExecutor([])
    with pytest.raises(TypeError, match="sequence"):
        ParallelExecutor("cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="booleans"):
        ParallelExecutor([True])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-negative"):
        ParallelExecutor([-1])
    with pytest.raises(ValueError, match="one device type"):
        ParallelExecutor(["cpu", "cuda:0"])
    with pytest.raises(ValueError, match="requires CUDA"):
        ParallelExecutor(["cpu", "cpu"], transport="p2p")
    with pytest.raises(ValueError, match="requires CUDA"):
        ParallelExecutor(["cpu", "cpu"], transport="host_staged")
    with pytest.raises(ValueError, match="unsupported transport"):
        ParallelExecutor(["cpu", "cpu"], transport="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="staging_bytes"):
        ParallelExecutor(["cpu", "cpu"], staging_bytes=0)
    with pytest.raises(ValueError, match="staging_bytes"):
        ParallelExecutor(["cpu", "cpu"], staging_bytes=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="collective_timeout"):
        ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta(0))

    ex = ParallelExecutor(["cpu", "cpu", "cpu"], staging_bytes=256, collective_timeout=timedelta(seconds=5))
    assert ex.devices == (torch.device("cpu"), torch.device("cpu"), torch.device("cpu"))
    assert ex.world_size == 3
    assert ex.resolved_transport == "cpu"
    assert ex.staging_bytes == 256
    assert ex.collective_timeout == timedelta(seconds=5)
    assert not ex.is_open


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_cpu_world_sizes_and_collectives(world_size: int) -> None:
    with ParallelExecutor(["cpu"] * world_size, staging_bytes=128) as ex:
        assert ex.is_open

        def worker(rank: int) -> tuple[torch.Tensor]:
            assert dist.get_rank() == rank
            assert dist.get_world_size() == world_size
            assert dist.get_backend() == "piper_parallel"

            dist.barrier()
            empty = torch.empty(0, dtype=torch.float32)
            dist.all_reduce(empty)

            _assert_reductions_and_broadcast(rank, world_size)
            _assert_gather_and_reduce_scatter(rank, world_size)
            _assert_alltoall_variants(rank, world_size)

            pg = dist.distributed_c10d._get_default_group()
            return (funcol_all_reduce(torch.full((8,), float(rank + 1)), "sum", pg),)

        results = ex.run(worker)
        expected = torch.full((8,), float(sum(r + 1 for r in range(world_size))))
        for (tensor_out,) in results:
            assert type(tensor_out) is torch.Tensor
            torch.testing.assert_close(tensor_out, expected)


def test_cpu_2d_mesh_subgroups_and_parallelize_module() -> None:
    torch.manual_seed(0)
    ref_mlp = _TwoLayerMLP(dim=8)
    x = torch.randn(2, 4, 8)
    expected = ref_mlp(x).detach()

    with ParallelExecutor(["cpu"] * 4) as ex:
        def worker(rank: int) -> tuple[float, float, torch.Tensor]:
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
            dp_pg = mesh.get_group("dp")
            tp_pg = mesh.get_group("tp")

            g01 = dist.new_group([0, 1])
            g23 = dist.new_group([2, 3])
            pair_group = g01 if rank < 2 else g23
            pair_t = torch.tensor([float(rank + 1)])
            dist.all_reduce(pair_t, group=pair_group)

            dp_t = torch.tensor([float(rank + 1)])
            tp_t = torch.tensor([float(rank + 1)])
            w_dp = dist.all_reduce(dp_t, group=dp_pg, async_op=True)
            w_tp = dist.all_reduce(tp_t, group=tp_pg, async_op=True)
            assert w_dp is not None and w_tp is not None
            w_dp.wait()
            w_tp.wait()

            dt = DTensor.from_local(torch.full((2, 2), float(rank + 1)), mesh, [Shard(0), Shard(1)])
            rep = dt.redistribute(mesh, [Replicate(), Replicate()]).to_local()
            assert rep.shape == (4, 4)

            tp_mesh = mesh["tp"]
            local_model = copy.deepcopy(ref_mlp)
            _parallelize_mlp(local_model, tp_mesh)
            tp_local = local_model(x)
            full_out = DTensor.from_local(tp_local, tp_mesh, [Shard(1)]).full_tensor()
            return pair_t.item(), dp_t.item(), full_out.detach()

        results = ex.run(worker)
        assert [r[0] for r in results] == [3.0, 3.0, 7.0, 7.0]
        assert [r[1] for r in results] == [4.0, 6.0, 4.0, 6.0]
        for row in results:
            torch.testing.assert_close(row[2], expected, rtol=1e-5, atol=1e-5)


def test_poll_only_completion_and_returned_dtensor_materialization() -> None:
    with ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta(seconds=2.0)) as ex:
        def poll_worker(rank: int) -> tuple[float, DTensor]:
            t = torch.full((8,), float(rank + 1))
            work = dist.all_reduce(t, async_op=True)
            assert work is not None
            deadline = time.monotonic() + 2.0
            while not work.is_completed():
                assert time.monotonic() < deadline
                time.sleep(0.002)
            assert work.is_success()

            mesh = init_device_mesh("cpu", (2,))
            dt = DTensor.from_local(torch.full((2,), float(rank + 1)), mesh, [Shard(0)])
            async_dt = dt.redistribute(mesh, [Replicate()], async_op=True)
            assert isinstance(async_dt._local_tensor, AsyncCollectiveTensor)
            return float(t[0].item()), async_dt

        res = ex.run(poll_worker)
        assert [row[0] for row in res] == [3.0, 3.0]
        for _, mat_dt in res:
            assert type(mat_dt._local_tensor) is torch.Tensor
            torch.testing.assert_close(mat_dt.to_local(), torch.tensor([1.0, 1.0, 2.0, 2.0]))


def test_caught_pre_mutation_vs_one_way_poison_and_failed_work_weakrefs() -> None:
    with ParallelExecutor(["cpu", "cpu"]) as ex:
        bufs = [torch.full((4,), 7.0), torch.full((8,), 9.0)]

        def caught_shape_mismatch(rank: int) -> str:
            with pytest.raises(ValueError, match="shapes must match"):
                dist.all_reduce(bufs[rank])
            return "recovered"

        assert ex.run(caught_shape_mismatch) == ("recovered", "recovered")
        torch.testing.assert_close(bufs[0], torch.full((4,), 7.0))
        torch.testing.assert_close(bufs[1], torch.full((8,), 9.0))
        assert ex.is_open
        assert ex.run(_run_simple_allreduce) == (3.0, 3.0)

        # Completed Work preserves its terminal success across later poison.
        completed_work_box: list[dist.Work] = []

        def grab_completed_work(rank: int) -> None:
            w = dist.all_reduce(torch.full((4,), float(rank + 1)), async_op=True)
            assert w is not None
            w.wait()
            completed_work_box.append(w)

        ex.run(grab_completed_work)

        # Escaping validation error (plus divergent new_group) must poison one-way and drop peer refs.
        held_work: list[dist.Work] = []
        weak_rank1: list[weakref.ref[torch.Tensor]] = []

        def escaping_failure(rank: int) -> None:
            if rank == 0:
                _ = dist.new_group([0, 1])
                t0 = torch.ones(4)
                w0 = dist.all_reduce(t0, async_op=True)
                assert w0 is not None
                held_work.append(w0)
                w0.wait()
            else:
                t1 = torch.ones(8)
                weak_rank1.append(weakref.ref(t1))
                dist.all_reduce(t1)

        with pytest.raises(ValueError, match="shapes must match"):
            ex.run(escaping_failure)
        assert not ex.is_open
        assert completed_work_box[0].is_success()
        assert completed_work_box[0].wait()

        with pytest.raises(RuntimeError, match="poisoned"):
            ex.run(_run_simple_allreduce)

        gc.collect()
        # Retaining rank 0's failed Work must not keep rank 1's tensor alive.
        assert weak_rank1[0]() is None
        with pytest.raises(ValueError, match="shapes must match"):
            held_work[0].wait()
        held_work.clear()

    # Reopening after poison cleanly starts a fresh executor with aligned c10d group names.
    with ParallelExecutor(["cpu", "cpu"]) as fresh:
        def check_new_group_aligned(_rank: int) -> str:
            pg = dist.new_group([0, 1])
            t = torch.ones(2)
            dist.all_reduce(t, group=pg)
            return pg.group_name

        names = fresh.run(check_new_group_aligned)
        assert names[0] == names[1]


def _assert_subgroup_destroy_lifecycle() -> None:
    # Subgroup destroy: reject unfinished local work, permit idle early detach, retire on last detach.
    with ParallelExecutor(["cpu", "cpu"]) as ex3:
        rank0_checked = threading.Event()
        early_detached = threading.Event()

        def subgroup_lifecycle(rank: int) -> None:
            pg_a = dist.new_group([0, 1])
            t = torch.full((4,), float(rank + 1))
            if rank == 1:
                assert rank0_checked.wait(timeout=2.0)
            w = dist.all_reduce(t, group=pg_a, async_op=True)
            assert w is not None
            if rank == 0:
                try:
                    with pytest.raises(RuntimeError, match="unfinished collectives"):
                        dist.destroy_process_group(pg_a)
                finally:
                    rank0_checked.set()
            w.wait()
            torch.testing.assert_close(t, torch.full((4,), 3.0))
            dist.destroy_process_group(pg_a)

            if rank == 0:
                pg_b = dist.new_group([0, 1])
                dist.destroy_process_group(pg_b)
                early_detached.set()
            else:
                assert early_detached.wait(timeout=2.0)
                pg_b = dist.new_group([0, 1])
                with pytest.raises(RuntimeError, match="detached"):
                    dist.all_reduce(torch.ones(4), group=pg_b)
                dist.destroy_process_group(pg_b)

        ex3.run(subgroup_lifecycle)
        assert list(ex3._channels.keys()) == ["0"]


def test_sole_stalled_op_deadline_explicit_wait_clamp_and_subgroup_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Sole 1-rank op stalling in transport must time out at collective_timeout without peer waiters.
    release_exec = threading.Event()
    with ParallelExecutor(["cpu"], collective_timeout=timedelta(seconds=0.15)) as ex1:
        assert ex1._transport is not None
        orig_exec = ex1._transport.execute_collective

        def stalled_exec(*args: Any, **kwargs: Any) -> None:
            release_exec.wait(timeout=5.0)
            orig_exec(*args, **kwargs)

        monkeypatch.setattr(ex1._transport, "execute_collective", stalled_exec)
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="timed out"):
            ex1.run(lambda _r: dist.all_reduce(torch.ones(4)))
        assert time.monotonic() - start < 1.0
        release_exec.set()

    # 2. Explicit positive Work.wait(timeout=5s) and subgroup timeout are clamped to collective_timeout.
    with ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta(seconds=0.15)) as ex2:
        def explicit_wait_too_long(rank: int) -> None:
            pg = dist.new_group([0, 1], timeout=timedelta(seconds=5))
            if rank == 1:
                time.sleep(0.35)
                return
            w = pg.allreduce([torch.ones(4)])
            w.wait(timeout=timedelta(seconds=5))

        start = time.monotonic()
        with pytest.raises(TimeoutError, match="timed out"):
            ex2.run(explicit_wait_too_long)
        assert time.monotonic() - start < 1.0

    _assert_subgroup_destroy_lifecycle()


def test_repeated_sessions_restore_state_and_isolate_mesh_ids() -> None:
    original_world = dist.distributed_c10d._world
    original_hook = sys.excepthook
    original_mesh_init = DeviceMesh.__init__
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    original_autotune = CachingAutotuner.autotune_to_one_config
    mesh_ids: set[int] = set()

    for size in (2, 3):
        with ParallelExecutor(["cpu"] * size) as ex:
            def inspect_mesh(_rank: int, mesh_size: int = size) -> int:
                mesh = init_device_mesh("cpu", (mesh_size,))
                same = init_device_mesh("cpu", (mesh_size,))
                assert mesh == same and hash(mesh) == hash(same)
                assert mesh._thread_id is not None
                return int(mesh._thread_id)

            ids = ex.run(inspect_mesh)
            assert len(set(ids)) == size
            assert mesh_ids.isdisjoint(ids)
            mesh_ids.update(ids)

        assert dist.distributed_c10d._world is original_world
        assert sys.excepthook is original_hook
        assert DeviceMesh.__init__ is original_mesh_init
        assert CachingAutotuner.autotune_to_one_config is original_autotune
        assert not dist.is_initialized()


def _assert_sequential_retryable_restore(
    monkeypatch: pytest.MonkeyPatch,
    original_world: object,
) -> None:
    ex_s = SequentialExecutor().__enter__()
    seq_fail = True
    assert ex_s._runtime is not None
    orig_seq_exit = ex_s._runtime.__exit__

    def flaky_seq_exit(*args: Any) -> Any:
        if seq_fail:
            raise KeyboardInterrupt("injected seq restore interrupt")
        return orig_seq_exit(*args)

    monkeypatch.setattr(ex_s._runtime, "__exit__", flaky_seq_exit)
    with pytest.raises(KeyboardInterrupt, match="injected seq restore interrupt"):
        ex_s.close()
    assert _sequential_runtime.session_lock.locked()
    assert dist.distributed_c10d._world is not original_world
    seq_fail = False
    ex_s.close()
    assert not _sequential_runtime.session_lock.locked()
    assert dist.distributed_c10d._world is original_world


def test_retryable_runtime_isolation_and_reentry_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    original_world = dist.distributed_c10d._world
    original_mesh_init = DeviceMesh.__init__

    # Re-entering an already open executor must fail without closing the active session.
    with ParallelExecutor(["cpu", "cpu"]) as ex:
        with pytest.raises(RuntimeError, match="cannot be reopened"):
            ex.__enter__()
        assert ex.is_open
        assert ex.run(_run_simple_allreduce) == (3.0, 3.0)

    # Interrupted restore() on ParallelExecutor keeps session_lock held until retry succeeds.
    ex_p = ParallelExecutor(["cpu", "cpu"]).__enter__()
    fail_restore = True
    orig_restore = ex_p._runtime.restore

    def flaky_restore() -> None:
        if fail_restore:
            raise KeyboardInterrupt("injected restore interrupt")
        orig_restore()

    monkeypatch.setattr(ex_p._runtime, "restore", flaky_restore)
    with pytest.raises(KeyboardInterrupt, match="injected restore interrupt"):
        ex_p.close()
    assert _sequential_runtime.session_lock.locked()
    assert dist.distributed_c10d._world is not original_world
    fail_restore = False
    ex_p.close()
    assert not _sequential_runtime.session_lock.locked()
    assert dist.distributed_c10d._world is original_world
    assert DeviceMesh.__init__ is original_mesh_init

    # Partial failure during _set_thread_isolation_mode on ParallelExecutor restores globals.
    set_iso = _sequential_runtime._set_thread_isolation_mode

    def partial_iso(enabled: bool) -> None:
        set_iso(enabled)
        if enabled:
            raise RuntimeError("boom partial iso")

    with monkeypatch.context() as patch:
        patch.setattr(_sequential_runtime, "_set_thread_isolation_mode", partial_iso)
        with pytest.raises(RuntimeError, match="boom partial iso"):
            with ParallelExecutor(["cpu", "cpu"]):
                pass
    assert not _sequential_runtime.session_lock.locked()
    assert dist.distributed_c10d._world is original_world

    if torch.cuda.is_available():
        _assert_sequential_retryable_restore(monkeypatch, original_world)


def test_startup_failures_weakref_and_nesting_rejection() -> None:
    original_world = dist.distributed_c10d._world

    with ParallelExecutor(["cpu", "cpu"]) as ex:
        released = [threading.Event(), threading.Event()]
        values = ex.run(lambda _rank: torch.empty(256))
        refs = [weakref.ref(v, lambda _, ev=ev: ev.set()) for v, ev in zip(values, released, strict=True)]
        del values
        assert all(ev.wait(5) for ev in released)
        assert all(ref() is None for ref in refs)

        with pytest.raises(RuntimeError, match="already open"):
            with ParallelExecutor(["cpu", "cpu"]):
                pass
        with pytest.raises(RuntimeError, match="overlap or nest"):
            ex.run(lambda _rank: ex.run(lambda r: r))

    init_pg = dist.init_process_group
    with pytest.MonkeyPatch.context() as patch:
        def failing_worker(*args: Any, **kwargs: Any) -> Any:
            if kwargs["rank"] == 1:
                raise RuntimeError("boom worker")
            return init_pg(*args, **kwargs)

        patch.setattr(dist, "init_process_group", failing_worker)
        with pytest.raises(RuntimeError, match="boom worker"):
            with ParallelExecutor(["cpu", "cpu"]):
                pass
    assert dist.distributed_c10d._world is original_world

    with ParallelExecutor(["cpu", "cpu"]) as reopened:
        assert reopened.run(lambda r: r) == (0, 1)


def _assert_controller_interrupt_after_worker_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    with ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta(seconds=0.5)) as ex_kb:
        orig_await = ex_kb._await_batch_locked
        workers_claimed = threading.Barrier(2)
        worker_entered = threading.Event()
        entered_wait = threading.Event()
        late_write_done = threading.Event()
        target = torch.zeros(4)

        def interrupting_await(_batch: Any) -> Any:
            ex_kb._cv.release()
            try:
                assert worker_entered.wait(timeout=2.0)
            finally:
                ex_kb._cv.acquire()
            entered_wait.set()
            raise KeyboardInterrupt("controller ctrl-c")

        monkeypatch.setattr(ex_kb, "_await_batch_locked", interrupting_await)

        def worker_with_unmatched_op(rank: int) -> None:
            workers_claimed.wait(timeout=2.0)
            if rank == 0:
                worker_entered.set()
                assert entered_wait.wait(timeout=2.0)
                dist.all_reduce(target)
                late_write_done.set()

        with pytest.raises(KeyboardInterrupt, match="controller ctrl-c"):
            ex_kb.run(worker_with_unmatched_op)
        monkeypatch.setattr(ex_kb, "_await_batch_locked", orig_await)
        assert not late_write_done.is_set()
        torch.testing.assert_close(target, torch.zeros(4))
        assert not ex_kb.is_open


def test_collective_timeout_keeps_isolation_and_controller_interrupt_poisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_world = dist.distributed_c10d._world
    release = threading.Event()
    executor = ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta(seconds=0.15)).__enter__()

    def forward(rank: int) -> None:
        if rank == 1:
            release.wait(10)
        else:
            dist.all_reduce(torch.ones(2))

    try:
        with pytest.raises(TimeoutError, match="timed out"):
            executor.run(forward)
        with pytest.raises(RuntimeError, match="poisoned"):
            executor.run(lambda r: r)
        with pytest.raises(TimeoutError, match="isolation is still active"):
            executor.close()
        assert dist.distributed_c10d._world is not original_world
        with pytest.raises(RuntimeError, match="open and not closing"):
            executor.run(lambda r: r)
        with pytest.raises(RuntimeError, match="already open"):
            with ParallelExecutor(["cpu", "cpu"]):
                pass
    finally:
        release.set()
        executor.close()

    assert dist.distributed_c10d._world is original_world
    _assert_controller_interrupt_after_worker_claim(monkeypatch)


def _followup_broadcast_on_rank(rank: int, reduced: torch.Tensor) -> torch.Tensor:
    val = torch.full((4,), float(reduced[0].item()) + 10.0) if rank == 0 else torch.zeros(4)
    dist.broadcast(val, src=0)
    return val


def test_future_callback_can_issue_followup_collective_and_close_rejection() -> None:
    entered = threading.Barrier(2)
    release = threading.Event()
    both_entered = threading.Event()
    entered_count = 0
    entered_lock = threading.Lock()

    with ParallelExecutor(["cpu", "cpu"]) as ex:
        def worker(rank: int) -> float:
            tensor = torch.full((16,), float(rank + 1))
            work = dist.all_reduce(tensor, async_op=True)
            assert work is not None
            fut = work.get_future()
            chained = fut.then(lambda done: _followup_broadcast_on_rank(rank, done.wait()[0]))
            return float(chained.wait()[0].item())

        assert ex.run(worker) == (13.0, 13.0)

        def nested_close(rank: int) -> None:
            if rank == 0:
                with pytest.raises(RuntimeError, match="outside its rank callbacks"):
                    ex.close()

        ex.run(nested_close)

        concurrent_error: list[BaseException] = []

        def hold_run(rank: int) -> int:
            nonlocal entered_count
            entered.wait(timeout=5)
            with entered_lock:
                entered_count += 1
                if entered_count == 2:
                    both_entered.set()
            release.wait(timeout=5)
            return rank

        runner = threading.Thread(target=lambda: ex.run(hold_run))
        runner.start()
        try:
            assert both_entered.wait(timeout=5)
            try:
                ex.close()
            except BaseException as exc:
                concurrent_error.append(exc)
        finally:
            release.set()
            runner.join(timeout=5)

        assert len(concurrent_error) == 1
        assert isinstance(concurrent_error[0], RuntimeError)
        assert "while run is active" in str(concurrent_error[0])


class _SubclassTensor(torch.Tensor):
    pass


def _exercise_mapped_ro_subrange(ro: torch.Tensor, ro_sub: torch.Tensor) -> None:
    with ParallelExecutor(["cpu"], staging_bytes=256) as one_ex:
        one_ex.run(lambda _r: dist.all_reduce(ro))

    with ParallelExecutor(["cpu", "cpu"], staging_bytes=256) as ex:
        def check_views(rank: int) -> None:
            pg = dist.distributed_c10d._get_default_group()
            z = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j], dtype=torch.complex64)
            with pytest.raises(ValueError, match="negative or conjugate"):
                dist.all_reduce(z.conj().imag)
            with pytest.raises(ValueError, match="negative or conjugate"):
                pg.broadcast([z.conj()], dist.BroadcastOptions())
            with pytest.raises(ValueError, match="read-only mapped"):
                dist.all_reduce(ro)
            # Independently wrapped subrange over the same MappedFile must also be rejected!
            with pytest.raises(ValueError, match="read-only mapped"):
                dist.all_reduce(ro_sub)

            b_buf = ro if rank == 0 else torch.empty_like(ro)
            dist.broadcast(b_buf, src=0)
            torch.testing.assert_close(b_buf, torch.arange(16, dtype=torch.float32))

            # Shared read-only CPU input with exact self-aliases in disjoint remote slots must succeed.
            remote_out = torch.empty_like(ro)
            outs = [ro, remote_out] if rank == 0 else [remote_out, ro]
            dist.all_gather(outs, ro)
            torch.testing.assert_close(remote_out, torch.arange(16, dtype=torch.float32))

            sub = _SubclassTensor(torch.ones(4))
            with pytest.raises(TypeError, match="plain torch.Tensor"):
                pg.allreduce([sub])
            with pytest.raises(TypeError, match="split sizes must be int"):
                pg.all_to_all_single(torch.ones(4, 2), torch.ones(4, 2), [True, True], [2, 2])  # type: ignore[list-item]

            # Active torch.device("meta") around wait() must still keep scratch on CPU.
            meta_in = torch.full((8,), float(rank + 1))
            with torch.device("meta"):
                w_meta = dist.all_reduce(meta_in, async_op=True)
                assert w_meta is not None
                w_meta.wait()
            torch.testing.assert_close(meta_in, torch.full((8,), 3.0))

        ex.run(check_views)


def test_validation_subrange_scatter_alias_singleton_and_uneven_allgather(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt_path = tmp_path / "readonly.bin"
    raw = torch.arange(16, dtype=torch.float32).numpy().tobytes()
    ckpt_path.write_bytes(raw)
    f = ckpt_path.open("rb")
    mm = mmap.mmap(f.fileno(), len(raw), access=mmap.ACCESS_READ)
    mapped = MappedFile(f, mm)
    ro_tensor = mapped.tensor(0, len(raw), torch.float32, (16,))
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        ro_sub = torch.frombuffer(memoryview(mapped)[8:24], dtype=torch.float32)

    try:
        _exercise_mapped_ro_subrange(ro_tensor, ro_sub)
    finally:
        del ro_sub, ro_tensor, mapped
        gc.collect()
        mm.close()
        f.close()

    with ParallelExecutor(["cpu", "cpu"], staging_bytes=256) as ex:
        # CPU scatter self-alias collision: root sources [x=11, y=22], root out=x, rank 1 out=x.
        x = torch.tensor([11.0])
        y = torch.tensor([22.0])

        def scatter_collision(rank: int) -> None:
            with pytest.raises(ValueError, match="overlap"):
                dist.scatter(x, [x, y] if rank == 0 else None, src=0)

        ex.run(scatter_collision)
        assert float(x.item()) == 11.0

        # Contiguous stride > 1 singleton broadcast.
        def stride2_singleton(rank: int) -> float:
            src_or_dst = (
                torch.tensor([13.0, -9.0], dtype=torch.float32)[::2][:1]
                if rank == 0
                else torch.zeros(2, dtype=torch.float32)[::2][:1]
            )
            assert src_or_dst.is_contiguous() and src_or_dst.stride() == (2,)
            dist.broadcast(src_or_dst, src=0)
            return float(src_or_dst.item())

        assert ex.run(stride2_singleton) == (13.0, 13.0)

        # Uneven all_gather list + shape check.
        def check_uneven_allgather(rank: int) -> tuple[list[float], list[float]]:
            src = torch.tensor([1.0, 2.0]) if rank == 0 else torch.tensor([3.0, 4.0, 5.0])
            outs = [torch.empty(2), torch.empty(3)]
            dist.all_gather(outs, src)
            return outs[0].tolist(), outs[1].tolist()

        assert ex.run(check_uneven_allgather) == (([1.0, 2.0], [3.0, 4.0, 5.0]), ([1.0, 2.0], [3.0, 4.0, 5.0]))

        def bad_uneven_shapes(rank: int) -> None:
            src = torch.ones((2, 3)) if rank == 0 else torch.ones(4)
            outs = (
                [torch.empty((2, 3)), torch.empty((2, 2))]
                if rank == 0
                else [torch.empty((3, 2)), torch.empty(4)]
            )
            with pytest.raises(ValueError, match="output shapes must match"):
                dist.all_gather(outs, src)

        ex.run(bad_uneven_shapes)

    monkeypatch.setitem(dist.Backend._plugins, "PIPER_PARALLEL", object())  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="different backend"):
        with ParallelExecutor(["cpu", "cpu"]):
            pass


def test_inductor_disk_cache_isolates_ranks_across_subprocesses(tmp_path: Path) -> None:
    cache_dir = tmp_path / "inductor_cache"
    base_env = dict(os.environ)
    base_env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    base_env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    base_env["HIP_VISIBLE_DEVICES"] = ""
    base_env["CUDA_VISIBLE_DEVICES"] = ""

    # One identical script for every invocation: ranks use implicit Replicate -> Shard(0)
    # slicing across distinct session histories so persistent disk cache keys stay rank-isolated.
    shared_script = """
import json, os, torch
import torch._dynamo.utils as du
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from piper_offload.parallel import ParallelExecutor

if os.environ.get("PIPER_PRELUDE") == "1":
    with ParallelExecutor(["cpu", "cpu"]) as pre:
        pre.run(lambda r: r)

with ParallelExecutor(["cpu"] * 4) as ex:
    def worker(_rank):
        mesh = init_device_mesh("cpu", (4,))
        fn = torch.compile(
            lambda x: x.redistribute(placements=[Shard(0)]).to_local().clone(),
            fullgraph=True,
            options={"compile_threads": 1},
        )
        rep = DTensor.from_local(torch.arange(8, dtype=torch.float32), mesh, [Replicate()])
        return fn(rep).tolist()
    res = ex.run(worker)
print(json.dumps({"shards": res, "hits": du.counters["inductor"].get("fxgraph_cache_hit", 0)}))
"""

    def _run_case(prelude: str) -> dict[str, Any]:
        run_env = dict(base_env)
        run_env["PIPER_PRELUDE"] = prelude
        proc = subprocess.run(
            [sys.executable, "-c", shared_script],
            env=run_env,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return json.loads(proc.stdout.strip().splitlines()[-1])

    first = _run_case("0")
    assert first["shards"] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]

    second = _run_case("1")
    assert second["shards"] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]

    third = _run_case("1")
    assert third["shards"] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
    assert third["hits"] >= 4


def test_triton_optional_import_cpu_canonicalization_and_unwaited_poison(tmp_path: Path) -> None:
    script = """
import sys
sys.modules["triton"] = None
sys.modules["triton.language"] = None
import torch, torch.distributed as dist
from piper_offload.parallel import ParallelExecutor

with ParallelExecutor(["cpu:0", "cpu:1"], staging_bytes=128) as ex:
    assert ex.devices == (torch.device("cpu"), torch.device("cpu"))
    def worker(rank):
        t = torch.full((8,), float(rank + 1))
        dist.all_reduce(t)
        return float(t[0].item())
    assert ex.run(worker) == (3.0, 3.0)
print("OK")
"""
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = ""
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert proc.stdout.strip().endswith("OK")

    retained: list[dist.Work] = []
    with ParallelExecutor(["cpu", "cpu"]) as ex:
        def leave_unwaited(rank: int) -> None:
            if rank == 0:
                w_coll = dist.all_reduce(torch.ones(4), async_op=True)
                assert w_coll is not None
                retained.append(w_coll)

        with pytest.raises(RuntimeError, match="aborted at end of run"):
            ex.run(leave_unwaited)
        assert not ex.is_open
        with pytest.raises(RuntimeError, match="aborted at end of run"):
            retained[0].wait()


def test_rejects_preexisting_process_group() -> None:
    if not dist.is_gloo_available():
        pytest.skip("Gloo needed to set up pre-existing group")
    original_hook = sys.excepthook
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        with pytest.raises(RuntimeError, match="no existing distributed"):
            with ParallelExecutor(["cpu", "cpu"]):
                pass
    finally:
        dist.destroy_process_group()
        sys.excepthook = original_hook


# ----------------------------------------------------------------------
# CUDA / 2-GPU ROCm tests
# ----------------------------------------------------------------------


@CUDA
def test_cuda_single_gpu_multirank_aliases_handoff_and_mutual_exclusion() -> None:
    with ParallelExecutor(["cuda:0", "cuda:0"], transport="p2p", staging_bytes=256) as ex:
        assert ex.resolved_transport == "p2p"

        caller_stream = torch.cuda.Stream(device="cuda:0")
        inp = torch.zeros(1024, device="cuda:0")
        with torch.cuda.device("cuda:0"), torch.cuda.stream(caller_stream):
            torch.cuda._sleep(15_000_000)
            inp.fill_(7.0)
            seen = ex.run(lambda _rank: inp.clone().cpu())
        for v in seen:
            torch.testing.assert_close(v, torch.full((1024,), 7.0))

        vals = [torch.empty(8, device="cuda:0") for _ in range(2)]
        vals[0][:4].fill_(5.0)
        vals[1][4:].fill_(9.0)
        ex.run(lambda rank: dist.all_gather_into_tensor(vals[rank], vals[rank][4 * rank : 4 * (rank + 1)]))
        expected = torch.tensor([5.0] * 4 + [9.0] * 4, device="cuda:0")
        torch.testing.assert_close(vals[0], expected)
        torch.testing.assert_close(vals[1], expected)

        shared = torch.arange(8, device="cuda:0", dtype=torch.float32)
        snap = shared.clone()
        with pytest.raises(ValueError, match="overlap"):
            ex.run(lambda rank: dist.all_reduce(shared[rank : rank + 4]))
        torch.testing.assert_close(shared, snap)

        with pytest.raises(RuntimeError, match="already open"):
            with SequentialExecutor():
                pass

    with SequentialExecutor():
        with pytest.raises(RuntimeError, match="already open"):
            with ParallelExecutor(["cuda:0", "cuda:0"]):
                pass

    with ParallelExecutor(["cuda:0"], transport="p2p", staging_bytes=256) as ex1:
        def check_1rank_2d(_rank: int) -> torch.Tensor:
            src = torch.arange(6, device="cuda:0", dtype=torch.float32).reshape(2, 3)
            out = torch.empty((2, 3), device="cuda:0", dtype=torch.float32)
            dist.reduce_scatter_tensor(out, src)
            return out.cpu()

        (out1,) = ex1.run(check_1rank_2d)
        assert out1.shape == (2, 3)
        torch.testing.assert_close(out1, torch.arange(6, dtype=torch.float32).reshape(2, 3))


@CUDA
def test_cuda_rank_order_triton_fallback_and_foreground_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ParallelExecutor(["cuda:0", "cuda:0", "cuda:0"], transport="p2p", staging_bytes=256) as ex3:
        vals_by_rank = (
            torch.tensor([1e20], device="cuda:0", dtype=torch.float32),
            torch.tensor([-1e20], device="cuda:0", dtype=torch.float32),
            torch.tensor([1.0], device="cuda:0", dtype=torch.float32),
        )

        def check_rank_order_f32_and_f64(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
            t = vals_by_rank[rank].clone()
            dist.all_reduce(t)
            src_val = [1e20, -1e20, 1.0][rank]
            t64 = torch.tensor([src_val], device="cuda:0", dtype=torch.float64)
            dist.all_reduce(t64)
            rs_in = torch.full((3,), src_val, device="cuda:0", dtype=torch.float64)
            rs_out = torch.empty(1, device="cuda:0", dtype=torch.float64)
            dist.reduce_scatter_tensor(rs_out, rs_in)
            assert float(rs_out.item()) == 1.0
            return t.cpu(), t64.cpu()

        res = ex3.run(check_rank_order_f32_and_f64)
        for out32, out64 in res:
            assert float(out32.item()) == 1.0
            assert float(out64.item()) == 1.0

        def _bf16_allreduce_cpu(rank: int) -> torch.Tensor:
            t = torch.full((8,), float(rank + 1), device="cuda:0", dtype=torch.bfloat16)
            dist.all_reduce(t)
            return t.cpu()

        with monkeypatch.context() as patch:
            patch.setattr(_parallel_collectives, "_load_triton_sum", lambda: None)
            bf16_outs = ex3.run(_bf16_allreduce_cpu)
            for out in bf16_outs:
                torch.testing.assert_close(out.float(), torch.full((8,), 6.0))

    with ParallelExecutor(["cuda:0", "cuda:0"], transport="p2p", staging_bytes=8) as tiny_ex:
        b0 = [torch.full((4,), float(r + 1), device="cuda:0", dtype=torch.float32) for r in range(2)]
        b1 = [torch.full((4,), float(r + 10), device="cuda:0", dtype=torch.float64) for r in range(2)]
        snap0, snap1 = b0[0].clone(), b1[0].clone()
        with pytest.raises(ValueError, match="staging_bytes"):
            tiny_ex.run(lambda r: dist.distributed_c10d._get_default_group().allreduce_coalesced([b0[r], b1[r]]).wait())
        torch.testing.assert_close(b0[0], snap0)
        torch.testing.assert_close(b1[0], snap1)

    foreground_during_sync: list[dict[Any, int]] = []
    orig_sync_events = _parallel_collectives._sync_events

    def spy_sync_events(events: Any) -> None:
        with _transfer_priority._foreground.condition:
            foreground_during_sync.append(dict(_transfer_priority._foreground.counts))
        orig_sync_events(events)

    monkeypatch.setattr(_parallel_collectives, "_sync_events", spy_sync_events)
    with ParallelExecutor(["cuda:0", "cuda:0"], transport="p2p", staging_bytes=256) as ex_fg:
        ex_fg.run(lambda r: dist.all_reduce(torch.full((8,), float(r + 1), device="cuda:0")))
    assert foreground_during_sync and all(counts == {} for counts in foreground_during_sync)


@TWO_GPUS
def test_cuda_auto_transport_probe_and_strict_p2p_error(monkeypatch: pytest.MonkeyPatch) -> None:
    ex_auto = ParallelExecutor(["cuda:0", "cuda:1"], transport="auto")
    expected = (
        "p2p"
        if (torch.cuda.can_device_access_peer(0, 1) and torch.cuda.can_device_access_peer(1, 0))
        else "host_staged"
    )
    assert ex_auto.resolved_transport == expected

    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda _s, _d: False)
    fallback = ParallelExecutor(["cuda:0", "cuda:1"], transport="auto")
    assert fallback.resolved_transport == "host_staged"
    with pytest.raises(RuntimeError, match="peer-to-peer access unavailable"):
        ParallelExecutor(["cuda:0", "cuda:1"], transport="p2p")


def _two_gpu_collective_checks(rank: int, dev: torch.device) -> tuple[float, float]:
    f32 = torch.full((128,), float(rank + 1), device=dev, dtype=torch.float32)
    bf16 = torch.full((128,), float(rank + 1), device=dev, dtype=torch.bfloat16)
    fp16 = torch.full((128,), float(rank + 1), device=dev, dtype=torch.float16)
    i64 = torch.full((64,), rank + 2, device=dev, dtype=torch.int64)
    dist.all_reduce(f32)
    dist.all_reduce(bf16)
    dist.all_reduce(fp16)
    dist.all_reduce(i64)
    torch.testing.assert_close(f32, torch.full((128,), 3.0, device=dev))
    torch.testing.assert_close(bf16.float(), torch.full((128,), 3.0, device=dev))
    torch.testing.assert_close(fp16.float(), torch.full((128,), 3.0, device=dev))
    assert torch.equal(i64, torch.full((64,), 5, device=dev, dtype=torch.int64))

    rs_in = torch.cat([torch.full((64,), float(rank + 10)), torch.full((64,), float(rank + 20))]).to(dev)
    rs_out = torch.empty(64, device=dev)
    dist.reduce_scatter_tensor(rs_out, rs_in)
    expected_rs = 21.0 if rank == 0 else 41.0
    torch.testing.assert_close(rs_out, torch.full((64,), expected_rs, device=dev))

    ag_out = torch.empty(128, device=dev)
    dist.all_gather_into_tensor(ag_out, rs_out)
    torch.testing.assert_close(ag_out[:64], torch.full((64,), 21.0, device=dev))
    torch.testing.assert_close(ag_out[64:], torch.full((64,), 41.0, device=dev))

    in_splits = [4, 2] if rank == 0 else [6, 3]
    out_splits = [4, 6] if rank == 0 else [2, 3]
    a2a_in = torch.cat([torch.full((in_splits[d], 8), float(rank * 10 + d), device=dev) for d in (0, 1)])
    a2a_out = torch.empty((sum(out_splits), 8), device=dev)
    dist.all_to_all_single(a2a_out, a2a_in, out_splits, in_splits)
    torch.testing.assert_close(a2a_out[: out_splits[0]], torch.full((out_splits[0], 8), float(rank), device=dev))
    torch.testing.assert_close(
        a2a_out[out_splits[0] :],
        torch.full((out_splits[1], 8), float(10 + rank), device=dev),
    )

    b_t = torch.full((96,), 77.0 if rank == 1 else 0.0, device=dev)
    dist.broadcast(b_t, src=1)
    torch.testing.assert_close(b_t, torch.full((96,), 77.0, device=dev))
    sc_list = [torch.full((64,), float(r + 3), device=dev) for r in (0, 1)] if rank == 0 else None
    sc_out = torch.empty(64, device=dev)
    dist.scatter(sc_out, sc_list, src=0)
    torch.testing.assert_close(sc_out, torch.full((64,), float(rank + 3), device=dev))
    dist.barrier()
    return f32[0].item(), bf16[0].item()


@TWO_GPUS
@pytest.mark.parametrize(
    "transport",
    [pytest.param("p2p", marks=TWO_GPUS_P2P), "host_staged"],
)
def test_two_gpu_collectives_and_foreground_transfer(
    transport: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_foreground: list[dict[Any, int]] = []
    seen_during_acquire: list[dict[Any, int]] = []
    if transport == "host_staged":
        orig_acquire = _parallel_collectives.host_pin_manager.acquire
        orig_reduce_chunk = _parallel_collectives._ParallelTransport._host_reduce_chunk

        def wrapped_acquire(*args: Any, **kwargs: Any) -> Any:
            with _transfer_priority._foreground.condition:
                seen_during_acquire.append(dict(_transfer_priority._foreground.counts))
            return orig_acquire(*args, **kwargs)

        def wrapped_reduce_chunk(self: Any, *args: Any, **kwargs: Any) -> None:
            with _transfer_priority._foreground.condition:
                seen_foreground.append(dict(_transfer_priority._foreground.counts))
            orig_reduce_chunk(self, *args, **kwargs)

        monkeypatch.setattr(_parallel_collectives.host_pin_manager, "acquire", wrapped_acquire)
        monkeypatch.setattr(_parallel_collectives._ParallelTransport, "_host_reduce_chunk", wrapped_reduce_chunk)
    else:
        orig_sum = _parallel_collectives._load_triton_sum()
        assert orig_sum is not None

        def wrapped_sum(output: torch.Tensor, inputs: Any) -> None:
            with _transfer_priority._foreground.condition:
                seen_foreground.append(dict(_transfer_priority._foreground.counts))
            orig_sum(output, inputs)

        monkeypatch.setattr(_parallel_collectives, "_load_triton_sum", lambda: wrapped_sum)

    with ParallelExecutor(["cuda:0", "cuda:1"], transport=transport, staging_bytes=256) as ex:  # type: ignore[arg-type]
        assert ex.resolved_transport == transport
        res = ex.run(lambda rank: _two_gpu_collective_checks(rank, torch.device(f"cuda:{rank}")))
        assert res == ((3.0, 3.0), (3.0, 3.0))

    expected_foreground = {torch.device("cuda", 0): 1, torch.device("cuda", 1): 1}
    assert any(counts == expected_foreground for counts in seen_foreground)
    assert all(counts == {} for counts in seen_during_acquire)
    assert _transfer_priority._foreground.counts == {}


@TWO_GPUS_P2P
def test_two_gpu_p2p_default_stream_stall_coalesced_and_tiny_fp64_reduce_scatter() -> None:
    with ParallelExecutor(["cuda:0", "cuda:1"], transport="p2p", collective_timeout=timedelta(seconds=0.35)) as ex:
        barrier = threading.Barrier(2)

        def default_stream_stall_worker(rank: int) -> float:
            dev = torch.device(f"cuda:{rank}")
            t = torch.tensor([42.0 if rank == 0 else 0.0], device=dev)
            barrier.wait(timeout=5.0)
            if rank == 0:
                with torch.cuda.device(0), torch.cuda.stream(torch.cuda.default_stream(0)):
                    torch.cuda._sleep(500_000_000)
            dist.broadcast(t, src=0)
            return float(t.item())

        assert ex.run(default_stream_stall_worker) == (42.0, 42.0)

    # Clear default stream after timed sleep before next tests.
    torch.cuda.synchronize(0)

    with ParallelExecutor(["cuda:0", "cuda:1"], transport="p2p", staging_bytes=16) as tiny_ex:
        def fp64_rs_and_coalesced(rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            dev = torch.device(f"cuda:{rank}")
            pg = dist.distributed_c10d._get_default_group()

            rs64_in = torch.tensor([float(rank + 1), float(rank + 10)], device=dev, dtype=torch.float64)
            rs64_out = torch.empty(1, device=dev, dtype=torch.float64)
            dist.reduce_scatter_tensor(rs64_out, rs64_in)
            assert float(rs64_out.item()) == (3.0 if rank == 0 else 21.0)

            a0 = torch.full((12,), float(rank + 1), device=dev, dtype=torch.bfloat16)
            a1 = torch.full((4,), float(2 * (rank + 1)), device=dev, dtype=torch.float32)
            pg.allreduce_coalesced([a0, a1]).wait()

            rs_in0 = torch.cat([torch.full((6,), float(rank + 3), device=dev, dtype=torch.bfloat16) for _ in (0, 1)])
            rs_in1 = torch.cat([torch.full((2,), float(rank + 5), device=dev, dtype=torch.float32) for _ in (0, 1)])
            rs_out0 = torch.empty(6, device=dev, dtype=torch.bfloat16)
            rs_out1 = torch.empty(2, device=dev, dtype=torch.float32)
            pg.reduce_scatter_tensor_coalesced([rs_out0, rs_out1], [rs_in0, rs_in1]).wait()
            return a0.cpu(), rs_out0.cpu(), rs_out1.cpu()

        res = tiny_ex.run(fp64_rs_and_coalesced)
        for bf16_all, bf16_rs, f32_rs in res:
            torch.testing.assert_close(bf16_all.float(), torch.full((12,), 3.0))
            torch.testing.assert_close(bf16_rs.float(), torch.full((6,), 7.0))
            torch.testing.assert_close(f32_rs, torch.full((2,), 11.0))


@TWO_GPUS
def test_two_gpu_host_staged_pin_budgets_and_cuda_device_guard() -> None:
    original_budget = host_pin_manager.max_pinned_bytes
    host_pin_manager.clear()
    host_pin_manager.max_pinned_bytes = 0
    try:
        with ParallelExecutor(["cuda:0", "cuda:1"], transport="host_staged", staging_bytes=256) as ex:
            def _pin_fallback(rank: int) -> torch.Tensor:
                tensor = torch.full((128,), float(rank + 2), device=f"cuda:{rank}")
                dist.all_reduce(tensor)
                return tensor.cpu()

            outs = ex.run(_pin_fallback)
        torch.testing.assert_close(outs[0], torch.full((128,), 5.0))
        torch.testing.assert_close(outs[1], torch.full((128,), 5.0))
        assert host_pin_manager.stats.active_leases == 0
    finally:
        host_pin_manager.clear()
        host_pin_manager.max_pinned_bytes = original_budget

    # Partial pin budget pins only dma_base and keeps FP32 acc pageable; verify device="cpu" inside device("cuda:1").
    host_pin_manager.clear()
    try:
        page_size = mmap.PAGESIZE
        host_pin_manager.max_pinned_bytes = 5 * page_size
        seen_leases: list[tuple[int, int, int]] = []
        orig_acquire = _parallel_collectives.host_pin_manager.acquire

        def record_acquire(tensors: Any, *args: Any, **kwargs: Any) -> Any:
            lease = orig_acquire(tensors, *args, **kwargs)
            seen_leases.append((len(tensors), lease.registered_bytes, lease.pageable_bytes))
            return lease

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_parallel_collectives.host_pin_manager, "acquire", record_acquire)
            elems = page_size
            with ParallelExecutor(["cuda:0", "cuda:1"], transport="host_staged", staging_bytes=8 * page_size) as ex:
                def bf16_reduce(rank: int) -> torch.Tensor:
                    t = torch.full((elems,), float(rank + 1), device=f"cuda:{rank}", dtype=torch.bfloat16)
                    with torch.device("cuda:1"):
                        dist.all_reduce(t)
                    return t.cpu()

                res = ex.run(bf16_reduce)
                for out in res:
                    torch.testing.assert_close(out.float(), torch.full((elems,), 3.0))
            assert seen_leases == [(1, 4 * page_size, 0)]
    finally:
        host_pin_manager.clear()
        host_pin_manager.max_pinned_bytes = original_budget


@TWO_GPUS
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "transport",
    [pytest.param("p2p", marks=TWO_GPUS_P2P), "host_staged"],
)
def test_two_gpu_eager_and_compiled_tp_mlp(compiled: bool, transport: str) -> None:
    if compiled:
        torch._dynamo.reset()

    torch.manual_seed(42)
    ref = _TwoLayerMLP(dim=16).to("cuda:0")
    x = torch.randn(2, 4, 16, device="cuda:0")
    with torch.no_grad():
        expected = ref(x).cpu()

    with ParallelExecutor(["cuda:0", "cuda:1"], transport=transport) as ex:  # type: ignore[arg-type]
        def run_rank(rank: int) -> torch.Tensor:
            dev = torch.device(f"cuda:{rank}")
            mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
            local_model = copy.deepcopy(ref).to(dev)
            _parallelize_mlp(local_model, mesh)
            x_rank = x.to(dev)

            def step(inp: torch.Tensor) -> torch.Tensor:
                out_sharded = local_model(inp)
                return DTensor.from_local(out_sharded, mesh, [Shard(1)]).redistribute(mesh, [Replicate()]).to_local()

            fn = (
                torch.compile(step, fullgraph=True, options={"triton.cudagraphs": False, "compile_threads": 1})
                if compiled
                else step
            )
            with torch.no_grad():
                _ = fn(x_rank)
                out = fn(x_rank)
            return out.cpu()

        outs = ex.run(run_rank)
        torch.testing.assert_close(outs[0], expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(outs[1], expected, rtol=1e-4, atol=1e-4)


@TWO_GPUS
def test_two_gpu_shared_model_store_concurrent_offloader_streaming() -> None:
    torch.manual_seed(7)
    cpu_ref = _StreamableModel(width=8, num_blocks=2)
    proto_ptrs = {name: param.untyped_storage().data_ptr() for name, param in cpu_ref.named_parameters()}
    x_cpu = torch.randn(3, 8)
    with torch.no_grad():
        expected_sum = cpu_ref(x_cpu) * 2.0

    store = ModelStore.from_module(cpu_ref)
    config = OffloadConfig(block_paths=("blocks",), block_mode="streaming")

    shell_0 = _StreamableModel(width=8, num_blocks=2).to("meta")
    shell_1 = _StreamableModel(width=8, num_blocks=2).to("meta")
    offloader_0 = store.bind(shell_0, config)
    offloader_1 = store.bind(shell_1, config)

    for name, ptr in proto_ptrs.items():
        assert shell_0.get_parameter(name).untyped_storage().data_ptr() == ptr
        assert shell_1.get_parameter(name).untyped_storage().data_ptr() == ptr

    offloaders = (offloader_0, offloader_1)
    with ParallelExecutor(["cuda:0", "cuda:1"], transport="host_staged", staging_bytes=256) as ex:
        def rank_forward(rank: int) -> torch.Tensor:
            dev = torch.device(f"cuda:{rank}")
            offloader = offloaders[rank]
            with activated_model(offloader, dev) as model, torch.inference_mode():
                y = model(x_cpu.to(dev))
                dist.all_reduce(y)
                return y.cpu()

        outs = ex.run(rank_forward)
        torch.testing.assert_close(outs[0], expected_sum, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(outs[1], expected_sum, rtol=1e-5, atol=1e-5)


@TWO_GPUS
def test_two_gpu_close_retry_after_failed_dma_fence_retains_lease_until_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_pin_manager.clear()
    original_world = dist.distributed_c10d._world
    fail_sync = True
    fail_poll = True
    orig_sync = _parallel_collectives._sync_streams
    orig_done = _parallel_collectives._is_transfer_done

    def flaky_sync(streams: Any) -> None:
        if fail_sync:
            raise RuntimeError("injected stream sync failure")
        orig_sync(streams)

    def flaky_done(record: Any) -> bool:
        if fail_poll:
            return False
        return orig_done(record)

    monkeypatch.setattr(_parallel_collectives, "_sync_streams", flaky_sync)
    monkeypatch.setattr(_parallel_collectives, "_is_transfer_done", flaky_done)
    ex_drain = ParallelExecutor(
        ["cuda:0", "cuda:1"],
        transport="host_staged",
        staging_bytes=256,
        collective_timeout=timedelta(seconds=0.15),
    ).__enter__()
    try:
        with pytest.raises(RuntimeError, match="injected stream sync failure"):
            ex_drain.run(
                lambda rank: dist.broadcast(torch.full((64,), float(rank + 1), device=f"cuda:{rank}"), src=0),
            )
        assert not ex_drain.is_open
        assert host_pin_manager.stats.active_leases > 0

        with pytest.raises(RuntimeError, match="poisoned"):
            ex_drain.run(lambda _rank: dist.barrier())
        with pytest.raises(TimeoutError, match="did not drain"):
            ex_drain.close()
        assert host_pin_manager.stats.active_leases > 0
        assert ex_drain._runtime is not None
        assert dist.distributed_c10d._world is not original_world

        fail_sync = False
        fail_poll = False
        ex_drain.close()
        assert host_pin_manager.stats.active_leases == 0
        assert ex_drain._runtime is None
        assert dist.distributed_c10d._world is original_world
    finally:
        fail_sync = False
        fail_poll = False
        if ex_drain._runtime is not None:
            ex_drain.close()
        host_pin_manager.clear()
    assert parallel_mod._state.active_executor is None


def test_retained_pg_admission_and_clean_reopen() -> None:
    with pytest.raises(ValueError, match="at most 365 days"):
        ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta.max)

    with ParallelExecutor(["cpu", "cpu"]) as ex:
        pgs = list(ex.run(lambda _r: dist.distributed_c10d._get_default_group()))
        stale_buf = torch.full((4,), 50.0)
        with pytest.raises(RuntimeError, match="active ParallelExecutor.run"):
            dist.all_reduce(stale_buf, group=pgs[0], async_op=True)
        torch.testing.assert_close(stale_buf, torch.full((4,), 50.0))

        stray_errors: list[BaseException] = []
        in_callback = threading.Event()
        rank1_gate = threading.Event()

        def stray_caller() -> None:
            try:
                assert in_callback.wait(timeout=2.0)
                pgs[0].allreduce([torch.full((4,), 50.0)])
            except BaseException as exc:
                stray_errors.append(exc)
            finally:
                rank1_gate.set()

        def only_rank1(rank: int) -> torch.Tensor:
            if rank == 0:
                in_callback.set()
                t = torch.full((4,), 1.0)
                dist.all_reduce(t)
                return t
            assert rank1_gate.wait(timeout=2.0)
            t = torch.full((4,), 2.0)
            dist.all_reduce(t)
            return t

        t = threading.Thread(target=stray_caller)
        t.start()
        try:
            outs = ex.run(only_rank1)
        finally:
            t.join(timeout=5.0)
        assert len(stray_errors) == 1 and isinstance(stray_errors[0], RuntimeError)
        assert "assigned rank worker thread" in str(stray_errors[0])
        for out in outs:
            torch.testing.assert_close(out, torch.full((4,), 3.0))

    pgs[0]._channel._detached_ranks.clear()  # Simulate teardown aborted before group shutdown.
    with ex:
        pgs[0].shutdown()
        assert 0 not in pgs[0]._channel._detached_ranks
        pgs[0].abort()  # Stale process group handle must not poison a reopened session.
        assert ex.is_open

        def reject_stale_handle(rank: int) -> None:
            if rank == 0:
                with pytest.raises(RuntimeError, match="handle has been destroyed"):
                    pgs[0].allreduce([torch.full((2,), 9.0)])

        ex.run(reject_stale_handle)
        assert ex.run(_run_simple_allreduce) == (3.0, 3.0)


def test_poll_only_deadlines_and_caught_wait_timeout() -> None:
    release_delay = threading.Event()
    with ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta(seconds=0.1)) as ex_delay:
        captured: list[dist.Work] = []

        def delayed_poll(rank: int) -> None:
            if rank == 1:
                release_delay.wait(timeout=1.0)
                return
            w = dist.all_reduce(torch.ones(4), async_op=True)
            assert w is not None
            captured.append(w)
            while not w.is_completed():
                time.sleep(0.002)

        try:
            start = time.monotonic()
            with pytest.raises(TimeoutError, match="timed out"):
                ex_delay.run(delayed_poll)
            assert time.monotonic() - start < 0.8
            assert captured[0].is_completed() and not captured[0].is_success()
        finally:
            release_delay.set()

    with ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta(seconds=0.1)) as ex_miss:
        def missing_poll(rank: int) -> None:
            if rank == 0:
                w = dist.all_reduce(torch.ones(4), async_op=True)
                assert w is not None
                while not w.is_completed():
                    time.sleep(0.002)

        with pytest.raises(TimeoutError, match="timed out"):
            ex_miss.run(missing_poll)

    with ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta(seconds=0.1)) as ex_caught:
        release_rank1 = threading.Event()

        def caught_wait(rank: int) -> None:
            if rank == 1:
                release_rank1.wait(timeout=2.0)
                return
            w = dist.all_reduce(torch.ones(4), async_op=True)
            assert w is not None
            with contextlib.suppress(TimeoutError):
                w.wait()

        try:
            start = time.monotonic()
            with pytest.raises(TimeoutError, match="timed out"):
                ex_caught.run(caught_wait)
            assert time.monotonic() - start < 0.8
        finally:
            release_rank1.set()


def test_running_op_late_completion_stays_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()
    work_box: list[dist.Work] = []
    with ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta(seconds=0.1)) as ex:
        assert ex._transport is not None
        orig_reduce = ex._transport._cpu_reduce_many

        def gated_reduce(*args: Any, **kwargs: Any) -> None:
            gate.wait(timeout=2.0)
            orig_reduce(*args, **kwargs)

        monkeypatch.setattr(ex._transport, "_cpu_reduce_many", gated_reduce)

        def submit_and_wait(rank: int) -> None:
            w = dist.all_reduce(torch.full((4,), float(rank + 1)), async_op=True)
            assert w is not None
            if rank == 0:
                work_box.append(w)
            w.wait()

        try:
            with pytest.raises(TimeoutError, match="timed out"):
                ex.run(submit_and_wait)
            assert not work_box[0].is_success()
            gate.set()
            deadline = time.monotonic() + 2.0
            while not work_box[0].is_completed():
                assert time.monotonic() < deadline
                time.sleep(0.002)
            assert not work_box[0].is_success()
            with pytest.raises(TimeoutError, match="timed out"):
                work_box[0].wait()
        finally:
            gate.set()


def _check_single_shapes(rank: int) -> None:
    ag_bad_in = torch.ones((2, 3)) if rank == 0 else torch.ones((3, 2))
    ag_bad_out = torch.empty((4, 3)) if rank == 0 else torch.empty((6, 2))
    with pytest.raises(ValueError, match="output shapes must match"):
        dist.all_gather_into_tensor(ag_bad_out, ag_bad_in)
    with pytest.raises(ValueError, match="must match stack"):
        dist.all_gather_into_tensor(torch.empty(12), ag_bad_in)

    rs_bad_in = torch.ones((4, 3)) if rank == 0 else torch.ones((6, 2))
    rs_bad_out = torch.empty((2, 3)) if rank == 0 else torch.empty((3, 2))
    with pytest.raises(ValueError, match="input shapes must match"):
        dist.reduce_scatter_tensor(rs_bad_out, rs_bad_in)
    with pytest.raises(ValueError, match="must match stack"):
        dist.reduce_scatter_tensor(rs_bad_out, torch.ones(12))

    scalar_out = torch.empty(2)
    dist.all_gather_into_tensor(scalar_out, torch.tensor(float(rank + 5)))
    torch.testing.assert_close(scalar_out, torch.tensor([5.0, 6.0]))


def test_dtensor_autograd_block_stream_single_shapes_errors_and_indices() -> None:
    with pytest.raises(ValueError, match="out of range"):
        ParallelExecutor([256])
    with pytest.raises(ValueError, match="out of range"):
        ParallelExecutor(["cuda:257"])
    with pytest.raises(ValueError, match="out of range"):
        ParallelExecutor(["cuda:-1"])

    with ParallelExecutor(["cpu", "cpu"]) as ex:
        def make_async_grad_dtensor(rank: int) -> tuple[DTensor, torch.Tensor]:
            mesh = init_device_mesh("cpu", (2,))
            local = torch.full((2,), float(rank + 1), requires_grad=True)
            dt = DTensor.from_local(local, mesh, [Shard(0)])
            async_dt = dt.redistribute(mesh, [Replicate()], async_op=True)
            return async_dt, local

        res = ex.run(make_async_grad_dtensor)
        for rank, (mat_dt, leaf) in enumerate(res):
            assert mat_dt.requires_grad and mat_dt.grad_fn is not None
            assert type(mat_dt._local_tensor) is torch.Tensor
            (mat_dt.to_local() * torch.tensor([1.0, 2.0, 3.0, 4.0])).sum().backward()
            expected_grad = [1.0, 2.0] if rank == 0 else [3.0, 4.0]
            assert leaf.grad is not None and leaf.grad.tolist() == expected_grad

        checked_block = threading.Event()

        def check_block_stream(rank: int) -> None:
            if rank == 0:
                w = dist.all_reduce(torch.ones(2), async_op=True)
                assert w is not None
                try:
                    with pytest.raises(NotImplementedError, match="block_current_stream"):
                        w.block_current_stream()
                finally:
                    checked_block.set()
                w.wait()
            else:
                assert checked_block.wait(timeout=2.0)
                w = dist.all_reduce(torch.ones(2), async_op=True)
                assert w is not None
                w.wait()

        ex.run(check_block_stream)
        ex.run(_check_single_shapes)
        assert ex.is_open

        class _UnprintableError(RuntimeError):
            def __str__(self) -> str:
                raise RuntimeError("broken __str__")

        def raise_unprintable(rank: int) -> None:
            if rank == 0:
                raise _UnprintableError()

        with pytest.raises(_UnprintableError):
            ex.run(raise_unprintable)


class _InterruptReleaseLock:
    def __init__(self, inner: threading.Lock) -> None:
        self._inner = inner
        self.fail_mode: str | None = None

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._inner.acquire(blocking, timeout)

    def release(self) -> None:
        if self.fail_mode == "before":
            raise KeyboardInterrupt("injected pre-release interrupt")
        self._inner.release()
        if self.fail_mode == "after":
            raise KeyboardInterrupt("injected post-release interrupt")

    def locked(self) -> bool:
        return self._inner.locked()

    def __enter__(self) -> bool:
        return self._inner.acquire()

    def __exit__(self, *args: object) -> None:
        self._inner.release()


def test_mapped_file_snapshot_and_session_lease_interrupt_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from piper_offload import _mapped_file

    ckpt_path = tmp_path / "snap.bin"
    ckpt_path.write_bytes(torch.arange(8, dtype=torch.float32).numpy().tobytes())
    with ckpt_path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 32, access=mmap.ACCESS_READ)
        try:
            mapped = MappedFile(f, mm)
            probe = torch.empty(4, dtype=torch.float32)

            class _MutatingEntry:
                @property
                def mmap(self) -> Any:
                    _mapped_file._mapping_ranges.pop(-999999, None)
                    return type("_Closed", (), {"closed": True})()

            with _mapped_file._lock:
                _mapped_file._mapping_ranges[-999998] = (
                    _MutatingEntry,  # type: ignore[dict-item]
                    probe.data_ptr(),
                    probe.data_ptr() + 16,
                )
                _mapped_file._mapping_ranges[-999999] = (
                    weakref.ref(mapped),
                    probe.data_ptr(),
                    probe.data_ptr() + 16,
                )
            try:
                assert _mapped_file.overlaps_mapped_file(probe)
            finally:
                with _mapped_file._lock:
                    _mapped_file._mapping_ranges.pop(-999999, None)
                    _mapped_file._mapping_ranges.pop(-999998, None)
        finally:
            del mapped
            gc.collect()
            mm.close()

    proxy_lock = _InterruptReleaseLock(_sequential_runtime.session_lock)
    monkeypatch.setattr(_sequential_runtime, "session_lock", proxy_lock)
    ex = ParallelExecutor(["cpu", "cpu"]).__enter__()
    try:
        proxy_lock.fail_mode = "before"
        with pytest.raises(KeyboardInterrupt, match="pre-release interrupt"):
            ex.close()
        assert _sequential_runtime.owns_session(ex)
        assert proxy_lock.locked()

        proxy_lock.fail_mode = "after"
        with pytest.raises(KeyboardInterrupt, match="post-release interrupt"):
            ex.close()
        assert _sequential_runtime.owns_session(ex)
        with pytest.raises(RuntimeError, match="already open"):
            with ParallelExecutor(["cpu", "cpu"]):
                pass

        proxy_lock.fail_mode = None
        ex.close()
        assert not _sequential_runtime.owns_session(ex)
        assert not proxy_lock.locked()
    finally:
        proxy_lock.fail_mode = None
        if _sequential_runtime.owns_session(ex) or ex._runtime is not None:
            ex.close()


@CUDA
def test_cuda_entry_events_preserved_on_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[float] = []
    worker_claimed = threading.Event()
    controller_cleared = threading.Event()
    worker_done = threading.Event()
    with ParallelExecutor(["cuda:0"], collective_timeout=timedelta(seconds=0.15)) as ex:
        orig_exec = ex._execute_worker_task

        def pause_after_claim(slot: Any, fn: Any, entry_events: Any) -> Any:
            worker_claimed.set()
            try:
                assert controller_cleared.wait(timeout=5.0)
                return orig_exec(slot, fn, entry_events)
            finally:
                worker_done.set()

        def interrupt_controller(_batch: Any) -> Any:
            ex._cv.release()
            try:
                assert worker_claimed.wait(timeout=5.0)
            finally:
                ex._cv.acquire()
            raise KeyboardInterrupt("controller ctrl-c")

        monkeypatch.setattr(ex, "_execute_worker_task", pause_after_claim)
        monkeypatch.setattr(ex, "_await_batch_locked", interrupt_controller)
        inp = torch.zeros(1024, device="cuda:0")
        caller_stream = torch.cuda.Stream(device="cuda:0")
        try:
            with torch.cuda.device("cuda:0"), torch.cuda.stream(caller_stream):
                torch.cuda._sleep(80_000_000)
                inp.fill_(9.0)
                with pytest.raises(KeyboardInterrupt, match="controller ctrl-c"):
                    ex.run(lambda _r: observed.append(float(inp[0].item())))
        finally:
            controller_cleared.set()
            assert worker_done.wait(timeout=5.0)
    assert observed == [9.0]


def _assert_concurrent_sequential_enter_close(monkeypatch: pytest.MonkeyPatch) -> None:
    ex_race = SequentialExecutor("cuda:0")
    orig_isolate = _sequential_runtime.isolate_runtime
    entered_iso = threading.Event()
    release_iso = threading.Event()

    def paused_isolate(*args: Any, **kwargs: Any) -> Any:
        entered_iso.set()
        assert release_iso.wait(timeout=5.0)
        return orig_isolate(*args, **kwargs)

    monkeypatch.setattr(_sequential_runtime, "isolate_runtime", paused_isolate)
    closer_errors: list[BaseException] = []
    opener = threading.Thread(target=ex_race.__enter__)
    opener.start()
    try:
        assert entered_iso.wait(timeout=5.0)
        try:
            ex_race.close()
        except BaseException as exc:
            closer_errors.append(exc)
    finally:
        release_iso.set()
        opener.join(timeout=5.0)
        if _sequential_runtime.owns_session(ex_race) or ex_race._runtime is not None:
            ex_race.close()
    assert len(closer_errors) == 1 and isinstance(closer_errors[0], RuntimeError)
    monkeypatch.setattr(_sequential_runtime, "isolate_runtime", orig_isolate)


@CUDA
def test_sequential_close_retries_sync_and_session_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    original_world = dist.distributed_c10d._world
    proxy_lock = _InterruptReleaseLock(_sequential_runtime.session_lock)
    monkeypatch.setattr(_sequential_runtime, "session_lock", proxy_lock)
    _assert_concurrent_sequential_enter_close(monkeypatch)

    ex_s = SequentialExecutor("cuda:0").__enter__()
    try:
        ex_s.run(lambda _r: dist.all_reduce(torch.ones(4, device="cuda:0")))
        assert ex_s._coordinator is not None
        fail_sync = True
        orig_sync = ex_s._coordinator.stream.synchronize

        def flaky_sync() -> None:
            if fail_sync:
                raise KeyboardInterrupt("injected seq sync interrupt")
            orig_sync()

        monkeypatch.setattr(ex_s._coordinator.stream, "synchronize", flaky_sync)
        proxy_lock.fail_mode = "before"
        with pytest.raises(KeyboardInterrupt, match="injected seq sync interrupt"):
            ex_s.close()
        assert ex_s._runtime is not None and ex_s._coordinator is not None
        assert proxy_lock.locked()
        assert dist.distributed_c10d._world is not original_world

        fail_sync = False
        with pytest.raises(KeyboardInterrupt, match="pre-release interrupt"):
            ex_s.close()
        assert _sequential_runtime.owns_session(ex_s) and proxy_lock.locked()
        assert dist.distributed_c10d._world is original_world

        proxy_lock.fail_mode = None
        ex_s.close()
        assert not proxy_lock.locked() and not _sequential_runtime.owns_session(ex_s)
    finally:
        fail_sync = False
        proxy_lock.fail_mode = None
        if _sequential_runtime.owns_session(ex_s) or ex_s._runtime is not None or ex_s._coordinator is not None:
            ex_s.close()


@TWO_GPUS
def test_two_gpu_fn_raise_and_retry_lease_close(monkeypatch: pytest.MonkeyPatch) -> None:
    host_pin_manager.clear()
    with ParallelExecutor(["cuda:0", "cuda:1"], transport="host_staged", staging_bytes=256) as ex:
        def boom_copy(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("injected copy failure")

        monkeypatch.setattr(_parallel_collectives._ParallelTransport, "_run_host_copy_chunks", boom_copy)
        with pytest.raises(RuntimeError, match="injected copy failure"):
            ex.run(lambda r: dist.broadcast(torch.full((16,), float(r), device=f"cuda:{r}"), src=0))
        assert host_pin_manager.stats.active_leases == 0
    assert host_pin_manager.stats.active_leases == 0

    monkeypatch.undo()
    orig_sync = _parallel_collectives._sync_streams
    orig_close_lease = host_pin_manager._close_lease
    fail_sync = True
    fail_close = True

    def flaky_sync(streams: Any) -> None:
        if fail_sync:
            raise RuntimeError("injected sync fail")
        orig_sync(streams)

    def flaky_manager_close_lease(key: int) -> None:
        if fail_close:
            raise KeyboardInterrupt("injected pin manager lock interrupt")
        orig_close_lease(key)

    monkeypatch.setattr(_parallel_collectives, "_sync_streams", flaky_sync)
    monkeypatch.setattr(host_pin_manager, "_close_lease", flaky_manager_close_lease)
    ex_retry = ParallelExecutor(
        ["cuda:0", "cuda:1"],
        transport="host_staged",
        staging_bytes=256,
        collective_timeout=timedelta(seconds=0.5),
    ).__enter__()
    try:
        with pytest.raises(RuntimeError, match="injected sync fail"):
            ex_retry.run(lambda r: dist.broadcast(torch.full((16,), float(r), device=f"cuda:{r}"), src=0))
        assert ex_retry._transport is not None and ex_retry._transport._pending is not None
        lease = ex_retry._transport._pending.lease
        assert lease is not None and not lease.closed
        assert host_pin_manager.stats.active_leases > 0
        fail_sync = False
        with pytest.raises(KeyboardInterrupt, match="injected pin manager lock interrupt"):
            ex_retry.close()
        assert not lease.closed
        assert host_pin_manager.stats.active_leases > 0
        fail_close = False
        ex_retry.close()
        assert lease.closed and host_pin_manager.stats.active_leases == 0
    finally:
        fail_sync = False
        fail_close = False
        if _sequential_runtime.owns_session(ex_retry) or ex_retry._runtime is not None:
            ex_retry.close()
        host_pin_manager.clear()


@TWO_GPUS
def test_two_gpu_uneven_all_gather_drops_stage_before_next_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    host_pin_manager.clear()
    observed_active_on_acquire: list[int] = []
    stage_refs: list[weakref.ref[torch.Tensor]] = []
    orig_acquire = _parallel_collectives.host_pin_manager.acquire
    orig_checkout = _parallel_collectives._HostStage.checkout
    orig_empty = torch.empty

    def check_checkout(self: Any, min_bytes: int, max_bytes: int) -> torch.Tensor:
        def checked_empty(*args: Any, **kwargs: Any) -> torch.Tensor:
            if stage_refs:
                assert stage_refs[-1]() is None
            return orig_empty(*args, **kwargs)

        with monkeypatch.context() as empty_patch:
            empty_patch.setattr(_parallel_collectives.torch, "empty", checked_empty)
            buf = orig_checkout(self, min_bytes, max_bytes)
        stage_refs.append(weakref.ref(buf))
        return buf

    def check_acquire(*args: Any, **kwargs: Any) -> Any:
        observed_active_on_acquire.append(host_pin_manager.stats.active_leases)
        return orig_acquire(*args, **kwargs)

    monkeypatch.setattr(_parallel_collectives._HostStage, "checkout", check_checkout)
    monkeypatch.setattr(_parallel_collectives.host_pin_manager, "acquire", check_acquire)
    with ParallelExecutor(["cuda:0", "cuda:1"], transport="host_staged", staging_bytes=1024) as ex_ag:
        def uneven_ag(rank: int) -> None:
            src = (
                torch.full((250,), 1.0, device="cuda:0")
                if rank == 0
                else torch.full((256,), 2.0, device="cuda:1")
            )
            outs = [torch.empty(250, device=f"cuda:{rank}"), torch.empty(256, device=f"cuda:{rank}")]
            dist.all_gather(outs, src)
            torch.testing.assert_close(outs[0].cpu(), torch.full((250,), 1.0))
            torch.testing.assert_close(outs[1].cpu(), torch.full((256,), 2.0))

        ex_ag.run(uneven_ag)
    assert observed_active_on_acquire == [0, 0]
    assert len(stage_refs) == 2
    assert host_pin_manager.stats.active_leases == 0


def test_alltoall_single_shapes_and_worker_finish_idempotency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ParallelExecutor(["cpu", "cpu"]) as ex:
        bufs = [
            torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
            torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
        ]
        snaps = [b.clone() for b in bufs]

        def bad_a2a(rank: int) -> None:
            out = torch.empty_like(bufs[rank])
            with pytest.raises(ValueError, match="slice shapes must agree"):
                dist.all_to_all_single(out, bufs[rank])

        ex.run(bad_a2a)
        for buf, snap in zip(bufs, snaps, strict=True):
            torch.testing.assert_close(buf, snap)
        assert ex.is_open and ex.run(_run_simple_allreduce) == (3.0, 3.0)

    with ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta(seconds=0.5)) as ex2:
        orig_handle = ex2._handle_worker_result
        rank1_entered = threading.Event()
        rank1_done = threading.Event()
        rank0_faulted = threading.Event()

        def fault_after_handle(slot: Any, batch: Any, out: Any, exc_caught: Any) -> None:
            orig_handle(slot, batch, out, exc_caught)
            if slot.rank == 0:
                rank0_faulted.set()
                raise RuntimeError("post-handle fault")

        monkeypatch.setattr(ex2, "_handle_worker_result", fault_after_handle)

        def worker_double(rank: int) -> None:
            if rank == 1:
                rank1_entered.set()
                assert rank0_faulted.wait(timeout=2.0)
                time.sleep(0.05)
                rank1_done.set()
            else:
                assert rank1_entered.wait(timeout=2.0)

        with pytest.raises(RuntimeError, match="post-handle fault"):
            ex2.run(worker_double)
        assert rank1_done.is_set()


def test_expired_slot_skipped_and_first_failure_time_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta(seconds=0.08)) as ex3:
        assert ex3._transport is not None
        executed: list[bool] = []
        orig_exec = ex3._transport.execute_collective
        orig_await = ex3._await_batch_locked

        def spy_exec(*args: Any, **kwargs: Any) -> None:
            executed.append(True)
            orig_exec(*args, **kwargs)

        def delayed_await(batch: Any) -> Any:
            time.sleep(0.12)
            return orig_await(batch)

        monkeypatch.setattr(ex3._transport, "execute_collective", spy_exec)
        monkeypatch.setattr(ex3, "_await_batch_locked", delayed_await)

        def late_submit(rank: int) -> None:
            if rank == 1:
                time.sleep(0.1)
            dist.all_reduce(torch.ones(2))

        with pytest.raises(TimeoutError, match="timed out"):
            ex3.run(late_submit)
        assert executed == []

    monkeypatch.undo()
    observed_failure_times: list[float] = []
    with ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta(seconds=0.2)) as ex4:
        orig_poison = ex4._poison_locked
        got_poison = threading.Event()

        def spy_poison(summary: Any) -> None:
            orig_poison(summary)
            if ex4._current_batch is not None and ex4._current_batch.failure_time is not None:
                observed_failure_times.append(ex4._current_batch.failure_time)
                got_poison.set()

        monkeypatch.setattr(ex4, "_poison_locked", spy_poison)

        def preserve_timestamp(rank: int) -> None:
            if rank == 0:
                w = dist.all_reduce(torch.ones(2), async_op=True)
                assert w is not None
                w.wait(timeout=timedelta(seconds=0.04))
            else:
                assert got_poison.wait(timeout=2.0)
                time.sleep(0.05)
                raise RuntimeError("later worker error")

        with pytest.raises(TimeoutError, match="timed out"):
            ex4.run(preserve_timestamp)
        assert len(set(observed_failure_times)) == 1


def test_startup_gate_cancels_late_worker_c10d_init(monkeypatch: pytest.MonkeyPatch) -> None:
    gate_cls = _sequential_runtime.StartupGate
    orig_try_admit = gate_cls.try_admit
    orig_init_pg = dist.init_process_group
    orig_wait = threading.Condition.wait
    controller = threading.current_thread()
    rollback_done = threading.Event()
    workers_entered = threading.Event()
    fired = threading.Event()
    late_rank1_admitted: list[bool] = []
    late_c10d_called = threading.Event()

    def gated_try_admit(self: Any) -> bool:
        if threading.current_thread().name.endswith("-1"):
            workers_entered.set()
            assert rollback_done.wait(timeout=5.0)
            ok = orig_try_admit(self)
            late_rank1_admitted.append(ok)
            return ok
        return orig_try_admit(self)

    def interrupt_startup_wait(self: threading.Condition, timeout: float | None = None) -> bool:
        if threading.current_thread() is controller and not fired.is_set() and workers_entered.is_set():
            fired.set()
            raise KeyboardInterrupt("injected startup abort")
        return orig_wait(self, timeout=timeout)

    def guarded_init_pg(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("rank") == 1:
            late_c10d_called.set()
            return orig_init_pg(*args, **kwargs)
        pg = orig_init_pg(*args, **kwargs)
        assert workers_entered.wait(timeout=5.0)
        return pg

    monkeypatch.setattr(gate_cls, "try_admit", gated_try_admit)
    monkeypatch.setattr(threading.Condition, "wait", interrupt_startup_wait)
    monkeypatch.setattr(dist, "init_process_group", guarded_init_pg)
    ex = ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta(seconds=0.15))
    try:
        with pytest.raises(TimeoutError, match="did not exit"):
            ex.__enter__()
        assert _sequential_runtime.owns_session(ex) and _sequential_runtime.session_lock.locked()
    finally:
        rollback_done.set()
        monkeypatch.undo()
        ex.close()
    assert not _sequential_runtime.owns_session(ex)
    assert not _sequential_runtime.session_lock.locked()
    assert late_rank1_admitted == [False]
    assert not late_c10d_called.is_set()


@CUDA
def test_cuda_close_drains_compute_stream_after_pre_drain_worker_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[torch.cuda.Event] = []
    with ParallelExecutor(["cuda:0"], collective_timeout=timedelta(seconds=5.0)) as ex:
        def boom_drain(_slot: Any) -> None:
            raise RuntimeError("failed before _drain_rank_stream")

        monkeypatch.setattr(ex, "_drain_rank_stream", boom_drain)

        def queue_gpu_and_finish(_rank: int) -> None:
            torch.cuda._sleep(40_000_000)
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream("cuda:0"))
            recorded.append(ev)

        with pytest.raises(RuntimeError, match="failed before _drain_rank_stream"):
            ex.run(queue_gpu_and_finish)
    assert len(recorded) == 1
    assert recorded[0].query()


def test_startup_timeout_rolls_back_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    original_world = dist.distributed_c10d._world
    orig_init_pg = dist.init_process_group
    orig_cancel = _sequential_runtime.StartupGate.cancel
    stall = threading.Event()
    rolled_back = threading.Event()

    def stalled_init_pg(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("rank") == 1:
            assert stall.wait(timeout=5.0)
        return orig_init_pg(*args, **kwargs)

    def cancel_and_release_stall(self: _sequential_runtime.StartupGate) -> None:
        rolled_back.set()
        stall.set()
        orig_cancel(self)

    monkeypatch.setattr(dist, "init_process_group", stalled_init_pg)
    monkeypatch.setattr(_sequential_runtime.StartupGate, "cancel", cancel_and_release_stall)
    ex = ParallelExecutor(["cpu", "cpu"], collective_timeout=timedelta(seconds=0.05))
    try:
        with pytest.raises(TimeoutError, match="did not initialize before collective_timeout"):
            ex.__enter__()
    finally:
        stall.set()
        monkeypatch.undo()
        if _sequential_runtime.owns_session(ex) or ex._runtime is not None:
            ex.close()
    assert rolled_back.is_set()
    assert dist.distributed_c10d._world is original_world
    assert not _sequential_runtime.owns_session(ex)
    assert not _sequential_runtime.session_lock.locked()
    with ParallelExecutor(["cpu", "cpu"]) as reopened:
        assert reopened.run(_run_simple_allreduce) == (3.0, 3.0)
