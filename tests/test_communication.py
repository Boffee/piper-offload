"""Exercise the registered relay through c10d and real DTensor operations."""

import functools
import gc
import math
import mmap
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import timedelta
from itertools import pairwise

import pytest
import weakref
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module

from piper_offload import _transfer_priority, communication
from piper_offload._devices import canonical_device
from piper_offload.communication import RelayOptions, register_relay_backend
from piper_offload._host_registration import RuntimeHostRegistration
from piper_offload.pin_manager import PinManager, host_pin_manager
from tests.conftest import Interrupt

pytestmark = pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo required")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
WAIT = timedelta(seconds=30)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _run_cases(rank: int, worker: Callable[..., None], cases: Sequence[tuple[object, ...]]) -> None:
    """Share worker startup; each case must release its process group before returning."""
    for args in cases:
        try:
            worker(rank, *args)
            assert not dist.is_initialized()
        except (Exception, pytest.fail.Exception) as error:
            # pytest failures also inherit BaseException. Make mp.spawn propagate
            # them with the case arguments and terminate peers still in a collective.
            raise RuntimeError(f"{worker.__name__}{(rank, *args)!r} failed") from error


def _run_relay(
    rank: int, store_path: str, device_indices: tuple[int, ...] | None,
    pipeline_buffers: int = 1, transport: str = "gloo",
) -> None:
    torch.set_num_threads(1)
    if device_indices is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", device_indices[rank])
        torch.cuda.set_device(device)
    register_relay_backend()
    register_relay_backend()
    original_pin_budget = host_pin_manager.max_pinned_bytes
    dist.init_process_group(
        "piper_relay",
        store=dist.FileStore(store_path, 2),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
        pg_options=RelayOptions(pipeline_buffers=pipeline_buffers, transport=transport),
    )
    try:
        assert dist.get_backend() == "piper_relay"
        mesh = DeviceMesh.from_group(dist.group.WORLD, device.type)

        # Repeat with registration disabled and enabled. Both paths must
        # deliver completed GPU results and release their transient pin leases.
        for budget in (0, 16 * 1024 * 1024):
            host_pin_manager.max_pinned_bytes = budget
            for dtype in DTYPES:
                _check_reduction(mesh, rank, device, dtype)
                _check_copy_collectives(rank, device, dtype)
                _check_dtensor_initialization(mesh, rank, device, dtype)
                assert host_pin_manager.stats.active_leases == 0

        for dtype in DTYPES:
            _check_mlp(mesh, rank, device, dtype)
        for dtype in (torch.float16, torch.bfloat16):
            _check_payload_bits(rank, device, dtype)
        _check_scatter_aliases(rank, device)
        _check_coalesced(device)
        # Integer metadata takes the same byte-preserving transport path.
        _check_copy_collectives(rank, device, torch.int64)
        if device.type == "cuda":
            _check_streams(device)

        # Unsupported accelerator collectives must fail without forwarding
        # GPU tensors to Gloo. A subsequent valid collective still succeeds.
        if device.type == "cuda":
            with pytest.raises(RuntimeError, match="[Bb]ackend"):
                dist.reduce(torch.ones(1, device=device), dst=0)
        x = torch.ones(1, device=device)
        dist.all_reduce(x)
        torch.testing.assert_close(x, torch.full_like(x, 2))
        dist.barrier()
    finally:
        dist.destroy_process_group()
        gc.collect()
        host_pin_manager.clear()
        host_pin_manager.max_pinned_bytes = original_pin_budget
    assert host_pin_manager.stats.active_leases == 0
    assert host_pin_manager.stats.pinned_bytes == 0


def _check_reduce_scatter_ops(mesh, rank, device, dtype):
    for shape in ((), (0,), (3, 7), (257,)):
        in_list = [torch.full(shape, float(rank + 1 + r), device=device, dtype=dtype) for r in range(2)]
        backups = [t.clone() for t in in_list]
        out = torch.empty(shape, device=device, dtype=dtype)
        work = dist.reduce_scatter(out, in_list, async_op=True)
        assert work.wait(WAIT)
        assert work.is_completed()
        assert work.get_future().done()
        assert work.get_future().wait()[0] is out
        expected = torch.full_like(out, float(3 + 2 * rank))
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        for src, backup in zip(in_list, backups, strict=True):
            torch.testing.assert_close(src, backup, rtol=0, atol=0)

        stacked = torch.stack(in_list)
        out_single = torch.empty_like(out)
        dist.reduce_scatter_single(out_single, stacked)
        torch.testing.assert_close(out_single, expected, rtol=0, atol=0)
        out_tensor = torch.empty_like(out)
        dist.reduce_scatter_tensor(out_tensor, stacked)
        torch.testing.assert_close(out_tensor, expected, rtol=0, atol=0)

    contributions = [torch.linspace(-5 + r, 7 + r, 12).to(dtype).reshape(2, 6) for r in range(2)]
    expected_2d = (contributions[0].float() + contributions[1].float()).to(dtype).to(device)
    local_partial = contributions[rank].to(device).clone()
    partial = DTensor.from_local(local_partial, mesh, [Partial()], run_check=False)
    for placement in (Shard(0), Shard(1)):
        expected_shard = torch.chunk(expected_2d, 2, dim=placement.dim)[rank]
        for asynchronous in (False, True):
            sharded = partial.redistribute(placements=[placement], async_op=asynchronous).to_local()
            torch.testing.assert_close(sharded, expected_shard, rtol=0, atol=0)
    torch.testing.assert_close(partial.to_local().cpu(), contributions[rank], rtol=0, atol=0)


def _check_reduction(mesh, rank, device, dtype):
    for shape in ((), (0,), (3, 7), (257,)):
        x = torch.full(shape, float(rank + 1), device=device, dtype=dtype)
        work = dist.all_reduce(x, async_op=True)
        assert work.wait(WAIT)
        assert work.is_completed()
        assert work.get_future().done()
        assert work.get_future().wait()[0] is x
        torch.testing.assert_close(x, torch.full_like(x, 3))

    # Compare against summation of the rounded low-precision inputs in FP32.
    contributions = [torch.linspace(-5 + r, 7 + r, 257).to(dtype) for r in range(2)]
    expected = (contributions[0].float() + contributions[1].float()).to(dtype).to(device)
    local = contributions[rank].to(device).clone()
    partial = DTensor.from_local(local, mesh, [Partial()], run_check=False)
    for asynchronous in (False, True):
        reduced = partial.redistribute(placements=[Replicate()], async_op=asynchronous).to_local()
        torch.testing.assert_close(reduced, expected, rtol=0, atol=0)
    torch.testing.assert_close(partial.to_local().cpu(), contributions[rank])
    _check_reduce_scatter_ops(mesh, rank, device, dtype)


def _check_alltoall_copy_ops(rank, device, dtype):
    single_in = torch.arange(12, dtype=torch.float32).reshape(4, 3).to(dtype=dtype, device=device) + rank * 10
    other_in = single_in - rank * 10 + (1 - rank) * 10
    single_out = torch.empty_like(single_in)
    dist.all_to_all_single(single_out, single_in)
    exp_equal = torch.cat([single_in[:2], other_in[:2]] if rank == 0 else [other_in[2:], single_in[2:]], dim=0)
    torch.testing.assert_close(single_out, exp_equal, rtol=0, atol=0)

    in_splits = [1, 3] if rank == 0 else [2, 2]
    out_splits = [1, 2] if rank == 0 else [3, 2]
    unequal_out = torch.empty((sum(out_splits), 3), dtype=dtype, device=device)
    dist.group.WORLD.alltoall_base(unequal_out, single_in, out_splits, in_splits).wait()
    exp_unequal = torch.cat([single_in[:1], other_in[:2]] if rank == 0 else [other_in[1:4], single_in[2:4]], dim=0)
    torch.testing.assert_close(unequal_out, exp_unequal, rtol=0, atol=0)
    fn_out = funcol.wait_tensor(funcol.all_to_all_single(single_in, out_splits, in_splits, dist.group.WORLD))
    torch.testing.assert_close(fn_out, exp_unequal, rtol=0, atol=0)

    global_grid = torch.arange(24, dtype=torch.float32).reshape(4, 6).to(dtype=dtype, device=device)
    shard0 = global_grid.chunk(2, dim=0)[rank].contiguous()
    shard1 = torch.ops._dtensor.shard_dim_alltoall(shard0, 0, 1, dist.group.WORLD.group_name)
    torch.testing.assert_close(shard1, global_grid.chunk(2, dim=1)[rank], rtol=0, atol=0)


def _check_copy_collectives(rank, device, dtype):
    for shape in ((), (0,), (2, 3)):
        payload = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).to(dtype=dtype, device=device)
        source = payload.clone() if rank == 1 else torch.full_like(payload, -1)
        work = dist.broadcast(source, src=1, async_op=True)
        assert work.wait(WAIT)
        assert work.is_completed()
        torch.testing.assert_close(source, payload, rtol=0, atol=0)

        received = torch.empty_like(payload)
        scatter_list = [payload, payload + 1] if rank == 1 else None
        work = dist.scatter(received, scatter_list, src=1, async_op=True)
        assert work.wait(WAIT)
        assert work.is_completed()
        torch.testing.assert_close(received, payload + rank, rtol=0, atol=0)

        local = payload + rank
        gathered = [torch.empty_like(local) for _ in range(2)]
        work = dist.all_gather(gathered, local, async_op=True)
        assert work.wait(WAIT)
        assert work.is_completed()
        assert len(work.get_future().wait()[0]) == 2
        for r, tensor in enumerate(gathered):
            torch.testing.assert_close(tensor, payload + r, rtol=0, atol=0)

        flat = local.reshape(-1)
        concatenated = torch.empty(flat.numel() * 2, dtype=dtype, device=device)
        dist.all_gather_single(concatenated, flat)
        expected = torch.cat([payload.reshape(-1), payload.reshape(-1) + 1])
        torch.testing.assert_close(concatenated, expected, rtol=0, atol=0)
        stacked = torch.empty((2, *flat.shape), dtype=dtype, device=device)
        dist.all_gather_into_tensor(stacked, flat)
        torch.testing.assert_close(stacked.reshape(-1), expected, rtol=0, atol=0)
        torch.testing.assert_close(local, payload + rank, rtol=0, atol=0)

        a2a_in = [payload + (rank * 2 + peer) for peer in range(2)]
        a2a_out = [torch.empty_like(payload) for _ in range(2)]
        a2a_work = dist.all_to_all(a2a_out, a2a_in, async_op=True)
        assert a2a_work.wait(WAIT)
        assert a2a_work.is_completed()
        assert len(a2a_work.get_future().wait()) == 2
        for peer, tensor in enumerate(a2a_out):
            torch.testing.assert_close(tensor, payload + (peer * 2 + rank), rtol=0, atol=0)

    _check_alltoall_copy_ops(rank, device, dtype)


def _check_scatter_aliases(rank, device):
    backing = torch.arange(12, dtype=torch.float32, device=device)
    other = torch.full((8,), -1.0, device=device)
    sources = [other, backing[:8]]
    if rank == 1:
        class UnexpectedTransport:
            def scatter(self, *args):
                pytest.fail("scatter entered transport before rejecting unsafe aliases")

        # Reproduce the root overwriting another rank's send buffer, including
        # partial overlap. Partial overlap with its own source is unsafe too.
        cases = (
            ([backing[:8], other], backing[:8]),
            ([backing[:8], other], backing[4:]),
            (sources, backing[4:]),
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(dist.group.WORLD, "_cpu_group", UnexpectedTransport())
            for inputs, output in cases:
                with pytest.raises(ValueError, match="scatter output must not overlap"):
                    dist.scatter(output, inputs, src=1)
        torch.testing.assert_close(backing, torch.arange(12, dtype=torch.float32, device=device))
        torch.testing.assert_close(other, torch.full_like(other, -1))

    # An exact alias of the root's own source remains valid, even when the
    # output is a distinct tensor view. The next collective must still work.
    received = sources[1].view_as(sources[1]) if rank == 1 else torch.empty_like(other)
    dist.scatter(received, sources if rank == 1 else None, src=1)
    torch.testing.assert_close(received, sources[rank])


def _check_dtensor_initialization(mesh, rank, device, dtype):
    for shape in ((5, 7), (1, 3), (0, 3)):
        expected = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).to(dtype=dtype, device=device)
        for placement in (Replicate(), Shard(0), Shard(1)):
            source = expected.clone() if rank == 1 else torch.full_like(expected, -1)
            dt = distribute_tensor(source, mesh, [placement], src_data_rank=1)
            for asynchronous in (False, True):
                result = dt.redistribute(placements=[Replicate()], async_op=asynchronous).to_local()
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
    # run_check exercises CPU metadata exchange and accelerator broadcast.
    local = torch.full((2, 3), float(rank), dtype=dtype, device=device)
    checked = DTensor.from_local(local, mesh, [Replicate()], run_check=True)
    torch.testing.assert_close(checked.to_local(), torch.zeros_like(local))


def _check_coalesced(device):
    rank = dist.get_rank()
    inputs = [torch.full((3,), rank + 1, device=device, dtype=torch.bfloat16),
              torch.full((2, 4), rank + 2, device=device, dtype=torch.float32)]
    reduced = funcol.all_reduce_coalesced(inputs, "sum", dist.group.WORLD)
    for i, tensor in enumerate(reduced):
        torch.testing.assert_close(tensor, torch.full_like(inputs[i], 3 + 2 * i))
    gathered = funcol.all_gather_single_coalesced(inputs, dist.group.WORLD)
    for i, tensor in enumerate(gathered):
        expected = torch.cat([torch.full_like(inputs[i], i + 1), torch.full_like(inputs[i], i + 2)])
        torch.testing.assert_close(tensor, expected)
        torch.testing.assert_close(inputs[i], torch.full_like(inputs[i], rank + i + 1))

    # PyTorch's inherited legacy alias must reach the coalesced override too.
    outputs = [torch.empty_like(tensor) for tensor in gathered]
    dist.group.WORLD.allgather_into_tensor_coalesced(outputs, inputs).wait()
    for output, expected in zip(outputs, gathered, strict=True):
        torch.testing.assert_close(output, expected)

    rs_inputs = [torch.cat([inputs[0] + 1, inputs[0] + 3]), torch.cat([inputs[1] + 1, inputs[1] + 3])]
    rs_backups = [t.clone() for t in rs_inputs]
    rs_single = funcol.wait_tensor(funcol.reduce_scatter_single(rs_inputs[0], "sum", 0, dist.group.WORLD))
    torch.testing.assert_close(rs_single, torch.full_like(inputs[0], 5 + 4 * rank))
    rs_coal = [
        funcol.wait_tensor(t)
        for t in funcol.reduce_scatter_single_coalesced(rs_inputs, "sum", [0, 0], dist.group.WORLD)
    ]
    for i, tensor in enumerate(rs_coal):
        torch.testing.assert_close(tensor, torch.full_like(inputs[i], 3 + 2 * i + 2 + 4 * rank))
        torch.testing.assert_close(rs_inputs[i], rs_backups[i])
    pg_rs = [torch.empty_like(t) for t in rs_coal]
    dist.group.WORLD.reduce_scatter_tensor_coalesced(pg_rs, rs_inputs).wait()
    for actual, expected in zip(pg_rs, rs_coal, strict=True):
        torch.testing.assert_close(actual, expected)
    base_rs = torch.empty_like(rs_coal[0])
    dist.group.WORLD._reduce_scatter_base(base_rs, rs_inputs[0]).wait()
    torch.testing.assert_close(base_rs, rs_coal[0])


def _check_payload_bits(rank, device, dtype):
    # Preserve signed zero, subnormals, infinity and a NaN payload without
    # routing copy-only collectives through floating-point conversion.
    infinity, nan = (0x7C00, 0x7E01) if dtype == torch.float16 else (0x7F80, 0x7FC1)
    bits = torch.tensor([0, -32768, 1, infinity, nan], dtype=torch.int16)
    local = bits.roll(rank).view(dtype).to(device)
    broadcast = local.clone()
    dist.broadcast(broadcast, src=1)
    assert torch.equal(broadcast.view(torch.int16).cpu(), bits.roll(1))
    output = torch.empty(local.numel() * 2, dtype=dtype, device=device)
    dist.all_gather_single(output, local)
    assert torch.equal(output.view(torch.int16).cpu(), torch.cat([bits, bits.roll(1)]))
    a2a_in = torch.cat([bits, bits.roll(1) + 2]).view(dtype).to(device)
    a2a_out = torch.empty_like(a2a_in)
    dist.all_to_all_single(a2a_out, a2a_in)
    exp_a2a = (
        torch.cat([bits, bits]) if rank == 0 else torch.cat([bits.roll(1) + 2, bits.roll(1) + 2])
    )
    assert torch.equal(a2a_out.view(torch.int16).cpu(), exp_a2a)


def _check_mlp(mesh: DeviceMesh, rank: int, device: torch.device, dtype: torch.dtype) -> None:
    # Standard initialization now exercises broadcast and scatter; a gathered
    # hidden activation additionally exercises functional all-gather.
    x = (torch.arange(24, dtype=torch.float32).reshape(3, 8) / 32).to(dtype)
    w1 = (torch.arange(96, dtype=torch.float32).reshape(8, 12) / 128 - 0.5).to(dtype)
    w2 = (torch.arange(48, dtype=torch.float32).reshape(12, 4) / 64 - 0.5).to(dtype)
    expected = (x @ w1).relu() @ w2
    dx = distribute_tensor(x.to(device), mesh, [Replicate()])
    dw1 = distribute_tensor(w1.to(device), mesh, [Shard(1)])
    dw2 = distribute_tensor(w2.to(device), mesh, [Shard(0)])
    with torch.inference_mode():
        hidden = (dx @ dw1).relu()
        torch.testing.assert_close(hidden.full_tensor().cpu(), (x @ w1).relu())
        result = (hidden @ dw2).redistribute(placements=[Replicate()]).to_local()
    tolerances = {torch.float32: (1e-5, 1e-5), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}
    rtol, atol = tolerances[dtype]
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)

    x_seq = (torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 32 - 0.3).to(dtype)
    w_row = (torch.arange(24, dtype=torch.float32).reshape(6, 4) / 32 - 0.4).to(dtype)
    bias_row = (torch.arange(6, dtype=torch.float32) / 16 - 0.1).to(dtype)
    expected_sp = torch.nn.functional.linear(x_seq, w_row, bias_row).chunk(2, dim=1)[rank]
    layer = torch.nn.Linear(4, 6, bias=True).to(device=device, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(w_row)
        layer.bias.copy_(bias_row)
    parallelize_module(layer, mesh, RowwiseParallel(output_layouts=Shard(1)))
    x_local = x_seq.chunk(2, dim=2)[rank].to(device)
    with torch.inference_mode():
        sp_out = layer(x_local)
    torch.testing.assert_close(sp_out.cpu(), expected_sp, rtol=rtol, atol=atol)


def _check_streams(device: torch.device) -> None:
    producer = torch.cuda.Stream(device=device)
    consumer = torch.cuda.Stream(device=device)
    with torch.cuda.stream(producer):
        # Operations immediately before the download and after the upload
        # exercise actual GPU ordering on a non-default stream.
        x = torch.arange(1024 * 1024, device=device, dtype=torch.float32).mul_(0.25)
        work = dist.all_reduce(x, async_op=True)
        gathered = torch.empty(x.numel() * 2, device=device)
        gather_work = dist.all_gather_single(gathered, x, async_op=True)
        broadcast = torch.full((1024,), float(dist.get_rank()), device=device, dtype=torch.float16)
        broadcast_work = dist.broadcast(broadcast, src=1, async_op=True)
        scattered = torch.empty_like(broadcast)
        scatter_list = [broadcast, broadcast + 1] if dist.get_rank() == 1 else None
        scatter_work = dist.scatter(scattered, scatter_list, src=1, async_op=True)
        rs_in = torch.arange(2 * 1024 * 1024, device=device, dtype=torch.float32).reshape(2, -1).mul_(0.25)
        rs_out = torch.empty(1024 * 1024, device=device, dtype=torch.float32)
        rs_work = dist.reduce_scatter_single(rs_out, rs_in, async_op=True)
        a2a_in = torch.arange(2 * 1024, device=device, dtype=torch.float32).reshape(2, -1) + dist.get_rank() * 10000
        a2a_out = torch.empty_like(a2a_in)
        a2a_work = dist.all_to_all_single(a2a_out, a2a_in, async_op=True)
        peer = 1 - dist.get_rank()
        p2p_src = torch.arange(257, device=device, dtype=torch.float16) + dist.get_rank() * 100
        p2p_dst = torch.empty_like(p2p_src)
        p2p_works = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, p2p_src, peer, tag=7),
            dist.P2POp(dist.irecv, p2p_dst, peer, tag=7),
        ])
    # Each collective starts after the producer's earlier work, and its
    # outputs are ready for any stream once wait() returns.
    for pending in (work, gather_work, broadcast_work, scatter_work, rs_work, a2a_work, *p2p_works):
        assert pending.wait(WAIT)
        assert pending.is_completed()
    assert producer.query()
    with torch.cuda.stream(consumer):
        result = x + 1
        rs_checked = rs_out + 1
        a2a_checked = a2a_out + 1
        p2p_checked = p2p_dst + 1
    consumer.synchronize()
    expected = torch.arange(x.numel(), dtype=torch.float32) * 0.5 + 1
    rs_expected = rs_in[dist.get_rank()].cpu().mul(2).add(1)
    rank = dist.get_rank()
    a2a_expected = (
        torch.arange(1024, dtype=torch.float32).repeat(2, 1)
        + rank * 1024
        + torch.tensor([[0.0], [10000.0]])
    )
    p2p_expected = torch.arange(257, dtype=torch.float16) + (1 - rank) * 100 + 1
    torch.testing.assert_close(result.cpu(), expected)
    torch.testing.assert_close(rs_checked.cpu(), rs_expected)
    torch.testing.assert_close(a2a_checked.cpu(), a2a_expected + 1)
    torch.testing.assert_close(p2p_checked.cpu(), p2p_expected)
    torch.testing.assert_close(gathered.cpu(), torch.cat([expected - 1, expected - 1]))
    torch.testing.assert_close(broadcast, torch.ones_like(broadcast))
    torch.testing.assert_close(scattered, torch.full_like(scattered, dist.get_rank() + 1))


def test_two_rank_cpu_dtensor(tmp_path):
    cases = [(str(tmp_path / f"cpu-{transport}"), None, 1, transport) for transport in ("gloo", "shared")]
    mp.spawn(_run_cases, args=(_run_relay, cases), nprocs=2)


@CUDA
def test_two_rank_dtensor_on_one_gpu(tmp_path):
    cases = [
        (str(tmp_path / f"one-gpu-{transport}-{buffers}"), (0, 0), buffers, transport)
        for transport, buffers in (("gloo", 1), ("gloo", 3), ("shared", 1))
    ]
    mp.spawn(_run_cases, args=(_run_relay, cases), nprocs=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two physical CUDA/HIP GPUs required")
@pytest.mark.parametrize("transport", ["gloo", "shared"])
def test_two_rank_dtensor_on_two_gpus(tmp_path, transport):
    mp.spawn(_run_relay, args=(str(tmp_path / "two-gpu-store"), (0, 1), 1, transport), nprocs=2)


def _compiled_forward(sharded, partial):
    gathered = sharded.redistribute(placements=[Replicate()]).to_local()
    reduced = partial.redistribute(placements=[Replicate()]).to_local()
    return (gathered + reduced).sin()


def _run_compiled_relay(rank, store_path, transport):
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(store_path, 2), rank=rank,
        world_size=2, timeout=timedelta(seconds=60),
        pg_options=RelayOptions(staging_bytes=512 if transport == "shared" else 192,
                                pipeline_buffers=3, transport=transport),
    )
    try:
        mesh = DeviceMesh.from_group(dist.group.WORLD, "cuda")
        compiled = torch.compile(
            _compiled_forward, fullgraph=True,
            options={"triton.cudagraphs": False, "compile_threads": 1},
        )
        for dtype in DTYPES:
            full = (torch.arange(24, dtype=torch.float32, device="cuda").reshape(4, 6) / 16).to(dtype)
            sharded = distribute_tensor(full, mesh, [Shard(0)])
            partial = DTensor.from_local(torch.full_like(full, rank + 1), mesh, [Partial()], run_check=False)
            for _ in range(2):
                result = compiled(sharded, partial)
                torch.testing.assert_close(result, (full + 3).sin())
    finally:
        dist.destroy_process_group()


@CUDA
def test_compiled_dtensor_collectives_on_one_gpu(tmp_path):
    cases = [(str(tmp_path / f"compiled-{transport}"), transport) for transport in ("gloo", "shared")]
    mp.spawn(_run_cases, args=(_run_compiled_relay, cases), nprocs=2)


@pytest.fixture
def single_rank_group(tmp_path, request):
    register_relay_backend()
    dist.init_process_group(
        "piper_relay",
        store=dist.FileStore(str(tmp_path / "validation-store"), 1),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=5),
        pg_options=getattr(request, "param", None),
    )
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@CUDA
def test_accelerator_collective_moves_data_as_a_foreground_transfer(single_rank_group, monkeypatch):
    # Block prefetch waits only for foreground transfers; a collective that
    # stopped marking itself would silently queue behind the next block's
    # weights again.
    seen = []

    device = canonical_device("cuda")

    class ObservingTransport:
        def allreduce(self, tensors, options):
            del tensors, options
            with _transfer_priority._foreground.condition:
                seen.append(dict(_transfer_priority._foreground.counts))
            return self

        def wait(self):
            pass

    monkeypatch.setattr(single_rank_group, "_cpu_group", ObservingTransport())
    dist.all_reduce(torch.ones(256, device="cuda"))
    assert seen == [{device: 1}]
    assert _transfer_priority._foreground.counts == {}


@CUDA
def test_barrier_waits_for_earlier_collectives(single_rank_group, monkeypatch):
    # A barrier that overtook a queued collective's Gloo messages would pair
    # different operations across ranks.
    release = threading.Event()
    order = []

    class OrderedTransport:
        def allreduce(self, tensors, options):
            del tensors, options
            release.wait(WAIT.total_seconds())
            order.append("allreduce")
            return self

        def barrier(self, options):
            del options
            order.append("barrier")
            return self

        def wait(self):
            pass

    monkeypatch.setattr(single_rank_group, "_cpu_group", OrderedTransport())
    reduction = dist.all_reduce(torch.ones(256, device="cuda"), async_op=True)
    barrier = dist.barrier(async_op=True)
    release.set()
    assert barrier.wait(WAIT)
    assert reduction.is_completed()
    assert order == ["allreduce", "barrier"]


@CUDA
def test_shutdown_lets_issued_collectives_finish_first(single_rank_group, monkeypatch):
    # Releasing Gloo or staging under a running collective would hang or corrupt it.
    release = threading.Event()

    class BlockingTransport:
        def allreduce(self, tensors, options):
            del tensors, options
            release.wait(WAIT.total_seconds())
            return self

        def wait(self):
            pass

    monkeypatch.setattr(single_rank_group, "_cpu_group", BlockingTransport())
    work = dist.all_reduce(torch.ones(256, device="cuda"), async_op=True)
    closer = threading.Thread(target=single_rank_group.shutdown)
    closer.start()
    closer.join(0.2)
    assert closer.is_alive()
    assert not work.is_completed()
    release.set()
    closer.join(WAIT.total_seconds())
    assert not closer.is_alive()
    assert work.wait(WAIT)


def test_future_callbacks_run_on_the_calling_thread(single_rank_group, monkeypatch):
    # DDP communication hooks such as PowerSGD run a collective inside a
    # future's callback and wait for it. Run on the group's worker, that
    # callback would wait for a collective queued behind itself.
    release = threading.Event()

    class HeldTransport:
        def allreduce(self, tensors, options):
            del tensors, options
            release.wait(WAIT.total_seconds())
            return self

        def wait(self):
            pass

    def chained(future):
        del future
        dist.all_reduce(torch.ones(4), async_op=True).wait(WAIT)
        return threading.current_thread()

    monkeypatch.setattr(single_rank_group, "_cpu_group", HeldTransport())
    work = dist.all_reduce(torch.ones(4), async_op=True)
    # The collective is still running when the callback is attached.
    releaser = threading.Timer(0.2, release.set)
    releaser.start()
    chained_future = work.get_future().then(chained)
    releaser.join()
    assert chained_future.wait() is threading.current_thread()


@CUDA
@pytest.mark.usefixtures("single_rank_group")
def test_collective_writes_outputs_created_in_inference_mode():
    # Piper Engine runs models under inference mode, so collective outputs are
    # inference tensors, which only inference mode may write in place. The
    # group's worker thread runs outside it; the relay writes through byte
    # views, which PyTorch permits.
    with torch.inference_mode():
        x = torch.full((256,), 2.0, device="cuda")
        dist.all_reduce(x)
    torch.testing.assert_close(x.cpu(), torch.full((256,), 2.0))


@CUDA
@pytest.mark.usefixtures("single_rank_group")
def test_waited_work_frees_its_tensors_without_garbage_collection():
    # Callers drop each Work right after waiting on it; buffers kept until the
    # cyclic collector runs would pile up across a forward pass of collectives.
    gc.disable()
    try:
        x = torch.ones(1024, device="cuda")
        work = dist.all_reduce(x, async_op=True)
        assert work.wait(WAIT)
        output = weakref.ref(x)
        del x, work
        # The worker drops a finished job just after waking its waiter; once it
        # has run the next collective, nothing else may hold the tensor.
        dist.all_reduce(torch.ones(1, device="cuda"))
        assert output() is None
    finally:
        gc.enable()


@pytest.mark.parametrize(
    ("tensor", "op", "message"),
    [
        (torch.ones(2), dist.ReduceOp.MAX, "only SUM"),
        (torch.ones(2, dtype=torch.float64), dist.ReduceOp.SUM, "supports FP32, FP16 and BF16"),
        (torch.ones(2, dtype=torch.int32), dist.ReduceOp.SUM, "supports FP32, FP16 and BF16"),
        (torch.ones(2, 3).T, dist.ReduceOp.SUM, "contiguous strided"),
        (torch.ones(2).to_sparse(), dist.ReduceOp.SUM, "contiguous strided"),
    ],
)
def test_reject_unsupported_allreduce(single_rank_group, tensor, op, message):
    with pytest.raises(NotImplementedError, match=message):
        dist.all_reduce(tensor, op=op)


def test_reject_multiple_local_tensors(single_rank_group):
    with pytest.raises(NotImplementedError, match="exactly one local tensor"):
        single_rank_group.allreduce([torch.ones(1), torch.ones(1)])


def test_allgather_validates_entire_batch_before_mutating(single_rank_group):
    inputs = [torch.ones(2), torch.ones(3)]
    outputs = [torch.full((2,), -1.0), torch.empty(4)]
    with pytest.raises(ValueError, match="expected element count"):
        single_rank_group.all_gather_single_coalesced(outputs, inputs)
    torch.testing.assert_close(outputs[0], torch.full((2,), -1.0))


def test_coalesced_reduction_rejects_overlapping_outputs(single_rank_group):
    tensor = torch.ones(4)
    with pytest.raises(ValueError, match="must not overlap"):
        single_rank_group.allreduce_coalesced([tensor[:3], tensor[2:]])


def test_coalesced_gather_rejects_cross_operation_aliases(single_rank_group):
    tensors = [torch.ones(2), torch.ones(2)]
    with pytest.raises(ValueError, match="overlap another input"):
        single_rank_group.all_gather_single_coalesced(tensors[::-1], tensors)


def test_scatter_rejects_mismatched_shapes(single_rank_group):
    with pytest.raises(ValueError, match="equal tensor shapes"):
        single_rank_group.scatter([torch.empty(2, 3)], [[torch.ones(6)]])


def test_broadcast_rejects_invalid_root(single_rank_group):
    options = dist.BroadcastOptions()
    options.rootRank = 1
    with pytest.raises(ValueError, match="root rank"):
        single_rank_group.broadcast([torch.ones(1)], options)


def test_reject_unsupported_reduce_scatter(single_rank_group, monkeypatch):
    manager = PinManager(1024 * 1024)
    monkeypatch.setattr(communication, "host_pin_manager", manager)
    out = torch.empty(2)
    for op in (dist.ReduceOp.MAX, dist.ReduceOp.PRODUCT):
        with pytest.raises(NotImplementedError, match="only SUM"):
            dist.reduce_scatter(out, [torch.ones(2)], op=op)
        with pytest.raises(NotImplementedError, match="only SUM"):
            dist.reduce_scatter_single(out, torch.ones(2), op=op)
    for bad in (
        torch.ones(2, dtype=torch.int64),
        torch.ones(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.bool),
    ):
        with pytest.raises(NotImplementedError, match="supports FP32, FP16 and BF16"):
            dist.reduce_scatter_single(torch.empty(2, dtype=bad.dtype), bad)
    with pytest.raises(NotImplementedError, match="contiguous strided"):
        dist.reduce_scatter_single(out, torch.ones(2, 2).T.reshape(1, 4)[:, ::2])
    class TensorSubclass(torch.Tensor):
        pass
    with pytest.raises(TypeError, match="plain local tensor"):
        single_rank_group.reduce_scatter([out], [[TensorSubclass(torch.ones(2))]])
    with pytest.raises(ValueError, match="one input tensor per rank"):
        single_rank_group.reduce_scatter([out], [[torch.ones(2), torch.ones(2)]])
    with pytest.raises(ValueError, match="equal tensor shapes"):
        single_rank_group.reduce_scatter([torch.empty(1, 2)], [[torch.ones(2)]])
    with pytest.raises(ValueError, match="expected element count"):
        dist.reduce_scatter_single(torch.empty(3), torch.ones(2))
    with pytest.raises(ValueError, match="input dtype"):
        dist.reduce_scatter_single(torch.empty(2, dtype=torch.float16), torch.ones(2))

    backing = torch.arange(5, dtype=torch.float32)
    with pytest.raises(ValueError, match="exact local input slice"):
        dist.reduce_scatter_single(backing[1:3], backing[:2])
    with pytest.raises(ValueError, match="must not overlap"):
        single_rank_group.reduce_scatter_single_coalesced([backing[:3], backing[2:4]], [torch.ones(3), torch.ones(2)])
    pair = [torch.ones(2), torch.ones(2)]
    with pytest.raises(ValueError, match="overlap another input"):
        single_rank_group.reduce_scatter_single_coalesced(pair[::-1], pair)
    bad_outputs = [torch.full((2,), -1.0), torch.empty(4)]
    bad_inputs = [torch.ones(2), torch.ones(3)]
    with pytest.raises(ValueError, match="expected element count"):
        single_rank_group.reduce_scatter_single_coalesced(bad_outputs, bad_inputs)
    torch.testing.assert_close(bad_outputs[0], torch.full((2,), -1.0))

    # Valid 1-rank calls (including exact own-slice alias and 0-element tensors) work cleanly.
    alias = torch.arange(4, dtype=torch.float16)
    dist.reduce_scatter_single(alias, alias)
    torch.testing.assert_close(alias, torch.arange(4, dtype=torch.float16))
    for dtype in DTYPES:
        empty = torch.empty(0, dtype=dtype)
        dist.reduce_scatter_single(empty, empty)
        dist.reduce_scatter(empty, [empty])
    assert manager.stats.active_leases == 0


@CUDA
def test_failed_reduction_releases_staging(single_rank_group, monkeypatch):
    manager = PinManager(1024 * 1024)
    monkeypatch.setattr(communication, "host_pin_manager", manager)
    expected = torch.arange(256, dtype=torch.float32)

    class FailingTransport:
        def allreduce(self, tensors, options):
            assert len(tensors) == 1
            assert tensors[0].device.type == "cpu"
            torch.testing.assert_close(tensors[0], expected)
            assert manager.stats.active_leases == 1
            return self

        def wait(self):
            raise RuntimeError("injected CPU transport failure")

    monkeypatch.setattr(single_rank_group, "_cpu_group", FailingTransport())
    x = expected.cuda()
    try:
        with pytest.raises(RuntimeError, match="injected CPU transport failure"):
            dist.all_reduce(x)
        assert manager.stats.active_leases == 0
        torch.testing.assert_close(x.cpu(), expected)
    finally:
        manager.clear()
    assert manager.stats.pinned_bytes == 0


def test_registration_does_not_initialize_cuda(monkeypatch):
    def unexpected_init():
        pytest.fail("backend registration initialized CUDA")

    monkeypatch.setattr(torch.cuda, "init", unexpected_init)
    register_relay_backend()
    register_relay_backend()


def test_registration_requires_gloo(monkeypatch):
    monkeypatch.setattr(dist, "is_gloo_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU Gloo"):
        register_relay_backend()


def test_registration_preserves_other_plugins(monkeypatch):
    plugin = dist.Backend._BackendPlugin(lambda *args: None, False)
    monkeypatch.setitem(dist.Backend._plugins, "PIPER_RELAY", plugin)
    with pytest.raises(RuntimeError, match="different backend"):
        communication.register_relay_backend()
    assert dist.Backend._plugins["PIPER_RELAY"] is plugin


class _CountingRegistration(RuntimeHostRegistration):
    def __init__(self):
        super().__init__()
        self.register_calls = 0
        self.unregister_calls = 0

    def register(self, pointer, size):
        self.register_calls += 1
        return super().register(pointer, size)

    def unregister(self, pointer):
        self.unregister_calls += 1
        return super().unregister(pointer)


class _MeasuredTransport:
    def __init__(self, inner, staging_bytes, accelerator, *, world_size: int = 3):
        self.inner = inner
        self.staging_bytes = staging_bytes
        self.accelerator = accelerator
        self.world_size = world_size
        ops = ("allreduce", "allgather", "broadcast", "scatter", "reduce_scatter", "send", "recv")
        if world_size != 2:
            ops = (*ops, "reduce")
        self.calls = dict.fromkeys(ops, 0)
        self.wire_bytes = dict.fromkeys(ops, 0)
        self.storage_pointers = set()

    def __getattr__(self, name):
        def call(*args):
            self.calls[name] += 1
            pending = list(args)
            meta_bytes = (2 * self.world_size + 1) * 8
            while pending:
                item = pending.pop()
                if isinstance(item, list):
                    pending.extend(item)
                elif isinstance(item, torch.Tensor):
                    assert item.device.type == "cpu"
                    assert item.nbytes <= self.staging_bytes
                    self.wire_bytes[name] += item.nbytes
                    is_a2a_meta = name == "allgather" and item.dtype == torch.int64 and item.nbytes == meta_bytes
                    if self.accelerator and not is_a2a_meta:
                        storage = item.untyped_storage()
                        assert storage.nbytes() == self.staging_bytes
                        self.storage_pointers.add(storage.data_ptr())
            return getattr(self.inner, name)(*args)
        return call


def _run_chunked_relay(rank, store_path, device_type, world_size, buffers=1, mixed_pinning=False):
    torch.set_num_threads(1)
    device = torch.device(device_type)
    if device_type == "cuda":
        torch.cuda.set_device(0)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(store_path, world_size), rank=rank, world_size=world_size,
        timeout=timedelta(seconds=20), pg_options=RelayOptions(staging_bytes=512, pipeline_buffers=buffers),
    )
    group = dist.group.WORLD
    assert group._staging_buffer is None
    registration = _CountingRegistration()
    initial_registrations = int(not mixed_pinning or rank != 0)
    manager = PinManager(16 * 1024 if initial_registrations else 0, backend=registration)
    measured = _MeasuredTransport(group._cpu_group, 512, device_type == "cuda", world_size=world_size)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(communication, "host_pin_manager", manager)
        patch.setattr(group, "_cpu_group", measured)
        pointer = None
        try:
            for dtype in DTYPES:
                _check_chunked_values(rank, world_size, device, dtype)
                if device_type == "cpu" and dtype == torch.float32:
                    assert group._staging_buffer is None
                if group._staging_buffer is not None:
                    assert group._staging_buffer.nbytes == 512
                    current = group._staging_buffer.data_ptr()
                    assert pointer in (None, current)
                    pointer = current
                assert manager.stats.active_leases == 0

            assert all(count >= 3 for count in measured.calls.values())
            if device_type == "cuda":
                assert measured.storage_pointers == {pointer}
                assert registration.register_calls == initial_registrations
                assert (group._pipeline is not None) == (buffers > 1 and bool(initial_registrations))
                assert registration.unregister_calls == 0
                # Idle registrations can be evicted without discarding the
                # allocation; restore the budget and register it again.
                manager.max_pinned_bytes = 0
                assert manager.stats.pinned_bytes == 0
                _check_chunked_values(rank, world_size, device, torch.bfloat16)
                assert registration.register_calls == initial_registrations
                manager.max_pinned_bytes = 16 * 1024
                _check_chunked_values(rank, world_size, device, torch.float16)
                assert registration.register_calls == initial_registrations + 1
                assert group._staging_buffer.data_ptr() == pointer
            else:
                assert registration.register_calls == 0
                _check_chunked_subgroup(rank)
        finally:
            dist.destroy_process_group()
        assert group._staging_buffer is None
        assert manager.stats.active_leases == 0
        assert manager.stats.pinned_bytes == 0
        assert registration.unregister_calls == registration.register_calls


def _check_chunked_subgroup(rank):
    subgroup = dist.new_group(ranks=[1, 2], pg_options=RelayOptions(staging_bytes=128))
    if rank in (1, 2):
        values = torch.full((257,), rank, dtype=torch.bfloat16)
        dist.all_reduce(values, group=subgroup)
        torch.testing.assert_close(values, torch.full_like(values, 3))
        assert subgroup._staging_buffer.nbytes == 128
        dist.destroy_process_group(subgroup)
        assert subgroup._staging_buffer is None


def _check_chunked_alltoall(rank, world_size, device, dtype, payload):
    a2a_sources = [(payload + rank * 10 + peer).to(dtype) for peer in range(world_size)]
    a2a_outputs = [torch.empty_like(a2a_sources[0]) for _ in range(world_size)]
    a2a_outputs[rank] = a2a_sources[rank]
    dist.all_to_all(a2a_outputs, a2a_sources)
    for peer in range(world_size):
        torch.testing.assert_close(a2a_outputs[peer], (payload + peer * 10 + rank).to(dtype), rtol=0, atol=0)
    with pytest.raises(ValueError, match="exact local slice"):
        dist.all_to_all_single(payload[:world_size], payload[:world_size])
    if rank == 0:
        a2a_backing = torch.arange(6, dtype=dtype, device=device)
        with pytest.raises(ValueError, match="exact local slice"):
            dist.all_to_all(
                [a2a_backing[1:5], *a2a_outputs[1:]], [a2a_backing[2:6], *a2a_sources[1:]],
            )


def _check_chunked_values(rank, world_size, device, dtype):
    # Odd length, a nonzero storage offset, and a payload larger than the pool.
    payload = (torch.arange(259, device=device, dtype=torch.float32) / 32).to(dtype)[1:-1]
    reduced = (payload + rank).clone()
    dist.all_reduce(reduced)
    expected = sum((payload + r).float() for r in range(world_size)).to(dtype)
    torch.testing.assert_close(reduced, expected, rtol=0, atol=0)

    root = world_size - 1
    broadcast = payload.clone() if rank == root else torch.zeros_like(payload)
    dist.broadcast(broadcast, src=root)
    torch.testing.assert_close(broadcast, payload, rtol=0, atol=0)
    scatter_sources = [payload + r for r in range(world_size)]
    received = scatter_sources[root] if rank == root else torch.empty_like(payload)
    dist.scatter(received, scatter_sources if rank == root else None, src=root)
    torch.testing.assert_close(received, payload + rank, rtol=0, atol=0)

    # Exact local in-place gather aliases are safe across chunk boundaries.
    output = torch.empty((world_size, payload.numel()), device=device, dtype=dtype)
    output[rank].copy_(payload + rank)
    dist.all_gather_single(output, output[rank])
    expected = torch.stack([payload + r for r in range(world_size)])
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="exact local output slice"):
        dist.all_gather_single(output, output[(rank + 1) % world_size])
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    pieces = [torch.empty_like(payload) for _ in range(world_size)]
    pieces[rank].copy_(payload + rank)
    dist.all_gather(pieces, pieces[rank])
    torch.testing.assert_close(torch.stack(pieces), expected, rtol=0, atol=0)

    rs_backing = torch.stack([payload + (rank + owner) for owner in range(world_size)])
    rs_out = torch.empty_like(payload)
    dist.reduce_scatter_single(rs_out, rs_backing)
    rs_expected = sum((payload + (r + rank)).float() for r in range(world_size)).to(dtype)
    torch.testing.assert_close(rs_out, rs_expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="exact local input slice"):
        dist.reduce_scatter_single(rs_backing[(rank + 1) % world_size], rs_backing)
    rs_list = [t.clone() for t in rs_backing.unbind(0)]
    dist.reduce_scatter(rs_list[rank], rs_list)
    torch.testing.assert_close(rs_list[rank], rs_expected, rtol=0, atol=0)
    for owner, orig in enumerate(rs_backing.unbind(0)):
        if owner != rank:
            torch.testing.assert_close(rs_list[owner], orig, rtol=0, atol=0)

    # Copy collectives preserve NaN payloads, signed zero and subnormals even
    # when raw bits cross many chunk boundaries.
    bits = torch.tensor([0, -32768, 1, 0x7E01, 0x7FC1], dtype=torch.int16).repeat(53)
    local = bits.roll(rank).view(torch.float16).to(device)
    gathered = torch.empty(local.numel() * world_size, device=device, dtype=torch.float16)
    dist.all_gather_single(gathered, local)
    assert torch.equal(gathered.view(torch.int16).cpu(), torch.cat([bits.roll(r) for r in range(world_size)]))
    _check_chunked_alltoall(rank, world_size, device, dtype, payload)


def test_chunked_relay_on_three_cpu_ranks(tmp_path):
    cases = [(str(tmp_path / f"chunked-cpu-{buffers}"), "cpu", 3, buffers) for buffers in (1, 3)]
    mp.spawn(_run_cases, args=(_run_chunked_relay, cases), nprocs=3)


@CUDA
def test_chunked_relay_reuses_gpu_staging(tmp_path):
    cases = [(str(tmp_path / f"chunked-cuda-{buffers}"), "cuda", 2, buffers) for buffers in (1, 2, 3)]
    mp.spawn(_run_cases, args=(_run_chunked_relay, cases), nprocs=2)


@CUDA
def test_pipeline_with_pageable_peer(tmp_path):
    mp.spawn(_run_chunked_relay, args=(str(tmp_path / "mixed-pins"), "cuda", 2, 3, True), nprocs=2)


def _run_mismatched_options(rank, store_path, field):
    register_relay_backend()
    with pytest.raises(ValueError, match="must match across all ranks"):
        dist.init_process_group(
            "piper_relay", store=dist.FileStore(store_path, 2), rank=rank, world_size=2,
            timeout=timedelta(seconds=5), pg_options=RelayOptions(
                staging_bytes=512 * (rank + 1) if field == "staging_bytes" else 512,
                pipeline_buffers=rank + 1 if field == "pipeline_buffers" else 1,
                transport="shared" if field == "transport" and rank == 1 else "gloo",
            ),
        )
    assert not dist.is_initialized()


def test_reject_mismatched_rank_staging_sizes(tmp_path):
    cases = [
        (str(tmp_path / f"mismatch-{field}"), field)
        for field in ("staging_bytes", "pipeline_buffers", "transport")
    ]
    mp.spawn(_run_cases, args=(_run_mismatched_options, cases), nprocs=2)


@pytest.mark.parametrize("size", [0, -16, 17, True, 64.0])
def test_reject_invalid_staging_size(size):
    with pytest.raises(ValueError, match="positive integer multiple of 16"):
        RelayOptions(staging_bytes=size)


@pytest.mark.parametrize(("size", "buffers", "transport"), [
    (16, 1, "gloo"), (32, 1, "gloo"), (80, 2, "gloo"), (128, 3, "gloo"),
    (32, 1, "shared"), (32, 3, "shared"),
])
def test_reject_insufficient_staging_slots(tmp_path, size, buffers, transport):
    register_relay_backend()
    with pytest.raises(ValueError, match="16 bytes per staging slot"):
        dist.init_process_group(
            "piper_relay", store=dist.FileStore(str(tmp_path / "small"), 1), rank=0, world_size=1,
            timeout=timedelta(seconds=5), pg_options=RelayOptions(
                staging_bytes=size, pipeline_buffers=buffers, transport=transport,
            ),
        )


@pytest.mark.parametrize(("size", "buffers"), [(80, 1), (80, 3)])
def test_reject_insufficient_shared_rank_slots(size, buffers):
    with pytest.raises(ValueError, match="16 bytes per staging slot in each rank region"):
        RelayOptions(staging_bytes=size, pipeline_buffers=buffers, transport="shared")._validate_group_size(2)


def test_accept_minimum_shared_rank_slots():
    assert RelayOptions().pipeline_buffers == 2
    RelayOptions(staging_bytes=48, pipeline_buffers=1, transport="shared")._validate_group_size(1)
    RelayOptions(staging_bytes=48, pipeline_buffers=3, transport="shared")._validate_group_size(1)
    RelayOptions(staging_bytes=96, pipeline_buffers=1, transport="shared")._validate_group_size(2)
    RelayOptions(staging_bytes=96, pipeline_buffers=3, transport="shared")._validate_group_size(2)


def test_staging_buffer_dies_with_the_group_while_chunk_views_linger(single_rank_group):
    group = single_rank_group
    with group._buffer(torch.device("cpu")) as (region, _pinned):
        view = region[:16]
    buffer_ref = weakref.ref(group._staging_buffer)
    group.shutdown()
    assert group._staging_buffer is None
    # A completed collective can hold a chunk view a little longer than the
    # group. A view keeps its base alive, so chunks are cut from a base-free
    # alias; otherwise the owner, and its registration, would outlive shutdown.
    assert view.numel() == 16
    assert buffer_ref() is None


def test_gather_rejects_partial_local_alias(single_rank_group):
    backing = torch.arange(6, dtype=torch.float32)
    with pytest.raises(ValueError, match="exact local output slice"):
        dist.all_gather_single(backing[2:], backing[:4])
    torch.testing.assert_close(backing, torch.arange(6, dtype=torch.float32))


@CUDA
@pytest.mark.parametrize("single_rank_group", [RelayOptions(staging_bytes=512, pipeline_buffers=n) for n in (1, 2, 3)],
                         indirect=True)
@pytest.mark.parametrize("operation", ["allreduce", "broadcast", "scatter", "allgather"])
def test_late_chunk_failure_releases_lease(single_rank_group, monkeypatch, operation):
    registration = _CountingRegistration()
    manager = PinManager(16 * 1024, backend=registration)
    monkeypatch.setattr(communication, "host_pin_manager", manager)
    original_transport = single_rank_group._cpu_group
    original_unregister = registration.unregister

    def unregister(pointer):
        pipeline = single_rank_group._pipeline
        if pipeline is not None:
            assert pipeline.download.query()
            assert pipeline.upload.query()
        original_unregister(pointer)

    monkeypatch.setattr(registration, "unregister", unregister)

    class FailSecondChunk:
        calls = 0

        def __getattr__(self, name):
            assert name == operation

            def call(tensors, *args):
                output = tensors[0][0] if name == "allgather" else tensors[0]
                output.fill_(9)
                assert manager.stats.active_leases == 1
                pipeline = single_rank_group._pipeline
                if pipeline is not None and self.calls == 0:
                    # Keep the first upload pending until after the next CPU
                    # chunk fails; unregister() above verifies explicit draining.
                    with torch.cuda.stream(pipeline.upload):
                        torch.cuda._sleep(5_000_000)
                return self
            return call

        def wait(self):
            self.calls += 1
            if self.calls == 2:
                manager.max_pinned_bytes = 0
                assert registration.unregister_calls == 0
                raise RuntimeError("injected second-chunk failure")

    transport = FailSecondChunk()
    monkeypatch.setattr(single_rank_group, "_cpu_group", transport)
    tensor = torch.ones(257, dtype=torch.float16, device="cuda")
    source = torch.zeros_like(tensor)
    calls = {
        "allreduce": lambda: dist.all_reduce(tensor),
        "broadcast": lambda: dist.broadcast(tensor, src=0),
        "scatter": lambda: dist.scatter(tensor, [source], src=0),
        "allgather": lambda: dist.all_gather_single(tensor, source),
    }
    with pytest.raises(RuntimeError, match="second-chunk failure"):
        calls[operation]()
    assert transport.calls == 2
    assert manager.stats.active_leases == 0
    assert registration.unregister_calls == 1
    # Completed chunks stay committed; neither the failed chunk nor the tail
    # is uploaded. Bounded staging deliberately does not provide rollback.
    expected = torch.ones_like(tensor, device="cpu")
    if operation == "allreduce":
        expected[:single_rank_group._slot_bytes(3) // 2].fill_(9)
    elif operation != "broadcast":
        expected.view(torch.uint8)[:single_rank_group._slot_bytes(2)].fill_(9)
    assert torch.equal(tensor.view(torch.int16).cpu(), expected.view(torch.int16))
    pointer = single_rank_group._staging_buffer.data_ptr()
    monkeypatch.setattr(single_rank_group, "_cpu_group", original_transport)
    manager.max_pinned_bytes = 16 * 1024
    dist.all_reduce(source)
    torch.testing.assert_close(source, torch.zeros_like(source))
    assert single_rank_group._staging_buffer.data_ptr() == pointer
    assert registration.register_calls == 2
    # Keep the group reference alive to verify shutdown still releases storage.
    single_rank_group.shutdown()
    assert single_rank_group._staging_buffer is None
    assert manager.stats.pinned_bytes == 0


@pytest.mark.parametrize("buffers", [0, -1, 4, True, 2.0])
def test_reject_invalid_pipeline_buffers(buffers):
    with pytest.raises(ValueError, match="pipeline_buffers must be"):
        RelayOptions(pipeline_buffers=buffers)


class _GlooGuard:
    def __init__(self, inner, *, allow_allreduce: bool, allow_control: bool) -> None:
        self.inner = inner
        self.allow_allreduce = allow_allreduce
        self.allow_control = allow_control
        self.gloo_calls = 0

    def allreduce(self, *args):
        if not self.allow_allreduce:
            pytest.fail("shared GPU reduction forwarded its payload to CPU Gloo")
        self.gloo_calls += 1
        return self.inner.allreduce(*args)

    def reduce_scatter(self, *args):
        if not self.allow_allreduce:
            pytest.fail("shared GPU reduction forwarded its payload to CPU Gloo")
        self.gloo_calls += 1
        return self.inner.reduce_scatter(*args)

    def reduce(self, *args):
        if not self.allow_allreduce:
            pytest.fail("shared GPU reduction forwarded its payload to CPU Gloo")
        self.gloo_calls += 1
        return self.inner.reduce(*args)

    def allgather(self, outputs, inputs):
        if not self.allow_control:
            pytest.fail("shared relay forwarded metadata or payload to Gloo allgather")
        assert inputs[0].dtype == torch.int64 and inputs[0].nbytes == 32
        return self.inner.allgather(outputs, inputs)

    def send(self, tensors, peer, tag):
        if self.allow_allreduce and tag == communication._GLOO_EXCHANGE_TAG:
            self.gloo_calls += 1
            return self.inner.send(tensors, peer, tag)
        if not self.allow_control:
            pytest.fail("shared relay sent control or payload over Gloo")
        assert tensors[0].nbytes == 8
        return self.inner.send(tensors, peer, tag)

    def recv(self, tensors, peer, tag):
        if self.allow_allreduce and tag == communication._GLOO_EXCHANGE_TAG:
            self.gloo_calls += 1
            return self.inner.recv(tensors, peer, tag)
        if not self.allow_control:
            pytest.fail("shared relay received control or payload over Gloo")
        assert tensors[0].nbytes == 8
        return self.inner.recv(tensors, peer, tag)

    def barrier(self, options):
        # The test's own synchronization, not a relay collective.
        return self.inner.barrier(options)


def _run_shared_chunks(rank, path, device_type, size, mixed_pinning, pipeline_buffers, staging_bytes):  # noqa: PLR0915
    from piper_offload._relay_signal import SharedControl

    torch.set_num_threads(1)
    device = torch.device(device_type)
    if device_type == "cuda":
        torch.cuda.set_device(0)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(path, size), rank=rank, world_size=size,
        timeout=timedelta(seconds=20),
        pg_options=RelayOptions(transport="shared", staging_bytes=staging_bytes, pipeline_buffers=pipeline_buffers),
    )
    group = dist.group.WORLD
    shared = group._shared
    assert shared.buffer_count == 3
    guard = _GlooGuard(
        group._cpu_group,
        allow_allreduce=(device_type != "cuda"),
        allow_control=(not isinstance(shared._control, SharedControl)),
    )
    registration = _CountingRegistration()
    pin_budget = max(16384, staging_bytes * 2)
    manager = PinManager(0 if mixed_pinning and rank == 0 else pin_budget, backend=registration)
    initial = int(device_type == "cuda" and manager.max_pinned_bytes != 0)
    pointer = shared.buffer.data_ptr()
    counts = [0, 0]
    control_counts = [0, 0]
    lanes = set()
    original_slot = shared._slot
    original_agree = shared._control.agree
    original_signal = shared._signal

    def slot(sender, lane, width):
        lanes.add(lane)
        counts[int(sender != rank)] += width
        view = original_slot(sender, lane, width)
        assert view.untyped_storage().data_ptr() == pointer
        assert view.untyped_storage().nbytes() == staging_bytes
        return view

    def tracked_agree(op, nbytes, kind, detail, timeout):
        control_counts[0] += 1
        return original_agree(op, nbytes, kind, detail, timeout)

    def tracked_signal(peers, tag, sequence):
        control_counts[1] += 1
        original_signal(peers, tag, sequence)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(communication, "host_pin_manager", manager)
        patch.setattr(group, "_cpu_group", guard)
        patch.setattr(shared, "_slot", slot)
        patch.setattr(shared._control, "agree", tracked_agree)
        patch.setattr(shared, "_signal", tracked_signal)
        # Exercise slot reuse and a partial last chunk at either alignment.
        local = torch.arange(shared.capacity + 1, dtype=torch.float32, device=device) + rank
        output = torch.empty((size, local.numel()), device=device)
        dist.all_gather_single(output, local)
        assert lanes == {0, 1, 2}
        assert counts == [local.nbytes, (size - 1) * local.nbytes]
        for peer in range(size):
            torch.testing.assert_close(output[peer], local - rank + peer)

        counts[:] = [0, 0]
        bcast = torch.arange(shared.capacity + 1, dtype=torch.float32, device=device)
        bcast_buf = bcast.clone() if rank == 0 else torch.empty_like(bcast)
        dist.broadcast(bcast_buf, src=0)
        torch.testing.assert_close(bcast_buf, bcast)
        assert counts == ([bcast.nbytes, 0] if rank == 0 else [0, bcast.nbytes])
        if size == 3:
            scatter_sources = [local - rank + peer for peer in range(size)]
            scatter_out = scatter_sources[1] if rank == 1 else torch.empty_like(local)
            dist.scatter(scatter_out, scatter_sources if rank == 1 else None, src=1)
            torch.testing.assert_close(scatter_out, local)
        # Both processes see the same physical host pages (distinct virtual
        # mappings). Each rank only owns its own region for Gloo reductions.
        shared.local_buffer().fill_(rank)
        dist.barrier()
        for peer in range(size):
            assert shared._slot(peer, 0, 1).item() == peer
        dist.barrier()
        for dtype in DTYPES:
            _check_chunked_values(rank, size, device, dtype)
        if device_type == "cuda":
            scratch = shared._reduction_buffer
            assert scratch.device == local.device
            assert scratch.nbytes == 3 * shared.capacity
            _check_shared_reduction_precision(rank, size, device, shared.capacity)
            assert shared._reduction_buffer is scratch
        _check_shared_coalesced(group, shared, control_counts, manager, counts, rank, size, device)
        if device_type == "cuda":
            assert shared._reduction_buffer is scratch
            assert guard.gloo_calls == 0
        assert control_counts[0] > 10 and control_counts[1] > 20
        assert group._staging_buffer is None
        assert shared.buffer.data_ptr() == pointer
        assert manager.stats.active_leases == 0
        assert registration.register_calls == initial
        if device_type == "cuda":
            manager.max_pinned_bytes = 0
            assert registration.unregister_calls == initial
            _check_chunked_values(rank, size, device, torch.bfloat16)
            manager.max_pinned_bytes = pin_budget
            _check_chunked_values(rank, size, device, torch.float16)
            assert registration.register_calls == initial + 1
            assert shared._reduction_buffer is scratch
        # Drop test-owned references so shutdown can retire the arena owner.
        del shared
        dist.destroy_process_group()
        assert group._shared is None
    gc.collect()
    manager.clear()
    assert manager.stats.active_leases == 0
    assert manager.stats.pinned_bytes == 0


def _check_shared_reduction_precision(rank, size, device, capacity):
    # Three ranks expose premature low-precision rounding and rank-dependent
    # FP32 addition order. Both buffers wrap, including a partial last chunk.
    values = {
        torch.float16: (65504, 65504, -65504),
        torch.bfloat16: (256, 1, -256),
        torch.float32: (2**24, 1, -(2**24)),
    }
    cases = [
        (dtype, capacity * 3 // dtype.itemsize + 1, contributions, False)
        for dtype, contributions in values.items()
    ]
    if size >= 3 and capacity >= 4096:
        cases.append((torch.bfloat16, 262144 // 2 + 1, values[torch.bfloat16], True))
    for dtype, count, contributions, vary_stride in cases:
        backing = torch.full((count + 2,), contributions[rank], dtype=dtype, device=device)
        local = backing[1:-1]
        expected_values = torch.tensor(contributions[:size], dtype=dtype).float()
        total = expected_values[0]
        for value in expected_values[1:]:
            total = total + value
        expected = torch.full_like(local, total.to(dtype).item())
        if vary_stride:
            varied = torch.arange(local[::997].numel(), device=device, dtype=torch.float32) % 19 - 9
            local[::997] = (varied + rank * 0.5).to(dtype)
            expected[::997] = functools.reduce(
                torch.add, ((varied + r * 0.5).to(dtype).float() for r in range(size)),
            ).to(dtype)
        dist.all_reduce(local)
        torch.testing.assert_close(local, expected, rtol=0, atol=0)
        torch.testing.assert_close(backing[[0, -1]], torch.full_like(backing[:2], contributions[rank]))

    for dtype, contributions in values.items():
        count = capacity * 3 // dtype.itemsize + 1
        flat_backing = torch.full((size * count + 2,), contributions[rank], dtype=dtype, device=device)
        rs_in = flat_backing[1:-1]
        out_backing = torch.full((count + 2,), contributions[rank], dtype=dtype, device=device)
        rs_out = out_backing[1:-1]
        expected_vals = torch.tensor(contributions[:size], dtype=dtype).float()
        total = expected_vals[0]
        for value in expected_vals[1:]:
            total = total + value
        dist.reduce_scatter_single(rs_out, rs_in)
        torch.testing.assert_close(rs_out, torch.full_like(rs_out, total.to(dtype).item()), rtol=0, atol=0)
        torch.testing.assert_close(flat_backing[[0, -1]], torch.full_like(flat_backing[:2], contributions[rank]))
        torch.testing.assert_close(out_backing[[0, -1]], torch.full_like(out_backing[:2], contributions[rank]))


def _check_shared_coalesced(group, shared, control_counts, manager, counts, rank, size, device):
    before_agree, before_signal = control_counts
    small_inputs = [
        torch.arange(3, dtype=torch.bfloat16, device=device) + rank,
        torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4) + rank,
        torch.empty((0,), dtype=torch.int64, device=device),
        torch.arange(5, dtype=torch.int64, device=device) + rank,
    ]
    small_outputs = [
        torch.empty((size * tensor.shape[0], *tensor.shape[1:]), dtype=tensor.dtype, device=device)
        for tensor in small_inputs
    ]
    group.allgather_into_tensor_coalesced(small_outputs, small_inputs).wait()
    for src, out in zip(small_inputs, small_outputs, strict=True):
        views = out.view(size, *src.shape)
        for peer in range(size):
            torch.testing.assert_close(views[peer], src - rank + peer, rtol=0, atol=0)
    assert control_counts[0] - before_agree == 1
    assert control_counts[1] - before_signal == 2

    large_inputs = [
        (torch.arange(shared.capacity // 2 + 3, dtype=torch.float16, device=device) % 17) + rank,
        torch.arange(shared.capacity // 4 + 5, dtype=torch.int32, device=device) + rank,
    ]
    counts[:] = [0, 0]
    gathered_large = funcol.all_gather_single_coalesced(large_inputs, group)
    for src, out in zip(large_inputs, gathered_large, strict=True):
        views = funcol.wait_tensor(out).view(size, *src.shape)
        for peer in range(size):
            torch.testing.assert_close(views[peer], src - rank + peer, rtol=0, atol=0)
    total_large_bytes = sum(tensor.nbytes for tensor in large_inputs)
    assert counts == [total_large_bytes, (size - 1) * total_large_bytes]

    empty_inputs = [
        torch.empty(0, dtype=torch.float32, device=device),
        torch.empty(0, dtype=torch.bfloat16, device=device),
    ]
    empty_outputs = [torch.empty(0, dtype=tensor.dtype, device=device) for tensor in empty_inputs]
    counts[:] = [0, 0]
    group.allgather_into_tensor_coalesced(empty_outputs, empty_inputs).wait()
    assert counts == [0, 0]
    if device.type == "cuda":
        group.allreduce_coalesced(empty_inputs).wait()
    else:
        shared.allreduce(empty_inputs, manager)
    assert counts == [0, 0]

    def make_reduce_batch(r: int) -> list[torch.Tensor]:
        bf16_rows = (
            torch.tensor([256.0, 1.0, -256.0], dtype=torch.bfloat16),
            torch.tensor([1.0, 2.0, 1.0], dtype=torch.bfloat16),
            torch.tensor([-256.0, 4.0, 256.0], dtype=torch.bfloat16),
        )
        n1 = shared.capacity // 2 + 3
        n2 = shared.capacity * 3 // 4 + 5
        return [
            bf16_rows[r].clone(),
            torch.empty((0,), dtype=torch.float32),
            (torch.arange(n1, dtype=torch.float32) % 13 - 4 + r * 0.5).to(torch.float16),
            (torch.arange(n2, dtype=torch.float32) % 29) * 0.25 + r,
        ]

    per_rank_cpu = [make_reduce_batch(r) for r in range(size)]
    reduce_inputs = [tensor.to(device) for tensor in per_rank_cpu[rank]]
    expected_outputs = [
        functools.reduce(torch.add, (per_rank_cpu[r][i].float() for r in range(size))).to(
            per_rank_cpu[0][i].dtype,
        )
        for i in range(len(reduce_inputs))
    ]
    if device.type == "cuda":
        group.allreduce_coalesced(reduce_inputs).wait()
    else:
        shared.allreduce(reduce_inputs, manager)
    for actual, expected in zip(reduce_inputs, expected_outputs, strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def _run_shared_adaptive_cpu(rank: int, path: str) -> None:
    torch.set_num_threads(1)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(path, 2), rank=rank, world_size=2,
        timeout=timedelta(seconds=20),
        pg_options=RelayOptions(transport="shared", staging_bytes=12 * 1024 * 1024),
    )
    try:
        shared = dist.group.WORLD._shared
        assert shared.capacity == 2 * 1024 * 1024
        lanes: set[int] = set()
        widths: set[int] = set()
        original_slot = shared._slot

        def tracked_slot(sender: int, lane: int, width: int) -> torch.Tensor:
            lanes.add(lane)
            widths.add(width)
            return original_slot(sender, lane, width)

        local = torch.full((1024 * 1024,), rank + 1, dtype=torch.float16)
        output = torch.empty((2, local.numel()), dtype=torch.float16)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(shared, "_slot", tracked_slot)
            dist.all_gather_single(output, local)
        assert lanes == {0, 1}
        assert widths == {1024 * 1024}
        for peer in range(2):
            torch.testing.assert_close(output[peer], torch.full_like(local, peer + 1))
    finally:
        dist.destroy_process_group()


def _run_shared_cpu_suite(rank, cases, adaptive_path):
    _run_cases(rank, _run_shared_chunks, cases)
    if adaptive_path is not None:
        _run_cases(rank, _run_shared_adaptive_cpu, [(adaptive_path,)])


@pytest.mark.parametrize(("size", "staging_sizes"), [(2, (1024,)), (3, (1024, 65536))],
                         ids=["two-ranks", "three-ranks"])
def test_shared_chunks_on_cpu(tmp_path, size, staging_sizes):
    cases = [
        (str(tmp_path / f"shared-cpu-{buffers}-{staging_bytes}"), "cpu", size, False, buffers, staging_bytes)
        for buffers in (1, 3)
        for staging_bytes in staging_sizes
    ]
    adaptive_path = str(tmp_path / "adaptive-cpu") if size == 2 else None
    mp.spawn(_run_shared_cpu_suite, args=(cases, adaptive_path), nprocs=size)


@CUDA
@pytest.mark.parametrize(("size", "options"), [
    # A nondefault Gloo pipeline option still uses three shared outgoing slots.
    (2, ((False, 1024, 1), (True, 1024, 1), (False, 1024, 3))),
    (3, ((False, 1024, 1), (True, 65536, 1))),
], ids=["two-ranks", "three-ranks"])
def test_shared_chunks_on_gpu(tmp_path, size, options):
    cases = [
        (str(tmp_path / f"shared-gpu-{mixed_pinning}-{staging_bytes}-{buffers}"),
         "cuda", size, mixed_pinning, buffers, staging_bytes)
        for mixed_pinning, staging_bytes, buffers in options
    ]
    mp.spawn(_run_cases, args=(_run_shared_chunks, cases), nprocs=size)


def _run_shared_registration_refusal(rank, path):
    from dataclasses import replace

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(path, 2), rank=rank, world_size=2,
        timeout=timedelta(seconds=10),
        pg_options=RelayOptions(transport="shared", staging_bytes=1024),
    )
    shared = dist.group.WORLD._shared
    registration = RuntimeHostRegistration()
    runtime = registration._runtime
    assert runtime is not None
    manager = PinManager(16384, backend=registration)
    refusing = True
    refusals = 0

    def register(pointer, size, flags):
        nonlocal refusals
        if rank == 0 and refusing and pointer == shared.buffer.data_ptr():
            refusals += 1
            return 1
        return runtime.register(pointer, size, flags)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(registration, "_runtime", replace(runtime, register=register))
            patch.setattr(communication, "host_pin_manager", manager)
            # One peer repeatedly refuses registration while the other pins.
            # All operations must finish and the same group must remain usable.
            for _ in range(2):
                _check_chunked_values(rank, 2, device, torch.float32)
                assert not shared.broken
                assert manager.stats.active_leases == 0
            assert manager.stats.registration_failures == refusals
            if rank == 0:
                assert refusals > 1
                assert manager.stats.pinned_bytes == 0
                assert shared.device is None
            else:
                assert manager.stats.registrations == 1
                assert shared.device == device
            # A later successful registration can use asynchronous copies.
            refusing = False
            _check_chunked_values(rank, 2, device, torch.float32)
            assert manager.stats.registrations == 1
            assert shared.device == device
            assert not shared.broken
    finally:
        dist.destroy_process_group()
        manager.clear()
    assert manager.stats.active_leases == manager.stats.pinned_bytes == 0


@CUDA
def test_shared_registration_refusal_keeps_group_usable(tmp_path):
    mp.spawn(_run_shared_registration_refusal, args=(str(tmp_path / "registration-refusal"),), nprocs=2)


def _run_shared_failure(rank, path, operation):
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(path, 2), rank=rank, world_size=2,
        timeout=timedelta(seconds=5),
        pg_options=RelayOptions(transport="shared", staging_bytes=512),
    )
    group = dist.group.WORLD
    shared = group._shared
    registration = _CountingRegistration()
    manager = PinManager(16384, backend=registration)
    signal = shared._signal
    unregister = registration.unregister

    def fail_release(peers, tag, sequence):
        if tag == 0 and sequence == 1:
            with torch.cuda.stream(shared.upload):
                torch.cuda._sleep(5_000_000)
        if tag == 1:
            manager.max_pinned_bytes = 0
            assert manager.stats.active_leases == 1
            assert registration.unregister_calls == 0
            raise RuntimeError("injected shared acknowledgement failure")
        signal(peers, tag, sequence)

    def check_unregister(pointer):
        assert shared.download.query()
        assert shared.upload.query()
        unregister(pointer)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(communication, "host_pin_manager", manager)
        patch.setattr(shared, "_signal", fail_release)
        patch.setattr(registration, "unregister", check_unregister)
        local = torch.full((257,), rank + 1, device="cuda", dtype=torch.bfloat16)
        output = torch.empty((2, 257), device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="acknowledgement failure"):
            if operation == "allreduce":
                dist.all_reduce(local)
            else:
                dist.all_gather_single(output, local)
        assert shared.broken
        assert manager.stats.active_leases == 0
        assert manager.stats.pinned_bytes == 0
        assert registration.unregister_calls == 1
        with pytest.raises(RuntimeError, match="failed previously"):
            dist.all_reduce(local)
        dist.destroy_process_group()


@CUDA
def test_shared_failure_drains_copies_before_unpinning(tmp_path):
    cases = [
        (str(tmp_path / f"shared-failure-{operation}"), operation)
        for operation in ("allgather", "allreduce")
    ]
    mp.spawn(_run_cases, args=(_run_shared_failure, cases), nprocs=2)


@pytest.mark.parametrize("transport", [None, "auto", "nccl", True])
def test_reject_invalid_relay_transport(transport):
    with pytest.raises(ValueError, match="transport must be"):
        RelayOptions(transport=transport)


def _run_ipc_option_mismatch(rank: int, path: str) -> None:
    torch.set_num_threads(1)
    register_relay_backend()
    with pytest.raises(ValueError, match="must match across all ranks"):
        dist.init_process_group(
            "piper_relay",
            store=dist.FileStore(path, 2),
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=5),
            pg_options=RelayOptions(transport="ipc" if rank == 0 else "shared", staging_bytes=512),
        )


def test_ipc_options_validation_and_mismatch(tmp_path):
    assert RelayOptions(transport="ipc").transport == "ipc"
    with pytest.raises(ValueError, match="staging slot"):
        RelayOptions(staging_bytes=16, transport="ipc")._validate_group_size(2)

    mp.spawn(_run_cases, args=(_run_ipc_option_mismatch, [(str(tmp_path / "ipc-opt-mismatch"),)]), nprocs=2)


def _run_ipc_cpu_collectives(rank: int, path: str) -> None:
    torch.set_num_threads(1)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay",
        store=dist.FileStore(path, 2),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=5),
        pg_options=RelayOptions(transport="ipc", staging_bytes=4096),
    )
    try:
        shared = dist.group.WORLD._shared
        x = torch.full((8,), float(rank + 1))
        dist.all_reduce(x)
        torch.testing.assert_close(x, torch.full((8,), 3.0))
        b = torch.full((8,), 5.0 if rank == 0 else 0.0)
        dist.broadcast(b, src=0)
        torch.testing.assert_close(b, torch.full((8,), 5.0))
        local = torch.full((4,), float(rank + 1))
        gathered = torch.empty((2, 4))
        dist.all_gather_single(gathered, local)
        torch.testing.assert_close(gathered, torch.tensor([[1.0] * 4, [2.0] * 4]))
        assert not shared.ipc_active
        assert not shared._ipc_checked
    finally:
        dist.destroy_process_group()


def test_ipc_transport_cpu_collectives_leave_ipc_inactive(tmp_path):
    mp.spawn(_run_cases, args=(_run_ipc_cpu_collectives, [(str(tmp_path / "ipc-cpu"),)]), nprocs=2)


def test_shared_mapping_survives_failed_unregistration(monkeypatch, tmp_path):
    import weakref
    from piper_offload import _relay_shared

    create = _relay_shared._create_mapping
    mappings = []

    def capture(size):
        mapping, name = create(size)
        mappings.append(weakref.ref(mapping))
        return mapping, name

    class Registration:
        fail = True

        def register(self, pointer, size):
            return True

        def unregister(self, pointer):
            if self.fail:
                raise RuntimeError("injected unregister failure")

    monkeypatch.setattr(_relay_shared, "_create_mapping", capture)
    store = dist.FileStore(str(tmp_path / "mapping-lifetime"), 1)
    tensor = _relay_shared.create_shared_buffer(store, 0, 1, 4096, timedelta(seconds=2))
    backend = Registration()
    manager = PinManager(backend=backend)
    manager.acquire([tensor]).close()
    del tensor
    gc.collect()
    assert mappings[0]() is not None
    assert not mappings[0]().closed
    assert manager.stats.pinned_bytes == 4096
    backend.fail = False
    manager.clear()
    gc.collect()
    assert mappings[0]() is None


def _run_shared_bad_mapping(rank, path, failure):
    from piper_offload import _relay_shared

    register_relay_backend()
    with pytest.MonkeyPatch.context() as patch:
        if rank == 1:
            if failure == "host":
                patch.setattr(_relay_shared.socket, "gethostname", lambda: "different-host")
            else:
                def unavailable(*args):
                    raise OSError("injected unavailable mapping")
                patch.setattr(_relay_shared, "_open_mapping", unavailable)
        with pytest.raises(RuntimeError, match="Could not initialize shared relay"):
            dist.init_process_group(
                "piper_relay", store=dist.FileStore(path, 2), rank=rank, world_size=2,
                timeout=timedelta(seconds=5), pg_options=RelayOptions(transport="shared"),
            )
        assert not dist.is_initialized()


def test_shared_initialization_fails_on_all_ranks(tmp_path):
    cases = [(str(tmp_path / f"bad-map-{failure}"), failure) for failure in ("host", "mapping")]
    mp.spawn(_run_cases, args=(_run_shared_bad_mapping, cases), nprocs=2)


def _run_shared_peer_failure(rank, path):
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(path, 2), rank=rank, world_size=2,
        timeout=timedelta(seconds=2), pg_options=RelayOptions(transport="shared", staging_bytes=512),
    )
    group = dist.group.WORLD
    shared = group._shared
    local = torch.arange(257, dtype=torch.int64) + rank
    output = torch.empty((2, 257), dtype=torch.int64)
    signal = shared._signal

    def fail_sender(peers, tag, sequence):
        if rank == 0 and tag == 0 and sequence == 1:
            raise RuntimeError("injected sender failure")
        signal(peers, tag, sequence)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(shared, "_signal", fail_sender)
        with pytest.raises(RuntimeError):
            dist.all_gather_single(output, local)
        assert shared.broken
        with pytest.raises(RuntimeError, match="failed previously"):
            dist.all_gather_single(output, local)
        dist.destroy_process_group()


def test_shared_peer_failure_times_out_without_reusing_slots(tmp_path):
    mp.spawn(_run_shared_peer_failure, args=(str(tmp_path / "peer-failure"),), nprocs=2)


def _run_shared_agreement_failure(rank: int, path: str, failure: str) -> None:
    from piper_offload._relay_signal import SharedControl, shared_control_supported

    torch.set_num_threads(1)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(path, 2), rank=rank, world_size=2,
        timeout=timedelta(seconds=15 if failure == "interrupted" and shared_control_supported() else 2),
        pg_options=RelayOptions(transport="shared", staging_bytes=512),
    )
    shared = dist.group.WORLD._shared
    store = dist.distributed_c10d._get_default_store()
    local = torch.arange(257, dtype=torch.int64) + rank
    output = torch.empty((2, 257), dtype=torch.int64)
    p2p_buf = torch.empty(8, dtype=torch.int64)
    try:
        if failure == "absent" and rank == 0:
            # Never joins the collective; rank 1's metadata exchange times out.
            store.wait(["checked"], timedelta(seconds=30))
            return
        with pytest.MonkeyPatch.context() as patch:
            if failure == "interrupted" and rank == 0:
                orig_agree = shared._control.agree

                def interrupted_agree(*args: object) -> None:
                    orig_agree(*args)
                    raise Interrupt

                patch.setattr(shared._control, "agree", interrupted_agree)
            start = time.monotonic()
            with pytest.raises(Interrupt if failure == "interrupted" and rank == 0 else RuntimeError):
                dist.all_gather_single(output, local)
            elapsed = time.monotonic() - start
            if failure == "interrupted" and rank == 1 and isinstance(shared._control, SharedControl):
                assert elapsed < 5.0
            assert shared.broken
            with pytest.raises(RuntimeError, match="failed previously"):
                dist.all_gather_single(output, local)
            with pytest.raises(RuntimeError, match="failed previously"):
                if rank == 0:
                    dist.send(p2p_buf, dst=1)
                else:
                    dist.recv(p2p_buf, src=0)
        if failure == "absent" and rank == 1:
            store.set("checked", "1")
    finally:
        dist.destroy_process_group()


def test_shared_failure_from_agreement_on_refuses_the_group(tmp_path):
    cases = [(str(tmp_path / f"agreement-{failure}"), failure) for failure in ("interrupted", "absent")]
    mp.spawn(_run_cases, args=(_run_shared_agreement_failure, cases), nprocs=2)


def _run_shared_metadata_mismatch(rank, path, device_type):
    if device_type == "cuda":
        torch.cuda.set_device(0)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(path, 2), rank=rank, world_size=2,
        timeout=timedelta(seconds=5), pg_options=RelayOptions(transport="shared", staging_bytes=512),
    )
    local = torch.full((rank + 1,), rank, dtype=torch.int64, device=device_type)
    output = torch.full((2, rank + 1), -1, dtype=torch.int64, device=device_type)
    with pytest.raises(ValueError, match="payload bytes"):
        dist.all_gather_single(output, local)
    assert torch.all(output == -1)
    copy_dtype = torch.float32 if rank == 0 else torch.int32
    copy_value = torch.full((1,), rank, dtype=copy_dtype, device=device_type)
    copy_output = torch.full((2, 1), -1, dtype=copy_dtype, device=device_type)
    with pytest.raises(ValueError, match="root/dtype"):
        dist.all_gather_single(copy_output, copy_value)
    assert torch.all(copy_output == -1)
    received = torch.empty((2, 1), dtype=torch.int64, device=device_type)
    dist.all_gather_single(received, local[:1])
    assert torch.equal(received.flatten().cpu(), torch.arange(2))
    if device_type == "cuda":
        # Equal byte counts can still disagree on reduction dtype or operation.
        value = torch.full((257,), rank + 1, dtype=DTYPES[rank + 1], device=device_type)
        with pytest.raises(ValueError, match="root/dtype"):
            dist.all_reduce(value)
        torch.testing.assert_close(value, torch.full_like(value, rank + 1))
        value = value.float()
        output = torch.full((2, value.numel()), -1, dtype=value.dtype, device=device_type)
        with pytest.raises(ValueError, match="collective operation"):
            if rank == 0:
                dist.all_reduce(value)
            else:
                dist.all_gather_single(output, value)
        torch.testing.assert_close(value, torch.full_like(value, rank + 1))
        assert torch.all(output == -1)
        dist.all_reduce(value)
        torch.testing.assert_close(value, torch.full_like(value, 3))

    _check_shared_coalesced_mismatch(dist.group.WORLD, local, rank, device_type)
    dist.destroy_process_group()


def _check_shared_coalesced_mismatch(group, local, rank, device_type):
    # Coalesced collectives validate tensor count and per-tensor split/dtype before writes.
    count_inputs = [local[:1]] if rank == 0 else [local[:1], local[:1]]
    count_outputs = [torch.full((2,), -1, dtype=torch.int64, device=device_type) for _ in count_inputs]
    with pytest.raises(ValueError, match="must match across ranks"):
        group.allgather_into_tensor_coalesced(count_outputs, count_inputs).wait()
    assert all(torch.all(out == -1) for out in count_outputs)

    split_shapes = (2, 2) if rank == 0 else (1, 3)
    split_inputs = [torch.full((n,), rank, dtype=torch.int32, device=device_type) for n in split_shapes]
    split_outputs = [torch.full((2 * n,), -1, dtype=torch.int32, device=device_type) for n in split_shapes]
    with pytest.raises(ValueError, match="must match across ranks"):
        group.allgather_into_tensor_coalesced(split_outputs, split_inputs).wait()
    assert all(torch.all(out == -1) for out in split_outputs)

    second_dtype = torch.float32 if rank == 0 else torch.int32
    dtype_inputs = [
        torch.full((2,), rank, dtype=torch.float32, device=device_type),
        torch.full((2,), rank, dtype=second_dtype, device=device_type),
    ]
    dtype_outputs = [torch.full((4,), -1, dtype=t.dtype, device=device_type) for t in dtype_inputs]
    with pytest.raises(ValueError, match="must match across ranks"):
        group.allgather_into_tensor_coalesced(dtype_outputs, dtype_inputs).wait()
    assert all(torch.all(out == -1) for out in dtype_outputs)

    if device_type == "cuda":
        reduce_pair = [
            torch.full((2,), float(rank + 1), dtype=torch.float32, device=device_type),
            torch.full((2,), float(rank + 1), dtype=DTYPES[rank + 1], device=device_type),
        ]
        with pytest.raises(ValueError, match="must match across ranks"):
            group.allreduce_coalesced(reduce_pair).wait()
        for tensor in reduce_pair:
            torch.testing.assert_close(tensor, torch.full_like(tensor, float(rank + 1)))

    valid_inputs = [local[:1], local[:1].to(torch.float32)]
    valid_outputs = [torch.empty((2,), dtype=t.dtype, device=device_type) for t in valid_inputs]
    group.allgather_into_tensor_coalesced(valid_outputs, valid_inputs).wait()
    torch.testing.assert_close(valid_outputs[0].cpu(), torch.arange(2, dtype=torch.int64))
    torch.testing.assert_close(valid_outputs[1].cpu(), torch.arange(2, dtype=torch.float32))


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_shared_metadata_mismatch_fails_before_writes(tmp_path, device_type):
    mp.spawn(_run_shared_metadata_mismatch, args=(str(tmp_path / "bad-metadata"), device_type), nprocs=2)


def test_shared_chunk_bytes_adapts_within_capacity():
    from piper_offload._relay_shared import _MIN_PIPELINE_CHUNK_BYTES, _TARGET_PIPELINE_CHUNKS, _shared_chunk_bytes

    large_capacity = 8 * 1024 * 1024
    assert _shared_chunk_bytes(0, large_capacity) == large_capacity
    assert _shared_chunk_bytes(1024, large_capacity) == large_capacity
    assert _shared_chunk_bytes(_MIN_PIPELINE_CHUNK_BYTES, large_capacity) == large_capacity
    assert _shared_chunk_bytes(1536 * 1024, large_capacity) == large_capacity
    assert _shared_chunk_bytes(2 * _MIN_PIPELINE_CHUNK_BYTES, 512) == 512

    default_2rank_capacity = 1_396_736
    assert _shared_chunk_bytes(1536 * 1024, default_2rank_capacity) == 768 * 1024

    assert _shared_chunk_bytes(2 * 1024 * 1024, large_capacity) == 1024 * 1024
    assert _shared_chunk_bytes(4 * 1024 * 1024, large_capacity) == 1024 * 1024

    eight_mib = 8 * 1024 * 1024
    ten_mib = 10 * 1024 * 1024
    chunk_8m = _shared_chunk_bytes(eight_mib, ten_mib)
    assert 0 < chunk_8m <= ten_mib
    assert chunk_8m % 4096 == 0
    assert -(-eight_mib // chunk_8m) == _TARGET_PIPELINE_CHUNKS

    slightly_over = 2 * 1024 * 1024 + 16
    chunk_over = _shared_chunk_bytes(slightly_over, 2 * 1024 * 1024)
    assert chunk_over == 1024 * 1024 + 4096
    assert -(-slightly_over // chunk_over) == 2

    fifty_mib = 50 * 1024 * 1024
    four_mib = 4 * 1024 * 1024
    chunk_50m = _shared_chunk_bytes(fifty_mib, four_mib)
    assert 0 < chunk_50m <= four_mib
    assert chunk_50m % 4096 == 0
    assert -(-fifty_mib // chunk_50m) == -(-fifty_mib // four_mib)


def _make_fake_shared_relay(tmp_path, name: str = "chunks", capacity: int = 64):
    from contextlib import nullcontext
    from piper_offload._relay_shared import SharedRelay

    class FakeEvent:
        def __init__(self, events: list[tuple[object, ...]], label: str) -> None:
            self.events = events
            self.label = label
            self.counter = 0
            self.token: tuple[str, int, object] | None = None

        def record(self, stream: object) -> None:
            self.counter += 1
            self.token = (self.label, self.counter, stream)
            self.events.append(("record", self.label, self.counter))

        def synchronize(self) -> None:
            self.events.append(("sync", self.label, self.token))

    events: list[tuple[object, ...]] = []
    shared = SharedRelay(
        dist.FileStore(str(tmp_path / name), 1),
        None,
        rank=0,
        world_size=1,
        nbytes=3 * capacity,
        timeout=timedelta(seconds=5),
    )
    shared.download = "download"
    shared.upload = "upload"
    original_slot = shared._slot

    def tracked_slot(sender: int, lane: int, width: int) -> torch.Tensor:
        events.append(("download", lane, width))
        return original_slot(sender, lane, width)

    def fake_signal(peers, tag, sequence) -> None:
        if any(send is not None for send, _ in peers):
            events.append(("ready_signal" if tag == 0 else "free_signal", sequence))

    def reset() -> None:
        events.clear()
        shared.ready = [FakeEvent(events, f"ready[{lane}]") for lane in range(3)]
        shared.uploaded = [FakeEvent(events, f"uploaded[{lane}]") for lane in range(3)]

    def make_fill(src: torch.Tensor):
        def fill(slot: torch.Tensor, start: int, width: int) -> None:
            slot.copy_(src[start:start + width], non_blocking=True)

        return fill

    shared._slot = tracked_slot
    shared._signal = fake_signal
    return shared, events, reset, make_fill, nullcontext


def test_shared_chunks_prefetches_ahead_and_releases_once(tmp_path):
    capacity = 64
    shared, events, reset, make_fill, nullcontext = _make_fake_shared_relay(tmp_path, capacity=capacity)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
        for count in (1, 2, 3, 4, 5):
            reset()
            source = torch.arange(count * capacity, dtype=torch.uint8)
            received = torch.empty_like(source)
            for index, start, width in shared._chunks(
                make_fill(source), source.numel(), [(1, 1)], asynchronous=True,
            ):
                events.append(("yield", index, start, width))
                lane_start = (index % 3) * capacity
                received[start:start + width].copy_(shared.buffer[lane_start:lane_start + width])
            assert torch.equal(received, source)

            expected_frees = {1: [0], 2: [1], 3: [2], 4: [0, 3], 5: [0, 1, 4]}[count]
            assert [item[1] for item in events if item[0] == "free_signal"] == expected_frees
            for old in expected_frees:
                sync_uploaded = ("sync", f"uploaded[{old % 3}]", (f"uploaded[{old % 3}]", old // 3 + 1, "upload"))
                assert events.index(sync_uploaded) < events.index(("free_signal", old))

            first_sync = events.index(("sync", "ready[0]", ("ready[0]", 1, "download")))
            initial_downloads = [idx for idx, ev in enumerate(events) if ev[0] == "download"][:min(3, count)]
            assert all(idx < first_sync for idx in initial_downloads)
            for index in range(count):
                lane = index % 3
                assert ("sync", f"ready[{lane}]", (f"ready[{lane}]", index // 3 + 1, "download")) in events

            if count == 5:
                downloads = [idx for idx, ev in enumerate(events) if ev[0] == "download"]
                ready2_sync = events.index(("sync", "ready[2]", ("ready[2]", 1, "download")))
                assert (
                    events.index(("free_signal", 0))
                    < downloads[3]
                    < ready2_sync
                    < events.index(("ready_signal", 2))
                )


def test_shared_chunks_skips_unused_direction_events(tmp_path):
    capacity = 64
    shared, events, reset, make_fill, nullcontext = _make_fake_shared_relay(tmp_path, name="unidir", capacity=capacity)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())

        # Receive-only: skips downloads and ready events, still syncs uploads before freeing.
        reset()
        for index, start, width in shared._chunks(None, 4 * capacity, [(None, 0)], asynchronous=True):
            events.append(("yield", index, start, width))
        assert not any(ev[0] == "download" or str(ev[1]).startswith("ready[") for ev in events)
        assert [ev[1] for ev in events if ev[0] == "free_signal"] == [0, 3]
        for old in (0, 3):
            sync_uploaded = ("sync", f"uploaded[{old % 3}]", (f"uploaded[{old % 3}]", old // 3 + 1, "upload"))
            assert events.index(sync_uploaded) < events.index(("free_signal", old))

        # Send-only: skips upload events, still prefetches ahead and syncs ready before sending.
        reset()
        source = torch.arange(4 * capacity, dtype=torch.uint8)
        received = torch.empty_like(source)
        for index, start, width in shared._chunks(
            make_fill(source), source.numel(), [(1, None)], asynchronous=True,
        ):
            lane_start = (index % 3) * capacity
            received[start:start + width].copy_(shared.buffer[lane_start:lane_start + width])
        assert torch.equal(received, source)
        assert not any(str(ev[1]).startswith("uploaded[") for ev in events)
        assert [ev[1] for ev in events if ev[0] == "ready_signal"] == [0, 1, 2, 3]
        downloads = [idx for idx, ev in enumerate(events) if ev[0] == "download"]
        assert downloads[3] < events.index(("sync", "ready[2]", ("ready[2]", 1, "download")))


def test_shared_control_agreement_and_abort():
    from piper_offload._relay_signal import SharedControl, control_bytes

    assert control_bytes(2) >= 4096
    assert control_bytes(2) % 4096 == 0
    assert control_bytes(64) % 4096 == 0

    mapping = mmap.mmap(-1, control_bytes(2))
    c0 = SharedControl(mapping, 0, 2)
    c1 = SharedControl(mapping, 1, 2)
    for seq in range(4):
        errors: list[BaseException] = []

        def worker(ctl: SharedControl, idx: int, s: int, errs: list[BaseException]) -> None:
            try:
                ctl.agree(1, 64, 0, s, timedelta(seconds=2))
                ctl.signal([(1 - idx, 1 - idx)], 0, s, timedelta(seconds=2))
                ctl.signal([(1 - idx, 1 - idx)], 1, s, timedelta(seconds=2))
            except BaseException as exc:
                errs.append(exc)

        t0 = threading.Thread(target=worker, args=(c0, 0, seq, errors))
        t1 = threading.Thread(target=worker, args=(c1, 1, seq, errors))
        t0.start()
        t1.start()
        t0.join()
        t1.join()
        assert not errors
        assert c0._expected_ready[1] == seq + 1 and c1._expected_ready[0] == seq + 1
        assert c0._expected_free[1] == seq + 1 and c1._expected_free[0] == seq + 1

    mismatch_errors: list[BaseException] = []

    def agree_detail(ctl: SharedControl, detail: int, errs: list[BaseException]) -> None:
        try:
            ctl.agree(1, 64, 0, detail, timedelta(seconds=2))
        except BaseException as exc:
            errs.append(exc)

    for details in ((10, 20), (30, 30)):
        mismatch_errors.clear()
        t0 = threading.Thread(target=agree_detail, args=(c0, details[0], mismatch_errors))
        t1 = threading.Thread(target=agree_detail, args=(c1, details[1], mismatch_errors))
        t0.start()
        t1.start()
        t0.join()
        t1.join()
        if details[0] != details[1]:
            assert len(mismatch_errors) == 2
            assert all(isinstance(err, ValueError) for err in mismatch_errors)
        else:
            assert not mismatch_errors

    c0.abort()
    with pytest.raises(RuntimeError, match="aborted by peer"):
        c1.wait_ready([(0, c0._local_ready + 1)], timedelta(seconds=1))


def test_shared_control_timeout_without_abort():
    from piper_offload._relay_signal import SharedControl, control_bytes

    mapping = mmap.mmap(-1, control_bytes(2))
    ctl = SharedControl(mapping, 0, 2)
    with pytest.raises(RuntimeError, match="timed out waiting for peer"):
        ctl.agree(0, 64, 0, 0, timedelta(milliseconds=50))
    with pytest.raises(RuntimeError, match="timed out waiting for peer"):
        ctl.wait_ready([(1, 1)], timedelta(milliseconds=50))


def _run_shared_wakeup(rank: int, path: str) -> None:
    from piper_offload._relay_signal import SharedControl

    torch.set_num_threads(1)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(path, 2), rank=rank, world_size=2,
        timeout=timedelta(seconds=15),
        pg_options=RelayOptions(transport="shared", staging_bytes=4096),
    )
    shared = dist.group.WORLD._shared
    try:
        if not isinstance(shared._control, SharedControl):
            return
        # 1. Delayed metadata agreement wakes peer sleeping past the spin window.
        if rank == 0:
            time.sleep(0.02)
        bcast = torch.full((4,), 7 if rank == 0 else 0, dtype=torch.int64)
        dist.broadcast(bcast, src=0)
        assert torch.all(bcast == 7)

        # 2. Delayed chunk ready signal wakes sleeping peer.
        original_publish = shared._control.publish_ready

        def delayed_publish() -> None:
            time.sleep(0.02)
            original_publish()

        with pytest.MonkeyPatch.context() as patch:
            if rank == 0:
                patch.setattr(shared._control, "publish_ready", delayed_publish)
            local = torch.full((4,), rank + 1, dtype=torch.int64)
            gathered = torch.empty((2, 4), dtype=torch.int64)
            dist.all_gather_single(gathered, local)
            assert torch.equal(gathered, torch.tensor([[1] * 4, [2] * 4], dtype=torch.int64))

        # 3. Mid-collective abort wakes sleeping peer well before the 15 s timeout.
        with pytest.MonkeyPatch.context() as patch:
            if rank == 0:
                def fail_signal(peers, tag, sequence):
                    del peers, tag, sequence
                    time.sleep(0.02)
                    raise RuntimeError("injected wakeup abort")

                patch.setattr(shared, "_signal", fail_signal)
            start = time.monotonic()
            with pytest.raises(RuntimeError):
                dist.all_gather_single(gathered, local)
            elapsed = time.monotonic() - start
            assert shared.broken
            if rank == 1:
                assert elapsed < 5.0
    finally:
        dist.destroy_process_group()


def test_shared_control_wakes_sleeping_peer(tmp_path):
    mp.spawn(_run_shared_wakeup, args=(str(tmp_path / "wakeup"),), nprocs=2)


def test_shared_relay_gloo_control_fallback(tmp_path, monkeypatch):
    from piper_offload import _relay_shared

    store = dist.FileStore(str(tmp_path / "fallback"), 1)
    prefix = dist.PrefixStore("piper_relay_shared", store)
    for peer in range(1, 4):
        prefix.set(str(peer), "")

    events: list[tuple[str, str, int, int]] = []
    replies: dict[int, int] = {1: 7, 2: 7, 3: 7}
    peer_detail: int | None = None
    agree_error: BaseException | None = None

    class RecordingWork:
        def __init__(self, kind: str, peer: int, tag: int, target: torch.Tensor | None = None) -> None:
            self.kind = kind
            self.peer = peer
            self.tag = tag
            self.target = target

        def wait(self, timeout: timedelta) -> None:
            del timeout
            if self.kind == "allgather" and agree_error is not None:
                raise agree_error
            events.append(("wait", self.kind, self.peer, self.tag))
            if self.target is not None:
                self.target.fill_(replies[self.peer])

    class RecordingGroup:
        def send(self, tensors: list[torch.Tensor], peer: int, tag: int) -> RecordingWork:
            del tensors
            events.append(("post", "send", peer, tag))
            return RecordingWork("send", peer, tag)

        def recv(self, tensors: list[torch.Tensor], peer: int, tag: int) -> RecordingWork:
            events.append(("post", "recv", peer, tag))
            return RecordingWork("recv", peer, tag, tensors[0])

        def allgather(self, gathered: list[list[torch.Tensor]], metadata: list[torch.Tensor]) -> RecordingWork:
            for item in gathered[0]:
                item.copy_(metadata[0])
            if peer_detail is not None:
                gathered[0][1][3] = peer_detail
            return RecordingWork("allgather", 0, 0)

    monkeypatch.setattr(_relay_shared, "shared_control_supported", lambda: False)
    shared = _relay_shared.SharedRelay(
        store, RecordingGroup(), rank=0, world_size=4, nbytes=192, timeout=timedelta(seconds=5),
    )
    assert isinstance(shared._control, _relay_shared._GlooControl)
    shared._signal(shared._peers, tag=0, sequence=7)
    assert len(events) == 12
    assert all(ev[0] == "post" for ev in events[:6])
    assert all(ev[0] == "wait" for ev in events[6:])
    assert shared._control._pending == []

    replies[2] = 6
    with pytest.raises(RuntimeError, match="control sequence mismatch"):
        shared._signal(shared._peers, tag=0, sequence=7)
    assert shared._control._pending == []

    sample = torch.empty(4, dtype=torch.float32)
    shared._agree_operation("allreduce", [sample], -1)
    assert not shared.broken
    peer_detail = 99
    with pytest.raises(ValueError, match="must match across ranks"):
        shared._agree_operation("allreduce", [sample], -1)
    assert not shared.broken
    peer_detail = None
    agree_error = Interrupt()
    with pytest.raises(Interrupt):
        shared._agree_operation("allreduce", [sample], -1)
    assert shared.broken
    shared.broken = False
    agree_error = RuntimeError("injected gloo agree failure")
    with pytest.raises(RuntimeError, match="injected gloo agree failure"):
        shared._agree_operation("allreduce", [sample], -1)
    assert shared.broken


def test_shared_prepare_streams_failure_propagates_original_error(tmp_path):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from piper_offload._relay_shared import SharedRelay

    class MetadataWork:
        def wait(self, timeout: timedelta) -> None:
            del timeout

    class MetadataGroup:
        def allgather(self, gathered: list[list[torch.Tensor]], metadata: list[torch.Tensor]) -> MetadataWork:
            gathered[0][0].copy_(metadata[0])
            return MetadataWork()

    class StubManager:
        def __init__(self) -> None:
            self.closed = False

        @contextmanager
        def acquire(self, tensors: list[torch.Tensor]):
            del tensors
            try:
                yield SimpleNamespace(pageable_bytes=0)
            finally:
                self.closed = True

    class DummyStream:
        def __init__(self) -> None:
            self.synchronized = False

        def synchronize(self) -> None:
            self.synchronized = True

    shared = SharedRelay(
        dist.FileStore(str(tmp_path / "streams"), 1),
        MetadataGroup(),
        rank=0,
        world_size=1,
        nbytes=192,
        timeout=timedelta(seconds=5),
    )
    manager = StubManager()
    current = DummyStream()
    fake_tensor = SimpleNamespace(device=torch.device("cuda", 0), nbytes=16, dtype=torch.float32)
    stream_calls = 0

    def failing_stream(device: torch.device) -> DummyStream:
        del device
        nonlocal stream_calls
        stream_calls += 1
        if stream_calls == 2:
            raise RuntimeError("injected stream creation failure")
        return DummyStream()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "Stream", failing_stream)
        patch.setattr(torch.cuda, "current_stream", lambda _device: current)
        with pytest.raises(RuntimeError, match="injected stream creation failure"):
            with shared._collective("allreduce", [fake_tensor], -1, manager):
                pytest.fail("collective body ran after stream creation failure")

    assert shared.broken
    assert manager.closed
    assert current.synchronized
    assert shared.device is None
    assert shared.download is None
    assert shared.upload is None
    assert shared.ready == []
    assert shared.uploaded == []


def _make_fake_collective_relay(tmp_path):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from piper_offload._relay_shared import SharedRelay

    class StubControl:
        def __init__(self) -> None:
            self.fail_on_signal = False
            self.abort_count = 0

        def agree(self, operation: int, nbytes: int, kind: int, detail: int, timeout: timedelta) -> None:
            del operation, nbytes, kind, detail, timeout

        def signal(self, peers: object, tag: int, sequence: int, timeout: timedelta) -> None:
            del peers, tag, sequence, timeout
            if self.fail_on_signal:
                raise RuntimeError("injected signal failure")

        def skip_ready(self, peer: int, count: int) -> None:
            del peer, count

        def abort(self) -> None:
            self.abort_count += 1

    class StubManager:
        def __init__(self) -> None:
            self.pageable_bytes = 0

        @contextmanager
        def acquire(self, tensors: list[torch.Tensor]):
            del tensors
            yield SimpleNamespace(pageable_bytes=self.pageable_bytes)

    class DummyStream:
        def __init__(self, name: str) -> None:
            self.name = name
            self.device = torch.device("cuda", 0)
            self.sync_count = 0

        def wait_stream(self, other: object) -> None:
            del other

        def synchronize(self) -> None:
            self.sync_count += 1

    class DummyEvent:
        def record(self, stream: object) -> None:
            del stream

        def synchronize(self) -> None:
            pass

    class FakeCudaTensor:
        device = torch.device("cuda", 0)

        def __init__(self, data: torch.Tensor) -> None:
            self.data = data
            self.dtype = data.dtype
            self.nbytes = data.nbytes

        def numel(self) -> int:
            return self.data.numel()

        def reshape(self, *shape: int) -> torch.Tensor:
            return self.data.reshape(*shape)

    download_stream = DummyStream("download")
    upload_stream = DummyStream("upload")
    current = DummyStream("current")
    stream_entries: list[str] = []

    @contextmanager
    def track_stream(stream: DummyStream):
        stream_entries.append(stream.name)
        yield

    shared = SharedRelay(
        dist.FileStore(str(tmp_path / "syncs"), 1),
        None,
        rank=0,
        world_size=1,
        nbytes=2 * 3 * 64,
        timeout=timedelta(seconds=5),
    )
    control = StubControl()
    manager = StubManager()
    shared._control = control
    shared.capacity = 64
    return (
        shared, control, manager, download_stream, upload_stream,
        current, stream_entries, track_stream, DummyEvent, FakeCudaTensor,
    )


def _check_async_stream_syncs(
    shared, manager, fake_cuda_tensor, syncs, reset_counts, copy_streams,
) -> None:
    # Async allgather (world_size=2): _local_copy on upload, 0 final stream syncs.
    shared.world_size = 2
    shared._peers = ((1, 1),)
    ag_src = torch.arange(8, dtype=torch.float32)
    ag_outs = [torch.zeros(8, dtype=torch.float32), torch.zeros(8, dtype=torch.float32)]
    shared._slot(1, 0, 32).copy_(torch.full((8,), 9.0, dtype=torch.float32).view(torch.uint8))
    shared.allgather([fake_cuda_tensor(ag_src)], [[fake_cuda_tensor(t) for t in ag_outs]], manager)
    assert syncs() == (0, 0, 0)
    assert copy_streams[-1] == (True, ["upload"])
    assert torch.equal(ag_outs[0], ag_src)
    assert torch.equal(ag_outs[1], torch.full((8,), 9.0, dtype=torch.float32))

    # Async broadcast root & non-root (world_size=2): 0 final stream syncs.
    reset_counts()
    bc = torch.arange(8, dtype=torch.float32)
    shared.broadcast(fake_cuda_tensor(bc), root=0, manager=manager)
    assert syncs() == (0, 0, 0)
    reset_counts()
    bc_recv = torch.zeros(8, dtype=torch.float32)
    shared._slot(1, 0, 32).copy_(torch.full((8,), 4.0, dtype=torch.float32).view(torch.uint8))
    shared.broadcast(fake_cuda_tensor(bc_recv), root=1, manager=manager)
    assert syncs() == (0, 0, 0)
    assert torch.equal(bc_recv, torch.full((8,), 4.0, dtype=torch.float32))

    # Async scatter root non-alias -> only upload synced; self-alias -> 0 stream syncs.
    reset_counts()
    sc_srcs = [torch.arange(8, dtype=torch.float32), torch.arange(8, dtype=torch.float32) + 1]
    sc_out = torch.zeros(8, dtype=torch.float32)
    shared.scatter([fake_cuda_tensor(t) for t in sc_srcs], fake_cuda_tensor(sc_out), root=0, manager=manager)
    assert syncs() == (0, 1, 0)
    assert copy_streams[-1] == (True, ["upload"])
    assert torch.equal(sc_out, sc_srcs[0])
    reset_counts()
    shared.scatter([fake_cuda_tensor(t) for t in sc_srcs], fake_cuda_tensor(sc_srcs[0]), root=0, manager=manager)
    assert syncs() == (0, 0, 0)

    # Async reduce_scatter (world_size=1, non-alias): _local_copy on upload; only upload synced.
    reset_counts()
    shared.world_size = 1
    shared._peers = ()
    rs_src = torch.arange(8, dtype=torch.float32) + 3
    rs_out = torch.zeros(8, dtype=torch.float32)
    shared.reduce_scatter([fake_cuda_tensor(rs_out)], [[fake_cuda_tensor(rs_src)]], manager)
    assert syncs() == (0, 1, 0)
    assert copy_streams[-1] == (True, ["upload"])
    assert torch.equal(rs_out, rs_src)


def test_shared_collective_synchronizes_only_dirty_or_failed_streams(tmp_path):
    (
        shared, control, manager, download_stream, upload_stream,
        current, stream_entries, track_stream, dummy_event, fake_cuda_tensor,
    ) = _make_fake_collective_relay(tmp_path)

    def syncs() -> tuple[int, int, int]:
        return (download_stream.sync_count, upload_stream.sync_count, current.sync_count)

    def reset_counts() -> None:
        download_stream.sync_count = upload_stream.sync_count = current.sync_count = 0
        stream_entries.clear()

    stream_iter = iter((download_stream, upload_stream))
    copy_streams: list[tuple[bool, list[str]]] = []
    orig_local_copy = shared._local_copy

    def tracked_local_copy(destination: torch.Tensor, source: torch.Tensor, *, asynchronous: bool) -> None:
        start_idx = len(stream_entries)
        orig_local_copy(destination, source, asynchronous=asynchronous)
        copy_streams.append((asynchronous, stream_entries[start_idx:]))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "Stream", lambda device: next(stream_iter))
        patch.setattr(torch.cuda, "Event", dummy_event)
        patch.setattr(torch.cuda, "current_stream", lambda _device: current)
        patch.setattr(torch.cuda, "stream", track_stream)
        patch.setattr(shared, "_local_copy", tracked_local_copy)

        _check_async_stream_syncs(shared, manager, fake_cuda_tensor, syncs, reset_counts, copy_streams)

        # Pageable fallback (asynchronous=False): _local_copy on current; only current synced.
        reset_counts()
        manager.pageable_bytes = 64
        pg_src = torch.arange(8, dtype=torch.float32) + 5
        pg_out = torch.zeros(8, dtype=torch.float32)
        shared.reduce_scatter([fake_cuda_tensor(pg_out)], [[fake_cuda_tensor(pg_src)]], manager)
        assert syncs() == (0, 0, 1)
        assert copy_streams[-1] == (False, [])
        assert torch.equal(pg_out, pg_src)

        # Exception mid-collective: download, upload, AND current are all synchronized.
        reset_counts()
        manager.pageable_bytes = 0
        shared.world_size = 2
        shared._peers = ((1, 1),)
        control.fail_on_signal = True
        with pytest.raises(RuntimeError, match="injected signal failure"):
            shared.broadcast(fake_cuda_tensor(torch.arange(8, dtype=torch.float32)), root=0, manager=manager)
        assert syncs() == (1, 1, 1)
        assert shared.broken


def _raise_after(error: BaseException, before: Callable[[], None] | None = None) -> Callable[[], None]:
    def raise_error() -> None:
        if before is not None:
            before()
        raise error

    return raise_error


def _patched_fake_collective_relay(tmp_path, monkeypatch):
    (
        shared, control, manager, download_stream, upload_stream,
        current, _stream_entries, track_stream, dummy_event, fake_cuda_tensor,
    ) = _make_fake_collective_relay(tmp_path)
    orig = (manager.acquire, download_stream.synchronize, upload_stream.synchronize, current.synchronize)
    stream_iter = iter((download_stream, upload_stream))
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: next(stream_iter))
    monkeypatch.setattr(torch.cuda, "Event", dummy_event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: current)
    monkeypatch.setattr(torch.cuda, "stream", track_stream)

    def reset() -> None:
        shared.broken = False
        control.abort_count = 0
        control.fail_on_signal = False
        shared.world_size = 1
        shared._peers = ()
        manager.pageable_bytes = 0
        monkeypatch.setattr(manager, "acquire", orig[0])
        monkeypatch.setattr(download_stream, "synchronize", orig[1])
        monkeypatch.setattr(upload_stream, "synchronize", orig[2])
        monkeypatch.setattr(current, "synchronize", orig[3])
        download_stream.sync_count = upload_stream.sync_count = current.sync_count = 0

    return shared, control, manager, download_stream, upload_stream, current, fake_cuda_tensor, reset, orig


def test_shared_collective_marks_broken_on_acquire_and_pre_acquire_failures(tmp_path, monkeypatch):
    from contextlib import contextmanager

    shared, control, manager, _dl, _ul, _cur, fake_cuda_tensor, reset, orig = (
        _patched_fake_collective_relay(tmp_path, monkeypatch)
    )
    sample = fake_cuda_tensor(torch.arange(8, dtype=torch.float32))

    @contextmanager
    def failing_enter(_tensors: list[torch.Tensor]):
        raise Interrupt
        yield  # pragma: no cover

    @contextmanager
    def failing_exit(tensors: list[torch.Tensor]):
        with orig[0](tensors) as lease:
            yield lease
            raise RuntimeError("injected release failure")

    # 1. Interrupt during manager.acquire.__enter__ after agreement.
    monkeypatch.setattr(manager, "acquire", failing_enter)
    with pytest.raises(Interrupt):
        with shared._collective("broadcast", [sample], 0, manager):
            pytest.fail("collective body ran after acquire failure")
    assert shared.broken
    assert control.abort_count == 1

    # 2. Failure during manager.acquire.__exit__ after body completes.
    reset()
    monkeypatch.setattr(manager, "acquire", failing_exit)
    with pytest.raises(RuntimeError, match="injected release failure"):
        shared.broadcast(sample, root=0, manager=manager)
    assert shared.broken
    assert control.abort_count == 1

    # 3. pre_acquire ValueError stays usable.
    reset()
    with pytest.raises(ValueError, match="mismatched splits"):
        with shared._collective(
            "alltoall", [sample], -1, manager, pre_acquire=_raise_after(ValueError("mismatched splits")),
        ):
            pytest.fail("collective body ran after pre_acquire ValueError")
    assert not shared.broken
    assert control.abort_count == 0

    # 4. pre_acquire Interrupt marks broken and aborts once.
    with pytest.raises(Interrupt):
        with shared._collective("alltoall", [sample], -1, manager, pre_acquire=_raise_after(Interrupt())):
            pytest.fail("collective body ran after pre_acquire Interrupt")
    assert shared.broken
    assert control.abort_count == 1


def test_shared_collective_marks_broken_on_sync_and_body_failures(tmp_path, monkeypatch):
    shared, control, manager, download_stream, upload_stream, current, fake_cuda_tensor, reset, orig = (
        _patched_fake_collective_relay(tmp_path, monkeypatch)
    )
    sample = fake_cuda_tensor(torch.arange(8, dtype=torch.float32))
    rs_src = fake_cuda_tensor(torch.arange(8, dtype=torch.float32))
    rs_out = fake_cuda_tensor(torch.zeros(8, dtype=torch.float32))

    # 1. Interrupt during finally upload.synchronize() after clean async body.
    monkeypatch.setattr(upload_stream, "synchronize", _raise_after(Interrupt(), before=orig[2]))
    with pytest.raises(Interrupt):
        shared.reduce_scatter([rs_out], [[rs_src]], manager)
    assert upload_stream.sync_count == 1
    assert shared.broken
    assert control.abort_count == 1

    # 2. download.synchronize() fails when both dirty; upload still syncs and relay aborts once.
    reset()
    monkeypatch.setattr(
        download_stream, "synchronize",
        _raise_after(RuntimeError("injected download sync failure"), before=orig[1]),
    )
    with pytest.raises(RuntimeError, match="injected download sync failure"):
        with shared._collective("allreduce", [sample], -1, manager) as asynchronous:
            assert asynchronous
            shared._download_dirty = True
            shared._upload_dirty = True
    assert (download_stream.sync_count, upload_stream.sync_count) == (1, 1)
    assert shared.broken
    assert control.abort_count == 1

    # 3. current_stream.synchronize() interrupt on pageable fallback.
    reset()
    manager.pageable_bytes = 64
    monkeypatch.setattr(current, "synchronize", _raise_after(Interrupt(), before=orig[3]))
    with pytest.raises(Interrupt):
        shared.reduce_scatter([rs_out], [[rs_src]], manager)
    assert current.sync_count == 1
    assert shared.broken
    assert control.abort_count == 1

    # 4. Body failure calls abort() once *before* draining DMA streams.
    reset()
    sync_states: list[tuple[str, bool, int]] = []
    monkeypatch.setattr(
        download_stream, "synchronize",
        lambda: sync_states.append(("download", shared.broken, control.abort_count)),
    )
    monkeypatch.setattr(
        upload_stream, "synchronize",
        lambda: sync_states.append(("upload", shared.broken, control.abort_count)),
    )
    shared.world_size = 2
    shared._peers = ((1, 1),)
    control.fail_on_signal = True
    with pytest.raises(RuntimeError, match="injected signal failure"):
        shared.broadcast(sample, root=0, manager=manager)
    assert sync_states == [("download", True, 1), ("upload", True, 1)]
    assert control.abort_count == 1


def test_shared_pack_spans_and_chunk_pieces():
    from piper_offload._relay_shared import _batch_signature, _chunk_pieces, _pack_spans

    assert _pack_spans([0, 0]) == ([], 0)
    assert _pack_spans([6, 0, 32], align=1) == ([(0, 0, 6), (2, 6, 32)], 38)

    spans, total = _pack_spans([6, 0, 32], align=16)
    assert (spans, total) == ([(0, 0, 6), (2, 16, 32)], 48)
    ends = [start + length for _, start, length in spans]
    assert _chunk_pieces(spans, ends, 0, 16) == [(0, 0, 0, 6)]
    assert _chunk_pieces(spans, ends, 16, 48) == [(2, 0, 0, 32)]
    assert _chunk_pieces(spans, ends, 0, 32) == [(0, 0, 0, 6), (2, 16, 0, 16)]
    assert _chunk_pieces(spans, ends, 32, 48) == [(2, 0, 16, 16)]

    a = [torch.empty(2, dtype=torch.int32), torch.empty(2, dtype=torch.int32)]
    b = [torch.empty(1, dtype=torch.int32), torch.empty(3, dtype=torch.int32)]
    c = [torch.empty(2, dtype=torch.int32), torch.empty(2, dtype=torch.float32)]
    assert _batch_signature(a) != _batch_signature(b)
    assert _batch_signature(a) != _batch_signature(c)


def _assert_fused_peer_lanes(buf: torch.Tensor, inputs: Sequence[torch.Tensor], rank: int) -> None:
    buf_start = buf.data_ptr()
    buf_end = buf_start + buf.nbytes
    ranges = sorted(
        (tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes)
        for peer, tensor in enumerate(inputs)
        if peer != rank
    )
    assert all(buf_start <= start and end <= buf_end for start, end in ranges)
    assert all(prev_end <= next_start for (_, prev_end), (next_start, _) in pairwise(ranges))


def _check_chunk_reduction_mode(world_size: int, rank: int, fused: bool) -> None:
    from piper_offload._relay_shared import SharedRelay, _bytes, _chunk_pieces, _pack_spans

    relay = object.__new__(SharedRelay)
    relay.rank = rank
    relay.world_size = world_size
    relay.capacity = 128

    rows = {
        torch.float16: (65504.0, 65504.0, -65504.0, 1.0),
        torch.bfloat16: (256.0, 1.0, -256.0, 2.0),
        torch.float32: (2.0**24, 1.0, -(2.0**24), 3.0),
    }
    backings = [
        [torch.full((37,), values[r], dtype=dtype) for dtype, values in rows.items()]
        for r in range(world_size)
    ]
    per_rank = [[b[1:-1] for b in rank_backings] for rank_backings in backings]
    local_backings = [b.clone() for b in backings[rank]]
    local = [b[1:-1] for b in local_backings]
    assert all(tensor.storage_offset() == 1 for tensor in local)
    byte_sources = [_bytes(t) for t in local]
    spans, total_bytes = _pack_spans([t.numel() for t in byte_sources], align=16)
    span_ends = [start + span_len for _, start, span_len in spans]

    buf = torch.empty(3 * relay.capacity, dtype=torch.uint8)
    calls = 0

    def cpu_sum_ranks(output: torch.Tensor, inputs: Sequence[torch.Tensor]) -> None:
        nonlocal calls
        calls += 1
        assert len(inputs) == world_size
        assert inputs[rank].data_ptr() == output.data_ptr()
        _assert_fused_peer_lanes(buf, inputs, rank)
        total = functools.reduce(torch.add, (tensor.float() for tensor in inputs))
        output.copy_(total)

    total_pieces = 0
    for start in range(0, total_bytes, relay.capacity):
        width = min(relay.capacity, total_bytes - start)
        pieces = _chunk_pieces(spans, span_ends, start, start + width)
        total_pieces += len(pieces)
        peer_slots: list[torch.Tensor | None] = [None] * world_size
        for peer in range(world_size):
            if peer != rank:
                slot = torch.empty(width, dtype=torch.uint8)
                for ti, slot_off, t_off, part_len in pieces:
                    slot[slot_off:slot_off + part_len].copy_(_bytes(per_rank[peer][ti])[t_off:t_off + part_len])
                peer_slots[peer] = slot
        if fused:
            relay._reduce_chunk_fused(local, byte_sources, buf, width, peer_slots, pieces, False, cpu_sum_ranks)
        else:
            relay._reduce_chunk_reference(local, byte_sources, buf, width, peer_slots, pieces, False)

    assert total_pieces > len(local)
    assert not fused or calls == total_pieces
    for i, actual in enumerate(local):
        expected = functools.reduce(torch.add, (per_rank[r][i].float() for r in range(world_size))).to(actual.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for sentinel_backing, values in zip(local_backings, rows.values(), strict=True):
        torch.testing.assert_close(
            sentinel_backing[[0, -1]], torch.full((2,), values[rank], dtype=sentinel_backing.dtype),
        )


def test_shared_allreduce_fused_and_reference_chunk_reduction(tmp_path, monkeypatch):
    from piper_offload import _relay_shared

    for world_size in (2, 3, 4):
        for rank in (0, world_size - 1):
            _check_chunk_reduction_mode(world_size, rank, fused=True)
            _check_chunk_reduction_mode(world_size, rank, fused=False)

    assert _relay_shared._select_fused_sum(torch.device("cpu"), 2) is None
    assert _relay_shared._select_fused_sum(torch.device("cuda", 0), 1) is None
    assert _relay_shared._select_fused_sum(torch.device("cuda", 0), 5) is None

    def fake_sum_ranks(output: torch.Tensor, inputs: Sequence[torch.Tensor]) -> None:
        del output, inputs

    with monkeypatch.context() as patch:
        patch.setattr(_relay_shared, "_load_triton_sum", lambda: fake_sum_ranks)
        for world_size in (2, 3, 4):
            assert _relay_shared._select_fused_sum(torch.device("cuda", 0), world_size) is fake_sum_ranks
        patch.setattr(_relay_shared, "_load_triton_sum", lambda: None)
        for world_size in (2, 3, 4):
            assert _relay_shared._select_fused_sum(torch.device("cuda", 0), world_size) is None

    class SingleRankControl:
        def agree(self, op, nbytes, kind, detail, timeout):
            del op, nbytes, kind, detail, timeout

        def signal(self, peers, tag, sequence, timeout):
            del peers, tag, sequence, timeout

        def skip_ready(self, peer, count):
            del peer, count

        def abort(self):
            return

    shared = _relay_shared.SharedRelay(
        dist.FileStore(str(tmp_path / "dispatch"), 1),
        None,
        rank=0,
        world_size=1,
        nbytes=1536,
        timeout=timedelta(seconds=5),
    )
    shared.world_size = 2
    shared._control = SingleRankControl()
    shared._peers = ((1, 1),)
    monkeypatch.setattr(shared, "_slot", lambda peer, lane, width: torch.ones(width, dtype=torch.uint8))
    ref_calls = 0

    def spy_reference(*args, **kwargs):
        del args, kwargs
        nonlocal ref_calls
        ref_calls += 1

    monkeypatch.setattr(shared, "_reduce_chunk_reference", spy_reference)
    shared.allreduce([torch.ones(8, dtype=torch.float16)], PinManager(0))
    assert ref_calls == 1

    _relay_shared._load_triton_sum.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.delitem(sys.modules, "piper_offload._triton_relay_sum", raising=False)
            patch.setitem(sys.modules, "triton", None)
            assert _relay_shared._load_triton_sum() is None
        _relay_shared._load_triton_sum.cache_clear()
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "piper_offload._triton_relay_sum", None)
            with pytest.raises(ModuleNotFoundError):
                _relay_shared._load_triton_sum()
    finally:
        _relay_shared._load_triton_sum.cache_clear()


def test_triton_relay_sum_validation():
    pytest.importorskip("triton")
    from piper_offload._triton_relay_sum import MAX_FUSED_RANKS, sum_ranks

    assert MAX_FUSED_RANKS == 4
    sample = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(ValueError, match="2..4"):
        sum_ranks(sample, [sample])
    with pytest.raises(ValueError, match="2..4"):
        sum_ranks(sample, [sample] * 5)
    for dtype in DTYPES:
        empty = torch.empty(0, dtype=dtype)
        sum_ranks(empty, [empty, empty])


@CUDA
def test_triton_relay_sum_matches_rank_order_fp32():
    pytest.importorskip("triton")
    from piper_offload._triton_relay_sum import sum_ranks

    rows = {
        torch.float16: (65504.0, 65504.0, -65504.0, 1.0),
        torch.bfloat16: (256.0, 1.0, -256.0, 2.0),
        torch.float32: (2.0**24, 1.0, -(2.0**24), 3.0),
    }
    for world_size in (2, 3, 4):
        for dtype, values in rows.items():
            # Odd length > 2048 crosses multiple Triton tiles and keeps a non-zero offset.
            length = 2053
            backings = [torch.full((length + 2,), values[r], dtype=dtype, device="cuda") for r in range(world_size)]
            stride_count = backings[0][1:-1][::7].numel()
            varied = torch.arange(stride_count, device="cuda", dtype=torch.float32) % 19 - 9
            for r, backing in enumerate(backings):
                backing[1:-1][::7] = (varied + (r + 1) * 0.5).to(dtype)
            inputs_ref = [b[1:-1].clone() for b in backings]
            expected = functools.reduce(torch.add, (t.float() for t in inputs_ref)).to(dtype)
            for rank in range(world_size):
                rank_inputs = [t.clone() for t in inputs_ref]
                backing = backings[rank].clone()
                rank_inputs[rank] = backing[1:-1]
                sum_ranks(rank_inputs[rank], rank_inputs)
                torch.testing.assert_close(rank_inputs[rank], expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    backing[[0, -1]], torch.full((2,), values[rank], dtype=dtype, device="cuda"),
                )


def test_shared_reduce_scatter_partition_and_selection():
    from piper_offload._relay_shared import (
        _REDUCE_SCATTER_MIN_BYTES,
        _partition_bytes,
        _reduce_scatter_sub_cap,
        _use_reduce_scatter,
    )

    for total_bytes, world_size in ((300, 3), (300, 4), (32, 4), (2, 3)):
        parts = _partition_bytes(total_bytes, world_size)
        assert len(parts) == world_size
        assert parts[0][0] == 0 and parts[-1][1] == total_bytes
        for (start, end), (next_start, _) in pairwise(parts):
            assert start <= end == next_start
            assert next_start % 16 == 0
    assert _partition_bytes(32, 4) == [(0, 0), (0, 16), (16, 16), (16, 32)]
    assert _partition_bytes(2, 3) == [(0, 0), (0, 0), (0, 2)]

    assert _reduce_scatter_sub_cap(48, 8) == 0
    assert _reduce_scatter_sub_cap(128, 3) == 64
    sub_4 = _reduce_scatter_sub_cap(696320, 4)
    assert sub_4 == 229376 and 3 * sub_4 <= 696320

    assert not _use_reduce_scatter(2, 1024 * 1024, 65536)
    assert not _use_reduce_scatter(3, _REDUCE_SCATTER_MIN_BYTES - 16, 65536)
    assert _use_reduce_scatter(3, _REDUCE_SCATTER_MIN_BYTES, 65536)
    assert not _use_reduce_scatter(8, 1024 * 1024, 48)


def _check_rs_small_fallback(shared, manager: PinManager) -> None:
    calls = [0, 0]
    orig_direct = shared._allreduce_direct
    orig_rs = shared._allreduce_reduce_scatter

    def tracked_direct(*args):
        calls[0] += 1
        orig_direct(*args)

    def tracked_rs(*args):
        calls[1] += 1
        orig_rs(*args)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(shared, "_allreduce_direct", tracked_direct)
        patch.setattr(shared, "_allreduce_reduce_scatter", tracked_rs)
        sample = torch.ones(64, dtype=torch.float32)
        shared.allreduce([sample], manager)
    assert calls == [1, 0]
    torch.testing.assert_close(sample, torch.full_like(sample, 4.0), rtol=0, atol=0)


def _check_rs_precision_and_sentinels(shared, rank: int, manager: PinManager) -> None:
    from piper_offload._relay_shared import _REDUCE_SCATTER_MIN_BYTES

    rows = {
        torch.float16: (65504.0, 65504.0, -65504.0, 1.0),
        torch.bfloat16: (256.0, 1.0, -256.0, 2.0),
        torch.float32: (2.0**24, 1.0, -(2.0**24), 3.0),
    }
    for dtype, contributions in rows.items():
        count = _REDUCE_SCATTER_MIN_BYTES // dtype.itemsize + 13
        backing = torch.full((count + 2,), contributions[rank], dtype=dtype)
        local = backing[1:-1]
        assert local.storage_offset() == 1
        shared.allreduce([local], manager)
        expected = functools.reduce(torch.add, (torch.tensor(v, dtype=dtype).float() for v in contributions)).to(dtype)
        torch.testing.assert_close(local, torch.full_like(local, expected.item()), rtol=0, atol=0)
        torch.testing.assert_close(backing[[0, -1]], torch.full((2,), contributions[rank], dtype=dtype))


def _check_rs_volume_accounting(shared, rank: int, manager: PinManager) -> None:
    from piper_offload._relay_shared import _partition_bytes

    total_bytes = 256 * 1024
    tensor = torch.full((total_bytes // 2,), float(rank + 1), dtype=torch.float16)
    my_start, my_end = _partition_bytes(total_bytes, shared.world_size)[rank]
    my_part = my_end - my_start
    reduced_bytes = 0
    phase1_peer_bytes = 0
    allgather_peer_bytes = 0
    orig_reduce_ref = shared._reduce_chunk_reference
    orig_allgather = shared._allgather_partitions
    orig_slot = shared._slot
    in_allgather = False

    def tracked_reduce_ref(tensors, byte_outputs, buffer, width, peer_slots, pieces, asynchronous, local_sources=None):
        nonlocal reduced_bytes, phase1_peer_bytes
        reduced_bytes += width
        phase1_peer_bytes += sum(slot.numel() for slot in peer_slots if slot is not None)
        orig_reduce_ref(tensors, byte_outputs, buffer, width, peer_slots, pieces, asynchronous, local_sources)

    def tracked_allgather(*args):
        nonlocal in_allgather
        in_allgather = True
        try:
            orig_allgather(*args)
        finally:
            in_allgather = False

    def tracked_slot(sender: int, lane: int, width: int) -> torch.Tensor:
        nonlocal allgather_peer_bytes
        if in_allgather and sender != rank:
            allgather_peer_bytes += width
        return orig_slot(sender, lane, width)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(shared, "_reduce_chunk_reference", tracked_reduce_ref)
        patch.setattr(shared, "_allgather_partitions", tracked_allgather)
        patch.setattr(shared, "_slot", tracked_slot)
        shared.allreduce([tensor], manager)

    assert reduced_bytes == my_part
    assert phase1_peer_bytes == (shared.world_size - 1) * my_part
    assert allgather_peer_bytes == total_bytes - my_part
    assert phase1_peer_bytes + allgather_peer_bytes == (3 * total_bytes) // 2
    torch.testing.assert_close(tensor, torch.full_like(tensor, 10.0), rtol=0, atol=0)


def _check_rs_coalesced_and_fused(shared, rank: int, manager: PinManager) -> None:
    from piper_offload import _relay_shared

    def make_batch(r: int) -> list[torch.Tensor]:
        bf16_rows = (
            torch.tensor([256.0, 1.0, -256.0], dtype=torch.bfloat16),
            torch.tensor([1.0, 2.0, 1.0], dtype=torch.bfloat16),
            torch.tensor([-256.0, 4.0, 256.0], dtype=torch.bfloat16),
            torch.tensor([2.0, 8.0, 1.0], dtype=torch.bfloat16),
        )
        return [
            torch.empty((0,), dtype=torch.float32),
            bf16_rows[r].clone(),
            ((torch.arange(70_003, dtype=torch.float32) % 13) - 4 + r * 0.5).to(torch.float16),
            (torch.arange(40_005, dtype=torch.float32) % 29) * 0.25 + r,
        ]

    per_rank = [make_batch(r) for r in range(shared.world_size)]
    expected = [
        functools.reduce(torch.add, (per_rank[r][i].float() for r in range(shared.world_size))).to(per_rank[0][i].dtype)
        for i in range(len(per_rank[0]))
    ]
    ref_inputs = [t.clone() for t in per_rank[rank]]
    shared.allreduce(ref_inputs, manager)
    for actual, exp in zip(ref_inputs, expected, strict=True):
        torch.testing.assert_close(actual, exp, rtol=0, atol=0)

    fused_inputs = [t.clone() for t in per_rank[rank]]
    fused_calls = 0

    def cpu_sum_ranks(output: torch.Tensor, inputs: Sequence[torch.Tensor]) -> None:
        nonlocal fused_calls
        fused_calls += 1
        assert len(inputs) == shared.world_size
        assert inputs[rank].data_ptr() == output.data_ptr()
        assert shared._reduction_buffer is not None
        _assert_fused_peer_lanes(shared._reduction_buffer, inputs, rank)
        output.copy_(functools.reduce(torch.add, (tensor.float() for tensor in inputs)))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_relay_shared, "_select_fused_sum", lambda _device, _world_size: cpu_sum_ranks)
        shared.allreduce(fused_inputs, manager)

    assert fused_calls > 0
    for actual, exp in zip(fused_inputs, expected, strict=True):
        torch.testing.assert_close(actual, exp, rtol=0, atol=0)


def _check_rs_empty_partitions_and_gloo_subgroup(shared, rank: int, manager: PinManager) -> None:
    from piper_offload import _relay_shared

    rs_calls = 0
    orig_rs = shared._allreduce_reduce_scatter

    def tracked_rs(*args):
        nonlocal rs_calls
        rs_calls += 1
        orig_rs(*args)

    small_vec = torch.arange(8, dtype=torch.float32) + (rank + 1)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_relay_shared, "_REDUCE_SCATTER_MIN_BYTES", 1)
        patch.setattr(shared, "_allreduce_reduce_scatter", tracked_rs)
        shared.allreduce([small_vec], manager)
    assert rs_calls == 1
    torch.testing.assert_close(small_vec, 4.0 * torch.arange(8, dtype=torch.float32) + 10.0, rtol=0, atol=0)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_relay_shared, "shared_control_supported", lambda: False)
        subgroup = dist.new_group(
            ranks=[0, 1, 2], pg_options=RelayOptions(transport="shared", staging_bytes=144 * 1024),
        )
    if rank in (0, 1, 2):
        try:
            sub_shared = subgroup._shared
            assert isinstance(sub_shared._control, _relay_shared._GlooControl)
            vals = (256.0, 1.0, -256.0)
            bf16 = torch.full((262144 // 2 + 13,), vals[rank], dtype=torch.bfloat16)
            varied = torch.arange(bf16[::997].numel(), dtype=torch.float32) % 19 - 9
            bf16[::997] = (varied + rank * 0.5).to(torch.bfloat16)
            expected = torch.full_like(bf16, 1.0)
            expected[::997] = functools.reduce(
                torch.add, ((varied + r * 0.5).to(torch.bfloat16).float() for r in range(3)),
            ).to(torch.bfloat16)
            sub_shared.allreduce([bf16], manager)
            torch.testing.assert_close(bf16, expected, rtol=0, atol=0)
        finally:
            dist.destroy_process_group(subgroup)


def _run_shared_allreduce_reduce_scatter(rank: int, path: str) -> None:
    torch.set_num_threads(1)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(path, 4), rank=rank, world_size=4,
        timeout=timedelta(seconds=20),
        pg_options=RelayOptions(transport="shared", staging_bytes=192 * 1024),
    )
    manager = PinManager(0)
    try:
        shared = dist.group.WORLD._shared
        _check_rs_small_fallback(shared, manager)
        _check_rs_precision_and_sentinels(shared, rank, manager)
        _check_rs_volume_accounting(shared, rank, manager)
        _check_rs_coalesced_and_fused(shared, rank, manager)
        _check_rs_empty_partitions_and_gloo_subgroup(shared, rank, manager)
    finally:
        dist.destroy_process_group()


def test_shared_allreduce_reduce_scatter(tmp_path):
    mp.spawn(_run_shared_allreduce_reduce_scatter, args=(str(tmp_path / "rs"),), nprocs=4)


def _check_shared_rs_precision_and_alias(shared, rank: int, manager: PinManager) -> None:
    rows = {
        torch.float16: (65504.0, 65504.0, -65504.0, 1.0),
        torch.bfloat16: (256.0, 1.0, -256.0, 2.0),
        torch.float32: (2.0**24, 1.0, -(2.0**24), 3.0),
    }
    for dtype, contributions in rows.items():
        count = 4096 // dtype.itemsize + 13
        src_backings = [torch.full((count + 2,), contributions[rank], dtype=dtype) for _ in range(shared.world_size)]
        varied = torch.arange(torch.empty(count)[::137].numel(), dtype=torch.float32) % 19 - 9
        for owner, sb in enumerate(src_backings):
            sb[1:-1][::137] = (varied + (rank + owner) * 0.5).to(dtype)
        sources = [sb[1:-1] for sb in src_backings]
        src_backups = [s.clone() for s in sources]

        out_backing = torch.full((count + 2,), contributions[rank], dtype=dtype)
        output = out_backing[1:-1]
        assert output.storage_offset() == 1
        shared.reduce_scatter([output], [sources], manager)

        const_total = functools.reduce(
            torch.add, (torch.tensor(v, dtype=dtype).float() for v in contributions),
        ).to(dtype)
        expected_full = torch.full((count,), const_total.item(), dtype=dtype)
        expected_full[::137] = functools.reduce(
            torch.add, ((varied + (r + rank) * 0.5).to(dtype).float() for r in range(shared.world_size)),
        ).to(dtype)
        torch.testing.assert_close(output, expected_full, rtol=0, atol=0)
        torch.testing.assert_close(out_backing[[0, -1]], torch.full((2,), contributions[rank], dtype=dtype))
        for src, backup in zip(sources, src_backups, strict=True):
            torch.testing.assert_close(src, backup, rtol=0, atol=0)

    # Exact own-rank input slice alias preserves all non-own slices.
    alias_sources = [t.clone() for t in sources]
    shared.reduce_scatter([alias_sources[rank]], [alias_sources], manager)
    torch.testing.assert_close(alias_sources[rank], expected_full, rtol=0, atol=0)
    for owner, backup in enumerate(src_backups):
        if owner != rank:
            torch.testing.assert_close(alias_sources[owner], backup, rtol=0, atol=0)


def _check_shared_rs_volume(shared, rank: int, manager: PinManager) -> None:
    from piper_offload import _relay_shared

    count = 32 * 1024  # 64 KiB per owner
    sources = [torch.full((count,), float(rank + owner + 1), dtype=torch.float16) for owner in range(shared.world_size)]
    output = torch.empty_like(sources[0])
    reduced_bytes = 0
    peer_read_bytes = 0
    copied_bytes = 0
    orig_reduce_ref = shared._reduce_chunk_reference
    orig_copy = _relay_shared._copy_packed_to_slot

    def tracked_reduce_ref(tensors, byte_sources, buffer, width, peer_slots, pieces, asynchronous, local_sources=None):
        nonlocal reduced_bytes, peer_read_bytes
        reduced_bytes += width
        peer_read_bytes += sum(slot.numel() for slot in peer_slots if slot is not None)
        orig_reduce_ref(tensors, byte_sources, buffer, width, peer_slots, pieces, asynchronous, local_sources)

    def tracked_copy(slot, srcs, spans, span_ends, start, end, asynchronous):
        nonlocal copied_bytes
        copied_bytes += sum(part[3] for part in _relay_shared._chunk_pieces(spans, span_ends, start, end))
        orig_copy(slot, srcs, spans, span_ends, start, end, asynchronous)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(shared, "_reduce_chunk_reference", tracked_reduce_ref)
        patch.setattr(_relay_shared, "_copy_packed_to_slot", tracked_copy)
        shared.reduce_scatter([output], [sources], manager)

    assert reduced_bytes == output.nbytes
    assert peer_read_bytes == (shared.world_size - 1) * output.nbytes
    assert copied_bytes == (shared.world_size - 1) * output.nbytes
    expected_val = float(sum(r + rank + 1 for r in range(shared.world_size)))
    torch.testing.assert_close(output, torch.full_like(output, expected_val), rtol=0, atol=0)


def _check_shared_rs_coalesced_and_fused(shared, rank: int, manager: PinManager) -> None:
    from piper_offload import _relay_shared

    def make_batch(src_rank: int, owner: int) -> list[torch.Tensor]:
        bf16_vec = (
            torch.tensor([256.0, 1.0, -256.0], dtype=torch.bfloat16)
            if src_rank == 0
            else torch.full((3,), float(owner + src_rank), dtype=torch.bfloat16)
        )
        return [
            torch.empty((0,), dtype=torch.float32),
            bf16_vec,
            ((torch.arange(2503, dtype=torch.float32) % 13) - 4 + src_rank + owner * 0.5).to(torch.float16),
            (torch.arange(1505, dtype=torch.float32) % 29) * 0.25 + src_rank + owner,
        ]

    per_src = [[make_batch(r, owner) for owner in range(shared.world_size)] for r in range(shared.world_size)]
    sources_list = [[per_src[rank][owner][ti] for owner in range(shared.world_size)] for ti in range(4)]
    expected = [
        functools.reduce(torch.add, (per_src[r][rank][ti].float() for r in range(shared.world_size))).to(
            per_src[0][0][ti].dtype,
        )
        for ti in range(4)
    ]
    ref_outputs = [torch.empty_like(expected[ti]) for ti in range(4)]
    shared.reduce_scatter(ref_outputs, sources_list, manager)
    for actual, exp in zip(ref_outputs, expected, strict=True):
        torch.testing.assert_close(actual, exp, rtol=0, atol=0)

    fused_outputs = [torch.empty_like(expected[ti]) for ti in range(4)]
    fused_calls = 0
    local_ranges = [
        (src_list[rank].data_ptr(), src_list[rank].data_ptr() + src_list[rank].nbytes)
        for src_list in sources_list
    ]

    def cpu_sum_ranks(output: torch.Tensor, inputs: Sequence[torch.Tensor]) -> None:
        nonlocal fused_calls
        fused_calls += 1
        assert len(inputs) == shared.world_size
        local_ptr = inputs[rank].data_ptr()
        assert local_ptr != output.data_ptr()
        assert any(lo <= local_ptr < hi for lo, hi in local_ranges if lo < hi)
        assert shared._reduction_buffer is not None
        _assert_fused_peer_lanes(shared._reduction_buffer, inputs, rank)
        output.copy_(functools.reduce(torch.add, (tensor.float() for tensor in inputs)))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_relay_shared, "_select_fused_sum", lambda _device, _world_size: cpu_sum_ranks)
        shared.reduce_scatter(fused_outputs, sources_list, manager)

    assert fused_calls > 0
    for actual, exp in zip(fused_outputs, expected, strict=True):
        torch.testing.assert_close(actual, exp, rtol=0, atol=0)


def _check_serial_subgroup(sub_shared, rank: int, force_gloo: bool, manager: PinManager) -> None:
    from piper_offload import _relay_shared

    assert isinstance(sub_shared._control, _relay_shared._GlooControl) == (
        force_gloo or not _relay_shared.shared_control_supported()
    )
    serial_calls = 0
    orig_serial = sub_shared._reduce_scatter_serial

    def tracked_serial(*args):
        nonlocal serial_calls
        serial_calls += 1
        orig_serial(*args)

    cases = (
        (torch.bfloat16, (256.0, 1.0, -256.0, 2.0)),
        (torch.float32, (2.0**24, 1.0, -(2.0**24), 3.0)),
    )
    for dtype, vals in cases:
        count = 257
        varied = torch.arange(torch.empty(count)[::19].numel(), dtype=torch.float32) % 17 - 8
        sources = []
        for owner in range(4):
            t = torch.full((count,), vals[rank], dtype=dtype)
            t[::19] = (varied + (rank + owner) * 0.5).to(dtype)
            sources.append(t)
        out = torch.empty_like(sources[0])
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(sub_shared, "_reduce_scatter_serial", tracked_serial)
            sub_shared.reduce_scatter([out], [sources], manager)
        const_val = functools.reduce(torch.add, (torch.tensor(v, dtype=dtype).float() for v in vals)).to(dtype)
        exp = torch.full((count,), const_val.item(), dtype=dtype)
        exp[::19] = functools.reduce(
            torch.add, ((varied + (r + rank) * 0.5).to(dtype).float() for r in range(4)),
        ).to(dtype)
        torch.testing.assert_close(out, exp, rtol=0, atol=0)
    assert serial_calls == 2


def _check_shared_rs_mismatch_and_serial(rank: int, shared, manager: PinManager) -> None:
    from piper_offload import _relay_shared

    bad_out = torch.empty(4, dtype=torch.float32 if rank == 0 else torch.float16)
    bad_srcs = [torch.ones(4, dtype=bad_out.dtype) for _ in range(shared.world_size)]
    with pytest.raises(ValueError, match="must match across ranks"):
        shared.reduce_scatter([bad_out], [bad_srcs], manager)
    assert not shared.broken
    ok_out = torch.empty(4, dtype=torch.float16)
    ok_srcs = [torch.full((4,), float(rank + owner + 1), dtype=torch.float16) for owner in range(shared.world_size)]
    shared.reduce_scatter([ok_out], [ok_srcs], manager)
    ok_val = float(sum(r + rank + 1 for r in range(shared.world_size)))
    torch.testing.assert_close(ok_out, torch.full_like(ok_out, ok_val), rtol=0, atol=0)

    for force_gloo in (False, True):
        with pytest.MonkeyPatch.context() as patch:
            if force_gloo:
                patch.setattr(_relay_shared, "shared_control_supported", lambda: False)
            subgroup = dist.new_group(
                ranks=[0, 1, 2, 3], pg_options=RelayOptions(transport="shared", staging_bytes=384),
            )
        try:
            _check_serial_subgroup(subgroup._shared, rank, force_gloo, manager)
        finally:
            dist.destroy_process_group(subgroup)


def _check_shared_rs_two_rank_subgroup(rank: int, manager: PinManager) -> None:
    from piper_offload import _relay_shared

    subgroup = dist.new_group(ranks=[0, 1], pg_options=RelayOptions(transport="shared", staging_bytes=4096))
    if rank in (0, 1):
        try:
            sub_shared = subgroup._shared
            packed_calls = 0
            copied_bytes = 0
            orig_packed = sub_shared._reduce_scatter_partitions
            orig_copy = _relay_shared._copy_packed_to_slot

            def tracked_packed(*args):
                nonlocal packed_calls
                packed_calls += 1
                orig_packed(*args)

            def tracked_copy(slot, srcs, spans, span_ends, start, end, asynchronous):
                nonlocal copied_bytes
                copied_bytes += sum(part[3] for part in _relay_shared._chunk_pieces(spans, span_ends, start, end))
                orig_copy(slot, srcs, spans, span_ends, start, end, asynchronous)

            count = 1025
            varied = torch.arange(torch.empty(count)[::31].numel(), dtype=torch.float32) % 19 - 9
            vals = (256.0, -256.0)
            sources = []
            for owner in range(2):
                t = torch.full((count,), vals[rank], dtype=torch.bfloat16)
                t[::31] = (varied + (rank + owner) * 0.5).to(torch.bfloat16)
                sources.append(t)
            out = torch.empty(count, dtype=torch.bfloat16)
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(sub_shared, "_reduce_scatter_partitions", tracked_packed)
                patch.setattr(_relay_shared, "_copy_packed_to_slot", tracked_copy)
                sub_shared.reduce_scatter([out], [sources], manager)
            assert packed_calls == 1
            assert copied_bytes == out.nbytes
            exp = torch.zeros(count, dtype=torch.bfloat16)
            exp[::31] = functools.reduce(
                torch.add, ((varied + (r + rank) * 0.5).to(torch.bfloat16).float() for r in range(2)),
            ).to(torch.bfloat16)
            torch.testing.assert_close(out, exp, rtol=0, atol=0)
        finally:
            dist.destroy_process_group(subgroup)


def _run_shared_reduce_scatter_suite(rank: int, path: str) -> None:
    torch.set_num_threads(1)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(path, 4), rank=rank, world_size=4,
        timeout=timedelta(seconds=20),
        pg_options=RelayOptions(transport="shared", staging_bytes=192 * 1024),
    )
    manager = PinManager(0)
    try:
        shared = dist.group.WORLD._shared
        _check_shared_rs_precision_and_alias(shared, rank, manager)
        _check_shared_rs_volume(shared, rank, manager)
        _check_shared_rs_coalesced_and_fused(shared, rank, manager)
        _check_shared_rs_mismatch_and_serial(rank, shared, manager)
        _check_shared_rs_two_rank_subgroup(rank, manager)
    finally:
        dist.destroy_process_group()


def test_shared_reduce_scatter_direct_and_serial(tmp_path):
    mp.spawn(_run_shared_reduce_scatter_suite, args=(str(tmp_path / "shared-rs"),), nprocs=4)


def _run_gloo_two_rank_wire_and_broadcast(rank: int, path: str) -> None:
    torch.set_num_threads(1)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(path, 2), rank=rank, world_size=2,
        timeout=timedelta(seconds=20), pg_options=RelayOptions(staging_bytes=512, pipeline_buffers=2),
    )
    group = dist.group.WORLD
    measured = _MeasuredTransport(group._cpu_group, 512, False, world_size=2)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(group, "_cpu_group", measured)
            # 1. Exact raw 2-byte wire volume on W=2 low-precision all_reduce + reduce_scatter_single.
            for dtype in (torch.float16, torch.bfloat16):
                measured.calls = dict.fromkeys(measured.calls, 0)
                measured.wire_bytes = dict.fromkeys(measured.wire_bytes, 0)
                # 41 elements = 82 bytes -> 80B chunk + 2B (1-element) tail chunk at capacity=80.
                values = (torch.arange(41, dtype=torch.float32) / 16 - 1.5 + rank * 0.25).to(dtype)
                reduced = values.clone()
                dist.all_reduce(reduced)
                peer_values = (torch.arange(41, dtype=torch.float32) / 16 - 1.5 + (1 - rank) * 0.25).to(dtype)
                exp_ar = (values.float() + peer_values.float()).to(dtype)
                torch.testing.assert_close(reduced, exp_ar, rtol=0, atol=0)

                rs_in = torch.stack([values + 0.5, values + 1.0])
                rs_out = torch.empty(41, dtype=dtype)
                dist.reduce_scatter_single(rs_out, rs_in)
                exp_rs = ((values + (rank + 1) * 0.5).float() + (peer_values + (rank + 1) * 0.5).float()).to(dtype)
                torch.testing.assert_close(rs_out, exp_rs, rtol=0, atol=0)

                assert measured.calls["allreduce"] == 0
                assert measured.calls["reduce_scatter"] == 0
                assert measured.calls["send"] == measured.calls["recv"] == 4
                assert measured.wire_bytes["send"] == measured.wire_bytes["recv"] == 2 * values.nbytes

            # 2. Broadcast uses slots=1 (capacity = 512 // (1 * 1 * 2) = 256 bytes) cleanly.
            measured.calls["broadcast"] = 0
            bc_payload = torch.arange(64, dtype=torch.float32) + 7.0  # 256 bytes == 1 chunk
            bc = bc_payload.clone() if rank == 0 else torch.zeros_like(bc_payload)
            dist.broadcast(bc, src=0)
            assert measured.calls["broadcast"] == 1
            torch.testing.assert_close(bc, bc_payload, rtol=0, atol=0)

            # 3. Full chunked matrix across all 3 dtypes on W=2.
            for dtype in DTYPES:
                _check_chunked_values(rank, 2, torch.device("cpu"), dtype)
    finally:
        dist.destroy_process_group()


def test_chunked_gloo_two_rank_wire_bytes_and_broadcast(tmp_path):
    mp.spawn(_run_gloo_two_rank_wire_and_broadcast, args=(str(tmp_path / "gloo-w2"),), nprocs=2)


def _make_fake_gloo_copy_env(group):
    from piper_offload._relay_staging import HostChunk

    class StubWork:
        def wait(self) -> None:
            pass

    class DummyCurrentStream:
        def __init__(self) -> None:
            self.syncs = 0

        def synchronize(self) -> None:
            self.syncs += 1

    class FakeCudaTensor:
        device = torch.device("cuda", 0)

        def __init__(self, data: torch.Tensor) -> None:
            with torch.inference_mode():
                self.data = data.clone()
            self.dtype = self.data.dtype
            self.nbytes = self.data.nbytes

        def data_ptr(self) -> int:
            return self.data.data_ptr()

        def numel(self) -> int:
            return self.data.numel()

        def reshape(self, *shape: int) -> torch.Tensor:
            return self.data.reshape(*shape)

    chunk_calls: list[tuple[int, int, int]] = []
    gloo_calls: list[tuple[str, list[torch.Tensor], list[torch.Tensor]]] = []
    state = {"fail_on_close": False}

    class FakeGlooGroup:
        def broadcast(self, tensors: list[torch.Tensor], options: object) -> StubWork:
            del options
            gloo_calls.append(("broadcast", [], tensors))
            return StubWork()

        def scatter(self, outputs: list[torch.Tensor], inputs: list[list[torch.Tensor]], options: object) -> StubWork:
            del options
            gloo_calls.append(("scatter", outputs, inputs[0]))
            return StubWork()

        def allgather(self, outputs: list[list[torch.Tensor]], inputs: list[torch.Tensor], options: object) -> StubWork:
            del options
            gloo_calls.append(("allgather", outputs[0], inputs))
            if inputs and inputs[0].dtype == torch.int64 and len(outputs[0]) == 2:
                meta = inputs[0].tolist()
                # Transpose rank 0's (send, recv) entries into rank 1's row for asymmetric exchange tests.
                outputs[0][0].copy_(inputs[0])
                outputs[0][1].copy_(torch.tensor([meta[3], meta[2], meta[1], meta[0], meta[4]], dtype=torch.int64))
            return StubWork()

        def send(self, tensors: list[torch.Tensor], peer: int, tag: int) -> StubWork:
            del peer, tag
            gloo_calls.append(("send", [], tensors))
            return StubWork()

        def recv(self, tensors: list[torch.Tensor], peer: int, tag: int) -> StubWork:
            del peer, tag
            gloo_calls.append(("recv", tensors, []))
            return StubWork()

    def gen_chunks(inputs: list[torch.Tensor], outputs: list[torch.Tensor], *, slots: int | None = None):
        slot_count = slots if slots is not None else len(inputs) + len(outputs)
        chunk_calls.append((len(inputs), len(outputs), slot_count))
        ref = outputs[0] if outputs else inputs[0]
        width = min(16, ref.numel() * 4)
        capacity = group._slot_bytes(slot_count)
        backing = torch.zeros(slot_count * capacity, dtype=torch.uint8)
        in_views = [backing[i * capacity:i * capacity + width] for i in range(len(inputs))]
        base = len(inputs)
        out_views = [backing[(base + i) * capacity:(base + i) * capacity + width] for i in range(len(outputs))]
        scratch = backing[(base + len(outputs)) * capacity:]
        yield HostChunk(in_views, out_views, scratch)
        if state["fail_on_close"]:
            raise RuntimeError("injected chunk teardown failure")

    return FakeGlooGroup(), DummyCurrentStream(), FakeCudaTensor, gen_chunks, chunk_calls, gloo_calls, state


def test_gloo_cuda_local_copy_paths_avoid_aliasing_and_roundtrips(single_rank_group):
    group = single_rank_group
    fake_gloo, current, fake_cuda_tensor, gen_chunks, chunk_calls, gloo_calls, state = _make_fake_gloo_copy_env(group)
    with pytest.MonkeyPatch.context() as patch, torch.no_grad():
        patch.setattr(group, "size", lambda: 2)
        patch.setattr(group, "rank", lambda: 0)
        patch.setattr(group, "_cpu_group", fake_gloo)
        patch.setattr(group, "_copy_chunks", gen_chunks)
        patch.setattr(torch.cuda, "current_stream", lambda _device: current)

        bc_tensor = torch.arange(4, dtype=torch.float32)
        opts_bc = dist.BroadcastOptions()
        opts_bc.rootRank = 0
        group.broadcast([bc_tensor], opts_bc).wait()
        assert chunk_calls[-1] == (1, 0, 1)
        opts_bc.rootRank = 1
        group.broadcast([bc_tensor], opts_bc).wait()
        assert chunk_calls[-1] == (0, 1, 1)

        sc_srcs = [fake_cuda_tensor(torch.full((4,), 11.0)), fake_cuda_tensor(torch.full((4,), 22.0))]
        sc_out = fake_cuda_tensor(torch.zeros(4))
        opts_sc = dist.ScatterOptions()
        opts_sc.rootRank = 0
        group._scatter_gloo(sc_out, sc_srcs, opts_sc)  # type: ignore[arg-type]
        assert chunk_calls[-1] == (1, 0, 3) and current.syncs == 1
        assert torch.equal(sc_out.data, sc_srcs[0].data)
        op, out_list, in_list = gloo_calls[-1]
        assert op == "scatter" and len(out_list) == 1 and len(in_list) == 2
        assert out_list[0].data_ptr() not in {t.data_ptr() for t in in_list}
        group._scatter_gloo(sc_srcs[0], sc_srcs, opts_sc)  # type: ignore[arg-type]
        empty_sc = [fake_cuda_tensor(torch.empty(0)), fake_cuda_tensor(torch.empty(0))]
        group._scatter_gloo(fake_cuda_tensor(torch.empty(0)), empty_sc, opts_sc)  # type: ignore[arg-type]
        assert current.syncs == 1

        ag_src = fake_cuda_tensor(torch.full((4,), 33.0))
        ag_outs = [fake_cuda_tensor(torch.zeros(4)), fake_cuda_tensor(torch.zeros(4))]
        group._gather(ag_src, ag_outs, communication.AllgatherOptions())  # type: ignore[arg-type]
        assert chunk_calls[-1] == (1, 1, 3) and current.syncs == 2
        assert torch.equal(ag_outs[0].data, ag_src.data)
        op, out_list, in_list = gloo_calls[-1]
        assert op == "allgather" and len(out_list) == 2 and len(in_list) == 1
        assert out_list[0].data_ptr() not in (in_list[0].data_ptr(), out_list[1].data_ptr())
        ag_alias = [ag_src, fake_cuda_tensor(torch.zeros(4))]
        group._gather(ag_src, ag_alias, communication.AllgatherOptions())  # type: ignore[arg-type]
        empty_ag = [fake_cuda_tensor(torch.empty(0)), fake_cuda_tensor(torch.empty(0))]
        group._gather(fake_cuda_tensor(torch.empty(0)), empty_ag, communication.AllgatherOptions())  # type: ignore[arg-type]
        assert current.syncs == 2

        state["fail_on_close"] = True
        with pytest.raises(RuntimeError, match="injected chunk teardown failure"):
            group._scatter_gloo(fake_cuda_tensor(torch.zeros(4)), sc_srcs, opts_sc)  # type: ignore[arg-type]
        fail_outs = [fake_cuda_tensor(torch.zeros(4)), fake_cuda_tensor(torch.zeros(4))]
        with pytest.raises(RuntimeError, match="injected chunk teardown failure"):
            group._gather(ag_src, fail_outs, communication.AllgatherOptions())  # type: ignore[arg-type]
        assert current.syncs == 4


def test_gloo_exchange_pair_drains_recv_on_send_failure(single_rank_group):
    group = single_rank_group

    class FailingWork:
        def __init__(self, name: str, log: list[str], error: BaseException) -> None:
            self.name = name
            self.log = log
            self.error = error

        def wait(self, *_args: object) -> bool:
            self.log.append(self.name)
            raise self.error

    class FakeGroup:
        def __init__(self, log: list[str], *, fail_send_post: bool = False) -> None:
            self.log = log
            self.fail_send_post = fail_send_post

        def recv(self, *_args: object) -> FailingWork:
            return FailingWork("recv", self.log, RuntimeError("recv failed"))

        def send(self, *_args: object) -> FailingWork:
            if self.fail_send_post:
                raise RuntimeError("send post failed")
            return FailingWork("send", self.log, RuntimeError("send failed"))

    buf = torch.empty(4, dtype=torch.uint8)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(group, "size", lambda: 2)
        patch.setattr(group, "rank", lambda: 0)

        wait_log: list[str] = []
        patch.setattr(group, "_cpu_group", FakeGroup(wait_log))
        with pytest.raises(RuntimeError, match="send failed"):
            group._exchange_pair(buf, buf, timedelta(0))
        assert wait_log == ["send", "recv"]

        post_log: list[str] = []
        patch.setattr(group, "_cpu_group", FakeGroup(post_log, fail_send_post=True))
        with pytest.raises(RuntimeError, match="send post failed"):
            group._exchange_pair(buf, buf, timedelta(seconds=1))
        assert post_log == ["recv"]


def _check_single_rank_alltoall_valid(group) -> None:
    alias = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    single_work = group.all_to_all_single(alias, alias, None, None)
    assert single_work.wait()
    assert single_work.get_future().wait() == [alias]
    torch.testing.assert_close(alias, torch.arange(6, dtype=torch.float32).reshape(2, 3))

    empty_rows = torch.empty(0, 4, dtype=torch.int64)
    empty_out = torch.empty(0, 4, dtype=torch.int64)
    dist.all_to_all_single(empty_out, empty_rows)

    real = torch.arange(4, dtype=torch.float32)
    imag = torch.arange(4, 8, dtype=torch.float32)
    for dtype in (torch.float64, torch.bool, torch.uint8, torch.complex64, torch.complex128):
        if dtype is torch.bool:
            src_dt = torch.tensor([[True, False], [False, True]])
        elif dtype in (torch.complex64, torch.complex128):
            src_dt = torch.complex(real, imag).to(dtype)
        else:
            src_dt = torch.arange(4, dtype=dtype)
        dst_dt = torch.empty_like(src_dt)
        list_work = group.alltoall([dst_dt], [src_dt])
        fut = list_work.get_future().wait()
        assert len(fut) == 1 and fut[0] is dst_dt
        assert torch.equal(dst_dt, src_dt)


def test_reject_unsupported_and_invalid_alltoall(single_rank_group):
    group = single_rank_group
    out = torch.empty(2, 2)
    src = torch.ones(2, 2)

    class TensorSubclass(torch.Tensor):
        pass

    with pytest.raises(TypeError, match="plain local tensor"):
        group.alltoall([out], [TensorSubclass(src)])
    with pytest.raises(NotImplementedError, match="contiguous strided"):
        dist.all_to_all_single(out, torch.ones(4, 2)[::2])
    with pytest.raises(NotImplementedError, match="does not support dtype"):
        dist.all_to_all_single(
            torch.empty(2, dtype=torch.bfloat16).view(torch.uint16),
            torch.ones(2, dtype=torch.int16).view(torch.uint16),
        )
    with pytest.raises(ValueError, match="one input and one output tensor per rank"):
        group.alltoall([out, out], [src])
    with pytest.raises(ValueError, match="one dtype across"):
        group.alltoall([out], [src.to(torch.float16)])
    with pytest.raises(ValueError, match="input and output dtypes must match"):
        group.all_to_all_single(torch.empty(4, dtype=torch.float16), torch.ones(4, dtype=torch.float32))
    with pytest.raises(ValueError, match="at least one dimension"):
        dist.all_to_all_single(torch.empty(()), torch.tensor(1.0))
    with pytest.raises(ValueError, match="trailing dimensions must match"):
        dist.all_to_all_single(torch.empty(2, 3), torch.ones(2, 4))
    with pytest.raises(ValueError, match="non-negative integer split sizes"):
        dist.all_to_all_single(torch.empty(2), torch.ones(2), [True], [2])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-negative integer split sizes"):
        dist.all_to_all_single(torch.empty(2), torch.ones(2), [-1], [2])
    with pytest.raises(ValueError, match="must sum to dim 0"):
        dist.all_to_all_single(torch.empty(3), torch.ones(3), [2], [3])
    with pytest.raises(ValueError, match="own-rank input and output byte sizes must match"):
        group.alltoall([torch.empty(2)], [torch.ones(3)])

    backing = torch.arange(6, dtype=torch.float32)
    with pytest.raises(ValueError, match="exact local slice"):
        group.alltoall([backing[1:5]], [backing[:4]])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(group, "size", lambda: 2)
        with pytest.raises(ValueError, match="divisible by world size"):
            dist.all_to_all_single(torch.empty(3, 2), torch.ones(3, 2))
        with pytest.raises(ValueError, match="outputs must not overlap"):
            group.alltoall([backing[:2], backing[1:3]], [torch.ones(2), torch.ones(2)])

    _check_single_rank_alltoall_valid(group)


def _check_alltoall_mismatch_and_shard_dim(rank: int, subgroup) -> None:
    sentinel = torch.full((4,), -9.0)
    bad_outputs = [sentinel.clone() for _ in range(4)]
    bad_inputs = [
        torch.ones(5 if (rank == 0 and dst == 1) else 4) for dst in range(4)
    ]
    with pytest.raises(ValueError, match="split sizes must match across ranks"):
        subgroup.alltoall(bad_outputs, bad_inputs).wait()
    for got in bad_outputs:
        torch.testing.assert_close(got, sentinel, rtol=0, atol=0)
    if subgroup._shared is not None:
        assert not subgroup._shared.broken

    bad_dtype_out = torch.empty(4, dtype=torch.float16 if rank == 1 else torch.float32)
    bad_dtype_in = torch.ones(4, dtype=bad_dtype_out.dtype)
    with pytest.raises(ValueError, match="must match"):
        dist.all_to_all_single(bad_dtype_out, bad_dtype_in, group=subgroup)
    if subgroup._shared is not None:
        assert not subgroup._shared.broken

    # Group remains healthy immediately after both error cases (with GlooGuard still active on shared).
    recovery_in = torch.arange(4, dtype=torch.int32) + rank * 10
    recovery_out = torch.empty_like(recovery_in)
    dist.all_to_all_single(recovery_out, recovery_in, group=subgroup)
    assert torch.equal(recovery_out, torch.arange(4, dtype=torch.int32) * 10 + rank)

    # Ulysses-style Shard(0) -> Shard(1) via shard_dim_alltoall across 4 ranks.
    global_grid = torch.arange(8 * 12, dtype=torch.float16).reshape(8, 12)
    shard0 = global_grid.chunk(4, dim=0)[rank].contiguous()
    shard1 = torch.ops._dtensor.shard_dim_alltoall(shard0, 0, 1, subgroup.group_name)
    torch.testing.assert_close(shard1, global_grid.chunk(4, dim=1)[rank], rtol=0, atol=0)


def _check_alltoall_four_rank_group(rank: int, transport: str, force_gloo_control: bool) -> None:
    from piper_offload import _relay_shared

    staging = 384 if transport == "shared" and force_gloo_control else (
        192 * 1024 if transport == "shared" else 512
    )
    opts = RelayOptions(transport=transport, staging_bytes=staging, pipeline_buffers=2 if transport == "gloo" else 1)
    with pytest.MonkeyPatch.context() as patch:
        if force_gloo_control:
            patch.setattr(_relay_shared, "shared_control_supported", lambda: False)
        subgroup = dist.new_group(ranks=[0, 1, 2, 3], pg_options=opts)
    try:
        rows_matrix = [
            [2, 0, 5, 1],
            [0, 3, 2, 0],
            [4, 1, 0, 7],
            [1, 6, 3, 2],
        ]
        width = 3
        in_splits = rows_matrix[rank]
        out_splits = [rows_matrix[src][rank] for src in range(4)]

        def make_chunk(src: int, dst: int) -> torch.Tensor:
            rows = rows_matrix[src][dst]
            base = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width)
            return (base + src * 100 + dst * 10).to(torch.bfloat16)

        src_tensor = torch.cat([make_chunk(rank, dst) for dst in range(4)], dim=0)
        expected_out = torch.cat([make_chunk(src, rank) for src in range(4)], dim=0)

        if transport == "shared":
            subgroup._cpu_group = _GlooGuard(
                subgroup._cpu_group, allow_allreduce=False, allow_control=force_gloo_control,
            )
        out_tensor = torch.empty((sum(out_splits), width), dtype=torch.bfloat16)
        dist.all_to_all_single(out_tensor, src_tensor, out_splits, in_splits, group=subgroup)
        torch.testing.assert_close(out_tensor, expected_out, rtol=0, atol=0)

        # Multi-chunk transfer exceeding serial (32B) and Gloo (128B) staging with sparse/zero pairs.
        large_matrix = [
            [0, 65, 0, 17],
            [33, 0, 71, 0],
            [0, 0, 0, 99],
            [45, 19, 0, 0],
        ]
        l_in_splits = large_matrix[rank]
        l_out_splits = [large_matrix[src][rank] for src in range(4)]
        l_src_list = [
            torch.arange(rows, dtype=torch.int64) + (rank * 1000 + dst * 100) for dst, rows in enumerate(l_in_splits)
        ]
        l_out_list = [torch.empty(rows, dtype=torch.int64) for rows in l_out_splits]
        subgroup.alltoall(l_out_list, l_src_list).wait()
        for src_r, got in enumerate(l_out_list):
            exp = torch.arange(l_out_splits[src_r], dtype=torch.int64) + (src_r * 1000 + rank * 100)
            assert torch.equal(got, exp)

        _check_alltoall_mismatch_and_shard_dim(rank, subgroup)
    finally:
        dist.destroy_process_group(subgroup)


def _check_shared_alltoall_path(rank: int, staging_bytes: int, expect_serial: bool) -> None:
    from piper_offload import _relay_shared

    sub = dist.new_group(
        ranks=[0, 1, 2, 3], pg_options=RelayOptions(transport="shared", staging_bytes=staging_bytes),
    )
    try:
        shared = sub._shared
        assert (_relay_shared._reduce_scatter_sub_cap(shared.capacity, 4) == 0) is expect_serial
        serial_calls = 0
        packed_calls = 0
        orig_serial = shared._alltoall_serial
        orig_packed = shared._alltoall_packed

        def tracked_serial(*args):
            nonlocal serial_calls
            serial_calls += 1
            orig_serial(*args)

        def tracked_packed(*args):
            nonlocal packed_calls
            packed_calls += 1
            orig_packed(*args)

        srcs = [torch.full((25,), float(rank * 10 + dst + 1)) for dst in range(4)]
        outs = [torch.empty_like(srcs[0]) for _ in range(4)]
        outs[rank] = srcs[rank]
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(shared, "_alltoall_serial", tracked_serial)
            patch.setattr(shared, "_alltoall_packed", tracked_packed)
            sub.alltoall(outs, srcs).wait()
        for peer in range(4):
            torch.testing.assert_close(outs[peer], torch.full((25,), float(peer * 10 + rank + 1)))
        assert serial_calls == (1 if expect_serial else 0)
        assert packed_calls == (0 if expect_serial else 1)
    finally:
        dist.destroy_process_group(sub)


def _run_alltoall_four_rank_worker(rank: int, path: str) -> None:
    torch.set_num_threads(1)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(path, 4), rank=rank, world_size=4,
        timeout=timedelta(seconds=20), pg_options=RelayOptions(transport="gloo", staging_bytes=4096),
    )
    try:
        for transport, force_gloo_control in (("gloo", False), ("shared", False), ("shared", True)):
            _check_alltoall_four_rank_group(rank, transport, force_gloo_control)
        for staging_bytes, expect_serial in ((384, True), (192 * 1024, False)):
            _check_shared_alltoall_path(rank, staging_bytes, expect_serial)
    finally:
        dist.destroy_process_group()


def test_shared_and_gloo_alltoall_unequal_mismatch_and_serial(tmp_path):
    mp.spawn(_run_alltoall_four_rank_worker, args=(str(tmp_path / "a2a-4"),), nprocs=4)


def test_alltoall_gloo_cuda_mock_inference_and_sync(single_rank_group):
    group = single_rank_group
    fake_gloo, current, fake_cuda_tensor, gen_chunks, chunk_calls, gloo_calls, state = _make_fake_gloo_copy_env(group)
    opts = communication.AllToAllOptions()
    manager = PinManager(0)
    with pytest.MonkeyPatch.context() as patch, torch.no_grad():
        patch.setattr(communication, "host_pin_manager", manager)
        patch.setattr(group, "_cpu_group", fake_gloo)
        patch.setattr(group, "_copy_chunks", gen_chunks)
        patch.setattr(group, "_staging_bytes", 512)
        patch.setattr(torch.cuda, "current_stream", lambda _device: current)

        w1_src = fake_cuda_tensor(torch.arange(4, dtype=torch.float32) + 3.0)
        w1_out = fake_cuda_tensor(torch.zeros(4))
        group._alltoall_gloo([w1_out], [w1_src], opts)  # type: ignore[arg-type]
        assert current.syncs == 1
        assert torch.equal(w1_out.data, w1_src.data)
        group._alltoall_gloo([w1_src], [w1_src], opts)  # type: ignore[arg-type]
        assert current.syncs == 1

        patch.setattr(group, "size", lambda: 2)
        patch.setattr(group, "rank", lambda: 0)
        srcs = [fake_cuda_tensor(torch.full((4,), 7.0)), fake_cuda_tensor(torch.full((4,), 8.0))]
        outs = [fake_cuda_tensor(torch.zeros(4)), fake_cuda_tensor(torch.zeros(4))]
        group._alltoall_gloo(outs, srcs, opts)  # type: ignore[arg-type]
        assert current.syncs == 2 and chunk_calls[-1] == (1, 1, 2)
        assert torch.equal(outs[0].data, srcs[0].data)
        assert gloo_calls[-1][0] == "send" and gloo_calls[-2][0] == "recv"

        # Multi-chunk unequal send/recv (160B vs 40B, slot capacity 128B) with an exact local alias:
        # exercises both asymmetric chunk counts and _exchange_unequal_gloo's unconditional CUDA sync.
        alias_t = fake_cuda_tensor(torch.full((4,), 9.0))
        unequal_srcs = [alias_t, fake_cuda_tensor(torch.arange(40, dtype=torch.float32))]
        unequal_outs = [alias_t, fake_cuda_tensor(torch.zeros(10, dtype=torch.float32))]
        calls_before = len(gloo_calls)
        group._alltoall_gloo(unequal_outs, unequal_srcs, opts)  # type: ignore[arg-type]
        new_calls = [name for name, _, _ in gloo_calls[calls_before:]]
        assert new_calls == ["allgather", "recv", "send", "send"]
        assert current.syncs == 3 and group._staging_buffer is not None
        group._staging_buffer = None

        state["fail_on_close"] = True
        with pytest.raises(RuntimeError, match="injected chunk teardown failure"):
            group._alltoall_gloo([fake_cuda_tensor(torch.zeros(4)), outs[1]], srcs, opts)  # type: ignore[arg-type]
        assert current.syncs == 4


def test_alltoall_shared_cuda_mock_stream_syncs(tmp_path):
    (
        shared, control, manager, download_stream, upload_stream,
        current_s, stream_entries, track_stream, dummy_event, shared_fake_cuda,
    ) = _make_fake_collective_relay(tmp_path)
    stream_iter = iter((download_stream, upload_stream))
    with pytest.MonkeyPatch.context() as patch, torch.no_grad():
        patch.setattr(torch.cuda, "Stream", lambda device: next(stream_iter))
        patch.setattr(torch.cuda, "Event", dummy_event)
        patch.setattr(torch.cuda, "current_stream", lambda _device: current_s)
        patch.setattr(torch.cuda, "stream", track_stream)

        a2a_src = torch.arange(8, dtype=torch.float32) + 2.0
        a2a_out = torch.zeros(8, dtype=torch.float32)
        shared.alltoall([shared_fake_cuda(a2a_out)], [shared_fake_cuda(a2a_src)], manager)
        assert (download_stream.sync_count, upload_stream.sync_count, current_s.sync_count) == (0, 1, 0)
        assert stream_entries == ["upload"]
        assert torch.equal(a2a_out, a2a_src)

        # W=1 exact alias and 0-element non-alias skip _local_copy and sync nothing.
        download_stream.sync_count = upload_stream.sync_count = current_s.sync_count = 0
        shared.alltoall([shared_fake_cuda(a2a_src)], [shared_fake_cuda(a2a_src)], manager)
        shared.alltoall([shared_fake_cuda(torch.empty(0))], [shared_fake_cuda(torch.empty(0))], manager)
        assert (download_stream.sync_count, upload_stream.sync_count, current_s.sync_count) == (0, 0, 0)

        # W=2 max_pair_bytes == 0 with non-alias local copy synchronizes only upload.
        shared.world_size = 2
        shared._peers = ((1, 1),)
        shared._slot(1, 0, 32).zero_()
        w2_local_out = torch.zeros(8, dtype=torch.float32)
        shared.alltoall(
            [shared_fake_cuda(w2_local_out), shared_fake_cuda(torch.empty(0))],
            [shared_fake_cuda(a2a_src), shared_fake_cuda(torch.empty(0))],
            manager,
        )
        assert (download_stream.sync_count, upload_stream.sync_count, current_s.sync_count) == (0, 1, 0)
        assert torch.equal(w2_local_out, a2a_src)
        shared.world_size = 1
        shared._peers = ()

        download_stream.sync_count = upload_stream.sync_count = current_s.sync_count = 0
        manager.pageable_bytes = 64
        pg_out = torch.zeros(8, dtype=torch.float32)
        shared.alltoall([shared_fake_cuda(pg_out)], [shared_fake_cuda(a2a_src)], manager)
        assert (download_stream.sync_count, upload_stream.sync_count, current_s.sync_count) == (0, 0, 1)
        assert torch.equal(pg_out, a2a_src)

        download_stream.sync_count = upload_stream.sync_count = current_s.sync_count = 0
        shared.world_size = 2
        shared._peers = ((1, 1),)
        control.fail_on_signal = True
        with pytest.raises(RuntimeError, match="injected signal failure"):
            shared.alltoall(
                [shared_fake_cuda(torch.zeros(4)), shared_fake_cuda(torch.zeros(4))],
                [shared_fake_cuda(torch.ones(4)), shared_fake_cuda(torch.ones(4))],
                manager,
            )
        assert (download_stream.sync_count, upload_stream.sync_count, current_s.sync_count) == (0, 0, 0)
        assert shared.broken


def test_reject_invalid_p2p(single_rank_group):
    group = single_rank_group
    t = torch.ones(4)

    class TensorSubclass(torch.Tensor):
        pass

    with pytest.raises(NotImplementedError, match="exactly one local tensor"):
        group.send([t, t], 0, 0)
    with pytest.raises(TypeError, match="plain local tensor"):
        group.send([TensorSubclass(t)], 0, 0)
    with pytest.raises(NotImplementedError, match="contiguous strided"):
        dist.isend(torch.ones(4, 2)[::2], dst=0)
    with pytest.raises(NotImplementedError, match="does not support dtype"):
        dist.irecv(torch.empty(2, dtype=torch.bfloat16).view(torch.uint16), src=0)
    with pytest.raises(ValueError, match="do not support the local rank"):
        dist.isend(t, dst=0)
    with pytest.raises(ValueError, match="do not support the local rank"):
        dist.irecv(t, src=0)
    with pytest.raises(ValueError, match="at least two ranks"):
        group.recv_anysource([t], 0)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(group, "size", lambda: 2)
        with pytest.raises(ValueError, match="outside the process group"):
            group.send([t], 2, 0)
        with pytest.raises(ValueError, match="outside the process group"):
            group.recv([t], -1, 0)
        for bad_tag in (-1, True, communication._MAX_P2P_TAG + 1):
            with pytest.raises(ValueError, match="point-to-point tag must be"):
                group.send([t], 1, bad_tag)  # type: ignore[arg-type]

    ar_work = dist.all_reduce(t, async_op=True)
    assert ar_work.wait()
    with pytest.raises(RuntimeError, match="only available on point-to-point receive"):
        ar_work.source_rank()

    group.shutdown()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(group, "size", lambda: 2)
        with pytest.raises(RuntimeError, match="shutdown"):
            group.send([t], 1, 0)


def _p2p_sample(dtype: torch.dtype, idx: int, side: int) -> torch.Tensor:
    if dtype is torch.bool:
        return torch.tensor([side == 0, side == 1, True])
    if dtype in (torch.complex64, torch.complex128):
        real = torch.arange(3, dtype=torch.float32) + side * 10
        return torch.complex(real, real + 1).to(dtype)
    return (torch.arange(5, dtype=torch.float32) + side * 10 + idx).to(dtype)


def _check_p2p_two_rank_dtypes(rank: int, group, global_ranks: tuple[int, int]) -> None:
    local_idx = global_ranks.index(rank)
    peer_global = global_ranks[1 - local_idx]

    dtypes = (
        torch.float32, torch.float16, torch.bfloat16, torch.float64,
        torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        torch.complex64, torch.complex128,
    )
    for idx, dtype in enumerate(dtypes):
        src = _p2p_sample(dtype, idx, local_idx)
        dst = torch.empty_like(src)
        for req in dist.batch_isend_irecv([
            dist.P2POp(dist.isend, src, peer_global, group=group, tag=idx),
            dist.P2POp(dist.irecv, dst, peer_global, group=group, tag=idx),
        ]):
            req.wait()
        assert torch.equal(dst, _p2p_sample(dtype, idx, 1 - local_idx))

    empty_out = torch.empty(0, dtype=torch.uint8)
    for empty_req in dist.batch_isend_irecv([
        dist.P2POp(dist.isend, torch.empty(0, dtype=torch.uint8), peer_global, group=group, tag=0),
        dist.P2POp(dist.irecv, empty_out, peer_global, group=group, tag=0),
    ]):
        empty_req.wait()

    bits = torch.tensor([0, -32768, 1, 0x7E01, 0x7FC1], dtype=torch.int16).repeat(60)
    bit_src = bits.roll(local_idx).view(torch.float16)
    bit_dst = torch.empty_like(bit_src)
    if local_idx == 0:
        dist.send(bit_src, dst=peer_global, group=group, tag=2)
        assert dist.recv(bit_dst, src=None, group=group, tag=1) == peer_global
    else:
        assert dist.recv(bit_dst, src=None, group=group, tag=2) == peer_global
        dist.send(bit_src, dst=peer_global, group=group, tag=1)
    assert torch.equal(bit_dst.view(torch.int16), bits.roll(1 - local_idx))


def _check_p2p_two_rank_ordering(rank: int, group, global_ranks: tuple[int, int]) -> None:
    local_idx = global_ranks.index(rank)
    peer_global = global_ranks[1 - local_idx]

    a_send = torch.arange(400, dtype=torch.float32) + local_idx * 1000
    b_send = torch.arange(400, dtype=torch.float32) + local_idx * 2000 + 500
    a_recv, b_recv = torch.empty_like(a_send), torch.empty_like(b_send)
    for req in [
        dist.isend(a_send, dst=peer_global, group=group, tag=0),
        dist.isend(b_send, dst=peer_global, group=group, tag=0),
        dist.irecv(a_recv, src=peer_global, group=group, tag=0),
        dist.irecv(b_recv, src=peer_global, group=group, tag=0),
    ]:
        req.wait()
    torch.testing.assert_close(a_recv, torch.arange(400, dtype=torch.float32) + (1 - local_idx) * 1000)
    torch.testing.assert_close(b_recv, torch.arange(400, dtype=torch.float32) + (1 - local_idx) * 2000 + 500)

    r1, r2 = torch.empty(300, dtype=torch.int64), torch.empty(300, dtype=torch.int64)
    s1 = torch.arange(300, dtype=torch.int64) + local_idx * 10 + 1
    s2 = torch.arange(300, dtype=torch.int64) + local_idx * 10 + 2
    ops = [
        dist.P2POp(dist.isend, s1, peer_global, group=group, tag=1),
        dist.P2POp(dist.isend, s2, peer_global, group=group, tag=2),
        dist.P2POp(dist.irecv, r1, peer_global, group=group, tag=1),
        dist.P2POp(dist.irecv, r2, peer_global, group=group, tag=2),
    ]
    for w in dist.batch_isend_irecv(ops if local_idx == 0 else list(reversed(ops))):
        w.wait()
    assert torch.equal(r1, torch.arange(300, dtype=torch.int64) + (1 - local_idx) * 10 + 1)
    assert torch.equal(r2, torch.arange(300, dtype=torch.int64) + (1 - local_idx) * 10 + 2)


def _check_p2p_ring_and_anysource(rank: int, group) -> None:
    # 4-rank Ring Attention exchange interleaved with collectives using tags 0, 1, 2.
    send_buf = torch.arange(300, dtype=torch.float32) + rank * 1000
    for step in range(3):
        recv_buf = torch.empty_like(send_buf)
        for req in dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_buf, (rank + 1) % 4, group=group, tag=step % 3),
            dist.P2POp(dist.irecv, recv_buf, (rank - 1) % 4, group=group, tag=step % 3),
        ]):
            req.wait()
        src_rank = (rank - 1 - step) % 4
        torch.testing.assert_close(recv_buf, torch.arange(300, dtype=torch.float32) + src_rank * 1000)
        send_buf = recv_buf
    probe = torch.full((4,), float(rank + 1))
    dist.all_reduce(probe, group=group)
    torch.testing.assert_close(probe, torch.full((4,), 10.0))
    a2a_in = torch.arange(4, dtype=torch.int32) + rank * 10
    a2a_out = torch.empty_like(a2a_in)
    dist.all_to_all_single(a2a_out, a2a_in, group=group)
    assert torch.equal(a2a_out, torch.arange(4, dtype=torch.int32) * 10 + rank)

    # Multi-chunk recv_anysource from three senders completes un-interleaved with exact sender ranks.
    if rank == 0:
        works = [
            dist.irecv(torch.empty(260, dtype=torch.int64), src=None, group=group, tag=9)
            for _ in range(3)
        ]
        got_by_src = {w.source_rank(): w.result()[0] for w in works}
        assert set(got_by_src) == {1, 2, 3}
        for sender, out_t in got_by_src.items():
            assert torch.equal(out_t, torch.arange(260, dtype=torch.int64) + sender * 500)
    else:
        dist.send(torch.arange(260, dtype=torch.int64) + rank * 500, dst=0, group=group, tag=9)


def _run_p2p_worker(rank: int, path: str) -> None:
    from piper_offload import _relay_shared

    torch.set_num_threads(1)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(path, 4), rank=rank, world_size=4,
        timeout=timedelta(seconds=20), pg_options=RelayOptions(transport="gloo", staging_bytes=512),
    )
    try:
        for opts in (
            RelayOptions(transport="gloo", staging_bytes=512),
            RelayOptions(transport="shared", staging_bytes=4096),
        ):
            sub = dist.new_group(ranks=[0, 1, 2, 3], pg_options=opts)
            try:
                _check_p2p_ring_and_anysource(rank, sub)
            finally:
                dist.destroy_process_group(sub)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(_relay_shared, "shared_control_supported", lambda: False)
            ctrl_sub = dist.new_group(ranks=[1, 2], pg_options=RelayOptions(transport="shared", staging_bytes=512))
        try:
            if rank in (1, 2):
                _check_p2p_two_rank_dtypes(rank, ctrl_sub, (1, 2))
                _check_p2p_two_rank_ordering(rank, ctrl_sub, (1, 2))
                # Verify tag isolation between P2P (tag >= 8) and _GlooControl (0/1) + _GLOO_EXCHANGE_TAG (2).
                idx = (1, 2).index(rank)
                ctrl_probe = torch.full((2,), float(idx + 1))
                dist.all_reduce(ctrl_probe, group=ctrl_sub)
                torch.testing.assert_close(ctrl_probe, torch.full((2,), 3.0))
                ctrl_a2a_in = torch.arange(2, dtype=torch.int32) + idx * 10
                ctrl_a2a_out = torch.empty_like(ctrl_a2a_in)
                dist.all_to_all_single(ctrl_a2a_out, ctrl_a2a_in, group=ctrl_sub)
                assert torch.equal(ctrl_a2a_out, torch.arange(2, dtype=torch.int32) * 10 + idx)
        finally:
            dist.destroy_process_group(ctrl_sub)
    finally:
        dist.destroy_process_group()


def test_p2p_gloo_and_shared(tmp_path):
    mp.spawn(_run_p2p_worker, args=(str(tmp_path / "p2p-4"),), nprocs=4)


def _make_mock_cuda_p2p_env(group, patch):
    from contextlib import nullcontext

    recorded_events: list[object] = []
    stream_syncs: list[int] = []
    streams: list[object] = []
    p2p_log: list[tuple[str, int, int, int]] = []
    state: dict[str, object] = {"fail_send": False, "fail_sync": False, "block_gate": None, "entered": 0, "target": 1}
    gate_lock = threading.Lock()
    entered_event = threading.Event()

    class DummyEvent:
        def __init__(self) -> None:
            self.synced = False

        def record(self, _stream: object) -> None:
            recorded_events.append(self)

        def synchronize(self) -> None:
            self.synced = True

    class DummyStream:
        def __init__(self, device: torch.device) -> None:
            self.device = device

        def synchronize(self) -> None:
            stream_syncs.append(1)
            if state["fail_sync"]:
                raise RuntimeError("injected stream synchronize failure")

    class MockWork(dist.Work):
        def __init__(self, fn: Callable[[], None], src: int = -1) -> None:
            super().__init__()
            self._fn = fn
            self._src = src

        def wait(self, *_args: object) -> bool:
            self._fn()
            return True

        def _source_rank(self) -> int:
            return self._src

    class FakeP2PGloo:
        def send(self, tensors: list[torch.Tensor], dst: int, tag: int) -> MockWork:
            def run() -> None:
                gate = state["block_gate"]
                if isinstance(gate, threading.Event):
                    with gate_lock:
                        state["entered"] = int(state["entered"]) + 1  # type: ignore[arg-type]
                        if int(state["entered"]) >= int(state["target"]):  # type: ignore[arg-type]
                            entered_event.set()
                    gate.wait(timeout=5)
                if state["fail_send"]:
                    raise RuntimeError("injected p2p send failure")
                p2p_log.append(("send", dst, tag, tensors[0].untyped_storage().data_ptr()))
            return MockWork(run)

        def recv(self, tensors: list[torch.Tensor], src: int, tag: int) -> MockWork:
            def run() -> None:
                p2p_log.append(("recv", src, tag, tensors[0].untyped_storage().data_ptr()))
                if tensors[0].numel():
                    tensors[0].fill_(42)
            return MockWork(run)

        def recv_anysource(self, tensors: list[torch.Tensor], tag: int) -> MockWork:
            def run() -> None:
                p2p_log.append(("any", -1, tag, tensors[0].untyped_storage().data_ptr()))
                tensors[0].fill_(7)
            return MockWork(run, src=1)

    def make_stream(device: torch.device) -> DummyStream:
        s = DummyStream(device)
        streams.append(s)
        return s

    class FakePinBackend:
        def __init__(self) -> None:
            self.register_calls = 0
            self.unregister_calls = 0

        def register(self, _pointer: int, _size: int) -> bool:
            self.register_calls += 1
            return True

        def unregister(self, _pointer: int) -> None:
            self.unregister_calls += 1

    reg = FakePinBackend()
    manager = PinManager(16 * 1024, backend=reg)
    patch.setattr(communication, "host_pin_manager", manager)
    patch.setattr(communication, "_validate_tensor", lambda _t: None)
    patch.setattr(group, "size", lambda: 2)
    patch.setattr(group, "rank", lambda: 0)
    patch.setattr(group, "_staging_bytes", 256)
    patch.setattr(group, "_cpu_group", FakeP2PGloo())
    patch.setattr(torch.cuda, "Event", DummyEvent)
    patch.setattr(torch.cuda, "Stream", make_stream)
    patch.setattr(torch.cuda, "device", lambda _d: nullcontext())
    patch.setattr(torch.cuda, "stream", lambda _s: nullcontext())
    patch.setattr(torch.cuda, "current_stream", DummyStream)
    return reg, manager, recorded_events, stream_syncs, streams, p2p_log, state, entered_event


def _check_p2p_pool_cap_and_shutdown(group, send_t, state, entered_event, streams, manager, reg) -> None:
    cap_gate = threading.Event()
    state.update({"block_gate": cap_gate, "entered": 0, "target": 3})
    cap_works = [group.send([send_t], 1, tag) for tag in (3, 4, 5)]
    assert entered_event.wait(timeout=5)
    cap_gate.set()
    for work in cap_works:
        assert work.wait()
    assert len(streams) == 4 and len(group._p2p_pool) == 2

    entered_event.clear()
    shut_gate = threading.Event()
    state.update({"block_gate": shut_gate, "entered": 0, "target": 1})
    pending = group.send([send_t], 1, 6)
    assert entered_event.wait(timeout=5)
    shut = threading.Thread(target=group.shutdown)
    shut.start()
    shut_gate.set()
    shut.join(timeout=5)
    assert not shut.is_alive() and pending.wait()
    assert group._p2p_pool == []
    assert manager.stats.pinned_bytes == 0 and reg.unregister_calls == reg.register_calls


def test_p2p_cuda_mock_batch_pinning_and_shutdown(single_rank_group):
    group = single_rank_group
    _, _, fake_cuda_tensor, _, _, _ = _make_fake_gloo_copy_env(group)[:6]
    assert group._get_backend(torch.device("cpu")) is group._cpu_group
    assert group._get_backend(torch.device("cuda", 0)) is group
    assert not getattr(group._get_backend(torch.device("cuda", 0)), "supports_coalescing", False)

    with pytest.MonkeyPatch.context() as patch:
        reg, manager, recorded_events, stream_syncs, streams, p2p_log, state, entered_event = _make_mock_cuda_p2p_env(
            group, patch,
        )
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="CUDA graph capture"):
            group.send([fake_cuda_tensor(torch.ones(4))], 1, 0)
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

        # Inference-mode FakeCudaTensor across multi-chunk send + recv + anysource reuses pooled slot.
        send_t = fake_cuda_tensor(torch.arange(20, dtype=torch.float32))
        recv_t = fake_cuda_tensor(torch.zeros(20, dtype=torch.float32))
        any_t = fake_cuda_tensor(torch.zeros(20, dtype=torch.float32))
        group.send([send_t], 1, 0).wait()
        group.recv([recv_t], 1, 1).wait()
        w_any = group.recv_anysource([any_t], 2)
        assert w_any.get_future().wait() == [any_t]
        assert w_any.source_rank() == 1
        assert len(streams) == 1 and len(group._p2p_pool) == 1
        assert len({entry[3] for entry in p2p_log}) == 1
        assert manager.stats.active_leases == 0 and reg.register_calls == 1
        assert len(stream_syncs) == 3 and all(ev.synced for ev in recorded_events)
        assert torch.all(recv_t.data.view(torch.uint8) == 42)
        any_bytes = any_t.data.view(torch.uint8)
        assert torch.all(any_bytes[:64] == 7) and torch.all(any_bytes[64:] == 42)

        # Transfer error synchronizes stream and returns pooled slot; sync error discards slot instead.
        state["fail_send"] = True
        with pytest.raises(RuntimeError, match="injected p2p send failure"):
            group.send([send_t], 1, 0).wait()
        assert len(stream_syncs) == 4 and manager.stats.active_leases == 0 and len(group._p2p_pool) == 1
        state["fail_send"] = False
        state["fail_sync"] = True
        with pytest.raises(RuntimeError, match="injected stream synchronize failure"):
            group.send([send_t], 1, 0).wait()
        assert len(group._p2p_pool) == 0 and manager.stats.active_leases == 0
        assert manager.stats.pinned_bytes == 0 and reg.unregister_calls == 1
        state["fail_sync"] = False

        _check_p2p_pool_cap_and_shutdown(group, send_t, state, entered_event, streams, manager, reg)


class _FakeIpcRuntime:
    """In-memory multi-rank fake implementing IpcRuntime over CPU uint8 tensors."""

    def __init__(self, *, world_size: int = 2) -> None:
        self.lock = threading.Lock()
        self.next_ptr = 0x1000_0000
        self.allocs: dict[int, torch.Tensor] = {}
        self.handles: dict[bytes, tuple[int, int]] = {}
        self.opened: dict[int, tuple[int, int, int]] = {}
        self.events: list[tuple[int, str, int]] = []
        self.can_access_calls: list[tuple[int, int]] = []
        self.bus_ids: dict[int, str] = {r: f"0000:0{r + 1}:00.0" for r in range(world_size)}
        self.bus_to_dev: dict[str, int | None] = {f"0000:0{r + 1}:00.0": r for r in range(world_size)}
        self.can_access: dict[tuple[int, int], bool] = {
            (a, b): True for a in range(world_size) for b in range(world_size) if a != b
        }
        self.prior_error: int = 0
        self.prior_errors: dict[int, int] = {}
        self.fail_malloc: set[int] = set()
        self.fail_get_handle: set[int] = set()
        self.raise_get_handle: dict[int, BaseException] = {}
        self.fail_open: set[int] = set()
        self.raise_open: dict[int, Exception] = {}
        self.local_rank = threading.local()

    def rank(self) -> int:
        return int(getattr(self.local_rank, "rank", 0))

    def bind_rank(self, rank: int) -> None:
        self.local_rank.rank = rank

    def check_prior_error(self) -> None:
        err = self.prior_errors.get(self.rank(), self.prior_error)
        if err:
            raise RuntimeError(f"Prior CUDA/HIP runtime work failed with error {err}")

    def device_get_pci_bus_id(self, device_index: int) -> str | None:
        del device_index
        self.check_prior_error()
        return self.bus_ids.get(self.rank(), "0000:01:00.0")

    def device_get_by_pci_bus_id(self, bus_id: str) -> int | None:
        self.check_prior_error()
        return self.bus_to_dev.get(bus_id)

    def device_can_access_peer(self, device_index: int, peer_device_index: int) -> bool:
        self.check_prior_error()
        with self.lock:
            self.can_access_calls.append((device_index, peer_device_index))
        return self.can_access.get((device_index, peer_device_index), False)

    def malloc(self, nbytes: int) -> int | None:
        self.check_prior_error()
        r = self.rank()
        if r in self.fail_malloc:
            return None
        buf = torch.zeros(nbytes, dtype=torch.uint8)
        with self.lock:
            self.allocs[buf.data_ptr()] = buf
            self.events.append((r, "malloc", buf.data_ptr()))
            return buf.data_ptr()

    def free(self, ptr: int) -> None:
        r = self.rank()
        with self.lock:
            self.events.append((r, "free", ptr))
            self.allocs.pop(ptr, None)

    def ipc_get_mem_handle(self, ptr: int) -> bytes | None:
        self.check_prior_error()
        r = self.rank()
        if r in self.raise_get_handle:
            raise self.raise_get_handle[r]
        if r in self.fail_get_handle:
            return None
        # Embed NUL bytes in the 64-byte handle so NUL preservation is always exercised.
        raw = bytes([0, r + 1, 0, 0xFF]) + (ptr & 0xFFFF_FFFF_FFFF_FFFF).to_bytes(8, "little") + bytes(52)
        with self.lock:
            size = self.allocs[ptr].numel()
            self.handles[raw] = (ptr, size)
            self.events.append((r, "get_handle", ptr))
        return raw

    def ipc_open_mem_handle(self, handle: bytes) -> int | None:
        self.check_prior_error()
        r = self.rank()
        if r in self.raise_open:
            raise self.raise_open[r]
        if r in self.fail_open or len(handle) != 64:
            return None
        with self.lock:
            entry = self.handles.get(handle)
            if entry is None:
                return None
            target, size = entry
            alias = self.next_ptr
            self.next_ptr += ((size + 4095) // 4096) * 4096 + 4096
            self.opened[alias] = (r, target, size)
            self.events.append((r, "open", alias))
            return alias

    def ipc_close_mem_handle(self, ptr: int) -> None:
        r = self.rank()
        with self.lock:
            self.events.append((r, "close", ptr))
            self.opened.pop(ptr, None)

    def memcpy_async(self, dst_ptr: int, src_ptr: int, nbytes: int, stream_ptr: int) -> None:
        import ctypes

        assert stream_ptr != 0
        if nbytes == 0:
            return
        with self.lock:
            real_dst = self._resolve(dst_ptr)
            real_src = self._resolve(src_ptr)
            self.events.append((self.rank(), "memcpy", nbytes))
        ctypes.memmove(real_dst, real_src, nbytes)

    def _resolve(self, ptr: int) -> int:
        for alias, (_r, target, size) in self.opened.items():
            if alias <= ptr < alias + size:
                return target + (ptr - alias)
        return ptr


class _FakeIpcTensor:
    """CPU-backed CUDA tensor stand-in supporting _relay_shared and IpcArena slice copies."""

    def __init__(self, data: torch.Tensor, device: torch.device) -> None:
        self.data = data
        self.device = device
        self.dtype = data.dtype
        self.nbytes = data.nbytes

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types
        unwrapped = [a.data if isinstance(a, _FakeIpcTensor) else a for a in args]
        kw = {k: (v.data if isinstance(v, _FakeIpcTensor) else v) for k, v in (kwargs or {}).items()}
        return func(*unwrapped, **kw)

    def numel(self) -> int:
        return self.data.numel()

    def data_ptr(self) -> int:
        return self.data.data_ptr()

    def is_contiguous(self) -> bool:
        return self.data.is_contiguous()

    def reshape(self, *shape: int) -> _FakeIpcTensor:
        return _FakeIpcTensor(self.data.reshape(*shape), self.device)

    def view(self, dtype: torch.dtype) -> _FakeIpcTensor:
        return _FakeIpcTensor(self.data.view(dtype), self.device)

    def __getitem__(self, key: object) -> _FakeIpcTensor:
        return _FakeIpcTensor(self.data[key], self.device)  # type: ignore[index]

    def copy_(self, src: object, non_blocking: bool = False) -> _FakeIpcTensor:
        del non_blocking
        raw = src.data if isinstance(src, _FakeIpcTensor) else src
        self.data.copy_(raw)  # type: ignore[arg-type]
        return self

    def add_(self, src: object) -> _FakeIpcTensor:
        raw = src.data if isinstance(src, _FakeIpcTensor) else src
        self.data.add_(raw)  # type: ignore[arg-type]
        return self


def _run_threaded(world_size: int, worker: Callable[[int], None]) -> None:
    errors: list[BaseException] = []

    def target(rank: int) -> None:
        try:
            worker(rank)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=target, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert not any(t.is_alive() for t in threads)
    if errors:
        raise errors[0]


class _FakeExchange:
    """In-memory multi-rank byte exchange matching ProcessGroupGloo.allgather."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.cond = threading.Condition()
        self.rank_round = [0] * world_size
        self.slots: dict[int, dict[int, bytes]] = {}

    @property
    def rounds_completed(self) -> int:
        return min(self.rank_round)

    def make_exchange(self, rank: int):
        def exchange(payload: bytes, timeout: timedelta) -> list[bytes]:
            deadline = time.monotonic() + timeout.total_seconds()
            with self.cond:
                rid = self.rank_round[rank]
                self.rank_round[rank] = rid + 1
                bucket = self.slots.setdefault(rid, {})
                bucket[rank] = payload
                self.cond.notify_all()
                while len(bucket) < self.world_size:
                    rem = deadline - time.monotonic()
                    if rem <= 0 or not self.cond.wait(rem):
                        raise RuntimeError("fake ipc exchange timed out")
                return [bucket[r] for r in range(self.world_size)]

        return exchange


class _FakeCloseBarrier:
    """Bounded thread barrier simulating SharedControl close rendezvous."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.cond = threading.Condition()
        self.arrived = 0

    def wait(self, timeout: timedelta) -> None:
        deadline = time.monotonic() + timeout.total_seconds()
        with self.cond:
            self.arrived += 1
            self.cond.notify_all()
            while self.arrived < self.world_size:
                rem = deadline - time.monotonic()
                if rem <= 0 or not self.cond.wait(rem):
                    raise RuntimeError("fake ipc close barrier timed out")


class _FakeGlooGroup:
    def __init__(self, rank: int, exchange: _FakeExchange) -> None:
        self._exchange = exchange.make_exchange(rank)

    def allgather(self, gathered: list[list[torch.Tensor]], local: list[torch.Tensor]):
        payload = bytes(local[0].tolist())
        exchange = self._exchange

        class Work:
            def wait(self, timeout: timedelta = timedelta(seconds=5)) -> bool:
                replies = exchange(payload, timeout)
                for dst, raw in zip(gathered[0], replies, strict=True):
                    dst.copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
                return True

        return Work()


class _FakeIpcControl:
    """In-memory per-peer monotonic signal and agreement controller matching SharedControl."""

    def __init__(self, rank: int, ws: int, cond: threading.Condition, shared_state: dict[str, object]) -> None:
        self.rank = rank
        self.ws = ws
        self.cond = cond
        self.shared_state = shared_state
        self.epoch = 0
        self.expected_ready = [0] * ws
        self.expected_free = [0] * ws

    def agree(self, op: int, nbytes: int, kind: int, detail: int, timeout: timedelta) -> None:
        deadline = time.monotonic() + timeout.total_seconds()
        agree_slots: dict[int, dict[int, tuple[tuple[int, int, int, int], int, int]]] = (
            self.shared_state["agree"]  # type: ignore[assignment]
        )
        ready_seq: list[int] = self.shared_state["ready"]  # type: ignore[assignment]
        free_seq: list[int] = self.shared_state["free"]  # type: ignore[assignment]
        with self.cond:
            self.epoch += 1
            ep = agree_slots.setdefault(self.epoch, {})
            ep[self.rank] = ((op, nbytes, kind, detail), ready_seq[self.rank], free_seq[self.rank])
            self.cond.notify_all()
            while len(ep) < self.ws:
                rem = deadline - time.monotonic()
                if rem <= 0 or not self.cond.wait(rem):
                    raise RuntimeError("fake control agree timed out")
            if any(meta != (op, nbytes, kind, detail) for meta, _, _ in ep.values()):
                raise ValueError(
                    "shared relay collective operation, device, payload bytes and root/dtype must match across ranks",
                )
            self.expected_ready = [ep[r][1] for r in range(self.ws)]
            self.expected_free = [ep[r][2] for r in range(self.ws)]

    def signal(
        self, peers: Sequence[tuple[int | None, int | None]], tag: int, sequence: int, timeout: timedelta,
    ) -> None:
        del sequence
        deadline = time.monotonic() + timeout.total_seconds()
        seq_key = "ready" if tag == 0 else "free"
        seq: list[int] = self.shared_state[seq_key]  # type: ignore[assignment]
        expected = self.expected_ready if tag == 0 else self.expected_free
        send = any(s is not None for s, _ in peers)
        recvs = [r for _, r in peers if r is not None]
        with self.cond:
            if send:
                seq[self.rank] += 1
                self.cond.notify_all()
            targets = [(p, expected[p] + 1) for p in recvs]
            for p, tgt in targets:
                expected[p] = tgt
            while any(seq[p] < tgt for p, tgt in targets):
                rem = deadline - time.monotonic()
                if rem <= 0 or not self.cond.wait(rem):
                    raise RuntimeError("fake control signal timed out")

    def skip_ready(self, peer: int, count: int) -> None:
        self.expected_ready[peer] += count

    def abort(self) -> None:
        with self.cond:
            self.cond.notify_all()


def _make_ipc_collective_env(world_size: int, staging_bytes: int, rt: _FakeIpcRuntime, tmp_path, patch):
    from contextlib import contextmanager
    from piper_offload import _relay_ipc, _relay_shared

    cond = threading.Condition()
    ctl_state: dict[str, object] = {"agree": {}, "ready": [0] * world_size, "free": [0] * world_size}

    class FakeStream:
        def __init__(self, device: torch.device, sid: int) -> None:
            self.device = device
            self.cuda_stream = sid

        def wait_stream(self, other: object) -> None:
            del other

        def synchronize(self) -> None:
            pass

    class FakeEvent:
        def record(self, stream: object) -> None:
            del stream

        def synchronize(self) -> None:
            pass

    rank_local = threading.local()
    stream_counter = threading.local()

    orig_empty = torch.empty

    def fake_empty(*args: object, **kwargs: object):
        dev = kwargs.get("device")
        if isinstance(dev, torch.device) and dev.type == "cuda":
            cpu_kw = {k: v for k, v in kwargs.items() if k != "device"}
            return _FakeIpcTensor(orig_empty(*args, **cpu_kw), dev)  # type: ignore[call-overload]
        return orig_empty(*args, **kwargs)  # type: ignore[call-overload]

    def fake_stream(device: torch.device) -> FakeStream:
        idx = getattr(stream_counter, "idx", 0) + 1
        stream_counter.idx = idx
        return FakeStream(device, 1000 + rank_local.rank * 10 + idx)

    @contextmanager
    def fake_ctx(obj: object):
        del obj
        yield

    patch.setattr(_relay_ipc.os, "getpid", lambda: 10_000 + getattr(rank_local, "rank", 0))
    patch.setattr(_relay_ipc, "load_ipc_runtime", lambda: rt)
    patch.setattr(_relay_shared, "_load_triton_sum", lambda: None)
    patch.setattr(torch, "empty", fake_empty)
    patch.setattr(torch.cuda, "Stream", fake_stream)
    patch.setattr(torch.cuda, "Event", FakeEvent)
    patch.setattr(torch.cuda, "current_stream", lambda device: FakeStream(device, 1))
    patch.setattr(torch.cuda, "stream", fake_ctx)
    patch.setattr(torch.cuda, "device", fake_ctx)

    store = dist.FileStore(str(tmp_path / "ipc-harness"), 1)
    patch.setattr(_relay_shared, "shared_control_supported", lambda: False)

    relays: dict[int, _relay_shared.SharedRelay] = {}
    exch = _FakeExchange(world_size)

    def create_relay(r: int) -> None:
        rank_local.rank = r
        relay = _relay_shared.SharedRelay(
            store,
            _FakeGlooGroup(r, exch),  # type: ignore[arg-type]
            rank=r,
            world_size=world_size,
            nbytes=staging_bytes,
            timeout=timedelta(seconds=5),
            ipc=True,
        )
        relay._control = _FakeIpcControl(r, world_size, cond, ctl_state)
        relays[r] = relay

    _run_threaded(world_size, create_relay)
    return relays, rank_local


def _check_ipc_collectives_on_rank(
    rank: int,
    world_size: int,
    relay,
    rank_local: threading.local,
    rt: _FakeIpcRuntime,
    manager: object,
    acquire_calls_ref: list[int],
) -> None:
    rt.bind_rank(rank)
    rank_local.rank = rank
    dev = torch.device("cuda", rank)

    def wrap(t: torch.Tensor) -> _FakeIpcTensor:
        return _FakeIpcTensor(t, dev)

    # 1. broadcast & scatter
    bcast = wrap(torch.arange(64, dtype=torch.int64) if rank == 0 else torch.zeros(64, dtype=torch.int64))
    relay.broadcast(bcast, root=0, manager=manager)
    assert torch.equal(bcast.data, torch.arange(64, dtype=torch.int64))

    sc_srcs = [wrap(torch.full((16,), float(r + 10))) for r in range(world_size)]
    sc_out = wrap(torch.zeros(16))
    relay.scatter(sc_srcs, sc_out, root=0, manager=manager)
    torch.testing.assert_close(sc_out.data, torch.full((16,), float(rank + 10)))

    # 2. allgather & coalesced allgather
    ag_in = [wrap(torch.full((16,), rank + 1, dtype=torch.float32)), wrap(torch.full((8,), rank + 2.0))]
    ag_outs = [
        [wrap(torch.zeros(16)) for _ in range(world_size)],
        [wrap(torch.zeros(8)) for _ in range(world_size)],
    ]
    relay.allgather(ag_in[:1], ag_outs[:1], manager)
    relay.allgather(ag_in, ag_outs, manager)
    for r in range(world_size):
        torch.testing.assert_close(ag_outs[0][r].data, torch.full((16,), float(r + 1)))
        torch.testing.assert_close(ag_outs[1][r].data, torch.full((8,), float(r + 2)))

    # 3. allreduce: direct (< 256 KiB), RS+AG (>= 256 KiB when W >= 3), and coalesced
    rs_calls = 0
    orig_rs = relay._allreduce_reduce_scatter

    def spy_rs(*args, **kwargs):
        nonlocal rs_calls
        rs_calls += 1
        return orig_rs(*args, **kwargs)

    relay._allreduce_reduce_scatter = spy_rs
    for numel in (32, 132 * 1024):
        ar = wrap(torch.full((numel,), float(rank + 1), dtype=torch.bfloat16))
        relay.allreduce([ar], manager)
        expected_sum = float(sum(r + 1 for r in range(world_size)))
        torch.testing.assert_close(ar.data, torch.full((numel,), expected_sum, dtype=torch.bfloat16))
    assert rs_calls == (1 if world_size == 4 else 0)

    ar_c = [wrap(torch.full((12,), float(rank + 1))), wrap(torch.full((6,), float(rank + 2)))]
    relay.allreduce(ar_c, manager)
    torch.testing.assert_close(ar_c[0].data, torch.full((12,), float(sum(r + 1 for r in range(world_size)))))
    torch.testing.assert_close(ar_c[1].data, torch.full((6,), float(sum(r + 2 for r in range(world_size)))))

    # 4. reduce_scatter single & coalesced
    rs_srcs = [
        wrap(torch.full((32,), float((rank + 1) * (o + 1)), dtype=torch.float16)) for o in range(world_size)
    ]
    rs_srcs2 = [wrap(torch.full((16,), float(rank + o + 1))) for o in range(world_size)]
    rs_outs = [wrap(torch.zeros(32, dtype=torch.float16)), wrap(torch.zeros(16))]
    relay.reduce_scatter(rs_outs[:1], [rs_srcs], manager)
    rs_expected = float(sum((r + 1) * (rank + 1) for r in range(world_size)))
    torch.testing.assert_close(rs_outs[0].data, torch.full((32,), rs_expected, dtype=torch.float16))
    relay.reduce_scatter(rs_outs, [rs_srcs, rs_srcs2], manager)
    torch.testing.assert_close(
        rs_outs[1].data, torch.full((16,), float(sum(r + rank + 1 for r in range(world_size)))),
    )
    assert torch.all(relay.buffer == 0)

    # 5. alltoall (equal + unequal splits): size exchange touches host buffer while payload uses IPC.
    _check_ipc_alltoall_on_rank(rank, world_size, relay, manager, wrap)
    assert relay.ipc_active and acquire_calls_ref[0] == 0
    relay.close()


def _check_ipc_alltoall_on_rank(
    rank: int, world_size: int, relay, manager: object, wrap: Callable[[torch.Tensor], _FakeIpcTensor],
) -> None:
    eq_ins = [wrap(torch.full((8,), float(rank * 10 + dst))) for dst in range(world_size)]
    eq_outs = [wrap(torch.zeros(8)) for _ in range(world_size)]
    relay.alltoall(eq_outs, eq_ins, manager)
    for src in range(world_size):
        torch.testing.assert_close(eq_outs[src].data, torch.full((8,), float(src * 10 + rank)))

    un_ins = [
        wrap(torch.full((rank + dst + 1,), rank * 10 + dst, dtype=torch.int64)) for dst in range(world_size)
    ]
    un_outs = [wrap(torch.zeros(src + rank + 1, dtype=torch.int64)) for src in range(world_size)]
    relay.alltoall(un_outs, un_ins, manager)
    for src in range(world_size):
        assert torch.equal(un_outs[src].data, torch.full((src + rank + 1,), src * 10 + rank, dtype=torch.int64))


@pytest.mark.parametrize("world_size", [2, 4])
def test_ipc_collectives_mocked_multi_rank(tmp_path, world_size):
    from contextlib import contextmanager

    rt = _FakeIpcRuntime(world_size=world_size)
    acquire_calls = [0]

    class CountingManager:
        @contextmanager
        def acquire(self, _tensors: object):
            acquire_calls[0] += 1
            yield

    manager = CountingManager()
    with pytest.MonkeyPatch.context() as patch:
        relays, rank_local = _make_ipc_collective_env(world_size, world_size * 3 * 128 * 1024, rt, tmp_path, patch)
        _run_threaded(
            world_size,
            lambda rank: _check_ipc_collectives_on_rank(
                rank, world_size, relays[rank], rank_local, rt, manager, acquire_calls,
            ),
        )
        assert acquire_calls[0] == 0
        for r in range(world_size):
            r_events = [op for rr, op, _ in rt.events if rr == r]
            close_positions = [i for i, op in enumerate(r_events) if op == "close"]
            free_pos = r_events.index("free")
            assert len(close_positions) == world_size - 1 and max(close_positions) < free_pos
        assert not rt.allocs and not rt.opened


def _make_fake_ipc_library(
    prefix: str, state: dict[str, int], opened_handles: list[bytes], closed_ptrs: list[int], freed_ptrs: list[int],
):
    import ctypes
    from types import SimpleNamespace

    def get_last_error() -> int:
        err = state["last_error"]
        state["last_error"] = 0
        return err

    def get_bus_id(buf_ptr: int, buf_len: int, dev: int) -> int:
        del dev
        text = b"0000:AB:00.0\x00"
        ctypes.memmove(buf_ptr, text, min(buf_len, len(text)))
        return 0

    def get_by_bus_id(out_ptr: object, _bus_id: bytes) -> int:
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_int))[0] = 1
        return 0

    def can_access_peer(out_ptr: object, d0: int, d1: int) -> int:
        del d0, d1
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_int))[0] = 1
        return 0

    def malloc(out_ptr: object, _nbytes: int) -> int:
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = 0xCAFE_0000
        return 0

    def free(ptr: int) -> int:
        freed_ptrs.append(ptr)
        return 0

    def get_mem_handle(handle_ptr: object, _dev_ptr: int) -> int:
        ctypes.memmove(handle_ptr, bytes([0, 0xAA, 0, 0xBB]) * 16, 64)
        return 0

    def open_mem_handle(out_ptr: object, handle: object, flags: int) -> int:
        assert flags == 0x01
        if state["open_rc"]:
            state["last_error"] = state["open_rc"]
            return state["open_rc"]
        opened_handles.append(bytes(handle.reserved))  # type: ignore[attr-defined]
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = 0xBEEF_0000
        return 0

    def close_mem_handle(ptr: int) -> int:
        closed_ptrs.append(ptr)
        return 0

    def memcpy_async(_dst: int, _src: int, _nbytes: int, kind: int, _stream: int) -> int:
        assert kind == 4
        if state["memcpy_rc"]:
            state["last_error"] = state["memcpy_rc"]
            return state["memcpy_rc"]
        return 0

    return SimpleNamespace(**{
        f"{prefix}GetLastError": get_last_error,
        f"{prefix}DeviceGetPCIBusId": get_bus_id,
        f"{prefix}DeviceGetByPCIBusId": get_by_bus_id,
        f"{prefix}DeviceCanAccessPeer": can_access_peer,
        f"{prefix}Malloc": malloc,
        f"{prefix}Free": free,
        f"{prefix}IpcGetMemHandle": get_mem_handle,
        f"{prefix}IpcOpenMemHandle": open_mem_handle,
        f"{prefix}IpcCloseMemHandle": close_mem_handle,
        f"{prefix}MemcpyAsync": memcpy_async,
    })


@pytest.mark.parametrize("prefix", ["cuda", "hip"])
def test_ipc_runtime_handles_nul_bytes_and_errors(monkeypatch, prefix):
    from piper_offload import _relay_ipc

    state = {"last_error": 0, "open_rc": 0, "memcpy_rc": 0}
    opened_handles: list[bytes] = []
    closed_ptrs: list[int] = []
    freed_ptrs: list[int] = []
    fake_lib = _make_fake_ipc_library(prefix, state, opened_handles, closed_ptrs, freed_ptrs)
    rt = _relay_ipc.DeviceIpcRuntime(fake_lib, prefix)  # type: ignore[arg-type]
    assert rt.device_get_pci_bus_id(0) == "0000:ab:00.0"
    raw_handle = rt.ipc_get_mem_handle(0xCAFE_0000)
    assert raw_handle == bytes([0, 0xAA, 0, 0xBB]) * 16
    assert rt.ipc_open_mem_handle(raw_handle) == 0xBEEF_0000
    assert opened_handles == [raw_handle]

    state["open_rc"] = 101
    assert rt.ipc_open_mem_handle(raw_handle) is None
    assert state["last_error"] == 0
    state["open_rc"] = 0

    state["memcpy_rc"] = 202
    with pytest.raises(RuntimeError, match="memcpyAsync failed with error 202"):
        rt.memcpy_async(0xBEEF_0000, 0xCAFE_0000, 64, 1)
    assert state["last_error"] == 0
    state["memcpy_rc"] = 0

    state["last_error"] = 77
    with pytest.raises(RuntimeError, match="Prior CUDA/HIP runtime work failed with error 77"):
        rt.malloc(64)
    rt.ipc_close_mem_handle(0xBEEF_0000)
    rt.free(0xCAFE_0000)
    assert closed_ptrs == [0xBEEF_0000] and freed_ptrs == [0xCAFE_0000]

    monkeypatch.setattr(_relay_ipc.sys, "platform", "win32")
    assert _relay_ipc.load_ipc_runtime() is None


def _open_two_ipc_arenas(
    rt: _FakeIpcRuntime, pid_fn: Callable[[int], int], *, slot_bytes: int = 64,
    errors_by_rank: dict[int, BaseException] | None = None,
):
    from piper_offload import _relay_ipc

    exch = _FakeExchange(2)
    barrier = _FakeCloseBarrier(2)
    out: list[_relay_ipc.IpcArena | None] = [None, None]
    tl = threading.local()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_relay_ipc.os, "getpid", lambda: pid_fn(getattr(tl, "rank", 0)))

        def worker(rank: int) -> None:
            tl.rank = rank
            rt.bind_rank(rank)
            try:
                out[rank] = _relay_ipc.open_ipc_arena(
                    exch.make_exchange(rank),
                    barrier.wait,
                    rank,
                    2,
                    torch.device("cuda", rank),
                    slot_bytes,
                    3,
                    timedelta(seconds=2),
                    runtime=rt,
                )
            except BaseException as exc:
                if errors_by_rank is None:
                    raise
                errors_by_rank[rank] = exc

        _run_threaded(2, worker)
    return out, exch


def test_ipc_arena_fallbacks_and_close_semantics():
    # 1. Same PID fallback returns None cleanly in one exchange round.
    rt_same_pid = _FakeIpcRuntime()
    same_pid_arenas, same_pid_exch = _open_two_ipc_arenas(rt_same_pid, lambda _r: 4242)
    assert same_pid_arenas == [None, None] and same_pid_exch.rounds_completed == 1
    assert not rt_same_pid.allocs

    # 2. Same PCI bus ID across distinct PIDs skips device_can_access_peer and succeeds.
    rt_same_bus = _FakeIpcRuntime()
    rt_same_bus.bus_ids = {0: "0000:01:00.0", 1: "0000:01:00.0"}
    arenas, _ = _open_two_ipc_arenas(rt_same_bus, lambda r: 100 + r)
    assert all(a is not None for a in arenas) and rt_same_bus.can_access_calls == []
    _run_threaded(2, lambda r: (rt_same_bus.bind_rank(r), arenas[r].close()))  # type: ignore[union-attr]
    assert not rt_same_bus.allocs and not rt_same_bus.opened

    # 3. device_get_by_pci_bus_id returning None (VISIBLE_DEVICES isolation) opens handle directly.
    rt_hidden = _FakeIpcRuntime()
    rt_hidden.bus_to_dev = {"0000:01:00.0": None, "0000:02:00.0": None}
    arenas_hidden, _ = _open_two_ipc_arenas(rt_hidden, lambda r: 200 + r)
    assert all(a is not None for a in arenas_hidden) and rt_hidden.can_access_calls == []
    _run_threaded(2, lambda r: (rt_hidden.bind_rank(r), arenas_hidden[r].close()))  # type: ignore[union-attr]

    # 4. Asymmetric open failure & peer-access denial close imported handles before rollback & free.
    rt_asym = _FakeIpcRuntime()
    rt_asym.fail_open.add(1)
    assert _open_two_ipc_arenas(rt_asym, lambda r: 300 + r)[0] == [None, None]
    r0_ops = [op for rr, op, _ in rt_asym.events if rr == 0]
    assert r0_ops.index("close") < r0_ops.index("free") and not rt_asym.allocs and not rt_asym.opened

    rt_no_p2p = _FakeIpcRuntime()
    rt_no_p2p.can_access = {(0, 1): False, (1, 0): False}
    assert _open_two_ipc_arenas(rt_no_p2p, lambda r: 400 + r)[0] == [None, None]
    assert not rt_no_p2p.allocs and not rt_no_p2p.opened

    # 5. Phase 1 single-rank error propagates to both ranks immediately and frees healthy allocations.
    rt_p1_err = _FakeIpcRuntime()
    rt_p1_err.prior_errors[0] = 9
    p1_errors: dict[int, BaseException] = {}
    _open_two_ipc_arenas(rt_p1_err, lambda r: 450 + r, errors_by_rank=p1_errors)
    assert "Prior CUDA/HIP runtime work failed with error 9" in str(p1_errors[0])
    assert "failed on a peer rank" in str(p1_errors[1])
    assert not rt_p1_err.allocs
    _check_ipc_handle_export_errors()

    # 6. Phase 2 single-rank exception wakes peer, closes imported handles before free, and propagates.
    rt_p2_err = _FakeIpcRuntime()
    rt_p2_err.raise_open[0] = RuntimeError("injected open crash")
    p2_errors: dict[int, BaseException] = {}
    _open_two_ipc_arenas(rt_p2_err, lambda r: 460 + r, errors_by_rank=p2_errors)
    assert "injected open crash" in str(p2_errors[0])
    assert "failed on a peer rank" in str(p2_errors[1])
    r1_ops = [op for rr, op, _ in rt_p2_err.events if rr == 1]
    assert r1_ops.index("close") < r1_ops.index("free") and not rt_p2_err.allocs and not rt_p2_err.opened

    # 7. close(broken=True) and close-rendezvous timeout close peer handles and skip local free.
    rt_close = _FakeIpcRuntime()
    arenas_close, _ = _open_two_ipc_arenas(rt_close, lambda r: 500 + r)
    rt_close.bind_rank(0)
    arenas_close[0].close(broken=True)  # type: ignore[union-attr]
    rt_close.bind_rank(1)
    arenas_close[1].close(timeout=timedelta(milliseconds=20))  # type: ignore[union-attr]
    assert not rt_close.opened and len(rt_close.allocs) == 2
    assert not any(op == "free" for _, op, _ in rt_close.events)


def _check_ipc_handle_export_errors() -> None:
    from piper_offload import _relay_ipc

    rt_export_err = _FakeIpcRuntime()
    rt_export_err.raise_get_handle[0] = RuntimeError("injected export failure")
    export_errors: dict[int, BaseException] = {}
    _open_two_ipc_arenas(rt_export_err, lambda r: 455 + r, errors_by_rank=export_errors)
    assert "injected export failure" in str(export_errors[0])
    assert "failed on a peer rank" in str(export_errors[1])
    assert not rt_export_err.allocs

    rt_interrupt = _FakeIpcRuntime()
    rt_interrupt.raise_get_handle[0] = Interrupt("injected export interrupt")
    rt_interrupt.bind_rank(0)
    with pytest.raises(Interrupt, match="injected export interrupt"):
        _relay_ipc._prepare_local_handle(rt_interrupt, torch.device("cuda", 0), 0, 192)
    assert not rt_interrupt.allocs


def test_ipc_slot_view_validation():
    from piper_offload import _relay_ipc

    rt = _FakeIpcRuntime()
    arenas, _ = _open_two_ipc_arenas(rt, lambda r: 600 + r, slot_bytes=16)
    arena = arenas[0]
    assert arena is not None
    dev = torch.device("cuda", 0)
    stream = type("S", (), {"cuda_stream": 77})()
    local_slot = _relay_ipc.IpcSlotView(arena, 0, 0, 0, 8, stream)  # type: ignore[arg-type]
    peer_slot = _relay_ipc.IpcSlotView(arena, 1, 0, 0, 8, stream)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="unit-stride slice"):
        _ = local_slot[::2]
    with pytest.raises(ValueError, match="Cannot write to a peer IPC slot"):
        peer_slot.copy_(_FakeIpcTensor(torch.zeros(8, dtype=torch.uint8), dev))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Cannot read local IPC slot"):
        local_slot.copy_to(_FakeIpcTensor(torch.zeros(8, dtype=torch.uint8), dev))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Expected 8 bytes"):
        local_slot.copy_(_FakeIpcTensor(torch.zeros(12, dtype=torch.uint8), dev))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Expected 8 bytes"):
        local_slot.copy_(_FakeIpcTensor(torch.zeros(8, dtype=torch.float32), dev))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Expected 8 bytes"):
        peer_slot.copy_to(_FakeIpcTensor(torch.zeros(8, dtype=torch.float32), dev))  # type: ignore[arg-type]
    with pytest.raises(IndexError, match="exceeds 16"):
        arena.copy_to_slot(0, 12, _FakeIpcTensor(torch.zeros(8, dtype=torch.uint8), dev), stream)  # type: ignore[arg-type]
    assert not any(op == "memcpy" for _, op, _ in rt.events)
    rt.bind_rank(0)
    arena.close(broken=True)
    rt.bind_rank(1)
    arenas[1].close(broken=True)  # type: ignore[union-attr]


def test_ipc_device_mismatch_and_switch(tmp_path):
    from contextlib import nullcontext

    class NoPinManager:
        def acquire(self, _tensors: object):
            return nullcontext()

    rt = _FakeIpcRuntime()
    rt.bus_ids = {0: "0000:01:00.0", 1: "0000:01:00.0"}
    with pytest.MonkeyPatch.context() as patch:
        relays, rank_local = _make_ipc_collective_env(2, 4096, rt, tmp_path, patch)

        def dev_worker(rank: int) -> None:
            rank_local.rank = rank
            rt.bind_rank(rank)
            # 1. CPU collective does not initialize IPC.
            relays[rank].broadcast(torch.full((4,), float(rank + 1)), root=0, manager=NoPinManager())  # type: ignore[arg-type]
            assert not relays[rank]._ipc_checked
            # 2. Rank 0 passes cuda:0 while rank 1 passes CPU -> ValueError mismatch without breaking.
            t = _FakeIpcTensor(torch.ones(4), torch.device("cuda", 0)) if rank == 0 else torch.ones(4)
            with pytest.raises(ValueError, match="must match across ranks"):
                relays[rank].broadcast(t, root=0, manager=NoPinManager())  # type: ignore[arg-type]
            assert not relays[rank].broken and not relays[rank]._ipc_checked
            # 3. Both open IPC on cuda:0.
            c0 = _FakeIpcTensor(torch.full((4,), float(rank + 1)), torch.device("cuda", 0))
            relays[rank].broadcast(c0, root=0, manager=NoPinManager())  # type: ignore[arg-type]
            assert relays[rank].ipc_active
            # 4. Single-rank switch (rank 0 cuda:1 vs rank 1 cuda:0) raises ValueError cleanly.
            sw = _FakeIpcTensor(torch.full((4,), float(rank + 5)), torch.device("cuda", 1 if rank == 0 else 0))
            with pytest.raises(ValueError, match="must match across ranks"):
                relays[rank].broadcast(sw, root=0, manager=NoPinManager())  # type: ignore[arg-type]
            assert not relays[rank].broken
            # 5. Both ranks switching together to cuda:1 agree cleanly and fall back to host.
            c1 = _FakeIpcTensor(torch.full((4,), float(rank + 5)), torch.device("cuda", 1))
            relays[rank].broadcast(c1, root=0, manager=NoPinManager())  # type: ignore[arg-type]
            assert torch.equal(c1.data, torch.full((4,), 5.0))
            relays[rank].close()

        _run_threaded(2, dev_worker)


def _exercise_ipc_cuda_collectives(rank: int, device: torch.device, shared, registration) -> None:
    # Multi-chunk broadcast & allreduce wrap all 3 ring lanes (staging_bytes=8192 -> 1360 B/slot).
    bc = (
        torch.arange(2048, device=device, dtype=torch.float32)
        if rank == 0
        else torch.zeros(2048, device=device)
    )
    dist.broadcast(bc, src=0)
    torch.testing.assert_close(bc, torch.arange(2048, device=device, dtype=torch.float32))

    if sys.platform == "linux" and (torch.cuda.device_count() == 1 or torch.cuda.can_device_access_peer(0, 1)):
        assert shared.ipc_active
    if shared.ipc_active:
        assert registration.register_calls == 0

    reduced_large = torch.full((2048,), float(rank + 1), device=device, dtype=torch.float32)
    dist.all_reduce(reduced_large)
    torch.testing.assert_close(reduced_large, torch.full((2048,), 3.0, device=device))

    local = torch.full((64,), float(rank + 1), device=device, dtype=torch.bfloat16)
    gathered = torch.empty((2, 64), device=device, dtype=torch.bfloat16)
    dist.all_gather_single(gathered, local)
    torch.testing.assert_close(
        gathered, torch.tensor([[1.0] * 64, [2.0] * 64], device=device, dtype=torch.bfloat16),
    )

    rs_in = torch.arange(64, device=device, dtype=torch.float32) + rank * 10
    rs_out = torch.empty(32, device=device, dtype=torch.float32)
    dist.reduce_scatter_tensor(rs_out, rs_in)
    expected_rs = (torch.arange(32, device=device, dtype=torch.float32) + rank * 32) * 2 + 10
    torch.testing.assert_close(rs_out, expected_rs)

    a2a_in = torch.full((2, 8), float(rank + 1), device=device)
    a2a_out = torch.empty((2, 8), device=device)
    dist.all_to_all_single(a2a_out, a2a_in)
    torch.testing.assert_close(a2a_out[0], torch.full((8,), 1.0, device=device))
    torch.testing.assert_close(a2a_out[1], torch.full((8,), 2.0, device=device))

    if shared.ipc_active:
        assert registration.register_calls == 0


def _run_ipc_collectives_cuda(rank: int, path: str, tcp_port: int | None) -> None:
    import logging
    from piper_offload import _relay_ipc

    torch.set_num_threads(1)
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)
    torch.cuda.set_device(device)
    register_relay_backend()
    registration = _CountingRegistration()
    manager = PinManager(16 * 1024 * 1024, backend=registration)
    pg_kwargs: dict[str, object] = (
        {"init_method": f"tcp://127.0.0.1:{tcp_port}"}
        if tcp_port is not None
        else {"store": dist.FileStore(path, 2)}
    )
    dist.init_process_group(
        "piper_relay",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
        pg_options=RelayOptions(transport="ipc", staging_bytes=8192),
        **pg_kwargs,  # type: ignore[arg-type]
    )
    shared = dist.group.WORLD._shared
    warnings: list[str] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            warnings.append(record.getMessage())

    handler = CaptureHandler(level=logging.WARNING)
    _relay_ipc.logger.addHandler(handler)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(communication, "host_pin_manager", manager)
        try:
            _exercise_ipc_cuda_collectives(rank, device, shared, registration)
            if tcp_port is not None and rank == 1:
                time.sleep(0.05)  # Force rank 0 TCPStore teardown before rank 1 destroys group.
        finally:
            try:
                dist.destroy_process_group()
            finally:
                _relay_ipc.logger.removeHandler(handler)
            manager.clear()
    assert not any("Skipping CUDA/HIP IPC buffer free" in msg for msg in warnings)


@CUDA
def test_ipc_transport_collectives_cuda(tmp_path):
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        tcp_port = sock.getsockname()[1]
    mp.spawn(
        _run_cases,
        args=(_run_ipc_collectives_cuda, [(str(tmp_path / "ipc-cuda"), None), ("", tcp_port)]),
        nprocs=2,
    )
