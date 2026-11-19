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
from piper_offload.communication import RelayOptions, register_relay_backend
from piper_offload._host_registration import RuntimeHostRegistration
from piper_offload.pin_manager import PinManager, host_pin_manager

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
    # Each collective starts after the producer's earlier work, and its
    # outputs are ready for any stream once wait() returns.
    for pending in (work, gather_work, broadcast_work, scatter_work, rs_work):
        assert pending.wait(WAIT)
        assert pending.is_completed()
    assert producer.query()
    with torch.cuda.stream(consumer):
        result = x + 1
        rs_checked = rs_out + 1
    consumer.synchronize()
    expected = torch.arange(x.numel(), dtype=torch.float32) * 0.5 + 1
    rs_expected = rs_in[dist.get_rank()].cpu().mul(2).add(1)
    torch.testing.assert_close(result.cpu(), expected)
    torch.testing.assert_close(rs_checked.cpu(), rs_expected)
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

    class ObservingTransport:
        def allreduce(self, tensors, options):
            del tensors, options
            seen.append(_transfer_priority._foreground.count)
            return self

        def wait(self):
            pass

    monkeypatch.setattr(single_rank_group, "_cpu_group", ObservingTransport())
    dist.all_reduce(torch.ones(256, device="cuda"))
    assert seen == [1]
    assert _transfer_priority._foreground.count == 0


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
    def __init__(self, inner, staging_bytes, accelerator):
        self.inner = inner
        self.staging_bytes = staging_bytes
        self.accelerator = accelerator
        self.calls = dict.fromkeys(("allreduce", "allgather", "broadcast", "scatter", "reduce_scatter", "reduce"), 0)
        self.storage_pointers = set()

    def __getattr__(self, name):
        def call(*args):
            self.calls[name] += 1
            pending = list(args)
            while pending:
                item = pending.pop()
                if isinstance(item, list):
                    pending.extend(item)
                elif isinstance(item, torch.Tensor):
                    assert item.device.type == "cpu"
                    assert item.nbytes <= self.staging_bytes
                    if self.accelerator:
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
    measured = _MeasuredTransport(group._cpu_group, 512, device_type == "cuda")
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

            assert all(count > 3 for count in measured.calls.values())
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
    (32, 1, "shared"), (128, 3, "shared"),
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


@pytest.mark.parametrize(("size", "buffers"), [(80, 1), (144, 2)])
def test_reject_insufficient_shared_rank_slots(size, buffers):
    with pytest.raises(ValueError, match="16 bytes per staging slot in each rank region"):
        RelayOptions(staging_bytes=size, pipeline_buffers=buffers, transport="shared")._validate_group_size(2)


def test_accept_minimum_shared_rank_slots():
    RelayOptions(staging_bytes=48, pipeline_buffers=1, transport="shared")._validate_group_size(1)
    RelayOptions(staging_bytes=96, pipeline_buffers=1, transport="shared")._validate_group_size(2)


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
    else:
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
        if not self.allow_control:
            pytest.fail("shared relay sent control or payload over Gloo")
        assert tensors[0].nbytes == 8
        return self.inner.send(tensors, peer, tag)

    def recv(self, tensors, peer, tag):
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
