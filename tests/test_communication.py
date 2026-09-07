"""Exercise the registered relay through c10d and real DTensor operations."""

import gc
import math
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor

from piper_offload import communication
from piper_offload.communication import register_relay_backend
from piper_offload.pin_manager import PinManager, host_pin_manager

pytestmark = pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo required")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _run_relay(rank: int, store_path: str, device_indices: tuple[int, ...] | None) -> None:
    torch.set_num_threads(1)
    if device_indices is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", device_indices[rank])
        torch.cuda.set_device(device)
    register_relay_backend()
    register_relay_backend()
    dist.init_process_group(
        "piper_relay",
        store=dist.FileStore(store_path, 2),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        assert dist.get_backend() == "piper_relay"
        mesh = DeviceMesh.from_group(dist.group.WORLD, device.type)

        # Repeat with registration disabled and enabled. Both paths must
        # deliver completed GPU results and release their transient pin leases.
        for budget in (0, 4 * 1024 * 1024):
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
    assert host_pin_manager.stats.active_leases == 0
    assert host_pin_manager.stats.pinned_bytes == 0


def _check_reduction(mesh, rank, device, dtype):
    for shape in ((), (0,), (3, 7), (257,)):
        x = torch.full(shape, float(rank + 1), device=device, dtype=dtype)
        work = dist.all_reduce(x, async_op=True)
        assert work.is_completed()
        assert work.wait(timedelta(seconds=1))
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


def _check_copy_collectives(rank, device, dtype):
    for shape in ((), (0,), (2, 3)):
        payload = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).to(dtype=dtype, device=device)
        source = payload.clone() if rank == 1 else torch.full_like(payload, -1)
        work = dist.broadcast(source, src=1, async_op=True)
        assert work.is_completed()
        assert work.wait(timedelta(seconds=1))
        torch.testing.assert_close(source, payload, rtol=0, atol=0)

        received = torch.empty_like(payload)
        scatter_list = [payload, payload + 1] if rank == 1 else None
        work = dist.scatter(received, scatter_list, src=1, async_op=True)
        assert work.is_completed()
        torch.testing.assert_close(received, payload + rank, rtol=0, atol=0)

        local = payload + rank
        gathered = [torch.empty_like(local) for _ in range(2)]
        work = dist.all_gather(gathered, local, async_op=True)
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
    # The blocking backend promises completion even before work.wait().
    assert work.is_completed()
    assert gather_work.is_completed()
    assert broadcast_work.is_completed()
    assert scatter_work.is_completed()
    assert producer.query()
    with torch.cuda.stream(consumer):
        result = x + 1
    consumer.synchronize()
    expected = torch.arange(x.numel(), dtype=torch.float32) * 0.5 + 1
    torch.testing.assert_close(result.cpu(), expected)
    torch.testing.assert_close(gathered.cpu(), torch.cat([expected - 1, expected - 1]))
    torch.testing.assert_close(broadcast, torch.ones_like(broadcast))
    torch.testing.assert_close(scattered, torch.full_like(scattered, dist.get_rank() + 1))


def test_two_rank_cpu_dtensor(tmp_path):
    mp.spawn(_run_relay, args=(str(tmp_path / "cpu-store"), None), nprocs=2)


@CUDA
def test_two_rank_dtensor_on_one_gpu(tmp_path):
    mp.spawn(_run_relay, args=(str(tmp_path / "shared-gpu-store"), (0, 0)), nprocs=2)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two physical CUDA/HIP GPUs required")
def test_two_rank_dtensor_on_two_gpus(tmp_path):
    mp.spawn(_run_relay, args=(str(tmp_path / "two-gpu-store"), (0, 1)), nprocs=2)


def _compiled_forward(sharded, partial):
    gathered = sharded.redistribute(placements=[Replicate()]).to_local()
    reduced = partial.redistribute(placements=[Replicate()]).to_local()
    return (gathered + reduced).sin()


def _run_compiled_relay(rank, store_path):
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay", store=dist.FileStore(store_path, 2), rank=rank,
        world_size=2, timeout=timedelta(seconds=60),
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
    mp.spawn(_run_compiled_relay, args=(str(tmp_path / "compiled-store"),), nprocs=2)


@pytest.fixture
def single_rank_group(tmp_path):
    register_relay_backend()
    dist.init_process_group(
        "piper_relay",
        store=dist.FileStore(str(tmp_path / "validation-store"), 1),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=5),
    )
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


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
