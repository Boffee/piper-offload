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
from piper_offload.communication import RelayOptions, register_relay_backend
from piper_offload._host_registration import RuntimeHostRegistration
from piper_offload.host_memory import HostMemoryManager

pytestmark = pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo required")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


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
    memory_manager = HostMemoryManager()
    register_relay_backend()
    register_relay_backend()
    dist.init_process_group(
        "piper_relay",
        store=dist.FileStore(store_path, 2),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
        pg_options=RelayOptions(pipeline_buffers=pipeline_buffers, transport=transport, memory_manager=memory_manager),
    )
    try:
        assert dist.get_backend() == "piper_relay"
        mesh = DeviceMesh.from_group(dist.group.WORLD, device.type)

        # Repeat with registration disabled and enabled. Both paths must
        # deliver completed GPU results and release their transient pin leases.
        for budget in (0, 16 * 1024 * 1024):
            memory_manager.max_pinned_bytes = budget
            for dtype in DTYPES:
                _check_reduction(mesh, rank, device, dtype)
                _check_copy_collectives(rank, device, dtype)
                _check_dtensor_initialization(mesh, rank, device, dtype)
                assert memory_manager.stats.active_leases == 0

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
        memory_manager.clear()
    assert memory_manager.stats.active_leases == 0
    assert memory_manager.stats.pinned_bytes == 0


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


@pytest.mark.parametrize("transport", ["gloo", "shared"])
def test_two_rank_cpu_dtensor(tmp_path, transport):
    mp.spawn(_run_relay, args=(str(tmp_path / "cpu-store"), None, 1, transport), nprocs=2)


@CUDA
@pytest.mark.parametrize(("transport", "buffers"), [("gloo", 1), ("gloo", 3), ("shared", 1)])
def test_two_rank_dtensor_on_one_gpu(tmp_path, buffers, transport):
    mp.spawn(_run_relay, args=(str(tmp_path / "shared-gpu-store"), (0, 0), buffers, transport), nprocs=2)


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
@pytest.mark.parametrize("transport", ["gloo", "shared"])
def test_compiled_dtensor_collectives_on_one_gpu(tmp_path, transport):
    mp.spawn(_run_compiled_relay, args=(str(tmp_path / "compiled-store"), transport), nprocs=2)


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
    manager = HostMemoryManager(1024 * 1024)
    monkeypatch.setattr(single_rank_group, "_memory_manager", manager)
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
        self.calls = dict.fromkeys(("allreduce", "allgather", "broadcast", "scatter"), 0)
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
    manager = HostMemoryManager(16 * 1024 if initial_registrations else 0, backend=registration)
    measured = _MeasuredTransport(group._cpu_group, 512, device_type == "cuda")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(group, "_memory_manager", manager)
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

    # Copy collectives preserve NaN payloads, signed zero and subnormals even
    # when raw bits cross many chunk boundaries.
    bits = torch.tensor([0, -32768, 1, 0x7E01, 0x7FC1], dtype=torch.int16).repeat(53)
    local = bits.roll(rank).view(torch.float16).to(device)
    gathered = torch.empty(local.numel() * world_size, device=device, dtype=torch.float16)
    dist.all_gather_single(gathered, local)
    assert torch.equal(gathered.view(torch.int16).cpu(), torch.cat([bits.roll(r) for r in range(world_size)]))


@pytest.mark.parametrize("buffers", [1, 3])
def test_chunked_relay_on_three_cpu_ranks(tmp_path, buffers):
    mp.spawn(_run_chunked_relay, args=(str(tmp_path / "chunked-cpu"), "cpu", 3, buffers), nprocs=3)


@CUDA
@pytest.mark.parametrize("buffers", [1, 2, 3])
def test_chunked_relay_reuses_gpu_staging(tmp_path, buffers):
    mp.spawn(_run_chunked_relay, args=(str(tmp_path / "chunked-cuda"), "cuda", 2, buffers), nprocs=2)


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


@pytest.mark.parametrize("field", ["staging_bytes", "pipeline_buffers", "transport"])
def test_reject_mismatched_rank_staging_sizes(tmp_path, field):
    mp.spawn(_run_mismatched_options, args=(str(tmp_path / "mismatch"), field), nprocs=2)


@pytest.mark.parametrize("size", [0, -16, 17, True, 64.0])
def test_reject_invalid_staging_size(size):
    with pytest.raises(ValueError, match="positive integer multiple of 16"):
        RelayOptions(staging_bytes=size)


@pytest.mark.parametrize(("size", "buffers", "transport"), [
    (16, 1, "gloo"), (32, 1, "gloo"), (80, 2, "gloo"), (128, 3, "gloo"),
    (48, 1, "shared"), (144, 3, "shared"),
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
    manager = HostMemoryManager(16 * 1024, backend=registration)
    monkeypatch.setattr(single_rank_group, "_memory_manager", manager)
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


class _ControlOnly:
    def __init__(self, inner):
        self.inner = inner
        self.messages = 0

    def allreduce(self, *args):
        pytest.fail("shared GPU reduction forwarded its payload to CPU Gloo")

    def allgather(self, outputs, inputs):
        assert inputs[0].dtype == torch.int64
        assert inputs[0].nbytes == 32
        self.messages += 1
        return self.inner.allgather(outputs, inputs)

    def send(self, tensors, peer, tag):
        assert tensors[0].nbytes == 8
        self.messages += 1
        return self.inner.send(tensors, peer, tag)

    def recv(self, tensors, peer, tag):
        assert tensors[0].nbytes == 8
        self.messages += 1
        return self.inner.recv(tensors, peer, tag)


def _run_shared_chunks(rank, path, device_type, size, mixed_pinning, pipeline_buffers, staging_bytes):  # noqa: PLR0915
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
    assert shared.buffer_count == 2
    control = _ControlOnly(shared.group)
    shared.group = control
    registration = _CountingRegistration()
    pin_budget = max(16384, staging_bytes * 2)
    manager = HostMemoryManager(0 if mixed_pinning and rank == 0 else pin_budget, backend=registration)
    initial = int(device_type == "cuda" and manager.max_pinned_bytes != 0)
    pointer = shared.buffer.data_ptr()
    counts = [0, 0]
    lanes = set()
    original_slot = shared._slot

    def slot(sender, lane, width):
        lanes.add(lane)
        counts[int(sender != rank)] += width
        view = original_slot(sender, lane, width)
        assert view.untyped_storage().data_ptr() == pointer
        assert view.untyped_storage().nbytes() == staging_bytes
        return view

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(group, "_memory_manager", manager)
        patch.setattr(shared, "_slot", slot)
        if device_type == "cuda":
            patch.setattr(group, "_cpu_group", control)
        # Exercise slot reuse and a partial last chunk at either alignment.
        local = torch.arange(shared.capacity + 1, dtype=torch.float32, device=device) + rank
        output = torch.empty((size, local.numel()), device=device)
        dist.all_gather_single(output, local)
        assert lanes == {0, 1}
        assert counts == [(size - 1) * local.nbytes] * 2
        for peer in range(size):
            torch.testing.assert_close(output[peer], local - rank + peer)
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
        assert control.messages > 20
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
    for dtype, contributions in values.items():
        count = capacity * 3 // dtype.itemsize + 1
        backing = torch.full((count + 2,), contributions[rank], dtype=dtype, device=device)
        local = backing[1:-1]
        expected = torch.tensor(contributions[:size], dtype=dtype).float()
        total = expected[0]
        for value in expected[1:]:
            total = total + value
        dist.all_reduce(local)
        torch.testing.assert_close(local, torch.full_like(local, total.to(dtype).item()), rtol=0, atol=0)
        torch.testing.assert_close(backing[[0, -1]], torch.full_like(backing[:2], contributions[rank]))


@pytest.mark.parametrize(("size", "staging_bytes"), [(2, 1024), (3, 1024), (3, 65536)])
@pytest.mark.parametrize("pipeline_buffers", [1, 3])
def test_shared_chunks_on_cpu(tmp_path, size, pipeline_buffers, staging_bytes):
    mp.spawn(_run_shared_chunks,
             args=(str(tmp_path / "shared-cpu"), "cpu", size, False, pipeline_buffers, staging_bytes), nprocs=size)


@CUDA
@pytest.mark.parametrize(("size", "mixed_pinning", "staging_bytes", "pipeline_buffers"), [
    (2, False, 1024, 1), (2, True, 1024, 1), (3, False, 1024, 1), (3, True, 65536, 1),
    # A nondefault Gloo pipeline option still uses two shared outgoing slots.
    (2, False, 1024, 3),
])
def test_shared_chunks_on_gpu(tmp_path, size, mixed_pinning, pipeline_buffers, staging_bytes):
    mp.spawn(
        _run_shared_chunks,
        args=(str(tmp_path / "shared-gpu"), "cuda", size, mixed_pinning, pipeline_buffers, staging_bytes), nprocs=size,
    )


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
    manager = HostMemoryManager(16384, backend=registration)
    signal = shared._signal
    unregister = registration.unregister

    def fail_release(send, receive, tag, sequence):
        if tag == 0 and sequence == 0:
            with torch.cuda.stream(shared.upload):
                torch.cuda._sleep(5_000_000)
        if tag == 1:
            manager.max_pinned_bytes = 0
            assert manager.stats.active_leases == 1
            assert registration.unregister_calls == 0
            raise RuntimeError("injected shared acknowledgement failure")
        signal(send, receive, tag, sequence)

    def check_unregister(pointer):
        assert shared.download.query()
        assert shared.upload.query()
        unregister(pointer)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(group, "_memory_manager", manager)
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
@pytest.mark.parametrize("operation", ["allgather", "allreduce"])
def test_shared_failure_drains_copies_before_unpinning(tmp_path, operation):
    mp.spawn(_run_shared_failure, args=(str(tmp_path / "shared-failure"), operation), nprocs=2)


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
    manager = HostMemoryManager(backend=backend)
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


@pytest.mark.parametrize("failure", ["host", "mapping"])
def test_shared_initialization_fails_on_all_ranks(tmp_path, failure):
    mp.spawn(_run_shared_bad_mapping, args=(str(tmp_path / "bad-map"), failure), nprocs=2)


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

    def fail_sender(send, receive, tag, sequence):
        if rank == 0 and tag == 0 and sequence == 1:
            raise RuntimeError("injected sender failure")
        signal(send, receive, tag, sequence)

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
    dist.destroy_process_group()


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_shared_metadata_mismatch_fails_before_writes(tmp_path, device_type):
    mp.spawn(_run_shared_metadata_mismatch, args=(str(tmp_path / "bad-metadata"), device_type), nprocs=2)
