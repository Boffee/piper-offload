"""Single-process DTensor scheduling and direct GPU collectives."""

import copy
import sys
import threading
import weakref
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import _functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from piper_offload import BlockCompileConfig, ModelOffloader
from piper_offload.host_module import HostModuleStore
from piper_offload.host_param import HostParam
from piper_offload.sequential import SequentialExecutor
from tests.conftest import activated_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def executor():
    pytest.importorskip("triton")
    with SequentialExecutor(timeout=timedelta(seconds=120)) as runtime:
        yield runtime
    assert not dist.is_initialized()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sum_reuses_inputs_without_device_workspace(executor, dtype):
    generator = torch.Generator().manual_seed(923)
    inputs = [torch.randn(65539, generator=generator).to(device="cuda", dtype=dtype) for _ in range(2)]
    expected = (inputs[0].float() + inputs[1].float()).to(dtype).cpu()
    work = [value.clone() for value in inputs]

    def forward(rank):
        work[rank].copy_(inputs[rank])
        result = dist.all_reduce(work[rank], async_op=True)
        result.wait()
        return work[rank]

    executor.run(forward)
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pointers = [value.data_ptr() for value in work]
    for _ in range(4):
        for rank, value in enumerate(executor.run(forward)):
            torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)
            assert value.data_ptr() == pointers[rank]
    # No tensor-sized SUM scratch, including low-precision accumulation.
    assert torch.cuda.max_memory_allocated() == baseline


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_copy_collectives_and_coalesced_sum(executor, device):
    def forward(rank):
        x = torch.full((5,), rank + 1.0, device=device)
        pg = dist.distributed_c10d._get_default_group()
        values = [x, torch.tensor(float(rank), device=device), torch.empty(0, device=device)]
        work = pg.allreduce_coalesced(values, dist.AllreduceCoalescedOptions())
        payload = work.get_future().wait()
        assert all(a is b for a, b in zip(payload, values, strict=True))
        torch.testing.assert_close(x, torch.full_like(x, 3))
        assert values[1].item() == 1
        for root in (0, 1):
            y = torch.full((5,), rank, dtype=torch.int64, device=device)
            dist.broadcast(y, root)
            torch.testing.assert_close(y, torch.full_like(y, root))
            sources = [torch.full_like(y, 7), torch.full_like(y, 9)] if rank == root else None
            dist.scatter(y, scatter_list=sources, src=root)
            torch.testing.assert_close(y, torch.full_like(y, 7 + 2 * rank))
            gathered = [torch.empty_like(y), torch.empty_like(y)]
            work = dist.all_gather(gathered, y, async_op=True)
            payload = work.get_future().wait()
            assert len(payload) == 1 and len(payload[0]) == 2
            assert all(a is b for a, b in zip(payload[0], gathered, strict=True))
            for peer, value in enumerate(gathered):
                torch.testing.assert_close(value, torch.full_like(y, 7 + 2 * peer))
            output = torch.empty(10, dtype=y.dtype, device=device)
            dist.all_gather_single(output, y)
            torch.testing.assert_close(output, torch.cat(gathered))
        gathered = funcol.all_gather_single_coalesced([x, values[1].reshape(1)], pg)
        torch.testing.assert_close(funcol.wait_tensor(gathered[0]), torch.full((10,), 3.0, device=device))
        torch.testing.assert_close(funcol.wait_tensor(gathered[1]), torch.ones(2, device=device))
        inputs = [torch.full((size,), float(rank + size), device=device) for size in (2, 3)]
        outputs = [[torch.empty_like(value) for _ in range(2)] for value in inputs]
        payload = pg.allgather(outputs, inputs).get_future().wait()
        assert len(payload) == 2
        for size, group in zip((2, 3), payload, strict=True):
            assert len(group) == 2
            for peer, value in enumerate(group):
                torch.testing.assert_close(value, torch.full((size,), float(peer + size), device=device))
        dist.barrier()

    executor.run(forward)


@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_functional_all_gather_out_reuses_output(executor, sequential, device):
    def forward(rank):
        value = torch.full((2,), rank + 1.0, device=device)
        output = torch.empty(4, device=device)
        pg = dist.distributed_c10d._get_default_group()
        result = torch.ops._c10d_functional.all_gather_into_tensor_out(value, 2, pg.group_name, out=output)
        result = torch.ops._c10d_functional.wait_tensor(result)
        assert result.data_ptr() == output.data_ptr()
        return result

    for output in executor.run(forward, sequential=sequential):
        torch.testing.assert_close(output.cpu(), torch.tensor([1.0, 1.0, 2.0, 2.0]))
    assert executor.run(lambda rank: torch._C._distributed_c10d._get_work_registry_size()) == (0, 0)


@pytest.mark.parametrize("size", [0, 1, 5, 8])
def test_ordinary_dtensor_initialization_and_redistribution(executor, size):
    def forward(rank):
        mesh = DeviceMesh("cuda", [0, 1])
        assert dist.get_rank() == rank
        assert tuple(mesh.get_coordinate()) == (rank,)
        original = torch.arange(size * 3, device="cuda", dtype=torch.float32).view(size, 3)
        for placement in (Replicate(), Shard(0), Shard(1)):
            value = distribute_tensor(original, mesh, [placement])
            torch.testing.assert_close(funcol.wait_tensor(value.full_tensor()), original)
        partial = DTensor.from_local(torch.full((3,), rank + 1.0, device="cuda"), mesh, [Partial()])
        torch.testing.assert_close(
            partial.redistribute(placements=[Replicate()]).to_local(), torch.full((3,), 3.0, device="cuda")
        )

    executor.run(forward)


def test_alternates_whole_regions_and_preserves_worker_identity(executor):
    events = []
    states = executor.run(
        lambda rank: (rank, threading.get_ident(), torch.cuda.current_stream().cuda_stream), sequential=False
    )
    assert states[0][1] != states[1][1]
    assert states[0][2] == states[1][2]

    def forward(rank):
        assert threading.get_ident() == states[rank][1]
        for region in range(3):
            events.append((rank, region))
            value = torch.tensor([float(rank)], device="cuda")
            dist.all_reduce(value)
        events.append((rank, "done"))

    executor.run(forward)
    assert events == [(r, i) for i in range(3) for r in range(2)] + [(0, "done"), (1, "done")]


class _FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(32, 64, bias=False)
        self.up = nn.Linear(32, 64, bias=False)
        self.down = nn.Linear(64, 32, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_FFN(), _FFN()])
        self.requires_grad_(False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _shard(model, mesh):
    for block in model.blocks:
        parallelize_module(
            block,
            mesh,
            {
                "gate": ColwiseParallel(use_local_output=False),
                "up": ColwiseParallel(use_local_output=False),
                "down": RowwiseParallel(use_local_output=False),
            },
        )
    return model


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
def test_model_forward(executor, compiled, streamed):
    torch.manual_seed(31)
    model = _Model().cuda()
    x = torch.randn(17, 32, device="cuda")
    expected = model(x)
    models = [copy.deepcopy(model), copy.deepcopy(model)]

    def setup(rank):
        mesh = DeviceMesh("cuda", [0, 1])
        module = _shard(models[rank], mesh)
        options = {"triton.cudagraphs": False, "compile_threads": 1}
        offloader = None
        if streamed:
            config = BlockCompileConfig(fullgraph=True, options=options) if compiled else None
            offloader = ModelOffloader.from_module(
                module,
                block_paths=["blocks"],
                block_mode="streaming",
                block_compile=config,
            )
            assert all(p.device_mesh.device_type == "cpu" for p in module.parameters())
        elif compiled:
            module = torch.compile(module, fullgraph=True, options=options)
        return module, offloader, DTensor.from_local(x.clone(), mesh, [Replicate()])

    states = executor.run(setup, sequential=False)

    def forward(rank):
        module, offloader, value = states[rank]
        with activated_model(offloader, "cuda") if offloader else nullcontext(module) as active:
            output = active(value).to_local()
        if offloader:
            assert all(p.device_mesh.device_type == "cpu" for p in module.parameters())
        return output

    executor.run(forward, sequential=False)  # Warm compilation before staggering rank progress.
    for _ in range(3):
        for actual in executor.run(forward):
            assert type(actual) is torch.Tensor
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
    assert executor.run(lambda rank: torch._C._distributed_c10d._get_work_registry_size()) == (0, 0)


def test_mmap_shard0_projection_retains_full_host_mapping(executor, tmp_path: Path):
    path = tmp_path / "weight.bin"
    path.touch()
    mapped = torch.from_file(str(path), shared=True, size=16 * 8, dtype=torch.float32)
    mapped.copy_(torch.arange(mapped.numel(), dtype=torch.float32))
    full = mapped.view(16, 8)
    source = HostParam(nn.Parameter(full, requires_grad=False))
    source_storage = source.storage_tensors()[0].untyped_storage()

    def setup(rank):
        mesh = DeviceMesh("cuda", [0, 1])
        target_local = torch.empty((8, 8), device="meta")
        target_dtensor = DTensor.from_local(
            target_local,
            mesh,
            [Shard(0)],
            run_check=False,
            shape=full.shape,
            stride=full.stride(),
        )
        module = nn.Module()
        module.weight = nn.Parameter(target_dtensor, requires_grad=False)
        projected = HostParam.project_dtensor(source, module.weight)
        instance = HostModuleStore(params={"weight": projected}, buffers={}).bind(module)

        host_local = module.weight.to_local()
        assert module.weight.device_mesh.device_type == "cpu"
        assert module.weight.placements == (Shard(0),)
        assert tuple(module.weight.shape) == tuple(full.shape)
        assert host_local.untyped_storage().data_ptr() == source_storage.data_ptr()
        assert host_local.untyped_storage().nbytes() == source_storage.nbytes()
        assert host_local.storage_offset() == rank * host_local.numel()
        assert projected.storage_tensors()[0].data_ptr() == host_local.data_ptr()
        assert projected.cache_bytes == host_local.nbytes
        return {"module": module, "instance": instance, "target": None}

    states = executor.run(setup, sequential=False)

    def activate(rank):
        state = states[rank]
        plan = state["instance"].resolve_load_plan()
        target = plan.allocate_target(torch.device("cuda"))
        plan.load_to_target(target)
        state["target"] = target
        weight = state["module"].weight
        assert weight.device_mesh.device_type == "cuda"
        assert weight.placements == (Shard(0),)
        assert state["instance"].params[
            "weight"
        ].target_layout == HostParam.target_layout_for(weight)
        torch.testing.assert_close(weight.to_local().cpu(), full[rank * 8 : (rank + 1) * 8])

    executor.run(activate, sequential=False)
    for gathered in executor.run(lambda rank: states[rank]["module"].weight.full_tensor()):
        torch.testing.assert_close(gathered.cpu(), full)

    def deactivate(rank):
        state = states[rank]
        state["instance"].install_host()
        state["target"] = None
        local = state["module"].weight.to_local()
        assert local.device.type == "cpu"
        assert local.untyped_storage().data_ptr() == source_storage.data_ptr()

    executor.run(deactivate, sequential=False)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_compiled_linear_slices_replicated_input_for_each_rank(executor, backend):
    torch.compiler.reset()
    x = torch.arange(7 * 64, device="cuda", dtype=torch.float32).view(7, 64) / 100
    weights = [torch.randn(16, 32, device="cuda") for _ in range(2)]

    def setup(rank):
        mesh = DeviceMesh("cuda", [0, 1])
        model = nn.Linear(64, 16, bias=False, device="cuda").requires_grad_(False)
        model.weight = nn.Parameter(DTensor.from_local(weights[rank], mesh, [Shard(1)]), False)
        options = {"triton.cudagraphs": False, "compile_threads": 1} if backend == "inductor" else None
        return torch.compile(model, backend=backend, fullgraph=True, options=options), mesh

    states = executor.run(setup, sequential=False)
    for step, sequential in enumerate((False, True, True, False)):
        value = x + step

        def forward(rank, value=value):
            model, mesh = states[rank]
            # DTensor inserts the rank-specific slice inside the compiled graph.
            replicated = DTensor.from_local(value, mesh, [Replicate()])
            return model(replicated).to_local()

        outputs = executor.run(forward, sequential=sequential)
        for rank, actual in enumerate(outputs):
            expected = torch.nn.functional.linear(value.chunk(2, dim=1)[rank], weights[rank])
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "failure",
    [
        "callback",
        "count",
        "kind",
        "shape",
        "dtype",
        "reduction",
        "stream",
        "root",
        "layout",
        "gather_out_shape",
        "gather_lists",
    ],
)
def test_rank_errors_release_peer_and_poison_executor(executor, failure):
    def forward(rank):
        if rank == 0 and failure == "callback":
            raise ValueError("test callback failed")
        if rank == 0 and failure == "count":
            return
        dtype = torch.float16 if rank == 1 and failure == "dtype" else torch.float32
        x = torch.ones(3 if rank == 1 and failure == "shape" else 2, device="cuda", dtype=dtype)
        if rank == 1 and failure == "kind":
            dist.broadcast(x, 0)
        elif failure == "reduction":
            dist.all_reduce(x, dist.ReduceOp.MAX)
        elif failure == "stream":
            with torch.cuda.stream(torch.cuda.Stream()):
                dist.all_reduce(x)
        elif failure == "root":
            dist.broadcast(x, rank)
        elif failure == "layout":
            dist.all_reduce(torch.ones(2, 3, device="cuda").T)
        elif failure == "gather_out_shape":
            torch.ops._c10d_functional.all_gather_into_tensor_out(
                x,
                2,
                dist.distributed_c10d._get_default_group().group_name,
                out=torch.empty(3, device="cuda"),
            )
        elif failure == "gather_lists":
            outputs = [[torch.empty_like(x)], [torch.empty_like(x) for _ in range(3)]]
            dist.distributed_c10d._get_default_group().allgather(outputs, [x, x])
        else:
            dist.all_reduce(x)

    with pytest.raises(RuntimeError, match="sequential rank"):
        executor.run(forward)
    with pytest.raises(RuntimeError, match="aborted"):
        executor.run(lambda rank: rank)


def test_cleanup_deactivates_rank_owned_dtensor_offloaders_after_error(executor):
    models = [_Model().cuda(), _Model().cuda()]
    offloaders = [None, None]

    def setup(rank):
        model = _shard(models[rank], DeviceMesh("cuda", [0, 1]))
        offloaders[rank] = offloader = ModelOffloader.from_module(model)
        offloader.activate("cuda")
        assert all(p.to_local().is_cuda for p in model.parameters())
        return threading.get_ident()

    identities = executor.run(setup, sequential=False)

    def forward(rank):
        raise ValueError("forward failed")

    def cleanup(rank):
        assert dist.get_rank() == rank
        assert threading.get_ident() == identities[rank]
        offloaders[rank].deactivate()
        assert all(p.device_mesh.device_type == "cpu" for p in models[rank].parameters())
        return rank

    try:
        with pytest.raises(RuntimeError, match="forward failed"):
            executor.run(forward)
    finally:
        assert executor.run(cleanup, sequential=False) == (0, 1)
    # Local cleanup does not reset failed distributed execution.
    with pytest.raises(RuntimeError, match="aborted"):
        executor.run(lambda rank: dist.barrier(), sequential=False)
    with pytest.raises(RuntimeError, match="aborted"):
        executor.run(lambda rank: rank)


def test_cleanup_error_still_runs_both_ranks(executor):
    def fail(rank):
        raise ValueError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        executor.run(fail)
    cleaned = []

    def cleanup(rank):
        cleaned.append(rank)
        if rank == 0:
            raise ValueError("cleanup failed")
        return rank

    with pytest.raises(RuntimeError, match="cleanup failed"):
        executor.run(cleanup, sequential=False)
    assert sorted(cleaned) == [0, 1]
    assert executor.run(lambda rank: rank, sequential=False) == (0, 1)


def test_cleanup_waits_for_previous_callbacks_to_exit():
    pytest.importorskip("triton")
    entered, release = threading.Event(), threading.Event()
    cleaned = []
    with SequentialExecutor(timeout=timedelta(seconds=1)) as executor:

        def forward(rank):
            if rank == 0:
                assert entered.wait(5)
                raise ValueError("forward failed")
            entered.set()
            assert release.wait(10)

        try:
            with pytest.raises(RuntimeError, match="forward failed"):
                executor.run(forward, sequential=False)
            with pytest.raises(TimeoutError, match="cleanup has not started"):
                executor.run(cleaned.append, sequential=False)
            assert cleaned == []
        finally:
            release.set()
            executor.run(cleaned.append, sequential=False)
        assert sorted(cleaned) == [0, 1]


def test_cleanup_timeout_preserves_pending_callback():
    pytest.importorskip("triton")
    release = threading.Event()
    cleaned = []
    with SequentialExecutor(timeout=timedelta(seconds=1)) as executor:

        def fail(rank):
            raise ValueError("forward failed")

        with pytest.raises(RuntimeError, match="forward failed"):
            executor.run(fail)

        def cleanup(rank):
            if rank == 1:
                assert release.wait(10)
            cleaned.append(rank)

        try:
            with pytest.raises(TimeoutError, match="cleanup callbacks have not exited"):
                executor.run(cleanup, sequential=False)
            assert cleaned == [0]
        finally:
            release.set()
            # This call waits for the pending cleanup before invoking either rank.
            executor.run(lambda rank: None, sequential=False)
        assert sorted(cleaned) == [0, 1]


def test_rejects_nested_executor_and_run(executor):
    with pytest.raises(RuntimeError, match="already open"):
        with SequentialExecutor():
            pass
    with pytest.raises(RuntimeError, match="overlap or nest"):
        executor.run(lambda rank: executor.run(lambda peer: peer))


@pytest.mark.parametrize("format_name", ["nvfp4", "convrot_nvfp4"])
@pytest.mark.parametrize("placement", [Replicate(), Shard(0), Shard(1)])
def test_high_first_dtensor_offload_roundtrip(executor, format_name, placement):
    from tests.test_nvfp4_adapter import _make_nvfp4_amax, _piper_nvfp4
    from tests.test_piper_convrot_nvfp4_adapter import _make_convrot_nvfp4

    if format_name == "nvfp4":
        source = _piper_nvfp4(_make_nvfp4_amax(), high_first=True)
    else:
        source, _ = _make_convrot_nvfp4(high_first=True)
    source = source.cuda()

    def forward(rank):
        mesh = DeviceMesh("cuda", [0, 1])
        weight = DTensor.from_local(source, mesh, [placement])
        host = HostParam(nn.Parameter(weight, requires_grad=False))
        state = host.allocate_gpu_storage(torch.device("cuda"))
        for _ in range(3):
            cpu = host.make_cpu_param().to_local()
            assert cpu.device.type == "cpu"
            assert cpu.high_first is True
            torch.testing.assert_close(cpu.qdata, source.qdata.cpu())
            host.copy_to_gpu(state)
            resident = host.make_gpu_param(state)
            assert resident.placements == weight.placements
            assert resident.device_mesh == mesh
            local = resident.to_local()
            assert local.high_first is True
            torch.testing.assert_close(local.qdata, source.qdata)
            torch.testing.assert_close(local.dequantize(), source.dequantize())

    executor.run(forward)


def test_restores_distributed_state_after_repeated_sessions():
    pytest.importorskip("triton")
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    original = dist.distributed_c10d._world
    original_hook = sys.excepthook
    original_mesh_init = DeviceMesh.__init__
    original_autotune = CachingAutotuner.autotune_to_one_config
    mesh_ids = set()
    for _ in range(2):
        with SequentialExecutor(timeout=timedelta(seconds=10)) as executor:
            assert executor.run(lambda rank: dist.get_rank()) == (0, 1)

            def setup(rank):
                mesh = DeviceMesh("cuda", [0, 1])
                same = DeviceMesh("cuda", [0, 1])
                cpu = DeviceMesh("cpu", [0, 1])
                assert mesh == same and hash(mesh) == hash(same)
                assert cpu._thread_id == mesh._thread_id
                return mesh._thread_id

            for identity in executor.run(setup, sequential=False):
                assert identity is not None and identity not in mesh_ids
                mesh_ids.add(identity)
        assert dist.distributed_c10d._world is original
        assert sys.excepthook is original_hook
        assert DeviceMesh.__init__ is original_mesh_init
        assert CachingAutotuner.autotune_to_one_config is original_autotune
        assert not dist.is_initialized()


@pytest.mark.parametrize("failure", ["isolation", "worker"])
def test_startup_failure_restores_runtime(monkeypatch, failure):
    pytest.importorskip("triton")
    from piper_offload import _sequential_runtime as runtime

    original = (
        dist.distributed_c10d._world,
        DeviceMesh.__init__,
        runtime.CachingAutotuner.autotune_to_one_config,
        sys.excepthook,
    )
    init_group = dist.init_process_group
    set_isolation = runtime._set_thread_isolation_mode

    def failing_isolation(enabled):
        set_isolation(enabled)
        if enabled:
            raise RuntimeError("startup failed")

    def failing_worker(*args, **kwargs):
        if kwargs["rank"] == 1:
            raise RuntimeError("startup failed")
        return init_group(*args, **kwargs)

    with monkeypatch.context() as patch:
        if failure == "isolation":
            patch.setattr(runtime, "_set_thread_isolation_mode", failing_isolation)
        else:
            patch.setattr(dist, "init_process_group", failing_worker)
        with pytest.raises(RuntimeError, match="startup failed"):
            with SequentialExecutor(timeout=timedelta(seconds=10)):
                pass

    restored = (
        dist.distributed_c10d._world,
        DeviceMesh.__init__,
        runtime.CachingAutotuner.autotune_to_one_config,
        sys.excepthook,
    )
    assert all(a is b for a, b in zip(original, restored, strict=True))
    with SequentialExecutor() as executor:
        assert executor.run(lambda rank: rank) == (0, 1)


def test_compiled_gather_redistribution(executor):
    def setup(rank):
        mesh = DeviceMesh("cuda", [0, 1])
        local = torch.full((3, 4), float(rank), device="cuda")
        value = DTensor.from_local(local, mesh, [Shard(0)])

        def forward(x):
            return x.redistribute(placements=[Replicate()]).to_local() + 2

        return torch.compile(forward, fullgraph=True, options={"triton.cudagraphs": False, "compile_threads": 1}), value

    states = executor.run(setup, sequential=False)
    executor.run(lambda rank: states[rank][0](states[rank][1]), sequential=False)
    expected = torch.cat([torch.full((3, 4), 2.0), torch.full((3, 4), 3.0)]).cuda()
    for output in executor.run(lambda rank: states[rank][0](states[rank][1])):
        torch.testing.assert_close(output, expected)


def test_idle_workers_release_callback_outputs(executor):
    released = [threading.Event(), threading.Event()]
    values = executor.run(lambda rank: torch.empty(1024, device="cuda"))
    references = [
        weakref.ref(value, lambda _, event=event: event.set()) for value, event in zip(values, released, strict=True)
    ]
    del values
    # Wait for the last Python reference to go away while workers remain idle.
    # A second run would itself overwrite stale job/result references.
    assert all(event.wait(5) for event in released)
    assert all(reference() is None for reference in references)


def test_exact_cross_rank_sum_alias(executor):
    value = torch.arange(2053, device="cuda", dtype=torch.bfloat16)
    expected = value * 2
    executor.run(lambda rank: dist.all_reduce(value))
    torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["sum", "coalesced", "gather"])
def test_rejects_aliases_before_mutation(executor, kind):
    backing = torch.arange(8, device="cuda", dtype=torch.float32)
    expected = backing.clone()

    def forward(rank):
        if kind == "sum":
            dist.all_reduce(backing[rank : rank + 4])
        elif kind == "coalesced":
            dist.distributed_c10d._get_default_group().allreduce_coalesced([backing[:4], backing[2:6]]).wait()
        else:
            # Rank 0's first output would destroy rank 1's original input.
            dist.all_gather_single(backing, backing[4 * (1 - rank) : 4 * (2 - rank)])

    with pytest.raises(RuntimeError, match="overlap|overwrite"):
        executor.run(forward)
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)


def test_gather_allows_direct_reads_from_output_slices(executor):
    values = [torch.empty(8, device="cuda") for _ in range(2)]
    values[0][:4].fill_(7)
    values[1][4:].fill_(9)

    def forward(rank):
        dist.all_gather_single(values[rank], values[rank][4 * rank : 4 * (rank + 1)])
        return values[rank]

    for value in executor.run(forward):
        torch.testing.assert_close(value.cpu(), torch.tensor([7.0] * 4 + [9.0] * 4))


def test_collective_timeout_and_reopen():
    pytest.importorskip("triton")
    release = threading.Event()
    with SequentialExecutor(timeout=timedelta(seconds=1)) as executor:

        def forward(rank):
            if rank == 1:
                release.wait(5)
            else:
                dist.all_reduce(torch.ones(2, device="cuda"))

        try:
            with pytest.raises(RuntimeError, match="timed out"):
                executor.run(forward)
        finally:
            release.set()
    with SequentialExecutor() as executor:
        assert executor.run(lambda rank: rank) == (0, 1)


def test_close_keeps_isolation_until_blocked_callback_exits():
    pytest.importorskip("triton")
    original = dist.distributed_c10d._world
    release = threading.Event()
    executor = SequentialExecutor(timeout=timedelta(seconds=1)).__enter__()

    def forward(rank):
        if rank == 1:
            release.wait(10)
        else:
            dist.all_reduce(torch.ones(2, device="cuda"))

    try:
        with pytest.raises(RuntimeError, match="timed out"):
            executor.run(forward)
        with pytest.raises(TimeoutError, match="isolation is still active"):
            executor.close()
        assert dist.distributed_c10d._world is not original
        with pytest.raises(RuntimeError, match="not closing"):
            executor.run(lambda rank: None, sequential=False)
        with pytest.raises(RuntimeError, match="already open"):
            with SequentialExecutor():
                pass
    finally:
        release.set()
        executor.close()
    assert dist.distributed_c10d._world is original


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo needed only to set up the pre-existing group")
def test_rejects_existing_group_without_changing_it():
    pytest.importorskip("triton")
    original_hook = sys.excepthook
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        group = dist.distributed_c10d._get_default_group()
        with pytest.raises(RuntimeError, match="no existing distributed"):
            with SequentialExecutor():
                pass
        assert dist.distributed_c10d._get_default_group() is group
        value = torch.ones(2)
        dist.all_reduce(value)
        torch.testing.assert_close(value, torch.ones(2))
    finally:
        dist.destroy_process_group()
        sys.excepthook = original_hook
