"""Copies from anonymous CPU storage preserve weight values and GPU scheduling."""

import gc
import struct
import threading
import weakref

import pytest
import torch
from torch import nn

import piper_offload._host_copy as host_copy
from piper_offload import BlockCompileConfig
from piper_offload.host_memory import HostMemoryManager
from piper_offload.composite_component import CompositeComponentStore
from piper_offload.host_buffer import HostBuffer
from piper_offload.host_module import HostModuleStore
from piper_offload.host_param import HostParam
from piper_offload.target_lease import CudaTargetLease
from tests._block_compile_helpers import _BlockModel, _make_offloader
from tests.conftest import activated_model, block_components
from tests.test_quantized_parameter_value import _QUANT_KINDS, _make_quantized

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class Completion:
    def __init__(self):
        self.done = False
        self.waited = False

    def query(self):
        return self.done

    def synchronize(self):
        self.waited = True
        self.done = True


def _bytes(tensor):
    storage = tensor.untyped_storage()
    return torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))


def test_handles_deduplicate_allocations_and_preserve_mixed_views():
    source = torch.arange(24, dtype=torch.int32).reshape(4, 6)
    view = source[1:, 1::2]
    as_bytes = _bytes(source)[3:17]
    memory_manager = HostMemoryManager()
    backings = memory_manager.capture((source, view, as_bytes))
    (backing,) = backings.values()
    assert tuple(memory_manager.capture((view,)).values()) == (backing,)
    assert backing.nbytes == source.untyped_storage().nbytes()
    anonymous_storage = _bytes(source).clone()
    assert backing.try_set_anonymous_storage(anonymous_storage)

    for tensor in (source, view, as_bytes, source[:0]):
        with backing.acquire(tensor) as lease:
            resolved = lease.tensor
            assert resolved.untyped_storage().data_ptr() == anonymous_storage.data_ptr()
            assert resolved.dtype == tensor.dtype
            assert resolved.shape == tensor.shape
            assert resolved.stride() == tensor.stride()
            assert resolved.storage_offset() == tensor.storage_offset()
            torch.testing.assert_close(resolved, tensor)

    assert backing.try_set_anonymous_storage(None)
    with backing.acquire(view) as lease:
        assert lease.tensor is view


def test_mmap_parameter_and_buffer_share_handle_and_keep_file_backed_wrappers(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(struct.pack("24f", *range(24)))
    source = torch.from_file(str(path), shared=False, size=24, dtype=torch.float32).reshape(4, 6)
    memory_manager = HostMemoryManager()
    host = HostParam(nn.Parameter(source, requires_grad=False), memory_manager=memory_manager)
    view = HostBuffer(source[:, 1::2], HostBuffer.target_layout_for(source[:, 1::2]), memory_manager=memory_manager)
    (backing,) = host.backing_handles()
    assert view.backing_handles() == (backing,)
    resting = host.make_cpu_param()
    anonymous_storage = _bytes(source).clone()
    assert backing.try_set_anonymous_storage(anonymous_storage)
    assert resting.data_ptr() == source.data_ptr()
    assert host.make_cpu_param().data_ptr() == source.data_ptr()
    assert host.storage_tensors()[0].data_ptr() == source.data_ptr()
    assert view.tensor.untyped_storage().data_ptr() == source.data_ptr()
    assert backing.try_set_anonymous_storage(None)
    torch.testing.assert_close(host.make_cpu_param(), source)


def test_independent_captures_do_not_share_anonymous_storage(monkeypatch):
    source = nn.Parameter(torch.arange(7.0), requires_grad=False)
    first, second = HostParam(source), HostParam(source)
    (first_backing,) = first.backing_handles()
    (second_backing,) = second.backing_handles()
    assert first_backing is not second_backing
    anonymous = source.detach().clone()
    assert first_backing.try_set_anonymous_storage(anonymous)
    observed = []
    raw_copy = host_copy._copy_host_to_device

    def observe(destination, source, *, non_blocking):
        observed.append(source.data_ptr())
        raw_copy(destination, source, non_blocking=non_blocking)

    monkeypatch.setattr(host_copy, "_copy_host_to_device", observe)
    for host in (first, second):
        target = host.allocate_gpu_storage(torch.device("cpu"))
        host.copy_to_gpu(target)
        torch.testing.assert_close(host.make_gpu_param(target), source)
    assert observed == [anonymous.data_ptr(), source.data_ptr()]


def test_shared_manager_does_not_keep_unrelated_parameters_alive():
    memory_manager = HostMemoryManager()
    first = HostParam(nn.Parameter(torch.arange(7.0)), memory_manager=memory_manager)
    second = HostParam(nn.Parameter(torch.arange(9.0)), memory_manager=memory_manager)
    second_storage = weakref.ref(second.storage_tensors()[0].untyped_storage())
    del second
    gc.collect()
    assert second_storage() is None
    torch.testing.assert_close(first.make_cpu_param(), torch.arange(7.0))


def test_model_capture_shares_backing_across_components_and_buffers():
    model = _BlockModel(num_blocks=2, width=8)
    weight = model.blocks[0].proj.weight
    model.blocks[1].proj.weight = nn.Parameter(weight.detach(), requires_grad=False)
    model.register_buffer("shared", weight.detach())
    model.transient = nn.Linear(8, 8, bias=False).requires_grad_(False)
    model.transient.weight = nn.Parameter(weight.detach(), requires_grad=False)
    store = CompositeComponentStore.from_module(model, block_paths=["blocks"], transient_paths=["transient"])
    block_stores = store.block_stores[0]._block_stores
    handles = [block.params["proj.weight"].backing_handles() for block in block_stores]
    handles.append(store.resident_store._module_store.buffers["shared"].backing_handles())
    handles.append(store.transient_stores[0][1]._module_store.params["transient.weight"].backing_handles())
    assert all(handle == handles[0] for handle in handles)


def test_copy_rejects_sources_outside_its_owner():
    source = torch.arange(7.0)
    backings = HostMemoryManager().capture((source,))
    with pytest.raises(ValueError, match="not captured"):
        host_copy.copy_host_to_device(torch.empty_like(source), source.clone(), backings=backings, non_blocking=False)


def test_source_acquisition_does_not_materialize_or_initialize_cuda(monkeypatch):
    source = torch.arange(7)
    backings = HostMemoryManager().capture((source,))
    (backing,) = backings.values()

    def unexpected(*args, **kwargs):
        raise AssertionError("source resolution must not allocate or use CUDA")

    monkeypatch.setattr(torch, "empty", unexpected)
    monkeypatch.setattr(torch.cuda, "current_stream", unexpected)
    with backing.acquire(source) as lease:
        assert lease.tensor is source


def test_setting_anonymous_storage_waits_for_leases_and_all_pending_copies():
    source = torch.arange(7)
    backings = HostMemoryManager().capture((source,))
    (backing,) = backings.values()
    first = backing.acquire(source)
    second = backing.acquire(source)
    anonymous_storage = source.clone()
    assert not backing.try_set_anonymous_storage(anonymous_storage)
    first_done, second_done = Completion(), Completion()
    first.close(first_done)
    first.close()
    assert not backing.try_set_anonymous_storage(anonymous_storage)
    second.close(second_done)
    first_done.done = True
    assert not backing.try_set_anonymous_storage(anonymous_storage)
    second_done.done = True
    assert backing.try_set_anonymous_storage(anonymous_storage)
    with pytest.raises(RuntimeError, match="closed"):
        _ = first.tensor


def test_active_reader_on_another_thread_prevents_setting_anonymous_storage():
    source = torch.arange(7)
    backings = HostMemoryManager().capture((source,))
    (backing,) = backings.values()
    acquired = threading.Event()
    release = threading.Event()

    def read():
        with backing.acquire(source):
            acquired.set()
            assert release.wait(5)

    reader = threading.Thread(target=read)
    reader.start()
    try:
        assert acquired.wait(5)
        assert not backing.try_set_anonymous_storage(source.clone())
    finally:
        release.set()
        reader.join(5)
    assert not reader.is_alive()
    assert backing.try_set_anonymous_storage(source.clone())


def test_last_owner_waits_for_copy_before_releasing_storage():
    source = torch.arange(7)
    backings = HostMemoryManager().capture((source,))
    (backing,) = backings.values()
    anonymous_storage = source.clone()
    assert backing.try_set_anonymous_storage(anonymous_storage)
    storage_ref = weakref.ref(anonymous_storage.untyped_storage())
    backing_ref = weakref.ref(backing)
    lease = backing.acquire(source)
    completion = Completion()
    lease.close(completion)
    del anonymous_storage, backing, backings
    gc.collect()
    assert completion.waited
    assert backing_ref() is None
    assert storage_ref() is None


def test_eviction_releases_anonymous_storage_even_while_store_and_cpu_wrapper_live():
    host = HostParam(nn.Parameter(torch.arange(7.0), requires_grad=False))
    (backing,) = host.backing_handles()
    resting = host.make_cpu_param()
    anonymous_storage = resting.detach().clone()
    storage_ref = weakref.ref(anonymous_storage.untyped_storage())
    assert backing.try_set_anonymous_storage(anonymous_storage)
    del anonymous_storage
    assert storage_ref() is not None
    assert backing.try_set_anonymous_storage(None)
    gc.collect()
    assert storage_ref() is None
    torch.testing.assert_close(host.make_cpu_param(), resting)


@pytest.mark.parametrize("invalid", [torch.empty(6), torch.empty(8), torch.empty(8)[1:], torch.empty(7, 2)[:, 0]])
def test_rejects_partial_or_mismatched_anonymous_storage_allocations(invalid):
    backings = HostMemoryManager().capture((torch.empty(7),))
    (backing,) = backings.values()
    with pytest.raises(ValueError, match="complete CPU allocation"):
        backing.try_set_anonymous_storage(invalid)


def test_rejects_unrelated_view_and_source_as_anonymous_storage():
    source = torch.arange(7)
    backings = HostMemoryManager().capture((source,))
    (backing,) = backings.values()
    with pytest.raises(ValueError, match="does not belong"):
        backing.acquire(source.clone())
    with pytest.raises(ValueError, match="source storage"):
        backing.try_set_anonymous_storage(source)


def test_empty_and_meta_parameters():
    empty = HostParam(nn.Parameter(torch.empty(0), requires_grad=False))
    (backing,) = empty.backing_handles()
    assert backing.nbytes == 0
    assert backing.try_set_anonymous_storage(torch.empty(0))
    with backing.acquire(empty.storage_tensors()[0]) as lease:
        assert lease.tensor.numel() == 0
    meta = HostParam(nn.Parameter(torch.empty(7, device="meta"), requires_grad=False))
    assert meta.backing_handles() == ()


@CUDA
def test_plain_parameter_and_buffer_copies_read_from_anonymous_storage(monkeypatch):
    module = nn.Linear(4, 3, bias=False).requires_grad_(False)
    module.register_buffer("scale", torch.arange(3.0))
    store = HostModuleStore.from_module(module)
    plan = store.bind(module).resolve_load_plan()
    expected_weight = module.weight.detach().clone()
    expected_scale = module.scale.clone()
    pointers = set()
    handles = []
    for host in (*store.params.values(), *store.buffers.values()):
        (backing,) = host.backing_handles()
        anonymous_storage = _bytes(host.storage_tensors()[0]).clone()
        pointers.add(anonymous_storage.data_ptr())
        assert backing.try_set_anonymous_storage(anonymous_storage)
        handles.append(backing)

    observed = set()
    raw_copy = host_copy._copy_host_to_device

    def observe(destination, source, *, non_blocking):
        observed.add(source.untyped_storage().data_ptr())
        raw_copy(destination, source, non_blocking=non_blocking)

    monkeypatch.setattr(host_copy, "_copy_host_to_device", observe)
    target = plan.allocate_target(torch.device("cuda"))
    plan.copy_to_target(target)
    assert observed == pointers
    torch.testing.assert_close(target.param_targets["weight"].param.cpu(), expected_weight)
    torch.testing.assert_close(target.buffer_targets["scale"].tensor.cpu(), expected_scale)
    torch.cuda.synchronize()
    for backing in handles:
        assert backing.try_set_anonymous_storage(None)
    observed.clear()
    plan.copy_to_target(target)
    assert not observed.intersection(pointers)


@CUDA
@pytest.mark.parametrize("kind", [*_QUANT_KINDS, "int4-tile", "gguf"])
def test_quantized_copies_resolve_physical_storage_without_rewriting_state(kind, monkeypatch):
    if kind == "int4-tile":
        from tests.test_int4_tile_adapter import _make_int4_tile

        source_param = _make_int4_tile()
    elif kind == "gguf":
        from tests.test_gguf_adapter import _quantized_weight

        source_param, _, _ = _quantized_weight(123)
    else:
        source_param = _make_quantized(kind)
    if not isinstance(source_param, nn.Parameter):
        source_param = nn.Parameter(source_param, requires_grad=False)
    memory_manager = HostMemoryManager()
    host = HostParam(source_param, memory_manager=memory_manager)
    state = host.host_state
    sources = host.storage_tensors()
    pointers = {}
    for source in sources:
        key = source.untyped_storage()._cdata
        if key in pointers:
            continue
        backings = memory_manager.capture((source,))
        (backing,) = backings.values()
        anonymous_storage = _bytes(source).clone()
        assert backing.try_set_anonymous_storage(anonymous_storage)
        pointers[key] = anonymous_storage.data_ptr()
    observed = set()
    raw_copy = host_copy._copy_host_to_device

    def observe(destination, source, *, non_blocking):
        assert source.untyped_storage().data_ptr() in pointers.values()
        observed.add(source.untyped_storage().data_ptr())
        raw_copy(destination, source, non_blocking=non_blocking)

    monkeypatch.setattr(host_copy, "_copy_host_to_device", observe)
    target = host.allocate_gpu_storage(torch.device("cuda"))
    host.copy_to_gpu(target, non_blocking=True)
    torch.cuda.synchronize()
    assert observed
    assert host.host_state is state
    assert all(a is b for a, b in zip(host.storage_tensors(), sources, strict=True))
    actual = host.make_gpu_param(target)
    for backing in host.backing_handles():
        assert backing.try_set_anonymous_storage(None)
    monkeypatch.setattr(host_copy, "_copy_host_to_device", raw_copy)
    expected_target = host.allocate_gpu_storage(torch.device("cuda"))
    host.copy_to_gpu(expected_target)
    # Compare physical recaptures so no format-specific dequantization tolerance
    # can hide a corrupt packed byte or scale.
    actual_host = HostParam(actual)
    expected_host = HostParam(host.make_gpu_param(expected_target))
    for actual_tensor, expected_tensor in zip(
        actual_host.storage_tensors(),
        expected_host.storage_tensors(),
        strict=True,
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@CUDA
@pytest.mark.parametrize("fail_after_copy", [False, True])
def test_async_copy_prevents_early_reuse_including_partial_failure(monkeypatch, fail_after_copy):
    source = torch.arange(1024.0)
    backings = HostMemoryManager().capture((source,))
    (backing,) = backings.values()
    anonymous_storage = source.pin_memory()
    assert backing.try_set_anonymous_storage(anonymous_storage)
    destination = torch.empty_like(source, device="cuda")
    stream = torch.cuda.Stream()
    # Warm the path before adding a delay, avoiding lazy initialization waits.
    with torch.cuda.stream(stream):
        host_copy.copy_host_to_device(destination, source, backings=backings, non_blocking=True)
        torch.cuda._sleep(1)
    stream.synchronize()
    raw_copy = host_copy._copy_host_to_device

    def copy_then_fail(destination, source, *, non_blocking):
        raw_copy(destination, source, non_blocking=non_blocking)
        raise RuntimeError("after enqueue")

    if fail_after_copy:
        monkeypatch.setattr(host_copy, "_copy_host_to_device", copy_then_fail)
    try:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(500_000_000)
            if fail_after_copy:
                with pytest.raises(RuntimeError, match="after enqueue"):
                    host_copy.copy_host_to_device(destination, source, backings=backings, non_blocking=True)
            else:
                host_copy.copy_host_to_device(destination, source, backings=backings, non_blocking=True)
        assert not backing.try_set_anonymous_storage(None)
    finally:
        stream.synchronize()
    assert backing.try_set_anonymous_storage(None)
    torch.testing.assert_close(destination.cpu(), source)


@CUDA
def test_existing_target_lease_copies_from_anonymous_storage_and_then_source():
    module = nn.Linear(16, 16, bias=False).requires_grad_(False)
    store = HostModuleStore.from_module(module)
    instance = store.bind(module)
    plan = instance.resolve_load_plan()
    host = store.params["weight"]
    (backing,) = host.backing_handles()
    assert backing.try_set_anonymous_storage(host.storage_tensors()[0].pin_memory())
    target = CudaTargetLease.allocate(plan, torch.device("cuda"))
    stream = torch.cuda.Stream()
    try:
        for _ in range(2):
            target.stage(plan, stream)
            with torch.cuda.stream(stream):
                active = target.acquire(stream)
                actual = active.param_targets["weight"].param.cpu()
            target.release()
            stream.synchronize()
            assert backing.try_set_anonymous_storage(None)
            torch.testing.assert_close(actual, module.weight)
    finally:
        target.close()


@CUDA
@pytest.mark.parametrize("mode", ["streaming", "rolling", "resident"])
def test_runtime_reuses_then_evicts_anonymous_storage_without_changing_weights(mode, monkeypatch):
    model = _BlockModel(num_blocks=3, width=64)
    value = torch.randn(2, 64)
    expected = model(value)
    offloader = _make_offloader(
        model,
        block_mode=mode,
        block_compile=BlockCompileConfig(fullgraph=True) if mode == "rolling" else None,
    )
    handles = []
    pointers = set()
    for instance in block_components(offloader)[0]._block_instances:
        for host in instance.params.values():
            (backing,) = host.backing_handles()
            anonymous_storage = host.storage_tensors()[0].pin_memory()
            pointers.add(anonymous_storage.data_ptr())
            assert backing.try_set_anonymous_storage(anonymous_storage)
            handles.append(backing)

    observed = set()
    raw_copy = host_copy._copy_host_to_device

    def observe(destination, source, *, non_blocking):
        observed.add(source.data_ptr())
        raw_copy(destination, source, non_blocking=non_blocking)

    monkeypatch.setattr(host_copy, "_copy_host_to_device", observe)
    try:
        for use_anonymous_storage in (True, True, False):
            if not use_anonymous_storage:
                for backing in handles:
                    assert backing.try_set_anonymous_storage(None)
            observed.clear()
            with activated_model(offloader, "cuda"), torch.inference_mode():
                actual = model(value.cuda()).cpu()
            torch.testing.assert_close(actual, expected)
            if use_anonymous_storage:
                assert observed == pointers
            else:
                assert not observed.intersection(pointers)
    finally:
        offloader.deactivate()
        if mode == "rolling":
            torch.compiler.reset()


@CUDA
def test_event_record_failure_waits_before_releasing_source(monkeypatch):
    source = torch.arange(1024.0)
    backings = HostMemoryManager().capture((source,))
    (backing,) = backings.values()
    assert backing.try_set_anonymous_storage(source.pin_memory())
    destination = torch.empty_like(source, device="cuda")
    actual_stream = torch.cuda.current_stream()

    class FailingStream:
        waited = False

        def record_event(self):
            raise RuntimeError("cannot record")

        def synchronize(self):
            actual_stream.synchronize()
            self.waited = True

    failing = FailingStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: failing)
    with pytest.raises(RuntimeError, match="cannot record"):
        host_copy.copy_host_to_device(destination, source, backings=backings, non_blocking=True)
    assert failing.waited
    assert backing.try_set_anonymous_storage(None)
    torch.testing.assert_close(destination.cpu(), source)


@CUDA
def test_copy_from_anonymous_storage_rejects_graph_capture_before_borrowed_pointer_escapes():
    source = torch.ones(16)
    backings = HostMemoryManager().capture((source,))
    (backing,) = backings.values()
    assert backing.try_set_anonymous_storage(source.pin_memory())
    destination = torch.empty_like(source, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        host_copy.copy_host_to_device(destination, source, backings=backings, non_blocking=True)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        with pytest.raises(RuntimeError, match="cannot be captured"):
            host_copy.copy_host_to_device(destination, source, backings=backings, non_blocking=True)
        # Include ordinary device work so capture is nonempty.
        destination.add_(1)
    assert backing.try_set_anonymous_storage(None)


def test_backing_handle_keeps_its_manager_alive_without_retaining_siblings():
    manager = HostMemoryManager()
    manager_ref = weakref.ref(manager)
    source = torch.arange(16.0)
    backings = manager.capture((source,))
    (backing,) = backings.values()
    assert backing.memory_manager is manager
    del manager, backings
    gc.collect()
    assert manager_ref() is backing.memory_manager
    with backing.acquire(source) as lease:
        torch.testing.assert_close(lease.tensor, source)
    del backing
    gc.collect()
    assert manager_ref() is None


def test_live_manager_does_not_retain_discarded_anonymous_copy():
    manager = HostMemoryManager()
    host = HostParam(nn.Parameter(torch.arange(8.0)), memory_manager=manager)
    (backing,) = host.backing_handles()
    anonymous = torch.arange(8.0)
    source_ref = weakref.ref(host.storage_tensors()[0].untyped_storage())
    anonymous_ref = weakref.ref(anonymous.untyped_storage())
    assert backing.try_set_anonymous_storage(anonymous)
    del anonymous, backing, host
    gc.collect()
    assert source_ref() is None
    assert anonymous_ref() is None
    assert not manager._backings


def test_adapter_capture_shares_the_explicit_manager_across_all_sources():
    from piper_offload import Adapter

    manager = HostMemoryManager(max_pinned_bytes=0)
    adapter = Adapter.from_state_dict({
        "projection.lora_A.weight": torch.randn(2, 8),
        "projection.lora_B.weight": torch.randn(4, 2),
        "projection.delta.weight": torch.randn(4, 8),
        "replacement.weight": torch.randn(4, 8),
    }, memory_manager=manager)
    delta = adapter.targets["projection.weight"]
    value = adapter.targets["replacement.weight"]
    for host in (delta.lora.a, delta.lora.b, delta.dense, value.backing):
        assert host.memory_manager is manager
        assert all(handle.memory_manager is manager for handle in host.backing_handles())
