"""Pinned copies of file-backed weights preserve values and GPU scheduling."""

import gc
import mmap
import struct
import threading
import weakref

import pytest
import torch
from torch import nn

import piper_offload._host_backing as host_backing_module
from piper_offload import BlockCompileConfig
from piper_offload.host_memory import HostMemoryManager
from piper_offload.composite_component import CompositeComponentStore
from piper_offload.host_buffer import HostBuffer
from piper_offload.host_module import HostModuleStore
from piper_offload.host_param import HostParam
from piper_offload.target_lease import CudaTargetLease
from tests._block_compile_helpers import _BlockModel, _make_offloader
from tests._host_memory_helpers import RecordingBackend
from tests.conftest import activated_model, block_components
from tests.test_host_memory import FakeBackend
from tests.test_quantized_parameter_value import _QUANT_KINDS, _make_quantized

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
BUDGET = 64 * 1024**2


class _Completion:
    def __init__(self):
        self.done = False
        self.waited = False

    def query(self):
        return self.done

    def synchronize(self):
        self.waited = True
        self.done = True


class _Stream:
    def __init__(self, completion):
        self.completion = completion

    def record_event(self):
        return self.completion

    def synchronize(self):
        self.completion.synchronize()


def _bytes(tensor):
    storage = tensor.untyped_storage()
    return torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))


def _file_tensor(tmp_path, tensor, name="weights.bin"):
    """A private file mapping holding ``tensor``'s bytes, as a checkpoint loader would produce."""
    path = tmp_path / name
    path.write_bytes(_bytes(tensor.contiguous()).numpy().tobytes())
    return torch.from_file(str(path), shared=False, size=tensor.numel(), dtype=tensor.dtype).view(tensor.shape)


def _observe_transfers(monkeypatch):
    observed = []
    raw_copy = host_backing_module._transfer

    def observe(destination, source, *, non_blocking):
        observed.append(source.untyped_storage().data_ptr())
        raw_copy(destination, source, non_blocking=non_blocking)

    monkeypatch.setattr(host_backing_module, "_transfer", observe)
    return observed


def test_pin_registers_anonymous_memory_in_place():
    backend = FakeBackend()
    source = torch.arange(24.0)
    (backing,) = HostMemoryManager(backend=backend).capture((source,)).values()
    assert backing.pin_in_place and not backing.needs_copy
    assert backing.pin()
    assert backing.pinned and backing.copy_bytes == 0
    assert backend.register_calls == [(source.data_ptr(), source.nbytes)]
    assert backing.evict()
    assert backend.unregister_calls == [source.data_ptr()]


def test_pin_copies_a_private_file_mapping(tmp_path):
    backend = FakeBackend()
    source = _file_tensor(tmp_path, torch.arange(24.0))
    (backing,) = HostMemoryManager(backend=backend).capture((source,)).values()
    assert not backing.pin_in_place and backing.needs_copy
    assert backing.pin()
    pointer, size = backing.span
    assert pointer != source.data_ptr() and size == source.nbytes
    assert pointer % mmap.PAGESIZE == 0
    assert backing.copy_bytes == source.nbytes
    assert backend.register_calls == [(pointer, size)]
    # Unpin keeps the copy; a later pin reuses it without another memcpy.
    assert backing.unpin()
    assert not backing.pinned and backing.copy_bytes == source.nbytes
    assert not backing.needs_copy
    assert backing.pin()
    assert backing.span == (pointer, size)
    assert backing.evict()
    assert backing.span == (source.data_ptr(), source.nbytes)
    assert backing.copy_bytes == 0


def test_trainable_mapped_weight_pins_in_place_and_write_back_is_guarded(tmp_path):
    manager = HostMemoryManager(BUDGET, backend=FakeBackend())
    trainable = HostParam(nn.Parameter(_file_tensor(tmp_path, torch.arange(8.0), "train.bin")), memory_manager=manager)
    frozen = HostParam(
        nn.Parameter(_file_tensor(tmp_path, torch.arange(8.0), "frozen.bin"), requires_grad=False),
        memory_manager=manager,
    )
    (trainable_backing,) = trainable.backing_handles()
    (frozen_backing,) = frozen.backing_handles()
    assert trainable_backing.pin_in_place and not frozen_backing.pin_in_place
    with manager.acquire([trainable_backing, frozen_backing]) as lease:
        assert lease.pinned
        assert trainable_backing.copy_bytes == 0 and frozen_backing.copy_bytes == 8 * 4
        target = trainable.allocate_gpu_storage(torch.device("cpu"))
        trainable.copy_to_gpu(target)
        trainable.copy_to_cpu(target)  # writes the source, which is what is pinned
        frozen_target = frozen.allocate_gpu_storage(torch.device("cpu"))
        frozen.copy_to_gpu(frozen_target)
        with pytest.raises(RuntimeError, match="pinned copy"):
            frozen.copy_to_cpu(frozen_target)
    manager.clear()


def test_in_flight_copy_blocks_unpin_and_evict_until_it_completes(tmp_path):
    source = _file_tensor(tmp_path, torch.arange(7.0))
    manager = HostMemoryManager(backend=FakeBackend())
    (backing,) = manager.capture((source,)).values()
    assert backing.pin()
    completion = _Completion()
    with backing._read(source, _Stream(completion)):
        pass
    assert backing.leases == 0 and backing.in_flight == 1 and not backing.idle
    assert not backing.unpin()
    assert not backing.evict()
    with manager.acquire([backing]) as lease:
        assert lease.pinned  # already pinned; nothing to admit
    completion.done = True
    assert backing.in_flight == 0 and backing.idle
    assert backing.evict()


def test_disposal_waits_for_in_flight_copies(tmp_path):
    source = _file_tensor(tmp_path, torch.arange(7.0))
    (backing,) = HostMemoryManager(backend=FakeBackend()).capture((source,)).values()
    completion = _Completion()
    with backing._read(source, _Stream(completion)):
        pass
    del backing
    gc.collect()
    assert completion.waited


def test_refused_registration_after_copying_keeps_no_copy(tmp_path):
    backend = FakeBackend()
    backend.capacity = 0
    source = _file_tensor(tmp_path, torch.arange(24.0))
    (backing,) = HostMemoryManager(backend=backend).capture((source,)).values()
    assert not backing.pin()
    assert not backing.pinned and backing.copy_bytes == 0
    assert len(backend.register_calls) == 1


def test_reads_resolve_mixed_views_onto_the_copy(tmp_path):
    source = _file_tensor(tmp_path, torch.arange(24, dtype=torch.int32).reshape(4, 6))
    view = source[1:, 1::2]
    as_bytes = _bytes(source)[3:17]
    memory_manager = HostMemoryManager(backend=FakeBackend())
    backings = memory_manager.capture((source, view, as_bytes))
    (backing,) = backings.values()
    assert tuple(memory_manager.capture((view,)).values()) == (backing,)
    assert backing.storage.nbytes() == source.untyped_storage().nbytes()
    assert backing.pin()
    copy_pointer = backing.span[0]

    for tensor in (source, view, as_bytes, source[:0]):
        with backing._read(tensor) as resolved:
            assert resolved.untyped_storage().data_ptr() == copy_pointer
            assert resolved.dtype == tensor.dtype
            assert resolved.shape == tensor.shape
            assert resolved.stride() == tensor.stride()
            assert resolved.storage_offset() == tensor.storage_offset()
            torch.testing.assert_close(resolved, tensor)

    assert backing.evict()
    with backing._read(view) as resolved:
        assert resolved is view


def test_mmap_parameter_and_buffer_share_handle_and_keep_file_backed_wrappers(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(struct.pack("24f", *range(24)))
    source = torch.from_file(str(path), shared=False, size=24, dtype=torch.float32).reshape(4, 6)
    memory_manager = HostMemoryManager(backend=FakeBackend())
    host = HostParam(nn.Parameter(source, requires_grad=False), memory_manager=memory_manager)
    view = HostBuffer(source[:, 1::2], HostBuffer.target_layout_for(source[:, 1::2]), memory_manager=memory_manager)
    (backing,) = host.backing_handles()
    assert view.backing_handles() == (backing,)
    resting = host.make_cpu_param()
    assert backing.pin()
    assert resting.data_ptr() == source.data_ptr()
    assert host.make_cpu_param().data_ptr() == source.data_ptr()
    assert host.storage_tensors()[0].data_ptr() == source.data_ptr()
    assert view.tensor.untyped_storage().data_ptr() == source.data_ptr()
    assert backing.evict()
    torch.testing.assert_close(host.make_cpu_param(), source)


def test_independent_captures_do_not_share_copies(tmp_path, monkeypatch):
    source = nn.Parameter(_file_tensor(tmp_path, torch.arange(7.0)), requires_grad=False)
    first = HostParam(source, memory_manager=HostMemoryManager(backend=FakeBackend()))
    second = HostParam(source, memory_manager=HostMemoryManager(backend=FakeBackend()))
    (first_backing,) = first.backing_handles()
    (second_backing,) = second.backing_handles()
    assert first_backing is not second_backing
    assert first_backing.pin()
    observed = _observe_transfers(monkeypatch)
    for host in (first, second):
        target = host.allocate_gpu_storage(torch.device("cpu"))
        host.copy_to_gpu(target)
        torch.testing.assert_close(host.make_gpu_param(target), source)
    assert observed == [first_backing.span[0], source.data_ptr()]
    assert observed[0] != observed[1]


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
    host = HostParam(nn.Parameter(source, requires_grad=False))
    (backing,) = host.backing_handles()
    with pytest.raises(ValueError, match="not captured"):
        host._copy_host(torch.empty_like(source), source.clone(), non_blocking=False)
    with pytest.raises(ValueError, match="does not belong"):
        backing.copy_to(torch.empty_like(source), source.clone(), non_blocking=False)


def test_source_reads_do_not_materialize_or_initialize_cuda(monkeypatch):
    source = torch.arange(7)
    destination = torch.zeros_like(source)
    backings = HostMemoryManager().capture((source,))
    (backing,) = backings.values()

    def unexpected(*args, **kwargs):
        raise AssertionError("source resolution must not allocate or use CUDA")

    monkeypatch.setattr(torch, "empty", unexpected)
    monkeypatch.setattr(torch.cuda, "current_stream", unexpected)
    with backing._read(source) as view:
        assert view is source
    backing.copy_to(destination, source, non_blocking=False)
    assert torch.equal(destination, source)


def test_unpin_and_evict_wait_for_every_reader_and_lease(tmp_path):
    source = _file_tensor(tmp_path, torch.arange(7.0))
    manager = HostMemoryManager(backend=FakeBackend())
    (backing,) = manager.capture((source,)).values()
    assert backing.pin()
    with backing._read(source), backing._read(source):
        assert backing.leases == 2
        assert not backing.unpin()
        assert not backing.evict()
    lease = manager.acquire([backing])
    assert not backing.evict()
    lease.close()
    assert backing.evict()
    assert not backing.pinned and backing.copy_bytes == 0


def test_active_reader_on_another_thread_prevents_unpinning(tmp_path):
    source = _file_tensor(tmp_path, torch.arange(7.0))
    (backing,) = HostMemoryManager(backend=FakeBackend()).capture((source,)).values()
    assert backing.pin()
    acquired = threading.Event()
    release = threading.Event()

    def read():
        with backing._read(source):
            acquired.set()
            assert release.wait(5)

    reader = threading.Thread(target=read)
    reader.start()
    try:
        assert acquired.wait(5)
        assert not backing.unpin()
    finally:
        release.set()
        reader.join(5)
    assert not reader.is_alive()
    assert backing.evict()


def test_eviction_releases_the_copy_while_store_and_cpu_wrapper_live(tmp_path):
    manager = HostMemoryManager(backend=FakeBackend())
    parameter = nn.Parameter(_file_tensor(tmp_path, torch.arange(7.0)), requires_grad=False)
    host = HostParam(parameter, memory_manager=manager)
    (backing,) = host.backing_handles()
    resting = host.make_cpu_param()
    assert backing.pin()
    assert manager.stats.copy_bytes == backing.storage.nbytes()
    assert backing.evict()
    assert manager.stats.copy_bytes == 0
    torch.testing.assert_close(host.make_cpu_param(), resting)


def test_empty_and_meta_parameters():
    empty = HostParam(nn.Parameter(torch.empty(0), requires_grad=False))
    (backing,) = empty.backing_handles()
    assert backing.storage.nbytes() == 0
    with backing._read(empty.storage_tensors()[0]) as view:
        assert view.numel() == 0
    meta = HostParam(nn.Parameter(torch.empty(7, device="meta"), requires_grad=False))
    assert meta.backing_handles() == ()


@CUDA
def test_plain_parameter_and_buffer_copies_read_from_the_copy(tmp_path, monkeypatch):
    module = nn.Linear(4, 3, bias=False).requires_grad_(False)
    module.weight.data = _file_tensor(tmp_path, module.weight.data, "weight.bin")
    module.register_buffer("scale", _file_tensor(tmp_path, torch.arange(3.0), "scale.bin"))
    manager = HostMemoryManager(BUDGET, backend=RecordingBackend())
    store = HostModuleStore.from_module(module, memory_manager=manager)
    plan = store.bind(module).resolve_load_plan()
    expected_weight = module.weight.detach().clone()
    expected_scale = module.scale.clone()
    hosts = (*store.params.values(), *store.buffers.values())
    handles = [backing for host in hosts for backing in host.backing_handles()]
    sources = {backing.storage.data_ptr() for backing in handles}
    observed = _observe_transfers(monkeypatch)
    target = plan.allocate_target(torch.device("cuda"))
    with manager.acquire(handles) as lease:
        assert lease.pinned
        copies = {backing.span[0] for backing in handles}
        assert copies.isdisjoint(sources)
        plan.copy_to_target(target)
        torch.cuda.synchronize()
    assert set(observed) == copies
    torch.testing.assert_close(target.param_targets["weight"].param.cpu(), expected_weight)
    torch.testing.assert_close(target.buffer_targets["scale"].tensor.cpu(), expected_scale)
    manager.clear()
    assert manager.stats.copy_bytes == 0
    observed.clear()
    plan.copy_to_target(target)
    assert set(observed) == sources


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
    manager = HostMemoryManager(BUDGET, backend=RecordingBackend())
    host = HostParam(source_param, memory_manager=manager)
    state = host.host_state
    sources = host.storage_tensors()
    raw_copy = host_backing_module._transfer
    observed = _observe_transfers(monkeypatch)
    target = host.allocate_gpu_storage(torch.device("cuda"))
    with manager.acquire(host.backing_handles()) as lease:
        assert lease.pinned
        pointers = {backing.span[0] for backing in host.backing_handles() if backing.storage.nbytes()}
        host.copy_to_gpu(target, non_blocking=True)
        torch.cuda.synchronize()
    assert observed and set(observed) <= pointers
    assert host.host_state is state
    assert all(a is b for a, b in zip(host.storage_tensors(), sources, strict=True))
    actual = host.make_gpu_param(target)
    manager.clear()
    monkeypatch.setattr(host_backing_module, "_transfer", raw_copy)
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
def test_async_copy_prevents_eviction_until_it_completes(tmp_path, monkeypatch, fail_after_copy):
    source = _file_tensor(tmp_path, torch.arange(1024.0))
    manager = HostMemoryManager(BUDGET, backend=RecordingBackend())
    (backing,) = manager.capture((source,)).values()
    with manager.acquire([backing]):
        assert backing.copy_bytes == source.nbytes  # pinned through a copy; stays after the lease
    destination = torch.empty_like(source, device="cuda")
    stream = torch.cuda.Stream()
    raw_copy = host_backing_module._transfer

    def copy_then_fail(destination, source, *, non_blocking):
        raw_copy(destination, source, non_blocking=non_blocking)
        raise RuntimeError("after enqueue")

    # Warm the path before adding a delay, avoiding lazy initialization waits.
    with torch.cuda.stream(stream):
        backing.copy_to(destination, source, non_blocking=True)
        torch.cuda._sleep(1)
    stream.synchronize()
    if fail_after_copy:
        monkeypatch.setattr(host_backing_module, "_transfer", copy_then_fail)
    # No lease is held: the in-flight copy alone must keep the copy alive.
    with torch.cuda.stream(stream):
        torch.cuda._sleep(500_000_000)
        if fail_after_copy:
            with pytest.raises(RuntimeError, match="after enqueue"):
                backing.copy_to(destination, source, non_blocking=True)
        else:
            backing.copy_to(destination, source, non_blocking=True)
    assert backing.in_flight and not backing.idle
    assert not backing.evict()
    manager.clear()
    assert backing.copy_bytes == source.nbytes
    stream.synchronize()
    assert backing.idle
    manager.clear()
    assert backing.copy_bytes == 0
    torch.testing.assert_close(destination.cpu(), source)


@CUDA
def test_copy_by_an_unleased_caller_survives_another_sessions_lease_closing(tmp_path):
    source = _file_tensor(tmp_path, torch.arange(1024.0))
    manager = HostMemoryManager(BUDGET, backend=RecordingBackend())
    (backing,) = manager.capture((source,)).values()
    destination = torch.empty_like(source, device="cuda")
    stream = torch.cuda.Stream()
    session = manager.acquire([backing])  # some other session's lease
    with torch.cuda.stream(stream):
        backing.copy_to(destination, source, non_blocking=True)
    stream.synchronize()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(500_000_000)
        backing.copy_to(destination, source, non_blocking=True)  # this caller holds no lease
    session.close()
    manager.clear()  # the session is gone; the copy is not
    assert backing.copy_bytes == source.nbytes
    stream.synchronize()
    manager.clear()
    assert backing.copy_bytes == 0
    torch.testing.assert_close(destination.cpu(), source)


@CUDA
def test_existing_target_lease_copies_from_the_copy_and_then_the_source(tmp_path, monkeypatch):
    module = nn.Linear(16, 16, bias=False).requires_grad_(False)
    module.weight.data = _file_tensor(tmp_path, module.weight.data)
    manager = HostMemoryManager(BUDGET, backend=RecordingBackend())
    store = HostModuleStore.from_module(module, memory_manager=manager)
    instance = store.bind(module)
    plan = instance.resolve_load_plan()
    (backing,) = store.params["weight"].backing_handles()
    observed = _observe_transfers(monkeypatch)
    target = CudaTargetLease.allocate(plan, torch.device("cuda"))
    stream = torch.cuda.Stream()
    try:
        for pinned in (True, False):
            lease = manager.acquire([backing]) if pinned else None
            expected_pointer = backing.span[0]
            assert (expected_pointer != backing.storage.data_ptr()) == pinned
            observed.clear()
            target.stage(plan, stream)
            with torch.cuda.stream(stream):
                active = target.acquire(stream)
                actual = active.param_targets["weight"].param.cpu()
            target.release()
            stream.synchronize()
            if lease is not None:
                lease.close()
                manager.clear()
            assert observed == [expected_pointer]
            torch.testing.assert_close(actual, module.weight)
    finally:
        target.close()


@CUDA
@pytest.mark.parametrize("mode", ["streaming", "rolling", "resident"])
def test_runtime_reuses_then_evicts_copies_without_changing_weights(tmp_path, mode, monkeypatch):
    model = _BlockModel(num_blocks=3, width=64)
    for name, parameter in model.named_parameters():
        parameter.data = _file_tensor(tmp_path, parameter.data, f"{name}.bin")
    value = torch.randn(2, 64)
    expected = model(value)
    backend = RecordingBackend()
    manager = HostMemoryManager(BUDGET, backend=backend)
    offloader = _make_offloader(
        model,
        memory_manager=manager,
        block_mode=mode,
        block_compile=BlockCompileConfig(fullgraph=True) if mode == "rolling" else None,
    )
    handles = [
        backing
        for instance in block_components(offloader)[0]._block_instances
        for host in instance.params.values()
        for backing in host.backing_handles()
    ]
    sources = {backing.storage.data_ptr() for backing in handles}
    assert not any(backing.pin_in_place for backing in handles)
    observed = _observe_transfers(monkeypatch)
    try:
        for step in ("pin", "reuse", "evicted"):
            if step == "evicted":
                manager.clear()
                assert manager.stats.copy_bytes == 0
                # Without a finite budget a file-backed weight is not copied
                # again, so the next session reads the mapping itself.
                manager.max_pinned_bytes = None
            observed.clear()
            registrations = len(backend.registrations)
            with activated_model(offloader, "cuda"), torch.inference_mode():
                copies = {backing.span[0] for backing in handles}
                actual = model(value.cuda()).cpu()
            torch.testing.assert_close(actual, expected)
            if mode == "resident":
                # Resident blocks copy once at activation without a host lease.
                assert set(observed) == sources
                continue
            if step == "evicted":
                assert set(observed) == sources
            else:
                assert copies.isdisjoint(sources)
                assert set(observed) == copies
                assert len(backend.registrations) == registrations + (len(handles) if step == "pin" else 0)
    finally:
        offloader.deactivate()
        if mode == "rolling":
            torch.compiler.reset()


@CUDA
def test_copy_from_an_evictable_copy_rejects_graph_capture(tmp_path):
    source = _file_tensor(tmp_path, torch.ones(16))
    manager = HostMemoryManager(BUDGET, backend=RecordingBackend())
    (backing,) = manager.capture((source,)).values()
    destination = torch.empty_like(source, device="cuda")
    stream = torch.cuda.Stream()
    with manager.acquire([backing]):
        assert backing.copy_bytes == source.nbytes
        with torch.cuda.stream(stream):
            backing.copy_to(destination, source, non_blocking=True)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            with pytest.raises(RuntimeError, match="cannot be captured"):
                backing.copy_to(destination, source, non_blocking=True)
            # Include ordinary device work so capture is nonempty.
            destination.add_(1)
    manager.clear()


def test_backing_handle_does_not_retain_its_manager():
    manager = HostMemoryManager()
    manager_ref = weakref.ref(manager)
    source = torch.arange(16.0)
    backings = manager.capture((source,))
    (backing,) = backings.values()
    del manager, backings
    gc.collect()
    assert manager_ref() is None
    destination = torch.empty_like(source)
    backing.copy_to(destination, source, non_blocking=False)
    torch.testing.assert_close(destination, source)


def test_disposing_the_owner_unregisters_and_frees_the_copy(tmp_path):
    backend = FakeBackend()
    manager = HostMemoryManager(backend=backend)
    host = HostParam(nn.Parameter(_file_tensor(tmp_path, torch.arange(8.0))), memory_manager=manager)
    (backing,) = host.backing_handles()
    assert backing.pin()
    copy_pointer = backing.span[0]
    source_ref = weakref.ref(host.storage_tensors()[0].untyped_storage())
    del backing, host
    gc.collect()
    assert source_ref() is None
    assert backend.unregister_calls == [copy_pointer]
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
        assert all(manager._backings.get(handle.storage._cdata) is handle for handle in host.backing_handles())
