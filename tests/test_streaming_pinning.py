"""Host registration across streaming sessions and asynchronous copies."""

import mmap

import pytest
import torch
from torch import nn

import piper_offload.block_component as block_component_module
from piper_offload import BlockCompileConfig, BlockComponentStore, ModelOffloader, ParameterValue, PinManager
from piper_offload._host_registration import RuntimeHostRegistration
from piper_offload.block_component import _host_transfer_tensors
from piper_offload.host_module import HostModuleStore, ParameterOverride
from piper_offload.host_param import HostParam
from piper_offload.target_lease import CudaTargetLease
from piper_offload.tensor_adapter_registry import param_representation, select_adapter
from tests._block_compile_helpers import _BlockModel, _make_offloader
from tests.conftest import activated_model, block_components
from tests.test_quantized_parameter_value import _QUANT_KINDS, _make_quantized

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")


class RecordingBackend(RuntimeHostRegistration):
    def __init__(self) -> None:
        self.registrations: list[tuple[int, int]] = []
        self.unregistrations: list[int] = []

    def register(self, pointer: int, size: int) -> bool:
        registered = super().register(pointer, size)
        if registered:
            self.registrations.append((pointer, size))
        return registered

    def unregister(self, pointer: int) -> None:
        super().unregister(pointer)
        self.unregistrations.append(pointer)


@pytest.fixture
def pins(monkeypatch: pytest.MonkeyPatch):
    backend = RecordingBackend()
    manager = PinManager(64 * 1024**2, backend=backend)
    monkeypatch.setattr(block_component_module, "host_pin_manager", manager)
    yield manager, backend
    manager.clear()
    assert manager.stats.active_leases == 0
    assert manager.stats.pinned_bytes == 0


@CUDA
@pytest.mark.parametrize("mode", ["streaming", "rolling"])
def test_reactivation_reuses_pins_and_preserves_results(mode: str, pins) -> None:
    manager, backend = pins
    model = _BlockModel(num_blocks=3, width=64)
    value = torch.randn(2, 64)
    expected = model(value)
    offloader = _make_offloader(
        model,
        block_mode=mode,
        block_compile=BlockCompileConfig(fullgraph=True) if mode == "rolling" else None,
    )
    try:
        for _ in range(2):
            with activated_model(offloader, "cuda"), torch.inference_mode():
                assert manager.stats.active_leases == 1
                assert manager.stats.idle_registrations == 0
                assert len(backend.registrations) == 3
                actual = model(value.cuda()).cpu()
            assert manager.stats.active_leases == 0
            assert manager.stats.idle_registrations == 3
            assert not backend.unregistrations
            torch.testing.assert_close(actual, expected)
    finally:
        offloader.deactivate()
        if mode == "rolling":
            torch.compiler.reset()


@CUDA
def test_working_set_release_also_returns_pins_to_idle_lru(pins) -> None:
    manager, backend = pins
    offloader = _make_offloader(_BlockModel())
    component = block_components(offloader)[0]
    with activated_model(offloader, "cuda"):
        component.release()
        assert manager.stats.active_leases == 0
        assert manager.stats.idle_registrations == 2
        count = len(backend.registrations)
        component.acquire()
        assert manager.stats.active_leases == 1
        assert len(backend.registrations) == count
        assert not backend.unregistrations


@CUDA
@pytest.mark.parametrize("budget", [0, 5 * mmap.PAGESIZE])
def test_budget_fallback_keeps_streamed_results_correct(budget: int, pins) -> None:
    manager, backend = pins
    manager.max_pinned_bytes = budget
    model = _BlockModel(num_blocks=3, width=64)
    value = torch.randn(2, 64)
    expected = model(value)
    offloader = _make_offloader(model)
    with activated_model(offloader, "cuda"), torch.inference_mode():
        lease = block_components(offloader)[0]._pin_lease
        assert lease is not None
        assert lease.pageable_bytes > 0
        assert (lease.registered_bytes > 0) == (budget > 0)
        assert manager.stats.pinned_bytes <= budget
        actual = model(value.cuda()).cpu()
    torch.testing.assert_close(actual, expected)
    if budget == 0:
        assert not backend.registrations


@CUDA
def test_another_component_evicts_only_released_registrations(pins) -> None:
    manager, backend = pins
    first = _make_offloader(_BlockModel(num_blocks=2, width=64))
    second = _make_offloader(_BlockModel(num_blocks=2, width=64))
    try:
        first.activate("cuda")
        manager.max_pinned_bytes = manager.stats.pinned_bytes
        second.activate("cuda")
        assert not backend.unregistrations
        lease = block_components(second)[0]._pin_lease
        assert lease is not None and lease.pageable_bytes > 0
        second.deactivate()
        first.deactivate()
        second.activate("cuda")
        assert backend.unregistrations
        lease = block_components(second)[0]._pin_lease
        assert lease is not None and lease.registered_bytes > 0
    finally:
        second.deactivate()
        first.deactivate()


@pytest.mark.parametrize(("device", "mode"), [("cpu", "streaming"), ("cuda", "resident"), ("cuda", "host")])
def test_cpu_and_resident_execution_do_not_acquire_pins(device: str, mode: str, pins, monkeypatch) -> None:
    manager, _backend = pins

    def unexpected_acquire(_tensors):
        raise AssertionError("this execution mode must not acquire host pins")

    monkeypatch.setattr(manager, "acquire", unexpected_acquire)
    offloader = ModelOffloader.from_module(
        _BlockModel(),
        block_paths=[] if mode == "host" else ["blocks"],
        block_mode="resident" if mode == "resident" else "streaming",
    )
    with activated_model(offloader, device):
        assert manager.stats.active_leases == 0


def test_transfer_sources_include_buffers_and_optimizer_backing() -> None:
    module = nn.Module()
    module.frozen = nn.Parameter(torch.randn(4, 4), requires_grad=False)
    module.trainable = nn.Parameter(torch.randn(4, 4))
    module.register_buffer("buffer", torch.ones(4))
    module.register_buffer("alias", module.buffer)
    store = HostModuleStore.from_module(module)
    frozen = HostParam(nn.Parameter(torch.randn(4, 4), requires_grad=False))
    trainable = HostParam(nn.Parameter(torch.randn(4, 4)))
    plan = store.bind(module).resolve_load_plan({
        "frozen": ParameterOverride(source=frozen),
        "trainable": ParameterOverride(source=trainable),
    })
    tensors = list(_host_transfer_tensors([plan]))
    identities = {id(tensor) for tensor in tensors}
    assert id(store.params["frozen"].storage_tensors()[0]) not in identities
    for host in (frozen, trainable, store.params["trainable"], store.buffers["buffer"]):
        assert all(id(tensor) in identities for tensor in host.storage_tensors())


@CUDA
@pytest.mark.parametrize("kind", _QUANT_KINDS)
def test_quantized_replacements_pin_payload_and_metadata_without_conversion(kind: str, pins) -> None:
    manager, backend = pins
    source = _make_quantized(kind)
    expected = select_adapter(source).dequantize(source).cpu()
    backing = ParameterValue.from_tensor(source).backing
    block = nn.Module()
    block.weight = nn.Parameter(torch.empty(backing.logical_shape, device="meta"), requires_grad=False)
    block.register_buffer("buffer", torch.arange(8, dtype=torch.float32))
    block.register_buffer("alias", block.buffer)
    model = nn.Module()
    model.blocks = nn.ModuleList([block])
    component = BlockComponentStore.from_module(model, blocks_path="blocks").bind(model)
    tensors = backing.storage_tensors()
    pointers = {tensor.untyped_storage().data_ptr() for tensor in tensors if tensor.numel()}
    try:
        component.activate(torch.device("cuda"), parameter_overrides={
            "blocks.0.weight": ParameterOverride(source=backing),
        })
        assert pointers <= {pointer for pointer, _size in backend.registrations}
        assert manager.stats.registrations == len(pointers) + 1  # tied buffer shares one registration
        assert all(tensor.is_pinned() for tensor in tensors if tensor.numel())
        target = param_representation(block.weight)
        adapter = select_adapter(target)
        assert type(adapter) is type(backing.adapter)
        actual = adapter.dequantize(target).cpu()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(block.buffer.cpu(), torch.arange(8, dtype=torch.float32))
    finally:
        component.deactivate()
    assert block.weight.is_meta
    assert pointers == {tensor.untyped_storage().data_ptr() for tensor in tensors if tensor.numel()}
    assert not backend.unregistrations


@CUDA
@pytest.mark.parametrize("mode", ["streaming", "rolling"])
def test_partial_activation_failure_releases_host_lease(mode: str, pins, monkeypatch) -> None:
    manager, backend = pins
    offloader = _make_offloader(
        _BlockModel(),
        block_mode=mode,
        block_compile=BlockCompileConfig(fullgraph=True) if mode == "rolling" else None,
    )
    runtime = block_components(offloader)[0]._runtime

    def fail_hooks(*_args):
        raise RuntimeError("injected hook failure after upload")

    with monkeypatch.context() as patch:
        patch.setattr(runtime, "_register_hooks", fail_hooks)
        with pytest.raises(RuntimeError, match="injected hook failure"):
            offloader.activate("cuda")
    assert manager.stats.active_leases == 0
    assert manager.stats.idle_registrations == 2
    count = len(backend.registrations)
    with activated_model(offloader, "cuda"):
        assert len(backend.registrations) == count


@CUDA
@pytest.mark.parametrize("mode", ["streaming", "rolling"])
def test_failed_runtime_quiescence_does_not_release_host_lease(
    mode: str,
    pins,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _backend = pins
    offloader = _make_offloader(
        _BlockModel(),
        block_mode=mode,
        block_compile=BlockCompileConfig(fullgraph=True) if mode == "rolling" else None,
    )
    component = block_components(offloader)[0]
    original_close = CudaTargetLease.close
    failed = False

    def fail_once(lease: CudaTargetLease) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected synchronization failure")
        original_close(lease)

    try:
        offloader.activate("cuda")
        runtime = component._active_runtime
        assert runtime is not None
        with monkeypatch.context() as patch:
            patch.setattr(CudaTargetLease, "close", fail_once)
            with pytest.raises(RuntimeError, match="injected synchronization failure"):
                offloader.deactivate()
        assert runtime.acquired
        assert component._active_device is None
        assert component._active_runtime is None
        assert component._pin_lease is not None
        assert manager.stats.active_leases == 1
        with pytest.raises(RuntimeError, match="Recreate the CUDA worker"):
            component.activate(torch.device("cuda"))

        runtime.release()
        component._pin_lease.close()
        component._pin_lease = None
    finally:
        offloader.deactivate()
        if mode == "rolling":
            torch.compiler.reset()


@CUDA
def test_prefetch_update_failure_waits_before_releasing_pins(pins) -> None:
    manager, backend = pins
    model = _BlockModel()
    component = BlockComponentStore.from_module(model, blocks_path="blocks").bind(model)

    def fail_update(_parameter):
        raise RuntimeError("injected update failure after upload")

    component.activate(torch.device("cuda"), parameter_overrides={
        "blocks.1.proj.weight": ParameterOverride(update=fail_update),
    })
    stream = component._runtime._stream
    try:
        with torch.inference_mode():
            model.blocks[0](torch.randn(2, 8, device="cuda"))
        manager.max_pinned_bytes = 0
        assert not backend.unregistrations
        with pytest.raises(RuntimeError, match="injected update failure"):
            component.deactivate()
        assert stream.query()
        assert manager.stats.active_leases == 0
        assert manager.stats.pinned_bytes == 0
    finally:
        component.deactivate()


@CUDA
def test_trainable_copy_back_preserves_registered_host_storage(pins) -> None:
    manager, backend = pins
    model = _BlockModel().requires_grad_(True)
    expected = [parameter.detach().clone() + 1 for parameter in model.parameters()]
    component = BlockComponentStore.from_module(
        model, blocks_path="blocks", include_block_trainables=True,
    ).bind(model)
    try:
        component.activate(torch.device("cuda"))
        count = len(backend.registrations)
        with component.optimizer_step(), torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
        assert manager.stats.active_leases == 1
        assert len(backend.registrations) == count
    finally:
        component.deactivate()
    for parameter, updated in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter, updated)
        assert parameter.is_pinned()
