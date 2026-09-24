"""Host registration across CUDA sessions and asynchronous copies."""

import mmap

import pytest
import torch
from torch import nn

import piper_offload.block_component as block_component_module
import piper_offload.host_component as host_component_module
import piper_offload.merge as merge_module
import piper_offload.model_offloader as model_offloader_module
import piper_offload.pin_manager as pin_module
from piper_offload import (
    Adapter,
    BlockCompileConfig,
    BlockComponentStore,
    HostComponentStore,
    LoRATransform,
    MappedCheckpoint,
    ModelOffloader,
    ParameterDelta,
    merge_adapter,
    ParameterDeltaTransform,
    ParameterValue,
    PinManager,
    ScaledLoRAFactor,
)
from piper_offload._host_registration import RuntimeHostRegistration
from piper_offload.host_module import HostModuleInstance, HostModuleStore, ParameterOverride
from piper_offload.host_param import HostParam
from piper_offload.target_lease import CudaTargetLease
from piper_offload.tensor_adapter_registry import param_representation, select_adapter
from tests._block_compile_helpers import _BlockModel, _make_offloader
from tests.conftest import CallbackParameterTransform, activated_model, block_components
from tests.test_pin_manager import install_fake_kernel
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
    monkeypatch.setattr(host_component_module, "host_pin_manager", manager)
    monkeypatch.setattr(pin_module, "host_pin_manager", manager)
    monkeypatch.setattr(model_offloader_module, "host_pin_manager", manager)
    monkeypatch.setattr(merge_module, "host_pin_manager", manager)
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


@pytest.mark.parametrize(
    ("device", "mode"),
    [
        ("cpu", "streaming"),
        ("cpu", "resident"),
        ("cpu", "host"),
    ],
)
def test_cpu_execution_does_not_acquire_pins(device: str, mode: str, pins, monkeypatch) -> None:
    manager, _backend = pins

    def unexpected_acquire(_tensors, **_kwargs):
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
    tensors = list(plan.storage_tensors())
    identities = {id(tensor) for tensor in tensors}
    assert id(store.params["frozen"].storage_tensors()[0]) not in identities
    for host in (frozen, trainable, store.params["trainable"], store.buffers["buffer"]):
        assert all(id(tensor) in identities for tensor in host.storage_tensors())


def test_transfer_sources_include_parameter_transform_backing() -> None:
    module = nn.Linear(8, 4, bias=False)
    module.requires_grad_(False)
    instance = HostModuleStore.from_module(module).bind(module)
    factor = ScaledLoRAFactor.from_tensors(
        torch.randn(2, 8),
        torch.randn(4, 2),
        0.5,
    )
    delta = ParameterDelta.from_tensors(
        a=torch.randn(2, 8),
        b=torch.randn(4, 2),
        dense=torch.randn(4, 8),
    )
    cases = (
        (
            LoRATransform([factor]),
            (*factor.a.storage_tensors(), *factor.b.storage_tensors()),
        ),
        (
            ParameterDeltaTransform([delta.scaled(0.25)]),
            (
                *delta.lora.a.storage_tensors(),
                *delta.lora.b.storage_tensors(),
                *delta.dense.storage_tensors(),
            ),
        ),
    )

    for transform, expected in cases:
        plan = instance.resolve_load_plan(
            {"weight": ParameterOverride(update=transform)}
        )
        identities = {id(tensor) for tensor in plan.storage_tensors()}
        assert all(id(tensor) in identities for tensor in expected)


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
        "blocks.1.proj.weight": ParameterOverride(
            update=CallbackParameterTransform(fail_update)
        ),
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


def _resident_component(model: nn.Module, mode: str):
    if mode == "host":
        return HostComponentStore.from_module(model).bind(model)
    return BlockComponentStore.from_module(
        model, blocks_path="blocks", include_block_trainables=True,
    ).bind(model, block_mode="resident")


@CUDA
@pytest.mark.parametrize("mode", ["resident", "host"])
@pytest.mark.parametrize("budget", [0, 64 * 1024**2])
def test_resident_and_host_uploads_lease_without_pinning(mode, budget, pins, monkeypatch) -> None:
    manager, backend = pins
    manager.max_pinned_bytes = budget
    model = _BlockModel()
    inputs = torch.randn(2, 8)
    expected = model(inputs)
    component = _resident_component(model, mode)
    original_stage = CudaTargetLease.stage
    staged: list[torch.cuda.Stream] = []

    def checked_stage(lease, plan, stream, **kwargs):
        # The upload runs under a lease that registers nothing.
        assert manager.stats.active_leases == 1
        assert backend.registrations == []
        assert not any(tensor.is_pinned() for tensor in plan.storage_tensors())
        original_stage(lease, plan, stream, **kwargs)
        staged.append(stream)

    monkeypatch.setattr(CudaTargetLease, "stage", checked_stage)
    try:
        component.activate(torch.device("cuda"))
        assert staged and all(stream.query() for stream in staged)
        assert manager.stats.active_leases == 0
        assert backend.registrations == []
        torch.testing.assert_close(model(inputs.cuda()).cpu(), expected)
        component.release()
        component.acquire()
        assert backend.registrations == []
        assert manager.stats.active_leases == 0
    finally:
        component.deactivate()


@CUDA
@pytest.mark.parametrize("mode", ["resident", "host"])
@pytest.mark.parametrize("fail_sync", [False, True])
def test_partial_upload_keeps_its_lease_until_cleanup_synchronizes(mode, fail_sync, pins, monkeypatch) -> None:
    manager, _backend = pins
    component = _resident_component(_BlockModel(), mode)
    original_stage = CudaTargetLease.stage
    original_sync = torch.cuda.synchronize
    device = torch.device("cuda", torch.cuda.current_device())

    def fail_after_upload(lease, plan, stream, **kwargs):
        original_stage(lease, plan, stream, **kwargs)
        assert manager.stats.active_leases == 1
        raise RuntimeError("injected upload failure")

    def failed_sync(*_args, **_kwargs):
        raise RuntimeError("injected synchronization failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(CudaTargetLease, "stage", fail_after_upload)
            if fail_sync:
                patch.setattr(torch.cuda, "synchronize", failed_sync)
            with pytest.raises(RuntimeError, match="injected"):
                component.activate(torch.device("cuda"))
            if fail_sync:
                # The upload may still be in flight: protection stays until a
                # synchronization succeeds.
                assert manager.stats.active_leases == 1
                with pytest.raises(RuntimeError, match="synchronization failure"):
                    component.release()
                assert manager.stats.active_leases == 1
                with pytest.raises(RuntimeError, match="synchronization failure"):
                    component.deactivate()
                assert component._active_device is None
                # The old target and transfer are still open; a new session
                # must not silently reuse them.
                with pytest.raises(RuntimeError, match="Recreate the CUDA worker"):
                    component.activate(torch.device("cuda"))
                assert manager.stats.active_leases == 1

        def checked_sync(sync_device):
            # Deactivation cleared the session, but the retry must still
            # synchronize the original transfer device, not the current one.
            assert sync_device == device
            original_sync(sync_device)

        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "synchronize", checked_sync)
            if mode == "host" or not fail_sync:
                component.release()
            else:
                # A block component has no session left to retry through; the
                # cleanup below is what recreating the worker would do.
                component._runtime.release()
                component._transfer.finish()
        assert manager.stats.active_leases == 0
    finally:
        component.deactivate()


@CUDA
def test_resident_copy_back_closes_at_the_runtime_completion_point(pins, monkeypatch) -> None:
    manager, _backend = pins
    model = _BlockModel().requires_grad_(True)
    expected = [parameter.detach().clone() + 1 for parameter in model.parameters()]
    component = _resident_component(model, "resident")

    def failed_sync(*_args, **_kwargs):
        raise RuntimeError("a device-wide synchronization is not needed here")

    try:
        component.activate(torch.device("cuda"))
        with monkeypatch.context() as patch:
            # The runtime synchronizes its own stream after the copy-back, which
            # is the completion point the lease closes at.
            patch.setattr(torch.cuda, "synchronize", failed_sync)
            with component.optimizer_step(), torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1)
        assert manager.stats.active_leases == 0
    finally:
        component.deactivate()
    for parameter, updated in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter, updated)


@CUDA
def test_failed_deactivation_refuses_resident_reactivation(pins, monkeypatch) -> None:
    manager, _backend = pins
    component = _resident_component(_BlockModel(), "resident")
    runtime = component._runtime

    def failed_sync(*_args, **_kwargs):
        raise RuntimeError("injected synchronization failure")

    try:
        component.activate(torch.device("cuda"))
        assert manager.stats.active_leases == 0
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "synchronize", failed_sync)
            with pytest.raises(RuntimeError, match="injected"):
                component.deactivate()
        assert runtime.acquired
        assert component._active_device is None
        # The runtime still holds the old targets and load plan; a new session
        # must not silently reuse them.
        with pytest.raises(RuntimeError, match="Recreate the CUDA worker"):
            component.activate(torch.device("cuda"))
    finally:
        runtime.release()
        component.deactivate()


@CUDA
@pytest.mark.parametrize("mode", ["resident", "host"])
def test_frozen_optimizer_step_takes_no_transfer_lease(mode, pins, monkeypatch) -> None:
    manager, _backend = pins
    component = _resident_component(_BlockModel().requires_grad_(False), mode)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("a frozen model has nothing to copy back")

    try:
        component.activate(torch.device("cuda"))
        with monkeypatch.context() as patch:
            patch.setattr(manager, "acquire", unexpected)
            patch.setattr(torch.cuda, "synchronize", unexpected)
            with component.optimizer_step():
                pass
    finally:
        component.deactivate()


@CUDA
@pytest.mark.parametrize("mode", ["resident", "host"])
@pytest.mark.parametrize("failure", [None, "copy", "sync"])
def test_optimizer_copy_back_has_its_own_pageable_lease(mode, failure, pins, monkeypatch) -> None:
    manager, backend = pins
    model = _BlockModel().requires_grad_(True)
    expected = [parameter.detach().clone() + 1 for parameter in model.parameters()]
    component = _resident_component(model, mode)
    original_copy = HostModuleInstance.copy_trainables_from_target
    copied = False

    def checked_copy(instance, target, **kwargs):
        nonlocal copied
        assert manager.stats.active_leases == 1
        assert backend.registrations == []
        assert not any(tensor.is_pinned() for tensor in instance.trainable_storage_tensors())
        original_copy(instance, target, **kwargs)
        copied = True
        if failure == "copy":
            raise RuntimeError("injected copy-back failure")

    def failed_sync(*_args, **_kwargs):
        raise RuntimeError("injected synchronization failure")

    def step():
        with component.optimizer_step(), torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)

    try:
        component.activate(torch.device("cuda"))
        assert manager.stats.active_leases == 0
        with monkeypatch.context() as patch:
            patch.setattr(HostModuleInstance, "copy_trainables_from_target", checked_copy)
            if failure == "sync":
                # Both the runtime's stream synchronization and the device-wide retry fail.
                patch.setattr(torch.cuda, "synchronize", failed_sync)
                patch.setattr(torch.cuda.Stream, "synchronize", failed_sync)
            if failure is None:
                step()
            else:
                with pytest.raises(RuntimeError, match="injected"):
                    step()
            assert copied
            assert manager.stats.active_leases == (1 if failure == "sync" else 0)
        component.release()
        assert manager.stats.active_leases == 0
        assert backend.registrations == []
    finally:
        component.deactivate()
    if failure != "copy":
        for parameter, updated in zip(model.parameters(), expected, strict=True):
            torch.testing.assert_close(parameter, updated)


@CUDA
@pytest.mark.parametrize("mode", ["streaming", "rolling"])
def test_mapped_checkpoint_streams_from_owned_copies(mode, pins, tmp_path, monkeypatch) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    manager, backend = pins
    model = _BlockModel(width=64)
    inputs = torch.randn(2, 64)
    expected = model(inputs)
    path = tmp_path / "model.safetensors"
    safetensors.save_file(model.state_dict(), str(path))
    reader = MappedCheckpoint(path)
    names = reader.keys()
    sources = {name: reader.get_tensor(name) for name in names}
    model.load_state_dict(sources, assign=True)
    mapping_pointers = {tensor.untyped_storage().data_ptr() for tensor in sources.values()}
    offloader = _make_offloader(
        model,
        block_mode=mode,
        block_compile=BlockCompileConfig(fullgraph=True) if mode == "rolling" else None,
    )
    resolved: list[torch.Tensor] = []
    original_view = pin_module._Copy.view

    def recording(copy: pin_module._Copy, tensor: torch.Tensor) -> torch.Tensor:
        view = original_view(copy, tensor)
        resolved.append(view)
        return view

    monkeypatch.setattr(pin_module._Copy, "view", recording)
    try:
        with activated_model(offloader, "cuda"), torch.inference_mode():
            actual = offloader.value(inputs.cuda()).cpu()
        torch.testing.assert_close(actual, expected)
        registered = {pointer for pointer, _size in backend.registrations}
        # The mappings are never registered; each storage has one page-aligned copy.
        assert registered.isdisjoint(mapping_pointers)
        assert len(registered) == len(mapping_pointers)
        assert all(pointer % mmap.PAGESIZE == 0 for pointer in registered)
        assert all(not tensor.is_pinned() for tensor in sources.values())
        # Every storage was transferred from its copy, not from the mapping.
        assert {view.untyped_storage().data_ptr() for view in resolved} == registered
        assert manager.stats.copy_bytes == manager.stats.pinned_bytes > 0
        manager.clear()
        assert manager.stats.pinned_bytes == 0
    finally:
        offloader.deactivate()
        if mode == "rolling":
            torch.compiler.reset()


@CUDA
@pytest.mark.parametrize("mode", ["resident", "streaming"])
def test_routed_lora_factors_are_leased_while_their_hooks_are_installed(mode, pins) -> None:
    manager, backend = pins
    model = _BlockModel()
    state = {}
    for index in range(2):
        state[f"blocks.{index}.proj.lora_A.weight"] = torch.randn(2, 8)
        state[f"blocks.{index}.proj.lora_B.weight"] = torch.randn(8, 2)
    adapter = Adapter.from_state_dict(state)
    factors = [
        tensor
        for target in adapter.targets.values()
        for host in (target.lora.a, target.lora.b)
        for tensor in host.storage_tensors()
    ]
    # Pin the factors elsewhere and leave them idle: staging them on every
    # forward is now a read of registered storage, which needs a lease.
    with manager.acquire(factors):
        pass
    assert manager.stats.idle_registrations == len(factors)
    registered = len(backend.registrations)
    offloader = _make_offloader(model, block_mode=mode)
    inputs = torch.randn(2, 8)
    try:
        offloader.activate("cuda", adapters=[adapter], adapter_strengths=[0.5], adapter_mode="routed")
        assert manager.stats.idle_registrations == 0
        expected_leases = 2 if mode == "streaming" else 1
        assert manager.stats.active_leases == expected_leases
        offloader.value(inputs.cuda())
        # Streaming registers its two block weights; the factors are not registered again.
        assert len(backend.registrations) == registered + (2 if mode == "streaming" else 0)
    finally:
        offloader.deactivate()
    assert manager.stats.active_leases == 0
    assert manager.stats.idle_registrations == len(factors) + (2 if mode == "streaming" else 0)
    with pytest.raises(RuntimeError, match="outside a lease"):
        adapter.targets["blocks.0.proj.weight"].lora.a.materialize(torch.device("cuda"), non_blocking=True)


def _routed_adapter() -> Adapter:
    state = {}
    for index in range(2):
        state[f"blocks.{index}.proj.lora_A.weight"] = torch.randn(2, 8)
        state[f"blocks.{index}.proj.lora_B.weight"] = torch.randn(8, 2)
    return Adapter.from_state_dict(state)


def _factor_tensors(adapter: Adapter) -> list[torch.Tensor]:
    return [
        tensor
        for target in adapter.targets.values()
        for host in (target.lora.a, target.lora.b)
        for tensor in host.storage_tensors()
    ]


def test_cpu_routed_activation_takes_no_lease(pins) -> None:
    manager, _backend = pins
    adapter = _routed_adapter()
    offloader = _make_offloader(_BlockModel(), block_mode="streaming")
    try:
        offloader.activate("cpu", adapters=[adapter], adapter_strengths=[0.5], adapter_mode="routed")
        assert manager.stats.active_leases == 0
        offloader.value(torch.randn(2, 8))
    finally:
        offloader.deactivate()


@CUDA
def test_permanent_merge_on_a_cuda_model_leases_pinned_factors(pins) -> None:
    manager, backend = pins
    adapter = _routed_adapter()
    with manager.acquire(_factor_tensors(adapter)):
        pass
    registered = len(backend.registrations)
    model = _BlockModel()
    expected = {}
    for name, target in adapter.targets.items():
        a, b = (host.make_cpu_param().data for host in (target.lora.a, target.lora.b))
        expected[name] = model.get_parameter(name).detach() + 0.5 * (b @ a)
    model.cuda()
    assert merge_adapter(model, [(adapter, 0.5)]) == 2
    assert manager.stats.active_leases == 0
    assert len(backend.registrations) == registered
    for name, weight in expected.items():
        torch.testing.assert_close(model.get_parameter(name).detach().cpu(), weight)


@CUDA
def test_permanent_merge_leases_sources_through_validation(pins, monkeypatch) -> None:
    from piper_offload import transfer_

    manager, _backend = pins
    adapter = _routed_adapter()
    with manager.acquire(_factor_tensors(adapter)):
        pass
    original_validate = merge_module._MergeOp.validate

    def staging_validate(op) -> None:
        # Quantized targets stage their sources while validating.
        original_validate(op)
        for tensor in op.transform.storage_tensors():
            transfer_(torch.empty_like(tensor, device="cuda"), tensor, non_blocking=True)

    monkeypatch.setattr(merge_module._MergeOp, "validate", staging_validate)
    model = _BlockModel().cuda()
    assert merge_adapter(model, [(adapter, 0.5)]) == 2
    assert manager.stats.active_leases == 0


@CUDA
def test_failed_synchronization_keeps_the_routed_lease(pins, monkeypatch) -> None:
    manager, _backend = pins
    adapter = _routed_adapter()
    offloader = _make_offloader(_BlockModel(), block_mode="resident")
    offloader.activate("cuda", adapters=[adapter], adapter_strengths=[0.5], adapter_mode="routed")
    offloader.value(torch.randn(2, 8).cuda())

    def failed_sync(*_args, **_kwargs):
        raise RuntimeError("injected synchronization failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "synchronize", failed_sync)
            with pytest.raises(RuntimeError, match="injected"):
                offloader.deactivate()
        # Staging may still be in flight: the routed factors stay leased, and
        # a new session must not silently take over.
        assert manager.stats.active_leases == 1
        with pytest.raises(RuntimeError, match="Recreate the CUDA worker"):
            offloader.activate("cuda", adapters=[adapter], adapter_strengths=[0.5], adapter_mode="routed")
        # The refused activation's cleanup synchronized the components for real,
        # which is what finally lets the routed lease close.
        assert manager.stats.active_leases == 0
    finally:
        offloader.deactivate()


@CUDA
@pytest.mark.parametrize("reclaim_code", [0, 170], ids=["intact", "discarded"])
def test_rotating_mapped_models_streams_from_offered_copies(reclaim_code, pins, tmp_path, monkeypatch) -> None:
    """Two streamed checkpoints over a budget that holds one: each switch offers a copy and takes the other back."""
    safetensors = pytest.importorskip("safetensors.torch")
    manager, backend = pins
    kernel = install_fake_kernel(monkeypatch, [])
    kernel.reclaim_code = reclaim_code
    rotation = []
    for name in ("first", "second"):
        model = _BlockModel(num_blocks=2, width=64)
        inputs = torch.randn(2, 64)
        expected = model(inputs)
        path = tmp_path / f"{name}.safetensors"
        safetensors.save_file(model.state_dict(), str(path))
        reader = MappedCheckpoint(path)
        names = reader.keys()
        model.load_state_dict({name: reader.get_tensor(name) for name in names}, assign=True)
        rotation.append((_make_offloader(model), inputs, expected))
    try:
        for round_index in range(3):
            for offloader, inputs, expected in rotation:
                with activated_model(offloader, "cuda"), torch.inference_mode():
                    actual = offloader.value(inputs.cuda()).cpu()
                    if round_index == 0:
                        # One model's copies are the whole budget, so activating
                        # the other must evict, and therefore offer, them.
                        manager.max_pinned_bytes = manager.stats.copy_bytes
                        manager.max_offered_bytes = manager.stats.copy_bytes
                torch.testing.assert_close(actual, expected)
        assert manager.stats.offered_bytes == manager.max_offered_bytes > 0
        counts = {name: sum(event == name for event, _pointer in kernel.events) for name in ("offer", "reclaim")}
        # Every offer but the ones still held was taken back by the next activation.
        assert counts["offer"] == counts["reclaim"] + len(manager._offered) > 0
        # Each storage's copy was built once and reused from the tier thereafter.
        copies = manager.stats.registrations + len(manager._offered)
        assert sum(event == "allocate" for event, _pointer in kernel.events) == copies
    finally:
        for offloader, _inputs, _expected in rotation:
            offloader.deactivate()
