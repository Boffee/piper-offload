"""Host registration across CUDA sessions and asynchronous copies."""

import mmap

import pytest
import torch
from torch import nn

import piper_offload._copy_memory as copy_memory
import piper_offload.merge as merge_module
import piper_offload.pin_manager as pin_module
from piper_offload import (
    Adapter,
    BlockCompileConfig,
    MappedCheckpoint,
    ModelOffloader,
    OffloadConfig,
    ParameterDelta,
    merge_adapter,
    ParameterValue,
    PinManager,
    ScaledLoRAFactor,
)
from piper_offload._host_registration import RuntimeHostRegistration
from piper_offload._parameter_update import build_parameter_transform
from piper_offload.host_module import HostModuleStore, ParameterOverride
from piper_offload.host_param import HostParam
from piper_offload.lora import LoRATransform
from piper_offload.target_lease import CudaTargetLease
from piper_offload.tensor_adapter_registry import param_representation, select_adapter
from tests._block_compile_helpers import _BlockModel, _make_offloader
from tests.conftest import CallbackParameterTransform, activated_model, residency_engine
from tests.test_pin_manager import install_fake_kernel
from tests.test_quantized_parameter_value import _QUANT_KINDS, _make_quantized, _to_dense

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
    monkeypatch.setattr(pin_module, "host_pin_manager", manager)
    monkeypatch.setattr(merge_module, "host_pin_manager", manager)
    yield manager, backend
    manager.clear()
    assert manager.stats.active_leases == 0
    assert manager.stats.pinned_bytes == 0


@CUDA
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_reactivation_reuses_pins_and_preserves_results(compiled: bool, pins) -> None:
    manager, backend = pins
    model = _BlockModel(num_blocks=3, width=64)
    value = torch.randn(2, 64)
    expected = model(value)
    offloader = _make_offloader(
        model,
        block_compile=BlockCompileConfig(fullgraph=True) if compiled else None,
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
        if compiled:
            torch.compiler.reset()


@CUDA
def test_working_set_release_also_returns_pins_to_idle_lru(pins) -> None:
    manager, backend = pins
    offloader = _make_offloader(_BlockModel())
    engine = residency_engine(offloader)
    with activated_model(offloader, "cuda"):
        engine.release()
        assert manager.stats.active_leases == 0
        assert manager.stats.idle_registrations == 2
        count = len(backend.registrations)
        engine.deactivate()
        engine.activate(torch.device("cuda"))
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
        lease = residency_engine(offloader)._pin_lease
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
        lease = residency_engine(second)._pin_lease
        assert lease is not None and lease.pageable_bytes > 0
        second.deactivate()
        first.deactivate()
        second.activate("cuda")
        assert backend.unregistrations
        lease = residency_engine(second)._pin_lease
        assert lease is not None and lease.registered_bytes > 0
    finally:
        second.deactivate()
        first.deactivate()


@pytest.mark.parametrize(
    ("device", "offload_paths", "vram_budget_bytes"),
    [
        ("cpu", ("blocks",), None),
        ("cpu", ("blocks",), 1024 * 1024 * 1024),
        ("cpu", (), None),
    ],
)
def test_cpu_execution_does_not_acquire_pins(
    device: str,
    offload_paths: tuple[str, ...],
    vram_budget_bytes: int | None,
    pins,
    monkeypatch,
) -> None:
    manager, _backend = pins

    def unexpected_acquire(_tensors, **_kwargs):
        raise AssertionError("this execution mode must not acquire host pins")

    monkeypatch.setattr(manager, "acquire", unexpected_acquire)
    offloader = ModelOffloader.from_module(
        _BlockModel(),
        config=OffloadConfig(
            offload_paths=offload_paths,
            vram_budget_bytes=vram_budget_bytes,
        ),
    )
    with activated_model(offloader, device):
        assert manager.stats.active_leases == 0


def test_transfer_sources_include_active_params_and_buffers() -> None:
    module = nn.Module()
    module.overridden = nn.Parameter(torch.randn(4, 4), requires_grad=False)
    module.kept = nn.Parameter(torch.randn(4, 4), requires_grad=False)
    module.register_buffer("buffer", torch.ones(4))
    module.register_buffer("alias", module.buffer)
    store = HostModuleStore.from_module(module)
    replacement = HostParam(nn.Parameter(torch.randn(4, 4), requires_grad=False))
    plan = store.bind(module).resolve_load_plan({
        "overridden": ParameterOverride(source=replacement),
    })
    tensors = list(plan.storage_tensors())
    identities = {id(tensor) for tensor in tensors}
    assert id(store.params["overridden"].storage_tensors()[0]) not in identities
    for host in (replacement, store.params["kept"], store.buffers["buffer"]):
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
            build_parameter_transform(
                [delta.scaled(0.25)],
                source_scale=1.0,
                stochastic_rounding=False,
                target_key="",
            ),
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
    expected = _to_dense(source).cpu()
    backing = ParameterValue.from_tensor(source).backing
    class _Unit(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(backing.logical_shape, device="meta"),
                requires_grad=False,
            )
            self.register_buffer("buffer", torch.arange(8, dtype=torch.float32))
            self.register_buffer("alias", self.buffer)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    block = _Unit()
    model = nn.Module()
    model.blocks = nn.ModuleList([block])
    offloader = _make_offloader(model)
    engine = residency_engine(offloader)
    tensors = backing.storage_tensors()
    pointers = {tensor.untyped_storage().data_ptr() for tensor in tensors if tensor.numel()}
    seen: list[tuple[torch.Tensor, type, torch.Tensor]] = []

    def inspect_hook(_mod: nn.Module, _args: tuple[object, ...], _out: object) -> None:
        rep = param_representation(block.weight)
        adapter = select_adapter(rep)
        seen.append((_to_dense(rep).cpu(), type(select_adapter(rep)), block.buffer.cpu()))

    handle = block.register_forward_hook(inspect_hook)
    try:
        engine.activate(
            torch.device("cuda"),
            unit_overrides=[{"weight": ParameterOverride(source=backing)}],
        )
        assert pointers <= {pointer for pointer, _size in backend.registrations}
        assert manager.stats.registrations == len(pointers) + 1  # tied buffer shares one registration
        assert all(tensor.is_pinned() for tensor in tensors if tensor.numel())
        with torch.inference_mode():
            block(torch.empty(1, device="cuda"))
        actual, adapter_type, buf = seen[0]
        assert adapter_type is type(backing.adapter)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(buf, torch.arange(8, dtype=torch.float32))
    finally:
        handle.remove()
        engine.deactivate()
    assert block.weight.is_meta
    assert pointers == {tensor.untyped_storage().data_ptr() for tensor in tensors if tensor.numel()}
    assert not backend.unregistrations


@CUDA
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_partial_activation_failure_releases_host_lease(compiled: bool, pins, monkeypatch) -> None:
    manager, backend = pins
    offloader = _make_offloader(
        _BlockModel(),
        block_compile=BlockCompileConfig(fullgraph=True) if compiled else None,
    )
    engine = residency_engine(offloader)

    def fail_hooks(*_args):
        raise RuntimeError("injected hook failure after upload")

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_register_unit_hooks", fail_hooks)
        with pytest.raises(RuntimeError, match="injected hook failure"):
            offloader.activate("cuda")
    assert manager.stats.active_leases == 0
    assert manager.stats.idle_registrations == 2
    count = len(backend.registrations)
    with activated_model(offloader, "cuda"):
        assert len(backend.registrations) == count


@CUDA
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_failed_runtime_quiescence_does_not_release_host_lease(
    compiled: bool,
    pins,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _backend = pins
    offloader = _make_offloader(
        _BlockModel(),
        block_compile=BlockCompileConfig(fullgraph=True) if compiled else None,
    )
    engine = residency_engine(offloader)
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
        with torch.inference_mode():
            offloader.value(torch.randn(2, 8, device="cuda"))
        assert engine.pool is not None
        with monkeypatch.context() as patch:
            patch.setattr(CudaTargetLease, "close", fail_once)
            with pytest.raises(RuntimeError, match="injected synchronization failure"):
                offloader.deactivate()
        assert engine.pool is not None and engine.pool.live_slot_count > 0
        assert engine.active_device is None
        assert engine._pin_lease is not None
        assert manager.stats.active_leases == 1
        with pytest.raises(RuntimeError, match="deactivate\\(\\) retries"):
            engine.activate(torch.device("cuda"))

        # The retry the refusal names releases what the failed one kept.
        offloader.deactivate()
        assert engine.pool is None
        assert not offloader.failed
        assert manager.stats.active_leases == 0
    finally:
        offloader.deactivate()
        if compiled:
            torch.compiler.reset()


@CUDA
def test_prefetch_update_failure_waits_before_releasing_pins(pins) -> None:
    manager, backend = pins
    model = _BlockModel()
    offloader = _make_offloader(model)
    engine = residency_engine(offloader)

    def fail_update(_parameter):
        raise RuntimeError("injected update failure after upload")

    engine.activate(
        torch.device("cuda"),
        unit_overrides=[
            None,
            {"proj.weight": ParameterOverride(update=CallbackParameterTransform(fail_update))},
        ],
    )
    stream = engine._copy_stream
    assert stream is not None
    try:
        with torch.inference_mode():
            model.blocks[0](torch.randn(2, 8, device="cuda"))
        manager.max_pinned_bytes = 0
        assert not backend.unregistrations
        with pytest.raises(RuntimeError, match="injected update failure"):
            engine.deactivate()
        assert stream.query()
        assert manager.stats.active_leases == 0
        assert manager.stats.pinned_bytes == 0
    finally:
        engine.deactivate()


def _resident_engine(model: nn.Module, mode: str):
    if mode == "host":
        return residency_engine(ModelOffloader.from_module(model))
    return residency_engine(
        _make_offloader(model, vram_budget_bytes=1024 * 1024 * 1024)
    )


@CUDA
@pytest.mark.parametrize("mode", ["all-resident", "host"])
@pytest.mark.parametrize("budget", [0, 64 * 1024**2])
def test_resident_and_host_uploads_lease_without_pinning(mode, budget, pins, monkeypatch) -> None:
    manager, backend = pins
    manager.max_pinned_bytes = budget
    model = _BlockModel()
    inputs = torch.randn(2, 8)
    expected = model(inputs)
    engine = _resident_engine(model, mode)
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
        for _ in range(2):
            staged.clear()
            engine.activate(torch.device("cuda"))
            assert staged and all(stream.query() for stream in staged)
            assert manager.stats.active_leases == 0
            assert backend.registrations == []
            torch.testing.assert_close(model(inputs.cuda()).cpu(), expected)
            engine.release()
            assert backend.registrations == []
            assert manager.stats.active_leases == 0
            engine.deactivate()
    finally:
        engine.deactivate()


@CUDA
@pytest.mark.parametrize("mode", ["all-resident", "host"])
@pytest.mark.parametrize("fail_sync", [False, True])
def test_partial_upload_keeps_its_lease_until_cleanup_synchronizes(mode, fail_sync, pins, monkeypatch) -> None:
    manager, _backend = pins
    engine = _resident_engine(_BlockModel(), mode)
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
                engine.activate(torch.device("cuda"))
            if fail_sync:
                # The upload may still be in flight: protection stays until a
                # synchronization succeeds.
                assert manager.stats.active_leases == 1
                with pytest.raises(RuntimeError, match="synchronization failure"):
                    engine.release()
                assert manager.stats.active_leases == 1
                with pytest.raises(RuntimeError, match="synchronization failure"):
                    engine.deactivate()
                assert engine.active_device is None
                # The old target and transfer are still open; a new session
                # must not silently reuse them.
                with pytest.raises(RuntimeError, match="deactivate\\(\\) retries"):
                    engine.activate(torch.device("cuda"))
                assert manager.stats.active_leases == 1

        def checked_sync(sync_device):
            # Deactivation cleared the session, but the retry must still
            # synchronize the original transfer device, not the current one.
            assert sync_device == device
            original_sync(sync_device)

        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "synchronize", checked_sync)
            engine.release()
        assert manager.stats.active_leases == 0
        assert not engine.failed
    finally:
        engine.deactivate()


@CUDA
def test_failed_deactivation_refuses_resident_reactivation(pins, monkeypatch) -> None:
    manager, _backend = pins
    engine = _resident_engine(_BlockModel(), "all-resident")

    def failed_sync(*_args, **_kwargs):
        raise RuntimeError("injected synchronization failure")

    try:
        engine.activate(torch.device("cuda"))
        assert manager.stats.active_leases == 0
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda.Stream, "synchronize", failed_sync)
            patch.setattr(torch.cuda, "synchronize", failed_sync)
            with pytest.raises(RuntimeError, match="injected"):
                engine.deactivate()
        assert engine.retains_resources
        assert engine.active_device is None
        with pytest.raises(RuntimeError, match="deactivate\\(\\) retries"):
            engine.activate(torch.device("cuda"))
        engine.deactivate()
        assert not engine.retains_resources
        assert not engine.failed
    finally:
        engine.deactivate()


@CUDA
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_mapped_checkpoint_streams_from_owned_copies(compiled, pins, tmp_path, monkeypatch) -> None:
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
        block_compile=BlockCompileConfig(fullgraph=True) if compiled else None,
    )
    resolved: list[torch.Tensor] = []
    original_view = copy_memory.Copy.view

    def recording(copy: copy_memory.Copy, tensor: torch.Tensor) -> torch.Tensor:
        view = original_view(copy, tensor)
        resolved.append(view)
        return view

    monkeypatch.setattr(copy_memory.Copy, "view", recording)
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
        if compiled:
            torch.compiler.reset()


def _sample_lora_adapter() -> Adapter:
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


@CUDA
def test_permanent_merge_on_a_cuda_model_leases_pinned_factors(pins) -> None:
    manager, backend = pins
    adapter = _sample_lora_adapter()
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
def test_permanent_merge_leases_sources_during_apply(pins, monkeypatch) -> None:
    from piper_offload import transfer_

    manager, _backend = pins
    adapter = _sample_lora_adapter()
    factors = _factor_tensors(adapter)
    with manager.acquire(factors):
        pass
    original_validate = LoRATransform.validate_parameter
    original_apply = LoRATransform.apply_parameter
    staged = []
    leases_seen: list[tuple[str, int]] = []

    def tracking_validate(update, param) -> None:
        leases_seen.append(("validate", manager.stats.active_leases))
        original_validate(update, param)

    def staging_apply(update, param) -> None:
        leases_seen.append(("apply", manager.stats.active_leases))
        original_apply(update, param)
        for tensor in update.storage_tensors():
            transfer_(torch.empty_like(tensor, device="cuda"), tensor, non_blocking=True)
            staged.append(tensor.data_ptr())

    monkeypatch.setattr(LoRATransform, "validate_parameter", tracking_validate)
    monkeypatch.setattr(LoRATransform, "apply_parameter", staging_apply)
    model = _BlockModel().cuda()
    assert merge_adapter(model, [(adapter, 0.5)]) == 2
    assert ("validate", 0) in leases_seen
    assert any(phase == "apply" and active > 0 for phase, active in leases_seen)
    assert sorted(staged) == sorted(tensor.data_ptr() for tensor in factors)
    assert manager.stats.active_leases == 0


@CUDA
@pytest.mark.parametrize("reclaim_code", [0, 170], ids=["intact", "discarded"])
def test_rotating_mapped_models_streams_from_offered_copies(reclaim_code, tmp_path, monkeypatch, request) -> None:
    """Two streamed checkpoints over a budget that holds one: each switch offers a copy and takes the other back."""
    safetensors = pytest.importorskip("safetensors.torch")
    kernel = install_fake_kernel(monkeypatch, [])
    # The platform component is selected when the manager is constructed.
    manager, backend = request.getfixturevalue("pins")
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
        assert counts["offer"] == counts["reclaim"] + len(manager._memory._offered) > 0
        # Each storage's copy was built once and reused from the tier thereafter.
        copies = manager.stats.registrations + len(manager._memory._offered)
        assert sum(event == "allocate" for event, _pointer in kernel.events) == copies
    finally:
        for offloader, _inputs, _expected in rotation:
            offloader.deactivate()
