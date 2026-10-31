"""Composition of complete values and updates from separate adapter resources."""

from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from piper_offload import (
    Adapter,
    AdapterSpec,
    BlockCompileConfig,
    MappedCheckpoint,
    ModelCache,
    ModelOffloader,
    ModelSpec,
    ParameterDelta,
    file_slice,
    merge_adapter,
)
from piper_offload.tensor_adapter_registry import param_representation, select_adapter
from piper_offload.tensor_adapters import RegularAdapter
from tests._block_compile_helpers import _BlockModel
from tests.conftest import activated_model
from tests.test_quantized_parameter_value import _make_quantized

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@contextmanager
def _application(model, requests, application):
    """Exercise the two public application paths with the same requests."""
    if application == "permanent":
        merge_adapter(model, requests, stochastic_rounding=False)
        yield model
    else:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        offloader = ModelOffloader.from_module(model)
        with activated_model(
            offloader,
            "cuda",
            adapters=[adapter for adapter, _ in requests],
            adapter_strengths=[strength for _, strength in requests],
            stochastic_rounding=False,
        ):
            yield model


@pytest.mark.parametrize("application", ["permanent", "activation"])
@pytest.mark.parametrize("kind", ["lora", "dense", "mixed"])
@pytest.mark.parametrize("scale_value", [False, True])
@pytest.mark.parametrize("shape", [(2, 3), (4, 2, 3)])
def test_composition_strengths_order_aliases_and_ownership(application, kind, scale_value, shape):
    source = torch.arange(6, dtype=torch.float64).reshape(2, 3).expand(shape).clone()
    original = source.clone()
    a = torch.tensor([[1., 2., 0.], [-1., 0., 1.]])
    b = torch.tensor([[1., 0.], [2., -1.]])
    dense = torch.full((2, 3), 0.75)
    extra = torch.full((2, 3), 0.125)
    value = Adapter.from_state_dict({"weight": source}, scale_parameter_values=scale_value)
    delta = Adapter({"weight": ParameterDelta.from_tensors(
        a=a if kind != "dense" else None,
        b=b if kind != "dense" else None,
        lora_alpha=3.0 if kind != "dense" else None,
        dense=dense if kind != "lora" else None,
    )})
    other = Adapter({"weight": ParameterDelta.from_tensors(dense=extra)})
    requests = [(value, -0.5), (delta, 0.25), (other, -1.5), (other, 0.5), (value, 0.0)]
    expected = original * (-0.5 if scale_value else 1.0) - extra.double()
    if kind != "dense":
        expected += 0.25 * (3.0 / 2.0) * (b.double() @ a.double())
    if kind != "lora":
        expected += 0.25 * dense.double()

    for ordered in (requests, list(reversed(requests))):
        model = nn.Module()
        model.weight = nn.Parameter(torch.empty(shape, device="meta"), requires_grad=False)
        model.alias = model.weight
        with _application(model, ordered, application):
            assert model.weight is model.alias
            assert model.weight.dtype == source.dtype
            assert not model.weight.requires_grad
            torch.testing.assert_close(model.weight.cpu(), expected)
            if application == "permanent":
                assert model.weight.data_ptr() != source.data_ptr()
        if application == "activation":
            assert model.weight.is_meta and model.alias.is_meta
        torch.testing.assert_close(source, original)


@pytest.mark.parametrize("application", ["permanent", "activation"])
@pytest.mark.parametrize("kind", ["lora", "dense"])
@pytest.mark.parametrize("base", ["absent", "disabled", "supplied", "physical"])
def test_partial_targets_share_updates_with_optional_head(application, kind, base):
    a = torch.tensor([[1., 2., 3.]])
    b = torch.tensor([[1.], [-2.]])
    dense = torch.arange(6, dtype=torch.float32).reshape(2, 3).clone()
    delta = (
        ParameterDelta.from_tensors(a=a, b=b, lora_alpha=2.0)
        if kind == "lora" else ParameterDelta.from_tensors(dense=dense)
    )
    adapter = Adapter({
        "weight": delta, "optional": delta, "missing": delta,
    }, allow_partial_targets=True)
    source = torch.full((4, 2, 3), 3.0)
    value = Adapter.from_state_dict({"optional": source})
    requests = [(adapter, 0.25)]
    if base in ("disabled", "supplied"):
        requests.append((value, 0.0 if base == "disabled" else 0.5))
    update = 0.25 * (2.0 * b @ a if kind == "lora" else dense)
    orders = (requests, list(reversed(requests))) if len(requests) > 1 else (requests,)
    for ordered in orders:
        model = nn.Module()
        model.weight = nn.Parameter(torch.ones(2, 3), requires_grad=False)
        model.optional = nn.Parameter(
            source.clone() if base == "physical" else torch.empty(source.shape, device="meta"),
            requires_grad=False,
        )
        model.alias = model.optional
        with _application(model, ordered, application):
            torch.testing.assert_close(model.weight.cpu(), 1.0 + update)
            assert model.optional is model.alias
            assert model.optional is not model.weight
            if base in ("absent", "disabled"):
                assert model.optional.is_meta
            else:
                torch.testing.assert_close(model.optional.cpu(), source + update)
        if application == "activation":
            torch.testing.assert_close(model.weight, torch.ones(2, 3))
            if base == "physical":
                torch.testing.assert_close(model.optional, source)
            else:
                assert model.optional.is_meta
    assert adapter.targets["weight"] is adapter.targets["optional"]
    torch.testing.assert_close(a, torch.tensor([[1., 2., 3.]]))
    torch.testing.assert_close(b, torch.tensor([[1.], [-2.]]))
    torch.testing.assert_close(dense, torch.arange(6, dtype=torch.float32).reshape(2, 3))
    torch.testing.assert_close(source, torch.full((4, 2, 3), 3.0))


@pytest.mark.parametrize("application", ["permanent", "activation"])
@pytest.mark.parametrize("kind", ["lora", "dense"])
def test_partial_targets_do_not_disable_strict_contributions(application, kind):
    delta = (
        ParameterDelta.from_tensors(a=torch.ones(1, 3), b=torch.ones(2, 1))
        if kind == "lora" else ParameterDelta.from_tensors(dense=torch.ones(2, 3))
    )
    partial = Adapter({"physical": delta, "weight": delta}, allow_partial_targets=True)
    strict = Adapter({"weight": delta})
    error = "Meta parameters" if kind == "lora" else "existing plain floating-point target"
    for requests in ([(partial, 1.0), (strict, 1.0)], [(strict, 1.0), (partial, 1.0)]):
        model = nn.Module()
        model.physical = nn.Parameter(torch.zeros(2, 3), requires_grad=False)
        model.weight = nn.Parameter(torch.empty(2, 3, device="meta"), requires_grad=False)
        with pytest.raises(ValueError, match=error), _application(model, requests, application):
            pytest.fail("strict contribution was skipped")
        assert model.weight.is_meta
        torch.testing.assert_close(model.physical, torch.zeros(2, 3))
    # A disabled strict contribution must not prevent partial application.
    with _application(model, [(strict, 0.0), (partial, 1.0)], application):
        assert model.weight.is_meta
        torch.testing.assert_close(model.physical.cpu(), torch.ones(2, 3))


def test_unpopulated_partial_targets_create_no_merge_work(monkeypatch):
    from piper_offload._parameter_update import ParameterUpdateTransform

    def unexpected_update(*args, **kwargs):
        pytest.fail("skipped targets must not stage merge updates")

    monkeypatch.setattr(ParameterUpdateTransform, "_stage_update_for_plan", unexpected_update)
    model = nn.Linear(3, 2, bias=False, device="meta").requires_grad_(False)
    # Even incompatible target shapes need no merge validation without a base.
    adapter = Adapter({"weight": ParameterDelta.from_tensors(dense=torch.ones(5, 7))}, allow_partial_targets=True)
    original = model.weight
    assert merge_adapter(model, [(adapter, 1.0)]) == 0
    assert model.weight is original
    offloader = ModelOffloader.from_module(model)
    updates = offloader._group_adapter_updates_by_param_name([(adapter, 1.0)])
    assert offloader._build_merge_parameter_overrides(torch.device("cuda"), updates) == {}
    assert offloader.cache_bytes == 0
    assert model.weight.is_meta


@pytest.mark.parametrize("application", ["permanent", "activation"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_merge_rejects_distinct_parameter_objects_sharing_storage(application, requires_grad):
    model = nn.Module()
    model.weight = nn.Parameter(torch.ones(2, 3), requires_grad=requires_grad)
    model.alias = nn.Parameter(model.weight.detach(), requires_grad=requires_grad)
    assert model.weight is not model.alias
    assert model.weight.data_ptr() == model.alias.data_ptr()
    delta = ParameterDelta.from_tensors(dense=torch.ones(2, 3))
    adapter = Adapter({"weight": delta, "alias": delta})
    with pytest.raises(ValueError, match="Tied parameter aliases"), _application(
        model, [(adapter, 1.0)], application,
    ):
        pytest.fail("distinct wrappers hid a shared parameter backing")
    torch.testing.assert_close(model.weight, torch.ones(2, 3))
    torch.testing.assert_close(model.alias, torch.ones(2, 3))


@pytest.mark.parametrize("application", ["permanent", "activation"])
@pytest.mark.parametrize("supply_value", [False, True])
def test_partial_meta_targets_preserve_tied_name_contract(application, supply_value):
    delta = ParameterDelta.from_tensors(dense=torch.ones(2, 3))
    partial = Adapter({"weight": delta}, allow_partial_targets=True)
    other = (
        Adapter.from_state_dict({"alias": torch.ones(2, 3)})
        if supply_value else Adapter({"alias": delta}, allow_partial_targets=True)
    )
    for requests in ([(partial, 1.0), (other, 1.0)], [(other, 1.0), (partial, 1.0)]):
        model = nn.Linear(3, 2, bias=False, device="meta").requires_grad_(False)
        model.alias = model.weight
        with pytest.raises(ValueError, match="[Tt]ied"), _application(model, requests, application):
            pytest.fail("different names for one tied target were accepted")
        assert model.weight.is_meta and model.alias.is_meta


@pytest.mark.parametrize("application", ["permanent", "activation"])
@pytest.mark.parametrize("failure", ["value_shape", "delta_shape", "inactive_value", "two_values", "tied_names"])
def test_composition_preflight_leaves_all_parameters_unchanged(application, failure):
    model = nn.Module()
    model.physical = nn.Parameter(torch.ones(2, 3), requires_grad=False)
    model.weight = nn.Parameter(torch.empty(2, 3, device="meta"), requires_grad=False)
    model.alias = model.weight
    value = Adapter.from_state_dict({
        "weight": torch.ones((3, 2) if failure == "value_shape" else (2, 3)),
    })
    delta = Adapter({
        "physical": ParameterDelta.from_tensors(dense=torch.ones(2, 3)),
        "alias" if failure == "tied_names" else "weight": ParameterDelta.from_tensors(
            dense=torch.ones((3, 2) if failure == "delta_shape" else (2, 3)),
        ),
    }, allow_partial_targets=failure != "inactive_value")
    requests = [(delta, 1.0), (value, 0.0 if failure == "inactive_value" else 1.0)]
    if failure == "two_values":
        requests.append((value, 0.5))
    error = {
        "value_shape": "shape mismatch",
        "delta_shape": "broadcast",
        "inactive_value": "existing plain floating-point target",
        "two_values": "multiple active parameter values",
        "tied_names": "[Tt]ied",
    }[failure]
    exception = RuntimeError if failure == "delta_shape" else ValueError
    with pytest.raises(exception, match=error), _application(model, requests, application):
        pytest.fail("invalid composition was applied")
    assert model.weight.is_meta and model.alias.is_meta
    torch.testing.assert_close(model.physical.cpu(), torch.ones(2, 3))


@pytest.mark.parametrize("application", ["permanent", "activation"])
@pytest.mark.parametrize("kind", ["torchao-int8", "quanto-qint8", "torchao-float8"])
@pytest.mark.parametrize("value_scale", [1.0, -0.5])
def test_quantized_composition_encodes_once(application, kind, value_scale, monkeypatch):
    source = _make_quantized(kind)
    adapter_type = type(select_adapter(source))
    base = adapter_type.dequantize(source).cpu().clone()
    rows, cols = base.shape
    a = torch.full((1, cols), 0.5)
    b = torch.full((rows, 1), 0.25)
    dense = torch.full((rows, cols), 0.125)
    value = Adapter.from_state_dict({"weight": source}, scale_parameter_values=True)
    delta = Adapter({"weight": ParameterDelta.from_tensors(a=a, b=b, dense=dense)})
    original_merge = adapter_type.merge_dense_
    calls = []

    def merge_once(target, update, alpha, *, beta=1.0, rounding_seed=None):
        calls.append(rounding_seed)
        original_merge(target, update, alpha, beta=beta, rounding_seed=rounding_seed)

    monkeypatch.setattr(adapter_type, "merge_dense_", staticmethod(merge_once))
    model = nn.Module()
    model.weight = nn.Parameter(torch.empty(rows, cols, device="meta"), requires_grad=False)
    with _application(model, [(delta, 0.5), (value, value_scale)], application):
        actual = adapter_type.dequantize(param_representation(model.weight)).cpu()
        expected = base.float() * value_scale + 0.5 * (b @ a + dense)
        # Allow terminal quantization error while comparing independent dense arithmetic.
        torch.testing.assert_close(
            actual.float(), expected, rtol=0.05, atol=max(base.abs().max().item() * 0.02, 0.02),
        )
    assert calls == [None]
    torch.testing.assert_close(adapter_type.dequantize(source).cpu(), base)


@pytest.mark.parametrize("application", ["permanent", "activation"])
@pytest.mark.parametrize(("scale_value", "value_strength"), [(False, 0.5), (True, -0.5), (True, 1.5)])
@pytest.mark.parametrize("shape", [(2, 3), (4, 2, 3)])
def test_value_with_loras_retains_one_factorized_merge(application, scale_value, value_strength, shape, monkeypatch):
    from piper_offload._parameter_update import ParameterUpdateTransform

    def unexpected_dense_merge(*args, **kwargs):
        pytest.fail("value plus LoRAs should retain factorized staging")

    monkeypatch.setattr(ParameterUpdateTransform, "_stage_update_for_plan", unexpected_dense_merge)
    calls = []
    original_merge = RegularAdapter.merge_lora_

    def record_merge(target, b, a, alpha, *, beta=1.0, rounding_seed=None):
        calls.append((b.shape, a.shape, alpha, beta))
        original_merge(target, b, a, alpha=alpha, beta=beta, rounding_seed=rounding_seed)

    monkeypatch.setattr(RegularAdapter, "merge_lora_", staticmethod(record_merge))
    model = nn.Module()
    model.weight = nn.Parameter(torch.empty(shape, device="meta"), requires_grad=False)
    source = torch.ones(shape)
    value = Adapter.from_state_dict({"weight": source}, scale_parameter_values=scale_value)
    delta = Adapter({"weight": ParameterDelta.from_tensors(a=torch.ones(1, 3), b=torch.ones(2, 1))})
    beta = value_strength if scale_value else 1.0
    with _application(model, [(value, value_strength), (delta, 0.25), (delta, -0.125)], application):
        torch.testing.assert_close(model.weight.cpu(), torch.full(shape, beta + 0.125))
    assert calls == [((2, 2), (2, 3), 1.0, beta)]
    torch.testing.assert_close(source, torch.ones(shape))


@pytest.mark.parametrize("application", ["permanent", "activation"])
@pytest.mark.parametrize("kind", ["piper-convrot-int8", "piper-convrot-nvfp4"])
def test_scaled_convrot_value_with_loras_uses_one_native_merge(application, kind, monkeypatch):
    from piper_offload._parameter_update import ParameterUpdateTransform

    source = _make_quantized(kind)
    adapter_type = type(select_adapter(source))
    base = adapter_type.dequantize(source).float().clone()
    rows, cols = base.shape
    value = Adapter.from_state_dict({"weight": source}, scale_parameter_values=True)
    delta = Adapter({"weight": ParameterDelta.from_tensors(
        a=torch.full((1, cols), 0.5, dtype=torch.bfloat16),
        b=torch.full((rows, 1), 0.25, dtype=torch.bfloat16),
    )})
    original_merge = adapter_type.merge_lora_
    calls = []

    def record_merge(target, b, a, alpha, *, beta=1.0, rounding_seed=None):
        pointers = (target.qdata.data_ptr(), target.scale.data_ptr())
        calls.append((b.shape, a.shape, alpha, beta))
        original_merge(target, b, a, alpha=alpha, beta=beta, rounding_seed=rounding_seed)
        assert pointers == (target.qdata.data_ptr(), target.scale.data_ptr())

    def unexpected_dense_staging(*args, **kwargs):
        pytest.fail("scaled ConvRot value plus LoRAs must use factorized staging")

    monkeypatch.setattr(adapter_type, "merge_lora_", staticmethod(record_merge))
    monkeypatch.setattr(ParameterUpdateTransform, "_stage_update_for_plan", unexpected_dense_staging)
    model = nn.Linear(cols, rows, bias=False, device="meta").requires_grad_(False)
    with _application(model, [(value, -0.5), (delta, 0.25), (delta, -0.125)], application):
        actual = adapter_type.dequantize(param_representation(model.weight)).cpu().float()
        expected = base * -0.5 + (0.25 - 0.125) * 0.25 * 0.5
        tolerance = 0.03 if kind == "piper-convrot-int8" else 0.2
        assert (actual - expected).abs().mean() < expected.abs().mean() * tolerance
    assert calls == [((rows, 2), (2, cols), 1.0, -0.5)]
    torch.testing.assert_close(adapter_type.dequantize(source).float(), base)


@CUDA
@pytest.mark.parametrize("mode", ["resident", "streaming", "rolling"])
def test_composed_values_reload_across_block_uses(mode):
    # Give this compiled case its own forward code, so unrelated tests do
    # not consume its Dynamo recompilation budget.
    class Block(nn.Linear):
        def forward(self, x):
            return F.relu(F.linear(x, self.weight))

    with torch.device("meta"):
        model = _BlockModel(blocks=[Block(4, 4, bias=False) for _ in range(2)])
    offloader = ModelOffloader.from_module(
        model,
        block_paths=("blocks",),
        block_mode=mode,
        block_compile=BlockCompileConfig(fullgraph=True, dynamic=False) if mode == "rolling" else None,
    )
    values = {f"blocks.{i}.weight": torch.eye(4) * (i + 1) for i in range(2)}
    value = Adapter.from_state_dict(values, scale_parameter_values=True)
    delta = Adapter({
        name: ParameterDelta.from_tensors(a=torch.ones(1, 4), b=torch.ones(4, 1))
        for name in values
    }, allow_partial_targets=True)
    with torch.inference_mode():
        x = torch.ones(2, 4, device="cuda")
    for strength in (0.25, -0.5, 0.0):
        expected = x.cpu()
        for i in range(2):
            expected = F.relu(F.linear(expected, torch.eye(4) * (i + 1) * 0.5 + strength))
        with activated_model(
            offloader, "cuda", adapters=[value, delta], adapter_strengths=[0.5, strength],
        ), torch.inference_mode():
            for _ in range(2):
                torch.testing.assert_close(model(x).cpu(), expected)
        assert all(param.is_meta for param in model.parameters())
    with activated_model(offloader, "cuda"):
        assert all(param.is_meta for param in model.parameters())
    with activated_model(offloader, "cuda", adapters=[delta, value], adapter_strengths=[1.0, 0.0]):
        assert all(param.is_meta for param in model.parameters())


@CUDA
def test_cache_composition_preserves_mappings_through_switching_and_exception(tmp_path):
    path = tmp_path / "value.safetensors"
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3).expand(4, 2, 3).clone()
    physical = torch.ones(2, 3)
    save_file({
        "weight": source, "physical": physical,
        "a": torch.ones(1, 3), "b": torch.ones(2, 1),
    }, path)
    original_file = path.read_bytes()
    with MappedCheckpoint(path) as checkpoint:
        mapped, mapped_physical, a, b = (
            checkpoint.get_tensor(name) for name in ("weight", "physical", "a", "b")
        )
    assert file_slice(mapped) is not None
    delta_spec = AdapterSpec(
        key="delta", factory=lambda: {
            "head.lora_A.weight": a, "head.lora_B.weight": b,
            "optional.lora_A.weight": a, "optional.lora_B.weight": b,
        },
        allow_partial_targets=True,
    )
    # Use canonical names without changing the mapped value tensor.
    value_spec = AdapterSpec(key="value", factory=lambda: {"optional.weight": mapped}, scale_parameter_values=True)
    def make_head():
        model = nn.Module()
        model.head = nn.Linear(3, 2, bias=False, device="meta").requires_grad_(False)
        model.head.weight = nn.Parameter(mapped_physical, requires_grad=False)
        model.optional = nn.Module()
        model.optional.weight = nn.Parameter(torch.empty(4, 2, 3, device="meta"), requires_grad=False)
        return model
    spec = ModelSpec(key="model", factory=make_head)
    cache = ModelCache()
    try:
        model_id = None
        for strength, value_strength in ((0.25, 0.5), (0.25, 0.0), (-0.5, 0.5), (-0.5, 0.0), (0.0, 0.5)):
            with cache.use(
                spec, device="cuda", adapter_specs=[delta_spec, value_spec],
                adapter_strengths=[strength, value_strength],
            ) as model:
                if model_id is None:
                    model_id = id(model)
                assert id(model) == model_id
                torch.testing.assert_close(model.head.weight.cpu(), physical + strength)
                if value_strength:
                    torch.testing.assert_close(model.optional.weight.cpu(), source * value_strength + strength)
                else:
                    assert model.optional.weight.is_meta
            assert model.optional.weight.is_meta
            torch.testing.assert_close(model.head.weight, physical)
            assert model.head.weight.data_ptr() == mapped_physical.data_ptr()
        with cache.use(spec, device="cuda", adapter_specs=[delta_spec]) as model:
            torch.testing.assert_close(model.head.weight.cpu(), physical + 1.0)
            assert model.optional.weight.is_meta
        with pytest.raises(RuntimeError, match="caller failed"):
            with cache.use(spec, device="cuda", adapter_specs=[value_spec, delta_spec]) as model:
                raise RuntimeError("caller failed")
        assert model.optional.weight.is_meta
        with cache.use(spec, device="cuda") as model:
            assert model.optional.weight.is_meta
            torch.testing.assert_close(model.head.weight.cpu(), physical)
        with cache.lease(value_spec) as captured:
            assert file_slice(captured.targets["optional.weight"].backing.make_cpu_param()) is not None
    finally:
        cache.clear()
    assert path.read_bytes() == original_file
    for actual, expected in (
        (mapped, source), (mapped_physical, physical), (a, torch.ones(1, 3)), (b, torch.ones(2, 1)),
    ):
        torch.testing.assert_close(actual, expected)


@CUDA
@pytest.mark.parametrize("application", ["permanent", "activation"])
def test_copy_only_value_rejects_composed_arithmetic_before_mutation(application):
    from tests.test_int4_tile_adapter import _make_int4_tile

    source = _make_int4_tile()
    value = Adapter.from_state_dict({"weight": source})
    rows, cols = source.shape
    delta = Adapter({
        "physical": ParameterDelta.from_tensors(dense=torch.ones(1)),
        "weight": ParameterDelta.from_tensors(a=torch.ones(1, cols), b=torch.ones(rows, 1)),
    }, allow_partial_targets=True)
    model = nn.Module()
    model.physical = nn.Parameter(torch.zeros(1), requires_grad=False)
    model.weight = nn.Parameter(torch.empty(rows, cols, device="meta"), requires_grad=False)
    with pytest.raises(ValueError, match="Int4TilePackedAdapter"), _application(
        model, [(delta, 1.0), (value, 1.0)], application,
    ):
        pytest.fail("copy-only representation accepted arithmetic")
    assert model.weight.is_meta
    torch.testing.assert_close(model.physical.cpu(), torch.zeros(1))
    # The rejected update must not disable exact copying of the same resource.
    with _application(model, [(value, 1.0)], application):
        assert torch.equal(model.weight.data.qdata.cpu(), source.qdata.cpu())
        assert torch.equal(model.weight.data.scale_and_zero.cpu(), source.scale_and_zero.cpu())


@CUDA
def test_composition_failure_releases_activation_and_preserves_rounding_rng(monkeypatch):
    from piper_offload import derive_seed, host_pin_manager

    source = torch.ones(2, 3)
    value = Adapter.from_state_dict({"weight": source}, scale_parameter_values=True)
    delta = Adapter({"weight": ParameterDelta.from_tensors(dense=torch.full((2, 3), 0.25))})
    offloader = ModelOffloader.from_module(
        nn.Linear(3, 2, bias=False, device="meta").requires_grad_(False),
    )
    original_merge = RegularAdapter.merge_dense_
    leases_before = host_pin_manager.stats.active_leases

    def fail_after_merge(*args, **kwargs):
        original_merge(*args, **kwargs)
        raise RuntimeError("merge failed")

    with monkeypatch.context() as patch:
        patch.setattr(RegularAdapter, "merge_dense_", staticmethod(fail_after_merge))
        with pytest.raises(RuntimeError, match="merge failed"):
            offloader.activate("cuda", adapters=[value, delta], adapter_strengths=[0.5, 1.0])
    assert offloader.active_device is None
    assert offloader.value.weight.is_meta
    assert host_pin_manager.stats.active_leases == leases_before

    seeds = []
    def record_seed(target, update, alpha, *, beta=1.0, rounding_seed=None):
        seeds.append(rounding_seed)
        original_merge(target, update, alpha, beta=beta, rounding_seed=rounding_seed)
    monkeypatch.setattr(RegularAdapter, "merge_dense_", staticmethod(record_seed))
    rng_before = torch.get_rng_state()
    cuda_rng_before = torch.cuda.get_rng_state()
    with activated_model(
        offloader, "cuda", adapters=[value, delta], adapter_strengths=[0.5, 1.0],
    ):
        torch.testing.assert_close(offloader.value.weight.cpu(), torch.full((2, 3), 0.75))
    assert seeds == [derive_seed("weight", 0)]
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng_before)
    assert host_pin_manager.stats.active_leases == leases_before
    torch.testing.assert_close(source, torch.ones(2, 3))


def test_permanent_composition_preserves_mapped_value_and_delta(tmp_path):
    path = tmp_path / "sources.safetensors"
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3).clone()
    dense = torch.full((2, 3), 0.25)
    save_file({"value": source, "delta": dense}, path)
    before = path.read_bytes()
    with MappedCheckpoint(path) as checkpoint:
        mapped_value = checkpoint.get_tensor("value")
        mapped_delta = checkpoint.get_tensor("delta")
    value = Adapter.from_state_dict({"weight": mapped_value}, scale_parameter_values=True)
    update = ParameterDelta.from_tensors(dense=mapped_delta)
    delta = Adapter({"weight": update, "optional": update}, allow_partial_targets=True)
    for strength in (0.5, -1.0):
        model = nn.Linear(3, 2, bias=False, device="meta").requires_grad_(False)
        model.optional = nn.Parameter(torch.empty(4, 2, 3, device="meta"), requires_grad=False)
        assert merge_adapter(model, [(delta, strength), (value, 0.5)]) == 1
        torch.testing.assert_close(model.weight, source * 0.5 + strength * dense)
        assert model.optional.is_meta
        assert file_slice(model.weight) is None
    assert file_slice(mapped_value) is not None
    assert file_slice(mapped_delta) is not None
    torch.testing.assert_close(mapped_value, source)
    torch.testing.assert_close(mapped_delta, dense)
    assert path.read_bytes() == before


@pytest.mark.parametrize("application", ["permanent", "activation"])
@pytest.mark.parametrize(("dtype", "base", "scale", "increment"), [
    (torch.float16, 40000.0, -1.0, 0.0),
    (torch.float16, 40000.0, -1.0, 1000.0),
    (torch.bfloat16, 1.0, 0.125, 0.001),
])
def test_scaled_composition_avoids_base_cancellation(application, dtype, base, scale, increment):
    source = torch.full((2, 3), base, dtype=dtype)
    dense = torch.full((2, 3), increment, dtype=dtype)
    value = Adapter.from_state_dict({"weight": source}, scale_parameter_values=True)
    delta = Adapter({"weight": ParameterDelta.from_tensors(dense=dense)})
    lora = Adapter({"weight": ParameterDelta.from_tensors(
        a=torch.full((1, 3), increment, dtype=dtype), b=torch.ones(2, 1, dtype=dtype),
    )})
    expected = (source.double() * scale + dense.double()).to(dtype)
    for requests in ([(value, scale)], [(value, scale), (delta, 1.0)], [(value, scale), (lora, 1.0)]):
        model = nn.Linear(3, 2, bias=False, device="meta").requires_grad_(False)
        with _application(model, requests, application):
            actual = model.weight.cpu()
            assert torch.isfinite(actual).all()
            reference = source * scale if len(requests) == 1 else expected
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(source, torch.full((2, 3), base, dtype=dtype))


@pytest.mark.parametrize("application", ["permanent", "activation"])
def test_scaled_quantized_lora_composition_avoids_intermediate_overflow(application):
    from piper_offload.int8_adapter import Int8Adapter
    from tests.test_int8_adapter import _make_affine_int8

    source = _make_affine_int8(torch.full((17, 128), 40000.0, dtype=torch.float16))
    base = Int8Adapter.dequantize(source)
    a = torch.full((1, 128), -40000.0, dtype=torch.float16)
    b = torch.ones((17, 1), dtype=torch.float16)
    # The scaled base exceeds FP16's range, but the combined result fits.
    assert torch.isinf(base * 2.0).all()
    expected = (base.double() * 2.0 + b.double() @ a.double()).to(base.dtype)
    assert torch.isfinite(expected).all()
    reference = Int8Adapter.dequantize(Int8Adapter.requantize(expected, like=source))
    value = Adapter.from_state_dict({"weight": source}, scale_parameter_values=True)
    lora = Adapter({"weight": ParameterDelta.from_tensors(a=a, b=b)})
    model = nn.Linear(128, 17, bias=False, device="meta").requires_grad_(False)

    with _application(model, [(value, 2.0), (lora, 1.0)], application):
        actual = Int8Adapter.dequantize(param_representation(model.weight)).cpu()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(Int8Adapter.dequantize(source), base, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [1.0, -1.0])
def test_value_transform_shares_plan_and_invalidates_failed_validation(scale, monkeypatch):
    from piper_offload import ParameterValue, ParameterValueTransform, derive_seed

    source = torch.full((2, 3), 40000.0, dtype=torch.float16)
    value = ParameterValue.from_tensor(source, scale_with_strength=True)
    transform = ParameterValueTransform(value.scaled(scale), stochastic_rounding=True, target_key="weight")
    slot = nn.Parameter(torch.empty(2, 3, device="meta"), requires_grad=False)
    with pytest.raises(RuntimeError, match="validated before application"):
        transform.materialize()
    seeds = []
    original_merge = RegularAdapter.merge_dense_

    def record_merge(target, update, alpha, *, beta=1.0, rounding_seed=None):
        seeds.append(rounding_seed)
        original_merge(target, update, alpha, beta=beta, rounding_seed=rounding_seed)

    monkeypatch.setattr(RegularAdapter, "merge_dense_", staticmethod(record_merge))
    for _ in range(2):
        transform.validate_parameter(slot)
        torch.testing.assert_close(transform.materialize(), source * scale, rtol=0, atol=0)
    # Revalidation keeps the repeated-merge rounding sequence.
    assert seeds == ([] if scale == 1.0 else [derive_seed("weight", i) for i in range(2)])
    with pytest.raises(ValueError, match="shape mismatch"):
        transform.validate_parameter(nn.Parameter(torch.empty(3, 2, device="meta"), requires_grad=False))
    with pytest.raises(RuntimeError, match="validated before application"):
        transform.apply_parameter(nn.Parameter(source.clone(), requires_grad=False))
    transform.validate_parameter(slot)
    loaded = nn.Parameter(source.clone(), requires_grad=False)
    transform.apply_parameter(loaded)
    torch.testing.assert_close(loaded, source * scale, rtol=0, atol=0)
