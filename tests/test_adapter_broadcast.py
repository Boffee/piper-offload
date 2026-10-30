"""Additive broadcasting: shape policy, arithmetic, and load lifetimes."""

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

import piper_offload.parameter_delta as parameter_delta_impl
from piper_offload import (
    Adapter,
    BlockCompileConfig,
    MappedCheckpoint,
    ModelOffloader,
    ParameterDelta,
    ParameterDeltaTransform,
    ParameterValue,
    file_slice,
    host_pin_manager,
    merge_adapter,
)
from piper_offload.tensor_adapters import RegularAdapter
from tests.conftest import activated_model

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    ("target_shape", "product_shapes"),
    [
        ((3, 4), [(1, 4)]),
        ((3, 4), [(3, 1)]),
        ((2, 3, 4), [(3, 4)]),
        ((2, 5, 3, 4), [(1, 4), (3, 1)]),
        ((0, 3, 4), [(3, 4)]),
        ((2, 0, 4), [(1, 4)]),
    ],
)
def test_permanent_lora_broadcast(target_shape, product_shapes):
    model = nn.Module()
    model.weight = nn.Parameter(torch.randn(target_shape, dtype=torch.float64), requires_grad=False)
    before = model.weight.detach().clone()
    original = model.weight
    pointer = original.data_ptr()
    expected = before.clone()
    adapters = []
    sources = []
    for index, (rows, cols) in enumerate(product_shapes):
        rank = index + 2
        a = torch.randn(rank, cols, dtype=torch.float64)
        b = torch.randn(rows, rank, dtype=torch.float64)
        alpha = rank * 1.5 if index == 0 else None
        strength = 0.375 if index == 0 else -0.5
        delta = ParameterDelta.from_tensors(a=a, b=b, lora_alpha=alpha)
        adapters.append((Adapter({"weight": delta}), strength))
        sources.extend([(a, a.clone()), (b, b.clone())])
        expected += strength * (1.5 if alpha is not None else 1.0) * (b @ a)

    assert merge_adapter(model, adapters) == 1
    assert model.weight is original
    assert model.weight.data_ptr() == pointer
    torch.testing.assert_close(model.weight, expected)
    for source, snapshot in sources:
        torch.testing.assert_close(source, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("target_shape", "source_shape"),
    [
        ((2, 3, 4), (3, 4)),
        ((2, 5, 3, 4), (1, 5, 1, 4)),
        ((2, 3), (3,)),
        ((2, 5, 3), (1,)),
        ((2, 3, 4), ()),
        ((), ()),
        ((2, 0, 4), (1, 4)),
    ],
)
def test_permanent_dense_broadcast(target_shape, source_shape):
    model = nn.Module()
    model.weight = nn.Parameter(torch.randn(target_shape), requires_grad=False)
    before = model.weight.detach().clone()
    source = torch.randn(source_shape)
    adapter = Adapter({"weight": ParameterDelta.from_tensors(dense=source)})

    assert merge_adapter(model, [(adapter, -0.375)]) == 1
    torch.testing.assert_close(model.weight, before - 0.375 * source)


def test_mixed_broadcast_combines_different_shapes_in_one_compact_update(monkeypatch):
    model = nn.Module()
    model.weight = nn.Parameter(torch.randn(2, 5, 3, 4, dtype=torch.float64), requires_grad=False)
    before = model.weight.detach().clone()
    a = torch.randn(2, 4, dtype=torch.float64)
    b = torch.randn(1, 2, dtype=torch.float64)
    column = torch.randn(3, 1, dtype=torch.float64)
    scalar = torch.tensor(0.125, dtype=torch.float64)
    first = Adapter({"weight": ParameterDelta.from_tensors(a=a, b=b, dense=column, lora_alpha=3.0)})
    second = Adapter({"weight": ParameterDelta.from_tensors(dense=scalar)})
    updates = []
    original = RegularAdapter.merge_dense_

    def record_merge(target, update, strength, *, rounding_seed=None):
        updates.append(update.shape)
        original(target, update, strength, rounding_seed=rounding_seed)

    monkeypatch.setattr(RegularAdapter, "merge_dense_", staticmethod(record_merge))
    assert merge_adapter(model, [(first, 0.5), (second, -0.75)]) == 1

    # Independent head updates exercise composition without reproducing packing.
    expected = torch.stack([
        head + 0.5 * (1.5 * (b @ a) + column) - 0.75 * scalar
        for head in before.reshape(-1, 3, 4)
    ]).reshape_as(before)
    torch.testing.assert_close(model.weight, expected)
    assert updates == [(3, 4)]  # No update allocation per stacked head.


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_broadcast_lora_rounds_product_before_addition(dtype):
    a = torch.tensor([[0.75, 0.333]], dtype=dtype)
    b = torch.tensor([[0.667], [1.25]], dtype=dtype)
    model = nn.Module()
    model.weight = nn.Parameter(torch.full((3, 2, 2), 2.0, dtype=dtype), requires_grad=False)
    base = model.weight.detach().clone()
    adapter = Adapter({"weight": ParameterDelta.from_tensors(a=a, b=b)})

    merge_adapter(model, [(adapter, -0.5)])

    rounded_product = (b.float() @ a.float()).to(dtype)
    expected = (base.float() - 0.5 * rounded_product.float()).to(dtype)
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("device", "autocast_dtype"),
    [
        pytest.param("cpu", torch.bfloat16, id="cpu-bf16"),
        pytest.param("cuda", torch.float16, marks=CUDA, id="cuda-fp16"),
    ],
)
@pytest.mark.parametrize("with_dense", [False, True], ids=["factor-only", "mixed"])
def test_broadcast_merge_preserves_compute_dtype_under_autocast(device, autocast_dtype, with_dense):
    model = nn.Module()
    shape = (3, 2, 2)
    model.weight = nn.Parameter(torch.ones(shape, device=device, dtype=torch.float32), requires_grad=False)
    # 300 * 300 overflows FP16 and loses precision in BF16, but is exact in FP32.
    # A full-shape dense term also exercises the broadcast product in mixed merges.
    delta = ParameterDelta.from_tensors(
        a=torch.full((1, 2), 300.0),
        b=torch.full((2, 1), 300.0),
        dense=torch.full(shape, 2.0) if with_dense else None,
    )

    with torch.autocast(device_type=device, dtype=autocast_dtype):
        merge_adapter(model, [(Adapter({"weight": delta}), 1.0)])
        assert torch.is_autocast_enabled(device)

    expected = torch.full_like(model.weight, 90003.0 if with_dense else 90001.0)
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("target_shape", "source_shape", "error", "message"),
    [
        ((2, 3, 4), (2, 4), RuntimeError, "broadcast"),
        ((1, 4), (3, 4), ValueError, "would expand the target"),
        ((3, 4), (1, 3, 4), ValueError, "would expand the target"),
    ],
)
def test_dense_shape_rejection_precedes_all_mutation(target_shape, source_shape, error, message):
    model = nn.Module()
    model.first = nn.Parameter(torch.zeros(3, 4), requires_grad=False)
    model.second = nn.Parameter(torch.zeros(target_shape), requires_grad=False)
    adapter = Adapter({
        "first": ParameterDelta.from_tensors(dense=torch.ones(())),
        "second": ParameterDelta.from_tensors(dense=torch.ones(source_shape)),
    })

    with pytest.raises(error, match=message):
        merge_adapter(model, [(adapter, 1.0)])
    assert torch.count_nonzero(model.first) == 0
    assert torch.count_nonzero(model.second) == 0


@pytest.mark.parametrize(
    ("target_shape", "error", "message"),
    [
        ((2, 3, 4), RuntimeError, "broadcast"),
        ((1, 4), ValueError, "would expand the target"),
        ((4,), ValueError, "would expand the target"),
    ],
)
def test_lora_cannot_expand_target_or_mismatch_dimensions(target_shape, error, message):
    model = nn.Module()
    model.weight = nn.Parameter(torch.zeros(target_shape), requires_grad=False)
    adapter = Adapter({"weight": ParameterDelta.from_tensors(a=torch.ones(1, 4), b=torch.ones(2, 1))})

    with pytest.raises(error, match=message):
        merge_adapter(model, [(adapter, 1.0)])
    assert torch.count_nonzero(model.weight) == 0


@pytest.mark.parametrize("source_shape", [(3, 4), (1, 3, 4), ()])
def test_complete_values_remain_exact_shape(source_shape):
    model = nn.Module()
    model.weight = nn.Parameter(torch.empty(2, 3, 4, device="meta"), requires_grad=False)
    adapter = Adapter({"weight": ParameterValue.from_tensor(torch.ones(source_shape))})

    with pytest.raises(ValueError, match="Parameter value shape mismatch"):
        merge_adapter(model, [(adapter, 1.0)])
    assert model.weight.is_meta


@pytest.mark.parametrize("kind", ["lora", "dense", "mixed"])
def test_quantized_broadcast_requires_explicit_capability(kind):
    quanto = pytest.importorskip("optimum.quanto")
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

    weight = WeightQBytesTensor.create(
        quanto.qint8, 0, (3, 4), (4, 1),
        torch.ones(3, 4, dtype=torch.int8), torch.ones(3, 1), None,
    )
    model = nn.Module()
    model.first = nn.Parameter(torch.zeros(3, 4), requires_grad=False)
    model.weight = nn.Parameter(weight, requires_grad=False)
    delta = ParameterDelta.from_tensors(
        a=torch.ones(1, 4) if kind != "dense" else None,
        b=torch.ones(1, 1) if kind != "dense" else None,
        dense=torch.ones(3, 4) if kind == "mixed" else (torch.ones(()) if kind == "dense" else None),
    )
    adapter = Adapter({"first": ParameterDelta.from_tensors(dense=torch.ones(())), "weight": delta})

    with pytest.raises(ValueError, match="QuantoAdapter does not support broadcast"):
        merge_adapter(model, [(adapter, 1.0)])
    assert torch.count_nonzero(model.first) == 0
    torch.testing.assert_close(weight.dequantize(), torch.ones(3, 4))


@CUDA
def test_prepared_local_sources_reuse_mapped_views(tmp_path, monkeypatch):
    # A local region with both a row offset and noncontiguous column slices.
    class LocalAdapter(RegularAdapter):
        @staticmethod
        def logical_shape(_target):
            return (6, 8)

        @staticmethod
        def merge_local_shape_and_offsets(_target):
            return (4, 3), (2, 5)

    adapter = LocalAdapter()
    monkeypatch.setattr(parameter_delta_impl, "select_adapter", lambda _target: adapter)
    sources = {
        "dense": torch.arange(48, dtype=torch.float32).reshape(6, 8) / 8,
        "a": torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16,
        "b": torch.arange(12, dtype=torch.float32).reshape(6, 2) / 8,
    }
    path = tmp_path / "delta.safetensors"
    save_file(sources, path)
    with MappedCheckpoint(path) as checkpoint:
        mapped = {name: checkpoint.get_tensor(name) for name in sources}
    delta = ParameterDelta.from_tensors(**mapped)
    transform = ParameterDeltaTransform([delta.scaled(0.25)])
    target = nn.Parameter(torch.zeros(4, 3, device="cuda"), requires_grad=False)
    transform.validate_parameter(target)
    plan = transform._dense_plan
    assert plan is not None
    assert plan.update_shape == (4, 3)
    assert not plan.dense_sources[0][0].is_contiguous()
    for view, name in (
        (plan.dense_sources[0][0], "dense"),
        (plan.factors[0].a, "a"),
        (plan.factors[0].b, "b"),
    ):
        assert view.untyped_storage().data_ptr() == mapped[name].untyped_storage().data_ptr()
        assert file_slice(view) is not None

    def unexpected_localization(*_args, **_kwargs):
        raise AssertionError("application repeated source localization")

    monkeypatch.setattr(parameter_delta_impl, "_local_dense_view", unexpected_localization)
    monkeypatch.setattr(parameter_delta_impl, "_localize_materialized_weight_factors", unexpected_localization)
    expected_update = 0.25 * (sources["dense"] + sources["b"] @ sources["a"])[2:6, 5:8]
    with host_pin_manager.acquire(transform.storage_tensors()):
        try:
            for merge_count in (1, 2):
                transform.apply_parameter(target)
                torch.testing.assert_close(target.cpu(), merge_count * expected_update)
        finally:
            torch.cuda.synchronize()
    for name, source in mapped.items():
        torch.testing.assert_close(source, sources[name], rtol=0, atol=0)


class _StackedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 3, 3) / 4, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(2, 3) / 4, requires_grad=False)

    def forward(self, x):
        return (x @ self.weight.transpose(-1, -2) + self.bias[:, None, :]).mean(0)


class _StackedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = _StackedBlock()
        self.blocks = nn.ModuleList([_StackedBlock() for _ in range(3)])

    def forward(self, x):
        x = self.head(x)
        for block in self.blocks:
            x = block(x)
        return x


def _assert_mapped_delta_sources(adapter, sources):
    pointers = {tensor.data_ptr() for tensor in sources.values()}
    for delta in adapter.targets.values():
        assert isinstance(delta, ParameterDelta)
        backings = [] if delta.lora is None else [delta.lora.a, delta.lora.b]
        if delta.dense is not None:
            backings.append(delta.dense)
        for backing in backings:
            tensor = backing.make_cpu_param()
            assert file_slice(tensor) is not None
            assert tensor.data_ptr() in pointers


@CUDA
@pytest.mark.parametrize("block_mode", ["resident", "streaming", "rolling"])
def test_broadcast_activation_restores_mapped_model_and_sources(tmp_path, block_mode):
    model = _StackedModel()
    base = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    prefixes = ["head", "blocks.0", "blocks.1", "blocks.2"]
    lora_prefixes = prefixes[:-1]
    dense_prefixes = ["head", "blocks.1", "blocks.2"]
    sources = {
        "a": torch.tensor([[0.125, -0.25, 0.375]]),
        "b": torch.tensor([[0.5]]),
        "second_a": torch.tensor([[-0.125], [0.25]]),
        "second_b": torch.tensor([[0.5, -0.25], [-0.125, 0.5], [0.375, 0.125]]),
        "weight_delta": torch.tensor([0.125, 0.0, -0.125]),
        "bias_delta": torch.tensor(0.25),
    }
    path = tmp_path / "model-and-adapter.safetensors"
    save_file(base | sources, path)
    with MappedCheckpoint(path) as checkpoint:
        mapped = {name: checkpoint.get_tensor(name) for name in base | sources}
    model.load_state_dict({name: mapped[name] for name in base}, assign=True)
    first = Adapter.from_state_dict({
        key: value
        for prefix in lora_prefixes
        for key, value in (
            (f"{prefix}.lora_A.weight", mapped["a"]),
            (f"{prefix}.lora_B.weight", mapped["b"]),
        )
    })
    second = Adapter.from_state_dict({
        key: value
        for prefix in dense_prefixes
        for key, value in (
            (f"{prefix}.delta.weight", mapped["weight_delta"]),
            (f"{prefix}.delta.bias", mapped["bias_delta"]),
        )
    } | {
        "blocks.0.lora_A.weight": mapped["second_a"],
        "blocks.0.lora_B.weight": mapped["second_b"],
    })
    _assert_mapped_delta_sources(first, {name: mapped[name] for name in sources})
    _assert_mapped_delta_sources(second, {name: mapped[name] for name in sources})
    offloader = ModelOffloader.from_module(
        model, block_paths=("blocks",), block_mode=block_mode,
        block_compile=BlockCompileConfig(dynamic=False, fullgraph=True) if block_mode == "rolling" else None,
    )
    x = torch.tensor([[0.5, -0.75, 0.125], [1.0, 0.25, -0.5]])
    for strengths in [(0.5, -0.25), (-0.75, 0.125), None]:
        expected = x
        for prefix in prefixes:
            weight = base[f"{prefix}.weight"].clone()
            bias = base[f"{prefix}.bias"].clone()
            if strengths is not None:
                if prefix in lora_prefixes:
                    weight += strengths[0] * (sources["b"] @ sources["a"])
                if prefix == "blocks.0":
                    weight += strengths[1] * (sources["second_b"] @ sources["second_a"])
                if prefix in dense_prefixes:
                    weight += strengths[1] * sources["weight_delta"]
                    bias += strengths[1] * sources["bias_delta"]
            expected = torch.stack([
                F.linear(expected, head_weight, head_bias)
                for head_weight, head_bias in zip(weight, bias, strict=True)
            ]).mean(0)

        with activated_model(
            offloader, "cuda", adapters=[first, second] if strengths is not None else [],
            adapter_strengths=strengths,
        ), torch.inference_mode():
            for _ in range(2):
                torch.testing.assert_close(model(x.cuda()).cpu(), expected, rtol=2e-5, atol=2e-5)

        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, base[name], rtol=0, atol=0)
            assert tensor.data_ptr() == mapped[name].data_ptr()
        for name, tensor in mapped.items():
            assert file_slice(tensor) is not None
            torch.testing.assert_close(tensor, (base | sources)[name], rtol=0, atol=0)
