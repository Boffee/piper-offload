"""Cross-format tests for quantized exact-name parameter values."""

import pytest
import torch
from torch import nn

from piper_offload import (
    Adapter,
    ModelOffloader,
    ParameterValue,
    merge_adapter,
)
from piper_offload.tensor_adapter_registry import (
    param_representation,
    select_adapter,
)
from piper_offload.tensor_adapters import DenseMergeTensorAdapter
from tests.conftest import activated_model


def _to_dense(tensor: torch.Tensor) -> torch.Tensor:
    dequantize = getattr(tensor, "dequantize", None)
    return dequantize() if callable(dequantize) else tensor.clone()

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_QUANT_KINDS = (
    "piper-nvfp4",
    "piper-convrot-int8",
    "piper-convrot-nvfp4",
)


def _make_quantized(kind: str) -> torch.Tensor:
    """Build one physical representation supported by dense merge."""
    if kind == "piper-nvfp4":
        from tests.test_piper_nvfp4_adapter import _make_piper_nvfp4

        return _make_piper_nvfp4()[0]
    elif kind == "piper-convrot-int8":
        from tests.test_piper_convrot_int8_adapter import _make_convrot

        return _make_convrot()
    elif kind == "piper-convrot-nvfp4":
        from tests.test_piper_convrot_nvfp4_adapter import (
            _make_convrot_nvfp4,
        )

        return _make_convrot_nvfp4()[0]
    else:
        raise AssertionError(f"unknown quant kind {kind!r}")


@CUDA
@pytest.mark.parametrize("kind", _QUANT_KINDS)
@pytest.mark.parametrize("strength", [1.0, 0.5], ids=["exact", "scaled"])
def test_quantized_parameter_value_uses_dense_merge_support(
    kind: str,
    strength: float,
) -> None:
    source = _make_quantized(kind)
    source_adapter = select_adapter(source)
    assert isinstance(source_adapter, DenseMergeTensorAdapter)
    dense_source = _to_dense(source).cpu()

    adapter = Adapter.from_state_dict(
        {"weight": source},
        scale_parameter_values=True,
    )
    value = adapter.targets["weight"]
    assert isinstance(value, ParameterValue)
    model = nn.Module()
    model.weight = nn.Parameter(
        torch.empty(value.backing.logical_shape, device="meta"),
        requires_grad=False,
    )
    offloader = ModelOffloader.from_module(model)

    with activated_model(
        offloader,
        "cuda",
        adapters=[adapter],
        adapter_strengths=[strength],
        stochastic_rounding=False,
    ):
        target = param_representation(model.weight)
        target_adapter = select_adapter(target)
        assert type(target_adapter) is type(source_adapter)
        dense_target = _to_dense(target).cpu()

    assert model.weight.is_meta
    assert tuple(dense_target.shape) == value.backing.logical_shape
    assert bool(torch.isfinite(dense_target).all())
    if strength == 1.0:
        torch.testing.assert_close(dense_target, dense_source)
    else:
        scaled_error = (dense_target - dense_source * strength).abs().mean()
        unchanged_error = (dense_target - dense_source).abs().mean()
        assert scaled_error < unchanged_error


@pytest.mark.parametrize("kind", _QUANT_KINDS)
def test_quantized_parameter_value_permanent_scaling(kind: str) -> None:
    source = _make_quantized(kind)
    source_adapter = select_adapter(source)
    dense_source = _to_dense(source).cpu()
    adapter = Adapter.from_state_dict(
        {"weight": source},
        scale_parameter_values=True,
    )
    value = adapter.targets["weight"]
    assert isinstance(value, ParameterValue)
    model = nn.Module()
    model.weight = nn.Parameter(
        torch.empty(value.backing.logical_shape, device="meta"),
        requires_grad=False,
    )

    assert (
        merge_adapter(
            model,
            [(adapter, 0.5)],
            stochastic_rounding=False,
        )
        == 1
    )

    target = param_representation(model.weight)
    target_adapter = select_adapter(target)
    assert type(target_adapter) is type(source_adapter)
    dense_target = _to_dense(target).cpu()
    scaled_error = (dense_target - dense_source * 0.5).abs().mean()
    unchanged_error = (dense_target - dense_source).abs().mean()
    assert scaled_error < unchanged_error


def test_structured_parameter_value_dtype_is_representation_owned() -> None:
    source = _make_quantized("piper-convrot-int8")
    value = ParameterValue.from_tensor(
        source,
        dtype=source.dtype,
    )

    assert value.backing.compute_dtype is source.dtype
    adapter = Adapter.from_state_dict(
        {"weight": source},
        dtype=source.dtype,
    )
    captured_value = adapter.targets["weight"]
    assert isinstance(captured_value, ParameterValue)
    assert captured_value.backing.compute_dtype is source.dtype
    with pytest.raises(ValueError, match="Prequantize"):
        ParameterValue.from_tensor(
            source,
            dtype=torch.float16,
        )


@pytest.mark.parametrize("kind", ["piper-nvfp4", "piper-convrot-int8", "piper-convrot-nvfp4"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_scaled_native_dense_merge_preserves_storage_calibration_and_seed(kind, device):
    source = _make_quantized(kind).to(device)
    source.act_per_tensor_scale = torch.tensor(0.75, device=device)
    adapter = select_adapter(source)
    before = _to_dense(source).float()
    update = torch.linspace(-2, 2, before.numel(), device=device).reshape(before.shape)
    update = update.to(adapter.compute_dtype(source))
    expected = before * -0.5 + update.float() * 0.25
    encodings = []
    for _ in range(2):
        target = source.clone()
        names = ["qdata", "scale", "act_per_tensor_scale"]
        if hasattr(target, "per_tensor_scale") and target.per_tensor_scale is not None:
            names.append("per_tensor_scale")
        pointers = [getattr(target, name).data_ptr() for name in names]
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
        adapter.merge_dense_(target, update, 0.25, beta=-0.5, rounding_seed=2**64 - 1)
        assert torch.equal(torch.get_rng_state(), rng)
        if cuda_rng is not None:
            assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
        assert pointers == [getattr(target, name).data_ptr() for name in names]
        torch.testing.assert_close(target.act_per_tensor_scale, source.act_per_tensor_scale)
        actual = _to_dense(target).float()
        # NVFP4's coarse code grid allows larger error than INT8.
        tolerance = 0.03 if kind == "piper-convrot-int8" else 0.2
        assert (actual - expected).abs().mean() < expected.abs().mean() * tolerance
        encodings.append([getattr(target, name).clone() for name in names])
    for first, replay in zip(*encodings, strict=True):
        assert torch.equal(first.reshape(-1).view(torch.uint8), replay.reshape(-1).view(torch.uint8))
    torch.testing.assert_close(_to_dense(source).float(), before)
