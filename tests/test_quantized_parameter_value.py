"""Cross-format tests for quantized exact-name parameter values."""

import pytest
import torch
from torch import nn

from piper_offload import (
    Adapter,
    BlockCompileConfig,
    ModelOffloader,
    ParameterValue,
    merge_adapter,
    register_adapter,
)
from piper_offload.tensor_adapter_registry import (
    param_representation,
    select_adapter,
)
from piper_offload.tensor_adapters import (
    DenseMergeTensorAdapter,
    DequantizeTensorAdapter,
)
from tests.conftest import activated_model, block_components

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_QUANT_KINDS = (
    "torchao-float8",
    "torchao-static-float8",
    "torchao-int8",
    "torchao-mxfp8",
    "torchao-mxfp4",
    "torchao-nvfp4",
    "quanto-qint8",
    "bnb4",
    "bnb8",
    "piper-convrot-int8",
    "piper-convrot-nvfp4",
)


def _make_quantized(kind: str) -> torch.Tensor:  # noqa: PLR0911, PLR0912
    """Build one physical representation supported by dense merge."""
    if kind == "torchao-float8":
        from tests.test_float8_adapter import _make_float8

        return _make_float8()
    elif kind == "torchao-static-float8":
        from tests.test_static_float8_adapter import _make_static_float8

        return _make_static_float8()
    elif kind == "torchao-int8":
        from tests.test_int8_adapter import _make_int8

        return _make_int8()
    elif kind == "torchao-mxfp8":
        from tests.test_mx_adapter import _make_mx

        return _make_mx(elem_dtype=torch.float8_e4m3fn)
    elif kind == "torchao-mxfp4":
        from tests.test_mx_adapter import _make_mx

        elem_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if elem_dtype is None:
            pytest.skip("torch lacks float4_e2m1fn_x2")
        return _make_mx(elem_dtype=elem_dtype)
    elif kind == "torchao-nvfp4":
        from tests.test_nvfp4_adapter import _make_nvfp4

        return _make_nvfp4()
    elif kind == "quanto-qint8":
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 16, 16
        return WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            torch.randint(-127, 128, (rows, cols), dtype=torch.int8),
            torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25),
            None,
        )
    elif kind == "bnb4":
        from tests.test_bnb4bit_adapter import _make_nf4

        return _make_nf4()
    elif kind == "bnb8":
        from tests.test_bnb8bit_adapter import _make_int8

        return _make_int8()
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
    assert isinstance(source_adapter, DequantizeTensorAdapter)
    dense_source = source_adapter.dequantize(source).cpu()

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
        dense_target = target_adapter.dequantize(target).cpu()

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
    assert isinstance(source_adapter, DequantizeTensorAdapter)
    dense_source = source_adapter.dequantize(source).cpu()
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
    dense_target = target_adapter.dequantize(target).cpu()
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


@CUDA
def test_auto_mode_falls_back_for_incompatible_value_adapter() -> None:
    from tests._block_compile_helpers import _BlockModel
    from tests.test_tensor_adapter_registry import (
        _ExternalAdapter,
        _ExternalTensor,
    )

    dim = 8
    with torch.device("meta"):
        model = _BlockModel(width=dim)
    offloader = ModelOffloader.from_module(
        model,
        block_paths=("blocks",),
        block_mode="auto",
        block_compile=BlockCompileConfig(fullgraph=True),
    )
    component = block_components(offloader)[0]
    assert component.block_mode == "rolling"
    assert component._auto_rolling
    assert component._auto_fallback_compile is None

    remove_adapter = register_adapter(_ExternalAdapter)
    try:
        incompatible = Adapter.from_state_dict(
            {
                f"blocks.{idx}.proj.weight": torch.Tensor._make_subclass(
                    _ExternalTensor,
                    torch.eye(dim),
                    False,
                )
                for idx in range(2)
            },
        )
        offloader.activate("cuda", adapters=[incompatible])
        assert component._active_runtime is component._eager_runtime
        assert component._auto_fallback_compile is not None
        assert component._auto_fallback_compile.installed
        assert not component._block_compile.installed
    finally:
        offloader.deactivate()
        remove_adapter()
    assert component._auto_fallback_compile is not None
    assert not component._auto_fallback_compile.installed
