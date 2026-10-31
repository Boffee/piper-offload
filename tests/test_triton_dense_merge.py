"""Dense-update specializations of the raw Triton quantized merge kernels."""

from collections.abc import Sequence
from types import ModuleType
from typing import Any

import pytest
import torch


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _exact_lora_update(
    rows: int,
    cols: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build factors whose product needs no accumulation or rounding."""
    b = torch.zeros((rows, 16), device="cuda", dtype=dtype)
    b[:, 0] = 1
    a = torch.zeros((16, cols), device="cuda", dtype=dtype)
    a[0] = torch.linspace(-0.5, 0.5, cols, device="cuda", dtype=dtype)
    return b, a, b @ a


def _assert_same_buffers(
    lora: Sequence[torch.Tensor | None],
    dense: Sequence[torch.Tensor | None],
) -> None:
    assert len(lora) == len(dense)
    for lora_buffer, dense_buffer in zip(lora, dense, strict=True):
        if lora_buffer is None:
            assert dense_buffer is None
        else:
            assert dense_buffer is not None
            assert torch.equal(lora_buffer, dense_buffer)


def _adapter_case(  # noqa: PLR0911, PLR0915
    backend: str,
) -> tuple[Any, ModuleType, torch.Tensor, torch.Tensor]:
    rows, cols = 17, 128
    if backend in ("bnb4", "bnb4_nested"):
        import piper_offload.bnb4bit_adapter as adapter_module
        from piper_offload.bnb4bit_adapter import Bnb4bitAdapter
        from tests.test_bnb4bit_adapter import _make_nf4

        target = _make_nf4(
            rows=rows,
            cols=cols,
            dtype=torch.bfloat16,
            double_quant=backend == "bnb4_nested",
            device="cuda",
        )
        update = torch.randn_like(Bnb4bitAdapter.dequantize(target))
        return Bnb4bitAdapter, adapter_module, target, update
    if backend == "bnb8":
        import piper_offload.bnb8bit_adapter as adapter_module
        from piper_offload.bnb8bit_adapter import Bnb8bitAdapter
        from tests.test_bnb8bit_adapter import _make_int8

        target = _make_int8(rows=rows, cols=cols, device="cuda")
        update = torch.randn((rows, cols), device="cuda", dtype=torch.float16)
        return Bnb8bitAdapter, adapter_module, target, update
    if backend == "float8":
        import piper_offload.float8_adapter as adapter_module
        from piper_offload.float8_adapter import Float8Adapter
        from tests.test_float8_adapter import _make_float8

        target = _make_float8(
            rows=rows,
            cols=cols,
            dtype=torch.bfloat16,
            per_tensor=True,
            dynamic_activation=False,
        ).cuda()
        update = torch.randn_like(Float8Adapter.dequantize(target))
        return Float8Adapter, adapter_module, target, update
    if backend == "int8":
        import piper_offload.int8_adapter as adapter_module
        from piper_offload.int8_adapter import Int8Adapter
        from tests.test_int8_adapter import _make_affine_int8

        weight = torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16)
        target = _make_affine_int8(weight, layout="row")
        return Int8Adapter, adapter_module, target, torch.randn_like(weight)
    if backend == "mx":
        import piper_offload.mx_adapter as adapter_module
        from piper_offload.mx_adapter import MxAdapter
        from tests.test_mx_adapter import _quantize_mx

        weight = torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16)
        target = _quantize_mx(weight, elem_dtype=torch.float8_e4m3fn)
        return MxAdapter, adapter_module, target, torch.randn_like(weight)
    if backend == "nvfp4":
        import piper_offload.nvfp4_adapter as adapter_module
        from piper_offload.nvfp4_adapter import Nvfp4Adapter
        from tests.test_nvfp4_adapter import _make_nvfp4_cuda

        weight = torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16)
        target = _make_nvfp4_cuda(weight, swizzled=False, two_level=True)
        return Nvfp4Adapter, adapter_module, target, torch.randn_like(weight)
    if backend == "static_float8":
        import piper_offload.static_float8_adapter as adapter_module
        from piper_offload.static_float8_adapter import StaticFloat8Adapter
        from tests.test_static_float8_adapter import _make_static_float8

        target = _make_static_float8(
            rows=rows,
            cols=cols,
            dtype=torch.bfloat16,
        ).cuda()
        update = torch.randn_like(StaticFloat8Adapter.dequantize(target))
        return StaticFloat8Adapter, adapter_module, target, update
    if backend == "quanto":
        quanto = pytest.importorskip("optimum.quanto")
        import piper_offload.quanto_adapter as adapter_module
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor
        from piper_offload.quanto_adapter import QuantoAdapter

        data = torch.randint(-100, 101, (rows, cols), device="cuda", dtype=torch.int8)
        scale = torch.rand((rows, 1), device="cuda", dtype=torch.bfloat16).add_(0.1)
        target = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            data,
            scale,
            None,
        )
        update = torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16)
        return QuantoAdapter, adapter_module, target, update
    raise AssertionError(f"unknown dense merge backend {backend}")


@CUDA
@pytest.mark.parametrize(
    "backend",
    ["bnb4", "bnb4_nested", "bnb8", "float8", "int8", "mx", "nvfp4", "quanto", "static_float8"],
)
@pytest.mark.parametrize("beta", [1.0, -0.75, 0.0, 1.5])
def test_supported_adapters_dispatch_dense_updates_to_triton(
    backend: str,
    beta: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    adapter, adapter_module, target, update = _adapter_case(backend)
    expected = adapter.dequantize(target).mul(beta).add_(update, alpha=0.125)
    reference = adapter.dequantize(adapter.requantize(expected, like=target))

    def fail_reference(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(f"supported CUDA {backend} dense merge must use Triton")

    monkeypatch.setattr(adapter_module, "merge_dense_requantize_", fail_reference)
    adapter.merge_dense_(target, update, 0.125, beta=beta)
    actual = adapter.dequantize(target)
    reference_error = (reference.float() - expected.float()).abs().mean()
    rounding_error = torch.finfo(expected.dtype).eps * expected.float().abs().mean()
    assert (actual.float() - expected.float()).abs().mean() <= reference_error * 1.1 + rounding_error


@CUDA
@pytest.mark.parametrize(
    "backend",
    ["bnb4", "bnb4_nested", "bnb8", "float8", "int8", "mx", "nvfp4", "quanto", "static_float8"],
)
@pytest.mark.parametrize("use_triton", [False, True], ids=["reference", "triton"])
def test_lora_adapter_applies_alpha_and_beta(
    backend: str,
    use_triton: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    adapter, adapter_module, target, _ = _adapter_case(backend)
    target = adapter.requantize(torch.ones_like(adapter.dequantize(target)), like=target)
    base = adapter.dequantize(target)
    # Cancel the base while retaining a residual smaller than one compute-dtype
    # step. Rounding beta * base before addition would erase this residual.
    torch.testing.assert_close(base, base[:1].expand_as(base), rtol=0, atol=0)
    b = torch.full((base.shape[0], 1), 8.0, device=base.device, dtype=base.dtype)
    a = -base[:1].contiguous()
    beta = 1.0 + torch.finfo(base.dtype).eps / 4
    expected = (base.double() * beta + (b.double() @ a.double()) * 0.125).to(base.dtype)
    assert torch.all(expected > 0)
    reference = adapter.dequantize(adapter.requantize(expected, like=target))
    kernel_name = "quanto_qint8" if backend == "quanto" else backend.removesuffix("_nested")
    entry_point = f"_triton_merge_{kernel_name}_lora"
    if backend == "mx":
        entry_point += "_"
    original_kernel = getattr(adapter_module, entry_point)
    calls = []

    def record_kernel(*args, **kwargs):
        calls.append(kwargs["beta"])
        return original_kernel(*args, **kwargs)

    monkeypatch.setattr(adapter_module, entry_point, record_kernel if use_triton else None)
    adapter.merge_lora_(target, b, a, alpha=0.125, beta=beta)
    assert calls == ([beta] if use_triton else [])
    actual = adapter.dequantize(target)
    reference_error = (reference.float() - expected.float()).abs().mean()
    rounding_error = torch.finfo(expected.dtype).eps * expected.float().abs().mean()
    assert (actual.float() - expected.float()).abs().mean() <= reference_error * 1.1 + rounding_error


@CUDA
@pytest.mark.parametrize(
    "backend", ["bnb4", "bnb8", "float8", "int8", "mx", "nvfp4", "quanto", "static_float8"],
)
@pytest.mark.parametrize("update_kind", ["dense", "lora"])
def test_scaled_triton_merge_replays_seed_without_reference_or_global_rng(
    backend: str,
    update_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from piper_offload import ParameterValue
    from piper_offload.tensor_adapter_registry import param_representation

    pytest.importorskip("triton")
    adapter, adapter_module, source, update = _adapter_case(backend)
    b, a, lora_update = _exact_lora_update(*update.shape, update.dtype)
    if update_kind == "lora":
        update = lora_update
    backing = ParameterValue.from_tensor(source).backing
    expected = adapter.dequantize(source).mul(-0.75).add_(update, alpha=0.125)
    cuda_rng = torch.cuda.get_rng_state()
    cpu_rng = torch.get_rng_state()
    results = []

    def fail_reference(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("scaled stochastic CUDA merge must use Triton")

    monkeypatch.setattr(adapter_module, "merge_dense_requantize_", fail_reference)
    for _ in range(2):
        target = param_representation(backing.materialize(torch.device("cuda")))
        if update_kind == "dense":
            adapter.merge_dense_(target, update, alpha=0.125, beta=-0.75, rounding_seed=2**64 - 1)
        else:
            adapter.merge_lora_(target, b, a, alpha=0.125, beta=-0.75, rounding_seed=2**64 - 1)
        actual = adapter.dequantize(target)
        assert (actual.float() - expected.float()).abs().mean() < expected.float().abs().mean() * 0.2
        results.append(actual)
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert torch.equal(cpu_rng, torch.get_rng_state())


@CUDA
@pytest.mark.parametrize("per_group", [False, True])
@pytest.mark.parametrize(("dtype", "base", "scale", "increment"), [
    (torch.float16, 40000.0, -1.0, 0.0),
    (torch.float16, 40000.0, 2.0, -40000.0),
    (torch.bfloat16, 1.0, 0.125, 0.001),
])
def test_fused_source_scaling_avoids_overflow_and_cancellation(
    per_group: bool,
    dtype: torch.dtype,
    base: float,
    scale: float,
    increment: float,
) -> None:
    from piper_offload.float8_adapter import Float8Adapter
    from tests.test_float8_adapter import _make_float8

    pytest.importorskip("triton")
    target = _make_float8(
        weight=torch.full((17, 128), base, dtype=dtype, device="cuda"),
        group_size=32 if per_group else None,
        per_tensor=not per_group,
    )
    source = Float8Adapter.dequantize(target)
    update = torch.full_like(source, increment)
    expected = (source.double() * scale + update.double()).to(dtype)
    b = torch.ones((source.shape[0], 1), device="cuda", dtype=dtype)
    a = torch.full((1, source.shape[1]), increment, device="cuda", dtype=dtype)
    for kind in ("dense", "lora"):
        result = target.clone()
        if kind == "dense":
            Float8Adapter.merge_dense_(result, update, alpha=1.0, beta=scale)
        else:
            Float8Adapter.merge_lora_(result, b, a, alpha=1.0, beta=scale)
        actual = Float8Adapter.dequantize(result)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("beta", [1.0, -0.75])
@CUDA
def test_bnb8_dense_specialization_matches_exact_lora_update(beta: float) -> None:
    pytest.importorskip("triton")
    from piper_offload._triton_bnb8_lora import (
        merge_bnb8_dense,
        merge_bnb8_lora,
    )

    rows, cols = 17, 130
    b, a, update = _exact_lora_update(rows, cols, torch.float16)
    cb = torch.randint(-100, 101, (rows, cols), device="cuda", dtype=torch.int8)
    scb = torch.rand(rows, device="cuda", dtype=torch.float32).add_(0.1)

    _assert_same_buffers(
        merge_bnb8_lora(cb, scb, b, a, 0.25, beta=beta),
        merge_bnb8_dense(cb, scb, update, 0.25, beta=beta),
    )


@pytest.mark.parametrize("beta", [1.0, -0.75])
@CUDA
def test_static_float8_dense_specialization_matches_exact_lora_update(beta: float) -> None:
    pytest.importorskip("triton")
    from piper_offload._triton_static_float8_lora import (
        merge_static_float8_dense,
        merge_static_float8_lora,
    )

    rows, cols = 17, 130
    b, a, update = _exact_lora_update(rows, cols, torch.float16)
    qdata = torch.randn(rows, cols, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.tensor(0.02, device="cuda")

    _assert_same_buffers(
        merge_static_float8_lora(qdata, scale, b, a, 0.25, beta=beta),
        merge_static_float8_dense(qdata, scale, update, 0.25, beta=beta),
    )


@pytest.mark.parametrize("beta", [1.0, -0.75])
@CUDA
def test_int8_dense_specialization_matches_exact_lora_update(beta: float) -> None:
    pytest.importorskip("triton")
    from piper_offload._triton_int8_lora import (
        merge_int8_dense,
        merge_int8_lora,
    )

    rows, cols = 17, 128
    b, a, update = _exact_lora_update(rows, cols, torch.bfloat16)
    qdata = torch.randint(-100, 101, (rows, cols), device="cuda", dtype=torch.int8)
    scale = torch.rand((rows, 4), device="cuda", dtype=torch.bfloat16).add_(0.1)
    kwargs = {"asymmetric": False, "reduce_range": False}

    _assert_same_buffers(
        merge_int8_lora(qdata, scale, None, (1, 32), b, a, 0.25, **kwargs, beta=beta),
        merge_int8_dense(qdata, scale, None, (1, 32), update, 0.25, **kwargs, beta=beta),
    )


@pytest.mark.parametrize("beta", [1.0, -0.75])
@CUDA
@pytest.mark.parametrize("floating", [False, True])
def test_quanto_dense_specialization_matches_exact_lora_update(
    beta: float,
    floating: bool,
) -> None:
    pytest.importorskip("triton")
    from piper_offload._triton_quanto_lora import (
        merge_quanto_qfloat8_dense,
        merge_quanto_qfloat8_lora,
        merge_quanto_qint8_dense,
        merge_quanto_qint8_lora,
    )

    rows, cols = 17, 128
    b, a, update = _exact_lora_update(rows, cols, torch.bfloat16)
    scale = torch.rand((rows, 1), device="cuda", dtype=torch.bfloat16).add_(0.1)
    if floating:
        qdata = torch.randn(rows, cols, device="cuda").to(torch.float8_e4m3fn)
        lora_merge = merge_quanto_qfloat8_lora
        dense_merge = merge_quanto_qfloat8_dense
    else:
        qdata = torch.randint(-100, 101, (rows, cols), device="cuda", dtype=torch.int8)
        lora_merge = merge_quanto_qint8_lora
        dense_merge = merge_quanto_qint8_dense

    _assert_same_buffers(
        lora_merge(qdata, scale, 0, b, a, 0.25, beta=beta),
        dense_merge(qdata, scale, 0, update, 0.25, beta=beta),
    )


@pytest.mark.parametrize("beta", [1.0, -0.75])
@CUDA
@pytest.mark.parametrize("per_group", [False, True])
def test_scaled_float8_dense_specialization_matches_exact_lora_update(
    beta: float,
    per_group: bool,
) -> None:
    pytest.importorskip("triton")
    from piper_offload._triton_float8_lora import (
        merge_float8_dense,
        merge_float8_lora,
    )

    rows, cols = 17, 128
    b, a, update = _exact_lora_update(rows, cols, torch.bfloat16)
    qdata = torch.randn(rows, cols, device="cuda").to(torch.float8_e4m3fn)
    if per_group:
        block_size = (1, 32)
        scale = torch.rand((rows, 4), device="cuda").add_(0.1)
    else:
        block_size = (rows, cols)
        scale = torch.tensor(0.1, device="cuda")

    _assert_same_buffers(
        merge_float8_lora(qdata, scale, block_size, b, a, 0.25, beta=beta),
        merge_float8_dense(qdata, scale, block_size, update, 0.25, beta=beta),
    )


@pytest.mark.parametrize("beta", [1.0, -0.75])
@CUDA
def test_bnb4_dense_specialization_matches_exact_lora_update(beta: float) -> None:
    pytest.importorskip("triton")
    from piper_offload._triton_bnb4_lora import (
        merge_bnb4_dense,
        merge_bnb4_lora,
    )

    rows, cols = 17, 128
    b, a, update = _exact_lora_update(rows, cols, torch.bfloat16)
    packed = torch.randint(0, 256, (rows * cols // 2,), device="cuda", dtype=torch.uint8)
    absmax = torch.rand(rows * (cols // 64), device="cuda").add_(0.1)
    code = torch.linspace(-1, 1, 16, device="cuda")
    common = (packed, absmax, code, None, None, None, (rows, cols), 64, "nf4")

    _assert_same_buffers(
        merge_bnb4_lora(*common, b, a, 0.25, beta=beta),
        merge_bnb4_dense(*common, update, 0.25, beta=beta),
    )


@pytest.mark.parametrize("beta", [1.0, -0.75])
@CUDA
def test_mx_dense_specialization_matches_exact_lora_update(beta: float) -> None:
    pytest.importorskip("triton")
    from piper_offload._triton_mx_lora import merge_mx_dense_, merge_mx_lora_

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        pytest.skip("E8M0 dtype unavailable")
    rows, cols = 17, 128
    b, a, update = _exact_lora_update(rows, cols, torch.bfloat16)
    qdata = torch.randn(rows, cols, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.ones((rows, cols // 32), device="cuda", dtype=e8m0_dtype)
    lora_buffers = (qdata.clone(), scale.clone())
    dense_buffers = (qdata.clone(), scale.clone())

    merge_mx_lora_(
        *lora_buffers,
        torch.float8_e4m3fn,
        32,
        torch.bfloat16,
        b,
        a,
        0.25,
        scaling_mode=0,
        swizzled=False,
        beta=beta,
    )
    merge_mx_dense_(
        *dense_buffers,
        torch.float8_e4m3fn,
        32,
        torch.bfloat16,
        update,
        0.25,
        scaling_mode=0,
        swizzled=False,
        beta=beta,
    )
    _assert_same_buffers(lora_buffers, dense_buffers)


@pytest.mark.parametrize("beta", [1.0, -0.75])
@CUDA
@pytest.mark.parametrize("global_scale", [False, True])
def test_nvfp4_dense_specialization_matches_exact_lora_update(
    beta: float,
    global_scale: bool,
) -> None:
    pytest.importorskip("triton")
    from piper_offload._triton_nvfp4_lora import (
        merge_nvfp4_dense,
        merge_nvfp4_lora,
    )

    rows, cols = 17, 128
    b, a, update = _exact_lora_update(rows, cols, torch.bfloat16)
    qdata = torch.randint(0, 256, (rows, cols // 2), device="cuda", dtype=torch.uint8)
    scale = torch.ones((rows, cols // 16), device="cuda", dtype=torch.float8_e4m3fn)
    per_tensor_scale = torch.tensor(0.01, device="cuda") if global_scale else None

    _assert_same_buffers(
        merge_nvfp4_lora(qdata, scale, per_tensor_scale, 16, False, b, a, 0.25, beta=beta),
        merge_nvfp4_dense(qdata, scale, per_tensor_scale, 16, False, update, 0.25, beta=beta),
    )
