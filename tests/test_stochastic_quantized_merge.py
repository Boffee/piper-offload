"""Correctness-first stochastic LoRA merge coverage across quant families."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

import piper_offload._torchao_nvfp4 as nvfp4_impl
import piper_offload.float8_adapter as float8_adapter_impl
import piper_offload.int8_adapter as int8_adapter_impl
import piper_offload.mx_adapter as mx_adapter_impl
import piper_offload.nvfp4_adapter as nvfp4_adapter_impl
import piper_offload.static_float8_adapter as static_float8_adapter_impl

from piper_offload import (
    LoRATransform,
    ScaledLoRAFactor,
)
from piper_offload.float8_adapter import Float8Adapter
from piper_offload.int8_adapter import Int8Adapter
from piper_offload.mx_adapter import MxAdapter
from piper_offload.nvfp4_adapter import Nvfp4Adapter
from piper_offload.static_float8_adapter import StaticFloat8Adapter
from piper_offload import derive_seed
from tests.test_float8_adapter import _make_float8
from tests.test_int8_adapter import _make_affine_int8, _with_act_pre_scale
from tests.test_mx_adapter import _quantize_mx
from tests.test_nvfp4_adapter import _make_nvfp4_cuda
from tests.test_static_float8_adapter import _make_static_float8

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _seed(*, key: str, merge_index: int = 0) -> int:
    return derive_seed(key, merge_index)


def _factors(
    shape: tuple[int, int],
    dtype: torch.dtype,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(2026)
    a = torch.randn(
        5,
        shape[1],
        dtype=dtype,
        device=device,
        generator=generator,
    ).mul_(0.2)
    b = torch.randn(
        shape[0],
        5,
        dtype=dtype,
        device=device,
        generator=generator,
    ).mul_(0.2)
    return b, a


def _assert_fused_replay(
    adapter: object,
    make_target: Callable[[], torch.Tensor],
    *,
    rows: int,
    cols: int,
    dtype: torch.dtype,
    qdata: Callable[[torch.Tensor], torch.Tensor],
    qparams: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
) -> None:
    """Assert backend-local replay and seed-independent finalized qparams."""
    b, a = _factors((rows, cols), dtype, device=torch.device("cuda"))
    seeds = (
        _seed(key="triton.weight"),
        _seed(key="triton.weight"),
        _seed(key="triton.other_weight"),
    )
    targets = []
    for seed in seeds:
        target = make_target()
        adapter.merge_lora_(target, b, a, 0.75, rounding_seed=seed)
        targets.append(target)

    first, replay, other = targets
    assert torch.equal(qdata(first), qdata(replay))
    assert not torch.equal(qdata(first), qdata(other))
    for first_param, replay_param, other_param in zip(
        qparams(first),
        qparams(replay),
        qparams(other),
        strict=True,
    ):
        assert torch.equal(first_param, replay_param)
        assert torch.equal(first_param, other_param)


def _run(
    adapter: object,
    make_target: Callable[[], torch.Tensor],
    *,
    key: str,
) -> torch.Tensor:
    target = make_target()
    shape = adapter.logical_shape(target)
    b, a = _factors(shape, adapter.compute_dtype(target), device=target.device)
    adapter.merge_lora_(
        target,
        b,
        a,
        0.75,
        rounding_seed=_seed(key=key),
    )
    return target


@pytest.mark.parametrize(
    "float8_dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
)
def test_scaled_float8_replays_codes_and_keeps_scales(
    float8_dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(30)
    weight = torch.randn(16, 64, dtype=torch.bfloat16, generator=generator)

    def make() -> torch.Tensor:
        return _make_float8(
            rows=16,
            cols=64,
            weight=weight.clone(),
            float8_dtype=float8_dtype,
            group_size=16,
            dynamic_activation=False,
        )

    first = _run(Float8Adapter, make, key="float8.weight")
    replay = _run(Float8Adapter, make, key="float8.weight")
    other = _run(Float8Adapter, make, key="float8.other_weight")
    assert torch.equal(first.qdata.view(torch.uint8), replay.qdata.view(torch.uint8))
    assert torch.equal(first.scale, other.scale)
    assert not torch.equal(first.qdata.view(torch.uint8), other.qdata.view(torch.uint8))


def test_static_float8_preserves_activation_calibration() -> None:
    generator = torch.Generator().manual_seed(40)
    weight = torch.randn(16, 32, dtype=torch.bfloat16, generator=generator)

    def make() -> torch.Tensor:
        return _make_static_float8(
            rows=16,
            cols=32,
            weight=weight.clone(),
            act_scale_value=0.03125,
        )

    first = _run(StaticFloat8Adapter, make, key="static.weight")
    other = _run(StaticFloat8Adapter, make, key="static.other_weight")
    assert torch.equal(first.scale, other.scale)
    assert torch.equal(first.act_quant_scale, other.act_quant_scale)
    assert first.act_quant_scale.item() == pytest.approx(0.03125)
    assert not torch.equal(first.qdata.view(torch.uint8), other.qdata.view(torch.uint8))


def test_affine_int8_preserves_qparams_and_activation_pre_scale() -> None:
    generator = torch.Generator().manual_seed(50)
    weight = torch.randn(18, 48, dtype=torch.bfloat16, generator=generator)
    pre_scale = torch.linspace(0.5, 1.5, 48, dtype=torch.float32)

    def make() -> torch.Tensor:
        return _with_act_pre_scale(
            _make_affine_int8(
                weight.clone(),
                layout="group",
                asymmetric=True,
                reduce_range=True,
            ),
            pre_scale,
        )

    first = _run(Int8Adapter, make, key="int8.weight")
    replay = _run(Int8Adapter, make, key="int8.weight")
    other = _run(Int8Adapter, make, key="int8.other_weight")
    assert torch.equal(first.qdata, replay.qdata)
    assert torch.equal(first.scale, other.scale)
    assert torch.equal(first.zero_point, other.zero_point)
    assert torch.equal(first.act_pre_scale, pre_scale)
    assert not torch.equal(first.qdata, other.qdata)


def test_affine_int8_factor_aware_transform_advances_rounding_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(51)
    weight = torch.randn(12, 48, dtype=torch.bfloat16, generator=generator)
    pre_scale = torch.linspace(0.5, 1.5, 48, dtype=torch.float32)
    qt = _with_act_pre_scale(
        _make_affine_int8(weight, layout="row"),
        pre_scale,
    )
    param = nn.Parameter(qt, requires_grad=False)
    a = torch.randn(4, 48, generator=generator)
    b = torch.randn(12, 4, generator=generator)
    transform = LoRATransform(
        [ScaledLoRAFactor.from_tensors(a, b, 0.25)],
        stochastic_rounding=True,
        target_key="int8.prepared.weight",
    )
    validated: list[int] = []
    merged: list[int] = []
    original_validate = Int8Adapter.validate_prepared_lora_merge
    original_merge = Int8Adapter.merge_prepared_lora_

    def tracked_validate(
        target: torch.Tensor,
        staged_b: torch.Tensor,
        staged_a: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        assert rounding_seed is not None
        validated.append(rounding_seed)
        original_validate(
            target,
            staged_b,
            staged_a,
            alpha,
            beta=beta,
            rounding_seed=rounding_seed,
        )

    def tracked_merge(
        target: torch.Tensor,
        staged_b: torch.Tensor,
        staged_a: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        assert rounding_seed is not None
        merged.append(rounding_seed)
        original_merge(
            target,
            staged_b,
            staged_a,
            alpha,
            beta=beta,
            rounding_seed=rounding_seed,
        )

    monkeypatch.setattr(
        Int8Adapter,
        "validate_prepared_lora_merge",
        staticmethod(tracked_validate),
    )
    monkeypatch.setattr(
        Int8Adapter,
        "merge_prepared_lora_",
        staticmethod(tracked_merge),
    )

    transform.validate_target(param)
    transform.apply(param)
    transform.apply(param)

    first_seed = _seed(key="int8.prepared.weight")
    second_seed = _seed(key="int8.prepared.weight", merge_index=1)
    assert validated == [first_seed]
    assert merged == [first_seed, second_seed]
    assert torch.isfinite(param.data.dequantize()).all()
    assert torch.equal(param.data.act_pre_scale, pre_scale)


@pytest.mark.parametrize(
    "elem_dtype",
    [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        pytest.param(getattr(torch, "float4_e2m1fn_x2", None)),
    ],
)
def test_mx_replays_packed_codes_and_keeps_e8m0_scales(
    elem_dtype: torch.dtype | None,
) -> None:
    if elem_dtype is None:
        pytest.skip("MXFP4 dtype unavailable")
    generator = torch.Generator().manual_seed(60)
    weight = torch.randn(16, 64, dtype=torch.bfloat16, generator=generator)

    def make() -> torch.Tensor:
        return _quantize_mx(
            weight.clone(),
            elem_dtype=elem_dtype,
            is_swizzled_scales=True,
        )

    first = _run(MxAdapter, make, key="mx.weight")
    replay = _run(MxAdapter, make, key="mx.weight")
    other = _run(MxAdapter, make, key="mx.other_weight")
    assert torch.equal(first.qdata.view(torch.uint8), replay.qdata.view(torch.uint8))
    assert torch.equal(first.scale.view(torch.uint8), other.scale.view(torch.uint8))
    assert not torch.equal(first.qdata.view(torch.uint8), other.qdata.view(torch.uint8))


def test_nvfp4_uses_final_two_level_swizzled_scales() -> None:
    pytest.importorskip("numpy")
    module = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    generator = torch.Generator().manual_seed(70)
    weight = torch.randn(16, 64, dtype=torch.bfloat16, generator=generator)

    def make() -> torch.Tensor:
        per_tensor_scale = module.per_tensor_amax_to_scale(
            weight.abs().max().to(torch.float32)
        )
        return module.NVFP4Tensor.to_nvfp4(
            weight.clone(),
            per_tensor_scale=per_tensor_scale,
            is_swizzled_scales=True,
            use_triton_kernel=False,
        )

    first = _run(Nvfp4Adapter, make, key="nvfp4.weight")
    replay = _run(Nvfp4Adapter, make, key="nvfp4.weight")
    other = _run(Nvfp4Adapter, make, key="nvfp4.other_weight")
    assert torch.equal(first.qdata, replay.qdata)
    assert torch.equal(first.scale.view(torch.uint8), other.scale.view(torch.uint8))
    assert torch.equal(first.per_tensor_scale, other.per_tensor_scale)
    assert not torch.equal(first.qdata, other.qdata)


def test_nvfp4_probabilities_use_final_block_times_global_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    generator = torch.Generator().manual_seed(71)
    values = torch.randn(16, 64, dtype=torch.bfloat16, generator=generator)
    per_tensor_scale = module.per_tensor_amax_to_scale(
        values.abs().max().to(torch.float32)
    )
    like = module.NVFP4Tensor.to_nvfp4(
        values,
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=True,
        use_triton_kernel=False,
    )
    captured: dict[str, torch.Tensor] = {}
    original = nvfp4_impl.stochastic_codebook_indices

    def capture_normalized(
        normalized: torch.Tensor,
        codebook: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        captured["normalized"] = normalized.clone()
        return original(normalized, codebook, **kwargs)

    monkeypatch.setattr(
        nvfp4_impl,
        "stochastic_codebook_indices",
        capture_normalized,
    )
    rounded = nvfp4_impl.requantize_nvfp4_tensor(
        values,
        like=like,
        rounding_seed=_seed(key="nv-final-scale.weight"),
    )

    element_scale = rounded.get_hp_scales().repeat_interleave(
        rounded.block_size,
        dim=-1,
    )
    expected = values.to(torch.float32) / element_scale.to(torch.float32)
    torch.testing.assert_close(
        captured["normalized"],
        expected,
        rtol=0,
        atol=0,
    )


@CUDA
def test_triton_int8_stochastic_merge_replays_and_keeps_qparams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, cols = 24, 69
    weight = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)

    def make() -> torch.Tensor:
        return _make_affine_int8(weight.clone(), layout="group", asymmetric=True)

    def fail_fallback(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("supported stochastic INT8 merge used the fallback")

    monkeypatch.setattr(int8_adapter_impl, "dequantize_int8_tensor", fail_fallback)
    _assert_fused_replay(
        Int8Adapter,
        make,
        rows=rows,
        cols=cols,
        dtype=torch.bfloat16,
        qdata=lambda target: target.qdata,
        qparams=lambda target: (target.scale, target.zero_point),
    )


@CUDA
def test_triton_int8_uses_fp32_probability_with_bfloat16_scales() -> None:
    pytest.importorskip("triton")
    from piper_offload._triton_int8_lora import merge_int8_lora

    rows, cols = 16, 8192
    qdata = torch.zeros(rows, cols, device="cuda", dtype=torch.int8)
    scale = torch.ones(rows, 1, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(rows, 1, device="cuda", dtype=torch.bfloat16)
    a = torch.full((1, cols), 128.0, device="cuda", dtype=torch.bfloat16)
    a[0, -1] = 254.0

    output, final_scale, _ = merge_int8_lora(
        qdata,
        scale,
        None,
        (1, cols),
        b,
        a,
        1.0,
        asymmetric=False,
        reduce_range=False,
        rounding_seed=_seed(key="int8-bfloat16-probability.weight"),
    )

    samples = output[:, :-1]
    assert bool(((samples == 64) | (samples == 65)).all())
    expected_upper_probability = 128.0 / float(final_scale[0, 0]) - 64.0
    observed_upper_probability = float((samples == 65).to(torch.float32).mean())
    assert observed_upper_probability == pytest.approx(
        expected_upper_probability,
        abs=0.01,
    )


@CUDA
def test_triton_float8_stochastic_merge_replays_and_keeps_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, cols = 16, 32
    weight = torch.randn(rows, cols, dtype=torch.bfloat16)

    def make() -> torch.Tensor:
        return _make_float8(
            rows=rows,
            cols=cols,
            group_size=8,
            weight=weight.clone(),
        ).cuda()

    def fail_fallback(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("supported stochastic Float8 merge used the fallback")

    monkeypatch.setattr(float8_adapter_impl, "dequantize_float8_tensor", fail_fallback)
    _assert_fused_replay(
        Float8Adapter,
        make,
        rows=rows,
        cols=cols,
        dtype=torch.bfloat16,
        qdata=lambda target: target.qdata.view(torch.uint8),
        qparams=lambda target: (target.scale,),
    )


@CUDA
def test_triton_static_float8_stochastic_merge_replays_and_keeps_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, cols = 16, 32
    weight = torch.randn(rows, cols, dtype=torch.bfloat16)

    def make() -> torch.Tensor:
        return _make_static_float8(
            rows=rows,
            cols=cols,
            weight=weight.clone(),
        ).cuda()

    def fail_fallback(*_args: object, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("supported stochastic static-FP8 merge used the fallback")

    monkeypatch.setattr(
        static_float8_adapter_impl,
        "_torch_merge_static_float8_lora",
        fail_fallback,
    )
    _assert_fused_replay(
        StaticFloat8Adapter,
        make,
        rows=rows,
        cols=cols,
        dtype=torch.bfloat16,
        qdata=lambda target: target.qdata.view(torch.uint8),
        qparams=lambda target: (target.scale,),
    )


@CUDA
@pytest.mark.parametrize(
    "elem_dtype",
    [torch.float8_e4m3fn, getattr(torch, "float4_e2m1fn_x2", None)],
)
def test_triton_mx_stochastic_merge_replays_and_keeps_scales(
    elem_dtype: torch.dtype | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if elem_dtype is None:
        pytest.skip("Torch build has no packed E2M1 dtype")
    rows, cols = 16, 64
    weight = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)

    def make() -> torch.Tensor:
        return _quantize_mx(weight.clone(), elem_dtype=elem_dtype)

    def fail_fallback(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("supported stochastic MX merge used the fallback")

    monkeypatch.setattr(mx_adapter_impl, "_torch_merge_mx_lora_", fail_fallback)
    _assert_fused_replay(
        MxAdapter,
        make,
        rows=rows,
        cols=cols,
        dtype=torch.bfloat16,
        qdata=lambda target: target.qdata.view(torch.uint8),
        qparams=lambda target: (target.scale.view(torch.uint8),),
    )


@CUDA
def test_triton_nvfp4_stochastic_merge_replays_and_keeps_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, cols = 16, 64
    weight = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)

    def make() -> torch.Tensor:
        return _make_nvfp4_cuda(
            weight.clone(),
            swizzled=True,
            two_level=True,
        )

    def fail_fallback(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("supported stochastic NVFP4 merge used the fallback")

    monkeypatch.setattr(nvfp4_adapter_impl, "_torch_merge_nvfp4_lora_", fail_fallback)
    _assert_fused_replay(
        Nvfp4Adapter,
        make,
        rows=rows,
        cols=cols,
        dtype=torch.bfloat16,
        qdata=lambda target: target.qdata,
        qparams=lambda target: (
            target.scale.view(torch.uint8),
            target.per_tensor_scale,
        ),
    )


@CUDA
def test_triton_integer_stochastic_rounding_is_unbiased() -> None:
    from piper_offload._triton_int8_lora import merge_int8_lora

    rows, cols = 16, 64
    qdata = torch.zeros(rows, cols, device="cuda", dtype=torch.int8)
    scale = torch.ones(rows, 1, device="cuda")
    b = torch.ones(rows, 1, device="cuda", dtype=torch.float16)
    a = torch.full((1, cols), 0.3, device="cuda", dtype=torch.float16)
    a[0, -1] = 1.0

    samples = []
    for sample in range(64):
        codes, scales, _ = merge_int8_lora(
            qdata,
            scale,
            None,
            (1, cols),
            b,
            a,
            1.0,
            asymmetric=False,
            reduce_range=False,
            rounding_seed=derive_seed("triton-statistical", sample),
        )
        # A row extremum of 1 sets a symmetric scale of 1 / 127.5.
        torch.testing.assert_close(scales, torch.full_like(scales, 2 / 255))
        samples.append(codes[:, :-1].float() * scales.reshape(rows, 1))

    observed = torch.stack(samples).mean()
    torch.testing.assert_close(observed, a[0, 0].float(), rtol=0, atol=5e-4)
