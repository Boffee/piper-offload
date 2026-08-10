"""Correctness-first stochastic LoRA merge coverage across quant families."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

import piper_offload._bnb as bnb_impl
import piper_offload._torchao_nvfp4 as nvfp4_impl
import piper_offload.bnb4bit_adapter as bnb4_adapter_impl
import piper_offload.bnb8bit_adapter as bnb8_adapter_impl
import piper_offload.float8_adapter as float8_adapter_impl
import piper_offload.int8_adapter as int8_adapter_impl
import piper_offload.mx_adapter as mx_adapter_impl
import piper_offload.nvfp4_adapter as nvfp4_adapter_impl
import piper_offload.quanto_adapter as quanto_adapter_impl
import piper_offload.static_float8_adapter as static_float8_adapter_impl

from piper_offload import (
    LoRATransform,
    ScaledLoRAFactor,
)
from piper_offload._bnb import requantize_params_4bit
from piper_offload.bnb4bit_adapter import Bnb4bitAdapter
from piper_offload.bnb8bit_adapter import Bnb8bitAdapter
from piper_offload.float8_adapter import Float8Adapter
from piper_offload.int8_adapter import Int8Adapter
from piper_offload.mx_adapter import MxAdapter
from piper_offload.nvfp4_adapter import Nvfp4Adapter
from piper_offload.quanto_adapter import QuantoAdapter
from piper_offload.static_float8_adapter import StaticFloat8Adapter
from piper_offload import derive_seed
from tests.test_bnb4bit_adapter import _make_nf4
from tests.test_bnb8bit_adapter import _make_int8 as _make_bnb8
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


@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("double_quant", [False, True])
def test_bnb4_replays_codes_and_keeps_final_qparams(
    quant_type: str,
    double_quant: bool,
) -> None:
    generator = torch.Generator().manual_seed(10)
    weight = torch.randn(17, 33, dtype=torch.bfloat16, generator=generator)

    def make() -> torch.Tensor:
        return _make_nf4(
            rows=17,
            cols=33,
            weight=weight.clone(),
            quant_type=quant_type,
            double_quant=double_quant,
            blocksize=64,
        )

    padding_nibble = make().data.view(torch.uint8)[-1] & 0x0F
    first = _run(Bnb4bitAdapter, make, key="bnb4.weight")
    replay = _run(Bnb4bitAdapter, make, key="bnb4.weight")
    other = _run(Bnb4bitAdapter, make, key="bnb4.other_weight")

    assert torch.equal(first.data.view(torch.uint8), replay.data.view(torch.uint8))
    assert torch.equal(first.quant_state.absmax, replay.quant_state.absmax)
    assert torch.equal(first.quant_state.absmax, other.quant_state.absmax)
    assert not torch.equal(first.data.view(torch.uint8), other.data.view(torch.uint8))
    assert torch.equal(first.data.view(torch.uint8)[-1] & 0x0F, padding_nibble)
    assert torch.equal(replay.data.view(torch.uint8)[-1] & 0x0F, padding_nibble)
    assert torch.equal(other.data.view(torch.uint8)[-1] & 0x0F, padding_nibble)
    if double_quant:
        assert torch.equal(
            first.quant_state.state2.absmax,
            other.quant_state.state2.absmax,
        )
        assert torch.equal(first.quant_state.offset, other.quant_state.offset)


def test_bnb4_nested_probabilities_use_decoded_final_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(11)
    values = torch.randn(17, 33, dtype=torch.bfloat16, generator=generator)
    like = _make_nf4(
        rows=17,
        cols=33,
        double_quant=True,
        blocksize=64,
    )
    captured: dict[str, torch.Tensor] = {}
    original = bnb_impl._stochastic_codebook_indices

    def capture_normalized(
        normalized: torch.Tensor,
        codebook: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        captured["normalized"] = normalized.clone()
        return original(normalized, codebook, **kwargs)

    monkeypatch.setattr(
        bnb_impl,
        "_stochastic_codebook_indices",
        capture_normalized,
    )
    rounded = requantize_params_4bit(
        values,
        like=like,
        rounding_seed=_seed(key="bnb-final-scale.weight"),
    )

    state = rounded.quant_state
    effective_scale = bnb_impl.bnb_functional.dequantize_blockwise(
        state.absmax,
        state.state2,
    ) + state.offset
    element_scale = (
        effective_scale.reshape(-1)
        .repeat_interleave(state.blocksize)[: values.numel()]
        .reshape_as(values)
    )
    expected = values.to(torch.float32) / element_scale.to(torch.float32)
    torch.testing.assert_close(
        captured["normalized"],
        expected,
        rtol=0,
        atol=0,
    )


def test_bnb8_replays_codes_and_keeps_row_scales() -> None:
    generator = torch.Generator().manual_seed(20)
    weight = torch.randn(32, 48, dtype=torch.float16, generator=generator)

    def make() -> torch.Tensor:
        return _make_bnb8(rows=32, cols=48, weight=weight.clone())

    first = _run(Bnb8bitAdapter, make, key="bnb8.weight")
    replay = _run(Bnb8bitAdapter, make, key="bnb8.weight")
    other = _run(Bnb8bitAdapter, make, key="bnb8.other_weight")
    assert torch.equal(first.CB, replay.CB)
    assert torch.equal(first.SCB, replay.SCB)
    assert torch.equal(first.SCB, other.SCB)
    assert not torch.equal(first.CB, other.CB)


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
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        assert rounding_seed is not None
        validated.append(rounding_seed)
        original_validate(
            target,
            staged_b,
            staged_a,
            strength,
            rounding_seed=rounding_seed,
        )

    def tracked_merge(
        target: torch.Tensor,
        staged_b: torch.Tensor,
        staged_a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        assert rounding_seed is not None
        merged.append(rounding_seed)
        original_merge(
            target,
            staged_b,
            staged_a,
            strength,
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
    assert validated
    assert merged == [first_seed, second_seed]
    assert set(validated) == {first_seed, second_seed}
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
    original = nvfp4_impl._stochastic_codebook_indices

    def capture_normalized(
        normalized: torch.Tensor,
        codebook: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        captured["normalized"] = normalized.clone()
        return original(normalized, codebook, **kwargs)

    monkeypatch.setattr(
        nvfp4_impl,
        "_stochastic_codebook_indices",
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


@pytest.mark.parametrize(
    "qtype_name",
    ["qint8", "qfloat8", "qfloat8_e4m3fn", "qfloat8_e5m2"],
)
def test_quanto_replays_qbytes_and_keeps_absmax_scales(qtype_name: str) -> None:
    quanto = pytest.importorskip("optimum.quanto")
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

    qtype = getattr(quanto, qtype_name)
    generator = torch.Generator().manual_seed(80)
    weight = torch.randn(16, 32, dtype=torch.float32, generator=generator)

    def make() -> torch.Tensor:
        template = WeightQBytesTensor.create(
            qtype,
            0,
            (16, 32),
            (32, 1),
            torch.zeros(16, 32, dtype=qtype.dtype),
            torch.ones(16, 1, dtype=torch.float32),
            None,
        )
        return QuantoAdapter.requantize(weight.clone(), like=template)

    first = _run(QuantoAdapter, make, key="quanto.weight")
    replay = _run(QuantoAdapter, make, key="quanto.weight")
    other = _run(QuantoAdapter, make, key="quanto.other_weight")
    assert torch.equal(first._data.view(torch.uint8), replay._data.view(torch.uint8))
    assert torch.equal(first._scale, other._scale)
    assert not torch.equal(first._data.view(torch.uint8), other._data.view(torch.uint8))


def test_quanto_zero_rows_remain_finite_and_exact_while_other_row_merges() -> None:
    quanto = pytest.importorskip("optimum.quanto")
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

    rows, cols = 4, 32
    template = WeightQBytesTensor.create(
        quanto.qint8,
        0,
        (rows, cols),
        (cols, 1),
        torch.zeros(rows, cols, dtype=torch.int8),
        torch.ones(rows, 1, dtype=torch.float32),
        None,
    )
    target = QuantoAdapter.requantize(
        torch.zeros(rows, cols),
        like=template,
    )
    b = torch.zeros(rows, 1)
    b[0] = 1
    a = torch.linspace(-0.25, 0.25, cols).reshape(1, cols)

    QuantoAdapter.merge_lora_(
        target,
        b,
        a,
        1.0,
        rounding_seed=_seed(key="quanto-zero-block.weight"),
    )

    assert bool((target._data[1:] == 0).all())
    assert bool((QuantoAdapter.dequantize(target)[1:] == 0).all())
    assert torch.isfinite(target._scale).all()
    assert torch.isfinite(QuantoAdapter.dequantize(target)).all()


@CUDA
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_triton_bnb4_stochastic_merge_replays_and_keeps_scales(
    quant_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, cols = 16, 64
    weight = torch.randn(rows, cols, dtype=torch.bfloat16)

    def make() -> torch.Tensor:
        return _make_nf4(
            rows=rows,
            cols=cols,
            weight=weight.clone(),
            quant_type=quant_type,
            blocksize=64,
            device="cuda",
        )

    def fail_fallback(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("supported stochastic BNB4 merge used the fallback")

    monkeypatch.setattr(bnb4_adapter_impl, "_torch_merge_bnb4_lora", fail_fallback)
    _assert_fused_replay(
        Bnb4bitAdapter,
        make,
        rows=rows,
        cols=cols,
        dtype=torch.bfloat16,
        qdata=lambda target: target.data.view(torch.uint8),
        qparams=lambda target: (target.quant_state.absmax,),
    )


@CUDA
def test_nested_bnb4_stochastic_merge_retains_reference_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _make_nf4(
        rows=16,
        cols=64,
        double_quant=True,
        blocksize=64,
        device="cuda",
    )
    b, a = _factors((16, 64), torch.bfloat16, device=torch.device("cuda"))

    def fail_triton(*_args: object, **_kwargs: object) -> tuple[torch.Tensor, ...]:
        raise AssertionError("nested stochastic BNB4 merge reached Triton")

    monkeypatch.setattr(bnb4_adapter_impl, "_triton_merge_bnb4_lora", fail_triton)
    Bnb4bitAdapter.merge_lora_(target, b, a, 0.75, rounding_seed=_seed(key="nested.weight"))


@CUDA
def test_triton_bnb4_rejects_nested_stochastic_rounding_directly() -> None:
    pytest.importorskip("triton")
    from piper_offload._triton_bnb4_lora import merge_bnb4_lora

    target = _make_nf4(
        rows=16,
        cols=64,
        double_quant=True,
        blocksize=64,
        device="cuda",
    )
    state = target.quant_state
    assert state.state2 is not None
    assert state.offset is not None
    b, a = _factors((16, 64), torch.bfloat16, device=torch.device("cuda"))

    with pytest.raises(ValueError, match="does not support nested scales"):
        merge_bnb4_lora(
            target.data.view(torch.uint8).reshape(-1),
            state.absmax,
            state.code,
            state.state2.absmax,
            state.state2.code,
            state.offset,
            tuple(state.shape),
            state.blocksize,
            state.quant_type,
            b,
            a,
            0.75,
            rounding_seed=_seed(key="nested-direct.weight"),
        )


@CUDA
def test_triton_bnb8_stochastic_merge_replays_and_keeps_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, cols = 32, 64
    weight = torch.randn(rows, cols, dtype=torch.float16)

    def make() -> torch.Tensor:
        return _make_bnb8(
            rows=rows,
            cols=cols,
            weight=weight.clone(),
            device="cuda",
        )

    def fail_fallback(*_args: object, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("supported stochastic BNB8 merge used the fallback")

    monkeypatch.setattr(bnb8_adapter_impl, "_torch_merge_bnb8_lora", fail_fallback)
    _assert_fused_replay(
        Bnb8bitAdapter,
        make,
        rows=rows,
        cols=cols,
        dtype=torch.float16,
        qdata=lambda target: target.CB,
        qparams=lambda target: (target.SCB,),
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
@pytest.mark.parametrize("qtype_name", ["qint8", "qfloat8_e5m2"])
def test_triton_quanto_stochastic_merge_replays_and_keeps_scales(
    qtype_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quanto = pytest.importorskip("optimum.quanto")
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

    rows, cols = 16, 32
    qtype = getattr(quanto, qtype_name)
    weight = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

    def make() -> torch.Tensor:
        template = WeightQBytesTensor.create(
            qtype,
            0,
            (rows, cols),
            (cols, 1),
            torch.zeros(rows, cols, device="cuda", dtype=qtype.dtype),
            torch.ones(rows, 1, device="cuda"),
            None,
        )
        return QuantoAdapter.requantize(weight.clone(), like=template)

    def fail_fallback(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("supported stochastic Quanto merge used the fallback")

    monkeypatch.setattr(quanto_adapter_impl, "_torch_merge_quanto_lora", fail_fallback)
    _assert_fused_replay(
        QuantoAdapter,
        make,
        rows=rows,
        cols=cols,
        dtype=torch.float32,
        qdata=lambda target: target._data.view(torch.uint8),
        qparams=lambda target: (target._scale,),
    )


@CUDA
def test_triton_quanto_uses_fp32_probability_with_bfloat16_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quanto = pytest.importorskip("optimum.quanto")
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

    rows, cols = 16, 8192
    target = WeightQBytesTensor.create(
        quanto.qint8,
        0,
        (rows, cols),
        (cols, 1),
        torch.zeros(rows, cols, device="cuda", dtype=torch.int8),
        torch.ones(rows, 1, device="cuda", dtype=torch.bfloat16),
        None,
    )
    b = torch.ones(rows, 1, device="cuda", dtype=torch.bfloat16)
    a = torch.full((1, cols), 128.0, device="cuda", dtype=torch.bfloat16)
    a[0, -1] = 253.0

    def fail_fallback(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("supported stochastic Quanto merge used the fallback")

    monkeypatch.setattr(quanto_adapter_impl, "_torch_merge_quanto_lora", fail_fallback)
    QuantoAdapter.merge_lora_(
        target,
        b,
        a,
        1.0,
        rounding_seed=_seed(key="quanto-bfloat16-probability.weight"),
    )

    samples = target._data[:, :-1]
    assert bool(((samples == 64) | (samples == 65)).all())
    expected_upper_probability = 128.0 / float(target._scale[0, 0]) - 64.0
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
    from piper_offload._triton_bnb8_lora import merge_bnb8_lora

    rows, cols = 16, 64
    cb = torch.zeros(rows, cols, device="cuda", dtype=torch.int8)
    scb = torch.zeros(rows, device="cuda", dtype=torch.float32)
    b = torch.ones(rows, 1, device="cuda", dtype=torch.float16)
    a = torch.full((1, cols), 0.3, device="cuda", dtype=torch.float16)
    a[0, -1] = 1.0

    samples = []
    for sample in range(64):
        qdata, scale = merge_bnb8_lora(
            cb,
            scb,
            b,
            a,
            1.0,
            rounding_seed=derive_seed("triton-statistical", sample),
        )
        assert torch.equal(scale, torch.ones_like(scale))
        samples.append(qdata[:, :-1].to(torch.float32) / 127.0)

    observed = torch.stack(samples).mean()
    torch.testing.assert_close(observed, a[0, 0].to(torch.float32), rtol=0, atol=5e-4)
