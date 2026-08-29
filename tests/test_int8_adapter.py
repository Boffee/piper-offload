"""Tests for TorchAO Int8 (``Int8Tensor``) adapter integration."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

import piper_offload.int8_adapter as int8_adapter_module
from piper_offload import (
    LoRA,
    LoRATransform,
    ModelOffloader,
    ScaledLoRAFactor,
    merge_lora,
)
from piper_offload.int8_adapter import Int8Adapter
from piper_offload.pinned_param import PinnedParam
from piper_offload.streamed_component import _param_target_layout
from piper_offload.tensor_adapter_registry import select_adapter, tensor_id
from tests.conftest import activated_model

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_model_offloader(
    model: nn.Module,
    *,
    block_paths: list[str] = [],
    stream_trainable_weights: bool = False,
) -> ModelOffloader:
    return ModelOffloader.from_module(
        model,
        block_paths=block_paths,
        stream_trainable_weights=stream_trainable_weights,
    )


def _int8_config(*, dynamic_activation: bool) -> object:
    pytest.importorskip("torchao")
    try:
        from torchao.quantization import (
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
        )
    except ImportError as exc:
        # The int8 adapter targets the torchao>=0.18 version-2 Int8Tensor
        # workflow; skip (don't error) when the installed torchao predates
        # it — or a future release moves it.
        pytest.skip(f"torchao int8 API unavailable: {exc}")

    return Int8DynamicActivationInt8WeightConfig(version=2) if dynamic_activation else Int8WeightOnlyConfig(version=2)


def _int8_tensor_cls() -> type:
    pytest.importorskip("torchao")
    try:
        from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
            Int8Tensor,
        )
    except ImportError as exc:
        pytest.skip(f"torchao Int8Tensor unavailable: {exc}")
    return Int8Tensor


def _make_int8(
    *,
    rows: int = 32,
    cols: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    dynamic_activation: bool = False,
    weight: torch.Tensor | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    from torchao.quantization import quantize_

    cfg = _int8_config(dynamic_activation=dynamic_activation)
    layer = nn.Linear(cols, rows, bias=False).to(dtype)
    if weight is not None:
        with torch.no_grad():
            layer.weight.copy_(weight)
    layer = layer.to(device)
    quantize_(layer, cfg)
    return layer.weight.data


def _make_int8_pergroup(
    *,
    rows: int = 32,
    cols: int = 128,
    group_size: int = 64,
) -> torch.Tensor:
    pytest.importorskip("torchao")
    try:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        from torchao.quantization.granularity import PerGroup
    except ImportError as exc:
        pytest.skip(f"torchao per-group int8 API unavailable: {exc}")

    layer = nn.Linear(cols, rows, bias=False).to(torch.bfloat16)
    quantize_(layer, Int8WeightOnlyConfig(granularity=PerGroup(group_size), version=2))
    return layer.weight.data


def _make_affine_int8(
    weight: torch.Tensor,
    *,
    layout: str = "row",
    asymmetric: bool = False,
    reduce_range: bool = False,
) -> torch.Tensor:
    pytest.importorskip("torchao")
    from torchao.quantization.granularity import PerGroup, PerRow, PerTensor
    from torchao.quantization.quant_primitives import MappingType

    int8_cls = _int8_tensor_cls()
    if layout == "tensor":
        granularity = PerTensor()
    elif layout == "row":
        granularity = PerRow()
    elif layout == "column":
        granularity = PerRow(dim=0)
    elif layout == "group":
        granularity = PerGroup(weight.shape[1] // 3)
    else:
        raise AssertionError(f"unknown test INT8 layout: {layout}")
    mapping_type = MappingType.ASYMMETRIC if asymmetric else MappingType.SYMMETRIC
    return int8_cls.from_hp(
        weight,
        granularity,
        mapping_type,
        reduce_range=reduce_range,
    )


def _with_act_pre_scale(
    qt: torch.Tensor,
    act_pre_scale: torch.Tensor,
) -> torch.Tensor:
    """Clone an INT8 test tensor and attach independent pre-scale storage."""
    int8_cls = _int8_tensor_cls()

    def cloned(t: torch.Tensor | None) -> torch.Tensor | None:
        return None if t is None else t.clone()

    return int8_cls(
        qt.qdata.clone(),
        qt.scale.clone(),
        list(qt.block_size),
        qt.dtype,
        zero_point=cloned(qt.zero_point),
        act_quant_scale=cloned(qt.act_quant_scale),
        act_quant_zero_point=cloned(qt.act_quant_zero_point),
        act_pre_scale=act_pre_scale.clone(),
        act_quant_kwargs=qt.act_quant_kwargs,
        reduce_range=qt.reduce_range,
    )


def _expected_int8_merge(
    qt: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    dense = Int8Adapter.dequantize(qt)
    pre_scale = qt.act_pre_scale
    if pre_scale is None:
        stored_a = a
        stored_strength = strength
    else:
        divisor = pre_scale.reshape(1, 1) if pre_scale.numel() == 1 else pre_scale.reshape(1, a.shape[1])
        stored_a = a.to(torch.float64).mul(strength).div(divisor.to(device=a.device, dtype=torch.float64)).to(a.dtype)
        stored_strength = 1.0
    dense.addmm_(b, stored_a, alpha=stored_strength)
    return Int8Adapter.requantize(dense, like=qt)


def _assert_triton_int8_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    torch.testing.assert_close(
        actual.scale,
        expected.scale,
        rtol=0.02,
        atol=torch.finfo(torch.float32).eps,
    )
    if actual.zero_point is not None and expected.zero_point is not None:
        zero_point_error = (actual.zero_point.to(torch.int16) - expected.zero_point.to(torch.int16)).abs()
        # Tiled BF16 GEMM accumulation can move a block extremum enough to
        # shift the independently recomputed affine zero point by a bucket
        # or two, while the represented dense values remain within the
        # quantization error checked below.
        assert zero_point_error.max().item() <= 2
    quantization_step = expected.scale.to(torch.float32).max().item()
    torch.testing.assert_close(
        actual.dequantize(torch.float32),
        expected.dequantize(torch.float32),
        rtol=0.03,
        atol=max(3 * quantization_step, torch.finfo(torch.float32).eps),
    )


class TestInt8Adapter:
    def test_matches_and_dispatches_int8_only(self) -> None:
        qt = _make_int8()
        assert Int8Adapter.matches(qt)
        assert not Int8Adapter.matches(torch.zeros(16, 16, dtype=torch.bfloat16))
        # Registry dispatch resolves Int8Tensor to this adapter (disjoint
        # from the other TorchAO structured adapters in the dispatch order).
        assert isinstance(select_adapter(qt), Int8Adapter)

    def test_pin_preserves_storage_and_metadata(self) -> None:
        int8_cls = _int8_tensor_cls()
        qt = _make_int8()
        pinned_param = PinnedParam(nn.Parameter(qt, requires_grad=False))

        pinned = pinned_param.make_cpu_param().data
        assert isinstance(pinned, int8_cls)
        assert pinned.qdata.is_pinned()
        assert pinned.scale.is_pinned()
        assert pinned.qdata.data_ptr() == pinned_param.pinned_state.storage[0].data_ptr()
        assert pinned.scale.data_ptr() == pinned_param.pinned_state.storage[1].data_ptr()
        if qt.zero_point is not None:
            assert pinned.zero_point is not None
            assert pinned.zero_point.is_pinned()
            assert pinned.zero_point.data_ptr() == pinned_param.pinned_state.storage[2].data_ptr()
        assert pinned.block_size == qt.block_size
        assert pinned.dtype == qt.dtype
        assert pinned.reduce_range == qt.reduce_range
        assert pinned_param.compute_dtype is torch.bfloat16
        assert torch.equal(pinned.dequantize(), qt.dequantize())

    def test_tensor_id_tracks_buffers(self) -> None:
        qt = _make_int8()
        key = tensor_id(qt)
        assert key[0] == "torchao-int8"
        assert key[1][0] == qt.qdata.device
        assert key[2][0] == qt.scale.device
        assert key == tensor_id(qt)
        assert key != tensor_id(_make_int8())

    def test_target_layout_ignores_tensor_id(self) -> None:
        p1 = nn.Parameter(_make_int8(), requires_grad=False)
        p2 = nn.Parameter(_make_int8(), requires_grad=False)

        assert _param_target_layout(p1) == _param_target_layout(p2)

    def test_target_layout_tracks_activation_quantization(self) -> None:
        with_activation = nn.Parameter(_make_int8(dynamic_activation=True), requires_grad=False)
        weight_only = nn.Parameter(_make_int8(dynamic_activation=False), requires_grad=False)

        assert _param_target_layout(with_activation) != _param_target_layout(weight_only)

    def test_reduced_range_is_preserved_and_part_of_layout(self) -> None:
        weight = torch.randn(32, 16, dtype=torch.bfloat16)
        full_range = _make_affine_int8(weight)
        reduced_range = _make_affine_int8(weight, reduce_range=True)

        assert reduced_range.scale.dtype is torch.float32
        assert reduced_range.reduce_range is True
        assert reduced_range.qdata.min().item() >= -64
        assert reduced_range.qdata.max().item() <= 63
        assert _param_target_layout(
            nn.Parameter(full_range, requires_grad=False),
        ) != _param_target_layout(
            nn.Parameter(reduced_range, requires_grad=False),
        )

        again = Int8Adapter.requantize(
            Int8Adapter.dequantize(reduced_range),
            like=reduced_range,
        )
        assert again.reduce_range is True
        assert again.qdata.min().item() >= -64
        assert again.qdata.max().item() <= 63

        pinned = (
            PinnedParam(
                nn.Parameter(reduced_range, requires_grad=False),
            )
            .make_cpu_param()
            .data
        )
        assert pinned.reduce_range is True

    def test_no_cpu_round_trip_or_trainable_swap_capability(self) -> None:
        pinned_param = PinnedParam(
            nn.Parameter(_make_int8(), requires_grad=True),
        )
        state = pinned_param.allocate_gpu_storage(torch.device("cpu"))

        with pytest.raises(NotImplementedError, match="CPU round-trip"):
            pinned_param.copy_to_cpu(state)
        with pytest.raises(NotImplementedError, match="Parameter.data-swap"):
            pinned_param.validate_parameter_data_swap_target()

    @pytest.mark.parametrize("dynamic_activation", [False, True])
    def test_dequantize_requantize_preserves_representation(self, dynamic_activation: bool) -> None:
        int8_cls = _int8_tensor_cls()
        qt = _make_int8(rows=32, cols=16, dynamic_activation=dynamic_activation)
        dense = Int8Adapter.dequantize(qt)
        assert dense.dtype is qt.dtype
        torch.testing.assert_close(dense, qt.dequantize())

        again = Int8Adapter.requantize(dense, like=qt)
        assert isinstance(again, int8_cls)
        assert again.block_size == qt.block_size
        assert again.dtype == qt.dtype
        assert again.qdata.dtype is torch.int8
        assert tuple(again.qdata.shape) == tuple(qt.qdata.shape)
        assert again.act_quant_kwargs == qt.act_quant_kwargs
        # int8's 256-level grid is too coarse for a bit-exact round trip
        # (boundary values flip ±1 bucket), but re-encoding stays within one
        # quantization step of the dense input it was given. Dequantize
        # directly to fp32: the default bf16 dequant rounds qdata*scale and
        # would occasionally push the error a hair past the bound.
        err = (again.dequantize(torch.float32) - dense).abs()
        assert err.max().item() <= again.scale.to(torch.float32).max().item()

    def test_requantize_rejects_shape_mismatch(self) -> None:
        qt = _make_int8(rows=32, cols=16)
        with pytest.raises(ValueError, match="Cannot requantize"):
            Int8Adapter.requantize(torch.randn(16, 32), like=qt)

    def test_requantize_recovers_per_group_granularity(self) -> None:
        # Int8WeightOnlyConfig(granularity=PerGroup(g)) gives block_size
        # [1, g] with g < in_features; requantize must recover PerGroup and
        # reproduce the partition rather than rejecting it as non-PerRow.
        int8_cls = _int8_tensor_cls()
        qt = _make_int8_pergroup(rows=32, cols=128, group_size=64)
        assert list(qt.block_size) == [1, 64]

        again = Int8Adapter.requantize(Int8Adapter.dequantize(qt), like=qt)
        assert isinstance(again, int8_cls)
        assert list(again.block_size) == [1, 64]
        assert tuple(again.scale.shape) == tuple(qt.scale.shape)

    def test_merge_lora_merges_per_group_int8_weight(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(128, 32, bias=False, dtype=torch.bfloat16)

        model = M()
        model.lin.weight.requires_grad = False
        model.lin.weight = nn.Parameter(
            _make_int8_pergroup(rows=32, cols=128, group_size=64),
            requires_grad=False,
        )
        original_qdata = model.lin.weight.data.qdata.clone()
        lora = LoRA.from_state_dict(
            state_dict={
                "lin.lora_A.weight": torch.randn(4, 128),
                "lin.lora_B.weight": torch.randn(32, 4),
            }
        )

        merged = merge_lora(model, [(lora, 1.0)])

        assert merged == 1
        assert list(model.lin.weight.data.block_size) == [1, 64]
        assert not torch.equal(model.lin.weight.data.qdata, original_qdata)

    def test_copy_into_preserves_absent_zero_point(self) -> None:
        # A symmetric int8 weight may carry zero_point=None, but
        # Int8Tensor.from_hp (used by requantize) always re-emits a zeros
        # zero_point. copy_into must fill only the slots the target has, not
        # assert on the recomputed zero_point the target lacks.
        int8_cls = _int8_tensor_cls()
        base = _make_int8(rows=32, cols=16)
        like = int8_cls(
            base.qdata,
            base.scale,
            list(base.block_size),
            base.dtype,
            zero_point=None,
            reduce_range=base.reduce_range,
        )
        assert like.zero_point is None

        dense = Int8Adapter.dequantize(like)
        dense.addmm_(
            torch.randn(32, 4, dtype=dense.dtype),
            torch.randn(4, 16, dtype=dense.dtype),
            alpha=0.5,
        )
        new = Int8Adapter.requantize(dense, like=like)
        assert new.zero_point is not None  # from_hp always emits one

        Int8Adapter.copy_into(new, target=like)  # must not raise
        assert like.zero_point is None  # target representation preserved
        assert torch.equal(like.qdata, new.qdata)

    def test_lora_transform_requantizes_param_in_place(self) -> None:
        int8_cls = _int8_tensor_cls()
        rows, cols, rank = 32, 16, 2
        qt = _make_int8(rows=rows, cols=cols, dynamic_activation=False)
        param = nn.Parameter(qt, requires_grad=False)
        a = torch.randn(rank, cols)
        b = torch.randn(rows, rank)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.5)])
        original_param = param
        original_qdata_ptr = param.data.qdata.data_ptr()

        # The merge path dequantizes, applies the delta, then requantizes;
        # mirror it exactly so the comparison is deterministic (not a lossy
        # round-trip property).
        expected_dense = Int8Adapter.dequantize(qt)
        expected_dense.addmm_(
            b.to(expected_dense.dtype),
            a.to(expected_dense.dtype),
            alpha=0.5,
        )
        expected = Int8Adapter.requantize(expected_dense, like=qt)

        transform.validate_target(param)
        transform.apply(param)

        assert param is original_param
        assert param.data.qdata.data_ptr() == original_qdata_ptr
        assert isinstance(param.data, int8_cls)
        assert torch.equal(param.data.qdata, expected.qdata)
        assert torch.equal(param.data.scale, expected.scale)

    def test_merge_lora_merges_int8_weight(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)

        model = M()
        model.lin.weight.requires_grad = False
        model.lin.weight = nn.Parameter(
            _make_int8(rows=16, cols=16, dynamic_activation=False),
            requires_grad=False,
        )
        # copy_into mutates the weight's storage in place, so snapshot the
        # original int8 bytes rather than holding a tensor ref.
        original_qdata = model.lin.weight.data.qdata.clone()
        lora = LoRA.from_state_dict(
            state_dict={
                "lin.lora_A.weight": torch.randn(4, 16),
                "lin.lora_B.weight": torch.randn(16, 4),
            }
        )

        merged = merge_lora(model, [(lora, 1.0)])

        assert merged == 1
        assert not torch.equal(model.lin.weight.data.qdata, original_qdata)

    @pytest.mark.parametrize(
        "shape",
        [(), (1,), (16,), (1, 16), (1, 1, 16)],
    )
    def test_lora_merge_accepts_scalar_and_per_input_act_pre_scale(
        self,
        shape: tuple[int, ...],
    ) -> None:
        base = _make_affine_int8(torch.randn(8, 16))
        pre_scale = torch.full(shape, 0.75)
        qt = _with_act_pre_scale(base, pre_scale)

        Int8Adapter.validate_lora_merge(
            qt,
            torch.ones(8, 1),
            torch.ones(1, 16),
            1.0,
        )

    @pytest.mark.parametrize(
        ("kind", "match"),
        [
            ("integer", "floating-point"),
            ("wrong-features", "scalar or one value per input feature"),
            ("batch-dependent", "scalar or one value per input feature"),
            ("zero", "non-zero"),
            ("nan", "finite"),
            ("infinite", "finite"),
        ],
    )
    def test_lora_merge_rejects_invalid_act_pre_scale_before_mutation(
        self,
        kind: str,
        match: str,
    ) -> None:
        rows, cols, rank = 8, 16, 3
        base = _make_affine_int8(torch.randn(rows, cols))
        if kind == "integer":
            pre_scale = torch.ones(cols, dtype=torch.int32)
        elif kind == "wrong-features":
            pre_scale = torch.ones(cols - 1)
        elif kind == "batch-dependent":
            pre_scale = torch.ones(2, cols)
        else:
            pre_scale = torch.ones(cols)
            pre_scale[3] = {
                "zero": 0.0,
                "nan": float("nan"),
                "infinite": float("inf"),
            }[kind]
        qt = _with_act_pre_scale(base, pre_scale)
        qdata_before = qt.qdata.clone()

        with pytest.raises(ValueError, match=match):
            Int8Adapter.validate_lora_merge(
                qt,
                torch.randn(rows, rank),
                torch.randn(rank, cols),
                0.5,
            )
        with pytest.raises(ValueError, match=match):
            Int8Adapter.merge_lora_(
                qt,
                torch.randn(rows, rank),
                torch.randn(rank, cols),
                0.5,
            )

        assert torch.equal(qt.qdata, qdata_before)

    @pytest.mark.parametrize(
        ("pre_scale", "a_value", "strength"),
        [
            (1e-40, 1.0, 0.0),
            (1e-40, 1.0, 1e-40),
            (1e30, 1e10, 1e30),
        ],
    )
    def test_act_pre_scale_validation_uses_strength_before_division(
        self,
        pre_scale: float,
        a_value: float,
        strength: float,
    ) -> None:
        rows, cols, rank = 8, 16, 3
        qt = _with_act_pre_scale(
            _make_affine_int8(torch.randn(rows, cols)),
            torch.full((cols,), pre_scale),
        )
        b = torch.randn(rows, rank)
        a = torch.full((rank, cols), a_value)
        expected = _expected_int8_merge(qt, b, a, strength)

        Int8Adapter.validate_lora_merge(qt, b, a, strength)
        Int8Adapter.merge_lora_(qt, b, a, strength)

        assert torch.equal(qt.qdata, expected.qdata)
        assert torch.equal(qt.scale, expected.scale)
        assert torch.isfinite(qt.dequantize()).all()

    def test_merge_lora_preflights_factor_dependent_overflow_before_mutation(
        self,
    ) -> None:
        rows, cols, rank = 8, 16, 3

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.first = nn.Linear(cols, rows, bias=False)
                self.second = nn.Linear(cols, rows, bias=False)

        model = M()
        model.first.weight = nn.Parameter(
            _make_affine_int8(torch.randn(rows, cols)),
            requires_grad=False,
        )
        model.second.weight = nn.Parameter(
            _with_act_pre_scale(
                _make_affine_int8(torch.randn(rows, cols)),
                torch.full((cols,), 1e-40),
            ),
            requires_grad=False,
        )
        before = {
            name: (
                param.data.qdata.clone(),
                param.data.scale.clone(),
            )
            for name, param in model.named_parameters()
        }
        lora = LoRA.from_state_dict(
            {
                "first.lora_A.weight": torch.ones(rank, cols),
                "first.lora_B.weight": torch.ones(rows, rank),
                "second.lora_A.weight": torch.ones(rank, cols),
                "second.lora_B.weight": torch.ones(rows, rank),
            }
        )

        with pytest.raises(
            ValueError,
            match="stored-coordinate LoRA factors must be finite",
        ):
            merge_lora(model, [(lora, 1.0)])

        for name, param in model.named_parameters():
            qdata, scale = before[name]
            assert torch.equal(param.data.qdata, qdata)
            assert torch.equal(param.data.scale, scale)

    def test_generic_merge_with_nonuniform_act_pre_scale_matches_routed_forward(
        self,
    ) -> None:
        torch.manual_seed(123)
        rows, cols, rank = 7, 8, 3
        pre_scale = torch.tensor(
            [0.5, 0.75, 1.0, 1.5, 2.0, 0.625, 1.25, 1.75],
        )
        qt = _with_act_pre_scale(
            _make_affine_int8(torch.randn(rows, cols) * 0.3),
            pre_scale,
        )
        x = torch.randn(13, cols)
        a = torch.randn(rank, cols) * 0.12
        b = torch.randn(rows, rank) * 0.12
        strength = 0.7
        expected_storage = _expected_int8_merge(qt, b, a, strength)
        routed_output = torch.nn.functional.linear(x, qt) + ((x @ a.T) * strength) @ b.T
        pre_scale_ptr = qt.act_pre_scale.data_ptr()
        pre_scale_before = qt.act_pre_scale.clone()

        Int8Adapter.merge_lora_(qt, b, a, strength)
        merged_output = torch.nn.functional.linear(x, qt)

        assert torch.equal(qt.qdata, expected_storage.qdata)
        assert torch.equal(qt.scale, expected_storage.scale)
        assert qt.act_pre_scale.data_ptr() == pre_scale_ptr
        assert torch.equal(qt.act_pre_scale, pre_scale_before)
        torch.testing.assert_close(
            merged_output,
            routed_output,
            rtol=0.01,
            atol=0.025,
        )

    @pytest.mark.parametrize(
        ("device", "expects_triton"),
        [
            ("cpu", False),
            pytest.param("cuda", True, marks=CUDA),
        ],
    )
    @pytest.mark.parametrize(
        ("a_value", "b_value", "strength", "pre_scale", "x_value"),
        [
            (1e10, 1e-10, 1e30, 1e30, 1e-30),
            (1e-20, 1e20, 1e-30, 1e-30, 1e30),
        ],
        ids=["multiply-would-overflow", "multiply-would-underflow"],
    )
    def test_packed_extreme_strengths_stage_before_low_precision_cast(
        self,
        device: str,
        expects_triton: bool,
        a_value: float,
        b_value: float,
        strength: float,
        pre_scale: float,
        x_value: float,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Packed factors keep strength boundaries through p-coordinate mapping."""
        torch.manual_seed(125)
        rows, cols = 4, 8
        base = _make_affine_int8(
            torch.randn(rows, cols, device=device) * 0.05,
        )
        qt = _with_act_pre_scale(
            base,
            torch.tensor(pre_scale, device=device),
        )
        param = nn.Parameter(qt, requires_grad=False)
        a = torch.full((1, cols), a_value)
        b = torch.full((rows, 1), b_value)
        factors = [ScaledLoRAFactor.from_tensors(a.clone(), b.clone(), strength) for _ in range(2)]
        x = torch.full((3, cols), x_value, device=device)

        # Independent logical-coordinate reference. The old packed path cast
        # A first and then multiplied strength, producing inf or zero before
        # the adapter had a chance to cancel strength against pre_scale.
        routed_output = torch.nn.functional.linear(x, qt)
        for _ in factors:
            routed_output = routed_output + ((x @ a.to(device).T) * strength) @ b.to(device).T

        prepared_a: list[torch.Tensor] = []
        original_prepared_merge = Int8Adapter.merge_prepared_lora_

        def record_prepared_merge(
            target: torch.Tensor,
            packed_b: torch.Tensor,
            packed_a: torch.Tensor,
            packed_strength: float,
            *,
            rounding_seed: int | None = None,
        ) -> None:
            prepared_a.append(packed_a.detach().clone())
            original_prepared_merge(
                target,
                packed_b,
                packed_a,
                packed_strength,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(
            Int8Adapter,
            "merge_prepared_lora_",
            staticmethod(record_prepared_merge),
        )

        triton_calls = 0
        original_triton_merge = int8_adapter_module._triton_merge_int8_lora

        if original_triton_merge is not None:

            def record_triton_merge(*args: object, **kwargs: object):
                nonlocal triton_calls
                triton_calls += 1
                return original_triton_merge(*args, **kwargs)

            monkeypatch.setattr(
                int8_adapter_module,
                "_triton_merge_int8_lora",
                record_triton_merge,
            )

        pre_scale_ptr = qt.act_pre_scale.data_ptr()
        transform = LoRATransform(factors)
        transform.validate_target(param)
        transform.apply(param)
        if device == "cuda":
            torch.cuda.synchronize()

        assert len(prepared_a) == 1
        assert torch.isfinite(prepared_a[0]).all()
        torch.testing.assert_close(
            prepared_a[0],
            torch.full_like(prepared_a[0], a_value),
        )
        assert triton_calls == int(expects_triton)
        assert param.data.act_pre_scale.data_ptr() == pre_scale_ptr
        assert torch.isfinite(param.data.dequantize(torch.float32)).all()
        merged_output = torch.nn.functional.linear(x, param.data)
        torch.testing.assert_close(
            merged_output,
            routed_output,
            rtol=0.01,
            atol=0.05,
        )

    @CUDA
    @pytest.mark.parametrize(
        "dtype",
        [torch.float16, torch.bfloat16, torch.float32],
    )
    def test_triton_merge_matches_eager_for_compute_dtypes(
        self,
        dtype: torch.dtype,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows, cols, rank = 35, 69, 7
        qt = _make_affine_int8(
            torch.randn(rows, cols, device="cuda", dtype=dtype),
        )
        a = torch.randn(cols, rank, device="cuda", dtype=dtype).t()
        b = torch.randn(rank, rows, device="cuda", dtype=dtype).t()
        strength = -0.375
        expected = _expected_int8_merge(qt, b, a, strength)
        qdata_ptr = qt.qdata.data_ptr()
        scale_ptr = qt.scale.data_ptr()
        zero_point_ptr = qt.zero_point.data_ptr()
        triton_merge = int8_adapter_module._triton_merge_int8_lora
        assert triton_merge is not None
        tracked_triton = Mock(wraps=triton_merge)
        monkeypatch.setattr(
            int8_adapter_module,
            "_triton_merge_int8_lora",
            tracked_triton,
        )

        assert not a.is_contiguous()
        assert not b.is_contiguous()
        Int8Adapter.merge_lora_(qt, b, a, strength)
        torch.cuda.synchronize()

        assert qt.qdata.data_ptr() == qdata_ptr
        assert qt.scale.data_ptr() == scale_ptr
        assert qt.zero_point.data_ptr() == zero_point_ptr
        assert tracked_triton.call_count == 1
        _assert_triton_int8_close(qt, expected)

    @CUDA
    @pytest.mark.parametrize(
        ("layout", "asymmetric"),
        [
            ("tensor", False),
            ("tensor", True),
            ("row", False),
            ("row", True),
            ("group", False),
            ("group", True),
        ],
    )
    def test_triton_merge_matches_affine_layouts_and_zero_points(
        self,
        layout: str,
        asymmetric: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows, cols, rank = 35, 69, 19
        dtype = torch.bfloat16
        weight = torch.randn(rows, cols, device="cuda", dtype=dtype) * 0.75 + 0.25
        qt = _make_affine_int8(
            weight,
            layout=layout,
            asymmetric=asymmetric,
        )
        a = torch.randn(rank, cols, device="cuda", dtype=dtype)
        b = torch.randn(rows, rank, device="cuda", dtype=dtype)
        strength = 0.1875
        expected = _expected_int8_merge(qt, b, a, strength)
        triton_merge = int8_adapter_module._triton_merge_int8_lora
        assert triton_merge is not None
        tracked_triton = Mock(wraps=triton_merge)
        monkeypatch.setattr(
            int8_adapter_module,
            "_triton_merge_int8_lora",
            tracked_triton,
        )

        Int8Adapter.merge_lora_(qt, b, a, strength)
        torch.cuda.synchronize()

        assert tracked_triton.call_count == 1
        _assert_triton_int8_close(qt, expected)

    @CUDA
    def test_triton_merge_preserves_absent_symmetric_zero_point(self) -> None:
        rows, cols, rank = 23, 39, 5
        base = _make_affine_int8(
            torch.randn(
                rows,
                cols,
                device="cuda",
                dtype=torch.bfloat16,
            ),
        )
        int8_cls = _int8_tensor_cls()
        qt = int8_cls(
            base.qdata,
            base.scale,
            list(base.block_size),
            base.dtype,
            zero_point=None,
            reduce_range=base.reduce_range,
        )
        a = torch.randn(rank, cols, device="cuda", dtype=qt.dtype)
        b = torch.randn(rows, rank, device="cuda", dtype=qt.dtype)
        expected = _expected_int8_merge(qt, b, a, 0.25)
        qdata_ptr = qt.qdata.data_ptr()
        scale_ptr = qt.scale.data_ptr()

        Int8Adapter.merge_lora_(qt, b, a, 0.25)
        torch.cuda.synchronize()

        assert qt.zero_point is None
        assert qt.qdata.data_ptr() == qdata_ptr
        assert qt.scale.data_ptr() == scale_ptr
        _assert_triton_int8_close(qt, expected)

    @CUDA
    def test_triton_transform_packs_multiple_loras_and_preserves_storage(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows, cols = 33, 69
        dtype = torch.bfloat16
        base = _make_affine_int8(
            torch.randn(rows, cols, device="cuda", dtype=dtype),
            layout="group",
            asymmetric=True,
        )
        int8_cls = _int8_tensor_cls()
        act_quant_scale = torch.tensor(0.03125, device="cuda")
        act_quant_zero_point = torch.tensor(-3, device="cuda", dtype=torch.int8)
        act_pre_scale = torch.tensor(0.75, device="cuda")
        qt = int8_cls(
            base.qdata,
            base.scale,
            list(base.block_size),
            base.dtype,
            zero_point=base.zero_point,
            act_quant_scale=act_quant_scale,
            act_quant_zero_point=act_quant_zero_point,
            act_pre_scale=act_pre_scale,
            act_quant_kwargs=base.act_quant_kwargs,
            reduce_range=base.reduce_range,
        )
        assert qt.scale.dtype is torch.float32
        assert qt.dtype is torch.bfloat16
        param = nn.Parameter(qt, requires_grad=False)
        factor_inputs = [
            (
                torch.randn(5, cols),
                torch.randn(rows, 5),
                0.5,
            ),
            (
                torch.randn(3, cols),
                torch.randn(rows, 3),
                -0.25,
            ),
        ]
        factors = [ScaledLoRAFactor.from_tensors(a, b, strength) for a, b, strength in factor_inputs]
        packed_a = torch.cat(
            [
                a.to(device="cuda", dtype=torch.float64).mul(strength).div(act_pre_scale.to(torch.float64)).to(dtype)
                for a, _b, strength in factor_inputs
            ],
            dim=0,
        )
        packed_b = torch.cat(
            [b.to(device="cuda", dtype=dtype) for _a, b, _strength in factor_inputs],
            dim=1,
        )
        expected_dense = Int8Adapter.dequantize(qt)
        expected_dense.addmm_(packed_b, packed_a)
        expected = Int8Adapter.requantize(expected_dense, like=qt)
        storage = (
            qt.qdata,
            qt.scale,
            qt.zero_point,
            qt.act_quant_scale,
            qt.act_quant_zero_point,
            qt.act_pre_scale,
        )
        storage_ptrs = tuple(t.data_ptr() for t in storage)
        activation_values = tuple(t.clone() for t in storage[3:])
        calls: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []
        triton_merge = int8_adapter_module._triton_merge_int8_lora
        assert triton_merge is not None

        def tracked_triton_merge(
            qdata: torch.Tensor,
            scale: torch.Tensor,
            zero_point: torch.Tensor | None,
            block_size: tuple[int, int],
            b: torch.Tensor,
            a: torch.Tensor,
            strength: float,
            *,
            asymmetric: bool,
            reduce_range: bool,
            rounding_seed: int | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            calls.append((tuple(b.shape), tuple(a.shape), strength))
            return triton_merge(
                qdata,
                scale,
                zero_point,
                block_size,
                b,
                a,
                strength,
                asymmetric=asymmetric,
                reduce_range=reduce_range,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(
            int8_adapter_module,
            "_triton_merge_int8_lora",
            tracked_triton_merge,
        )
        transform = LoRATransform(factors)
        transform.validate_target(param)
        transform.apply(param)
        torch.cuda.synchronize()

        assert calls == [((rows, 8), (8, cols), 1.0)]
        merged = param.data
        merged_storage = (
            merged.qdata,
            merged.scale,
            merged.zero_point,
            merged.act_quant_scale,
            merged.act_quant_zero_point,
            merged.act_pre_scale,
        )
        assert tuple(t.data_ptr() for t in merged_storage) == storage_ptrs
        for actual, original in zip(
            merged_storage[3:],
            activation_values,
            strict=True,
        ):
            assert torch.equal(actual, original)
        _assert_triton_int8_close(merged, expected)

    @CUDA
    def test_nonuniform_act_pre_scale_generic_and_triton_merges_match(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(124)
        rows, cols, rank = 19, 64, 5
        dtype = torch.bfloat16
        base = _make_affine_int8(
            torch.randn(rows, cols, device="cuda", dtype=dtype) * 0.3,
        )
        pre_scale = torch.linspace(
            0.5,
            2.0,
            cols,
            device="cuda",
            dtype=dtype,
        ).reshape(1, cols)
        generic = _with_act_pre_scale(base, pre_scale)
        fused = _with_act_pre_scale(base, pre_scale)
        a = torch.randn(rank, cols, device="cuda", dtype=dtype) * 0.1
        b = torch.randn(rows, rank, device="cuda", dtype=dtype) * 0.1
        strength = -0.375
        triton_merge = int8_adapter_module._triton_merge_int8_lora
        assert triton_merge is not None

        monkeypatch.setattr(int8_adapter_module, "_triton_merge_int8_lora", None)
        Int8Adapter.merge_lora_(generic, b, a, strength)

        calls = 0

        def tracked_triton_merge(*args: object, **kwargs: object):
            nonlocal calls
            calls += 1
            return triton_merge(*args, **kwargs)

        monkeypatch.setattr(
            int8_adapter_module,
            "_triton_merge_int8_lora",
            tracked_triton_merge,
        )
        Int8Adapter.merge_lora_(fused, b, a, strength)
        torch.cuda.synchronize()

        assert calls == 1
        assert torch.equal(generic.act_pre_scale, pre_scale)
        assert torch.equal(fused.act_pre_scale, pre_scale)
        _assert_triton_int8_close(fused, generic)

    @CUDA
    @pytest.mark.parametrize("asymmetric", [False, True])
    def test_triton_merge_preserves_reduced_range(
        self,
        asymmetric: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows, cols, rank = 35, 69, 7
        dtype = torch.bfloat16
        qt = _make_affine_int8(
            torch.randn(rows, cols, device="cuda", dtype=dtype),
            layout="group",
            asymmetric=asymmetric,
            reduce_range=True,
        )
        a = torch.randn(rank, cols, device="cuda", dtype=dtype)
        b = torch.randn(rows, rank, device="cuda", dtype=dtype)
        expected = _expected_int8_merge(qt, b, a, 0.25)
        triton_merge = int8_adapter_module._triton_merge_int8_lora
        assert triton_merge is not None
        tracked_triton = Mock(wraps=triton_merge)
        monkeypatch.setattr(
            int8_adapter_module,
            "_triton_merge_int8_lora",
            tracked_triton,
        )

        assert qt.scale.dtype is torch.float32
        Int8Adapter.merge_lora_(qt, b, a, 0.25)
        torch.cuda.synchronize()

        assert qt.reduce_range is True
        assert qt.qdata.min().item() >= -64
        assert qt.qdata.max().item() <= 63
        assert tracked_triton.call_count == 1
        _assert_triton_int8_close(qt, expected)

    @CUDA
    @pytest.mark.parametrize("fallback", ["unavailable", "unsupported-layout"])
    def test_fused_merge_uses_exact_generic_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fallback: str,
    ) -> None:
        rows, cols, rank = 24, 30, 5
        layout = "column" if fallback == "unsupported-layout" else "row"
        qt = _make_affine_int8(
            torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16),
            layout=layout,
        )
        param = nn.Parameter(qt, requires_grad=False)
        a = torch.randn(rank, cols)
        b = torch.randn(rows, rank)
        a_cuda = a.to(device="cuda", dtype=qt.dtype)
        b_cuda = b.to(device="cuda", dtype=qt.dtype)
        expected = _expected_int8_merge(qt, b_cuda, a_cuda, 0.5)
        qdata_ptr = qt.qdata.data_ptr()

        if fallback == "unavailable":
            monkeypatch.setattr(
                int8_adapter_module,
                "_triton_merge_int8_lora",
                None,
            )
        else:

            def fail_triton_merge(*_args, **_kwargs):
                raise AssertionError("unsupported INT8 layout reached Triton")

            monkeypatch.setattr(
                int8_adapter_module,
                "_triton_merge_int8_lora",
                fail_triton_merge,
            )

        transform = LoRATransform(
            [ScaledLoRAFactor.from_tensors(a, b, 0.5)]
        )
        transform.validate_target(param)
        transform.apply(param)

        assert param.data.qdata.data_ptr() == qdata_ptr
        assert torch.equal(param.data.qdata, expected.qdata)
        assert torch.equal(param.data.scale, expected.scale)
        assert torch.equal(param.data.zero_point, expected.zero_point)

    @CUDA
    @pytest.mark.parametrize("asymmetric", [False, True])
    @pytest.mark.parametrize("values", ["zero", "extreme"])
    def test_triton_merge_handles_zero_and_extreme_values(
        self,
        asymmetric: bool,
        values: str,
    ) -> None:
        rows, cols, rank = 17, 33, 5
        dtype = torch.bfloat16
        if values == "zero":
            weight = torch.zeros(rows, cols, device="cuda", dtype=dtype)
            a = torch.zeros(rank, cols, device="cuda", dtype=dtype)
            b = torch.zeros(rows, rank, device="cuda", dtype=dtype)
        else:
            weight = torch.linspace(
                -2048,
                3072,
                rows * cols,
                device="cuda",
                dtype=dtype,
            ).reshape(rows, cols)
            a = torch.linspace(
                -64,
                96,
                rank * cols,
                device="cuda",
                dtype=dtype,
            ).reshape(rank, cols)
            b = torch.linspace(
                80,
                -48,
                rows * rank,
                device="cuda",
                dtype=dtype,
            ).reshape(rows, rank)
        qt = _make_affine_int8(
            weight,
            layout="group",
            asymmetric=asymmetric,
        )
        expected = _expected_int8_merge(qt, b, a, 0.75)

        Int8Adapter.merge_lora_(qt, b, a, 0.75)
        torch.cuda.synchronize()

        assert torch.isfinite(qt.scale).all()
        assert torch.isfinite(qt.dequantize(torch.float32)).all()
        if values == "zero":
            assert torch.count_nonzero(qt.dequantize()).item() == 0
            assert torch.count_nonzero(qt.scale).item() == qt.scale.numel()
        _assert_triton_int8_close(qt, expected)

    def test_reconstructed_cpu_forward_matches(self) -> None:
        # int8 matmul runs on CPU, so reconstruction correctness is checked
        # without a GPU: the rebuilt wrapper must produce the same output.
        int8_cls = _int8_tensor_cls()
        weight = torch.randn(32, 16, dtype=torch.bfloat16)
        qt = _make_int8(rows=32, cols=16, weight=weight)
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, qt)

        pinned_param = PinnedParam(nn.Parameter(qt, requires_grad=False))
        reconstructed = pinned_param.make_cpu_param().data
        assert isinstance(reconstructed, int8_cls)
        out = torch.nn.functional.linear(x, reconstructed)
        torch.testing.assert_close(out, ref)

    @CUDA
    def test_allocate_copy_make_gpu_param_preserves_wrapper(self) -> None:
        int8_cls = _int8_tensor_cls()
        qt = _make_int8()
        pinned_param = PinnedParam(nn.Parameter(qt, requires_grad=False))

        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()
        pinned = pinned_param.make_cpu_param().data

        assert isinstance(gpu_param.data, int8_cls)
        assert gpu_param.data.qdata.is_cuda
        assert gpu_param.data.scale.is_cuda
        assert gpu_param.data.block_size == pinned.block_size
        assert gpu_param.data.dtype == pinned.dtype
        assert torch.equal(gpu_param.data.qdata.cpu(), pinned.qdata)
        assert torch.equal(gpu_param.data.scale.cpu(), pinned.scale)
        if pinned.zero_point is not None:
            assert gpu_param.data.zero_point is not None
            assert torch.equal(gpu_param.data.zero_point.cpu(), pinned.zero_point)

    @CUDA
    @pytest.mark.parametrize("dynamic_activation", [False, True])
    def test_model_offloader_cuda_forward(self, dynamic_activation: bool) -> None:
        layer = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
        layer.weight.requires_grad = False
        layer.weight = nn.Parameter(
            _make_int8(rows=128, cols=64, dynamic_activation=dynamic_activation),
            requires_grad=False,
        )
        strategy = _make_model_offloader(layer)

        try:
            x = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
            with activated_model(strategy, "cuda") as active:
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (128, 128)
            assert y.dtype is torch.bfloat16
        finally:
            strategy.deactivate()

    @CUDA
    def test_model_offloader_routed_lora_on_int8(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.Linear(128, 128, bias=False, dtype=torch.bfloat16),
                        nn.Linear(128, 128, bias=False, dtype=torch.bfloat16),
                    ]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = block(x)
                return x

        model = M()
        for block in model.blocks:
            block.weight.requires_grad = False
            block.weight = nn.Parameter(
                _make_int8(rows=128, cols=128, dynamic_activation=True),
                requires_grad=False,
            )
        offloader = _make_model_offloader(
            model,
            block_paths=["blocks"],
        )
        lora = LoRA.from_state_dict(
            state_dict={
                "blocks.0.lora_A.weight": torch.randn(4, 128),
                "blocks.0.lora_B.weight": torch.randn(128, 4),
            }
        )
        try:
            x = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
            with activated_model(
                offloader,
                "cuda",
                loras=[lora],
                lora_strengths=[0.25],
                lora_mode="routed",
            ) as active:
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (128, 128)
            assert y.dtype is torch.bfloat16
        finally:
            offloader.deactivate()

    @CUDA
    def test_streamed_int8_merge_requantizes_on_activate(self) -> None:
        int8_cls = _int8_tensor_cls()

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.Linear(32, 32, bias=False, dtype=torch.bfloat16),
                        nn.Linear(32, 32, bias=False, dtype=torch.bfloat16),
                    ]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = block(x)
                return x

        model = M()
        for block in model.blocks:
            block.weight.requires_grad = False
            block.weight = nn.Parameter(
                _make_int8(rows=32, cols=32, dynamic_activation=False),
                requires_grad=False,
            )
        qt = model.blocks[0].weight.data
        rank = 4
        a = torch.randn(rank, 32)
        b = torch.randn(32, rank)
        lora = LoRA.from_state_dict(
            state_dict={
                "blocks.0.lora_A.weight": a,
                "blocks.0.lora_B.weight": b,
            }
        )
        # Reference on CUDA, matching the device the offloader merges on: the
        # offloader merges into a byte-identical GPU copy of the weight.
        qt_cuda = qt.cuda()
        expected_dense = Int8Adapter.dequantize(qt_cuda)
        expected_dense.addmm_(
            b.cuda().to(expected_dense.dtype),
            a.cuda().to(expected_dense.dtype),
            alpha=0.5,
        )
        expected = Int8Adapter.requantize(expected_dense, like=qt_cuda)

        offloader = _make_model_offloader(
            model,
            block_paths=["blocks"],
        )
        try:
            x = torch.randn(8, 32, dtype=torch.bfloat16, device="cuda")
            with activated_model(
                offloader,
                "cuda",
                loras=[lora],
                lora_strengths=[0.5],
                lora_mode="merge",
            ) as active:
                merged = active.blocks[0].weight.data
                assert isinstance(merged, int8_cls)
                _assert_triton_int8_close(merged, expected)
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (8, 32)
        finally:
            offloader.deactivate()
