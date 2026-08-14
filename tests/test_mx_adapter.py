"""Tests for TorchAO MX (MXFP8 / MXFP4) adapter integration."""

import pytest
import torch
from torch import nn

import piper_offload.mx_adapter as mx_adapter_impl
from piper_offload import (
    LoRA,
    LoRATransform,
    ModelOffloader,
    ScaledLoRAFactor,
    StreamConfig,
    merge_lora,
)
from piper_offload._torchao_mx import is_supported_mx_elem_dtype
from piper_offload.mx_adapter import MxAdapter
from piper_offload.pinned_param import PinnedParam
from piper_offload.streamed_component import _param_target_layout
from piper_offload.tensor_adapter_registry import select_adapter, tensor_id
from tests.conftest import activated_model

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# MXFP4 packs two elements per byte under ``torch.float4_e2m1fn_x2``, which
# only exists on new-enough torch builds. Probe it so the suite degrades to
# MXFP8-only rather than failing to import.
_FP4 = getattr(torch, "float4_e2m1fn_x2", None)

ELEM_DTYPES = [
    pytest.param(torch.float8_e4m3fn, id="mxfp8-e4m3"),
    pytest.param(torch.float8_e5m2, id="mxfp8-e5m2"),
    pytest.param(
        _FP4,
        id="mxfp4",
        marks=pytest.mark.skipif(_FP4 is None, reason="torch lacks float4_e2m1fn_x2"),
    ),
]


def _make_model_offloader(
    model: nn.Module,
    *,
    blocks_attr: list[str] = [],
    stream_trainable_weights: bool = False,
) -> ModelOffloader:
    return ModelOffloader.from_module(
        model,
        blocks_attr=blocks_attr,
        stream_trainable_weights=stream_trainable_weights,
    )


def _mx_tensor_cls():
    pytest.importorskip("numpy")
    mod = pytest.importorskip("torchao.prototype.mx_formats.mx_tensor")
    return mod.MXTensor


def _mx_kwargs_cls():
    mod = pytest.importorskip("torchao.prototype.mx_formats.mx_tensor")
    return mod.QuantizeTensorToMXKwargs


def _quantize_mx(
    data: torch.Tensor,
    *,
    elem_dtype: torch.dtype,
    dynamic_activation: bool = False,
    block_size: int = 32,
    scaling_mode: object | None = None,
    is_swizzled_scales: bool = False,
) -> torch.Tensor:
    mx_cls = _mx_tensor_cls()
    # Weight-only MX cannot run a matmul on its own; a forward needs the
    # activation-quant kwargs that the dynamic-activation MX recipe carries.
    kwargs_cls = _mx_kwargs_cls()
    kwargs = {} if scaling_mode is None else {"scaling_mode": scaling_mode}
    act_quant_kwargs = (
        kwargs_cls(
            elem_dtype=elem_dtype,
            block_size=block_size,
            is_swizzled_scales=is_swizzled_scales,
            **kwargs,
        )
        if dynamic_activation
        else None
    )
    return mx_cls.to_mx(
        data,
        elem_dtype,
        block_size=block_size,
        act_quant_kwargs=act_quant_kwargs,
        is_swizzled_scales=is_swizzled_scales,
        **kwargs,
    )


def _make_mx(
    *,
    elem_dtype: torch.dtype,
    rows: int = 16,
    cols: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    dynamic_activation: bool = False,
) -> torch.Tensor:
    return _quantize_mx(
        torch.randn(rows, cols, dtype=dtype),
        elem_dtype=elem_dtype,
        dynamic_activation=dynamic_activation,
    )


def _scale_mode(name: str) -> object:
    mod = pytest.importorskip("torchao.prototype.mx_formats.config")
    return getattr(mod.ScaleCalculationMode, name)


def _clone_mx(mx: torch.Tensor) -> torch.Tensor:
    mx_cls = _mx_tensor_cls()
    return mx_cls(
        mx.qdata.clone(),
        mx.scale.clone(),
        mx.elem_dtype,
        mx.block_size,
        mx.orig_dtype,
        mx.kernel_preference,
        mx.act_quant_kwargs,
        mx.is_swizzled_scales,
    )


def _reference_merge(
    mx: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    dense = MxAdapter.dequantize(mx)
    dense.addmm_(b, a, alpha=strength)
    return MxAdapter.requantize(dense, like=mx)


def _assert_mx_merge_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    assert torch.equal(
        actual.scale.view(torch.uint8),
        expected.scale.view(torch.uint8),
    )
    torch.testing.assert_close(
        actual.dequantize(torch.float32),
        expected.dequantize(torch.float32),
        rtol=0.15,
        atol=0.25,
    )
    differing_bytes = torch.count_nonzero(actual.qdata.view(torch.uint8) != expected.qdata.view(torch.uint8)).item()
    assert differing_bytes <= actual.qdata.numel() // 50 + 1


class TestMxAdapter:
    def test_supported_elem_dtype_gate(self) -> None:
        assert is_supported_mx_elem_dtype(torch.float8_e4m3fn)
        assert is_supported_mx_elem_dtype(torch.float8_e5m2)
        if _FP4 is not None:
            assert is_supported_mx_elem_dtype(_FP4)
        # MXFP6 (string elem dtypes) and plain dtypes are out of scope.
        assert not is_supported_mx_elem_dtype("fp6_e2m3")
        assert not is_supported_mx_elem_dtype("fp6_e3m2")
        assert not is_supported_mx_elem_dtype(torch.bfloat16)
        assert not is_supported_mx_elem_dtype(None)

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_matches_mx_only(self, elem_dtype: torch.dtype) -> None:
        qt = _make_mx(elem_dtype=elem_dtype)
        assert MxAdapter.matches(qt)
        assert not MxAdapter.matches(torch.zeros(16, 64, dtype=torch.bfloat16))

    def test_rejects_mxfp6_with_clear_error(self) -> None:
        # MXFP6 is a real MXTensor but intentionally unsupported: it must
        # not dispatch to MxAdapter, and with no other adapter matching it
        # should surface the registry's "no adapter" error.
        mx_cls = _mx_tensor_cls()
        try:
            f6 = mx_cls.to_mx(
                torch.randn(16, 64, dtype=torch.bfloat16),
                "fp6_e2m3",
                block_size=32,
            )
        except Exception:  # pragma: no cover - torchao build without fp6
            pytest.skip("this torchao build cannot construct MXFP6")
        assert not MxAdapter.matches(f6)
        with pytest.raises(NotImplementedError, match="No TensorAdapter"):
            select_adapter(f6)

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_pin_preserves_storage_and_metadata(self, elem_dtype: torch.dtype) -> None:
        mx_cls = _mx_tensor_cls()
        qt = _make_mx(elem_dtype=elem_dtype)
        pinned_param = PinnedParam(nn.Parameter(qt, requires_grad=False))

        pinned = pinned_param.make_cpu_param().data
        assert isinstance(pinned, mx_cls)
        assert pinned.qdata.is_pinned()
        assert pinned.scale.is_pinned()
        assert pinned.qdata.data_ptr() == pinned_param.pinned_state.storage[0].data_ptr()
        assert pinned.scale.data_ptr() == pinned_param.pinned_state.storage[1].data_ptr()
        assert pinned.elem_dtype == qt.elem_dtype
        assert pinned.block_size == qt.block_size
        assert pinned.orig_dtype == qt.orig_dtype
        assert pinned.is_swizzled_scales == qt.is_swizzled_scales
        assert pinned_param.compute_dtype is torch.bfloat16

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_transposed_storage_stride_is_preserved(self, elem_dtype: torch.dtype) -> None:
        qt = _make_mx(elem_dtype=elem_dtype, rows=16, cols=64).t()
        pinned_param = PinnedParam(nn.Parameter(qt, requires_grad=False))
        pinned = pinned_param.make_cpu_param().data

        assert pinned.shape == qt.shape
        assert pinned.qdata.stride() == qt.qdata.stride()
        assert pinned.scale.stride() == qt.scale.stride()

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_tensor_id_is_stable_and_keyed(self, elem_dtype: torch.dtype) -> None:
        qt = _make_mx(elem_dtype=elem_dtype)
        key = tensor_id(qt)
        assert key[0] == "torchao-mx"
        assert key[1][0] == qt.qdata.device
        assert key[2][0] == qt.scale.device
        assert key == tensor_id(qt)

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_target_layout_ignores_tensor_id(self, elem_dtype: torch.dtype) -> None:
        p1 = nn.Parameter(_make_mx(elem_dtype=elem_dtype), requires_grad=False)
        p2 = nn.Parameter(_make_mx(elem_dtype=elem_dtype), requires_grad=False)

        assert _param_target_layout(p1) == _param_target_layout(p2)

    def test_target_layout_distinguishes_mxfp8_and_mxfp4(self) -> None:
        if _FP4 is None:
            pytest.skip("torch lacks float4_e2m1fn_x2")
        p8 = nn.Parameter(_make_mx(elem_dtype=torch.float8_e4m3fn), requires_grad=False)
        p4 = nn.Parameter(_make_mx(elem_dtype=_FP4), requires_grad=False)

        assert _param_target_layout(p8) != _param_target_layout(p4)

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_dynamic_activation_metadata_is_keyed(self, elem_dtype: torch.dtype) -> None:
        # The dynamic-activation recipe carries ``act_quant_kwargs`` (a
        # dataclass) that flows through ``metadata_key`` in tensor_id /
        # layout_signature. Exercise it on CPU so the path is covered
        # without a GPU (the forward tests that use it are CUDA-gated).
        weight_only = nn.Parameter(
            _make_mx(elem_dtype=elem_dtype, dynamic_activation=False),
            requires_grad=False,
        )
        dynamic = nn.Parameter(
            _make_mx(elem_dtype=elem_dtype, dynamic_activation=True),
            requires_grad=False,
        )

        key = tensor_id(dynamic.data)
        assert key[0] == "torchao-mx"
        assert key == tensor_id(dynamic.data)
        # Activation quantization changes the matmul dispatch, so the
        # block-pool layout must distinguish it from the weight-only base.
        assert _param_target_layout(dynamic) != _param_target_layout(weight_only)

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_no_cpu_round_trip_or_trainable_swap_capability(self, elem_dtype: torch.dtype) -> None:
        pinned_param = PinnedParam(
            nn.Parameter(_make_mx(elem_dtype=elem_dtype), requires_grad=True),
        )
        state = pinned_param.allocate_gpu_storage(torch.device("cpu"))

        with pytest.raises(NotImplementedError, match="CPU round-trip"):
            pinned_param.copy_to_cpu(state)
        with pytest.raises(NotImplementedError, match="Parameter.data-swap"):
            pinned_param.validate_parameter_data_swap_target()

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_dequantize_requantize_preserves_representation(self, elem_dtype: torch.dtype) -> None:
        mx = _make_mx(elem_dtype=elem_dtype, rows=16, cols=64)
        dense = MxAdapter.dequantize(mx)
        assert dense.dtype is mx.orig_dtype
        torch.testing.assert_close(dense, mx.dequantize(mx.orig_dtype))

        # MX uses deterministic FLOOR scaling onto power-of-two (E8M0)
        # block scales, so an unmodified round trip reproduces the packed
        # bytes and scales exactly.
        again = MxAdapter.requantize(dense, like=mx)
        assert again.elem_dtype == mx.elem_dtype
        assert again.block_size == mx.block_size
        assert again.orig_dtype == mx.orig_dtype
        assert again.is_swizzled_scales == mx.is_swizzled_scales
        assert again.act_quant_kwargs == mx.act_quant_kwargs
        assert torch.equal(again.qdata.view(torch.uint8), mx.qdata.view(torch.uint8))
        assert torch.equal(again.scale.view(torch.uint8), mx.scale.view(torch.uint8))

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_requantize_preserves_scaling_mode(self, elem_dtype: torch.dtype) -> None:
        # The default MX inference recipe quantizes with RCEIL (not to_mx's
        # FLOOR default) and records the mode on act_quant_kwargs.
        # requantize must recover it so the merge re-encodes with the same
        # scale-rounding policy, not silently fall back to FLOOR.
        mx_cls = _mx_tensor_cls()
        kwargs_cls = _mx_kwargs_cls()
        try:
            from torchao.prototype.mx_formats.config import ScaleCalculationMode
        except ImportError as exc:
            pytest.skip(f"torchao ScaleCalculationMode unavailable: {exc}")

        weight = torch.randn(16, 64, dtype=torch.bfloat16)
        akw = kwargs_cls(
            elem_dtype=elem_dtype,
            block_size=32,
            scaling_mode=ScaleCalculationMode.RCEIL,
        )
        like = mx_cls.to_mx(
            weight,
            elem_dtype,
            block_size=32,
            scaling_mode=ScaleCalculationMode.RCEIL,
            act_quant_kwargs=akw,
        )

        # Requantize a fresh dense tensor (not the dequant of ``like``, whose
        # values already sit on the grid where RCEIL and FLOOR can coincide)
        # so the two modes genuinely diverge and the recovered mode is
        # observable.
        fresh = torch.randn(16, 64, dtype=torch.float32)
        again = MxAdapter.requantize(fresh, like=like)

        def _reencode(mode: object) -> torch.Tensor:
            return mx_cls.to_mx(
                fresh.to(like.orig_dtype),
                like.elem_dtype,
                block_size=like.block_size,
                scaling_mode=mode,
                act_quant_kwargs=like.act_quant_kwargs,
                is_swizzled_scales=like.is_swizzled_scales,
            )

        as_rceil = _reencode(ScaleCalculationMode.RCEIL)
        as_floor = _reencode(ScaleCalculationMode.FLOOR)
        # The two modes differ on this input, and requantize reproduces the
        # RCEIL re-encode — i.e. it recovered the mode rather than using the
        # FLOOR default.
        assert not torch.equal(as_rceil.scale.view(torch.uint8), as_floor.scale.view(torch.uint8))
        assert torch.equal(again.scale.view(torch.uint8), as_rceil.scale.view(torch.uint8))

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_requantize_rejects_shape_mismatch(self, elem_dtype: torch.dtype) -> None:
        mx = _make_mx(elem_dtype=elem_dtype, rows=16, cols=64)
        with pytest.raises(ValueError, match="Cannot requantize"):
            MxAdapter.requantize(torch.randn(64, 16), like=mx)

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_merge_rejects_transposed_weight(self, elem_dtype: torch.dtype) -> None:
        # A transposed MX weight has non-contiguous packed qdata, which the
        # standard-layout re-encode cannot fill. The adapter preserves this
        # layout for movement but rejects it for merge with a clear error
        # (rather than an opaque kernel assertion); routed LoRA still works.
        transposed = _make_mx(elem_dtype=elem_dtype, rows=16, cols=64).t()
        assert not transposed.qdata.is_contiguous()
        with pytest.raises(ValueError, match="non-contiguous.*MX"):
            MxAdapter.requantize(
                torch.randn(*transposed.shape, dtype=torch.float32),
                like=transposed,
            )

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_lora_transform_requantizes_param_in_place(self, elem_dtype: torch.dtype) -> None:
        mx_cls = _mx_tensor_cls()
        rows, cols, rank = 16, 64, 2
        mx = _make_mx(elem_dtype=elem_dtype, rows=rows, cols=cols)
        param = nn.Parameter(mx, requires_grad=False)
        a = torch.randn(rank, cols)
        b = torch.randn(rows, rank)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.5)])
        original_param = param
        original_qdata_ptr = param.data.qdata.data_ptr()

        expected_dense = MxAdapter.dequantize(mx)
        expected_dense.addmm_(
            b.to(expected_dense.dtype),
            a.to(expected_dense.dtype),
            alpha=0.5,
        )
        expected = MxAdapter.requantize(expected_dense, like=mx)

        transform.validate_target(param)
        transform.apply(param)

        # copy_into mutates the existing wrapper's storage in place, so the
        # Parameter object and its packed-element buffer keep their identity.
        assert param is original_param
        assert param.data.qdata.data_ptr() == original_qdata_ptr
        assert isinstance(param.data, mx_cls)
        assert torch.equal(param.data.qdata.view(torch.uint8), expected.qdata.view(torch.uint8))
        assert torch.equal(param.data.scale.view(torch.uint8), expected.scale.view(torch.uint8))

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_merge_lora_merges_mx_weight(self, elem_dtype: torch.dtype) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)

        model = M()
        model.lin.weight.requires_grad = False
        model.lin.weight = nn.Parameter(
            _make_mx(elem_dtype=elem_dtype, rows=16, cols=64),
            requires_grad=False,
        )
        # copy_into mutates the weight's storage in place, so snapshot the
        # original packed bytes rather than holding a tensor ref.
        original_qdata = model.lin.weight.data.qdata.view(torch.uint8).clone()
        lora = LoRA.from_state_dict(
            state_dict={
                "lin.lora_A.weight": torch.randn(4, 64),
                "lin.lora_B.weight": torch.randn(16, 4),
            }
        )

        merged = merge_lora(model, [(lora, 1.0)])

        assert merged == 1
        assert not torch.equal(model.lin.weight.data.qdata.view(torch.uint8), original_qdata)

    @CUDA
    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    @pytest.mark.parametrize("scaling_mode_name", ["FLOOR", "RCEIL"])
    @pytest.mark.parametrize("is_swizzled_scales", [False, True])
    def test_triton_merge_matches_generic_for_odd_shape_and_rank(
        self,
        elem_dtype: torch.dtype,
        scaling_mode_name: str,
        is_swizzled_scales: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(31)
        rows, cols, rank = 19, 96, 7
        scaling_mode = _scale_mode(scaling_mode_name)
        base = _quantize_mx(
            torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16),
            elem_dtype=elem_dtype,
            dynamic_activation=True,
            scaling_mode=scaling_mode,
            is_swizzled_scales=is_swizzled_scales,
        )
        target = _clone_mx(base)
        a = torch.randn(rank, cols, device="cuda", dtype=torch.bfloat16).mul_(0.15)
        b = torch.randn(rows, rank, device="cuda", dtype=torch.bfloat16).mul_(0.15)
        expected = _reference_merge(base, b, a, 0.7)
        qdata_ptr = target.qdata.data_ptr()
        scale_ptr = target.scale.data_ptr()
        metadata = (
            target.elem_dtype,
            target.block_size,
            target.orig_dtype,
            target.kernel_preference,
            target.act_quant_kwargs,
            target.is_swizzled_scales,
        )

        def fail_fallback(
            _target: torch.Tensor,
            _b: torch.Tensor,
            _a: torch.Tensor,
            _strength: float,
        ) -> None:
            raise AssertionError("supported CUDA MX merge must use Triton")

        monkeypatch.setattr(mx_adapter_impl, "_torch_merge_mx_lora_", fail_fallback)
        result = MxAdapter.merge_lora_(target, b, a, 0.7)
        torch.cuda.synchronize()

        assert result is None
        assert target.qdata.data_ptr() == qdata_ptr
        assert target.scale.data_ptr() == scale_ptr
        assert (
            target.elem_dtype,
            target.block_size,
            target.orig_dtype,
            target.kernel_preference,
            target.act_quant_kwargs,
            target.is_swizzled_scales,
        ) == metadata
        _assert_mx_merge_close(target, expected)

    @CUDA
    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    @pytest.mark.parametrize("scaling_mode_name", ["CEIL", "EVEN"])
    def test_triton_merge_supports_additional_scale_modes(
        self,
        elem_dtype: torch.dtype,
        scaling_mode_name: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(32)
        rows, cols, rank = 17, 64, 5
        scaling_mode = _scale_mode(scaling_mode_name)
        base = _quantize_mx(
            torch.randn(rows, cols, device="cuda", dtype=torch.float32),
            elem_dtype=elem_dtype,
            dynamic_activation=True,
            scaling_mode=scaling_mode,
        )
        target = _clone_mx(base)
        a = torch.randn(rank, cols, device="cuda").mul_(0.1)
        b = torch.randn(rows, rank, device="cuda").mul_(0.1)
        expected = _reference_merge(base, b, a, -0.4)

        def fail_fallback(
            _target: torch.Tensor,
            _b: torch.Tensor,
            _a: torch.Tensor,
            _strength: float,
        ) -> None:
            raise AssertionError("supported MX scale mode must use Triton")

        monkeypatch.setattr(mx_adapter_impl, "_torch_merge_mx_lora_", fail_fallback)
        MxAdapter.merge_lora_(target, b, a, -0.4)
        torch.cuda.synchronize()

        _assert_mx_merge_close(target, expected)

    @CUDA
    def test_triton_merge_handles_multiple_swizzled_scale_tiles(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(36)
        rows, cols, rank = 131, 160, 3
        base = _quantize_mx(
            torch.randn(rows, cols, device="cuda"),
            elem_dtype=torch.float8_e5m2,
            scaling_mode=_scale_mode("RCEIL"),
            is_swizzled_scales=True,
        )
        target = _clone_mx(base)
        a = torch.randn(rank, cols, device="cuda").mul_(0.1)
        b = torch.randn(rows, rank, device="cuda").mul_(0.1)
        expected = _reference_merge(base, b, a, 0.6)

        def fail_fallback(
            _target: torch.Tensor,
            _b: torch.Tensor,
            _a: torch.Tensor,
            _strength: float,
        ) -> None:
            raise AssertionError("supported swizzled MX layout must use Triton")

        monkeypatch.setattr(mx_adapter_impl, "_torch_merge_mx_lora_", fail_fallback)
        MxAdapter.merge_lora_(target, b, a, 0.6)
        torch.cuda.synchronize()

        _assert_mx_merge_close(target, expected)

    @CUDA
    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_triton_zero_update_matches_generic(
        self,
        elem_dtype: torch.dtype,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(37)
        rows, cols, rank = 17, 64, 5
        target = _quantize_mx(
            torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16),
            elem_dtype=elem_dtype,
        )
        zeros_b = torch.zeros(rows, rank, device="cuda", dtype=torch.bfloat16)
        zeros_a = torch.zeros(rank, cols, device="cuda", dtype=torch.bfloat16)
        expected = _reference_merge(target, zeros_b, zeros_a, 1.0)

        def fail_fallback(
            _target: torch.Tensor,
            _b: torch.Tensor,
            _a: torch.Tensor,
            _strength: float,
        ) -> None:
            raise AssertionError("supported CUDA MX merge must use Triton")

        monkeypatch.setattr(mx_adapter_impl, "_torch_merge_mx_lora_", fail_fallback)
        MxAdapter.merge_lora_(
            target,
            zeros_b,
            zeros_a,
            1.0,
        )
        torch.cuda.synchronize()

        assert torch.equal(
            target.qdata.view(torch.uint8),
            expected.qdata.view(torch.uint8),
        )
        assert torch.equal(
            target.scale.view(torch.uint8),
            expected.scale.view(torch.uint8),
        )

    @CUDA
    @pytest.mark.parametrize(
        "elem_dtype",
        [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            pytest.param(
                _FP4,
                marks=pytest.mark.skipif(
                    _FP4 is None,
                    reason="torch lacks float4_e2m1fn_x2",
                ),
            ),
        ],
    )
    def test_triton_preserves_e8m0_code_zero_subnormal(
        self,
        elem_dtype: torch.dtype,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows, cols, rank = 3, 32, 2
        target = _quantize_mx(
            torch.full(
                (rows, cols),
                torch.finfo(torch.float32).tiny,
                device="cuda",
            ),
            elem_dtype=elem_dtype,
        )
        assert torch.all(target.scale.view(torch.uint8) == 0)
        zeros_b = torch.zeros(rows, rank, device="cuda")
        zeros_a = torch.zeros(rank, cols, device="cuda")
        expected = _reference_merge(target, zeros_b, zeros_a, 1.0)
        assert torch.count_nonzero(expected.dequantize()).item() > 0

        def fail_fallback(
            _target: torch.Tensor,
            _b: torch.Tensor,
            _a: torch.Tensor,
            _strength: float,
        ) -> None:
            raise AssertionError("standard MX storage must use Triton")

        monkeypatch.setattr(mx_adapter_impl, "_torch_merge_mx_lora_", fail_fallback)
        MxAdapter.merge_lora_(target, zeros_b, zeros_a, 1.0)
        torch.cuda.synchronize()

        assert torch.equal(
            target.qdata.view(torch.uint8),
            expected.qdata.view(torch.uint8),
        )
        assert torch.equal(
            target.scale.view(torch.uint8),
            expected.scale.view(torch.uint8),
        )

    @CUDA
    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_triton_merge_handles_multiple_factors(
        self,
        elem_dtype: torch.dtype,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(33)
        rows, cols = 19, 96
        base = _quantize_mx(
            torch.randn(rows, cols, device="cuda", dtype=torch.float32),
            elem_dtype=elem_dtype,
        )
        target = nn.Parameter(_clone_mx(base), requires_grad=False)
        factor_specs = [
            (3, 0.25),
            (5, -0.4),
        ]
        factors = []
        packed_a = []
        packed_b = []
        for rank, strength in factor_specs:
            a = torch.randn(rank, cols)
            b = torch.randn(rows, rank)
            factors.append(ScaledLoRAFactor.from_tensors(a, b, strength))
            packed_a.append(a.cuda().mul(strength))
            packed_b.append(b.cuda())
        expected = _reference_merge(
            base,
            torch.cat(packed_b, dim=1),
            torch.cat(packed_a, dim=0),
            1.0,
        )
        calls = 0
        triton_merge = mx_adapter_impl._triton_merge_mx_lora_
        assert triton_merge is not None

        def record_triton(*args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            triton_merge(*args, **kwargs)

        monkeypatch.setattr(
            mx_adapter_impl,
            "_triton_merge_mx_lora_",
            record_triton,
        )
        param_id = id(target)
        qdata_ptr = target.data.qdata.data_ptr()
        scale_ptr = target.data.scale.data_ptr()
        transform = LoRATransform(factors)
        transform.validate_target(target)
        transform.apply(target)
        torch.cuda.synchronize()

        assert calls == 1
        assert id(target) == param_id
        assert target.data.qdata.data_ptr() == qdata_ptr
        assert target.data.scale.data_ptr() == scale_ptr
        _assert_mx_merge_close(target.data, expected)

    @CUDA
    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_merge_falls_back_for_nonstandard_block_size(
        self,
        elem_dtype: torch.dtype,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(34)
        rows, cols, rank = 13, 64, 3
        base = _quantize_mx(
            torch.randn(rows, cols, device="cuda", dtype=torch.float32),
            elem_dtype=elem_dtype,
            block_size=16,
        )
        target = _clone_mx(base)
        a = torch.randn(rank, cols, device="cuda")
        b = torch.randn(rows, rank, device="cuda")
        expected = _reference_merge(base, b, a, 0.3)

        def fail_triton(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("nonstandard MX block size must use fallback")

        monkeypatch.setattr(
            mx_adapter_impl,
            "_triton_merge_mx_lora_",
            fail_triton,
        )
        MxAdapter.merge_lora_(target, b, a, 0.3)

        assert torch.equal(
            target.qdata.view(torch.uint8),
            expected.qdata.view(torch.uint8),
        )
        assert torch.equal(
            target.scale.view(torch.uint8),
            expected.scale.view(torch.uint8),
        )

    @CUDA
    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_merge_falls_back_when_triton_is_unavailable(
        self,
        elem_dtype: torch.dtype,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(35)
        rows, cols, rank = 11, 64, 3
        base = _quantize_mx(
            torch.randn(rows, cols, device="cuda", dtype=torch.float32),
            elem_dtype=elem_dtype,
        )
        target = _clone_mx(base)
        a = torch.randn(rank, cols, device="cuda")
        b = torch.randn(rows, rank, device="cuda")
        expected = _reference_merge(base, b, a, -0.2)

        monkeypatch.setattr(mx_adapter_impl, "_triton_merge_mx_lora_", None)
        MxAdapter.merge_lora_(target, b, a, -0.2)

        assert torch.equal(
            target.qdata.view(torch.uint8),
            expected.qdata.view(torch.uint8),
        )
        assert torch.equal(
            target.scale.view(torch.uint8),
            expected.scale.view(torch.uint8),
        )

    @CUDA
    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_allocate_copy_make_gpu_param_preserves_wrapper(self, elem_dtype: torch.dtype) -> None:
        mx_cls = _mx_tensor_cls()
        pinned_param = PinnedParam(
            nn.Parameter(_make_mx(elem_dtype=elem_dtype), requires_grad=False),
        )

        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()
        pinned = pinned_param.make_cpu_param().data

        assert isinstance(gpu_param.data, mx_cls)
        assert gpu_param.data.qdata.is_cuda
        assert gpu_param.data.scale.is_cuda
        assert gpu_param.data.elem_dtype == pinned.elem_dtype
        assert gpu_param.data.block_size == pinned.block_size
        assert gpu_param.data.orig_dtype == pinned.orig_dtype
        # Compare the raw bytes: fp8 / e8m0 dtypes carry NaN encodings that
        # break value equality, so view as uint8 for a bitwise check.
        assert torch.equal(
            gpu_param.data.qdata.view(torch.uint8).cpu(),
            pinned.qdata.view(torch.uint8),
        )
        assert torch.equal(
            gpu_param.data.scale.view(torch.uint8).cpu(),
            pinned.scale.view(torch.uint8),
        )

    @CUDA
    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_model_offloader_cuda_forward_dynamic_mx(self, elem_dtype: torch.dtype) -> None:
        layer = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
        layer.weight.requires_grad = False
        weight = layer.weight.detach().contiguous()
        layer.weight = nn.Parameter(
            _quantize_mx(weight, elem_dtype=elem_dtype, dynamic_activation=True),
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
    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_model_offloader_routed_lora_on_dynamic_mx(self, elem_dtype: torch.dtype) -> None:
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
            weight = block.weight.detach().contiguous()
            block.weight = nn.Parameter(
                _quantize_mx(weight, elem_dtype=elem_dtype, dynamic_activation=True),
                requires_grad=False,
            )
        offloader = _make_model_offloader(
            model,
            blocks_attr=["blocks"],
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
                stream_config=StreamConfig(num_resident_blocks=1, num_prefetch_blocks=0),
            ) as active:
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (128, 128)
            assert y.dtype is torch.bfloat16
        finally:
            offloader.deactivate()

    @CUDA
    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_streamed_mx_merge_requantizes_on_activate(self, elem_dtype: torch.dtype) -> None:
        mx_cls = _mx_tensor_cls()

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.Linear(64, 64, bias=False, dtype=torch.bfloat16),
                        nn.Linear(64, 64, bias=False, dtype=torch.bfloat16),
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
                _quantize_mx(
                    block.weight.detach().contiguous(),
                    elem_dtype=elem_dtype,
                    dynamic_activation=True,
                ),
                requires_grad=False,
            )
        mx = model.blocks[0].weight.data
        rank = 4
        a = torch.randn(rank, 64)
        b = torch.randn(64, rank)
        lora = LoRA.from_state_dict(
            state_dict={
                "blocks.0.lora_A.weight": a,
                "blocks.0.lora_B.weight": b,
            }
        )
        # Compute the reference on CUDA, matching the device the offloader
        # merges on (CPU vs CUDA rounding can flip boundary elements). The
        # offloader merges into a byte-identical GPU copy of the original
        # weight, so move that same tensor — don't re-quantize from dense.
        mx_cuda = mx.cuda()
        expected_dense = MxAdapter.dequantize(mx_cuda)
        expected_dense.addmm_(
            b.cuda().to(expected_dense.dtype),
            a.cuda().to(expected_dense.dtype),
            alpha=0.5,
        )
        expected = MxAdapter.requantize(expected_dense, like=mx_cuda)

        offloader = _make_model_offloader(
            model,
            blocks_attr=["blocks"],
        )
        try:
            x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
            with activated_model(
                offloader,
                "cuda",
                loras=[lora],
                lora_strengths=[0.5],
                lora_mode="merge",
                stochastic_rounding=False,
                stream_config=StreamConfig(num_resident_blocks=1, num_prefetch_blocks=0),
            ) as active:
                merged = active.blocks[0].weight.data
                assert isinstance(merged, mx_cls)
                torch.testing.assert_close(
                    merged.dequantize(mx.orig_dtype).to(torch.float32),
                    expected.dequantize(mx.orig_dtype).to(torch.float32),
                )
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (8, 64)
        finally:
            offloader.deactivate()
