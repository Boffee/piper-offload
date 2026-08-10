"""Tests for TorchAO NVFP4 adapter integration."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

import piper_offload.nvfp4_adapter as nvfp4_adapter_module
from piper_offload import (
    LoRA,
    LoRATransform,
    ModelOffloader,
    ScaledLoRAFactor,
    StreamConfig,
    merge_lora,
)
from piper_offload.nvfp4_adapter import Nvfp4Adapter
from piper_offload.pinned_param import PinnedParam
from piper_offload.tensor_adapter_registry import tensor_id
from piper_offload.streamed_component import _param_target_layout
from tests.conftest import activated_model

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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


def _nvfp4_modules():
    pytest.importorskip("numpy")
    mod = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    return mod.NVFP4Tensor, mod.QuantizeTensorToNVFP4Kwargs


def _make_nvfp4(
    *,
    rows: int = 16,
    cols: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    dynamic_activation: bool = True,
    swizzled: bool = False,
) -> torch.Tensor:
    nvfp4_tensor_cls, kwargs_cls = _nvfp4_modules()
    act_quant_kwargs = (
        kwargs_cls(
            is_swizzled_scales=swizzled,
            use_dynamic_per_tensor_scale=True,
        )
        if dynamic_activation
        else None
    )
    return nvfp4_tensor_cls.to_nvfp4(
        torch.randn(rows, cols, dtype=dtype),
        per_tensor_scale=torch.tensor(0.01, dtype=torch.float32),
        is_swizzled_scales=swizzled,
        use_triton_kernel=False,
        act_quant_kwargs=act_quant_kwargs,
    )


def _make_nvfp4_amax(
    *,
    rows: int = 16,
    cols: int = 64,
    swizzled: bool = True,
) -> torch.Tensor:
    """NVFP4 weight whose two-level ``per_tensor_scale`` is amax-derived,
    matching the real ``quantize_`` configs — so a dequant/requant round
    trip reproduces the representation exactly (requantize recomputes the
    same amax-derived global scale)."""
    nvfp4_tensor_cls, _ = _nvfp4_modules()
    mod = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    weight = torch.randn(rows, cols, dtype=torch.bfloat16)
    per_tensor_scale = mod.per_tensor_amax_to_scale(weight.abs().max().to(torch.float32))
    return nvfp4_tensor_cls.to_nvfp4(
        weight,
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=swizzled,
        use_triton_kernel=False,
    )


def _make_nvfp4_cuda(
    weight: torch.Tensor,
    *,
    swizzled: bool,
    two_level: bool,
    dynamic_activation: bool = False,
    use_triton_kernel: bool = False,
    act_per_tensor_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    nvfp4_tensor_cls, kwargs_cls = _nvfp4_modules()
    mod = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    per_tensor_scale = (
        mod.per_tensor_amax_to_scale(weight.abs().max()).clamp_min(torch.finfo(torch.float32).eps)
        if two_level
        else None
    )
    act_quant_kwargs = (
        kwargs_cls(
            is_swizzled_scales=swizzled,
            use_dynamic_per_tensor_scale=True,
            use_triton_kernel=use_triton_kernel,
        )
        if dynamic_activation
        else None
    )
    nv = nvfp4_tensor_cls.to_nvfp4(
        weight,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=act_per_tensor_scale,
        is_swizzled_scales=swizzled,
        use_triton_kernel=False,
        act_quant_kwargs=act_quant_kwargs,
    )
    nv.use_triton_kernel = use_triton_kernel
    return nv


def _expected_nvfp4_merge(
    nv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    dense = Nvfp4Adapter.dequantize(nv)
    dense.addmm_(b, a, alpha=strength)
    return Nvfp4Adapter.requantize(dense, like=nv)


def _assert_triton_nvfp4_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    if actual.per_tensor_scale is not None:
        assert expected.per_tensor_scale is not None
        torch.testing.assert_close(
            actual.per_tensor_scale,
            expected.per_tensor_scale,
            rtol=0.02,
            atol=torch.finfo(torch.float32).eps,
        )
    torch.testing.assert_close(
        actual.get_hp_scales().to(torch.float32),
        expected.get_hp_scales().to(torch.float32),
        rtol=0.13,
        atol=torch.finfo(torch.float32).eps,
    )
    block_step = expected.get_hp_scales().to(torch.float32).max().item()
    torch.testing.assert_close(
        actual.dequantize(torch.float32),
        expected.dequantize(torch.float32),
        rtol=0.2,
        atol=max(2 * block_step, torch.finfo(torch.float32).eps),
    )


class TestNvfp4Adapter:
    def test_matches_nvfp4_only(self) -> None:
        qt = _make_nvfp4()
        assert Nvfp4Adapter.matches(qt)
        assert not Nvfp4Adapter.matches(torch.zeros(16, 16, dtype=torch.bfloat16))

    def test_pin_preserves_storage_and_metadata(self) -> None:
        nvfp4_tensor_cls, _ = _nvfp4_modules()
        qt = _make_nvfp4(swizzled=True)
        p = nn.Parameter(qt, requires_grad=False)
        pinned_param = PinnedParam(p)

        pinned = pinned_param.make_cpu_param().data
        assert isinstance(pinned, nvfp4_tensor_cls)
        assert pinned.qdata.is_pinned()
        assert pinned.scale.is_pinned()
        assert pinned.per_tensor_scale is not None
        assert pinned.per_tensor_scale.is_pinned()
        assert pinned.qdata.data_ptr() == pinned_param.pinned_state.storage[0].data_ptr()
        assert pinned.scale.data_ptr() == pinned_param.pinned_state.storage[1].data_ptr()
        assert pinned.per_tensor_scale.data_ptr() == pinned_param.pinned_state.storage[2].data_ptr()
        assert pinned.block_size == qt.block_size
        assert pinned.orig_dtype == qt.orig_dtype
        assert pinned.is_swizzled_scales == qt.is_swizzled_scales
        assert pinned.use_triton_kernel == qt.use_triton_kernel
        assert pinned.act_quant_kwargs == qt.act_quant_kwargs
        assert pinned_param.compute_dtype is torch.bfloat16

    def test_transposed_qdata_stride_is_preserved(self) -> None:
        qt = _make_nvfp4(rows=16, cols=32).t()
        pinned_param = PinnedParam(nn.Parameter(qt, requires_grad=False))
        pinned = pinned_param.make_cpu_param().data

        assert pinned.shape == qt.shape
        assert pinned.qdata.stride() == qt.qdata.stride()
        assert pinned.scale.stride() == qt.scale.stride()
        assert pinned.dequantize().shape == qt.dequantize().shape

    def test_tensor_id_tracks_optional_scale_tensor(self) -> None:
        qt = _make_nvfp4()
        key = tensor_id(qt)
        assert key[0] == "torchao-nvfp4"
        assert key[1][0] == qt.qdata.device
        assert key[2][0] == qt.scale.device
        assert key[3][0] == qt.per_tensor_scale.device
        assert key == tensor_id(qt)

    def test_target_layout_ignores_tensor_id(self) -> None:
        p1 = nn.Parameter(_make_nvfp4(), requires_grad=False)
        p2 = nn.Parameter(_make_nvfp4(), requires_grad=False)

        assert _param_target_layout(p1) == _param_target_layout(p2)

    def test_target_layout_tracks_activation_quantization(self) -> None:
        with_activation = nn.Parameter(_make_nvfp4(dynamic_activation=True), requires_grad=False)
        weight_only = nn.Parameter(_make_nvfp4(dynamic_activation=False), requires_grad=False)

        assert _param_target_layout(with_activation) != _param_target_layout(weight_only)

    def test_no_cpu_round_trip_or_trainable_swap_capability(self) -> None:
        pinned_param = PinnedParam(
            nn.Parameter(_make_nvfp4(), requires_grad=True),
        )
        state = pinned_param.allocate_gpu_storage(torch.device("cpu"))

        with pytest.raises(NotImplementedError, match="CPU round-trip"):
            pinned_param.copy_to_cpu(state)
        with pytest.raises(NotImplementedError, match="Parameter.data-swap"):
            pinned_param.validate_parameter_data_swap_target()

    @pytest.mark.parametrize("swizzled", [False, True])
    def test_dequantize_requantize_preserves_representation(self, swizzled: bool) -> None:
        nvfp4_cls, _ = _nvfp4_modules()
        # Amax-derived per-tensor scale (as the real configs produce) makes
        # the round trip exact: requantize recomputes the same global scale.
        nv = _make_nvfp4_amax(rows=16, cols=64, swizzled=swizzled)
        dense = Nvfp4Adapter.dequantize(nv)
        assert dense.dtype is nv.orig_dtype
        torch.testing.assert_close(dense, nv.dequantize(nv.orig_dtype))

        again = Nvfp4Adapter.requantize(dense, like=nv)
        assert isinstance(again, nvfp4_cls)
        assert again.block_size == nv.block_size
        assert again.orig_dtype == nv.orig_dtype
        assert again.is_swizzled_scales == nv.is_swizzled_scales
        assert (again.per_tensor_scale is None) == (nv.per_tensor_scale is None)
        # Re-encoding uses the torch path; the swizzled layout and packed
        # bytes match the original regardless of its use_triton_kernel flag.
        assert torch.equal(again.qdata, nv.qdata)
        assert torch.equal(again.scale.view(torch.uint8), nv.scale.view(torch.uint8))

    def test_requantize_rejects_shape_mismatch(self) -> None:
        nv = _make_nvfp4(rows=16, cols=64)
        with pytest.raises(ValueError, match="Cannot requantize"):
            Nvfp4Adapter.requantize(torch.randn(64, 16), like=nv)

    def test_requantize_rejects_non_scalar_per_tensor_scale(self) -> None:
        # A non-scalar per_tensor_scale (per-expert grouped/MoE scales) is
        # not constructible via to_nvfp4 today, and a 3-D weight is rejected
        # earlier by LoRA factor-shape validation — but guard requantize
        # itself so a future per-expert layout fails loudly instead of
        # collapsing every expert to one global scale. Build it through the
        # raw wrapper to exercise the guard.
        from piper_offload._torchao_nvfp4 import create_nvfp4_tensor

        nvfp4_cls, _ = _nvfp4_modules()
        mod = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
        experts, n, k = 4, 64, 64
        weight = torch.randn(experts, n, k, dtype=torch.bfloat16)
        base = nvfp4_cls.to_nvfp4(
            weight,
            per_tensor_scale=mod.per_tensor_amax_to_scale(weight.abs().max().to(torch.float32)),
            is_swizzled_scales=False,
            use_triton_kernel=False,
        )
        per_expert = create_nvfp4_tensor(
            base.qdata,
            base.scale,
            base.block_size,
            base.orig_dtype,
            torch.rand(experts, 1, 1, dtype=torch.float32),
            None,
            base.is_swizzled_scales,
            base.use_triton_kernel,
            base.act_quant_kwargs,
        )
        assert per_expert.per_tensor_scale.numel() == experts
        with pytest.raises(ValueError, match="non-scalar per_tensor_scale"):
            Nvfp4Adapter.requantize(
                torch.randn(experts, n, k, dtype=torch.float32),
                like=per_expert,
            )

    def test_requantize_zero_dense_does_not_nan(self) -> None:
        # An all-zero merged weight (e.g. a LoRA delta that exactly cancels
        # the base) recomputes a per_tensor_scale of 0; to_nvfp4's two-level
        # path would then divide block scales by it and emit NaN. requantize
        # must fall back to a valid scale and produce a clean all-zero
        # tensor.
        nvfp4_cls, _ = _nvfp4_modules()
        nv = _make_nvfp4(rows=16, cols=64)
        assert nv.per_tensor_scale is not None  # two-level scaling
        again = Nvfp4Adapter.requantize(torch.zeros(16, 64, dtype=torch.float32), like=nv)
        assert isinstance(again, nvfp4_cls)
        assert not torch.isnan(again.scale.float()).any()
        dequant = again.dequantize(nv.orig_dtype)
        assert not torch.isnan(dequant.float()).any()
        assert torch.count_nonzero(dequant).item() == 0

    def test_merge_rejects_transposed_weight(self) -> None:
        # A transposed NVFP4 weight has non-contiguous packed qdata, which
        # the standard-layout re-encode cannot fill. The adapter preserves
        # this layout for movement but rejects it for merge with a clear
        # error (rather than an opaque kernel assertion); routed LoRA still
        # works.
        transposed = _make_nvfp4(rows=16, cols=64).t()
        assert not transposed.qdata.is_contiguous()
        with pytest.raises(ValueError, match="non-contiguous.*NVFP4"):
            Nvfp4Adapter.requantize(
                torch.randn(*transposed.shape, dtype=torch.float32),
                like=transposed,
            )

    def test_lora_transform_requantizes_param_in_place(self) -> None:
        nvfp4_cls, _ = _nvfp4_modules()
        rows, cols, rank = 16, 64, 2
        nv = _make_nvfp4(rows=rows, cols=cols, dynamic_activation=False)
        param = nn.Parameter(nv, requires_grad=False)
        a = torch.randn(rank, cols)
        b = torch.randn(rows, rank)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.5)])
        original_param = param
        original_qdata_ptr = param.data.qdata.data_ptr()

        # Mirror the merge path exactly so the comparison is deterministic.
        expected_dense = Nvfp4Adapter.dequantize(nv)
        expected_dense.addmm_(
            b.to(expected_dense.dtype),
            a.to(expected_dense.dtype),
            alpha=0.5,
        )
        expected = Nvfp4Adapter.requantize(expected_dense, like=nv)

        transform.apply(param)

        # copy_into mutates the existing wrapper's storage in place, so the
        # Parameter object and its packed-FP4 buffer keep their identity.
        assert param is original_param
        assert param.data.qdata.data_ptr() == original_qdata_ptr
        assert isinstance(param.data, nvfp4_cls)
        assert torch.equal(param.data.qdata, expected.qdata)
        assert torch.equal(param.data.scale.view(torch.uint8), expected.scale.view(torch.uint8))

    def test_merge_lora_merges_nvfp4_weight(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)

        model = M()
        model.lin.weight.requires_grad = False
        model.lin.weight = nn.Parameter(
            _make_nvfp4(rows=16, cols=64, dynamic_activation=False),
            requires_grad=False,
        )
        # copy_into mutates the weight's storage in place, so snapshot the
        # original packed bytes rather than holding a tensor ref.
        original_qdata = model.lin.weight.data.qdata.clone()
        lora = LoRA.from_state_dict(
            state_dict={
                "lin.lora_A.weight": torch.randn(4, 64),
                "lin.lora_B.weight": torch.randn(16, 4),
            }
        )

        merged = merge_lora(model, [(lora, 1.0)])

        assert merged == 1
        assert isinstance(model.lin.weight.data, _nvfp4_modules()[0])
        assert not torch.equal(model.lin.weight.data.qdata, original_qdata)

    @CUDA
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("swizzled", [False, True])
    @pytest.mark.parametrize("two_level", [False, True])
    def test_triton_merge_matches_supported_representations(
        self,
        dtype: torch.dtype,
        swizzled: bool,
        two_level: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # NVFP4 requires K divisible by 16. Odd M and rank exercise both
        # Triton tail masks while retaining a valid packed representation.
        rows, cols, rank = 19, 80, 13
        weight = torch.randn(rows, cols, device="cuda", dtype=dtype)
        nv = _make_nvfp4_cuda(
            weight,
            swizzled=swizzled,
            two_level=two_level,
        )
        a = torch.randn(cols, rank, device="cuda", dtype=dtype).t()
        b = torch.randn(rank, rows, device="cuda", dtype=dtype).t()
        strength = -0.234375
        expected = _expected_nvfp4_merge(nv, b, a, strength)
        qdata_ptr = nv.qdata.data_ptr()
        scale_ptr = nv.scale.data_ptr()
        global_scale_ptr = nv.per_tensor_scale.data_ptr() if nv.per_tensor_scale is not None else None
        triton_merge = nvfp4_adapter_module._triton_merge_nvfp4_lora
        assert triton_merge is not None
        tracked_triton = Mock(wraps=triton_merge)
        monkeypatch.setattr(
            nvfp4_adapter_module,
            "_triton_merge_nvfp4_lora",
            tracked_triton,
        )

        assert not a.is_contiguous()
        assert not b.is_contiguous()
        Nvfp4Adapter.merge_lora_(nv, b, a, strength)
        torch.cuda.synchronize()

        assert nv.qdata.data_ptr() == qdata_ptr
        assert nv.scale.data_ptr() == scale_ptr
        if nv.per_tensor_scale is not None:
            assert nv.per_tensor_scale.data_ptr() == global_scale_ptr
        assert nv.is_swizzled_scales is swizzled
        assert tracked_triton.call_count == 1
        _assert_triton_nvfp4_close(nv, expected)

    @CUDA
    def test_triton_transform_packs_loras_and_preserves_all_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows, cols = 21, 80
        dtype = torch.bfloat16
        act_per_tensor_scale = torch.tensor(
            0.0078125,
            device="cuda",
            dtype=torch.float32,
        )
        nv = _make_nvfp4_cuda(
            torch.randn(rows, cols, device="cuda", dtype=dtype),
            swizzled=True,
            two_level=True,
            dynamic_activation=True,
            use_triton_kernel=True,
            act_per_tensor_scale=act_per_tensor_scale,
        )
        param = nn.Parameter(nv, requires_grad=False)
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
            [a.to(device="cuda", dtype=dtype).mul_(strength) for a, _b, strength in factor_inputs],
            dim=0,
        )
        packed_b = torch.cat(
            [b.to(device="cuda", dtype=dtype) for _a, b, _strength in factor_inputs],
            dim=1,
        )
        dense = Nvfp4Adapter.dequantize(nv)
        dense.addmm_(packed_b, packed_a)
        expected = Nvfp4Adapter.requantize(dense, like=nv)
        storage = (
            nv.qdata,
            nv.scale,
            nv.per_tensor_scale,
            nv.act_per_tensor_scale,
        )
        storage_ptrs = tuple(t.data_ptr() for t in storage)
        activation_value = nv.act_per_tensor_scale.clone()
        metadata = (
            nv.block_size,
            nv.orig_dtype,
            nv.is_swizzled_scales,
            nv.use_triton_kernel,
            nv.act_quant_kwargs,
        )
        calls: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []
        triton_merge = nvfp4_adapter_module._triton_merge_nvfp4_lora
        assert triton_merge is not None

        def tracked_triton_merge(
            qdata: torch.Tensor,
            scale: torch.Tensor,
            per_tensor_scale: torch.Tensor | None,
            block_size: int,
            is_swizzled_scales: bool,
            b: torch.Tensor,
            a: torch.Tensor,
            strength: float,
            *,
            rounding_seed: int | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            calls.append((tuple(b.shape), tuple(a.shape), strength))
            return triton_merge(
                qdata,
                scale,
                per_tensor_scale,
                block_size,
                is_swizzled_scales,
                b,
                a,
                strength,
                rounding_seed=rounding_seed,
            )

        def fail_dequantize(_tensor: torch.Tensor) -> torch.Tensor:
            raise AssertionError("supported NVFP4 merge materialized the generic dense path")

        monkeypatch.setattr(
            nvfp4_adapter_module,
            "_triton_merge_nvfp4_lora",
            tracked_triton_merge,
        )
        monkeypatch.setattr(
            nvfp4_adapter_module,
            "dequantize_nvfp4_tensor",
            fail_dequantize,
        )
        LoRATransform(factors).apply(param)
        torch.cuda.synchronize()

        assert calls == [((rows, 8), (8, cols), 1.0)]
        merged = param.data
        merged_storage = (
            merged.qdata,
            merged.scale,
            merged.per_tensor_scale,
            merged.act_per_tensor_scale,
        )
        assert tuple(t.data_ptr() for t in merged_storage) == storage_ptrs
        assert torch.equal(merged.act_per_tensor_scale, activation_value)
        assert (
            merged.block_size,
            merged.orig_dtype,
            merged.is_swizzled_scales,
            merged.use_triton_kernel,
            merged.act_quant_kwargs,
        ) == metadata
        _assert_triton_nvfp4_close(merged, expected)

    @CUDA
    def test_fused_merge_uses_exact_fallback_when_triton_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows, cols, rank = 16, 64, 5
        nv = _make_nvfp4_cuda(
            torch.randn(
                rows,
                cols,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            swizzled=True,
            two_level=True,
        )
        param = nn.Parameter(nv, requires_grad=False)
        a = torch.randn(rank, cols)
        b = torch.randn(rows, rank)
        expected = _expected_nvfp4_merge(
            nv,
            b.to(device="cuda", dtype=nv.orig_dtype),
            a.to(device="cuda", dtype=nv.orig_dtype),
            0.5,
        )

        monkeypatch.setattr(
            nvfp4_adapter_module,
            "_triton_merge_nvfp4_lora",
            None,
        )
        LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.5)]).apply(param)

        assert torch.equal(param.data.qdata, expected.qdata)
        assert torch.equal(
            param.data.scale.view(torch.uint8),
            expected.scale.view(torch.uint8),
        )
        assert torch.equal(
            param.data.per_tensor_scale,
            expected.per_tensor_scale,
        )

    @CUDA
    def test_fused_merge_falls_back_for_transposed_storage(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        base = _make_nvfp4_cuda(
            torch.randn(16, 64, device="cuda", dtype=torch.bfloat16),
            swizzled=False,
            two_level=True,
        )
        transposed = base.t()
        rows, cols = transposed.shape
        assert not transposed.qdata.is_contiguous()

        def fail_triton_merge(*_args, **_kwargs):
            raise AssertionError("unsupported transposed storage reached Triton")

        monkeypatch.setattr(
            nvfp4_adapter_module,
            "_triton_merge_nvfp4_lora",
            fail_triton_merge,
        )
        with pytest.raises(ValueError, match="non-contiguous.*NVFP4"):
            Nvfp4Adapter.merge_lora_(
                transposed,
                torch.randn(
                    rows,
                    3,
                    device="cuda",
                    dtype=transposed.orig_dtype,
                ),
                torch.randn(
                    3,
                    cols,
                    device="cuda",
                    dtype=transposed.orig_dtype,
                ),
                0.5,
            )

    @CUDA
    @pytest.mark.parametrize("two_level", [False, True])
    def test_triton_merge_encodes_zero_weight_safely(
        self,
        two_level: bool,
    ) -> None:
        rows, cols, rank = 17, 48, 7
        nv = _make_nvfp4_cuda(
            torch.zeros(
                rows,
                cols,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            swizzled=True,
            two_level=two_level,
        )
        a = torch.zeros(rank, cols, device="cuda", dtype=nv.orig_dtype)
        b = torch.zeros(rows, rank, device="cuda", dtype=nv.orig_dtype)

        Nvfp4Adapter.merge_lora_(nv, b, a, 1.0)
        torch.cuda.synchronize()

        assert torch.isfinite(nv.scale.to(torch.float32)).all()
        logical_scales = nv.get_hp_scales()
        assert torch.count_nonzero(logical_scales).item() == logical_scales.numel()
        assert torch.count_nonzero(nv.dequantize()).item() == 0
        if nv.per_tensor_scale is not None:
            assert torch.equal(
                nv.per_tensor_scale,
                torch.full_like(
                    nv.per_tensor_scale,
                    torch.finfo(torch.float32).eps,
                ),
            )

    @CUDA
    def test_allocate_copy_make_gpu_param_preserves_wrapper(self) -> None:
        nvfp4_tensor_cls, _ = _nvfp4_modules()
        pinned_param = PinnedParam(
            nn.Parameter(_make_nvfp4(swizzled=True), requires_grad=False),
        )

        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()
        pinned = pinned_param.make_cpu_param().data

        assert isinstance(gpu_param.data, nvfp4_tensor_cls)
        assert gpu_param.data.qdata.is_cuda
        assert gpu_param.data.scale.is_cuda
        assert gpu_param.data.per_tensor_scale is not None
        assert gpu_param.data.per_tensor_scale.is_cuda
        assert gpu_param.data.block_size == pinned.block_size
        assert gpu_param.data.orig_dtype == pinned.orig_dtype
        assert gpu_param.data.is_swizzled_scales == pinned.is_swizzled_scales
        assert torch.equal(gpu_param.data.qdata.cpu(), pinned.qdata)
        assert torch.equal(gpu_param.data.scale.cpu(), pinned.scale)
        assert torch.equal(
            gpu_param.data.per_tensor_scale.cpu(),
            pinned.per_tensor_scale,
        )

    @CUDA
    def test_model_offloader_cuda_forward_dynamic_nvfp4(self) -> None:
        nvfp4_mod = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
        _, kwargs_cls = _nvfp4_modules()
        layer = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
        layer.weight.requires_grad = False
        weight = layer.weight.detach().contiguous()
        layer.weight = nn.Parameter(
            nvfp4_mod.NVFP4Tensor.to_nvfp4(
                weight,
                per_tensor_scale=nvfp4_mod.per_tensor_amax_to_scale(torch.max(torch.abs(weight))),
                is_swizzled_scales=True,
                use_triton_kernel=False,
                act_quant_kwargs=kwargs_cls(
                    is_swizzled_scales=True,
                    use_dynamic_per_tensor_scale=True,
                    use_triton_kernel=False,
                ),
            ),
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
    def test_streamed_nvfp4_merge_requantizes_on_activate(self) -> None:
        nvfp4_mod = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
        nvfp4_cls, kwargs_cls = _nvfp4_modules()

        def _quantize(weight: torch.Tensor) -> torch.Tensor:
            return nvfp4_cls.to_nvfp4(
                weight,
                per_tensor_scale=nvfp4_mod.per_tensor_amax_to_scale(torch.max(torch.abs(weight))),
                is_swizzled_scales=True,
                use_triton_kernel=False,
                act_quant_kwargs=kwargs_cls(
                    is_swizzled_scales=True,
                    use_dynamic_per_tensor_scale=True,
                    use_triton_kernel=False,
                ),
            )

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
                _quantize(block.weight.detach().contiguous()),
                requires_grad=False,
            )
        nv = model.blocks[0].weight.data
        rank = 4
        a = torch.randn(rank, 64)
        b = torch.randn(64, rank)
        lora = LoRA.from_state_dict(
            state_dict={
                "blocks.0.lora_A.weight": a,
                "blocks.0.lora_B.weight": b,
            }
        )
        # Reference on CUDA, matching the device the offloader merges on: the
        # offloader merges into a byte-identical GPU copy of the weight.
        nv_cuda = nv.cuda()
        expected_dense = Nvfp4Adapter.dequantize(nv_cuda)
        expected_dense.addmm_(
            b.cuda().to(expected_dense.dtype),
            a.cuda().to(expected_dense.dtype),
            alpha=0.5,
        )
        expected = Nvfp4Adapter.requantize(expected_dense, like=nv_cuda)

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
                stream_config=StreamConfig(num_resident_blocks=1, num_prefetch_blocks=0),
            ) as active:
                merged = active.blocks[0].weight.data
                assert isinstance(merged, nvfp4_cls)
                # Triton's tiled BF16 update can move a handful of values
                # across an FP4 rounding boundary relative to addmm.
                _assert_triton_nvfp4_close(merged, expected)
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (8, 64)
        finally:
            offloader.deactivate()

    @CUDA
    def test_model_offloader_routed_lora_on_dynamic_nvfp4(self) -> None:
        nvfp4_mod = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
        _, kwargs_cls = _nvfp4_modules()

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
                nvfp4_mod.NVFP4Tensor.to_nvfp4(
                    weight,
                    per_tensor_scale=nvfp4_mod.per_tensor_amax_to_scale(torch.max(torch.abs(weight))),
                    is_swizzled_scales=True,
                    use_triton_kernel=False,
                    act_quant_kwargs=kwargs_cls(
                        is_swizzled_scales=True,
                        use_dynamic_per_tensor_scale=True,
                        use_triton_kernel=False,
                    ),
                ),
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
