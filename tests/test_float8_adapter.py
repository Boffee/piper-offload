"""Tests for TorchAO scaled-fp8 (``Float8Tensor``) adapter integration."""

import pytest
import torch
from torch import nn

import piper_offload.float8_adapter as float8_adapter_module
from piper_offload import (
    Adapter,
    LoRATransform,
    ModelOffloader,
    ScaledLoRAFactor,
    merge_adapter,
)
from piper_offload.float8_adapter import Float8Adapter
from piper_offload.host_param import HostParam
from piper_offload.block_component import _param_target_layout
from piper_offload.tensor_adapter_registry import tensor_id
from tests.conftest import activated_model

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_model_offloader(
    model: nn.Module,
    *,
    block_paths: list[str] = [],
    include_block_trainables: bool = False,
) -> ModelOffloader:
    return ModelOffloader.from_module(
        model,
        block_paths=block_paths,
        include_block_trainables=include_block_trainables,
    )


def _float8_modules():
    pytest.importorskip("torchao")
    try:
        from torchao.quantization.granularity import PerGroup, PerRow, PerTensor
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            Float8Tensor,
            QuantizeTensorToFloat8Kwargs,
        )
    except ImportError as exc:
        # The float8 adapter targets the torchao>=0.18 Float8Tensor workflow;
        # skip (don't error) when the installed torchao predates it — or a
        # future release moves it — matching the importorskip above.
        pytest.skip(f"torchao float8 API unavailable: {exc}")

    return Float8Tensor, QuantizeTensorToFloat8Kwargs, PerGroup, PerRow, PerTensor


def _mm_config() -> object:
    # The quantize_(...) workflow always sets mm_config on weights; the
    # scaled-mm forward path asserts it is present. Match that here.
    from torchao.float8.inference import Float8MMConfig

    return Float8MMConfig(use_fast_accum=True)


def _make_float8(
    *,
    rows: int = 16,
    cols: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    float8_dtype: torch.dtype = torch.float8_e4m3fn,
    per_tensor: bool = False,
    group_size: int | None = None,
    dynamic_activation: bool = True,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    float8_tensor_cls, kwargs_cls, per_group_cls, per_row_cls, per_tensor_cls = _float8_modules()
    if per_tensor and group_size is not None:
        raise ValueError("per_tensor and group_size are mutually exclusive")
    if group_size is not None:
        granularity = per_group_cls(group_size)
    else:
        granularity = per_tensor_cls() if per_tensor else per_row_cls()
    act_quant_kwargs = kwargs_cls(granularity=granularity) if dynamic_activation else None
    if weight is None:
        weight = torch.randn(rows, cols, dtype=dtype)
    return float8_tensor_cls.from_hp(
        weight,
        float8_dtype=float8_dtype,
        granularity=granularity,
        mm_config=_mm_config(),
        act_quant_kwargs=act_quant_kwargs,
    )


def _assert_float8_merge_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    float8_dtype: torch.dtype,
) -> None:
    torch.testing.assert_close(
        actual.scale,
        expected.scale,
        rtol=0.02,
        atol=0,
    )
    torch.testing.assert_close(
        actual.dequantize().to(torch.float32),
        expected.dequantize().to(torch.float32),
        rtol=0.3 if float8_dtype is torch.float8_e5m2 else 0.13,
        atol=0.15 if float8_dtype is torch.float8_e5m2 else 0.05,
    )


class TestFloat8Adapter:
    def test_matches_float8_only(self) -> None:
        f8 = _make_float8()
        assert Float8Adapter.matches(f8)
        assert not Float8Adapter.matches(torch.zeros(16, 16, dtype=torch.bfloat16))

    def test_capture_preserves_storage_and_metadata(self) -> None:
        float8_tensor_cls, _, _, _, _ = _float8_modules()
        f8 = _make_float8()
        p = nn.Parameter(f8, requires_grad=False)
        host_param = HostParam(p)

        host = host_param.make_cpu_param().data
        assert isinstance(host, float8_tensor_cls)
        assert not host.qdata.is_pinned()
        assert not host.scale.is_pinned()
        assert host.qdata.data_ptr() == host_param.host_state.storage[0].data_ptr()
        assert host.scale.data_ptr() == host_param.host_state.storage[1].data_ptr()
        assert host.block_size == f8.block_size
        assert host.mm_config == f8.mm_config
        assert host.kernel_preference == f8.kernel_preference
        assert host.act_quant_kwargs == f8.act_quant_kwargs
        assert host.dtype == f8.dtype
        assert host_param.compute_dtype is torch.bfloat16
        assert torch.equal(host.dequantize(), f8.dequantize())

    def test_tensor_id_tracks_both_buffers(self) -> None:
        f8 = _make_float8()
        key = tensor_id(f8)
        assert key[0] == "torchao-float8"
        assert key[1][0] == f8.qdata.device
        assert key[2][0] == f8.scale.device
        assert key == tensor_id(f8)
        assert key != tensor_id(_make_float8())

    def test_target_layout_ignores_tensor_id(self) -> None:
        p1 = nn.Parameter(_make_float8(), requires_grad=False)
        p2 = nn.Parameter(_make_float8(), requires_grad=False)

        assert _param_target_layout(p1) == _param_target_layout(p2)

    def test_target_layout_tracks_granularity(self) -> None:
        per_row = nn.Parameter(_make_float8(per_tensor=False), requires_grad=False)
        per_tensor = nn.Parameter(_make_float8(per_tensor=True), requires_grad=False)

        assert _param_target_layout(per_row) != _param_target_layout(per_tensor)

    def test_target_layout_tracks_activation_quantization(self) -> None:
        with_activation = nn.Parameter(_make_float8(dynamic_activation=True), requires_grad=False)
        weight_only = nn.Parameter(_make_float8(dynamic_activation=False), requires_grad=False)

        assert _param_target_layout(with_activation) != _param_target_layout(weight_only)

    def test_cpu_round_trip_restores_host_bytes(self) -> None:
        host_param = HostParam(
            nn.Parameter(_make_float8(), requires_grad=False),
        )
        state = host_param.allocate_gpu_storage(torch.device("cpu"))
        host_param.copy_to_gpu(state)

        original_qdata = host_param.host_state.storage[0].view(torch.uint8).clone()
        original_scale = host_param.host_state.storage[1].clone()
        host_param.host_state.storage[0].view(torch.uint8).zero_()
        host_param.host_state.storage[1].zero_()
        host_param.copy_to_cpu(state)

        assert torch.equal(host_param.host_state.storage[0].view(torch.uint8), original_qdata)
        assert torch.equal(host_param.host_state.storage[1], original_scale)

    def test_no_trainable_swap_capability(self) -> None:
        host_param = HostParam(
            nn.Parameter(_make_float8(), requires_grad=True),
        )

        with pytest.raises(NotImplementedError, match="Parameter.data-swap"):
            host_param.validate_parameter_data_swap_target()

    @pytest.mark.parametrize("per_tensor", [False, True])
    def test_dequantize_requantize_preserves_representation(self, per_tensor: bool) -> None:
        f8 = _make_float8(per_tensor=per_tensor)
        dense = Float8Adapter.dequantize(f8)
        assert dense.dtype is f8.dtype
        torch.testing.assert_close(dense, f8.dequantize())

        again = Float8Adapter.requantize(dense, like=f8)
        assert again.block_size == f8.block_size
        assert again.qdata.dtype == f8.qdata.dtype
        assert again.dtype == f8.dtype
        assert again.kernel_preference == f8.kernel_preference
        assert again.mm_config == f8.mm_config
        assert again.act_quant_kwargs == f8.act_quant_kwargs
        assert torch.equal(again.qdata.view(torch.uint8), f8.qdata.view(torch.uint8))
        assert torch.equal(again.scale, f8.scale)

    def test_requantize_rejects_shape_mismatch(self) -> None:
        f8 = _make_float8(rows=4, cols=8)
        with pytest.raises(ValueError, match="Cannot requantize"):
            Float8Adapter.requantize(torch.randn(8, 4), like=f8)

    def test_requantize_zero_row_does_not_nan(self) -> None:
        # torchao's from_hp computes scale = amax / fp8_max with no eps
        # floor, so an all-zero row (per-row scaling) gets scale 0 and
        # qdata 0/0 = NaN. requantize must repair that to an exact zero row
        # while leaving the other rows intact.
        f8 = _make_float8(rows=16, cols=64, dynamic_activation=False)
        dense = Float8Adapter.dequantize(f8)
        dense[3] = 0  # one fully cancelled row

        again = Float8Adapter.requantize(dense, like=f8)
        deq = again.dequantize().to(torch.float32)
        assert not torch.isnan(deq).any()
        assert torch.count_nonzero(deq[3]).item() == 0
        assert torch.count_nonzero(deq[[0, 1, 2, 4]]).item() > 0

    @pytest.mark.parametrize(
        "float8_dtype",
        [torch.float8_e4m3fn, torch.float8_e5m2],
    )
    def test_requantize_zero_group_does_not_nan(
        self,
        float8_dtype: torch.dtype,
    ) -> None:
        rows, cols, group_size = 7, 32, 8
        f8 = _make_float8(
            rows=rows,
            cols=cols,
            dtype=torch.float32,
            float8_dtype=float8_dtype,
            group_size=group_size,
            dynamic_activation=False,
        )
        dense = Float8Adapter.dequantize(f8)
        dense[3, 16:24] = 0

        again = Float8Adapter.requantize(dense, like=f8)

        dequantized = again.dequantize().to(torch.float32)
        assert torch.isfinite(dequantized).all()
        assert torch.count_nonzero(dequantized[3, 16:24]).item() == 0
        assert again.scale[3, 2].item() == torch.finfo(torch.float32).eps
        assert torch.count_nonzero(dequantized[3, :16]).item() > 0
        assert torch.count_nonzero(dequantized[3, 24:]).item() > 0

    def test_requantize_all_zero_does_not_nan(self) -> None:
        # Per-tensor scaling: a fully cancelled weight gives a scalar scale
        # of 0; the repair must still yield a clean all-zero tensor.
        f8 = _make_float8(rows=16, cols=16, per_tensor=True)
        again = Float8Adapter.requantize(torch.zeros(16, 16, dtype=torch.float32), like=f8)
        deq = again.dequantize().to(torch.float32)
        assert not torch.isnan(deq).any()
        assert torch.count_nonzero(deq).item() == 0

    @CUDA
    @pytest.mark.parametrize("per_tensor", [False, True])
    @pytest.mark.parametrize(
        ("dtype", "float8_dtype"),
        [
            (torch.bfloat16, torch.float8_e4m3fn),
            (torch.bfloat16, torch.float8_e5m2),
            (torch.float16, torch.float8_e4m3fn),
            (torch.float16, torch.float8_e5m2),
            (torch.float32, torch.float8_e4m3fn),
            (torch.float32, torch.float8_e5m2),
        ],
    )
    def test_triton_lora_merge_matches_eager_round_trip(
        self,
        monkeypatch: pytest.MonkeyPatch,
        per_tensor: bool,
        dtype: torch.dtype,
        float8_dtype: torch.dtype,
    ) -> None:
        rows, cols, rank = 70, 130, 7
        f8 = _make_float8(
            rows=rows,
            cols=cols,
            dtype=dtype,
            float8_dtype=float8_dtype,
            per_tensor=per_tensor,
            dynamic_activation=False,
        ).cuda()
        a = torch.randn(cols, rank, device="cuda", dtype=dtype).t()
        b = torch.randn(rank, rows, device="cuda", dtype=dtype).t()
        strength = 0.25
        dense = Float8Adapter.dequantize(f8)
        dense.addmm_(b, a, alpha=strength)
        expected = Float8Adapter.requantize(dense, like=f8)
        qdata_ptr = f8.qdata.data_ptr()
        scale_ptr = f8.scale.data_ptr()
        original_block_size = f8.block_size
        original_mm_config = f8.mm_config
        original_kernel_preference = f8.kernel_preference
        original_act_quant_kwargs = f8.act_quant_kwargs

        assert not a.is_contiguous()
        assert not b.is_contiguous()

        def fail_dequantize(_tensor: torch.Tensor) -> torch.Tensor:
            raise AssertionError("supported CUDA layouts must use raw Triton")

        monkeypatch.setattr(
            "piper_offload.float8_adapter.dequantize_float8_tensor",
            fail_dequantize,
        )
        result = Float8Adapter.merge_lora_(
            f8,
            b,
            a,
            strength,
        )
        torch.cuda.synchronize()

        assert result is None
        assert f8.qdata.data_ptr() == qdata_ptr
        assert f8.scale.data_ptr() == scale_ptr
        assert f8.block_size is original_block_size
        assert f8.mm_config is original_mm_config
        assert f8.kernel_preference is original_kernel_preference
        assert f8.act_quant_kwargs is original_act_quant_kwargs
        _assert_float8_merge_close(f8, expected, float8_dtype)

    @CUDA
    @pytest.mark.parametrize("layout", ["row", "tensor", "group"])
    def test_lora_transform_falls_back_when_triton_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        layout: str,
    ) -> None:
        rows, cols, rank = 16, 32, 4
        f8 = _make_float8(
            rows=rows,
            cols=cols,
            per_tensor=layout == "tensor",
            group_size=8 if layout == "group" else None,
            dynamic_activation=False,
        ).cuda()
        param = nn.Parameter(f8, requires_grad=False)
        a = torch.randn(rank, cols, dtype=f8.dtype)
        b = torch.randn(rows, rank, dtype=f8.dtype)
        expected_dense = Float8Adapter.dequantize(f8)
        expected_dense.addmm_(
            b.cuda(),
            a.cuda(),
            alpha=0.5,
        )
        expected = Float8Adapter.requantize(expected_dense, like=f8)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.5)])

        monkeypatch.setattr(
            "piper_offload.float8_adapter._triton_merge_float8_lora",
            None,
        )
        transform.validate_target(param)
        transform.apply(param)

        assert torch.equal(
            param.data.qdata.view(torch.uint8),
            expected.qdata.view(torch.uint8),
        )
        assert torch.equal(param.data.scale, expected.scale)

    @CUDA
    @pytest.mark.parametrize(
        ("dtype", "float8_dtype"),
        [
            (torch.bfloat16, torch.float8_e4m3fn),
            (torch.bfloat16, torch.float8_e5m2),
            (torch.float16, torch.float8_e4m3fn),
            (torch.float16, torch.float8_e5m2),
            (torch.float32, torch.float8_e4m3fn),
            (torch.float32, torch.float8_e5m2),
        ],
    )
    def test_triton_group_lora_merge_matches_eager_round_trip(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dtype: torch.dtype,
        float8_dtype: torch.dtype,
    ) -> None:
        rows, cols, rank = 19, 40, 7
        f8 = _make_float8(
            rows=rows,
            cols=cols,
            dtype=dtype,
            float8_dtype=float8_dtype,
            group_size=8,
            dynamic_activation=False,
        ).cuda()
        a = torch.randn(cols, rank, device="cuda", dtype=dtype).t()
        b = torch.randn(rank, rows, device="cuda", dtype=dtype).t()
        strength = -0.3125
        dense = Float8Adapter.dequantize(f8)
        dense.addmm_(b, a, alpha=strength)
        expected = Float8Adapter.requantize(dense, like=f8)
        qdata_ptr = f8.qdata.data_ptr()
        scale_ptr = f8.scale.data_ptr()
        metadata = (
            f8.block_size,
            f8.mm_config,
            f8.kernel_preference,
            f8.act_quant_kwargs,
        )

        def fail_dequantize(_tensor: torch.Tensor) -> torch.Tensor:
            raise AssertionError("standard PerGroup layouts must use raw Triton")

        monkeypatch.setattr(
            float8_adapter_module,
            "dequantize_float8_tensor",
            fail_dequantize,
        )
        result = Float8Adapter.merge_lora_(f8, b, a, strength)
        torch.cuda.synchronize()

        assert result is None
        assert f8.qdata.data_ptr() == qdata_ptr
        assert f8.scale.data_ptr() == scale_ptr
        assert (
            f8.block_size,
            f8.mm_config,
            f8.kernel_preference,
            f8.act_quant_kwargs,
        ) == metadata
        _assert_float8_merge_close(f8, expected, float8_dtype)

    @CUDA
    @pytest.mark.parametrize(
        ("group_size", "cols"),
        [
            (2, 34),
            (3, 39),
            (16, 48),
            (96, 192),
            (256, 512),
        ],
    )
    def test_triton_group_lora_merge_supports_valid_group_sizes(
        self,
        group_size: int,
        cols: int,
    ) -> None:
        rows, rank = 7, 5
        f8 = _make_float8(
            rows=rows,
            cols=cols,
            group_size=group_size,
            dynamic_activation=False,
        ).cuda()
        a = torch.randn(rank, cols, device="cuda", dtype=f8.dtype)
        b = torch.randn(rows, rank, device="cuda", dtype=f8.dtype)
        expected_dense = Float8Adapter.dequantize(f8)
        expected_dense.addmm_(b, a, alpha=0.1875)
        expected = Float8Adapter.requantize(expected_dense, like=f8)

        Float8Adapter.merge_lora_(f8, b, a, 0.1875)
        torch.cuda.synchronize()

        assert f8.block_size == [1, group_size]
        assert tuple(f8.scale.shape) == (rows, cols // group_size)
        _assert_float8_merge_close(f8, expected, f8.qdata.dtype)

    @CUDA
    def test_triton_group_transform_packs_loras_and_preserves_storage(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows, cols = 21, 40
        dtype = torch.bfloat16
        f8 = _make_float8(
            rows=rows,
            cols=cols,
            dtype=dtype,
            group_size=8,
            dynamic_activation=True,
        )
        f8 = f8.cuda()
        param = nn.Parameter(f8, requires_grad=False)
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
        dense = Float8Adapter.dequantize(f8)
        dense.addmm_(packed_b, packed_a)
        expected = Float8Adapter.requantize(dense, like=f8)
        qdata_ptr = f8.qdata.data_ptr()
        scale_ptr = f8.scale.data_ptr()
        metadata = (
            f8.block_size,
            f8.mm_config,
            f8.kernel_preference,
            f8.act_quant_kwargs,
        )
        calls: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []
        triton_merge = float8_adapter_module._triton_merge_float8_lora
        assert triton_merge is not None

        def tracked_triton_merge(
            qdata: torch.Tensor,
            scale: torch.Tensor,
            block_size: tuple[int, ...],
            b: torch.Tensor,
            a: torch.Tensor,
            strength: float,
            *,
            rounding_seed: int | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append((tuple(b.shape), tuple(a.shape), strength))
            return triton_merge(
                qdata,
                scale,
                block_size,
                b,
                a,
                strength,
                rounding_seed=rounding_seed,
            )

        def fail_dequantize(_tensor: torch.Tensor) -> torch.Tensor:
            raise AssertionError("packed PerGroup merge materialized the generic dense path")

        monkeypatch.setattr(
            float8_adapter_module,
            "_triton_merge_float8_lora",
            tracked_triton_merge,
        )
        monkeypatch.setattr(
            float8_adapter_module,
            "dequantize_float8_tensor",
            fail_dequantize,
        )
        transform = LoRATransform(factors)
        transform.validate_target(param)
        transform.apply(param)
        torch.cuda.synchronize()

        assert calls == [((rows, 8), (8, cols), 1.0)]
        assert param.data.qdata.data_ptr() == qdata_ptr
        assert param.data.scale.data_ptr() == scale_ptr
        assert (
            param.data.block_size,
            param.data.mm_config,
            param.data.kernel_preference,
            param.data.act_quant_kwargs,
        ) == metadata
        _assert_float8_merge_close(
            param.data,
            expected,
            param.data.qdata.dtype,
        )

    @CUDA
    def test_lora_merge_falls_back_for_oversized_group_layout(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows, cols, rank = 7, 1024, 4
        f8 = _make_float8(
            rows=rows,
            cols=cols,
            group_size=512,
            dynamic_activation=False,
        ).cuda()
        a = torch.randn(rank, cols, device="cuda", dtype=f8.dtype)
        b = torch.randn(rows, rank, device="cuda", dtype=f8.dtype)
        expected_dense = Float8Adapter.dequantize(f8)
        expected_dense.addmm_(b, a, alpha=0.5)
        expected = Float8Adapter.requantize(expected_dense, like=f8)

        def fail_triton(*_args: object) -> tuple[torch.Tensor, torch.Tensor]:
            raise AssertionError("oversized groups must use generic fallback")

        monkeypatch.setattr(
            float8_adapter_module,
            "_triton_merge_float8_lora",
            fail_triton,
        )
        Float8Adapter.merge_lora_(f8, b, a, 0.5)

        assert torch.equal(
            f8.qdata.view(torch.uint8),
            expected.qdata.view(torch.uint8),
        )
        assert torch.equal(f8.scale, expected.scale)

    @CUDA
    @pytest.mark.parametrize(
        "float8_dtype",
        [torch.float8_e4m3fn, torch.float8_e5m2],
    )
    def test_group_lora_fallback_repairs_cancelled_block(
        self,
        monkeypatch: pytest.MonkeyPatch,
        float8_dtype: torch.dtype,
    ) -> None:
        rows, cols, group_size = 7, 32, 8
        f8 = _make_float8(
            rows=rows,
            cols=cols,
            dtype=torch.float32,
            float8_dtype=float8_dtype,
            group_size=group_size,
            dynamic_activation=False,
        ).cuda()
        dense = Float8Adapter.dequantize(f8)
        a = torch.zeros(1, cols, device="cuda", dtype=f8.dtype)
        b = torch.zeros(rows, 1, device="cuda", dtype=f8.dtype)
        a[0, 16:24] = -dense[3, 16:24]
        b[3, 0] = 1
        expected_dense = dense.clone()
        expected_dense.addmm_(b, a)
        assert torch.count_nonzero(expected_dense[3, 16:24]).item() == 0
        expected = Float8Adapter.requantize(expected_dense, like=f8)
        fallback_calls = 0
        original_fallback = float8_adapter_module._torch_merge_float8_lora_

        def tracked_fallback(
            target: torch.Tensor,
            staged_b: torch.Tensor,
            staged_a: torch.Tensor,
            strength: float,
            *,
            rounding_seed: int | None = None,
        ) -> None:
            nonlocal fallback_calls
            fallback_calls += 1
            original_fallback(
                target,
                staged_b,
                staged_a,
                strength,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(
            float8_adapter_module,
            "_triton_merge_float8_lora",
            None,
        )
        monkeypatch.setattr(
            float8_adapter_module,
            "_torch_merge_float8_lora_",
            tracked_fallback,
        )

        Float8Adapter.merge_lora_(f8, b, a, 1.0)

        assert fallback_calls == 1
        assert torch.equal(
            f8.qdata.view(torch.uint8),
            expected.qdata.view(torch.uint8),
        )
        assert torch.equal(f8.scale, expected.scale)
        assert torch.isfinite(f8.dequantize()).all()
        assert torch.count_nonzero(f8.dequantize()[3, 16:24]).item() == 0

    @CUDA
    def test_lora_merge_rejects_transposed_group_layout_clearly(self) -> None:
        rows, cols, rank = 16, 32, 4
        f8 = (
            _make_float8(
                rows=rows,
                cols=cols,
                group_size=4,
                dynamic_activation=False,
            )
            .cuda()
            .t()
        )
        assert tuple(f8.block_size) == (4, 1)
        assert not f8.qdata.is_contiguous()
        a = torch.randn(rank, rows, device="cuda", dtype=f8.dtype)
        b = torch.randn(cols, rank, device="cuda", dtype=f8.dtype)

        with pytest.raises(
            ValueError,
            match="transposed PerGroup.*routed LoRA",
        ):
            Float8Adapter.merge_lora_(f8, b, a, 0.5)

    @CUDA
    @pytest.mark.parametrize("per_tensor", [False, True])
    def test_triton_lora_merge_repairs_zero_scaling_block(
        self,
        per_tensor: bool,
    ) -> None:
        rows, cols = 16, 32
        f8 = _make_float8(
            rows=rows,
            cols=cols,
            per_tensor=per_tensor,
            dynamic_activation=False,
        ).cuda()
        if per_tensor:
            f8.qdata.zero_()
            f8.scale.fill_(torch.finfo(torch.float32).eps)
        else:
            f8.qdata[3].zero_()
            f8.scale[3].fill_(torch.finfo(torch.float32).eps)
        a = torch.zeros(4, cols, device="cuda", dtype=f8.dtype)
        b = torch.zeros(rows, 4, device="cuda", dtype=f8.dtype)

        Float8Adapter.merge_lora_(f8, b, a, 1.0)
        torch.cuda.synchronize()

        dequantized = f8.dequantize().to(torch.float32)
        assert torch.isfinite(dequantized).all()
        if per_tensor:
            assert torch.count_nonzero(dequantized).item() == 0
            assert torch.count_nonzero(f8.scale).item() == 1
        else:
            assert torch.count_nonzero(dequantized[3]).item() == 0
            assert torch.count_nonzero(f8.scale[3]).item() == 1
            assert torch.count_nonzero(dequantized[[0, 1, 2, 4]]).item() > 0

    @CUDA
    @pytest.mark.parametrize(
        "float8_dtype",
        [torch.float8_e4m3fn, torch.float8_e5m2],
    )
    def test_triton_group_lora_merge_repairs_zero_scaling_blocks(
        self,
        float8_dtype: torch.dtype,
    ) -> None:
        rows, cols, rank = 17, 40, 7
        f8 = _make_float8(
            rows=rows,
            cols=cols,
            float8_dtype=float8_dtype,
            group_size=8,
            dynamic_activation=False,
        ).cuda()
        f8.qdata[3, 16:24].zero_()
        f8.scale[3, 2].fill_(torch.finfo(torch.float32).eps)
        expected = Float8Adapter.requantize(
            Float8Adapter.dequantize(f8),
            like=f8,
        )
        a = torch.zeros(rank, cols, device="cuda", dtype=f8.dtype)
        b = torch.zeros(rows, rank, device="cuda", dtype=f8.dtype)

        Float8Adapter.merge_lora_(f8, b, a, 1.0)
        torch.cuda.synchronize()

        dequantized = f8.dequantize().to(torch.float32)
        assert torch.isfinite(dequantized).all()
        assert torch.count_nonzero(dequantized[3, 16:24]).item() == 0
        assert f8.scale[3, 2].item() == torch.finfo(torch.float32).eps
        assert torch.count_nonzero(dequantized[3, :16]).item() > 0
        assert torch.equal(
            f8.qdata.view(torch.uint8),
            expected.qdata.view(torch.uint8),
        )
        torch.testing.assert_close(f8.scale, expected.scale, rtol=0.02, atol=0)

    def test_lora_transform_requantizes_param_in_place(self) -> None:
        float8_tensor_cls, _, _, per_row_cls, _ = _float8_modules()
        rows, cols, rank = 4, 8, 2
        f8 = _make_float8(rows=rows, cols=cols, dynamic_activation=False)
        param = nn.Parameter(f8, requires_grad=False)
        a = torch.randn(rank, cols)
        b = torch.randn(rows, rank)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.5)])
        original_param = param
        original_qdata_ptr = param.data.qdata.data_ptr()

        expected_dense = Float8Adapter.dequantize(f8)
        expected_dense.addmm_(
            b.to(expected_dense.dtype),
            a.to(expected_dense.dtype),
            alpha=0.5,
        )
        expected = float8_tensor_cls.from_hp(
            expected_dense.to(f8.dtype),
            granularity=per_row_cls(),
        )

        transform.validate_target(param)
        transform.apply(param)

        assert param is original_param
        assert param.data.qdata.data_ptr() == original_qdata_ptr
        assert isinstance(param.data, float8_tensor_cls)
        assert torch.equal(
            param.data.qdata.view(torch.uint8),
            expected.qdata.view(torch.uint8),
        )
        assert torch.equal(param.data.scale, expected.scale)

    def test_merge_lora_merges_float8_weight(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)

        model = M()
        model.lin.weight.requires_grad = False
        model.lin.weight = nn.Parameter(_make_float8(dynamic_activation=False), requires_grad=False)
        # copy_into mutates the weight's storage in place, so snapshot
        # the original packed bytes rather than holding a tensor ref.
        original_qdata = model.lin.weight.data.qdata.view(torch.uint8).clone()
        lora = Adapter.from_state_dict(
            state_dict={
                "lin.lora_A.weight": torch.randn(4, 16),
                "lin.lora_B.weight": torch.randn(16, 4),
            }
        )

        merged = merge_adapter(model, [(lora, 1.0)])

        assert merged == 1
        assert not torch.equal(model.lin.weight.data.qdata.view(torch.uint8), original_qdata)

    def test_merge_lora_preflights_transposed_group_before_mutation(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.first = nn.Linear(16, 16, bias=False, dtype=torch.float32)
                self.second = nn.Linear(16, 16, bias=False, dtype=torch.float32)

        model = M()
        model.first.weight = nn.Parameter(
            _make_float8(
                dtype=torch.float32,
                dynamic_activation=False,
            ),
            requires_grad=False,
        )
        model.second.weight = nn.Parameter(
            _make_float8(
                dtype=torch.float32,
                group_size=4,
                dynamic_activation=False,
            ).t(),
            requires_grad=False,
        )
        first_qdata = model.first.weight.data.qdata.view(torch.uint8).clone()
        first_scale = model.first.weight.data.scale.clone()
        second_qdata = model.second.weight.data.qdata.view(torch.uint8).clone()
        second_scale = model.second.weight.data.scale.clone()
        lora = Adapter.from_state_dict(
            state_dict={
                "first.lora_A.weight": torch.randn(4, 16),
                "first.lora_B.weight": torch.randn(16, 4),
                "second.lora_A.weight": torch.randn(4, 16),
                "second.lora_B.weight": torch.randn(16, 4),
            }
        )

        with pytest.raises(
            ValueError,
            match="transposed PerGroup.*routed LoRA",
        ):
            merge_adapter(model, [(lora, 1.0)])

        assert torch.equal(
            model.first.weight.data.qdata.view(torch.uint8),
            first_qdata,
        )
        assert torch.equal(model.first.weight.data.scale, first_scale)
        assert torch.equal(
            model.second.weight.data.qdata.view(torch.uint8),
            second_qdata,
        )
        assert torch.equal(model.second.weight.data.scale, second_scale)

    @CUDA
    def test_allocate_copy_make_gpu_param_preserves_wrapper(self) -> None:
        float8_tensor_cls, _, _, _, _ = _float8_modules()
        host_param = HostParam(
            nn.Parameter(_make_float8(), requires_grad=False),
        )

        gpu_state = host_param.allocate_gpu_storage(torch.device("cuda"))
        host_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = host_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()
        host = host_param.make_cpu_param().data

        assert isinstance(gpu_param.data, float8_tensor_cls)
        assert gpu_param.data.qdata.is_cuda
        assert gpu_param.data.scale.is_cuda
        assert gpu_param.data.block_size == host.block_size
        assert gpu_param.data.kernel_preference == host.kernel_preference
        assert gpu_param.data.act_quant_kwargs == host.act_quant_kwargs
        assert gpu_param.data.dtype == host.dtype
        assert torch.equal(
            gpu_param.data.qdata.view(torch.uint8).cpu(),
            host.qdata.view(torch.uint8),
        )
        assert torch.equal(gpu_param.data.scale.cpu(), host.scale)

    @CUDA
    def test_model_offloader_cuda_forward_dynamic_float8(self) -> None:
        layer = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
        layer.weight.requires_grad = False
        weight = layer.weight.detach().contiguous()
        layer.weight = nn.Parameter(
            _make_float8(weight=weight, dynamic_activation=True),
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
    def test_streamed_float8_merge_requantizes_on_activate(self) -> None:
        float8_tensor_cls, _, _, per_row_cls, _ = _float8_modules()

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.Linear(16, 16, bias=False, dtype=torch.bfloat16),
                        nn.Linear(16, 16, bias=False, dtype=torch.bfloat16),
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
                _make_float8(dynamic_activation=False),
                requires_grad=False,
            )
        f8 = model.blocks[0].weight.data
        rank = 4
        a = torch.randn(rank, 16)
        b = torch.randn(16, rank)
        lora = Adapter.from_state_dict(
            state_dict={
                "blocks.0.lora_A.weight": a,
                "blocks.0.lora_B.weight": b,
            }
        )
        # Compute the reference on CUDA, matching the device the offloader
        # merges on. A CPU reference flips a couple of float8 boundary elements
        # relative to the CUDA merge (CPU vs CUDA round-to-nearest at bucket
        # edges), making the tight tolerance RNG/CUDA-state sensitive.
        f8_cuda = f8.cuda()
        expected_dense = Float8Adapter.dequantize(f8_cuda)
        expected_dense.addmm_(
            b.cuda().to(expected_dense.dtype),
            a.cuda().to(expected_dense.dtype),
            alpha=0.5,
        )
        expected = float8_tensor_cls.from_hp(
            expected_dense.to(f8.dtype),
            granularity=per_row_cls(),
        )

        offloader = _make_model_offloader(
            model,
            block_paths=["blocks"],
        )
        try:
            x = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
            with activated_model(
                offloader,
                "cuda",
                adapters=[lora],
                adapter_strengths=[0.5],
                adapter_mode="merge",
            ) as active:
                merged = active.blocks[0].weight.data
                assert isinstance(merged, float8_tensor_cls)
                _assert_float8_merge_close(
                    merged,
                    expected,
                    merged.qdata.dtype,
                )
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (8, 16)
        finally:
            offloader.deactivate()
