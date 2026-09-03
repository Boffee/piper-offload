"""Tests for Piper ``ConvRotNVFP4Tensor`` offload support."""

from typing import Any

import pytest
import torch
from torch import nn

from piper_offload import (
    Adapter,
    LoRATransform,
    ModelOffloader,
    ScaledLoRAFactor,
    merge_adapter,
)
from piper_offload.block_component import _param_target_layout
from piper_offload.nvfp4_adapter import Nvfp4Adapter
from piper_offload.pinned_param import PinnedParam
from piper_offload.piper_convrot_nvfp4_adapter import (
    PiperConvRotNVFP4Adapter,
)
from piper_offload.rolling_runtime import _ROLLING_ADAPTER_TYPES
from piper_offload.tensor_adapter_registry import select_adapter, tensor_id
from piper_offload.tensor_adapters import (
    CpuRoundTripTensorAdapter,
    DequantRequantTensorAdapter,
    DenseMergeTargetValidationTensorAdapter,
    DenseMergeTensorAdapter,
    LoRAMergeTensorAdapter,
    LoRAMergeValidationTensorAdapter,
    ParameterDataSwapTensorAdapter,
    TensorCopyIntoAdapter,
)
from tests.conftest import activated_model


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


SM120 = pytest.mark.skipif(
    not _exact_sm120_available(),
    reason="exact SM120 required",
)


def _modules() -> tuple[Any, Any, Any, Any, Any]:
    nvfp4 = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    convrot = pytest.importorskip("piper_kernels.linear.convrot.nvfp4")
    rotation = pytest.importorskip("piper_kernels.linear.convrot._rotation")
    return (
        nvfp4.NVFP4Tensor,
        nvfp4.QuantizeTensorToNVFP4Kwargs,
        nvfp4.per_tensor_amax_to_scale,
        convrot.ConvRotNVFP4Tensor,
        rotation.rotate_groups,
    )


def _make_convrot_nvfp4(
    *,
    rows: int = 16,
    cols: int = 64,
    group_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    dense: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    nvfp4_cls, kwargs_cls, amax_to_scale, convrot_cls, rotate_groups = _modules()
    if dense is None:
        dense = torch.randn(rows, cols, device=device, dtype=dtype)
    else:
        dense = dense.to(device=device, dtype=dtype)
    rotated = rotate_groups(dense, group_size)
    global_scale = amax_to_scale(rotated.abs().amax().to(torch.float32)).clamp_min(torch.finfo(torch.float32).eps)
    packed = nvfp4_cls.to_nvfp4(
        rotated,
        per_tensor_scale=global_scale,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        act_quant_kwargs=kwargs_cls(
            block_size=16,
            is_swizzled_scales=True,
            use_triton_kernel=False,
            use_dynamic_per_tensor_scale=True,
        ),
    )
    return convrot_cls.from_torchao(packed, group_size=group_size), dense


class TestPiperConvRotNVFP4Adapter:
    def test_dedicated_adapter_precedes_generic_nvfp4(self) -> None:
        convrot, _dense = _make_convrot_nvfp4()

        assert PiperConvRotNVFP4Adapter.matches(convrot)
        assert Nvfp4Adapter.matches(convrot)
        assert isinstance(select_adapter(convrot), PiperConvRotNVFP4Adapter)

    def test_pin_preserves_storage_and_all_metadata(self) -> None:
        convrot, _dense = _make_convrot_nvfp4(group_size=64)
        pinned_param = PinnedParam(nn.Parameter(convrot, requires_grad=False))
        pinned = pinned_param.make_cpu_param().data

        assert type(pinned) is type(convrot)
        assert pinned.qdata.is_pinned()
        assert pinned.scale.is_pinned()
        assert pinned.per_tensor_scale is not None
        assert pinned.per_tensor_scale.is_pinned()
        assert pinned.group_size == 64
        assert pinned.block_size == convrot.block_size
        assert pinned.orig_dtype is convrot.orig_dtype
        assert pinned.is_swizzled_scales is convrot.is_swizzled_scales
        assert pinned.use_triton_kernel is convrot.use_triton_kernel
        assert pinned.act_quant_kwargs == convrot.act_quant_kwargs
        assert pinned_param.compute_dtype is convrot.orig_dtype
        assert pinned_param.cache_bytes == sum(
            storage.nbytes
            for storage in (
                convrot.qdata,
                convrot.scale,
                convrot.per_tensor_scale,
                convrot.act_per_tensor_scale,
            )
            if storage is not None
        )

    def test_reconstructs_same_wrapper_into_reusable_device_storage(self) -> None:
        convrot, _dense = _make_convrot_nvfp4(group_size=16)
        pinned_param = PinnedParam(nn.Parameter(convrot, requires_grad=False))
        state = pinned_param.allocate_gpu_storage(torch.device("cpu"))

        pinned_param.copy_to_gpu(state)
        reconstructed = pinned_param.make_gpu_param(state).data

        assert type(reconstructed) is type(convrot)
        assert reconstructed.group_size == 16
        assert torch.equal(reconstructed.qdata, convrot.qdata)
        assert torch.equal(
            reconstructed.scale.view(torch.uint8),
            convrot.scale.view(torch.uint8),
        )

    def test_identity_and_pool_layout_track_rotation_group(self) -> None:
        convrot_cls = _modules()[3]
        group_64, _dense = _make_convrot_nvfp4(group_size=64)
        group_16 = convrot_cls(
            group_64.qdata,
            group_64.scale,
            group_64.block_size,
            group_64.orig_dtype,
            16,
            group_64.per_tensor_scale,
            group_64.act_per_tensor_scale,
            group_64.is_swizzled_scales,
            group_64.use_triton_kernel,
            group_64.act_quant_kwargs,
        )

        assert tensor_id(group_64)[0] == "piper-kernels-convrot-nvfp4"
        assert tensor_id(group_64) != tensor_id(group_16)
        assert _param_target_layout(nn.Parameter(group_64, requires_grad=False)) != _param_target_layout(
            nn.Parameter(group_16, requires_grad=False)
        )

    def test_lora_merge_delegates_to_kernel_addmm_in_place(self) -> None:
        rows, cols, rank = 16, 64, 4
        convrot, _dense = _make_convrot_nvfp4(rows=rows, cols=cols)
        param = nn.Parameter(convrot, requires_grad=False)
        a = torch.randn(rank, cols, dtype=convrot.orig_dtype)
        b = torch.randn(rows, rank, dtype=convrot.orig_dtype)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.25)])
        expected = convrot.clone()
        expected.addmm_(b, a, alpha=0.25)
        qdata_ptr = convrot.qdata.data_ptr()
        scale_ptr = convrot.scale.data_ptr()

        transform.validate_target(param)
        transform.apply(param)

        assert param.data.qdata.data_ptr() == qdata_ptr
        assert param.data.scale.data_ptr() == scale_ptr
        assert param.data.group_size == convrot.group_size
        assert torch.equal(param.data.qdata, expected.qdata)
        assert torch.equal(
            param.data.scale.view(torch.uint8),
            expected.scale.view(torch.uint8),
        )
        torch.testing.assert_close(param.data.dequantize(), expected.dequantize())

    @pytest.mark.parametrize("with_lora", [False, True])
    def test_parameter_delta_uses_kernel_add_in_place(
        self,
        with_lora: bool,
    ) -> None:
        rows, cols, rank = 16, 64, 4
        convrot, _dense = _make_convrot_nvfp4(rows=rows, cols=cols)
        model = nn.Module()
        model.lin = nn.Linear(cols, rows, bias=False, dtype=convrot.orig_dtype)
        model.lin.weight = nn.Parameter(convrot, requires_grad=False)
        dense = torch.randn(rows, cols)
        strength = -0.25
        state_dict = {"lin.delta.weight": dense}
        expected_update = torch.zeros(rows, cols, dtype=convrot.orig_dtype)
        expected_update.add_(dense.to(convrot.orig_dtype), alpha=strength)
        if with_lora:
            a = torch.randn(rank, cols)
            b = torch.randn(rows, rank)
            state_dict.update(
                {
                    "lin.lora_A.weight": a,
                    "lin.lora_B.weight": b,
                }
            )
            scaled_a = a.to(convrot.orig_dtype)
            scaled_a.mul_(strength)
            expected_update.addmm_(b.to(convrot.orig_dtype), scaled_a)
        expected = convrot.clone()
        expected.add_(expected_update)
        qdata_ptr = convrot.qdata.data_ptr()
        scale_ptr = convrot.scale.data_ptr()
        adapter = Adapter.from_state_dict(
            state_dict,
            host_backing="adopt",
        )

        assert (
            merge_adapter(
                model,
                [(adapter, strength)],
                stochastic_rounding=False,
            )
            == 1
        )

        assert model.lin.weight.data.qdata.data_ptr() == qdata_ptr
        assert model.lin.weight.data.scale.data_ptr() == scale_ptr
        assert torch.equal(model.lin.weight.data.qdata, expected.qdata)
        assert torch.equal(
            model.lin.weight.data.scale.view(torch.uint8),
            expected.scale.view(torch.uint8),
        )

    def test_dense_merge_forwards_strength_and_rounding_seed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        convrot_cls = _modules()[3]
        convrot, _dense = _make_convrot_nvfp4()
        update = torch.randn(16, 64, dtype=convrot.orig_dtype)
        calls: list[tuple[float, int | None]] = []
        original_add = convrot_cls.add_

        def recording_add(
            target: torch.Tensor,
            other: torch.Tensor,
            *,
            alpha: float = 1,
            rounding_seed: int | None = None,
        ) -> torch.Tensor:
            calls.append((alpha, rounding_seed))
            return original_add(
                target,
                other,
                alpha=alpha,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(convrot_cls, "add_", recording_add)

        PiperConvRotNVFP4Adapter.merge_dense_(
            convrot,
            update,
            0.125,
            rounding_seed=456,
        )

        assert calls == [(0.125, 456)]

    def test_merge_forwards_rounding_seed_to_kernel_addmm(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        convrot_cls = _modules()[3]
        convrot, _dense = _make_convrot_nvfp4()
        mat1 = torch.randn(16, 4, dtype=torch.bfloat16)
        mat2 = torch.randn(4, 64, dtype=torch.bfloat16)
        calls: list[tuple[float, int | None]] = []
        original_addmm = convrot_cls.addmm_

        def recording_addmm(
            target: torch.Tensor,
            b: torch.Tensor,
            a: torch.Tensor,
            *,
            beta: float = 1,
            alpha: float = 1,
            rounding_seed: int | None = None,
        ) -> torch.Tensor:
            calls.append((alpha, rounding_seed))
            return original_addmm(
                target,
                b,
                a,
                beta=beta,
                alpha=alpha,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(convrot_cls, "addmm_", recording_addmm)

        PiperConvRotNVFP4Adapter.merge_lora_(
            convrot,
            mat1,
            mat2,
            0.125,
            rounding_seed=456,
        )

        assert calls == [(0.125, 456)]

    def test_stochastic_merge_replays(self) -> None:
        generator = torch.Generator().manual_seed(123)
        dense = torch.randn(16, 64, dtype=torch.bfloat16, generator=generator)
        a = torch.randn(4, 64, dtype=torch.bfloat16, generator=generator)
        b = torch.randn(16, 4, dtype=torch.bfloat16, generator=generator)

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            convrot, _dense = _make_convrot_nvfp4(dense=dense)
            PiperConvRotNVFP4Adapter.merge_lora_(
                convrot,
                b,
                a,
                0.125,
                rounding_seed=456,
            )
            return convrot.qdata.clone(), convrot.scale.clone()

        first = run()
        replay = run()
        assert torch.equal(first[0], replay[0])
        assert torch.equal(first[1].view(torch.uint8), replay[1].view(torch.uint8))

    def test_advertises_only_movement_and_merge_capabilities(self) -> None:
        adapter = PiperConvRotNVFP4Adapter()

        assert isinstance(adapter, LoRAMergeTensorAdapter)
        assert isinstance(adapter, LoRAMergeValidationTensorAdapter)
        assert isinstance(adapter, DenseMergeTensorAdapter)
        assert isinstance(adapter, DenseMergeTargetValidationTensorAdapter)
        assert not isinstance(adapter, DequantRequantTensorAdapter)
        assert not isinstance(adapter, TensorCopyIntoAdapter)
        assert not isinstance(adapter, CpuRoundTripTensorAdapter)
        assert not isinstance(adapter, ParameterDataSwapTensorAdapter)
        assert PiperConvRotNVFP4Adapter in _ROLLING_ADAPTER_TYPES

    def test_validates_mutated_rotation_metadata(self) -> None:
        convrot, _dense = _make_convrot_nvfp4()
        mutated: Any = convrot
        mutated.group_size = 32

        with pytest.raises(ValueError, match="group size"):
            PiperConvRotNVFP4Adapter.matches(mutated)

    def test_merge_fails_clearly_when_kernel_addmm_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        convrot_cls = _modules()[3]
        convrot, _dense = _make_convrot_nvfp4()
        monkeypatch.delattr(convrot_cls, "addmm_")

        with pytest.raises(RuntimeError, match=r"piper-kernels>=0\.6\.1"):
            PiperConvRotNVFP4Adapter.validate_lora_merge(
                convrot,
                torch.empty(16, 4, dtype=torch.bfloat16),
                torch.empty(4, 64, dtype=torch.bfloat16),
                1.0,
            )

    def test_dense_merge_fails_clearly_when_kernel_add_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        convrot_cls = _modules()[3]
        convrot, _dense = _make_convrot_nvfp4()
        monkeypatch.delattr(convrot_cls, "add_")

        with pytest.raises(RuntimeError, match=r"piper-kernels>=0\.7\.0rc1"):
            PiperConvRotNVFP4Adapter.validate_dense_merge_target(convrot)

    @SM120
    def test_model_offloader_cuda_forward_preserves_convrot_nvfp4(self) -> None:
        convrot_cls = _modules()[3]
        weight, _dense = _make_convrot_nvfp4(rows=128, cols=256)
        layer = nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
        layer.weight = nn.Parameter(weight, requires_grad=False)
        inputs = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        expected = torch.nn.functional.linear(inputs, weight.cuda())
        offloader = ModelOffloader.from_module(layer)

        try:
            with activated_model(offloader, "cuda") as active:
                actual = active(inputs)
                assert isinstance(active.weight.data, convrot_cls)
                assert active.weight.data.group_size == weight.group_size
                torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected)
        finally:
            offloader.deactivate()
