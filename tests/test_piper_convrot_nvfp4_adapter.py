"""Tests for Piper ``ConvRotNVFP4Tensor`` offload support."""

from typing import Any

import pytest
import torch
from torch import nn

from piper_offload import (
    Adapter,
    ModelOffloader,
    ScaledLoRAFactor,
    merge_adapter,
)
from piper_offload.host_param import HostParam
from piper_offload.lora import LoRATransform
from piper_offload.piper_convrot_nvfp4_adapter import (
    PiperConvRotNVFP4Adapter,
)
from piper_offload.piper_nvfp4_adapter import PiperNvfp4Adapter
from piper_offload.tensor_adapter_registry import select_adapter, tensor_id
from piper_offload.tensor_adapters import (
    DenseMergeTensorAdapter,
    DenseMergeValidationTensorAdapter,
    LoRAMergeTensorAdapter,
    LoRAMergeValidationTensorAdapter,
)
from tests.conftest import activated_model

_param_target_layout = HostParam.target_layout_for


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


SM120 = pytest.mark.skipif(
    not _exact_sm120_available(),
    reason="exact SM120 required",
)


def _modules() -> tuple[Any, Any, Any]:
    nvfp4 = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    convrot = pytest.importorskip("piper_kernels.weights.convrot.nvfp4")
    return (
        nvfp4.NVFP4Tensor,
        nvfp4.QuantizeTensorToNVFP4Kwargs,
        convrot.ConvRotNVFP4Tensor,
    )


def _make_convrot_nvfp4(
    *,
    rows: int = 16,
    cols: int = 64,
    group_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    dense: torch.Tensor | None = None,
    high_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    _nvfp4_cls, kwargs_cls, convrot_cls = _modules()
    if dense is None:
        dense = torch.randn(rows, cols, device=device, dtype=dtype)
    else:
        dense = dense.to(device=device, dtype=dtype)
    convrot = convrot_cls.from_hp(
        dense,
        group_size=group_size,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        act_quant_kwargs=kwargs_cls(
            block_size=16,
            is_swizzled_scales=True,
            use_triton_kernel=False,
            use_dynamic_per_tensor_scale=True,
        ),
    )
    if high_first:
        qdata = (((convrot.qdata & 0x0F) << 4) | (convrot.qdata >> 4)).to(torch.uint8)
        convrot = convrot_cls(
            qdata,
            convrot.scale,
            convrot.block_size,
            convrot.orig_dtype,
            convrot.group_size,
            convrot.per_tensor_scale,
            convrot.act_per_tensor_scale,
            convrot.is_swizzled_scales,
            convrot.use_triton_kernel,
            convrot.act_quant_kwargs,
            high_first=True,
        )
    return convrot, dense


class TestPiperConvRotNVFP4Adapter:
    def test_dedicated_adapter_selects_convrot_and_not_plain_nvfp4(self) -> None:
        nvfp4_cls = _modules()[0]
        convrot, dense = _make_convrot_nvfp4()
        torchao_w = nvfp4_cls.to_nvfp4(dense, is_swizzled_scales=True, use_triton_kernel=False)

        assert PiperConvRotNVFP4Adapter.matches(convrot)
        assert not PiperNvfp4Adapter.matches(convrot)
        assert isinstance(select_adapter(convrot), PiperConvRotNVFP4Adapter)
        with pytest.raises(NotImplementedError, match="register_adapter"):
            select_adapter(torchao_w)

    def test_capture_preserves_storage_and_all_metadata(self) -> None:
        convrot, _dense = _make_convrot_nvfp4(group_size=64)
        host_param = HostParam(nn.Parameter(convrot, requires_grad=False))
        host = host_param.make_cpu_param().data

        assert type(host) is type(convrot)
        assert not host.qdata.is_pinned()
        assert not host.scale.is_pinned()
        assert host.per_tensor_scale is not None
        assert not host.per_tensor_scale.is_pinned()
        assert host.group_size == 64
        assert host.block_size == convrot.block_size
        assert host.orig_dtype is convrot.orig_dtype
        assert host.is_swizzled_scales is convrot.is_swizzled_scales
        assert host.use_triton_kernel is convrot.use_triton_kernel
        assert host.act_quant_kwargs == convrot.act_quant_kwargs
        assert host_param.compute_dtype is convrot.orig_dtype
        assert host_param.cache_bytes == sum(
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
        host_param = HostParam(nn.Parameter(convrot, requires_grad=False))
        state = host_param.allocate_gpu_storage(torch.device("cpu"))

        host_param.copy_to_gpu(state)
        reconstructed = host_param.make_gpu_param(state).data

        assert type(reconstructed) is type(convrot)
        assert reconstructed.group_size == 16
        assert torch.equal(reconstructed.qdata, convrot.qdata)
        assert torch.equal(
            reconstructed.scale.view(torch.uint8),
            convrot.scale.view(torch.uint8),
        )

    def test_high_first_metadata_survives_movement(self) -> None:
        convrot, _dense = _make_convrot_nvfp4(high_first=True)
        host_param = HostParam(nn.Parameter(convrot, requires_grad=False))
        state = host_param.allocate_gpu_storage(torch.device("cpu"))

        host_param.copy_to_gpu(state)
        reconstructed = host_param.make_gpu_param(state).data

        assert reconstructed.high_first is True
        assert torch.equal(reconstructed.qdata, convrot.qdata)
        torch.testing.assert_close(reconstructed.dequantize(), convrot.dequantize())

    def test_identity_and_pool_layout_track_rotation_group(self) -> None:
        convrot_cls = _modules()[2]
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

    @pytest.mark.parametrize("beta", [1.0, -0.5, 0.0])
    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=SM120)])
    def test_lora_merge_delegates_to_kernel_addmm_in_place(self, beta: float, device: str) -> None:
        rows, cols, rank = 16, 64, 4
        convrot, _dense = _make_convrot_nvfp4(rows=rows, cols=cols)
        convrot = convrot.to(device)
        param = nn.Parameter(convrot, requires_grad=False)
        a = torch.randn(rank, cols, dtype=convrot.orig_dtype)
        b = torch.randn(rows, rank, dtype=convrot.orig_dtype)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.25)], beta=beta)
        expected = convrot.clone()
        expected.addmm_(b.to(device), a.to(device), alpha=0.25, beta=beta)
        qdata_ptr = convrot.qdata.data_ptr()
        scale_ptr = convrot.scale.data_ptr()

        transform.validate_parameter(param)
        transform.apply_parameter(param)

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

    @pytest.mark.parametrize("beta", [1.0, -0.5])
    def test_dense_merge_forwards_coefficients_and_rounding_seed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        beta: float,
    ) -> None:
        convrot_cls = _modules()[2]
        convrot, _dense = _make_convrot_nvfp4()
        update = torch.randn(16, 64, dtype=convrot.orig_dtype)
        calls: list[tuple[float, float, int | None]] = []
        original_add = convrot_cls.add_

        def recording_add(
            target: torch.Tensor,
            other: torch.Tensor,
            *,
            alpha: float = 1,
            beta: float = 1,
            rounding_seed: int | None = None,
        ) -> torch.Tensor:
            calls.append((alpha, beta, rounding_seed))
            return original_add(
                target,
                other,
                alpha=alpha,
                beta=beta,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(convrot_cls, "add_", recording_add)

        PiperConvRotNVFP4Adapter.merge_dense_(
            convrot,
            update,
            0.125,
            beta=beta,
            rounding_seed=456,
        )

        assert calls == [(0.125, beta, 456)]

    def test_merge_forwards_rounding_seed_to_kernel_addmm(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        convrot_cls = _modules()[2]
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
        assert isinstance(adapter, DenseMergeValidationTensorAdapter)

    def test_validates_mutated_rotation_metadata(self) -> None:
        convrot, _dense = _make_convrot_nvfp4()
        mutated: Any = convrot
        mutated.group_size = 32

        with pytest.raises(ValueError, match="group size"):
            PiperConvRotNVFP4Adapter.matches(mutated)

    def test_preflight_rejects_transposed_and_non_contiguous_atomically(self) -> None:
        convrot_cls = _modules()[2]
        base, _ = _make_convrot_nvfp4(rows=16, cols=64, group_size=64)
        non_contig = convrot_cls(
            base.qdata[::2],
            base.scale.clone(),
            base.block_size,
            base.orig_dtype,
            group_size=base.group_size,
            per_tensor_scale=base.per_tensor_scale,
            act_per_tensor_scale=base.act_per_tensor_scale,
            is_swizzled_scales=base.is_swizzled_scales,
            use_triton_kernel=base.use_triton_kernel,
            act_quant_kwargs=base.act_quant_kwargs,
            high_first=base.high_first,
        )
        for bad_weight, rows, cols in ((base.t(), 64, 16), (non_contig, 8, 64)):
            model = nn.Module()
            model.good = nn.Linear(16, 4, bias=False, dtype=torch.bfloat16).requires_grad_(False)
            model.good.weight.fill_(1.0)
            model.bad = nn.Linear(cols, rows, bias=False, dtype=torch.bfloat16).requires_grad_(False)
            model.bad.weight = nn.Parameter(bad_weight, requires_grad=False)
            lora = Adapter.from_state_dict(
                {
                    "good.lora_A.weight": torch.ones(2, 16, dtype=torch.bfloat16),
                    "good.lora_B.weight": torch.ones(4, 2, dtype=torch.bfloat16),
                    "bad.lora_A.weight": torch.ones(2, cols, dtype=torch.bfloat16),
                    "bad.lora_B.weight": torch.ones(rows, 2, dtype=torch.bfloat16),
                }
            )
            dense = Adapter.from_state_dict(
                {
                    "good.delta.weight": torch.ones(4, 16, dtype=torch.bfloat16),
                    "bad.delta.weight": torch.ones(rows, cols, dtype=torch.bfloat16),
                }
            )
            for adapter in (lora, dense):
                with pytest.raises(ValueError, match="contiguous untransposed packed storage"):
                    merge_adapter(model, [(adapter, 1.0)])
                torch.testing.assert_close(model.good.weight, torch.ones_like(model.good.weight), rtol=0, atol=0)

    @SM120
    def test_model_offloader_cuda_forward_preserves_convrot_nvfp4(self) -> None:
        convrot_cls = _modules()[2]
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
