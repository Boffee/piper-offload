"""Tests for Piper ``PiperNVFP4Tensor`` offload support."""

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
from piper_offload.piper_quantized_adapter import PiperQuantizedAdapter
from piper_offload.tensor_adapter_registry import select_adapter, tensor_id
from piper_offload.tensor_adapters import (
    DenseMergeTensorAdapter,
    DenseMergeValidationTensorAdapter,
    LoRAMergeTensorAdapter,
    LoRAMergeValidationTensorAdapter,
    ShardProjectionTensorAdapter,
)
from tests.conftest import activated_model

_param_target_layout = HostParam.target_layout_for


def _exact_sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


SM120 = pytest.mark.skipif(
    not _exact_sm120_available(),
    reason="exact SM120 required",
)


def _piper_cls() -> Any:
    return pytest.importorskip("piper_kernels.weights.nvfp4").PiperNVFP4Tensor


def _make_piper_nvfp4(
    *,
    rows: int = 16,
    cols: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    dense: torch.Tensor | None = None,
    high_first: bool = False,
    is_swizzled_scales: bool = True,
    per_tensor_scale: torch.Tensor | None = None,
    act_per_tensor_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    piper_cls = _piper_cls()
    if dense is None:
        dense = torch.randn(rows, cols, device=device, dtype=dtype)
    else:
        dense = dense.to(device=device, dtype=dtype)
    packed = piper_cls.from_hp(
        dense,
        per_tensor_scale=per_tensor_scale,
        compute_per_tensor_scale=per_tensor_scale is None,
        act_per_tensor_scale=act_per_tensor_scale,
        is_swizzled_scales=is_swizzled_scales,
        act_quant="dynamic",
    )
    if high_first:
        qdata = (((packed.qdata & 0x0F) << 4) | (packed.qdata >> 4)).to(torch.uint8)
        packed = piper_cls(
            qdata,
            packed.scale,
            packed.block_size,
            packed.dtype,
            packed.per_tensor_scale,
            packed.act_per_tensor_scale,
            packed.is_swizzled_scales,
            packed.act_quant,
            high_first=True,
        )
    return packed, dense


class TestPiperNvfp4Adapter:
    def test_matches_piper_nvfp4_and_convrot_nvfp4_under_unified_adapter(self) -> None:
        convrot_mod = pytest.importorskip("piper_kernels.weights.convrot.nvfp4")

        piper_w, _dense = _make_piper_nvfp4()
        convrot_w = convrot_mod.ConvRotNVFP4Tensor.from_quantized(
            piper_w.qdata,
            piper_w.scale,
            group_size=64,
            logical_dtype=piper_w.dtype,
            per_tensor_scale=piper_w.per_tensor_scale,
            act_per_tensor_scale=piper_w.act_per_tensor_scale,
            is_swizzled_scales=piper_w.is_swizzled_scales,
            act_quant=piper_w.act_quant,
            high_first=piper_w.high_first,
        )

        assert PiperQuantizedAdapter.matches(piper_w)
        assert PiperQuantizedAdapter.matches(convrot_w)
        assert isinstance(select_adapter(piper_w), PiperQuantizedAdapter)
        assert isinstance(select_adapter(convrot_w), PiperQuantizedAdapter)
        assert tensor_id(piper_w) != tensor_id(convrot_w)
        assert PiperQuantizedAdapter.layout_signature(piper_w) != PiperQuantizedAdapter.layout_signature(
            convrot_w
        )

    def test_capture_and_reconstruct_preserve_storage_and_metadata(self) -> None:
        piper_w, _dense = _make_piper_nvfp4(high_first=True, act_per_tensor_scale=torch.tensor(0.25))
        host_param = HostParam(nn.Parameter(piper_w, requires_grad=False))
        host = host_param.make_cpu_param().data

        assert type(host) is type(piper_w)
        assert not host.qdata.is_pinned()
        assert not host.scale.is_pinned()
        assert host.high_first is True
        assert host.per_tensor_scale is not None
        assert host.act_per_tensor_scale is not None
        assert host_param.compute_dtype is piper_w.dtype
        assert host_param.cache_bytes == sum(
            s.nbytes
            for s in (piper_w.qdata, piper_w.scale, piper_w.per_tensor_scale, piper_w.act_per_tensor_scale)
            if s is not None
        )

        state = host_param.allocate_gpu_storage(torch.device("cpu"))
        host_param.copy_to_gpu(state)
        reconstructed = host_param.make_gpu_param(state).data
        assert type(reconstructed) is type(piper_w)
        assert reconstructed.high_first is True
        assert torch.equal(reconstructed.qdata, piper_w.qdata)
        assert torch.equal(reconstructed.scale.view(torch.uint8), piper_w.scale.view(torch.uint8))
        torch.testing.assert_close(reconstructed.dequantize(), piper_w.dequantize())

    def test_identity_and_pool_layout_track_high_first(self) -> None:
        low_first, dense = _make_piper_nvfp4(high_first=False)
        high_first, _ = _make_piper_nvfp4(dense=dense, high_first=True)

        assert tensor_id(low_first)[0] == "piper-kernels-quantized"
        assert tensor_id(low_first) != tensor_id(high_first)
        assert _param_target_layout(nn.Parameter(low_first, requires_grad=False)) != _param_target_layout(
            nn.Parameter(high_first, requires_grad=False)
        )

    @pytest.mark.parametrize("beta", [1.0, -0.5, 0.0])
    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=SM120)])
    def test_lora_merge_delegates_to_kernel_addmm_in_place(self, beta: float, device: str) -> None:
        rows, cols, rank = 16, 64, 4
        piper_w, _dense = _make_piper_nvfp4(rows=rows, cols=cols)
        piper_w = piper_w.to(device)
        param = nn.Parameter(piper_w, requires_grad=False)
        a = torch.randn(rank, cols, dtype=piper_w.dtype)
        b = torch.randn(rows, rank, dtype=piper_w.dtype)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.25)], beta=beta)
        expected = piper_w.clone()
        expected.addmm_(b.to(device), a.to(device), alpha=0.25, beta=beta)
        qdata_ptr = piper_w.qdata.data_ptr()
        scale_ptr = piper_w.scale.data_ptr()

        transform.validate_parameter(param)
        transform.apply_parameter(param)

        assert param.data.qdata.data_ptr() == qdata_ptr
        assert param.data.scale.data_ptr() == scale_ptr
        assert torch.equal(param.data.qdata, expected.qdata)
        assert torch.equal(param.data.scale.view(torch.uint8), expected.scale.view(torch.uint8))
        torch.testing.assert_close(param.data.dequantize(), expected.dequantize())

    @pytest.mark.parametrize("with_lora", [False, True])
    def test_parameter_delta_uses_kernel_add_in_place(self, with_lora: bool) -> None:
        rows, cols, rank = 16, 64, 4
        piper_w, _dense = _make_piper_nvfp4(rows=rows, cols=cols)
        model = nn.Module()
        model.lin = nn.Linear(cols, rows, bias=False, dtype=piper_w.dtype)
        model.lin.weight = nn.Parameter(piper_w, requires_grad=False)
        dense = torch.randn(rows, cols)
        strength = -0.25
        state_dict = {"lin.delta.weight": dense}
        expected_update = torch.zeros(rows, cols, dtype=piper_w.dtype)
        expected_update.add_(dense.to(piper_w.dtype), alpha=strength)
        if with_lora:
            a = torch.randn(rank, cols)
            b = torch.randn(rows, rank)
            state_dict.update({"lin.lora_A.weight": a, "lin.lora_B.weight": b})
            scaled_a = a.to(piper_w.dtype)
            scaled_a.mul_(strength)
            expected_update.addmm_(b.to(piper_w.dtype), scaled_a)
        expected = piper_w.clone()
        expected.add_(expected_update)
        qdata_ptr = piper_w.qdata.data_ptr()
        scale_ptr = piper_w.scale.data_ptr()
        adapter = Adapter.from_state_dict(state_dict)

        assert merge_adapter(model, [(adapter, strength)], stochastic_rounding=False) == 1

        assert model.lin.weight.data.qdata.data_ptr() == qdata_ptr
        assert model.lin.weight.data.scale.data_ptr() == scale_ptr
        assert torch.equal(model.lin.weight.data.qdata, expected.qdata)
        assert torch.equal(model.lin.weight.data.scale.view(torch.uint8), expected.scale.view(torch.uint8))

    def test_merge_forwards_coefficients_and_rounding_seed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        piper_cls = _piper_cls()
        piper_w, _ = _make_piper_nvfp4()
        b = torch.randn(16, 4, dtype=torch.bfloat16)
        a = torch.randn(4, 64, dtype=torch.bfloat16)
        update = torch.randn(16, 64, dtype=torch.bfloat16)
        addmm_calls: list[tuple[float, float, int | None]] = []
        add_calls: list[tuple[float, float, int | None]] = []
        orig_addmm = piper_cls.addmm_
        orig_add = piper_cls.add_

        def rec_addmm(
            target: torch.Tensor,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            *,
            beta: float = 1,
            alpha: float = 1,
            rounding_seed: int | None = None,
        ) -> torch.Tensor:
            addmm_calls.append((alpha, beta, rounding_seed))
            return orig_addmm(target, mat1, mat2, beta=beta, alpha=alpha, rounding_seed=rounding_seed)

        def rec_add(
            target: torch.Tensor,
            other: torch.Tensor,
            *,
            alpha: float = 1,
            beta: float = 1,
            rounding_seed: int | None = None,
        ) -> torch.Tensor:
            add_calls.append((alpha, beta, rounding_seed))
            return orig_add(target, other, alpha=alpha, beta=beta, rounding_seed=rounding_seed)

        monkeypatch.setattr(piper_cls, "addmm_", rec_addmm)
        monkeypatch.setattr(piper_cls, "add_", rec_add)

        PiperQuantizedAdapter.merge_lora_(piper_w, b, a, 0.125, beta=-0.5, rounding_seed=456)
        PiperQuantizedAdapter.merge_dense_(piper_w, update, 0.25, beta=-0.75, rounding_seed=789)

        assert addmm_calls == [(0.125, -0.5, 456)]
        assert add_calls == [(0.25, -0.75, 789)]

    def test_preflight_rejects_3d_and_transposed_updates(self) -> None:
        piper_cls = _piper_cls()
        w3 = piper_cls.from_hp(
            torch.randn(2, 16, 64, dtype=torch.bfloat16),
            per_tensor_scale=torch.tensor([0.2, 0.4], dtype=torch.float32).view(2, 1, 1),
            is_swizzled_scales=True,
        )
        with pytest.raises(ValueError, match="2-D weight"):
            PiperQuantizedAdapter.validate_lora_merge(w3)
        with pytest.raises(ValueError, match="2-D weight"):
            PiperQuantizedAdapter.validate_dense_merge(w3)

        w2, _ = _make_piper_nvfp4(rows=16, cols=64, is_swizzled_scales=False)
        wt = w2.t()
        assert wt.transposed
        with pytest.raises(NotImplementedError, match="transposed weight"):
            PiperQuantizedAdapter.validate_lora_merge(wt)
        with pytest.raises(NotImplementedError, match="transposed weight"):
            PiperQuantizedAdapter.validate_dense_merge(wt)

    def test_preflight_rejects_non_scalar_per_tensor_scale_atomically(self) -> None:
        base: Any = _make_piper_nvfp4(rows=16, cols=64, is_swizzled_scales=False)[0]
        base.per_tensor_scale = torch.tensor([0.2, 0.4], dtype=torch.float32).view(2, 1, 1)
        model = nn.Module()
        model.good = nn.Linear(16, 4, bias=False, dtype=torch.bfloat16).requires_grad_(False)
        model.good.weight.fill_(1.0)
        model.bad = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16).requires_grad_(False)
        model.bad.weight = nn.Parameter(base, requires_grad=False)
        lora = Adapter.from_state_dict(
            {
                "good.lora_A.weight": torch.ones(2, 16, dtype=torch.bfloat16),
                "good.lora_B.weight": torch.ones(4, 2, dtype=torch.bfloat16),
                "bad.lora_A.weight": torch.ones(2, 64, dtype=torch.bfloat16),
                "bad.lora_B.weight": torch.ones(16, 2, dtype=torch.bfloat16),
            }
        )
        dense = Adapter.from_state_dict(
            {
                "good.delta.weight": torch.ones(4, 16, dtype=torch.bfloat16),
                "bad.delta.weight": torch.ones(16, 64, dtype=torch.bfloat16),
            }
        )
        for adapter in (lora, dense):
            with pytest.raises(ValueError, match="per_tensor_scale"):
                merge_adapter(model, [(adapter, 1.0)])
            torch.testing.assert_close(model.good.weight, torch.ones_like(model.good.weight), rtol=0, atol=0)

    def test_preflight_rejects_non_canonical_flat_scale_atomically(self) -> None:
        piper_cls = _piper_cls()
        base, _ = _make_piper_nvfp4(rows=16, cols=64, is_swizzled_scales=False)
        flat_weight = piper_cls.from_quantized(
            base.qdata,
            base.scale.reshape(-1),
            logical_dtype=base.dtype,
            is_swizzled_scales=False,
        )
        assert PiperQuantizedAdapter.matches(flat_weight)
        with pytest.raises(ValueError, match="canonical 2-D scales"):
            PiperQuantizedAdapter.validate_lora_merge(flat_weight)
        with pytest.raises(ValueError, match="canonical 2-D scales"):
            PiperQuantizedAdapter.validate_dense_merge(flat_weight)

        model = nn.Module()
        model.good = nn.Linear(16, 4, bias=False, dtype=torch.bfloat16).requires_grad_(False)
        model.good.weight.fill_(1.0)
        model.bad = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16).requires_grad_(False)
        model.bad.weight = nn.Parameter(flat_weight, requires_grad=False)
        lora = Adapter.from_state_dict(
            {
                "good.lora_A.weight": torch.ones(2, 16, dtype=torch.bfloat16),
                "good.lora_B.weight": torch.ones(4, 2, dtype=torch.bfloat16),
                "bad.lora_A.weight": torch.ones(2, 64, dtype=torch.bfloat16),
                "bad.lora_B.weight": torch.ones(16, 2, dtype=torch.bfloat16),
            }
        )
        dense = Adapter.from_state_dict(
            {
                "good.delta.weight": torch.ones(4, 16, dtype=torch.bfloat16),
                "bad.delta.weight": torch.ones(16, 64, dtype=torch.bfloat16),
            }
        )
        for adapter in (lora, dense):
            with pytest.raises(ValueError, match="canonical 2-D scales"):
                merge_adapter(model, [(adapter, 1.0)])
            torch.testing.assert_close(model.good.weight, torch.ones_like(model.good.weight), rtol=0, atol=0)

    def test_from_storage_validates_subclass_identity(self) -> None:
        cr_mod = pytest.importorskip("piper_kernels.weights.convrot.nvfp4")
        base, _ = _make_piper_nvfp4(rows=16, cols=64, is_swizzled_scales=False)
        tensors, metadata = base.as_storage()
        with pytest.raises(ValueError, match="group_size"):
            cr_mod.ConvRotNVFP4Tensor.from_storage(tensors, metadata)

    def test_stochastic_merge_replays(self) -> None:
        generator = torch.Generator().manual_seed(123)
        dense = torch.randn(16, 64, dtype=torch.bfloat16, generator=generator)
        a = torch.randn(4, 64, dtype=torch.bfloat16, generator=generator)
        b = torch.randn(16, 4, dtype=torch.bfloat16, generator=generator)

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            piper_w, _ = _make_piper_nvfp4(dense=dense)
            PiperQuantizedAdapter.merge_lora_(piper_w, b, a, 0.125, rounding_seed=456)
            return piper_w.qdata.clone(), piper_w.scale.clone()

        first = run()
        replay = run()
        assert torch.equal(first[0], replay[0])
        assert torch.equal(first[1].view(torch.uint8), replay[1].view(torch.uint8))

    def test_3d_per_expert_swizzled_projection_roundtrip(self) -> None:
        piper_cls = _piper_cls()
        torch.manual_seed(11)
        w3 = piper_cls.from_hp(
            torch.randn(4, 192, 80, dtype=torch.bfloat16),
            per_tensor_scale=torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32).view(4, 1, 1),
            is_swizzled_scales=True,
        )
        host = PiperQuantizedAdapter.capture_host(w3)
        proj, _ = PiperQuantizedAdapter.project_shard(
            host, offsets=(2, 128, 64), local_shape=(2, 64, 16)
        )
        assert proj.target_shapes is not None
        assert tuple(proj.storage[2].shape) == (2, 1, 1)  # type: ignore[union-attr]
        slots = tuple(
            None
            if tensor is None
            else (
                tensor.contiguous()
                if shape is None
                else tensor.contiguous().view(shape)
            )
            for tensor, shape in zip(proj.storage, proj.target_shapes, strict=True)
        )
        rebuilt = PiperQuantizedAdapter._reconstruct(slots, proj.meta)
        ref = w3.dequantize()[2:4, 128:192, 64:80]
        torch.testing.assert_close(rebuilt.dequantize(), ref, rtol=0, atol=0)

    def test_advertises_expected_capabilities(self) -> None:
        adapter = PiperQuantizedAdapter()

        assert isinstance(adapter, LoRAMergeTensorAdapter)
        assert isinstance(adapter, LoRAMergeValidationTensorAdapter)
        assert isinstance(adapter, DenseMergeTensorAdapter)
        assert isinstance(adapter, DenseMergeValidationTensorAdapter)
        assert isinstance(adapter, ShardProjectionTensorAdapter)

    @SM120
    def test_model_offloader_cuda_forward_preserves_piper_nvfp4(self) -> None:
        piper_cls = _piper_cls()
        weight, _dense = _make_piper_nvfp4(rows=128, cols=256)
        layer = nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
        layer.weight = nn.Parameter(weight, requires_grad=False)
        inputs = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        expected = torch.nn.functional.linear(inputs, weight.cuda())
        offloader = ModelOffloader.from_module(layer)

        try:
            with activated_model(offloader, "cuda") as active:
                actual = active(inputs)
                assert isinstance(active.weight.data, piper_cls)
                torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected)
        finally:
            offloader.deactivate()
