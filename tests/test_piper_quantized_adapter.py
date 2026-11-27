"""Tests for ``PiperQuantizedAdapter`` across ``ConvRotTrellisTensor`` and custom ``QuantizedWeight``."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from piper_kernels.weights import QuantizedWeight
from piper_kernels.weights._dispatch import register_quantized_weight_ops
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.trellis import ConvRotTrellisTensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor
from piper_kernels.weights.sharding import trellis_row_shard_byte_range
from torch import nn

from tests.test_checkpoint import _write as _write_safetensors

from piper_offload import (
    Adapter,
    MappedCheckpoint,
    ModelCache,
    ModelOffloader,
    ModelSpec,
    ModelStore,
    OffloadConfig,
    ParameterDelta,
    file_slice,
    host_pin_manager,
    merge_adapter,
    requires_activation,
)
from piper_offload._dtensor import DTENSOR_AVAILABLE
from piper_offload.dtensor_adapter import _local_shape_and_offsets
from piper_offload.host_param import HostParam
from piper_offload.piper_quantized_adapter import PiperQuantizedAdapter, _PiperQuantizedMeta
from piper_offload.structured_adapter import StructuredHost
from piper_offload.tensor_adapter_registry import select_adapter
from tests.conftest import activated_model, offloaded_units, resident_instance

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_trellis_weight(
    out_features: int = 32,
    in_features: int = 128,
    *,
    bpw: int = 3,
    bytes_per_block: int | None = None,
    block_size: int = 16,
    group_size: int = 64,
    logical_dtype: torch.dtype = torch.bfloat16,
    row_widths: torch.Tensor | None = None,
    seed: int = 0,
) -> ConvRotTrellisTensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    blocks_per_row = in_features // block_size
    scale_choices = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.bfloat16)
    idx = torch.randint(0, 4, (out_features, blocks_per_row), generator=gen)
    block_scales = scale_choices[idx].to(torch.float8_e4m3fn)
    scale = (
        torch.rand((out_features, 1), generator=gen, dtype=torch.float32) * 0.5 + 0.25
    )
    if row_widths is None:
        resolved_bpb = (bpw * block_size) // 8 if bytes_per_block is None else bytes_per_block
        bitstreams = torch.randint(
            0, 256, (out_features, blocks_per_row, resolved_bpb), generator=gen, dtype=torch.uint8
        )
    else:
        total_bytes = int(row_widths.sum().item()) * blocks_per_row
        bitstreams = torch.randint(0, 256, (total_bytes,), generator=gen, dtype=torch.uint8)
    return ConvRotTrellisTensor.from_quantized(
        bitstreams,
        block_scales,
        scale,
        block_size=block_size,
        group_size=group_size,
        logical_dtype=logical_dtype,
        row_widths=row_widths,
    )


class _TrellisNet(nn.Module):
    def __init__(
        self,
        *,
        out_features: int = 32,
        in_features: int = 128,
        weights: list[torch.Tensor],
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for w in weights:
            lin = nn.Linear(in_features, out_features, bias=False, device="meta")
            lin.weight = nn.Parameter(w, requires_grad=False)
            self.blocks.append(lin)
        self.eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class _SyntheticQuantizedWeight(QuantizedWeight):
    """Minimal third-party-style ``QuantizedWeight`` testing generic adapter support."""

    _format_name = "synthetic_quant"
    _update_format_name = "synthetic_quant"
    tensor_data_names = ["qdata", "scale"]
    optional_tensor_data_names = ["extra_scale"]
    tensor_attribute_names = ["dtype", "tag"]
    optional_tensor_attribute_names = ["transposed"]

    qdata: torch.Tensor
    scale: torch.Tensor
    extra_scale: torch.Tensor | None
    tag: int

    @staticmethod
    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
        tag: int = 1,
        *,
        extra_scale: torch.Tensor | None = None,
        transposed: bool = False,
    ) -> _SyntheticQuantizedWeight:
        shape = (qdata.shape[1], qdata.shape[0]) if transposed else tuple(qdata.shape)
        self = cast(
            _SyntheticQuantizedWeight,
            torch.Tensor._make_wrapper_subclass(cls, shape, dtype=dtype, device=qdata.device),
        )
        self.qdata = qdata
        self.scale = scale
        self.tag = int(tag)
        self.extra_scale = extra_scale
        self.transposed = bool(transposed)
        return self

    def _validate_storage_for_export(self) -> None:
        for name in ("qdata", "scale", "extra_scale"):
            tensor = getattr(self, name)
            if tensor is not None and not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

    def dequantize(self, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        dt = self.dtype if output_dtype is None else output_dtype
        w = self.qdata.to(dt) * self.scale.to(dt)
        if self.extra_scale is not None:
            w = w * self.extra_scale.to(dt)
        return w.t() if self.transposed else w

    def _transpose(self) -> _SyntheticQuantizedWeight:
        return _SyntheticQuantizedWeight(
            self.qdata,
            self.scale,
            self.dtype,
            self.tag,
            extra_scale=self.extra_scale,
            transposed=not self.transposed,
        )


register_quantized_weight_ops(_SyntheticQuantizedWeight)


@_SyntheticQuantizedWeight.implements_torch_function(F.linear)
def _synthetic_linear(
    _func: object,
    _types: tuple[type, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    weight = cast(_SyntheticQuantizedWeight, args[1])
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    return F.linear(args[0], weight.dequantize(), bias)


def _with_meta(meta: _PiperQuantizedMeta, **updates: Any) -> _PiperQuantizedMeta:
    meta_map = dict(meta.metadata)
    meta_map.update(updates)
    return replace(meta, metadata=tuple(sorted(meta_map.items())))


def _reconstruct_cpu(state: StructuredHost[_PiperQuantizedMeta]) -> torch.Tensor:
    targets = state.target_shapes
    slots = tuple(
        None
        if tensor is None
        else (
            tensor.contiguous()
            if targets is None or targets[idx] is None
            else tensor.contiguous().view(targets[idx])  # type: ignore[arg-type]
        )
        for idx, tensor in enumerate(state.storage)
    )
    return PiperQuantizedAdapter._reconstruct(slots, state.meta)


def _make_meta_shard_dtensor(
    mesh: Any,
    global_shape: tuple[int, ...],
    local_shape: tuple[int, ...],
    placements: list[Any],
) -> nn.Parameter:
    from torch.distributed.tensor import DTensor

    stride: list[int] = [1] * len(global_shape)
    for dim in range(len(global_shape) - 2, -1, -1):
        stride[dim] = stride[dim + 1] * global_shape[dim + 1]
    return nn.Parameter(
        DTensor.from_local(
            torch.empty(local_shape, device="meta"),
            mesh,
            placements,
            shape=global_shape,
            stride=tuple(stride),
            run_check=False,
        ),
        requires_grad=False,
    )


def _assert_trellis_shard_projection_parity(
    state: StructuredHost[_PiperQuantizedMeta],
    ref: torch.Tensor,
    x_full: torch.Tensor,
    *,
    offsets: tuple[int, int],
    local_shape: tuple[int, int],
    expect_active: bool,
    expected_ptrs: tuple[int, ...] | None,
) -> StructuredHost[_PiperQuantizedMeta]:
    proj, sig = PiperQuantizedAdapter.project_shard(state, offsets=offsets, local_shape=local_shape)
    if expected_ptrs is not None:
        proj_ptrs = tuple(
            t.untyped_storage().data_ptr() for t in PiperQuantizedAdapter.storage_tensors(proj)
        )
        assert proj_ptrs == expected_ptrs
    cpu_param = PiperQuantizedAdapter.cpu_param(proj)
    assert requires_activation(cpu_param) is expect_active
    rebuilt_cpu = cast(ConvRotTrellisTensor, _reconstruct_cpu(proj))
    expected_slice = ref[offsets[0] : offsets[0] + local_shape[0], offsets[1] : offsets[1] + local_shape[1]]
    torch.testing.assert_close(rebuilt_cpu.dequantize(), expected_slice, rtol=0, atol=0)
    assert PiperQuantizedAdapter.layout_signature(rebuilt_cpu) == sig

    x_slice = x_full[:, offsets[1] : offsets[1] + local_shape[1]]
    torch.testing.assert_close(
        F.linear(x_slice, rebuilt_cpu), F.linear(x_slice, expected_slice), rtol=2e-2, atol=2e-2
    )
    if torch.cuda.is_available():
        gpu_state = PiperQuantizedAdapter.alloc_gpu(proj, torch.device("cuda"))
        PiperQuantizedAdapter.copy_to_gpu(proj, gpu_state, non_blocking=False)
        rebuilt_gpu = cast(ConvRotTrellisTensor, PiperQuantizedAdapter.gpu_param(proj, gpu_state).data)
        assert rebuilt_gpu.bitstreams.is_cuda and rebuilt_gpu.bitstreams.is_contiguous()
        assert PiperQuantizedAdapter.layout_signature(rebuilt_gpu) == sig
        torch.testing.assert_close(rebuilt_gpu.dequantize().cpu(), expected_slice, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            F.linear(x_slice.cuda(), rebuilt_gpu).cpu(),
            F.linear(x_slice, expected_slice),
            rtol=2e-2,
            atol=2e-2,
        )
    return proj


def _assert_trellis_2d_rank(
    store: ModelStore,
    u_ref: torch.Tensor,
    v_ref: torch.Tensor,
    cuda_lock: threading.Lock,
) -> tuple[int, tuple[int, ...]]:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import Shard

    mesh = DeviceMesh("cpu", [[0, 1], [2, 3]])
    coord = tuple(mesh.get_coordinate() or ())
    placements = [Shard(0), Shard(1)]
    local_shape, offsets = _local_shape_and_offsets((32, 128), tuple(mesh.shape), coord, tuple(placements))
    shell = nn.Module()
    shell.u = _make_meta_shard_dtensor(mesh, (32, 128), local_shape, placements)
    shell.v = _make_meta_shard_dtensor(mesh, (32, 128), local_shape, placements)
    shell.eval().requires_grad_(False)
    offloader = store.bind(shell)
    try:
        resident = resident_instance(offloader)
        assert resident is not None
        assert offloader.cache_bytes < store.cache_bytes
        u_hp = resident.params["u"]
        v_hp = resident.params["v"]
        assert requires_activation(shell.u.data)
        assert not requires_activation(shell.v.data)
        assert isinstance(shell.v.to_local(), ConvRotTrellisTensor)
        u_expected = u_ref[offsets[0] : offsets[0] + local_shape[0], offsets[1] : offsets[1] + local_shape[1]]
        v_expected = v_ref[offsets[0] : offsets[0] + local_shape[0], offsets[1] : offsets[1] + local_shape[1]]
        u_rebuilt = cast(ConvRotTrellisTensor, _reconstruct_cpu(cast(Any, u_hp.host_state).inner_state))
        torch.testing.assert_close(u_rebuilt.dequantize(), u_expected, rtol=0, atol=0)
        torch.testing.assert_close(shell.v.to_local().dequantize(), v_expected, rtol=0, atol=0)
        if torch.cuda.is_available():
            with cuda_lock:
                dev = torch.device("cuda:0")
                for hp, exp_slice in ((u_hp, u_expected), (v_hp, v_expected)):
                    hs = cast(Any, hp.host_state)
                    gs = cast(Any, hp.allocate_gpu_storage(dev))
                    hp.copy_to_gpu(gs, non_blocking=False)
                    gpu_w = cast(ConvRotTrellisTensor, hs.inner.gpu_param(hs.inner_state, gs.inner_gpu).data)
                    assert gpu_w.bitstreams.is_cuda and gpu_w.bitstreams.is_contiguous()
                    torch.testing.assert_close(gpu_w.dequantize().cpu(), exp_slice, rtol=2e-2, atol=2e-2)
    finally:
        offloader.deactivate()
    return offloader.cache_bytes, offsets


class TestConvRotTrellisOffload:
    def test_selection_host_param_and_metadata_signatures(self) -> None:
        uniform = _make_trellis_weight(32, 128, bpw=3, group_size=64, block_size=16, seed=1)
        var_rw = torch.tensor([4, 6, 8, 10, 12] * 6 + [6, 10], dtype=torch.uint8)
        variable = _make_trellis_weight(32, 128, group_size=64, block_size=16, row_widths=var_rw, seed=1)

        for weight, expected_slots in ((uniform, 3), (variable, 4)):
            adapter = select_adapter(weight)
            assert isinstance(adapter, PiperQuantizedAdapter)
            assert not requires_activation(weight)

            hp = HostParam(nn.Parameter(weight, requires_grad=False))
            assert len(hp.storage_tensors()) == expected_slots
            rebuilt = cast(ConvRotTrellisTensor, hp.make_cpu_param().data)
            assert isinstance(rebuilt, ConvRotTrellisTensor)
            assert rebuilt.group_size == weight.group_size
            assert rebuilt.block_size == weight.block_size
            assert rebuilt.dtype == weight.dtype
            assert PiperQuantizedAdapter.tensor_id(rebuilt) == PiperQuantizedAdapter.tensor_id(weight)
            assert PiperQuantizedAdapter.layout_signature(rebuilt) == PiperQuantizedAdapter.layout_signature(
                weight
            )
            torch.testing.assert_close(rebuilt.dequantize(), weight.dequantize(), rtol=0, atol=0)

        diff_group = _make_trellis_weight(32, 128, bpw=3, group_size=16, seed=1)
        diff_dtype = _make_trellis_weight(
            32, 128, bpw=3, group_size=64, logical_dtype=torch.float16, seed=1
        )
        other_storage = _make_trellis_weight(32, 128, bpw=3, group_size=64, seed=2)
        assert PiperQuantizedAdapter.tensor_id(uniform) != PiperQuantizedAdapter.tensor_id(variable)
        assert PiperQuantizedAdapter.tensor_id(uniform) != PiperQuantizedAdapter.tensor_id(diff_group)
        assert PiperQuantizedAdapter.tensor_id(uniform) != PiperQuantizedAdapter.tensor_id(diff_dtype)
        assert PiperQuantizedAdapter.tensor_id(uniform) != PiperQuantizedAdapter.tensor_id(other_storage)
        assert PiperQuantizedAdapter.layout_signature(uniform) != PiperQuantizedAdapter.layout_signature(
            diff_group
        )
        assert PiperQuantizedAdapter.layout_signature(uniform) != PiperQuantizedAdapter.layout_signature(
            diff_dtype
        )
        assert PiperQuantizedAdapter.layout_signature(uniform) != PiperQuantizedAdapter.layout_signature(
            variable
        )
        assert PiperQuantizedAdapter.layout_signature(uniform) == PiperQuantizedAdapter.layout_signature(
            other_storage
        )

    @pytest.mark.parametrize("bpw", [2, 3, 4, 5, 6])
    def test_cpu_offloader_resident_and_streaming_parity(self, bpw: int) -> None:
        w0 = _make_trellis_weight(64, 64, bpw=bpw, group_size=64, seed=10 + bpw)
        rw = torch.tensor([4, 6, 8, 10, 12] * 12 + [4, 6, 8, 10], dtype=torch.uint8)
        w1 = _make_trellis_weight(64, 64, group_size=64, row_widths=rw, seed=20 + bpw)

        proto = _TrellisNet(out_features=64, in_features=64, weights=[w0, w1])
        x = torch.randn(3, 64, dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = proto(x)
        store = ModelStore.from_module(proto)

        for cfg in (None, OffloadConfig(offload_paths=("blocks",))):
            shell = _TrellisNet(
                out_features=64,
                in_features=64,
                weights=[torch.empty((64, 64), dtype=torch.bfloat16, device="meta") for _ in range(2)],
            )
            offloader = store.bind(shell, config=cfg)
            assert (resident_instance(offloader) is None) is (cfg is not None)
            assert len(offloaded_units(offloader)) == (2 if cfg is not None else 0)
            with activated_model(offloader, "cpu") as active, torch.inference_mode():
                actual = active(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert not shell.blocks[0].weight.is_meta

    def test_tied_weights_store_and_model_cache_lifecycle(self) -> None:
        rw = torch.tensor([4, 6, 8, 10, 12] * 6 + [4, 12], dtype=torch.uint8)
        shared = _make_trellis_weight(32, 64, group_size=64, row_widths=rw, seed=7)

        class TiedModel(nn.Module):
            def __init__(self, w: torch.Tensor) -> None:
                super().__init__()
                self.a = nn.Linear(64, 32, bias=False, device="meta")
                self.b = nn.Linear(64, 32, bias=False, device="meta")
                self.a.weight = nn.Parameter(w, requires_grad=False)
                self.b.weight = self.a.weight
                self.eval().requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.linear(x, self.a.weight) + F.linear(x, self.b.weight)

        proto = TiedModel(shared)
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = proto(x)

        store = ModelStore.from_module(proto)
        single_hp = HostParam(nn.Parameter(shared, requires_grad=False))
        assert store.cache_bytes == single_hp.cache_bytes

        shell = TiedModel(torch.empty((32, 64), dtype=torch.bfloat16, device="meta"))
        bound = store.bind(shell)
        with activated_model(bound, "cpu") as active, torch.inference_mode():
            assert active.a.weight.data.bitstreams.data_ptr() == active.b.weight.data.bitstreams.data_ptr()
            torch.testing.assert_close(active(x), expected, rtol=0, atol=0)

        cache = ModelCache()
        spec = ModelSpec(
            key="trellis",
            factory=lambda: TiedModel(_make_trellis_weight(32, 64, group_size=64, row_widths=rw, seed=7)),
        )
        with cache.use(spec, device="cpu") as cached, torch.inference_mode():
            torch.testing.assert_close(cached(x), expected, rtol=0, atol=0)
        cache.clear()

    @CUDA
    def test_cuda_execution_uniform_bpw_and_variable_row(self) -> None:
        torch.manual_seed(42)
        weights = [
            _make_trellis_weight(64, 64, bpw=bpw, group_size=64, seed=30 + bpw) for bpw in (2, 3, 4, 5, 6)
        ]
        rw = torch.tensor([4, 6, 8, 10, 12] * 12 + [4, 6, 8, 10], dtype=torch.uint8)
        weights.append(_make_trellis_weight(64, 64, group_size=64, row_widths=rw, seed=40))

        model = _TrellisNet(out_features=64, in_features=64, weights=weights)
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        with torch.inference_mode():
            cpu_linear_refs = [F.linear(x, b.weight.data) for b in model.blocks]
            cpu_deq_refs = [
                F.linear(x, cast(ConvRotTrellisTensor, b.weight.data).dequantize()) for b in model.blocks
            ]
            gpu_expected = x.cuda()
            for b in model.blocks:
                gpu_expected = F.linear(gpu_expected, cast(ConvRotTrellisTensor, b.weight.data).cuda())
            gpu_expected = gpu_expected.cpu()

        store = ModelStore.from_module(model)
        for cfg in (None, OffloadConfig(offload_paths=("blocks",))):
            shell = _TrellisNet(
                out_features=64,
                in_features=64,
                weights=[torch.empty((64, 64), dtype=torch.bfloat16, device="meta") for _ in weights],
            )
            offloader = store.bind(shell, config=cfg)
            assert (resident_instance(offloader) is None) is (cfg is not None)
            assert len(offloaded_units(offloader)) == (len(weights) if cfg is not None else 0)
            with activated_model(offloader, "cuda") as active, torch.inference_mode():
                actual = active(x.cuda()).cpu()
                torch.testing.assert_close(actual, gpu_expected, rtol=0, atol=0)
                if cfg is None:
                    for b, lin_ref, deq_ref in zip(active.blocks, cpu_linear_refs, cpu_deq_refs, strict=True):
                        gpu_out = F.linear(x.cuda(), b.weight).cpu()
                        torch.testing.assert_close(gpu_out, lin_ref, rtol=2e-2, atol=2e-2)
                        torch.testing.assert_close(gpu_out, deq_ref, rtol=5e-2, atol=5e-2)
            assert not shell.blocks[0].weight.is_meta

        cache = ModelCache()
        spec = ModelSpec(
            key="trellis-cuda",
            factory=lambda: _TrellisNet(out_features=64, in_features=64, weights=weights),
            config=OffloadConfig(offload_paths=("blocks",)),
        )
        with cache.use(spec, device="cuda") as active, torch.inference_mode():
            torch.testing.assert_close(active(x.cuda()).cpu(), gpu_expected, rtol=0, atol=0)
        cache.clear()

    def test_parameter_value_update_cpu_and_cuda(self) -> None:
        rw = torch.tensor([4, 6, 8, 12] * 8, dtype=torch.uint8)
        w_new = _make_trellis_weight(32, 64, group_size=64, row_widths=rw, seed=52)

        model = nn.Module()
        model.w = nn.Parameter(torch.empty((32, 64), dtype=torch.bfloat16, device="meta"), requires_grad=False)
        model.requires_grad_(False)
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        with torch.inference_mode():
            expected_cpu = F.linear(x, w_new)
            expected_deq = F.linear(x, w_new.dequantize())

        val_adapter = Adapter.from_state_dict({"w": w_new}, scale_parameter_values=False)
        if torch.cuda.is_available():
            offloader = ModelOffloader.from_module(model)
            expected_cuda = F.linear(x.cuda(), w_new.cuda()).cpu()
            with activated_model(offloader, "cuda", adapters=[val_adapter], adapter_strengths=[1.0]):
                assert isinstance(model.w.data, ConvRotTrellisTensor) and model.w.is_cuda
                torch.testing.assert_close(F.linear(x.cuda(), model.w).cpu(), expected_cuda, rtol=0, atol=0)
                torch.testing.assert_close(F.linear(x.cuda(), model.w).cpu(), expected_deq, rtol=2e-2, atol=2e-2)

        cpu_model = nn.Module()
        cpu_model.w = nn.Parameter(
            torch.empty((32, 64), dtype=torch.bfloat16, device="meta"), requires_grad=False
        )
        cpu_model.requires_grad_(False)
        merge_adapter(cpu_model, [(val_adapter, 1.0)])
        assert isinstance(cpu_model.w.data, ConvRotTrellisTensor) and cpu_model.w.device.type == "cpu"
        torch.testing.assert_close(F.linear(x, cpu_model.w), expected_cpu, rtol=0, atol=0)
        torch.testing.assert_close(F.linear(x, cpu_model.w), expected_deq, rtol=2e-2, atol=2e-2)

    def test_preflight_rejections_preserve_sibling_weights(self) -> None:
        w = _make_trellis_weight(32, 64, bpw=4, group_size=64, seed=61)

        wt = w.t()
        with pytest.raises(NotImplementedError, match="transposed weight"):
            select_adapter(wt)
        with pytest.raises(NotImplementedError, match="transposed weight"):
            HostParam(nn.Parameter(wt, requires_grad=False))

        strided_bits = torch.randint(0, 256, (32, 8, 8), dtype=torch.uint8)[:, ::2, :]
        assert not strided_bits.is_contiguous()
        non_contig = ConvRotTrellisTensor.from_quantized(
            w.bitstreams.clone(),
            w.block_scales.clone(),
            w.scale.clone(),
            block_size=16,
            group_size=64,
            logical_dtype=torch.bfloat16,
        )
        non_contig.bitstreams = strided_bits
        with pytest.raises(ValueError, match="contiguous"):
            select_adapter(non_contig)

        class MixedNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.plain = nn.Linear(64, 32, bias=False, dtype=torch.bfloat16)
                self.trellis = nn.Linear(64, 32, bias=False, device="meta")
                self.trellis.weight = nn.Parameter(w, requires_grad=False)
                self.eval().requires_grad_(False)

        net = MixedNet()
        plain_before = net.plain.weight.detach().clone()

        dense = Adapter({
            "plain.weight": ParameterDelta.from_tensors(dense=torch.randn(32, 64, dtype=torch.bfloat16) * 0.1),
            "trellis.weight": ParameterDelta.from_tensors(
                dense=torch.randn(32, 64, dtype=torch.bfloat16) * 0.1
            ),
        })
        lora = Adapter.from_state_dict({
            "plain.lora_A.weight": torch.randn(4, 64, dtype=torch.bfloat16) * 0.1,
            "plain.lora_B.weight": torch.randn(32, 4, dtype=torch.bfloat16) * 0.1,
            "trellis.lora_A.weight": torch.randn(4, 64, dtype=torch.bfloat16) * 0.1,
            "trellis.lora_B.weight": torch.randn(32, 4, dtype=torch.bfloat16) * 0.1,
        })
        for upd in (dense, lora):
            with pytest.raises(ValueError, match="ConvRotTrellisTensor does not support in-place"):
                merge_adapter(net, [(upd, 0.5)])
            torch.testing.assert_close(net.plain.weight, plain_before)

        if torch.cuda.is_available():
            offloader = ModelOffloader.from_module(net)
            for upd in (dense, lora):
                with pytest.raises(ValueError, match="ConvRotTrellisTensor does not support in-place"):
                    offloader.activate("cuda", adapters=[upd], adapter_strengths=[0.5])
                assert offloader.active_device is None
                torch.testing.assert_close(net.plain.weight, plain_before)

    def test_non_integer_bpw_byte_accounting_and_scaled_value_rejection(self) -> None:
        w_frac = _make_trellis_weight(
            32, 64, bytes_per_block=5, block_size=16, group_size=64, seed=105
        )
        rw = torch.tensor([4, 5, 8, 12] * 8, dtype=torch.uint8)
        w_var = _make_trellis_weight(32, 64, block_size=16, group_size=64, row_widths=rw, seed=106)
        for w in (w_frac, w_var):
            hp = HostParam(nn.Parameter(w, requires_grad=False))
            expected_bytes = sum(t.nbytes for t in hp.storage_tensors())
            assert hp.cache_bytes == hp.gpu_target_bytes == expected_bytes

        x = torch.randn(2, 64, dtype=torch.bfloat16)
        model = _TrellisNet(out_features=32, in_features=64, weights=[w_frac])
        with torch.inference_mode():
            expected_cpu = model(x)
        offloader = ModelOffloader.from_module(model)
        with activated_model(offloader, "cpu") as active, torch.inference_mode():
            torch.testing.assert_close(active(x), expected_cpu, rtol=0, atol=0)
        if torch.cuda.is_available():
            expected_cuda = F.linear(x.cuda(), w_frac.cuda()).cpu()
            with activated_model(offloader, "cuda") as active, torch.inference_mode():
                torch.testing.assert_close(active(x.cuda()).cpu(), expected_cuda, rtol=0, atol=0)

        scaled = Adapter.from_state_dict({"w": w_frac}, scale_parameter_values=True)
        meta_m = nn.Module()
        meta_m.w = nn.Parameter(
            torch.empty((32, 64), dtype=torch.bfloat16, device="meta"), requires_grad=False
        )
        meta_m.requires_grad_(False)
        with pytest.raises(ValueError, match="ConvRotTrellisTensor does not support in-place"):
            merge_adapter(meta_m, [(scaled, 0.5)])
        assert meta_m.w.is_meta
        if torch.cuda.is_available():
            meta_off = ModelOffloader.from_module(meta_m)
            with pytest.raises(ValueError, match="ConvRotTrellisTensor does not support in-place"):
                meta_off.activate("cuda", adapters=[scaled], adapter_strengths=[0.5])
            assert meta_off.active_device is None and meta_m.w.is_meta


class TestConvRotTrellisDTensorProjection:
    def test_project_shard_uniform_and_variable_row_cpu_and_cuda(self) -> None:
        torch.manual_seed(71)
        u_w = _make_trellis_weight(32, 128, bpw=3, block_size=16, group_size=64, seed=71)
        u_state = PiperQuantizedAdapter.capture_host(u_w)
        u_ptrs = tuple(t.untyped_storage().data_ptr() for t in PiperQuantizedAdapter.storage_tensors(u_state))
        u_ref = u_w.dequantize()
        x_full = torch.randn(2, 128, dtype=torch.bfloat16)

        for offsets, local_shape, expect_active in (
            ((16, 0), (16, 128), False),
            ((0, 64), (32, 64), True),
            ((16, 64), (16, 64), True),
        ):
            _assert_trellis_shard_projection_parity(
                u_state,
                u_ref,
                x_full,
                offsets=offsets,
                local_shape=local_shape,
                expect_active=expect_active,
                expected_ptrs=u_ptrs,
            )

        rw = torch.tensor([4, 6, 8, 10, 12] * 6 + [6, 10], dtype=torch.uint8)
        v_w = _make_trellis_weight(32, 128, block_size=16, group_size=64, row_widths=rw, seed=72)
        v_state = PiperQuantizedAdapter.capture_host(v_w)
        v_ptrs = tuple(t.untyped_storage().data_ptr() for t in PiperQuantizedAdapter.storage_tensors(v_state))
        v_ref = v_w.dequantize()

        # Shard(0): zero-copy narrowed bitstreams via trellis_row_shard_byte_range.
        v_s0 = _assert_trellis_shard_projection_parity(
            v_state,
            v_ref,
            x_full,
            offsets=(8, 0),
            local_shape=(12, 128),
            expect_active=False,
            expected_ptrs=v_ptrs,
        )
        expected_range = trellis_row_shard_byte_range(
            rw, blocks_per_row=128 // 16, start=8, length=12, block_size=16
        )
        assert v_s0.storage[0] is not None
        assert v_s0.storage[0].storage_offset() == expected_range.offset
        assert v_s0.storage[0].numel() == expected_range.length

        # Shard(1) & 2-D: variable-row column sharding re-packs contiguous bitstreams via shard_quantized_weight.
        for offsets, local_shape in (((0, 64), (32, 64)), ((12, 64), (16, 64))):
            _assert_trellis_shard_projection_parity(
                v_state,
                v_ref,
                x_full,
                offsets=offsets,
                local_shape=local_shape,
                expect_active=False,
                expected_ptrs=None,
            )

    def test_model_store_bind_dtensor_1d_and_2d_parallel_executor(self) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import Shard

        from piper_offload.parallel import ParallelExecutor

        rw = torch.tensor([4, 6, 8, 10, 12] * 6 + [4, 8], dtype=torch.uint8)
        u_glb = _make_trellis_weight(32, 128, bpw=4, block_size=16, group_size=64, seed=81)
        v_glb = _make_trellis_weight(32, 128, block_size=16, group_size=64, row_widths=rw, seed=82)
        u_ref = u_glb.dequantize()
        v_ref = v_glb.dequantize()

        proto = nn.Module()
        proto.u = nn.Parameter(u_glb, requires_grad=False)
        proto.v = nn.Parameter(v_glb, requires_grad=False)
        proto.eval().requires_grad_(False)
        store = ModelStore.from_module(proto)
        u_ptr = store._module_store.params["u"].storage_tensors()[0].untyped_storage().data_ptr()
        v_ptr = store._module_store.params["v"].storage_tensors()[0].untyped_storage().data_ptr()
        cuda_lock = threading.Lock()

        with ParallelExecutor(["cpu"] * 4) as ex:
            results = ex.run(lambda _rank: _assert_trellis_2d_rank(store, u_ref, v_ref, cuda_lock))
        assert {offsets for _, offsets in results} == {(0, 0), (0, 64), (16, 0), (16, 64)}

        def verify_1d_and_rejection(_rank: int) -> None:
            mesh = DeviceMesh("cpu", [0, 1])
            bad_shell = nn.Module()
            bad_shell.u = _make_meta_shard_dtensor(mesh, (32, 128), (32, 32), [Shard(1)])
            bad_shell.v = _make_meta_shard_dtensor(mesh, (32, 128), (32, 32), [Shard(1)])
            with pytest.raises(ValueError, match="layout mismatch"):
                store.bind(bad_shell)
            assert bad_shell.u.is_meta and bad_shell.v.is_meta

            s0_shell = nn.Module()
            s0_shell.u = _make_meta_shard_dtensor(mesh, (32, 128), (16, 128), [Shard(0)])
            s0_shell.v = _make_meta_shard_dtensor(mesh, (32, 128), (16, 128), [Shard(0)])
            off = store.bind(s0_shell)
            try:
                assert not requires_activation(s0_shell.u.data)
                assert not requires_activation(s0_shell.v.data)
                assert s0_shell.u.to_local().bitstreams.untyped_storage().data_ptr() == u_ptr
                assert s0_shell.v.to_local().bitstreams.untyped_storage().data_ptr() == v_ptr
            finally:
                off.deactivate()

        with ParallelExecutor(["cpu", "cpu"]) as ex:
            ex.run(verify_1d_and_rejection)

    def test_trellis_and_convrot_int8_projection_rejections(self) -> None:
        u_w = _make_trellis_weight(32, 128, bpw=3, block_size=16, group_size=64, seed=91)
        u_state = PiperQuantizedAdapter.capture_host(u_w)
        rw = torch.tensor([4, 6, 8, 10] * 8, dtype=torch.uint8)
        v_w = _make_trellis_weight(32, 128, block_size=16, group_size=64, row_widths=rw, seed=92)
        v_state = PiperQuantizedAdapter.capture_host(v_w)

        with pytest.raises(ValueError, match="2-D matrices only"):
            PiperQuantizedAdapter.project_shard(
                StructuredHost(storage=u_state.storage, meta=u_state.meta, logical_shape=(2, 16, 128)),
                offsets=(0, 0, 0),
                local_shape=(1, 16, 128),
            )
        with pytest.raises(ValueError, match="non-empty local shards"):
            PiperQuantizedAdapter.project_shard(u_state, offsets=(0, 0), local_shape=(0, 128))
        with pytest.raises(ValueError, match="exceeds weight shape"):
            PiperQuantizedAdapter.project_shard(u_state, offsets=(16, 0), local_shape=(32, 128))
        with pytest.raises(ValueError, match="rotation group_size"):
            PiperQuantizedAdapter.project_shard(u_state, offsets=(0, 16), local_shape=(32, 64))

        fine_group = PiperQuantizedAdapter.capture_host(
            _make_trellis_weight(32, 128, bpw=3, block_size=32, group_size=16, seed=93)
        )
        with pytest.raises(ValueError, match="block_size"):
            PiperQuantizedAdapter.project_shard(fine_group, offsets=(0, 16), local_shape=(32, 64))

        bad_bits = StructuredHost(
            storage=(u_state.storage[0].transpose(0, 1), u_state.storage[1], u_state.storage[2]),  # type: ignore[union-attr]
            meta=u_state.meta,
            logical_shape=u_state.logical_shape,
        )
        with pytest.raises(ValueError, match="contiguous"):
            PiperQuantizedAdapter.project_shard(bad_bits, offsets=(0, 0), local_shape=(16, 128))

        bad_bscale = StructuredHost(
            storage=(u_state.storage[0], u_state.storage[1][:, :4].clone(), u_state.storage[2]),  # type: ignore[index]
            meta=u_state.meta,
            logical_shape=u_state.logical_shape,
        )
        with pytest.raises(ValueError, match="incompatible scale shapes"):
            PiperQuantizedAdapter.project_shard(bad_bscale, offsets=(0, 0), local_shape=(16, 128))

        bad_uniform_ndim = StructuredHost(
            storage=(u_state.storage[0].reshape(-1), u_state.storage[1], u_state.storage[2]),  # type: ignore[union-attr]
            meta=u_state.meta,
            logical_shape=u_state.logical_shape,
        )
        with pytest.raises(ValueError, match="3-D bitstreams"):
            PiperQuantizedAdapter.project_shard(bad_uniform_ndim, offsets=(0, 0), local_shape=(16, 128))

        bad_var_len = StructuredHost(
            storage=(
                v_state.storage[0][:-1],  # type: ignore[index]
                v_state.storage[1],
                v_state.storage[2],
                v_state.storage[3],
            ),
            meta=v_state.meta,
            logical_shape=v_state.logical_shape,
        )
        with pytest.raises(ValueError, match="shorter than row_widths"):
            PiperQuantizedAdapter.project_shard(bad_var_len, offsets=(0, 0), local_shape=(32, 128))

        for bad_field, bad_val, match in (
            ("block_size", 0, "block_size must be positive"),
            ("group_size", 0, "group_size must be positive"),
        ):
            with pytest.raises(ValueError, match=match):
                PiperQuantizedAdapter.project_shard(
                    StructuredHost(
                        storage=u_state.storage,
                        meta=_with_meta(u_state.meta, **{bad_field: bad_val}),
                        logical_shape=u_state.logical_shape,
                    ),
                    offsets=(0, 0),
                    local_shape=(16, 128),
                )

        int8_state = PiperQuantizedAdapter.capture_host(
            ConvRotInt8Tensor.from_hp(torch.randn(16, 128, dtype=torch.bfloat16), group_size=64)
        )
        with pytest.raises(ValueError, match="group_size must be positive"):
            PiperQuantizedAdapter.project_shard(
                StructuredHost(
                    storage=int8_state.storage,
                    meta=_with_meta(int8_state.meta, group_size=0),
                    logical_shape=int8_state.logical_shape,
                ),
                offsets=(0, 64),
                local_shape=(16, 64),
            )

    def test_variable_row_projection_rejections(self) -> None:
        rw = torch.tensor([4, 6, 8, 10] * 8, dtype=torch.uint8)
        v_w = _make_trellis_weight(32, 128, block_size=16, group_size=64, row_widths=rw, seed=94)
        v_state = PiperQuantizedAdapter.capture_host(v_w)

        bad_var_ndim = StructuredHost(
            storage=(v_state.storage[0].view(1, -1), v_state.storage[1], v_state.storage[2], v_state.storage[3]),  # type: ignore[union-attr]
            meta=v_state.meta,
            logical_shape=v_state.logical_shape,
        )
        with pytest.raises(ValueError, match="1-D bitstreams"):
            PiperQuantizedAdapter.project_shard(bad_var_ndim, offsets=(0, 0), local_shape=(16, 128))

        bad_rw_shape = StructuredHost(
            storage=(v_state.storage[0], v_state.storage[1], v_state.storage[2], rw[:-1]),
            meta=v_state.meta,
            logical_shape=v_state.logical_shape,
        )
        with pytest.raises(ValueError, match="row_widths"):
            PiperQuantizedAdapter.project_shard(bad_rw_shape, offsets=(0, 0), local_shape=(16, 128))

        bad_var_rw = torch.full((32,), 3, dtype=torch.uint8)
        for bad_width in (3, 13):
            bad_rw = StructuredHost(
                storage=(
                    v_state.storage[0],
                    v_state.storage[1],
                    v_state.storage[2],
                    bad_var_rw.fill_(bad_width).clone(),
                ),
                meta=v_state.meta,
                logical_shape=v_state.logical_shape,
            )
            with pytest.raises(ValueError, match="variable-row widths"):
                PiperQuantizedAdapter.project_shard(bad_rw, offsets=(0, 0), local_shape=(16, 128))

    def test_replicate_dtensor_and_active_shard1_merge_rejection(self) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import Replicate, Shard

        from piper_offload.parallel import ParallelExecutor

        rw = torch.tensor([4, 6, 8, 10] * 8, dtype=torch.uint8)
        v_w = _make_trellis_weight(32, 128, block_size=16, group_size=64, row_widths=rw, seed=94)
        u_glb = _make_trellis_weight(32, 128, bpw=4, block_size=16, group_size=64, seed=95)
        proto = nn.Module()
        proto.u = nn.Parameter(u_glb, requires_grad=False)
        proto.v = nn.Parameter(v_w, requires_grad=False)
        proto.eval().requires_grad_(False)
        x = torch.randn(2, 128, dtype=torch.bfloat16)
        with torch.inference_mode():
            expected_rep = F.linear(x, u_glb) + F.linear(x, v_w)
        store = ModelStore.from_module(proto)

        def verify_replicate(_rank: int) -> None:
            mesh = DeviceMesh("cpu", [0, 1])
            shell = nn.Module()
            shell.u = _make_meta_shard_dtensor(mesh, (32, 128), (32, 128), [Replicate()])
            shell.v = _make_meta_shard_dtensor(mesh, (32, 128), (32, 128), [Replicate()])
            shell.eval().requires_grad_(False)
            off = store.bind(shell)
            assert off.cache_bytes == store.cache_bytes
            with activated_model(off, "cpu") as active, torch.inference_mode():
                out = F.linear(x, active.u.to_local()) + F.linear(x, active.v.to_local())
                torch.testing.assert_close(out, expected_rep, rtol=0, atol=0)

        with ParallelExecutor(["cpu", "cpu"]) as ex:
            ex.run(verify_replicate)

        if torch.cuda.is_available():
            u_only = nn.Module()
            u_only.lin = nn.Linear(128, 32, bias=False, device="meta")
            u_only.lin.weight = nn.Parameter(u_glb, requires_grad=False)
            u_only.eval().requires_grad_(False)
            u_store = ModelStore.from_module(u_only)

            def verify_active_shard1_merge_rejected(_rank: int) -> None:
                mesh = DeviceMesh("cpu", [0, 1])
                sh = nn.Module()
                sh.lin = nn.Linear(128, 32, bias=False, device="meta")
                sh.lin.weight = _make_meta_shard_dtensor(mesh, (32, 128), (32, 64), [Shard(1)])
                sh.eval().requires_grad_(False)
                off = u_store.bind(sh)
                lora = Adapter.from_state_dict({
                    "lin.lora_A.weight": torch.randn(4, 128, dtype=torch.bfloat16) * 0.1,
                    "lin.lora_B.weight": torch.randn(32, 4, dtype=torch.bfloat16) * 0.1,
                })
                with pytest.raises(ValueError, match="ConvRotTrellisTensor does not support in-place"):
                    off.activate("cuda", adapters=[lora], adapter_strengths=[0.5])
                assert off.active_device is None

            with ParallelExecutor(["cpu", "cpu"]) as ex:
                ex.run(verify_active_shard1_merge_rejected)

    def test_mapped_checkpoint_variable_row_trellis_shard_projection(self, tmp_path: Path) -> None:
        torch.manual_seed(197)
        rw = torch.tensor([4, 6, 8, 10, 12] * 6 + [6, 10], dtype=torch.uint8)
        v_w = _make_trellis_weight(32, 128, block_size=16, group_size=64, row_widths=rw, seed=197)
        v_ref = v_w.dequantize()
        ckpt_path = tmp_path / "trellis_var.safetensors"
        _write_safetensors(
            ckpt_path,
            {
                "pad": torch.ones(17, dtype=torch.uint8),
                "bitstreams": v_w.bitstreams,
                "block_scales": v_w.block_scales,
                "scale": v_w.scale,
                "row_widths": cast(torch.Tensor, v_w.row_widths),
            },
        )
        original_file = ckpt_path.read_bytes()
        host_pin_manager.clear()
        ckpt = MappedCheckpoint(ckpt_path)
        try:
            mapped_w = ConvRotTrellisTensor.from_quantized(
                ckpt.get_tensor("bitstreams"),
                ckpt.get_tensor("block_scales"),
                ckpt.get_tensor("scale"),
                block_size=v_w.block_size,
                group_size=v_w.group_size,
                logical_dtype=v_w.dtype,
                row_widths=ckpt.get_tensor("row_widths"),
            )
            host_state = PiperQuantizedAdapter.capture_host(mapped_w)
            assert file_slice(cast(torch.Tensor, host_state.storage[0])) is not None
            host_bit_ptr = cast(torch.Tensor, host_state.storage[0]).untyped_storage().data_ptr()

            s0, _ = PiperQuantizedAdapter.project_shard(
                host_state, offsets=(8, 0), local_shape=(12, 128)
            )
            s0_bits = cast(torch.Tensor, s0.storage[0])
            expected_range = trellis_row_shard_byte_range(
                rw, blocks_per_row=128 // 16, start=8, length=12, block_size=16
            )
            assert file_slice(s0_bits) is not None
            assert s0_bits.untyped_storage().data_ptr() == host_bit_ptr
            assert s0_bits.storage_offset() == expected_range.offset
            assert s0_bits.numel() == expected_range.length
            torch.testing.assert_close(
                cast(ConvRotTrellisTensor, _reconstruct_cpu(s0)).dequantize(),
                v_ref[8:20, :],
                rtol=0,
                atol=0,
            )

            if torch.cuda.is_available():
                s0_gpu = PiperQuantizedAdapter.alloc_gpu(s0, torch.device("cuda"))
                with host_pin_manager.acquire(PiperQuantizedAdapter.storage_tensors(s0), pin=True):
                    PiperQuantizedAdapter.copy_to_gpu(s0, s0_gpu, non_blocking=True)
                torch.cuda.synchronize()
                rebuilt_gpu = cast(
                    ConvRotTrellisTensor, PiperQuantizedAdapter.gpu_param(s0, s0_gpu).data
                )
                torch.testing.assert_close(
                    rebuilt_gpu.dequantize().cpu(), v_ref[8:20, :], rtol=2e-2, atol=2e-2
                )

            s1, _ = PiperQuantizedAdapter.project_shard(
                host_state, offsets=(0, 64), local_shape=(32, 64)
            )
            s1_bits = cast(torch.Tensor, s1.storage[0])
            assert file_slice(s1_bits) is None
            assert s1_bits.untyped_storage().resizable()
            torch.testing.assert_close(
                cast(ConvRotTrellisTensor, _reconstruct_cpu(s1)).dequantize(),
                v_ref[:, 64:128],
                rtol=0,
                atol=0,
            )
            assert ckpt_path.read_bytes() == original_file
        finally:
            host_pin_manager.clear()
            ckpt.close()


def _assert_custom_subclass_cuda_lifecycle(
    store: ModelStore,
    shell_stream: _TrellisNet,
    offloader: ModelOffloader,
    val_adapter: Adapter,
    w_new: _SyntheticQuantizedWeight,
    x: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    seen_cuda: list[tuple[bool, int]] = []
    handle = shell_stream.blocks[0].register_forward_pre_hook(
        lambda mod, _inp: seen_cuda.append(
            (mod.weight.data.is_cuda, cast(_SyntheticQuantizedWeight, mod.weight.data).tag)
        )
    )
    try:
        with activated_model(offloader, "cuda") as active, torch.inference_mode():
            torch.testing.assert_close(active(x.cuda()).cpu(), expected, rtol=0, atol=0)
    finally:
        handle.remove()
    assert seen_cuda == [(True, 7)]

    shell_res = _TrellisNet(
        out_features=16,
        in_features=32,
        weights=[torch.empty((16, 32), dtype=torch.bfloat16, device="meta")],
    )
    res_offloader = store.bind(shell_res)
    with activated_model(res_offloader, "cuda") as active, torch.inference_mode():
        rebuilt_gpu = cast(_SyntheticQuantizedWeight, active.blocks[0].weight.data)
        assert rebuilt_gpu.tag == 7 and rebuilt_gpu.is_cuda
        torch.testing.assert_close(active(x.cuda()).cpu(), expected, rtol=0, atol=0)

    meta_target = _TrellisNet(
        out_features=16,
        in_features=32,
        weights=[torch.empty((16, 32), dtype=torch.bfloat16, device="meta")],
    )
    meta_offloader = ModelOffloader.from_module(meta_target)
    with activated_model(
        meta_offloader, "cuda", adapters=[val_adapter], adapter_strengths=[1.0]
    ) as active, torch.inference_mode():
        updated_gpu = cast(_SyntheticQuantizedWeight, active.blocks[0].weight.data)
        assert updated_gpu.tag == 13 and updated_gpu.is_cuda
        torch.testing.assert_close(
            active(x.cuda()).cpu(), F.linear(x, w_new.dequantize()), rtol=0, atol=0
        )


class TestCustomQuantizedWeightSubclass:
    @pytest.mark.parametrize("with_extra", [False, True])
    def test_custom_subclass_offload_and_parameter_value(self, with_extra: bool) -> None:
        torch.manual_seed(101)
        qdata = torch.randint(-8, 8, (16, 32), dtype=torch.int8)
        scale = torch.rand((16, 1), dtype=torch.float32) * 0.5 + 0.1
        extra = torch.tensor(0.5, dtype=torch.float32) if with_extra else None
        w = _SyntheticQuantizedWeight(qdata, scale, torch.bfloat16, 7, extra_scale=extra)

        assert isinstance(select_adapter(w), PiperQuantizedAdapter)
        assert not requires_activation(w)
        assert HostParam(nn.Parameter(w, requires_grad=False)).cache_bytes == (
            qdata.nbytes + scale.nbytes + (extra.nbytes if extra is not None else 0)
        )

        w_tag = _SyntheticQuantizedWeight(qdata.clone(), scale.clone(), torch.bfloat16, 9, extra_scale=extra)
        w_with = _SyntheticQuantizedWeight(
            qdata.clone(), scale.clone(), torch.bfloat16, 7, extra_scale=torch.tensor(0.25)
        )
        w_without = _SyntheticQuantizedWeight(qdata.clone(), scale.clone(), torch.bfloat16, 7, extra_scale=None)
        assert PiperQuantizedAdapter.tensor_id(w) != PiperQuantizedAdapter.tensor_id(w_tag)
        assert PiperQuantizedAdapter.layout_signature(w) != PiperQuantizedAdapter.layout_signature(w_tag)
        assert PiperQuantizedAdapter.tensor_id(w_with) != PiperQuantizedAdapter.tensor_id(w_without)
        assert PiperQuantizedAdapter.layout_signature(w_with) != PiperQuantizedAdapter.layout_signature(
            w_without
        )

        model = _TrellisNet(out_features=16, in_features=32, weights=[w])
        x = torch.randn(2, 32, dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = model(x)

        store = ModelStore.from_module(model)
        shell_stream = _TrellisNet(
            out_features=16,
            in_features=32,
            weights=[torch.empty((16, 32), dtype=torch.bfloat16, device="meta")],
        )
        offloader = store.bind(shell_stream, config=OffloadConfig(offload_paths=("blocks",)))
        with activated_model(offloader, "cpu") as active, torch.inference_mode():
            rebuilt = cast(_SyntheticQuantizedWeight, active.blocks[0].weight.data)
            assert isinstance(rebuilt, _SyntheticQuantizedWeight) and rebuilt.tag == 7
            assert (rebuilt.extra_scale is not None) is with_extra
            torch.testing.assert_close(active(x), expected, rtol=0, atol=0)

        w_new = _SyntheticQuantizedWeight(
            torch.randint(-8, 8, (16, 32), dtype=torch.int8),
            torch.full((16, 1), 0.25, dtype=torch.float32),
            torch.bfloat16,
            13,
            extra_scale=extra,
        )
        val_adapter = Adapter.from_state_dict({"blocks.0.weight": w_new}, scale_parameter_values=False)
        if torch.cuda.is_available():
            _assert_custom_subclass_cuda_lifecycle(
                store, shell_stream, offloader, val_adapter, w_new, x, expected
            )

        cpu_target = _TrellisNet(
            out_features=16,
            in_features=32,
            weights=[torch.empty((16, 32), dtype=torch.bfloat16, device="meta")],
        )
        merge_adapter(cpu_target, [(val_adapter, 1.0)])
        updated = cast(_SyntheticQuantizedWeight, cpu_target.blocks[0].weight.data)
        assert isinstance(updated, _SyntheticQuantizedWeight) and updated.tag == 13
        with torch.inference_mode():
            torch.testing.assert_close(cpu_target(x), F.linear(x, w_new.dequantize()), rtol=0, atol=0)

    def test_custom_subclass_rejects_merges_sharding_transpose_and_non_contiguous(self) -> None:
        w = _SyntheticQuantizedWeight(
            torch.randint(-4, 5, (16, 32), dtype=torch.int8),
            torch.ones((16, 1), dtype=torch.float32),
            torch.bfloat16,
            3,
        )
        with pytest.raises(NotImplementedError, match="transposed weight"):
            select_adapter(w.t())
        non_contig = _SyntheticQuantizedWeight(
            torch.randint(-4, 5, (16, 64), dtype=torch.int8)[:, ::2],
            torch.ones((16, 1), dtype=torch.float32),
            torch.bfloat16,
            3,
        )
        with pytest.raises(ValueError, match="contiguous"):
            select_adapter(non_contig)

        state = PiperQuantizedAdapter.capture_host(w)
        with pytest.raises(NotImplementedError, match="not supported"):
            PiperQuantizedAdapter.project_shard(state, offsets=(0, 0), local_shape=(8, 32))

        class MixedNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.plain = nn.Linear(32, 16, bias=False, dtype=torch.bfloat16)
                self.custom = nn.Linear(32, 16, bias=False, device="meta")
                self.custom.weight = nn.Parameter(w, requires_grad=False)
                self.eval().requires_grad_(False)

        net = MixedNet()
        plain_before = net.plain.weight.detach().clone()
        dense = Adapter({
            "plain.weight": ParameterDelta.from_tensors(dense=torch.randn(16, 32, dtype=torch.bfloat16) * 0.1),
            "custom.weight": ParameterDelta.from_tensors(
                dense=torch.randn(16, 32, dtype=torch.bfloat16) * 0.1
            ),
        })
        lora = Adapter.from_state_dict({
            "plain.lora_A.weight": torch.randn(4, 32, dtype=torch.bfloat16) * 0.1,
            "plain.lora_B.weight": torch.randn(16, 4, dtype=torch.bfloat16) * 0.1,
            "custom.lora_A.weight": torch.randn(4, 32, dtype=torch.bfloat16) * 0.1,
            "custom.lora_B.weight": torch.randn(16, 4, dtype=torch.bfloat16) * 0.1,
        })
        for upd in (dense, lora):
            with pytest.raises(ValueError, match="_SyntheticQuantizedWeight does not support in-place"):
                merge_adapter(net, [(upd, 0.5)])
            torch.testing.assert_close(net.plain.weight, plain_before)

        if torch.cuda.is_available():
            offloader = ModelOffloader.from_module(net)
            for upd in (dense, lora):
                with pytest.raises(ValueError, match="_SyntheticQuantizedWeight does not support in-place"):
                    offloader.activate("cuda", adapters=[upd], adapter_strengths=[0.5])
                assert offloader.active_device is None
                torch.testing.assert_close(net.plain.weight, plain_before)

        if DTENSOR_AVAILABLE:
            from torch.distributed.device_mesh import DeviceMesh
            from torch.distributed.tensor import Replicate, Shard

            from piper_offload.parallel import ParallelExecutor

            proto = nn.Module()
            proto.w = nn.Parameter(w, requires_grad=False)
            proto.eval().requires_grad_(False)
            store = ModelStore.from_module(proto)

            def verify(_rank: int) -> None:
                mesh = DeviceMesh("cpu", [0, 1])
                rep_shell = nn.Module()
                rep_shell.w = _make_meta_shard_dtensor(mesh, (16, 32), (16, 32), [Replicate()])
                off = store.bind(rep_shell)
                assert off.cache_bytes == store.cache_bytes
                local = rep_shell.w.to_local()
                assert isinstance(local, _SyntheticQuantizedWeight) and local.tag == 3
                x = torch.randn(2, 32, dtype=torch.bfloat16)
                with activated_model(off, "cpu") as active, torch.inference_mode():
                    torch.testing.assert_close(
                        F.linear(x, active.w.to_local()), F.linear(x, w.dequantize()), rtol=0, atol=0
                    )

                shard_shell = nn.Module()
                shard_shell.w = _make_meta_shard_dtensor(mesh, (16, 32), (8, 32), [Shard(0)])
                with pytest.raises(ValueError, match="layout mismatch"):
                    store.bind(shard_shell)
                assert shard_shell.w.is_meta

            with ParallelExecutor(["cpu", "cpu"]) as ex:
                ex.run(verify)

    def test_optional_slot_tensor_id_isolation_and_subclass_defense_in_depth(self) -> None:
        torch.manual_seed(111)
        base = PiperNVFP4Tensor.from_hp(
            torch.randn(16, 64, dtype=torch.bfloat16), is_swizzled_scales=False
        )
        shared_scalar = torch.tensor(0.5, dtype=torch.float32)
        w1 = PiperNVFP4Tensor(
            base.qdata, base.scale, base.block_size, base.dtype, shared_scalar, None, False, None
        )
        w2 = PiperNVFP4Tensor(
            base.qdata, base.scale, base.block_size, base.dtype, None, shared_scalar, False, None
        )
        assert PiperQuantizedAdapter.tensor_id(w1) != PiperQuantizedAdapter.tensor_id(w2)
        assert PiperQuantizedAdapter.layout_signature(w1) != PiperQuantizedAdapter.layout_signature(w2)

        model = _TrellisNet(out_features=16, in_features=64, weights=[w1, w2])
        store = ModelStore.from_module(model)
        shell = _TrellisNet(
            out_features=16,
            in_features=64,
            weights=[torch.empty((16, 64), dtype=torch.bfloat16, device="meta") for _ in range(2)],
        )
        offloader = store.bind(shell, config=OffloadConfig(offload_paths=("blocks",)))
        with activated_model(offloader, "cpu") as active:
            first = cast(PiperNVFP4Tensor, active.blocks[0].weight.data)
            second = cast(PiperNVFP4Tensor, active.blocks[1].weight.data)
            assert first.per_tensor_scale is not None and first.act_per_tensor_scale is None
            assert second.per_tensor_scale is None and second.act_per_tensor_scale is not None

        class _ExtraSlotInt8(ConvRotInt8Tensor):
            optional_tensor_data_names = ["act_per_tensor_scale", "extra"]

        int8_state = PiperQuantizedAdapter.capture_host(
            ConvRotInt8Tensor.from_hp(torch.randn(16, 64, dtype=torch.bfloat16), group_size=64)
        )
        extra_state = StructuredHost(
            storage=(*int8_state.storage, torch.ones(1)),
            meta=replace(
                int8_state.meta,
                wrapper_type=_ExtraSlotInt8,
                storage_names=(*int8_state.meta.storage_names, "extra"),
            ),
            logical_shape=int8_state.logical_shape,
        )
        with pytest.raises(NotImplementedError, match="not supported"):
            PiperQuantizedAdapter.project_shard(extra_state, offsets=(0, 0), local_shape=(8, 64))

        class _NoContigCheckWeight(QuantizedWeight):
            tensor_data_names = ["qdata", "scale"]
            tensor_attribute_names = ["dtype"]

            qdata: torch.Tensor
            scale: torch.Tensor

            @staticmethod
            def __new__(cls, qdata: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> _NoContigCheckWeight:
                self = cast(
                    _NoContigCheckWeight,
                    torch.Tensor._make_wrapper_subclass(cls, tuple(qdata.shape), dtype=dtype, device=qdata.device),
                )
                self.qdata = qdata
                self.scale = scale
                self.transposed = False
                return self

            def _validate_storage_for_export(self) -> None:
                pass

        bad_w = _NoContigCheckWeight(
            torch.randint(-4, 5, (8, 16), dtype=torch.int8)[:, ::2],
            torch.ones((8, 1), dtype=torch.float32),
            torch.bfloat16,
        )
        with pytest.raises(ValueError, match="must be contiguous"):
            select_adapter(bad_w)

        synth_w = _SyntheticQuantizedWeight(
            torch.randint(-4, 5, (16, 32), dtype=torch.int8),
            torch.ones((16, 1), dtype=torch.float32),
            torch.bfloat16,
            5,
        )
        scaled_custom = Adapter.from_state_dict({"blocks.0.weight": synth_w}, scale_parameter_values=True)
        meta_custom = _TrellisNet(
            out_features=16,
            in_features=32,
            weights=[torch.empty((16, 32), dtype=torch.bfloat16, device="meta")],
        )
        with pytest.raises(ValueError, match="_SyntheticQuantizedWeight does not support in-place"):
            merge_adapter(meta_custom, [(scaled_custom, 0.5)])
        assert meta_custom.blocks[0].weight.is_meta

    def test_quantized_subclass_merge_rejection_and_single_pass_as_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _SubInt8(ConvRotInt8Tensor):
            pass

        class _SubNVFP4(PiperNVFP4Tensor):
            pass

        base_int8 = ConvRotInt8Tensor.from_hp(
            torch.randn(16, 64, dtype=torch.bfloat16), group_size=64
        )
        sub_int8 = _SubInt8(
            base_int8.qdata, base_int8.scale, base_int8.group_size, base_int8.dtype
        )
        base_nv = PiperNVFP4Tensor.from_hp(
            torch.randn(16, 64, dtype=torch.bfloat16), is_swizzled_scales=False
        )
        sub_nv = _SubNVFP4(
            base_nv.qdata,
            base_nv.scale,
            base_nv.block_size,
            base_nv.dtype,
            is_swizzled_scales=False,
        )
        for sub_weight in (sub_int8, sub_nv):
            expected_msg = f"{type(sub_weight).__name__} does not support in-place"
            with pytest.raises(ValueError, match=expected_msg):
                PiperQuantizedAdapter.validate_lora_merge(sub_weight)
            with pytest.raises(ValueError, match=expected_msg):
                PiperQuantizedAdapter.validate_dense_merge(sub_weight)

        calls = 0
        orig_as_storage = QuantizedWeight.as_storage

        def counted_as_storage(
            self: QuantizedWeight,
        ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
            nonlocal calls
            calls += 1
            return orig_as_storage(self)

        monkeypatch.setattr(QuantizedWeight, "as_storage", counted_as_storage)
        trellis_w = _make_trellis_weight(32, 128, bpw=3, block_size=16, group_size=64, seed=5)
        for weight in (base_nv, trellis_w):
            for op in (
                PiperQuantizedAdapter.tensor_id,
                PiperQuantizedAdapter.layout_signature,
                PiperQuantizedAdapter.capture_host,
            ):
                calls = 0
                op(weight)
                assert calls == 1

    def test_projection_defensive_missing_slots_and_nvfp4_block_size(self) -> None:
        int8_state = PiperQuantizedAdapter.capture_host(
            ConvRotInt8Tensor.from_hp(torch.randn(16, 64, dtype=torch.bfloat16), group_size=64)
        )
        for idx, slot_name in ((0, "qdata"), (1, "scale")):
            bad_storage = tuple(None if i == idx else t for i, t in enumerate(int8_state.storage))
            with pytest.raises(RuntimeError, match=f"host state is missing {slot_name}"):
                PiperQuantizedAdapter.project_shard(
                    StructuredHost(
                        storage=bad_storage,
                        meta=replace(int8_state.meta, storage_names=("qdata", "scale")),
                        logical_shape=int8_state.logical_shape,
                    ),
                    offsets=(0, 0),
                    local_shape=(8, 64),
                )

        u_state = PiperQuantizedAdapter.capture_host(
            _make_trellis_weight(32, 128, bpw=3, block_size=16, group_size=64, seed=6)
        )
        for idx, slot_name in ((0, "bitstreams"), (1, "block_scales"), (2, "scale")):
            bad_storage = tuple(None if i == idx else t for i, t in enumerate(u_state.storage))
            with pytest.raises(RuntimeError, match=f"host state is missing {slot_name}"):
                PiperQuantizedAdapter.project_shard(
                    StructuredHost(
                        storage=bad_storage,
                        meta=u_state.meta,
                        logical_shape=u_state.logical_shape,
                    ),
                    offsets=(0, 0),
                    local_shape=(16, 128),
                )

        nv_state = PiperQuantizedAdapter.capture_host(
            PiperNVFP4Tensor.from_hp(
                torch.randn(16, 64, dtype=torch.bfloat16), is_swizzled_scales=False
            )
        )
        for bad_bs in (True, None):
            with pytest.raises(ValueError, match="block_size must be positive"):
                PiperQuantizedAdapter.project_shard(
                    StructuredHost(
                        storage=nv_state.storage,
                        meta=_with_meta(nv_state.meta, block_size=bad_bs),
                        logical_shape=nv_state.logical_shape,
                    ),
                    offsets=(0, 0),
                    local_shape=(8, 64),
                )
