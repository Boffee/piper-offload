"""Tests for direct GGUF-to-ConvRot INT8 offload."""

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from piper_offload import Adapter, ModelOffloader, ModelStore, OffloadConfig, merge_adapter
from piper_offload.gguf_adapter import GgufAdapter
from piper_offload.lora import LoRATransform, ScaledLoRAFactor
from piper_offload.parameter_delta import ParameterDelta, ParameterDeltaTransform
from piper_offload.host_param import HostParam
from piper_offload.tensor_adapter_registry import param_representation
from tests._gguf_helpers import GGUFParameter, packed_q4_0, packed_q8_0

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
QUANT_TYPE = 2  # Q4_0: 32 values per 18-byte block.


@pytest.fixture()
def w() -> GGUFParameter:
    return GGUFParameter(
        torch.zeros((4, 36), dtype=torch.uint8),
        requires_grad=False,
        quant_type=QUANT_TYPE,
    )


def _quantized_weight(
    seed: int,
    *,
    rows: int = 64,
    features: int = 64,
    quant_type: int = QUANT_TYPE,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    generator = torch.Generator().manual_seed(seed)
    packed = (
        packed_q8_0(rows, features, generator=generator)
        if quant_type == 8
        else packed_q4_0(rows, features, generator=generator)
    )
    return GGUFParameter(packed, requires_grad=False, quant_type=quant_type), packed, quant_type


def _materialize(weight: torch.Tensor) -> tuple[HostParam, Any, torch.Tensor]:
    assert isinstance(weight, nn.Parameter)
    host = HostParam(weight)
    state = host.allocate_gpu_storage(torch.device("cuda"))
    param = host.make_gpu_param(state)
    host.copy_to_gpu(state)
    torch.cuda.synchronize()
    return host, state, param_representation(param)


class TestGGUFSource:
    def test_adapter_derives_logical_layout(self, w: GGUFParameter) -> None:
        assert GgufAdapter.logical_shape(w) == (4, 64)
        assert GgufAdapter.compute_dtype(w) is torch.bfloat16

        state = GgufAdapter.capture_host(w)
        assert state.data.data_ptr() == w.as_tensor().data_ptr()
        torch.testing.assert_close(state.data, w.as_tensor().view(torch.uint8))
        assert state.quant_type == QUANT_TYPE
        assert state.logical_shape == (4, 64)
        assert state.group_size == 64
        assert GgufAdapter.cache_bytes(state) == state.data.nbytes

        (packed,) = GgufAdapter.storage_tensors(state)
        assert packed is state.data
        assert type(packed) is torch.Tensor
        assert packed.dtype is torch.uint8
        assert tuple(packed.shape) == tuple(w.as_tensor().shape)

    def test_matches_existing_gguf_parameter_contract(self) -> None:
        weight, packed, _quant_type = _quantized_weight(0)
        assert isinstance(weight, GGUFParameter)
        assert GgufAdapter.matches(weight)

        state = GgufAdapter.capture_host(weight)
        rebuilt = GgufAdapter.cpu_param(state)
        assert isinstance(rebuilt, GGUFParameter)
        assert rebuilt.quant_type == weight.quant_type
        assert rebuilt.quant_shape == weight.quant_shape
        assert rebuilt.as_tensor().data_ptr() == state.data.data_ptr()
        assert rebuilt.as_tensor().data_ptr() == packed.data_ptr()
        torch.testing.assert_close(rebuilt.as_tensor(), packed)

    def test_does_not_match_arbitrary_packed_tensor(self) -> None:
        assert not GgufAdapter.matches(torch.zeros((4, 36), dtype=torch.uint8))

    def test_tensor_id_tracks_packed_storage(self, w: GGUFParameter) -> None:
        alias = GGUFParameter(
            w.as_tensor(),
            requires_grad=False,
            quant_type=w.quant_type,
        )
        clone = GGUFParameter(
            w.as_tensor().clone(),
            requires_grad=False,
            quant_type=w.quant_type,
        )
        conflicting_shape = GGUFParameter(
            w.as_tensor(),
            requires_grad=False,
            quant_type=w.quant_type,
        )
        conflicting_shape.quant_shape = torch.Size((2, 128))

        assert GgufAdapter.tensor_id(w) == GgufAdapter.tensor_id(alias)
        assert GgufAdapter.tensor_id(w) != GgufAdapter.tensor_id(clone)
        assert GgufAdapter.tensor_id(w) != GgufAdapter.tensor_id(conflicting_shape)

    @pytest.mark.parametrize(
        ("features", "expected_group_size"),
        [(32, 16), (128, 64), (256, 256)],
    )
    def test_derives_largest_supported_group_size(
        self,
        features: int,
        expected_group_size: int,
    ) -> None:
        weight, _packed, _quant_type = _quantized_weight(
            features,
            rows=4,
            features=features,
        )

        state = GgufAdapter.capture_host(weight)

        assert state.group_size == expected_group_size

    def test_rejects_width_without_supported_group(self) -> None:
        weight = GGUFParameter(
            torch.zeros((4, 7)),
            requires_grad=False,
            quant_type=0,
        )

        with pytest.raises(ValueError, match="in_features divisible by 16"):
            GgufAdapter.capture_host(weight)

    def test_layout_signature_ignores_quant_type_and_tracks_target_layout(self) -> None:
        q4, _p4, _qt4 = _quantized_weight(11, rows=64, features=64, quant_type=2)
        q8, _p8, _qt8 = _quantized_weight(12, rows=64, features=64, quant_type=8)
        wider, _pw, _qtw = _quantized_weight(13, rows=64, features=128, quant_type=2)

        assert GgufAdapter.tensor_id(q4) != GgufAdapter.tensor_id(q8)
        assert GgufAdapter.layout_signature(q4) == GgufAdapter.layout_signature(q8)
        assert GgufAdapter.layout_signature(q4) != GgufAdapter.layout_signature(wider)

    def test_model_store_bind_accepts_differing_quant_types_and_rejects_shape_mismatch(
        self,
    ) -> None:
        def _shell(weight: torch.Tensor) -> nn.Linear:
            assert isinstance(weight, nn.Parameter)
            mod = nn.Linear(64, 64, bias=False, device="meta")
            mod.weight = weight
            return mod.requires_grad_(False)

        q4, q4_packed, _qt4 = _quantized_weight(14, rows=64, features=64, quant_type=2)
        q8, _q8_packed, _qt8 = _quantized_weight(15, rows=64, features=64, quant_type=8)
        wider, _pw, _qtw = _quantized_weight(16, rows=64, features=128, quant_type=2)
        store = ModelStore.from_module(_shell(q4))

        shell_q8 = _shell(q8)
        store.bind(shell_q8)
        assert isinstance(shell_q8.weight, GGUFParameter)
        assert shell_q8.weight.quant_type == 2
        torch.testing.assert_close(shell_q8.weight.as_tensor(), q4_packed)

        with pytest.raises(ValueError, match="layout mismatch"):
            store.bind(_shell(wider))


class TestPermanentMerge:
    def test_rejects_before_mutating_another_parameter(self) -> None:
        class Model(nn.Module):
            def __init__(self, gguf_weight: nn.Parameter) -> None:
                super().__init__()
                self.first = nn.Linear(64, 64, bias=False)
                self.second = nn.Linear(64, 64, bias=False)
                self.second.weight = gguf_weight

        weight, _packed, _quant_type = _quantized_weight(8)
        assert isinstance(weight, nn.Parameter)
        model = Model(weight)
        model.requires_grad_(False)
        first_before = model.first.weight.detach().clone()
        adapter = Adapter.from_state_dict(
            {
                "first.lora_A.weight": torch.randn(4, 64),
                "first.lora_B.weight": torch.randn(64, 4),
                "second.lora_A.weight": torch.randn(4, 64),
                "second.lora_B.weight": torch.randn(64, 4),
            }
        )

        with pytest.raises(ValueError, match="Permanent updates to packed GGUF"):
            merge_adapter(model, [(adapter, 1.0)])

        torch.testing.assert_close(model.first.weight, first_before)

    def test_rejects_scaled_parameter_value(self) -> None:
        weight, _packed, _quant_type = _quantized_weight(9)
        model = nn.Linear(
            64,
            64,
            bias=False,
            device="meta",
            dtype=torch.bfloat16,
        )
        model.weight.requires_grad_(False)
        adapter = Adapter.from_state_dict(
            {"weight": weight},
            scale_parameter_values=True,
        )

        with pytest.raises(ValueError, match="Permanent updates to packed GGUF"):
            merge_adapter(model, [(adapter, 0.5)])

        assert model.weight.is_meta

    def test_allows_exact_parameter_value(self) -> None:
        weight, packed, quant_type = _quantized_weight(10)
        model = nn.Linear(
            64,
            64,
            bias=False,
            device="meta",
            dtype=torch.bfloat16,
        )
        model.weight.requires_grad_(False)
        adapter = Adapter.from_state_dict({"weight": weight})

        assert merge_adapter(model, [(adapter, 0.5)]) == 1

        assert isinstance(model.weight, GGUFParameter)
        assert model.weight.quant_type == quant_type
        torch.testing.assert_close(model.weight.as_tensor(), packed)


class TestDirectConversion:
    @CUDA
    def test_external_parameter_activates_as_convrot(self) -> None:
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        weight, packed, quant_type = _quantized_weight(1)
        model = nn.Linear(64, 64, bias=False)
        model.weight = weight
        offloader = ModelOffloader.from_module(model)

        offloader.activate("cuda")
        try:
            actual = param_representation(model.weight)
            expected = ConvRotInt8Tensor.from_gguf(
                packed.cuda(),
                quant_type=quant_type,
                group_size=64,
            )
            assert isinstance(actual, ConvRotInt8Tensor)
            torch.testing.assert_close(actual.qdata, expected.qdata)
            torch.testing.assert_close(actual.scale, expected.scale)
        finally:
            offloader.deactivate()

        assert isinstance(model.weight, GGUFParameter)

    @CUDA
    def test_parameter_value_activates_as_convrot(self) -> None:
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        weight, packed, quant_type = _quantized_weight(2)
        model = nn.Linear(
            64,
            64,
            bias=False,
            device="meta",
            dtype=torch.bfloat16,
        )
        model.weight.requires_grad_(False)
        offloader = ModelOffloader.from_module(model)
        adapter = Adapter.from_state_dict(
            {"weight": weight},
        )

        offloader.activate(
            "cuda",
            adapters=[adapter],
            stochastic_rounding=False,
        )
        try:
            actual = param_representation(model.weight)
            expected = ConvRotInt8Tensor.from_gguf(
                packed.cuda(),
                quant_type=quant_type,
                group_size=64,
            )
            assert isinstance(actual, ConvRotInt8Tensor)
            torch.testing.assert_close(actual.qdata, expected.qdata)
            torch.testing.assert_close(actual.scale, expected.scale)
        finally:
            offloader.deactivate()

        assert model.weight.is_meta

    @CUDA
    def test_scaled_parameter_value_uses_convrot_dense_merge(self) -> None:
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        weight, packed, quant_type = _quantized_weight(3)
        model = nn.Linear(
            64,
            64,
            bias=False,
            device="meta",
            dtype=torch.bfloat16,
        )
        model.weight.requires_grad_(False)
        offloader = ModelOffloader.from_module(model)
        adapter = Adapter.from_state_dict(
            {"weight": weight},
            scale_parameter_values=True,
        )

        offloader.activate(
            "cuda",
            adapters=[adapter],
            adapter_strengths=[0.5],
            stochastic_rounding=False,
        )
        try:
            actual = param_representation(model.weight)
            expected = ConvRotInt8Tensor.from_gguf(
                packed.cuda(),
                quant_type=quant_type,
                group_size=64,
            )
            dense = expected.dequantize()
            expected.addmm_(
                torch.zeros(expected.shape[0], 1, dtype=expected.dtype, device=expected.device),
                torch.zeros(1, expected.shape[1], dtype=expected.dtype, device=expected.device),
                beta=0.5,
            )
            torch.testing.assert_close(actual.qdata, expected.qdata)
            torch.testing.assert_close(actual.scale, expected.scale)
        finally:
            offloader.deactivate()

    @CUDA
    def test_conversion_reuses_storage(self) -> None:
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        first, first_packed, first_quant_type = _quantized_weight(4, quant_type=2)
        second, second_packed, second_quant_type = _quantized_weight(5, quant_type=8)
        assert tuple(first_packed.shape) != tuple(second_packed.shape)
        _first_backing, state, actual = _materialize(first)
        assert not hasattr(state, "staging")
        expected = ConvRotInt8Tensor.from_gguf(
            first_packed.cuda(),
            quant_type=first_quant_type,
            group_size=64,
        )

        torch.testing.assert_close(actual.qdata, expected.qdata)
        torch.testing.assert_close(actual.scale, expected.scale)
        qdata_ptr = actual.qdata.data_ptr()
        scale_ptr = actual.scale.data_ptr()
        assert isinstance(second, nn.Parameter)
        second_backing = HostParam(second)
        second_backing.copy_to_gpu(state)
        torch.cuda.synchronize()
        expected_second = ConvRotInt8Tensor.from_gguf(
            second_packed.cuda(),
            quant_type=second_quant_type,
            group_size=64,
        )

        assert actual.qdata.data_ptr() == qdata_ptr
        assert actual.scale.data_ptr() == scale_ptr
        torch.testing.assert_close(actual.qdata, expected_second.qdata)
        torch.testing.assert_close(actual.scale, expected_second.scale)

    @CUDA
    def test_lora_merges_into_converted_target(self) -> None:
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        weight, packed, quant_type = _quantized_weight(6)
        assert isinstance(weight, nn.Parameter)
        source_param = weight
        backing = HostParam(source_param)
        rank = 4
        a = torch.randn(rank, 64)
        b = torch.randn(64, rank)
        strength = 0.25
        transform = LoRATransform(
            [ScaledLoRAFactor.from_tensors(a, b, strength)],
        )
        transform.validate_parameter(source_param)
        state = backing.allocate_gpu_storage(torch.device("cuda"))
        active = backing.make_gpu_param(state)
        backing.copy_to_gpu(state)

        transform.apply_parameter(active)
        actual = param_representation(active)
        expected = ConvRotInt8Tensor.from_gguf(
            packed.cuda(),
            quant_type=quant_type,
            group_size=64,
        )
        expected.addmm_(
            b.to(device="cuda", dtype=torch.bfloat16),
            a.to(device="cuda", dtype=torch.bfloat16),
            alpha=strength,
        )

        torch.testing.assert_close(actual.qdata, expected.qdata)
        torch.testing.assert_close(actual.scale, expected.scale)

    @CUDA
    def test_dense_delta_merges_into_converted_target(self) -> None:
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        weight, packed, quant_type = _quantized_weight(7)
        assert isinstance(weight, nn.Parameter)
        source_param = weight
        backing = HostParam(source_param)
        dense = torch.randn(64, 64)
        strength = 0.125
        transform = ParameterDeltaTransform(
            [
                ParameterDelta.from_tensors(
                    dense=dense,
                ).scaled(strength)
            ]
        )
        transform.validate_parameter(source_param)
        state = backing.allocate_gpu_storage(torch.device("cuda"))
        active = backing.make_gpu_param(state)
        backing.copy_to_gpu(state)

        transform.apply_parameter(active)
        actual = param_representation(active)
        expected = ConvRotInt8Tensor.from_gguf(
            packed.cuda(),
            quant_type=quant_type,
            group_size=64,
        )
        expected.add_(
            dense.to(device="cuda", dtype=torch.bfloat16),
            alpha=strength,
        )

        torch.testing.assert_close(actual.qdata, expected.qdata)
        torch.testing.assert_close(actual.scale, expected.scale)

    @CUDA
    def test_mixed_quant_streaming_blocks_share_two_target_pool(self) -> None:
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        class _BlockModel(nn.Module):
            def __init__(self, weights: tuple[nn.Parameter, ...]) -> None:
                super().__init__()
                blocks = []
                for w in weights:
                    layer = nn.Linear(64, 64, bias=False)
                    layer.weight = w
                    blocks.append(layer)
                self.blocks = nn.ModuleList(blocks)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.blocks[0](x)
                for block in self.blocks[1:]:
                    out = out + block(x)
                return out

        items = (
            _quantized_weight(21, quant_type=2),
            _quantized_weight(22, quant_type=8),
            _quantized_weight(23, quant_type=2),
        )
        weights = tuple(item[0] for item in items)
        assert all(isinstance(w, nn.Parameter) for w in weights)
        model = _BlockModel(weights)  # type: ignore[arg-type]

        x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
        expected = torch.zeros((8, 64), device="cuda", dtype=torch.bfloat16)
        for _w, packed, qt in items:
            decoded = ConvRotInt8Tensor.from_gguf(
                packed.cuda(), quant_type=qt, group_size=64
            )
            expected = expected + F.linear(x, decoded)

        seen_ptrs: list[int] = []
        handles = [
            block.register_forward_hook(
                lambda mod, _inp, _out: seen_ptrs.append(
                    param_representation(mod.weight).qdata.data_ptr()
                )
            )
            for block in model.blocks
        ]

        offloader = ModelOffloader.from_module(
            model,
            OffloadConfig(offload_paths=("blocks",)),
        )
        offloader.activate("cuda")
        try:
            with torch.inference_mode():
                actual = model(x)
                torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected)
            # Active + lookahead share a 2-target pool across Q4_0 and Q8_0 blocks.
            assert len(seen_ptrs) == 3
            assert len(set(seen_ptrs)) == 2
            assert seen_ptrs[2] == seen_ptrs[0]
        finally:
            for handle in handles:
                handle.remove()
            offloader.deactivate()

    @CUDA
    def test_resident_activation_retains_no_permanent_staging_vram(self) -> None:
        items = [_quantized_weight(200 + idx, rows=512, features=512, quant_type=2) for idx in range(8)]
        layers = []
        for weight, _packed, _qt in items:
            assert isinstance(weight, nn.Parameter)
            layer = nn.Linear(512, 512, bias=False)
            layer.weight = weight
            layers.append(layer)
        model = nn.Sequential(*layers)
        offloader = ModelOffloader.from_module(model)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        offloader.activate("cuda")
        try:
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() - baseline
            target_bytes = sum(
                param_representation(layer.weight).qdata.nbytes
                + param_representation(layer.weight).scale.nbytes
                for layer in layers
            )
            packed_bytes = sum(packed.nbytes for _w, packed, _qt in items)
            assert allocated <= target_bytes + 16 * 1024
            assert allocated < target_bytes + packed_bytes // 2
        finally:
            offloader.deactivate()
