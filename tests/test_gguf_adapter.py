"""Tests for direct GGUF-to-ConvRot INT8 offload."""

from typing import Any

import pytest
import torch
from torch import nn

from piper_offload import Adapter, ModelOffloader
from piper_offload.gguf_adapter import GgufAdapter
from piper_offload.lora import LoRATransform, ScaledLoRAFactor
from piper_offload.parameter_delta import ParameterDelta, ParameterDeltaTransform
from piper_offload.pinned_param import PinnedParam
from piper_offload.tensor_adapter_registry import param_representation
from tests._gguf_helpers import GGUFParameter

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
) -> tuple[torch.Tensor, torch.Tensor, int]:
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    dense = np.random.default_rng(seed).standard_normal(
        (rows, features),
        dtype=np.float32,
    )
    quant_type = int(gguf.GGMLQuantizationType.Q4_0)
    packed = torch.from_numpy(gguf.quantize(dense, gguf.GGMLQuantizationType.Q4_0))
    return (
        GGUFParameter(packed, requires_grad=False, quant_type=quant_type),
        packed,
        quant_type,
    )


def _materialize(weight: torch.Tensor) -> tuple[PinnedParam, Any, torch.Tensor]:
    assert isinstance(weight, nn.Parameter)
    pinned = PinnedParam(weight, pin_memory=False)
    state = pinned.allocate_gpu_storage(torch.device("cuda"))
    param = pinned.make_gpu_param(state)
    pinned.copy_to_gpu(state)
    torch.cuda.synchronize()
    return pinned, state, param_representation(param)


class TestGGUFSource:
    def test_adapter_derives_logical_layout(self, w: GGUFParameter) -> None:
        assert GgufAdapter.logical_shape(w) == (4, 64)
        assert GgufAdapter.compute_dtype(w) is torch.bfloat16

        state = GgufAdapter.adopt_host(w)
        assert state.data.data_ptr() == w.as_tensor().data_ptr()
        assert state.quant_type == QUANT_TYPE
        assert state.logical_shape == (4, 64)
        assert state.group_size == 64
        assert GgufAdapter.cache_bytes(state) == state.data.nbytes

    def test_matches_existing_gguf_parameter_contract(self) -> None:
        weight, packed, _quant_type = _quantized_weight(0)
        assert isinstance(weight, GGUFParameter)
        assert GgufAdapter.matches(weight)

        state = GgufAdapter.adopt_host(weight)
        rebuilt = GgufAdapter.cpu_param(state)
        assert isinstance(rebuilt, GGUFParameter)
        assert rebuilt.quant_type == weight.quant_type
        assert rebuilt.quant_shape == weight.quant_shape
        assert rebuilt.as_tensor().data_ptr() == packed.data_ptr()

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

        assert GgufAdapter.tensor_id(w) == GgufAdapter.tensor_id(alias)
        assert GgufAdapter.tensor_id(w) != GgufAdapter.tensor_id(clone)

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

        state = GgufAdapter.adopt_host(weight)

        assert state.group_size == expected_group_size

    def test_rejects_width_without_supported_group(self) -> None:
        weight = GGUFParameter(
            torch.zeros((4, 7)),
            requires_grad=False,
            quant_type=0,
        )

        with pytest.raises(ValueError, match="in_features divisible by 16"):
            GgufAdapter.adopt_host(weight)


class TestDirectConversion:
    @CUDA
    def test_external_parameter_activates_as_convrot(self) -> None:
        from piper_kernels.linear.convrot import ConvRotInt8Tensor

        weight, packed, quant_type = _quantized_weight(1)
        model = nn.Linear(64, 64, bias=False)
        model.weight = weight
        offloader = ModelOffloader.from_module(model, host_backing="adopt")

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
        from piper_kernels.linear.convrot import ConvRotInt8Tensor

        weight, packed, quant_type = _quantized_weight(2)
        model = nn.Linear(
            64,
            64,
            bias=False,
            device="meta",
            dtype=torch.bfloat16,
        )
        model.weight.requires_grad_(False)
        offloader = ModelOffloader.from_module(model, host_backing="adopt")
        adapter = Adapter.from_state_dict(
            {"weight": weight},
            host_backing="adopt",
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
        from piper_kernels.linear.convrot import ConvRotInt8Tensor

        weight, packed, quant_type = _quantized_weight(3)
        model = nn.Linear(
            64,
            64,
            bias=False,
            device="meta",
            dtype=torch.bfloat16,
        )
        model.weight.requires_grad_(False)
        offloader = ModelOffloader.from_module(model, host_backing="adopt")
        adapter = Adapter.from_state_dict(
            {"weight": weight},
            host_backing="adopt",
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
            expected.add_(dense, alpha=-0.5)
            torch.testing.assert_close(actual.qdata, expected.qdata)
            torch.testing.assert_close(actual.scale, expected.scale)
        finally:
            offloader.deactivate()

    @CUDA
    def test_conversion_reuses_storage(self) -> None:
        from piper_kernels.linear.convrot import ConvRotInt8Tensor

        first, first_packed, first_quant_type = _quantized_weight(4)
        second, second_packed, second_quant_type = _quantized_weight(5)
        first_backing, state, actual = _materialize(first)
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
        second_backing = PinnedParam(second, pin_memory=False)
        second_backing.copy_to_gpu(state)
        torch.cuda.synchronize()
        expected_second = ConvRotInt8Tensor.from_gguf(
            second_packed.cuda(),
            quant_type=second_quant_type,
            group_size=64,
        )

        assert actual.qdata.data_ptr() == qdata_ptr
        assert actual.scale.data_ptr() == scale_ptr
        assert tuple(state.staging.shape) == tuple(first_packed.shape)
        assert state.staging.nbytes < actual.dequantize().nbytes
        torch.testing.assert_close(actual.qdata, expected_second.qdata)
        torch.testing.assert_close(actual.scale, expected_second.scale)
        with pytest.raises(NotImplementedError, match="CPU round-trip"):
            first_backing.copy_to_cpu(state)

    @CUDA
    def test_lora_merges_into_converted_target(self) -> None:
        from piper_kernels.linear.convrot import ConvRotInt8Tensor

        weight, packed, quant_type = _quantized_weight(6)
        assert isinstance(weight, nn.Parameter)
        source_param = weight
        backing = PinnedParam(source_param, pin_memory=False)
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
        from piper_kernels.linear.convrot import ConvRotInt8Tensor

        weight, packed, quant_type = _quantized_weight(7)
        assert isinstance(weight, nn.Parameter)
        source_param = weight
        backing = PinnedParam(source_param, pin_memory=False)
        dense = torch.randn(64, 64)
        strength = 0.125
        transform = ParameterDeltaTransform(
            [
                ParameterDelta.from_tensors(
                    dense=dense,
                    pin_memory=False,
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
