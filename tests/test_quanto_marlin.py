"""Real optimum-quanto Marlin FP8 adapter lifecycle tests."""

import pytest
import torch
from torch import nn

import piper_offload.quanto_adapter as quanto_adapter_module
from piper_offload import Adapter, ModelOffloader
from piper_offload.host_param import HostParam
from piper_offload.quanto_adapter import QuantoAdapter
from piper_offload.tensor_adapter_registry import tensor_id
from tests.conftest import activated_model

try:
    from optimum import quanto
    from optimum.quanto.tensor.weights.marlin.fp8 import (
        MarlinF8QBytesTensor,
    )
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor
except ImportError:
    quanto = None
    MarlinF8QBytesTensor = None
    WeightQBytesTensor = None


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
pytestmark = pytest.mark.skipif(
    quanto is None,
    reason="optimum-quanto is required",
)


def _make_marlin_model() -> nn.Sequential:
    assert quanto is not None
    assert MarlinF8QBytesTensor is not None
    model = nn.Sequential(
        nn.Linear(
            128,
            128,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
        )
    )
    quanto.quantize(model, weights=quanto.qfloat8)
    quanto.freeze(model)
    model[0].weight.requires_grad_(False)
    if not isinstance(model[0].weight.data, MarlinF8QBytesTensor):
        pytest.skip("optimum-quanto Marlin FP8 kernel is unavailable")
    return model


def _assert_qbytes_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    torch.testing.assert_close(actual._scale, expected._scale)
    torch.testing.assert_close(
        actual._data.to(torch.float32),
        expected._data.to(torch.float32),
        rtol=0.15,
        atol=2.0,
    )


def _quanto_absmax_oracle(
    dense: torch.Tensor,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    """Use Quanto's optimizer and quantizer as the merge oracle."""
    from optimum.quanto.tensor.optimizers import AbsmaxOptimizer

    assert WeightQBytesTensor is not None
    canonical = like.weight_qbytes_tensor()
    scale = AbsmaxOptimizer()(
        dense,
        qtype=canonical.qtype,
        axis=canonical.axis,
    ).to(canonical._scale.dtype)
    zero = scale == 0
    eps = torch.finfo(torch.float32).eps
    scale = torch.where(zero, torch.full_like(scale, eps), scale)
    return WeightQBytesTensor.quantize(
        dense,
        canonical.qtype,
        canonical.axis,
        scale,
        canonical.activation_qtype,
        optimized=False,
    )


class TestQuantoMarlin:
    @CUDA
    def test_tensor_id_uses_marlin_physical_backing(self) -> None:
        assert MarlinF8QBytesTensor is not None
        model = _make_marlin_model()
        first = model[0].weight.data
        packed_copy = type(first._data)(
            first._data._data.clone(),
            first._data.size(),
            first._data.stride(),
        )
        second = MarlinF8QBytesTensor(
            first.qtype,
            first.axis,
            first.size(),
            first.stride(),
            packed_copy,
            first._scale,
        )

        first_id = tensor_id(first)
        second_id = tensor_id(second)
        assert first._scale.data_ptr() == second._scale.data_ptr()
        assert first_id[2] == first._data._data.data_ptr()
        assert second_id[2] == second._data._data.data_ptr()
        assert first_id != second_id

    @CUDA
    def test_capture_canonicalizes_marlin_before_gpu_wrapper_construction(
        self,
    ) -> None:
        assert WeightQBytesTensor is not None
        model = _make_marlin_model()
        marlin = model[0].weight.data
        original = marlin.dequantize().clone()
        assert marlin.activation_qtype is quanto.qfloat8

        host = HostParam(model[0].weight)
        assert host.host_state.data.dtype is torch.float8_e4m3fn
        assert tuple(host.host_state.data.shape) == (128, 128)
        assert tuple(host.host_state.scale.shape) == (128, 1)

        cpu_param = host.make_cpu_param()
        assert type(cpu_param.data) is WeightQBytesTensor
        assert cpu_param.data._data.data_ptr() == host.host_state.data.data_ptr()
        assert cpu_param.data._scale.data_ptr() == host.host_state.scale.data_ptr()
        assert cpu_param.data.activation_qtype is marlin.activation_qtype
        assert torch.equal(cpu_param.data.dequantize().cuda(), original)

        gpu_state = host.allocate_gpu_storage(torch.device("cuda"))
        # Pool targets construct their wrapper before the first H2D fill.
        gpu_param = host.make_gpu_param(gpu_state)
        assert type(gpu_param.data) is WeightQBytesTensor
        assert gpu_param.data._data.data_ptr() == gpu_state.data.data_ptr()
        assert gpu_param.data._scale.data_ptr() == gpu_state.scale.data_ptr()
        assert gpu_param.data.activation_qtype is marlin.activation_qtype
        host.copy_to_gpu(gpu_state)
        torch.cuda.synchronize()

        assert torch.equal(gpu_param.data.dequantize(), original)
        assert HostParam.target_layout_for(gpu_param) == host.target_layout

    @CUDA
    def test_copy_into_repacks_existing_marlin_storage_in_place(self) -> None:
        assert MarlinF8QBytesTensor is not None
        model = _make_marlin_model()
        target = model[0].weight.data
        original = target.dequantize()
        updated = QuantoAdapter.requantize(
            original + torch.randn_like(original) * 0.125,
            like=target,
        )
        packed_ptr = target._data._data.data_ptr()
        scale_ptr = target._scale.data_ptr()
        workspace_ptr = target._workspace.data_ptr()
        packed_before = target._data._data.clone()

        QuantoAdapter.copy_into(updated, target=target)
        torch.cuda.synchronize()

        assert isinstance(target, MarlinF8QBytesTensor)
        assert target._data._data.data_ptr() == packed_ptr
        assert target._scale.data_ptr() == scale_ptr
        assert target._workspace.data_ptr() == workspace_ptr
        assert not torch.equal(target._data._data, packed_before)
        canonical = target.weight_qbytes_tensor()
        assert torch.equal(canonical._data, updated._data)
        assert torch.equal(canonical._scale, updated._scale)

    @CUDA
    def test_merge_lora_uses_generic_repack_for_existing_marlin_storage(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert MarlinF8QBytesTensor is not None
        model = _make_marlin_model()
        target = model[0].weight.data
        rank = 7
        a = torch.randn(
            rank,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        b = torch.randn(
            128,
            rank,
            device="cuda",
            dtype=torch.bfloat16,
        )
        expected_dense = target.dequantize()
        expected_dense.addmm_(b, a, alpha=-0.1875)
        expected = _quanto_absmax_oracle(expected_dense, like=target)
        packed_ptr = target._data._data.data_ptr()
        scale_ptr = target._scale.data_ptr()
        workspace_ptr = target._workspace.data_ptr()
        packed_before = target._data._data.clone()

        def fail_triton(*_args: object) -> torch.Tensor:
            raise AssertionError("packed Marlin FP8 must not use raw Triton")

        monkeypatch.setattr(
            quanto_adapter_module,
            "_triton_merge_quanto_qfloat8_lora",
            fail_triton,
        )

        QuantoAdapter.merge_lora_(target, b, a, -0.1875)
        torch.cuda.synchronize()

        assert isinstance(target, MarlinF8QBytesTensor)
        assert target._data._data.data_ptr() == packed_ptr
        assert target._scale.data_ptr() == scale_ptr
        assert target._workspace.data_ptr() == workspace_ptr
        assert not torch.equal(target._data._data, packed_before)
        _assert_qbytes_close(target.weight_qbytes_tensor(), expected)

    @CUDA
    def test_model_offloader_canonicalizes_and_merges_real_marlin_weight(
        self,
    ) -> None:
        assert WeightQBytesTensor is not None
        model = _make_marlin_model()
        target = model[0].weight.data
        rank = 7
        strength = 0.25
        a = torch.randn(rank, 128)
        b = torch.randn(128, rank)
        expected_dense = target.dequantize()
        expected_dense.addmm_(
            b.cuda().to(expected_dense.dtype),
            a.cuda().to(expected_dense.dtype),
            alpha=strength,
        )
        expected = _quanto_absmax_oracle(expected_dense, like=target)
        lora = Adapter.from_state_dict(
            state_dict={
                "0.lora_A.weight": a,
                "0.lora_B.weight": b,
            }
        )
        offloader = ModelOffloader.from_module(model)

        assert type(model[0].weight.data) is WeightQBytesTensor
        assert model[0].weight.data._data.device.type == "cpu"
        assert tuple(model[0].weight.data._scale.shape) == (128, 1)

        x = torch.randn(
            3,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        with activated_model(
            offloader,
            "cuda",
            adapters=[lora],
            adapter_strengths=[strength],
            adapter_mode="merge",
        ) as active:
            merged = active[0].weight.data
            assert type(merged) is WeightQBytesTensor
            assert merged._data.is_cuda
            assert tuple(merged._scale.shape) == (128, 1)
            _assert_qbytes_close(merged, expected)
            output = active(x)
            reference = torch.nn.functional.linear(
                x,
                merged.dequantize(),
            )
            torch.cuda.synchronize()

        assert output.shape == (3, 128)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(
            output,
            reference,
            rtol=0.02,
            atol=0.02,
        )
        assert type(model[0].weight.data) is WeightQBytesTensor
        assert model[0].weight.data._data.device.type == "cpu"


class TestQuantoScaleLayout:
    @CUDA
    def test_reordered_axis_zero_scale_is_rejected_before_merge(
        self,
    ) -> None:
        assert WeightQBytesTensor is not None
        rows = cols = 8
        target = WeightQBytesTensor(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            torch.randint(
                -32,
                33,
                (rows, cols),
                device="cuda",
                dtype=torch.int8,
            ),
            torch.rand(
                1,
                rows,
                device="cuda",
                dtype=torch.bfloat16,
            ).add_(0.25),
            None,
        )
        a = torch.randn(3, cols, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(rows, 3, device="cuda", dtype=torch.bfloat16)

        with pytest.raises(ValueError, match=r"shape \(rows, 1\)"):
            QuantoAdapter.validate_lora_merge(target, b, a, 0.25)
