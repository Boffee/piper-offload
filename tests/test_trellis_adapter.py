"""Tests for trellis-coded ConvRot INT8 offload."""

import pytest
import torch
from torch import nn

from piper_offload import Adapter, ModelOffloader, TrellisParameter, merge_adapter
from piper_offload.host_param import HostParam
from piper_offload.lora import LoRATransform, ScaledLoRAFactor
from piper_offload.tensor_adapter_registry import (
    param_representation,
    requires_activation,
    select_adapter,
)
from piper_offload.trellis_adapter import TrellisAdapter

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm required")
ROWS = 64
BLOCK_SIZE = 256
GROUP_SIZE = 256


def _weight(
    seed: int, *, blocks: int = 1, num_bytes: int = 143, act_scale: float | None = None
) -> TrellisParameter:
    """Arbitrary trellis storage: every bit pattern is a valid walk."""
    generator = torch.Generator().manual_seed(seed)
    bitstreams = torch.randint(
        0, 256, (ROWS, blocks, num_bytes), dtype=torch.uint8, generator=generator
    )
    # E4M3 codes 0x20-0x3F are finite scales from 0.125 to 1.875.
    codes = torch.randint(0x20, 0x40, (ROWS, blocks), dtype=torch.uint8, generator=generator)
    scale = torch.rand(ROWS, 1, generator=generator) * 1e-3 + 1e-4
    return TrellisParameter(
        bitstreams,
        codes.view(torch.float8_e4m3fn),
        scale,
        block_size=BLOCK_SIZE,
        group_size=GROUP_SIZE,
        act_per_tensor_scale=None if act_scale is None else torch.tensor(act_scale),
    )


def _decode(weight: TrellisParameter) -> torch.Tensor:
    from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

    return ConvRotInt8Tensor.from_trellis(
        weight.bitstreams.cuda(),
        weight.block_scales.cuda(),
        weight.scale.cuda(),
        block_size=weight.block_size,
        group_size=weight.group_size,
    )


class TestTrellisSource:
    def test_parameter_presents_the_logical_weight(self) -> None:
        weight = _weight(0, blocks=2)

        assert weight.shape == (ROWS, 2 * BLOCK_SIZE)
        assert weight.dtype is torch.bfloat16
        assert weight.storage_tensors() == (weight.bitstreams, weight.block_scales, weight.scale)
        assert "block_size=256" in repr(weight)
        assert requires_activation(weight)
        assert isinstance(select_adapter(weight), TrellisAdapter)

    def test_detach_shares_and_clone_copies_storage(self) -> None:
        weight = _weight(1, act_scale=0.05)

        detached, cloned = weight.detach(), weight.clone()

        for copy, shares in ((detached, True), (cloned, False)):
            assert isinstance(copy, TrellisParameter)
            assert copy.shape == weight.shape
            for a, b in zip(copy.storage_tensors(), weight.storage_tensors(), strict=True):
                assert (a.data_ptr() == b.data_ptr()) is shares
                assert torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))

    def test_other_operations_require_split_storage_or_activation(self) -> None:
        with pytest.raises(NotImplementedError, match="split checkpoint storage"):
            _weight(2)[:8]

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"bitstreams": torch.zeros(ROWS, 1, 143)}, "uint8"),
            ({"block_scales": torch.zeros(ROWS, 2, dtype=torch.float8_e4m3fn)}, "block_scales"),
            ({"scale": torch.zeros(ROWS)}, "scale must be float32"),
        ],
    )
    def test_parameter_rejects_inconsistent_storage(
        self, change: dict[str, torch.Tensor], message: str
    ) -> None:
        weight = _weight(3)
        tensors = {
            "bitstreams": weight.bitstreams,
            "block_scales": weight.block_scales,
            "scale": weight.scale,
        } | change

        with pytest.raises(ValueError, match=message):
            TrellisParameter(**tensors, block_size=BLOCK_SIZE, group_size=GROUP_SIZE)

    def test_adapter_captures_packed_storage(self) -> None:
        weight = _weight(4, act_scale=0.05)

        state = TrellisAdapter.capture_host(weight)

        captured = TrellisAdapter.storage_tensors(state)
        assert [t.data_ptr() for t in captured] == [t.data_ptr() for t in weight.storage_tensors()]
        # Bitstreams, FP8 block scales, FP32 row scales, then the FP32 activation scale.
        assert TrellisAdapter.cache_bytes(state) == ROWS * (143 + 1 + 4) + 4
        first, second = TrellisAdapter.cpu_param(state), TrellisAdapter.cpu_param(state)
        assert isinstance(first, TrellisParameter)
        assert first is not second
        assert [t.data_ptr() for t in first.storage_tensors()] == [t.data_ptr() for t in captured]
        assert TrellisAdapter.logical_shape(weight) == (ROWS, BLOCK_SIZE)
        assert TrellisAdapter.compute_dtype(weight) is torch.bfloat16

    def test_tensor_id_and_layout_follow_storage_and_rate(self) -> None:
        weight = _weight(5)
        other_rate = _weight(5, num_bytes=111)

        assert TrellisAdapter.tensor_id(weight) == TrellisAdapter.tensor_id(weight.detach())
        assert TrellisAdapter.tensor_id(weight) != TrellisAdapter.tensor_id(weight.clone())
        assert TrellisAdapter.layout_signature(weight) == TrellisAdapter.layout_signature(
            weight.clone()
        )
        assert TrellisAdapter.layout_signature(weight) != TrellisAdapter.layout_signature(
            other_rate
        )

    def test_rejects_permanent_merge(self) -> None:
        model = nn.Sequential(nn.Linear(BLOCK_SIZE, ROWS, bias=False))
        model[0].weight = _weight(6)
        adapter = Adapter.from_state_dict(
            {
                "0.lora_A.weight": torch.randn(4, BLOCK_SIZE),
                "0.lora_B.weight": torch.randn(ROWS, 4),
            }
        )

        with pytest.raises(ValueError, match="Permanent updates to trellis-coded"):
            merge_adapter(model, [(adapter, 1.0)])


class TestTrellisActivation:
    @CUDA
    def test_activation_decodes_into_convrot_int8(self) -> None:
        from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

        weight = _weight(7)
        model = nn.Linear(BLOCK_SIZE, ROWS, bias=False)
        model.weight = weight
        offloader = ModelOffloader.from_module(model)

        offloader.activate("cuda")
        try:
            actual = param_representation(model.weight)
            expected = _decode(weight)
            assert isinstance(actual, ConvRotInt8Tensor)
            assert actual.group_size == GROUP_SIZE
            assert torch.equal(actual.qdata, expected.qdata)
            assert torch.equal(actual.scale, expected.scale)
        finally:
            offloader.deactivate()

        assert isinstance(model.weight, TrellisParameter)

    @CUDA
    def test_refill_reuses_the_target_and_carries_the_activation_scale(self) -> None:
        first, second = _weight(8, act_scale=0.05), _weight(9, act_scale=0.25)
        backing = HostParam(first)
        state = backing.allocate_gpu_storage(torch.device("cuda"))
        actual = param_representation(backing.make_gpu_param(state))
        backing.copy_to_gpu(state)
        qdata_pointer = actual.qdata.data_ptr()

        HostParam(second).copy_to_gpu(state)
        torch.cuda.synchronize()

        expected = _decode(second)
        assert actual.qdata.data_ptr() == qdata_pointer
        assert torch.equal(actual.qdata, expected.qdata)
        assert torch.equal(actual.scale, expected.scale)
        assert float(actual.act_per_tensor_scale) == pytest.approx(0.25)
        assert sum(t.nbytes for t in state.staging) < actual.qdata.nbytes

    @CUDA
    def test_lora_merges_into_the_decoded_target(self) -> None:
        weight = _weight(10)
        backing = HostParam(weight)
        a, b, strength = torch.randn(4, BLOCK_SIZE), torch.randn(ROWS, 4), 0.25
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, strength)])
        transform.validate_parameter(weight)
        state = backing.allocate_gpu_storage(torch.device("cuda"))
        active = backing.make_gpu_param(state)
        backing.copy_to_gpu(state)

        transform.apply_parameter(active)

        actual = param_representation(active)
        expected = _decode(weight)
        expected.addmm_(
            b.to(device="cuda", dtype=torch.bfloat16),
            a.to(device="cuda", dtype=torch.bfloat16),
            alpha=strength,
        )
        torch.testing.assert_close(actual.qdata, expected.qdata)
        torch.testing.assert_close(actual.scale, expected.scale)
