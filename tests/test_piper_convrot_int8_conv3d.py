"""Static ConvRot INT8 convolution weights retain every scale through offload."""

import pytest
import torch
from torch import nn

from piper_offload._piper_convrot_int8 import create_convrot_int8_tensor
from piper_offload.host_param import HostParam
from piper_offload.piper_convrot_int8_adapter import PiperConvRotInt8Adapter
from piper_offload.tensor_adapter_registry import tensor_id

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _convrot_cls():
    return pytest.importorskip("piper_kernels.weights.convrot.int8").ConvRotInt8Tensor


class TestConv3dWeight:
    def _weight(self):
        return _convrot_cls().from_hp(
            torch.randn(8, 64, 3, 3, 3, dtype=torch.float16) * 0.02,
            group_size=64,
            act_per_tensor_scale=torch.tensor(0.02),
        )

    def test_mmap_capture_keeps_all_storage_and_counts_activation_scale(self, tmp_path):
        weight = self._weight()
        path = tmp_path / "weight.pt"
        torch.save(weight, path)
        with torch.serialization.safe_globals([_convrot_cls()]):
            loaded = torch.load(path, weights_only=True, mmap=True)
        host = HostParam(nn.Parameter(loaded, requires_grad=False))
        restored = host.make_cpu_param()
        assert tuple(restored.shape) == (8, 64, 3, 3, 3)
        assert host.cache_bytes == loaded.qdata.nbytes + loaded.scale.nbytes + 4
        assert len(host.storage_tensors()) == 3
        for name in ("qdata", "scale", "act_per_tensor_scale"):
            assert getattr(restored, name).data_ptr() == getattr(loaded, name).data_ptr()
            assert not getattr(restored, name).untyped_storage().resizable()
        torch.testing.assert_close(restored.dequantize(), weight.dequantize())

    def test_activation_scale_participates_in_identity_and_layout(self):
        weight = self._weight()
        original_id = tensor_id(weight)
        original_layout = PiperConvRotInt8Adapter.layout_signature(weight)
        weight.act_per_tensor_scale = torch.tensor(0.03)
        assert tensor_id(weight) != original_id
        assert PiperConvRotInt8Adapter.layout_signature(weight) == original_layout
        weight.act_per_tensor_scale = None
        assert PiperConvRotInt8Adapter.layout_signature(weight) != original_layout

    @CUDA
    def test_offloaded_weight_preserves_scale_and_convolution_execution(self):
        from piper_kernels.conv3d.convrot.int8 import ConvRotInt8Conv3d

        source = self._weight()
        host = HostParam(nn.Parameter(source, requires_grad=False))
        state = host.allocate_gpu_storage(torch.device("cuda"))
        host.copy_to_gpu(state)
        gpu = host.make_gpu_param(state)
        assert gpu.act_per_tensor_scale.is_cuda
        torch.testing.assert_close(gpu.act_per_tensor_scale.cpu(), source.act_per_tensor_scale)
        layer = ConvRotInt8Conv3d(gpu, padding="reflect")
        activation = torch.randn(1, 64, 2, 4, 4, device="cuda", dtype=torch.float16)
        expected = ConvRotInt8Conv3d(source.to(device="cuda"), padding="reflect")(activation)
        torch.testing.assert_close(layer(activation), expected, atol=0, rtol=0)

    @pytest.mark.parametrize("storage", ["qdata", "scale"])
    @pytest.mark.parametrize("operation", ["validate", "reconstruct"])
    def test_offload_rejects_storage_that_would_require_copying(self, storage, operation):
        weight = self._weight()
        if storage == "qdata":
            weight.qdata = weight.qdata.transpose(1, 2)
        else:
            weight.scale = torch.ones(weight.shape[0], 2)[:, ::2]
        assert not getattr(weight, storage).is_contiguous()
        with pytest.raises(ValueError, match="must be contiguous"):
            if operation == "validate":
                PiperConvRotInt8Adapter.matches(weight)
            else:
                create_convrot_int8_tensor(
                    weight.qdata,
                    weight.scale,
                    weight.group_size,
                    weight.dtype,
                    weight.act_per_tensor_scale,
                )

    @pytest.mark.parametrize("operation", ["dense", "lora"])
    def test_weight_updates_are_rejected_before_staging(self, operation):
        weight = self._weight()

        def validate():
            if operation == "dense":
                return PiperConvRotInt8Adapter.validate_dense_merge_target(weight)
            return PiperConvRotInt8Adapter.validate_lora_merge(weight, torch.ones(8, 1), torch.ones(1, 64), 1.0)

        with pytest.raises(NotImplementedError, match="updates require a 2-D weight"):
            validate()

    def test_offload_rejects_unrepresented_matrix_transpose(self):
        weight = _convrot_cls().from_hp(torch.ones(8, 64), group_size=64).t()
        with pytest.raises(NotImplementedError, match="untransposed weight"):
            HostParam(nn.Parameter(weight, requires_grad=False))
