"""Real CPU compilation over mapped weights and across device activations."""

import pytest
import torch
from torch import nn

from piper_offload import BlockCompileConfig, MappedCheckpoint, file_slice
from piper_offload.pin_manager import host_pin_manager
from tests._block_compile_helpers import _BlockModel, _make_offloader
from tests.conftest import activated_model, block_components
from tests.test_checkpoint import _write


@pytest.mark.parametrize("block_mode", ["streaming", "resident", "rolling", "auto"])
def test_cpu_compilation_preserves_quantized_checkpoint_storage(tmp_path, monkeypatch, block_mode):
    int8 = pytest.importorskip("piper_kernels.weights.convrot.int8")
    torch.manual_seed(2081)
    model = _BlockModel(width=64)
    storage = {}
    for index, block in enumerate(model.blocks):
        weight = int8.ConvRotInt8Tensor.from_hp(block.proj.weight, group_size=16)
        storage[f"{index}.qdata"] = weight.qdata
        storage[f"{index}.scale"] = weight.scale
    path = tmp_path / "weights.safetensors"
    _write(path, storage)
    original_bytes = path.read_bytes()
    with MappedCheckpoint(path) as reader:
        for index, block in enumerate(model.blocks):
            block.proj.weight = nn.Parameter(
                int8.ConvRotInt8Tensor.from_quantized(
                    reader.get_tensor(f"{index}.qdata"), reader.get_tensor(f"{index}.scale"),
                    group_size=16, logical_dtype=torch.float32,
                ),
                requires_grad=False,
            )

    def backing():
        return [
            (tensor.data_ptr(), file_slice(tensor))
            for block in model.blocks
            for tensor in (block.proj.weight.qdata, block.proj.weight.scale)
        ]

    original_backing = backing()
    assert all(piece is not None for _pointer, piece in original_backing)
    inputs = [torch.randn(2, 64), torch.randn(3, 64)]
    with torch.inference_mode():
        expected = [model(value) for value in inputs]

    def forbid_pin(*args, **kwargs):
        pytest.fail("CPU block compilation must not acquire a pin lease")

    monkeypatch.setattr(host_pin_manager, "acquire", forbid_pin)
    offloader = _make_offloader(
        model, block_mode=block_mode, block_compile=BlockCompileConfig(fullgraph=True),
    )
    assert backing() == original_backing
    for _ in range(2):
        with activated_model(offloader, "cpu"), torch.inference_mode():
            for value, reference in zip(inputs, expected, strict=True):
                torch.testing.assert_close(model(value), reference, rtol=1e-4, atol=1e-5)
            component = block_components(offloader)[0]
            assert component._active_runtime is None
            assert component._inductor_compile.installed
            assert backing() == original_backing
        assert all("forward" not in block.__dict__ for block in model.blocks)
        assert backing() == original_backing
    assert path.read_bytes() == original_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("block_mode", ["streaming", "resident", "rolling", "auto"])
def test_compilation_switches_between_cpu_and_cuda(block_mode):
    torch.manual_seed(2082)
    model = _BlockModel()
    value = torch.randn(3, 8)
    with torch.inference_mode():
        expected = model(value)
    offloader = _make_offloader(
        model, block_mode=block_mode, block_compile=BlockCompileConfig(fullgraph=True),
    )
    host_pointers = [param.data_ptr() for param in model.parameters()]
    for device in ("cpu", "cuda", "cpu", "cuda"):
        with activated_model(offloader, device), torch.inference_mode():
            actual = model(value.to(device)).cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            assert all("forward" in block.__dict__ for block in model.blocks)
        assert all("forward" not in block.__dict__ for block in model.blocks)
        assert [param.data_ptr() for param in model.parameters()] == host_pointers
