"""Physical host storage enumeration without materialization or movement."""

import pytest
import torch
from torch import nn

from piper_offload import ParameterValue
from piper_offload.host_buffer import HostBuffer
from piper_offload.host_module import HostModuleStore
from piper_offload.host_param import HostParam
from tests.test_quantized_parameter_value import _QUANT_KINDS, _make_quantized


def _unexpected_materialization(*args: object, **kwargs: object) -> None:
    raise AssertionError("storage enumeration must not materialize tensors")


@pytest.mark.parametrize("kind", _QUANT_KINDS)
def test_quantized_storage_is_complete_and_reuses_backing(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backing = ParameterValue.from_tensor(_make_quantized(kind)).backing
    expected_bytes = backing.cache_bytes
    adapter_type = type(backing.adapter)
    for method in ("capture_host", "cpu_param", "alloc_gpu", "dequantize"):
        if hasattr(adapter_type, method):
            monkeypatch.setattr(adapter_type, method, staticmethod(_unexpected_materialization))

    tensors = backing.storage_tensors()
    repeated = backing.storage_tensors()

    assert tensors
    assert all(type(tensor) is torch.Tensor for tensor in tensors)
    assert all(tensor.device.type == "cpu" and tensor.layout is torch.strided for tensor in tensors)
    assert all(not tensor.is_pinned() for tensor in tensors)
    assert sum(tensor.nbytes for tensor in tensors) == expected_bytes
    assert len({id(tensor) for tensor in tensors}) == len(tensors)
    assert tuple(map(id, repeated)) == tuple(map(id, tensors))


@pytest.mark.parametrize("shape", [(3, 4), (0, 4)])
def test_plain_storage_reuses_backing_including_empty_tensors(
    shape: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = nn.Parameter(torch.empty(shape), requires_grad=False)
    backing = HostParam(parameter)
    monkeypatch.setattr(HostParam, "make_cpu_param", _unexpected_materialization)

    (tensor,) = backing.storage_tensors()

    assert tensor is backing.host_state.data
    assert tensor.untyped_storage().data_ptr() == parameter.untyped_storage().data_ptr()
    assert tensor.shape == shape
    assert tensor.nbytes == backing.cache_bytes


def test_meta_parameter_has_no_physical_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    backing = HostParam(nn.Parameter(torch.empty(3, 4, device="meta"), requires_grad=False))
    monkeypatch.setattr(type(backing.adapter), "storage_tensors", _unexpected_materialization)

    assert backing.storage_tensors() == ()


def test_buffer_enumeration_preserves_shared_storage_and_view_layout() -> None:
    backing = HostBuffer.capture(torch.arange(12))
    view = backing.tensor[3::2]
    viewed = HostBuffer(view, HostBuffer.target_layout_for(view))

    (whole,) = backing.storage_tensors()
    (part,) = viewed.storage_tensors()

    assert whole is backing.tensor
    assert part is view
    assert part.untyped_storage().data_ptr() == whole.untyped_storage().data_ptr()
    assert part.data_ptr() != whole.data_ptr()
    assert part.storage_offset() == 3
    assert part.stride() == (2,)
    assert part.untyped_storage().nbytes() > part.nbytes


def test_tied_names_enumerate_the_same_backing() -> None:
    module = nn.Module()
    module.weight = nn.Parameter(torch.randn(3, 4), requires_grad=False)
    module.tied_weight = module.weight
    buffer = torch.randn(4)
    module.register_buffer("buffer", buffer)
    module.register_buffer("tied_buffer", buffer)
    store = HostModuleStore.from_module(module)

    assert store.params["weight"].storage_tensors()[0] is store.params["tied_weight"].storage_tensors()[0]
    assert store.buffers["buffer"].storage_tensors()[0] is store.buffers["tied_buffer"].storage_tensors()[0]
