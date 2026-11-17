"""GGUF mapping, logical parameters, and their pin/activation integration."""

import gc
import mmap
import struct
import sys
import weakref
from pathlib import Path

import pytest
import torch
from torch import nn

from piper_offload import (
    CheckpointError, GgufParameter, MappedCheckpoint, ModelOffloader, PinManager, file_slice, requires_activation,
)
from piper_offload.gguf_adapter import GgufAdapter
from tests._block_compile_helpers import _BlockModel
from tests.test_pin_manager import FakeBackend, _transferred
from tests.test_streaming_pinning import pins as pins  # noqa: PLC0414 — re-export a shared pytest fixture

gguf = pytest.importorskip("gguf")
np = pytest.importorskip("numpy")


def _write(writer: gguf.GGUFWriter) -> None:
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def checkpoint(tmp_path: Path):
    path = tmp_path / "model.gguf"
    dense = np.random.default_rng(42).standard_normal((64, 64), dtype=np.float32)
    packed = gguf.quantize(dense, gguf.GGMLQuantizationType.Q4_0)
    writer = gguf.GGUFWriter(path, "test")
    writer.add_tensor("weight", packed, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
    writer.add_tensor("weight2", packed[::-1].copy(), raw_dtype=gguf.GGMLQuantizationType.Q4_0)
    writer.add_tensor("bias", np.arange(64, dtype=np.float32))
    writer.add_tensor("half", np.arange(5, dtype=np.float16))
    bf16 = torch.arange(4, dtype=torch.bfloat16)
    writer.add_tensor("bf16", bf16.view(torch.uint8).numpy(), raw_dtype=gguf.GGMLQuantizationType.BF16)
    _write(writer)
    return path, torch.from_numpy(packed)


def test_reader_describes_logical_weights_and_actual_storage(checkpoint):
    path, packed = checkpoint
    with MappedCheckpoint(path) as reader:
        assert reader.metadata() is None
        descriptor = reader.get_slice("weight")
        assert descriptor.get_shape() == [64, 64]
        assert descriptor.get_dtype() == "BF16"
        assert descriptor.get_nbytes() == packed.nbytes
        assert descriptor.get_nbytes(dtype=torch.float32) == packed.nbytes
        weight = reader.get_tensor("weight")
        assert isinstance(weight, GgufParameter)
        assert weight.shape == (64, 64)
        assert weight.dtype is torch.bfloat16
        assert requires_activation(weight)
        assert not requires_activation(reader.get_tensor("bias"))
        torch.testing.assert_close(weight.as_tensor(), packed)
        for name, dtype, count in (
            ("bias", torch.float32, 64), ("half", torch.float16, 5), ("bf16", torch.bfloat16, 4),
        ):
            tensor = reader.get_tensor(name)
            assert type(tensor) is torch.Tensor
            torch.testing.assert_close(tensor, torch.arange(count, dtype=dtype))
            assert file_slice(tensor) is not None
            assert reader.get_slice(name).get_nbytes(dtype=torch.float32) == tensor.nbytes


def test_selective_read_can_skip_an_unsupported_encoding(tmp_path):
    path = tmp_path / "mixed.gguf"
    writer = gguf.GGUFWriter(path, "test")
    writer.add_tensor("supported", np.ones(4, dtype=np.float32))
    writer.add_tensor("unsupported", np.ones(4, dtype=np.int8))
    _write(writer)
    with MappedCheckpoint(path) as reader:
        assert reader.keys() == ["supported", "unsupported"]
        torch.testing.assert_close(reader.get_tensor("supported"), torch.ones(4))
        with pytest.raises(CheckpointError, match="unsupported GGUF quantization type"):
            reader.get_slice("unsupported")
        with pytest.raises(CheckpointError, match="unsupported GGUF quantization type"):
            reader.get_tensor("unsupported")


def test_header_matches_reference_parser_without_importing_it(tmp_path, monkeypatch):
    path = tmp_path / "model.gguf"
    writer = gguf.GGUFWriter(path, "test")
    writer.add_custom_alignment(1024)
    # Every metadata value type: a wrong size misplaces everything after it.
    for index, add in enumerate((
        writer.add_uint8, writer.add_int8, writer.add_uint16, writer.add_int16, writer.add_uint32,
        writer.add_int32, writer.add_float32, writer.add_uint64, writer.add_int64, writer.add_float64,
    )):
        add(f"scalar.{index}", 1)
    writer.add_bool("flag", True)
    writer.add_array("tokenizer.ggml.tokens", ["a", "", "\u00e9t\u00e9", "<|im_end|>"])
    writer.add_array("tokenizer.ggml.token_type", [1, 2, 3, 4])
    writer.add_array("nested", [["a", "bc"], ["d"]])
    rng = np.random.default_rng(0)
    q = gguf.GGMLQuantizationType
    writer.add_tensor("q4_k", rng.integers(0, 256, (2, 2 * 144), dtype=np.uint8), raw_dtype=q.Q4_K)
    writer.add_tensor("q8_0", rng.integers(0, 256, (3, 2 * 34), dtype=np.uint8), raw_dtype=q.Q8_0)
    writer.add_tensor("iq3_s", rng.integers(0, 256, (2, 110), dtype=np.uint8), raw_dtype=q.IQ3_S)
    writer.add_tensor("f32", rng.standard_normal((3, 5), dtype=np.float32))
    writer.add_tensor("f16", np.arange(7, dtype=np.float16))
    writer.add_tensor("bf16", rng.integers(0, 256, (2, 12), dtype=np.uint8), raw_dtype=q.BF16)
    _write(writer)
    expected = gguf.GGUFReader(path).tensors
    # The gguf extra no longer installs the package, so reading must not import it.
    monkeypatch.setitem(sys.modules, "gguf", None)
    with MappedCheckpoint(path) as reader:
        assert reader.keys() == sorted(tensor.name for tensor in expected)
        for tensor in expected:
            if tensor.tensor_type is q.IQ3_S:
                with pytest.raises(CheckpointError, match="unsupported GGUF quantization type 21"):
                    reader.get_slice(tensor.name)
                continue
            descriptor = reader.get_slice(tensor.name)
            assert descriptor.get_shape() == [int(dim) for dim in reversed(tensor.shape)]
            assert descriptor.get_nbytes() == tensor.n_bytes
            stored = reader.get_tensor(tensor.name)
            if isinstance(stored, GgufParameter):
                assert stored.quant_type == tensor.tensor_type
                stored = stored.as_tensor()
                assert stored.shape == tensor.data.shape
            piece = file_slice(stored)
            assert piece is not None
            assert piece.offset == tensor.data_offset
            assert stored.view(torch.uint8).numpy().tobytes() == tensor.data.tobytes()


def _alignment(value: int) -> bytes:
    return b"general.alignment" + struct.pack("<II", 4, value)


@pytest.mark.parametrize(("endianess", "damage", "message"), [
    ("BIG", lambda data: data, "byte-swapped"),
    ("LITTLE", lambda data: data[:10], "shorter than its 24-byte header"),
    ("LITTLE", lambda data: b"GGML" + data[4:], "does not begin with 'GGUF'"),
    ("LITTLE", lambda data: data[:4] + struct.pack("<I", 4) + data[8:], "unsupported GGUF version 4"),
    ("LITTLE", lambda data: data.replace(_alignment(64), _alignment(48)), "alignment 48 is not a power of two"),
    ("LITTLE", lambda data: data[:60], "malformed GGUF header"),
    ("LITTLE", lambda data: data[:24] + struct.pack("<Q", 1 << 40) + data[32:], "string extends past the end"),
    ("LITTLE", lambda data: data.replace(b"two", b"one"), "duplicate GGUF tensor 'one'"),
    ("LITTLE", lambda data: data.rstrip(b"\0")[:-1], "'two' extends outside the checkpoint"),
])
def test_rejects_unreadable_headers(tmp_path, endianess, damage, message):
    path = tmp_path / "model.gguf"
    writer = gguf.GGUFWriter(path, "test", endianess=gguf.GGUFEndian[endianess])
    writer.add_custom_alignment(64)
    writer.add_tensor("one", np.ones(4, dtype=np.float32))
    writer.add_tensor("two", np.ones(4, dtype=np.float32))
    _write(writer)
    path.write_bytes(damage(path.read_bytes()))
    with pytest.raises(CheckpointError, match=message):
        MappedCheckpoint(path)


def test_mapping_and_open_file_outlive_reader_and_die_with_last_view(checkpoint):
    path, packed = checkpoint
    with MappedCheckpoint(path) as reader:
        mapping = weakref.ref(reader._mapping)
        file = weakref.ref(reader._mapping.file)
        assert memoryview(reader._mapping).readonly
        weight = reader.get_tensor("weight")
        first, second = weight.split(32)
        source = first.as_tensor()
        piece = file_slice(source)
        assert piece is not None
        assert piece.length == packed.nbytes
        assert piece is file_slice(second.as_tensor())
        assert source.untyped_storage().nbytes() == packed.nbytes
        assert piece.file.readable()
    del reader, weight, first, piece, source
    gc.collect()
    assert mapping() is not None
    assert file() is not None and not file().closed
    torch.testing.assert_close(second.as_tensor(), packed[32:])
    del second
    gc.collect()
    assert mapping() is None
    assert file() is None


def test_capture_assignment_and_detach_keep_provenance(checkpoint):
    path, _ = checkpoint
    with MappedCheckpoint(path) as reader:
        weight = reader.get_tensor("weight")
    model = nn.Linear(64, 64, bias=False, device="meta")
    model.load_state_dict({"weight": weight}, assign=True)
    model.requires_grad_(False)
    assert model.weight.shape == (64, 64)
    captured = GgufAdapter.capture_host(model.weight)
    assert captured.data.data_ptr() == weight.as_tensor().data_ptr()
    assert file_slice(captured.data) is file_slice(weight.as_tensor())
    rebuilt = GgufAdapter.cpu_param(captured)
    assert isinstance(rebuilt, GgufParameter)
    assert rebuilt.as_tensor().data_ptr() == captured.data.data_ptr()
    assert file_slice(model.state_dict()["weight"].as_tensor()) is file_slice(captured.data)


def test_grouped_projection_split_copies_only_when_required(checkpoint):
    path, packed = checkpoint
    weight = MappedCheckpoint(path).get_tensor("weight")
    grouped = weight.unflatten(0, (2, 32))
    first, second = (part.flatten(0, 1) for part in grouped.split((16, 16), dim=1))
    for result, expected in ((first, torch.cat((packed[:16], packed[32:48]))),
                             (second, torch.cat((packed[16:32], packed[48:])))):
        assert result.shape == (32, 64)
        assert file_slice(result.as_tensor()) is None
        torch.testing.assert_close(result.as_tensor(), expected)
    assert file_slice(weight.clone().as_tensor()) is None
    assert file_slice(weight.detach().as_tensor()) is not None


def test_row_indexing_preserves_packed_values(checkpoint):
    path, packed = checkpoint
    weight = MappedCheckpoint(path).get_tensor("weight")
    torch.testing.assert_close(weight[3:7].as_tensor(), packed[3:7])
    torch.testing.assert_close(weight[3].as_tensor(), packed[3])
    torch.testing.assert_close(weight[:, :].as_tensor(), packed)
    with pytest.raises(ValueError, match="complete encoded rows"):
        weight[:, :32]


def test_row_concatenation_preserves_encoding_and_has_owned_storage(checkpoint):
    path, packed = checkpoint
    weight = MappedCheckpoint(path).get_tensor("weight")
    first, second = weight.chunk(2)
    swapped = torch.cat((second, first))
    assert swapped.shape == weight.shape
    assert swapped.quant_type == weight.quant_type
    assert file_slice(swapped.as_tensor()) is None
    torch.testing.assert_close(swapped.as_tensor(), torch.cat((packed[32:], packed[:32])))
    different_encoding = GgufParameter(packed, quant_type=int(gguf.GGMLQuantizationType.IQ4_NL))
    with pytest.raises(ValueError, match="same encoding"):
        torch.cat((weight, different_encoding))


def test_pins_owned_copy_and_eviction_releases_it(checkpoint):
    path, expected = checkpoint
    weight = MappedCheckpoint(path).get_tensor("weight")
    raw = GgufAdapter.capture_host(weight).data
    backend = FakeBackend()
    manager = PinManager(4 * mmap.PAGESIZE, backend=backend)
    with manager.acquire([raw]) as lease:
        assert lease.registered_bytes == raw.nbytes
        assert backend.register_calls[0][0] != raw.data_ptr()
        region = manager._registrations[raw.data_ptr()].copy.region
        assert manager.stats.copy_bytes > 0
        torch.testing.assert_close(_transferred(manager, raw), expected)
    manager.clear()
    assert region.closed
    assert manager.stats.copy_bytes == manager.stats.pinned_bytes == 0
    torch.testing.assert_close(raw, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mapped_parameter_activates_and_restores_source(checkpoint):
    from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

    path, packed = checkpoint
    original_bytes = path.read_bytes()
    model = nn.Linear(64, 64, bias=False, device="meta")
    model.load_state_dict({"weight": MappedCheckpoint(path).get_tensor("weight")}, assign=True)
    model.requires_grad_(False)
    offloader = ModelOffloader.from_module(model)
    offloader.activate("cuda")
    try:
        actual = model.weight
        assert isinstance(actual, ConvRotInt8Tensor)
        expected = ConvRotInt8Tensor.from_gguf(packed.cuda(), quant_type=2, group_size=64)
        torch.testing.assert_close(actual.qdata, expected.qdata)
        torch.testing.assert_close(actual.scale, expected.scale)
    finally:
        offloader.deactivate()
    assert isinstance(model.weight, GgufParameter)
    assert file_slice(model.weight.as_tensor()) is not None
    torch.testing.assert_close(model.weight.as_tensor(), packed)
    assert path.read_bytes() == original_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["streaming", "resident"])
def test_mapped_blocks_execute_and_preserve_pinning_policy(checkpoint, pins, mode):
    from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

    path, packed = checkpoint
    manager, backend = pins
    model = _BlockModel(blocks=[nn.Linear(64, 64, bias=False, device="meta") for _ in range(2)])
    with MappedCheckpoint(path) as reader:
        model.load_state_dict({
            "blocks.0.weight": reader.get_tensor("weight"),
            "blocks.1.weight": reader.get_tensor("weight2"),
        }, assign=True)
    sources = [parameter.as_tensor() for parameter in model.parameters()]
    offloader = ModelOffloader.from_module(model, block_paths=["blocks"], block_mode=mode)
    value = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    expected = value
    for data in (packed, packed.flip(0)):
        weight = ConvRotInt8Tensor.from_gguf(data.cuda(), quant_type=2, group_size=64)
        expected = torch.nn.functional.linear(expected, weight)
    for _ in range(2):
        offloader.activate("cuda")
        try:
            with torch.inference_mode():
                actual = model(value)
            torch.testing.assert_close(actual, expected)
            assert (manager.stats.copy_bytes > 0) == (mode == "streaming")
        finally:
            offloader.deactivate()
    assert len(backend.registrations) == (2 if mode == "streaming" else 0)
    assert all(pointer not in {source.data_ptr() for source in sources} for pointer, _ in backend.registrations)
    manager.clear()
    assert manager.stats.pinned_bytes == manager.stats.copy_bytes == 0
    for parameter, source in zip(model.parameters(), sources, strict=True):
        assert file_slice(parameter.as_tensor()) is file_slice(source)
