"""Read-only checkpoint mapping: safe_open surface, validation, lifetime, and provenance."""

import gc
import json
import struct
import weakref
from pathlib import Path

import pytest
import torch

import piper_offload.checkpoint as checkpoint_module
from piper_offload import CheckpointError, MappedCheckpoint, file_slice

_TAGS = {dtype: tag for tag, dtype in checkpoint_module._DTYPES.items()}


def _write(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str] | None = None) -> bytes:
    """Write a safetensors file by hand and return its data section."""
    header: dict[str, object] = {}
    data = bytearray()
    for name, tensor in tensors.items():
        raw = _raw_bytes(tensor)
        header[name] = {
            "dtype": _TAGS[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    if metadata is not None:
        header["__metadata__"] = metadata
    encoded = json.dumps(header).encode()
    encoded += b" " * (-(8 + len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(data))
    return bytes(data)


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes() if tensor.numel() else b""


def _write_raw(path: Path, header: object, data: bytes = b"") -> None:
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


@pytest.fixture
def sample(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    tensors = {
        "block.0.weight": torch.arange(24, dtype=torch.float32).reshape(4, 6),
        "block.0.scale": torch.tensor(0.5, dtype=torch.bfloat16),
        "block.0.bias": torch.arange(4, dtype=torch.int8),
        "block.1.empty": torch.empty(0, 3, dtype=torch.float16),
        "flags": torch.tensor([True, False, True]),
    }
    path = tmp_path / "model.safetensors"
    _write(path, tensors, {"format": "pt", "note": "hand-written"})
    return path, tensors


def test_reading_surface_matches_the_written_tensors(sample) -> None:
    path, tensors = sample
    with MappedCheckpoint(path) as reader:
        assert reader.keys() == sorted(tensors)
        assert reader.metadata() == {"format": "pt", "note": "hand-written"}
        for name, expected in tensors.items():
            piece = reader.get_slice(name)
            assert piece.get_dtype() == _TAGS[expected.dtype]
            assert piece.get_shape() == list(expected.shape)
            actual = reader.get_tensor(name)
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            torch.testing.assert_close(actual, expected)
        with pytest.raises(KeyError, match="no tensor 'missing'"):
            reader.get_tensor("missing")


def test_matches_safetensors_reader(sample) -> None:
    safetensors = pytest.importorskip("safetensors")
    path, tensors = sample
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as reference, MappedCheckpoint(path) as reader:
        assert reader.keys() == list(reference.keys())
        assert reader.metadata() == reference.metadata()
        for name in tensors:
            torch.testing.assert_close(reader.get_tensor(name), reference.get_tensor(name))
            assert reader.get_slice(name).get_dtype() == reference.get_slice(name).get_dtype()
            assert reader.get_slice(name).get_shape() == reference.get_slice(name).get_shape()


def test_no_metadata_reads_as_none(tmp_path: Path) -> None:
    path = tmp_path / "bare.safetensors"
    _write(path, {"w": torch.ones(2)})
    assert MappedCheckpoint(path).metadata() is None


def test_tensors_are_views_into_a_read_only_mapping_with_their_own_storage(sample) -> None:
    path, tensors = sample
    reader = MappedCheckpoint(path)
    weight = reader.get_tensor("block.0.weight")
    bias = reader.get_tensor("block.0.bias")
    assert not weight.untyped_storage().resizable()
    assert weight.untyped_storage().nbytes() == weight.numel() * weight.element_size()
    assert bias.untyped_storage().data_ptr() != weight.untyped_storage().data_ptr()
    # Two reads of one tensor map the same bytes.
    assert reader.get_tensor("block.0.weight").data_ptr() == weight.data_ptr()


def test_provenance_names_the_open_file_and_exact_byte_range(sample) -> None:
    path, tensors = sample
    data = path.read_bytes()
    reader = MappedCheckpoint(path)
    weight = reader.get_tensor("block.0.weight")
    piece = file_slice(weight)
    assert piece is not None
    assert piece.length == weight.numel() * weight.element_size()
    assert data[piece.offset:piece.offset + piece.length] == _raw_bytes(weight)
    piece.file.seek(piece.offset)
    assert piece.file.read(piece.length) == data[piece.offset:piece.offset + piece.length]
    # A view shares its base's storage and so its slice; other storage has none.
    assert file_slice(weight[1:3]) is piece
    assert file_slice(torch.ones(4)) is None
    assert file_slice(reader.get_tensor("block.1.empty")) is None


def test_tensors_and_provenance_outlive_the_reader_and_die_with_the_last_tensor(sample) -> None:
    path, tensors = sample
    with MappedCheckpoint(path) as reader:
        weight = reader.get_tensor("block.0.weight")
        mapping_ref = weakref.ref(reader._mapping)
        file_ref = weakref.ref(mapping_ref().file)
    del reader
    gc.collect()
    assert mapping_ref() is not None
    assert file_ref() is not None and not file_ref().closed
    torch.testing.assert_close(weight, tensors["block.0.weight"])
    assert file_slice(weight) is not None
    pointer = weight.untyped_storage().data_ptr()
    del weight
    gc.collect()
    assert mapping_ref() is None
    assert file_ref() is None
    assert pointer not in checkpoint_module._provenance


def test_closed_reader_refuses_new_tensors_but_keeps_existing_ones(sample) -> None:
    path, tensors = sample
    reader = MappedCheckpoint(path)
    bias = reader.get_tensor("block.0.bias")
    reader.close()
    for name in ("block.0.weight", "block.1.empty"):
        with pytest.raises(RuntimeError, match="is closed"):
            reader.get_tensor(name)
    torch.testing.assert_close(bias, tensors["block.0.bias"])
    assert reader.get_slice("block.0.weight").get_shape() == [4, 6]


_ENTRY = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
_BAD_HEADERS = [
    ({"w": _ENTRY}, b"\0" * 4, "invalid data_offsets"),
    ({"w": {**_ENTRY, "shape": [3]}}, b"\0" * 8, "holds 8 bytes but its shape and dtype need 12"),
    ({"w": {**_ENTRY, "dtype": "F4"}}, b"\0" * 8, "unsupported dtype 'F4'"),
    ({"w": {**_ENTRY, "shape": [-2]}}, b"\0" * 8, "invalid shape"),
    ({"w": {**_ENTRY, "data_offsets": [8, 0]}}, b"\0" * 8, "invalid data_offsets"),
    ({"w": {**_ENTRY, "data_offsets": [0]}}, b"\0" * 8, "invalid data_offsets"),
    ({"w": _ENTRY, "v": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}, b"\0" * 8, "overlaps"),
    ({"w": "not an object"}, b"", "must be a JSON object"),
    ({"__metadata__": {"a": 1}, "w": _ENTRY}, b"\0" * 8, "__metadata__ must map strings to strings"),
    ([], b"", "must begin with"),
    ({"w": {**_ENTRY, "data_offsets": [4, 12]}}, b"\0" * 12, "4 unclaimed bytes before tensor 'w'"),
    ({"w": _ENTRY}, b"\0" * 12, "4 unclaimed bytes after the last tensor"),
    (
        {"w": _ENTRY, "v": {"dtype": "F32", "shape": [1], "data_offsets": [12, 16]}},
        b"\0" * 16,
        "4 unclaimed bytes before tensor 'v'",
    ),
    ({}, b"\0" * 4, "4 unclaimed bytes after the last tensor"),
    ({"w": {"dtype": "F32", "shape": [0, 2**63], "data_offsets": [0, 0]}}, b"", "cannot represent"),
    ({"w": {"dtype": "F32", "shape": [0, 2**62, 4], "data_offsets": [0, 0]}}, b"", "cannot represent"),
    ({"w": {"dtype": "F32", "shape": [2, -1], "data_offsets": [0, 0]}}, b"", "invalid shape"),
    ({"w": _ENTRY, "e": {"dtype": "F32", "shape": [0], "data_offsets": [4, 4]}}, b"\0" * 8, "overlaps"),
    ({"w": {**_ENTRY, "note": float("nan")}}, b"\0" * 8, "NaN is not valid JSON"),
    ({"w": {**_ENTRY, "note": float("inf")}}, b"\0" * 8, "Infinity is not valid JSON"),
]


@pytest.mark.parametrize(("header", "data", "message"), _BAD_HEADERS)
def test_malformed_headers_are_rejected_before_any_tensor_exists(tmp_path: Path, header, data, message) -> None:
    path = tmp_path / "bad.safetensors"
    _write_raw(path, header, data)
    with pytest.raises(CheckpointError, match=message):
        MappedCheckpoint(path)


def test_duplicate_keys_truncation_and_bad_length_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.safetensors"
    entry = b'"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}'
    encoded = b"{" + entry + b", " + entry + b"}"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 4)
    with pytest.raises(CheckpointError, match="duplicate header key 'w'"):
        MappedCheckpoint(path)
    path.write_bytes(b"\0\0\0")
    with pytest.raises(CheckpointError, match="shorter than its 8-byte header length"):
        MappedCheckpoint(path)
    path.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    with pytest.raises(CheckpointError, match="exceeds the file"):
        MappedCheckpoint(path)
    path.write_bytes(struct.pack("<Q", 2) + b"{]")
    with pytest.raises(CheckpointError, match="not valid JSON"):
        MappedCheckpoint(path)


def test_empty_tensor_at_a_shared_offset_is_valid_in_either_header_order(tmp_path: Path) -> None:
    for order in (("w", "e"), ("e", "w")):
        entries = {
            "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "e": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
        }
        path = tmp_path / f"{'-'.join(order)}.safetensors"
        _write_raw(path, {name: entries[name] for name in order}, b"\0" * 8)
        reader = MappedCheckpoint(path)
        assert reader.get_tensor("e").shape == (0,)
        assert reader.get_tensor("w").shape == (2,)


def test_header_must_be_strict_utf8_starting_at_the_brace(tmp_path: Path) -> None:
    path = tmp_path / "enc.safetensors"
    for raw, message in (
        (b" {}", "must begin with"),
        (b"\xef\xbb\xbf{}", "must begin with"),
        ("{}".encode("utf-16-le"), "not valid JSON"),  # starts with the brace byte, then NULs
        ("{}".encode("utf-16-be"), "must begin with"),
        (b"{\x00}\x00", "not valid JSON"),
        (b'{"\xff": 1}', "not valid JSON"),
    ):
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
        with pytest.raises(CheckpointError, match=message):
            MappedCheckpoint(path)
    assert checkpoint_module._HEADER_LIMIT == 100_000_000


def test_scalars_and_empty_tensors(tmp_path: Path) -> None:
    path = tmp_path / "edge.safetensors"
    tensors = {"scalar": torch.tensor(3.0, dtype=torch.float64), "empty": torch.empty(0, dtype=torch.int64)}
    _write(path, tensors)
    reader = MappedCheckpoint(path)
    scalar = reader.get_tensor("scalar")
    assert scalar.shape == () and scalar.item() == 3.0
    assert file_slice(scalar) is not None and file_slice(scalar).length == 8
    empty = reader.get_tensor("empty")
    assert empty.shape == (0,) and empty.dtype == torch.int64
    assert reader.get_slice("empty").get_shape() == [0]


def test_every_supported_dtype_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "dtypes.safetensors"
    tensors = {}
    for tag, dtype in checkpoint_module._DTYPES.items():
        base = torch.arange(4, dtype=torch.float32)
        if dtype is torch.bool:
            tensors[tag] = base.bool()
        elif dtype is torch.float8_e8m0fnu:
            tensors[tag] = torch.arange(4, dtype=torch.uint8).view(dtype)  # E8M0 has no conversion from float
        else:
            tensors[tag] = base.to(dtype)
    assert "F8_E8M0" in tensors
    _write(path, tensors)
    reader = MappedCheckpoint(path)
    for tag, expected in tensors.items():
        actual = reader.get_tensor(tag)
        assert actual.dtype == expected.dtype
        assert _raw_bytes(actual) == _raw_bytes(expected)
