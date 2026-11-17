"""Read-only safetensors and GGUF mappings with file provenance per storage.

``MappedCheckpoint`` maps a checkpoint file read-only and hands out tensors
that are views into the mapping, like ``safetensors.safe_open`` with
``device="cpu"``. Two things differ, both for the pin manager. The mapping
can never be written, so registering or transferring it can never make the
kernel copy a page. And every tensor's storage records the open file and the
byte range it came from (:func:`file_slice`), so a pinned copy can be filled
by positional reads instead of through the mapping.

The mapping and the open file live exactly as long as any tensor over them
and are never closed explicitly. The file must not change while any tensor
maps it.
"""

import contextlib
import io
import json
import math
import mmap
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import EllipsisType
from typing import Any, NoReturn, Self, TypeGuard

import torch
from piper_kernels.gguf import GGUF_QUANT_SIZES, GGUFQuantizationType

from ._mapped_file import FileSlice, MappedFile, file_slice
from .gguf_parameter import GgufParameter

_DTYPES: dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "BOOL": torch.bool,
    "C64": torch.complex64,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E4M3FNUZ": torch.float8_e4m3fnuz,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E5M2FNUZ": torch.float8_e5m2fnuz,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "I8": torch.int8,
    "U16": torch.uint16,
    "U32": torch.uint32,
    "U64": torch.uint64,
    "U8": torch.uint8,
}
_HEADER_LIMIT = 100_000_000  # safetensors' own cap on the header


class CheckpointError(ValueError):
    """A checkpoint whose header does not describe supported tensor bytes."""


type _Index = int | slice | EllipsisType | torch.Tensor | None


@dataclass(frozen=True, slots=True)
class TensorSlice:
    """One tensor's header entry, as ``safe_open().get_slice()`` returns.

    Indexing it indexes the mapped tensor, so the result is a view whose
    pages are read when the view is.
    """

    checkpoint: MappedCheckpoint
    name: str
    tag: str
    shape: tuple[int, ...]

    def get_dtype(self) -> str:
        """The safetensors dtype tag, such as ``"BF16"``."""
        return self.tag

    def get_shape(self) -> list[int]:
        return list(self.shape)

    def get_nbytes(self, *, dtype: torch.dtype | None = None) -> int:
        """Stored bytes, optionally estimating a cast of ordinary safetensors floats.

        GGUF loads preserve their stored encodings, including their plain float
        tensors, so a dtype override does not change their storage estimate.
        """
        entry = self.checkpoint._entry(self.name)
        if (
            dtype is not None
            and self.checkpoint.path.suffix.lower() != ".gguf"
            and self.tag in {"F32", "F16", "BF16"}
        ):
            return entry.length // entry.storage_dtype.itemsize * dtype.itemsize
        return entry.length

    def __getitem__(self, key: _Index | tuple[_Index, ...]) -> torch.Tensor:
        return self.checkpoint.get_tensor(self.name)[key]


@dataclass(frozen=True, slots=True)
class _TensorEntry:
    """Logical descriptor and the byte layout used to build its mapped tensor."""

    storage_dtype: torch.dtype
    tag: str  # logical dtype, in safetensors notation
    shape: tuple[int, ...]  # logical shape
    offset: int  # from the start of the file
    length: int
    quant_type: int | None = None
    packed_shape: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class _UnsupportedTensor:
    """A GGUF tensor in an encoding Piper Kernels cannot convert; requesting it raises."""

    quant_type: int


class MappedCheckpoint:
    """A safetensors or GGUF file mapped read-only, with ``safe_open``'s reading surface.

    ``keys()``, ``metadata()``, ``get_tensor()`` and ``get_slice()`` are named
    and behave as on ``safetensors.safe_open(path, framework="pt",
    device="cpu")`` so the reader is a drop-in replacement; the getter names
    are deliberate. The safetensors header is validated in full before any
    tensor can be built. GGUF encoding support is checked when a tensor's
    descriptor or value is requested, allowing selective loads to skip unused
    encodings. Tensors are views into the mapping and outlive the reader.

    GGUF files must be little-endian, version 2 or 3. Quantized GGUF tensors
    are returned as ``GgufParameter`` objects with logical shape and BF16
    compute dtype. GGUF ``metadata()`` returns None: its key-value metadata is
    not safetensors' application metadata and is not decoded.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        file = self.path.open("rb")
        try:
            size = file.seek(0, io.SEEK_END)
            file.seek(0)
            self._entries: Mapping[str, _TensorEntry | _UnsupportedTensor]
            if self.path.suffix.lower() == ".gguf":
                self._entries, self._metadata = _read_gguf_header(file, size), None
            else:
                self._entries, self._metadata = _read_header(file, size)
            mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            if hasattr(os, "posix_fadvise"):
                # Doubles the kernel's readahead window for this file, as
                # PyTorch's own file mapping does; cold loads run twice as
                # fast. It is only advice: a file system or sandbox that
                # rejects it costs the readahead, not the checkpoint.
                with contextlib.suppress(OSError):
                    os.posix_fadvise(file.fileno(), 0, size, os.POSIX_FADV_SEQUENTIAL)
        except BaseException:
            file.close()
            raise
        self._mapping: MappedFile | None = MappedFile(file, mapping)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Drop the reader's hold on the mapping. Tensors already built keep it alive."""
        self._mapping = None

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def metadata(self) -> dict[str, str] | None:
        """The ``__metadata__`` entry, or ``None`` when the file has none."""
        return dict(self._metadata) if self._metadata is not None else None

    def get_slice(self, name: str) -> TensorSlice:
        entry = self._entry(name)
        return TensorSlice(self, name, entry.tag, entry.shape)

    def get_tensor(self, name: str) -> torch.Tensor:
        """A tensor over the mapping whose storage is exactly its bytes and records their file slice."""
        entry = self._entry(name)
        mapping = self._mapping
        if mapping is None:
            raise RuntimeError(f"{self.path} is closed")
        tensor = mapping.tensor(
            entry.offset, entry.length, entry.storage_dtype, entry.packed_shape or entry.shape,
        )
        if entry.quant_type is not None:
            return GgufParameter(tensor, quant_type=entry.quant_type)
        return tensor

    def _entry(self, name: str) -> _TensorEntry:
        try:
            entry = self._entries[name]
        except KeyError:
            raise KeyError(f"{self.path} has no tensor {name!r}") from None
        if isinstance(entry, _UnsupportedTensor):
            raise CheckpointError(f"unsupported GGUF quantization type {entry.quant_type!r} for {name!r}")
        if entry.quant_type is not None and len(entry.shape) != 2:
            raise CheckpointError(f"quantized GGUF tensor {name!r} must be a matrix, got {entry.shape}")
        return entry


# The GGUF v2 and v3 header, all little-endian: the magic "GGUF", a u32
# version and u64 tensor and metadata counts. Then each metadata entry: a
# string key, a u32 value type and the value. Then each tensor's info: a
# string name, a u32 dimension count, u64 dimensions innermost first, a u32
# encoding type and a u64 data offset from the aligned start of tensor data.
# A string is a u64 byte length and UTF-8 bytes; an array value is a u32
# element type, a u64 count and the elements. Value types 0-7 and 10-12 are
# fixed-size numbers and booleans, 8 is a string and 9 an array.
_GGUF_PREFIX = struct.Struct("<4sIQQ")
_GGUF_U32 = struct.Struct("<I")
_GGUF_U64 = struct.Struct("<Q")
_GGUF_TYPE_AND_COUNT = struct.Struct("<IQ")  # an array's element type and count; a tensor's encoding and offset
_GGUF_UINT32, _GGUF_STRING, _GGUF_ARRAY = 4, 8, 9
_GGUF_SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_PLAIN_TAGS: dict[int, str] = {
    GGUFQuantizationType.F32: "F32",
    GGUFQuantizationType.F16: "F16",
    GGUFQuantizationType.BF16: "BF16",
}


def _read_gguf_header(file: io.BufferedReader, size: int) -> dict[str, _TensorEntry | _UnsupportedTensor]:
    """Tensor entries from a little-endian GGUF v2 or v3 header.

    Reads the counts, the ``general.alignment`` value and each tensor's info.
    Every other metadata value is skipped by its length, never decoded: a
    chat model's tokenizer keeps hundreds of thousands of strings there.
    Tensors whose encoding Piper Kernels cannot convert are listed but not
    laid out, so a selective load can skip them.
    """
    if size < _GGUF_PREFIX.size:
        raise CheckpointError(f"not a GGUF file: shorter than its {_GGUF_PREFIX.size}-byte header")
    with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        try:
            return _parse_gguf_header(data)
        except (struct.error, UnicodeDecodeError) as error:
            raise CheckpointError(f"malformed GGUF header: {error}") from None


def _parse_gguf_header(data: mmap.mmap) -> dict[str, _TensorEntry | _UnsupportedTensor]:
    magic, version, tensor_count, metadata_count = _GGUF_PREFIX.unpack_from(data)
    if magic != b"GGUF":
        raise CheckpointError("not a GGUF file: it does not begin with 'GGUF'")
    if version & 0xFFFF == 0:  # a big-endian version number, such as 3 << 24
        raise CheckpointError("GGUF byte-swapped weights cannot be used by the native conversion kernels")
    if version not in (2, 3):
        raise CheckpointError(f"unsupported GGUF version {version}")
    pos = _GGUF_PREFIX.size
    alignment = 32
    for _ in range(metadata_count):
        key, pos = _gguf_string(data, pos)
        (value_type,) = _GGUF_U32.unpack_from(data, pos)
        pos += 4
        if key == b"general.alignment":
            if value_type != _GGUF_UINT32:
                raise CheckpointError("GGUF general.alignment must be a UINT32")
            (alignment,) = _GGUF_U32.unpack_from(data, pos)
            if alignment == 0 or alignment & (alignment - 1):
                raise CheckpointError(f"GGUF alignment {alignment} is not a power of two")
        pos = _skip_gguf_value(data, pos, value_type)
    infos: list[tuple[str, tuple[int, ...], int, int]] = []
    for _ in range(tensor_count):
        name, pos = _gguf_string(data, pos)
        (dimension_count,) = _GGUF_U32.unpack_from(data, pos)
        dims = struct.unpack_from(f"<{dimension_count}Q", data, pos + 4)
        quant_type, offset = _GGUF_TYPE_AND_COUNT.unpack_from(data, pos + 4 + 8 * dimension_count)
        pos += 4 + 8 * dimension_count + _GGUF_TYPE_AND_COUNT.size
        infos.append((name.decode(), tuple(reversed(dims)), quant_type, offset))
    data_start = -(-pos // alignment) * alignment
    entries: dict[str, _TensorEntry | _UnsupportedTensor] = {}
    for name, shape, quant_type, offset in infos:
        if name in entries:
            raise CheckpointError(f"duplicate GGUF tensor {name!r}")
        entries[name] = _gguf_entry(name, shape, quant_type, data_start + offset, len(data))
    return entries


def _gguf_string(data: mmap.mmap, pos: int) -> tuple[bytes, int]:
    (length,) = _GGUF_U64.unpack_from(data, pos)
    end = pos + 8 + length
    if end > len(data):  # slicing would clamp and copy the rest of the file
        raise CheckpointError("malformed GGUF header: a string extends past the end of the file")
    return data[pos + 8:end], end


def _skip_gguf_value(data: mmap.mmap, pos: int, value_type: int) -> int:
    """The position after a metadata value, found from lengths alone."""
    if value_type == _GGUF_STRING:
        return pos + 8 + _GGUF_U64.unpack_from(data, pos)[0]
    if value_type != _GGUF_ARRAY:
        try:
            return pos + _GGUF_SCALAR_BYTES[value_type]
        except KeyError:
            raise CheckpointError(f"unknown GGUF metadata value type {value_type}") from None
    item_type, count = _GGUF_TYPE_AND_COUNT.unpack_from(data, pos)
    pos += _GGUF_TYPE_AND_COUNT.size
    if item_type in _GGUF_SCALAR_BYTES:
        return pos + count * _GGUF_SCALAR_BYTES[item_type]
    if item_type == _GGUF_STRING:
        # Inlined: tokenizer vocabularies and merges hold most of the strings.
        for _ in range(count):
            pos += 8 + _GGUF_U64.unpack_from(data, pos)[0]
        return pos
    for _ in range(count):
        pos = _skip_gguf_value(data, pos, item_type)
    return pos


def _gguf_entry(
    name: str, shape: tuple[int, ...], quant_type: int, offset: int, size: int,
) -> _TensorEntry | _UnsupportedTensor:
    if quant_type in _GGUF_PLAIN_TAGS:
        tag = _GGUF_PLAIN_TAGS[quant_type]
        dtype = _DTYPES[tag]
        entry = _TensorEntry(dtype, tag, shape, offset, math.prod(shape) * dtype.itemsize)
    elif quant_type in GGUF_QUANT_SIZES:
        block_size, type_size = GGUF_QUANT_SIZES[GGUFQuantizationType(quant_type)]
        if not shape or shape[-1] % block_size:
            raise CheckpointError(f"GGUF tensor {name!r} has rows that are not whole {block_size}-value blocks")
        packed_shape = (*shape[:-1], shape[-1] // block_size * type_size)
        entry = _TensorEntry(
            torch.uint8, "BF16", shape, offset, math.prod(packed_shape),
            quant_type=quant_type, packed_shape=packed_shape,
        )
    else:
        return _UnsupportedTensor(quant_type)
    if offset + entry.length > size:
        raise CheckpointError(f"GGUF tensor {name!r} extends outside the checkpoint")
    return entry


def _read_header(file: io.BufferedReader, size: int) -> tuple[dict[str, _TensorEntry], dict[str, str] | None]:
    if size < 8:
        raise CheckpointError("not a safetensors file: shorter than its 8-byte header length")
    (header_size,) = struct.unpack("<Q", file.read(8))
    if header_size > _HEADER_LIMIT or 8 + header_size > size:
        raise CheckpointError(f"header length {header_size} exceeds the file or the {_HEADER_LIMIT}-byte limit")
    raw = file.read(header_size)
    # Strict UTF-8 starting at the brace, as safetensors requires: no BOM, no
    # leading whitespace, no other encoding json.loads would guess.
    if not raw.startswith(b"{"):
        raise CheckpointError("header must begin with '{'")
    try:
        header = json.loads(raw.decode(), object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as error:
        raise CheckpointError(f"header is not valid JSON: {error}") from None
    metadata = header.pop("__metadata__", None)
    if metadata is not None and not (
        isinstance(metadata, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())
    ):
        raise CheckpointError("__metadata__ must map strings to strings")
    data_start, data_size = 8 + header_size, size - 8 - header_size
    entries = {name: _parse_entry(name, raw, data_start, data_size) for name, raw in header.items()}
    # The data section must be claimed exactly, in order, without gaps or
    # overlap; unclaimed bytes are how a file hides a payload behind its
    # header. Zero-length entries sort first at a shared offset so header
    # order cannot decide validity.
    end = data_start
    for name, entry in sorted(entries.items(), key=lambda item: (item[1].offset, item[1].length)):
        if entry.offset < end:
            raise CheckpointError(f"tensor {name!r} overlaps the previous tensor")
        if entry.offset > end:
            raise CheckpointError(f"{entry.offset - end} unclaimed bytes before tensor {name!r}")
        end += entry.length
    if end != data_start + data_size:
        raise CheckpointError(f"{data_start + data_size - end} unclaimed bytes after the last tensor")
    return entries, metadata


def _parse_entry(name: str, raw: object, data_start: int, data_size: int) -> _TensorEntry:
    if not isinstance(raw, dict):
        raise CheckpointError(f"tensor {name!r} must be a JSON object")
    tag, shape, offsets = raw.get("dtype"), raw.get("shape"), raw.get("data_offsets")
    if not isinstance(tag, str) or tag not in _DTYPES:
        raise CheckpointError(f"tensor {name!r} has unsupported dtype {tag!r}")
    if not _is_integer_list(shape) or any(n < 0 for n in shape):
        raise CheckpointError(f"tensor {name!r} has an invalid shape {shape!r}")
    if not _is_integer_list(offsets) or len(offsets) != 2 or not 0 <= offsets[0] <= offsets[1] <= data_size:
        raise CheckpointError(
            f"tensor {name!r} has invalid data_offsets {offsets!r} for a {data_size}-byte data section",
        )
    dtype = _DTYPES[tag]
    try:
        # A meta tensor allocates nothing but runs every size and stride check
        # PyTorch applies, so a shape it cannot represent fails here.
        meta = torch.empty(shape, dtype=dtype, device="meta")
    except (RuntimeError, TypeError, OverflowError) as error:
        raise CheckpointError(f"tensor {name!r} has a shape PyTorch cannot represent: {error}") from None
    start, end = offsets
    if end - start != meta.numel() * meta.element_size():
        raise CheckpointError(f"tensor {name!r} holds {end - start} bytes but its shape and dtype need {meta.nbytes}")
    return _TensorEntry(dtype, tag, tuple(shape), data_start + start, end - start)


def _is_integer_list(value: object) -> TypeGuard[list[int]]:
    return isinstance(value, list) and all(isinstance(n, int) and not isinstance(n, bool) for n in value)


def _reject_constant(constant: str) -> NoReturn:
    raise CheckpointError(f"{constant} is not valid JSON")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"duplicate header key {key!r}")
        result[key] = value
    return result


__all__ = ["CheckpointError", "FileSlice", "MappedCheckpoint", "TensorSlice", "file_slice"]
