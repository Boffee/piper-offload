"""Read-only safetensors mapping with file provenance per storage.

``MappedCheckpoint`` maps a safetensors file read-only and hands out tensors
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
import mmap
import os
import struct
import threading
import warnings
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import EllipsisType
from typing import Any, NoReturn, Self, TypeGuard

import torch

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
    """A safetensors file whose header does not describe its bytes."""


@dataclass(frozen=True, slots=True)
class FileSlice:
    """Where a storage's bytes live: ``length`` bytes at ``offset`` in the reader's open ``file``."""

    file: io.BufferedReader
    offset: int
    length: int


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

    def __getitem__(self, key: _Index | tuple[_Index, ...]) -> torch.Tensor:
        return self.checkpoint.get_tensor(self.name)[key]


@dataclass(frozen=True, slots=True)
class _TensorEntry:
    dtype: torch.dtype
    tag: str
    shape: tuple[int, ...]
    offset: int  # from the start of the file
    length: int


class _MappedFile:
    """Buffer the tensors are built over.

    Every tensor's storage references this object through the buffer
    protocol, so the file and the mapping live exactly as long as the tensors.
    Provenance references it weakly.
    """

    __slots__ = ("__weakref__", "file", "mmap")

    def __init__(self, file: io.BufferedReader, mapping: mmap.mmap) -> None:
        self.file = file
        self.mmap = mapping

    def __buffer__(self, flags: int, /) -> memoryview:
        return memoryview(self.mmap)


_lock = threading.Lock()
# Storage pointer -> (its mapping, weakly, and its slice).
_provenance: dict[int, tuple[weakref.ReferenceType[_MappedFile], FileSlice]] = {}


def file_slice(source: torch.Tensor | torch.UntypedStorage) -> FileSlice | None:
    """The checkpoint bytes behind a tensor's storage, or ``None`` for storage that is not a mapped checkpoint.

    Views share their base tensor's storage and so its slice. Storage that
    merely reuses a dead mapping's address does not match.
    """
    storage = source if isinstance(source, torch.UntypedStorage) else source.untyped_storage()
    with _lock:
        entry = _provenance.get(storage.data_ptr())
    if entry is None:
        return None
    mapping, piece = entry
    return piece if mapping() is not None and piece.length == storage.nbytes() else None


def _forget_dead_mappings() -> None:
    with _lock:
        for pointer in [pointer for pointer, (mapping, _) in _provenance.items() if mapping() is None]:
            del _provenance[pointer]


class MappedCheckpoint:
    """A safetensors file mapped read-only, with ``safe_open``'s reading surface.

    ``keys()``, ``metadata()``, ``get_tensor()`` and ``get_slice()`` are named
    and behave as on ``safetensors.safe_open(path, framework="pt",
    device="cpu")`` so the reader is a drop-in replacement; the getter names
    are deliberate. The header is validated in full before any tensor can be
    built. Tensors are views into the mapping and outlive the reader.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        file = self.path.open("rb")
        try:
            size = file.seek(0, io.SEEK_END)
            file.seek(0)
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
        self._mapping: _MappedFile | None = _MappedFile(file, mapping)
        weakref.finalize(self._mapping, _forget_dead_mappings)

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
        if entry.length == 0:
            return torch.empty(entry.shape, dtype=entry.dtype)
        window = memoryview(mapping)[entry.offset:entry.offset + entry.length]
        with warnings.catch_warnings():
            # Non-writable is the point: nothing may write through a checkpoint mapping.
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            tensor = torch.frombuffer(window, dtype=entry.dtype).reshape(entry.shape)
        with _lock:
            _provenance[tensor.untyped_storage().data_ptr()] = (
                weakref.ref(mapping),
                FileSlice(mapping.file, entry.offset, entry.length),
            )
        return tensor

    def _entry(self, name: str) -> _TensorEntry:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(f"{self.path} has no tensor {name!r}") from None


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
