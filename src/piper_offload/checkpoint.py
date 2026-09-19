"""Read-only safetensors mapping with file provenance per storage.

``MappedCheckpoint`` maps a safetensors file read-only and hands out tensors
that are views into that mapping, the way ``safetensors.safe_open`` does,
with two differences the pin manager relies on. The mapping can never be
written, so nothing that registers or transfers it can make the kernel copy
a page. And every tensor's storage records the open file and the byte range
it came from, so a pinned copy can be filled by positional reads instead of
through the mapping (:func:`file_slice`).

Lifetime: the mapping and the open file live exactly as long as any tensor
over them. The reader itself may be closed or discarded at any time. The
mapping is never closed explicitly, because an explicit close would leave
live tensors pointing into an unmapped range. The file must not change while
any tensor maps it.
"""

import io
import json
import math
import mmap
import struct
import threading
import warnings
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, NoReturn, Self

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
# safetensors refuses headers above this many bytes; a larger length field means a corrupt file.
_HEADER_LIMIT = 100_000_000


class CheckpointError(ValueError):
    """A safetensors file whose header does not describe its bytes."""


@dataclass(frozen=True, slots=True)
class FileSlice:
    """Where a storage's bytes live in its checkpoint file.

    ``file`` is the reader's original open file, kept open for as long as any
    tensor maps the checkpoint, so a positional read of ``length`` bytes at
    ``offset`` yields the storage's bytes even if the path is replaced.
    """

    file: io.BufferedReader
    offset: int
    length: int


@dataclass(frozen=True, slots=True)
class _Entry:
    dtype: torch.dtype
    tag: str
    shape: tuple[int, ...]
    offset: int  # from the start of the file
    length: int


class _Mapping:
    """The buffer the tensors are built over.

    Exports the read-only mapping through the buffer protocol so that every
    tensor's storage holds a reference to this object, which keeps the file
    and the mapping alive. Provenance holds it weakly.
    """

    __slots__ = ("__weakref__", "file", "mmap")

    def __init__(self, file: io.BufferedReader, mapping: mmap.mmap) -> None:
        self.file = file
        self.mmap = mapping

    def __buffer__(self, flags: int, /) -> memoryview:
        return memoryview(self.mmap)


_registry_lock = threading.Lock()
# storage pointer -> (the mapping, weakly, and the slice). Entries of a dead
# mapping are purged by its finalizer and skipped by lookups in between.
_slices: dict[int, tuple[weakref.ReferenceType[_Mapping], FileSlice]] = {}


def _record(mapping: _Mapping, pointer: int, piece: FileSlice) -> None:
    with _registry_lock:
        _slices[pointer] = (weakref.ref(mapping), piece)


def _purge_dead() -> None:
    with _registry_lock:
        for pointer in [pointer for pointer, (ref, _piece) in _slices.items() if ref() is None]:
            del _slices[pointer]


def file_slice(tensor: torch.Tensor) -> FileSlice | None:
    """The checkpoint bytes behind ``tensor``'s storage, or ``None`` for any other storage.

    Views share their base tensor's storage and so its slice. A storage that
    merely reuses a dead mapping's address is not matched.
    """
    return storage_slice(tensor.untyped_storage())


def storage_slice(storage: torch.UntypedStorage) -> FileSlice | None:
    """The checkpoint bytes behind ``storage``, or ``None`` when it is not a mapped checkpoint."""
    with _registry_lock:
        entry = _slices.get(storage.data_ptr())
    if entry is None:
        return None
    mapping, piece = entry
    if mapping() is None or piece.length != storage.nbytes():
        return None
    return piece


class TensorSlice:
    """Header-only view of one tensor, mirroring ``safe_open``'s slices."""

    __slots__ = ("_checkpoint", "_entry", "_name")

    def __init__(self, checkpoint: MappedCheckpoint, name: str, entry: _Entry) -> None:
        self._checkpoint = checkpoint
        self._name = name
        self._entry = entry

    def get_dtype(self) -> str:
        """The safetensors dtype tag, such as ``"BF16"``."""
        return self._entry.tag

    def get_shape(self) -> list[int]:
        return list(self._entry.shape)

    def __getitem__(self, index: object) -> torch.Tensor:
        return self._checkpoint.get_tensor(self._name)[index]  # type: ignore[index]


class MappedCheckpoint:
    """A safetensors file mapped read-only, with ``safe_open``'s reading surface.

    ``keys()``, ``metadata()``, ``get_tensor()`` and ``get_slice()`` behave as
    they do on ``safetensors.safe_open(path, framework="pt", device="cpu")``.
    Tensors are views into the mapping and outlive the reader.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        file = self.path.open("rb")
        try:
            size = file.seek(0, io.SEEK_END)
            file.seek(0)
            entries, metadata = _read_header(file, size)
            mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            file.close()
            raise
        self._entries = entries
        self._metadata = metadata
        self._mapping: _Mapping | None = _Mapping(file, mapping)
        weakref.finalize(self._mapping, _purge_dead)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Drop the reader's hold on the mapping. Tensors already built keep it alive."""
        self._mapping = None

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def metadata(self) -> dict[str, str] | None:
        """The ``__metadata__`` entry, or ``None`` when the file has none."""
        return dict(self._metadata) if self._metadata is not None else None

    def get_slice(self, name: str) -> TensorSlice:
        return TensorSlice(self, name, self._entry(name))

    def get_tensor(self, name: str) -> torch.Tensor:
        """A tensor over the mapping. Its storage is exactly the tensor's bytes and records their file slice."""
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
        _record(mapping, tensor.untyped_storage().data_ptr(), FileSlice(mapping.file, entry.offset, entry.length))
        return tensor

    def _entry(self, name: str) -> _Entry:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(f"{self.path} has no tensor {name!r}") from None


def _read_header(file: io.BufferedReader, size: int) -> tuple[dict[str, _Entry], dict[str, str] | None]:
    """Parse and validate the header. Every byte range is checked before any tensor is built over it."""
    if size < 8:
        raise CheckpointError("not a safetensors file: shorter than its 8-byte header length")
    (header_size,) = struct.unpack("<Q", file.read(8))
    if header_size > _HEADER_LIMIT or 8 + header_size > size:
        raise CheckpointError(f"header length {header_size} exceeds the file or the {_HEADER_LIMIT}-byte limit")
    raw = file.read(header_size)
    # safetensors requires strict UTF-8 starting at the object brace: no BOM,
    # no leading whitespace, and no other encoding json.loads would guess.
    if not raw.startswith(b"{"):
        raise CheckpointError("header must begin with '{'")
    try:
        header = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise CheckpointError(f"header is not valid JSON: {error}") from None
    if not isinstance(header, dict):
        raise CheckpointError("header must be a JSON object")
    data_start = 8 + header_size
    data_size = size - data_start
    metadata = header.pop("__metadata__", None)
    if metadata is not None and not (
        isinstance(metadata, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())
    ):
        raise CheckpointError("__metadata__ must map strings to strings")
    entries = {name: _parse_entry(name, raw, data_start, data_size) for name, raw in header.items()}
    # The data section must be covered exactly, in order and without gaps:
    # unclaimed bytes are how a file smuggles a payload past its header.
    previous_end = data_start
    # Zero-length entries sort before a tensor starting at the same offset;
    # header order must not decide whether the file is valid.
    for name, entry in sorted(entries.items(), key=lambda item: (item[1].offset, item[1].length)):
        if entry.offset < previous_end:
            raise CheckpointError(f"tensor {name!r} overlaps the previous tensor's bytes")
        if entry.offset > previous_end:
            raise CheckpointError(f"{entry.offset - previous_end} unclaimed bytes before tensor {name!r}")
        previous_end = entry.offset + entry.length
    if previous_end != data_start + data_size:
        raise CheckpointError(f"{data_start + data_size - previous_end} unclaimed bytes after the last tensor")
    return entries, metadata


def _parse_entry(name: str, raw: object, data_start: int, data_size: int) -> _Entry:
    if not isinstance(raw, dict):
        raise CheckpointError(f"tensor {name!r} must be a JSON object")
    tag = raw.get("dtype")
    if not isinstance(tag, str) or tag not in _DTYPES:
        raise CheckpointError(f"tensor {name!r} has unsupported dtype {tag!r}")
    dtype = _DTYPES[tag]
    shape = raw.get("shape")
    if not isinstance(shape, list) or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in shape):
        raise CheckpointError(f"tensor {name!r} has an invalid shape {shape!r}")
    try:
        # A meta tensor allocates nothing but runs every size and stride
        # check, so a shape PyTorch cannot represent fails here, not later.
        torch.empty(shape, dtype=dtype, device="meta")
    except (RuntimeError, TypeError, OverflowError) as error:
        raise CheckpointError(f"tensor {name!r} has a shape PyTorch cannot represent: {error}") from None
    offsets = raw.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(n, int) and not isinstance(n, bool) for n in offsets)
    ):
        raise CheckpointError(f"tensor {name!r} has invalid data_offsets {offsets!r}")
    start, end = offsets
    if not 0 <= start <= end <= data_size:
        raise CheckpointError(
            f"tensor {name!r} data_offsets {offsets!r} fall outside the {data_size}-byte data section",
        )
    expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if end - start != expected:
        raise CheckpointError(f"tensor {name!r} holds {end - start} bytes but its shape and dtype need {expected}")
    return _Entry(dtype, tag, tuple(shape), data_start + start, end - start)


def _reject_constant(constant: str) -> NoReturn:
    raise CheckpointError(f"{constant} is not valid JSON")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"duplicate header key {key!r}")
        result[key] = value
    return result


__all__ = ["CheckpointError", "FileSlice", "MappedCheckpoint", "TensorSlice", "file_slice", "storage_slice"]
