"""Shared read-only mapping lifetime and storage provenance for checkpoint readers."""

import io
import mmap
import threading
import warnings
import weakref
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class FileSlice:
    """Where a storage's bytes live: ``length`` bytes at ``offset`` in the reader's open ``file``."""

    file: io.BufferedReader
    offset: int
    length: int


class MappedFile:
    """Buffer the tensors are built over.

    Every tensor's storage references this object through the buffer
    protocol, so the file and the mapping live exactly as long as the tensors.
    Provenance references it weakly.
    """

    __slots__ = ("__weakref__", "file", "mmap")

    def __init__(self, file: io.BufferedReader, mapping: mmap.mmap) -> None:
        self.file = file
        self.mmap = mapping
        weakref.finalize(self, _forget_dead_mappings)

    def __buffer__(self, flags: int, /) -> memoryview:
        return memoryview(self.mmap)

    def tensor(self, offset: int, length: int, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        """A tensor-sized storage over this mapping, recording its exact file bytes."""
        if length == 0:
            return torch.empty(shape, dtype=dtype)
        window = memoryview(self)[offset:offset + length]
        with warnings.catch_warnings():
            # Non-writable is the point: callers never write through a checkpoint mapping.
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            tensor = torch.frombuffer(window, dtype=dtype).reshape(shape)
        with _lock:
            _provenance[tensor.untyped_storage().data_ptr()] = (
                weakref.ref(self), FileSlice(self.file, offset, length),
            )
        return tensor


# Reentrant: ``_forget_dead_mappings`` is a garbage-collection finalizer, so it can
# run in a thread that already holds this lock, at any allocation inside a locked
# block. A plain Lock deadlocks that thread against itself.
_lock = threading.RLock()
# Storage pointer -> (its mapping, weakly, and its slice).
_provenance: dict[int, tuple[weakref.ReferenceType[MappedFile], FileSlice]] = {}


def file_slice(source: torch.Tensor | torch.UntypedStorage) -> FileSlice | None:
    """The checkpoint bytes behind a tensor's storage, or ``None`` for storage that is not a mapped checkpoint.

    Views share their base tensor's storage and so its slice. Storage that
    merely reuses a dead mapping's address does not match.
    For structured parameters, query their underlying storage tensor (for
    example ``GgufParameter.as_tensor()``), not the logical wrapper.
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
            _provenance.pop(pointer, None)
