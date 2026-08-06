"""Per-buffer pinned-CPU storage primitive."""

from dataclasses import dataclass
from typing import Self

import torch

from .tensor_adapters import adopt_cpu_storage, clone_to_pinned_cpu


@dataclass(slots=True, eq=False)
class PinnedBuffer:
    """Pinned host storage for one registered buffer."""

    tensor: torch.Tensor
    target_layout: tuple[object, ...]

    @classmethod
    def clone(cls, buffer: torch.Tensor, *, pin_memory: bool = True) -> Self:
        """Clone into pinned RAM or adopt existing CPU storage."""
        materialize = clone_to_pinned_cpu if pin_memory else adopt_cpu_storage
        tensor = materialize(
            buffer,
            memory_format=torch.contiguous_format,
        )
        return cls(
            tensor=tensor,
            target_layout=cls.target_layout_for(tensor),
        )

    @staticmethod
    def target_layout_for(buffer: torch.Tensor) -> tuple[object, ...]:
        """Opaque target-compatibility layout for ``buffer``."""
        return (
            tuple(buffer.shape),
            tuple(buffer.stride()),
            buffer.dtype,
            buffer.layout,
        )

    @staticmethod
    def bind_layout_for(buffer: torch.Tensor) -> tuple[object, ...]:
        """Opaque bind-compatibility layout for ``buffer``.

        dtype excluded: binding replaces the module's buffer with the
        pinned tensor, so a placeholder's dtype carries no information
        past validation (mirrors :meth:`PinnedParam.bind_layout_for`).
        """
        return (
            tuple(buffer.shape),
            tuple(buffer.stride()),
            buffer.layout,
        )

    @property
    def cache_bytes(self) -> int:
        return self.tensor.nbytes


__all__ = ["PinnedBuffer"]
