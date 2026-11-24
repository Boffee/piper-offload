"""Per-buffer CPU storage primitive."""

from dataclasses import dataclass
from typing import Self

import torch

from .tensor_adapters import capture_host_tensor


@dataclass(slots=True, eq=False)
class HostBuffer:
    """Host storage for one registered buffer."""

    tensor: torch.Tensor
    target_layout: tuple[object, ...]

    @classmethod
    def capture(cls, buffer: torch.Tensor) -> Self:
        """Capture contiguous pageable CPU backing, retaining compatible storage."""
        tensor = capture_host_tensor(
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

        ``dtype`` and ``stride`` are excluded: binding replaces the module's
        buffer with the normalized contiguous host tensor, so placeholder
        dtype and layout strides carry no constraint past shape and layout
        (mirrors :meth:`HostParam.accepts_bind_target`).
        """
        return (
            tuple(buffer.shape),
            buffer.layout,
        )

    @property
    def cache_bytes(self) -> int:
        return self.tensor.nbytes

    def storage_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the existing backing tensor, preserving its storage and view."""
        return (self.tensor,)


__all__ = ["HostBuffer"]
