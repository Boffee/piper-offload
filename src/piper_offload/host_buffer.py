"""Per-buffer CPU storage primitive."""

from dataclasses import dataclass, field
from typing import Self

import torch

from ._host_backing import HostBacking
from .host_memory import HostMemoryManager
from .tensor_adapters import capture_host_tensor


@dataclass(frozen=True, slots=True, eq=False)
class HostBuffer:
    """Storage for a buffer whose host bytes remain immutable while captured.

    Offload does not persist device-side buffer updates back to host storage.
    Stateful buffers that require such updates are outside this contract.
    """

    tensor: torch.Tensor
    target_layout: tuple[object, ...]
    memory_manager: HostMemoryManager = field(default_factory=HostMemoryManager, repr=False)
    _backings: dict[int, HostBacking] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_backings", self.memory_manager.capture(self.storage_tensors()))

    @classmethod
    def capture(cls, buffer: torch.Tensor, *, memory_manager: HostMemoryManager | None = None) -> Self:
        """Capture contiguous pageable CPU backing, retaining compatible storage."""
        tensor = capture_host_tensor(
            buffer,
            memory_format=torch.contiguous_format,
        )
        return cls(
            tensor=tensor,
            target_layout=cls.target_layout_for(tensor),
            memory_manager=memory_manager if memory_manager is not None else HostMemoryManager(),
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
        host tensor, so a placeholder's dtype carries no information
        past validation (mirrors :meth:`HostParam.bind_layout_for`).
        """
        return (
            tuple(buffer.shape),
            tuple(buffer.stride()),
            buffer.layout,
        )

    @property
    def cache_bytes(self) -> int:
        return self.tensor.nbytes

    def storage_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the existing backing tensor, preserving its storage and view."""
        return (self.tensor,)

    def backing_handles(self) -> tuple[HostBacking, ...]:
        """Shared allocation handles; the CPU buffer keeps its source storage."""
        return tuple(self._backings.values())

    def copy_to_gpu(self, destination: torch.Tensor, *, non_blocking: bool = False) -> None:
        """Copy this buffer using its explicitly owned backing handles."""
        backing = self._backings[self.tensor.untyped_storage()._cdata]
        backing.copy_to(destination, self.tensor, non_blocking=non_blocking)


__all__ = ["HostBuffer"]
