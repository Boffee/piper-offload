"""Shared contract for CUDA block-residency runtimes."""

import contextlib
from typing import Protocol

import torch

from .block_compile import CompileBackend


class BlockRuntime(Protocol):
    """CUDA working-set lifecycle shared by block-residency strategies.

    Implementations allocate accelerator resources only during
    :meth:`acquire`. Acquisition may fail after partially initializing a
    runtime, so :meth:`release` must be safe for released and partial states
    and release every resource it can before propagating a cleanup error.
    """

    @property
    def acquired(self) -> bool: ...

    @property
    def compile_backend(self) -> CompileBackend:
        """The ``torch.compile`` backend required by this strategy."""
        ...

    def acquire(self, device: torch.device) -> None:
        """Allocate the CUDA working set and install execution hooks."""
        ...

    def release(self) -> None:
        """Idempotently release the CUDA working set from any state."""
        ...

    def optimizer_step(self) -> contextlib.AbstractContextManager[None]: ...


__all__ = ["BlockRuntime"]
