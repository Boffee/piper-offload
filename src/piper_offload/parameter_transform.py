"""Shared execution contract for activation-scoped parameter updates."""

from typing import Protocol

import torch
from torch import nn


class ParameterTransform(Protocol):
    """A validated update that can be reapplied after parameter copies."""

    def validate_parameter(self, param: nn.Parameter) -> None:
        """Validate ``param`` and prepare any reusable execution state."""
        ...

    def apply_parameter(self, param: nn.Parameter) -> None:
        """Apply the prepared update to ``param`` in place."""
        ...

    def storage_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return physical CPU tensors read while applying the update."""
        ...


__all__ = ["ParameterTransform"]
