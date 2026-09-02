"""Shared execution contract for activation-scoped parameter updates."""

from typing import Protocol

from torch import nn


class ParameterTransform(Protocol):
    """A validated update that can be reapplied after parameter copies."""

    def validate_parameter(self, param: nn.Parameter) -> None:
        """Validate ``param`` and prepare any reusable execution state."""
        ...

    def apply_parameter(self, param: nn.Parameter) -> None:
        """Apply the prepared update to ``param`` in place."""
        ...


__all__ = ["ParameterTransform"]
