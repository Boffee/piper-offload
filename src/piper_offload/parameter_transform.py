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


class ParameterTransformSequence:
    """Ordered composition of parameter transforms behind one copy hook."""

    __slots__ = ("_transforms",)

    def __init__(self, *transforms: ParameterTransform) -> None:
        if not transforms:
            raise ValueError("ParameterTransformSequence requires a transform")
        self._transforms = transforms

    def validate_parameter(self, param: nn.Parameter) -> None:
        for transform in self._transforms:
            transform.validate_parameter(param)

    def apply_parameter(self, param: nn.Parameter) -> None:
        for transform in self._transforms:
            transform.apply_parameter(param)


__all__ = ["ParameterTransform", "ParameterTransformSequence"]
