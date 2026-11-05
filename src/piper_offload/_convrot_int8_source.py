"""Shared behavior of packed sources that activate as ConvRot INT8 targets.

GGUF and trellis-coded sources keep their packed bytes in host backing, and each
refill decodes them into a reusable ConvRot INT8 target. Once active, the
parameter is an ordinary ``ConvRotInt8Tensor``, so dequantization and LoRA and
dense updates delegate to :class:`PiperConvRotInt8Adapter`. The packed source is
inference-only and cannot be updated permanently. Each subclass owns its capture,
identity, staging, and refill.
"""

from __future__ import annotations

from typing import ClassVar, Protocol

import torch
from torch import nn

from .piper_convrot_int8_adapter import PiperConvRotInt8Adapter


class _TargetState(Protocol):
    """GPU state holding the reusable ConvRot INT8 target."""

    @property
    def target(self) -> torch.Tensor: ...


class ConvRotInt8SourceAdapter:
    """Base adapter for packed sources whose active representation is ConvRot INT8."""

    _SOURCE_NAME: ClassVar[str]
    """The source format's name in error messages, such as ``"packed GGUF"``."""

    @classmethod
    def _require_frozen(cls, requires_grad: bool) -> None:
        if requires_grad:
            raise ValueError(f"{cls._SOURCE_NAME} parameters are inference-only")

    @classmethod
    def gpu_param(
        cls,
        host: object,
        gpu_state: _TargetState,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        del host
        cls._require_frozen(requires_grad)
        return nn.Parameter(gpu_state.target, requires_grad=False)

    @classmethod
    def validate_permanent_update(cls) -> None:
        raise ValueError(
            f"Permanent updates to {cls._SOURCE_NAME} parameters are unsupported; "
            "use activation-time adapter merge instead."
        )

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        """Delegate active-value dequantization to ConvRot INT8."""
        return PiperConvRotInt8Adapter.dequantize(t)

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        """Delegate an activation-time LoRA update to ConvRot INT8."""
        PiperConvRotInt8Adapter.merge_lora_(
            target, b, a, alpha, beta=beta, rounding_seed=rounding_seed
        )

    @staticmethod
    def merge_dense_(
        target: torch.Tensor,
        update: torch.Tensor,
        alpha: float,
        *,
        beta: float = 1.0,
        rounding_seed: int | None = None,
    ) -> None:
        """Delegate an activation-time dense update to ConvRot INT8."""
        PiperConvRotInt8Adapter.merge_dense_(
            target, update, alpha, beta=beta, rounding_seed=rounding_seed
        )


__all__ = ["ConvRotInt8SourceAdapter"]
