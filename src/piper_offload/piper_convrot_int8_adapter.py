"""Piper ``ConvRotInt8Tensor`` offload adapter.

``piper-kernels`` owns ConvRot's tensor semantics and execution backends.
Piper Offload owns the movement and merge integration: pin the public ``qdata``
and ``scale`` storage tensors, move those bytes, reconstruct the wrapper with
its ``group_size`` and logical ``dtype`` metadata, and delegate staged LoRA
updates to the public in-place ``ConvRotInt8Tensor.addmm_`` operation.
When stochastic rounding is requested, Piper Offload forwards its stable
per-target seed to the kernel-owned terminal INT8 code selection.

The adapter remains frozen-only: it does not advertise CPU round-trip or
trainable ``Parameter.data`` swap. Permanent and activation-time LoRA merge
preserve the existing target wrapper and storage identities. Piper selects its
optimized Triton backend on supported CUDA devices and its portable reference
backend elsewhere. Routed LoRA remains available when the base must remain
untouched.
"""

from dataclasses import dataclass
from typing import Any

import torch

from ._piper_convrot_int8 import (
    create_convrot_int8_tensor,
    is_convrot_int8_tensor,
    require_convrot_int8_tensor,
    validate_layout,
)
from .torchao_structured_adapter import TorchaoStructuredAdapter


@dataclass(slots=True, frozen=True)
class _PiperConvRotInt8Meta:
    """ConvRot metadata needed to reconstruct a storage wrapper."""

    group_size: int
    dtype: torch.dtype


class PiperConvRotInt8Adapter(
    TorchaoStructuredAdapter[_PiperConvRotInt8Meta]
):
    """Adapter for ``piper_kernels.convrot.ConvRotInt8Tensor`` weights."""

    _TAG = "piper-kernels-convrot-int8"
    _STORAGE_NAMES = ("qdata", "scale")

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_convrot_int8_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_convrot_int8_tensor(t)

    @staticmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        return (t.qdata, t.scale)

    @staticmethod
    def _meta_of(t: Any) -> _PiperConvRotInt8Meta:  # noqa: ANN401
        return _PiperConvRotInt8Meta(
            group_size=t.group_size,
            dtype=t.dtype,
        )

    @staticmethod
    def _reconstruct(
        storage: tuple[torch.Tensor | None, ...],
        meta: _PiperConvRotInt8Meta,
    ) -> torch.Tensor:
        qdata, scale = storage
        assert qdata is not None
        assert scale is not None
        return create_convrot_int8_tensor(
            qdata,
            scale,
            meta.group_size,
            meta.dtype,
        )

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return (t.group_size, t.dtype)

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return t.dtype

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        PiperConvRotInt8Adapter.validate_lora_merge(
            target,
            b,
            a,
            strength,
            rounding_seed=rounding_seed,
        )
        require_convrot_int8_tensor(target).addmm_(
            b,
            a,
            alpha=strength,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def validate_lora_merge(
        target: torch.Tensor,
        _b: torch.Tensor,
        _a: torch.Tensor,
        _strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        del rounding_seed
        require_convrot_int8_tensor(target)
