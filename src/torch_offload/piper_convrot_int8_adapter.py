"""Piper ``ConvRotInt8Tensor`` offload adapter.

``piper-kernels`` owns ConvRot's tensor semantics and execution backends.
torch-offload owns only the movement integration: pin the public ``qdata`` and
``scale`` storage tensors, move those bytes, and reconstruct the wrapper with
its ``group_size`` and logical ``dtype`` metadata.

The adapter intentionally exposes frozen-inference movement only. It does not
advertise CPU round-trip, trainable ``Parameter.data`` swap, requantization,
copy, or staged LoRA merge capabilities. Routed LoRA remains available when
the owning module is a compatible logical ``nn.Linear``.
"""

from __future__ import annotations

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
