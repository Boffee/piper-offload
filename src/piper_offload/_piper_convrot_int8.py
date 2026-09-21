"""Internal optional-import boundary for Piper ConvRot INT8 support.

``piper-kernels`` owns the :class:`ConvRotInt8Tensor` representation and its
linear, convolution, and matrix-update execution backends. Piper
Offload uses the public wrapper constructor and storage fields to preserve the
representation during movement; its adapter delegates LoRA and dense merges,
including optional stochastic rounding, to those public operations.

``piper-kernels`` is a required dependency; TorchAO is not. ``ConvRotInt8Tensor``
subclasses ``TorchAOBaseTensor``, so this import fails without the ``torchao``
extra and the adapter then reports the format unavailable.
"""

from typing import Any

import torch

LAYOUT_ATTRS = (
    "qdata",
    "scale",
    "act_per_tensor_scale",
    "group_size",
    "dtype",
    "transposed",
)
"""Public ``ConvRotInt8Tensor`` fields preserved by Piper Offload."""


try:
    from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

    PIPER_CONVROT_AVAILABLE = True
except ImportError:
    PIPER_CONVROT_AVAILABLE = False
    ConvRotInt8Tensor: Any = None


def is_convrot_int8_tensor(t: object) -> bool:
    """Return whether ``t`` is a Piper ``ConvRotInt8Tensor``."""
    return PIPER_CONVROT_AVAILABLE and isinstance(t, ConvRotInt8Tensor)


def require_convrot_int8_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated ConvRot tensor, or raise."""
    if not is_convrot_int8_tensor(t):
        raise TypeError(f"expected piper_kernels.weights.convrot.int8.ConvRotInt8Tensor, got {type(t).__name__}")
    validate_layout(t)
    return t


def require_convrot_int8_matrix(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Require a canonical matrix weight before staging a weight update."""
    tensor = require_convrot_int8_tensor(t)
    if tensor.ndim != 2:
        raise NotImplementedError("ConvRot INT8 weight updates require a 2-D weight")
    return tensor


def create_convrot_int8_tensor(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
    act_per_tensor_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rebuild a ConvRot wrapper from canonical storage without copying or repacking."""
    if not PIPER_CONVROT_AVAILABLE:
        raise RuntimeError("piper-kernels[convrot] is required to create a ConvRotInt8Tensor")
    return ConvRotInt8Tensor(
        qdata,
        scale,
        group_size=group_size,
        dtype=dtype,
        act_per_tensor_scale=act_per_tensor_scale,
    )


def validate_layout(t: torch.Tensor) -> None:
    """Validate the public ConvRot storage contract used by the adapter."""
    missing = [attr for attr in LAYOUT_ATTRS if not hasattr(t, attr)]
    if missing:
        raise RuntimeError(
            f"ConvRotInt8Tensor is missing expected attributes {missing!r}; "
            f"piper-offload expects the public layout {LAYOUT_ATTRS}. "
            "piper-kernels likely refactored the wrapper class — upgrade "
            "piper-offload to match."
        )

    # Use the same strict, storage-preserving constructor as reconstruction.
    wrapped: Any = t
    if wrapped.transposed:
        raise NotImplementedError("ConvRot INT8 offload requires an untransposed weight")
    create_convrot_int8_tensor(
        wrapped.qdata,
        wrapped.scale,
        wrapped.group_size,
        wrapped.dtype,
        wrapped.act_per_tensor_scale,
    )
