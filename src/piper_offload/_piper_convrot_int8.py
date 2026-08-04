"""Internal optional-import boundary for Piper ConvRot INT8 support.

``piper-kernels`` owns the :class:`ConvRotInt8Tensor` representation and its
linear and in-place ``addmm_`` execution backends. Piper Offload uses the
public wrapper constructor and storage fields to preserve the representation
during movement; its adapter delegates LoRA merges to the public ``addmm_``.

The dependency remains optional: importing :mod:`piper_offload` succeeds when
``piper-kernels`` (or its ``convrot`` extra) is absent.
"""

from typing import Any

import torch

LAYOUT_ATTRS = (
    "qdata",
    "scale",
    "group_size",
    "dtype",
)
"""Public ``ConvRotInt8Tensor`` fields preserved by Piper Offload."""


try:
    from piper_kernels.convrot import ConvRotInt8Tensor

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
        raise TypeError(
            f"expected piper_kernels.convrot.ConvRotInt8Tensor, "
            f"got {type(t).__name__}"
        )
    validate_layout(t)
    return t


def create_convrot_int8_tensor(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rebuild a ConvRot wrapper from its public storage and metadata."""
    if not PIPER_CONVROT_AVAILABLE:
        raise RuntimeError(
            "piper-kernels[convrot] is required to create a "
            "ConvRotInt8Tensor"
        )
    return ConvRotInt8Tensor(qdata, scale, group_size, dtype)


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

    # The public constructor is also piper-kernels' storage validator. Rebuild
    # a cheap wrapper so mutations to public fields cannot enter the offload
    # path with invalid qdata/scale/group metadata.
    wrapped: Any = t
    create_convrot_int8_tensor(
        wrapped.qdata,
        wrapped.scale,
        wrapped.group_size,
        wrapped.dtype,
    )
