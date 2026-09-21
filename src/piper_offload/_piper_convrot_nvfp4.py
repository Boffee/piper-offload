"""Internal optional-import boundary for Piper ConvRot NVFP4 support.

``piper-kernels`` owns the semantic tensor, grouped rotation, and in-place
``addmm_`` and ``add_`` updates. Piper Offload only preserves its public packed
storage and metadata during movement, then delegates LoRA and dense merges to
those operations.

``piper-kernels`` is a required dependency; TorchAO is not. This import reaches
TorchAO's NVFP4 encoder, so it fails without the ``torchao`` extra and the
adapter then reports the format unavailable.
"""

from typing import Any, cast

import torch

from ._torchao_nvfp4 import LAYOUT_ATTRS as NVFP4_LAYOUT_ATTRS

LAYOUT_ATTRS = (*NVFP4_LAYOUT_ATTRS, "group_size", "high_first")
"""Public ``ConvRotNVFP4Tensor`` fields preserved by Piper Offload."""


try:
    from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor

    PIPER_CONVROT_NVFP4_AVAILABLE = True
except ImportError:
    PIPER_CONVROT_NVFP4_AVAILABLE = False
    ConvRotNVFP4Tensor: Any = None


def is_convrot_nvfp4_tensor(t: object) -> bool:
    """Return whether ``t`` is a Piper ``ConvRotNVFP4Tensor``."""
    return PIPER_CONVROT_NVFP4_AVAILABLE and isinstance(t, ConvRotNVFP4Tensor)


def require_convrot_nvfp4_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated ConvRot NVFP4 tensor, or raise."""
    if not is_convrot_nvfp4_tensor(t):
        raise TypeError(f"expected piper_kernels.weights.convrot.nvfp4.ConvRotNVFP4Tensor, got {type(t).__name__}")
    validate_layout(t)
    return t


def create_convrot_nvfp4_tensor(  # noqa: PLR0913, PLR0917
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_size: int,
    orig_dtype: torch.dtype,
    group_size: int,
    per_tensor_scale: torch.Tensor | None,
    act_per_tensor_scale: torch.Tensor | None,
    is_swizzled_scales: bool,
    use_triton_kernel: bool,
    act_quant_kwargs: object | None,
    high_first: bool,
    *,
    wrapper_type: type[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Rebuild a ConvRot NVFP4 wrapper from storage and metadata."""
    if not PIPER_CONVROT_NVFP4_AVAILABLE:
        raise RuntimeError("piper-kernels[convrot]>=0.6 is required to create a ConvRotNVFP4Tensor")
    constructor = cast(
        Any,
        ConvRotNVFP4Tensor if wrapper_type is None else wrapper_type,
    )
    if not issubclass(constructor, ConvRotNVFP4Tensor):
        raise TypeError(f"ConvRot NVFP4 wrapper type must subclass ConvRotNVFP4Tensor, got {constructor.__name__}")
    return constructor(
        qdata=qdata,
        scale=scale,
        block_size=block_size,
        orig_dtype=orig_dtype,
        group_size=group_size,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=act_per_tensor_scale,
        is_swizzled_scales=is_swizzled_scales,
        use_triton_kernel=use_triton_kernel,
        act_quant_kwargs=act_quant_kwargs,
        high_first=high_first,
    )


def validate_layout(t: torch.Tensor) -> None:
    """Validate the public ConvRot NVFP4 storage contract."""
    missing = [attr for attr in LAYOUT_ATTRS if not hasattr(t, attr)]
    if missing:
        raise RuntimeError(
            f"ConvRotNVFP4Tensor is missing expected attributes {missing!r}; "
            f"piper-offload expects the public layout {LAYOUT_ATTRS}. "
            "Upgrade piper-offload to match piper-kernels."
        )

    wrapped: Any = t
    create_convrot_nvfp4_tensor(
        wrapped.qdata,
        wrapped.scale,
        wrapped.block_size,
        wrapped.orig_dtype,
        wrapped.group_size,
        wrapped.per_tensor_scale,
        wrapped.act_per_tensor_scale,
        wrapped.is_swizzled_scales,
        wrapped.use_triton_kernel,
        wrapped.act_quant_kwargs,
        wrapped.high_first,
        wrapper_type=type(wrapped),
    )
