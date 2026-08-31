"""Internal optional-import module for TorchAO NVFP4 support.

Single source of truth for the private-ish TorchAO NVFP4 layout this
repo needs to move packed weights through :class:`PinnedParam` and to
expose the dequantize/requantize adapter capability. TorchAO's public
workflow creates ``NVFP4Tensor`` weights via
``quantize_(..., NVFP4WeightOnlyConfig/NVFP4DynamicActivation...)``;
the adapter only preserves, moves, and (for LoRA merge) re-encodes those
already-quantized tensors.
"""

from typing import Any, cast

import torch

from ._stochastic_quantization import _stochastic_codebook_indices

LAYOUT_ATTRS = (
    "qdata",
    "scale",
    "block_size",
    "orig_dtype",
    "per_tensor_scale",
    "act_per_tensor_scale",
    "is_swizzled_scales",
    "use_triton_kernel",
    "act_quant_kwargs",
)
"""Attributes this repo reads from a TorchAO ``NVFP4Tensor``."""


try:
    from torchao.prototype.mx_formats.kernels import (
        f4_unpacked_to_f32,
        pack_uint4,
        unpack_uint4,
    )
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    TORCHAO_NVFP4_AVAILABLE = True
except ImportError:
    TORCHAO_NVFP4_AVAILABLE = False
    NVFP4Tensor: Any = None
    f4_unpacked_to_f32: Any = None
    pack_uint4: Any = None
    per_tensor_amax_to_scale: Any = None
    unpack_uint4: Any = None


def is_nvfp4_tensor(t: object) -> bool:
    """Return whether ``t`` is a TorchAO NVFP4 tensor."""
    return TORCHAO_NVFP4_AVAILABLE and isinstance(t, NVFP4Tensor)


def require_nvfp4_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated TorchAO NVFP4 tensor, or raise."""
    if not is_nvfp4_tensor(t):
        raise TypeError(f"expected TorchAO NVFP4Tensor, got {type(t).__name__}")
    validate_layout(t)
    return t


def create_nvfp4_tensor(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_size: int,
    orig_dtype: torch.dtype,
    per_tensor_scale: torch.Tensor | None,
    act_per_tensor_scale: torch.Tensor | None,
    is_swizzled_scales: bool,
    use_triton_kernel: bool,
    act_quant_kwargs: object | None,
    *,
    wrapper_type: type[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Rebuild a compatible NVFP4 wrapper from raw storage + metadata."""
    if not TORCHAO_NVFP4_AVAILABLE:
        raise RuntimeError("torchao is required to create an NVFP4Tensor")
    constructor = cast(Any, NVFP4Tensor if wrapper_type is None else wrapper_type)
    if not issubclass(constructor, NVFP4Tensor):
        raise TypeError(f"NVFP4 wrapper type must subclass TorchAO NVFP4Tensor, got {constructor.__name__}")
    return constructor(
        qdata=qdata,
        scale=scale,
        block_size=block_size,
        orig_dtype=orig_dtype,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=act_per_tensor_scale,
        is_swizzled_scales=is_swizzled_scales,
        use_triton_kernel=use_triton_kernel,
        act_quant_kwargs=act_quant_kwargs,
    )


def validate_layout(t: torch.Tensor) -> None:
    """Raise if ``t`` is missing the NVFP4 attributes we preserve."""
    missing = [a for a in LAYOUT_ATTRS if not hasattr(t, a)]
    if not missing:
        return
    raise RuntimeError(
        f"NVFP4Tensor is missing expected attributes {missing!r}; "
        f"this repo is pinned to a layout that exposes {LAYOUT_ATTRS}. "
        "TorchAO likely refactored the wrapper class — upgrade "
        "piper-offload to match."
    )


def dequantize_nvfp4_tensor(t: torch.Tensor) -> torch.Tensor:
    """Return the dense logical value in the wrapper's original dtype."""
    nv = require_nvfp4_tensor(t)
    return nv.dequantize(nv.orig_dtype)


def requantize_nvfp4_tensor(
    t: torch.Tensor,
    *,
    like: torch.Tensor,
    rounding_seed: int | None = None,
) -> torch.Tensor:
    """Encode dense ``t`` using the NVFP4 layout and metadata from ``like``.

    Goes through the public ``NVFP4Tensor.to_nvfp4``, which recomputes the
    FP8 (E4M3) block scales from the new values. When ``like`` uses
    two-level scaling, the global ``per_tensor_scale`` is recomputed from
    the merged amax via ``per_tensor_amax_to_scale`` so a LoRA merge that
    grows the global range is absorbed rather than clipped; the
    activation-side ``act_per_tensor_scale`` and ``act_quant_kwargs`` carry
    over from ``like`` unchanged, since a weight-side merge does not touch
    activation calibration.

    Re-encoding always uses the torch reference path
    (``use_triton_kernel=False``): it produces the identical swizzled-scale
    layout without NVFP4's optional Triton/mslk dependency, and
    ``copy_into`` writes the bytes into ``like``'s wrapper, which keeps its
    own ``use_triton_kernel`` flag for the forward matmul.

    When ``rounding_seed`` is provided, that deterministic re-encode still
    establishes the final global/block scales and metadata first. Only the
    terminal E2M1 codes are then replaced in place using stochastic rounding.
    """
    nv = require_nvfp4_tensor(like)
    if tuple(t.shape) != tuple(nv.shape):
        raise ValueError(
            f"Cannot requantize tensor with shape {tuple(t.shape)} like NVFP4Tensor with shape {tuple(nv.shape)}."
        )
    if not nv.qdata.is_contiguous():
        # A transposed/strided NVFP4 weight has a packed layout that the
        # re-encode (which always produces the standard contiguous packing)
        # can neither consume nor fill — to_nvfp4 rejects non-contiguous
        # input, and even past that the packed shape would not match the
        # target's. Reject early with an actionable error rather than an
        # opaque kernel assertion deep in the merge.
        raise ValueError(
            "Cannot merge LoRA into a non-contiguous (e.g. transposed) "
            "NVFP4 weight: requantization produces the standard packed "
            "layout, which cannot fill a transposed target. Use routed LoRA "
            "for this weight."
        )
    per_tensor_scale = None
    if nv.per_tensor_scale is not None:
        if nv.per_tensor_scale.numel() != 1:
            # The merge recomputes one global per-tensor scale from the
            # merged amax. A non-scalar per_tensor_scale (e.g. per-expert
            # scales on a 3-D grouped/MoE weight) would be collapsed to a
            # single global range — silently dropping per-group precision,
            # with copy_into broadcasting the scalar across every slot.
            # TorchAO's to_nvfp4 only accepts a scalar per_tensor_scale
            # today (and a 3-D weight is rejected earlier by LoRA
            # factor-shape validation), so this guards a future layout:
            # reject it loudly rather than collapse it.
            raise ValueError(
                "Cannot merge LoRA into an NVFP4 weight with a non-scalar "
                "per_tensor_scale (e.g. per-expert grouped/MoE scales); the "
                "merge recomputes a single global scale and would drop "
                "per-group precision. Use routed LoRA for this weight."
            )
        # An all-zero merged weight (e.g. a LoRA delta that exactly cancels
        # the base) has amax 0, so the recomputed global scale is 0 — and
        # to_nvfp4's two-level path then divides block scales by it, emitting
        # NaN. Floor the scale to a minimum epsilon: the standard zero-amax
        # guard (matching int8's choose_qparams eps). A positive global scale
        # encodes an all-zero weight identically — every block scale is 0 —
        # so the result is an exact zero tensor.
        per_tensor_scale = per_tensor_amax_to_scale(t.detach().abs().max()).clamp_min(torch.finfo(torch.float32).eps)
    out = NVFP4Tensor.to_nvfp4(
        t.to(dtype=nv.orig_dtype),
        block_size=nv.block_size,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=nv.act_per_tensor_scale,
        is_swizzled_scales=nv.is_swizzled_scales,
        use_triton_kernel=False,
        act_quant_kwargs=nv.act_quant_kwargs,
    )
    out = create_nvfp4_tensor(
        out.qdata,
        out.scale,
        out.block_size,
        out.orig_dtype,
        out.per_tensor_scale,
        out.act_per_tensor_scale,
        out.is_swizzled_scales,
        out.use_triton_kernel,
        out.act_quant_kwargs,
        wrapper_type=type(nv),
    )
    if rounding_seed is not None:
        _stochastic_recode_nvfp4_(out, t, rounding_seed=rounding_seed)
    return out


def _stochastic_recode_nvfp4_(
    out: torch.Tensor,
    source: torch.Tensor,
    *,
    rounding_seed: int,
) -> None:
    """Replace only ``out``'s terminal codes using its finalized scales."""
    nv = require_nvfp4_tensor(out)
    hp_scale = nv.get_hp_scales()
    element_scale = hp_scale.repeat_interleave(nv.block_size, dim=-1)
    valid_scale = torch.isfinite(element_scale) & (element_scale > 0)
    normalized = torch.where(
        valid_scale,
        source.to(torch.float32) / element_scale.to(torch.float32),
        torch.zeros_like(source, dtype=torch.float32),
    )
    deterministic_codes = unpack_uint4(nv.qdata.contiguous().view(torch.uint8))
    codebook = f4_unpacked_to_f32(torch.arange(16, device=source.device, dtype=torch.uint8))
    codes = _stochastic_codebook_indices(
        normalized,
        codebook,
        seed=rounding_seed,
        deterministic=deterministic_codes,
    )
    codes = torch.where(
        valid_scale,
        codes,
        deterministic_codes.to(torch.int64),
    )
    nv.qdata.view(torch.uint8).copy_(pack_uint4(codes.to(torch.uint8)))
