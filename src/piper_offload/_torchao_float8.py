"""Internal optional-import module for TorchAO scaled-FP8 support.

Single source of truth for the TorchAO ``Float8Tensor`` layout this
repo needs to move scaled-fp8 weights through :class:`PinnedParam` and
to expose the dequantize/requantize adapter capability. TorchAO's
public workflow creates ``Float8Tensor`` weights via
``quantize_(..., Float8WeightOnlyConfig/Float8DynamicActivation...)``;
the adapter only preserves, moves, and (for LoRA merge) re-encodes
those already-quantized tensors.
"""

from typing import Any

import torch

from ._stochastic_quantization import _stochastic_cast_float8
from ._torchao_granularity import (
    expand_block_parameter,
    granularity_from_block_size,
)

LAYOUT_ATTRS = (
    "qdata",
    "scale",
    "block_size",
    "mm_config",
    "act_quant_kwargs",
    "kernel_preference",
)
"""Attributes this repo reads from a TorchAO ``Float8Tensor``."""


try:
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
        Float8Tensor,
    )

    TORCHAO_FLOAT8_AVAILABLE = True
except ImportError:
    TORCHAO_FLOAT8_AVAILABLE = False
    Float8Tensor: Any = None


def is_float8_tensor(t: object) -> bool:
    """Return whether ``t`` is a TorchAO scaled-fp8 tensor."""
    return TORCHAO_FLOAT8_AVAILABLE and isinstance(t, Float8Tensor)


def require_float8_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated TorchAO Float8Tensor, or raise."""
    if not is_float8_tensor(t):
        raise TypeError(f"expected TorchAO Float8Tensor, got {type(t).__name__}")
    validate_layout(t)
    return t


def create_float8_tensor(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_size: list[int],
    mm_config: object | None,
    act_quant_kwargs: object | None,
    kernel_preference: object,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rebuild a TorchAO ``Float8Tensor`` from raw storage + metadata."""
    if not TORCHAO_FLOAT8_AVAILABLE:
        raise RuntimeError("torchao is required to create a Float8Tensor")
    return Float8Tensor(
        qdata,
        scale,
        block_size=block_size,
        mm_config=mm_config,
        act_quant_kwargs=act_quant_kwargs,
        kernel_preference=kernel_preference,
        dtype=dtype,
    )


def validate_layout(t: torch.Tensor) -> None:
    """Raise if ``t`` is missing the Float8 attributes we preserve."""
    missing = [a for a in LAYOUT_ATTRS if not hasattr(t, a)]
    if not missing:
        return
    raise RuntimeError(
        f"Float8Tensor is missing expected attributes {missing!r}; "
        f"this repo is pinned to a layout that exposes {LAYOUT_ATTRS}. "
        "TorchAO likely refactored the wrapper class — upgrade "
        "piper-offload to match."
    )


def dequantize_float8_tensor(t: torch.Tensor) -> torch.Tensor:
    """Return the dense logical value in the wrapper's compute dtype."""
    f8 = require_float8_tensor(t)
    return f8.dequantize()


def validate_float8_requantize_layout(t: torch.Tensor) -> object:
    """Return the recoverable granularity or reject an unencodable layout.

    This is also the adapter's LoRA-merge preflight: the generic fallback can
    only rebuild layouts that TorchAO's public ``from_hp`` constructor can
    express. In particular, transposing a PerGroup weight moves its groups off
    the last axis and cannot be represented by ``PerGroup`` on re-encode.
    """
    f8 = require_float8_tensor(t)
    try:
        return granularity_from_block_size(
            tuple(f8.block_size),
            tuple(f8.shape),
        )
    except ValueError as exc:
        raise ValueError(
            "Cannot re-encode this Float8Tensor layout. TorchAO can rebuild "
            "PerTensor, PerRow, and last-axis PerGroup weights, but not a "
            "transposed PerGroup layout. Use routed LoRA instead of merging "
            "into this weight."
        ) from exc


def requantize_float8_tensor(
    t: torch.Tensor,
    *,
    like: torch.Tensor,
    rounding_seed: int | None = None,
) -> torch.Tensor:
    """Encode dense ``t`` using the fp8 layout and metadata from ``like``.

    Goes through the public ``Float8Tensor.from_hp`` so the scale is
    recomputed for the new values (a LoRA merge can grow the per-block
    amax past what ``like``'s scale covers). Granularity is recovered
    from ``like.block_size``; all dispatch metadata (mm config, kernel
    preference, activation quant kwargs, fp8 dtype) carries over.

    Zero blocks are repaired afterwards: ``from_hp`` computes
    ``scale = amax / fp8_max`` with no epsilon floor, so an all-zero block
    (a zeroed group or row, or a fully cancelled weight) gets ``scale = 0``
    and ``qdata = 0 / 0 = NaN``. See :func:`_repair_zero_scale_blocks`.
    When ``rounding_seed`` is supplied, that finalized representation keeps its
    scales and metadata while only its terminal FP8 codes are replaced.
    """
    f8 = require_float8_tensor(like)
    if tuple(t.shape) != tuple(f8.shape):
        raise ValueError(
            f"Cannot requantize tensor with shape {tuple(t.shape)} like Float8Tensor with shape {tuple(f8.shape)}."
        )
    granularity = validate_float8_requantize_layout(f8)
    out = Float8Tensor.from_hp(
        t.to(dtype=f8.dtype),
        float8_dtype=f8.qdata.dtype,
        granularity=granularity,
        mm_config=f8.mm_config,
        kernel_preference=f8.kernel_preference,
        act_quant_kwargs=f8.act_quant_kwargs,
    )
    out = _repair_zero_scale_blocks(out)
    if rounding_seed is not None:
        _stochastic_recode_float8_(t, target=out, rounding_seed=rounding_seed)
    return out


def _stochastic_recode_float8_(
    t: torch.Tensor,
    *,
    target: torch.Tensor,
    rounding_seed: int,
) -> None:
    """Replace only finalized TorchAO FP8 terminal codes in place."""
    out = require_float8_tensor(target)
    expanded_scale = expand_block_parameter(
        out.scale,
        block_size=tuple(out.block_size),
        shape=tuple(out.shape),
    )
    valid_scale = torch.isfinite(expanded_scale) & (expanded_scale > 0)
    scaled = torch.where(
        valid_scale,
        t.to(torch.float32) / expanded_scale.to(torch.float32),
        torch.zeros_like(t, dtype=torch.float32),
    )
    qdata = _stochastic_cast_float8(
        scaled,
        out.qdata.dtype,
        seed=rounding_seed,
        deterministic=out.qdata,
    )
    out.qdata.copy_(torch.where(valid_scale, qdata, out.qdata))


def _repair_zero_scale_blocks(f8: Any) -> torch.Tensor:  # noqa: ANN401
    """Repair NaN produced by an all-zero scaling block.

    TorchAO's ``Float8Tensor.from_hp`` computes ``scale = amax / fp8_max``
    with no epsilon floor, so a block whose values are all zero gets
    ``scale = 0`` and ``qdata = 0 / 0 = NaN`` (inherited from torchao —
    ``quantize_`` of a zeroed row produces the same NaN). Floor those zero
    scales to a minimum epsilon — the standard zero-amax guard, matching
    int8's ``choose_qparams`` ``eps`` — and zero the NaN ``qdata``: a zero
    block dequantizes to 0 under any positive scale, so the result is an
    exact zero rather than NaN. Non-zero blocks are untouched.
    """
    zero = f8.scale == 0
    if not bool(zero.any()):
        return f8
    eps = torch.finfo(torch.float32).eps
    scale = torch.where(zero, torch.full_like(f8.scale, eps), f8.scale)
    qdata_zero = _expand_block_mask(
        zero,
        block_size=tuple(f8.block_size),
        shape=tuple(f8.qdata.shape),
    )
    qdata = torch.where(qdata_zero, torch.zeros_like(f8.qdata), f8.qdata)
    return create_float8_tensor(
        qdata,
        scale,
        list(f8.block_size),
        f8.mm_config,
        f8.act_quant_kwargs,
        f8.kernel_preference,
        f8.dtype,
    )


def _expand_block_mask(
    mask: torch.Tensor,
    *,
    block_size: tuple[int, ...],
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Expand one boolean value per quantization block to logical elements."""
    if len(block_size) != len(shape) or any(
        block <= 0 or size % block != 0 for size, block in zip(shape, block_size, strict=True)
    ):
        raise RuntimeError(
            "Float8Tensor has an invalid block_size for zero-scale repair: "
            f"block_size={block_size!r}, qdata shape={shape!r}."
        )

    block_grid = tuple(size // block for size, block in zip(shape, block_size, strict=True))
    expected_numel = 1
    for size in block_grid:
        expected_numel *= size
    if mask.numel() != expected_numel:
        raise RuntimeError(
            "Float8Tensor scale layout does not match its quantization blocks: "
            f"scale shape={tuple(mask.shape)!r}, expected block grid={block_grid!r}."
        )

    expanded = mask.reshape(block_grid)
    for dim, repeat in enumerate(block_size):
        if repeat != 1:
            expanded = expanded.repeat_interleave(repeat, dim=dim)
    if tuple(expanded.shape) != shape:
        raise RuntimeError(
            "Float8Tensor zero-scale mask did not expand to qdata shape: "
            f"expanded={tuple(expanded.shape)!r}, qdata={shape!r}."
        )
    return expanded
