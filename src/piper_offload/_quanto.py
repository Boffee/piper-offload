"""Internal optional-import module for ``optimum-quanto`` support.

Single source of truth for everything this repo depends on from
optimum-quanto:

- The ``from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor``
  optional import and the ``QUANTO_AVAILABLE`` flag.
- :data:`LAYOUT_ATTRS` — the private-attr names this repo reads on a
  ``WeightQBytesTensor``.
- :func:`dequantize_qbytes_tensor` and
  :func:`requantize_qbytes_tensor` — the pieces used by
  :class:`~piper_offload.quanto_adapter.QuantoAdapter` to expose a
  dequantize/requantize adapter capability.

Both pin/move/wrap and dequantize/requantize support consume from here
through :mod:`quanto_adapter`, so the layout assumption only has to be
updated once when optimum-quanto refactors.

Pinned to optimum-quanto's internal layout. Not part of the public API.
"""

from typing import Any

import torch

from ._stochastic_quantization import (
    _stochastic_cast_float8,
    _stochastic_round_to_int,
)

LAYOUT_ATTRS = ("_data", "_scale", "qtype", "axis")
"""Attributes this repo reads from a ``WeightQBytesTensor``.

If optimum-quanto refactors and any of these vanishes, callers that
go through :func:`validate_layout` get a framed :class:`RuntimeError`
naming the missing attribute(s); callers that don't would otherwise
fail with a generic ``AttributeError`` later in the access path.
"""


try:
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

    QUANTO_AVAILABLE = True
except ImportError:
    QUANTO_AVAILABLE = False
    WeightQBytesTensor: Any = None

try:
    from optimum.quanto.tensor.weights.marlin.fp8 import MarlinF8QBytesTensor
except ImportError:
    MarlinF8QBytesTensor: Any = None


def is_weight_qbytes_tensor(t: object) -> bool:
    """Return whether ``t`` is an optimum-quanto WeightQBytesTensor."""
    return QUANTO_AVAILABLE and isinstance(t, WeightQBytesTensor)


def require_qbytes_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated quanto tensor, or raise."""
    if not is_weight_qbytes_tensor(t):
        raise TypeError(f"expected optimum-quanto WeightQBytesTensor, got {type(t).__name__}")
    validate_layout(t)
    return t


def qbytes_activation_qtype(t: Any) -> object | None:  # noqa: ANN401
    """Optional activation quant type stored by some quanto versions."""
    return getattr(t, "activation_qtype", None)


def is_marlin_f8_qbytes_tensor(t: object) -> bool:
    """Return whether ``t`` uses optimum-quanto's optimized FP8 packing."""
    return MarlinF8QBytesTensor is not None and isinstance(
        t,
        MarlinF8QBytesTensor,
    )


def canonicalize_qbytes_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return an unoptimized ``WeightQBytesTensor`` with natural storage.

    Optimized Quanto subclasses may wrap a device-specific packed tensor and
    reorder their scales. Their ``weight_qbytes_tensor`` conversion is the
    serialization contract for recovering raw qbytes and natural scale order.
    """
    qbytes = require_qbytes_tensor(t)
    if type(qbytes) is WeightQBytesTensor:
        return qbytes
    if not is_marlin_f8_qbytes_tensor(qbytes):
        raise RuntimeError(
            "piper-offload does not support this optimized optimum-quanto "
            f"WeightQBytesTensor subclass: {type(qbytes).__name__}. Convert it "
            "to WeightQBytesTensor before offloading."
        )
    canonical = qbytes.weight_qbytes_tensor()
    if type(canonical) is not WeightQBytesTensor:
        raise RuntimeError(
            "optimum-quanto did not convert MarlinF8QBytesTensor to the "
            "expected unoptimized WeightQBytesTensor representation."
        )
    validate_layout(canonical)
    return canonical


def create_qbytes_tensor(
    qtype: object,
    axis: int | None,
    size: torch.Size,
    stride: tuple[int, ...],
    data: torch.Tensor,
    scale: torch.Tensor,
    activation_qtype: object | None,
) -> torch.Tensor:
    """Create the canonical, kernel-agnostic Quanto representation."""
    if not QUANTO_AVAILABLE:
        raise RuntimeError("optimum-quanto is required to create a WeightQBytesTensor")
    return WeightQBytesTensor(
        qtype,
        axis,
        size,
        stride,
        data,
        scale,
        activation_qtype,
    )


def qbytes_data_storage(t: torch.Tensor) -> torch.Tensor:
    """Return the physical data leaf used for identity and deduplication."""
    qbytes = require_qbytes_tensor(t)
    if is_marlin_f8_qbytes_tensor(qbytes):
        return qbytes._data._data
    return qbytes._data


def canonical_qbytes_storage_layout(
    t: torch.Tensor,
) -> tuple[
    tuple[int, ...],
    torch.dtype,
    tuple[int, ...],
    torch.dtype,
]:
    """Return raw data/scale layout after optimized-subclass conversion."""
    qbytes = require_qbytes_tensor(t)
    if is_marlin_f8_qbytes_tensor(qbytes):
        rows, cols = qbytes.size()
        return (
            (rows, cols),
            qbytes.qtype.dtype,
            (rows, 1),
            qbytes._scale.dtype,
        )
    return (
        tuple(qbytes._data.shape),
        qbytes._data.dtype,
        tuple(qbytes._scale.shape),
        qbytes._scale.dtype,
    )


def copy_qbytes_tensor_(src: torch.Tensor, target: torch.Tensor) -> None:
    """Copy canonical qbytes into a base or Marlin target in place."""
    source = canonicalize_qbytes_tensor(src)
    target_qbytes = require_qbytes_tensor(target)
    if (
        source.qtype is not target_qbytes.qtype
        or source.axis != target_qbytes.axis
        or tuple(source.size()) != tuple(target_qbytes.size())
        or source.dtype is not target_qbytes.dtype
    ):
        raise ValueError(
            "Cannot copy between incompatible optimum-quanto qbytes representations."
        )

    if is_marlin_f8_qbytes_tensor(target_qbytes):
        packed = MarlinF8QBytesTensor(
            target_qbytes.qtype,
            target_qbytes.axis,
            target_qbytes.size(),
            target_qbytes.stride(),
            source._data,
            source._scale,
        )
        target_qbytes._data._data.copy_(packed._data._data)
        target_qbytes._scale.copy_(packed._scale)
        return
    if type(target_qbytes) is not WeightQBytesTensor:
        raise RuntimeError(
            "piper-offload cannot update optimized optimum-quanto subclass "
            f"{type(target_qbytes).__name__} in place."
        )
    target_qbytes._data.copy_(source._data)
    target_qbytes._scale.copy_(source._scale)


def validate_layout(qt: torch.Tensor) -> None:
    """Raise if ``qt`` is missing any of :data:`LAYOUT_ATTRS`.

    The streaming adapter (:meth:`QuantoAdapter.matches`) calls this so
    a layout drift in optimum-quanto is reported uniformly rather than
    as a generic ``AttributeError`` partway through the dispatch. The
    check itself is four ``hasattr`` calls — cheap to run on every
    dispatch, no caching.
    """
    missing = [a for a in LAYOUT_ATTRS if not hasattr(qt, a)]
    if not missing:
        return
    raise RuntimeError(
        f"WeightQBytesTensor is missing expected attributes {missing!r}; "
        f"this repo is pinned to a layout that exposes {LAYOUT_ATTRS}. "
        "optimum-quanto likely refactored the wrapper class — upgrade "
        "piper-offload to match."
    )


def dequantize_qbytes_tensor(qt: torch.Tensor) -> torch.Tensor:
    """Return the dense logical value in the wrapper's compute dtype.

    Quanto's FP8 operator emits NaN qbytes for an exact-zero block because it
    evaluates ``0 / 0``. A zero scale nevertheless represents an exact-zero
    block, so repair those logical values after dequantization. This also lets
    a later nonzero LoRA update recover a weight originally quantized from
    zeros instead of propagating the encoding NaNs.
    """
    canonical = canonicalize_qbytes_tensor(qt)
    dense = canonical.dequantize()
    scale = _broadcast_scale(
        canonical._scale,
        axis=canonical.axis,
        ndim=dense.ndim,
    )
    return dense.masked_fill_(scale == 0, 0)


def requantize_qbytes_tensor(
    t: torch.Tensor,
    *,
    like: torch.Tensor,
    rounding_seed: int | None = None,
) -> torch.Tensor:
    """Encode dense ``t`` using ``like``'s Quanto quantization recipe.

    Quanto's 8-bit weight scale is an absmax statistic of the represented
    values, not fixed calibration metadata. Recompute it for ``t`` while
    preserving the reference qtype, axis, scale dtype/shape, stride, and
    activation qtype. Exact-zero blocks receive a small positive scale so
    their zero qbytes dequantize finitely instead of taking a ``0 / 0`` path.
    When ``rounding_seed`` is supplied, finalize that ordinary representation first
    and then stochastically replace only its terminal qbytes.
    """
    qbytes = canonicalize_qbytes_tensor(like)
    if tuple(t.shape) != tuple(qbytes.size()):
        raise ValueError(
            f"Cannot requantize tensor with shape {tuple(t.shape)} like "
            f"WeightQBytesTensor with shape {tuple(qbytes.size())}."
        )
    scale = _absmax_scale(t, reference=qbytes)
    out = create_qbytes_tensor(
        qbytes.qtype,
        qbytes.axis,
        qbytes.size(),
        qbytes.stride(),
        _quantize_to_qbytes(t, qbytes, scale),
        scale,
        qbytes_activation_qtype(qbytes),
    )
    if rounding_seed is not None:
        _stochastic_recode_qbytes_(t, target=out, rounding_seed=rounding_seed)
    return out


def _stochastic_recode_qbytes_(
    t: torch.Tensor,
    *,
    target: torch.Tensor,
    rounding_seed: int,
) -> None:
    """Replace only finalized Quanto terminal codes in place."""
    qbytes = require_qbytes_tensor(target)
    scale = qbytes._scale
    broadcast_scale = _broadcast_scale(
        scale,
        axis=qbytes.axis,
        ndim=t.dim(),
    )
    valid_scale = torch.isfinite(broadcast_scale) & (broadcast_scale > 0)
    scaled = torch.where(
        valid_scale,
        t.to(torch.float32) / broadcast_scale.to(torch.float32),
        torch.zeros_like(t, dtype=torch.float32),
    )
    storage_dtype = qbytes._data.dtype
    if storage_dtype.is_floating_point:
        qdata = _stochastic_cast_float8(
            scaled,
            storage_dtype,
            seed=rounding_seed,
            deterministic=qbytes._data,
        )
    else:
        limits = torch.iinfo(storage_dtype)
        qdata = _stochastic_round_to_int(
            scaled,
            seed=rounding_seed,
            quant_min=limits.min,
            quant_max=limits.max,
            deterministic=qbytes._data,
        ).to(storage_dtype)
    qbytes._data.copy_(torch.where(valid_scale, qdata, qbytes._data))


def _validate_absmax_reference(
    float_data: torch.Tensor,
    *,
    reference: Any,  # noqa: ANN401
) -> tuple[int | None, int | float]:
    """Validate metadata needed to recompute or retain an absmax scale."""
    if getattr(reference.qtype, "bits", None) != 8:
        raise ValueError("Quanto qbytes requantization requires an 8-bit qtype.")
    if getattr(reference.qtype, "dtype", None) is not reference._data.dtype:
        raise ValueError(
            "Quanto qbytes storage dtype does not match its qtype metadata."
        )
    if not reference._scale.dtype.is_floating_point:
        raise ValueError("Quanto qbytes requantization requires floating-point scales.")

    axis = reference.axis
    if axis is None:
        valid_scale_layout = reference._scale.numel() == 1
        expected_scale = "one scale value"
    elif axis == 0 and float_data.ndim > 0:
        expected_shape = (float_data.shape[0], *([1] * (float_data.ndim - 1)))
        valid_scale_layout = tuple(reference._scale.shape) == expected_shape
        expected_scale = f"shape {expected_shape}"
    elif float_data.ndim > 0 and axis in (-1, float_data.ndim - 1):
        expected_shape = (*([1] * (float_data.ndim - 1)), float_data.shape[-1])
        valid_scale_layout = tuple(reference._scale.shape) == expected_shape
        expected_scale = f"shape {expected_shape}"
    else:
        raise ValueError(
            "Quanto qbytes requantization supports per-tensor, first-axis, "
            f"or last-axis scales, got axis {axis}."
        )
    if not valid_scale_layout:
        raise ValueError(
            "Quanto qbytes scale storage does not match its quantization axis: "
            f"expected {expected_scale}, stored {tuple(reference._scale.shape)}."
        )

    qmax = getattr(reference.qtype, "qmax", None)
    if not isinstance(qmax, (int, float)) or qmax <= 0:
        raise ValueError(
            "Quanto qbytes requantization requires a qtype with a positive qmax."
        )
    return axis, qmax


def _absmax_scale(
    float_data: torch.Tensor,
    *,
    reference: Any,  # noqa: ANN401
) -> torch.Tensor:
    """Return Quanto's absmax scale in ``reference``'s storage layout."""
    axis, qmax = _validate_absmax_reference(float_data, reference=reference)

    if float_data.numel() == 0:
        # An empty logical shard has no quantization blocks from which to
        # derive a new statistic. Retain its existing scale layout and values,
        # including exact zeros, while placing the result beside the new data.
        return reference._scale.to(device=float_data.device).clone()

    if axis is None:
        amax = float_data.abs().amax()
    elif axis == 0:
        amax = float_data.abs().amax(
            dim=tuple(range(1, float_data.ndim)),
            keepdim=True,
        )
    else:
        amax = float_data.abs().amax(
            dim=tuple(range(float_data.ndim - 1)),
            keepdim=True,
        )
    scale = (amax / qmax).to(dtype=reference._scale.dtype)
    if scale.numel() != reference._scale.numel():
        raise ValueError(
            "Quanto qbytes scale storage does not match its quantization axis: "
            f"computed {tuple(scale.shape)}, stored {tuple(reference._scale.shape)}."
        )
    scale = scale.reshape(reference._scale.shape)
    zero = scale == 0
    eps = torch.finfo(torch.float32).eps
    return torch.where(zero, torch.full_like(scale, eps), scale)


def _quantize_to_qbytes(
    float_data: torch.Tensor,
    reference: Any,  # noqa: ANN401
    scale: torch.Tensor,
) -> torch.Tensor:
    """Quantize float data with a scale shaped like ``reference._scale``."""
    scaled = float_data / _broadcast_scale(
        scale,
        axis=reference.axis,
        ndim=float_data.dim(),
    )
    storage_dtype = reference._data.dtype
    if storage_dtype.is_floating_point:
        limits = torch.finfo(storage_dtype)
    else:
        scaled = scaled.round()
        limits = torch.iinfo(storage_dtype)
    return scaled.clamp(limits.min, limits.max).to(storage_dtype)


def _broadcast_scale(
    scale: torch.Tensor,
    *,
    axis: int | None,
    ndim: int,
) -> torch.Tensor:
    """Shape a Quanto scale for elementwise dense-value operations."""
    if axis == 0:
        return scale.view(-1, *([1] * (ndim - 1)))
    return scale
