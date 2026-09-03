"""TorchAO ``MXTensor`` adapter (OCP microscaling MXFP8 / MXFP4).

TorchAO's MX workflow stores weights as a tensor subclass with packed
element bytes (``qdata`` — ``float8_e4m3fn``/``float8_e5m2`` for MXFP8,
packed ``uint8`` for MXFP4), E8M0 power-of-two block scales (``scale``),
and metadata controlling the matmul dispatch (``elem_dtype``,
``block_size``, ``kernel_preference``, ``act_quant_kwargs``,
``is_swizzled_scales``). The shared
:class:`~piper_offload.torchao_structured_adapter.TorchaoStructuredAdapter`
base preserves that representation across pinned CPU and GPU storage;
this module supplies the MX-specific hooks. One adapter covers both MXFP8
and MXFP4 because TorchAO models them as the same ``MXTensor`` subclass
parameterized by ``elem_dtype``.

Beyond inference movement, this adapter opts into a format-specific CUDA
LoRA merge plus dequantize/requantize and ``copy_into``. Contiguous MXFP8
and MXFP4 weights use one raw Triton kernel that updates each 32-element
block directly, including its E8M0 scale, without a full dense weight
temporary. Unsupported layouts or environments without Triton retain the
public ``MXTensor.to_mx`` fallback. Like any merge into a quantized base it
is lossy; MXFP4's 4-bit grid makes a permanent merge far coarser than
MXFP8, so routed LoRA remains the non-destructive alternative.

It does not opt into CPU round-trip or trainable ``Parameter.data`` swap:
the quant state lives in the wrapper object, not its bytes, so MX weights
stay frozen for streaming/training. Routed LoRA remains the
non-destructive alternative when the owning module is a logical
``nn.Linear`` with compatible shape/dtype.
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch

from ._dense_merge import merge_dense_requantize_
from ._torchao_mx import (
    create_mx_tensor,
    dequantize_mx_tensor,
    is_mx_tensor,
    requantize_mx_tensor,
    require_mx_tensor,
    validate_layout,
)
from .tensor_adapters import metadata_key
from .torchao_structured_adapter import TorchaoStructuredAdapter, copy_storage_into

try:
    from ._triton_mx_lora import (
        merge_mx_dense_ as _triton_merge_mx_dense_,
    )
    from ._triton_mx_lora import (
        merge_mx_lora_ as _triton_merge_mx_lora_,
    )
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_merge_mx_dense_ = None
    _triton_merge_mx_lora_ = None


_MX_SCALING_MODES = {
    "floor": 0,
    "rceil": 1,
    "ceil": 2,
    "even": 3,
}
_TRITON_COMPUTE_DTYPES = (torch.bfloat16, torch.float32)
_FP4_ELEM_DTYPE = getattr(torch, "float4_e2m1fn_x2", None)
_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)


def _scaling_mode_id(mx: Any) -> int | None:  # noqa: ANN401
    mode = getattr(mx.act_quant_kwargs, "scaling_mode", None)
    if mode is None:
        return _MX_SCALING_MODES["floor"]
    value = getattr(mode, "value", None)
    if value is None:
        return None
    return _MX_SCALING_MODES.get(value)


def _expected_scale_shape(
    rows: int,
    cols: int,
    *,
    swizzled: bool,
) -> tuple[int, int]:
    if not swizzled:
        return rows, cols // 32
    return (
        ((rows + 127) // 128) * 32,
        ((cols + 127) // 128) * 16,
    )


def _can_use_triton_merge(
    mx: Any,  # noqa: ANN401
    b: torch.Tensor,
    a: torch.Tensor,
) -> bool:
    if (
        _triton_merge_mx_lora_ is None
        or mx.qdata.device.type != "cuda"
        or mx.qdata.ndim != 2
        or mx.scale.ndim != 2
        or not mx.qdata.is_contiguous()
        or not mx.scale.is_contiguous()
        or mx.scale.dtype is not _E8M0_DTYPE
        or mx.block_size != 32
        or mx.orig_dtype not in _TRITON_COMPUTE_DTYPES
        or b.ndim != 2
        or a.ndim != 2
        or b.dtype is not a.dtype
        or b.dtype is not mx.orig_dtype
        or mx.qdata.device != mx.scale.device
        or mx.qdata.device != b.device
        or mx.qdata.device != a.device
        or _scaling_mode_id(mx) is None
    ):
        return False

    rows, cols = tuple(mx.shape)
    rank = a.shape[0]
    if rows == 0 or cols == 0 or rank == 0 or cols % 32 != 0 or b.shape != (rows, rank) or a.shape[1] != cols:
        return False

    if mx.elem_dtype is _FP4_ELEM_DTYPE and _FP4_ELEM_DTYPE is not None:
        expected_qdata_shape = (rows, cols // 2)
        if mx.qdata.dtype is not torch.uint8:
            return False
    elif mx.elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        expected_qdata_shape = (rows, cols)
        if mx.qdata.dtype is not mx.elem_dtype:
            return False
    else:
        return False

    return tuple(mx.qdata.shape) == expected_qdata_shape and tuple(mx.scale.shape) == _expected_scale_shape(
        rows,
        cols,
        swizzled=mx.is_swizzled_scales,
    )


def _can_use_triton_dense_merge(
    mx: Any,  # noqa: ANN401
    update: torch.Tensor,
) -> bool:
    if (
        _triton_merge_mx_dense_ is None
        or mx.qdata.device.type != "cuda"
        or mx.qdata.ndim != 2
        or mx.scale.ndim != 2
        or not mx.qdata.is_contiguous()
        or not mx.scale.is_contiguous()
        or mx.scale.dtype is not _E8M0_DTYPE
        or mx.block_size != 32
        or mx.orig_dtype not in _TRITON_COMPUTE_DTYPES
        or update.ndim != 2
        or update.dtype is not mx.orig_dtype
        or mx.qdata.device != mx.scale.device
        or mx.qdata.device != update.device
        or _scaling_mode_id(mx) is None
    ):
        return False

    rows, cols = tuple(mx.shape)
    if rows == 0 or cols == 0 or cols % 32 != 0 or tuple(update.shape) != (rows, cols):
        return False

    if mx.elem_dtype is _FP4_ELEM_DTYPE and _FP4_ELEM_DTYPE is not None:
        expected_qdata_shape = (rows, cols // 2)
        if mx.qdata.dtype is not torch.uint8:
            return False
    elif mx.elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        expected_qdata_shape = (rows, cols)
        if mx.qdata.dtype is not mx.elem_dtype:
            return False
    else:
        return False

    return tuple(mx.qdata.shape) == expected_qdata_shape and tuple(mx.scale.shape) == _expected_scale_shape(
        rows,
        cols,
        swizzled=mx.is_swizzled_scales,
    )


def _torch_merge_mx_lora_(
    target: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    rounding_seed: int | None = None,
) -> None:
    """Apply the established wrapper-based MX merge in place."""
    dense = dequantize_mx_tensor(target)
    dense.addmm_(b, a, alpha=strength)
    requantized = requantize_mx_tensor(
        dense,
        like=target,
        rounding_seed=rounding_seed,
    )
    copy_storage_into(
        MxAdapter._storage_of(require_mx_tensor(requantized)),
        MxAdapter._storage_of(require_mx_tensor(target)),
        non_blocking=False,
    )


@dataclass(slots=True, frozen=True)
class _MxMeta:
    """Reconstruction metadata snapshot for a TorchAO MX tensor."""

    elem_dtype: object
    block_size: int
    orig_dtype: torch.dtype
    kernel_preference: object
    act_quant_kwargs: object | None
    is_swizzled_scales: bool


class MxAdapter(TorchaoStructuredAdapter[_MxMeta]):
    """Adapter for TorchAO ``MXTensor`` (MXFP8 / MXFP4) weights."""

    _TAG = "torchao-mx"
    _STORAGE_NAMES = ("qdata", "scale")

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_mx_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_mx_tensor(t)

    @staticmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        return (t.qdata, t.scale)

    @staticmethod
    def _meta_of(t: Any) -> _MxMeta:  # noqa: ANN401
        return _MxMeta(
            elem_dtype=t.elem_dtype,
            block_size=t.block_size,
            orig_dtype=t.orig_dtype,
            kernel_preference=t.kernel_preference,
            act_quant_kwargs=t.act_quant_kwargs,
            is_swizzled_scales=t.is_swizzled_scales,
        )

    @staticmethod
    def _reconstruct(storage: tuple[torch.Tensor | None, ...], meta: _MxMeta) -> torch.Tensor:
        qdata, scale = storage
        assert qdata is not None
        assert scale is not None
        return create_mx_tensor(
            qdata,
            scale,
            meta.elem_dtype,
            meta.block_size,
            meta.orig_dtype,
            meta.kernel_preference,
            meta.act_quant_kwargs,
            meta.is_swizzled_scales,
        )

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return (
            t.elem_dtype,
            t.block_size,
            t.orig_dtype,
            t.is_swizzled_scales,
            metadata_key(t.kernel_preference),
            metadata_key(t.act_quant_kwargs),
        )

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return t.orig_dtype

    # --- capabilities beyond inference movement ---------------------------

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        return dequantize_mx_tensor(t)

    @staticmethod
    def requantize(
        t: torch.Tensor,
        *,
        like: torch.Tensor,
        rounding_seed: int | None = None,
    ) -> torch.Tensor:
        return requantize_mx_tensor(t, like=like, rounding_seed=rounding_seed)

    @staticmethod
    def _validate_merge_target(
        target: torch.Tensor,
        *,
        kind: Literal["lora", "dense"],
    ) -> None:
        """Reject layouts the standard MX re-encode cannot refill."""
        label = "LoRA" if kind == "lora" else "a dense update"
        mx = require_mx_tensor(target)
        if not mx.qdata.is_contiguous():
            guidance = " Use routed LoRA for this weight." if kind == "lora" else ""
            raise ValueError(
                f"Cannot merge {label} into a non-contiguous (e.g. transposed) "
                "MX weight: requantization produces the standard packed layout, "
                f"which cannot fill a transposed target.{guidance}"
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
        MxAdapter._validate_merge_target(target, kind="lora")

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge a validated staged update while preserving target storage."""
        mx = require_mx_tensor(target)
        if _can_use_triton_merge(mx, b, a):
            assert _triton_merge_mx_lora_ is not None
            scaling_mode = _scaling_mode_id(mx)
            assert scaling_mode is not None
            _triton_merge_mx_lora_(
                mx.qdata,
                mx.scale,
                mx.elem_dtype,
                mx.block_size,
                mx.orig_dtype,
                b,
                a,
                strength,
                scaling_mode=scaling_mode,
                swizzled=mx.is_swizzled_scales,
                rounding_seed=rounding_seed,
            )
            return
        _torch_merge_mx_lora_(
            target,
            b,
            a,
            strength,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def validate_dense_merge_target(
        target: torch.Tensor,
        *,
        rounding_seed: int | None = None,
    ) -> bool:
        del rounding_seed
        MxAdapter._validate_merge_target(target, kind="dense")
        return False

    @staticmethod
    def merge_dense_(
        target: torch.Tensor,
        update: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge a full-rank update, preferring raw Triton storage."""
        mx = require_mx_tensor(target)
        if _can_use_triton_dense_merge(mx, update):
            assert _triton_merge_mx_dense_ is not None
            scaling_mode = _scaling_mode_id(mx)
            assert scaling_mode is not None
            _triton_merge_mx_dense_(
                mx.qdata,
                mx.scale,
                mx.elem_dtype,
                mx.block_size,
                mx.orig_dtype,
                update,
                strength,
                scaling_mode=scaling_mode,
                swizzled=mx.is_swizzled_scales,
                rounding_seed=rounding_seed,
            )
            return
        merge_dense_requantize_(
            MxAdapter,
            target,
            update,
            strength,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        # Copy the packed elements + E8M0 block scales into target's
        # existing buffers, preserving its wrapper/object identity. MX has
        # no optional storage, so both slots are always present.
        copy_storage_into(
            MxAdapter._storage_of(require_mx_tensor(src)),
            MxAdapter._storage_of(require_mx_tensor(target)),
            non_blocking=False,
        )
