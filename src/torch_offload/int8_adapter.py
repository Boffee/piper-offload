"""TorchAO ``Int8Tensor`` adapter (int8 weight-only / int8 dynamic-act).

TorchAO's INT8 workflow stores weights as a tensor subclass with int8
bytes (``qdata``), per-row/per-tensor ``scale``, an optional
``zero_point`` (asymmetric quant), optional static-activation tensors
(``act_quant_scale`` / ``act_quant_zero_point`` / ``act_pre_scale``), and
metadata controlling the matmul dispatch (``block_size``, ``dtype``,
``act_quant_kwargs``). The shared
:class:`~torch_offload.torchao_structured_adapter.TorchaoStructuredAdapter`
base preserves that representation across pinned CPU and GPU storage;
this module supplies the INT8-specific hooks. One adapter covers both
``Int8WeightOnlyConfig`` and ``Int8DynamicActivationInt8WeightConfig``
because TorchAO models them as the same ``Int8Tensor`` parameterized by
``act_quant_kwargs``.

Beyond inference movement, this adapter opts into dequantize/requantize,
``copy_into``, and a format-specific CUDA LoRA merge. Supported
per-tensor, per-row, and per-group layouts use raw Triton kernels when
available; other layouts and environments use the exact generic
dequantize/GEMM/requantize path. Both recompute the per-block weight
scale and preserve the existing wrapper and storage tensors. Like any
merge into a quantized base it is lossy; choosing merge vs routed
(non-destructive) LoRA is the caller's tradeoff.

It does not opt into CPU round-trip, trainable ``Parameter.data`` swap,
or activation-scoped dense ``addmm_`` merge: the quant state lives in the
wrapper object, not its bytes, so int8 weights stay frozen for
streaming/training. Routed LoRA remains the non-destructive alternative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ._torchao_int8 import (
    create_int8_tensor,
    dequantize_int8_tensor,
    is_int8_tensor,
    requantize_int8_tensor,
    require_int8_tensor,
    validate_layout,
)
from .tensor_adapters import metadata_key
from .torchao_structured_adapter import TorchaoStructuredAdapter, copy_storage_into

try:
    from ._triton_int8_lora import merge_int8_lora as _triton_merge_int8_lora
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_merge_int8_lora = None


_TRITON_COMPUTE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _triton_int8_layout_supported(
    qt: Any,  # noqa: ANN401
    b: torch.Tensor,
    a: torch.Tensor,
) -> bool:
    """Return whether the raw buffers fit the Triton affine-INT8 pipeline."""
    if (
        _triton_merge_int8_lora is None
        or qt.qdata.device.type != "cuda"
        or qt.qdata.dtype is not torch.int8
        or qt.qdata.ndim != 2
        or b.ndim != 2
        or a.ndim != 2
        or b.dtype is not a.dtype
        or b.dtype is not qt.scale.dtype
        or b.dtype not in _TRITON_COMPUTE_DTYPES
        or qt.qdata.device != qt.scale.device
        or qt.qdata.device != b.device
        or qt.qdata.device != a.device
    ):
        return False

    rows, cols = qt.qdata.shape
    rank = a.shape[0]
    if rows == 0 or cols == 0 or rank == 0 or b.shape != (rows, rank) or a.shape[1] != cols:
        return False

    block_size = tuple(qt.block_size)
    if block_size == (rows, cols):
        expected_qparam_shape = (1, 1)
    elif len(block_size) == 2 and block_size[0] == 1 and 0 < block_size[1] <= cols and cols % block_size[1] == 0:
        expected_qparam_shape = (rows, cols // block_size[1])
    else:
        return False

    if tuple(qt.scale.shape) != expected_qparam_shape:
        return False
    if qt.zero_point is None:
        return True
    return (
        qt.zero_point.dtype is torch.int8
        and qt.zero_point.device == qt.qdata.device
        and tuple(qt.zero_point.shape) == expected_qparam_shape
    )


@dataclass(slots=True, frozen=True)
class _Int8Meta:
    """Reconstruction metadata snapshot for a TorchAO Int8 tensor."""

    block_size: tuple[int, ...]
    dtype: torch.dtype  # logical (pre-quantization) dtype
    act_quant_kwargs: object | None


class Int8Adapter(TorchaoStructuredAdapter[_Int8Meta]):
    """Adapter for TorchAO ``Int8Tensor`` weights."""

    _TAG = "torchao-int8"
    _STORAGE_NAMES = (
        "qdata",
        "scale",
        "zero_point",
        "act_quant_scale",
        "act_quant_zero_point",
        "act_pre_scale",
    )

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_int8_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_int8_tensor(t)

    @staticmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        return (
            t.qdata,
            t.scale,
            t.zero_point,
            t.act_quant_scale,
            t.act_quant_zero_point,
            t.act_pre_scale,
        )

    @staticmethod
    def _meta_of(t: Any) -> _Int8Meta:  # noqa: ANN401
        return _Int8Meta(
            block_size=tuple(t.block_size),
            dtype=t.dtype,
            act_quant_kwargs=t.act_quant_kwargs,
        )

    @staticmethod
    def _reconstruct(storage: tuple[torch.Tensor | None, ...], meta: _Int8Meta) -> torch.Tensor:
        qdata, scale, zero_point, act_quant_scale, act_quant_zero_point, act_pre_scale = storage
        assert qdata is not None
        assert scale is not None
        return create_int8_tensor(
            qdata,
            scale,
            list(meta.block_size),
            meta.dtype,
            zero_point,
            act_quant_scale,
            act_quant_zero_point,
            act_pre_scale,
            meta.act_quant_kwargs,
        )

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return (
            tuple(t.block_size),
            t.dtype,
            metadata_key(t.act_quant_kwargs),
        )

    @classmethod
    def _layout_metadata(cls, t: Any) -> tuple[object, ...]:  # noqa: ANN401
        # Drop the logical dtype, which layout_signature's standard dtype
        # slot already carries (mirrors Float8Adapter).
        return (
            tuple(t.block_size),
            metadata_key(t.act_quant_kwargs),
        )

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return t.dtype

    # --- capabilities beyond inference movement ---------------------------

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        return dequantize_int8_tensor(t)

    @staticmethod
    def requantize(t: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        return requantize_int8_tensor(t, like=like)

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
    ) -> None:
        """Merge a staged LoRA update while preserving target storage."""
        qt = require_int8_tensor(target)
        if _triton_int8_layout_supported(qt, b, a):
            assert _triton_merge_int8_lora is not None
            asymmetric = qt.zero_point is not None and bool(qt.zero_point.any())
            qdata, scale, zero_point = _triton_merge_int8_lora(
                qt.qdata,
                qt.scale,
                qt.zero_point,
                tuple(qt.block_size),
                b,
                a,
                strength,
                asymmetric=asymmetric,
            )
            qt.qdata.copy_(qdata)
            qt.scale.copy_(scale)
            if qt.zero_point is not None:
                qt.zero_point.copy_(zero_point)
            return

        dense = dequantize_int8_tensor(target)
        dense.addmm_(b, a, alpha=strength)
        new_data = requantize_int8_tensor(dense, like=target)
        Int8Adapter.copy_into(new_data, target=target)

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        # Fill target's present storage slots (qdata/scale, optional
        # zero_point, and any activation-quant tensors) from the requantized
        # src. Driven by target presence: a symmetric base stored with
        # zero_point=None keeps it, even though from_hp always re-emits a
        # zeros zero_point that carries nothing the target lacks.
        copy_storage_into(
            Int8Adapter._storage_of(require_int8_tensor(src)),
            Int8Adapter._storage_of(require_int8_tensor(target)),
            non_blocking=False,
        )
