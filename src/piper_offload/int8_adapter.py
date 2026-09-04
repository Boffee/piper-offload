"""TorchAO ``Int8Tensor`` adapter (int8 weight-only / int8 dynamic-act).

TorchAO's INT8 workflow stores weights as a tensor subclass with int8
bytes (``qdata``), per-row/per-tensor ``scale``, an optional
``zero_point`` (asymmetric quant), optional static-activation tensors
(``act_quant_scale`` / ``act_quant_zero_point`` / ``act_pre_scale``), and
metadata controlling the matmul dispatch (``block_size``, ``dtype``,
``act_quant_kwargs``, ``reduce_range``). The shared
:class:`~piper_offload.torchao_structured_adapter.TorchaoStructuredAdapter`
base preserves that representation across pinned CPU and GPU storage;
this module supplies the INT8-specific hooks. One adapter covers both
``Int8WeightOnlyConfig`` and ``Int8DynamicActivationInt8WeightConfig``
because TorchAO models them as the same ``Int8Tensor`` parameterized by
``act_quant_kwargs``.

Beyond inference movement, this adapter opts into dequantize/requantize,
``copy_into``, a format-specific CUDA LoRA merge, and full-rank dense merge.
Supported per-tensor, per-row, and per-group layouts use raw Triton kernels when
available; other layouts and environments use the exact generic
dequantize/GEMM/requantize path. Both recompute the per-block weight
scale and preserve the existing wrapper and storage tensors. Like any
merge into a quantized base it is lossy; choosing merge vs routed
(non-destructive) LoRA is the caller's tradeoff.

It does not opt into CPU round-trip or trainable ``Parameter.data`` swap: the
quant state lives in the wrapper object, not its bytes, so int8 weights stay
frozen for streaming/training. Routed LoRA remains the non-destructive
alternative.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ._dense_merge import merge_dense_requantize_
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
    from ._triton_int8_lora import merge_int8_dense as _triton_merge_int8_dense
    from ._triton_int8_lora import merge_int8_lora as _triton_merge_int8_lora
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_merge_int8_dense = None
    _triton_merge_int8_lora = None


_TRITON_COMPUTE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _normalized_act_pre_scale(qt: Any) -> torch.Tensor | None:  # noqa: ANN401
    """Return ``act_pre_scale`` in scalar or per-input merge shape.

    TorchAO applies this tensor to the activation before its weight matmul.
    A permanent logical-weight update must therefore be divided by the same
    scale along the input dimension. Restrict accepted layouts to forms whose
    meaning does not depend on the activation rank: one scalar, or one value
    per input feature with singleton leading dimensions.
    """
    pre_scale = qt.act_pre_scale
    if pre_scale is None:
        return None
    if not pre_scale.dtype.is_floating_point:
        raise ValueError(
            "TorchAO INT8 act_pre_scale must be floating-point for additive "
            f"merge, got {pre_scale.dtype}."
        )
    if pre_scale.device != qt.qdata.device:
        raise ValueError(
            "TorchAO INT8 act_pre_scale must be on the weight device for additive "
            f"merge, got {pre_scale.device} and {qt.qdata.device}."
        )

    input_features = qt.shape[-1]
    if pre_scale.numel() == 1:
        normalized = pre_scale.reshape(1, 1)
    elif (
        pre_scale.ndim >= 1
        and pre_scale.shape[-1] == input_features
        and all(size == 1 for size in pre_scale.shape[:-1])
    ):
        normalized = pre_scale.reshape(1, input_features)
    else:
        raise ValueError(
            "TorchAO INT8 act_pre_scale must be a scalar or one value per "
            "input feature with only singleton leading dimensions for additive "
            f"merge; got shape {tuple(pre_scale.shape)} for {input_features} "
            "input features."
        )

    if bool((normalized == 0).any()):
        raise ValueError(
            "TorchAO INT8 act_pre_scale must contain only non-zero values for "
            "additive merge."
        )
    return normalized


def _lora_a_in_stored_weight_coordinates(
    qt: Any,  # noqa: ANN401
    a: torch.Tensor,
    strength: float,
) -> tuple[torch.Tensor, float]:
    """Map logical LoRA ``A`` through TorchAO's activation pre-scaling."""
    pre_scale = _normalized_act_pre_scale(qt)
    if pre_scale is None:
        return a, strength

    # Base execution is (x * p) @ W_stored.T, while routed LoRA is
    # strength * x @ (B @ A).T. Thus delta(W_stored) is
    # B @ ((strength * A) / p). Fold strength into A before division so both
    # backends receive unit strength. Numerical validity is caller-owned.
    stored_a = a.mul(strength).div(
        pre_scale.to(device=a.device, dtype=a.dtype)
    )
    return stored_a.contiguous(), 1.0


def _prepare_lora_merge(
    target: torch.Tensor,
    a: torch.Tensor,
    strength: float,
) -> tuple[Any, torch.Tensor, float]:
    """Validate and express one update in stored-weight coordinates."""
    qt = require_int8_tensor(target)
    stored_a, stored_strength = _lora_a_in_stored_weight_coordinates(
        qt,
        a,
        strength,
    )
    return qt, stored_a, stored_strength


def _prepare_dense_merge(
    target: torch.Tensor,
    update: torch.Tensor,
    strength: float,
) -> tuple[Any, torch.Tensor, float]:
    """Validate and express one logical dense update in stored coordinates."""
    qt = require_int8_tensor(target)
    pre_scale = _normalized_act_pre_scale(qt)
    if pre_scale is None:
        return qt, update, strength

    stored_update = update.mul(strength).div(
        pre_scale.to(device=update.device, dtype=update.dtype)
    )
    return qt, stored_update, 1.0


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
        or b.dtype is not qt.dtype
        or b.dtype not in _TRITON_COMPUTE_DTYPES
        or qt.scale.dtype not in _TRITON_COMPUTE_DTYPES
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


def _triton_int8_dense_layout_supported(
    qt: Any,  # noqa: ANN401
    update: torch.Tensor,
) -> bool:
    """Return whether a dense update fits the Triton affine-INT8 pipeline."""
    if (
        _triton_merge_int8_dense is None
        or qt.qdata.device.type != "cuda"
        or qt.qdata.dtype is not torch.int8
        or qt.qdata.ndim != 2
        or update.ndim != 2
        or update.dtype is not qt.dtype
        or update.dtype not in _TRITON_COMPUTE_DTYPES
        or qt.scale.dtype not in _TRITON_COMPUTE_DTYPES
        or qt.qdata.device != qt.scale.device
        or qt.qdata.device != update.device
    ):
        return False

    rows, cols = qt.qdata.shape
    if rows == 0 or cols == 0 or tuple(update.shape) != (rows, cols):
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
    reduce_range: bool | None


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
            reduce_range=t.reduce_range,
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
            meta.reduce_range,
        )

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return (
            tuple(t.block_size),
            t.dtype,
            metadata_key(t.act_quant_kwargs),
            t.reduce_range,
        )

    @classmethod
    def _layout_metadata(cls, t: Any) -> tuple[object, ...]:  # noqa: ANN401
        # Drop the logical dtype, which layout_signature's standard dtype
        # slot already carries (mirrors Float8Adapter).
        return (
            tuple(t.block_size),
            metadata_key(t.act_quant_kwargs),
            t.reduce_range,
        )

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return t.dtype

    # --- capabilities beyond inference movement ---------------------------

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        return dequantize_int8_tensor(t)

    @staticmethod
    def requantize(
        t: torch.Tensor,
        *,
        like: torch.Tensor,
        rounding_seed: int | None = None,
    ) -> torch.Tensor:
        return requantize_int8_tensor(t, like=like, rounding_seed=rounding_seed)

    @staticmethod
    def stage_lora_factors(
        target: torch.Tensor,
        factors: Sequence[tuple[float, torch.Tensor, torch.Tensor]],
        *,
        logical_shape: tuple[int, ...],
        compute_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, float] | None:
        """Transform validated logical ``A`` factors before packing."""
        qt = require_int8_tensor(target)
        pre_scale = _normalized_act_pre_scale(qt)
        if pre_scale is None:
            # Keep the ordinary zero-overhead staging path for INT8 tensors
            # whose stored and logical weight coordinates are identical.
            return None

        total_rank = sum(a.shape[0] for _strength, a, _b in factors)
        a_packed = torch.empty(
            (total_rank, logical_shape[1]),
            device=qt.device,
            dtype=compute_dtype,
        )
        b_packed = torch.empty(
            (logical_shape[0], total_rank),
            device=qt.device,
            dtype=compute_dtype,
        )
        stored_pre_scale = pre_scale.to(device=qt.device, dtype=compute_dtype)

        rank_offset = 0
        for strength, a, b in factors:
            next_offset = rank_offset + a.shape[0]
            stored_a = a_packed[rank_offset:next_offset]
            stored_a.copy_(a, non_blocking=True)
            stored_a.mul_(strength).div_(stored_pre_scale)
            b_packed[:, rank_offset:next_offset].copy_(
                b,
                non_blocking=True,
            )
            rank_offset = next_offset

        return b_packed, a_packed, 1.0

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge a staged LoRA update while preserving target storage."""
        qt, a, strength = _prepare_lora_merge(
            target,
            a,
            strength,
        )
        Int8Adapter._merge_stored_lora_(
            qt,
            b,
            a,
            strength,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def validate_dense_merge(
        target: torch.Tensor,
        _update: torch.Tensor,
        _strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        del rounding_seed
        _normalized_act_pre_scale(require_int8_tensor(target))

    @staticmethod
    def merge_dense_(
        target: torch.Tensor,
        update: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge a logical full-rank update into stored-weight coordinates."""
        qt, stored_update, stored_strength = _prepare_dense_merge(
            target,
            update,
            strength,
        )
        if _triton_int8_dense_layout_supported(qt, stored_update):
            assert _triton_merge_int8_dense is not None
            asymmetric = qt.zero_point is not None and bool(qt.zero_point.any())
            qdata, scale, zero_point = _triton_merge_int8_dense(
                qt.qdata,
                qt.scale,
                qt.zero_point,
                tuple(qt.block_size),
                stored_update,
                stored_strength,
                asymmetric=asymmetric,
                reduce_range=bool(qt.reduce_range),
                rounding_seed=rounding_seed,
            )
            qt.qdata.copy_(qdata)
            qt.scale.copy_(scale)
            if qt.zero_point is not None:
                qt.zero_point.copy_(zero_point)
            return
        merge_dense_requantize_(
            Int8Adapter,
            qt,
            stored_update,
            stored_strength,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def _merge_stored_lora_(
        qt: Any,  # noqa: ANN401
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge factors already expressed in stored-weight coordinates."""
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
                reduce_range=bool(qt.reduce_range),
                rounding_seed=rounding_seed,
            )
            qt.qdata.copy_(qdata)
            qt.scale.copy_(scale)
            if qt.zero_point is not None:
                qt.zero_point.copy_(zero_point)
            return

        dense = dequantize_int8_tensor(qt)
        dense.addmm_(b, a, alpha=strength)
        new_data = Int8Adapter.requantize(
            dense,
            like=qt,
            rounding_seed=rounding_seed,
        )
        Int8Adapter.copy_into(new_data, target=qt)

    @staticmethod
    def validate_lora_merge(
        target: torch.Tensor,
        _b: torch.Tensor,
        _a: torch.Tensor,
        _strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Validate activation pre-scale structure."""
        del rounding_seed
        _normalized_act_pre_scale(require_int8_tensor(target))

    @staticmethod
    def validate_prepared_lora_merge(
        target: torch.Tensor,
        _b: torch.Tensor,
        _a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Validate factor-aware staging without applying pre-scale twice."""
        del rounding_seed
        qt = require_int8_tensor(target)
        _normalized_act_pre_scale(qt)
        if strength != 1.0:
            raise ValueError(f"TorchAO INT8 prepared LoRA merge requires unit strength, got {strength}.")

    @staticmethod
    def merge_prepared_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge validated factor-aware data without pre-scaling twice."""
        qt = require_int8_tensor(target)
        Int8Adapter._merge_stored_lora_(
            qt,
            b,
            a,
            strength,
            rounding_seed=rounding_seed,
        )

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
