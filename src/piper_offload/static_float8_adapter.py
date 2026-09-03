"""TorchAO static-activation ``PrototypeFloat8Tensor`` adapter.

The prototype static FP8 representation is deliberately separate from
:class:`~piper_offload.float8_adapter.Float8Adapter`: TorchAO's regular
``Float8Tensor`` owns dynamic/weight-only metadata, while
``PrototypeFloat8Tensor`` additionally owns a checkpoint-calibrated
``act_quant_scale``.  This adapter keeps all three storage tensors
(``qdata``, weight ``scale``, and ``act_quant_scale``) quantized and intact
through pinning, block streaming, cache reuse, and lossless CPU round trips.

CUDA LoRA updates use a format-specific Triton merge when available, with a
pure-Torch dequantize/GEMM/requantize fallback. Both recompute only the weight
scale and install only weight bytes and weight scales in the existing target.
The calibrated activation scale is never replaced or recalibrated. Routed LoRA
uses the ordinary logical shape and compute dtype exposed by the shared adapter
contract.
"""

from dataclasses import dataclass
from typing import Any

import torch

from ._dense_merge import merge_dense_requantize_
from ._torchao_static_float8 import (
    create_static_float8_tensor,
    dequantize_static_float8_tensor,
    is_static_float8_tensor,
    requantize_static_float8_tensor,
    require_static_float8_tensor,
    validate_layout,
)
from .tensor_adapters import metadata_key
from .torchao_structured_adapter import (
    TorchaoGpu,
    TorchaoPinned,
    TorchaoStructuredAdapter,
    copy_storage,
)

try:
    from ._triton_static_float8_lora import (
        merge_static_float8_dense as _triton_merge_static_float8_dense,
    )
    from ._triton_static_float8_lora import (
        merge_static_float8_lora as _triton_merge_static_float8_lora,
    )
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    _triton_merge_static_float8_dense = None
    _triton_merge_static_float8_lora = None


def _torch_merge_static_float8_lora(
    target: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    rounding_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge static FP8 through its ordinary Torch reference path."""
    dense = dequantize_static_float8_tensor(target)
    dense.addmm_(b, a, alpha=strength)

    requantized = requantize_static_float8_tensor(
        dense,
        like=target,
        rounding_seed=rounding_seed,
    )
    source = require_static_float8_tensor(requantized)
    return source.qdata, source.scale


def _merge_static_float8_lora(
    target: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    rounding_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefer Triton, falling back to the wrapper-based Torch merge."""
    f8 = require_static_float8_tensor(target)
    if (
        _triton_merge_static_float8_lora is not None
        and f8.qdata.device.type == "cuda"
    ):
        return _triton_merge_static_float8_lora(
            f8.qdata,
            f8.scale,
            b,
            a,
            strength,
            rounding_seed=rounding_seed,
        )
    return _torch_merge_static_float8_lora(
        target,
        b,
        a,
        strength,
        rounding_seed=rounding_seed,
    )


@dataclass(slots=True, frozen=True)
class _StaticFloat8Meta:
    """Reconstruction metadata for a static PrototypeFloat8Tensor."""

    block_size: tuple[int, ...]
    mm_config: object | None
    act_quant_kwargs: object
    kernel_preference: object
    dtype: torch.dtype


class StaticFloat8Adapter(TorchaoStructuredAdapter[_StaticFloat8Meta]):
    """Adapter for calibrated TorchAO ``PrototypeFloat8Tensor`` weights."""

    _TAG = "torchao-static-float8"
    _STORAGE_NAMES = ("qdata", "scale", "act_quant_scale")

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_static_float8_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_static_float8_tensor(t)

    @staticmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        return (t.qdata, t.scale, t.act_quant_scale)

    @staticmethod
    def _meta_of(t: Any) -> _StaticFloat8Meta:  # noqa: ANN401
        return _StaticFloat8Meta(
            block_size=tuple(t.block_size),
            mm_config=t.mm_config,
            act_quant_kwargs=t.act_quant_kwargs,
            kernel_preference=t.kernel_preference,
            dtype=t.dtype,
        )

    @staticmethod
    def _reconstruct(
        storage: tuple[torch.Tensor | None, ...],
        meta: _StaticFloat8Meta,
    ) -> torch.Tensor:
        qdata, scale, act_quant_scale = storage
        assert qdata is not None
        assert scale is not None
        assert act_quant_scale is not None
        return create_static_float8_tensor(
            qdata,
            scale,
            act_quant_scale,
            list(meta.block_size),
            meta.mm_config,
            meta.act_quant_kwargs,
            meta.kernel_preference,
            meta.dtype,
        )

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return (
            tuple(t.block_size),
            t.dtype,
            metadata_key(t.mm_config),
            metadata_key(t.kernel_preference),
            metadata_key(t.act_quant_kwargs),
        )

    @classmethod
    def _layout_metadata(cls, t: Any) -> tuple[object, ...]:  # noqa: ANN401
        # The standard layout dtype slot already records the logical dtype.
        return (
            tuple(t.block_size),
            metadata_key(t.mm_config),
            metadata_key(t.kernel_preference),
            metadata_key(t.act_quant_kwargs),
        )

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return t.dtype

    # --- capabilities beyond inference movement -------------------------

    @staticmethod
    def copy_to_cpu(
        src: TorchaoGpu,
        dst: TorchaoPinned[_StaticFloat8Meta],
        *,
        non_blocking: bool = False,
    ) -> None:
        # All three GPU tensors use the identical representation on CPU.
        copy_storage(src.storage, dst.storage, non_blocking=non_blocking)

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        return dequantize_static_float8_tensor(t)

    @staticmethod
    def requantize(
        t: torch.Tensor,
        *,
        like: torch.Tensor,
        rounding_seed: int | None = None,
    ) -> torch.Tensor:
        return requantize_static_float8_tensor(
            t,
            like=like,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def merge_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Run and install the preferred static-FP8 LoRA merge."""
        f8 = require_static_float8_tensor(target)
        merged = _merge_static_float8_lora(
            target,
            b,
            a,
            strength,
            rounding_seed=rounding_seed,
        )
        qdata, scale = merged
        f8.qdata.copy_(qdata)
        f8.scale.copy_(scale)

    @staticmethod
    def merge_dense_(
        target: torch.Tensor,
        update: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Merge a full-rank update without changing activation calibration."""
        f8 = require_static_float8_tensor(target)
        if _triton_merge_static_float8_dense is not None and f8.qdata.device.type == "cuda":
            qdata, scale = _triton_merge_static_float8_dense(
                f8.qdata,
                f8.scale,
                update,
                strength,
                rounding_seed=rounding_seed,
            )
            f8.qdata.copy_(qdata)
            f8.scale.copy_(scale)
            return
        merge_dense_requantize_(
            StaticFloat8Adapter,
            target,
            update,
            strength,
            rounding_seed=rounding_seed,
        )

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        src_f8 = require_static_float8_tensor(src)
        target_f8 = require_static_float8_tensor(target)
        # A LoRA changes the weight range, not the checkpoint's calibrated
        # input range.  Intentionally leave target_f8.act_quant_scale alone.
        target_f8.qdata.copy_(src_f8.qdata)
        target_f8.scale.copy_(src_f8.scale)


__all__ = ["StaticFloat8Adapter"]
