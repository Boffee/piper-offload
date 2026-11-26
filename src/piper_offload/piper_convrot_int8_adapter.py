"""Piper ``ConvRotInt8Tensor`` offload adapter.

``piper-kernels`` owns ConvRot's tensor semantics and execution backends.
Piper Offload owns the movement and merge integration: capture the public ``qdata``
and ``scale`` storage tensors, move those bytes, reconstruct the wrapper with
its ``group_size`` and logical ``dtype`` metadata, and delegate staged LoRA
and dense updates to the public in-place ``ConvRotInt8Tensor.addmm_`` and
``ConvRotInt8Tensor.add_`` operations, including both update and base scaling.
When stochastic rounding is requested, Piper Offload forwards its stable
per-target seed to the kernel-owned terminal INT8 code selection.

Permanent and activation-time LoRA merge
preserve the existing target wrapper and storage identities. Piper selects its
optimized Triton backend on supported CUDA devices and its portable reference
backend elsewhere.
"""

from dataclasses import dataclass
from typing import Any

import torch

from ._piper_convrot_int8 import (
    create_convrot_int8_tensor,
    is_convrot_int8_tensor,
    require_convrot_int8_matrix,
    require_convrot_int8_tensor,
    validate_layout,
)
from .structured_adapter import StructuredHost, _KernelMergeStructuredAdapter


@dataclass(slots=True, frozen=True)
class _PiperConvRotInt8Meta:
    """ConvRot metadata needed to reconstruct a storage wrapper."""

    group_size: int
    dtype: torch.dtype


class PiperConvRotInt8Adapter(_KernelMergeStructuredAdapter[_PiperConvRotInt8Meta]):
    """Adapter for ``piper_kernels.weights.convrot.int8.ConvRotInt8Tensor`` weights."""

    _TAG = "piper-kernels-convrot-int8"
    _STORAGE_NAMES = ("qdata", "scale", "act_per_tensor_scale")

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_convrot_int8_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_convrot_int8_tensor(t)

    @staticmethod
    def _require_matrix(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_convrot_int8_matrix(t)

    @staticmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        return (t.qdata, t.scale, t.act_per_tensor_scale)

    @staticmethod
    def _meta_of(t: Any) -> _PiperConvRotInt8Meta:  # noqa: ANN401
        return _PiperConvRotInt8Meta(
            group_size=t.group_size,
            dtype=t.dtype,
        )

    @staticmethod
    def _reconstruct(
        storage: tuple[torch.Tensor | None, ...],
        meta: _PiperConvRotInt8Meta,
    ) -> torch.Tensor:
        qdata, scale, act_per_tensor_scale = storage
        assert qdata is not None
        assert scale is not None
        return create_convrot_int8_tensor(
            qdata,
            scale,
            meta.group_size,
            meta.dtype,
            act_per_tensor_scale,
        )

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return (t.group_size, t.dtype)

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return t.dtype

    @classmethod
    def project_shard(
        cls,
        state: StructuredHost[_PiperConvRotInt8Meta],
        *,
        offsets: tuple[int, ...],
        local_shape: tuple[int, ...],
    ) -> tuple[StructuredHost[_PiperConvRotInt8Meta], tuple[object, ...]]:
        """Project a zero-copy host shard for 2-D ``ConvRotInt8Tensor`` weights."""
        qdata, scale, act_per_tensor_scale = state.storage
        assert qdata is not None
        assert scale is not None
        global_shape = state.logical_shape
        group_size = state.meta.group_size
        if len(global_shape) != 2 or qdata.ndim != 2:
            raise ValueError(
                "ConvRot INT8 DTensor shard projection supports 2-D matrices "
                f"only; got shape {global_shape}."
            )
        if len(offsets) != 2 or len(local_shape) != 2:
            raise ValueError(
                "ConvRot INT8 shard projection requires 2-D offsets and shape; "
                f"got offsets={offsets}, local_shape={local_shape}."
            )
        global_rows, global_cols = global_shape
        row_offset, col_offset = offsets
        local_rows, local_cols = local_shape
        if local_rows <= 0 or local_cols <= 0:
            raise ValueError(
                "ConvRot INT8 DTensor shard projection requires non-empty local "
                f"shards; got local_shape={local_shape}."
            )
        if (
            row_offset < 0
            or row_offset + local_rows > global_rows
            or col_offset < 0
            or col_offset + local_cols > global_cols
        ):
            raise ValueError(
                "ConvRot INT8 shard projection range exceeds weight shape: "
                f"global_shape={global_shape}, offsets={offsets}, local_shape={local_shape}."
            )
        if (
            not qdata.is_contiguous()
            or not scale.is_contiguous()
            or tuple(qdata.shape) != global_shape
            or tuple(scale.shape) != (global_rows, 1)
        ):
            raise ValueError(
                "ConvRot INT8 DTensor shard projection requires contiguous, "
                "untransposed qdata and scale storage; "
                f"got qdata={tuple(qdata.shape)}, scale={tuple(scale.shape)}."
            )
        if (col_offset != 0 or local_cols != global_cols) and (
            col_offset % group_size != 0 or local_cols % group_size != 0
        ):
            raise ValueError(
                "ConvRot INT8 Shard(1) projection along columns "
                f"(offset={col_offset}, size={local_cols}) must align to "
                f"rotation group_size {group_size}."
            )

        qdata_local = qdata
        scale_local = scale
        if row_offset != 0 or local_rows != global_rows:
            qdata_local = qdata_local.narrow(0, row_offset, local_rows)
            scale_local = scale_local.narrow(0, row_offset, local_rows)
        if col_offset != 0 or local_cols != global_cols:
            qdata_local = qdata_local.narrow(1, col_offset, local_cols)

        return cls._finalize_projected_shard(
            (qdata_local, scale_local, act_per_tensor_scale),
            state.meta,
            local_shape,
        )
