"""Unified structured adapter for ``piper_kernels.weights.QuantizedWeight`` subclasses.

All Piper Kernels quantized weight subclasses implement the storage boundary
(``as_storage()`` / ``from_storage(tensors, metadata)``), so one stateless
adapter moves ``ConvRotInt8Tensor``, ``PiperNVFP4Tensor``, ``ConvRotNVFP4Tensor``,
``ConvRotTrellisTensor``, and future ``QuantizedWeight`` formats uniformly across
CPU and GPU without format-specific transport glue.

Per-format merge eligibility and DTensor shard projection are validated per
target weight (e.g., 2-D ConvRot INT8 and 2-D canonical NVFP4 support in-place
``addmm_`` / ``add_``, whereas 5-D Conv3D weights, 3-D MoE NVFP4 weights, and
``ConvRotTrellisTensor`` reject in-place merges during preflight).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from piper_kernels.weights import QuantizedWeight
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.convrot.trellis import ConvRotTrellisTensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor
from piper_kernels.weights.sharding import shard_quantized_weight, trellis_row_shard_byte_range

from ._nvfp4_projection import project_nvfp4_storage
from .structured_adapter import StructuredHost, StructuredSnapshot, _KernelMergeStructuredAdapter
from .tensor_adapters import metadata_key

__all__ = ["PiperQuantizedAdapter", "_PiperQuantizedMeta"]

_NVFP4_SLOTS = ("qdata", "scale", "per_tensor_scale", "act_per_tensor_scale")


@dataclass(slots=True, frozen=True)
class _PiperQuantizedMeta:
    """Generic ``QuantizedWeight`` snapshot preserved across host movement."""

    wrapper_type: type[QuantizedWeight]
    storage_names: tuple[str, ...]
    # Metadata values across QuantizedWeight subclasses can include ints, bools, dtypes, or configs.
    metadata: tuple[tuple[str, Any], ...]


def _validate_matrix_shard(
    label: str,
    global_shape: tuple[int, ...],
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if len(global_shape) != 2:
        raise ValueError(
            f"{label} shard projection supports 2-D matrices only; got shape {global_shape}."
        )
    if len(offsets) != 2 or len(local_shape) != 2:
        raise ValueError(
            f"{label} shard projection requires 2-D offsets and shape; "
            f"got offsets={offsets}, local_shape={local_shape}."
        )
    global_rows, global_cols = global_shape
    row_offset, col_offset = offsets
    local_rows, local_cols = local_shape
    if local_rows <= 0 or local_cols <= 0:
        raise ValueError(
            f"{label} shard projection requires non-empty local shards; got local_shape={local_shape}."
        )
    if (
        row_offset < 0
        or row_offset + local_rows > global_rows
        or col_offset < 0
        or col_offset + local_cols > global_cols
    ):
        raise ValueError(
            f"{label} shard projection range exceeds weight shape: "
            f"global_shape={global_shape}, offsets={offsets}, local_shape={local_shape}."
        )
    return (global_rows, global_cols), (row_offset, col_offset), (local_rows, local_cols)


def _require_positive_int(value: object, what: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{what} must be positive, got {value}.")
    return value


def _require_slot(
    storage_map: Mapping[str, torch.Tensor | None],
    name: str,
    label: str,
) -> torch.Tensor:
    tensor = storage_map.get(name)
    if tensor is None:
        raise RuntimeError(f"{label} host state is missing {name}.")
    return tensor


def _pack_projected_slots(
    state: StructuredHost[_PiperQuantizedMeta],
    projected: Mapping[str, torch.Tensor | None],
    target_shapes: Mapping[str, tuple[int, ...] | None] | None = None,
) -> tuple[tuple[torch.Tensor | None, ...], tuple[tuple[int, ...] | None, ...] | None]:
    storage = tuple(projected[name] for name in state.meta.storage_names)
    if target_shapes is None:
        return storage, None
    return storage, tuple(target_shapes[name] for name in state.meta.storage_names)


def _require_2d_mergeable_nvfp4(weight: PiperNVFP4Tensor, label: str) -> PiperNVFP4Tensor:
    if weight.ndim != 2:
        raise ValueError(f"{label} weight updates require a 2-D weight, got {weight.ndim}-D")
    rows, features = weight.shape
    if rows <= 0 or features <= 0 or features % weight.block_size != 0:
        raise ValueError(
            f"{label} weight updates require nonempty block-{weight.block_size} storage, "
            f"got shape {tuple(weight.shape)}"
        )
    if weight.scale.ndim != 2:
        raise ValueError(
            f"{label} weight updates require canonical 2-D scales, got shape {tuple(weight.scale.shape)}"
        )
    if weight.per_tensor_scale is not None and weight.per_tensor_scale.ndim != 0:
        raise ValueError(f"{label} weight updates require a scalar per_tensor_scale")
    return weight


def _project_convrot_int8(
    state: StructuredHost[_PiperQuantizedMeta],
    storage_map: dict[str, torch.Tensor | None],
    meta_map: dict[str, Any],
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> tuple[tuple[torch.Tensor | None, ...], tuple[tuple[int, ...] | None, ...] | None]:
    qdata = _require_slot(storage_map, "qdata", "ConvRot INT8")
    scale = _require_slot(storage_map, "scale", "ConvRot INT8")
    if len(state.logical_shape) != 2 or qdata.ndim != 2:
        raise ValueError(
            "ConvRot INT8 DTensor shard projection supports 2-D matrices "
            f"only; got shape {state.logical_shape}."
        )
    (global_rows, global_cols), (row_offset, col_offset), (local_rows, local_cols) = (
        _validate_matrix_shard("ConvRot INT8", state.logical_shape, offsets, local_shape)
    )
    if (
        not qdata.is_contiguous()
        or not scale.is_contiguous()
        or tuple(qdata.shape) != (global_rows, global_cols)
        or tuple(scale.shape) != (global_rows, 1)
    ):
        raise ValueError(
            "ConvRot INT8 DTensor shard projection requires contiguous, untransposed "
            f"qdata and scale storage; got qdata={tuple(qdata.shape)}, scale={tuple(scale.shape)}."
        )
    group_size = _require_positive_int(
        meta_map.get("group_size"), "ConvRot INT8 rotation group_size"
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
    return _pack_projected_slots(
        state,
        {
            "qdata": qdata_local,
            "scale": scale_local,
            "act_per_tensor_scale": storage_map.get("act_per_tensor_scale"),
        },
    )


def _project_nvfp4(
    cls_name: str,
    state: StructuredHost[_PiperQuantizedMeta],
    storage_map: dict[str, torch.Tensor | None],
    meta_map: dict[str, Any],
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> tuple[tuple[torch.Tensor | None, ...], tuple[tuple[int, ...] | None, ...] | None]:
    is_convrot = state.meta.wrapper_type is ConvRotNVFP4Tensor
    group_size = 0
    if is_convrot:
        if len(state.logical_shape) != 2:
            raise ValueError(
                f"{cls_name} shard projection only supports 2-D ConvRot NVFP4 tensors, "
                f"got shape {state.logical_shape}."
            )
        group_size = _require_positive_int(
            meta_map.get("group_size"), f"{cls_name} rotation group_size"
        )
    block_size = _require_positive_int(meta_map.get("block_size"), f"{cls_name} block_size")
    nvfp4_storage = tuple(storage_map.get(name) for name in _NVFP4_SLOTS)
    proj_storage, target_shapes = project_nvfp4_storage(
        adapter_name_label=cls_name,
        storage=nvfp4_storage,
        logical_shape=state.logical_shape,
        offsets=offsets,
        local_shape=local_shape,
        block_size=block_size,
        is_swizzled_scales=bool(meta_map["is_swizzled_scales"]),
    )
    if is_convrot:
        col_off, local_cols = offsets[1], local_shape[1]
        if (col_off != 0 or local_cols != state.logical_shape[1]) and (
            col_off % group_size != 0 or local_cols % group_size != 0
        ):
            raise ValueError(
                f"{cls_name} Shard(1) offset/width ({col_off}, {local_cols}) "
                f"must align to rotation group_size {group_size}."
            )
    return _pack_projected_slots(
        state,
        dict(zip(_NVFP4_SLOTS, proj_storage, strict=True)),
        dict(zip(_NVFP4_SLOTS, target_shapes, strict=True)),
    )


def _project_convrot_trellis(
    state: StructuredHost[_PiperQuantizedMeta],
    storage_map: dict[str, torch.Tensor | None],
    meta_map: dict[str, Any],
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> tuple[tuple[torch.Tensor | None, ...], tuple[tuple[int, ...] | None, ...] | None]:
    bitstreams = _require_slot(storage_map, "bitstreams", "ConvRot Trellis")
    block_scales = _require_slot(storage_map, "block_scales", "ConvRot Trellis")
    scale = _require_slot(storage_map, "scale", "ConvRot Trellis")
    row_widths = storage_map.get("row_widths")

    (global_rows, global_cols), (row_offset, col_offset), (local_rows, local_cols) = (
        _validate_matrix_shard("ConvRot Trellis", state.logical_shape, offsets, local_shape)
    )
    block_size = _require_positive_int(meta_map.get("block_size"), "ConvRot Trellis block_size")
    group_size = _require_positive_int(
        meta_map.get("group_size"), "ConvRot Trellis rotation group_size"
    )
    if global_cols % block_size != 0:
        raise ValueError(
            f"ConvRot Trellis width {global_cols} must divide by block_size {block_size}."
        )
    if any(
        t is not None and not t.is_contiguous()
        for t in (bitstreams, block_scales, scale, row_widths)
    ):
        raise ValueError("ConvRot Trellis shard projection requires contiguous storage.")

    blocks_per_row = global_cols // block_size
    if (
        tuple(block_scales.shape) != (global_rows, blocks_per_row)
        or tuple(scale.shape) != (global_rows, 1)
    ):
        raise ValueError(
            "ConvRot Trellis shard projection encountered incompatible scale shapes: "
            f"block_scales={tuple(block_scales.shape)}, scale={tuple(scale.shape)}."
        )

    col_sharded = col_offset != 0 or local_cols != global_cols
    row_sharded = row_offset != 0 or local_rows != global_rows
    if col_sharded:
        if col_offset % group_size != 0 or local_cols % group_size != 0:
            raise ValueError(
                "ConvRot Trellis Shard(1) projection along columns "
                f"(offset={col_offset}, size={local_cols}) must align to "
                f"rotation group_size {group_size}."
            )
        if col_offset % block_size != 0 or local_cols % block_size != 0:
            raise ValueError(
                "ConvRot Trellis Shard(1) projection along columns "
                f"(offset={col_offset}, size={local_cols}) must align to "
                f"block_size {block_size}."
            )

    if row_widths is None:
        if bitstreams.ndim != 3 or tuple(bitstreams.shape[:2]) != (global_rows, blocks_per_row):
            raise ValueError(
                "ConvRot Trellis uniform shard projection requires 3-D bitstreams "
                f"[out, blocks, bytes], got shape {tuple(bitstreams.shape)}."
            )
        bit_local = bitstreams.narrow(0, row_offset, local_rows) if row_sharded else bitstreams
        bscale_local = block_scales.narrow(0, row_offset, local_rows) if row_sharded else block_scales
        scale_local = scale.narrow(0, row_offset, local_rows) if row_sharded else scale
        if col_sharded:
            b_start = col_offset // block_size
            b_len = local_cols // block_size
            bit_local = bit_local.narrow(1, b_start, b_len)
            bscale_local = bscale_local.narrow(1, b_start, b_len)
        return _pack_projected_slots(
            state,
            {"bitstreams": bit_local, "block_scales": bscale_local, "scale": scale_local},
        )

    if bitstreams.ndim != 1 or tuple(row_widths.shape) != (global_rows,):
        raise ValueError(
            "ConvRot Trellis variable-row shard projection requires 1-D bitstreams and "
            f"row_widths ({global_rows},), got bitstreams={tuple(bitstreams.shape)}, "
            f"row_widths={tuple(row_widths.shape)}."
        )
    byte_range = trellis_row_shard_byte_range(
        row_widths,
        blocks_per_row=blocks_per_row,
        start=row_offset,
        length=local_rows,
        block_size=block_size,
    )
    if byte_range.offset + byte_range.length > bitstreams.numel():
        raise ValueError(
            "ConvRot Trellis variable-row bitstreams is shorter than row_widths requires: "
            f"need {byte_range.offset + byte_range.length} bytes, got {bitstreams.numel()}."
        )
    row_tensors = {
        "bitstreams": bitstreams.narrow(0, byte_range.offset, byte_range.length),
        "block_scales": block_scales.narrow(0, row_offset, local_rows),
        "scale": scale.narrow(0, row_offset, local_rows),
        "row_widths": row_widths.narrow(0, row_offset, local_rows),
    }
    if not col_sharded:
        return _pack_projected_slots(state, row_tensors)

    row_weight = state.meta.wrapper_type.from_storage(row_tensors, meta_map)
    col_weight = shard_quantized_weight(row_weight, dim=1, start=col_offset, length=local_cols)
    sharded_tensors, _ = col_weight.as_storage()
    return _pack_projected_slots(state, sharded_tensors)


class PiperQuantizedAdapter(_KernelMergeStructuredAdapter[_PiperQuantizedMeta]):
    """Generic structured adapter for all ``piper_kernels.weights.QuantizedWeight`` formats."""

    _TAG = "piper-kernels-quantized"

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return isinstance(t, QuantizedWeight)

    @staticmethod
    def _validate_and_unpack(
        t: torch.Tensor,
    ) -> tuple[QuantizedWeight, dict[str, torch.Tensor], dict[str, Any]]:
        if not isinstance(t, QuantizedWeight):
            raise TypeError(f"expected QuantizedWeight, got {type(t).__name__}")
        tensors, metadata = t.as_storage()
        for name, tensor in tensors.items():
            if not tensor.is_contiguous():
                raise ValueError(
                    f"{type(t).__name__} storage tensor {name!r} must be contiguous"
                )
        return t, tensors, metadata

    @staticmethod
    def _require(t: torch.Tensor) -> QuantizedWeight:
        return PiperQuantizedAdapter._validate_and_unpack(t)[0]

    @staticmethod
    def _extract(t: torch.Tensor) -> StructuredSnapshot[_PiperQuantizedMeta]:
        weight, tensors, metadata = PiperQuantizedAdapter._validate_and_unpack(t)
        sorted_meta = tuple(sorted(metadata.items()))
        return StructuredSnapshot(
            tensor=weight,
            named_storage=tuple(tensors.items()),
            meta=_PiperQuantizedMeta(
                wrapper_type=type(weight),
                storage_names=tuple(tensors.keys()),
                metadata=sorted_meta,
            ),
            id_metadata=(
                type(weight),
                tuple((key, metadata_key(val)) for key, val in sorted_meta),
            ),
        )

    @staticmethod
    def _reconstruct(
        storage: tuple[torch.Tensor | None, ...],
        meta: _PiperQuantizedMeta,
    ) -> torch.Tensor:
        tensor_map = {
            name: tensor
            for name, tensor in zip(meta.storage_names, storage, strict=True)
            if tensor is not None
        }
        return meta.wrapper_type.from_storage(tensor_map, dict(meta.metadata))

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return cast(QuantizedWeight, t).dtype

    @staticmethod
    def _require_matrix(t: torch.Tensor) -> QuantizedWeight:
        weight = PiperQuantizedAdapter._require(t)
        weight_type = type(weight)
        if weight_type is ConvRotInt8Tensor:
            if weight.ndim != 2:
                raise NotImplementedError("ConvRot INT8 weight updates require a 2-D weight")
            return weight
        if weight_type in (PiperNVFP4Tensor, ConvRotNVFP4Tensor):
            label = "ConvRot NVFP4" if weight_type is ConvRotNVFP4Tensor else "Piper NVFP4"
            return _require_2d_mergeable_nvfp4(cast(PiperNVFP4Tensor, weight), label)
        raise ValueError(f"{weight_type.__name__} does not support in-place weight updates.")

    @classmethod
    def project_shard(
        cls,
        state: StructuredHost[_PiperQuantizedMeta],
        *,
        offsets: tuple[int, ...],
        local_shape: tuple[int, ...],
    ) -> tuple[StructuredHost[_PiperQuantizedMeta], tuple[object, ...]]:
        """Project a host shard across supported ``QuantizedWeight`` subclasses."""
        storage_map = dict(zip(state.meta.storage_names, state.storage, strict=True))
        meta_map = dict(state.meta.metadata)
        weight_type = state.meta.wrapper_type

        if weight_type is ConvRotInt8Tensor:
            proj_storage, target_shapes = _project_convrot_int8(
                state, storage_map, meta_map, offsets, local_shape
            )
        elif weight_type in (PiperNVFP4Tensor, ConvRotNVFP4Tensor):
            proj_storage, target_shapes = _project_nvfp4(
                cls.__name__, state, storage_map, meta_map, offsets, local_shape
            )
        elif weight_type is ConvRotTrellisTensor:
            proj_storage, target_shapes = _project_convrot_trellis(
                state, storage_map, meta_map, offsets, local_shape
            )
        else:
            raise NotImplementedError(
                f"{weight_type.__name__} DTensor shard projection is not supported."
            )

        return cls._finalize_projected_shard(
            proj_storage,
            state.meta,
            local_shape,
            target_shapes=target_shapes,
        )
