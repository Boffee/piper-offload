"""Zero-copy DTensor shard projection helpers for 2-D and 3-D NVFP4 weights."""

from __future__ import annotations

import torch

_SWIZZLED_ROW_GROUP = 128
_SWIZZLED_COL_GROUPS_PER_TILE = 4
_SWIZZLED_TILE_ROWS = 32
_SWIZZLED_TILE_COLS = 16
_SWIZZLED_TILE_ELEMS = _SWIZZLED_TILE_ROWS * _SWIZZLED_TILE_COLS


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _narrow_axes(
    tensor: torch.Tensor,
    axis_slices: tuple[tuple[int, int, int], ...],
) -> torch.Tensor:
    """Narrow ``tensor`` zero-copy along each axis where the slice is proper."""
    view = tensor
    for axis, (dim_size, off, length) in enumerate(axis_slices):
        if off != 0 or length != dim_size:
            view = view.narrow(axis, off, length)
    return view


def _validate_swizzled_axis(
    adapter_name_label: str,
    axis: int,
    dim_size: int,
    offset: int,
    length: int,
    tile_span: int,
) -> None:
    """Ensure a swizzled shard respects 128x4 tile boundaries (allowing a tail block)."""
    if offset % tile_span != 0:
        raise ValueError(
            f"{adapter_name_label} swizzled Shard({axis}) offset {offset} must align "
            f"to {tile_span}."
        )
    if length % tile_span != 0 and offset + length != dim_size:
        raise ValueError(
            f"{adapter_name_label} swizzled Shard({axis}) length {length} at offset "
            f"{offset} (dim size {dim_size}) must align to {tile_span} unless it is "
            "the final tail shard."
        )


def _validate_shard_inputs(
    adapter_name_label: str,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    logical_shape: tuple[int, ...],
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
    block_size: int,
) -> None:
    rank = len(logical_shape)
    if rank not in (2, 3):
        raise ValueError(
            f"{adapter_name_label} shard projection only supports 2-D and 3-D tensors, "
            f"got shape {logical_shape}."
        )
    if len(offsets) != rank or len(local_shape) != rank:
        raise ValueError(
            f"{adapter_name_label} expected {rank}-D offsets and local_shape, "
            f"got offsets={offsets}, local_shape={local_shape}."
        )
    for axis, (dim_size, off, length) in enumerate(
        zip(logical_shape, offsets, local_shape, strict=True)
    ):
        if length <= 0:
            raise ValueError(
                f"{adapter_name_label} Shard({axis}) produced empty shard "
                f"(local length {length})."
            )
        if off < 0 or off + length > dim_size:
            raise ValueError(
                f"{adapter_name_label} Shard({axis}) slice [{off}:{off + length}] "
                f"is out of bounds for dimension size {dim_size}."
            )
    if not qdata.is_contiguous() or not scale.is_contiguous():
        raise ValueError(
            f"{adapter_name_label} shard projection requires contiguous qdata and scale "
            f"storage (qdata contiguous={qdata.is_contiguous()}, "
            f"scale contiguous={scale.is_contiguous()})."
        )
    if block_size <= 0:
        raise ValueError(
            f"{adapter_name_label} block_size must be positive, got {block_size}."
        )


def _project_qdata_view(
    adapter_name_label: str,
    qdata: torch.Tensor,
    logical_shape: tuple[int, ...],
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> torch.Tensor:
    leading = logical_shape[:-2]
    leading_slices = tuple(zip(leading, offsets[:-2], local_shape[:-2], strict=True))
    rows, cols = logical_shape[-2], logical_shape[-1]
    col_axis = len(logical_shape) - 1
    col_off, local_cols = offsets[col_axis], local_shape[col_axis]

    if cols % 2 != 0:
        raise ValueError(f"{adapter_name_label} packed FP4 width {cols} must be even.")
    if col_off % 2 != 0 or local_cols % 2 != 0:
        raise ValueError(
            f"{adapter_name_label} Shard({col_axis}) offset/width "
            f"({col_off}, {local_cols}) must be even for packed FP4 storage."
        )
    expected_q_cols = cols // 2
    expected_q_shape = (*leading, rows, expected_q_cols)
    if tuple(qdata.shape) != expected_q_shape:
        raise ValueError(
            f"{adapter_name_label} qdata shape {tuple(qdata.shape)} does not match "
            f"expected {expected_q_shape} for logical shape {logical_shape}."
        )
    return _narrow_axes(
        qdata,
        (
            *leading_slices,
            (rows, offsets[-2], local_shape[-2]),
            (expected_q_cols, col_off // 2, local_cols // 2),
        ),
    )


def _project_scale_view(
    adapter_name_label: str,
    scale: torch.Tensor,
    logical_shape: tuple[int, ...],
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
    block_size: int,
    *,
    is_swizzled_scales: bool,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    leading = logical_shape[:-2]
    local_leading = local_shape[:-2]
    leading_slices = tuple(zip(leading, offsets[:-2], local_leading, strict=True))
    row_axis, col_axis = len(logical_shape) - 2, len(logical_shape) - 1
    rows, cols = logical_shape[-2], logical_shape[-1]
    row_off, local_rows = offsets[row_axis], local_shape[row_axis]
    col_off, local_cols = offsets[col_axis], local_shape[col_axis]

    if cols % block_size != 0:
        raise ValueError(
            f"{adapter_name_label} column dimension {cols} must divide evenly by "
            f"block_size {block_size}."
        )
    if col_off % block_size != 0 or local_cols % block_size != 0:
        raise ValueError(
            f"{adapter_name_label} Shard({col_axis}) offset/width ({col_off}, "
            f"{local_cols}) must align to block_size {block_size}."
        )

    if not is_swizzled_scales:
        scale_cols = cols // block_size
        expected_scale_shape = (*leading, rows, scale_cols)
        if tuple(scale.shape) != expected_scale_shape:
            raise ValueError(
                f"{adapter_name_label} unswizzled scale shape {tuple(scale.shape)} "
                f"does not match expected {expected_scale_shape}."
            )
        scale_view = _narrow_axes(
            scale,
            (
                *leading_slices,
                (rows, row_off, local_rows),
                (scale_cols, col_off // block_size, local_cols // block_size),
            ),
        )
        return scale_view, tuple(scale_view.shape)

    col_tile_span = _SWIZZLED_COL_GROUPS_PER_TILE * block_size
    _validate_swizzled_axis(
        adapter_name_label, row_axis, rows, row_off, local_rows, _SWIZZLED_ROW_GROUP
    )
    _validate_swizzled_axis(
        adapter_name_label, col_axis, cols, col_off, local_cols, col_tile_span
    )
    n_rb = _ceil_div(rows, _SWIZZLED_ROW_GROUP)
    n_cb = _ceil_div(cols // block_size, _SWIZZLED_COL_GROUPS_PER_TILE)
    expected_scale_shape = (
        *leading,
        n_rb * _SWIZZLED_TILE_ROWS,
        n_cb * _SWIZZLED_TILE_COLS,
    )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"{adapter_name_label} swizzled scale shape {tuple(scale.shape)} "
            f"does not match expected {expected_scale_shape} for logical shape "
            f"{logical_shape}."
        )
    local_n_rb = _ceil_div(local_rows, _SWIZZLED_ROW_GROUP)
    local_n_cb = _ceil_div(local_cols // block_size, _SWIZZLED_COL_GROUPS_PER_TILE)
    scale_view = _narrow_axes(
        scale.view(*leading, n_rb, n_cb, _SWIZZLED_TILE_ELEMS),
        (
            *leading_slices,
            (n_rb, row_off // _SWIZZLED_ROW_GROUP, local_n_rb),
            (n_cb, col_off // col_tile_span, local_n_cb),
        ),
    )
    target_scale_shape = (
        *local_leading,
        local_n_rb * _SWIZZLED_TILE_ROWS,
        local_n_cb * _SWIZZLED_TILE_COLS,
    )
    if scale_view.is_contiguous():
        scale_view = scale_view.view(target_scale_shape)
    return scale_view, target_scale_shape


def _project_optional_global_scale(
    *,
    adapter_name_label: str,
    scale_name: str,
    tensor: torch.Tensor | None,
    logical_shape: tuple[int, ...],
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
    allow_per_expert_3d: bool = False,
) -> torch.Tensor | None:
    """Project an optional global scalar or 3-D ``(E, 1, 1)`` scale without copying."""
    if tensor is None:
        return None
    if not tensor.is_contiguous():
        raise ValueError(
            f"{adapter_name_label} {scale_name} must be contiguous for shard projection."
        )
    if tensor.ndim == 0:
        return tensor
    if (
        allow_per_expert_3d
        and len(logical_shape) == 3
        and tensor.ndim == 3
        and tuple(tensor.shape) == (logical_shape[0], 1, 1)
    ):
        return _narrow_axes(tensor, ((logical_shape[0], offsets[0], local_shape[0]),))
    raise ValueError(
        f"{adapter_name_label} {scale_name} shape {tuple(tensor.shape)} cannot be "
        f"projected for logical shape {logical_shape}."
    )


def project_nvfp4_storage(
    *,
    adapter_name_label: str,
    storage: tuple[torch.Tensor | None, ...],
    logical_shape: tuple[int, ...],
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
    block_size: int,
    is_swizzled_scales: bool,
) -> tuple[tuple[torch.Tensor | None, ...], tuple[tuple[int, ...] | None, ...]]:
    """Project zero-copy 2-D/3-D NVFP4 storage views and canonical GPU buffer shapes."""
    qdata, scale, per_tensor_scale, act_per_tensor_scale = storage
    if qdata is None or scale is None:
        raise RuntimeError(f"{adapter_name_label} host state is missing qdata or scale.")
    _validate_shard_inputs(
        adapter_name_label,
        qdata,
        scale,
        logical_shape,
        offsets,
        local_shape,
        block_size,
    )
    q_view = _project_qdata_view(
        adapter_name_label, qdata, logical_shape, offsets, local_shape
    )
    scale_view, target_scale_shape = _project_scale_view(
        adapter_name_label,
        scale,
        logical_shape,
        offsets,
        local_shape,
        block_size,
        is_swizzled_scales=is_swizzled_scales,
    )
    proj_per_tensor = _project_optional_global_scale(
        adapter_name_label=adapter_name_label,
        scale_name="per_tensor_scale",
        tensor=per_tensor_scale,
        logical_shape=logical_shape,
        offsets=offsets,
        local_shape=local_shape,
        allow_per_expert_3d=True,
    )
    proj_act_per_tensor = _project_optional_global_scale(
        adapter_name_label=adapter_name_label,
        scale_name="act_per_tensor_scale",
        tensor=act_per_tensor_scale,
        logical_shape=logical_shape,
        offsets=offsets,
        local_shape=local_shape,
        allow_per_expert_3d=False,
    )
    return (
        (q_view, scale_view, proj_per_tensor, proj_act_per_tensor),
        (tuple(q_view.shape), target_scale_shape, None, None),
    )
