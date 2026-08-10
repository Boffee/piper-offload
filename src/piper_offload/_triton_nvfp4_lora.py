"""Triton kernels for TorchAO NVFP4 LoRA merges."""

# Triton JIT kernel signatures intentionally use untyped pointer parameters
# and upper-case constexpr names.
# ruff: noqa: ANN001, ANN202, N803, PLR0913
# pyright: reportCallIssue=false, reportGeneralTypeIssues=false

import torch
import triton
import triton.language as tl

from ._triton_stochastic_quantization import (
    _seed_argument,
    _stochastic_e2m1_code,
)

_COMPUTE_BF16 = 0
_COMPUTE_FP32 = 1
_BLOCK_SIZE = 16
_PACKED_BLOCK_SIZE = 8
_REDUCTION_BLOCK = 8192


@triton.jit
def _decode_fp4(code):
    magnitude_code = code & 0x7
    magnitude = tl.where(
        magnitude_code == 0,
        0.0,
        tl.where(
            magnitude_code == 1,
            0.5,
            tl.where(
                magnitude_code == 2,
                1.0,
                tl.where(
                    magnitude_code == 3,
                    1.5,
                    tl.where(
                        magnitude_code == 4,
                        2.0,
                        tl.where(
                            magnitude_code == 5,
                            3.0,
                            tl.where(magnitude_code == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return tl.where((code & 0x8) != 0, -magnitude, magnitude)


@triton.jit
def _encode_fp4(value):
    magnitude = tl.abs(value)
    magnitude_code = tl.where(
        magnitude <= 0.25,
        0,
        tl.where(
            magnitude < 0.75,
            1,
            tl.where(
                magnitude <= 1.25,
                2,
                tl.where(
                    magnitude < 1.75,
                    3,
                    tl.where(
                        magnitude <= 2.5,
                        4,
                        tl.where(
                            magnitude < 3.5,
                            5,
                            tl.where(magnitude <= 5.0, 6, 7),
                        ),
                    ),
                ),
            ),
        ),
    )
    value_bits = value.to(tl.int32, bitcast=True)
    sign = ((value_bits >> 31) & 1) << 3
    return magnitude_code | sign


@triton.jit
def _scale_offset(
    row,
    scale_col,
    NUM_SWIZZLE_COL_BLOCKS,
    SCALE_COLS: tl.constexpr,
    SWIZZLED: tl.constexpr,
):
    if SWIZZLED:
        row_block = row // 128
        col_block = scale_col // 4
        row_in_block = row % 128
        row_group = row_in_block // 32
        row_lane = row_in_block % 32
        col_in_block = scale_col % 4
        return (row_block * NUM_SWIZZLE_COL_BLOCKS + col_block) * 512 + row_lane * 16 + row_group * 4 + col_in_block
    return row * SCALE_COLS + scale_col


@triton.jit
def _merged_tile(
    qdata_ptr,
    scale_ptr,
    input_global_scale_ptr,
    b_ptr,
    a_ptr,
    strength,
    M,
    PACKED_N,
    NUM_SWIZZLE_COL_BLOCKS,
    R: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    HAS_GLOBAL_SCALE: tl.constexpr,
    SWIZZLED: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * 16 + tl.arange(0, 16)
    row_mask = offsets_m < M
    mask = row_mask[:, None]

    packed_offsets = offsets_m[:, None] * PACKED_N + offsets_n[None, :] // 2
    packed = tl.load(qdata_ptr + packed_offsets, mask=mask, other=0)
    code = tl.where(
        (offsets_n[None, :] & 1) == 0,
        packed & 0xF,
        packed >> 4,
    )
    base = _decode_fp4(code).to(tl.float32)

    scale_offsets = _scale_offset(
        offsets_m,
        pid_n,
        NUM_SWIZZLE_COL_BLOCKS,
        SCALE_COLS,
        SWIZZLED,
    )
    block_scale = tl.load(
        scale_ptr + scale_offsets,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    if COMPUTE_DTYPE == 0:
        block_scale = block_scale.to(tl.bfloat16).to(tl.float32)
    if HAS_GLOBAL_SCALE:
        global_scale = tl.load(input_global_scale_ptr).to(tl.float32)
        block_scale = block_scale * global_scale
        if COMPUTE_DTYPE == 0:
            block_scale = block_scale.to(tl.bfloat16).to(tl.float32)
    base *= block_scale[:, None]
    if COMPUTE_DTYPE == 0:
        base = base.to(tl.bfloat16)

    accumulator = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    for rank_start in range(0, R, BLOCK_R):
        offsets_r = rank_start + tl.arange(0, BLOCK_R)
        b = tl.load(
            b_ptr + offsets_m[:, None] * R + offsets_r[None, :],
            mask=row_mask[:, None] & (offsets_r[None, :] < R),
            other=0.0,
        )
        a = tl.load(
            a_ptr + offsets_r[:, None] * (PACKED_N * 2) + offsets_n[None, :],
            mask=(offsets_r[:, None] < R),
            other=0.0,
        )
        if COMPUTE_DTYPE == 1:
            accumulator += tl.dot(b, a, input_precision="ieee")
        else:
            accumulator += tl.dot(b, a)

    update = accumulator * strength
    merged = tl.where(update == 0.0, base.to(tl.float32), base + update)
    if COMPUTE_DTYPE == 0:
        merged = merged.to(tl.bfloat16)
    return merged, offsets_m, row_mask


@triton.jit
def _merged_amax_kernel(
    qdata_ptr,
    scale_ptr,
    input_global_scale_ptr,
    b_ptr,
    a_ptr,
    partial_max_ptr,
    strength,
    M,
    PACKED_N,
    NUM_SWIZZLE_COL_BLOCKS,
    R: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    HAS_GLOBAL_SCALE: tl.constexpr,
    SWIZZLED: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    merged, _offsets_m, row_mask = _merged_tile(
        qdata_ptr,
        scale_ptr,
        input_global_scale_ptr,
        b_ptr,
        a_ptr,
        strength,
        M,
        PACKED_N,
        NUM_SWIZZLE_COL_BLOCKS,
        R,
        SCALE_COLS,
        HAS_GLOBAL_SCALE,
        SWIZZLED,
        COMPUTE_DTYPE,
        BLOCK_M,
        BLOCK_R,
    )
    absolute = tl.where(row_mask[:, None], tl.abs(merged.to(tl.float32)), 0.0)
    tile_max = tl.max(tl.max(absolute, axis=1), axis=0)
    tile_offset = tl.program_id(axis=0) * SCALE_COLS + tl.program_id(axis=1)
    tl.store(partial_max_ptr + tile_offset, tile_max)


@triton.jit
def _reduce_max_kernel(
    input_ptr,
    output_ptr,
    NUM_VALUES,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        input_ptr + offsets,
        mask=offsets < NUM_VALUES,
        other=0.0,
    )
    tl.store(output_ptr + tl.program_id(axis=0), tl.max(values, axis=0))


@triton.jit
def _global_scale_kernel(
    partial_max_ptr,
    output_global_scale_ptr,
    NUM_VALUES,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        partial_max_ptr + offsets,
        mask=offsets < NUM_VALUES,
        other=0.0,
    )
    max_abs = tl.max(values, axis=0)
    global_scale = max_abs / 2688.0
    global_scale = tl.maximum(global_scale, 1.1920928955078125e-07)
    tl.store(output_global_scale_ptr, global_scale)


@triton.jit
def _quantize_kernel(
    qdata_ptr,
    scale_ptr,
    input_global_scale_ptr,
    output_global_scale_ptr,
    b_ptr,
    a_ptr,
    output_qdata_ptr,
    output_scale_ptr,
    strength,
    rounding_seed,
    M,
    PACKED_N,
    NUM_SWIZZLE_COL_BLOCKS,
    R: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    HAS_GLOBAL_SCALE: tl.constexpr,
    SWIZZLED: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    merged, offsets_m, row_mask = _merged_tile(
        qdata_ptr,
        scale_ptr,
        input_global_scale_ptr,
        b_ptr,
        a_ptr,
        strength,
        M,
        PACKED_N,
        NUM_SWIZZLE_COL_BLOCKS,
        R,
        SCALE_COLS,
        HAS_GLOBAL_SCALE,
        SWIZZLED,
        COMPUTE_DTYPE,
        BLOCK_M,
        BLOCK_R,
    )
    merged_f32 = merged.to(tl.float32)
    max_abs = tl.max(
        tl.where(row_mask[:, None], tl.abs(merged_f32), 0.0),
        axis=1,
    )
    block_scale = max_abs / 6.0

    if HAS_GLOBAL_SCALE:
        output_global_scale = tl.load(output_global_scale_ptr).to(tl.float32)
        fp8_scale = block_scale / output_global_scale
    else:
        output_global_scale = 1.0
        fp8_scale = block_scale
    fp8_scale = tl.minimum(tl.maximum(fp8_scale, 0.015625), 448.0)
    fp8_scale = fp8_scale.to(tl.float8e4nv).to(tl.float32)

    scale_offsets = _scale_offset(
        offsets_m,
        tl.program_id(axis=1),
        NUM_SWIZZLE_COL_BLOCKS,
        SCALE_COLS,
        SWIZZLED,
    )
    tl.store(
        output_scale_ptr + scale_offsets,
        fp8_scale,
        mask=row_mask,
    )

    if HAS_GLOBAL_SCALE:
        reciprocal_scale = (1.0 / output_global_scale) / fp8_scale
        normalized = merged_f32 * reciprocal_scale[:, None]
    else:
        normalized = merged_f32 / fp8_scale[:, None]
    normalized = tl.minimum(tl.maximum(normalized, -6.0), 6.0)
    code = _encode_fp4(normalized)
    if STOCHASTIC:
        logical_offsets = offsets_m[:, None] * (PACKED_N * 2) + (
            tl.program_id(axis=1) * 16 + tl.arange(0, 16)
        )[None, :]
        code = _stochastic_e2m1_code(
            normalized,
            code,
            rounding_seed,
            logical_offsets,
        )
    code_pairs = code.reshape(BLOCK_M, 8, 2)
    shifts = (tl.arange(0, 2) * 4)[None, None, :]
    packed = tl.sum(code_pairs << shifts, axis=2)

    packed_cols = tl.program_id(axis=1) * 8 + tl.arange(0, 8)
    packed_offsets = offsets_m[:, None] * PACKED_N + packed_cols[None, :]
    tl.store(
        output_qdata_ptr + packed_offsets,
        packed,
        mask=row_mask[:, None],
    )


def _compute_dtype_id(dtype: torch.dtype) -> int:
    if dtype is torch.bfloat16:
        return _COMPUTE_BF16
    if dtype is torch.float32:
        return _COMPUTE_FP32
    raise ValueError(f"Triton NVFP4 merge supports bfloat16 and float32 LoRA factors, got {dtype}.")


def _validate_inputs(  # noqa: PLR0912
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor | None,
    block_size: int,
    is_swizzled_scales: bool,
    b: torch.Tensor,
    a: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Validate a raw NVFP4 launch and return M, N, rank, and dtype id."""
    if qdata.device.type != "cuda":
        raise ValueError("Triton NVFP4 merge requires CUDA tensors.")
    if qdata.dtype is not torch.uint8:
        raise ValueError(f"Triton NVFP4 merge expects packed uint8 data, got {qdata.dtype}.")
    if scale.dtype is not torch.float8_e4m3fn:
        raise ValueError("Triton NVFP4 merge expects E4M3FN block scales.")
    if qdata.ndim != 2 or scale.ndim != 2 or b.ndim != 2 or a.ndim != 2:
        raise ValueError("Triton NVFP4 merge expects rank-two tensors.")
    if not qdata.is_contiguous() or not scale.is_contiguous():
        raise ValueError("Triton NVFP4 merge requires contiguous packed data and scales.")
    if b.dtype is not a.dtype:
        raise ValueError("Triton NVFP4 merge requires matching factor dtypes.")
    compute_dtype = _compute_dtype_id(b.dtype)
    if (
        qdata.device != scale.device
        or qdata.device != b.device
        or qdata.device != a.device
        or (per_tensor_scale is not None and qdata.device != per_tensor_scale.device)
    ):
        raise ValueError("Triton NVFP4 merge requires all tensors on one CUDA device.")
    if block_size != _BLOCK_SIZE:
        raise ValueError("Triton NVFP4 merge requires block_size=16.")

    rows, packed_cols = qdata.shape
    cols = packed_cols * 2
    rank = a.shape[0]
    if rows == 0 or cols == 0 or rank == 0:
        raise ValueError("Triton NVFP4 merge requires non-empty tensors.")
    if packed_cols % _PACKED_BLOCK_SIZE != 0:
        raise ValueError("Triton NVFP4 merge requires columns divisible by 16.")
    if b.shape != (rows, rank) or a.shape[1] != cols:
        raise ValueError("LoRA factors do not match the NVFP4 weight shape.")

    scale_cols = cols // _BLOCK_SIZE
    expected_scale_shape = (
        (
            (rows + 127) // 128 * 32,
            (cols + 63) // 64 * 16,
        )
        if is_swizzled_scales
        else (rows, scale_cols)
    )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError("NVFP4 scale shape does not match its swizzle metadata.")
    if per_tensor_scale is not None and (per_tensor_scale.dtype is not torch.float32 or per_tensor_scale.numel() != 1):
        raise ValueError("Triton NVFP4 merge expects one float32 per-tensor scale.")
    return rows, cols, rank, compute_dtype


def _recompute_global_scale(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    input_global_scale: torch.Tensor,
    is_swizzled_scales: bool,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    rows: int,
    cols: int,
    rank: int,
    compute_dtype: int,
    block_m: int,
    block_r: int,
) -> torch.Tensor:
    """Recompute the two-level global scale without materializing dense data."""
    scale_cols = cols // _BLOCK_SIZE
    num_pid_m = (rows + block_m - 1) // block_m
    num_tiles = num_pid_m * scale_cols
    partial_max = torch.empty(
        num_tiles,
        device=qdata.device,
        dtype=torch.float32,
    )
    num_swizzle_col_blocks = (scale_cols + 3) // 4
    _merged_amax_kernel[(num_pid_m, scale_cols)](
        qdata,
        scale,
        input_global_scale,
        b,
        a,
        partial_max,
        strength,
        M=rows,
        PACKED_N=cols // 2,
        NUM_SWIZZLE_COL_BLOCKS=num_swizzle_col_blocks,
        R=rank,
        SCALE_COLS=scale_cols,
        HAS_GLOBAL_SCALE=True,
        SWIZZLED=is_swizzled_scales,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_M=block_m,
        BLOCK_R=block_r,
        num_warps=4,
    )

    reduction_input = partial_max
    num_values = num_tiles
    while num_values > _REDUCTION_BLOCK:
        output_count = (num_values + _REDUCTION_BLOCK - 1) // _REDUCTION_BLOCK
        reduced = torch.empty(
            output_count,
            device=qdata.device,
            dtype=torch.float32,
        )
        _reduce_max_kernel[(output_count,)](
            reduction_input,
            reduced,
            NUM_VALUES=num_values,
            BLOCK_SIZE=_REDUCTION_BLOCK,
            num_warps=8,
        )
        reduction_input = reduced
        num_values = output_count

    output_global_scale = torch.empty_like(input_global_scale)
    final_reduction_block = int(triton.next_power_of_2(num_values))
    _global_scale_kernel[(1,)](
        reduction_input,
        output_global_scale,
        NUM_VALUES=num_values,
        BLOCK_SIZE=final_reduction_block,
        num_warps=4 if final_reduction_block <= 2048 else 8,
    )
    return output_global_scale


def merge_nvfp4_lora(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor | None,
    block_size: int,
    is_swizzled_scales: bool,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    rounding_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return packed NVFP4 buffers after one raw LoRA merge."""
    rows, cols, rank, compute_dtype = _validate_inputs(
        qdata,
        scale,
        per_tensor_scale,
        block_size,
        is_swizzled_scales,
        b,
        a,
    )
    qdata = qdata.contiguous()
    scale = scale.contiguous()
    b = b.contiguous()
    a = a.contiguous()

    block_m = 16
    block_r = 16 if rank <= 16 else 32
    output_global_scale = (
        _recompute_global_scale(
            qdata,
            scale,
            per_tensor_scale,
            is_swizzled_scales,
            b,
            a,
            strength,
            rows=rows,
            cols=cols,
            rank=rank,
            compute_dtype=compute_dtype,
            block_m=block_m,
            block_r=block_r,
        )
        if per_tensor_scale is not None
        else None
    )

    output_qdata = torch.empty_like(qdata)
    output_scale = torch.zeros_like(scale)
    scale_cols = cols // _BLOCK_SIZE
    num_pid_m = (rows + block_m - 1) // block_m
    num_swizzle_col_blocks = (scale_cols + 3) // 4
    input_global_scale = per_tensor_scale if per_tensor_scale is not None else scale
    quant_global_scale = output_global_scale if output_global_scale is not None else scale
    _quantize_kernel[(num_pid_m, scale_cols)](
        qdata,
        scale,
        input_global_scale,
        quant_global_scale,
        b,
        a,
        output_qdata,
        output_scale,
        strength,
        _seed_argument(rounding_seed),
        M=rows,
        PACKED_N=cols // 2,
        NUM_SWIZZLE_COL_BLOCKS=num_swizzle_col_blocks,
        R=rank,
        SCALE_COLS=scale_cols,
        HAS_GLOBAL_SCALE=per_tensor_scale is not None,
        SWIZZLED=is_swizzled_scales,
        COMPUTE_DTYPE=compute_dtype,
        STOCHASTIC=rounding_seed is not None,
        BLOCK_M=block_m,
        BLOCK_R=block_r,
        num_warps=4,
    )
    return output_qdata, output_scale, output_global_scale
