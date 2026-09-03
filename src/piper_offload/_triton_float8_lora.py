"""Triton kernels for TorchAO scaled-FP8 LoRA and dense merges."""

# Triton JIT kernel signatures intentionally use untyped pointer parameters
# and upper-case constexpr names.
# ruff: noqa: ANN001, ANN202, N803, PLR0912, PLR0913
# pyright: reportCallIssue=false, reportIndexIssue=false

import torch
import triton
import triton.language as tl

from ._triton_stochastic_quantization import (
    _seed_argument,
    _stochastic_float8,
)

_COMPUTE_FP16 = 0
_COMPUTE_BF16 = 1
_COMPUTE_FP32 = 2
_REDUCTION_BLOCK = 8192
# A group maps to one output tile. Bound that tile so unusual very large
# groups use the generic path instead of producing an impractical kernel.
MAX_GROUP_SIZE = 256


@triton.jit
def _merge_update_kernel(
    qdata_ptr,
    scale_ptr,
    b_ptr,
    a_ptr,
    update_ptr,
    dense_ptr,
    partial_max_ptr,
    strength,
    M,
    N,
    K: tl.constexpr,
    DENSE_UPDATE: tl.constexpr,
    PER_ROW: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = offsets_m[:, None] * N + offsets_n[None, :]
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    if DENSE_UPDATE:
        accumulator = tl.load(
            update_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offsets_k = k_start + tl.arange(0, BLOCK_K)
            b = tl.load(
                b_ptr + offsets_m[:, None] * K + offsets_k[None, :],
                mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K),
                other=0.0,
            )
            a = tl.load(
                a_ptr + offsets_k[:, None] * N + offsets_n[None, :],
                mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N),
                other=0.0,
            )
            if COMPUTE_DTYPE == 2:
                accumulator += tl.dot(b, a, input_precision="ieee")
            else:
                accumulator += tl.dot(b, a)
    if PER_ROW:
        weight_scale = tl.load(
            scale_ptr + offsets_m,
            mask=offsets_m < M,
            other=1.0,
        )[:, None]
    else:
        weight_scale = tl.load(scale_ptr)
    base = tl.load(qdata_ptr + offsets, mask=mask, other=0.0)
    base = base.to(tl.float32) * weight_scale
    if COMPUTE_DTYPE == 0:
        base = base.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        base = base.to(tl.bfloat16)

    merged = base.to(tl.float32) + accumulator * strength
    if COMPUTE_DTYPE == 0:
        merged = merged.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        merged = merged.to(tl.bfloat16)
    tl.store(dense_ptr + offsets, merged, mask=mask)

    absolute = tl.where(mask, tl.abs(merged.to(tl.float32)), 0.0)
    if PER_ROW:
        row_max = tl.max(absolute, axis=1)
        tl.store(
            partial_max_ptr + offsets_m * num_pid_n + pid_n,
            row_max,
            mask=offsets_m < M,
        )
    else:
        tile_max = tl.max(tl.max(absolute, axis=1), axis=0)
        tl.store(partial_max_ptr + pid, tile_max)


@triton.jit
def _merge_group_kernel(
    qdata_ptr,
    scale_ptr,
    b_ptr,
    a_ptr,
    update_ptr,
    output_qdata_ptr,
    output_scale_ptr,
    strength,
    rounding_seed,
    M,
    N,
    K: tl.constexpr,
    DENSE_UPDATE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    FP8_LIMIT: tl.constexpr,
    E4M3: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    group = tl.program_id(axis=1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_in_group = tl.arange(0, BLOCK_N)
    offsets_n = group * GROUP_SIZE + offsets_in_group
    row_mask = offsets_m < M
    group_mask = offsets_in_group < GROUP_SIZE
    mask = row_mask[:, None] & group_mask[None, :]

    offsets = offsets_m[:, None] * N + offsets_n[None, :]
    if DENSE_UPDATE:
        accumulator = tl.load(
            update_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offsets_k = k_start + tl.arange(0, BLOCK_K)
            b = tl.load(
                b_ptr + offsets_m[:, None] * K + offsets_k[None, :],
                mask=row_mask[:, None] & (offsets_k[None, :] < K),
                other=0.0,
            )
            a = tl.load(
                a_ptr + offsets_k[:, None] * N + offsets_n[None, :],
                mask=(offsets_k[:, None] < K) & group_mask[None, :],
                other=0.0,
            )
            if COMPUTE_DTYPE == 2:
                accumulator += tl.dot(b, a, input_precision="ieee")
            else:
                accumulator += tl.dot(b, a)
    scale_offsets = offsets_m * NUM_GROUPS + group
    weight_scale = tl.load(
        scale_ptr + scale_offsets,
        mask=row_mask,
        other=1.0,
    )
    base = tl.load(qdata_ptr + offsets, mask=mask, other=0.0)
    base = base.to(tl.float32) * weight_scale[:, None]
    if COMPUTE_DTYPE == 0:
        base = base.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        base = base.to(tl.bfloat16)

    merged = base.to(tl.float32) + accumulator * strength
    if COMPUTE_DTYPE == 0:
        merged = merged.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        merged = merged.to(tl.bfloat16)

    absolute = tl.where(mask, tl.abs(merged.to(tl.float32)), 0.0)
    output_scale = tl.max(absolute, axis=1) / FP8_LIMIT
    if COMPUTE_DTYPE == 0:
        output_scale = output_scale.to(tl.float16).to(tl.float32)
    elif COMPUTE_DTYPE == 1:
        output_scale = output_scale.to(tl.bfloat16).to(tl.float32)
    output_scale = tl.where(
        output_scale == 0.0,
        1.1920928955078125e-07,
        output_scale,
    )
    tl.store(
        output_scale_ptr + scale_offsets,
        output_scale,
        mask=row_mask,
    )

    scaled = merged.to(tl.float32) / output_scale[:, None]
    scaled = tl.minimum(tl.maximum(scaled, -FP8_LIMIT), FP8_LIMIT)
    if STOCHASTIC:
        scaled = _stochastic_float8(
            scaled,
            rounding_seed,
            offsets,
            E4M3,
            FP8_LIMIT,
        )
    tl.store(output_qdata_ptr + offsets, scaled, mask=mask)


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
def _reduce_tensor_scale_kernel(
    partial_max_ptr,
    output_scale_ptr,
    NUM_VALUES,
    FP8_LIMIT: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        partial_max_ptr + offsets,
        mask=offsets < NUM_VALUES,
        other=0.0,
    )
    max_abs = tl.max(values, axis=0)
    scale = max_abs / FP8_LIMIT
    if COMPUTE_DTYPE == 0:
        scale = scale.to(tl.float16).to(tl.float32)
    elif COMPUTE_DTYPE == 1:
        scale = scale.to(tl.bfloat16).to(tl.float32)
    scale = tl.where(scale == 0.0, 1.1920928955078125e-07, scale)
    tl.store(output_scale_ptr, scale)


@triton.jit
def _reduce_row_scale_kernel(
    partial_max_ptr,
    output_scale_ptr,
    NUM_TILES_N,
    FP8_LIMIT: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        partial_max_ptr + row * NUM_TILES_N + offsets,
        mask=offsets < NUM_TILES_N,
        other=0.0,
    )
    max_abs = tl.max(values, axis=0)
    scale = max_abs / FP8_LIMIT
    if COMPUTE_DTYPE == 0:
        scale = scale.to(tl.float16).to(tl.float32)
    elif COMPUTE_DTYPE == 1:
        scale = scale.to(tl.bfloat16).to(tl.float32)
    scale = tl.where(scale == 0.0, 1.1920928955078125e-07, scale)
    tl.store(output_scale_ptr + row, scale)


@triton.jit
def _quantize_kernel(
    dense_ptr,
    scale_ptr,
    output_ptr,
    rounding_seed,
    NUMEL,
    N,
    PER_ROW: tl.constexpr,
    FP8_LIMIT: tl.constexpr,
    E4M3: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL
    dense = tl.load(dense_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if PER_ROW:  # noqa: SIM108 - constexpr branches are pruned by Triton.
        scale = tl.load(scale_ptr + offsets // N, mask=mask, other=1.0)
    else:
        scale = tl.load(scale_ptr)
    scaled = dense / scale
    scaled = tl.minimum(
        tl.maximum(scaled, -FP8_LIMIT),
        FP8_LIMIT,
    )
    if STOCHASTIC:
        scaled = _stochastic_float8(
            scaled,
            rounding_seed,
            offsets,
            E4M3,
            FP8_LIMIT,
        )
    tl.store(output_ptr + offsets, scaled, mask=mask)


def _compute_dtype_id(dtype: torch.dtype) -> int:
    if dtype is torch.float16:
        return _COMPUTE_FP16
    if dtype is torch.bfloat16:
        return _COMPUTE_BF16
    if dtype is torch.float32:
        return _COMPUTE_FP32
    raise ValueError(f"Triton scaled-FP8 merge supports float16, bfloat16, and float32 LoRA factors, got {dtype}.")


def _validate_inputs(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, ...],
    b: torch.Tensor,
    a: torch.Tensor,
    update: torch.Tensor,
    *,
    dense_update: bool,
) -> tuple[int, int, int, int, bool, int | None]:
    """Validate a scaled-FP8 launch and return its normalized dimensions."""
    if qdata.device.type != "cuda":
        raise ValueError("Triton scaled-FP8 merge requires CUDA tensors.")
    if qdata.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"Triton scaled-FP8 merge supports E4M3FN and E5M2 storage, got {qdata.dtype}.")
    compute_tensor = update if dense_update else b
    if not dense_update and b.dtype is not a.dtype:
        raise ValueError("Triton scaled-FP8 merge requires matching LoRA factor dtypes.")
    compute_dtype = _compute_dtype_id(compute_tensor.dtype)
    if qdata.ndim != 2 or update.ndim != 2 or (not dense_update and (b.ndim != 2 or a.ndim != 2)):
        raise ValueError("Triton scaled-FP8 merge expects rank-two tensors.")
    operands = (update,) if dense_update else (b, a)
    if qdata.device != scale.device or any(qdata.device != operand.device for operand in operands):
        raise ValueError("Triton scaled-FP8 merge requires all tensors on one CUDA device.")
    if scale.dtype is not torch.float32:
        raise ValueError("Triton scaled-FP8 merge expects float32 scales.")

    rows, cols = qdata.shape
    rank = 1 if dense_update else a.shape[0]
    per_row = block_size == (1, cols)
    per_tensor = block_size == (rows, cols)
    group_size = (
        block_size[1]
        if (len(block_size) == 2 and block_size[0] == 1 and 0 < block_size[1] < cols and cols % block_size[1] == 0)
        else None
    )
    if per_row:
        if tuple(scale.shape) != (rows, 1):
            raise ValueError(
                f"Triton per-row scaled-FP8 merge expects scale shape {(rows, 1)}, got {tuple(scale.shape)}."
            )
    elif per_tensor:
        if scale.numel() != 1:
            raise ValueError("Triton per-tensor scaled-FP8 merge expects one scale.")
    elif group_size is not None:
        if group_size > MAX_GROUP_SIZE:
            raise ValueError(
                f"Triton per-group scaled-FP8 merge supports group sizes up to {MAX_GROUP_SIZE}, got {group_size}."
            )
        expected_scale_shape = (rows, cols // group_size)
        if tuple(scale.shape) != expected_scale_shape:
            raise ValueError(
                "Triton per-group scaled-FP8 merge expects scale shape "
                f"{expected_scale_shape}, got {tuple(scale.shape)}."
            )
    else:
        raise ValueError(
            "Triton scaled-FP8 merge supports per-row, per-tensor, and "
            f"standard per-group layouts, got block_size={block_size!r}."
        )
    if rows == 0 or cols == 0 or rank == 0:
        raise ValueError("Triton scaled-FP8 merge requires non-empty weight and factors.")
    if dense_update and tuple(update.shape) != (rows, cols):
        raise ValueError("Dense update does not match the scaled-FP8 weight shape.")
    if not dense_update and (b.shape != (rows, rank) or a.shape[1] != cols):
        raise ValueError("LoRA factors do not match the scaled-FP8 weight shape.")
    return rows, cols, rank, compute_dtype, per_row, group_size


def _merge_group(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    update: torch.Tensor,
    strength: float,
    *,
    rows: int,
    cols: int,
    rank: int,
    dense_update: bool,
    group_size: int,
    compute_dtype: int,
    rounding_seed: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge and requantize independent row groups without a dense buffer."""
    output_qdata = torch.empty_like(qdata)
    output_scale = torch.empty_like(scale)
    block_m = 16
    block_n = max(16, int(triton.next_power_of_2(group_size)))
    block_k = 16 if rank <= 16 else 32
    num_pid_m = (rows + block_m - 1) // block_m
    num_groups = cols // group_size
    _merge_group_kernel[(num_pid_m, num_groups)](
        qdata,
        scale,
        b,
        a,
        update,
        output_qdata,
        output_scale,
        strength,
        _seed_argument(rounding_seed),
        M=rows,
        N=cols,
        K=rank,
        DENSE_UPDATE=dense_update,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        FP8_LIMIT=torch.finfo(qdata.dtype).max,
        E4M3=qdata.dtype is torch.float8_e4m3fn,
        STOCHASTIC=rounding_seed is not None,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output_qdata, output_scale


def _reduce_output_scale(
    partial_max: torch.Tensor,
    output_scale: torch.Tensor,
    *,
    rows: int,
    num_pid_n: int,
    num_tiles: int,
    per_row: bool,
    fp8_limit: float,
    compute_dtype: int,
) -> None:
    """Reduce merge maxima into the layout's new FP8 scale tensor."""
    if per_row:
        reduction_block = triton.next_power_of_2(num_pid_n)
        _reduce_row_scale_kernel[(rows,)](
            partial_max,
            output_scale,
            NUM_TILES_N=num_pid_n,
            FP8_LIMIT=fp8_limit,
            COMPUTE_DTYPE=compute_dtype,
            BLOCK_SIZE=reduction_block,
            num_warps=8,
        )
        return

    reduction_input = partial_max
    num_values = num_tiles
    while num_values > _REDUCTION_BLOCK:
        output_count = (num_values + _REDUCTION_BLOCK - 1) // _REDUCTION_BLOCK
        reduced = torch.empty(
            output_count,
            device=partial_max.device,
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

    _reduce_tensor_scale_kernel[(1,)](
        reduction_input,
        output_scale,
        NUM_VALUES=num_values,
        FP8_LIMIT=fp8_limit,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_SIZE=_REDUCTION_BLOCK,
        num_warps=8,
    )


def _merge_float8(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, ...],
    b: torch.Tensor,
    a: torch.Tensor,
    update: torch.Tensor,
    strength: float,
    *,
    dense_update: bool,
    rounding_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw scaled-FP8 buffers after one supported-layout merge."""
    rows, cols, rank, compute_dtype, per_row, group_size = _validate_inputs(
        qdata,
        scale,
        block_size,
        b,
        a,
        update,
        dense_update=dense_update,
    )

    qdata = qdata.contiguous()
    scale = scale.contiguous()
    if dense_update:
        update = update.contiguous()
        b = update
        a = update
    else:
        b = b.contiguous()
        a = a.contiguous()
        update = b

    if group_size is not None:
        return _merge_group(
            qdata,
            scale,
            b,
            a,
            update,
            strength,
            rows=rows,
            cols=cols,
            rank=rank,
            dense_update=dense_update,
            group_size=group_size,
            compute_dtype=compute_dtype,
            rounding_seed=rounding_seed,
        )

    block_m = 64
    block_n = 128
    block_k = 16 if rank <= 16 else 32
    num_pid_m = (rows + block_m - 1) // block_m
    num_pid_n = (cols + block_n - 1) // block_n
    num_tiles = num_pid_m * num_pid_n
    num_partial_max = rows * num_pid_n if per_row else num_tiles
    dense = torch.empty_like(qdata, dtype=update.dtype if dense_update else b.dtype)
    partial_max = torch.empty(
        num_partial_max,
        device=qdata.device,
        dtype=torch.float32,
    )
    output_qdata = torch.empty_like(qdata)
    output_scale = torch.empty_like(scale)

    _merge_update_kernel[(num_tiles,)](
        qdata,
        scale,
        b,
        a,
        update,
        dense,
        partial_max,
        strength,
        M=rows,
        N=cols,
        K=rank,
        DENSE_UPDATE=dense_update,
        PER_ROW=per_row,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=8,
    )

    fp8_limit = torch.finfo(qdata.dtype).max
    _reduce_output_scale(
        partial_max,
        output_scale,
        rows=rows,
        num_pid_n=num_pid_n,
        num_tiles=num_tiles,
        per_row=per_row,
        fp8_limit=fp8_limit,
        compute_dtype=compute_dtype,
    )

    quant_block = 1024
    quant_grid = (qdata.numel() + quant_block - 1) // quant_block
    _quantize_kernel[(quant_grid,)](
        dense,
        output_scale,
        output_qdata,
        _seed_argument(rounding_seed),
        NUMEL=qdata.numel(),
        N=cols,
        PER_ROW=per_row,
        FP8_LIMIT=fp8_limit,
        E4M3=qdata.dtype is torch.float8_e4m3fn,
        STOCHASTIC=rounding_seed is not None,
        BLOCK_SIZE=quant_block,
        num_warps=8,
    )
    return output_qdata, output_scale


def merge_float8_lora(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, ...],
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    rounding_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw scaled-FP8 buffers after one supported-layout LoRA merge."""
    return _merge_float8(
        qdata,
        scale,
        block_size,
        b,
        a,
        b,
        strength,
        dense_update=False,
        rounding_seed=rounding_seed,
    )


def merge_float8_dense(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, ...],
    update: torch.Tensor,
    strength: float,
    *,
    rounding_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw scaled-FP8 buffers after one supported-layout dense merge."""
    return _merge_float8(
        qdata,
        scale,
        block_size,
        update,
        update,
        update,
        strength,
        dense_update=True,
        rounding_seed=rounding_seed,
    )
