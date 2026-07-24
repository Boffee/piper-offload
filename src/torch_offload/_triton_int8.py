"""Triton kernels for TorchAO affine-INT8 LoRA merges."""

# Triton JIT kernel signatures intentionally use untyped pointer parameters
# and upper-case constexpr names.
# ruff: noqa: ANN001, ANN202, N803, PLR0912, PLR0913
# pyright: reportCallIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_COMPUTE_FP16 = 0
_COMPUTE_BF16 = 1
_COMPUTE_FP32 = 2
_STATS_BLOCK = 8192


@triton.jit
def _merge_dense_kernel(
    qdata_ptr,
    scale_ptr,
    zero_point_ptr,
    b_ptr,
    a_ptr,
    dense_ptr,
    strength,
    M,
    N,
    K: tl.constexpr,
    BLOCK_NUMEL: tl.constexpr,
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

    offsets = offsets_m[:, None] * N + offsets_n[None, :]
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    qparam_offsets = offsets // BLOCK_NUMEL
    qdata = tl.load(qdata_ptr + offsets, mask=mask, other=0).to(tl.float32)
    zero_point = tl.load(
        zero_point_ptr + qparam_offsets,
        mask=mask,
        other=0,
    ).to(tl.float32)
    scale = tl.load(scale_ptr + qparam_offsets, mask=mask, other=0.0)
    base = qdata - zero_point
    if COMPUTE_DTYPE == 0:
        base = base.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        base = base.to(tl.bfloat16)
    base = base * scale
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


@triton.jit
def _block_stats_kernel(
    dense_ptr,
    partial_min_ptr,
    partial_max_ptr,
    BLOCK_NUMEL: tl.constexpr,
    CHUNKS_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    qparam_id = pid // CHUNKS_PER_BLOCK
    chunk_id = pid % CHUNKS_PER_BLOCK
    block_offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = qparam_id * BLOCK_NUMEL + block_offsets
    values = tl.load(
        dense_ptr + offsets,
        mask=block_offsets < BLOCK_NUMEL,
        other=0.0,
    ).to(tl.float32)
    min_value = tl.min(tl.minimum(values, 0.0), axis=0)
    max_value = tl.max(tl.maximum(values, 0.0), axis=0)
    tl.store(partial_min_ptr + pid, min_value)
    tl.store(partial_max_ptr + pid, max_value)


@triton.jit
def _choose_qparams_kernel(
    partial_min_ptr,
    partial_max_ptr,
    output_scale_ptr,
    output_zero_point_ptr,
    CHUNKS_PER_BLOCK: tl.constexpr,
    ASYMMETRIC: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    qparam_id = tl.program_id(axis=0)
    chunk_offsets = tl.arange(0, BLOCK_SIZE)
    partial_offsets = qparam_id * CHUNKS_PER_BLOCK + chunk_offsets
    mask = chunk_offsets < CHUNKS_PER_BLOCK
    min_value = tl.min(
        tl.load(partial_min_ptr + partial_offsets, mask=mask, other=0.0),
        axis=0,
    )
    max_value = tl.max(
        tl.load(partial_max_ptr + partial_offsets, mask=mask, other=0.0),
        axis=0,
    )

    if COMPUTE_DTYPE == 0:
        min_value = min_value.to(tl.float16).to(tl.float32)
        max_value = max_value.to(tl.float16).to(tl.float32)
    elif COMPUTE_DTYPE == 1:
        min_value = min_value.to(tl.bfloat16).to(tl.float32)
        max_value = max_value.to(tl.bfloat16).to(tl.float32)

    if ASYMMETRIC:
        value_range = max_value - min_value
        if COMPUTE_DTYPE == 0:
            value_range = value_range.to(tl.float16).to(tl.float32)
        elif COMPUTE_DTYPE == 1:
            value_range = value_range.to(tl.bfloat16).to(tl.float32)
        scale = value_range / 255.0
    else:
        scale = tl.maximum(-min_value, max_value) / 127.5

    if COMPUTE_DTYPE == 0:
        scale = scale.to(tl.float16).to(tl.float32)
    elif COMPUTE_DTYPE == 1:
        scale = scale.to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(scale, 1.1920928955078125e-07)
    if COMPUTE_DTYPE == 0:
        scale = scale.to(tl.float16).to(tl.float32)
    elif COMPUTE_DTYPE == 1:
        scale = scale.to(tl.bfloat16).to(tl.float32)

    zero_point = 0
    if ASYMMETRIC:
        ratio = min_value / scale
        if COMPUTE_DTYPE == 0:
            ratio = ratio.to(tl.float16).to(tl.float32)
        elif COMPUTE_DTYPE == 1:
            ratio = ratio.to(tl.bfloat16).to(tl.float32)
        zero_point = -128 - libdevice.llrint(ratio)
        zero_point = tl.minimum(tl.maximum(zero_point, -128), 127)

    tl.store(output_scale_ptr + qparam_id, scale)
    tl.store(output_zero_point_ptr + qparam_id, zero_point)


@triton.jit
def _quantize_kernel(
    dense_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    NUMEL,
    BLOCK_NUMEL: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL
    qparam_offsets = offsets // BLOCK_NUMEL
    values = tl.load(dense_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + qparam_offsets, mask=mask, other=1.0).to(tl.float32)
    inverse_scale = 1.0 / scale
    if COMPUTE_DTYPE == 0:
        inverse_scale = inverse_scale.to(tl.float16).to(tl.float32)
    elif COMPUTE_DTYPE == 1:
        inverse_scale = inverse_scale.to(tl.bfloat16).to(tl.float32)
    scaled = values * inverse_scale
    if COMPUTE_DTYPE == 0:
        scaled = scaled.to(tl.float16).to(tl.float32)
    elif COMPUTE_DTYPE == 1:
        scaled = scaled.to(tl.bfloat16).to(tl.float32)
    quantized = libdevice.llrint(scaled)
    zero_point = tl.load(
        zero_point_ptr + qparam_offsets,
        mask=mask,
        other=0,
    )
    quantized += zero_point
    quantized = tl.minimum(tl.maximum(quantized, -128), 127)
    tl.store(output_ptr + offsets, quantized, mask=mask)


def _compute_dtype_id(dtype: torch.dtype) -> int:
    if dtype is torch.float16:
        return _COMPUTE_FP16
    if dtype is torch.bfloat16:
        return _COMPUTE_BF16
    if dtype is torch.float32:
        return _COMPUTE_FP32
    raise ValueError(f"Triton INT8 merge supports float16, bfloat16, and float32 LoRA factors, got {dtype}.")


def _block_numel(
    shape: tuple[int, int],
    block_size: tuple[int, int],
) -> int:
    rows, cols = shape
    block_rows, block_cols = block_size
    if block_size == shape:
        return rows * cols
    if block_rows == 1 and 0 < block_cols <= cols and cols % block_cols == 0:
        return block_cols
    raise ValueError(
        "Triton INT8 merge supports two-dimensional per-tensor, per-row, "
        f"and per-group layouts, got shape {shape} with block size {block_size}."
    )


def merge_int8_lora(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    block_size: tuple[int, int],
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    asymmetric: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw affine-INT8 buffers after one LoRA merge."""
    if qdata.device.type != "cuda":
        raise ValueError("Triton INT8 merge requires CUDA tensors.")
    if qdata.dtype is not torch.int8:
        raise ValueError(f"Triton INT8 merge requires int8 weight storage, got {qdata.dtype}.")
    if qdata.ndim != 2 or b.ndim != 2 or a.ndim != 2:
        raise ValueError("Triton INT8 merge expects rank-two tensors.")
    if b.dtype is not a.dtype or scale.dtype is not b.dtype:
        raise ValueError("Triton INT8 merge requires matching scale and LoRA factor dtypes.")
    compute_dtype = _compute_dtype_id(b.dtype)
    if (
        qdata.device != scale.device
        or qdata.device != b.device
        or qdata.device != a.device
        or (zero_point is not None and qdata.device != zero_point.device)
    ):
        raise ValueError("Triton INT8 merge requires all tensors on one CUDA device.")

    rows, cols = qdata.shape
    rank = a.shape[0]
    if rows == 0 or cols == 0 or rank == 0:
        raise ValueError("Triton INT8 merge requires non-empty weight and factors.")
    if b.shape != (rows, rank) or a.shape[1] != cols:
        raise ValueError("LoRA factors do not match the INT8 weight shape.")

    block_numel = _block_numel((rows, cols), block_size)
    num_qparams = qdata.numel() // block_numel
    if scale.numel() != num_qparams:
        raise ValueError("INT8 scale shape does not match the quantization block layout.")
    if zero_point is not None and (zero_point.dtype is not torch.int8 or zero_point.numel() != num_qparams):
        raise ValueError("INT8 zero-point storage does not match the quantization blocks.")

    qdata = qdata.contiguous()
    scale = scale.contiguous()
    zero_point = torch.zeros_like(scale, dtype=torch.int8) if zero_point is None else zero_point.contiguous()
    b = b.contiguous()
    a = a.contiguous()

    dense = torch.empty(qdata.shape, device=qdata.device, dtype=b.dtype)
    output_qdata = torch.empty(qdata.shape, device=qdata.device, dtype=torch.int8)
    output_scale = torch.empty(
        scale.shape,
        device=scale.device,
        dtype=scale.dtype,
    )
    output_zero_point = torch.empty(
        scale.shape,
        device=scale.device,
        dtype=torch.int8,
    )

    block_m = 64
    block_n = 128
    block_k = 16 if rank <= 16 else 32
    merge_grid = ((rows + block_m - 1) // block_m) * ((cols + block_n - 1) // block_n)
    _merge_dense_kernel[(merge_grid,)](
        qdata,
        scale,
        zero_point,
        b,
        a,
        dense,
        strength,
        M=rows,
        N=cols,
        K=rank,
        BLOCK_NUMEL=block_numel,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=8,
    )

    stats_block = int(min(_STATS_BLOCK, triton.next_power_of_2(block_numel)))
    chunks_per_block = (block_numel + stats_block - 1) // stats_block
    num_partials = num_qparams * chunks_per_block
    partial_min = torch.empty(
        num_partials,
        device=qdata.device,
        dtype=torch.float32,
    )
    partial_max = torch.empty_like(partial_min)
    _block_stats_kernel[(num_partials,)](
        dense,
        partial_min,
        partial_max,
        BLOCK_NUMEL=block_numel,
        CHUNKS_PER_BLOCK=chunks_per_block,
        BLOCK_SIZE=stats_block,
        num_warps=4 if stats_block <= 2048 else 8,
    )

    qparam_reduction_block = triton.next_power_of_2(chunks_per_block)
    _choose_qparams_kernel[(num_qparams,)](
        partial_min,
        partial_max,
        output_scale,
        output_zero_point,
        CHUNKS_PER_BLOCK=chunks_per_block,
        ASYMMETRIC=asymmetric,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_SIZE=qparam_reduction_block,
        num_warps=4 if qparam_reduction_block <= 2048 else 8,
    )

    quant_block = 1024
    quant_grid = (qdata.numel() + quant_block - 1) // quant_block
    _quantize_kernel[(quant_grid,)](
        dense,
        output_scale,
        output_zero_point,
        output_qdata,
        NUMEL=qdata.numel(),
        BLOCK_NUMEL=block_numel,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_SIZE=quant_block,
        num_warps=8,
    )
    return output_qdata, output_scale, output_zero_point
