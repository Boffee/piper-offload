"""Triton kernel for fixed-scale Quanto qint8 LoRA merges."""

# Triton JIT kernel signatures intentionally use untyped pointer parameters
# and upper-case constexpr names.
# ruff: noqa: ANN001, ANN202, N803, PLR0913
# pyright: reportCallIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_AXIS_TENSOR = 0
_AXIS_ROW = 1
_AXIS_COLUMN = 2
_COMPUTE_FP16 = 0
_COMPUTE_BF16 = 1
_COMPUTE_FP32 = 2


@triton.jit
def _merge_qint8_kernel(
    qdata_ptr,
    scale_ptr,
    b_ptr,
    a_ptr,
    output_ptr,
    strength,
    M,
    N,
    K: tl.constexpr,
    SCALE_AXIS: tl.constexpr,
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
    qdata = tl.load(qdata_ptr + offsets, mask=mask, other=0).to(tl.float32)
    if SCALE_AXIS == 1:
        scale = tl.load(
            scale_ptr + offsets_m,
            mask=offsets_m < M,
            other=1.0,
        )[:, None]
    elif SCALE_AXIS == 2:
        scale = tl.load(
            scale_ptr + offsets_n,
            mask=offsets_n < N,
            other=1.0,
        )[None, :]
    else:
        scale = tl.load(scale_ptr)

    base = qdata * scale.to(tl.float32)
    if COMPUTE_DTYPE == 0:
        base = base.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        base = base.to(tl.bfloat16)

    merged = base.to(tl.float32) + accumulator * strength
    if COMPUTE_DTYPE == 0:
        merged = merged.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        merged = merged.to(tl.bfloat16)

    scale_f32 = scale.to(tl.float32)
    safe_scale = tl.where(scale_f32 == 0.0, 1.0, scale_f32)
    scaled = merged.to(tl.float32) / safe_scale
    if COMPUTE_DTYPE == 0:
        scaled = scaled.to(tl.float16).to(tl.float32)
    elif COMPUTE_DTYPE == 1:
        scaled = scaled.to(tl.bfloat16).to(tl.float32)
    zero_scale_value = tl.where(
        merged.to(tl.float32) > 0.0,
        127.0,
        tl.where(merged.to(tl.float32) < 0.0, -128.0, 0.0),
    )
    scaled = tl.where(scale_f32 == 0.0, zero_scale_value, scaled)
    quantized = libdevice.rint(scaled)
    quantized = tl.minimum(tl.maximum(quantized, -128.0), 127.0)
    tl.store(output_ptr + offsets, quantized, mask=mask)


def _compute_dtype_id(dtype: torch.dtype) -> int:
    if dtype is torch.float16:
        return _COMPUTE_FP16
    if dtype is torch.bfloat16:
        return _COMPUTE_BF16
    if dtype is torch.float32:
        return _COMPUTE_FP32
    raise ValueError(
        "Triton Quanto qint8 merge supports float16, bfloat16, and float32 "
        f"scales and factors, got {dtype}."
    )


def _scale_axis_id(
    axis: int | None,
    scale: torch.Tensor,
    rows: int,
    cols: int,
) -> int:
    if axis is None and scale.numel() == 1:
        return _AXIS_TENSOR
    if axis == 0 and scale.numel() == rows:
        return _AXIS_ROW
    if axis in (-1, 1) and scale.numel() == cols:
        return _AXIS_COLUMN
    raise ValueError(
        "Triton Quanto qint8 merge expects a scalar scale, one scale per "
        "row for axis 0, or one scale per column for the last axis."
    )


def merge_quanto_qint8_lora(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    axis: int | None,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Return qint8 storage after one fixed-scale Quanto LoRA merge."""
    if qdata.device.type != "cuda":
        raise ValueError("Triton Quanto qint8 merge requires CUDA tensors.")
    if qdata.dtype is not torch.int8:
        raise ValueError("Triton Quanto qint8 merge expects int8 storage.")
    if scale.dtype is not b.dtype or b.dtype is not a.dtype:
        raise ValueError(
            "Triton Quanto qint8 merge requires matching scale and factor dtypes."
        )
    compute_dtype = _compute_dtype_id(b.dtype)
    if qdata.ndim != 2 or b.ndim != 2 or a.ndim != 2:
        raise ValueError("Triton Quanto qint8 merge expects rank-two tensors.")
    if (
        qdata.device != scale.device
        or qdata.device != b.device
        or qdata.device != a.device
    ):
        raise ValueError(
            "Triton Quanto qint8 merge requires all tensors on one CUDA device."
        )

    rows, cols = qdata.shape
    rank = a.shape[0]
    if rows == 0 or cols == 0 or rank == 0:
        raise ValueError(
            "Triton Quanto qint8 merge requires non-empty tensors."
        )
    if b.shape != (rows, rank) or a.shape[1] != cols:
        raise ValueError("LoRA factors do not match the Quanto weight shape.")
    scale_axis = _scale_axis_id(axis, scale, rows, cols)

    qdata = qdata.contiguous()
    scale = scale.contiguous()
    b = b.contiguous()
    a = a.contiguous()

    block_m = 16
    block_n = 128
    block_k = 16 if rank <= 16 else 32
    grid = (
        ((rows + block_m - 1) // block_m)
        * ((cols + block_n - 1) // block_n),
    )
    output = torch.empty_like(qdata)
    _merge_qint8_kernel[grid](
        qdata,
        scale,
        b,
        a,
        output,
        strength,
        M=rows,
        N=cols,
        K=rank,
        SCALE_AXIS=scale_axis,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output
