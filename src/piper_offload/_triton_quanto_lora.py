"""Triton kernels for absmax-requantized Quanto qint8/qfloat8 merges."""

# Triton JIT kernel signatures intentionally use untyped pointer parameters
# and upper-case constexpr names.
# ruff: noqa: ANN001, ANN202, N803, PLR0913
# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ._triton_stochastic_quantization import (
    _seed_argument,
    _stochastic_float8,
    _stochastic_round_to_int,
)

_AXIS_TENSOR = 0
_AXIS_ROW = 1
_AXIS_COLUMN = 2
_COMPUTE_FP16 = 0
_COMPUTE_BF16 = 1
_COMPUTE_FP32 = 2


@triton.jit
def _merge_update_kernel(
    qdata_ptr,
    scale_ptr,
    b_ptr,
    a_ptr,
    update_ptr,
    output_ptr,
    strength,
    M,
    N,
    K: tl.constexpr,
    DENSE_UPDATE: tl.constexpr,
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
    qdata = tl.load(qdata_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
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

    scale_f32 = scale.to(tl.float32)
    # Quanto's FP8 quantizer stores NaN qbytes for an exact-zero scale
    # (0 / 0). The represented block is nevertheless exactly zero; select it
    # explicitly so a later nonzero LoRA update can recover the block.
    base = tl.where(scale_f32 == 0.0, 0.0, qdata * scale_f32)
    if COMPUTE_DTYPE == 0:
        base = base.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        base = base.to(tl.bfloat16)

    merged = base.to(tl.float32) + accumulator * strength
    if COMPUTE_DTYPE == 0:
        merged = merged.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        merged = merged.to(tl.bfloat16)
    tl.store(output_ptr + offsets, merged, mask=mask)


@triton.jit
def _quantize_qbytes_kernel(
    dense_ptr,
    scale_ptr,
    output_ptr,
    rounding_seed,
    M,
    N,
    SCALE_AXIS: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    INTEGER_STORAGE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    E4M3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N
    offsets_m = offsets // N
    offsets_n = offsets % N
    merged = tl.load(dense_ptr + offsets, mask=mask, other=0.0)

    if SCALE_AXIS == 1:
        scale = tl.load(scale_ptr + offsets_m, mask=mask, other=1.0)
    elif SCALE_AXIS == 2:
        scale = tl.load(scale_ptr + offsets_n, mask=mask, other=1.0)
    else:
        scale = tl.load(scale_ptr)

    stochastic_scaled = merged.to(tl.float32) / scale.to(tl.float32)
    scaled = stochastic_scaled
    if COMPUTE_DTYPE == 0:
        scaled = scaled.to(tl.float16).to(tl.float32)
    elif COMPUTE_DTYPE == 1:
        scaled = scaled.to(tl.bfloat16).to(tl.float32)
    if INTEGER_STORAGE:
        deterministic = libdevice.rint(scaled)
        if STOCHASTIC:
            scaled = _stochastic_round_to_int(
                stochastic_scaled,
                deterministic,
                rounding_seed,
                offsets,
                QMIN,
                QMAX,
            )
        else:
            scaled = deterministic
    if STOCHASTIC and not INTEGER_STORAGE:
        stochastic_scaled = tl.minimum(
            tl.maximum(stochastic_scaled, QMIN),
            QMAX,
        )
        scaled = _stochastic_float8(
            stochastic_scaled,
            rounding_seed,
            offsets,
            E4M3,
            QMAX,
        )
    quantized = tl.minimum(tl.maximum(scaled, QMIN), QMAX)
    tl.store(output_ptr + offsets, quantized, mask=mask)


def _compute_dtype_id(dtype: torch.dtype) -> int:
    if dtype is torch.float16:
        return _COMPUTE_FP16
    if dtype is torch.bfloat16:
        return _COMPUTE_BF16
    if dtype is torch.float32:
        return _COMPUTE_FP32
    raise ValueError(
        "Triton Quanto qbytes merge supports float16, bfloat16, and float32 "
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
        "Triton Quanto qbytes merge expects a scalar scale, one scale per "
        "row for axis 0, or one scale per column for the last axis."
    )


def _absmax_scale(
    dense: torch.Tensor,
    scale: torch.Tensor,
    scale_axis: int,
    qmax: float,
) -> torch.Tensor:
    """Recompute Quanto's data-dependent scale without leaving CUDA."""
    if scale_axis == _AXIS_ROW:
        amax = dense.abs().amax(dim=1, keepdim=True)
    elif scale_axis == _AXIS_COLUMN:
        amax = dense.abs().amax(dim=0, keepdim=True)
    else:
        amax = dense.abs().amax()
    output = (amax / qmax).to(dtype=scale.dtype).reshape(scale.shape)
    eps = torch.finfo(torch.float32).eps
    return torch.where(output == 0, torch.full_like(output, eps), output)


def _merge_quanto_qbytes(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    axis: int | None,
    b: torch.Tensor,
    a: torch.Tensor,
    update: torch.Tensor,
    strength: float,
    *,
    dense_update: bool,
    integer_storage: bool,
    qmin: float,
    qmax: float,
    rounding_seed: int | None,
    e4m3: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return qbytes data and a fresh scale after a LoRA or dense merge."""
    if qdata.device.type != "cuda":
        raise ValueError("Triton Quanto qbytes merge requires CUDA tensors.")
    compute_tensor = update if dense_update else b
    if scale.dtype is not compute_tensor.dtype or (not dense_update and b.dtype is not a.dtype):
        raise ValueError(
            "Triton Quanto qbytes merge requires matching scale and factor dtypes."
        )
    compute_dtype = _compute_dtype_id(compute_tensor.dtype)
    if qdata.ndim != 2 or update.ndim != 2 or (not dense_update and (b.ndim != 2 or a.ndim != 2)):
        raise ValueError("Triton Quanto qbytes merge expects rank-two tensors.")
    operands = (update,) if dense_update else (b, a)
    if qdata.device != scale.device or any(qdata.device != operand.device for operand in operands):
        raise ValueError(
            "Triton Quanto qbytes merge requires all tensors on one CUDA device."
        )

    rows, cols = qdata.shape
    rank = 1 if dense_update else a.shape[0]
    if rows == 0 or cols == 0 or rank == 0:
        raise ValueError(
            "Triton Quanto qbytes merge requires non-empty tensors."
        )
    if dense_update and tuple(update.shape) != (rows, cols):
        raise ValueError("Dense update does not match the Quanto weight shape.")
    if not dense_update and (b.shape != (rows, rank) or a.shape[1] != cols):
        raise ValueError("LoRA factors do not match the Quanto weight shape.")
    scale_axis = _scale_axis_id(axis, scale, rows, cols)

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

    block_m = 16
    block_n = 128
    block_k = 16 if rank <= 16 else 32
    grid = (
        ((rows + block_m - 1) // block_m)
        * ((cols + block_n - 1) // block_n),
    )
    dense = torch.empty(qdata.shape, device=qdata.device, dtype=compute_tensor.dtype)
    _merge_update_kernel[grid](
        qdata,
        scale,
        b,
        a,
        update,
        dense,
        strength,
        M=rows,
        N=cols,
        K=rank,
        DENSE_UPDATE=dense_update,
        SCALE_AXIS=scale_axis,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    output_scale = _absmax_scale(dense, scale, scale_axis, qmax)
    output = torch.empty_like(qdata)
    quant_block = 1024
    quant_grid = (triton.cdiv(qdata.numel(), quant_block),)
    _quantize_qbytes_kernel[quant_grid](
        dense,
        output_scale,
        output,
        _seed_argument(rounding_seed),
        M=rows,
        N=cols,
        SCALE_AXIS=scale_axis,
        COMPUTE_DTYPE=compute_dtype,
        INTEGER_STORAGE=integer_storage,
        QMIN=qmin,
        QMAX=qmax,
        STOCHASTIC=rounding_seed is not None,
        E4M3=e4m3,
        BLOCK_SIZE=quant_block,
        num_warps=8,
    )
    return output, output_scale


def merge_quanto_qint8_lora(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    axis: int | None,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    rounding_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return qint8 storage and its recomputed absmax scale."""
    if qdata.dtype is not torch.int8:
        raise ValueError("Triton Quanto qint8 merge expects int8 storage.")
    return _merge_quanto_qbytes(
        qdata,
        scale,
        axis,
        b,
        a,
        b,
        strength,
        dense_update=False,
        integer_storage=True,
        qmin=-128.0,
        qmax=127.0,
        rounding_seed=rounding_seed,
        e4m3=False,
    )


def merge_quanto_qfloat8_lora(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    axis: int | None,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    rounding_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return qfloat8 storage and its recomputed absmax scale."""
    if qdata.dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        raise ValueError(
            "Triton Quanto qfloat8 merge expects E4M3FN or E5M2 storage, "
            f"got {qdata.dtype}."
        )
    limits = torch.finfo(qdata.dtype)
    return _merge_quanto_qbytes(
        qdata,
        scale,
        axis,
        b,
        a,
        b,
        strength,
        dense_update=False,
        integer_storage=False,
        qmin=limits.min,
        qmax=limits.max,
        rounding_seed=rounding_seed,
        e4m3=qdata.dtype is torch.float8_e4m3fn,
    )


def merge_quanto_qint8_dense(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    axis: int | None,
    update: torch.Tensor,
    strength: float,
    *,
    rounding_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return qint8 storage and its recomputed scale after a dense merge."""
    if qdata.dtype is not torch.int8:
        raise ValueError("Triton Quanto qint8 merge expects int8 storage.")
    return _merge_quanto_qbytes(
        qdata,
        scale,
        axis,
        update,
        update,
        update,
        strength,
        dense_update=True,
        integer_storage=True,
        qmin=-128.0,
        qmax=127.0,
        rounding_seed=rounding_seed,
        e4m3=False,
    )


def merge_quanto_qfloat8_dense(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    axis: int | None,
    update: torch.Tensor,
    strength: float,
    *,
    rounding_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return qfloat8 storage and its recomputed scale after a dense merge."""
    if qdata.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"Triton Quanto qfloat8 merge expects E4M3FN or E5M2 storage, got {qdata.dtype}.")
    limits = torch.finfo(qdata.dtype)
    return _merge_quanto_qbytes(
        qdata,
        scale,
        axis,
        update,
        update,
        update,
        strength,
        dense_update=True,
        integer_storage=False,
        qmin=limits.min,
        qmax=limits.max,
        rounding_seed=rounding_seed,
        e4m3=qdata.dtype is torch.float8_e4m3fn,
    )
