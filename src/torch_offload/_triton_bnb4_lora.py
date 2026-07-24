"""Triton kernels for bitsandbytes blockwise 4-bit LoRA merges."""

# Triton JIT kernel signatures intentionally use untyped pointer parameters
# and upper-case constexpr names.
# ruff: noqa: ANN001, ANN202, N803, PLR0912, PLR0913, PLR0915
# pyright: reportCallIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

_COMPUTE_FP16 = 0
_COMPUTE_BF16 = 1
_COMPUTE_FP32 = 2
_NESTED_BLOCK = 256
_REDUCTION_BLOCK = 8192


@triton.jit
def _block_scale(
    block_indices,
    absmax_ptr,
    nested_absmax_ptr,
    nested_code_ptr,
    offset_ptr,
    NESTED: tl.constexpr,
):
    if NESTED:
        scale_codes = tl.load(absmax_ptr + block_indices).to(tl.int32)
        quantized_scales = tl.load(nested_code_ptr + scale_codes)
        nested_scales = tl.load(
            nested_absmax_ptr + block_indices // 256,
        )
        return quantized_scales * nested_scales + tl.load(offset_ptr)
    return tl.load(absmax_ptr + block_indices).to(tl.float32)


@triton.jit
def _merged_values(
    indices,
    mask,
    packed_ptr,
    absmax_ptr,
    code_ptr,
    nested_absmax_ptr,
    nested_code_ptr,
    offset_ptr,
    b_ptr,
    a_ptr,
    strength,
    N,
    K: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    NESTED: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    packed = tl.load(
        packed_ptr + indices // 2,
        mask=mask,
        other=0,
    ).to(tl.int32)
    code_indices = tl.where(
        indices % 2 == 0,
        packed >> 4,
        packed & 0xF,
    )
    codes = tl.load(code_ptr + code_indices)
    scales = _block_scale(
        indices // QUANT_BLOCK,
        absmax_ptr,
        nested_absmax_ptr,
        nested_code_ptr,
        offset_ptr,
        NESTED,
    )
    base = codes * scales
    if COMPUTE_DTYPE == 0:
        base = base.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        base = base.to(tl.bfloat16)

    rows = indices // N
    cols = indices % N
    update = tl.zeros(indices.shape, dtype=tl.float32)
    for rank_index in range(K):
        b = tl.load(
            b_ptr + rows * K + rank_index,
            mask=mask,
            other=0.0,
        )
        a = tl.load(
            a_ptr + rank_index * N + cols,
            mask=mask,
            other=0.0,
        )
        update += b.to(tl.float32) * a.to(tl.float32)

    merged = base.to(tl.float32) + update * strength
    if COMPUTE_DTYPE == 0:
        merged = merged.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        merged = merged.to(tl.bfloat16)
    return merged


@triton.jit
def _merge_block_max_kernel(
    packed_ptr,
    absmax_ptr,
    code_ptr,
    nested_absmax_ptr,
    nested_code_ptr,
    offset_ptr,
    b_ptr,
    a_ptr,
    output_absmax_ptr,
    strength,
    NUMEL,
    N,
    K: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    NESTED: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    block = tl.program_id(axis=0)
    indices = block * QUANT_BLOCK + tl.arange(0, QUANT_BLOCK)
    mask = indices < NUMEL
    merged = _merged_values(
        indices,
        mask,
        packed_ptr,
        absmax_ptr,
        code_ptr,
        nested_absmax_ptr,
        nested_code_ptr,
        offset_ptr,
        b_ptr,
        a_ptr,
        strength,
        N,
        K,
        QUANT_BLOCK,
        NESTED,
        COMPUTE_DTYPE,
    )
    absolute = tl.where(mask, tl.abs(merged.to(tl.float32)), 0.0)
    tl.store(output_absmax_ptr + block, tl.max(absolute, axis=0))


@triton.jit
def _nearest_main_code(values, code_ptr):
    code_offsets = tl.arange(0, 16)
    codes = tl.load(code_ptr + code_offsets)
    differences = tl.abs(values[:, None] - codes[None, :])
    return tl.argmin(differences, axis=1)


@triton.jit
def _merge_quantize_kernel(
    packed_ptr,
    absmax_ptr,
    code_ptr,
    nested_absmax_ptr,
    nested_code_ptr,
    offset_ptr,
    b_ptr,
    a_ptr,
    output_absmax_ptr,
    output_packed_ptr,
    strength,
    NUMEL,
    N,
    K: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    NESTED: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    block = tl.program_id(axis=0)
    byte_offsets = (
        block * (QUANT_BLOCK // 2)
        + tl.arange(0, QUANT_BLOCK // 2)
    )
    even_indices = byte_offsets * 2
    odd_indices = even_indices + 1
    even_mask = even_indices < NUMEL
    odd_mask = odd_indices < NUMEL

    even = _merged_values(
        even_indices,
        even_mask,
        packed_ptr,
        absmax_ptr,
        code_ptr,
        nested_absmax_ptr,
        nested_code_ptr,
        offset_ptr,
        b_ptr,
        a_ptr,
        strength,
        N,
        K,
        QUANT_BLOCK,
        NESTED,
        COMPUTE_DTYPE,
    ).to(tl.float32)
    odd = _merged_values(
        odd_indices,
        odd_mask,
        packed_ptr,
        absmax_ptr,
        code_ptr,
        nested_absmax_ptr,
        nested_code_ptr,
        offset_ptr,
        b_ptr,
        a_ptr,
        strength,
        N,
        K,
        QUANT_BLOCK,
        NESTED,
        COMPUTE_DTYPE,
    ).to(tl.float32)

    scale = tl.load(output_absmax_ptr + block)
    safe_scale = tl.where(scale == 0.0, 1.0, scale)
    even = tl.minimum(tl.maximum(even / safe_scale, -1.0), 1.0)
    odd = tl.minimum(tl.maximum(odd / safe_scale, -1.0), 1.0)
    even_codes = _nearest_main_code(even, code_ptr)
    odd_codes = _nearest_main_code(odd, code_ptr)
    packed = (even_codes << 4) | odd_codes
    tl.store(output_packed_ptr + byte_offsets, packed, mask=even_mask)


@triton.jit
def _sum_chunks_kernel(
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
    tl.store(output_ptr + tl.program_id(axis=0), tl.sum(values, axis=0))


@triton.jit
def _mean_kernel(
    input_ptr,
    output_ptr,
    NUM_VALUES,
    TOTAL_VALUES,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        input_ptr + offsets,
        mask=offsets < NUM_VALUES,
        other=0.0,
    )
    tl.store(output_ptr, tl.sum(values, axis=0) / TOTAL_VALUES)


@triton.jit
def _nearest_nested_code(values, code_ptr):
    code_offsets = tl.arange(0, 256)
    codes = tl.load(code_ptr + code_offsets)
    differences = tl.abs(values[:, None] - codes[None, :])
    return tl.argmin(differences, axis=1)


@triton.jit
def _quantize_nested_scales_kernel(
    absmax_ptr,
    offset_ptr,
    code_ptr,
    output_qabsmax_ptr,
    output_nested_absmax_ptr,
    NUM_VALUES,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(axis=0)
    indices = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < NUM_VALUES
    centered = (
        tl.load(absmax_ptr + indices, mask=mask, other=0.0)
        - tl.load(offset_ptr)
    )
    absolute = tl.where(mask, tl.abs(centered), 0.0)
    nested_absmax = tl.max(absolute, axis=0)
    safe_absmax = tl.where(nested_absmax == 0.0, 1.0, nested_absmax)
    scaled = tl.minimum(
        tl.maximum(centered / safe_absmax, -1.0),
        1.0,
    )
    quantized = _nearest_nested_code(scaled, code_ptr)
    tl.store(output_qabsmax_ptr + indices, quantized, mask=mask)
    tl.store(output_nested_absmax_ptr + block, nested_absmax)


def _compute_dtype_id(dtype: torch.dtype) -> int:
    if dtype is torch.float16:
        return _COMPUTE_FP16
    if dtype is torch.bfloat16:
        return _COMPUTE_BF16
    if dtype is torch.float32:
        return _COMPUTE_FP32
    raise ValueError(
        "Triton BNB4 merge supports float16, bfloat16, and float32 "
        f"compute dtypes, got {dtype}."
    )


def _mean_absmax(absmax: torch.Tensor) -> torch.Tensor:
    reduction_inputs = [absmax]
    num_values = absmax.numel()
    while num_values > _REDUCTION_BLOCK:
        output_count = (
            num_values + _REDUCTION_BLOCK - 1
        ) // _REDUCTION_BLOCK
        reduced = torch.empty(
            output_count,
            device=absmax.device,
            dtype=torch.float32,
        )
        _sum_chunks_kernel[(output_count,)](
            reduction_inputs[-1],
            reduced,
            NUM_VALUES=num_values,
            BLOCK_SIZE=_REDUCTION_BLOCK,
            num_warps=8,
        )
        reduction_inputs.append(reduced)
        num_values = output_count

    output = torch.empty((), device=absmax.device, dtype=torch.float32)
    reduction_block = 1 << (num_values - 1).bit_length()
    _mean_kernel[(1,)](
        reduction_inputs[-1],
        output,
        NUM_VALUES=num_values,
        TOTAL_VALUES=absmax.numel(),
        BLOCK_SIZE=reduction_block,
        num_warps=8,
    )
    return output


def merge_bnb4_lora(
    packed: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    nested_absmax: torch.Tensor | None,
    nested_code: torch.Tensor | None,
    offset: torch.Tensor | None,
    logical_shape: tuple[int, int],
    blocksize: int,
    quant_type: str,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Merge by recomputing each block twice, without a dense weight buffer."""
    if packed.device.type != "cuda":
        raise ValueError("Triton BNB4 merge requires CUDA tensors.")
    if packed.dtype is not torch.uint8:
        raise ValueError("Triton BNB4 merge expects a raw uint8 packed view.")
    if quant_type not in ("nf4", "fp4"):
        raise ValueError("Triton BNB4 merge supports NF4 and FP4.")
    if code.dtype is not torch.float32 or code.numel() != 16:
        raise ValueError("Triton BNB4 merge expects a 16-value float32 codebook.")
    if b.dtype is not a.dtype:
        raise ValueError("Triton BNB4 merge requires matching factor dtypes.")
    compute_dtype = _compute_dtype_id(b.dtype)
    if b.ndim != 2 or a.ndim != 2:
        raise ValueError("Triton BNB4 merge expects rank-two factors.")

    rows, cols = logical_shape
    rank = a.shape[0]
    numel = rows * cols
    num_blocks = (numel + blocksize - 1) // blocksize
    nested = nested_absmax is not None
    if rows == 0 or cols == 0 or rank == 0:
        raise ValueError("Triton BNB4 merge requires non-empty tensors.")
    if numel % 2 != 0 or packed.numel() != numel // 2:
        raise ValueError("Triton BNB4 merge requires exact two-values-per-byte storage.")
    if blocksize != 64:
        raise ValueError("Triton BNB4 merge expects the standard 64-value blocks.")
    if b.shape != (rows, rank) or a.shape[1] != cols:
        raise ValueError("LoRA factors do not match the BNB4 weight shape.")
    if absmax.numel() != num_blocks:
        raise ValueError("BNB4 absmax shape does not match the logical weight.")
    if (
        packed.device != absmax.device
        or packed.device != code.device
        or packed.device != b.device
        or packed.device != a.device
    ):
        raise ValueError(
            "Triton BNB4 merge requires all tensors on one CUDA device."
        )

    if nested:
        if nested_code is None or offset is None:
            raise ValueError("Nested BNB4 merge requires its codebook and offset.")
        expected_nested_blocks = (
            num_blocks + _NESTED_BLOCK - 1
        ) // _NESTED_BLOCK
        if (
            absmax.dtype is not torch.uint8
            or nested_absmax.dtype is not torch.float32
            or nested_absmax.numel() != expected_nested_blocks
            or nested_code.dtype is not torch.float32
            or nested_code.numel() != 256
            or offset.dtype is not torch.float32
            or offset.numel() != 1
        ):
            raise ValueError("Nested BNB4 scale metadata has an unsupported layout.")
        if (
            packed.device != nested_absmax.device
            or packed.device != nested_code.device
            or packed.device != offset.device
        ):
            raise ValueError(
                "Triton BNB4 merge requires nested metadata on the weight device."
            )
    elif absmax.dtype is not torch.float32:
        raise ValueError("Non-nested BNB4 merge expects float32 absmax values.")

    packed = packed.contiguous()
    absmax = absmax.contiguous()
    code = code.contiguous()
    b = b.contiguous()
    a = a.contiguous()
    nested_absmax_input = (
        nested_absmax.contiguous() if nested_absmax is not None else absmax
    )
    nested_code_input = (
        nested_code.contiguous() if nested_code is not None else code
    )
    offset_input = offset.contiguous() if offset is not None else absmax

    raw_absmax = torch.empty(
        num_blocks,
        device=packed.device,
        dtype=torch.float32,
    )
    output_packed = torch.empty_like(packed)
    _merge_block_max_kernel[(num_blocks,)](
        packed,
        absmax,
        code,
        nested_absmax_input,
        nested_code_input,
        offset_input,
        b,
        a,
        raw_absmax,
        strength,
        NUMEL=numel,
        N=cols,
        K=rank,
        QUANT_BLOCK=blocksize,
        NESTED=nested,
        COMPUTE_DTYPE=compute_dtype,
        num_warps=8,
    )
    _merge_quantize_kernel[(num_blocks,)](
        packed,
        absmax,
        code,
        nested_absmax_input,
        nested_code_input,
        offset_input,
        b,
        a,
        raw_absmax,
        output_packed,
        strength,
        NUMEL=numel,
        N=cols,
        K=rank,
        QUANT_BLOCK=blocksize,
        NESTED=nested,
        COMPUTE_DTYPE=compute_dtype,
        num_warps=8,
    )

    if not nested:
        return output_packed, raw_absmax, None, None

    assert nested_absmax is not None
    assert nested_code is not None
    output_offset = _mean_absmax(raw_absmax)
    output_qabsmax = torch.empty_like(absmax)
    output_nested_absmax = torch.empty_like(nested_absmax)
    _quantize_nested_scales_kernel[(output_nested_absmax.numel(),)](
        raw_absmax,
        output_offset,
        nested_code,
        output_qabsmax,
        output_nested_absmax,
        NUM_VALUES=num_blocks,
        BLOCK_SIZE=_NESTED_BLOCK,
        num_warps=8,
    )
    return (
        output_packed,
        output_qabsmax,
        output_nested_absmax,
        output_offset,
    )
