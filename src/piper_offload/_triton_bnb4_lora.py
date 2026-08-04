"""Triton kernels for bitsandbytes blockwise 4-bit LoRA merges."""

# Triton JIT kernel signatures intentionally use untyped pointer parameters
# and upper-case constexpr names.
# ruff: noqa: ANN001, ANN202, N803, PLR0912, PLR0913, PLR0915
# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl

_COMPUTE_FP16 = 0
_COMPUTE_BF16 = 1
_COMPUTE_FP32 = 2
_QUANT_NF4 = 0
_QUANT_FP4 = 1
_QUANT_BLOCK = 64
_NESTED_BLOCK = 256
_REDUCTION_BLOCK = 8192


@triton.jit
def _nearest_sorted_code(
    values,
    code_ptr,
    MAX_INDEX: tl.constexpr,
    STEPS: tl.constexpr,
):
    """Return nearest indices in a monotonically increasing codebook."""
    low = tl.zeros(values.shape, dtype=tl.int32)
    high = low + MAX_INDEX
    for _ in range(STEPS):
        middle = (low + high) // 2
        middle_value = tl.load(code_ptr + middle)
        move_right = middle_value < values
        low = tl.where(move_right, middle + 1, low)
        high = tl.where(move_right, high, middle)

    upper_index = low
    lower_index = tl.maximum(upper_index - 1, 0)
    lower = tl.load(code_ptr + lower_index)
    upper = tl.load(code_ptr + upper_index)
    choose_lower = tl.abs(values - lower) <= tl.abs(values - upper)
    return tl.where(choose_lower, lower_index, upper_index)


@triton.jit
def _nearest_fp4_code(values):
    """Encode the fixed bitsandbytes FP4 magnitude/sign code layout."""
    magnitude = tl.abs(values)
    level = tl.zeros(values.shape, dtype=tl.int32)
    level += magnitude > 0.0026041667442768812
    level += magnitude > 0.08593750256113708
    level += magnitude > 0.2083333358168602
    level += magnitude > 0.2916666716337204
    level += magnitude > 0.4166666716337204
    level += magnitude > 0.5833333432674408
    level += magnitude > 0.8333333432674408

    code = tl.where(
        level < 2,
        level,
        tl.where(
            level < 4,
            level + 4,
            tl.where(level < 6, level, level - 4),
        ),
    )
    return tl.where((values < 0.0) & (level != 0), code + 8, code)


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
    M,
    N,
    K: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    NESTED: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    NESTED_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * QUANT_BLOCK + tl.arange(0, QUANT_BLOCK)
    row_mask = offsets_m < M

    accumulator = tl.zeros((BLOCK_M, QUANT_BLOCK), dtype=tl.float32)
    for rank_start in range(0, K, BLOCK_K):
        offsets_k = rank_start + tl.arange(0, BLOCK_K)
        b = tl.load(
            b_ptr + offsets_m[:, None] * K + offsets_k[None, :],
            mask=row_mask[:, None] & (offsets_k[None, :] < K),
            other=0.0,
        )
        a = tl.load(
            a_ptr + offsets_k[:, None] * N + offsets_n[None, :],
            mask=offsets_k[:, None] < K,
            other=0.0,
        )
        if COMPUTE_DTYPE == 2:
            accumulator += tl.dot(b, a, input_precision="ieee")
        else:
            accumulator += tl.dot(b, a)

    packed_cols = pid_n * (QUANT_BLOCK // 2) + tl.arange(
        0,
        QUANT_BLOCK // 2,
    )
    packed_offsets = offsets_m[:, None] * (N // 2) + packed_cols[None, :]
    packed = tl.load(
        packed_ptr + packed_offsets,
        mask=row_mask[:, None],
        other=0,
    ).to(tl.int32)
    code_indices = tl.interleave(packed >> 4, packed & 0xF)
    base = tl.load(code_ptr + code_indices)

    blocks_per_row = N // QUANT_BLOCK
    scale_offsets = offsets_m * blocks_per_row + pid_n
    if NESTED:
        scale_codes = tl.load(
            absmax_ptr + scale_offsets,
            mask=row_mask,
            other=0,
        ).to(tl.int32)
        quantized_scales = tl.load(nested_code_ptr + scale_codes)
        nested_scales = tl.load(
            nested_absmax_ptr + scale_offsets // NESTED_BLOCK,
            mask=row_mask,
            other=0.0,
        )
        scales = quantized_scales * nested_scales + tl.load(offset_ptr)
    else:
        scales = tl.load(
            absmax_ptr + scale_offsets,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

    base *= scales[:, None]
    if COMPUTE_DTYPE == 0:
        base = base.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        base = base.to(tl.bfloat16)

    merged = base.to(tl.float32) + accumulator * strength
    if COMPUTE_DTYPE == 0:
        merged = merged.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        merged = merged.to(tl.bfloat16)
    merged_f32 = merged.to(tl.float32)

    output_scale = tl.max(tl.abs(merged_f32), axis=1)
    safe_scale = tl.where(output_scale == 0.0, 1.0, output_scale)
    normalized = tl.minimum(
        tl.maximum(merged_f32 / safe_scale[:, None], -1.0),
        1.0,
    )
    if QUANT_TYPE == 0:
        output_codes = _nearest_sorted_code(
            normalized,
            code_ptr,
            15,
            4,
        )
    else:
        output_codes = _nearest_fp4_code(normalized)

    high, low = tl.split(
        output_codes.reshape(
            BLOCK_M,
            QUANT_BLOCK // 2,
            2,
        )
    )
    output_packed = (high << 4) | low
    tl.store(
        output_packed_ptr + packed_offsets,
        output_packed,
        mask=row_mask[:, None],
    )
    tl.store(
        output_absmax_ptr + scale_offsets,
        output_scale,
        mask=row_mask,
    )


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
    quantized = _nearest_sorted_code(scaled, code_ptr, 255, 8)
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


def _quant_type_id(quant_type: str) -> int:
    if quant_type == "nf4":
        return _QUANT_NF4
    if quant_type == "fp4":
        return _QUANT_FP4
    raise ValueError("Triton BNB4 merge supports NF4 and FP4.")


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
    """Merge each row-aligned quantization block without a dense weight."""
    if packed.device.type != "cuda":
        raise ValueError("Triton BNB4 merge requires CUDA tensors.")
    if packed.dtype is not torch.uint8:
        raise ValueError("Triton BNB4 merge expects a raw uint8 packed view.")
    quant_type_id = _quant_type_id(quant_type)
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
    if cols % blocksize != 0:
        raise ValueError("Triton BNB4 merge requires row-aligned 64-value blocks.")
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
    block_m = 16
    block_k = 16 if rank <= 16 else 32
    grid = (
        triton.cdiv(rows, block_m),
        cols // _QUANT_BLOCK,
    )
    _merge_quantize_kernel[grid](
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
        M=rows,
        N=cols,
        K=rank,
        QUANT_TYPE=quant_type_id,
        NESTED=nested,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        QUANT_BLOCK=_QUANT_BLOCK,
        NESTED_BLOCK=_NESTED_BLOCK,
        num_warps=4,
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
