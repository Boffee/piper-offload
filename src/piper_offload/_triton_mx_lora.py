"""Triton kernel for TorchAO MXFP8 / MXFP4 LoRA merges."""

# Triton JIT kernel signatures intentionally use untyped pointer parameters
# and upper-case constexpr names.
# ruff: noqa: ANN001, ANN202, N803, PLR0124, PLR0912, PLR0913
# pyright: reportCallIssue=false

import torch
import triton
import triton.language as tl

_COMPUTE_BF16 = 0
_COMPUTE_FP32 = 1
_FORMAT_E4M3 = 0
_FORMAT_E5M2 = 1
_FORMAT_E2M1 = 2
_SCALE_FLOOR = 0
_SCALE_RCEIL = 1
_SCALE_CEIL = 2
_SCALE_EVEN = 3
_MX_BLOCK_SIZE = 32


@triton.jit
def _e8m0_to_fp32(scale):
    # E8M0 code 0 is 2**-127, a valid FP32 subnormal. ``tl.exp2(-127)``
    # flushes it to zero on CUDA, so construct the exact IEEE-754 bits.
    bits = scale.to(tl.int32) << 23
    bits = tl.where(scale == 0, 1 << 22, bits)
    value = bits.to(tl.float32, bitcast=True)
    return tl.where(scale != 255, value, float("nan"))


@triton.jit
def _decode_fp4(code):
    magnitude_code = code & 7
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
                        tl.where(magnitude_code == 5, 3.0, tl.where(magnitude_code == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where((code & 8) != 0, -magnitude, magnitude)


@triton.jit
def _encode_fp4(value):
    value = value.to(tl.float32)
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
                        tl.where(magnitude < 3.5, 5, tl.where(magnitude <= 5.0, 6, 7)),
                    ),
                ),
            ),
        ),
    )
    bits = value.to(tl.int32, bitcast=True)
    sign = tl.where(bits < 0, 8, 0)
    return (magnitude_code | sign).to(tl.uint8)


@triton.jit
def _scale_offset(
    rows,
    scale_block,
    NUM_SCALE_BLOCKS,
    SWIZZLED: tl.constexpr,
):
    if SWIZZLED:
        row_block = rows // 128
        row_in_block = rows % 128
        scale_col_block = scale_block // 4
        scale_col_in_block = scale_block % 4
        num_scale_col_blocks = tl.cdiv(NUM_SCALE_BLOCKS, 4)
        tile_offset = (row_block * num_scale_col_blocks + scale_col_block) * 512
        return tile_offset + (row_in_block % 32) * 16 + (row_in_block // 32) * 4 + scale_col_in_block
    return rows * NUM_SCALE_BLOCKS + scale_block


@triton.jit
def _calculate_scale(
    max_abs,
    TARGET_MAX_POW2: tl.constexpr,
    MAX_POS: tl.constexpr,
    MBITS: tl.constexpr,
    SCALING_MODE: tl.constexpr,
):
    if SCALING_MODE == 1:
        descale = max_abs.to(tl.float32) / MAX_POS
        unbiased = tl.ceil(tl.log2(descale))
        unbiased = tl.minimum(tl.maximum(unbiased, -127.0), 127.0)
        encoded = (unbiased + 127.0).to(tl.uint8)
        encoded = tl.where(descale != descale, 255, encoded)
        inverse_scale = tl.where(
            encoded == 0,
            1.0,
            tl.exp2(-unbiased.to(tl.float32)),
        )
        return encoded, inverse_scale

    max_abs_f32 = max_abs.to(tl.float32)
    max_abs_bits = max_abs_f32.to(tl.int32, bitcast=True)
    if SCALING_MODE == 3:
        if MBITS == 3:
            rounding_bias = 524288
        elif MBITS == 2:
            rounding_bias = 1048576
        else:
            rounding_bias = 2097152
        max_abs_bits = (max_abs_bits + rounding_bias) & -8388608

    extracted_pow2 = ((max_abs_bits >> 23) & 255) - 127
    if SCALING_MODE == 2:
        extracted_pow2 += (max_abs_bits & 8388607) > 0

    unbiased = extracted_pow2 - TARGET_MAX_POW2
    unbiased = tl.minimum(tl.maximum(unbiased, -127), 128)
    encoded = (unbiased + 127).to(tl.uint8)
    encoded = tl.where(max_abs_f32 != max_abs_f32, 255, encoded)

    scale_bits = encoded.to(tl.int32) << 23
    scale = scale_bits.to(tl.float32, bitcast=True)
    scale = tl.maximum(scale, 1.1754943508222875e-38)
    return encoded, 1.0 / scale


@triton.jit
def _merge_mx_kernel(
    qdata_ptr,
    scale_ptr,
    b_ptr,
    a_ptr,
    strength,
    M,
    N,
    NUM_SCALE_BLOCKS,
    K: tl.constexpr,
    FORMAT: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    SCALING_MODE: tl.constexpr,
    SWIZZLED: tl.constexpr,
    TARGET_MAX_POW2: tl.constexpr,
    MAX_POS: tl.constexpr,
    MBITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * 32 + tl.arange(0, 32)
    row_mask = offsets_m < M

    accumulator = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
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
        if COMPUTE_DTYPE == 1:
            accumulator += tl.dot(b, a, input_precision="ieee")
        else:
            accumulator += tl.dot(b, a)

    if FORMAT == 2:
        packed_offsets_n = pid_n * 16 + tl.arange(0, 16)
        packed_offsets = offsets_m[:, None] * (N // 2) + packed_offsets_n[None, :]
        packed = tl.load(
            qdata_ptr + packed_offsets,
            mask=row_mask[:, None],
            other=0,
        )
        base_lp = tl.interleave(packed & 15, packed >> 4)
        base_lp = _decode_fp4(base_lp)
    else:
        qdata_offsets = offsets_m[:, None] * N + offsets_n[None, :]
        base_lp = tl.load(
            qdata_ptr + qdata_offsets,
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)

    scale_offsets = _scale_offset(
        offsets_m,
        pid_n,
        NUM_SCALE_BLOCKS,
        SWIZZLED,
    )
    encoded_scale = tl.load(
        scale_ptr + scale_offsets,
        mask=row_mask,
        other=127,
    )
    base_scale = _e8m0_to_fp32(encoded_scale)
    if COMPUTE_DTYPE == 0:
        base = (base_lp.to(tl.bfloat16) * base_scale[:, None].to(tl.bfloat16)).to(tl.bfloat16)
    else:
        base = base_lp.to(tl.float32) * base_scale[:, None]

    merged = base.to(tl.float32) + accumulator * strength
    if COMPUTE_DTYPE == 0:
        merged = merged.to(tl.bfloat16)

    max_abs = tl.max(tl.abs(merged.to(tl.float32)), axis=1)
    output_scale, inverse_scale = _calculate_scale(
        max_abs,
        TARGET_MAX_POW2,
        MAX_POS,
        MBITS,
        SCALING_MODE,
    )
    normalized = merged.to(tl.float32) * inverse_scale[:, None]
    normalized = tl.minimum(tl.maximum(normalized, -MAX_POS), MAX_POS)

    tl.store(
        scale_ptr + scale_offsets,
        output_scale,
        mask=row_mask,
    )
    if FORMAT == 2:
        codes = _encode_fp4(normalized)
        low, high = tl.split(codes.reshape(BLOCK_M, 16, 2))
        packed = low | (high << 4)
        tl.store(
            qdata_ptr + packed_offsets,
            packed,
            mask=row_mask[:, None],
        )
    else:
        tl.store(
            qdata_ptr + qdata_offsets,
            normalized,
            mask=row_mask[:, None],
        )


def _format_parameters(
    elem_dtype: torch.dtype,
) -> tuple[int, int, float, int]:
    if elem_dtype is torch.float8_e4m3fn:
        return _FORMAT_E4M3, 8, 448.0, 3
    if elem_dtype is torch.float8_e5m2:
        return _FORMAT_E5M2, 15, 57344.0, 2
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if elem_dtype is fp4_dtype and fp4_dtype is not None:
        return _FORMAT_E2M1, 2, 6.0, 1
    raise ValueError(f"Triton MX merge does not support element dtype {elem_dtype}.")


def _compute_dtype_id(dtype: torch.dtype) -> int:
    if dtype is torch.bfloat16:
        return _COMPUTE_BF16
    if dtype is torch.float32:
        return _COMPUTE_FP32
    raise ValueError(f"Triton MX merge supports bfloat16 and float32 weights and factors, got {dtype}.")


def _expected_scale_shape(
    rows: int,
    cols: int,
    *,
    swizzled: bool,
) -> tuple[int, int]:
    if not swizzled:
        return rows, cols // _MX_BLOCK_SIZE
    return (
        ((rows + 127) // 128) * 32,
        ((cols + 127) // 128) * 16,
    )


def merge_mx_lora_(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    elem_dtype: torch.dtype,
    block_size: int,
    orig_dtype: torch.dtype,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    scaling_mode: int,
    swizzled: bool,
) -> None:
    """Merge one staged LoRA update directly into raw MX storage."""
    if qdata.device.type != "cuda":
        raise ValueError("Triton MX merge requires CUDA tensors.")
    if qdata.ndim != 2 or scale.ndim != 2 or b.ndim != 2 or a.ndim != 2:
        raise ValueError("Triton MX merge expects rank-two tensors.")
    if block_size != _MX_BLOCK_SIZE:
        raise ValueError("Triton MX merge supports block size 32.")
    if b.dtype is not a.dtype or b.dtype is not orig_dtype:
        raise ValueError("Triton MX merge requires factors in the weight's original dtype.")
    compute_dtype = _compute_dtype_id(orig_dtype)
    if qdata.device != scale.device or qdata.device != b.device or qdata.device != a.device:
        raise ValueError("Triton MX merge requires all tensors on one CUDA device.")
    if not qdata.is_contiguous() or not scale.is_contiguous():
        raise ValueError("Triton MX merge requires contiguous MX storage.")
    if scaling_mode not in (
        _SCALE_FLOOR,
        _SCALE_RCEIL,
        _SCALE_CEIL,
        _SCALE_EVEN,
    ):
        raise ValueError(f"Unsupported MX scaling mode id {scaling_mode}.")

    rows, rank = b.shape
    if a.shape[0] != rank:
        raise ValueError("LoRA factor inner dimensions do not match.")
    cols = a.shape[1]
    if rows == 0 or cols == 0 or rank == 0:
        raise ValueError("Triton MX merge requires non-empty weight and factors.")
    if cols % _MX_BLOCK_SIZE != 0:
        raise ValueError("Triton MX merge requires columns divisible by 32.")

    format_id, target_max_pow2, max_pos, mbits = _format_parameters(elem_dtype)
    expected_qdata_shape = (rows, cols // 2) if format_id == _FORMAT_E2M1 else (rows, cols)
    if tuple(qdata.shape) != expected_qdata_shape:
        raise ValueError("MX qdata shape does not match its element format and LoRA factors.")
    expected_scale_shape = _expected_scale_shape(rows, cols, swizzled=swizzled)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError("MX scale shape does not match its block layout.")

    expected_scale_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scale.dtype is not expected_scale_dtype:
        raise ValueError(f"Triton MX merge expects E8M0 scales, got {scale.dtype}.")
    if format_id == _FORMAT_E2M1:
        if qdata.dtype is not torch.uint8:
            raise ValueError("Triton MXFP4 merge expects packed uint8 qdata.")
    elif qdata.dtype is not elem_dtype:
        raise ValueError("MXFP8 qdata dtype does not match elem_dtype.")

    b = b.contiguous()
    a = a.contiguous()
    block_m = 16
    block_k = 16 if rank <= 16 else 32
    grid = (
        triton.cdiv(rows, block_m),
        cols // _MX_BLOCK_SIZE,
    )
    _merge_mx_kernel[grid](
        qdata,
        scale.view(torch.uint8),
        b,
        a,
        strength,
        M=rows,
        N=cols,
        NUM_SCALE_BLOCKS=cols // _MX_BLOCK_SIZE,
        K=rank,
        FORMAT=format_id,
        COMPUTE_DTYPE=compute_dtype,
        SCALING_MODE=scaling_mode,
        SWIZZLED=swizzled,
        TARGET_MAX_POW2=target_max_pow2,
        MAX_POS=max_pos,
        MBITS=mbits,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
    )
