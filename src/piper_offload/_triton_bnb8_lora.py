"""Triton kernels for bitsandbytes row-wise int8 LoRA merges."""

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
    _stochastic_round_to_int,
)


@triton.jit
def _merge_dense_kernel(
    cb_ptr,
    scb_ptr,
    b_ptr,
    a_ptr,
    dense_ptr,
    row_tile_max_ptr,
    strength,
    M,
    N,
    K: tl.constexpr,
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
        accumulator += tl.dot(b, a)

    offsets = offsets_m[:, None] * N + offsets_n[None, :]
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    cb = tl.load(cb_ptr + offsets, mask=mask, other=0).to(tl.float16)
    scb = tl.load(
        scb_ptr + offsets_m,
        mask=offsets_m < M,
        other=0.0,
    ).to(tl.float16)
    normalized = (cb / 127.0).to(tl.float16)
    base = (normalized * scb[:, None]).to(tl.float16)

    merged = base.to(tl.float32) + accumulator * strength
    merged = tl.minimum(tl.maximum(merged, -65504.0), 65504.0)
    merged = merged.to(tl.float16)
    tl.store(dense_ptr + offsets, merged, mask=mask)

    absolute = tl.where(mask, tl.abs(merged.to(tl.float32)), 0.0)
    row_max = tl.max(absolute, axis=1)
    tl.store(
        row_tile_max_ptr + offsets_m * num_pid_n + pid_n,
        row_max,
        mask=offsets_m < M,
    )


@triton.jit
def _reduce_row_max_kernel(
    row_tile_max_ptr,
    output_scb_ptr,
    NUM_TILES,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_TILES)
    values = tl.load(
        row_tile_max_ptr + row * NUM_TILES + offsets,
        mask=offsets < NUM_TILES,
        other=0.0,
    )
    tl.store(output_scb_ptr + row, tl.max(values, axis=0))


@triton.jit
def _quantize_kernel(
    dense_ptr,
    scb_ptr,
    output_cb_ptr,
    rounding_seed,
    M,
    N,
    STOCHASTIC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = offsets_m[:, None] * N + offsets_n[None, :]
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)

    dense = tl.load(dense_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scb = tl.load(
        scb_ptr + offsets_m,
        mask=offsets_m < M,
        other=0.0,
    )
    safe_scb = tl.where(scb == 0.0, 1.0, scb)
    scaled = dense / safe_scb[:, None] * 127.0
    scaled = tl.where(scb[:, None] == 0.0, 0.0, scaled)
    quantized = libdevice.rint(scaled)
    if STOCHASTIC:
        quantized = _stochastic_round_to_int(
            scaled,
            quantized,
            rounding_seed,
            offsets,
            -127,
            127,
        )
    quantized = tl.minimum(tl.maximum(quantized, -127.0), 127.0)
    tl.store(output_cb_ptr + offsets, quantized, mask=mask)


def merge_bnb8_lora(
    cb: torch.Tensor,
    scb: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    rounding_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw ``CB`` and ``SCB`` buffers after one LoRA merge."""
    if cb.device.type != "cuda":
        raise ValueError("Triton BNB8 merge requires CUDA tensors.")
    if cb.dtype is not torch.int8 or scb.dtype is not torch.float32:
        raise ValueError(
            "Triton BNB8 merge expects int8 CB and float32 SCB tensors."
        )
    if b.dtype is not torch.float16 or a.dtype is not torch.float16:
        raise ValueError("Triton BNB8 merge expects float16 LoRA factors.")
    if cb.ndim != 2 or b.ndim != 2 or a.ndim != 2:
        raise ValueError("Triton BNB8 merge expects rank-two tensors.")
    if (
        cb.device != scb.device
        or cb.device != b.device
        or cb.device != a.device
    ):
        raise ValueError(
            "Triton BNB8 merge requires all tensors on one CUDA device."
        )

    rows, cols = cb.shape
    rank = a.shape[0]
    if rows == 0 or cols == 0 or rank == 0:
        raise ValueError("Triton BNB8 merge requires non-empty tensors.")
    if scb.numel() != rows:
        raise ValueError("Triton BNB8 merge expects one SCB value per row.")
    if b.shape != (rows, rank) or a.shape[1] != cols:
        raise ValueError("LoRA factors do not match the BNB8 weight shape.")

    cb = cb.contiguous()
    scb = scb.contiguous()
    b = b.contiguous()
    a = a.contiguous()

    block_m = 16
    block_n = 128
    block_k = 16 if rank <= 16 else 32
    num_pid_m = (rows + block_m - 1) // block_m
    num_pid_n = (cols + block_n - 1) // block_n
    num_tiles = num_pid_m * num_pid_n
    reduction_block = 1 << (num_pid_n - 1).bit_length()
    if reduction_block > 65536:
        raise ValueError("Triton BNB8 merge supports at most 8M columns.")

    dense = torch.empty_like(cb, dtype=torch.float16)
    row_tile_max = torch.empty(
        (rows, num_pid_n),
        device=cb.device,
        dtype=torch.float32,
    )
    output_cb = torch.empty_like(cb)
    output_scb = torch.empty_like(scb)

    _merge_dense_kernel[(num_tiles,)](
        cb,
        scb,
        b,
        a,
        dense,
        row_tile_max,
        strength,
        M=rows,
        N=cols,
        K=rank,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    _reduce_row_max_kernel[(rows,)](
        row_tile_max,
        output_scb,
        NUM_TILES=num_pid_n,
        BLOCK_TILES=reduction_block,
        num_warps=4,
    )
    _quantize_kernel[(num_tiles,)](
        dense,
        output_scb,
        output_cb,
        _seed_argument(rounding_seed),
        M=rows,
        N=cols,
        STOCHASTIC=rounding_seed is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return output_cb, output_scb
