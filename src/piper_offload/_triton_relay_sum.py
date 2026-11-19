"""Fused rank-ordered FP32-accumulated SUM for 2 to 4 shared-relay ranks."""

# ruff: noqa: ANN001, ANN202, N803
# pyright: reportCallIssue=false, reportIndexIssue=false

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

MAX_FUSED_RANKS = 4
_BLOCK = 1024


@triton.jit
def _sum_ranks_kernel(out_ptr, p0, p1, p2, p3, size, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < size
    # All rank loads complete before storing so out_ptr can safely alias one input.
    acc = tl.load(p0 + offsets, valid, other=0).to(tl.float32)
    acc = acc + tl.load(p1 + offsets, valid, other=0).to(tl.float32)
    if N >= 3:
        acc = acc + tl.load(p2 + offsets, valid, other=0).to(tl.float32)
    if N >= 4:
        acc = acc + tl.load(p3 + offsets, valid, other=0).to(tl.float32)
    tl.store(out_ptr + offsets, acc, valid)


def sum_ranks(output: torch.Tensor, inputs: Sequence[torch.Tensor]) -> None:
    """Accumulate 2 to 4 contiguous matching tensors left-to-right in FP32 into ``output``."""
    count = len(inputs)
    if not 2 <= count <= MAX_FUSED_RANKS:
        raise ValueError(f"fused relay SUM supports 2..{MAX_FUSED_RANKS} ranks, got {count}")
    size = output.numel()
    if size == 0:
        return
    p0, p1 = inputs[0], inputs[1]
    p2 = inputs[2] if count >= 3 else p0
    p3 = inputs[3] if count >= 4 else p0
    grid = (triton.cdiv(size, _BLOCK),)
    _sum_ranks_kernel[grid](output, p0, p1, p2, p3, size, N=tl.constexpr(count), BLOCK=tl.constexpr(_BLOCK))
