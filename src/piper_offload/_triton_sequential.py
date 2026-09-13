"""Direct two-rank SUM on one device, with no tensor-sized workspace."""

# ruff: noqa: ANN001, ANN202, N803
# pyright: reportCallIssue=false, reportIndexIssue=false

import torch
import triton
import triton.language as tl


@triton.jit
def _sum_kernel(left, right, size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < size
    a = tl.load(left + offsets, valid, other=0).to(tl.float32)
    b = tl.load(right + offsets, valid, other=0).to(tl.float32)
    total = a + b
    # Each program owns disjoint elements in both tensors. Both original
    # contributions are loaded before either corresponding output is written.
    tl.store(left + offsets, total, valid)
    tl.store(right + offsets, total, valid)


def sum_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    """Sum matching, contiguous FP32/FP16/BF16 tensors into both inputs."""
    if left.numel():
        _sum_kernel[(triton.cdiv(left.numel(), 1024),)](left, right, left.numel(), BLOCK=tl.constexpr(1024))
