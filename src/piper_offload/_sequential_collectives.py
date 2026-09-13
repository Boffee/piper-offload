"""Validated local collectives for two ranks sharing one accelerator."""

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import torch

_SUM_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_COPY_DTYPES = (
    *_SUM_DTYPES,
    torch.float64,
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.complex64,
    torch.complex128,
)


type CollectiveKind = Literal["sum", "broadcast", "scatter", "allgather", "barrier"]


@dataclass(slots=True, frozen=True)
class LocalCollective:
    kind: CollectiveKind
    inputs: tuple[torch.Tensor, ...] = ()
    outputs: tuple[torch.Tensor, ...] = ()
    root: int = 0


def _overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.device == right.device
        and bool(left.nbytes and right.nbytes)
        and (left.data_ptr() < right.data_ptr() + right.nbytes and right.data_ptr() < left.data_ptr() + left.nbytes)
    )


def _same_region(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.device == right.device and left.data_ptr() == right.data_ptr() and left.nbytes == right.nbytes


def _match(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
        raise ValueError("sequential collective tensor shapes, dtypes and devices must match")


def _validate(call: LocalCollective, device: torch.device) -> None:
    for tensor in (*call.inputs, *call.outputs):
        if type(tensor) is not torch.Tensor:
            raise TypeError("sequential collectives require plain local tensors")
        if tensor.device not in (torch.device("cpu"), device):
            raise ValueError("sequential collectives require CPU tensors or the executor's GPU")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError("sequential collectives require contiguous strided tensors")
        if tensor.dtype not in _COPY_DTYPES:
            raise NotImplementedError(f"unsupported sequential collective dtype: {tensor.dtype}")
    if call.root not in (0, 1):
        raise ValueError("sequential collective root must be 0 or 1")


def _copies(calls: tuple[LocalCollective, LocalCollective]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    left, right = calls
    copies = []
    if left.kind == "broadcast":
        if any(len(c.outputs) != 1 for c in calls):
            raise ValueError("broadcast requires one tensor per rank")
        copies = [(c.outputs[0], calls[left.root].outputs[0]) for c in calls]
    elif left.kind == "scatter":
        source = calls[left.root]
        if len(source.inputs) != 2 or calls[1 - left.root].inputs or any(len(c.outputs) != 1 for c in calls):
            raise ValueError("scatter requires two source tensors on the root and one output per rank")
        copies = [(c.outputs[0], source.inputs[r]) for r, c in enumerate(calls)]
    elif left.kind == "allgather":
        if not left.inputs or len(left.inputs) != len(right.inputs):
            raise ValueError("all-gather requires matching nonempty input lists")
        if any(len(c.outputs) != 2 * len(left.inputs) for c in calls):
            raise ValueError("all-gather requires two output slices per input")
        for index, pair in enumerate(zip(left.inputs, right.inputs, strict=True)):
            _match(*pair)
            copies.extend((c.outputs[2 * index + rank], src) for c in calls for rank, src in enumerate(pair))
    else:
        raise NotImplementedError(f"unsupported sequential collective: {left.kind}")
    for output, source in copies:
        _match(output, source)
    _validate_copy_aliases(copies)
    return copies


def _validate_copy_aliases(copies: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    # Cross-rank aliases can occur in one address space. Exact aliases are
    # safe only when they represent the same intended value; partial overlap
    # or overwriting another source would make ordering affect the result.
    for (output, source), (other_output, other_source) in combinations(copies, 2):
        if _overlap(output, other_output) and not (
            _same_region(output, other_output) and _same_region(source, other_source)
        ):
            raise ValueError("sequential collective outputs must not overlap")
    for output, source in copies:
        for _, other_source in copies:
            if _overlap(output, other_source) and not (
                _same_region(output, source) and _same_region(source, other_source)
            ):
                raise ValueError("sequential collective output would overwrite another input")


def execute(calls: tuple[LocalCollective, LocalCollective], device: torch.device) -> None:
    """Validate both ranks before enqueuing any mutation on the shared stream."""
    left, right = calls
    if (left.kind, left.root) != (right.kind, right.root):
        raise ValueError("sequential ranks requested different collectives or roots")
    for call in calls:
        _validate(call, device)
    if left.kind == "barrier":
        return
    if left.kind == "sum":
        _sum(left, right)
        return
    for output, source in _copies(calls):
        if not _same_region(output, source):
            output.copy_(source, non_blocking=True)


def _sum(left: LocalCollective, right: LocalCollective) -> None:
    if not left.outputs or len(left.outputs) != len(right.outputs):
        raise ValueError("all-reduce requires matching nonempty tensor lists")
    pairs = list(zip(left.outputs, right.outputs, strict=True))
    for a, b in pairs:
        _match(a, b)
        if a.dtype not in _SUM_DTYPES:
            raise NotImplementedError("sequential SUM supports FP32, FP16 and BF16")
        if _overlap(a, b) and not _same_region(a, b):
            raise ValueError("sequential SUM inputs must not partially overlap")
    for first, second in combinations(pairs, 2):
        if any(_overlap(a, b) for a in first for b in second):
            raise ValueError("coalesced sequential SUM tensors must not overlap")
    for a, b in pairs:
        if a.device.type == "cuda":
            from ._triton_sequential import sum_pair  # noqa: PLC0415

            sum_pair(a, b)
        else:
            torch.add(a, b, out=a)
            b.copy_(a)
