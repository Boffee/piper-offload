"""Validation, lowering, and bounded multi-transport execution for ParallelExecutor collectives."""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.distributed as dist

from ._mapped_file import overlaps_mapped_file
from ._transfer_priority import ForegroundProducer, foreground_transfer, register_foreground_producer
from .pin_manager import PinLease, host_pin_manager

TransportPreference = Literal["auto", "p2p", "host_staged"]
TransportMode = Literal["cpu", "p2p", "host_staged"]
CollectiveOp = Literal[
    "barrier",
    "allreduce",
    "broadcast",
    "scatter",
    "allgather",
    "reduce_scatter",
    "alltoall",
    "alltoall_single",
]

_MAX_FUSED_RANKS = 4
_SUM_FLOAT_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})
_SUM_DTYPES = _SUM_FLOAT_DTYPES | frozenset(
    {torch.float64, torch.int8, torch.uint8, torch.int32, torch.int64}
)


class _PreMutationError(Exception):
    """Wraps a collective validation or budget error raised before any state mutation."""

    def __init__(self, cause: ValueError | TypeError) -> None:
        super().__init__(str(cause))
        self.cause = cause


@functools.cache
def _load_triton_sum() -> Callable[[torch.Tensor, Sequence[torch.Tensor]], None] | None:
    """Lazy-load optional Triton fused SUM so CPU and non-Triton environments work."""
    try:
        from ._triton_relay_sum import sum_ranks  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name != "triton":
            raise
        return None
    return sum_ranks


@dataclass(frozen=True)
class CollectiveCall:
    """Per-rank descriptor submitted to a group collective channel."""

    op: CollectiveOp
    rank: int
    tensors: tuple[torch.Tensor, ...] = ()
    grouped_tensors: tuple[tuple[torch.Tensor, ...], ...] = ()
    root: int = 0


@dataclass(frozen=True, slots=True)
class _CopySpec:
    """Normalized byte-copy work item from one source rank to disjoint destinations."""

    src_rank: int
    src_bytes: torch.Tensor
    dsts: tuple[tuple[int, torch.Tensor], ...]


@dataclass(frozen=True, slots=True)
class _ReduceSpec:
    """Normalized 1D reduction work item with inputs ordered by ascending group rank."""

    inputs: tuple[torch.Tensor, ...]
    outputs: tuple[tuple[int, torch.Tensor], ...]
    inplace_allreduce: bool


@dataclass(slots=True)
class _HostStage:
    """Single reusable host DMA staging buffer guarded by _ParallelTransport._lock."""

    dma_base: torch.Tensor | None = None
    quarantined: bool = False

    def checkout(self, min_bytes: int, max_bytes: int) -> torch.Tensor:
        if self.quarantined:
            raise RuntimeError("piper_parallel host stage is unavailable after failed GPU drain")
        buf = self.dma_base
        self.dma_base = None
        if buf is not None and min_bytes <= buf.numel() <= max_bytes:
            return buf
        buf = None  # Drop oversized/undersized buffer before allocating replacement.
        return torch.empty(min_bytes, dtype=torch.uint8, device="cpu")

    def release(self, buf: torch.Tensor) -> None:
        if not self.quarantined:
            self.dma_base = buf

    def quarantine(self) -> None:
        self.quarantined = True
        self.dma_base = None


@dataclass(slots=True, eq=False)
class _PendingTransfer:
    """Durable pre-allocated ownership record for active/unreclaimed GPU transfers."""

    streams: tuple[torch.cuda.Stream, ...]
    keep_alive: tuple[torch.Tensor, ...]
    lease: PinLease | None = None
    dma_base: torch.Tensor | None = None
    events: tuple[torch.cuda.Event, ...] = field(default=())


def _all_peers_accessible(devices: Sequence[torch.device]) -> tuple[bool, tuple[int, int] | None]:
    distinct_indices = sorted({int(dev.index) for dev in devices if dev.index is not None})
    for src_idx in distinct_indices:
        for dst_idx in distinct_indices:
            if src_idx != dst_idx and not torch.cuda.can_device_access_peer(src_idx, dst_idx):
                return False, (src_idx, dst_idx)
    return True, None


def _validate_cuda_devices(devices: Sequence[torch.device]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for ParallelExecutor CUDA devices")
    device_count = torch.cuda.device_count()
    for dev in devices:
        if dev.index is None or dev.index < 0 or dev.index >= device_count:
            raise ValueError(f"invalid CUDA device {dev!r}; available device count is {device_count}")


def resolve_transport(
    devices: Sequence[torch.device],
    transport: TransportPreference,
) -> TransportMode:
    """Resolve user transport preference against normalized rank devices."""
    if transport not in ("auto", "p2p", "host_staged"):
        raise ValueError(f"unsupported transport {transport!r}; expected 'auto', 'p2p', or 'host_staged'")
    if not devices:
        raise ValueError("devices must not be empty")
    dev_type = devices[0].type
    if dev_type == "cpu":
        if transport != "auto":
            raise ValueError(f"{transport} transport requires CUDA devices")
        return "cpu"
    if dev_type != "cuda":
        raise ValueError(f"unsupported device type {dev_type!r}; expected 'cpu' or 'cuda'")
    _validate_cuda_devices(devices)
    if transport == "host_staged":
        return "host_staged"
    accessible, blocked_pair = _all_peers_accessible(devices)
    if accessible:
        return "p2p"
    if transport == "p2p":
        assert blocked_pair is not None
        raise RuntimeError(
            f"CUDA peer-to-peer access unavailable between cuda:{blocked_pair[0]} and cuda:{blocked_pair[1]}",
        )
    return "host_staged"


def _flat_1d(tensor: torch.Tensor) -> torch.Tensor:
    """Return a unit-stride 1D view preserving exact storage pointer and offset."""
    return tensor.as_strided((tensor.numel(),), (1,)) if tensor.numel() <= 1 else tensor.reshape(-1)


def as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Return a unit-stride 1D uint8 byte view over a contiguous tensor."""
    return _flat_1d(tensor).view(torch.uint8)


def overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return True when two contiguous tensors share device byte ranges."""
    if left.device != right.device or left.numel() == 0 or right.numel() == 0:
        return False
    left_start = left.data_ptr()
    left_end = left_start + left.numel() * left.element_size()
    right_start = right.data_ptr()
    right_end = right_start + right.numel() * right.element_size()
    return left_start < right_end and right_start < left_end


def single_tensor(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Extract the single tensor from a pybind ProcessGroup argument list."""
    if len(tensors) != 1:
        raise ValueError("piper_parallel expected exactly one tensor")
    return tensors[0]


def validate_tensors(tensors: Sequence[torch.Tensor], expected_device: torch.device) -> None:
    """Validate that every tensor is a plain contiguous non-neg/non-conj strided tensor on expected_device."""
    for tensor in tensors:
        if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
            raise TypeError("piper_parallel expects plain torch.Tensor or Parameter objects")
        if tensor.layout != torch.strided:
            raise ValueError("piper_parallel supports only strided tensors")
        if tensor.is_quantized:
            raise ValueError("piper_parallel does not support quantized tensors")
        if tensor.is_neg() or tensor.is_conj():
            raise ValueError("piper_parallel does not support negative or conjugate tensor views")
        if tensor.device != expected_device:
            raise ValueError(
                f"piper_parallel tensor device {tensor.device} does not match rank device {expected_device}",
            )
        if not tensor.is_contiguous():
            raise ValueError("piper_parallel requires contiguous tensors")


def validate_writable(tensors: Sequence[torch.Tensor]) -> None:
    """Reject Piper read-only mapped checkpoint storage before collective mutation."""
    for tensor in tensors:
        if overlaps_mapped_file(tensor):
            raise ValueError("piper_parallel cannot write into read-only mapped checkpoint storage")


def validate_reduction_tensors(
    tensors: Sequence[torch.Tensor],
    expected_device: torch.device,
    reduce_op: object,
) -> None:
    """Validate reduction op and tensor storage/dtype preconditions."""
    if reduce_op != dist.ReduceOp.SUM:
        raise ValueError("piper_parallel supports only ReduceOp.SUM")
    validate_tensors(tensors, expected_device)
    for tensor in tensors:
        if tensor.dtype not in _SUM_DTYPES:
            raise ValueError(f"piper_parallel does not support SUM reduction for dtype {tensor.dtype}")


def validate_disjoint(tensors: Sequence[torch.Tensor], message: str) -> None:
    """Ensure tensors in a sequence have mutually disjoint byte spans."""
    for idx, left in enumerate(tensors):
        for right in tensors[idx + 1:]:
            if overlap(left, right):
                raise ValueError(message)


def validate_batched_outputs(outputs: Sequence[torch.Tensor], inputs: Sequence[torch.Tensor]) -> None:
    """Ensure batched outputs are disjoint from one another and non-paired inputs."""
    validate_disjoint(outputs, "piper_parallel coalesced outputs must not overlap")
    for out_idx, output in enumerate(outputs):
        for in_idx, source in enumerate(inputs):
            if out_idx != in_idx and overlap(output, source):
                raise ValueError("piper_parallel coalesced output must not overlap another batch entry's input")


def validate_own_slice_aliases(
    rank: int,
    single: torch.Tensor,
    per_rank: Sequence[torch.Tensor],
    message: str,
) -> None:
    """Allow an in-place alias only when single matches rank's exact slice pointer."""
    for peer, slice_tensor in enumerate(per_rank):
        if overlap(slice_tensor, single) and (peer != rank or slice_tensor.data_ptr() != single.data_ptr()):
            raise ValueError(message)


def validate_alltoall_aliases(
    rank: int,
    outputs: Sequence[torch.Tensor],
    sources: Sequence[torch.Tensor],
) -> None:
    """Allow in-place all-to-all only on the exact own-rank slice."""
    if outputs[rank].nbytes != sources[rank].nbytes:
        raise ValueError("piper_parallel all-to-all own-rank input and output byte sizes must match")
    for out_rank, output in enumerate(outputs):
        for in_rank, source in enumerate(sources):
            if overlap(output, source) and not (
                out_rank == in_rank == rank
                and output.data_ptr() == source.data_ptr()
                and output.nbytes == source.nbytes
            ):
                raise ValueError("piper_parallel all-to-all output may overlap inputs only at its exact local slice")


def resolve_alltoall_splits(
    tensor: torch.Tensor,
    split_sizes: Sequence[int] | None,
    group_size: int,
) -> list[int]:
    """Resolve and validate dim-0 split sizes for all_to_all_single."""
    dim0 = tensor.shape[0]
    if split_sizes is None or len(split_sizes) == 0:
        if dim0 % group_size != 0:
            raise ValueError(
                f"piper_parallel equal all_to_all_single requires dim 0 ({dim0}) "
                f"divisible by world size ({group_size})",
            )
        return [dim0 // group_size] * group_size
    if len(split_sizes) != group_size:
        raise ValueError("piper_parallel all_to_all_single requires one split size per rank")
    for s in split_sizes:
        if type(s) is not int:
            raise TypeError("piper_parallel all_to_all_single split sizes must be int")
    resolved = list(split_sizes)
    if any(s < 0 for s in resolved) or sum(resolved) != dim0:
        raise ValueError("piper_parallel all_to_all_single split sizes must be non-negative and sum to dim 0")
    return resolved


def validate_pair(
    output: torch.Tensor,
    source: torch.Tensor,
    expected_numel: int,
    *,
    same_shape: bool = False,
) -> None:
    """Validate dtype and element count / shape agreement between two tensors."""
    if output.dtype != source.dtype:
        raise ValueError("piper_parallel collective tensor dtypes must match")
    if same_shape and output.shape != source.shape:
        raise ValueError("piper_parallel collective tensor shapes must match")
    if output.numel() != expected_numel:
        raise ValueError("piper_parallel collective tensor sizes must match")


def validate_single_collective_shape(big: torch.Tensor, small: torch.Tensor, group_size: int) -> None:
    """Validate PyTorch's stack or dim-0 concatenation contract for single-tensor collectives."""
    if big.dtype != small.dtype:
        raise ValueError("piper_parallel collective tensor dtypes must match")
    stack_shape = (group_size, *small.shape)
    concat_shape = (group_size * small.shape[0], *small.shape[1:]) if small.ndim >= 1 else None
    if big.shape == stack_shape or (concat_shape is not None and big.shape == concat_shape):
        return
    if group_size == 1 and big.shape == small.shape:
        return
    raise ValueError(
        f"piper_parallel tensor shape {tuple(big.shape)} must match stack {stack_shape}"
        + (f" or concatenation {concat_shape}" if concat_shape is not None else ""),
    )


def _validate_allreduce_calls(calls: Sequence[CollectiveCall]) -> None:
    count = len(calls[0].tensors)
    for call in calls[1:]:
        if len(call.tensors) != count:
            raise ValueError("piper_parallel allreduce tensor list lengths must match across ranks")
    for idx in range(count):
        ref = calls[0].tensors[idx]
        for call in calls[1:]:
            validate_pair(call.tensors[idx], ref, ref.numel(), same_shape=True)


def _validate_broadcast_calls(calls: Sequence[CollectiveCall]) -> None:
    root = calls[0].root
    ref = calls[root].tensors[0]
    for call in calls:
        if call.root != root:
            raise ValueError("piper_parallel broadcast root rank must match across ranks")
        validate_pair(call.tensors[0], ref, ref.numel(), same_shape=True)


def _validate_scatter_calls(calls: Sequence[CollectiveCall]) -> None:
    root = calls[0].root
    for call in calls[1:]:
        if call.root != root:
            raise ValueError("piper_parallel scatter root rank must match across ranks")
    root_sources = calls[root].grouped_tensors[0]
    for r, call in enumerate(calls):
        validate_pair(call.tensors[0], root_sources[r], root_sources[r].numel(), same_shape=True)


def _validate_allgather_calls(calls: Sequence[CollectiveCall]) -> None:
    group_size = len(calls)
    batch_count = len(calls[0].tensors)
    for call in calls[1:]:
        if len(call.tensors) != batch_count or len(call.grouped_tensors) != batch_count:
            raise ValueError("piper_parallel allgather batch lengths must match across ranks")
    for b in range(batch_count):
        dtype = calls[0].tensors[b].dtype
        for r in range(group_size):
            src = calls[r].tensors[b]
            if src.dtype != dtype:
                raise ValueError("piper_parallel allgather tensor dtypes must match across ranks")
            for dst_rank in range(group_size):
                dst = calls[dst_rank].grouped_tensors[b][r]
                if dst.dtype != src.dtype or dst.shape != src.shape:
                    raise ValueError("piper_parallel allgather output shapes must match rank inputs")


def _validate_reduce_scatter_calls(calls: Sequence[CollectiveCall]) -> None:
    group_size = len(calls)
    batch_count = len(calls[0].tensors)
    for call in calls[1:]:
        if len(call.tensors) != batch_count or len(call.grouped_tensors) != batch_count:
            raise ValueError("piper_parallel reduce_scatter batch lengths must match across ranks")
    for b in range(batch_count):
        dtype = calls[0].tensors[b].dtype
        for r in range(group_size):
            out = calls[r].tensors[b]
            if out.dtype != dtype:
                raise ValueError("piper_parallel reduce_scatter tensor dtypes must match across ranks")
            for peer in range(group_size):
                src = calls[peer].grouped_tensors[b][r]
                if src.dtype != out.dtype or src.shape != out.shape:
                    raise ValueError("piper_parallel reduce_scatter input shapes must match peer outputs")


def _validate_alltoall_calls(calls: Sequence[CollectiveCall]) -> None:
    group_size = len(calls)
    dtype = calls[0].tensors[0].dtype
    for call in calls[1:]:
        if call.tensors[0].dtype != dtype:
            raise ValueError("piper_parallel all-to-all dtypes must match across ranks")
    for src in range(group_size):
        for dst in range(group_size):
            if calls[src].grouped_tensors[0][dst].nbytes != calls[dst].tensors[src].nbytes:
                raise ValueError("piper_parallel all-to-all split byte sizes must agree between peers")


def _validate_alltoall_single_calls(calls: Sequence[CollectiveCall]) -> None:
    group_size = len(calls)
    dtype = calls[0].tensors[0].dtype
    for call in calls[1:]:
        if call.tensors[0].dtype != dtype:
            raise ValueError("piper_parallel all-to-all dtypes must match across ranks")
    for src in range(group_size):
        for dst in range(group_size):
            if calls[src].grouped_tensors[0][dst].shape != calls[dst].tensors[src].shape:
                raise ValueError("piper_parallel all_to_all_single slice shapes must agree between peers")


_CALL_VALIDATORS: dict[CollectiveOp, Callable[[Sequence[CollectiveCall]], None]] = {
    "barrier": lambda _calls: None,
    "allreduce": _validate_allreduce_calls,
    "broadcast": _validate_broadcast_calls,
    "scatter": _validate_scatter_calls,
    "allgather": _validate_allgather_calls,
    "reduce_scatter": _validate_reduce_scatter_calls,
    "alltoall": _validate_alltoall_calls,
    "alltoall_single": _validate_alltoall_single_calls,
}


def _lower_broadcast_and_scatter(calls: Sequence[CollectiveCall]) -> list[_CopySpec]:
    group_size = len(calls)
    root = calls[0].root
    if calls[0].op == "broadcast":
        if group_size <= 1:
            return []
        src = as_bytes(calls[root].tensors[0])
        dsts = tuple((r, as_bytes(calls[r].tensors[0])) for r in range(group_size) if r != root)
        return [_CopySpec(root, src, dsts)] if src.numel() > 0 and dsts else []

    root_sources = calls[root].grouped_tensors[0]
    specs: list[_CopySpec] = []
    for r, call in enumerate(calls):
        dst = call.tensors[0]
        src = root_sources[r]
        if dst.numel() == 0 or (r == root and dst.data_ptr() == src.data_ptr()):
            continue
        specs.append(_CopySpec(root, as_bytes(src), ((r, as_bytes(dst)),)))
    return specs


def _lower_allgather(calls: Sequence[CollectiveCall]) -> list[_CopySpec]:
    group_size = len(calls)
    specs: list[_CopySpec] = []
    for b in range(len(calls[0].tensors)):
        for src_rank in range(group_size):
            src = calls[src_rank].tensors[b]
            if src.numel() == 0:
                continue
            dsts = tuple(
                (dst_rank, as_bytes(calls[dst_rank].grouped_tensors[b][src_rank]))
                for dst_rank in range(group_size)
                if not (
                    dst_rank == src_rank
                    and calls[dst_rank].grouped_tensors[b][src_rank].data_ptr() == src.data_ptr()
                )
            )
            if dsts:
                specs.append(_CopySpec(src_rank, as_bytes(src), dsts))
    return specs


def _lower_alltoall(calls: Sequence[CollectiveCall]) -> list[_CopySpec]:
    group_size = len(calls)
    specs: list[_CopySpec] = []
    for src_rank in range(group_size):
        for dst_rank in range(group_size):
            src_slice = calls[src_rank].grouped_tensors[0][dst_rank]
            dst_slice = calls[dst_rank].tensors[src_rank]
            if src_slice.nbytes == 0 or (src_rank == dst_rank and src_slice.data_ptr() == dst_slice.data_ptr()):
                continue
            specs.append(_CopySpec(src_rank, as_bytes(src_slice), ((dst_rank, as_bytes(dst_slice)),)))
    return specs


def _lower_copy_specs(calls: Sequence[CollectiveCall]) -> list[_CopySpec]:
    op = calls[0].op
    if op in ("broadcast", "scatter"):
        return _lower_broadcast_and_scatter(calls)
    if op == "allgather":
        return _lower_allgather(calls)
    return _lower_alltoall(calls)


def _lower_reduce_specs(calls: Sequence[CollectiveCall]) -> list[_ReduceSpec]:
    op = calls[0].op
    group_size = len(calls)
    if op == "allreduce":
        if group_size <= 1:
            return []
        specs: list[_ReduceSpec] = []
        for idx in range(len(calls[0].tensors)):
            flats = tuple(_flat_1d(calls[r].tensors[idx]) for r in range(group_size))
            if flats[0].numel() > 0:
                specs.append(_ReduceSpec(flats, tuple(enumerate(flats)), inplace_allreduce=True))
        return specs

    specs = []
    for b in range(len(calls[0].tensors)):
        for dst_rank in range(group_size):
            out = _flat_1d(calls[dst_rank].tensors[b])
            if out.numel() == 0:
                continue
            inputs = tuple(_flat_1d(calls[src_rank].grouped_tensors[b][dst_rank]) for src_rank in range(group_size))
            if group_size == 1 and out.data_ptr() == inputs[0].data_ptr():
                continue
            specs.append(_ReduceSpec(inputs, ((dst_rank, out),), inplace_allreduce=False))
    return specs


def _logical_operands_for_rank(call: CollectiveCall) -> tuple[torch.Tensor, ...]:
    """Return every declared input and output tensor (including kept self-aliases) for one rank."""
    return (*call.tensors, *(t for batch in call.grouped_tensors for t in batch))


def _validate_shared_device_disjointness(
    calls: Sequence[CollectiveCall],
    writes_by_rank: Sequence[Sequence[torch.Tensor]],
    group_devices: Sequence[torch.device],
) -> None:
    group_size = len(calls)
    logical_by_rank = [_logical_operands_for_rank(call) for call in calls]
    for r, writes in enumerate(writes_by_rank):
        if not writes:
            continue
        validate_writable(writes)
        dev_r = group_devices[r]
        for q in range(group_size):
            if q == r or group_devices[q] != dev_r:
                continue
            peer_operands = logical_by_rank[q]
            for write_tensor in writes:
                if any(overlap(write_tensor, peer_tensor) for peer_tensor in peer_operands):
                    raise ValueError(
                        "piper_parallel rank output must not overlap another rank's tensor on the same device",
                    )


def validate_and_lower(
    calls: Sequence[CollectiveCall],
    group_devices: Sequence[torch.device],
) -> tuple[list[_CopySpec], list[_ReduceSpec]]:
    """Validate structure, lower specs, and verify actual writes vs peer logical operands & writability."""
    op = calls[0].op
    for call in calls[1:]:
        if call.op != op:
            raise ValueError(
                f"piper_parallel collective mismatch: rank 0 called {op}, rank {call.rank} called {call.op}",
            )
    _CALL_VALIDATORS[op](calls)
    if op == "barrier":
        return [], []

    group_size = len(calls)
    writes_by_rank: list[list[torch.Tensor]] = [[] for _ in range(group_size)]
    if op in ("allreduce", "reduce_scatter"):
        copy_specs: list[_CopySpec] = []
        reduce_specs = _lower_reduce_specs(calls)
        for spec in reduce_specs:
            for dst_rank, out in spec.outputs:
                writes_by_rank[dst_rank].append(out)
    else:
        reduce_specs = []
        copy_specs = _lower_copy_specs(calls)
        for spec in copy_specs:
            for dst_rank, dst_bytes in spec.dsts:
                writes_by_rank[dst_rank].append(dst_bytes)

    _validate_shared_device_disjointness(calls, writes_by_rank, group_devices)
    return copy_specs, reduce_specs


@contextmanager
def _on_stream(stream: torch.cuda.Stream) -> Iterator[None]:
    """Run operations under stream's own CUDA device and stream context."""
    with torch.cuda.device(stream.device), torch.cuda.stream(stream):
        yield


def _copy_d2d(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_stream: torch.cuda.Stream,
    src_stream: torch.cuda.Stream,
) -> None:
    """Enqueue D2D copy selecting private streams on both source and destination devices."""
    if src.device == dst.device:
        with _on_stream(dst_stream):
            dst.copy_(src, non_blocking=True)
        return
    with _on_stream(src_stream), _on_stream(dst_stream):
        dst.copy_(src, non_blocking=True)


def _sync_events(events: Iterable[torch.cuda.Event | None]) -> None:
    """Synchronize every non-None CUDA event before raising the first encountered error."""
    first_error: BaseException | None = None
    for event in events:
        if event is None:
            continue
        try:
            event.synchronize()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _sync_streams(streams: Iterable[torch.cuda.Stream]) -> None:
    """Synchronize every unique stream before raising the first encountered error."""
    seen: dict[int, torch.cuda.Stream] = {}
    for stream in streams:
        seen[id(stream)] = stream
    first_error: BaseException | None = None
    for stream in seen.values():
        try:
            stream.synchronize()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _record_completion_events(record: _PendingTransfer) -> None:
    recorded: list[torch.cuda.Event] = []
    for stream in record.streams:
        try:
            with torch.cuda.device(stream.device):
                ev = torch.cuda.Event()
                ev.record(stream)
            recorded.append(ev)
        except Exception:
            record.events = ()
            return
    record.events = tuple(recorded)


def _is_transfer_done(record: _PendingTransfer) -> bool:
    if record.events:
        for stream, ev in zip(record.streams, record.events, strict=True):
            with torch.cuda.device(stream.device):
                if not ev.query():
                    return False
        return True
    for stream in record.streams:
        with torch.cuda.device(stream.device):
            if not stream.query():
                return False
    return True


def _sorted_unique_devices(devices: Sequence[torch.device]) -> list[torch.device]:
    return sorted(set(devices), key=lambda d: int(d.index if d.index is not None else 0))


def _unique_streams(streams: Iterable[torch.cuda.Stream]) -> tuple[torch.cuda.Stream, ...]:
    seen: dict[int, torch.cuda.Stream] = {}
    for stream in streams:
        seen[id(stream)] = stream
    return tuple(seen.values())


def _collect_reduce_tensors(spec: _ReduceSpec) -> list[torch.Tensor]:
    return [*spec.inputs, *(out for _, out in spec.outputs)]


def _collect_copy_tensors(specs: Sequence[_CopySpec]) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for spec in specs:
        tensors.append(spec.src_bytes)
        tensors.extend(dst for _, dst in spec.dsts)
    return tensors


class _ParallelTransport:
    """Bounded collective transfer engine over cpu, p2p, or host_staged."""

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        mode: TransportMode,
        staging_bytes: int,
    ) -> None:
        self._devices = devices
        self._mode = mode
        self._staging_bytes = staging_bytes
        self._lock = threading.Lock()
        self._stage = _HostStage()
        self._pending: _PendingTransfer | None = None
        self._poisoned = False
        self._rank_streams: dict[int, torch.cuda.Stream] = {}
        self._device_streams: dict[int, torch.cuda.Stream] = {}
        self._foreground_producer: ForegroundProducer | None = (
            register_foreground_producer(devices) if mode != "cpu" else None
        )

    def _device_stream(self, device: torch.device) -> torch.cuda.Stream:
        assert device.index is not None
        idx = int(device.index)
        stream = self._device_streams.get(idx)
        if stream is None:
            stream = torch.cuda.Stream(device)
            self._device_streams[idx] = stream
        return stream

    def _rank_stream(self, global_rank: int) -> torch.cuda.Stream:
        stream = self._rank_streams.get(global_rank)
        if stream is None:
            stream = torch.cuda.Stream(self._devices[global_rank])
            self._rank_streams[global_rank] = stream
        return stream

    def _ensure_healthy(self) -> None:
        with self._lock:
            if self._poisoned or self._pending is not None:
                raise RuntimeError("piper_parallel transport is poisoned or pending drain")

    def _abort_poisoned_entry(self, lease: PinLease | None, dma_base: torch.Tensor | None) -> None:
        try:
            if lease is not None:
                lease.close()
        finally:
            if dma_base is not None:
                with self._lock:
                    self._stage.release(dma_base)

    def _finalize_pending_transfer(self, record: _PendingTransfer, *, release_stage: bool) -> None:
        lease_ref = record.lease
        try:
            if lease_ref is not None:
                lease_ref.close()
        except BaseException:
            with self._lock:
                self._poisoned = True
                self._stage.quarantine()
            raise
        record.lease = None
        dma_ref = record.dma_base
        record.dma_base = None
        record.keep_alive = ()
        record.events = ()
        record.streams = ()
        with self._lock:
            if self._pending is record:
                self._pending = None
            if release_stage and dma_ref is not None:
                self._stage.release(dma_ref)

    def _run_and_sync_transfer(
        self,
        record: _PendingTransfer,
        fn: Callable[[], None],
    ) -> tuple[bool, BaseException | None]:
        synced = False
        fn_exc: BaseException | None = None
        try:
            fn()
        except BaseException as exc:
            fn_exc = exc
        finally:
            try:
                _record_completion_events(record)
                _sync_streams(record.streams)
                synced = True
            finally:
                if not synced:
                    with self._lock:
                        self._poisoned = True
                        self._stage.quarantine()
        return synced, fn_exc

    def _run_gpu_transfer(
        self,
        streams: Sequence[torch.cuda.Stream],
        keep_alive: Sequence[torch.Tensor],
        fn: Callable[[], None],
        *,
        lease: PinLease | None = None,
        dma_base: torch.Tensor | None = None,
    ) -> None:
        record = _PendingTransfer(
            streams=_unique_streams(streams),
            keep_alive=tuple(keep_alive),
            lease=lease,
            dma_base=dma_base,
        )
        with self._lock:
            poisoned = self._poisoned or self._pending is not None
        if poisoned:
            self._abort_poisoned_entry(lease, dma_base)
            raise RuntimeError("piper_parallel transport is poisoned or pending drain")

        published = False
        synced = False
        fn_exc: BaseException | None = None
        try:
            with ExitStack() as stack:
                for dev in _sorted_unique_devices([s.device for s in record.streams]):
                    stack.enter_context(foreground_transfer(dev))
                with self._lock:
                    if self._poisoned or self._pending is not None:
                        raise RuntimeError("piper_parallel transport is poisoned or pending drain")
                    self._pending = record
                    published = True
                synced, fn_exc = self._run_and_sync_transfer(record, fn)
            if fn_exc is not None:
                raise fn_exc
        finally:
            if not published:
                self._abort_poisoned_entry(lease, dma_base)
            elif synced:
                if fn_exc is not None:
                    with self._lock:
                        self._poisoned = True
                        self._stage.quarantine()
                self._finalize_pending_transfer(record, release_stage=(fn_exc is None))

    def drain_unreclaimed(self, deadline: float) -> bool:
        """Poll completion fences for any interrupted GPU transfer; call only from close()."""
        with self._lock:
            record = self._pending
        if record is None:
            return True

        while not _is_transfer_done(record):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            time.sleep(min(0.002, remaining))

        self._finalize_pending_transfer(record, release_stage=False)
        return True

    def close(self) -> None:
        """Release cached staging buffers and streams after drain completes."""
        try:
            with self._lock:
                self._stage.quarantine()
            self._rank_streams.clear()
            self._device_streams.clear()
        finally:
            if self._foreground_producer is not None:
                self._foreground_producer.close()
                self._foreground_producer = None

    def _plan_budgets(
        self,
        calls: Sequence[CollectiveCall],
        group_devices: tuple[torch.device, ...],
    ) -> tuple[
        list[_CopySpec],
        list[_ReduceSpec],
        Callable[[torch.Tensor, Sequence[torch.Tensor]], None] | None,
        list[tuple[int, int, bool]],
        list[tuple[int, int, torch.dtype]],
    ]:
        fused_sum: Callable[[torch.Tensor, Sequence[torch.Tensor]], None] | None = None
        p2p_plans: list[tuple[int, int, bool]] = []
        host_plans: list[tuple[int, int, torch.dtype]] = []
        try:
            copy_specs, reduce_specs = validate_and_lower(calls, group_devices)
            if self._mode == "cpu":
                for spec in reduce_specs:
                    self._validate_cpu_reduce_budget(spec)
            elif self._mode == "p2p" and reduce_specs:
                fused_sum = _load_triton_sum()
                group_size = len(calls)
                p2p_plans = [self._plan_p2p_reduce(spec, group_size, fused_sum) for spec in reduce_specs]
            elif self._mode == "host_staged" and reduce_specs:
                host_plans = [
                    self._host_reduce_layout(len(spec.inputs), spec.inputs[0].dtype, spec.inputs[0].numel())
                    for spec in reduce_specs
                    if len(spec.inputs) > 1
                ]
        except (ValueError, TypeError) as exc:
            raise _PreMutationError(exc) from exc
        return copy_specs, reduce_specs, fused_sum, p2p_plans, host_plans

    @torch.inference_mode()
    def execute_collective(
        self,
        calls: Sequence[CollectiveCall],
        global_ranks: tuple[int, ...],
        ready_events: Sequence[torch.cuda.Event | None],
    ) -> None:
        """Validate across all ranks and execute one synchronized collective."""
        group_devices = tuple(self._devices[g] for g in global_ranks)
        copy_specs, reduce_specs, fused_sum, p2p_plans, host_plans = self._plan_budgets(calls, group_devices)

        if calls[0].op == "barrier":
            if self._mode != "cpu":
                self._ensure_healthy()
                _sync_events(ready_events)
            return

        if self._mode == "cpu":
            self._execute_cpu(copy_specs, reduce_specs)
            return

        self._ensure_healthy()
        _sync_events(ready_events)
        self._ensure_healthy()
        if self._mode == "p2p":
            if reduce_specs:
                self._execute_p2p_reduce(reduce_specs, p2p_plans, fused_sum, global_ranks, group_devices)
            elif copy_specs:
                self._execute_p2p_copies(copy_specs, global_ranks)
        elif reduce_specs:
            self._execute_host_reduce(reduce_specs, host_plans, group_devices)
        elif copy_specs:
            self._execute_host_copies(copy_specs, group_devices)

    # ------------------------------------------------------------------
    # CPU execution (bounded to self._staging_bytes)
    # ------------------------------------------------------------------

    def _execute_cpu(self, copy_specs: Sequence[_CopySpec], reduce_specs: Sequence[_ReduceSpec]) -> None:
        for spec in reduce_specs:
            self._cpu_reduce_many(spec.inputs, [out for _, out in spec.outputs])
        for copy_spec in copy_specs:
            for _dst_rank, dst_bytes in copy_spec.dsts:
                dst_bytes.copy_(copy_spec.src_bytes)

    def _validate_cpu_reduce_budget(self, spec: _ReduceSpec) -> None:
        if len(spec.inputs) <= 1:
            return
        dtype = spec.inputs[0].dtype
        acc_dtype = torch.float32 if dtype in _SUM_FLOAT_DTYPES else dtype
        if self._staging_bytes < acc_dtype.itemsize:
            raise ValueError(f"staging_bytes ({self._staging_bytes}) too small for dtype {dtype}")

    def _cpu_reduce_many(self, inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor]) -> None:
        n = outputs[0].numel()
        if n == 0:
            return
        if len(inputs) == 1:
            src0 = inputs[0]
            for out in outputs:
                if out.data_ptr() != src0.data_ptr():
                    out.copy_(src0)
            return
        dtype = outputs[0].dtype
        acc_dtype = torch.float32 if dtype in _SUM_FLOAT_DTYPES else dtype
        chunk_elems = min(n, self._staging_bytes // acc_dtype.itemsize)
        acc = torch.empty(chunk_elems, dtype=acc_dtype, device="cpu")
        for offset in range(0, n, chunk_elems):
            end = min(n, offset + chunk_elems)
            buf = acc[: end - offset]
            buf.copy_(inputs[0][offset:end])
            for src in inputs[1:]:
                buf.add_(src[offset:end])
            for out in outputs:
                out[offset:end].copy_(buf)

    # ------------------------------------------------------------------
    # Direct CUDA P2P execution
    # ------------------------------------------------------------------

    def _plan_p2p_reduce(
        self,
        spec: _ReduceSpec,
        group_size: int,
        fused_sum: Callable[[torch.Tensor, Sequence[torch.Tensor]], None] | None,
    ) -> tuple[int, int, bool]:
        dtype = spec.inputs[0].dtype
        n = spec.inputs[0].numel()
        if group_size <= 1:
            return 1, n, False
        use_fused = fused_sum is not None and dtype in _SUM_FLOAT_DTYPES and group_size <= _MAX_FUSED_RANKS
        acc_bytes = 4 if dtype in _SUM_FLOAT_DTYPES else dtype.itemsize
        per_dst_elem = (group_size - 1) * dtype.itemsize if use_fused else dtype.itemsize + acc_bytes
        if spec.inplace_allreduce:
            needed = group_size * per_dst_elem
            if self._staging_bytes < needed:
                raise ValueError(
                    f"staging_bytes ({self._staging_bytes}) too small for P2P reduction of {group_size} ranks "
                    f"with dtype {dtype}",
                )
            chunk = min(n, self._staging_bytes // needed)
            return group_size, chunk, use_fused
        if self._staging_bytes < per_dst_elem:
            raise ValueError(
                f"staging_bytes ({self._staging_bytes}) too small for P2P reduce_scatter of {group_size} ranks "
                f"with dtype {dtype}",
            )
        concurrency = min(group_size, self._staging_bytes // per_dst_elem)
        chunk = min(n, self._staging_bytes // (concurrency * per_dst_elem))
        return concurrency, chunk, use_fused

    def _execute_p2p_copies(self, copy_specs: Sequence[_CopySpec], global_ranks: tuple[int, ...]) -> None:
        streams = [self._rank_stream(g) for g in global_ranks]
        touched_streams: list[torch.cuda.Stream] = []
        for spec in copy_specs:
            src_stream = streams[spec.src_rank]
            touched_streams.append(src_stream)
            for dst_rank, _ in spec.dsts:
                touched_streams.append(streams[dst_rank])

        def _issue() -> None:
            for spec in copy_specs:
                src_stream = streams[spec.src_rank]
                for dst_rank, dst_bytes in spec.dsts:
                    _copy_d2d(dst_bytes, spec.src_bytes, streams[dst_rank], src_stream)

        self._run_gpu_transfer(touched_streams, _collect_copy_tensors(copy_specs), _issue)

    def _execute_p2p_reduce(
        self,
        specs: Sequence[_ReduceSpec],
        plans: Sequence[tuple[int, int, bool]],
        fused_sum: Callable[[torch.Tensor, Sequence[torch.Tensor]], None] | None,
        global_ranks: tuple[int, ...],
        group_devices: tuple[torch.device, ...],
    ) -> None:
        streams = [self._rank_stream(g) for g in global_ranks]
        group_size = len(global_ranks)
        # Group reduce_scatter specs by op batch: run disjoint dst ranks up to `concurrency` together.
        idx = 0
        while idx < len(specs):
            spec = specs[idx]
            concurrency, chunk, use_fused = plans[idx]
            if spec.inplace_allreduce or group_size == 1:
                if spec.inplace_allreduce:
                    self._p2p_allreduce_spec(spec, chunk, use_fused, fused_sum, group_devices, streams)
                else:
                    dst_rank, out = spec.outputs[0]
                    src0 = spec.inputs[0]
                    self._run_gpu_transfer(
                        (streams[dst_rank],),
                        _collect_reduce_tensors(spec),
                        lambda s=streams[dst_rank], o=out, src=src0: self._copy_on_stream(s, o, src),
                    )
                idx += 1
                continue

            # Collect consecutive reduce_scatter specs up to `concurrency` sharing plan, dtype, and distinct dst ranks.
            batch = list(zip(specs[idx : idx + concurrency], plans[idx : idx + concurrency], strict=True))
            head_plan = batch[0][1]
            head_dtype = spec.inputs[0].dtype
            compatible: list[_ReduceSpec] = []
            seen_dsts: set[int] = set()
            for item_spec, item_plan in batch:
                dst_rank = item_spec.outputs[0][0]
                if (
                    item_spec.inplace_allreduce
                    or item_plan != head_plan
                    or item_spec.inputs[0].dtype != head_dtype
                    or dst_rank in seen_dsts
                ):
                    break
                seen_dsts.add(dst_rank)
                compatible.append(item_spec)
            self._p2p_reduce_scatter_batch(compatible, chunk, use_fused, fused_sum, group_devices, streams)
            idx += len(compatible)

    def _copy_on_stream(self, stream: torch.cuda.Stream, dst: torch.Tensor, src: torch.Tensor) -> None:
        if dst.data_ptr() != src.data_ptr():
            with _on_stream(stream):
                dst.copy_(src, non_blocking=True)

    def _allocate_scratch_for_ranks(
        self,
        dst_ranks: Sequence[int],
        group_size: int,
        chunk: int,
        dtype: torch.dtype,
        *,
        use_fused: bool,
        group_devices: tuple[torch.device, ...],
        streams: Sequence[torch.cuda.Stream],
    ) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
        acc_dtype = torch.float32 if dtype in _SUM_FLOAT_DTYPES else dtype
        slots_per_dst = group_size - 1 if use_fused else 1
        scratches: list[list[torch.Tensor]] = []
        accs: list[torch.Tensor] = []
        for dst_rank in dst_ranks:
            dev = group_devices[dst_rank]
            with _on_stream(streams[dst_rank]):
                scratches.append([torch.empty(chunk, dtype=dtype, device=dev) for _ in range(slots_per_dst)])
                if not use_fused:
                    accs.append(torch.empty(chunk, dtype=acc_dtype, device=dev))
        return scratches, accs

    def _reduce_for_rank(
        self,
        dst_rank: int,
        inputs: Sequence[torch.Tensor],
        offset: int,
        end: int,
        peer_slots: list[torch.Tensor],
        acc_buf: torch.Tensor | None,
        streams: Sequence[torch.cuda.Stream],
        *,
        use_fused: bool,
        fused_sum: Callable[[torch.Tensor, Sequence[torch.Tensor]], None] | None,
        out_slice: torch.Tensor | None,
    ) -> None:
        """Pull peers for one rank in ascending rank order into `out_slice` or `acc_buf`."""
        dst_stream = streams[dst_rank]
        length = end - offset
        if use_fused:
            assert fused_sum is not None
            assert out_slice is not None
            with _on_stream(dst_stream):
                ordered: list[torch.Tensor] = []
                slot_idx = 0
                for k, src in enumerate(inputs):
                    if k == dst_rank:
                        ordered.append(src[offset:end])
                    else:
                        slot = peer_slots[slot_idx][:length]
                        _copy_d2d(slot, src[offset:end], dst_stream, streams[k])
                        ordered.append(slot)
                        slot_idx += 1
                fused_sum(out_slice, ordered)
            return

        assert acc_buf is not None
        acc_slice = acc_buf[:length]
        peer_buf = peer_slots[0][:length]
        with _on_stream(dst_stream):
            for k, src in enumerate(inputs):
                if k == dst_rank:
                    operand = src[offset:end]
                else:
                    _copy_d2d(peer_buf, src[offset:end], dst_stream, streams[k])
                    operand = peer_buf
                if k == 0:
                    acc_slice.copy_(operand)
                else:
                    acc_slice.add_(operand)
            if out_slice is not None:
                out_slice.copy_(acc_slice)

    def _p2p_reduce_scatter_batch(
        self,
        specs: Sequence[_ReduceSpec],
        chunk: int,
        use_fused: bool,
        fused_sum: Callable[[torch.Tensor, Sequence[torch.Tensor]], None] | None,
        group_devices: tuple[torch.device, ...],
        streams: Sequence[torch.cuda.Stream],
    ) -> None:
        group_size = len(group_devices)
        dst_ranks = [spec.outputs[0][0] for spec in specs]
        dtype = specs[0].outputs[0][1].dtype
        scratches, accs = self._allocate_scratch_for_ranks(
            dst_ranks,
            group_size,
            chunk,
            dtype,
            use_fused=use_fused,
            group_devices=group_devices,
            streams=streams,
        )
        keep_alive: list[torch.Tensor] = [t for slots in scratches for t in slots]
        keep_alive.extend(accs)
        for spec in specs:
            keep_alive.extend(_collect_reduce_tensors(spec))

        def _issue() -> None:
            for spec_idx, spec in enumerate(specs):
                dst_rank, out = spec.outputs[0]
                n = out.numel()
                peer_slots = scratches[spec_idx]
                acc_buf = accs[spec_idx] if accs else None
                for offset in range(0, n, chunk):
                    end = min(n, offset + chunk)
                    self._reduce_for_rank(
                        dst_rank,
                        spec.inputs,
                        offset,
                        end,
                        peer_slots,
                        acc_buf,
                        streams,
                        use_fused=use_fused,
                        fused_sum=fused_sum,
                        out_slice=out[offset:end],
                    )

        self._run_gpu_transfer(streams, keep_alive, _issue)

    def _p2p_allreduce_spec(
        self,
        spec: _ReduceSpec,
        chunk: int,
        use_fused: bool,
        fused_sum: Callable[[torch.Tensor, Sequence[torch.Tensor]], None] | None,
        group_devices: tuple[torch.device, ...],
        streams: Sequence[torch.cuda.Stream],
    ) -> None:
        flats = spec.inputs
        group_size = len(flats)
        n = flats[0].numel()
        dtype = flats[0].dtype
        scratches, accs = self._allocate_scratch_for_ranks(
            list(range(group_size)),
            group_size,
            chunk,
            dtype,
            use_fused=use_fused,
            group_devices=group_devices,
            streams=streams,
        )
        keep_alive: list[torch.Tensor] = [t for slots in scratches for t in slots]
        keep_alive.extend(accs)
        keep_alive.extend(flats)

        def _issue() -> None:
            for offset in range(0, n, chunk):
                end = min(n, offset + chunk)
                if use_fused:
                    assert fused_sum is not None
                    self._p2p_allreduce_chunk_fused(flats, offset, end, scratches, streams, fused_sum)
                else:
                    self._p2p_allreduce_chunk_acc(flats, offset, end, scratches, accs, streams)

        self._run_gpu_transfer(streams, keep_alive, _issue)

    def _p2p_allreduce_chunk_fused(
        self,
        flats: Sequence[torch.Tensor],
        offset: int,
        end: int,
        scratches: list[list[torch.Tensor]],
        streams: Sequence[torch.cuda.Stream],
        fused_sum: Callable[[torch.Tensor, Sequence[torch.Tensor]], None],
    ) -> None:
        group_size = len(flats)
        length = end - offset
        pulled_events: list[torch.cuda.Event] = []
        # Phase 1: copy peer slices into private rank slots before any output writes occur.
        for r in range(group_size):
            with _on_stream(streams[r]):
                slot_idx = 0
                for k in range(group_size):
                    if k != r:
                        _copy_d2d(
                            scratches[r][slot_idx][:length],
                            flats[k][offset:end],
                            streams[r],
                            streams[k],
                        )
                        slot_idx += 1
                ev = torch.cuda.Event()
                ev.record(streams[r])
                pulled_events.append(ev)
        # Phase 2: fence peer reads, then sum in ascending rank order 0..W-1.
        for r in range(group_size):
            with _on_stream(streams[r]):
                for q in range(group_size):
                    if q != r:
                        streams[r].wait_event(pulled_events[q])
                ordered: list[torch.Tensor] = []
                slot_idx = 0
                for k in range(group_size):
                    if k == r:
                        ordered.append(flats[r][offset:end])
                    else:
                        ordered.append(scratches[r][slot_idx][:length])
                        slot_idx += 1
                fused_sum(flats[r][offset:end], ordered)

    def _p2p_allreduce_chunk_acc(
        self,
        flats: Sequence[torch.Tensor],
        offset: int,
        end: int,
        scratches: list[list[torch.Tensor]],
        accs: list[torch.Tensor],
        streams: Sequence[torch.cuda.Stream],
    ) -> None:
        group_size = len(flats)
        length = end - offset
        pulled_events: list[torch.cuda.Event] = []
        # Phase 1: accumulate into private acc in ascending rank order 0..W-1.
        for r in range(group_size):
            self._reduce_for_rank(
                r,
                flats,
                offset,
                end,
                scratches[r],
                accs[r],
                streams,
                use_fused=False,
                fused_sum=None,
                out_slice=None,
            )
            with _on_stream(streams[r]):
                ev = torch.cuda.Event()
                ev.record(streams[r])
                pulled_events.append(ev)
        # Phase 2: wait until all peers finished reading flats[r], then store result.
        for r in range(group_size):
            with _on_stream(streams[r]):
                for q in range(group_size):
                    if q != r:
                        streams[r].wait_event(pulled_events[q])
                flats[r][offset:end].copy_(accs[r][:length])

    # ------------------------------------------------------------------
    # Host-staged CUDA execution (single bounded staging buffer)
    # ------------------------------------------------------------------

    def _host_reduce_layout(
        self,
        group_size: int,
        dtype: torch.dtype,
        total_elems: int,
    ) -> tuple[int, int, torch.dtype]:
        acc_dtype = torch.float32 if dtype in _SUM_FLOAT_DTYPES else dtype
        elem_bytes = dtype.itemsize
        acc_bytes = acc_dtype.itemsize
        if group_size * elem_bytes + acc_bytes <= self._staging_bytes:
            slots = group_size
        elif elem_bytes + acc_bytes <= self._staging_bytes:
            slots = 1
        else:
            raise ValueError(
                f"staging_bytes ({self._staging_bytes}) too small for host_staged reduction of dtype {dtype}",
            )
        chunk_elems = min(total_elems, self._staging_bytes // (slots * elem_bytes + acc_bytes))
        return chunk_elems, slots, acc_dtype

    def _acquire_stage_and_lease(self, min_bytes: int, max_bytes: int) -> tuple[torch.Tensor, PinLease]:
        with self._lock:
            dma_base = self._stage.checkout(min_bytes, max_bytes)
        try:
            lease = host_pin_manager.acquire([dma_base], pin=True)
        except BaseException:
            with self._lock:
                self._stage.release(dma_base)
            raise
        return dma_base, lease

    def _execute_host_copies(self, copy_specs: Sequence[_CopySpec], group_devices: tuple[torch.device, ...]) -> None:
        for copy_spec in copy_specs:
            self._host_copy_spec(copy_spec, group_devices)

    def _host_copy_spec(self, copy_spec: _CopySpec, group_devices: tuple[torch.device, ...]) -> None:
        nbytes = copy_spec.src_bytes.numel()
        if nbytes == 0 or not copy_spec.dsts:
            return
        chunk = min(nbytes, self._staging_bytes)
        dma_base, lease = self._acquire_stage_and_lease(chunk, self._staging_bytes)
        try:
            src_stream = self._device_stream(group_devices[copy_spec.src_rank])
            dsts = [(self._device_stream(group_devices[r]), dst_bytes) for r, dst_bytes in copy_spec.dsts]
            all_streams = [src_stream, *(s for s, _ in dsts)]
            keep_alive = [copy_spec.src_bytes, *(t for _, t in dsts), dma_base]
            stage = dma_base.detach()[:chunk]
            non_blocking = lease.pageable_bytes == 0
        except BaseException:
            try:
                lease.close()
            finally:
                with self._lock:
                    self._stage.release(dma_base)
            raise
        self._run_gpu_transfer(
            all_streams,
            keep_alive,
            lambda s_b=copy_spec.src_bytes, s_s=src_stream, ds=dsts, stg=stage, nb=non_blocking: (
                self._run_host_copy_chunks(s_b, s_s, ds, stg, non_blocking=nb)
            ),
            lease=lease,
            dma_base=dma_base,
        )

    def _run_host_copy_chunks(
        self,
        src_bytes: torch.Tensor,
        src_stream: torch.cuda.Stream,
        dsts: Sequence[tuple[torch.cuda.Stream, torch.Tensor]],
        stage: torch.Tensor,
        *,
        non_blocking: bool,
    ) -> None:
        nbytes = src_bytes.numel()
        chunk = stage.numel()
        for offset in range(0, nbytes, chunk):
            end = min(nbytes, offset + chunk)
            buf = stage[: end - offset]
            with _on_stream(src_stream):
                buf.copy_(src_bytes[offset:end], non_blocking=non_blocking)
            src_stream.synchronize()
            for dst_stream, dst_tensor in dsts:
                with _on_stream(dst_stream):
                    dst_tensor[offset:end].copy_(buf, non_blocking=non_blocking)
            _sync_streams(dst_stream for dst_stream, _ in dsts)

    def _execute_host_reduce(
        self,
        specs: Sequence[_ReduceSpec],
        plans: Sequence[tuple[int, int, torch.dtype]],
        group_devices: tuple[torch.device, ...],
    ) -> None:
        plan_iter = iter(plans)
        for spec in specs:
            if len(spec.inputs) == 1:
                dev = group_devices[spec.outputs[0][0]]
                stream = self._device_stream(dev)
                self._run_gpu_transfer(
                    (stream,),
                    _collect_reduce_tensors(spec),
                    lambda s=stream, out=spec.outputs[0][1], src=spec.inputs[0]: self._copy_on_stream(s, out, src),
                )
            else:
                chunk_elems, num_slots, acc_dtype = next(plan_iter)
                self._host_reduce_spec(spec, group_devices, chunk_elems, num_slots, acc_dtype)

    def _host_reduce_spec(
        self,
        spec: _ReduceSpec,
        group_devices: tuple[torch.device, ...],
        chunk_elems: int,
        num_slots: int,
        acc_dtype: torch.dtype,
    ) -> None:
        dtype = spec.inputs[0].dtype
        n = spec.inputs[0].numel()
        inputs = [(self._device_stream(group_devices[r]), t) for r, t in enumerate(spec.inputs)]
        outputs = [(self._device_stream(group_devices[r]), out) for r, out in spec.outputs]
        acc_in_dma = dtype == acc_dtype
        dma_elem_count = chunk_elems * (num_slots + (1 if acc_in_dma else 0))
        dma_bytes = dma_elem_count * dtype.itemsize
        max_dma_bytes = self._staging_bytes if acc_in_dma else self._staging_bytes - chunk_elems * acc_dtype.itemsize
        dma_base, lease = self._acquire_stage_and_lease(dma_bytes, max_dma_bytes)
        try:
            dma_view = dma_base.detach()[:dma_bytes].view(dtype)
            peer_slots = [dma_view[i * chunk_elems : (i + 1) * chunk_elems] for i in range(num_slots)]
            acc = (
                dma_view[num_slots * chunk_elems : (num_slots + 1) * chunk_elems]
                if acc_in_dma
                else torch.empty(chunk_elems, dtype=acc_dtype, device="cpu")
            )
        except BaseException:
            try:
                lease.close()
            finally:
                with self._lock:
                    self._stage.release(dma_base)
            raise

        all_streams = [*(s for s, _ in inputs), *(s for s, _ in outputs)]
        keep_alive = [*_collect_reduce_tensors(spec), dma_base, *([acc] if not acc_in_dma else [])]

        def _issue() -> None:
            non_blocking = lease.pageable_bytes == 0
            for offset in range(0, n, chunk_elems):
                end = min(n, offset + chunk_elems)
                self._host_reduce_chunk(
                    inputs, outputs, peer_slots, acc, offset, end,
                    dtype=dtype, acc_dtype=acc_dtype, non_blocking=non_blocking,
                )

        self._run_gpu_transfer(all_streams, keep_alive, _issue, lease=lease, dma_base=dma_base)

    def _host_reduce_chunk(
        self,
        inputs: Sequence[tuple[torch.cuda.Stream, torch.Tensor]],
        outputs: Sequence[tuple[torch.cuda.Stream, torch.Tensor]],
        peer_slots: list[torch.Tensor],
        acc: torch.Tensor,
        offset: int,
        end: int,
        *,
        dtype: torch.dtype,
        acc_dtype: torch.dtype,
        non_blocking: bool,
    ) -> None:
        length = end - offset
        acc_buf = acc[:length]
        if len(peer_slots) == len(inputs):
            for (in_stream, in_tensor), slot in zip(inputs, peer_slots, strict=True):
                with _on_stream(in_stream):
                    slot[:length].copy_(in_tensor[offset:end], non_blocking=non_blocking)
            _sync_streams(in_stream for in_stream, _ in inputs)
            acc_buf.copy_(peer_slots[0][:length])
            for slot in peer_slots[1:]:
                acc_buf.add_(slot[:length])
        else:
            slot = peer_slots[0][:length]
            for idx, (in_stream, in_tensor) in enumerate(inputs):
                with _on_stream(in_stream):
                    slot.copy_(in_tensor[offset:end], non_blocking=non_blocking)
                in_stream.synchronize()
                if idx == 0:
                    acc_buf.copy_(slot)
                else:
                    acc_buf.add_(slot)

        send_buf = acc_buf if dtype == acc_dtype else peer_slots[0][:length]
        if dtype != acc_dtype:
            send_buf.copy_(acc_buf)
        for out_stream, out_tensor in outputs:
            with _on_stream(out_stream):
                out_tensor[offset:end].copy_(send_buf, non_blocking=non_blocking)
        _sync_streams(out_stream for out_stream, _ in outputs)
