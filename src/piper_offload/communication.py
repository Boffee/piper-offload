"""Experimental, blocking host-relay process group for DTensor inference.

Call ``register_relay_backend()`` in each worker before initializing a
``piper_relay`` process group. It supports SUM all-reduce in FP32, FP16 and
BF16, plus broadcast, scatter and all-gather of contiguous tensors. Select
``transport="shared"`` for same-machine transfers through shared host slots
and GPU-local summation; Gloo then carries only control messages and reductions
of CPU tensors. The default ``transport="gloo"`` carries payloads through CPU
Gloo as well. Gloo never receives an accelerator tensor.

Collectives stream through a reusable, bounded CPU buffer and complete all
device copies before returning, including when ``async_op=True``. No offloader,
NCCL, or compiled extension is required. Gloo pipelining is opt-in; shared
transfers use two outgoing slots per rank. Public calls remain blocking.
"""

import threading
from collections.abc import Generator, Sequence
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import timedelta
from itertools import pairwise

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import AllgatherOptions, _DistributedBackendOptions

from ._relay_shared import BUFFERS_PER_RANK, COPY_DTYPES, REDUCTION_DTYPES, SharedRelay, shared_slot_bytes
from ._relay_staging import HostChunk, TransferPipeline, chunk_ranges, make_host_chunk
from .host_memory import HostMemoryManager

_BACKEND_NAME = "piper_relay"
_registration_lock = threading.Lock()


@dataclass(frozen=True)
class RelayOptions:
    """Pass as ``pg_options`` to ``init_process_group`` or ``new_group``.

    ``staging_bytes`` bounds this group's reusable CPU allocation, including
    reduction accumulators, but excludes Gloo workspace and caller tensors.
    All ranks must agree. Gloo staging is allocated lazily; shared staging is
    mapped at group initialization. Both are released on group shutdown.
    ``memory_manager`` owns the page-rounded pin budget. Pass the same manager
    used by host captures to share that budget; otherwise the group owns one.
    For Gloo transfers, ``pipeline_buffers=2`` or ``3`` divides the same allocation
    into chunk sets to overlap pinned CUDA/HIP copies with CPU communication.
    The default, 1, is serial. CPU and pageable copies keep the same chunk size
    and run serially.

    ``transport="shared"`` instead maps one ``staging_bytes`` arena across all
    ranks on the same machine. Copy collectives always use two outgoing slots
    per rank, reused across peer rounds. Every process registers its full
    mapping under its own pin budget; physical payload storage is shared once.
    GPU reductions use the same two slots plus at most three slot sizes of
    reusable device scratch per rank, independent of tensor size. FP16/BF16
    accumulate in FP32. CPU tensor reductions use Gloo and borrow this rank's
    region. ``pipeline_buffers`` only affects the Gloo path. The default
    transport remains ``"gloo"``.
    """

    staging_bytes: int = 8 * 1024 * 1024
    pipeline_buffers: int = 1
    transport: str = "gloo"
    memory_manager: HostMemoryManager = field(default_factory=HostMemoryManager, compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.staging_bytes) is not int or self.staging_bytes <= 0 or self.staging_bytes % 16:
            raise ValueError("piper_relay staging_bytes must be a positive integer multiple of 16")
        if type(self.pipeline_buffers) is not int or self.pipeline_buffers not in (1, 2, 3):
            raise ValueError("piper_relay pipeline_buffers must be 1, 2, or 3")
        if self.transport not in ("gloo", "shared"):
            raise ValueError("piper_relay transport must be 'gloo' or 'shared'")

    def _validate_group_size(self, size: int) -> None:
        minimum = 16 * max(3, size + 1) * self.pipeline_buffers
        if self.staging_bytes < minimum:
            raise ValueError("piper_relay staging_bytes must allow 16 bytes per staging slot")
        if self.transport == "shared" and shared_slot_bytes(self.staging_bytes, size) * BUFFERS_PER_RANK < minimum:
            raise ValueError(
                "piper_relay shared staging_bytes must allow 16 bytes per staging slot in each rank region",
            )


def register_relay_backend() -> None:
    """Register the experimental ``piper_relay`` backend in this process.

    Registration is idempotent and does not initialize an accelerator or a
    process group. Each spawned worker must call this function. Initialize
    with ``backend="piper_relay"`` after selecting the worker's accelerator
    with ``torch.cuda.set_device``; omit ``device_id`` from initialization.
    The Python process group does not implement eager accelerator connection.
    """
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("piper_relay requires a PyTorch build with CPU Gloo support")
    with _registration_lock:
        plugin = dist.Backend._plugins.get(_BACKEND_NAME.upper())
        if plugin is not None:
            if plugin.creator_fn is not _create_relay_group:
                raise RuntimeError("A different backend is already registered as piper_relay")
            return
        dist.Backend.register_backend(
            _BACKEND_NAME,
            _create_relay_group,
            extended_api=True,
            devices=["cpu", "cuda"],
        )


def _create_relay_group(backend: _DistributedBackendOptions, options: RelayOptions | None) -> dist.ProcessGroup:
    if options is None:
        options = RelayOptions()
    if not isinstance(options, RelayOptions):
        raise TypeError("piper_relay pg_options must be RelayOptions")
    return _RelayProcessGroup(backend.store, backend.group_rank, backend.group_size, backend.timeout, options)


class _RelayProcessGroup(dist.ProcessGroup):
    # Implement c10d's tensor-list/options virtual methods. Its Python
    # convenience overloads (tensor/root/timeout) are not backend entry points.
    def __init__(self, store: dist.Store, rank: int, size: int, timeout: timedelta, options: RelayOptions) -> None:
        # The rank/size overload constructs the Python trampoline. The
        # store/rank/size overload cannot construct subclasses.
        super().__init__(rank, size)  # type: ignore[call-arg]
        self._memory_manager = options.memory_manager
        self._collective_lock = threading.Lock()
        self._staging_bytes = options.staging_bytes
        self._pipeline_buffers = options.pipeline_buffers
        self._staging_buffer: torch.Tensor | None = None
        self._pipeline: TransferPipeline | None = None
        self._shared: SharedRelay | None = None
        self._closed = False
        options._validate_group_size(size)
        # Different chunk sizes would issue incompatible Gloo collectives.
        settings = dist.PrefixStore("piper_relay_options", store)
        settings.set(str(rank), f"{self._staging_bytes}:{self._pipeline_buffers}:{options.transport}")
        keys = [str(r) for r in range(size)]
        settings.wait(keys, timeout)
        if len({settings.get(key) for key in keys}) != 1:
            raise ValueError("piper_relay staging_bytes, pipeline_buffers and transport must match across all ranks")
        self._cpu_group = dist.ProcessGroupGloo(store, rank, size, timeout)
        # Give the base process group ownership of Gloo for shutdown and CPU
        # control operations. Deliberately do not register Gloo for CUDA.
        self._register_backend(
            torch.device("cpu"),
            dist.ProcessGroup.BackendType.GLOO,
            self._cpu_group,
        )
        if options.transport == "shared":
            self._shared = SharedRelay(
                store, self._cpu_group, rank, size, self._staging_bytes, timeout,
            )

    def getBackendName(self) -> str:  # noqa: N802 -- PyTorch trampoline name
        return _BACKEND_NAME

    def shutdown(self) -> None:
        with self._collective_lock:
            if self._closed:
                return
            self._closed = True
            try:
                super().shutdown()
            finally:
                self._staging_buffer = None
                self._pipeline = None
                self._shared = None

    @contextmanager
    def _collective(self, tensor: torch.Tensor) -> Generator[None]:
        device = tensor.device
        with self._collective_lock, torch.no_grad(), (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        ):
            if self._closed:
                raise RuntimeError("piper_relay process group is shut down")
            if self._shared is not None and self._shared.broken:
                raise RuntimeError("shared relay failed previously; destroy and recreate the process group")
            if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("piper_relay blocking transfers do not support CUDA graph capture")
            yield

    @contextmanager
    def _buffer(self, device: torch.device) -> Generator[tuple[torch.Tensor, bool]]:
        if self._shared is not None:
            # CPU reductions use only this rank's region. No second host
            # payload allocation is necessary.
            owner = self._shared.buffer
            region = self._shared.local_buffer()
        else:
            if self._staging_buffer is None:
                self._staging_buffer = torch.empty(self._staging_bytes, dtype=torch.uint8, device="cpu")
            owner = region = self._staging_buffer
        # Lease the persistent owner, never temporary views: view destruction
        # must not retire the reusable registration. Idle pins remain evictable.
        with self._memory_manager.acquire([owner]) if device.type == "cuda" else nullcontext() as lease:
            yield region, lease is not None and lease.pageable_bytes == 0

    def _slot_bytes(self, slots: int) -> int:
        size = self._staging_bytes if self._shared is None else self._shared.capacity * self._shared.buffer_count
        return size // (16 * slots * self._pipeline_buffers) * 16

    def allreduce(  # type: ignore[override]
        self,
        tensors: list[torch.Tensor],
        opts: dist.AllreduceOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else dist.AllreduceOptions()
        tensor = _single_tensor(tensors)
        _validate_reduction([tensor], options)
        with self._collective(tensor):
            self._reduce(tensor, options)
        return _CompletedWork([tensor])

    def allreduce_coalesced(
        self,
        tensors: list[torch.Tensor],
        opts: dist.AllreduceCoalescedOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else dist.AllreduceCoalescedOptions()
        _validate_reduction(tensors, options)
        _validate_disjoint(tensors)
        # PyTorch's Python bindings do not accept coalesced options in the
        # single-tensor Gloo overload, despite their common C++ fields.
        single_options = dist.AllreduceOptions()
        single_options.reduceOp = options.reduceOp
        single_options.timeout = options.timeout
        single_options.asyncOp = options.asyncOp
        with self._collective(tensors[0]):
            for tensor in tensors:
                self._reduce(tensor, single_options)
        return _CompletedWork(tensors)

    def _reduce(self, tensor: torch.Tensor, options: dist.AllreduceOptions) -> None:
        if self._shared is not None and tensor.device.type == "cuda":
            self._shared.reduce(tensor, self._memory_manager)
            return
        low_precision = tensor.dtype != torch.float32
        with closing(self._copy_chunks(
            [tensor], [tensor], slots=3 if low_precision else 1, inplace=True, stage_cpu=low_precision,
        )) as chunks:
            for chunk in chunks:
                host = chunk.inputs[0].view(tensor.dtype)
                if low_precision:
                    assert chunk.scratch is not None
                    accumulator = chunk.scratch[:host.numel() * 4].view(torch.float32)
                    accumulator.copy_(host)
                else:
                    accumulator = host
                # PyTorch's Gloo stubs omit the collective bindings.
                self._cpu_group.allreduce([accumulator], options).wait()  # type: ignore[attr-defined]
                if low_precision:
                    host.copy_(accumulator)

    def broadcast(  # type: ignore[override]
        self,
        tensors: list[torch.Tensor],
        opts: dist.BroadcastOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else dist.BroadcastOptions()
        tensor = _single_tensor(tensors)
        _validate_tensors([tensor])
        self._validate_root(options.rootRank)
        if options.rootTensor != 0:
            raise ValueError("piper_relay broadcast requires rootTensor=0")
        sources = [tensor] if self.rank() == options.rootRank else []
        with self._collective(tensor):
            if self._shared is not None:
                self._shared.copy("broadcast", sources, [tensor], options.rootRank, self._memory_manager)
            else:
                with closing(self._copy_chunks(sources, [tensor], slots=2)) as chunks:
                    for chunk in chunks:
                        if chunk.inputs:
                            chunk.outputs[0].copy_(chunk.inputs[0])
                        self._cpu_group.broadcast(chunk.outputs, options).wait()  # type: ignore[attr-defined]
        return _CompletedWork([tensor])

    def scatter(  # type: ignore[override]
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: dist.ScatterOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else dist.ScatterOptions()
        output = _single_tensor(output_tensors)
        self._validate_root(options.rootRank)
        sources: list[torch.Tensor] = []
        if self.rank() == options.rootRank:
            if len(input_tensors) != 1 or len(input_tensors[0]) != self.size():
                raise ValueError("piper_relay scatter root must provide one tensor per rank")
            sources = input_tensors[0]
        elif input_tensors:
            raise ValueError("piper_relay scatter inputs belong only on the root rank")
        _validate_tensors([output, *sources])
        for source_rank, tensor in enumerate(sources):
            _validate_pair(output, tensor, tensor.numel(), same_shape=True)
            # Gloo may copy the root's result while another rank's send still
            # reads its source. Only an exact alias of the root's own slice
            # is safe; enforce the same contract for CPU and accelerator use.
            if _overlap(output, tensor) and (
                source_rank != options.rootRank or output.data_ptr() != tensor.data_ptr()
            ):
                raise ValueError("piper_relay scatter output must not overlap inputs except its exact root slice")
        with self._collective(output):
            if self._shared is not None:
                self._shared.copy("scatter", sources, [output], options.rootRank, self._memory_manager)
            else:
                with closing(self._copy_chunks(sources, [output], slots=self.size() + 1)) as chunks:
                    for chunk in chunks:
                        self._cpu_group.scatter(  # type: ignore[attr-defined]
                            chunk.outputs, [chunk.inputs] if sources else [], options,
                        ).wait()
        return _CompletedWork([output])

    def allgather(  # type: ignore[override]
        self,
        output_tensors: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: AllgatherOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else AllgatherOptions()
        source = _single_tensor(input_tensors)
        if len(output_tensors) != 1 or len(output_tensors[0]) != self.size():
            raise ValueError("piper_relay all-gather requires one output tensor per rank")
        destinations = output_tensors[0]
        _validate_tensors([source, *destinations])
        _validate_disjoint(destinations)
        for output in destinations:
            _validate_pair(output, source, source.numel(), same_shape=True)
        self._validate_gather_aliases(source, destinations)
        with self._collective(source):
            self._gather(source, destinations, options)
        return _CompletedWork(destinations, nested=True)

    def all_gather_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch parameter name
        opts: AllgatherOptions | None = None,
    ) -> dist.Work:
        return self.all_gather_single_coalesced([output], [input], opts)

    def _allgather_base(
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch parameter name
        opts: AllgatherOptions | None = None,
    ) -> dist.Work:
        return self.all_gather_single(output, input, opts)

    def all_gather_single_coalesced(
        self,
        output_lists: list[torch.Tensor],
        input_list: list[torch.Tensor],
        opts: AllgatherOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else AllgatherOptions()
        if len(output_lists) != len(input_list) or not input_list:
            raise ValueError("piper_relay all-gather requires matching nonempty input/output lists")
        _validate_tensors([*input_list, *output_lists])
        _validate_disjoint(output_lists)
        for i, output in enumerate(output_lists):
            for j, source in enumerate(input_list):
                if i != j and _overlap(output, source):
                    raise ValueError("piper_relay batched outputs must not overlap another input")
        # Validate the whole batch before starting communication or mutation.
        destinations: list[list[torch.Tensor]] = []
        for output, source in zip(output_lists, input_list, strict=True):
            _validate_pair(output, source, source.numel() * self.size())
            views = list(output.view(self.size(), source.numel()).unbind(0))
            self._validate_gather_aliases(source, views)
            destinations.append(views)
        with self._collective(input_list[0]):
            for source, outputs in zip(input_list, destinations, strict=True):
                self._gather(source, outputs, options)
        return _CompletedWork(output_lists)

    def _validate_gather_aliases(self, source: torch.Tensor, outputs: list[torch.Tensor]) -> None:
        for rank, output in enumerate(outputs):
            if _overlap(output, source) and (rank != self.rank() or output.data_ptr() != source.data_ptr()):
                raise ValueError("piper_relay all-gather input may overlap only its exact local output slice")

    def _gather(self, source: torch.Tensor, outputs: list[torch.Tensor], options: AllgatherOptions) -> None:
        if self._shared is not None:
            self._shared.copy("allgather", [source], outputs, -1, self._memory_manager)
            return
        with closing(self._copy_chunks([source], outputs)) as chunks:
            for chunk in chunks:
                self._cpu_group.allgather([chunk.outputs], chunk.inputs, options).wait()  # type: ignore[attr-defined]

    def _copy_chunks(
        self, inputs: list[torch.Tensor], outputs: list[torch.Tensor], *, slots: int | None = None,
        inplace: bool = False, stage_cpu: bool = False,
    ) -> Generator[HostChunk]:
        sources, destinations = [_bytes(t) for t in inputs], [_bytes(t) for t in outputs]
        # The root/non-root scatter plans must use identical chunk sizes.
        slots = slots if slots is not None else len(inputs) + len(outputs)
        capacity = self._slot_bytes(slots)
        device = outputs[0].device
        if device.type == "cpu" and not stage_cpu:
            for start, end in chunk_ranges(destinations[0].numel(), capacity):
                yield HostChunk([t[start:end] for t in sources], [t[start:end] for t in destinations])
            return
        with self._buffer(device) as (buffer, pinned):
            if pinned and self._pipeline_buffers > 1 and destinations[0].numel() > capacity:
                if self._pipeline is None or self._pipeline.device != device:
                    self._pipeline = TransferPipeline(device, self._pipeline_buffers)
                # Closing the inner generator drains DMA before _buffer releases
                # its lease, including when the caller's CPU collective raises.
                with closing(self._pipeline.chunks(sources, destinations, buffer, capacity, slots, inplace)) as chunks:
                    yield from chunks
                return
            # CPU and pageable fallback retain the configured chunk size so
            # ranks with different registration outcomes issue matching calls.
            region = buffer[:slots * capacity]
            for start, end in chunk_ranges(destinations[0].numel(), capacity):
                chunk = make_host_chunk(region, capacity, end - start, len(inputs), len(outputs), inplace)
                for source, host in zip(sources, chunk.inputs, strict=True):
                    host.copy_(source[start:end], non_blocking=False)
                yield chunk
                for destination, host in zip(destinations, chunk.outputs, strict=True):
                    destination[start:end].copy_(host, non_blocking=False)

    def _validate_root(self, root: int) -> None:
        if not 0 <= root < self.size():
            raise ValueError("piper_relay root rank is outside the process group")


class _CompletedWork(dist.Work):
    """Expose an already completed collective to Python and c10d callers."""

    def __init__(self, tensors: list[torch.Tensor], *, nested: bool = False) -> None:
        super().__init__()
        self._tensors = tensors
        self._future: torch.futures.Future[list[torch.Tensor] | list[list[torch.Tensor]]] = torch.futures.Future()
        self._future.set_result([tensors] if nested else tensors)

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        del timeout
        return True

    def is_completed(self) -> bool:
        return True

    def is_success(self) -> bool:
        return True

    def get_future(self) -> torch.futures.Future[list[torch.Tensor] | list[list[torch.Tensor]]]:
        return self._future

    def result(self) -> list[torch.Tensor]:
        return self._tensors

    def synchronize(self) -> None:
        pass

    def block_current_stream(self) -> None:
        pass


def _validate_reduction(tensors: list[torch.Tensor], options: dist.AllreduceOptions) -> None:
    if options.reduceOp != dist.ReduceOp.SUM:
        raise NotImplementedError("piper_relay currently supports only SUM all-reduce")
    _validate_tensors(tensors)
    for tensor in tensors:
        if tensor.dtype not in REDUCTION_DTYPES:
            raise NotImplementedError("piper_relay SUM all-reduce supports FP32, FP16 and BF16")


def _single_tensor(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) != 1:
        raise NotImplementedError("piper_relay requires exactly one local tensor for this collective")
    return tensors[0]


def _validate_tensors(tensors: Sequence[torch.Tensor]) -> None:
    if not tensors:
        raise ValueError("piper_relay requires a nonempty tensor list")
    for tensor in tensors:
        _validate_tensor(tensor)
        if tensor.device != tensors[0].device:
            raise ValueError("piper_relay requires one local device per collective")


def _validate_tensor(tensor: torch.Tensor) -> None:
    if type(tensor) is not torch.Tensor:
        raise TypeError("piper_relay expects a plain local tensor, not a tensor subclass")
    if tensor.device.type not in ("cpu", "cuda"):
        raise NotImplementedError("piper_relay supports only CPU and CUDA/HIP tensors")
    if tensor.dtype not in COPY_DTYPES:
        raise NotImplementedError(f"piper_relay does not support dtype {tensor.dtype}")
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise NotImplementedError("piper_relay requires a contiguous strided tensor")


def _validate_pair(
    output: torch.Tensor, source: torch.Tensor, expected_numel: int, *, same_shape: bool = False,
) -> None:
    if output.dtype != source.dtype or output.numel() != expected_numel:
        raise ValueError("piper_relay output must have the expected element count and input dtype")
    if same_shape and output.shape != source.shape:
        raise ValueError("piper_relay list collectives require equal tensor shapes")


def _overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(left.nbytes and right.nbytes) and (
        left.data_ptr() < right.data_ptr() + right.nbytes
        and right.data_ptr() < left.data_ptr() + left.nbytes
    )


def _validate_disjoint(tensors: Sequence[torch.Tensor]) -> None:
    ordered = sorted((tensor for tensor in tensors if tensor.nbytes), key=lambda tensor: tensor.data_ptr())
    for left, right in pairwise(ordered):
        if _overlap(left, right):
            raise ValueError("piper_relay collective outputs must not overlap")


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Transport payload bits unchanged, including BF16 and integer metadata."""
    return tensor.reshape(-1).view(torch.uint8)


__all__ = ["RelayOptions", "register_relay_backend"]
