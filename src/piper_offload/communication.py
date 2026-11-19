"""Experimental host-relay process group for DTensor inference.

Call ``register_relay_backend()`` in each worker before initializing a
``piper_relay`` process group. The application supplies its launcher, device
mesh, and sharding. With ``local_rank`` provided by the launcher::

    import torch
    import torch.distributed as dist
    from piper_offload.communication import RelayOptions, register_relay_backend

    torch.cuda.set_device(local_rank)
    register_relay_backend()
    dist.init_process_group(
        "piper_relay",
        pg_options=RelayOptions(staging_bytes=8 * 1024 * 1024),
    )

Omit ``device_id``; this backend does not implement eager accelerator connection.
Select ``transport="shared"`` for same-machine transfers through shared host slots
and GPU-local summation; on x86_64/AMD64, Gloo then carries only barriers and
reductions of CPU tensors (other architectures also use Gloo for control
messages). The default ``transport="gloo"`` carries payloads through CPU Gloo as
well. Gloo never receives an accelerator tensor.

Collectives stream through a reusable, bounded CPU buffer. Each group runs its
collectives, including ``barrier``, on one worker thread in issue order. With
``async_op=True`` a call returns at once; until ``wait()`` returns, the caller
must not read the outputs or write the inputs or outputs, from any stream.
``get_future()`` also waits, so the future's callbacks run on the calling thread.
Without it, the call returns once the collective has completed, and the outputs
are then ready for every stream. A collective on accelerator tensors starts once
the caller's current stream has produced its inputs and moves its data as a
foreground transfer, which background copies such as block prefetch wait for
(see ``piper_offload._transfer_priority``). No offloader, NCCL, or compiled
extension is required. Gloo transfers default to two pipeline buffers; shared
transfers use three outgoing slots per rank.

SUM all-reduce and reduce-scatter support FP32, FP16 and BF16, accumulating
reduced precision in FP32. Broadcast, scatter and equal-size all-gather also
preserve FP64, bool, uint8, signed integers, and complex64/complex128.
All-reduce, reduce-scatter and all-gather include coalesced variants; all-gather
and reduce-scatter support tensor-list and single-buffer forms. Inputs and
outputs must be contiguous plain tensors on one local device. Outputs must not
overlap, except that an all-gather input may occupy its exact own-rank output
slice, a reduce-scatter output may occupy its exact own-rank input slice, and a
scatter root may receive into its exact source slice. Coalesced outputs must not
overwrite other inputs.

``RelayOptions`` describes staging and pin budgets. If the staging allocation
exceeds the pin budget or registration reports invalid-value, copies run
synchronously through pageable storage. All-to-all, non-SUM reductions,
accelerator point-to-point transfers, and CUDA graph capture are unsupported.
Direct C++ functional all-gather ``out`` calls bypass the Python override and
are unsupported for accelerator tensors.

After a shared-exchange failure, outputs may be partial; destroy and recreate
the process group. Windows GPU, ROCm, and distinct-GPU behavior need validation
on their target hardware.
"""

import threading
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    AllgatherOptions,
    ReduceOptions,
    ReduceScatterOptions,
    _DistributedBackendOptions,
)

from ._relay_shared import BUFFERS_PER_RANK, COPY_DTYPES, REDUCTION_DTYPES, SharedRelay, shared_slot_bytes
from ._relay_staging import HostChunk, TransferPipeline, chunk_ranges, make_host_chunk
from ._transfer_priority import foreground_transfer
from .pin_manager import host_pin_manager

_BACKEND_NAME = "piper_relay"
_GLOO_EXCHANGE_TAG = 2
_registration_lock = threading.Lock()


def _sum_pair_in_fp32(
    output: torch.Tensor, local: torch.Tensor, peer: torch.Tensor, scratch_f32: torch.Tensor, rank: int,
) -> None:
    """Sum ``local`` and ``peer`` in rank order through ``scratch_f32``."""
    first, second = (local, peer) if rank == 0 else (peer, local)
    step = scratch_f32.numel()
    for start in range(0, output.numel(), step):
        end = min(start + step, output.numel())
        acc = scratch_f32[:end - start]
        acc.copy_(first[start:end])
        acc.add_(second[start:end])
        output[start:end].copy_(acc)


@dataclass(frozen=True)
class RelayOptions:
    """Pass as ``pg_options`` to ``init_process_group`` or ``new_group``.

    ``staging_bytes`` defaults to 8 MiB per rank with Gloo or per group with
    shared transport. It bounds reusable CPU allocation, including reduction
    accumulators, but excludes Gloo workspace, caller tensors, and allocator
    overhead. All ranks within a group must agree on its options, including
    when creating subgroups. Gloo staging is allocated lazily; shared staging
    is mapped at group initialization. Both are released on group shutdown.
    Pinning remains subject to ``host_pin_manager``'s separate page-rounded budget.
    For Gloo transfers, ``pipeline_buffers=2`` (the default) or ``3`` divides
    the same allocation into chunk sets to overlap pinned CUDA/HIP copies with
    CPU communication; ``1`` runs chunks serially. CPU and pageable copies keep
    the same chunk size and run serially.

    ``transport="shared"`` instead maps one ``staging_bytes`` arena across all
    ranks on the same machine. Copy collectives always use three outgoing slots
    per rank, reused across peer rounds; all-gather and broadcast publish each
    chunk once for all peers, and coalesced shared batches pack into one transfer.
    Every process registers its full payload mapping under its own pin budget;
    physical payload storage is shared once. On x86_64/AMD64, rank agreement and
    chunk signaling use a separate small unpinned shared-memory control mapping;
    other architectures use Gloo control messages. GPU reductions use the same
    three slots plus at most three slot sizes of reusable device scratch per
    rank, at most ``staging_bytes / world_size``, independent of tensor size.
    FP16/BF16 accumulate in FP32. CPU tensor reductions use Gloo and borrow this
    rank's region. ``pipeline_buffers`` affects only the Gloo path and does not
    affect shared slot sizing or minimum ``staging_bytes``. The default
    transport remains ``"gloo"``.
    """

    staging_bytes: int = 8 * 1024 * 1024
    pipeline_buffers: int = 2
    transport: str = "gloo"

    def __post_init__(self) -> None:
        if type(self.staging_bytes) is not int or self.staging_bytes <= 0 or self.staging_bytes % 16:
            raise ValueError("piper_relay staging_bytes must be a positive integer multiple of 16")
        if type(self.pipeline_buffers) is not int or self.pipeline_buffers not in (1, 2, 3):
            raise ValueError("piper_relay pipeline_buffers must be 1, 2, or 3")
        if self.transport not in ("gloo", "shared"):
            raise ValueError("piper_relay transport must be 'gloo' or 'shared'")

    def _validate_group_size(self, size: int) -> None:
        buffers = 1 if self.transport == "shared" else self.pipeline_buffers
        minimum = 16 * max(3, size + 1) * buffers
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
        super().__init__(rank, size)
        self._staging_bytes = options.staging_bytes
        self._pipeline_buffers = options.pipeline_buffers
        self._staging_buffer: torch.Tensor | None = None
        self._pipeline: TransferPipeline | None = None
        self._shared: SharedRelay | None = None
        self._closed = False  # shutdown() released Gloo and staging
        # Runs every collective in issue order and refuses new ones once shut
        # down. Only this thread touches the staging, shared slots, Gloo
        # transport and stream.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piper-relay")
        self._stream: torch.cuda.Stream | None = None
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
        # Gloo is also the default backend, which PyTorch's per-group hooks
        # resolve: 2.15 attaches its flight recorder to every group that
        # registers a backend.
        self._set_default_backend(dist.ProcessGroup.BackendType.GLOO)
        if options.transport == "shared":
            self._shared = SharedRelay(store, self._cpu_group, rank, size, self._staging_bytes, timeout)

    def getBackendName(self) -> str:  # noqa: N802 -- PyTorch trampoline name
        return _BACKEND_NAME

    def shutdown(self) -> None:
        # Collectives already issued finish before what they use is released.
        # An interrupted wait releases nothing, so a later call can finish.
        self._executor.shutdown(wait=True)
        if self._closed:
            return
        self._closed = True
        try:
            super().shutdown()
        finally:
            self._staging_buffer = None
            self._pipeline = None
            self._shared = None

    def _submit(
        self, device: torch.device, body: Callable[[], None], outputs: list[torch.Tensor], *, nested: bool = False,
    ) -> dist.Work:
        """Queue a validated collective over ``device`` tensors on the group's worker; return its Work."""
        ready: torch.cuda.Event | None = None
        if device.type == "cuda":
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("piper_relay transfers do not support CUDA graph capture")
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))

        def job() -> None:
            with torch.no_grad():
                if self._shared is not None and self._shared.broken:
                    raise RuntimeError("shared relay failed previously; destroy and recreate the process group")
                if ready is None:
                    body()
                    return
                ready.synchronize()
                stream = self._stream
                if stream is None or stream.device != device:
                    # A private stream: synchronizing the worker's default
                    # stream would also wait for the caller's work on it.
                    stream = self._stream = torch.cuda.Stream(device=device)
                with torch.cuda.device(device), torch.cuda.stream(stream), foreground_transfer():
                    body()

        return _PendingWork(self._executor.submit(job), outputs, nested=nested)

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
        # Chunk views are cut from a base-free alias: a view keeps its base
        # tensor alive, and a completed collective can hold a chunk view a
        # little longer than the group, which would keep the owner, and so
        # its registration, alive past shutdown.
        with host_pin_manager.acquire([owner]) if device.type == "cuda" else nullcontext() as lease:
            yield region.detach(), lease is not None and lease.pageable_bytes == 0

    def _slot_bytes(self, slots: int) -> int:
        if self._shared is not None:
            size = self._shared.capacity * self._shared.buffer_count
            return size // (16 * slots) * 16
        return self._staging_bytes // (16 * slots * self._pipeline_buffers) * 16

    def _exchange_pair(self, send_buf: torch.Tensor, recv_buf: torch.Tensor, timeout: timedelta) -> None:
        peer = 1 - self.rank()
        # Post receive before send and wait in order so both ranks make concurrent progress.
        works: list[dist.Work] = [
            self._cpu_group.recv([recv_buf], peer, _GLOO_EXCHANGE_TAG),  # type: ignore[attr-defined]
        ]
        first_error: BaseException | None = None
        try:
            works.insert(0, self._cpu_group.send([send_buf], peer, _GLOO_EXCHANGE_TAG))  # type: ignore[attr-defined]
        except BaseException as error:
            first_error = error
        finally:
            for work in works:
                try:
                    if timeout > timedelta(0):
                        work.wait(timeout)
                    else:
                        work.wait()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error

    def barrier(self, opts: dist.BarrierOptions | None = None) -> dist.Work:  # type: ignore[override]
        options = opts if opts is not None else dist.BarrierOptions()
        # Queued like a collective: sent to Gloo directly, it could overtake an
        # earlier collective's control messages and pair different operations
        # across ranks.
        return self._submit(
            torch.device("cpu"), lambda: self._cpu_group.barrier(options).wait(), [],  # type: ignore[attr-defined]
        )

    def allreduce(  # type: ignore[override]
        self,
        tensors: list[torch.Tensor],
        opts: dist.AllreduceOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else dist.AllreduceOptions()
        tensor = _single_tensor(tensors)
        _validate_reduction([tensor], options)
        return self._submit(tensor.device, lambda: self._reduce(tensor, options), [tensor])

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
        def body() -> None:
            if self._shared is not None and tensors[0].device.type == "cuda":
                self._shared.allreduce(tensors, host_pin_manager)
                return
            for tensor in tensors:
                self._reduce(tensor, single_options)

        return self._submit(tensors[0].device, body, tensors)

    def _reduce(self, tensor: torch.Tensor, options: dist.AllreduceOptions) -> None:
        if self._shared is not None and tensor.device.type == "cuda":
            self._shared.allreduce([tensor], host_pin_manager)
            return
        low_precision = tensor.dtype != torch.float32
        if low_precision and self.size() == 2:
            capacity = self._slot_bytes(3)
            rank = self.rank()
            with closing(self._copy_chunks([tensor], [tensor], slots=3, inplace=True, stage_cpu=True)) as chunks:
                for chunk in chunks:
                    raw = chunk.inputs[0]
                    assert chunk.scratch is not None
                    peer_raw = chunk.scratch[:raw.numel()]
                    scratch_f32 = chunk.scratch[capacity:].view(torch.float32)
                    self._exchange_pair(raw, peer_raw, options.timeout)
                    host = raw.view(tensor.dtype)
                    _sum_pair_in_fp32(host, host, peer_raw.view(tensor.dtype), scratch_f32, rank)
            return
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

        def body() -> None:
            if self._shared is not None:
                self._shared.broadcast(tensor, options.rootRank, host_pin_manager)
                return
            is_root = self.rank() == options.rootRank
            sources = [tensor] if is_root else []
            destinations = [] if is_root else [tensor]
            with closing(self._copy_chunks(sources, destinations, slots=1)) as chunks:
                for chunk in chunks:
                    payload = chunk.inputs if is_root else chunk.outputs
                    self._cpu_group.broadcast(payload, options).wait()  # type: ignore[attr-defined]

        return self._submit(tensor.device, body, [tensor])

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

        def body() -> None:
            if self._shared is not None:
                self._shared.scatter(sources, output, options.rootRank, host_pin_manager)
                return
            self._scatter_gloo(output, sources, options)

        return self._submit(output.device, body, [output])

    def _scatter_gloo(
        self, output: torch.Tensor, sources: list[torch.Tensor], options: dist.ScatterOptions,
    ) -> None:
        rank = self.rank()
        slots = self.size() + 1
        if output.device.type == "cuda" and self.size() > 1 and rank == options.rootRank:
            remote_sources = [tensor for source_rank, tensor in enumerate(sources) if source_rank != rank]
            capacity = self._slot_bytes(slots)
            need_local_copy = bool(output.numel()) and output.data_ptr() != sources[rank].data_ptr()
            if need_local_copy:
                _bytes(output).copy_(_bytes(sources[rank]), non_blocking=True)
            try:
                with closing(self._copy_chunks(remote_sources, [], slots=slots)) as chunks:
                    for chunk in chunks:
                        width = chunk.inputs[0].numel()
                        assert chunk.scratch is not None
                        dummy_in = chunk.scratch[:width]
                        dummy_out = chunk.scratch[capacity:capacity + width]
                        gloo_inputs = list(chunk.inputs)
                        gloo_inputs.insert(rank, dummy_in)
                        self._cpu_group.scatter(  # type: ignore[attr-defined]
                            [dummy_out], [gloo_inputs], options,
                        ).wait()
            finally:
                if need_local_copy:
                    torch.cuda.current_stream(output.device).synchronize()
            return
        with closing(self._copy_chunks(sources, [output], slots=slots)) as chunks:
            for chunk in chunks:
                self._cpu_group.scatter(  # type: ignore[attr-defined]
                    chunk.outputs, [chunk.inputs] if sources else [], options,
                ).wait()

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
        self._validate_own_slice_aliases(
            source, destinations, "piper_relay all-gather input may overlap only its exact local output slice",
        )
        return self._submit(
            source.device, lambda: self._gather(source, destinations, options), destinations, nested=True,
        )

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
        _validate_batched_outputs(output_lists, input_list)
        # Validate the whole batch before starting communication or mutation.
        destinations: list[list[torch.Tensor]] = []
        for output, source in zip(output_lists, input_list, strict=True):
            _validate_pair(output, source, source.numel() * self.size())
            views = list(output.view(self.size(), source.numel()).unbind(0))
            self._validate_own_slice_aliases(
                source, views, "piper_relay all-gather input may overlap only its exact local output slice",
            )
            destinations.append(views)
        def body() -> None:
            if self._shared is not None:
                self._shared.allgather(input_list, destinations, host_pin_manager)
                return
            for source, outputs in zip(input_list, destinations, strict=True):
                self._gather(source, outputs, options)

        return self._submit(input_list[0].device, body, output_lists)

    def _validate_own_slice_aliases(
        self, single: torch.Tensor, per_rank: Sequence[torch.Tensor], message: str,
    ) -> None:
        for rank, slice_tensor in enumerate(per_rank):
            if _overlap(slice_tensor, single) and (
                rank != self.rank() or slice_tensor.data_ptr() != single.data_ptr()
            ):
                raise ValueError(message)

    def reduce_scatter(  # type: ignore[override]
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: ReduceScatterOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else ReduceScatterOptions()
        output = _single_tensor(output_tensors)
        if len(input_tensors) != 1 or len(input_tensors[0]) != self.size():
            raise ValueError("piper_relay reduce-scatter requires one input tensor per rank")
        sources = input_tensors[0]
        _validate_reduction([output, *sources], options)
        for source in sources:
            _validate_pair(output, source, source.numel(), same_shape=True)
        self._validate_own_slice_aliases(
            output, sources, "piper_relay reduce-scatter output may overlap only its exact local input slice",
        )

        def body() -> None:
            if self._shared is not None and output.device.type == "cuda":
                self._shared.reduce_scatter([output], [sources], host_pin_manager)
                return
            self._reduce_scatter_gloo(output, sources, options)

        return self._submit(output.device, body, [output])

    def reduce_scatter_single(  # type: ignore[override]
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch parameter name
        opts: ReduceScatterOptions | None = None,
    ) -> dist.Work:
        return self.reduce_scatter_single_coalesced([output], [input], opts)

    def _reduce_scatter_base(  # type: ignore[override]
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch parameter name
        opts: ReduceScatterOptions | None = None,
    ) -> dist.Work:
        return self.reduce_scatter_single(output, input, opts)

    def reduce_scatter_single_coalesced(  # type: ignore[override]
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: ReduceScatterOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else ReduceScatterOptions()
        if len(output_tensors) != len(input_tensors) or not output_tensors:
            raise ValueError("piper_relay reduce-scatter requires matching nonempty output/input lists")
        _validate_reduction([*output_tensors, *input_tensors], options)
        _validate_batched_outputs(output_tensors, input_tensors)
        # Validate the whole batch before starting communication or mutation.
        sources_list: list[list[torch.Tensor]] = []
        for output, source in zip(output_tensors, input_tensors, strict=True):
            _validate_pair(source, output, output.numel() * self.size())
            views = list(source.view(self.size(), output.numel()).unbind(0))
            self._validate_own_slice_aliases(
                output, views, "piper_relay reduce-scatter output may overlap only its exact local input slice",
            )
            sources_list.append(views)

        def body() -> None:
            if self._shared is not None and output_tensors[0].device.type == "cuda":
                self._shared.reduce_scatter(output_tensors, sources_list, host_pin_manager)
                return
            for output, sources in zip(output_tensors, sources_list, strict=True):
                self._reduce_scatter_gloo(output, sources, options)

        return self._submit(output_tensors[0].device, body, output_tensors)

    def _reduce_scatter_gloo(
        self, output: torch.Tensor, sources: list[torch.Tensor], options: ReduceScatterOptions,
    ) -> None:
        if output.dtype != torch.float32 and self.size() == 2:
            rank = self.rank()
            peer = 1 - rank
            with closing(self._copy_chunks(
                [sources[rank], sources[peer]], [output], slots=3, inplace=True, stage_cpu=True,
            )) as chunks:
                for chunk in chunks:
                    width = chunk.inputs[0].numel()
                    assert chunk.scratch is not None
                    peer_raw = chunk.scratch[:width]
                    self._exchange_pair(chunk.inputs[1], peer_raw, options.timeout)
                    chunk.inputs[1].copy_(peer_raw)
                    _sum_pair_in_fp32(
                        chunk.outputs[0].view(output.dtype),
                        chunk.inputs[0].view(output.dtype),
                        chunk.inputs[1].view(output.dtype),
                        chunk.scratch.view(torch.float32),
                        rank,
                    )
            return
        if output.dtype == torch.float32:
            with closing(self._copy_chunks(sources, [output], slots=self.size() + 1)) as chunks:
                for chunk in chunks:
                    # PyTorch's Gloo stubs omit the collective bindings.
                    self._cpu_group.reduce_scatter(  # type: ignore[attr-defined]
                        [chunk.outputs[0].view(torch.float32)],
                        [[t.view(torch.float32) for t in chunk.inputs]],
                        options,
                    ).wait()
            return
        reduce_options = ReduceOptions()
        reduce_options.reduceOp = options.reduceOp
        reduce_options.timeout = options.timeout
        reduce_options.asyncOp = options.asyncOp
        rank = self.rank()
        for target in range(self.size()):
            reduce_options.rootRank = target
            with closing(self._copy_chunks(
                [sources[target]], [output] if rank == target else [], slots=3, inplace=True, stage_cpu=True,
            )) as chunks:
                for chunk in chunks:
                    host = chunk.inputs[0].view(output.dtype)
                    assert chunk.scratch is not None
                    accumulator = chunk.scratch[:host.numel() * 4].view(torch.float32)
                    accumulator.copy_(host)
                    # PyTorch's Gloo stubs omit the collective bindings.
                    self._cpu_group.reduce([accumulator], reduce_options).wait()  # type: ignore[attr-defined]
                    if rank == target:
                        host.copy_(accumulator)

    def _gather(self, source: torch.Tensor, outputs: list[torch.Tensor], options: AllgatherOptions) -> None:
        if self._shared is not None:
            self._shared.allgather([source], [outputs], host_pin_manager)
            return
        rank = self.rank()
        if source.device.type == "cuda" and self.size() > 1:
            remote_outputs = [out for r, out in enumerate(outputs) if r != rank]
            need_local_copy = bool(source.numel()) and outputs[rank].data_ptr() != source.data_ptr()
            if need_local_copy:
                _bytes(outputs[rank]).copy_(_bytes(source), non_blocking=True)
            try:
                with closing(self._copy_chunks([source], remote_outputs, slots=self.size() + 1)) as chunks:
                    for chunk in chunks:
                        assert chunk.scratch is not None
                        gloo_outputs = list(chunk.outputs)
                        gloo_outputs.insert(rank, chunk.scratch[:chunk.inputs[0].numel()])
                        self._cpu_group.allgather([gloo_outputs], chunk.inputs, options).wait()  # type: ignore[attr-defined]
            finally:
                if need_local_copy:
                    torch.cuda.current_stream(source.device).synchronize()
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
        primary = destinations[0] if destinations else sources[0]
        device = (outputs or inputs)[0].device
        if device.type == "cpu" and not stage_cpu:
            for start, end in chunk_ranges(primary.numel(), capacity):
                yield HostChunk([t[start:end] for t in sources], [t[start:end] for t in destinations])
            return
        with self._buffer(device) as (buffer, pinned):
            if pinned and self._pipeline_buffers > 1 and primary.numel() > capacity:
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
            for start, end in chunk_ranges(primary.numel(), capacity):
                chunk = make_host_chunk(region, capacity, end - start, len(inputs), len(outputs), inplace)
                for source, host in zip(sources, chunk.inputs, strict=True):
                    host.copy_(source[start:end], non_blocking=False)
                yield chunk
                for destination, host in zip(destinations, chunk.outputs, strict=True):
                    destination[start:end].copy_(host, non_blocking=False)

    def _validate_root(self, root: int) -> None:
        if not 0 <= root < self.size():
            raise ValueError("piper_relay root rank is outside the process group")


class _PendingWork(dist.Work):
    """A collective queued on its group's worker; waiting blocks until it has completed.

    The job synchronizes its streams before finishing, so once ``wait`` returns the
    outputs are ready for any stream. An interrupted ``wait`` leaves the collective
    running; its job keeps the tensors referenced until it finishes. Nothing refers
    back to the Work, so its tensors are freed as soon as the caller drops it and
    the job ends, without waiting for the cyclic garbage collector.
    """

    def __init__(self, job: Future[None], tensors: list[torch.Tensor], *, nested: bool = False) -> None:
        super().__init__()
        self._job = job
        self._tensors = tensors
        self._nested = nested

    def get_future(self) -> torch.futures.Future[list[torch.Tensor] | list[list[torch.Tensor]]]:
        """Wait for the collective, then return its completed future.

        Callbacks then run on the calling thread, as for any completed future.
        Run on the group's worker, a callback that waits for a later collective,
        as DDP communication hooks such as PowerSGD do, would block the only
        thread that can run it.
        """
        future: torch.futures.Future[list[torch.Tensor] | list[list[torch.Tensor]]] = torch.futures.Future()
        error = self._job.exception()
        if error is not None:
            # torch's stub types the exception as the result type.
            future.set_exception(error)  # pyright: ignore[reportArgumentType]
        else:
            future.set_result([self._tensors] if self._nested else self._tensors)
        return future

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        # c10d's default, zero, means no timeout.
        self._job.result(timeout=timeout.total_seconds() or None)
        return True

    def is_completed(self) -> bool:
        return self._job.done()

    def is_success(self) -> bool:
        return self._job.done() and self._job.exception() is None

    def result(self) -> list[torch.Tensor]:
        self.wait()
        return self._tensors

    def synchronize(self) -> None:
        self.wait()

    def block_current_stream(self) -> None:
        self.wait()


def _validate_reduction(
    tensors: list[torch.Tensor], options: dist.AllreduceOptions | ReduceScatterOptions,
) -> None:
    if options.reduceOp != dist.ReduceOp.SUM:
        raise NotImplementedError("piper_relay currently supports only SUM reductions")
    _validate_tensors(tensors)
    for tensor in tensors:
        if tensor.dtype not in REDUCTION_DTYPES:
            raise NotImplementedError("piper_relay SUM reduction supports FP32, FP16 and BF16")


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


def _validate_batched_outputs(outputs: Sequence[torch.Tensor], inputs: Sequence[torch.Tensor]) -> None:
    _validate_disjoint(outputs)
    for i, output in enumerate(outputs):
        for j, source in enumerate(inputs):
            if i != j and _overlap(output, source):
                raise ValueError("piper_relay batched outputs must not overlap another input")


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Transport payload bits unchanged, including BF16 and integer metadata."""
    return tensor.reshape(-1).view(torch.uint8)


__all__ = ["RelayOptions", "register_relay_backend"]
