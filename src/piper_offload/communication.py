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
and GPU-local summation, or ``transport="ipc"`` on Linux to add peer GPU-to-GPU
IPC staging slots (CUDA/HIP) with automatic fallback to shared host memory on
Windows, CPU collectives, or unsupported P2P topologies; on x86_64/AMD64, Gloo
then carries barriers, CPU tensor reductions, and point-to-point transfers
(other architectures also use Gloo for control messages). The default
``transport="gloo"`` carries collective payloads through CPU Gloo as well. Gloo
never receives an accelerator tensor.

Collectives run on one worker thread per process group in issue order; point-to-point
operations (``send``, ``recv``, ``isend``, ``irecv``, ``recv_anysource``, and
``batch_isend_irecv``) run concurrently per ``(direction, peer, tag)`` lane so
bidirectional transfers and ring exchanges make progress without deadlocking on
collectives or each other. With ``async_op=True`` or async P2P calls
(``isend``, ``irecv``, ``batch_isend_irecv``), a call returns at once; until
``wait()`` returns, the caller must not read the outputs or write the inputs or
outputs, from any stream. ``get_future()`` on collectives and P2P work waits, so
the future's callbacks run on the calling thread. Without ``async_op``, the call
returns once the transfer has completed, and the outputs are then ready for
every stream.
Transfers on accelerator tensors start once the caller's current stream has
produced its inputs and move their data as foreground transfers, which
background copies targeting the same device (such as block prefetch) wait for
(see ``piper_offload._transfer_priority``). No offloader, NCCL, or compiled extension
is required. Gloo collectives default to two pipeline buffers; shared
collectives use three outgoing slots per rank.

SUM all-reduce and reduce-scatter support FP32, FP16 and BF16, accumulating
reduced precision in FP32. Broadcast, scatter, equal-size all-gather,
all-to-all (equal and unequal splits), and point-to-point transfers also
preserve FP64, bool, uint8, signed integers, and complex64/complex128.
All-reduce, reduce-scatter, and all-gather include coalesced variants;
all-gather, reduce-scatter, and all-to-all support tensor-list and
single-buffer forms. Inputs and outputs must be contiguous plain tensors on one
local device. Outputs must not overlap, except that an all-gather input may
occupy its exact own-rank output slice, a reduce-scatter output may occupy its
exact own-rank input slice, a scatter root may receive into its exact source
slice, and an all-to-all output may occupy its exact own-rank input slice.
Coalesced outputs must not overwrite other inputs. Point-to-point transfers
require non-negative integer tags up to ``(1 << 31) - 1 - 8``, distinct peer
ranks (no self-rank ``isend``/``irecv``), and must not overlap ``recv_anysource``
with explicit ``recv`` on the same tag concurrently.

``RelayOptions`` describes staging and pin budgets. If the staging allocation
exceeds the pin budget or registration reports invalid-value, copies run
synchronously through pageable storage. Non-SUM reductions and CUDA graph
capture are unsupported. Direct C++ functional all-gather ``out`` calls bypass
the Python override and are unsupported for accelerator tensors.

If a shared- or IPC-transport collective fails or is interrupted, its outputs may
be partial and every later operation on the group raises; destroy and recreate
the process group. Linux ROCm distinct-GPU IPC has been validated; NVIDIA
distinct-GPU and Windows GPU behavior need validation on their target hardware.
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
    AllToAllOptions,
    Backend,
    ReduceOptions,
    ReduceScatterOptions,
    _DistributedBackendOptions,
)

from ._relay_shared import BUFFERS_PER_RANK, COPY_DTYPES, REDUCTION_DTYPES, SharedRelay, shared_slot_bytes
from ._relay_staging import HostChunk, TransferPipeline, chunk_ranges, make_host_chunk
from ._transfer_priority import ForegroundProducer, foreground_transfer, register_foreground_producer
from .pin_manager import host_pin_manager

_BACKEND_NAME = "piper_relay"
_SHARED_BROKEN_ERROR = "shared relay failed previously; destroy and recreate the process group"
_GLOO_EXCHANGE_TAG = 2
_P2P_TAG_BASE = 8
_MAX_P2P_TAG = (1 << 31) - 1 - _P2P_TAG_BASE
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
    affect shared slot sizing or minimum ``staging_bytes``. On CUDA, P2P ops
    lazily cache up to two host staging buffers (sized up to
    ``staging_bytes / 2`` with Gloo or ``staging_bytes / (2 * world_size)`` with
    shared transport), allocating transient overflow chunks only while more than
    two CUDA P2P transfers run concurrently, and release cached buffers on
    group shutdown.

    ``transport="ipc"`` builds on ``transport="shared"`` by mapping the shared host
    staging arena at group initialization (for CPU collectives, all-to-all size
    headers, and fallback calls, kept unpinned while IPC is active) and lazily
    allocating a per-rank CUDA/HIP device staging ring (``staging_bytes /
    world_size``, three slots per rank) plus up to ``staging_bytes / world_size``
    of reduction scratch during GPU reductions (up to ``2 * staging_bytes /
    world_size`` peak device memory per rank) on the first GPU collective,
    opening peer IPC handles on Linux. Each rank binds its IPC arena once to
    the device of its first GPU collective; later collectives where all ranks
    switch to another device fall back to shared host memory, while single-rank
    device switches or mixed CPU/GPU calls raise ``ValueError`` without breaking
    the group. GPU collective payloads then move GPU-to-GPU through device IPC
    slots without host copies or pin-budget usage, while still releasing on
    group shutdown. CPU collectives, all-to-all size headers, barriers, P2P ops,
    and ranks or topologies without IPC/P2P support (including Windows,
    same-process ranks, or fallback calls) automatically use shared host memory
    and Gloo as applicable. The default transport remains ``"gloo"``.
    """

    staging_bytes: int = 8 * 1024 * 1024
    pipeline_buffers: int = 2
    transport: str = "gloo"

    def __post_init__(self) -> None:
        if type(self.staging_bytes) is not int or self.staging_bytes <= 0 or self.staging_bytes % 16:
            raise ValueError("piper_relay staging_bytes must be a positive integer multiple of 16")
        if type(self.pipeline_buffers) is not int or self.pipeline_buffers not in (1, 2, 3):
            raise ValueError("piper_relay pipeline_buffers must be 1, 2, or 3")
        if self.transport not in ("gloo", "shared", "ipc"):
            raise ValueError("piper_relay transport must be 'gloo', 'shared', or 'ipc'")

    def _validate_group_size(self, size: int) -> None:
        shared_like = self.transport in ("shared", "ipc")
        buffers = 1 if shared_like else self.pipeline_buffers
        minimum = 16 * max(3, size + 1) * buffers
        if self.staging_bytes < minimum:
            raise ValueError("piper_relay staging_bytes must allow 16 bytes per staging slot")
        if shared_like and shared_slot_bytes(self.staging_bytes, size) * BUFFERS_PER_RANK < minimum:
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
        self._p2p_lock = threading.Lock()
        self._p2p_tails: dict[tuple[str, int, int], threading.Event] = {}
        self._p2p_inflight: set[threading.Thread] = set()
        self._p2p_pool: list[tuple[torch.Tensor, torch.cuda.Stream]] = []
        self._shutting_down = False
        self._foreground_producer: ForegroundProducer | None = None
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
        if options.transport in ("shared", "ipc"):
            self._shared = SharedRelay(
                store, self._cpu_group, rank, size, self._staging_bytes, timeout, ipc=options.transport == "ipc",
            )
        self._foreground_producer = register_foreground_producer()

    def getBackendName(self) -> str:  # noqa: N802 -- PyTorch trampoline name
        return _BACKEND_NAME

    def _get_backend(self, device: torch.device) -> Backend | _RelayProcessGroup:  # type: ignore[override]
        # dist.batch_isend_irecv queries the device backend for supports_coalescing.
        # Returning self keeps CUDA batched P2P on the per-op path without
        # registering Gloo for CUDA (so unsupported CUDA collectives still fail).
        if device.type == "cuda":
            return self
        return super()._get_backend(device)

    def shutdown(self) -> None:
        # Collectives and P2P transfers already issued finish before what they
        # use is released. An interrupted wait releases nothing, so a later call
        # can finish.
        with self._p2p_lock:
            self._shutting_down = True
            inflight = list(self._p2p_inflight)
        self._executor.shutdown(wait=True)
        for thread in inflight:
            thread.join()
        if self._closed:
            return
        self._closed = True
        try:
            try:
                if self._shared is not None:
                    self._shared.close()
            finally:
                super().shutdown()
        finally:
            self._staging_buffer = None
            self._pipeline = None
            self._shared = None
            if self._foreground_producer is not None:
                self._foreground_producer.close()
                self._foreground_producer = None
            with self._p2p_lock:
                self._p2p_pool.clear()

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

        def job() -> int | None:
            with torch.no_grad():
                if self._shared is not None and self._shared.broken:
                    raise RuntimeError(_SHARED_BROKEN_ERROR)
                if ready is None:
                    body()
                    return None
                ready.synchronize()
                stream = self._stream
                if stream is None or stream.device != device:
                    # A private stream: synchronizing the worker's default
                    # stream would also wait for the caller's work on it.
                    stream = self._stream = torch.cuda.Stream(device=device)
                with torch.cuda.device(device), torch.cuda.stream(stream), foreground_transfer(device):
                    body()
            return None

        return _PendingWork(self._executor.submit(job), outputs, nested=nested)

    def _validate_p2p_peer(self, peer: int) -> None:
        if type(peer) is not int or not 0 <= peer < self.size():
            raise ValueError("piper_relay point-to-point peer rank is outside the process group")
        if peer == self.rank():
            raise ValueError("piper_relay point-to-point transfers do not support the local rank")

    def _acquire_p2p_slot(self, device: torch.device, chunk_bytes: int) -> tuple[torch.Tensor, torch.cuda.Stream]:
        with self._p2p_lock:
            for index, (buf, stream) in enumerate(self._p2p_pool):
                if stream.device == device and buf.numel() == chunk_bytes:
                    return self._p2p_pool.pop(index)
        return (
            torch.empty(chunk_bytes, dtype=torch.uint8, device="cpu"),
            torch.cuda.Stream(device=device),
        )

    def _release_p2p_slot(self, slot: tuple[torch.Tensor, torch.cuda.Stream]) -> None:
        with self._p2p_lock:
            if not self._shutting_down and not self._closed and len(self._p2p_pool) < 2:
                self._p2p_pool.append(slot)

    def _run_p2p_device(
        self,
        device: torch.device,
        chunk_bytes: int,
        transfer: Callable[[torch.Tensor | None], int | None] | None,
    ) -> int | None:
        assert transfer is not None
        if device.type == "cpu":
            return transfer(None)
        slot: tuple[torch.Tensor, torch.cuda.Stream] | None = self._acquire_p2p_slot(device, chunk_bytes)
        buf: torch.Tensor | None
        stream: torch.cuda.Stream | None
        buf, stream = slot
        synced = False
        try:
            with (
                torch.cuda.device(device),
                torch.cuda.stream(stream),
                foreground_transfer(device),
                host_pin_manager.acquire([buf]),
            ):
                try:
                    return transfer(buf.detach())
                finally:
                    stream.synchronize()
                    synced = True
        finally:
            if synced and slot is not None:
                self._release_p2p_slot(slot)
            # Clear local references before unwinding so an exception traceback
            # never retains a staging buffer past lease release or shutdown.
            buf = None
            stream = None
            slot = None
            transfer = None

    @staticmethod
    def _run_p2p_lane(
        prev: threading.Event | None,
        ready: torch.cuda.Event | None,
        future: Future[int | None] | None,
        body: Callable[[], int | None] | None,
        finish: Callable[[], None] | None,
        shared: SharedRelay | None = None,
    ) -> None:
        assert future is not None
        assert body is not None
        assert finish is not None
        try:
            if prev is not None:
                prev.wait()
            prev = None
            with torch.no_grad():
                if shared is not None and shared.broken:
                    raise RuntimeError(_SHARED_BROKEN_ERROR)
                if ready is not None:
                    ready.synchronize()
                ready = None
                future.set_result(body())
        except BaseException as error:
            future.set_exception(error)
        finally:
            # Break any error -> traceback -> frame -> future/body/shared cycle so
            # worker buffers and caller tensors release without cyclic GC.
            body = None
            future = None
            shared = None
            try:
                finish()
            finally:
                finish = None

    def _submit_p2p(
        self,
        key: tuple[str, int, int],
        device: torch.device,
        body: Callable[[], int | None],
        outputs: list[torch.Tensor],
    ) -> dist.Work:
        ready: torch.cuda.Event | None = None
        if device.type == "cuda":
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("piper_relay transfers do not support CUDA graph capture")
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
        future: Future[int | None] = Future()
        done = threading.Event()

        def finish() -> None:
            current = threading.current_thread()
            with self._p2p_lock:
                if self._p2p_tails.get(key) is done:
                    del self._p2p_tails[key]
                self._p2p_inflight.discard(current)
            done.set()

        with self._p2p_lock:
            if self._shutting_down:
                raise RuntimeError("cannot schedule new futures after shutdown")
            prev = self._p2p_tails.get(key)
            thread = threading.Thread(
                target=self._run_p2p_lane,
                args=(prev, ready, future, body, finish, self._shared),
                name="piper-relay-p2p",
                daemon=True,
            )
            thread.start()
            self._p2p_tails[key] = done
            self._p2p_inflight.add(thread)
        return _PendingWork(future, outputs)

    def send(self, tensors: list[torch.Tensor], dstRank: int, tag: int) -> dist.Work:  # noqa: N803 -- PyTorch API
        tensor = _single_tensor(tensors)
        _validate_tensors([tensor])
        self._validate_p2p_peer(dstRank)
        wire_tag = _wire_p2p_tag(tag)
        chunk_bytes = self._slot_bytes(2)

        def transfer(stage_buf: torch.Tensor | None) -> int | None:
            slice_buf: torch.Tensor | None = None
            try:
                data = _bytes(tensor)
                for start, end in chunk_ranges(data.numel(), chunk_bytes):
                    if stage_buf is None:
                        slice_buf = data[start:end]
                    else:
                        slice_buf = stage_buf[:end - start]
                        if end > start:
                            slice_buf.copy_(data[start:end], non_blocking=False)
                    self._cpu_group.send([slice_buf], dstRank, wire_tag).wait()  # type: ignore[attr-defined]
                return None
            finally:
                slice_buf = None
                stage_buf = None

        return self._submit_p2p(
            ("send", dstRank, wire_tag),
            tensor.device,
            lambda: self._run_p2p_device(tensor.device, chunk_bytes, transfer),
            [tensor],
        )

    def _recv_p2p(self, tensors: list[torch.Tensor], src: int | None, tag: int) -> dist.Work:
        tensor = _single_tensor(tensors)
        _validate_tensors([tensor])
        wire_tag = _wire_p2p_tag(tag)
        if src is None:
            if self.size() < 2:
                raise ValueError("piper_relay recv_anysource requires at least two ranks")
            key = ("recv", -1, wire_tag)
        else:
            self._validate_p2p_peer(src)
            key = ("recv", src, wire_tag)
        chunk_bytes = self._slot_bytes(2)

        def transfer(stage_buf: torch.Tensor | None) -> int:
            slice_buf: torch.Tensor | None = None
            try:
                data = _bytes(tensor)
                resolved_src = src
                for index, (start, end) in enumerate(chunk_ranges(data.numel(), chunk_bytes)):
                    slice_buf = data[start:end] if stage_buf is None else stage_buf[:end - start]
                    if resolved_src is None and index == 0:
                        work = self._cpu_group.recv_anysource([slice_buf], wire_tag)  # type: ignore[attr-defined]
                        work.wait()
                        resolved_src = int(work._source_rank())
                    else:
                        assert resolved_src is not None
                        self._cpu_group.recv([slice_buf], resolved_src, wire_tag).wait()  # type: ignore[attr-defined]
                    if stage_buf is not None and end > start:
                        data[start:end].copy_(slice_buf, non_blocking=False)
                assert resolved_src is not None
                return resolved_src
            finally:
                slice_buf = None
                stage_buf = None

        return self._submit_p2p(
            key,
            tensor.device,
            lambda: self._run_p2p_device(tensor.device, chunk_bytes, transfer),
            [tensor],
        )

    def recv(self, tensors: list[torch.Tensor], srcRank: int, tag: int) -> dist.Work:  # noqa: N803 -- PyTorch API
        return self._recv_p2p(tensors, srcRank, tag)

    def recv_anysource(self, tensors: list[torch.Tensor], tag: int) -> dist.Work:
        return self._recv_p2p(tensors, None, tag)

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

    def _exchange_peers(
        self,
        send_peer: int,
        send_buf: torch.Tensor | None,
        recv_peer: int,
        recv_buf: torch.Tensor | None,
        timeout: timedelta,
    ) -> None:
        # Post receive before send and wait in order so both ranks make concurrent progress.
        works: list[dist.Work] = []
        if recv_buf is not None:
            works.append(self._cpu_group.recv([recv_buf], recv_peer, _GLOO_EXCHANGE_TAG))  # type: ignore[attr-defined]
        first_error: BaseException | None = None
        try:
            if send_buf is not None:
                works.insert(0, self._cpu_group.send([send_buf], send_peer, _GLOO_EXCHANGE_TAG))  # type: ignore[attr-defined]
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

    def _exchange_pair(self, send_buf: torch.Tensor, recv_buf: torch.Tensor, timeout: timedelta) -> None:
        peer = 1 - self.rank()
        self._exchange_peers(peer, send_buf, peer, recv_buf, timeout)

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

    def _validate_alltoall_aliases(self, outputs: Sequence[torch.Tensor], sources: Sequence[torch.Tensor]) -> None:
        rank = self.rank()
        if outputs[rank].nbytes != sources[rank].nbytes:
            raise ValueError("piper_relay all-to-all own-rank input and output byte sizes must match")
        for out_rank, output in enumerate(outputs):
            for in_rank, source in enumerate(sources):
                if _overlap(output, source) and not (
                    out_rank == in_rank == rank
                    and output.data_ptr() == source.data_ptr()
                    and output.nbytes == source.nbytes
                ):
                    raise ValueError(
                        "piper_relay all-to-all output may overlap inputs only at its exact local slice",
                    )

    def alltoall(  # type: ignore[override]
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: AllToAllOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else AllToAllOptions()
        outputs = list(output_tensors)
        sources = list(input_tensors)
        if len(outputs) != self.size() or len(sources) != self.size():
            raise ValueError("piper_relay all-to-all requires one input and one output tensor per rank")
        _validate_tensors([*outputs, *sources])
        dtype = outputs[0].dtype
        if any(tensor.dtype != dtype for tensor in (*outputs, *sources)):
            raise ValueError("piper_relay all-to-all requires one dtype across all input and output tensors")
        _validate_disjoint(outputs)
        self._validate_alltoall_aliases(outputs, sources)
        return self._submit(outputs[0].device, lambda: self._alltoall(outputs, sources, options), outputs)

    def all_to_all_single(  # type: ignore[override]
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch parameter name
        output_split_sizes: Sequence[int] | None = None,
        input_split_sizes: Sequence[int] | None = None,
        opts: AllToAllOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else AllToAllOptions()
        _validate_tensors([output, input])
        if output.dtype != input.dtype:
            raise ValueError("piper_relay all-to-all input and output dtypes must match")
        if output.ndim < 1 or input.ndim < 1:
            raise ValueError("piper_relay all_to_all_single requires tensors with at least one dimension")
        if output.shape[1:] != input.shape[1:]:
            raise ValueError("piper_relay all_to_all_single input and output trailing dimensions must match")
        out_splits = _resolve_alltoall_splits(output, output_split_sizes, self.size())
        in_splits = _resolve_alltoall_splits(input, input_split_sizes, self.size())
        out_views = list(output.split(out_splits, dim=0))
        in_views = list(input.split(in_splits, dim=0))
        self._validate_alltoall_aliases(out_views, in_views)
        return self._submit(output.device, lambda: self._alltoall(out_views, in_views, options), [output])

    def alltoall_base(  # type: ignore[override]
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch parameter name
        output_split_sizes: Sequence[int] | None = None,
        input_split_sizes: Sequence[int] | None = None,
        opts: AllToAllOptions | None = None,
    ) -> dist.Work:
        return self.all_to_all_single(output, input, output_split_sizes, input_split_sizes, opts)

    def _alltoall(
        self, outputs: Sequence[torch.Tensor], sources: Sequence[torch.Tensor], options: AllToAllOptions,
    ) -> None:
        if self._shared is not None:
            self._shared.alltoall(outputs, sources, host_pin_manager)
            return
        self._alltoall_gloo(outputs, sources, options)

    def _exchange_unequal_gloo(
        self,
        send_peer: int,
        source: torch.Tensor,
        recv_peer: int,
        destination: torch.Tensor,
        timeout: timedelta,
    ) -> None:
        capacity = self._slot_bytes(2)
        src_bytes = _bytes(source)
        dst_bytes = _bytes(destination)
        send_len = src_bytes.numel()
        recv_len = dst_bytes.numel()
        total = max(send_len, recv_len)
        if source.device.type == "cpu":
            for start in range(0, total, capacity):
                send_slice = src_bytes[start:min(send_len, start + capacity)] if start < send_len else None
                recv_slice = dst_bytes[start:min(recv_len, start + capacity)] if start < recv_len else None
                self._exchange_peers(send_peer, send_slice, recv_peer, recv_slice, timeout)
            return
        with self._buffer(source.device) as (buffer, _):
            send_stage = buffer[:capacity]
            recv_stage = buffer[capacity:2 * capacity]
            try:
                for start in range(0, total, capacity):
                    send_end = min(send_len, start + capacity)
                    recv_end = min(recv_len, start + capacity)
                    send_slice = send_stage[:send_end - start] if start < send_len else None
                    recv_slice = recv_stage[:recv_end - start] if start < recv_len else None
                    if send_slice is not None:
                        send_slice.copy_(src_bytes[start:send_end], non_blocking=False)
                    self._exchange_peers(send_peer, send_slice, recv_peer, recv_slice, timeout)
                    if recv_slice is not None:
                        dst_bytes[start:recv_end].copy_(recv_slice, non_blocking=False)
            finally:
                torch.cuda.current_stream(source.device).synchronize()

    def _agree_alltoall_gloo(
        self, outputs: Sequence[torch.Tensor], sources: Sequence[torch.Tensor], timeout: timedelta,
    ) -> None:
        world_size = self.size()
        rank = self.rank()
        local_meta = torch.tensor(
            [*[src.nbytes for src in sources], *[out.nbytes for out in outputs], COPY_DTYPES.index(outputs[0].dtype)],
            dtype=torch.int64,
        )
        gathered = [torch.empty_like(local_meta) for _ in range(world_size)]
        gather_opts = AllgatherOptions()
        if timeout > timedelta(0):
            gather_opts.timeout = timeout
        self._cpu_group.allgather([gathered], [local_meta], gather_opts).wait()  # type: ignore[attr-defined]
        matrix: list[list[int]] = [row.tolist() for row in gathered]
        expected_dtype = matrix[rank][-1]
        if any(row[-1] != expected_dtype for row in matrix) or any(
            matrix[src][dst] != matrix[dst][world_size + src]
            for src in range(world_size)
            for dst in range(world_size)
        ):
            raise ValueError("piper_relay all-to-all input and output dtypes and split sizes must match across ranks")

    def _alltoall_gloo(
        self, outputs: Sequence[torch.Tensor], sources: Sequence[torch.Tensor], options: AllToAllOptions,
    ) -> None:
        device = outputs[0].device
        world_size = self.size()
        rank = self.rank()
        if world_size > 1:
            self._agree_alltoall_gloo(outputs, sources, options.timeout)
        need_local_copy = bool(sources[rank].numel()) and outputs[rank].data_ptr() != sources[rank].data_ptr()
        try:
            if need_local_copy:
                _bytes(outputs[rank]).copy_(_bytes(sources[rank]), non_blocking=device.type == "cuda")
            for step in range(1, world_size):
                send_peer = (rank + step) % world_size
                recv_peer = (rank - step) % world_size
                src = sources[send_peer]
                dst = outputs[recv_peer]
                send_len = src.nbytes
                recv_len = dst.nbytes
                if send_len == 0 and recv_len == 0:
                    continue
                if send_len != recv_len and send_len and recv_len:
                    self._exchange_unequal_gloo(send_peer, src, recv_peer, dst, options.timeout)
                else:
                    chunk_inputs = [src] if send_len else []
                    chunk_outputs = [dst] if recv_len else []
                    with closing(self._copy_chunks(chunk_inputs, chunk_outputs, slots=2)) as chunks:
                        for chunk in chunks:
                            self._exchange_peers(
                                send_peer,
                                chunk.inputs[0] if chunk.inputs else None,
                                recv_peer,
                                chunk.outputs[0] if chunk.outputs else None,
                                options.timeout,
                            )
        finally:
            if need_local_copy and device.type == "cuda":
                torch.cuda.current_stream(device).synchronize()

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

    def __init__(self, job: Future[int | None], tensors: list[torch.Tensor], *, nested: bool = False) -> None:
        super().__init__()
        self._job = job
        self._tensors = tensors
        self._nested = nested

    def source_rank(self) -> int:
        source = self._job.result()
        if source is None:
            raise RuntimeError("source_rank is only available on point-to-point receive work")
        return source

    def _source_rank(self) -> int:
        return self.source_rank()

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


def _wire_p2p_tag(tag: int) -> int:
    if type(tag) is not int or not 0 <= tag <= _MAX_P2P_TAG:
        raise ValueError(f"piper_relay point-to-point tag must be an integer in [0, {_MAX_P2P_TAG}]")
    return _P2P_TAG_BASE + tag


def _resolve_alltoall_splits(
    tensor: torch.Tensor, split_sizes: Sequence[int] | None, world_size: int,
) -> list[int]:
    if not split_sizes:
        if tensor.shape[0] % world_size:
            raise ValueError(
                "piper_relay all_to_all_single dim 0 must be divisible by world size when splits are omitted",
            )
        return [tensor.shape[0] // world_size] * world_size
    if len(split_sizes) != world_size or any(type(size) is not int or size < 0 for size in split_sizes):
        raise ValueError("piper_relay all_to_all_single requires world_size non-negative integer split sizes")
    splits = list(split_sizes)
    if sum(splits) != tensor.shape[0]:
        raise ValueError("piper_relay all_to_all_single split sizes must sum to dim 0")
    return splits


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Transport payload bits unchanged, including BF16 and integer metadata."""
    return tensor.reshape(-1).view(torch.uint8)


__all__ = ["RelayOptions", "register_relay_backend"]
