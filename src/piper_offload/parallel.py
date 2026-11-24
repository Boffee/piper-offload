"""Concurrent in-process multi-rank runtime and ProcessGroup (``piper_parallel``).

``ParallelExecutor`` creates ``N >= 1`` persistent rank worker threads bound
concurrently to CPU or CUDA/HIP devices, one shared progress thread executing
ready collectives, and isolates PyTorch's global distributed and compiler state
until ``close()`` succeeds. The process-wide ``piper_parallel`` backend name is
registered once and creates groups only while a ``ParallelExecutor`` is open.

Contracts:

* **Process exclusivity:** ``ParallelExecutor`` and
  :class:`~piper_offload.sequential.SequentialExecutor` share one process-wide
  session lock and require no pre-existing default or named process groups.
* **Device topologies & disjointness:** All devices in one executor must share
  one type (``"cpu"`` or ``"cuda"``; indexed CPU strings such as ``"cpu:0"``
  canonicalize to ``"cpu"``). Independent multi-GPU placements
  (``["cuda:0", "cuda:1"]``) and shared-GPU placements (``["cuda:0", "cuda:0"]``)
  are supported. Whenever logical ranks share one physical device, every actual
  write destination must be disjoint from other ranks' declared inputs and
  outputs before collective mutation begins.
* **CUDA compute stream handoff:** Each CUDA rank worker owns one dedicated
  compute stream. Before worker execution, every rank waits for events captured
  from the caller's current stream on every executor CUDA device (rejecting
  CUDA graph capture). Callbacks using additional side streams must join those
  streams back onto the rank's current stream before returning or raising.
  Workers synchronize their compute streams before ``run()`` returns.
* **Supported operations:** Collectives only — ``barrier``, coalesced/single
  ``allreduce`` (``ReduceOp.SUM`` only), ``broadcast`` (``rootTensor=0``),
  ``scatter``, ``allgather`` (supporting uneven per-rank tensor lists and
  equal-chunk ``all_gather_single`` / ``all_gather_single_coalesced`` /
  ``_allgather_base`` / ``allgather_into_tensor_coalesced``),
  ``reduce_scatter`` / ``reduce_scatter_single`` /
  ``reduce_scatter_single_coalesced`` (and ``_reduce_scatter_base`` /
  ``reduce_scatter_tensor_coalesced`` over ``ReduceOp.SUM``), and ``alltoall`` /
  ``all_to_all_single`` / ``alltoall_base``. Point-to-point messaging
  (``send``/``recv``) is unsupported. Inputs and outputs must be contiguous
  strided ``torch.Tensor`` or ``Parameter`` objects without the negative-view
  or conjugate-view bit (``is_neg()`` / ``is_conj()``), not tensor subclasses or
  quantized tensors, and actual CPU write destinations must not overlap live
  Piper read-only checkpoint mappings (callers remain responsible for writable
  storage imported from other sources). CUDA graph capture is rejected.
* **In-place aliases & rank order:** Exact local own-slice aliases are allowed
  for ``scatter``, ``allgather``, ``reduce_scatter``, and ``alltoall``; partial
  overlaps and conflicting writes are rejected before mutation. Floating-point
  reductions accumulate across ranks in strict ascending group-rank order.
* **Transports & offloading coordination:** ``transport="auto"`` selects direct
  cross-GPU peer transfers (``"p2p"``) when ``can_device_access_peer`` holds for
  every distinct ordered GPU pair, or falls back to pinned host staging bounded
  by ``staging_bytes`` (``"host_staged"``); on CPU devices ``"auto"`` uses
  bounded CPU staging. CUDA transfers wait for producer readiness before
  entering :func:`~piper_offload._transfer_priority.foreground_transfer`.
* **Futures, subgroups, lifetimes & one-way poison:** ``Work.get_future()``
  waits synchronously for completion so ``.then()`` callbacks run on the calling
  rank thread with active runtime isolation; callers must unblock peer ranks
  before calling ``get_future()``. Explicit ``Work.wait(timeout=...)`` and
  subgroup timeouts are clamped by ``collective_timeout``.
  ``Work.block_current_stream()`` is unsupported and raises
  ``NotImplementedError``. Returned ``AsyncCollectiveTensor`` and ``DTensor``
  wrappers carrying async local tensors materialize before ``run()`` returns.
  Explicit ``destroy_process_group(pg)`` rejects unfinished local work while
  open; once idle ranks detach cleanly. A matched collective failing
  pre-mutation validation completes cleanly per rank; if caught without leaving
  broken state, the executor stays open. Any unhandled exception escaping a
  callback, timeout, controller interrupt, unfinished collective, or
  transport/sync error poisons the executor one-way: callers must ``close()``
  the executor before reopening it or creating a new ``ParallelExecutor``. If
  ``close()`` times out waiting for workers, transfers, or runtime
  restoration, isolation and the session lock remain held until a subsequent
  ``close()`` succeeds (affected buffers from an outlived CPU or GPU op may
  change until ``close()`` succeeds).
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, auto
from typing import Any, Literal

import torch
import torch.distributed as dist

# PyTorch 2.15 internal c10d classes are absent from type stubs.
from torch._C._distributed_c10d import (  # type: ignore[attr-defined]
    AllgatherOptions,
    AllToAllOptions,
    ReduceScatterOptions,
    _DistributedBackendOptions,
)
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_map_only

from ._devices import canonical_device
from ._parallel_collectives import (
    CollectiveCall,
    TransportMode,
    TransportPreference,
    _flat_1d,
    _ParallelTransport,
    _PreMutationError,
    resolve_alltoall_splits,
    resolve_transport,
    single_tensor,
    validate_alltoall_aliases,
    validate_batched_outputs,
    validate_disjoint,
    validate_own_slice_aliases,
    validate_pair,
    validate_reduction_tensors,
    validate_single_collective_shape,
    validate_tensors,
    validate_writable,
)
from ._sequential_runtime import (
    StartupGate,
    acquire_session,
    isolate_runtime,
    make_in_process_store,
    owns_session,
    release_session,
)

__all__ = ["ParallelExecutor"]

_BACKEND = "piper_parallel"
_MAX_COLLECTIVE_TIMEOUT = timedelta(days=365)
_MAX_WAIT_SECONDS = 3600.0


class _State(Enum):
    CLOSED = auto()
    STARTING = auto()
    OPEN = auto()
    RUNNING = auto()
    POISONED = auto()
    CLOSING = auto()


@dataclass
class _ModuleState:
    register_lock: threading.Lock = field(default_factory=threading.Lock)
    active_executor: ParallelExecutor | None = None


_state = _ModuleState()


@dataclass(frozen=True, slots=True)
class _ErrorSummary:
    """Traceback-free failure summary retained without holding tensor or frame refs."""

    exc_type: type[Exception]
    message: str

    def build(self) -> Exception:
        return self.exc_type(self.message)


def _summarize_error(exc: BaseException) -> _ErrorSummary:
    try:
        msg = str(exc) or type(exc).__name__
    except BaseException:
        msg = type(exc).__name__
    for candidate in (TimeoutError, ValueError, TypeError):
        if isinstance(exc, candidate):
            return _ErrorSummary(candidate, msg)
    return _ErrorSummary(RuntimeError, msg)


def _capture_ready_event(device: torch.device) -> torch.cuda.Event | None:
    if device.type != "cuda":
        return None
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("piper_parallel transfers do not support CUDA graph capture")
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        return event


def _capture_entry_events(devices: Sequence[torch.device]) -> dict[torch.device, torch.cuda.Event]:
    events: dict[torch.device, torch.cuda.Event] = {}
    for dev in devices:
        if dev.type == "cuda" and dev not in events:
            event = _capture_ready_event(dev)
            assert event is not None
            events[dev] = event
    return events


def _wait_stream_events(stream: torch.cuda.Stream, events: Sequence[torch.cuda.Event]) -> None:
    with torch.cuda.device(stream.device), torch.cuda.stream(stream):
        for event in events:
            stream.wait_event(event)


def _join_threads_until(threads: Sequence[threading.Thread], deadline: float) -> bool:
    alive = False
    for thread in threads:
        if thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=min(max(0.0, deadline - time.monotonic()), _MAX_WAIT_SECONDS))
            if thread.is_alive():
                alive = True
    return not alive


def _materialize_async(value: object) -> object:
    if isinstance(value, AsyncCollectiveTensor):
        return value.wait()
    if isinstance(value, DTensor) and isinstance(value._local_tensor, AsyncCollectiveTensor):
        value._local_tensor = value._local_tensor.wait()
    return value


SlotState = Literal["waiting", "queued", "running", "done"]


@dataclass(slots=True, eq=False)
class _CollectiveSlot:
    """Rendezvous slot for one ordered collective across all group ranks."""

    seq: int
    calls: dict[int, CollectiveCall] = field(default_factory=dict)
    events: dict[int, torch.cuda.Event | None] = field(default_factory=dict)
    contributed_ranks: set[int] = field(default_factory=set)
    state: SlotState = "waiting"
    error: _ErrorSummary | None = None
    deadline: float = float("inf")

    def clear_payload(self) -> None:
        self.calls.clear()
        self.events.clear()


class _PendingWork(dist.Work):
    """Passive observer over a channel's rendezvous slot."""

    def __init__(
        self,
        executor: ParallelExecutor,
        slot: _CollectiveSlot,
        tensors: list[torch.Tensor],
        *,
        nested: bool = False,
        default_timeout: timedelta,
    ) -> None:
        super().__init__()
        self._executor = executor
        self._slot = slot
        self._tensors = list(tensors)
        self._nested = nested
        self._default_timeout = default_timeout

    def source_rank(self) -> int:
        raise RuntimeError("piper_parallel source_rank is unavailable for collective operations")

    def _source_rank(self) -> int:
        return self.source_rank()

    def get_future(self) -> torch.futures.Future[Any]:
        """Wait for completion before returning a resolved future so callbacks run on caller thread."""
        torch_future: torch.futures.Future[Any] = torch.futures.Future()
        try:
            self.wait()
        except Exception as exc:
            torch_future.set_exception(exc)  # pyright: ignore[reportArgumentType]
        else:
            payload: Any = [self._tensors] if self._nested else self._tensors
            torch_future.set_result(payload)
        return torch_future

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        limit = min(timeout, self._default_timeout) if timeout > timedelta(0) else self._default_timeout
        self._executor._wait_slot(self._slot, limit.total_seconds())
        return True

    def is_completed(self) -> bool:
        with self._executor._cv:
            return self._slot.state == "done"

    def is_success(self) -> bool:
        with self._executor._cv:
            return self._slot.state == "done" and self._slot.error is None

    def result(self) -> list[torch.Tensor]:
        self.wait()
        return list(self._tensors)

    def synchronize(self) -> None:
        self.wait()

    def block_current_stream(self) -> None:
        raise NotImplementedError("piper_parallel does not support Work.block_current_stream(); call wait() instead")


class _GroupChannel:
    """Lock-free-buffer collective slot table synchronized under executor._cv."""

    def __init__(
        self,
        executor: ParallelExecutor,
        group_id: str,
        global_ranks: tuple[int, ...],
        default_timeout: timedelta,
    ) -> None:
        self._executor = executor
        self.group_id = group_id
        self.global_ranks = global_ranks
        self.size = len(global_ranks)
        self._default_timeout = default_timeout

        self._rank_seqs = [0] * self.size
        self._slots: dict[int, _CollectiveSlot] = {}
        self._next_exec_seq = 0
        self._running_slot: _CollectiveSlot | None = None
        self._registered_ranks: set[int] = set()
        self._detached_ranks: set[int] = set()
        self._retired = False

    def register_locked(self, group_rank: int) -> None:
        if group_rank in self._detached_ranks:
            raise RuntimeError(f"piper_parallel rank {group_rank} already detached from group {self.group_id!r}")
        self._registered_ranks.add(group_rank)

    def submit_locked(
        self,
        group_rank: int,
        call: CollectiveCall,
        ready_event: torch.cuda.Event | None,
        timeout_seconds: float,
    ) -> _CollectiveSlot:
        seq = self._rank_seqs[group_rank]
        self._rank_seqs[group_rank] += 1
        slot = self._slots.get(seq)
        if slot is None:
            slot = _CollectiveSlot(seq=seq)
            self._slots[seq] = slot
        slot.contributed_ranks.add(group_rank)
        slot.calls[group_rank] = call
        slot.events[group_rank] = ready_event
        now = time.monotonic()
        if slot.deadline <= now:
            summary = _ErrorSummary(TimeoutError, "piper_parallel operation timed out")
            self._fail_slot_locked(slot, summary)
            self._executor._poison_locked(summary)
            return slot
        op_deadline = now + timeout_seconds
        tightened = op_deadline < slot.deadline
        if tightened:
            slot.deadline = op_deadline

        if self._detached_ranks:
            self._fail_slot_locked(
                slot,
                _ErrorSummary(
                    RuntimeError,
                    f"piper_parallel group {self.group_id!r} has a detached peer rank",
                ),
            )
            return slot

        self._maybe_enqueue_head_locked()
        if tightened:
            self._executor._cv.notify_all()
        return slot

    def _fail_slot_locked(self, slot: _CollectiveSlot, summary: _ErrorSummary) -> None:
        if slot.state == "done":
            return
        slot.error = summary
        slot.state = "done"
        slot.clear_payload()
        self._complete_slot_locked(slot)

    def _maybe_enqueue_head_locked(self) -> None:
        if self._running_slot is not None or self._detached_ranks:
            return
        if self._executor._state in (_State.POISONED, _State.CLOSING, _State.CLOSED):
            return
        while True:
            head = self._slots.get(self._next_exec_seq)
            if head is None:
                return
            if head.state == "done":
                del self._slots[self._next_exec_seq]
                self._next_exec_seq += 1
                continue
            if head.state == "waiting" and len(head.calls) == self.size:
                head.state = "queued"
                self._executor._ready_queue.append((self, head))
                self._executor._cv.notify_all()
            return

    def _complete_slot_locked(self, slot: _CollectiveSlot) -> None:
        if self._slots.get(slot.seq) is slot:
            del self._slots[slot.seq]
        while self._next_exec_seq not in self._slots and self._next_exec_seq <= slot.seq:
            self._next_exec_seq += 1
        self._maybe_enqueue_head_locked()
        self._maybe_retire_locked()
        self._executor._cv.notify_all()

    def earliest_deadline_locked(self) -> float | None:
        deadlines: list[float] = []
        if self._running_slot is not None and self._running_slot.error is None:
            deadlines.append(self._running_slot.deadline)
        deadlines.extend(
            slot.deadline
            for slot in self._slots.values()
            if slot.state != "done" and slot.error is None
        )
        return min(deadlines) if deadlines else None

    def abort_pending_locked(self, summary: _ErrorSummary) -> None:
        if self._running_slot is not None and self._running_slot.error is None:
            self._running_slot.error = summary
        for slot in list(self._slots.values()):
            if slot.state in ("waiting", "queued"):
                slot.error = summary
                slot.state = "done"
                slot.clear_payload()
                if self._slots.get(slot.seq) is slot:
                    del self._slots[slot.seq]
        self._maybe_retire_locked()

    def detach_locked(self, group_rank: int) -> None:
        if group_rank in self._detached_ranks:
            return
        if self._executor._state in (_State.STARTING, _State.OPEN, _State.RUNNING):
            running_has = self._running_slot is not None and group_rank in self._running_slot.contributed_ranks
            waiting_has = any(
                s.state != "done" and group_rank in s.contributed_ranks for s in self._slots.values()
            )
            if running_has or waiting_has:
                raise RuntimeError(
                    f"cannot destroy piper_parallel group {self.group_id!r} while rank {group_rank} "
                    "has unfinished collectives",
                )

        self._detached_ranks.add(group_rank)
        detached_err = _ErrorSummary(
            RuntimeError,
            f"piper_parallel group {self.group_id!r} peer rank {group_rank} destroyed group early",
        )
        for slot in list(self._slots.values()):
            if slot.state in ("waiting", "queued") and group_rank not in slot.contributed_ranks:
                self._fail_slot_locked(slot, detached_err)
        self._maybe_retire_locked()
        self._executor._cv.notify_all()

    def has_unfinished_locked(self) -> bool:
        if self._running_slot is not None:
            return True
        return any(slot.state != "done" for slot in self._slots.values())

    def _maybe_retire_locked(self) -> None:
        if self._retired:
            return
        if len(self._detached_ranks) >= self.size and not self.has_unfinished_locked():
            self._retired = True
            if self._executor._channels.get(self.group_id) is self:
                del self._executor._channels[self.group_id]


class _ParallelProcessGroup(dist.ProcessGroup):
    """Per-rank PyTorch ProcessGroup backed by an in-process _GroupChannel."""

    def __init__(
        self,
        executor: ParallelExecutor,
        rank: int,
        size: int,
        channel: _GroupChannel,
        group_id: str,
        device: torch.device,
        default_timeout: timedelta,
    ) -> None:
        super().__init__(rank, size)
        self._executor = executor
        self._channel = channel
        self._group_id = group_id
        self._device = device
        self._default_timeout = default_timeout
        self._set_group_name(group_id)  # type: ignore[arg-type]

    def getBackendName(self) -> str:  # noqa: N802 -- PyTorch ProcessGroup virtual API
        return _BACKEND

    @property
    def group_name(self) -> str:  # type: ignore[override]
        return self._group_id

    def _get_backend(self, device: torch.device) -> _ParallelProcessGroup:  # type: ignore[override]  # noqa: ARG002
        return self

    def _is_active_handle_locked(self) -> bool:
        return (
            self._executor._channels.get(self._group_id) is self._channel
            and self.rank() not in self._channel._detached_ranks
        )

    def shutdown(self) -> None:
        with self._executor._cv:
            if self._executor._channels.get(self._group_id) is not self._channel:
                return
            self._channel.detach_locked(self.rank())

    def abort(self) -> None:
        with self._executor._cv:
            if not self._is_active_handle_locked():
                return
            self._executor._poison_locked(_ErrorSummary(RuntimeError, "process group aborted"))

    def _effective_timeout(self, opts: object) -> timedelta:
        opt_timeout = getattr(opts, "timeout", None)
        if isinstance(opt_timeout, timedelta) and opt_timeout > timedelta(0):
            return min(opt_timeout, self._default_timeout)
        return self._default_timeout

    def _submit(
        self,
        call: CollectiveCall,
        outputs: list[torch.Tensor],
        opts: object,
        *,
        nested: bool = False,
    ) -> dist.Work:
        ready_event = _capture_ready_event(self._device)
        timeout = self._effective_timeout(opts)
        with self._executor._cv:
            if self._executor._state in (_State.POISONED, _State.CLOSING, _State.CLOSED):
                err = self._executor._poison_error or _ErrorSummary(RuntimeError, "process group shut down")
                raise err.build()
            if self._executor._state is not _State.RUNNING:
                raise RuntimeError("piper_parallel collectives require an active ParallelExecutor.run() call")
            global_rank = self._channel.global_ranks[self.rank()]
            if (
                global_rank >= len(self._executor._workers)
                or threading.current_thread() is not self._executor._workers[global_rank].thread
            ):
                raise RuntimeError(
                    f"piper_parallel rank {global_rank} collective must run on its assigned rank worker thread",
                )
            if not self._is_active_handle_locked():
                raise RuntimeError(f"piper_parallel group {self._group_id!r} handle has been destroyed")
            slot = self._channel.submit_locked(self.rank(), call, ready_event, timeout.total_seconds())
        return _PendingWork(
            self._executor,
            slot,
            outputs,
            nested=nested,
            default_timeout=timeout,
        )

    def _validate_rank_arg(self, peer: int, name: str) -> None:
        if type(peer) is not int or peer < 0 or peer >= self.size():
            raise ValueError(f"piper_parallel invalid {name} rank {peer} for group size {self.size()}")

    def barrier(self, opts: dist.BarrierOptions | None = None) -> dist.Work:  # type: ignore[override]
        options = opts if opts is not None else dist.BarrierOptions()
        return self._submit(CollectiveCall(op="barrier", rank=self.rank()), [], options)

    def allreduce(  # type: ignore[override]
        self,
        tensors: list[torch.Tensor],
        opts: dist.AllreduceOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else dist.AllreduceOptions()
        tensor = single_tensor(tensors)
        validate_reduction_tensors([tensor], self._device, options.reduceOp)
        if self.size() > 1:
            validate_writable([tensor])
        return self._submit(
            CollectiveCall(op="allreduce", rank=self.rank(), tensors=(tensor,)),
            [tensor],
            options,
        )

    def allreduce_coalesced(
        self,
        tensors: list[torch.Tensor],
        opts: dist.AllreduceCoalescedOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else dist.AllreduceCoalescedOptions()
        if not tensors:
            raise ValueError("piper_parallel allreduce_coalesced requires a non-empty tensor list")
        validate_reduction_tensors(tensors, self._device, options.reduceOp)
        validate_disjoint(tensors, "piper_parallel allreduce_coalesced tensors must not overlap")
        if self.size() > 1:
            validate_writable(tensors)
        items = list(tensors)
        return self._submit(
            CollectiveCall(op="allreduce", rank=self.rank(), tensors=tuple(items)),
            items,
            options,
        )

    def broadcast(  # type: ignore[override]
        self,
        tensors: list[torch.Tensor],
        opts: dist.BroadcastOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else dist.BroadcastOptions()
        tensor = single_tensor(tensors)
        validate_tensors([tensor], self._device)
        self._validate_rank_arg(options.rootRank, "root")
        if options.rootTensor != 0:
            raise ValueError("piper_parallel broadcast requires rootTensor=0")
        if self.rank() != options.rootRank:
            validate_writable([tensor])
        return self._submit(
            CollectiveCall(op="broadcast", rank=self.rank(), tensors=(tensor,), root=options.rootRank),
            [tensor],
            options,
        )

    def scatter(  # type: ignore[override]
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: dist.ScatterOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else dist.ScatterOptions()
        output = single_tensor(output_tensors)
        self._validate_rank_arg(options.rootRank, "root")
        sources: list[torch.Tensor] = []
        if self.rank() == options.rootRank:
            if len(input_tensors) != 1 or len(input_tensors[0]) != self.size():
                raise ValueError("piper_parallel scatter root must provide one tensor per rank")
            sources = list(input_tensors[0])
        elif input_tensors:
            raise ValueError("piper_parallel scatter inputs belong only on the root rank")
        validate_tensors([output, *sources], self._device)
        for src_tensor in sources:
            validate_pair(output, src_tensor, src_tensor.numel(), same_shape=True)
        if sources:
            validate_own_slice_aliases(
                options.rootRank,
                output,
                sources,
                "piper_parallel scatter output must not overlap inputs except its exact root slice",
            )
            if output.data_ptr() != sources[options.rootRank].data_ptr():
                validate_writable([output])
        else:
            validate_writable([output])
        grouped = (tuple(sources),) if sources else ()
        return self._submit(
            CollectiveCall(
                op="scatter",
                rank=self.rank(),
                tensors=(output,),
                grouped_tensors=grouped,
                root=options.rootRank,
            ),
            [output],
            options,
        )

    def allgather(  # type: ignore[override]
        self,
        output_tensors: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: AllgatherOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else AllgatherOptions()
        source = single_tensor(input_tensors)
        if len(output_tensors) != 1 or len(output_tensors[0]) != self.size():
            raise ValueError("piper_parallel all-gather requires one output tensor per rank")
        destinations = list(output_tensors[0])
        validate_tensors([source, *destinations], self._device)
        validate_disjoint(destinations, "piper_parallel all-gather outputs must not overlap")
        if any(out.dtype != source.dtype for out in destinations):
            raise ValueError("piper_parallel collective tensor dtypes must match")
        validate_pair(destinations[self.rank()], source, source.numel(), same_shape=True)
        validate_own_slice_aliases(
            self.rank(),
            source,
            destinations,
            "piper_parallel all-gather input may overlap only its exact local output slice",
        )
        validate_writable(
            [
                out
                for r, out in enumerate(destinations)
                if not (r == self.rank() and out.data_ptr() == source.data_ptr())
            ],
        )
        return self._submit(
            CollectiveCall(
                op="allgather",
                rank=self.rank(),
                tensors=(source,),
                grouped_tensors=(tuple(destinations),),
            ),
            destinations,
            options,
            nested=True,
        )

    def all_gather_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch ProcessGroup API name
        opts: AllgatherOptions | None = None,
    ) -> dist.Work:
        return self.all_gather_single_coalesced([output], [input], opts)

    def _allgather_base(
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch ProcessGroup API name
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
            raise ValueError("piper_parallel all-gather requires matching nonempty input/output lists")
        validate_tensors([*input_list, *output_lists], self._device)
        validate_batched_outputs(output_lists, input_list)
        destinations: list[tuple[torch.Tensor, ...]] = []
        writable: list[torch.Tensor] = []
        for output, source in zip(output_lists, input_list, strict=True):
            validate_single_collective_shape(output, source, self.size())
            views = tuple(_flat_1d(output).view(self.size(), *source.shape).unbind(0))
            validate_own_slice_aliases(
                self.rank(),
                source,
                views,
                "piper_parallel all-gather input may overlap only its exact local output slice",
            )
            writable.extend(
                v for r, v in enumerate(views) if not (r == self.rank() and v.data_ptr() == source.data_ptr())
            )
            destinations.append(views)
        validate_writable(writable)
        return self._submit(
            CollectiveCall(
                op="allgather",
                rank=self.rank(),
                tensors=tuple(input_list),
                grouped_tensors=tuple(destinations),
            ),
            list(output_lists),
            options,
        )

    def allgather_into_tensor_coalesced(
        self,
        output_lists: list[torch.Tensor],
        input_list: list[torch.Tensor],
        opts: AllgatherOptions | None = None,
    ) -> dist.Work:
        return self.all_gather_single_coalesced(output_lists, input_list, opts)

    def reduce_scatter(  # type: ignore[override]
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: ReduceScatterOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else ReduceScatterOptions()
        output = single_tensor(output_tensors)
        if len(input_tensors) != 1 or len(input_tensors[0]) != self.size():
            raise ValueError("piper_parallel reduce-scatter requires one input tensor per rank")
        sources = list(input_tensors[0])
        validate_reduction_tensors([output, *sources], self._device, options.reduceOp)
        for src in sources:
            validate_pair(output, src, src.numel(), same_shape=True)
        validate_own_slice_aliases(
            self.rank(),
            output,
            sources,
            "piper_parallel reduce-scatter output may overlap only its exact local input slice",
        )
        if not (self.size() == 1 and output.data_ptr() == sources[0].data_ptr()):
            validate_writable([output])
        return self._submit(
            CollectiveCall(
                op="reduce_scatter",
                rank=self.rank(),
                tensors=(output,),
                grouped_tensors=(tuple(sources),),
            ),
            [output],
            options,
        )

    def reduce_scatter_single(  # type: ignore[override]
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch ProcessGroup API name
        opts: ReduceScatterOptions | None = None,
    ) -> dist.Work:
        return self.reduce_scatter_single_coalesced([output], [input], opts)

    def _reduce_scatter_base(  # type: ignore[override]
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch ProcessGroup API name
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
            raise ValueError("piper_parallel reduce-scatter requires matching nonempty output/input lists")
        validate_reduction_tensors([*output_tensors, *input_tensors], self._device, options.reduceOp)
        validate_batched_outputs(output_tensors, input_tensors)
        sources_list: list[tuple[torch.Tensor, ...]] = []
        writable: list[torch.Tensor] = []
        for output, source in zip(output_tensors, input_tensors, strict=True):
            validate_single_collective_shape(source, output, self.size())
            views = tuple(_flat_1d(source).view(self.size(), *output.shape).unbind(0))
            validate_own_slice_aliases(
                self.rank(),
                output,
                views,
                "piper_parallel reduce-scatter output may overlap only its exact local input slice",
            )
            if not (self.size() == 1 and output.data_ptr() == views[0].data_ptr()):
                writable.append(output)
            sources_list.append(views)
        validate_writable(writable)
        return self._submit(
            CollectiveCall(
                op="reduce_scatter",
                rank=self.rank(),
                tensors=tuple(output_tensors),
                grouped_tensors=tuple(sources_list),
            ),
            list(output_tensors),
            options,
        )

    def reduce_scatter_tensor_coalesced(
        self,
        outputTensors: list[torch.Tensor],  # noqa: N803 -- PyTorch ProcessGroup API name
        inputTensors: list[torch.Tensor],  # noqa: N803 -- PyTorch ProcessGroup API name
        opts: ReduceScatterOptions | None = None,
    ) -> dist.Work:
        return self.reduce_scatter_single_coalesced(outputTensors, inputTensors, opts)

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
            raise ValueError("piper_parallel all-to-all requires one input and one output tensor per rank")
        validate_tensors([*outputs, *sources], self._device)
        dtype = outputs[0].dtype
        if any(tensor.dtype != dtype for tensor in (*outputs, *sources)):
            raise ValueError("piper_parallel all-to-all requires one dtype across all input and output tensors")
        validate_disjoint(outputs, "piper_parallel all-to-all outputs must not overlap")
        validate_alltoall_aliases(self.rank(), outputs, sources)
        validate_writable(
            [
                out
                for r, out in enumerate(outputs)
                if not (r == self.rank() and out.data_ptr() == sources[r].data_ptr())
            ],
        )
        return self._submit(
            CollectiveCall(
                op="alltoall",
                rank=self.rank(),
                tensors=tuple(outputs),
                grouped_tensors=(tuple(sources),),
            ),
            outputs,
            options,
        )

    def all_to_all_single(  # type: ignore[override]
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch ProcessGroup API name
        output_split_sizes: Sequence[int] | None = None,
        input_split_sizes: Sequence[int] | None = None,
        opts: AllToAllOptions | None = None,
    ) -> dist.Work:
        options = opts if opts is not None else AllToAllOptions()
        validate_tensors([output, input], self._device)
        if output.dtype != input.dtype:
            raise ValueError("piper_parallel all-to-all input and output dtypes must match")
        if output.ndim < 1 or input.ndim < 1:
            raise ValueError("piper_parallel all_to_all_single requires tensors with at least one dimension")
        if output.shape[1:] != input.shape[1:]:
            raise ValueError("piper_parallel all_to_all_single input and output trailing dimensions must match")
        out_splits = resolve_alltoall_splits(output, output_split_sizes, self.size())
        in_splits = resolve_alltoall_splits(input, input_split_sizes, self.size())
        out_views = tuple(output.split(out_splits, dim=0))
        in_views = tuple(input.split(in_splits, dim=0))
        validate_alltoall_aliases(self.rank(), out_views, in_views)
        validate_writable(
            [
                out
                for r, out in enumerate(out_views)
                if not (r == self.rank() and out.data_ptr() == in_views[r].data_ptr())
            ],
        )
        return self._submit(
            CollectiveCall(
                op="alltoall_single",
                rank=self.rank(),
                tensors=out_views,
                grouped_tensors=(in_views,),
            ),
            [output],
            options,
        )

    def alltoall_base(  # type: ignore[override]
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch ProcessGroup API name
        output_split_sizes: Sequence[int] | None = None,
        input_split_sizes: Sequence[int] | None = None,
        opts: AllToAllOptions | None = None,
    ) -> dist.Work:
        return self.all_to_all_single(output, input, output_split_sizes, input_split_sizes, opts)


def _create_parallel_group(
    backend_options: _DistributedBackendOptions,
    _pg_options: object,
) -> dist.ProcessGroup:
    executor = _state.active_executor
    if executor is None:
        raise RuntimeError("piper_parallel backend requires an active ParallelExecutor")
    return executor._create_group(backend_options)


def _ensure_backend_registered() -> None:
    with _state.register_lock:
        plugin = dist.Backend._plugins.get(_BACKEND.upper())  # type: ignore[attr-defined]
        if plugin is not None and getattr(plugin, "creator_fn", None) is not _create_parallel_group:
            raise RuntimeError("a different backend is already registered as piper_parallel")
        if plugin is None:
            dist.Backend.register_backend(
                _BACKEND,
                _create_parallel_group,
                extended_api=True,
                devices=["cpu", "cuda"],
            )


_MAX_CUDA_INDEX = 127  # PyTorch c10::DeviceIndex is signed int8_t.


def _parse_device_entry(entry: torch.device | str | int) -> torch.device:
    if isinstance(entry, bool):
        raise ValueError("device entries cannot be booleans")
    if isinstance(entry, int):
        if entry < 0:
            raise ValueError(f"CUDA device index must be non-negative, got {entry}")
        if entry > _MAX_CUDA_INDEX:
            raise ValueError(f"CUDA device index out of range: {entry}")
        return torch.device("cuda", entry)
    if isinstance(entry, str):
        dev_type, sep, idx_text = entry.strip().partition(":")
        if sep and dev_type.lower() == "cuda":
            try:
                raw_idx = int(idx_text.strip())
            except ValueError:
                pass
            else:
                if raw_idx < 0 or raw_idx > _MAX_CUDA_INDEX:
                    raise ValueError(f"CUDA device index out of range: {raw_idx}")
    dev = canonical_device(entry)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_available() else 0)
    return dev


def _normalize_devices(devices: Sequence[torch.device | str | int] | None) -> tuple[torch.device, ...]:
    if devices is None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return tuple(torch.device("cuda", idx) for idx in range(torch.cuda.device_count()))
        return (torch.device("cpu"), torch.device("cpu"))

    if isinstance(devices, (str, bytes, torch.device)):
        raise TypeError("devices must be a sequence of devices, not a single device")

    normalized = [_parse_device_entry(entry) for entry in devices]
    if not normalized:
        raise ValueError("devices must not be empty")
    first_type = normalized[0].type
    if first_type not in ("cpu", "cuda"):
        raise ValueError(f"unsupported device type {first_type!r}; expected 'cpu' or 'cuda'")
    if any(dev.type != first_type for dev in normalized[1:]):
        raise ValueError("ParallelExecutor devices must all share one device type ('cpu' or 'cuda')")
    return tuple(normalized)


@dataclass(slots=True, eq=False)
class _PublishedBatch:
    run_id: int
    fn: Callable[[int], Any] | None
    entry_events: tuple[torch.cuda.Event, ...]
    results: list[Any]
    finished_ranks: set[int] = field(default_factory=set)
    discarded: bool = False
    first_error: BaseException | None = None
    failure_time: float | None = None

    def clear_inputs(self) -> None:
        self.fn = None
        self.entry_events = ()


@dataclass(slots=True, eq=False)
class _WorkerSlot:
    rank: int
    device: torch.device
    thread: threading.Thread | None = None
    last_run_id: int = 0
    compute_stream: torch.cuda.Stream | None = None
    stream_needs_drain: bool = False


class ParallelExecutor:
    """Run ``N`` logical ranks concurrently in persistent worker threads.

    Ranks execute simultaneously across ``N`` CPU or CUDA/HIP device placements
    (including shared-GPU topologies such as ``["cuda:0", "cuda:0"]``) and
    run collectives over the process-registered ``piper_parallel`` backend using
    direct GPU peer copies (``"p2p"``), bounded pinned host staging
    (``"host_staged"``), or bounded CPU staging.

    Parameters
    ----------
    devices:
        Sequence of CPU or CUDA/HIP device targets (one per rank). Defaults to
        every visible CUDA device when available, or ``("cpu", "cpu")``.
    transport:
        ``"auto"`` (default) picks ``"p2p"`` when all distinct GPU pairs support
        peer access and otherwise falls back to ``"host_staged"`` (or CPU
        staging for CPU devices). Passing ``"p2p"`` raises when any distinct
        pair lacks peer access; passing ``"p2p"`` or ``"host_staged"`` rejects
        CPU devices.
    staging_bytes:
        Maximum host staging or GPU scratch footprint in bytes (strictly
        positive ``int``, default 16 MiB). ``"host_staged"`` keeps at most one
        reusable host DMA buffer up to this limit while open.
    collective_timeout:
        Positive deadline governing collective completion, explicit waits,
        subgroup timeouts, failure draining, and ``close()`` cleanup.

    Raises
    ------
    RuntimeError
        On ``__enter__`` if another ``ParallelExecutor`` or
        ``SequentialExecutor`` is already open or if any PyTorch distributed
        process group is already initialized.
    TimeoutError
        On ``close()`` if workers, progress execution, or pending transfers fail
        to finish before ``collective_timeout``. Isolation, channels, pin leases,
        and the session lock remain held until a subsequent ``close()`` succeeds.
    """

    def __init__(
        self,
        devices: Sequence[torch.device | str | int] | None = None,
        *,
        transport: TransportPreference = "auto",
        staging_bytes: int = 16 * 1024 * 1024,
        collective_timeout: timedelta = timedelta(seconds=30),
    ) -> None:
        if type(staging_bytes) is not int or staging_bytes <= 0:
            raise ValueError("staging_bytes must be a positive integer")
        if not isinstance(collective_timeout, timedelta) or collective_timeout <= timedelta(0):
            raise ValueError("collective_timeout must be a positive timedelta")
        if collective_timeout > _MAX_COLLECTIVE_TIMEOUT:
            raise ValueError("collective_timeout must be at most 365 days")

        self._devices = _normalize_devices(devices)
        self._transport_pref = transport
        self._resolved_transport: TransportMode = resolve_transport(self._devices, transport)
        self._staging_bytes = staging_bytes
        self._collective_timeout = collective_timeout

        self._controller_lock = threading.Lock()
        self._cv = threading.Condition()
        self._state = _State.CLOSED
        self._startup_gate: StartupGate | None = None
        self._runtime: Any = None
        self._transport: _ParallelTransport | None = None
        self._workers: list[_WorkerSlot] = []
        self._progress_thread: threading.Thread | None = None
        self._workers_stop = False
        self._progress_stop = False
        self._run_id = 0
        self._next_run_id = 1
        self._current_batch: _PublishedBatch | None = None
        self._channels: dict[str, _GroupChannel] = {}
        self._ready_queue: deque[tuple[_GroupChannel, _CollectiveSlot]] = deque()
        self._poison_error: _ErrorSummary | None = None

    @property
    def devices(self) -> tuple[torch.device, ...]:
        return self._devices

    @property
    def world_size(self) -> int:
        return len(self._devices)

    @property
    def resolved_transport(self) -> TransportMode:
        return self._resolved_transport

    @property
    def staging_bytes(self) -> int:
        return self._staging_bytes

    @property
    def collective_timeout(self) -> timedelta:
        return self._collective_timeout

    @property
    def is_open(self) -> bool:
        return self._state in (_State.OPEN, _State.RUNNING)

    def _is_internal_thread(self) -> bool:
        curr = threading.current_thread()
        if curr is self._progress_thread:
            return True
        return any(slot.thread is curr for slot in self._workers)

    def _poison_locked(self, summary: _ErrorSummary) -> None:
        if self._poison_error is None:
            self._poison_error = summary
        if self._state not in (_State.CLOSING, _State.CLOSED):
            self._state = _State.POISONED
        if self._current_batch is not None and self._current_batch.failure_time is None:
            self._current_batch.failure_time = time.monotonic()
        self._ready_queue.clear()
        for channel in list(self._channels.values()):
            channel.abort_pending_locked(self._poison_error)
        self._cv.notify_all()

    def _wait_slot(self, slot: _CollectiveSlot, timeout_seconds: float) -> None:
        local_deadline = time.monotonic() + timeout_seconds
        with self._cv:
            while slot.state != "done":
                if self._state in (_State.POISONED, _State.CLOSING, _State.CLOSED):
                    err = slot.error or self._poison_error or _ErrorSummary(RuntimeError, "process group shut down")
                    raise err.build()
                deadline = min(local_deadline, slot.deadline)
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    err = TimeoutError("piper_parallel operation timed out")
                    self._poison_locked(_summarize_error(err))
                    raise err
                self._cv.wait(timeout=min(remaining, _MAX_WAIT_SECONDS))
            error = slot.error
        if error is not None:
            raise error.build()

    def __enter__(self) -> ParallelExecutor:
        if not self._controller_lock.acquire(blocking=False):
            raise RuntimeError("ParallelExecutor cannot open while run or close is active")
        try:
            with self._cv:
                if (
                    self._state is not _State.CLOSED
                    or self._workers
                    or self._progress_thread is not None
                    or owns_session(self)
                    or self._startup_gate is not None
                ):
                    raise RuntimeError("ParallelExecutor cannot be reopened; close() first if still closing")
            if not acquire_session(self):
                raise RuntimeError("another executor is already open")

            try:
                with self._cv:
                    self._state = _State.STARTING
                    self._poison_error = None
                    self._workers_stop = False
                    self._progress_stop = False
                if dist.is_initialized() or dist.distributed_c10d._world.pg_map:
                    raise RuntimeError("ParallelExecutor requires no existing distributed process groups")

                _ensure_backend_registered()
                self._transport = _ParallelTransport(self._devices, self._resolved_transport, self._staging_bytes)
                self._runtime = isolate_runtime(backend=_BACKEND)
                self._runtime.enter()
                _state.active_executor = self
                self._progress_thread = threading.Thread(
                    target=self._progress_loop,
                    name="piper-parallel-progress",
                    daemon=False,
                )
                self._progress_thread.start()
                self._startup_gate = StartupGate()
                self._start_workers(make_in_process_store())
                with self._cv:
                    self._state = _State.OPEN
            except BaseException:
                self._close_locked()
                raise
        finally:
            self._controller_lock.release()
        return self

    def _run_worker_entry(
        self,
        slot: _WorkerSlot,
        store: dist.Store,
        gate: StartupGate,
        startup_lock: threading.Lock,
        startup_done: threading.Condition,
        signal_ready: Callable[[BaseException | None], None],
    ) -> None:
        if not gate.try_admit():
            return
        try:
            try:
                if slot.device.type == "cuda":
                    torch.cuda.set_device(slot.device)
                    slot.compute_stream = torch.cuda.Stream(slot.device)
                dist.init_process_group(
                    backend=_BACKEND,
                    store=store,
                    rank=slot.rank,
                    world_size=self.world_size,
                    timeout=self._collective_timeout,
                )
            except BaseException as exc:
                with startup_lock:
                    signal_ready(exc)
                    startup_done.notify_all()
                return

            with startup_lock:
                signal_ready(None)
                startup_done.notify_all()

            try:
                self._worker_loop(slot)
            except BaseException as fatal:
                now = time.monotonic()
                summary = _summarize_error(fatal)
                with self._cv:
                    batch = self._current_batch
                    if batch is not None and slot.last_run_id == batch.run_id:
                        if batch.first_error is None:
                            batch.first_error = fatal
                        if batch.failure_time is None:
                            batch.failure_time = now
                        batch.finished_ranks.add(slot.rank)
                    self._poison_locked(summary)
            finally:
                with contextlib.suppress(BaseException):
                    if dist.is_initialized():
                        dist.destroy_process_group()
        finally:
            gate.mark_exited()

    def _start_workers(self, store: dist.Store) -> None:
        gate = self._startup_gate
        assert gate is not None
        startup_lock = threading.Lock()
        startup_done = threading.Condition(startup_lock)
        ready_count = 0
        startup_error: list[BaseException] = []

        def _signal_ready(exc: BaseException | None) -> None:
            nonlocal ready_count
            if exc is not None and not startup_error:
                startup_error.append(exc)
            ready_count += 1

        self._workers = [
            _WorkerSlot(rank=r, device=self._devices[r], last_run_id=self._run_id)
            for r in range(self.world_size)
        ]
        for slot in self._workers:
            thread = threading.Thread(
                target=self._run_worker_entry,
                args=(slot, store, gate, startup_lock, startup_done, _signal_ready),
                name=f"piper-parallel-rank-{slot.rank}",
                daemon=False,
            )
            slot.thread = thread
            thread.start()

        deadline = time.monotonic() + self._collective_timeout.total_seconds()
        with startup_lock:
            while ready_count < self.world_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("ParallelExecutor workers did not initialize before collective_timeout")
                startup_done.wait(timeout=min(remaining, _MAX_WAIT_SECONDS))
            if startup_error:
                raise startup_error[0]

    def _expire_slot_locked(self, channel: _GroupChannel, slot: _CollectiveSlot) -> None:
        summary = _ErrorSummary(TimeoutError, "piper_parallel operation timed out")
        if slot.error is None:
            slot.error = summary
        slot.state = "done"
        slot.clear_payload()
        self._poison_locked(summary)
        channel._complete_slot_locked(slot)

    def _finish_progress_slot_locked(
        self,
        channel: _GroupChannel,
        slot: _CollectiveSlot,
        pre_mutation: ValueError | TypeError | None,
        exc_caught: BaseException | None,
    ) -> None:
        channel._running_slot = None
        if pre_mutation is not None:
            if slot.error is None:
                slot.error = _summarize_error(pre_mutation)
        elif exc_caught is not None:
            summary = _summarize_error(exc_caught)
            if slot.error is None:
                slot.error = summary
            self._poison_locked(summary)
        elif slot.error is None and time.monotonic() >= slot.deadline:
            summary = _ErrorSummary(TimeoutError, "piper_parallel operation timed out")
            slot.error = summary
            self._poison_locked(summary)
        slot.state = "done"
        channel._complete_slot_locked(slot)

    def _progress_loop(self) -> None:
        while True:
            channel: _GroupChannel | None = None
            slot: _CollectiveSlot | None = None
            calls: list[CollectiveCall] = []
            events: list[torch.cuda.Event | None] = []
            with self._cv:
                while not self._progress_stop and not self._ready_queue:
                    self._cv.wait()
                if self._progress_stop:
                    return
                channel, slot = self._ready_queue.popleft()
                if slot.state != "queued":
                    continue
                if time.monotonic() >= slot.deadline:
                    self._expire_slot_locked(channel, slot)
                    continue
                slot.state = "running"
                calls = [slot.calls[r] for r in range(channel.size)]
                events = [slot.events[r] for r in range(channel.size)]
                slot.clear_payload()
                channel._running_slot = slot
                self._cv.notify_all()

            exc_caught: BaseException | None = None
            pre_mutation: ValueError | TypeError | None = None
            try:
                assert self._transport is not None
                self._transport.execute_collective(calls, channel.global_ranks, events)
            except _PreMutationError as err:
                pre_mutation = err.cause
            except BaseException as exc:
                exc_caught = exc
            finally:
                calls.clear()
                events.clear()

            with self._cv:
                self._finish_progress_slot_locked(channel, slot, pre_mutation, exc_caught)
            pre_mutation = None
            exc_caught = None
            channel = None
            slot = None

    def _drain_rank_stream(self, slot: _WorkerSlot) -> None:
        stream = slot.compute_stream
        if stream is None:
            return
        with self._cv:
            slot.stream_needs_drain = True
        stream.synchronize()
        with self._cv:
            slot.stream_needs_drain = False

    def _execute_worker_task(
        self,
        slot: _WorkerSlot,
        fn: Callable[[int], Any],
        entry_events: tuple[torch.cuda.Event, ...],
    ) -> tuple[Any, BaseException | None]:
        out: Any = None
        exc_caught: BaseException | None = None
        try:
            if slot.compute_stream is not None:
                _wait_stream_events(slot.compute_stream, entry_events)
                with torch.cuda.device(slot.device), torch.cuda.stream(slot.compute_stream):
                    out = tree_map_only((AsyncCollectiveTensor, DTensor), _materialize_async, fn(slot.rank))
            else:
                out = tree_map_only((AsyncCollectiveTensor, DTensor), _materialize_async, fn(slot.rank))
        except BaseException as exc:
            exc_caught = exc
        return out, exc_caught

    def _handle_worker_result(
        self,
        slot: _WorkerSlot,
        batch: _PublishedBatch,
        out: Any,  # noqa: ANN401
        exc_caught: BaseException | None,
    ) -> None:
        if exc_caught is not None:
            now = time.monotonic()
            summary = _summarize_error(exc_caught)
            with self._cv:
                if batch.first_error is None:
                    batch.first_error = exc_caught
                if batch.failure_time is None:
                    batch.failure_time = now
                self._poison_locked(summary)
            with contextlib.suppress(BaseException):
                self._drain_rank_stream(slot)
            with self._cv:
                batch.finished_ranks.add(slot.rank)
                self._cv.notify_all()
            return

        try:
            self._drain_rank_stream(slot)
        except BaseException as sync_exc:
            now = time.monotonic()
            summary = _summarize_error(sync_exc)
            with self._cv:
                if batch.first_error is None:
                    batch.first_error = sync_exc
                if batch.failure_time is None:
                    batch.failure_time = now
                self._poison_locked(summary)
                batch.finished_ranks.add(slot.rank)
                self._cv.notify_all()
            return
        with self._cv:
            if batch.first_error is None and not batch.discarded:
                batch.results[slot.rank] = out
            batch.finished_ranks.add(slot.rank)
            self._cv.notify_all()

    def _worker_loop(self, slot: _WorkerSlot) -> None:
        while True:
            batch: _PublishedBatch | None = None
            fn: Callable[[int], Any] | None = None
            entry_events: tuple[torch.cuda.Event, ...] = ()
            skip_poisoned = False
            with self._cv:
                while self._run_id == slot.last_run_id and not self._workers_stop:
                    self._cv.wait()
                if self._workers_stop:
                    return
                batch = self._current_batch
                slot.last_run_id = self._run_id
                if batch is None:
                    continue
                skip_poisoned = (
                    self._state in (_State.POISONED, _State.CLOSING, _State.CLOSED)
                    or batch.fn is None
                )
                if not skip_poisoned:
                    fn = batch.fn
                    entry_events = batch.entry_events
            if skip_poisoned:
                with self._cv:
                    batch.finished_ranks.add(slot.rank)
                    self._cv.notify_all()
                batch = None
                continue

            assert fn is not None
            try:
                out, exc_caught = self._execute_worker_task(slot, fn, entry_events)
                fn = None
                entry_events = ()
                self._handle_worker_result(slot, batch, out, exc_caught)
                out = None
                exc_caught = None
            except BaseException as fatal:
                now = time.monotonic()
                summary = _summarize_error(fatal)
                with self._cv:
                    if batch.first_error is None:
                        batch.first_error = fatal
                    if batch.failure_time is None:
                        batch.failure_time = now
                    self._poison_locked(summary)
                    batch.finished_ranks.add(slot.rank)
                    self._cv.notify_all()
            finally:
                fn = None
                entry_events = ()
                batch = None

    def _create_group(self, options: _DistributedBackendOptions) -> _ParallelProcessGroup:
        global_ranks = (
            tuple(int(r) for r in options.global_ranks_in_group)
            if options.global_ranks_in_group
            else tuple(range(self.world_size))
        )
        if len(global_ranks) != options.group_size:
            raise RuntimeError("piper_parallel subgroup global_ranks_in_group length mismatch")
        if any(r < 0 or r >= self.world_size for r in global_ranks) or len(set(global_ranks)) != len(global_ranks):
            raise RuntimeError(f"piper_parallel invalid subgroup global ranks: {global_ranks}")
        if options.group_rank < 0 or options.group_rank >= options.group_size:
            raise RuntimeError(f"piper_parallel invalid group_rank {options.group_rank}")

        opt_timeout = (
            min(options.timeout, self._collective_timeout)
            if options.timeout > timedelta(0)
            else self._collective_timeout
        )
        with self._cv:
            if self._transport is None or self._state not in (_State.STARTING, _State.RUNNING):
                raise RuntimeError("ParallelExecutor is not open or was poisoned")
            global_rank = global_ranks[options.group_rank]
            if (
                global_rank >= len(self._workers)
                or threading.current_thread() is not self._workers[global_rank].thread
            ):
                raise RuntimeError(
                    f"piper_parallel rank {global_rank} group creation must run on its assigned rank worker thread",
                )
            channel = self._channels.get(options.group_id)
            if channel is None:
                channel = _GroupChannel(self, options.group_id, global_ranks, opt_timeout)
                self._channels[options.group_id] = channel
            elif channel.global_ranks != global_ranks:
                raise RuntimeError(
                    f"piper_parallel group {options.group_id!r} rank mapping mismatch: "
                    f"{channel.global_ranks} != {global_ranks}",
                )
            channel.register_locked(options.group_rank)
        dev = self._devices[global_rank]
        return _ParallelProcessGroup(
            self,
            options.group_rank,
            options.group_size,
            channel,
            options.group_id,
            dev,
            opt_timeout,
        )

    def _earliest_slot_deadline_locked(self) -> float | None:
        deadlines = [
            d for ch in self._channels.values() if (d := ch.earliest_deadline_locked()) is not None
        ]
        return min(deadlines) if deadlines else None

    def _await_batch_locked(self, batch: _PublishedBatch) -> tuple[Any, ...]:
        timeout_secs = self._collective_timeout.total_seconds()
        while len(batch.finished_ranks) < self.world_size:
            slot_deadline = self._earliest_slot_deadline_locked() if self._state is _State.RUNNING else None
            fail_deadline = (batch.failure_time + timeout_secs) if batch.failure_time is not None else None
            if slot_deadline is None and fail_deadline is None:
                self._cv.wait()
                continue
            deadline = (
                min(slot_deadline, fail_deadline)
                if slot_deadline is not None and fail_deadline is not None
                else (slot_deadline if slot_deadline is not None else fail_deadline)
            )
            assert deadline is not None
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                now = time.monotonic()
                if slot_deadline is not None and now >= slot_deadline:
                    timeout_err = TimeoutError("piper_parallel operation timed out")
                    self._poison_locked(_summarize_error(timeout_err))
                    if batch.first_error is None:
                        batch.first_error = timeout_err
                    continue
                batch.discarded = True
                batch.results.clear()
                err = batch.first_error or (
                    self._poison_error.build()
                    if self._poison_error is not None
                    else TimeoutError("piper_parallel operation timed out")
                )
                self._poison_locked(_summarize_error(err))
                if isinstance(err, TimeoutError):
                    raise err
                raise TimeoutError("ParallelExecutor worker thread did not finish after failure") from err
            self._cv.wait(timeout=min(remaining, _MAX_WAIT_SECONDS))

        if batch.first_error is not None:
            self._poison_locked(_summarize_error(batch.first_error))
            batch.results.clear()
            raise batch.first_error
        if any(ch.has_unfinished_locked() for ch in self._channels.values()):
            err = RuntimeError("piper_parallel unfinished operation aborted at end of run")
            self._poison_locked(_summarize_error(err))
            batch.results.clear()
            raise err
        if self._state is _State.POISONED or any(slot.stream_needs_drain for slot in self._workers):
            summary = self._poison_error or _ErrorSummary(RuntimeError, "ParallelExecutor failed during run")
            self._poison_locked(summary)
            batch.results.clear()
            raise summary.build()

        self._state = _State.OPEN
        results = tuple(batch.results)
        batch.results.clear()
        return results

    def _wait_batch_drain_locked(self, batch: _PublishedBatch, deadline: float) -> None:
        while len(batch.finished_ranks) < self.world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                batch.discarded = True
                batch.results.clear()
                return
            self._cv.wait(timeout=min(remaining, _MAX_WAIT_SECONDS))
        batch.discarded = True
        batch.results.clear()

    def _run_batch(self, batch: _PublishedBatch) -> tuple[Any, ...]:
        published = False
        drain_attempted = False
        try:
            with self._cv:
                if self._state is not _State.OPEN:
                    raise RuntimeError("ParallelExecutor must be open and not closing to run callbacks")
                published = True
                self._state = _State.RUNNING
                self._current_batch = batch
                self._run_id = batch.run_id
                self._cv.notify_all()
                try:
                    return self._await_batch_locked(batch)
                except TimeoutError:
                    drain_attempted = True
                    raise
        except BaseException as exc:
            if published:
                with self._cv:
                    self._poison_locked(_summarize_error(exc))
                    if self._run_id != batch.run_id:
                        batch.finished_ranks.update(range(self.world_size))
                    elif not drain_attempted:
                        deadline = time.monotonic() + self._collective_timeout.total_seconds()
                        self._wait_batch_drain_locked(batch, deadline)
            raise
        finally:
            if published:
                with self._cv:
                    self._current_batch = None
                    if self._state is _State.RUNNING:
                        self._poison_locked(_ErrorSummary(RuntimeError, "ParallelExecutor run aborted unexpectedly"))
                    batch.clear_inputs()
                    self._cv.notify_all()
            else:
                batch.clear_inputs()

    def run[T](self, callback: Callable[[int], T]) -> tuple[T, ...]:
        """Execute ``callback(rank)`` concurrently across all ranks and return their results."""
        if not callable(callback):
            raise TypeError("ParallelExecutor.run callback must be callable")
        if self._is_internal_thread() or not self._controller_lock.acquire(blocking=False):
            raise RuntimeError("ParallelExecutor.run cannot overlap or nest")
        try:
            with self._cv:
                if self._state is _State.POISONED:
                    reason = self._poison_error.message if self._poison_error is not None else "unknown error"
                    raise RuntimeError(
                        f"ParallelExecutor was poisoned by a prior failure ({reason}); close and recreate it",
                    )
                if self._state is not _State.OPEN:
                    raise RuntimeError("ParallelExecutor must be open and not closing to run callbacks")

            entry_events = tuple(_capture_entry_events(self._devices).values())
            batch = _PublishedBatch(
                run_id=self._next_run_id,
                fn=callback,
                entry_events=entry_events,
                results=[None] * self.world_size,
            )
            self._next_run_id += 1
            return self._run_batch(batch)
        finally:
            self._controller_lock.release()

    def _drain_worker_streams(self, deadline: float) -> bool:
        for slot in self._workers:
            stream = slot.compute_stream
            if stream is None:
                continue
            with torch.cuda.device(slot.device):
                while not stream.query():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return False
                    time.sleep(min(0.002, remaining))
            slot.stream_needs_drain = False
        return True

    def _close_locked(self) -> None:
        with self._cv:
            if (
                self._state is _State.CLOSED
                and not owns_session(self)
                and self._runtime is None
                and self._startup_gate is None
            ):
                return
            self._state = _State.CLOSING
            self._poison_locked(_ErrorSummary(RuntimeError, "ParallelExecutor closed"))
            self._workers_stop = True
            self._progress_stop = True
            gate = self._startup_gate
            if gate is not None:
                gate.cancel()
            self._cv.notify_all()
            workers = [slot.thread for slot in self._workers if slot.thread is not None]
            progress = self._progress_thread

        deadline = time.monotonic() + self._collective_timeout.total_seconds()
        if not _join_threads_until(workers, deadline) or (
            gate is not None and not gate.wait_exited(deadline)
        ):
            raise TimeoutError(
                "ParallelExecutor worker threads did not exit before collective_timeout; isolation is still active",
            )
        if progress is not None and not _join_threads_until((progress,), deadline):
            raise TimeoutError(
                "ParallelExecutor progress thread did not exit before collective_timeout; isolation is still active",
            )
        if not self._drain_worker_streams(deadline):
            raise TimeoutError(
                "ParallelExecutor worker CUDA streams did not drain before collective_timeout; "
                "isolation is still active",
            )
        if self._transport is not None and not self._transport.drain_unreclaimed(deadline):
            raise TimeoutError(
                "ParallelExecutor GPU transfers did not drain before collective_timeout; isolation is still active",
            )

        with self._cv:
            self._channels.clear()
            self._ready_queue.clear()
            self._workers.clear()
            self._progress_thread = None
            self._startup_gate = None

        if self._transport is not None:
            self._transport.close()
            self._transport = None

        if self._runtime is not None:
            self._runtime.restore()
            self._runtime = None

        with self._cv:
            self._poison_error = None
            self._current_batch = None
            self._workers_stop = False
            self._progress_stop = False
            self._run_id = 0
            self._next_run_id = 1
            self._state = _State.CLOSED
        if _state.active_executor is self:
            _state.active_executor = None
        release_session(self)

    def close(self) -> None:
        """Stop workers and progress thread, drain transfers, and restore global runtime."""
        if self._is_internal_thread():
            raise RuntimeError("close ParallelExecutor outside its rank callbacks")
        if not self._controller_lock.acquire(blocking=False):
            raise RuntimeError("cannot close ParallelExecutor while run is active")
        try:
            self._close_locked()
        finally:
            self._controller_lock.release()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:  # noqa: ANN401
        self.close()
