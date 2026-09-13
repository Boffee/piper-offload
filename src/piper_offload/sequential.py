"""Experimental two-rank DTensor inference in one process on one CUDA/HIP GPU.

Requires PyTorch 2.14 and the ``triton`` extra; CUDA graphs are unsupported.
"""

import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, Future, wait
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import timedelta
from types import TracebackType
from typing import Self, cast

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    AllgatherOptions,
    _create_work_from_future,
    _DistributedBackendOptions,
)
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.distributed_c10d import GroupName
from torch.utils._pytree import tree_map_only

from ._sequential_collectives import LocalCollective, execute

_BACKEND = "piper_sequential"
_session_lock = threading.Lock()
_active_executor: SequentialExecutor | None = None


@dataclass(slots=True)
class _Round:
    group: GroupName
    calls: dict[int, LocalCollective] = field(default_factory=dict)
    complete: bool = False


class _Coordinator:
    def __init__(self, device: torch.device, stream: torch.cuda.Stream, timeout: float) -> None:
        self.device, self.stream, self.timeout = device, stream, timeout
        self.condition = threading.Condition()
        self.sequential = False
        self.turn = 0
        self.finished = False
        self.error: BaseException | None = None
        self.pending: _Round | None = None

    def begin(self, sequential: bool) -> None:
        with self.condition:
            self._check_error()
            self.sequential, self.turn = sequential, 0
            self.finished = False

    def _check_error(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"sequential execution aborted: {self.error}") from self.error

    def _wait(self, ready: Callable[[], bool]) -> None:
        if not self.condition.wait_for(lambda: self.error is not None or ready(), timeout=self.timeout):
            raise TimeoutError("sequential rank timed out waiting for its peer")
        self._check_error()

    def enter(self, rank: int) -> None:
        with self.condition:
            self._wait(lambda: not self.sequential or self.turn == rank or self.finished)

    def finish(self) -> None:
        with self.condition:
            self._check_error()
            if self.pending is not None:
                raise RuntimeError("sequential ranks reached different numbers of collectives")
            self.finished = True
            self.condition.notify_all()

    def abort(self, error: BaseException) -> None:
        with self.condition:
            if self.error is None:
                self.error = error
            self.condition.notify_all()

    def collective(self, rank: int, group: GroupName, call: LocalCollective) -> None:
        with self.condition:
            self._check_error()
            if self.finished:
                raise RuntimeError("sequential ranks reached different numbers of collectives")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("sequential collectives do not support CUDA graph capture")
            if torch.cuda.current_stream(self.device) != self.stream:
                raise RuntimeError("sequential collectives must use the executor's shared compute stream")
            if self.pending is None:
                self.pending = _Round(group)
            current = self.pending
            if current.group != group or rank in current.calls:
                raise RuntimeError("sequential ranks requested different process groups or collective order")
            current.calls[rank] = call
            self.turn = 1 - rank
            self.condition.notify_all()
            if len(current.calls) == 2:
                execute((current.calls[0], current.calls[1]), self.device)
                current.calls.clear()
                current.complete = True
                self.pending = None
                self.condition.notify_all()
            self._wait(lambda: current.complete)
            self._wait(lambda: not self.sequential or self.turn == rank or self.finished)


class _SequentialProcessGroup(dist.ProcessGroup):
    # Implement c10d's tensor-list/options virtual methods. Its Python
    # convenience overloads (tensor/root/timeout) are not backend entry points.
    def __init__(self, rank: int, group_id: GroupName, coordinator: _Coordinator) -> None:
        super().__init__(rank, 2)  # type: ignore[call-arg]
        self._group_id = group_id
        self._coordinator = coordinator

    def getBackendName(self) -> str:  # noqa: N802 -- PyTorch trampoline name
        return _BACKEND

    @property
    def group_name(self) -> GroupName:
        # The native getter reads a registered C++ device backend. This Python
        # process group owns its local operations without such a backend.
        return self._group_id

    def _call(self, call: LocalCollective, *, nested_outputs: bool = False) -> dist.Work:
        try:
            self._coordinator.collective(self.rank(), self._group_id, call)
            # Record this rank's result on the shared compute stream. Work.wait
            # orders consumers without synchronizing the device at each handoff.
            future = torch.futures.Future(devices=[self._coordinator.device])
            outputs = list(call.outputs)
            future.set_result([outputs[i : i + 2] for i in range(0, len(outputs), 2)] if nested_outputs else outputs)
            return _create_work_from_future(future)
        except BaseException as error:
            self._coordinator.abort(error)
            raise

    def allreduce(  # type: ignore[override]
        self,
        tensors: list[torch.Tensor],
        opts: dist.AllreduceOptions | dist.AllreduceCoalescedOptions | None = None,
    ) -> dist.Work:
        if opts is not None and opts.reduceOp != dist.ReduceOp.SUM:
            raise NotImplementedError("sequential execution supports only SUM all-reduce")
        return self._call(LocalCollective("sum", outputs=tuple(tensors)))

    allreduce_coalesced = allreduce

    def broadcast(  # type: ignore[override]
        self,
        tensors: list[torch.Tensor],
        opts: dist.BroadcastOptions | None = None,
    ) -> dist.Work:
        options = dist.BroadcastOptions() if opts is None else opts
        if options.rootTensor != 0:
            raise ValueError("sequential broadcast requires rootTensor=0")
        return self._call(LocalCollective("broadcast", outputs=tuple(tensors), root=options.rootRank))

    def scatter(  # type: ignore[override]
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: dist.ScatterOptions | None = None,
    ) -> dist.Work:
        options = dist.ScatterOptions() if opts is None else opts
        if len(input_tensors) > 1:
            raise ValueError("sequential scatter requires one local tensor")
        return self._call(
            LocalCollective(
                "scatter",
                tuple(input_tensors[0]) if input_tensors else (),
                tuple(output_tensors),
                options.rootRank,
            )
        )

    def allgather(  # type: ignore[override]
        self,
        output_tensors: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: AllgatherOptions | None = None,
    ) -> dist.Work:
        del opts
        if len(output_tensors) != len(input_tensors) or any(len(group) != 2 for group in output_tensors):
            raise ValueError("sequential all-gather requires two outputs per input")
        return self._call(
            LocalCollective(
                "allgather",
                tuple(input_tensors),
                tuple(t for group in output_tensors for t in group),
            ),
            nested_outputs=True,
        )

    def all_gather_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,  # noqa: A002 -- PyTorch parameter name
        opts: AllgatherOptions | None = None,
    ) -> dist.Work:
        del opts
        return self.all_gather_single_coalesced([output], [input])

    def all_gather_single_coalesced(
        self,
        output_lists: list[torch.Tensor],
        input_list: list[torch.Tensor],
        opts: AllgatherOptions | None = None,
    ) -> dist.Work:
        del opts
        if len(output_lists) != len(input_list):
            raise ValueError("sequential all-gather requires matching input/output lists")
        slices = []
        for output, tensor in zip(output_lists, input_list, strict=True):
            if not output.is_contiguous() or output.numel() != 2 * tensor.numel():
                raise ValueError("sequential all-gather output must hold two contiguous input slices")
            slices.extend(output.view(2, *tensor.shape).unbind(0))
        return self._call(LocalCollective("allgather", tuple(input_list), tuple(slices)))

    def barrier(self, opts: dist.BarrierOptions | None = None) -> dist.Work:  # type: ignore[override]
        del opts
        return self._call(LocalCollective("barrier"))


def _create_group(backend: _DistributedBackendOptions, options: object) -> dist.ProcessGroup:
    del options
    executor = _active_executor
    if executor is None or executor._coordinator is None or backend.group_size != 2:
        raise RuntimeError("piper_sequential groups require an active two-rank SequentialExecutor")
    return _SequentialProcessGroup(backend.group_rank, backend.group_id, executor._coordinator)


@dataclass(slots=True)
class _Job:
    callback: Callable[[int], object]
    local_only: bool = False
    results: tuple[Future[object], Future[object]] = field(default_factory=lambda: (Future(), Future()))

    def wait(self, timeout: float, message: str) -> None:
        _, pending = wait(self.results, timeout=timeout)
        if pending:
            raise TimeoutError(message)


class SequentialExecutor:
    """Run ordinary rank callbacks in two persistent threads on one GPU.

    ``run(callback)`` returns one result per rank and schedules computation
    between matching collective calls. Use ``sequential=False`` for setup,
    compilation warmup, and cleanup that need unrestricted rank progress.
    Use a separate model/offloader per rank and close offloaders in callbacks.
    Keep DTensor operations inside callbacks; return local tensors for use outside.

    After a rank error or collective timeout, ``run(..., sequential=False)``
    remains available for local cleanup, after both prior callbacks have exited.
    Collectives stay aborted. Close joins workers before restoring PyTorch state;
    if a callback does not exit within ``timeout``, close raises and retains
    isolation until a later successful close. Other distributed activity must
    not run concurrently.
    """

    def __init__(self, device: torch.device | str = "cuda", *, timeout: timedelta = timedelta(minutes=5)) -> None:
        if timeout.total_seconds() <= 0:
            raise ValueError("sequential timeout must be positive")
        self.device = torch.device(device)
        self.timeout = timeout
        self._coordinator: _Coordinator | None = None
        self._threads: list[threading.Thread] = []
        self._queues: list[queue.SimpleQueue[_Job | None]] = []
        self._call_lock = threading.Lock()
        self._pending_job: _Job | None = None
        self._runtime: AbstractContextManager[None] | None = None
        self._closing = False

    def __enter__(self) -> Self:
        global _active_executor  # noqa: PLW0603 -- one scoped process-wide rank runtime
        if self._coordinator is not None or self._closing:
            raise RuntimeError("sequential executor cannot be reopened")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("SequentialExecutor requires one CUDA/HIP device")
        if not _session_lock.acquire(blocking=False):
            raise RuntimeError("another SequentialExecutor is already open")
        try:
            if dist.is_initialized() or dist.distributed_c10d._world.pg_map:
                raise RuntimeError("SequentialExecutor requires no existing distributed process groups")
            # Fail before changing process-wide state if the direct SUM kernel
            # dependency is unavailable.
            from . import _triton_sequential  # noqa: PLC0415, F401
            from ._sequential_runtime import isolate_runtime  # noqa: PLC0415

            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            self.device = torch.device("cuda", index)
            self._coordinator = _Coordinator(
                self.device,
                torch.cuda.default_stream(self.device),
                self.timeout.total_seconds(),
            )
            plugin = dist.Backend._plugins.get(_BACKEND.upper())
            if plugin is not None and plugin.creator_fn is not _create_group:
                raise RuntimeError("a different backend is already registered as piper_sequential")
            if plugin is None:
                dist.Backend.register_backend(_BACKEND, _create_group, extended_api=True, devices=["cpu", "cuda"])
            runtime = isolate_runtime()
            runtime.__enter__()
            self._runtime = runtime
            _active_executor = self
            store = dist.HashStore()
            ready: list[Future[None]] = [Future(), Future()]
            self._queues = [queue.SimpleQueue(), queue.SimpleQueue()]
            for rank in range(2):
                thread = threading.Thread(target=self._worker, args=(rank, store, ready[rank]), daemon=True)
                thread.start()
                self._threads.append(thread)
            for result in ready:
                result.result(timeout=self.timeout.total_seconds())
        except BaseException:
            if self._runtime is not None:
                self.close()
            else:
                _session_lock.release()
            raise
        return self

    def _worker(self, rank: int, store: dist.Store, ready: Future[None]) -> None:
        coordinator = self._coordinator
        assert coordinator is not None
        try:
            torch.cuda.set_device(self.device)
            with torch.no_grad(), torch.cuda.stream(coordinator.stream):
                dist.init_process_group(_BACKEND, store=store, rank=rank, world_size=2, timeout=self.timeout)
                ready.set_result(None)
                while (job := self._queues[rank].get()) is not None:
                    try:
                        if not job.local_only:
                            coordinator.enter(rank)
                        result = tree_map_only(AsyncCollectiveTensor, lambda tensor: tensor.wait(), job.callback(rank))
                        if not job.local_only:
                            coordinator.finish()
                        job.results[rank].set_result(result)
                        del result
                    except BaseException as error:
                        coordinator.abort(error)
                        job.results[rank].set_exception(error)
                    finally:
                        # Idle workers must not keep the previous forward's
                        # outputs (or its callback's captured model) alive.
                        del job
        except BaseException as error:
            coordinator.abort(error)
            if not ready.done():
                ready.set_exception(error)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def run[T](self, callback: Callable[[int], T], *, sequential: bool = True) -> tuple[T, T]:
        """Invoke both ranks; after an error, only local ``sequential=False`` cleanup is allowed."""
        if self._runtime is None or self._closing:
            raise RuntimeError("SequentialExecutor must be open and not closing to run callbacks")
        assert self._coordinator is not None
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("sequential executor callbacks cannot overlap or nest")
        try:
            local_only = self._coordinator.error is not None and not sequential
            if not local_only:
                self._coordinator.begin(sequential)
            elif self._pending_job is not None:
                # A failed run returns promptly while its peer may still be
                # unwinding. Cleanup must not race either rank's previous work.
                self._pending_job.wait(
                    self.timeout.total_seconds(),
                    "sequential callbacks have not exited; cleanup has not started",
                )
            self._coordinator.stream.wait_stream(torch.cuda.current_stream(self.device))
            job = self._pending_job = _Job(callback, local_only)
            for inbox in self._queues:
                inbox.put(job)
            if local_only:
                job.wait(self.timeout.total_seconds(), "sequential cleanup callbacks have not exited")
            else:
                wait(job.results, return_when=FIRST_EXCEPTION)
            for rank, result in enumerate(job.results):
                if result.done() and (error := result.exception()) is not None:
                    raise RuntimeError(f"sequential rank {rank} failed: {error}") from error
            self._coordinator.stream.synchronize()
            return cast(tuple[T, T], tuple(result.result() for result in job.results))
        except BaseException as error:
            self._coordinator.abort(error)
            raise
        finally:
            if self._pending_job is not None and all(result.done() for result in self._pending_job.results):
                self._pending_job = None
            self._call_lock.release()

    def close(self) -> None:
        """Join rank workers and then restore the process's distributed state."""
        global _active_executor  # noqa: PLW0603 -- restore the scoped process-wide rank runtime
        if self._runtime is None:
            return
        if threading.current_thread() in self._threads:
            raise RuntimeError("close SequentialExecutor outside its rank callbacks")
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("cannot close SequentialExecutor while run is active")
        try:
            assert self._coordinator is not None
            self._closing = True
            self._coordinator.abort(RuntimeError("sequential executor is closing"))
            for inbox in self._queues:
                inbox.put(None)
            deadline = time.monotonic() + self.timeout.total_seconds()
            for thread in self._threads:
                thread.join(max(0, deadline - time.monotonic()))
            if any(thread.is_alive() for thread in self._threads):
                raise TimeoutError("sequential callbacks have not exited; distributed isolation is still active")
            try:
                self._coordinator.stream.synchronize()
            finally:
                self._runtime.__exit__(None, None, None)
                self._runtime = None
                self._coordinator.pending = None
                self._coordinator.error = None
                self._coordinator = None
                self._pending_job = None
                _active_executor = None
                _session_lock.release()
        finally:
            self._call_lock.release()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["SequentialExecutor"]
