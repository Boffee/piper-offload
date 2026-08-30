"""CUDA execution strategies for streamed block components."""

import contextlib
import functools
import logging
import weakref
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol

import torch
from torch import nn

from .block_compile import CompileBackend
from .module_names import group_names
from .pinned_module import PinnedModuleInstance, PinnedModuleTarget
from .target_lease import _CudaTargetLease

logger = logging.getLogger(__name__)

type BlockSignature = tuple[object, ...]
type _LoadedTrainableBlock = tuple[PinnedModuleInstance, PinnedModuleTarget]


class StreamingRuntime(Protocol):
    """CUDA working-set lifecycle shared by streaming strategies.

    Implementations allocate accelerator resources only during
    :meth:`acquire`. Acquisition may fail after partially initializing a
    runtime, so :meth:`release` must also be safe for released and partial
    states and must release every resource it can before propagating a cleanup
    error.
    """

    @property
    def acquired(self) -> bool: ...

    @property
    def compile_backend(self) -> CompileBackend:
        """The ``torch.compile`` backend required by this strategy."""
        ...

    def acquire(self, device: torch.device) -> None:
        """Allocate the CUDA working set and install execution hooks."""
        ...

    def release(self) -> None:
        """Idempotently release the CUDA working set from any state."""
        ...

    def optimizer_step(self) -> contextlib.AbstractContextManager[None]: ...


def _instance_target_signature(instance: PinnedModuleInstance) -> BlockSignature:
    """Return the layout-equivalence key for one block's GPU targets."""
    params = instance.params
    param_sig = tuple(
        (
            tuple(names),
            params[names[0]].requires_grad,
            params[names[0]].target_layout,
        )
        for names in group_names(params.keys(), lambda name: id(params[name]))
    )
    buffers = instance.buffers
    buffer_sig = tuple(
        (tuple(names), buffers[names[0]].target_layout)
        for names in group_names(buffers.keys(), lambda name: id(buffers[name]))
    )
    return (param_sig, buffer_sig)


class _MorphingTargetPool:
    """Signature-keyed pool of reusable stream-aware GPU targets."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._free: dict[BlockSignature, list[_CudaTargetLease]] = {}

    def acquire(
        self,
        signature: BlockSignature,
        instance: PinnedModuleInstance,
        stream: torch.cuda.Stream,
    ) -> _CudaTargetLease:
        free = self._free.get(signature)
        if free:
            return free.pop()
        return _CudaTargetLease.allocate(
            instance,
            self._device,
            allocation_stream=stream,
        )

    def release(
        self,
        signature: BlockSignature,
        lease: _CudaTargetLease,
    ) -> None:
        self._free.setdefault(signature, []).append(lease)

    def close(self) -> None:
        for leases in self._free.values():
            for lease in leases:
                lease.close()
        self._free.clear()


class BlockStreamingRuntime:
    """One active block plus one whole-block lookahead target."""

    def __init__(
        self,
        instances: Sequence[PinnedModuleInstance],
        *,
        log_label: str,
    ) -> None:
        self._instances = tuple(instances)
        self._blocks = tuple(instance.module for instance in instances)
        self._signatures = tuple(_instance_target_signature(instance) for instance in instances)
        self._log_label = log_label
        self._device: torch.device | None = None
        self._pool: _MorphingTargetPool | None = None
        self._block_to_lease: dict[int, _CudaTargetLease] = {}
        self._active_idx: int | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._executor: ThreadPoolExecutor | None = None
        self._stream: torch.cuda.Stream | None = None
        self._pending: dict[int, Future[None]] = {}
        self._last_idx = -1
        self._optimizer_step_active = False
        self._move_trainable_grads_to(torch.device("cpu"))

    @property
    def acquired(self) -> bool:
        return self._device is not None

    @property
    def compile_backend(self) -> CompileBackend:
        return "inductor"

    def acquire(self, device: torch.device) -> None:
        if self.acquired:
            raise RuntimeError("block streaming runtime is already acquired")

        num_blocks = len(self._instances)
        self._device = device
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._stream = torch.cuda.Stream(device=device, priority=-1)
        self._pending = {}
        self._last_idx = -1

        self._pool = _MorphingTargetPool(device)
        self._move_trainable_grads_to(device)

        current_stream = torch.cuda.current_stream(device)
        self._stage_block(0, current_stream)
        self._acquire_block(0, current_stream)
        self._active_idx = 0

        self._register_hooks()

        logger.info(
            f"{self._log_label} acquired: one block on GPU plus one "
            f"lookahead target across {num_blocks} blocks"
        )

    def release(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        first_prefetch_exc = self._drain_and_evict_all()
        self._stream = None
        self._move_trainable_grads_to(torch.device("cpu"))
        if self._pool is not None:
            self._pool.close()
        self._pool = None
        self._block_to_lease.clear()
        self._active_idx = None
        self._last_idx = -1
        self._device = None

        if first_prefetch_exc is not None:
            raise first_prefetch_exc

    @contextlib.contextmanager
    def optimizer_step(self) -> Iterator[None]:
        if not self.acquired:
            raise RuntimeError(
                "StreamedComponent.optimizer_step() called while its CUDA "
                "working set is released. Acquire the component before "
                "entering the optimizer step."
            )
        if self._optimizer_step_active:
            raise RuntimeError(
                "StreamedComponent.optimizer_step() does not support "
                "reentrant entry. A nested optimizer-step boundary would "
                "scatter the outer step's stale pinned bytes on top of "
                "the inner update."
            )
        if not any(instance.has_trainables for instance in self._instances):
            yield
            return

        first_prefetch_exc = self._drain_and_evict_all()
        if first_prefetch_exc is not None:
            raise first_prefetch_exc

        device = self._require_device()
        step_stream = self._stream
        assert step_stream is not None, "stream allocated in acquire()"

        self._optimizer_step_active = True
        try:
            with contextlib.ExitStack() as stack:
                loaded = self._load_trainables_for_step(device, step_stream, stack)
                try:
                    yield
                finally:
                    self._scatter_trainables_after_step(loaded, step_stream, device)
        finally:
            self._optimizer_step_active = False

    def _require_device(self) -> torch.device:
        device = self._device
        if device is None:
            raise RuntimeError("block streaming runtime is released")
        return device

    def _load_trainables_for_step(
        self,
        device: torch.device,
        step_stream: torch.cuda.Stream,
        stack: contextlib.ExitStack,
    ) -> list[_LoadedTrainableBlock]:
        loaded: list[_LoadedTrainableBlock] = []
        with torch.cuda.stream(step_stream):
            for instance in self._instances:
                if not instance.has_trainables:
                    continue
                stack.callback(instance.install_pinned)
                target = instance.allocate_target(
                    device,
                    param_names=instance.trainable_param_names,
                    buffer_names=(),
                )
                instance.load_to_target(target, non_blocking=True)
                instance.move_trainable_grads_to(device)
                loaded.append((instance, target))
        torch.cuda.current_stream(device).wait_stream(step_stream)
        return loaded

    def _scatter_trainables_after_step(
        self,
        loaded: list[_LoadedTrainableBlock],
        step_stream: torch.cuda.Stream,
        device: torch.device,
    ) -> None:
        step_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(step_stream):
            for instance, target in loaded:
                instance.copy_trainables_from_target(target, non_blocking=False)
        step_stream.synchronize()

    def _move_trainable_grads_to(self, device: torch.device) -> None:
        for instance in self._instances:
            instance.move_trainable_grads_to(device)

    def _stage_block(
        self,
        block_idx: int,
        stream: torch.cuda.Stream,
        *,
        non_blocking: bool = False,
    ) -> None:
        assert self._pool is not None, "runtime is not acquired"
        instance = self._instances[block_idx]
        lease = self._block_to_lease.get(block_idx)
        if lease is None:
            lease = self._pool.acquire(
                self._signatures[block_idx],
                instance,
                stream,
            )
            self._block_to_lease[block_idx] = lease
        lease.stage(
            instance,
            stream,
            run_post_copy_hooks=True,
            non_blocking=non_blocking,
        )

    def _acquire_block(
        self,
        block_idx: int,
        stream: torch.cuda.Stream,
    ) -> None:
        lease = self._block_to_lease[block_idx]
        self._instances[block_idx].install_target(lease.acquire(stream))

    def _release_block(
        self,
        block_idx: int,
    ) -> None:
        self._instances[block_idx].install_pinned()
        assert self._pool is not None, "runtime is not acquired"
        lease = self._block_to_lease.pop(block_idx, None)
        if lease is not None:
            lease.release()
            self._pool.release(self._signatures[block_idx], lease)

    def _drain_and_evict_all(self) -> BaseException | None:
        first_prefetch_exc: BaseException | None = None
        for future in list(self._pending.values()):
            try:
                future.result()
            except BaseException as exc:
                if first_prefetch_exc is None:
                    first_prefetch_exc = exc
        self._pending.clear()

        for block_idx in list(self._block_to_lease):
            self._release_block(block_idx)
        self._active_idx = None
        return first_prefetch_exc

    def _submit_prefetch(self, idx: int) -> None:
        assert self._executor is not None
        assert self._stream is not None
        if idx == self._active_idx or self._pending:
            return
        self._pending[idx] = self._executor.submit(
            self._stage_block,
            idx,
            self._stream,
            non_blocking=True,
        )

    def _ensure_on_gpu(self, idx: int) -> None:
        current_stream = torch.cuda.current_stream(self._require_device())
        future = self._pending.pop(idx, None)
        if future is not None:
            future.result()
        else:
            self._stage_block(idx, current_stream)
        self._acquire_block(idx, current_stream)
        self._active_idx = idx

    def _before_block_forward(
        self,
        idx: int,
    ) -> None:
        if not self.acquired:
            return

        current_stream = torch.cuda.current_stream(self._require_device())
        if idx != self._active_idx:
            if self._active_idx is not None:
                self._release_block(self._active_idx)
                self._active_idx = None
            self._ensure_on_gpu(idx)
        self._block_to_lease[idx].record_stream(current_stream)

        last = self._last_idx
        self._last_idx = idx
        num_blocks = len(self._instances)
        if last < 0:
            direction = 1
        else:
            diff = idx - last
            direction = (-1 if diff > 0 else 1) if abs(diff) > num_blocks // 2 else 1 if diff >= 0 else -1
        self._submit_prefetch((idx + direction) % num_blocks)

    def _register_hooks(self) -> None:
        runtime_ref = weakref.ref(self)

        def _pre_hook(_module: nn.Module, _args: tuple[object, ...], *, idx: int) -> None:
            runtime = runtime_ref()
            if runtime is not None:
                runtime._before_block_forward(idx)

        last_idx_by_module = {id(module): idx for idx, module in enumerate(self._blocks)}
        for idx in last_idx_by_module.values():
            handle = self._blocks[idx].register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            self._hooks.append(handle)


__all__ = ["BlockStreamingRuntime", "StreamingRuntime"]
