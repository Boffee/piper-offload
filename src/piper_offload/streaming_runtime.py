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

logger = logging.getLogger(__name__)

type BlockSignature = tuple[object, ...]
type _LoadedTrainableBlock = tuple[PinnedModuleInstance, PinnedModuleTarget]


class StreamingRuntime(Protocol):
    """Lifecycle shared by CUDA streaming strategies.

    Implementations allocate accelerator resources only during
    :meth:`activate`. Activation may fail after partially initializing a
    runtime, so :meth:`deactivate` must also be safe for inactive and partial
    states and must release every resource it can before propagating a cleanup
    error.
    """

    @property
    def active(self) -> bool: ...

    @property
    def compile_backend(self) -> CompileBackend:
        """The ``torch.compile`` backend required by this strategy."""
        ...

    def activate(self, device: torch.device) -> None:
        """Allocate resources and install hooks for one CUDA activation."""
        ...

    def deactivate(self) -> None:
        """Idempotently release resources from any activation state."""
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
    """Signature-keyed pool of reusable GPU block targets."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._free: dict[BlockSignature, list[PinnedModuleTarget]] = {}
        self._events: dict[int, torch.cuda.Event] = {}

    def acquire(
        self,
        signature: BlockSignature,
        instance: PinnedModuleInstance,
    ) -> PinnedModuleTarget:
        free = self._free.get(signature)
        if free:
            return free.pop()
        return instance.allocate_target(self._device)

    def release(
        self,
        signature: BlockSignature,
        target: PinnedModuleTarget,
    ) -> None:
        self._free.setdefault(signature, []).append(target)

    def set_compute_event(
        self,
        target: PinnedModuleTarget,
        event: torch.cuda.Event,
    ) -> None:
        self._events[id(target)] = event

    def wait_if_needed(
        self,
        target: PinnedModuleTarget,
        stream: torch.cuda.Stream | None,
    ) -> None:
        event = self._events.pop(id(target), None)
        if event is not None and stream is not None and not event.query():
            event.wait(stream)


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
        self._block_to_target: dict[int, PinnedModuleTarget] = {}
        self._active_idx: int | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._executor: ThreadPoolExecutor | None = None
        self._stream: torch.cuda.Stream | None = None
        self._pending: dict[int, Future[None]] = {}
        self._prefetch_events: dict[int, torch.cuda.Event] = {}
        self._last_idx = -1
        self._optimizer_step_active = False
        self._move_trainable_grads_to(torch.device("cpu"))

    @property
    def active(self) -> bool:
        return self._device is not None

    @property
    def compile_backend(self) -> CompileBackend:
        return "inductor"

    def activate(self, device: torch.device) -> None:
        if self.active:
            raise RuntimeError("block streaming runtime is already active")

        num_blocks = len(self._instances)
        self._device = device
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._stream = torch.cuda.Stream(device=device, priority=-1)
        self._pending = {}
        self._prefetch_events = {i: torch.cuda.Event() for i in range(num_blocks)}
        self._last_idx = -1

        self._pool = _MorphingTargetPool(device)
        self._move_trainable_grads_to(device)

        self._load_block(0)
        self._active_idx = 0

        self._register_hooks()

        logger.info(f"{self._log_label} active: one block on GPU plus one lookahead target across {num_blocks} blocks")

    def deactivate(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        first_prefetch_exc = self._drain_and_evict_all()
        self._prefetch_events.clear()
        self._stream = None
        self._move_trainable_grads_to(torch.device("cpu"))
        self._pool = None
        self._block_to_target.clear()
        self._active_idx = None
        self._last_idx = -1
        self._device = None

        if first_prefetch_exc is not None:
            raise first_prefetch_exc

    @contextlib.contextmanager
    def optimizer_step(self) -> Iterator[None]:
        if not self.active:
            raise RuntimeError(
                "StreamedComponent.optimizer_step() called on inactive "
                "streamer. Use it inside the offloader's context "
                "manager, between backward and the next forward."
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
        assert step_stream is not None, "stream allocated in activate()"

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
            raise RuntimeError("block streaming runtime is inactive")
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

    def _load_block(
        self,
        block_idx: int,
        *,
        non_blocking: bool = False,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        assert self._pool is not None, "runtime is not active"
        instance = self._instances[block_idx]
        target = self._block_to_target.get(block_idx)
        if target is None:
            target = self._pool.acquire(self._signatures[block_idx], instance)
            self._block_to_target[block_idx] = target
        self._pool.wait_if_needed(target, stream)
        instance.load_to_target(
            target,
            run_post_copy_hooks=True,
            non_blocking=non_blocking,
        )

    def _release_block(self, block_idx: int) -> None:
        self._instances[block_idx].install_pinned()
        assert self._pool is not None, "runtime is not active"
        target = self._block_to_target.pop(block_idx, None)
        if target is not None:
            self._pool.release(self._signatures[block_idx], target)

    def _drain_and_evict_all(self) -> BaseException | None:
        first_prefetch_exc: BaseException | None = None
        for future in list(self._pending.values()):
            try:
                future.result()
            except BaseException as exc:
                if first_prefetch_exc is None:
                    first_prefetch_exc = exc
        self._pending.clear()

        if self._stream is not None:
            try:
                self._stream.synchronize()
            except BaseException as exc:
                if first_prefetch_exc is None:
                    first_prefetch_exc = exc

        for block_idx in list(self._block_to_target):
            self._release_block(block_idx)
        self._active_idx = None
        return first_prefetch_exc

    def _evict_active(self, compute_event: torch.cuda.Event | None = None) -> None:
        victim = self._active_idx
        if victim is None:
            return
        if compute_event is not None:
            assert self._pool is not None, "runtime is not active"
            target = self._block_to_target.get(victim)
            if target is not None:
                self._pool.set_compute_event(target, compute_event)
        self._release_block(victim)
        self._active_idx = None

    def _do_prefetch(self, idx: int) -> None:
        assert self._stream is not None
        with torch.cuda.stream(self._stream):
            self._load_block(idx, non_blocking=True, stream=self._stream)
            self._prefetch_events[idx].record(self._stream)

    def _submit_prefetch(self, idx: int) -> None:
        assert self._executor is not None
        if idx == self._active_idx or idx in self._pending:
            return
        if self._pending:
            return
        self._pending[idx] = self._executor.submit(self._do_prefetch, idx)

    def _ensure_on_gpu(self, idx: int) -> None:
        future = self._pending.pop(idx, None)
        if future is not None:
            future.result()
            event = self._prefetch_events[idx]
            if not event.query():
                event.wait(torch.cuda.current_stream(self._require_device()))
            self._active_idx = idx
            return
        self._load_block(idx)
        self._active_idx = idx

    def _before_block_forward(
        self,
        idx: int,
    ) -> None:
        if not self.active:
            return

        if idx != self._active_idx:
            compute_event = torch.cuda.Event()
            compute_event.record(torch.cuda.current_stream(self._require_device()))
            self._evict_active(compute_event)
            self._ensure_on_gpu(idx)

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
