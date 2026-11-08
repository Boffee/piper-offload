"""CUDA execution strategy that streams whole block targets."""

import contextlib
import functools
import logging
import weakref
from collections.abc import Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass

import torch
from torch import nn

from .block_compile import CompileBackend
from .block_runtime import validate_load_plans
from .host_module import (
    HostModuleInstance,
    HostModuleLoadPlan,
    HostModuleTarget,
)
from .module_names import group_names
from .target_lease import CudaTargetLease

logger = logging.getLogger(__name__)

type BlockSignature = tuple[object, ...]
type _LoadedTrainableBlock = tuple[HostModuleInstance, HostModuleTarget]


def _wait_for_prefetch(future: Future[None]) -> None:
    """Wait for a prefetch; return or raise only once its worker has finished.

    The worker owns its block's target until the job ends. An interrupt such
    as Ctrl-C can end the wait first, so it is re-raised only after the
    worker finishes staging; otherwise the caller could hand the target to
    another block while the worker still writes into it.
    """
    try:
        future.result()
    except BaseException:
        while not future.done():
            try:
                wait((future,))
            except BaseException:
                # A repeated interrupt; the first one is re-raised below.
                continue
        raise


def _prefetch_direction(last_idx: int, idx: int, num_blocks: int) -> int:
    """Guess the traversal direction from the previous block to ``idx``.

    A step longer than half the list wraps around the end, so it continues
    the other way: from the last block back to block 0 still runs forward.
    """
    if last_idx < 0:
        return 1
    step = idx - last_idx
    if abs(step) > num_blocks // 2:
        return -1 if step > 0 else 1
    return 1 if step >= 0 else -1


def _plan_target_signature(plan: HostModuleLoadPlan) -> BlockSignature:
    """Return the layout-equivalence key for one block's GPU targets."""
    instance = plan.instance
    params = plan.sources
    param_sig = tuple(
        (
            tuple(names),
            params[names[0]].requires_grad,
            params[names[0]].target_layout,
        )
        for names in group_names(
            params.keys(),
            lambda name: id(instance.params[name]),
        )
    )
    buffers = instance.buffers
    buffer_sig = tuple(
        (tuple(names), buffers[names[0]].target_layout)
        for names in group_names(buffers.keys(), lambda name: id(buffers[name]))
    )
    return (param_sig, buffer_sig)


class _TargetPool:
    """Reusable stream-aware GPU targets, keyed by block layout signature.

    Only the runtime's forward thread takes targets from or returns them to
    the pool, so it needs no lock.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._free: dict[BlockSignature, list[CudaTargetLease]] = {}

    def acquire(
        self,
        signature: BlockSignature,
        plan: HostModuleLoadPlan,
        stream: torch.cuda.Stream,
    ) -> CudaTargetLease:
        free = self._free.get(signature)
        if free:
            return free.pop()
        # The prefetch worker refills pooled targets outside the caller's
        # inference mode, which cannot update inference tensors in place.
        with torch.inference_mode(False):
            return CudaTargetLease.allocate(
                plan,
                self._device,
                allocation_stream=stream,
            )

    def release(
        self,
        signature: BlockSignature,
        lease: CudaTargetLease,
    ) -> None:
        self._free.setdefault(signature, []).append(lease)

    def close(self) -> None:
        for leases in self._free.values():
            for lease in leases:
                lease.close()
        self._free.clear()


@dataclass(eq=False, slots=True)
class _BlockTarget:
    """A target leased to one block and, while the worker stages it, that prefetch.

    ``installed`` turns true only after the block's module points at the
    target, and false before its release starts, so a switch that stopped
    partway is released before the block runs again.
    """

    block_idx: int
    lease: CudaTargetLease
    prefetch: Future[None] | None = None
    installed: bool = False


class StreamingBlockRuntime:
    """One active block plus one whole-block lookahead target.

    Every target lease is in the pool, the active target, or the lookahead
    target. The active target's block is installed on it. The lookahead target
    belongs to the prefetch worker until :func:`_wait_for_prefetch` returns, so
    only then does it become active or return to the pool. Only the forward
    thread moves targets between these places, and each step keeps every lease
    in one of them even when it raises.
    """

    def __init__(
        self,
        instances: Sequence[HostModuleInstance],
        *,
        wraparound: bool = True,
    ) -> None:
        self._instances = tuple(instances)
        self._wraparound = wraparound
        self._device: torch.device | None = None
        self._load_plans: tuple[HostModuleLoadPlan, ...] = ()
        self._signatures: tuple[BlockSignature, ...] = ()
        self._pool: _TargetPool | None = None
        self._active: _BlockTarget | None = None
        self._lookahead: _BlockTarget | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._executor: ThreadPoolExecutor | None = None
        self._stream: torch.cuda.Stream | None = None
        self._last_idx = -1
        self._optimizer_step_active = False
        self._move_trainable_grads_to(torch.device("cpu"))

    @property
    def acquired(self) -> bool:
        return self._device is not None

    @property
    def compile_backend(self) -> CompileBackend:
        return "inductor"

    def acquire(
        self,
        device: torch.device,
        load_plans: Sequence[HostModuleLoadPlan],
    ) -> None:
        if self.acquired:
            raise RuntimeError("streaming block runtime is already acquired")

        plans = validate_load_plans(self._instances, load_plans)

        num_blocks = len(self._instances)
        self._device = device
        self._load_plans = plans
        self._signatures = tuple(_plan_target_signature(plan) for plan in plans)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._stream = torch.cuda.Stream(device=device, priority=-1)
        self._last_idx = -1
        self._pool = _TargetPool(device)
        self._move_trainable_grads_to(device)

        current_stream = torch.cuda.current_stream(device)
        self._acquire_block(self._stage_block(0, current_stream), current_stream)

        self._register_hooks()

        logger.info(
            "streaming block runtime acquired: one block on GPU plus one lookahead target across %d blocks",
            num_blocks,
        )

    def release(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        try:
            self._release_blocks()
        finally:
            # After a failed prefetch both targets are back in the pool, so
            # cleanup finishes before the error propagates. A block that could
            # not return to host storage still holds its target: stay acquired,
            # as after a failed synchronization, so nothing reuses that target.
            if self._active is None and self._lookahead is None:
                self._stream = None
                self._move_trainable_grads_to(torch.device("cpu"))
                if self._pool is not None:
                    self._pool.close()
                self._pool = None
                self._last_idx = -1
                self._device = None
                self._load_plans = ()
                self._signatures = ()

    @contextlib.contextmanager
    def optimizer_step(self) -> Generator[None]:
        if not self.acquired:
            raise RuntimeError(
                "BlockComponent.optimizer_step() called while its CUDA "
                "working set is released. Acquire the component before "
                "entering the optimizer step."
            )
        if self._optimizer_step_active:
            raise RuntimeError(
                "BlockComponent.optimizer_step() does not support "
                "reentrant entry. A nested optimizer-step boundary would "
                "scatter the outer step's stale host bytes on top of "
                "the inner update."
            )
        if not any(instance.has_trainables for instance in self._instances):
            yield
            return

        self._release_blocks()

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
            raise RuntimeError("streaming block runtime is released")
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
                stack.callback(instance.install_host)
                plan = instance.resolve_load_plan().select_parameters(
                    instance.trainable_param_names,
                )
                target = plan.allocate_target(
                    device,
                    buffer_names=(),
                )
                plan.load_to_target(target, non_blocking=True)
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

    # Targets move only on the forward thread. Each step keeps every lease in
    # the pool, the active target, or the lookahead target, also when it raises.
    # A target leaves its place only after the step that frees it returns: a
    # block's target once the block is back on host storage, the lookahead's
    # once its prefetch has finished.

    def _take_target(self, block_idx: int, stream: torch.cuda.Stream) -> _BlockTarget:
        """Take a pooled target for ``block_idx``, allocating it on ``stream`` if none fits."""
        assert self._pool is not None, "runtime is not acquired"
        plan = self._load_plans[block_idx]
        return _BlockTarget(block_idx, self._pool.acquire(self._signatures[block_idx], plan, stream))

    def _return_target(self, block_target: _BlockTarget) -> None:
        """Return a target to the pool once its block is uninstalled and no prefetch writes it."""
        assert self._pool is not None, "runtime is not acquired"
        block_target.lease.release()
        self._pool.release(self._signatures[block_target.block_idx], block_target.lease)

    def _stage_block(self, block_idx: int, stream: torch.cuda.Stream) -> _BlockTarget:
        """Stage ``block_idx`` on ``stream`` from the forward thread."""
        block_target = self._take_target(block_idx, stream)
        try:
            block_target.lease.stage(self._load_plans[block_idx], stream, non_blocking=False)
        except BaseException:
            self._return_target(block_target)
            raise
        return block_target

    def _acquire_block(self, block_target: _BlockTarget, stream: torch.cuda.Stream) -> None:
        """Install a staged target on its block and make it the active target."""
        self._active = block_target
        instance = self._load_plans[block_target.block_idx].instance
        instance.install_target(block_target.lease.acquire(stream))
        block_target.installed = True

    def _release_block(self, block_target: _BlockTarget) -> None:
        """Restore a block to host storage, then return its target to the pool.

        If the restore fails, some parameters may still point at the target,
        so it is not returned; the caller keeps tracking it for a retry.
        """
        block_target.installed = False
        self._load_plans[block_target.block_idx].instance.install_host()
        self._return_target(block_target)

    def _submit_prefetch(self, block_idx: int) -> None:
        """Stage ``block_idx`` into the lookahead target on the worker, unless one is in use."""
        assert self._executor is not None
        stream = self._stream
        assert stream is not None
        active = self._active
        if self._lookahead is not None or (active is not None and active.block_idx == block_idx):
            return
        lookahead = self._lookahead = self._take_target(block_idx, stream)
        lookahead.prefetch = self._executor.submit(
            lookahead.lease.stage,
            self._load_plans[block_idx],
            stream,
            non_blocking=True,
        )

    def _take_lookahead(self) -> _BlockTarget | None:
        """Empty the lookahead once its prefetch finishes and return it.

        A failed prefetch returns its target to the pool and raises.
        """
        lookahead, self._lookahead = self._lookahead, None
        if lookahead is not None and lookahead.prefetch is not None:
            try:
                _wait_for_prefetch(lookahead.prefetch)
            except BaseException:
                self._return_target(lookahead)
                raise
            lookahead.prefetch = None
        return lookahead

    def _take_staged(self, block_idx: int, stream: torch.cuda.Stream) -> _BlockTarget:
        """Return ``block_idx`` staged: the lookahead if it holds that block, else staged on ``stream``."""
        lookahead = self._take_lookahead()
        if lookahead is not None:
            if lookahead.block_idx == block_idx:
                return lookahead
            # A skipped or out-of-order block. Returning its target lets the
            # next prefetch start instead of waiting until that block runs.
            self._return_target(lookahead)
        return self._stage_block(block_idx, stream)

    def _release_blocks(self) -> None:
        """Return both targets to the pool; a failed prefetch raises after both are back."""
        active = self._active
        if active is not None:
            self._release_block(active)
            self._active = None
        lookahead = self._take_lookahead()
        if lookahead is not None:
            self._return_target(lookahead)

    def _before_block_forward(self, idx: int) -> None:
        device = self._device
        if device is None:
            return

        current_stream = torch.cuda.current_stream(device)
        active = self._active
        if active is None or active.block_idx != idx or not active.installed:
            if active is not None:
                self._release_block(active)
                self._active = None
            active = self._take_staged(idx, current_stream)
            self._acquire_block(active, current_stream)
        active.lease.record_stream(current_stream)

        last_idx, self._last_idx = self._last_idx, idx
        num_blocks = len(self._instances)
        if self._wraparound or idx != num_blocks - 1:
            next_idx = (idx + _prefetch_direction(last_idx, idx, num_blocks)) % num_blocks
            self._submit_prefetch(next_idx)

    def _register_hooks(self) -> None:
        runtime_ref = weakref.ref(self)

        def _pre_hook(_module: nn.Module, _args: tuple[object, ...], *, idx: int) -> None:
            runtime = runtime_ref()
            if runtime is not None:
                runtime._before_block_forward(idx)

        last_idx_by_module = {
            id(plan.instance.module): idx
            for idx, plan in enumerate(self._load_plans)
        }
        for idx in last_idx_by_module.values():
            module = self._load_plans[idx].instance.module
            handle = module.register_forward_pre_hook(
                functools.partial(_pre_hook, idx=idx)
            )
            self._hooks.append(handle)


__all__ = ["StreamingBlockRuntime"]
