"""Layer 2 Generic Residency Engine: execution prediction, async prefetch, and VRAM residency."""

from __future__ import annotations

import functools
import logging
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass

import torch
from torch import nn

from . import pin_manager as _pin_manager
from ._devices import canonical_device
from ._transfer_priority import background_transfers
from ._working_set import WorkingSetState
from .block_compile import BlockCompileConfig, _BlockCompileState
from .host_module import HostModuleInstance, HostModuleLoadPlan, ParameterOverride
from .offload_config import _USE_CONFIG_BUDGET, _UseConfigBudget, validate_vram_budget
from .target_lease import CudaTargetLease
from .vram_pool import (
    ElasticVramPool,
    VramSlot,
    estimate_plan_gpu_bytes,
    plan_target_signature,
)

logger = logging.getLogger(__name__)


def _stage_lookahead(
    lease: CudaTargetLease,
    plan: HostModuleLoadPlan,
    stream: torch.cuda.Stream,
) -> None:
    """Run lookahead host-to-CUDA staging under background transfer priority."""
    with background_transfers():
        lease.stage(plan, stream, non_blocking=True)


def _wait_for_prefetch(future: Future[None]) -> None:
    """Wait until a lookahead worker finishes, even if the caller is interrupted.

    Returning a slot to the pool or closing its CUDA lease while the worker
    is still staging into that target would race on the lease and its
    storage. If waiting is interrupted, keep waiting until the worker exits
    before re-raising the interrupt.
    """
    try:
        future.result()
    except BaseException:
        while not future.done():
            try:
                wait((future,))
            except BaseException:
                continue
        raise


def _shutdown_executor(executor: ThreadPoolExecutor) -> None:
    """Shut down ``executor`` and wait for its worker thread even if interrupted."""
    interrupt: BaseException | None = None
    while True:
        try:
            executor.shutdown(wait=True)
            break
        except BaseException as exc:
            if interrupt is None:
                interrupt = exc
    if interrupt is not None:
        raise interrupt


def _predict_next(
    history: Sequence[int],
    tail: Sequence[int],
    num_units: int,
) -> int:
    """Predict the unit following ``tail`` using recent suffix transitions.

    Searches ``history[:-1]`` from most recent to oldest for the longest
    suffix match (up to length 4), falling back to observed +/-1 stride in
    declaration order on unseen Step-0 transitions.
    """
    hist_len = len(history)
    max_k = min(4, len(tail), hist_len - 1)
    for k in range(max_k, 0, -1):
        pattern = tail[-k:]
        for start in range(hist_len - 1 - k, -1, -1):
            matched = True
            for offset in range(k):
                if history[start + offset] != pattern[offset]:
                    matched = False
                    break
            if matched:
                return history[start + k]

    curr = tail[-1]
    prev = tail[-2] if len(tail) >= 2 else None
    if prev is not None and prev - curr == 1:
        return (curr - 1) % num_units
    return (curr + 1) % num_units


class _ExecutionPredictor:
    """Online execution-order trace predictor and Belady MIN distance estimator."""

    def __init__(self, num_units: int) -> None:
        if num_units <= 0:
            raise ValueError(f"num_units must be positive, got {num_units}.")
        self._num_units = num_units
        self._max_history = max(256, 8 * num_units)
        self._history: list[int] = []
        self._cached_distances: dict[int, int] | None = None

    def record(self, unit_idx: int) -> None:
        """Record execution of ``unit_idx`` and invalidate cached distances."""
        self._history.append(unit_idx)
        if len(self._history) > self._max_history:
            del self._history[: len(self._history) - self._max_history]
        self._cached_distances = None

    def next_after(self, unit_idx: int) -> int | None:
        """Predict the next unit likely to execute after ``unit_idx``."""
        if self._num_units <= 1:
            return None
        tail = self._history if self._history and self._history[-1] == unit_idx else [unit_idx]
        nxt = _predict_next(self._history, tail, self._num_units)
        if nxt == unit_idx:
            return None
        return nxt

    def _compute_distances(self) -> dict[int, int]:
        horizon = max(2 * self._num_units, len(self._history), 32)
        window = list(self._history) if self._history else [self._num_units - 1]
        distances: dict[int, int] = {}
        for step in range(1, horizon + 1):
            nxt = _predict_next(self._history, window, self._num_units)
            if nxt not in distances:
                distances[nxt] = step
            window.append(nxt)
            if len(window) > self._max_history:
                del window[: len(window) - self._max_history]
        return distances

    def distance(self, unit_idx: int) -> float:
        """Estimated forward steps until ``unit_idx`` is next used (``inf`` if far)."""
        if self._cached_distances is None:
            self._cached_distances = self._compute_distances()
        dist = self._cached_distances.get(unit_idx)
        return float("inf") if dist is None else float(dist)


@dataclass(slots=True)
class _ActiveUnit:
    unit_idx: int
    slot: VramSlot
    installed: bool = False


@dataclass(slots=True)
class _LookaheadTask:
    unit_idx: int
    slot: VramSlot
    future: Future[None]


class ResidencyEngine:
    """Manages host and dynamic CUDA residency for partitioned model instances.

    Unifies the non-offloaded resident remainder (uploaded once without pinning)
    and offloaded units (cached up to ``vram_budget_bytes`` using recorded-order
    Belady eviction, or strictly streamed across 2 reusable slots when
    ``vram_budget_bytes is None``).
    """

    def __init__(
        self,
        *,
        resident: HostModuleInstance | None,
        units: Sequence[HostModuleInstance],
        unit_paths: Sequence[str],
        vram_budget_bytes: int | None = None,
        block_compile: BlockCompileConfig | None = None,
    ) -> None:
        if len(units) != len(unit_paths):
            raise ValueError("units and unit_paths must have equal length.")
        self._resident = resident
        self._units = tuple(units)
        self._unit_paths = tuple(unit_paths)
        self._default_budget_bytes = validate_vram_budget(vram_budget_bytes)
        self._block_compile = _BlockCompileState.create(
            [unit.module for unit in self._units],
            block_compile,
            backend="inductor",
        )

        self._working_set_state = WorkingSetState.RELEASED
        self._active_device: torch.device | None = None
        self._session_device: torch.device | None = None

        self._resident_plan: HostModuleLoadPlan | None = None
        self._resident_lease: CudaTargetLease | None = None
        self._resident_transfer = _pin_manager.TransferLease()
        self._resident_use_hook: torch.utils.hooks.RemovableHandle | None = None

        self._unit_plans: tuple[HostModuleLoadPlan, ...] = ()
        self._pool: ElasticVramPool | None = None
        self._predictor: _ExecutionPredictor | None = None
        self._active: _ActiveUnit | None = None
        self._lookahead: _LookaheadTask | None = None
        self._pin_lease: _pin_manager.PinLease | None = None
        self._units_transfer = _pin_manager.TransferLease()
        self._copy_stream: torch.cuda.Stream | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._unit_hooks: list[torch.utils.hooks.RemovableHandle] = []

    @property
    def active_device(self) -> torch.device | None:
        return self._active_device

    @property
    def resident(self) -> HostModuleInstance | None:
        return self._resident

    @property
    def units(self) -> tuple[HostModuleInstance, ...]:
        return self._units

    @property
    def unit_paths(self) -> tuple[str, ...]:
        return self._unit_paths

    @property
    def block_compile(self) -> BlockCompileConfig | None:
        return self._block_compile.config

    @property
    def vram_budget_bytes(self) -> int | None:
        return self._pool.budget_bytes if self._pool is not None else self._default_budget_bytes

    @property
    def pool(self) -> ElasticVramPool | None:
        return self._pool

    @property
    def failed(self) -> bool:
        """Whether an ended session left unreleased CUDA, pin, or compile state."""
        return self._active_device is None and (
            self._working_set_state is not WorkingSetState.RELEASED
            or self._block_compile.installed
        )

    @property
    def retains_resources(self) -> bool:
        """Whether any GPU targets, hooks, pin leases, or compiled forwards remain open."""
        return (
            self._working_set_state is not WorkingSetState.RELEASED
            or self._block_compile.installed
            or self._resident_lease is not None
            or (self._pool is not None and self._pool.live_slot_count > 0)
            or self._pin_lease is not None
            or self._resident_transfer.open
            or self._units_transfer.open
        )

    def _distance_fn(self) -> Callable[[int], float]:
        predictor = self._predictor
        if predictor is None:
            return lambda _idx: float("inf")
        return predictor.distance

    def _close_pin_lease(self) -> None:
        if self._pin_lease is not None:
            self._pin_lease.close()
            self._pin_lease = None

    def _remove_resident_use_hook(self) -> None:
        hook = self._resident_use_hook
        self._resident_use_hook = None
        if hook is not None:
            hook.remove()

    def _remove_unit_hooks(self) -> None:
        for hook in self._unit_hooks:
            hook.remove()
        self._unit_hooks.clear()

    def _all_unit_storage_tensors(self) -> Iterable[torch.Tensor]:
        for plan in self._unit_plans:
            yield from plan.storage_tensors()

    def _ensure_streaming_resources(self) -> tuple[torch.cuda.Stream, ThreadPoolExecutor]:
        """Lazily acquire pinned host storage, H2D copy stream, and prefetch worker."""
        device = self._session_device
        if device is None:
            raise RuntimeError("ResidencyEngine CUDA session is not active.")
        if self._pin_lease is None and self._unit_plans:
            self._pin_lease = _pin_manager.host_pin_manager.acquire(
                self._all_unit_storage_tensors(),
                pin=True,
            )
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=device, priority=-1)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        return self._copy_stream, self._executor

    def _install_resident_use_hook(self) -> None:
        if self._resident is None or self._resident_lease is None:
            return
        engine_ref = weakref.ref(self)

        def record_stream(_module: nn.Module, _args: tuple[object, ...]) -> None:
            engine = engine_ref()
            if engine is None:
                return
            device = engine._active_device
            lease = engine._resident_lease
            if device is not None and device.type == "cuda" and lease is not None:
                lease.record_stream(torch.cuda.current_stream(device))

        self._resident_use_hook = self._resident.module.register_forward_pre_hook(
            record_stream,
            prepend=True,
        )

    def _register_unit_hooks(self) -> None:
        if not self._units:
            return
        engine_ref = weakref.ref(self)

        def _pre_hook(_module: nn.Module, _args: tuple[object, ...], *, idx: int) -> None:
            engine = engine_ref()
            if engine is not None:
                engine._before_unit_forward(idx)

        def _post_hook(
            _module: nn.Module,
            _args: tuple[object, ...],
            _output: object,
            *,
            idx: int,
        ) -> None:
            engine = engine_ref()
            if engine is not None:
                engine._after_unit_forward(idx)

        for idx, unit in enumerate(self._units):
            self._unit_hooks.append(
                unit.module.register_forward_pre_hook(
                    functools.partial(_pre_hook, idx=idx),
                    prepend=True,
                )
            )
            self._unit_hooks.append(
                unit.module.register_forward_hook(
                    functools.partial(_post_hook, idx=idx),
                    prepend=False,
                    always_call=True,
                )
            )

    def _activate_resident_cuda(
        self,
        active_device: torch.device,
        resident_plan: HostModuleLoadPlan | None,
        stream: torch.cuda.Stream,
    ) -> None:
        if (
            resident_plan is None
            or self._resident is None
            or not (resident_plan.loads or self._resident.buffers)
        ):
            return
        self._resident_transfer.start(
            _pin_manager.host_pin_manager,
            resident_plan.storage_tensors(),
            active_device,
        )
        lease = CudaTargetLease.allocate(
            resident_plan,
            active_device,
            allocation_stream=stream,
        )
        self._resident_lease = lease
        lease.stage(resident_plan, stream, non_blocking=True)
        stream.synchronize()
        self._resident_transfer.close()
        target = lease.acquire(stream)
        self._resident.install_target(target)
        self._install_resident_use_hook()

    def _activate_units_cuda(
        self,
        active_device: torch.device,
        unit_plans: tuple[HostModuleLoadPlan, ...],
        stream: torch.cuda.Stream,
        *,
        vram_budget_bytes: int | None,
        compile_blocks: bool,
    ) -> None:
        if not self._units:
            return
        self._predictor = _ExecutionPredictor(len(self._units))
        pool = ElasticVramPool(
            active_device,
            vram_budget_bytes=vram_budget_bytes,
        )
        self._pool = pool

        total_unit_bytes = sum(estimate_plan_gpu_bytes(plan) for plan in unit_plans)
        all_resident = (
            vram_budget_bytes is not None
            and total_unit_bytes <= vram_budget_bytes
        )
        if all_resident:
            self._units_transfer.start(
                _pin_manager.host_pin_manager,
                self._all_unit_storage_tensors(),
                active_device,
            )
            for idx, plan in enumerate(unit_plans):
                lease = CudaTargetLease.allocate(
                    plan,
                    active_device,
                    allocation_stream=stream,
                )
                pool.admit_preloaded_slot(
                    idx,
                    lease,
                    plan_target_signature(plan),
                )
                lease.stage(plan, stream, non_blocking=True)
            stream.synchronize()
            self._units_transfer.close()
        else:
            self._ensure_streaming_resources()

        self._register_unit_hooks()
        self._block_compile.install(compile_blocks)

    def _activate_cpu(
        self,
        active_device: torch.device,
        resident_overrides: Mapping[str, ParameterOverride] | None,
        per_unit_overrides: tuple[Mapping[str, ParameterOverride] | None, ...],
        *,
        compile_blocks: bool,
    ) -> None:
        if resident_overrides or any(bool(ovr) for ovr in per_unit_overrides):
            raise ValueError("Parameter overrides require CUDA activation.")
        if self._resident is not None:
            self._resident.install_host()
        for unit in self._units:
            unit.install_host()
        try:
            self._block_compile.install(compile_blocks)
        except BaseException:
            self._block_compile.restore()
            raise
        self._active_device = active_device

    def activate(
        self,
        device: torch.device | str,
        *,
        compile_blocks: bool = True,
        resident_overrides: Mapping[str, ParameterOverride] | None = None,
        unit_overrides: Sequence[Mapping[str, ParameterOverride] | None] | None = None,
        vram_budget_bytes: int | _UseConfigBudget | None = _USE_CONFIG_BUDGET,
    ) -> None:
        """Activate residency on ``device`` (CPU or CUDA)."""
        if self.failed:
            raise RuntimeError(
                "ResidencyEngine cannot activate after its prior session failed "
                "to finish releasing; deactivate() retries that release. Recreate "
                "the worker only if the retry fails too."
            )
        if self._active_device is not None:
            raise RuntimeError("ResidencyEngine is already active; call deactivate() first.")

        session_budget = (
            self._default_budget_bytes
            if isinstance(vram_budget_bytes, _UseConfigBudget)
            else validate_vram_budget(vram_budget_bytes)
        )
        active_device = canonical_device(device)
        if active_device.type not in ("cpu", "cuda"):
            raise ValueError(
                f"ResidencyEngine supports CUDA or CPU; got {active_device}."
            )

        per_unit_overrides = (
            (None,) * len(self._units)
            if unit_overrides is None
            else tuple(unit_overrides)
        )
        if len(per_unit_overrides) != len(self._units):
            raise ValueError("unit_overrides length must match offloaded units.")

        if active_device.type == "cpu":
            self._activate_cpu(
                active_device,
                resident_overrides,
                per_unit_overrides,
                compile_blocks=compile_blocks,
            )
            return

        resident_plan = (
            self._resident.resolve_load_plan(resident_overrides)
            if self._resident is not None
            else None
        )
        unit_plans = tuple(
            unit.resolve_load_plan(ovr)
            for unit, ovr in zip(self._units, per_unit_overrides, strict=True)
        )

        self._active_device = active_device
        self._session_device = active_device
        self._resident_plan = resident_plan
        self._unit_plans = unit_plans
        self._working_set_state = WorkingSetState.CHANGING

        try:
            stream = torch.cuda.current_stream(active_device)
            self._activate_resident_cuda(active_device, resident_plan, stream)
            self._activate_units_cuda(
                active_device,
                unit_plans,
                stream,
                vram_budget_bytes=session_budget,
                compile_blocks=compile_blocks,
            )
        except BaseException:
            self.deactivate()
            raise

        self._working_set_state = WorkingSetState.ACQUIRED

    def _release_active_unit(self) -> None:
        active = self._active
        if active is None:
            return
        active.installed = False
        self._units[active.unit_idx].install_host()
        active.slot.lease.release()
        self._active = None
        assert self._pool is not None
        self._pool.return_slot(
            active.slot,
            staged_unit_idx=active.unit_idx,
            distance_fn=self._distance_fn(),
        )

    def _drain_lookahead(self) -> BaseException | None:
        lookahead, self._lookahead = self._lookahead, None
        if lookahead is None:
            return None
        prefetch_error: BaseException | None = None
        try:
            _wait_for_prefetch(lookahead.future)
        except BaseException as exc:
            prefetch_error = exc
        staged_idx = lookahead.unit_idx if prefetch_error is None else None
        assert self._pool is not None
        self._pool.return_slot(
            lookahead.slot,
            staged_unit_idx=staged_idx,
            distance_fn=self._distance_fn(),
        )
        return prefetch_error

    def _reuse_active_unit(self, idx: int, device: torch.device) -> bool:
        active = self._active
        if active is None:
            return False
        if active.unit_idx != idx or not active.installed:
            self._release_active_unit()
            return False
        assert self._predictor is not None
        assert self._pool is not None
        current_stream = torch.cuda.current_stream(device)
        self._predictor.record(idx)
        active.slot.lease.record_stream(current_stream)
        if self._lookahead is None and self._pool.budget_bytes != 0:
            next_idx = self._predictor.next_after(idx)
            if (
                next_idx is not None
                and next_idx != idx
                and not self._pool.has_cached_unit(next_idx)
            ):
                self._submit_lookahead(next_idx)
        return True

    def _before_unit_forward(self, idx: int) -> None:
        device = self._active_device
        if (
            device is None
            or device.type != "cuda"
            or self._pool is None
            or self._predictor is None
        ):
            return

        if self._reuse_active_unit(idx, device):
            return

        self._predictor.record(idx)
        current_stream = torch.cuda.current_stream(device)

        slot: VramSlot | None = None
        lookahead, self._lookahead = self._lookahead, None
        if lookahead is not None:
            try:
                _wait_for_prefetch(lookahead.future)
            except BaseException:
                self._pool.return_slot(
                    lookahead.slot,
                    staged_unit_idx=None,
                    distance_fn=self._distance_fn(),
                )
                raise
            if lookahead.unit_idx == idx:
                slot = lookahead.slot
            else:
                # Keep mispredicted prefetched unit parked if budget permits.
                self._pool.return_slot(
                    lookahead.slot,
                    staged_unit_idx=lookahead.unit_idx,
                    distance_fn=self._distance_fn(),
                )

        if slot is None:
            slot, cache_hit = self._pool.checkout(
                idx,
                self._unit_plans[idx],
                current_stream,
                distance_fn=self._distance_fn(),
                is_lookahead=False,
            )
            assert slot is not None
            if not cache_hit:
                try:
                    self._ensure_streaming_resources()
                    slot.lease.stage(
                        self._unit_plans[idx],
                        current_stream,
                        non_blocking=False,
                    )
                except BaseException:
                    self._pool.return_slot(
                        slot,
                        staged_unit_idx=None,
                        distance_fn=self._distance_fn(),
                    )
                    raise

        active = _ActiveUnit(unit_idx=idx, slot=slot, installed=False)
        self._active = active
        target = slot.lease.acquire(current_stream)
        self._units[idx].install_target(target)
        active.installed = True

        # Submit overlapped lookahead for predicted successor if not already cached.
        if self._pool.budget_bytes != 0:
            next_idx = self._predictor.next_after(idx)
            if next_idx is not None and not self._pool.has_cached_unit(next_idx):
                self._submit_lookahead(next_idx)

    def _submit_lookahead(self, next_idx: int) -> None:
        assert self._pool is not None
        assert self._predictor is not None
        copy_stream, executor = self._ensure_streaming_resources()
        slot, cache_hit = self._pool.checkout(
            next_idx,
            self._unit_plans[next_idx],
            copy_stream,
            distance_fn=self._predictor.distance,
            is_lookahead=True,
        )
        if slot is None:
            return
        if cache_hit:
            self._pool.return_slot(
                slot,
                staged_unit_idx=next_idx,
                distance_fn=self._predictor.distance,
            )
            return
        try:
            future = executor.submit(
                _stage_lookahead,
                slot.lease,
                self._unit_plans[next_idx],
                copy_stream,
            )
        except BaseException:
            self._pool.return_slot(
                slot,
                staged_unit_idx=None,
                distance_fn=self._predictor.distance,
            )
            raise
        self._lookahead = _LookaheadTask(
            unit_idx=next_idx,
            slot=slot,
            future=future,
        )

    def _after_unit_forward(self, idx: int) -> None:
        active = self._active
        if active is None or active.unit_idx != idx:
            return
        if self._pool is not None and self._pool.budget_bytes == 0:
            self._release_active_unit()

    def _release_units(self) -> tuple[BaseException | None, BaseException | None]:
        cleanup_error: BaseException | None = None
        prefetch_error: BaseException | None = None

        def _capture(op: Callable[[], object]) -> None:
            nonlocal cleanup_error
            try:
                op()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc

        _capture(self._remove_unit_hooks)
        if self._pool is not None:

            def _drain() -> None:
                nonlocal prefetch_error
                prefetch_error = self._drain_lookahead()

            _capture(_drain)
        if self._executor is not None:
            executor, self._executor = self._executor, None
            _capture(lambda: _shutdown_executor(executor))
        _capture(self._release_active_unit)
        _capture(self._units_transfer.finish)
        if self._pool is not None:
            pool = self._pool
            _capture(pool.close_all if self._active is None else pool.close_idle)
            if pool.live_slot_count == 0:
                self._pool = None
        if (
            self._executor is None
            and self._lookahead is None
            and self._active is None
            and self._pool is None
            and not self._units_transfer.open
        ):
            _capture(self._close_pin_lease)
            if self._pin_lease is None:
                self._copy_stream = None
                self._predictor = None
                self._unit_plans = ()
        return cleanup_error, prefetch_error

    def _release_resident(self) -> BaseException | None:
        cleanup_error: BaseException | None = None

        def _capture(op: Callable[[], object]) -> None:
            nonlocal cleanup_error
            try:
                op()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc

        _capture(self._resident_transfer.finish)
        if self._resident_lease is not None:
            _capture(self._remove_resident_use_hook)

            def _close_resident() -> None:
                if self._resident is not None:
                    self._resident.install_host()
                assert self._resident_lease is not None
                self._resident_lease.close()
                self._resident_lease = None
                self._resident_plan = None

            _capture(_close_resident)
        else:
            self._resident_plan = None
        return cleanup_error

    def release(self) -> None:
        """Idempotently release CUDA targets, prefetch work, and pin leases."""
        if self._working_set_state is WorkingSetState.RELEASED:
            return
        self._working_set_state = WorkingSetState.CHANGING

        unit_error, prefetch_error = self._release_units()
        resident_error = self._release_resident()
        cleanup_error = unit_error if unit_error is not None else resident_error

        all_released = (
            self._executor is None
            and self._lookahead is None
            and self._active is None
            and self._pool is None
            and self._pin_lease is None
            and self._resident_lease is None
            and not self._units_transfer.open
            and not self._resident_transfer.open
        )
        if all_released:
            self._session_device = None
            self._working_set_state = WorkingSetState.RELEASED

        if cleanup_error is not None:
            raise cleanup_error
        if prefetch_error is not None:
            raise prefetch_error

    def deactivate(self) -> None:
        """Restore compiled forwards, release CUDA storage, and end the session."""
        first_error: BaseException | None = None
        try:
            try:
                self._block_compile.restore()
            except BaseException as exc:
                first_error = exc
            try:
                self.release()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        finally:
            self._active_device = None
        if first_error is not None:
            raise first_error


__all__ = [
    "ResidencyEngine",
    "_ExecutionPredictor",
]
