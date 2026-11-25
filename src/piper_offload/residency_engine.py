"""Layer 2 Generic Residency Engine: execution prediction, async prefetch, and VRAM residency."""

from __future__ import annotations

import contextlib
import functools
import logging
import threading
import weakref
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass

import torch
from torch import nn

from . import pin_manager as _pin_manager
from ._devices import canonical_device
from ._transfer_priority import background_transfers
from ._working_set import WorkingSetState
from .block_compile import BlockCompileConfig, _BlockCompileState
from .device_transfer import DeviceTransferContext
from .host_module import HostModuleInstance, HostModuleLoadPlan, ParameterOverride
from .offload_config import _USE_CONFIG_BUDGET, _UseConfigBudget, validate_vram_budget
from .target_lease import CudaTargetLease
from .vram_budget import BudgetTenantHandle, VramBudget, VramReservation
from .vram_pool import (
    ElasticVramPool,
    VramSlot,
    estimate_plan_gpu_bytes,
    plan_target_signature,
    target_gpu_bytes,
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


class _ExecutionPredictor:
    """Online execution-order trace predictor and Belady MIN distance estimator.

    Indexes observed suffix transitions of length 1..4 inside the bounded
    history window so both next-unit prediction and Belady distance rollouts run
    in O(1) per lookup and steady-state traversals advance the cached forecast
    incrementally in O(1) per unit instead of rebuilding on every step.
    """

    def __init__(self, num_units: int) -> None:
        if num_units <= 0:
            raise ValueError(f"num_units must be positive, got {num_units}.")
        self._num_units = num_units
        self._max_history = max(256, 8 * num_units)
        self._history: deque[int] = deque()
        self._seq = 0
        # Map k-gram suffix (k in 1..4) -> (predicted_next_unit, write_seq).
        self._tables: tuple[dict[tuple[int, ...], tuple[int, int]], ...] = (
            {},
            {},
            {},
            {},
            {},
        )
        self._forecast: deque[int] | None = None
        self._forecast_states: deque[tuple[int, int, int, int] | None] = deque()
        self._seen_states: set[tuple[int, int, int, int]] = set()
        self._unit_steps: dict[int, deque[int]] = {}
        self._forecast_offset = 0
        self._rollout_tail: tuple[int, ...] = ()
        self._lock = threading.Lock()

    def _invalidate_forecast(self) -> None:
        self._forecast = None
        self._forecast_states.clear()
        self._seen_states.clear()
        self._unit_steps.clear()
        self._forecast_offset = 0
        self._rollout_tail = ()

    def _update_table(self, k: int, pattern: tuple[int, ...], nxt: int, seq: int) -> bool:
        table = self._tables[k]
        prev = table.get(pattern)
        table[pattern] = (nxt, seq)
        return prev is None or prev[0] != nxt

    def _evict_table(self, k: int, pattern: tuple[int, ...], nxt: int, seq: int) -> bool:
        table = self._tables[k]
        if table.get(pattern) == (nxt, seq):
            del table[pattern]
            return True
        return False

    def record(self, unit_idx: int) -> None:
        """Record execution of ``unit_idx`` and advance or invalidate cached distances."""
        with self._lock:
            seq = self._seq
            self._seq += 1
            hist = self._history
            hlen = len(hist)
            changed = False
            if hlen > 0:
                tail = tuple(hist[i] for i in range(max(0, hlen - 4), hlen))
                for k in range(1, len(tail) + 1):
                    changed |= self._update_table(k, tail[-k:], unit_idx, seq)
            hist.append(unit_idx)
            if len(hist) > self._max_history:
                evicted = hist.popleft()
                evict_seq = seq - self._max_history
                head = tuple(hist[i] for i in range(min(4, len(hist))))
                for k in range(1, len(head) + 1):
                    changed |= self._evict_table(
                        k,
                        (evicted, *head[: k - 1]),
                        head[k - 1],
                        evict_seq + k,
                    )

            if (
                not changed
                and self._forecast is not None
                and len(self._forecast) > 0
                and len(hist) >= 4
                and self._forecast[0] == unit_idx
            ):
                self._forecast.popleft()
                old_state = self._forecast_states.popleft()
                if old_state is not None:
                    self._seen_states.discard(old_state)
                self._forecast_offset += 1
                steps = self._unit_steps.get(unit_idx)
                if steps is not None:
                    steps.popleft()
                    if not steps:
                        del self._unit_steps[unit_idx]
                self._extend_forecast()
            else:
                self._invalidate_forecast()

    def _predict_from_tail(self, tail: tuple[int, ...]) -> int:
        for k in range(min(4, len(tail)), 0, -1):
            entry = self._tables[k].get(tail[-k:])
            if entry is not None:
                return entry[0]
        curr = tail[-1]
        if len(tail) >= 2 and tail[-2] - curr == 1:
            return (curr - 1) % self._num_units
        return (curr + 1) % self._num_units

    def next_after(self, unit_idx: int) -> int | None:
        """Predict the next unit likely to execute after ``unit_idx``."""
        with self._lock:
            if self._num_units <= 1:
                return None
            hist = self._history
            if hist and hist[-1] == unit_idx:
                hlen = len(hist)
                tail = tuple(hist[i] for i in range(max(0, hlen - 4), hlen))
            else:
                tail = (unit_idx,)
            nxt = self._predict_from_tail(tail)
            if nxt == unit_idx:
                return None
            return nxt

    def _extend_forecast(self) -> None:
        assert self._forecast is not None
        horizon = max(2 * self._num_units, len(self._history), 32)
        tail = self._rollout_tail
        num_units = self._num_units
        forecast = self._forecast
        f_states = self._forecast_states
        seen = self._seen_states
        unit_steps = self._unit_steps
        offset = self._forecast_offset
        while len(forecast) < horizon and len(unit_steps) < num_units:
            state: tuple[int, int, int, int] | None = None
            if len(tail) >= 4:
                state = (tail[-4], tail[-3], tail[-2], tail[-1])
                if state in seen:
                    break
                seen.add(state)
            nxt = self._predict_from_tail(tail)
            abs_step = offset + len(forecast) + 1
            forecast.append(nxt)
            f_states.append(state)
            steps = unit_steps.get(nxt)
            if steps is None:
                unit_steps[nxt] = deque((abs_step,))
            else:
                steps.append(abs_step)
            tail = (*tail[-3:], nxt)
        self._rollout_tail = tail

    def _ensure_forecast(self) -> None:
        if self._forecast is not None:
            return
        hist = self._history
        if hist:
            hlen = len(hist)
            self._rollout_tail = tuple(hist[i] for i in range(max(0, hlen - 4), hlen))
        else:
            self._rollout_tail = (self._num_units - 1,)
        self._forecast = deque()
        self._forecast_states.clear()
        self._seen_states.clear()
        self._unit_steps.clear()
        self._forecast_offset = 0
        self._extend_forecast()

    def distance(self, unit_idx: int) -> float:
        """Estimated forward steps until ``unit_idx`` is next used (``inf`` if far)."""
        with self._lock:
            self._ensure_forecast()
            steps = self._unit_steps.get(unit_idx)
            if not steps:
                return float("inf")
            return float(steps[0] - self._forecast_offset)


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
        self._resident_reservation: VramReservation | None = None
        self._resident_transfer = _pin_manager.TransferLease()
        self._resident_use_hook: torch.utils.hooks.RemovableHandle | None = None

        self._unit_plans: tuple[HostModuleLoadPlan, ...] = ()
        self._pool: ElasticVramPool | None = None
        self._predictor: _ExecutionPredictor | None = None
        self._active: _ActiveUnit | None = None
        self._lookahead: _LookaheadTask | None = None
        self._pin_lease: _pin_manager.PinLease | None = None
        self._units_transfer = _pin_manager.TransferLease()
        self._transfer_context: DeviceTransferContext | None = None
        self._owns_transfer_context = False
        self._transfer_context_borrowed = False
        self._budget_handle: BudgetTenantHandle | None = None
        self._no_record_trace_depth = 0
        self._unit_hooks: list[torch.utils.hooks.RemovableHandle] = []

    @property
    def _copy_stream(self) -> torch.cuda.Stream | None:
        return None if self._transfer_context is None else self._transfer_context.copy_stream

    @property
    def _executor(self) -> ThreadPoolExecutor | None:
        return None if self._transfer_context is None else self._transfer_context.executor

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
            or self._resident_reservation is not None
            or (self._pool is not None and self._pool.live_slot_count > 0)
            or self._pin_lease is not None
            or self._budget_handle is not None
            or (
                self._owns_transfer_context
                and self._transfer_context is not None
                and self._transfer_context.is_open
            )
            or self._transfer_context_borrowed
            or self._resident_transfer.open
            or self._units_transfer.open
        )

    @property
    def is_recording_trace(self) -> bool:
        """Whether unit forwards currently record execution-trace history."""
        return self._no_record_trace_depth == 0

    @contextlib.contextmanager
    def no_record_trace(self) -> Iterator[None]:
        """Temporarily suppress execution-trace history recording (re-entrant)."""
        self._no_record_trace_depth += 1
        try:
            yield
        finally:
            self._no_record_trace_depth -= 1

    def _distance_fn(self) -> Callable[[int], float]:
        predictor = self._predictor
        if predictor is None:
            return lambda _idx: float("inf")
        return predictor.distance

    def _close_pin_lease(self) -> None:
        if self._pin_lease is not None:
            self._pin_lease.close()
            self._pin_lease = None

    def _unregister_budget_handle(self) -> None:
        if self._budget_handle is not None:
            self._budget_handle.unregister()
            self._budget_handle = None

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
        if self._transfer_context is None:
            self._transfer_context = DeviceTransferContext(device)
            self._owns_transfer_context = True
        return self._transfer_context.ensure()

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
        *,
        vram_budget: VramBudget | None = None,
    ) -> None:
        if (
            resident_plan is None
            or self._resident is None
            or not (resident_plan.loads or self._resident.buffers)
        ):
            return
        if vram_budget is not None:
            est_bytes = estimate_plan_gpu_bytes(resident_plan)
            self._resident_reservation = vram_budget.reserve(
                est_bytes,
                name="residency_engine.resident",
            )
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
        if self._resident_reservation is not None:
            actual_bytes = target_gpu_bytes(lease.target)
            if actual_bytes != self._resident_reservation.nbytes:
                self._resident_reservation.resize(actual_bytes)
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
        vram_budget: VramBudget | None,
        compile_blocks: bool,
    ) -> None:
        if not self._units:
            return
        predictor = _ExecutionPredictor(len(self._units))
        self._predictor = predictor
        pool = ElasticVramPool(
            active_device,
            vram_budget_bytes=vram_budget_bytes,
        )
        if vram_budget is not None:
            handle = vram_budget.register_tenant(
                "residency_engine",
                priority=10,
                evict_cb=lambda needed: pool.evict_idle_bytes(
                    needed,
                    distance_fn=predictor.distance,
                ),
                usage_fn=lambda: pool.used_bytes,
                yield_to_peers=True,
            )
            self._budget_handle = handle
            pool.set_budget_callbacks(
                budget_fn=handle.ceiling_bytes,
                claim_bytes_fn=handle.try_claim,
            )
        self._pool = pool

        total_unit_bytes = sum(estimate_plan_gpu_bytes(plan) for plan in unit_plans)
        all_resident = (
            vram_budget is None
            and vram_budget_bytes is not None
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
        vram_budget: VramBudget | None = None,
        transfer_context: DeviceTransferContext | None = None,
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
        if vram_budget is not None and not isinstance(vram_budget, VramBudget):
            raise TypeError(
                f"vram_budget must be a VramBudget or None; got {type(vram_budget).__name__}."
            )
        if transfer_context is not None and not isinstance(
            transfer_context, DeviceTransferContext
        ):
            raise TypeError(
                "transfer_context must be a DeviceTransferContext or None; "
                f"got {type(transfer_context).__name__}."
            )
        active_device = canonical_device(device)
        if active_device.type not in ("cpu", "cuda"):
            raise ValueError(
                f"ResidencyEngine supports CUDA or CPU; got {active_device}."
            )
        if transfer_context is not None and transfer_context.device != active_device:
            raise ValueError(
                f"transfer_context device {transfer_context.device} does not match "
                f"activation device {active_device}."
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
            if transfer_context is not None:
                transfer_context._acquire_borrow()
                self._transfer_context = transfer_context
                self._owns_transfer_context = False
                self._transfer_context_borrowed = True
            stream = torch.cuda.current_stream(active_device)
            self._activate_resident_cuda(
                active_device,
                resident_plan,
                stream,
                vram_budget=vram_budget,
            )
            self._activate_units_cuda(
                active_device,
                unit_plans,
                stream,
                vram_budget_bytes=session_budget,
                vram_budget=vram_budget,
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
        if self._no_record_trace_depth == 0:
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

        if self._no_record_trace_depth == 0:
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
        pool = self._pool
        if pool is not None:
            budget = pool.budget_bytes
            if budget is not None and (budget == 0 or active.slot.nbytes > budget):
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
        if self._owns_transfer_context and self._transfer_context is not None:
            _capture(self._transfer_context._shutdown_executor)
        _capture(self._release_active_unit)
        _capture(self._units_transfer.finish)
        if self._pool is not None:
            pool = self._pool
            _capture(pool.close_all if self._active is None else pool.close_idle)
            if pool.live_slot_count == 0:
                self._unregister_budget_handle()
                self._pool = None
        else:
            self._unregister_budget_handle()
        if (
            self._lookahead is None
            and self._active is None
            and self._pool is None
            and not self._units_transfer.open
        ):
            if self._owns_transfer_context and self._transfer_context is not None:
                _capture(self._transfer_context.close)
                if self._transfer_context.is_closed:
                    self._transfer_context = None
                    self._owns_transfer_context = False
            owns_open_ctx = (
                self._owns_transfer_context and self._transfer_context is not None
            )
            if not owns_open_ctx:
                _capture(self._close_pin_lease)
                if self._pin_lease is None:
                    if (
                        self._transfer_context_borrowed
                        and self._transfer_context is not None
                    ):
                        self._transfer_context._release_borrow()
                        self._transfer_context_borrowed = False
                    if not self._owns_transfer_context:
                        self._transfer_context = None
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
        if self._resident_lease is None and self._resident_reservation is not None:
            self._resident_reservation.release()
            self._resident_reservation = None
        return cleanup_error

    def release(self) -> None:
        """Idempotently release CUDA targets, prefetch work, and pin leases."""
        if self._working_set_state is WorkingSetState.RELEASED:
            return
        self._working_set_state = WorkingSetState.CHANGING

        unit_error, prefetch_error = self._release_units()
        resident_error = self._release_resident()
        cleanup_error = unit_error if unit_error is not None else resident_error

        owns_open_ctx = (
            self._owns_transfer_context and self._transfer_context is not None
        )
        all_released = (
            not owns_open_ctx
            and not self._transfer_context_borrowed
            and self._lookahead is None
            and self._active is None
            and self._pool is None
            and self._pin_lease is None
            and self._budget_handle is None
            and self._resident_lease is None
            and self._resident_reservation is None
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
