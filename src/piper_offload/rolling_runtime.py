"""CUDA execution strategy for compiled single-target parameter rolling."""

import functools
import logging
import math
import weakref
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .block_compile import CompileBackend
from .block_runtime import validate_load_plans
from .float8_adapter import Float8Adapter
from .gguf_adapter import GgufAdapter
from .host_module import (
    HostModuleInstance,
    HostModuleLoadPlan,
    ParameterLoad,
)
from .int4_tile_adapter import Int4TilePackedAdapter
from .int8_adapter import Int8Adapter
from .mx_adapter import MxAdapter
from .nvfp4_adapter import Nvfp4Adapter
from .piper_convrot_int8_adapter import PiperConvRotInt8Adapter
from .piper_convrot_nvfp4_adapter import PiperConvRotNVFP4Adapter
from .rolling_compile import (
    register_rolling_target,
    rolling_inductor_backend,
    unregister_rolling_target,
)
from .static_float8_adapter import StaticFloat8Adapter
from .target_lease import CudaTargetLease
from .tensor_adapters import RegularAdapter
from .trellis_adapter import TrellisAdapter

logger = logging.getLogger(__name__)

_ROLLING_ADAPTER_TYPES = (
    RegularAdapter,
    Float8Adapter,
    StaticFloat8Adapter,
    Int8Adapter,
    Int4TilePackedAdapter,
    MxAdapter,
    PiperConvRotNVFP4Adapter,
    Nvfp4Adapter,
    GgufAdapter,
    TrellisAdapter,
    PiperConvRotInt8Adapter,
)


def _resolve_rolling_loads(
    instances: Sequence[HostModuleInstance],
    load_plans: Sequence[HostModuleLoadPlan],
) -> tuple[tuple[HostModuleLoadPlan, ...], HostModuleLoadPlan]:
    """Build the shared allocation plan in the order used by compiled slot indices."""
    plans = validate_load_plans(instances, load_plans)
    source_names = {name for plan in plans for name in plan.loads}
    allocation_loads: dict[str, ParameterLoad] = {}
    for name in instances[0].params:
        if name not in source_names:
            continue
        active = [
            load
            for plan in plans
            if (load := plan.loads.get(name)) is not None
        ]
        assert active
        source = active[0].source
        if any(
            candidate.source.target_layout != source.target_layout
            for candidate in active[1:]
        ):
            raise NotImplementedError(
                "rolling compilation requires identical active parameter "
                f"layouts for slot {name!r} in every block with a source"
            )
        if type(source.adapter) not in _ROLLING_ADAPTER_TYPES:
            raise NotImplementedError(
                "rolling compilation does not support active parameter "
                f"adapter {type(source.adapter).__name__} for slot {name!r}"
            )
        if math.prod(source.logical_shape) == 0:
            raise NotImplementedError(
                "rolling compilation does not support zero-sized parameter slots"
            )
        # This synthetic plan defines shared target storage only. Block
        # updates remain exclusively on their activation load plans.
        allocation_loads[name] = ParameterLoad(source)
    return plans, HostModuleLoadPlan(instances[0], allocation_loads)


class RollingBlockRuntime:
    """One shared CUDA target refilled parameter-by-parameter."""

    def __init__(
        self,
        instances: tuple[HostModuleInstance, ...],
        *,
        wraparound: bool = True,
    ) -> None:
        self._instances = instances
        self._wraparound = wraparound
        self._reset_acquired_state()

    def _reset_acquired_state(self) -> None:
        self._lease: CudaTargetLease | None = None
        self._stream: torch.cuda.Stream | None = None
        self._events: tuple[torch.cuda.Event, ...] = ()
        self._ready_events: tuple[torch.cuda.Event, ...] = ()
        self._fallback_event: torch.cuda.Event | None = None
        self._owners: list[int] | None = None
        self._slot_names: tuple[str, ...] = ()
        self._load_plans: tuple[HostModuleLoadPlan, ...] = ()
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    @property
    def acquired(self) -> bool:
        return self._lease is not None or self._stream is not None

    @property
    def compile_backend(self) -> CompileBackend:
        return rolling_inductor_backend

    def validate_load_plans(
        self,
        load_plans: Sequence[HostModuleLoadPlan],
    ) -> None:
        """Raise when active loads cannot use one compiled rolling target."""
        _resolve_rolling_loads(self._instances, load_plans)

    def acquire(
        self,
        device: torch.device,
        load_plans: Sequence[HostModuleLoadPlan],
    ) -> None:
        acquire = self._prepare_acquire(device, load_plans)
        acquire()

    def _prepare_acquire(
        self, device: torch.device, load_plans: Sequence[HostModuleLoadPlan],
    ) -> Callable[[], None]:
        """Bind this activation's layout for its initial acquire and all reacquires."""
        if self.acquired:
            raise RuntimeError("rolling block runtime is already acquired")
        plans, allocation = _resolve_rolling_loads(self._instances, load_plans)
        return functools.partial(self._acquire, device, plans, allocation, tuple(allocation.loads))

    def _acquire(
        self,
        device: torch.device,
        plans: tuple[HostModuleLoadPlan, ...],
        allocation: HostModuleLoadPlan,
        slot_names: tuple[str, ...],
    ) -> None:
        self._load_plans = plans
        self._slot_names = slot_names
        self._stream = torch.cuda.Stream(device=device, priority=-1)
        self._events = tuple(torch.cuda.Event() for _name in self._slot_names)
        self._ready_events = tuple(torch.cuda.Event() for _name in self._slot_names)
        self._owners = [0] * len(self._slot_names)

        if not self._slot_names:
            logger.info(
                "rolling block runtime acquired with no active parameter slots across %d blocks",
                len(self._instances),
            )
            return

        self._fallback_event = torch.cuda.Event()
        self._lease = CudaTargetLease.allocate(
            allocation,
            device,
            buffer_names=(),
        )
        target = self._lease.target
        register_rolling_target(
            self,
            [target.param_targets[name].param for name in self._slot_names],
        )

        self._lease.stage(
            plans[0],
            self._stream,
            non_blocking=True,
        )
        with torch.cuda.stream(self._stream):
            for ready_event in self._ready_events:
                ready_event.record(self._stream)
        self._lease.acquire(torch.cuda.current_stream(device))
        self._lease.record_stream(self._stream)
        for instance in self._instances:
            instance.install_target(target)
        self._register_hooks(device)

        logger.info(
            "rolling block runtime acquired: one parameter target for %d compiled blocks",
            len(self._instances),
        )

    def _register_hooks(self, device: torch.device) -> None:
        runtime_ref = weakref.ref(self)

        def _pre_hook(
            _module: nn.Module,
            _args: tuple[object, ...],
            *,
            idx: int,
        ) -> None:
            runtime = runtime_ref()
            if runtime is not None:
                runtime.before_block(idx, device)

        for idx, instance in enumerate(self._instances):
            handle = instance.module.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            self._hooks.append(handle)

    def before_block(self, block_idx: int, device: torch.device) -> None:
        prefetch_stream = self._stream
        fallback_event = self._fallback_event
        owners = self._owners
        if self._lease is None or prefetch_stream is None or fallback_event is None or owners is None:
            return

        missing = [idx for idx, owner in enumerate(owners) if owner != block_idx]
        if not missing:
            return

        # Handles skipped or out-of-order blocks. The sequential hot path
        # arrives with every refill enqueued; compiled waits provide
        # per-parameter readiness.
        fallback_event.record(torch.cuda.current_stream(device))
        with torch.cuda.stream(prefetch_stream):
            prefetch_stream.wait_event(fallback_event)
            for param_idx in missing:
                self._refill(block_idx, param_idx)

    def wait_param(self, param_idx: int) -> None:
        lease = self._lease
        if lease is None:
            raise RuntimeError("rolling wait executed while runtime is released")
        target = lease.target
        name = self._slot_names[param_idx]
        device = target.param_targets[name].param.device
        current_stream = torch.cuda.current_stream(device)
        lease.record_stream(current_stream)
        current_stream.wait_event(self._ready_events[param_idx])

    def rollover_param(self, param_idx: int) -> None:
        lease = self._lease
        prefetch_stream = self._stream
        owners = self._owners
        if lease is None or prefetch_stream is None or owners is None:
            raise RuntimeError("rolling refill executed while runtime is released")
        target = lease.target
        block_idx = owners[param_idx]
        next_idx = block_idx + 1
        if next_idx == len(self._instances):
            if not self._wraparound:
                return
            next_idx = 0

        name = self._slot_names[param_idx]
        device = target.param_targets[name].param.device
        current_stream = torch.cuda.current_stream(device)
        compute_done = self._events[param_idx]
        compute_done.record(current_stream)
        with torch.cuda.stream(prefetch_stream):
            prefetch_stream.wait_event(compute_done)
            self._refill(next_idx, param_idx)

    def _refill(self, block_idx: int, param_idx: int) -> None:
        lease = self._lease
        prefetch_stream = self._stream
        owners = self._owners
        assert lease is not None
        target = lease.target
        assert prefetch_stream is not None
        assert owners is not None
        name = self._slot_names[param_idx]
        param_target = target.param_targets[name]
        # The slot holds no known block until the refill is enqueued, so one
        # that stopped partway is refilled instead of trusted.
        owners[param_idx] = -1
        self._load_plans[block_idx].refill_param_target(name, param_target)
        self._ready_events[param_idx].record(prefetch_stream)
        owners[param_idx] = block_idx

    def release(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        lease = self._lease
        if lease is not None:
            unregister_rolling_target(self)
        # Every block returns to host storage before the shared target closes.
        # A failure keeps the target and leaves the runtime acquired, so
        # nothing reuses storage a block may still point at.
        for instance in self._instances:
            instance.install_host()
        if lease is not None:
            lease.close()
        self._reset_acquired_state()


@dataclass(frozen=True, slots=True)
class RollingSlotSpec:
    """Normalized per-slot metadata used to validate rolling compatibility."""

    identity: Hashable
    adapter_type: type
    shape: torch.Size
    target_layout: tuple[object, object]


def _validate_rolling_block_slots(
    params: Mapping[str, RollingSlotSpec],
    param_names: tuple[str, ...],
    reference_layouts: tuple[tuple[object, object], ...],
) -> None:
    if tuple(params) != param_names:
        raise NotImplementedError("rolling compilation requires identical parameter names and ordering in every block")
    if len({spec.identity for spec in params.values()}) != len(params):
        raise NotImplementedError("rolling compilation does not yet support tied parameters inside blocks")
    layouts = tuple(spec.target_layout for spec in params.values())
    if layouts != reference_layouts:
        raise NotImplementedError("rolling compilation requires identical parameter layouts in every block")
    if any(spec.adapter_type not in _ROLLING_ADAPTER_TYPES for spec in params.values()):
        raise NotImplementedError(
            "rolling compilation supports only regular dense, TorchAO-family, "
            "GGUF, trellis-coded, and Piper ConvRot INT8/NVFP4 parameters"
        )
    if any(spec.shape.numel() == 0 for spec in params.values()):
        raise NotImplementedError("rolling compilation does not support zero-sized parameter slots")


def validate_rolling_block_specs(
    block_module_ids: Sequence[int],
    block_params: Sequence[Mapping[str, RollingSlotSpec]],
    *,
    has_buffers: bool,
) -> None:
    """Validate rolling constraints prior to or during runtime construction."""
    if not block_params:
        raise ValueError("rolling compilation requires at least one block")
    if len(set(block_module_ids)) != len(block_module_ids):
        raise NotImplementedError("rolling compilation does not support aliased block modules")
    if has_buffers:
        raise NotImplementedError("rolling compilation does not yet support block buffers")

    param_names = tuple(block_params[0])
    if not param_names:
        raise NotImplementedError("rolling compilation requires block parameters")
    reference_layouts = tuple(spec.target_layout for spec in block_params[0].values())
    for params in block_params:
        _validate_rolling_block_slots(params, param_names, reference_layouts)


def create_rolling_block_runtime(
    instances: Sequence[HostModuleInstance],
    *,
    wraparound: bool = True,
) -> RollingBlockRuntime:
    """Validate and create a rolling block runtime."""
    validate_rolling_block_specs(
        [id(instance.module) for instance in instances],
        [
            {
                name: RollingSlotSpec(
                    identity=id(host),
                    adapter_type=type(host.adapter),
                    shape=host.shape,
                    target_layout=host.target_layout,
                )
                for name, host in instance.params.items()
            }
            for instance in instances
        ],
        has_buffers=any(instance.buffers for instance in instances),
    )

    return RollingBlockRuntime(
        tuple(instances),
        wraparound=wraparound,
    )


__all__ = [
    "RollingBlockRuntime",
    "RollingSlotSpec",
    "create_rolling_block_runtime",
    "validate_rolling_block_specs",
]
