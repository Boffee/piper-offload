"""CUDA execution strategy for compiled single-target parameter rolling."""

import contextlib
import functools
import logging
import math
import weakref
from collections.abc import Generator, Sequence

import torch
from torch import nn

from .block_compile import CompileBackend
from .float8_adapter import Float8Adapter
from .gguf_adapter import GgufAdapter
from .int4_tile_adapter import Int4TilePackedAdapter
from .int8_adapter import Int8Adapter
from .mx_adapter import MxAdapter
from .nvfp4_adapter import Nvfp4Adapter
from .pinned_module import PinnedModuleInstance
from .pinned_param import PinnedParam
from .piper_convrot_int8_adapter import PiperConvRotInt8Adapter
from .piper_convrot_nvfp4_adapter import PiperConvRotNVFP4Adapter
from .quanto_adapter import QuantoAdapter
from .rolling_compile import (
    register_rolling_target,
    rolling_inductor_backend,
    unregister_rolling_target,
)
from .static_float8_adapter import StaticFloat8Adapter
from .target_lease import _CudaTargetLease
from .tensor_adapters import RegularAdapter

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
    QuantoAdapter,
    GgufAdapter,
    PiperConvRotInt8Adapter,
)


class RollingBlockRuntime:
    """One shared CUDA target refilled parameter-by-parameter."""

    def __init__(
        self,
        instances: tuple[PinnedModuleInstance, ...],
        *,
        wraparound: bool = True,
    ) -> None:
        self._instances = instances
        self._wraparound = wraparound
        self._reset_acquired_state()

    def _reset_acquired_state(self) -> None:
        self._lease: _CudaTargetLease | None = None
        self._stream: torch.cuda.Stream | None = None
        self._events: tuple[torch.cuda.Event, ...] = ()
        self._ready_events: tuple[torch.cuda.Event, ...] = ()
        self._fallback_event: torch.cuda.Event | None = None
        self._owners: list[int] | None = None
        self._slot_names: tuple[str, ...] = ()
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    @property
    def acquired(self) -> bool:
        return self._lease is not None or self._stream is not None

    @property
    def compile_backend(self) -> CompileBackend:
        return rolling_inductor_backend

    def acquire(self, device: torch.device) -> None:
        if self.acquired:
            raise RuntimeError("rolling block runtime is already acquired")
        self._stream = torch.cuda.Stream(device=device, priority=-1)
        materialized_by_instance = tuple(
            instance.materialized_params for instance in self._instances
        )
        materialized_names = {
            name
            for params in materialized_by_instance
            for name in params
        }
        self._slot_names = tuple(
            name
            for name in self._instances[0].params
            if name in materialized_names
        )
        self._events = tuple(torch.cuda.Event() for _name in self._slot_names)
        self._ready_events = tuple(torch.cuda.Event() for _name in self._slot_names)
        self._owners = [0] * len(self._slot_names)

        if not self._slot_names:
            logger.info(
                "rolling block runtime acquired with no materialized parameter slots across %d blocks",
                len(self._instances),
            )
            return

        self._fallback_event = torch.cuda.Event()
        slot_backings: dict[str, PinnedParam] = {}
        for name in self._slot_names:
            active = [
                backing for params in materialized_by_instance
                if (backing := params.get(name)) is not None
            ]
            assert active
            backing = active[0]
            if any(
                candidate.target_layout != backing.target_layout
                for candidate in active[1:]
            ):
                raise NotImplementedError(
                    "rolling compilation requires identical active parameter "
                    f"layouts for slot {name!r} in every materialized block"
                )
            if type(backing.adapter) not in _ROLLING_ADAPTER_TYPES:
                raise NotImplementedError(
                    "rolling compilation does not support active parameter "
                    f"adapter {type(backing.adapter).__name__} for slot {name!r}"
                )
            if math.prod(backing.logical_shape) == 0:
                raise NotImplementedError(
                    "rolling compilation does not support zero-sized parameter slots"
                )
            slot_backings[name] = backing
        self._lease = _CudaTargetLease.allocate(
            self._instances[0],
            device,
            param_names=self._slot_names,
            buffer_names=(),
            param_backings=slot_backings,
        )
        target = self._lease.target
        register_rolling_target(
            self,
            [target.param_targets[name].param for name in self._slot_names],
        )

        self._lease.stage(
            self._instances[0],
            self._stream,
            run_post_copy_hooks=True,
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
        self._instances[block_idx].refill_param_target(name, param_target)
        self._ready_events[param_idx].record(prefetch_stream)
        owners[param_idx] = block_idx

    def release(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        first_error: BaseException | None = None
        lease = self._lease
        if lease is not None:
            unregister_rolling_target(self)
        for instance in self._instances:
            try:
                instance.install_pinned()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if lease is not None:
            try:
                lease.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._reset_acquired_state()
        if first_error is not None:
            raise first_error

    @contextlib.contextmanager
    def optimizer_step(self) -> Generator[None]:
        if not self.acquired:
            raise RuntimeError(
                "BlockComponent.optimizer_step() called while its CUDA "
                "working set is released. Acquire the component before "
                "entering the optimizer step."
            )
        yield


def _validate_instance(
    instance: PinnedModuleInstance,
    param_names: tuple[str, ...],
    reference_layouts: tuple[tuple[object, object], ...],
) -> None:
    params = instance.params
    if instance.has_trainables:
        raise NotImplementedError("rolling compilation is inference-only")
    if instance.buffers:
        raise NotImplementedError("rolling compilation does not yet support block buffers")
    if tuple(params) != param_names:
        raise NotImplementedError("rolling compilation requires identical parameter names and ordering in every block")
    if len({id(pinned) for pinned in params.values()}) != len(params):
        raise NotImplementedError("rolling compilation does not yet support tied parameters inside blocks")
    layouts = tuple(pinned.target_layout for pinned in params.values())
    if layouts != reference_layouts:
        raise NotImplementedError("rolling compilation requires identical parameter layouts in every block")
    if any(type(pinned.adapter) not in _ROLLING_ADAPTER_TYPES for pinned in params.values()):
        raise NotImplementedError(
            "rolling compilation supports only regular dense, TorchAO-family, "
            "Quanto, GGUF, and Piper ConvRot INT8 parameters"
        )
    if any(pinned.shape.numel() == 0 for pinned in params.values()):
        raise NotImplementedError("rolling compilation does not support zero-sized parameter slots")


def create_rolling_block_runtime(
    instances: Sequence[PinnedModuleInstance],
    *,
    wraparound: bool = True,
) -> RollingBlockRuntime:
    """Validate and create a rolling block runtime."""
    if not instances:
        raise ValueError("rolling compilation requires at least one block")
    if len({id(instance.module) for instance in instances}) != len(instances):
        raise NotImplementedError("rolling compilation does not support aliased block modules")

    reference_params = instances[0].params
    param_names = tuple(reference_params)
    if not param_names:
        raise NotImplementedError("rolling compilation requires block parameters")
    reference_layouts = tuple(pinned.target_layout for pinned in reference_params.values())
    for instance in instances:
        _validate_instance(
            instance,
            param_names,
            reference_layouts,
        )

    return RollingBlockRuntime(
        tuple(instances),
        wraparound=wraparound,
    )


__all__ = ["RollingBlockRuntime", "create_rolling_block_runtime"]
