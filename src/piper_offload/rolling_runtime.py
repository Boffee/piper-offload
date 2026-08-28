"""CUDA execution strategy for compiled single-target parameter rolling."""

import contextlib
import functools
import logging
import weakref
from collections.abc import Iterator, Sequence

import torch
from torch import nn

from .block_compile import BlockCompileConfig, CompileBackend
from .float8_adapter import Float8Adapter
from .gguf_adapter import GgufAdapter
from .int4_tile_adapter import Int4TilePackedAdapter
from .int8_adapter import Int8Adapter
from .mx_adapter import MxAdapter
from .nvfp4_adapter import Nvfp4Adapter
from .pinned_module import (
    PinnedModuleInstance,
    PinnedModuleTarget,
)
from .piper_convrot_int8_adapter import PiperConvRotInt8Adapter
from .quanto_adapter import QuantoAdapter
from .rolling_compile import (
    register_rolling_target,
    rolling_inductor_backend,
    unregister_rolling_target,
)
from .static_float8_adapter import StaticFloat8Adapter
from .stream_config import StreamConfig
from .tensor_adapters import RegularAdapter

logger = logging.getLogger(__name__)

_ROLLING_ADAPTER_TYPES = (
    RegularAdapter,
    Float8Adapter,
    StaticFloat8Adapter,
    Int8Adapter,
    Int4TilePackedAdapter,
    MxAdapter,
    Nvfp4Adapter,
    QuantoAdapter,
    GgufAdapter,
    PiperConvRotInt8Adapter,
)


class _RollingTargetRuntime:
    """One shared CUDA target refilled parameter-by-parameter."""

    def __init__(
        self,
        instances: tuple[PinnedModuleInstance, ...],
        param_names: tuple[str, ...],
        log_label: str,
    ) -> None:
        self._instances = instances
        self._param_names = param_names
        self._log_label = log_label
        self._reset_active_state()

    def _reset_active_state(self) -> None:
        self._target: PinnedModuleTarget | None = None
        self._stream: torch.cuda.Stream | None = None
        self._events: tuple[torch.cuda.Event, ...] = ()
        self._ready_events: tuple[torch.cuda.Event, ...] = ()
        self._fallback_event: torch.cuda.Event | None = None
        self._owners: list[int] | None = None
        self._cyclic = False
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    @property
    def active(self) -> bool:
        return self._target is not None or self._stream is not None

    @property
    def compile_backend(self) -> CompileBackend:
        return rolling_inductor_backend

    def activate(self, device: torch.device, config: StreamConfig) -> None:
        if self.active:
            raise RuntimeError("rolling target runtime is already active")
        if config.num_resident_blocks != 1 or config.num_prefetch_blocks != 0:
            raise ValueError(
                "rolling compilation uses exactly one shared target and "
                "requires StreamConfig(num_resident_blocks=1, "
                "num_prefetch_blocks=0)"
            )
        self._cyclic = config.cyclic
        self._stream = torch.cuda.Stream(device=device, priority=-1)
        self._events = tuple(torch.cuda.Event() for _ in self._param_names)
        self._ready_events = tuple(torch.cuda.Event() for _ in self._param_names)
        self._fallback_event = torch.cuda.Event()
        self._target = self._instances[0].allocate_target(device)
        register_rolling_target(
            self,
            [self._target.param_targets[name].param for name in self._param_names],
        )
        self._owners = [-1] * len(self._param_names)

        with torch.cuda.stream(self._stream):
            self._instances[0].load_to_target(
                self._target,
                run_post_copy_hooks=True,
                non_blocking=True,
            )
            for ready_event in self._ready_events:
                ready_event.record(self._stream)
        torch.cuda.current_stream(device).wait_stream(self._stream)
        self._owners[:] = [0] * len(self._param_names)
        for instance in self._instances:
            instance.install_target(self._target)
        self._register_hooks(device)

        logger.info(
            f"{self._log_label} active: one rolling parameter target for {len(self._instances)} compiled blocks"
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
        if self._target is None or prefetch_stream is None or fallback_event is None or owners is None:
            return

        missing = [idx for idx, owner in enumerate(owners) if owner != block_idx]
        if not missing:
            return

        # Handles skipped/out-of-order blocks and the first block of a new
        # non-cyclic invocation. The sequential hot path arrives with every
        # refill enqueued; compiled waits provide per-parameter readiness.
        fallback_event.record(torch.cuda.current_stream(device))
        with torch.cuda.stream(prefetch_stream):
            prefetch_stream.wait_event(fallback_event)
            for param_idx in missing:
                self._refill(block_idx, param_idx)

    def wait_param(self, param_idx: int) -> None:
        target = self._target
        if target is None:
            raise RuntimeError("rolling wait executed while runtime is inactive")
        device = target.param_targets[self._param_names[param_idx]].param.device
        torch.cuda.current_stream(device).wait_event(self._ready_events[param_idx])

    def rollover_param(self, param_idx: int) -> None:
        target = self._target
        prefetch_stream = self._stream
        owners = self._owners
        if target is None or prefetch_stream is None or owners is None:
            raise RuntimeError("rolling refill executed while runtime is inactive")
        block_idx = owners[param_idx]
        if block_idx < 0:
            name = self._param_names[param_idx]
            raise RuntimeError(f"rolling graph used {name!r} before its slot had an owner")

        next_idx = block_idx + 1
        if next_idx == len(self._instances):
            if not self._cyclic:
                return
            next_idx = 0

        name = self._param_names[param_idx]
        device = target.param_targets[name].param.device
        current_stream = torch.cuda.current_stream(device)
        compute_done = self._events[param_idx]
        compute_done.record(current_stream)
        with torch.cuda.stream(prefetch_stream):
            prefetch_stream.wait_event(compute_done)
            self._refill(next_idx, param_idx)

    def _refill(self, block_idx: int, param_idx: int) -> None:
        target = self._target
        prefetch_stream = self._stream
        owners = self._owners
        assert target is not None
        assert prefetch_stream is not None
        assert owners is not None
        name = self._param_names[param_idx]
        param_target = target.param_targets[name]
        self._instances[block_idx].refill_param_target(name, param_target)
        self._ready_events[param_idx].record(prefetch_stream)
        owners[param_idx] = block_idx

    def deactivate(self) -> None:
        for handle in self._hooks:
            handle.remove()
        first_error: BaseException | None = None
        if self._stream is not None:
            try:
                self._stream.synchronize()
            except BaseException as exc:
                first_error = exc
        if self._target is not None:
            unregister_rolling_target(self)
        for instance in self._instances:
            try:
                instance.install_pinned()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._reset_active_state()
        if first_error is not None:
            raise first_error

    @contextlib.contextmanager
    def optimizer_step(self) -> Iterator[None]:
        if not self.active:
            raise RuntimeError(
                "StreamedComponent.optimizer_step() called on inactive "
                "streamer. Use it inside the offloader's context manager."
            )
        yield


def _validate_instance(
    instance: PinnedModuleInstance,
    block_idx: int,
    param_names: tuple[str, ...],
    reference_layouts: tuple[tuple[object, object], ...],
) -> None:
    params = instance.params
    if instance.has_trainables:
        raise NotImplementedError(
            f"rolling compilation is inference-only; streamed block {block_idx} contains trainable parameters"
        )
    if instance.buffers:
        raise NotImplementedError("rolling compilation does not yet support streamed buffers")
    if tuple(params) != param_names:
        raise NotImplementedError("rolling compilation requires identical parameter names and ordering in every block")
    if len({id(pinned) for pinned in params.values()}) != len(params):
        raise NotImplementedError(f"rolling compilation does not yet support tied parameters inside block {block_idx}")
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


def create_rolling_runtime(
    instances: Sequence[PinnedModuleInstance],
    config: BlockCompileConfig | None,
    *,
    log_label: str,
) -> _RollingTargetRuntime | None:
    """Validate and create the configured rolling runtime, if enabled."""
    if config is None or not config.rolling:
        return None
    if not instances:
        raise ValueError("rolling compilation requires at least one streamed block")
    if len({id(instance.module) for instance in instances}) != len(instances):
        raise NotImplementedError("rolling compilation does not support aliased block modules")

    reference_params = instances[0].params
    param_names = tuple(reference_params)
    if not param_names:
        raise ValueError("rolling compilation requires streamed parameters")
    reference_layouts = tuple(pinned.target_layout for pinned in reference_params.values())
    for block_idx, instance in enumerate(instances):
        _validate_instance(
            instance,
            block_idx,
            param_names,
            reference_layouts,
        )

    return _RollingTargetRuntime(tuple(instances), param_names, log_label)


__all__ = ["create_rolling_runtime"]
