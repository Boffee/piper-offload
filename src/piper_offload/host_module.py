"""Name-based host module store and instance primitives.

This module supports sharing one CPU cache across multiple
concrete model instances. Names are the durable relationship between a
store and an instance.
"""

from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self, cast

import torch
from torch import nn

from .host_buffer import HostBuffer
from .host_param import HostParam
from .module_names import group_names, resolve_parent_leaf
from .tensor_adapter_registry import (
    buffer_tensor_id,
    param_representation,
    param_tensor_id,
)

type ParameterUpdate = Callable[[nn.Parameter], None]


@dataclass(frozen=True, slots=True)
class ParameterOverride:
    """Activation-scoped replacement source and/or in-place update."""

    source: HostParam | None = None
    update: ParameterUpdate | None = None

    def __post_init__(self) -> None:
        if self.source is None and self.update is None:
            raise ValueError(
                "ParameterOverride requires a replacement source, an "
                "update, or both."
            )


@dataclass(frozen=True, slots=True)
class ParameterLoad:
    """Resolved source and optional update for one active parameter."""

    source: HostParam
    update: ParameterUpdate | None = None


@dataclass(slots=True)
class HostParamTarget:
    """Active adapter storage for one host parameter backing."""

    _state: object
    param: nn.Parameter
    _backing: HostParam


@dataclass(slots=True)
class HostBufferTarget:
    """Active tensor storage for one host buffer backing."""

    tensor: torch.Tensor


@dataclass(slots=True)
class HostModuleTarget:
    """Name-keyed active storage for a :class:`HostModuleStore`.

    Targets may contain the whole store or a validated subset of it.
    Names mapped to the same host object also point at the same
    target object.
    """

    param_targets: dict[str, HostParamTarget]
    buffer_targets: dict[str, HostBufferTarget]


@dataclass(slots=True)
class HostModuleStore:
    """Host backing bytes for one module layout.

    ``params`` and ``buffers`` are keyed by PyTorch logical names from
    ``named_parameters(remove_duplicate=False)`` and
    ``named_buffers(remove_duplicate=False)``. Selected names sharing
    storage point at the same host object.
    """

    params: dict[str, HostParam]
    buffers: dict[str, HostBuffer]

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        include_param_names: Iterable[str] | None = None,
        include_buffer_names: Iterable[str] | None = None,
    ) -> Self:
        """Capture and install owned CPU copies keyed by module names."""
        all_params = _named_parameters(module)
        params = _select_known_names(
            all_params,
            include_param_names,
        )

        all_buffers = _named_buffers(module)
        buffers = _select_known_names(
            all_buffers,
            include_buffer_names,
        )

        store = cls(
            params=_capture_params(params),
            buffers=_capture_buffers(buffers),
        )
        _validate_trainable_param_data_swaps(store.params)
        _install_host_params(module, store.params)
        _install_host_buffers(module, store.buffers)
        return store

    @property
    def cache_bytes(self) -> int:
        return _unique_cache_bytes(self.params) + _unique_cache_bytes(self.buffers)

    @property
    def has_trainables(self) -> bool:
        return bool(self.trainable_param_names)

    @property
    def trainable_param_names(self) -> tuple[str, ...]:
        return tuple(name for name, host in self.params.items() if host.requires_grad)

    def bind(self, module: nn.Module) -> HostModuleInstance:
        """Validate ``module`` and bind this store's backing bytes to it.

        The sole instance factory: layout-checks ``module`` against this
        store, constructs a :class:`HostModuleInstance` that owns
        ``module`` and shares this store's host bytes, then installs
        those host bytes onto ``module`` via
        :meth:`HostModuleInstance.install_host`.
        """
        _validate_module_matches(self.params, self.buffers, module)
        instance = HostModuleInstance(
            module=module,
            params=self.params,
            buffers=self.buffers,
        )
        instance.install_host()
        return instance


@dataclass(slots=True)
class HostModuleInstance:
    """One concrete module bound to host parameter and buffer backings.

    Owns the :class:`nn.Module` whose managed params and buffers are backed by
    this instance's host bytes, including trainable target-to-host
    synchronization. :meth:`resolve_load_plan` combines activation-scoped
    overrides with immutable model backing; the resulting plan owns target
    allocation and loading. :meth:`install_host` restores the host
    bytes onto :attr:`module`.
    """

    module: nn.Module
    params: Mapping[str, HostParam]
    buffers: Mapping[str, HostBuffer]

    @property
    def has_trainables(self) -> bool:
        return bool(self.trainable_param_names)

    @property
    def trainable_param_names(self) -> tuple[str, ...]:
        return tuple(name for name, host in self.params.items() if host.requires_grad)

    def install_host(self) -> None:
        """Install the host bytes onto :attr:`module`'s attributes.

        Pure CPU repoint: materializes the CPU wrappers on
        demand (deduped by ``id(host)`` so tied names share one wrapper)
        and installs them onto :attr:`module`, leaving its managed state on
        the host bytes. Mutates :attr:`module` in place.
        """
        _install_host_params(self.module, self.params)
        _install_host_buffers(self.module, self.buffers)

    def install_target(self, target: HostModuleTarget) -> None:
        """Install already-filled active storage without copying into it."""
        _validate_target_names_known(self.params, self.buffers, target)
        _set_params(
            self.module,
            {name: param_target.param for name, param_target in target.param_targets.items()},
        )
        _set_buffers(
            self.module,
            {name: buffer_target.tensor for name, buffer_target in target.buffer_targets.items()},
        )

    def resolve_load_plan(
        self,
        overrides: Mapping[str, ParameterOverride] | None = None,
    ) -> HostModuleLoadPlan:
        """Resolve one immutable activation plan against model alias groups."""
        selected = {} if overrides is None else dict(overrides)
        unknown = sorted(set(selected) - set(self.params))
        if unknown:
            raise ValueError(
                f"Cannot override unknown parameter names: {_format_names(unknown)}."
            )

        overrides_by_host: dict[int, ParameterOverride] = {}
        for name, override in selected.items():
            if not isinstance(override, ParameterOverride):
                raise ValueError(
                    "Parameter overrides must be ParameterOverride "
                    f"instances; {name!r} has {type(override).__name__}."
                )
            key = id(self.params[name])
            previous = overrides_by_host.get(key)
            if previous is not None and not _same_load_override(previous, override):
                raise ValueError(
                    "Tied parameter aliases cannot use different activation "
                    f"overrides; conflict at {name!r}."
                )
            overrides_by_host[key] = override

        loads_by_host: dict[int, ParameterLoad | None] = {}
        parameters: dict[str, ParameterLoad] = {}
        for name, base in self.params.items():
            key = id(base)
            if key not in loads_by_host:
                override = overrides_by_host.get(key)
                loads_by_host[key] = _resolve_parameter_load(
                    name,
                    base,
                    override,
                )
            load = loads_by_host[key]
            if load is not None:
                parameters[name] = load
        return HostModuleLoadPlan(self, parameters)

    def copy_trainables_from_target(
        self,
        target: HostModuleTarget,
        *,
        non_blocking: bool = False,
    ) -> None:
        """Copy trainable target params back into host storage.

        This is the explicit host-cache mutation path for optimizer-step
        sync. Frozen params and buffers are intentionally not copied back.
        """
        _validate_target_names_known(self.params, self.buffers, target)
        _validate_target_has_trainable_params(self.params, target)
        _copy_trainable_params_from_target(
            self.params,
            target.param_targets,
            non_blocking=non_blocking,
        )

    def move_trainable_grads_to(self, device: torch.device) -> None:
        """Move each trainable param's ``.grad`` (if any) to ``device``.

        During backward, PyTorch's native ``AccumulateGrad`` writes grads on
        the param's data device. As ``.data`` is moved between CPU and
        a GPU target, ``.grad`` keeps living wherever ``AccumulateGrad`` placed
        it; this realigns grad with data so the optimizer reads both on the
        same device. Tied params are deduplicated, and ``None`` grads (no
        backward yet, or ``zero_grad(set_to_none=True)``) are skipped.
        """
        for param in self._iter_trainable_params():
            grad = param.grad
            if grad is None or grad.device == device:
                continue
            moved = grad.to(device)
            if param.data.device == device:
                param.grad = moved
            else:
                # PyTorch's grad setter rejects cross-device grad/data pairs.
                # A trainable can transiently have offloaded (CPU) data
                # and a GPU grad, so move the grad storage in place instead.
                grad.data = moved.data

    def _iter_trainable_params(self) -> Iterator[nn.Parameter]:
        params = dict(self.module.named_parameters(remove_duplicate=False))
        seen: set[int] = set()
        for name, host in self.params.items():
            if not host.requires_grad:
                continue
            param = params[name]
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            yield param


@dataclass(frozen=True, slots=True, init=False)
class HostModuleLoadPlan:
    """Resolved parameter loads for one module activation.

    The bound instance retains model-name and alias ownership. This plan owns
    only the activation's effective sources and updates, so target allocation,
    copying, and mutation all consume the same immutable description.
    """

    instance: HostModuleInstance
    loads: Mapping[str, ParameterLoad]

    def __init__(
        self,
        instance: HostModuleInstance,
        loads: Mapping[str, ParameterLoad],
    ) -> None:
        selected = dict(loads)
        unknown = sorted(set(selected) - set(instance.params))
        if unknown:
            raise ValueError(
                f"Parameter load plan contains unknown names: {_format_names(unknown)}."
            )
        for name, load in selected.items():
            if not isinstance(load, ParameterLoad):
                raise ValueError(
                    "Resolved parameter loads must be ParameterLoad instances; "
                    f"{name!r} has {type(load).__name__}."
                )
        _validate_alias_loads(instance.params, selected)
        object.__setattr__(self, "instance", instance)
        object.__setattr__(self, "loads", MappingProxyType(selected))

    @property
    def sources(self) -> dict[str, HostParam]:
        """Effective source backing for every parameter loaded by this plan."""
        return {name: load.source for name, load in self.loads.items()}

    def select_parameters(
        self,
        names: Iterable[str],
    ) -> HostModuleLoadPlan:
        """Return a plan containing only the requested active parameters."""
        return HostModuleLoadPlan(
            self.instance,
            _select_known_names(self.loads, names),
        )

    def allocate_target(
        self,
        device: torch.device,
        *,
        buffer_names: Iterable[str] | None = None,
    ) -> HostModuleTarget:
        """Allocate target storage from this plan's effective sources."""
        _validate_cuda_device(device)
        params = self.sources
        buffers = _select_known_names(self.instance.buffers, buffer_names)
        return HostModuleTarget(
            param_targets=_allocate_param_targets(
                params,
                _items_for_names(self.instance.params, params),
                device,
            ),
            buffer_targets=_allocate_buffer_targets(buffers, device),
        )

    def refill_param_target(
        self,
        name: str,
        target: HostParamTarget,
        *,
        non_blocking: bool = True,
    ) -> None:
        """Refill one target and immediately apply its planned update."""
        if name not in self.instance.params:
            raise ValueError(
                f"param name {name!r} is not owned by this HostModuleInstance"
            )
        load = self.loads.get(name)
        if load is None:
            return
        _load_param_target(load, target, non_blocking=non_blocking)

    def load_to_target(
        self,
        target: HostModuleTarget,
        *,
        non_blocking: bool = False,
    ) -> None:
        """Load this plan into ``target`` and install it on the module."""
        self.copy_to_target(target, non_blocking=non_blocking)
        self.instance.install_target(target)

    def copy_to_target(
        self,
        target: HostModuleTarget,
        *,
        non_blocking: bool = False,
    ) -> None:
        """Copy sources, apply updates, and leave module registries untouched."""
        instance = self.instance
        _validate_target_names_known(instance.params, instance.buffers, target)
        loads = _items_for_names(self.loads, target.param_targets)
        buffers = _items_for_names(instance.buffers, target.buffer_targets)
        _load_params_to_target(
            loads,
            target.param_targets,
            non_blocking=non_blocking,
        )
        _copy_buffers_to_target(
            buffers,
            target.buffer_targets,
            non_blocking=non_blocking,
        )


def _capture_params(
    params: Mapping[str, nn.Parameter],
) -> dict[str, HostParam]:
    host_by_name: dict[str, HostParam] = {}
    for names in group_names(
        params.keys(),
        lambda name: param_tensor_id(params[name]),
    ):
        _validate_param_storage_group_requires_grad(names, params)
        host = HostParam(params[names[0]])
        _validate_param_storage_group_tieable(names, host)
        for name in names:
            host_by_name[name] = host
    return host_by_name


def _validate_param_storage_group_tieable(names: Sequence[str], host: HostParam) -> None:
    """Reject tied weights whose adapter migrates wrapper state on forward.

    A migrate-state adapter (bitsandbytes int8) shares one reconstructed
    wrapper across all tied leaves, but the first module's forward migrates
    its quant state onto that module and nulls it on the wrapper — so a
    second tied module computes against missing state (garbage or a crash).
    The per-load rearm cannot fix this: it fires once per load, not once per
    consuming module. ``_needs_rearm`` flags exactly these adapters.
    """
    if len(names) > 1 and host._needs_rearm:
        raise NotImplementedError(
            f"Tied weights {sorted(names)!r} use an adapter whose quant state "
            f"migrates onto the owning module on first forward "
            f"({type(host.adapter).__name__}); one shared wrapper cannot "
            "serve multiple tied modules. Untie these weights, or keep them "
            "resident instead of offloading."
        )


def _capture_buffers(
    buffers: Mapping[str, torch.Tensor],
) -> dict[str, HostBuffer]:
    host_by_name: dict[str, HostBuffer] = {}
    for names in group_names(
        buffers.keys(),
        lambda name: buffer_tensor_id(buffers[name]),
    ):
        host = HostBuffer.clone(
            buffers[names[0]],
        )
        for name in names:
            host_by_name[name] = host
    return host_by_name


def _select_known_names[NamedT](
    items: Mapping[str, NamedT],
    names: Iterable[str] | None,
) -> dict[str, NamedT]:
    if names is None:
        return dict(items)

    included = set(names)
    missing = sorted(included - set(items))
    if missing:
        raise ValueError(f"Cannot select unknown names: {_format_names(missing)}.")
    return {name: value for name, value in items.items() if name in included}


def _items_for_names[NamedT](
    items: Mapping[str, NamedT],
    names: Iterable[str],
) -> dict[str, NamedT]:
    included = set(names)
    return {name: value for name, value in items.items() if name in included}


def _validate_module_matches(
    host_params: Mapping[str, HostParam],
    host_buffers: Mapping[str, HostBuffer],
    module: nn.Module,
) -> None:
    """Validate that ``module`` is a structurally compatible bind target.

    Compares bind layouts, not full target layouts: binding replaces every
    managed tensor with the host backing storage, so placeholder fields
    the bind overwrites (dtype, for plain tensors) are not required to
    match — a config-built meta skeleton binds against bytes captured from
    natively loaded weights.
    """
    params = _named_parameters(module)
    buffers = _named_buffers(module)

    _validate_names_present(host_params, host_buffers, params, buffers)

    for name, host in host_params.items():
        param = params[name]
        if param.requires_grad != host.requires_grad:
            raise ValueError(
                f"Param {name!r} requires_grad mismatch: store has "
                f"{host.requires_grad}, module has {param.requires_grad}."
            )
        if host.is_meta:
            representation = param_representation(param)
            host_representation = cast(torch.Tensor, host.host_state)
            if (
                type(representation) is not torch.Tensor
                or not representation.is_meta
                or representation.layout is not torch.strided
            ):
                raise ValueError(
                    f"Param {name!r} meta layout mismatch: store requires a "
                    "plain strided meta tensor, module has "
                    f"type={type(representation).__name__}, "
                    f"device={representation.device}, layout={representation.layout}."
                )
            if (
                tuple(representation.shape) != tuple(host.shape)
                or representation.dtype is not host.compute_dtype
                or representation.stride() != host_representation.stride()
                or representation.storage_offset()
                != host_representation.storage_offset()
            ):
                raise ValueError(
                    f"Param {name!r} meta layout mismatch: store has "
                    f"shape={tuple(host.shape)}, dtype={host.compute_dtype}, "
                    f"stride={host_representation.stride()}, "
                    f"storage_offset={host_representation.storage_offset()}, "
                    f"module has type={type(representation).__name__}, "
                    f"device={representation.device}, shape={tuple(representation.shape)}, "
                    f"dtype={representation.dtype}, stride={representation.stride()}, "
                    f"storage_offset={representation.storage_offset()}."
                )
            continue
        layout = HostParam.bind_layout_for(param)
        if layout != host.bind_layout:
            raise ValueError(
                f"Param {name!r} layout mismatch: store has {host.bind_layout!r}, module has {layout!r}."
            )

    for name, host in host_buffers.items():
        layout = HostBuffer.bind_layout_for(buffers[name])
        if layout != HostBuffer.bind_layout_for(host.tensor):
            raise ValueError(
                f"Buffer {name!r} layout mismatch: store has "
                f"{HostBuffer.bind_layout_for(host.tensor)!r}, module has {layout!r}."
            )


def _validate_target_has_trainable_params(
    params: Mapping[str, HostParam],
    target: HostModuleTarget,
) -> None:
    trainable_params = _trainable_params(params)
    expected_names = set(trainable_params)
    actual_names = set(target.param_targets)
    missing = sorted(expected_names - actual_names)
    if missing:
        raise ValueError(f"HostModuleTarget trainable param target names mismatch: missing {_format_names(missing)}.")


def _validate_target_names_known(
    params: Mapping[str, HostParam],
    buffers: Mapping[str, HostBuffer],
    target: HostModuleTarget,
) -> None:
    extra_params = sorted(set(target.param_targets) - set(params))
    extra_buffers = sorted(set(target.buffer_targets) - set(buffers))
    if not extra_params and not extra_buffers:
        return

    details = []
    if extra_params:
        details.append(f"params {_format_names(extra_params)}")
    if extra_buffers:
        details.append(f"buffers {_format_names(extra_buffers)}")
    raise ValueError(f"HostModuleTarget contains entries outside the store: {'; '.join(details)}.")


def _validate_param_storage_group_requires_grad(
    names: Iterable[str],
    params: Mapping[str, nn.Parameter],
) -> None:
    names = list(names)
    requires_grad = {params[name].requires_grad for name in names}
    if len(requires_grad) <= 1:
        return
    raise ValueError(f"HostModuleStore cannot group params with mixed requires_grad: {_format_names(names)}.")


def _validate_trainable_param_data_swaps(
    params: Mapping[str, HostParam],
) -> None:
    seen: set[int] = set()
    for name, host in params.items():
        if not host.requires_grad:
            continue
        key = id(host)
        if key in seen:
            continue
        seen.add(key)
        try:
            host.validate_parameter_data_swap_target()
        except NotImplementedError as exc:
            raise NotImplementedError(f"Trainable param {name!r} cannot use Parameter.data swap: {exc}") from exc


def _validate_names_present(
    host_params: Mapping[str, HostParam],
    host_buffers: Mapping[str, HostBuffer],
    params: Mapping[str, nn.Parameter],
    buffers: Mapping[str, torch.Tensor],
) -> None:
    missing_params = sorted(set(host_params) - set(params))
    missing_buffers = sorted(set(host_buffers) - set(buffers))
    if not missing_params and not missing_buffers:
        return

    details = []
    if missing_params:
        details.append(f"params {_format_names(missing_params)}")
    if missing_buffers:
        details.append(f"buffers {_format_names(missing_buffers)}")
    raise ValueError(f"Module is missing host names: {'; '.join(details)}.")


def _same_load_override(
    left: ParameterOverride,
    right: ParameterOverride,
) -> bool:
    return left.source is right.source and left.update is right.update


def _resolve_parameter_load(
    name: str,
    base: HostParam,
    override: ParameterOverride | None,
) -> ParameterLoad | None:
    if override is None:
        return None if base.is_meta else ParameterLoad(base)

    source = override.source
    if source is None:
        if base.is_meta:
            raise ValueError(
                f"Meta parameter {name!r} requires a physical replacement source."
            )
        source = base
    elif not isinstance(source, HostParam):
        raise ValueError(
            f"Parameter load source for {name!r} must be a HostParam; "
            f"got {type(source).__name__}."
        )
    if source.is_meta:
        raise ValueError(
            f"Parameter load source for {name!r} must own physical storage."
        )
    if source.logical_shape != base.logical_shape:
        raise ValueError(
            f"Parameter load source shape mismatch for {name!r}: source has "
            f"{source.logical_shape}, model slot has {base.logical_shape}."
        )
    if source.requires_grad != base.requires_grad:
        raise ValueError(
            f"Parameter load source requires_grad mismatch for {name!r}: "
            f"source has {source.requires_grad}, model slot has {base.requires_grad}."
        )
    return ParameterLoad(source, override.update)


def _validate_alias_loads(
    identities: Mapping[str, HostParam],
    loads: Mapping[str, ParameterLoad],
) -> None:
    by_identity: dict[int, ParameterLoad] = {}
    for name, load in loads.items():
        key = id(identities[name])
        previous = by_identity.get(key)
        if previous is not None and (
            previous.source is not load.source
            or previous.update is not load.update
        ):
            raise ValueError(
                "Tied parameter aliases cannot resolve to different loads; "
                f"conflict at {name!r}."
            )
        by_identity[key] = load


def _allocate_param_targets(
    params: Mapping[str, HostParam],
    identities: Mapping[str, HostParam],
    device: torch.device,
) -> dict[str, HostParamTarget]:
    targets_by_host_id: dict[int, HostParamTarget] = {}
    targets_by_name: dict[str, HostParamTarget] = {}
    for name, host in params.items():
        key = id(identities[name])
        target = targets_by_host_id.get(key)
        if target is None:
            state = host.allocate_gpu_storage(device)
            target = HostParamTarget(
                _state=state,
                param=host.make_gpu_param(state),
                _backing=host,
            )
            targets_by_host_id[key] = target
        targets_by_name[name] = target
    return targets_by_name


def _validate_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError(f"HostModuleTarget requires a CUDA device; got {device}.")


def _trainable_params(
    params: Mapping[str, HostParam],
) -> dict[str, HostParam]:
    return {name: host for name, host in params.items() if host.requires_grad}


def _allocate_buffer_targets(
    buffers: Mapping[str, HostBuffer],
    device: torch.device,
) -> dict[str, HostBufferTarget]:
    targets_by_host_id: dict[int, HostBufferTarget] = {}
    targets_by_name: dict[str, HostBufferTarget] = {}
    for name, host in buffers.items():
        key = id(host)
        target = targets_by_host_id.get(key)
        if target is None:
            target = HostBufferTarget(
                tensor=torch.empty_like(host.tensor, device=device),
            )
            targets_by_host_id[key] = target
        targets_by_name[name] = target
    return targets_by_name


def _load_params_to_target(
    loads: Mapping[str, ParameterLoad],
    targets: Mapping[str, HostParamTarget],
    *,
    non_blocking: bool,
) -> None:
    copied: set[int] = set()
    for name, load in loads.items():
        target = targets[name]
        key = id(target)
        if key in copied:
            continue
        _load_param_target(
            load,
            target,
            non_blocking=non_blocking,
        )
        copied.add(key)


def _load_param_target(
    load: ParameterLoad,
    target: HostParamTarget,
    *,
    non_blocking: bool,
) -> None:
    """Copy one source and immediately apply its optional update."""
    source = load.source
    if source.target_layout != target._backing.target_layout:
        raise RuntimeError(
            "Parameter target layout does not match the active load source. "
            "Reallocate the target for the current activation plan."
        )
    source.copy_to_gpu(target._state, non_blocking=non_blocking)
    # Re-arm the reused wrapper at the freshly-loaded buffers (no-op unless
    # the adapter migrates state off the wrapper, e.g. bitsandbytes int8).
    source.rearm_after_load(target.param, target._state)
    if load.update is not None:
        load.update(target.param)


def _copy_buffers_to_target(
    buffers: Mapping[str, HostBuffer],
    targets: Mapping[str, HostBufferTarget],
    *,
    non_blocking: bool,
) -> None:
    copied: set[int] = set()
    for name, host in buffers.items():
        key = id(host)
        if key in copied:
            continue
        targets[name].tensor.copy_(host.tensor, non_blocking=non_blocking)
        copied.add(key)


def _copy_trainable_params_from_target(
    params: Mapping[str, HostParam],
    targets: Mapping[str, HostParamTarget],
    *,
    non_blocking: bool,
) -> None:
    copied: set[int] = set()
    for name, host in params.items():
        if not host.requires_grad:
            continue
        key = id(host)
        if key in copied:
            continue
        host.copy_to_cpu(targets[name]._state, non_blocking=non_blocking)
        copied.add(key)


def _install_host_params(
    module: nn.Module,
    params: Mapping[str, HostParam],
) -> None:
    # Build the materialized CPU params on demand, deduped by ``id(host)``
    # so tied names share one wrapper (preserving tied-weight behavior).
    # ``make_cpu_param`` is cheap/zero-copy for every adapter (a plain
    # wrapper / metadata reconstruction aliasing the host tensors), so
    # per-install construction is fine.
    materialized: dict[str, nn.Parameter] = {}
    by_host: dict[int, nn.Parameter] = {}
    for name, host in params.items():
        cpu_param = by_host.get(id(host))
        if cpu_param is None:
            cpu_param = host.make_cpu_param()
            by_host[id(host)] = cpu_param
        materialized[name] = cpu_param
    _set_params(module, materialized)


def _set_params(
    module: nn.Module,
    materialized_params: Mapping[str, nn.Parameter],
) -> None:
    # Both materialized sources carry the correct ``requires_grad`` (GPU
    # param via ``make_gpu_param``, CPU wrapper via ``make_cpu_param``), so
    # the swap-vs-replace decision reads off the materialized param itself.
    for name, materialized in materialized_params.items():
        parent, leaf = resolve_parent_leaf(module, name)
        if materialized.requires_grad:
            # Trainable: keep the user's wrapper, swap only ``.data``.
            _get_param(parent, leaf).data = materialized.data
        else:
            # Frozen: replace the registry entry outright.
            _set_param(parent, leaf, materialized)


def _install_host_buffers(
    module: nn.Module,
    buffers: Mapping[str, HostBuffer],
) -> None:
    _set_buffers(
        module,
        {name: host.tensor for name, host in buffers.items()},
    )


def _set_buffers(
    module: nn.Module,
    buffers: Mapping[str, torch.Tensor],
) -> None:
    for name, tensor in buffers.items():
        parent, leaf = resolve_parent_leaf(module, name)
        persistent = leaf not in parent._non_persistent_buffers_set
        parent.register_buffer(leaf, tensor, persistent=persistent)


def _named_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    return _unique_name_dict(module.named_parameters(remove_duplicate=False))


def _named_buffers(module: nn.Module) -> dict[str, torch.Tensor]:
    return _unique_name_dict(module.named_buffers(remove_duplicate=False))


def _unique_name_dict[NamedT](
    items: Iterable[tuple[str, NamedT]],
) -> dict[str, NamedT]:
    values: dict[str, NamedT] = {}
    for name, value in items:
        if name in values:
            raise ValueError(f"Module yielded duplicate name {name!r}.")
        values[name] = value
    return values


def _get_param(parent: nn.Module, leaf: str) -> nn.Parameter:
    param = parent._parameters.get(leaf)
    if param is None:
        raise RuntimeError(f"Parameter {leaf!r} is unexpectedly missing.")
    return param


def _set_param(parent: nn.Module, leaf: str, param: nn.Parameter) -> None:
    if leaf not in parent._parameters:
        raise RuntimeError(f"Parameter {leaf!r} is unexpectedly missing.")
    parent._parameters[leaf] = param


def _unique_cache_bytes(
    items: Mapping[str, HostParam] | Mapping[str, HostBuffer],
) -> int:
    total = 0
    seen: set[int] = set()
    for value in items.values():
        key = id(value)
        if key in seen:
            continue
        seen.add(key)
        total += value.cache_bytes
    return total


def _format_names(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in names)


__all__ = [
    "HostBufferTarget",
    "HostModuleInstance",
    "HostModuleLoadPlan",
    "HostModuleStore",
    "HostModuleTarget",
    "HostParamTarget",
    "ParameterLoad",
    "ParameterOverride",
    "ParameterUpdate",
]
