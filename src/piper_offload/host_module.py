"""Name-based host module store and instance primitives.

This module supports sharing one CPU cache across multiple
concrete model instances. Names are the durable relationship between a
store and an instance.
"""

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from types import MappingProxyType
from typing import NoReturn, Self, cast

import torch
from torch import nn

from ._dtensor import is_dtensor, mesh_signature, placements_key, require_dtensor
from .dtensor_adapter import DTensorAdapter, require_initialized_mesh
from .host_buffer import HostBuffer
from .host_param import HostParam
from .module_names import group_names, resolve_parent_leaf
from .parameter_transform import ParameterTransform
from .tensor_adapter_registry import (
    buffer_tensor_id,
    param_representation,
    param_tensor_id,
)
from .tensor_adapters import transfer_


@dataclass(frozen=True, slots=True)
class ParameterOverride:
    """Activation-scoped replacement source and/or in-place update."""

    source: HostParam | None = None
    update: ParameterTransform | None = None

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
    update: ParameterTransform | None = None


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
        """Capture host backing and install it on the source module."""
        store, _ = cls.capture(
            module, include_param_names=include_param_names, include_buffer_names=include_buffer_names,
        )
        return store

    @classmethod
    def capture(
        cls,
        module: nn.Module,
        *,
        include_param_names: Iterable[str] | None = None,
        include_buffer_names: Iterable[str] | None = None,
    ) -> tuple[Self, HostModuleInstance]:
        """Capture host backing incrementally per alias group and install it."""
        param_groups, buffer_groups = _collect_capture_groups(
            module,
            include_param_names=include_param_names,
            include_buffer_names=include_buffer_names,
        )
        host_params = _capture_install_param_groups(module, param_groups)
        host_buffers = _capture_install_buffer_groups(module, buffer_groups)
        store = cls(params=host_params, buffers=host_buffers)
        instance = HostModuleInstance(
            module=module,
            params=store.params,
            buffers=store.buffers,
        )
        return store, instance

    @property
    def cache_bytes(self) -> int:
        return _unique_cache_bytes(self.params) + _unique_cache_bytes(self.buffers)

    def _resolve_bind(self, module: nn.Module) -> HostModuleStore:
        """Validate ``module`` and project unsharded params onto DTensor targets."""
        resolved_params = _resolve_module_bind(self.params, self.buffers, module)
        if resolved_params is None:
            return self
        return HostModuleStore(params=resolved_params, buffers=self.buffers)

    def bind(self, module: nn.Module) -> HostModuleInstance:
        """Validate a compatible module and install this store's backing on it."""
        return self._resolve_bind(module)._install(module)

    def _install(self, module: nn.Module) -> HostModuleInstance:
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
    this instance's host bytes. :meth:`resolve_load_plan` combines
    activation-scoped overrides with immutable model backing; the resulting plan
    owns target allocation and loading. :meth:`install_host` restores the host
    bytes onto :attr:`module`.
    """

    module: nn.Module
    params: Mapping[str, HostParam]
    buffers: Mapping[str, HostBuffer]

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
        install_parameters(
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

    def storage_tensors(self) -> Iterator[torch.Tensor]:
        """Host transfer sources without alias deduplication.

        Replacements supersede base sources; the pin manager deduplicates
        aliases and whole storage allocations.
        """
        for load in self.loads.values():
            yield from load.source.storage_tensors()
            if load.update is not None:
                yield from load.update.storage_tensors()
        for buffer in self.instance.buffers.values():
            yield from buffer.storage_tensors()

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


def _collect_capture_groups(
    module: nn.Module,
    *,
    include_param_names: Iterable[str] | None = None,
    include_buffer_names: Iterable[str] | None = None,
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    all_params = _named_parameters(module)
    params = _select_known_names(all_params, include_param_names)
    all_buffers = _named_buffers(module)
    buffers = _select_known_names(all_buffers, include_buffer_names)

    trainable = sorted(name for name, p in params.items() if p.requires_grad)
    if trainable:
        raise ValueError(
            "Piper Offload is inference-only and requires frozen parameters "
            f"(requires_grad=False); found trainable parameter(s): {_format_names(trainable)}."
        )

    param_groups = group_names(params.keys(), lambda name: param_tensor_id(params[name]))
    buffer_groups = group_names(buffers.keys(), lambda name: buffer_tensor_id(buffers[name]))
    return param_groups, buffer_groups


def _capture_one_param_group(module: nn.Module, names: tuple[str, ...]) -> HostParam:
    parent, leaf = resolve_parent_leaf(module, names[0])
    source = parent._parameters[leaf]
    if source is None:
        raise RuntimeError(f"Parameter {names[0]!r} is unexpectedly missing.")
    del parent
    host = HostParam(source)
    del source
    cpu_param = host.make_cpu_param()
    for name in names:
        target_parent, target_leaf = resolve_parent_leaf(module, name)
        _set_param(target_parent, target_leaf, cpu_param)
    return host


def _capture_install_param_groups(
    module: nn.Module,
    groups: Sequence[tuple[str, ...]],
) -> dict[str, HostParam]:
    host_by_name: dict[str, HostParam] = {}
    for names in groups:
        host = _capture_one_param_group(module, names)
        for name in names:
            host_by_name[name] = host
    return host_by_name


def _capture_one_buffer_group(module: nn.Module, names: tuple[str, ...]) -> HostBuffer:
    parent, leaf = resolve_parent_leaf(module, names[0])
    source = parent._buffers[leaf]
    if source is None:
        raise RuntimeError(f"Buffer {names[0]!r} is unexpectedly missing.")
    del parent
    host = HostBuffer.capture(source)
    del source
    for name in names:
        target_parent, target_leaf = resolve_parent_leaf(module, name)
        persistent = target_leaf not in target_parent._non_persistent_buffers_set
        target_parent.register_buffer(target_leaf, host.tensor, persistent=persistent)
    return host


def _capture_install_buffer_groups(
    module: nn.Module,
    groups: Sequence[tuple[str, ...]],
) -> dict[str, HostBuffer]:
    host_by_name: dict[str, HostBuffer] = {}
    for names in groups:
        host = _capture_one_buffer_group(module, names)
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


def _record_target_slot[HostT](
    owners: dict[tuple[int, str], tuple[str, HostT]],
    module: nn.Module,
    name: str,
    host: HostT,
    kind: str,
) -> None:
    parent, leaf = resolve_parent_leaf(module, name)
    slot = (id(parent), leaf)
    existing = owners.get(slot)
    if existing is not None and existing[1] is not host:
        raise ValueError(
            f"{kind}s {existing[0]!r} and {name!r} share the same target "
            f"module slot but map to distinct store {kind.lower()}s."
        )
    owners[slot] = (name, host)


def _raise_param_bind_mismatch(
    name: str,
    host: HostParam,
    representation: torch.Tensor,
) -> NoReturn:
    if host.is_meta:
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
        raise ValueError(
            f"Param {name!r} meta layout mismatch: store has "
            f"shape={tuple(host.shape)}, dtype={host_representation.dtype}, "
            f"stride={host_representation.stride()}, "
            f"storage_offset={host_representation.storage_offset()}, "
            f"module has type={type(representation).__name__}, "
            f"device={representation.device}, shape={tuple(representation.shape)}, "
            f"dtype={representation.dtype}, stride={representation.stride()}, "
            f"storage_offset={representation.storage_offset()}."
        )
    raise ValueError(
        f"Param {name!r} layout mismatch: store has {host.target_layout!r}, "
        f"module has type={type(representation).__name__}, "
        f"shape={tuple(representation.shape)}, dtype={representation.dtype}, "
        f"device={representation.device}, layout={representation.layout}."
    )


def _resolve_projected_param(
    name: str,
    host: HostParam,
    param: nn.Parameter,
    representation: torch.Tensor,
    projected_by_key: dict[tuple[object, ...], HostParam],
) -> HostParam:
    dt = require_dtensor(representation)
    mesh = dt.device_mesh
    try:
        require_initialized_mesh(mesh)
    except ValueError as exc:
        raise ValueError(f"Param {name!r} layout mismatch: {exc}") from exc
    key = (
        id(host),
        tuple(dt.shape),
        mesh_signature(mesh),
        mesh.mesh_dim_names,
        tuple(id(mesh.get_group(dim)) for dim in range(mesh.ndim)),
        placements_key(dt.placements),
    )
    existing = projected_by_key.get(key)
    if existing is not None:
        if existing.accepts_bind_target(param):
            return existing
        raise ValueError(
            f"Param {name!r} layout mismatch: target does not match projected "
            f"DTensor layout {existing.target_layout!r}."
        )
    try:
        projected = HostParam.project_dtensor(host, param)
    except (NotImplementedError, TypeError, ValueError) as exc:
        raise ValueError(f"Param {name!r} layout mismatch: {exc}") from exc
    projected_by_key[key] = projected
    return projected


def _resolve_module_bind(
    host_params: Mapping[str, HostParam],
    host_buffers: Mapping[str, HostBuffer],
    module: nn.Module,
) -> dict[str, HostParam] | None:
    """Validate ``module`` and project unsharded host params onto DTensor targets.

    Compares bind compatibility via :meth:`HostParam.accepts_bind_target`
    (and :meth:`HostBuffer.bind_layout_for` for buffers): binding replaces
    every managed tensor with the host backing storage, so placeholder fields
    overwritten on bind (such as plain tensor dtype or a plain strided
    ``meta`` shape matching a frozen quantized store) do not reject a fresh
    skeleton. When a non-meta unsharded store param targets a ``DTensor``,
    projects a zero-copy rank-local view (deduplicated per host source and
    target distributed layout so tied parameters stay tied). Returns a new
    parameter mapping when any projection occurred, otherwise ``None``.
    """
    params = _named_parameters(module)
    buffers = _named_buffers(module)

    _validate_names_present(host_params, host_buffers, params, buffers)

    param_slots: dict[tuple[int, str], tuple[str, HostParam]] = {}
    resolved_params: dict[str, HostParam] = {}
    projected_by_key: dict[tuple[object, ...], HostParam] = {}
    projected = False

    for name, host in host_params.items():
        param = params[name]
        if param.requires_grad:
            raise ValueError(
                f"Param {name!r} must be frozen (requires_grad=False)."
            )
        if host.accepts_bind_target(param):
            resolved_host = host
        else:
            representation = param_representation(param)
            if (
                not host.is_meta
                and not isinstance(host.adapter, DTensorAdapter)
                and is_dtensor(representation)
            ):
                resolved_host = _resolve_projected_param(
                    name,
                    host,
                    param,
                    representation,
                    projected_by_key,
                )
                projected = True
            else:
                _raise_param_bind_mismatch(name, host, representation)
        _record_target_slot(param_slots, module, name, resolved_host, "Param")
        resolved_params[name] = resolved_host

    buffer_slots: dict[tuple[int, str], tuple[str, HostBuffer]] = {}
    for name, host in host_buffers.items():
        layout = HostBuffer.bind_layout_for(buffers[name])
        if layout != HostBuffer.bind_layout_for(host.tensor):
            raise ValueError(
                f"Buffer {name!r} layout mismatch: store has "
                f"{HostBuffer.bind_layout_for(host.tensor)!r}, module has {layout!r}."
            )
        _record_target_slot(buffer_slots, module, name, host, "Buffer")

    return resolved_params if projected else None


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
    if load.update is not None:
        load.update.apply_parameter(target.param)


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
        transfer_(targets[name].tensor, host.tensor, non_blocking=non_blocking)
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
    install_parameters(module, materialized)


def install_parameters(
    module: nn.Module,
    materialized_params: Mapping[str, nn.Parameter],
) -> None:
    """Install frozen parameter wrappers into ``module``'s registry."""
    for name, materialized in materialized_params.items():
        parent, leaf = resolve_parent_leaf(module, name)
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
]
