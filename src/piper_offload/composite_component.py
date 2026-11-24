"""Internal composition of resident, transient, and block components."""

import contextlib
import threading
import weakref
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property, partial

import torch
from torch import nn

from .block_component import BlockComponent, _block_is_empty, _resolve_blocks
from .host_component import HostComponent
from .host_module import HostModuleInstance, HostModuleStore, ParameterOverride
from .host_param import HostParam
from .module_names import buffer_names, parameter_names, resolve_parent_leaf
from .offload_config import OffloadConfig
from .rolling_runtime import RollingSlotSpec, validate_rolling_block_specs
from .tensor_adapter_registry import param_tensor_id

type _OwnedEntry = tuple[weakref.ref[nn.Module], weakref.ref["_ShellClaim"]]

_OWNERSHIP_LOCK = threading.RLock()
_OWNED_MODULES: dict[int, _OwnedEntry] = {}


def _iter_modules_by_id(root: nn.Module) -> list[nn.Module]:
    """Return ``root`` and registered descendants deduplicated by object identity."""
    modules: list[nn.Module] = []
    seen_ids: set[int] = set()
    stack: list[nn.Module] = [root]
    while stack:
        module = stack.pop()
        mod_id = id(module)
        if mod_id in seen_ids:
            continue
        seen_ids.add(mod_id)
        modules.append(module)
        for child in reversed(list(module._modules.values())):
            if child is not None and id(child) not in seen_ids:
                stack.append(child)
    return modules


def _cleanup_dead_module(
    ref: weakref.ref[nn.Module],
    *,
    mod_id: int,
) -> None:
    with _OWNERSHIP_LOCK:
        current = _OWNED_MODULES.get(mod_id)
        if current is not None and current[0] is ref:
            del _OWNED_MODULES[mod_id]


def _release_entries(entries: list[tuple[int, _OwnedEntry]]) -> None:
    with _OWNERSHIP_LOCK:
        for mod_id, entry in entries:
            if _OWNED_MODULES.get(mod_id) is entry:
                del _OWNED_MODULES[mod_id]
        entries.clear()


class _ShellClaim:
    """Exclusive live ownership reservation over a module subtree."""

    def __init__(self, entries: list[tuple[int, _OwnedEntry]]) -> None:
        self._entries = entries
        finalizer = weakref.finalize(self, _release_entries, entries)
        finalizer.atexit = False
        self._finalizer = finalizer

    def release(self) -> None:
        self._finalizer()


def claim_shell(model: nn.Module) -> _ShellClaim:
    """Atomically claim exclusive live ownership across ``model``'s subtree."""
    with _OWNERSHIP_LOCK:
        modules = _iter_modules_by_id(model)
        for submodule in modules:
            existing = _OWNED_MODULES.get(id(submodule))
            if existing is None:
                continue
            mod_ref, claim_ref = existing
            if mod_ref() is submodule and claim_ref() is not None:
                raise ValueError(
                    "Model (or a submodule) is already bound to a live ModelOffloader "
                    "(or in use by capture/bind). Deactivate and drop the existing "
                    "offloader first, or bind a fresh shell."
                )

        entries: list[tuple[int, _OwnedEntry]] = []
        claim = _ShellClaim(entries)
        claim_ref = weakref.ref(claim)
        for submodule in modules:
            mod_id = id(submodule)
            mod_ref = weakref.ref(
                submodule,
                partial(_cleanup_dead_module, mod_id=mod_id),
            )
            entry: _OwnedEntry = (mod_ref, claim_ref)
            _OWNED_MODULES[mod_id] = entry
            entries.append((mod_id, entry))
        return claim


@contextlib.contextmanager
def reserve_shell(model: nn.Module) -> Iterator[None]:
    """Reserve exclusive ownership across ``model`` for one scoped operation."""
    claim = claim_shell(model)
    try:
        yield
    finally:
        claim.release()


class CompositeComponent:
    """Resident, transient-path, and block components."""

    def __init__(
        self,
        *,
        resident: HostComponent | None,
        blocks: Sequence[BlockComponent],
        transient: Sequence[tuple[str, HostComponent]] = (),
        transient_blocks: Sequence[BlockComponent] = (),
    ) -> None:
        self._resident = resident
        self._transient = tuple(transient)
        self._blocks = tuple(blocks)
        self._transient_blocks = tuple(transient_blocks)
        self._active = False

    @property
    def resident(self) -> HostComponent | None:
        return self._resident

    @property
    def blocks(self) -> tuple[BlockComponent, ...]:
        return self._blocks

    @property
    def transient(self) -> tuple[tuple[str, HostComponent], ...]:
        return self._transient

    @property
    def transient_blocks(self) -> tuple[BlockComponent, ...]:
        return self._transient_blocks

    def _components(self) -> Iterator[HostComponent | BlockComponent]:
        if self._resident is not None:
            yield self._resident
        for _path, component in self._transient:
            yield component
        yield from self._blocks

    @cached_property
    def param_names(self) -> frozenset[str]:
        return frozenset(name for component in self._components() for name in component.param_names)

    @property
    def buffer_names(self) -> frozenset[str]:
        return frozenset(name for component in self._components() for name in component.buffer_names)

    @property
    def failed(self) -> bool:
        """Whether any component still holds device work from a prior session."""
        return any(component.failed for component in self._components())

    def activate(
        self,
        device: torch.device,
        *,
        compile_blocks: bool = True,
        parameter_overrides: Mapping[str, ParameterOverride] | None = None,
    ) -> None:
        """Activate every component in order.

        A failure leaves the components as far as they got; the caller's
        :meth:`deactivate` releases them, as it releases any ended session.
        """
        if self._active:
            raise RuntimeError("CompositeComponent.activate() called while already active; deactivate() first.")

        overrides = {} if parameter_overrides is None else dict(parameter_overrides)
        if overrides:
            unknown = sorted(overrides.keys() - self.param_names)
            if unknown:
                raise ValueError(
                    f"Parameter overrides contain unmanaged names: {unknown!r}."
                )

        self._active = True
        for component in self._components():
            component.activate(
                device,
                compile_blocks=compile_blocks,
                parameter_overrides=(
                    {
                        name: override
                        for name, override in overrides.items()
                        if name in component.param_names
                    }
                    if overrides
                    else None
                ),
            )

    def deactivate(self) -> None:
        """Deactivate every component in reverse order, continuing past failures.

        Component deactivation is idempotent, safe before activation, and
        retries whatever an earlier failed release kept, so this both ends
        a session and finishes releasing a failed one.
        """
        self._active = False
        with contextlib.ExitStack() as stack:
            for component in self._components():
                stack.callback(component.deactivate)


type _SlotKey = tuple[int, str]
# Each block slot records: (block_local_name, store_key, slot_key)
type _BlockSlot = tuple[str, str, _SlotKey]


@dataclass(frozen=True, slots=True)
class _BlockPlan:
    module: nn.Module
    block_index: int
    params: tuple[_BlockSlot, ...]
    buffers: tuple[_BlockSlot, ...]


@dataclass(frozen=True, slots=True)
class _BlockGroupPlan:
    path: str
    blocks: tuple[_BlockPlan, ...]
    wraparound: bool


@dataclass(frozen=True, slots=True)
class _ModelPartition:
    resident: tuple[tuple[str, ...], tuple[str, ...]] | None
    transient: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]
    block_groups: tuple[_BlockGroupPlan, ...]
    param_slot_names: Mapping[_SlotKey, list[str]]
    buffer_slot_names: Mapping[_SlotKey, list[str]]


@dataclass(slots=True)
class _SlotIndex:
    ordered_names: list[str]
    slot_names: dict[_SlotKey, list[str]]
    managed: set[str] | None
    owned_slots: set[_SlotKey] = field(default_factory=set)
    managed_cache: dict[tuple[int, str], bool] = field(default_factory=dict)


def _collect_slot_index(
    model: nn.Module,
    entries: Iterable[tuple[str, object]],
    allowed: set[str] | None,
) -> _SlotIndex:
    ordered_names: list[str] = []
    seen_names: set[str] = set()
    slot_names: dict[_SlotKey, list[str]] = {}
    for name, _ in entries:
        if name in seen_names:
            continue
        if allowed is not None and name not in allowed:
            continue
        seen_names.add(name)
        ordered_names.append(name)
        parent, leaf = resolve_parent_leaf(model, name)
        slot_names.setdefault((id(parent), leaf), []).append(name)
    return _SlotIndex(
        ordered_names=ordered_names,
        slot_names=slot_names,
        managed=allowed,
    )


def _collect_module_prefixes(root: nn.Module) -> dict[int, list[str]]:
    """Collect dotted registered paths from ``root`` to each reachable module."""
    prefixes: dict[int, list[str]] = {id(root): [""]}
    stack: list[tuple[nn.Module, str, frozenset[int]]] = [
        (root, "", frozenset((id(root),)))
    ]
    while stack:
        module, prefix, ancestor_ids = stack.pop()
        for child_name, child in module._modules.items():
            if child is None:
                continue
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            prefixes.setdefault(id(child), []).append(child_prefix)
            child_id = id(child)
            if child_id not in ancestor_ids:
                stack.append((child, child_prefix, ancestor_ids | {child_id}))
    return prefixes


def _resolve_block_slots(
    block: nn.Module,
    entries: Iterable[tuple[str, object]],
    slot_index: _SlotIndex,
    *,
    path: str,
    block_index: int,
    kind: str,
    get_prefixes: Callable[[], Mapping[int, list[str]]],
) -> list[_BlockSlot]:
    resolved: list[_BlockSlot] = []
    seen_locals: set[str] = set()
    block_id = id(block)
    allowed = slot_index.managed
    for local_name, _ in entries:
        if local_name in seen_locals:
            continue
        seen_locals.add(local_name)
        if allowed is not None and f"{path}.{block_index}.{local_name}" not in allowed:
            cache_key = (block_id, local_name)
            is_managed = slot_index.managed_cache.get(cache_key)
            if is_managed is None:
                prefixes = get_prefixes().get(block_id, ())
                is_managed = any(
                    (f"{p}.{local_name}" if p else local_name) in allowed
                    for p in prefixes
                )
                slot_index.managed_cache[cache_key] = is_managed
            if not is_managed:
                continue
        parent, leaf = resolve_parent_leaf(block, local_name)
        slot = (id(parent), leaf)
        names = slot_index.slot_names.get(slot)
        if names is None:
            if allowed is None:
                raise ValueError(
                    f"Block {block_index} at {path!r} contains {kind} {local_name!r} "
                    "that is not registered on the root model; register block modules "
                    "as submodules of the root model."
                )
            continue
        slot_index.owned_slots.add(slot)
        resolved.append((local_name, names[0], slot))
    return resolved


def _slot_of(model: nn.Module, name: str) -> _SlotKey:
    parent, leaf = resolve_parent_leaf(model, name)
    return (id(parent), leaf)


def _plan_block_group(
    model: nn.Module,
    path: str,
    *,
    wraparound: bool,
    param_index: _SlotIndex,
    buffer_index: _SlotIndex,
    get_prefixes: Callable[[], Mapping[int, list[str]]],
) -> _BlockGroupPlan:
    filter_unmanaged = param_index.managed is not None
    blocks = _resolve_blocks(model, path)
    if not wraparound and len({id(block) for block in blocks}) != len(blocks):
        raise ValueError(
            "transient_block_paths does not support aliased block modules; "
            f"{path!r} contains repeated module objects."
        )

    block_plans: list[_BlockPlan] = []
    unmanaged: list[int] = []
    expected_param_set: set[str] = set()
    expected_buffer_set: set[str] = set()

    for idx, block in enumerate(blocks):
        block_params = _resolve_block_slots(
            block,
            block.named_parameters(remove_duplicate=False),
            param_index,
            path=path,
            block_index=idx,
            kind="parameter",
            get_prefixes=get_prefixes,
        )
        block_buffers = _resolve_block_slots(
            block,
            block.named_buffers(remove_duplicate=False),
            buffer_index,
            path=path,
            block_index=idx,
            kind="buffer",
            get_prefixes=get_prefixes,
        )
        if not block_params and not block_buffers:
            if filter_unmanaged and not _block_is_empty(block):
                unmanaged.append(idx)
            continue

        param_set = {local for local, _store_key, _slot in block_params}
        buffer_set = {local for local, _store_key, _slot in block_buffers}
        if block_plans:
            first_idx = block_plans[0].block_index
            if param_set != expected_param_set:
                difference = sorted(param_set ^ expected_param_set)
                raise ValueError(
                    f"Block {idx} parameter names differ from block {first_idx}: {difference}."
                )
            if buffer_set != expected_buffer_set:
                difference = sorted(buffer_set ^ expected_buffer_set)
                raise ValueError(
                    f"Block {idx} buffer names differ from block {first_idx}: {difference}."
                )
        else:
            expected_param_set = param_set
            expected_buffer_set = buffer_set

        block_plans.append(
            _BlockPlan(
                module=block,
                block_index=idx,
                params=tuple(block_params),
                buffers=tuple(block_buffers),
            )
        )

    if unmanaged:
        occupied = [plan.block_index for plan in block_plans]
        raise ValueError(
            f"ModelStore.bind() bind model has non-empty block(s) {unmanaged} "
            f"at {path!r} that this store does not occupy (it occupies {occupied}); "
            "those blocks would never be moved or streamed. Bind a model "
            "whose only non-empty blocks are the occupied ones."
        )
    if not block_plans:
        raise ValueError(
            f"Block path {path!r} has no streamable blocks (every block is structurally empty)."
        )

    return _BlockGroupPlan(
        path=path,
        blocks=tuple(block_plans),
        wraparound=wraparound,
    )


def _plan_model_groups(
    model: nn.Module,
    config: OffloadConfig,
    *,
    store: HostModuleStore | None = None,
) -> _ModelPartition:
    managed_params = None if store is None else set(store.params)
    managed_buffers = None if store is None else set(store.buffers)
    param_index = _collect_slot_index(
        model,
        model.named_parameters(remove_duplicate=False),
        managed_params,
    )
    buffer_index = _collect_slot_index(
        model,
        model.named_buffers(remove_duplicate=False),
        managed_buffers,
    )
    transient_modules = [(path, model.get_submodule(path)) for path in config.transient_paths]

    prefixes_cache: dict[int, list[str]] | None = None

    def get_prefixes() -> Mapping[int, list[str]]:
        nonlocal prefixes_cache
        if prefixes_cache is None:
            prefixes_cache = _collect_module_prefixes(model)
        return prefixes_cache

    block_groups = tuple(
        _plan_block_group(
            model,
            path,
            wraparound=wraparound,
            param_index=param_index,
            buffer_index=buffer_index,
            get_prefixes=get_prefixes,
        )
        for path, wraparound in (
            *((p, True) for p in config.block_paths),
            *((p, False) for p in config.transient_block_paths),
        )
    )

    remaining_params = [
        name for name in param_index.ordered_names if _slot_of(model, name) not in param_index.owned_slots
    ]
    remaining_buffers = [
        name for name in buffer_index.ordered_names if _slot_of(model, name) not in buffer_index.owned_slots
    ]
    remaining_param_set = set(remaining_params)
    remaining_buffer_set = set(remaining_buffers)

    transient: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for path, module in transient_modules:
        prefix = f"{path}." if path else ""
        selected_param_set = {f"{prefix}{name}" for name in parameter_names(module)} & remaining_param_set
        selected_buffer_set = {f"{prefix}{name}" for name in buffer_names(module)} & remaining_buffer_set
        if not selected_param_set and not selected_buffer_set:
            continue
        selected_params = tuple(n for n in remaining_params if n in selected_param_set)
        selected_buffers = tuple(n for n in remaining_buffers if n in selected_buffer_set)
        transient.append((path, selected_params, selected_buffers))
        remaining_param_set -= selected_param_set
        remaining_buffer_set -= selected_buffer_set
        remaining_params = [n for n in remaining_params if n in remaining_param_set]
        remaining_buffers = [n for n in remaining_buffers if n in remaining_buffer_set]

    resident = (
        (tuple(remaining_params), tuple(remaining_buffers))
        if remaining_params or remaining_buffers
        else None
    )
    if resident is None and not transient and not block_groups:
        raise ValueError("Offloading requires at least one parameter, registered buffer, or block to manage.")

    return _ModelPartition(
        resident=resident,
        transient=tuple(transient),
        block_groups=block_groups,
        param_slot_names=param_index.slot_names,
        buffer_slot_names=buffer_index.slot_names,
    )


def _validate_no_cross_group_ties(
    model: nn.Module,
    partition: _ModelPartition,
    param_key: Callable[[str], Hashable],
) -> None:
    param_owners: dict[Hashable, tuple[object, str]] = {}
    buffer_owners: dict[_SlotKey, tuple[object, str]] = {}

    def record_param(group_key: object, name: str) -> None:
        key = param_key(name)
        existing = param_owners.get(key)
        if existing is None:
            param_owners[key] = (group_key, name)
            return
        if existing[0] != group_key:
            raise ValueError(
                "Parameter storage shared across offload groups is unsupported: "
                f"{existing[1]!r} and {name!r}. Use whole-model offloading "
                "when cross-group ties must be preserved."
            )

    def record_buffer(group_key: object, name: str) -> None:
        slot = _slot_of(model, name)
        existing = buffer_owners.get(slot)
        if existing is None:
            buffer_owners[slot] = (group_key, name)
            return
        if existing[0] != group_key:
            raise ValueError(
                "Buffer module slot shared across offload groups is unsupported: "
                f"{existing[1]!r} and {name!r}. Use whole-model offloading "
                "when cross-group module aliases must be preserved."
            )

    if partition.resident is not None:
        for name in partition.resident[0]:
            record_param("resident", name)
        for name in partition.resident[1]:
            record_buffer("resident", name)
    for path, t_params, t_buffers in partition.transient:
        group_key = ("transient", path)
        for name in t_params:
            record_param(group_key, name)
        for name in t_buffers:
            record_buffer(group_key, name)
    for group_idx, group in enumerate(partition.block_groups):
        for bp in group.blocks:
            group_key = ("block", group_idx, id(bp.module))
            for _local, store_key, _all in bp.params:
                record_param(group_key, store_key)
            for _local, store_key, _all in bp.buffers:
                record_buffer(group_key, store_key)


def _validate_rolling_groups(
    partition: _ModelPartition,
    raw_params: Mapping[str, nn.Parameter],
) -> None:
    for group in partition.block_groups:
        block_ids = [id(bp.module) for bp in group.blocks]
        block_specs = []
        has_buffers = any(bp.buffers for bp in group.blocks)
        for bp in group.blocks:
            specs: dict[str, RollingSlotSpec] = {}
            for local, store_key, _all in bp.params:
                param = raw_params[store_key]
                adapter_type, shape, target_layout = HostParam.rolling_slot_metadata_for(param)
                specs[local] = RollingSlotSpec(
                    identity=param_tensor_id(param),
                    adapter_type=adapter_type,
                    shape=shape,
                    target_layout=target_layout,
                )
            block_specs.append(specs)
        validate_rolling_block_specs(
            block_ids,
            block_specs,
            has_buffers=has_buffers,
        )


def _build_host_component(
    store: HostModuleStore,
    module: nn.Module,
    param_names_seq: Sequence[str],
    buffer_names_seq: Sequence[str],
    instances_to_install: list[HostModuleInstance],
) -> HostComponent:
    instance = HostModuleInstance(
        module=module,
        params={name: store.params[name] for name in param_names_seq},
        buffers={name: store.buffers[name] for name in buffer_names_seq},
    )
    instances_to_install.append(instance)
    return HostComponent(instance)


def _build_group_name_indices(
    group: _BlockGroupPlan,
    param_slot_names: Mapping[_SlotKey, list[str]],
    buffer_slot_names: Mapping[_SlotKey, list[str]],
) -> tuple[dict[str, tuple[int, str]], frozenset[str]]:
    param_index: dict[str, tuple[int, str]] = {}
    slot_local_reps: dict[_SlotKey, dict[str, tuple[int, str]]] = {}
    slot_name_sets: dict[_SlotKey, set[str]] = {}

    for compact_idx, bp in enumerate(group.blocks):
        for local, _store_key, slot in bp.params:
            ref = (compact_idx, local)
            slot_local_reps.setdefault(slot, {}).setdefault(local, ref)
            names_set = slot_name_sets.get(slot)
            if names_set is None:
                names_set = set(param_slot_names[slot])
                slot_name_sets[slot] = names_set
            exact = f"{group.path}.{bp.block_index}.{local}"
            if exact in names_set:
                param_index[exact] = ref

    for slot, local_reps in slot_local_reps.items():
        reps = tuple(local_reps.values())
        for reg_name in param_slot_names[slot]:
            if reg_name in param_index:
                continue
            matched = next(
                (
                    (compact_idx, local)
                    for compact_idx, local in reps
                    if reg_name == local or reg_name.endswith(f".{local}")
                ),
                reps[0],
            )
            param_index[reg_name] = matched

    unique_buf_slots: set[_SlotKey] = {
        slot for bp in group.blocks for _local, _store_key, slot in bp.buffers
    }
    group_buffers = {
        reg_name
        for slot in unique_buf_slots
        for reg_name in buffer_slot_names[slot]
    }
    return param_index, frozenset(group_buffers)


def _assemble_composite(
    store: HostModuleStore,
    model: nn.Module,
    partition: _ModelPartition,
    config: OffloadConfig,
    *,
    install: bool,
) -> CompositeComponent:
    instances_to_install: list[HostModuleInstance] = []
    resident = (
        _build_host_component(
            store, model, partition.resident[0], partition.resident[1], instances_to_install,
        )
        if partition.resident is not None
        else None
    )
    transient = tuple(
        (
            path,
            _build_host_component(store, model, t_params, t_buffers, instances_to_install),
        )
        for path, t_params, t_buffers in partition.transient
    )

    persistent: list[BlockComponent] = []
    transient_blocks: list[BlockComponent] = []
    for group in partition.block_groups:
        instances: list[HostModuleInstance] = []
        for bp in group.blocks:
            instance = HostModuleInstance(
                module=bp.module,
                params={local: store.params[store_key] for local, store_key, _slot in bp.params},
                buffers={local: store.buffers[store_key] for local, store_key, _slot in bp.buffers},
            )
            instances.append(instance)
            instances_to_install.append(instance)
        param_index, group_buffers = _build_group_name_indices(
            group,
            partition.param_slot_names,
            partition.buffer_slot_names,
        )
        component = BlockComponent(
            instances,
            name=group.path,
            block_indices=[bp.block_index for bp in group.blocks],
            param_name_index=param_index,
            buffer_names=group_buffers,
            block_compile=config.block_compile,
            wraparound=group.wraparound,
            block_mode=config.block_mode,
        )
        if group.wraparound:
            persistent.append(component)
        else:
            transient_blocks.append(component)

    composite = CompositeComponent(
        resident=resident,
        transient=transient,
        blocks=(*persistent, *transient_blocks),
        transient_blocks=tuple(transient_blocks),
    )
    if install:
        for instance in instances_to_install:
            instance.install_host()
    return composite


def _bind_module_store(
    store: HostModuleStore,
    model: nn.Module,
    config: OffloadConfig,
) -> tuple[CompositeComponent, _ShellClaim]:
    claim = claim_shell(model)
    try:
        store._validate_bind(model)
        partition = _plan_model_groups(model, config, store=store)
        _validate_no_cross_group_ties(model, partition, lambda name: id(store.params[name]))
        composite = _assemble_composite(store, model, partition, config, install=True)
    except BaseException:
        claim.release()
        raise
    return composite, claim


def _validate_capture_inputs(
    model: nn.Module,
    partition: _ModelPartition,
    config: OffloadConfig,
) -> None:
    raw_params = dict(model.named_parameters(remove_duplicate=False))
    trainable = sorted(name for name, param in raw_params.items() if param.requires_grad)
    if trainable:
        formatted = ", ".join(repr(name) for name in trainable)
        raise ValueError(
            "Piper Offload is inference-only and requires frozen parameters "
            f"(requires_grad=False); found trainable parameter(s): {formatted}."
        )
    _validate_no_cross_group_ties(model, partition, lambda name: param_tensor_id(raw_params[name]))
    if config.block_mode == "rolling":
        _validate_rolling_groups(partition, raw_params)


def _capture_and_bind(
    model: nn.Module,
    config: OffloadConfig,
) -> tuple[HostModuleStore, CompositeComponent, _ShellClaim]:
    claim = claim_shell(model)
    try:
        partition = _plan_model_groups(model, config, store=None)
        _validate_capture_inputs(model, partition, config)
        store, _ = HostModuleStore.capture(model)
        composite = _assemble_composite(store, model, partition, config, install=False)
    except BaseException:
        claim.release()
        raise
    return store, composite, claim


__all__ = [
    "CompositeComponent",
    "_ShellClaim",
    "_bind_module_store",
    "_capture_and_bind",
    "claim_shell",
    "reserve_shell",
]
