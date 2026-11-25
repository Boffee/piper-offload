"""Model topology partitioning, shell ownership, and store binding."""

from __future__ import annotations

import contextlib
import threading
import weakref
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import partial

from torch import nn

from .host_module import HostModuleInstance, HostModuleStore, ParameterOverride
from .module_names import resolve_parent_leaf, walk_attr_path
from .offload_config import OffloadConfig
from .residency_engine import ResidencyEngine
from .tensor_adapter_registry import param_tensor_id

type _OwnedEntry = tuple[weakref.ref[nn.Module], weakref.ref["_ShellClaim"]]
type _SlotKey = tuple[int, str]
type _UnitSlot = tuple[str, str, _SlotKey]

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


def _module_is_empty(module: nn.Module) -> bool:
    """Return whether ``module`` holds no parameters or buffers at any depth."""
    return next(module.parameters(), None) is None and next(module.buffers(), None) is None


@dataclass(slots=True)
class _SlotIndex:
    ordered_names: list[str]
    slot_names: dict[_SlotKey, list[str]]
    managed: set[str] | None
    owned_slots: set[_SlotKey] = field(default_factory=set)
    managed_cache: dict[tuple[int, str], bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _UnitPlan:
    unit_idx: int
    path: str
    module: nn.Module
    params: tuple[_UnitSlot, ...]
    buffers: tuple[_UnitSlot, ...]


@dataclass(frozen=True, slots=True)
class _PlannedPartition:
    resident: tuple[tuple[str, ...], tuple[str, ...]] | None
    units: tuple[_UnitPlan, ...]
    unit_paths: tuple[str, ...]
    param_slot_names: Mapping[_SlotKey, list[str]]
    buffer_slot_names: Mapping[_SlotKey, list[str]]


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


def _slot_of(model: nn.Module, name: str) -> _SlotKey:
    parent, leaf = resolve_parent_leaf(model, name)
    return (id(parent), leaf)


def _resolve_unit_slots(
    unit_module: nn.Module,
    entries: Iterable[tuple[str, object]],
    slot_index: _SlotIndex,
    *,
    unit_path: str,
    kind: str,
    get_prefixes: Callable[[], Mapping[int, list[str]]],
) -> list[_UnitSlot]:
    resolved: list[_UnitSlot] = []
    seen_locals: set[str] = set()
    mod_id = id(unit_module)
    allowed = slot_index.managed
    for local_name, _ in entries:
        if local_name in seen_locals:
            continue
        seen_locals.add(local_name)
        if allowed is not None and f"{unit_path}.{local_name}" not in allowed:
            cache_key = (mod_id, local_name)
            is_managed = slot_index.managed_cache.get(cache_key)
            if is_managed is None:
                prefixes = get_prefixes().get(mod_id, ())
                is_managed = any(
                    (f"{p}.{local_name}" if p else local_name) in allowed
                    for p in prefixes
                )
                slot_index.managed_cache[cache_key] = is_managed
            if not is_managed:
                continue
        parent, leaf = resolve_parent_leaf(unit_module, local_name)
        slot = (id(parent), leaf)
        names = slot_index.slot_names.get(slot)
        if names is None:
            if allowed is None:
                raise ValueError(
                    f"Unit {unit_path!r} contains {kind} {local_name!r} "
                    "that is not registered on the root model; register offloaded "
                    "modules as submodules of the root model."
                )
            continue
        resolved.append((local_name, names[0], slot))
    return resolved


def _resolve_offload_target(model: nn.Module, path: str) -> nn.Module:
    try:
        obj = walk_attr_path(model, path)
    except AttributeError as exc:
        raise ValueError(
            f"offload_paths entry {path!r} does not exist on {type(model).__name__}."
        ) from exc
    if obj is model:
        raise ValueError(
            f"offload_paths entry {path!r} cannot refer to the root model itself; "
            "leave offload_paths=() for whole-model resident mode or list submodules."
        )
    if not isinstance(obj, nn.Module):
        raise TypeError(
            f"Expected nn.Module or nn.ModuleList at {path!r}, got {type(obj).__name__}."
        )
    return obj


def _validate_no_path_prefix_overlap(paths: tuple[str, ...]) -> None:
    for i, p1 in enumerate(paths):
        for p2 in paths[i + 1 :]:
            if p1 == p2 or p2.startswith(f"{p1}.") or p1.startswith(f"{p2}."):
                raise ValueError(
                    f"Overlapping offload_paths entries: {p1!r} and {p2!r}."
                )


@dataclass(slots=True)
class _UnitAccumulator:
    param_index: _SlotIndex
    buffer_index: _SlotIndex
    filter_unmanaged: bool
    get_prefixes: Callable[[], Mapping[int, list[str]]]
    unit_plans: list[_UnitPlan] = field(default_factory=list)
    unit_paths: list[str] = field(default_factory=list)
    unit_idx_by_mod_id: dict[int, int] = field(default_factory=dict)
    owner_top_path_by_mod_id: dict[int, str] = field(default_factory=dict)

    def _validate_disjoint(
        self,
        candidate_path: str,
        unit_params: list[_UnitSlot],
        unit_buffers: list[_UnitSlot],
    ) -> None:
        if any(slot in self.param_index.owned_slots for _, _, slot in unit_params):
            raise ValueError(
                f"Overlapping offload_paths entry {candidate_path!r} shares "
                "parameter slots with another offloaded unit."
            )
        if any(slot in self.buffer_index.owned_slots for _, _, slot in unit_buffers):
            raise ValueError(
                f"Overlapping offload_paths entry {candidate_path!r} shares "
                "buffer slots with another offloaded unit."
            )

    def register_candidate(
        self,
        candidate_mod: nn.Module,
        candidate_path: str,
        container_path: str | None,
    ) -> tuple[bool, bool]:
        """Register one candidate unit; returns ``(added_or_aliased, unmanaged_nonempty)``."""
        mod_id = id(candidate_mod)
        existing_idx = self.unit_idx_by_mod_id.get(mod_id)
        if (
            existing_idx is not None
            and container_path is not None
            and self.owner_top_path_by_mod_id.get(mod_id) == container_path
        ):
            return True, False

        unit_params = _resolve_unit_slots(
            candidate_mod,
            candidate_mod.named_parameters(remove_duplicate=False),
            self.param_index,
            unit_path=candidate_path,
            kind="parameter",
            get_prefixes=self.get_prefixes,
        )
        unit_buffers = _resolve_unit_slots(
            candidate_mod,
            candidate_mod.named_buffers(remove_duplicate=False),
            self.buffer_index,
            unit_path=candidate_path,
            kind="buffer",
            get_prefixes=self.get_prefixes,
        )
        if not unit_params and not unit_buffers:
            unmanaged = self.filter_unmanaged and not _module_is_empty(candidate_mod)
            return False, unmanaged

        self._validate_disjoint(candidate_path, unit_params, unit_buffers)

        self.param_index.owned_slots.update(slot for _, _, slot in unit_params)
        self.buffer_index.owned_slots.update(slot for _, _, slot in unit_buffers)

        unit_idx = len(self.unit_plans)
        self.unit_idx_by_mod_id[mod_id] = unit_idx
        self.owner_top_path_by_mod_id[mod_id] = (
            container_path if container_path is not None else candidate_path
        )
        self.unit_plans.append(
            _UnitPlan(
                unit_idx=unit_idx,
                path=candidate_path,
                module=candidate_mod,
                params=tuple(unit_params),
                buffers=tuple(unit_buffers),
            )
        )
        self.unit_paths.append(candidate_path)
        return True, False


def _register_module_list_path(
    acc: _UnitAccumulator,
    path: str,
    target: nn.ModuleList,
) -> None:
    if len(target) == 0:
        raise ValueError(f"Block path {path!r} resolved to an empty nn.ModuleList.")
    added_in_list = 0
    occupied: list[str] = []
    unmanaged: list[str] = []
    for child_name, block in target._modules.items():
        if block is None:
            continue
        ok, is_unmanaged = acc.register_candidate(
            block,
            f"{path}.{child_name}",
            container_path=path,
        )
        if ok:
            added_in_list += 1
            occupied.append(child_name)
        elif is_unmanaged:
            unmanaged.append(child_name)
    if unmanaged:
        raise ValueError(
            f"ModelStore.bind() bind model has non-empty block(s) {unmanaged} "
            f"at {path!r} that this store does not occupy (it occupies {occupied}); "
            "those blocks would never be moved or streamed. Bind a model "
            "whose only non-empty blocks are the occupied ones."
        )
    if added_in_list == 0:
        raise ValueError(
            f"Block path {path!r} has no streamable blocks (every block is structurally empty)."
        )


def _plan_model_units(
    model: nn.Module,
    config: OffloadConfig,
    *,
    store: HostModuleStore | None = None,
) -> _PlannedPartition:
    paths = tuple(config.offload_paths)
    _validate_no_path_prefix_overlap(paths)

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

    prefixes_cache: dict[int, list[str]] | None = None

    def get_prefixes() -> Mapping[int, list[str]]:
        nonlocal prefixes_cache
        if prefixes_cache is None:
            prefixes_cache = _collect_module_prefixes(model)
        return prefixes_cache

    acc = _UnitAccumulator(
        param_index=param_index,
        buffer_index=buffer_index,
        filter_unmanaged=store is not None,
        get_prefixes=get_prefixes,
    )

    for path in paths:
        target = _resolve_offload_target(model, path)
        if isinstance(target, nn.ModuleList):
            _register_module_list_path(acc, path, target)
            continue
        ok, is_unmanaged = acc.register_candidate(target, path, container_path=None)
        if is_unmanaged:
            raise ValueError(
                f"ModelStore.bind() bind model has non-empty module at {path!r} "
                "that this store does not occupy; bind a compatible model store."
            )
        if not ok:
            raise ValueError(
                f"offload_paths entry {path!r} has no parameters or registered buffers to offload."
            )

    _validate_no_resident_descendant_aliases(model, acc, param_index, buffer_index)

    remaining_params = [
        name
        for name in param_index.ordered_names
        if _slot_of(model, name) not in param_index.owned_slots
    ]
    remaining_buffers = [
        name
        for name in buffer_index.ordered_names
        if _slot_of(model, name) not in buffer_index.owned_slots
    ]
    resident = (
        (tuple(remaining_params), tuple(remaining_buffers))
        if remaining_params or remaining_buffers
        else None
    )
    if resident is None and not acc.unit_plans:
        raise ValueError(
            "Offloading requires at least one parameter, registered buffer, or offloaded module to manage."
        )

    return _PlannedPartition(
        resident=resident,
        units=tuple(acc.unit_plans),
        unit_paths=tuple(acc.unit_paths),
        param_slot_names=param_index.slot_names,
        buffer_slot_names=buffer_index.slot_names,
    )


def _validate_non_unit_module_slots(
    module: nn.Module,
    param_index: _SlotIndex,
    buffer_index: _SlotIndex,
) -> None:
    mod_id = id(module)
    for leaf, param in module._parameters.items():
        if param is not None and (mod_id, leaf) in param_index.owned_slots:
            name = param_index.slot_names[(mod_id, leaf)][0]
            raise ValueError(
                "Parameter storage shared across offload boundaries is unsupported: "
                f"{name!r}."
            )
    for leaf, buf in module._buffers.items():
        if buf is not None and (mod_id, leaf) in buffer_index.owned_slots:
            name = buffer_index.slot_names[(mod_id, leaf)][0]
            raise ValueError(
                "Buffer module slot shared across offload boundaries is unsupported: "
                f"{name!r}."
            )


def _validate_no_resident_descendant_aliases(
    model: nn.Module,
    acc: _UnitAccumulator,
    param_index: _SlotIndex,
    buffer_index: _SlotIndex,
) -> None:
    unit_root_ids = {id(unit.module) for unit in acc.unit_plans}
    if not unit_root_ids:
        return
    seen_ids: set[int] = set()
    stack: list[nn.Module] = [model]
    while stack:
        module = stack.pop()
        mod_id = id(module)
        if mod_id in seen_ids:
            continue
        seen_ids.add(mod_id)
        if mod_id != id(model):
            _validate_non_unit_module_slots(module, param_index, buffer_index)
        for child in module._modules.values():
            if child is None:
                continue
            child_id = id(child)
            if child_id not in unit_root_ids and child_id not in seen_ids:
                stack.append(child)


def _validate_no_cross_unit_ties(
    model: nn.Module,
    partition: _PlannedPartition,
    param_key: Callable[[str], Hashable],
) -> None:
    param_owners: dict[Hashable, tuple[object, str]] = {}
    buffer_slot_owners: dict[_SlotKey, tuple[object, str]] = {}

    def record_param(group_key: object, name: str) -> None:
        key = param_key(name)
        existing = param_owners.get(key)
        if existing is None:
            param_owners[key] = (group_key, name)
            return
        if existing[0] != group_key:
            raise ValueError(
                "Parameter storage shared across offload boundaries is unsupported: "
                f"{existing[1]!r} and {name!r}. Keep tied weights together outside "
                "offload_paths or inside a single offloaded unit."
            )

    def record_buffer(group_key: object, name: str) -> None:
        slot = _slot_of(model, name)
        existing_slot = buffer_slot_owners.get(slot)
        if existing_slot is None:
            buffer_slot_owners[slot] = (group_key, name)
        elif existing_slot[0] != group_key:
            raise ValueError(
                "Buffer module slot shared across offload boundaries is unsupported: "
                f"{existing_slot[1]!r} and {name!r}."
            )

    if partition.resident is not None:
        for name in partition.resident[0]:
            record_param("resident", name)
        for name in partition.resident[1]:
            record_buffer("resident", name)
    for unit in partition.units:
        group_key = ("unit", unit.unit_idx)
        for _local, store_key, _slot in unit.params:
            record_param(group_key, store_key)
        for _local, store_key, _slot in unit.buffers:
            record_buffer(group_key, store_key)


class ModelPartition:
    """Assembled partition owning host instances, flat-to-unit routing, and engine."""

    def __init__(
        self,
        *,
        engine: ResidencyEngine,
        resident_params: frozenset[str],
        resident_buffers: frozenset[str],
        unit_param_index: Mapping[str, tuple[int, str]],
        unit_buffer_names: frozenset[str],
    ) -> None:
        self.engine = engine
        self._resident_params = resident_params
        self._resident_buffers = resident_buffers
        self._unit_param_index = dict(unit_param_index)
        self.param_names = resident_params | frozenset(unit_param_index)
        self.buffer_names = resident_buffers | unit_buffer_names

    def route_overrides(
        self,
        parameter_overrides: Mapping[str, ParameterOverride] | None,
    ) -> tuple[dict[str, ParameterOverride] | None, list[dict[str, ParameterOverride] | None]]:
        """Split flat model parameter overrides into ``(resident, per_unit)`` dicts."""
        num_units = len(self.engine.units)
        unit_ovrs: list[dict[str, ParameterOverride] | None] = [None] * num_units
        if not parameter_overrides:
            return None, unit_ovrs

        unknown = sorted(parameter_overrides.keys() - self.param_names)
        if unknown:
            raise ValueError(
                f"Parameter overrides contain unmanaged names: {unknown!r}."
            )

        resident_ovr: dict[str, ParameterOverride] | None = None
        for full_name, override in parameter_overrides.items():
            if full_name in self._resident_params:
                if resident_ovr is None:
                    resident_ovr = {}
                resident_ovr[full_name] = override
                continue
            unit_idx, local_name = self._unit_param_index[full_name]
            target = unit_ovrs[unit_idx]
            if target is None:
                target = {}
                unit_ovrs[unit_idx] = target
            existing = target.get(local_name)
            if existing is not None and (
                existing.source is not override.source
                or existing.update is not override.update
            ):
                raise ValueError(
                    "Tied parameter aliases cannot resolve to different loads; "
                    f"conflict at {full_name!r}."
                )
            target[local_name] = override
        return resident_ovr, unit_ovrs


def _build_unit_param_index(
    units: tuple[_UnitPlan, ...],
    param_slot_names: Mapping[_SlotKey, list[str]],
    buffer_slot_names: Mapping[_SlotKey, list[str]],
) -> tuple[dict[str, tuple[int, str]], frozenset[str]]:
    param_index: dict[str, tuple[int, str]] = {}
    slot_local_reps: dict[_SlotKey, dict[str, tuple[int, str]]] = {}
    slot_name_sets: dict[_SlotKey, set[str]] = {}

    for unit in units:
        for local, _store_key, slot in unit.params:
            ref = (unit.unit_idx, local)
            slot_local_reps.setdefault(slot, {}).setdefault(local, ref)
            names_set = slot_name_sets.get(slot)
            if names_set is None:
                names_set = set(param_slot_names[slot])
                slot_name_sets[slot] = names_set
            exact = f"{unit.path}.{local}"
            if exact in names_set:
                param_index[exact] = ref

    for slot, local_reps in slot_local_reps.items():
        reps = tuple(local_reps.values())
        for reg_name in param_slot_names[slot]:
            if reg_name in param_index:
                continue
            matched = next(
                (
                    (u_idx, local)
                    for u_idx, local in reps
                    if reg_name == local or reg_name.endswith(f".{local}")
                ),
                reps[0],
            )
            param_index[reg_name] = matched

    unique_buf_slots: set[_SlotKey] = {
        slot for unit in units for _local, _store_key, slot in unit.buffers
    }
    unit_buffers = {
        reg_name
        for slot in unique_buf_slots
        for reg_name in buffer_slot_names[slot]
    }
    return param_index, frozenset(unit_buffers)


def _assemble_partition(
    store: HostModuleStore,
    model: nn.Module,
    partition: _PlannedPartition,
    config: OffloadConfig,
    *,
    install: bool,
) -> ModelPartition:
    instances_to_install: list[HostModuleInstance] = []
    resident_instance: HostModuleInstance | None = None
    resident_params: frozenset[str] = frozenset()
    resident_buffers: frozenset[str] = frozenset()
    if partition.resident is not None:
        r_param_names, r_buffer_names = partition.resident
        resident_instance = HostModuleInstance(
            module=model,
            params={name: store.params[name] for name in r_param_names},
            buffers={name: store.buffers[name] for name in r_buffer_names},
        )
        resident_params = frozenset(r_param_names)
        resident_buffers = frozenset(r_buffer_names)
        instances_to_install.append(resident_instance)

    unit_instances: list[HostModuleInstance] = []
    for unit in partition.units:
        instance = HostModuleInstance(
            module=unit.module,
            params={local: store.params[store_key] for local, store_key, _slot in unit.params},
            buffers={local: store.buffers[store_key] for local, store_key, _slot in unit.buffers},
        )
        unit_instances.append(instance)
        instances_to_install.append(instance)

    unit_param_index, unit_buffers = _build_unit_param_index(
        partition.units,
        partition.param_slot_names,
        partition.buffer_slot_names,
    )
    engine = ResidencyEngine(
        resident=resident_instance,
        units=unit_instances,
        unit_paths=partition.unit_paths,
        vram_budget_bytes=config.vram_budget_bytes,
        block_compile=config.block_compile,
    )
    result = ModelPartition(
        engine=engine,
        resident_params=resident_params,
        resident_buffers=resident_buffers,
        unit_param_index=unit_param_index,
        unit_buffer_names=unit_buffers,
    )
    if install:
        for instance in instances_to_install:
            instance.install_host()
    return result


def _bind_module_store(
    store: HostModuleStore,
    model: nn.Module,
    config: OffloadConfig,
) -> tuple[ModelPartition, _ShellClaim, HostModuleStore]:
    claim = claim_shell(model)
    try:
        resolved_store = store._resolve_bind(model)
        planned = _plan_model_units(model, config, store=resolved_store)
        _validate_no_cross_unit_ties(
            model,
            planned,
            lambda name: id(resolved_store.params[name]),
        )
        partition = _assemble_partition(
            resolved_store, model, planned, config, install=True
        )
    except BaseException:
        claim.release()
        raise
    return partition, claim, resolved_store


def _validate_freeze_and_ties(
    model: nn.Module,
    planned: _PlannedPartition,
) -> None:
    raw_params = dict(model.named_parameters(remove_duplicate=False))
    trainable = sorted(
        name for name, param in raw_params.items() if param.requires_grad
    )
    if trainable:
        formatted = ", ".join(repr(name) for name in trainable)
        raise ValueError(
            "Piper Offload is inference-only and requires frozen parameters "
            f"(requires_grad=False); found trainable parameter(s): {formatted}."
        )
    _validate_no_cross_unit_ties(
        model,
        planned,
        lambda name: param_tensor_id(raw_params[name]),
    )


def _capture_and_bind(
    model: nn.Module,
    config: OffloadConfig,
) -> tuple[HostModuleStore, ModelPartition, _ShellClaim]:
    claim = claim_shell(model)
    try:
        planned = _plan_model_units(model, config, store=None)
        _validate_freeze_and_ties(model, planned)
        store, _ = HostModuleStore.capture(model)
        partition = _assemble_partition(store, model, planned, config, install=False)
    except BaseException:
        claim.release()
        raise
    return store, partition, claim


__all__ = [
    "ModelPartition",
    "_ShellClaim",
    "_bind_module_store",
    "_capture_and_bind",
    "claim_shell",
    "reserve_shell",
]
