"""Inductor integration for experimental per-parameter block rollover."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Protocol, cast
from weakref import ReferenceType, ref

import torch
from torch import fx, nn
from torch._dynamo import lookup_backend
from torch._inductor import config as inductor_config
from torch._inductor.custom_graph_pass import (
    CustomInferenceAwareGraphPass,
    CustomSchedulerPass,
)
from torch._inductor.dependencies import WeakDep
from torch._inductor.virtualized import V
from torch._library.effects import EffectType

if TYPE_CHECKING:
    from torch._inductor.scheduler import BaseSchedulerNode


class RollingRuntime(Protocol):
    """Runtime called by the side-effecting op inserted into each graph."""

    def wait_param(self, param_idx: int) -> None: ...

    def rollover_param(self, param_idx: int) -> None: ...


type _SlotEntry = tuple[ReferenceType[object], int]
type _ObjectSlotEntry = tuple[ReferenceType[object], ReferenceType[object], int]
type _TensorLayout = tuple[tuple[int, ...], torch.dtype, tuple[int, ...]]
type _ParamSpec = tuple[int, tuple[int, ...], tuple[_TensorLayout, ...]]

_SLOTS_BY_OBJECT_ID: dict[int, _ObjectSlotEntry] = {}
_SLOTS_BY_DATA_PTR: dict[int, _SlotEntry] = {}
# ``param.data`` wrappers are transient, so a bare object ID is unsafe: Python
# may reuse it for an unrelated activation tensor. Object entries retain a
# weak reference and every lookup verifies identity before accepting the ID.
_SLOT_REGISTRY_LOCK = RLock()
_WAIT_KERNEL = "piper_offload.rolling_wait.default"
_REFILL_KERNEL = "piper_offload.rolling_refill.default"


def _remove_slot_entries(matches: Callable[[_SlotEntry], bool]) -> None:
    with _SLOT_REGISTRY_LOCK:
        for key, (_value_ref, runtime_ref, param_idx) in tuple(_SLOTS_BY_OBJECT_ID.items()):
            if matches((runtime_ref, param_idx)):
                _SLOTS_BY_OBJECT_ID.pop(key, None)
        for key, entry in tuple(_SLOTS_BY_DATA_PTR.items()):
            if matches(entry):
                _SLOTS_BY_DATA_PTR.pop(key, None)


def _tensor_layout(tensor: torch.Tensor) -> _TensorLayout:
    return (tuple(tensor.shape), tensor.dtype, tensor.stride())


def rolling_storage_tensors(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return the physical tensors AOT exposes for one logical slot.

    Ordinary parameters stay as one compiler argument. TorchAO-style tensor
    subclasses are flattened into their named storage tensors before the
    post-grad Inductor graph, so registration and liveness tracking must use
    the same public ``__tensor_flatten__`` contract.
    """
    flatten = getattr(tensor, "__tensor_flatten__", None)
    if flatten is None:
        return (tensor,)
    names, _context = flatten()
    storage = tuple(getattr(tensor, name) for name in names)
    if not storage or any(not isinstance(item, torch.Tensor) for item in storage):
        raise RuntimeError("rolling compilation received a tensor subclass with invalid __tensor_flatten__ storage")
    return storage


def _registered_slot_objects(param: nn.Parameter) -> tuple[torch.Tensor, ...]:
    # Dynamo identifies the logical wrapper; AOT may identify its flattened
    # storage wrappers, whose identities remain significant across graph reuse.
    representation = param.data
    values = (param, representation, *rolling_storage_tensors(representation))
    return tuple(dict.fromkeys(values))


def register_rolling_target(
    runtime: RollingRuntime,
    params: Sequence[nn.Parameter],
) -> None:
    """Associate active slot wrappers/storage with their weak runtime owner."""
    objects_by_param = tuple(_registered_slot_objects(param) for param in params)

    def remove_dead_runtime(runtime_ref: ReferenceType[object]) -> None:
        _remove_slot_entries(lambda entry: entry[0] is runtime_ref)

    runtime_owner = cast(object, runtime)
    runtime_ref = ref(runtime_owner, remove_dead_runtime)
    with _SLOT_REGISTRY_LOCK:
        for param_idx, values in enumerate(objects_by_param):
            for value in values:
                object_entry = _SLOTS_BY_OBJECT_ID.get(id(value))
                if object_entry is not None:
                    registered_value = object_entry[0]()
                    existing_owner = object_entry[1]() if registered_value is value else None
                    if existing_owner is not None and (
                        existing_owner is not runtime_owner or object_entry[2] != param_idx
                    ):
                        raise RuntimeError(f"rolling target registration collided on object identity {id(value)}")

                data_ptr = value.data_ptr()
                data_entry = _SLOTS_BY_DATA_PTR.get(data_ptr) if data_ptr != 0 else None
                existing_owner = None if data_entry is None else data_entry[0]()
                if (
                    data_entry is not None
                    and existing_owner is not None
                    and (existing_owner is not runtime_owner or data_entry[1] != param_idx)
                ):
                    raise RuntimeError(f"rolling target registration collided on storage pointer {data_ptr}")

        for param_idx, values in enumerate(objects_by_param):
            entry = (runtime_ref, param_idx)
            for value in values:
                object_id = id(value)

                def remove_dead_value(
                    value_ref: ReferenceType[object],
                    *,
                    key: int = object_id,
                ) -> None:
                    with _SLOT_REGISTRY_LOCK:
                        current = _SLOTS_BY_OBJECT_ID.get(key)
                        if current is not None and current[0] is value_ref:
                            _SLOTS_BY_OBJECT_ID.pop(key, None)

                value_ref = ref(cast(object, value), remove_dead_value)
                _SLOTS_BY_OBJECT_ID[object_id] = (value_ref, runtime_ref, param_idx)
                data_ptr = value.data_ptr()
                if data_ptr != 0:
                    _SLOTS_BY_DATA_PTR[data_ptr] = entry


def unregister_rolling_target(
    runtime: RollingRuntime,
) -> None:
    """Remove active-slot associations before their allocations can be reused."""
    runtime_owner = cast(object, runtime)
    _remove_slot_entries(lambda entry: entry[0]() is runtime_owner)


def _registered_slot_entry(
    value: object,
    storage: Sequence[torch.Tensor] = (),
) -> _SlotEntry | None:
    with _SLOT_REGISTRY_LOCK:
        object_entry = _SLOTS_BY_OBJECT_ID.get(id(value))
        if object_entry is not None and object_entry[0]() is value:
            return (object_entry[1], object_entry[2])
        for item in storage:
            object_entry = _SLOTS_BY_OBJECT_ID.get(id(item))
            if object_entry is not None and object_entry[0]() is item:
                return (object_entry[1], object_entry[2])
            if item.data_ptr() != 0:
                data_entry = _SLOTS_BY_DATA_PTR.get(item.data_ptr())
                if data_entry is not None:
                    return data_entry
    return None


def _runtime_for_slot(
    slots: list[torch.Tensor],
    param_idx: int,
    operation: str,
) -> RollingRuntime:
    if not slots:
        raise RuntimeError(f"compiled rolling {operation} received no storage slots")
    with _SLOT_REGISTRY_LOCK:
        entry = _SLOTS_BY_DATA_PTR.get(slots[0].data_ptr())
        runtime = None if entry is None else entry[0]()
    if runtime is None or entry is None:
        raise RuntimeError("compiled rolling lifecycle callback could not resolve its active target runtime")
    if entry[1] != param_idx:
        raise RuntimeError(f"compiled rolling {operation} parameter index does not match its active target slot")
    return cast(RollingRuntime, runtime)


@torch.library.custom_op(
    "piper_offload::rolling_wait",
    mutates_args=(),
)
def _rolling_wait(
    slots: list[torch.Tensor],
    param_idx: int,
) -> None:
    _runtime_for_slot(slots, param_idx, "wait").wait_param(param_idx)


@torch.library.register_fake("piper_offload::rolling_wait")
def _rolling_wait_fake(
    slots: list[torch.Tensor],
    param_idx: int,
) -> None:
    del slots, param_idx


@torch.library.custom_op(
    "piper_offload::rolling_refill",
    mutates_args=(),
)
def _rolling_refill(
    slots: list[torch.Tensor],
    param_idx: int,
) -> None:
    _runtime_for_slot(slots, param_idx, "refill").rollover_param(param_idx)


@torch.library.register_fake("piper_offload::rolling_refill")
def _rolling_refill_fake(
    slots: list[torch.Tensor],
    param_idx: int,
) -> None:
    del slots, param_idx


torch.library._register_effectful_op(torch.ops.piper_offload.rolling_wait.default, EffectType.ORDERED)
torch.library._register_effectful_op(torch.ops.piper_offload.rolling_refill.default, EffectType.ORDERED)


def _contains_node(value: object, wanted: fx.Node) -> bool:
    if value is wanted:
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_node(item, wanted) for item in value)
    if isinstance(value, dict):
        return any(_contains_node(item, wanted) for item in value.values())
    return False


def _is_read_only_alias(user: fx.Node, source: fx.Node) -> bool:
    """Whether ``user`` only creates another view of ``source``.

    Inductor commonly lowers a linear weight through ``aten.permute`` before
    ``aten.mm``. Treating the permute as the final read would overwrite the
    shared storage before the matrix multiply. Operator alias annotations let
    us follow such view chains to the actual consumer without encoding a list
    of particular view operators.
    """
    schema = getattr(user.target, "_schema", None)
    if schema is None:
        return False

    input_aliases: set[str] = set()
    input_writes = False
    for position, argument in enumerate(schema.arguments):
        value = user.args[position] if position < len(user.args) else user.kwargs.get(argument.name)
        if not _contains_node(value, source):
            continue
        alias_info = argument.alias_info
        if alias_info is None:
            continue
        input_aliases.update(alias_info.before_set)
        input_writes |= alias_info.is_write

    if not input_aliases or input_writes:
        return False
    return any(
        result.alias_info is not None and bool(input_aliases & set(result.alias_info.after_set))
        for result in schema.returns
    )


def _storage_readers(
    graph: fx.Graph,
    placeholders: Sequence[fx.Node],
) -> tuple[fx.Node, ...]:
    aliases = set(placeholders)
    readers: list[fx.Node] = []
    output: fx.Node | None = None

    for node in graph.nodes:
        if node.op == "output":
            output = node
        sources = [alias for alias in aliases if alias in node.all_input_nodes]
        if not sources:
            continue
        if node.op == "output":
            raise RuntimeError("rolling compilation cannot refill a parameter returned from the compiled block graph")
        if all(_is_read_only_alias(node, source) for source in sources):
            aliases.add(node)
        else:
            readers.append(node)

    if output is None:
        raise RuntimeError("rolling compilation received a graph without output")

    return tuple(readers)


@dataclass(frozen=True, slots=True)
class _RollingLifecyclePass(CustomInferenceAwareGraphPass):
    """Bracket each parameter's storage readers with lifecycle effects."""

    param_specs: tuple[_ParamSpec, ...]

    def uuid(self) -> None:
        # Active target discovery is process-local, so persistent Inductor
        # code caching stays disabled. Dynamo can still reuse the compiled
        # graph in-process because runtime lookup is keyed by slot storage.
        return None

    def __call__(self, graph: fx.Graph, is_inference: bool) -> None:
        if not is_inference:
            raise RuntimeError("rolling compilation is inference-only")

        placeholders = [node for node in graph.nodes if node.op == "placeholder"]
        for param_idx, argument_indices, expected_layouts in self.param_specs:
            if not argument_indices or max(argument_indices) >= len(placeholders):
                raise RuntimeError("Inductor changed compiled argument ordering before the rolling lifecycle pass")
            storage_placeholders = tuple(placeholders[argument_idx] for argument_idx in argument_indices)
            actual_layouts = tuple(
                _tensor_layout(value)
                for placeholder in storage_placeholders
                if isinstance((value := placeholder.meta.get("val")), torch.Tensor)
            )
            if actual_layouts != expected_layouts:
                raise RuntimeError("AOT changed a rolling parameter's flattened storage layout or argument ordering")
            readers = _storage_readers(graph, storage_placeholders)
            if not readers:
                continue
            first_reader = readers[0]
            last_reader = readers[-1]
            # The wait is an ordered host effect. A late scheduler-only edge
            # orders it before the storage readers without making Inductor
            # treat an immutable parameter as mutated. The latter changes
            # coalescing scores and can select numerically different reduction
            # launch configurations even when the arithmetic graph is the same.
            with graph.inserting_before(first_reader):
                graph.call_function(
                    torch.ops.piper_offload.rolling_wait.default,
                    args=(list(storage_placeholders), param_idx),
                )
            # The refill is likewise an ordered host effect. Scheduler-only
            # reader-to-refill edges keep it after every physical storage read
            # without consuming a reader output, so ordinary fusion survives.
            with graph.inserting_after(last_reader):
                graph.call_function(
                    torch.ops.piper_offload.rolling_refill.default,
                    args=(list(storage_placeholders), param_idx),
                )
        graph.lint()


class _RollingSchedulerPass(CustomSchedulerPass):
    """Order lifecycle effects around readers without tensor materialization."""

    def uuid(self) -> None:
        return None

    @staticmethod
    def _is_kernel(node: BaseSchedulerNode, suffix: str | tuple[str, ...]) -> bool:
        return str(getattr(node.node, "python_kernel_name", "")).endswith(suffix)

    @classmethod
    def _is_lifecycle_node(cls, node: BaseSchedulerNode) -> bool:
        return cls._is_kernel(node, (_WAIT_KERNEL, _REFILL_KERNEL))

    @classmethod
    def _is_refill(cls, node: BaseSchedulerNode) -> bool:
        return cls._is_kernel(node, _REFILL_KERNEL)

    @classmethod
    def _is_wait(cls, node: BaseSchedulerNode) -> bool:
        return cls._is_kernel(node, _WAIT_KERNEL)

    @classmethod
    def _storage_readers(
        cls,
        lifecycle: BaseSchedulerNode,
        nodes: Sequence[BaseSchedulerNode],
    ) -> list[BaseSchedulerNode]:
        lifecycle_outputs = {
            output.get_name() for node in nodes if cls._is_lifecycle_node(node) for output in node.get_outputs()
        }
        slot_names = {
            dependency.name
            for dependency in lifecycle.read_writes.reads
            if not isinstance(dependency, WeakDep) and dependency.name not in lifecycle_outputs
        }
        return [
            node
            for node in nodes
            if node is not lifecycle
            and not cls._is_lifecycle_node(node)
            and any(
                not isinstance(dependency, WeakDep) and dependency.name in slot_names
                for dependency in node.read_writes.reads
            )
        ]

    def __call__(
        self,
        nodes: list[BaseSchedulerNode],
    ) -> list[BaseSchedulerNode]:
        for wait in (node for node in nodes if self._is_wait(node)):
            readers = self._storage_readers(wait, nodes)
            if not readers:
                raise RuntimeError("rolling scheduler could not resolve a parameter reader")
            wait_outputs = wait.get_outputs()
            if not wait_outputs:
                raise RuntimeError("rolling scheduler wait has no effect token")
            wait_token = wait_outputs[0].get_name()
            for reader in readers:
                reader_outputs = reader.get_outputs()
                if not reader_outputs:
                    raise RuntimeError("rolling scheduler reader has no ordering output")
                reader.add_fake_dep(
                    WeakDep(
                        wait_token,
                        mutating_buf=reader_outputs[0].get_name(),
                        is_fake=True,
                    )
                )

        for refill in (node for node in nodes if self._is_refill(node)):
            readers = self._storage_readers(refill, nodes)
            if not readers:
                raise RuntimeError("rolling scheduler could not resolve a parameter reader")
            refill_outputs = refill.get_outputs()
            if not refill_outputs:
                raise RuntimeError("rolling scheduler refill has no effect token")
            mutating_buffer = refill_outputs[0].get_name()
            for reader in readers:
                reader_outputs = reader.get_outputs()
                if not reader_outputs:
                    raise RuntimeError("rolling scheduler reader has no ordering output")
                for output in reader_outputs:
                    refill.add_fake_dep(
                        WeakDep(
                            output.get_name(),
                            mutating_buf=mutating_buffer,
                            is_fake=True,
                        )
                    )
        scheduler = V.graph.scheduler
        if scheduler is None:
            raise RuntimeError("rolling scheduler pass ran without a scheduler")
        ordered = scheduler.topological_sort_schedule(nodes)
        scheduler.nodes = ordered
        scheduler.compute_ancestors()
        scheduler.compute_input_distances()
        return ordered


def _append_post_grad_pass(
    existing: object,
    lifecycle_pass: _RollingLifecyclePass,
) -> object:
    if existing is None:
        return lifecycle_pass
    if isinstance(existing, (tuple, list)):
        return (*existing, lifecycle_pass)
    return (existing, lifecycle_pass)


def _append_pre_fusion_pass(
    existing: object,
    scheduler_pass: _RollingSchedulerPass,
) -> object:
    if existing is None:
        return scheduler_pass
    if not callable(existing):
        raise TypeError("_pre_fusion_custom_pass must be callable")
    existing_pass = cast(
        Callable[[list[BaseSchedulerNode]], list[BaseSchedulerNode]],
        existing,
    )

    def composed(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        return scheduler_pass(existing_pass(nodes))

    return composed


def rolling_inductor_backend(
    graph_module: fx.GraphModule,
    example_inputs: Sequence[object],
    **kwargs: object,
) -> Callable[..., object]:
    """Compile a graph with per-parameter rollover lifecycle effects."""
    if sum(node.op == "placeholder" for node in graph_module.graph.nodes) != len(example_inputs):
        raise RuntimeError("Dynamo supplied mismatched placeholders and example inputs to rolling compilation")

    param_specs: dict[int, tuple[tuple[int, ...], tuple[_TensorLayout, ...]]] = {}
    flat_argument_idx = 0
    for value in example_inputs:
        storage = rolling_storage_tensors(value) if isinstance(value, torch.Tensor) else ()
        width = len(storage) or 1
        entry = _registered_slot_entry(value, storage)
        if entry is not None and entry[0]() is not None:
            param_specs[entry[1]] = (
                tuple(range(flat_argument_idx, flat_argument_idx + width)),
                tuple(_tensor_layout(item) for item in storage),
            )
        flat_argument_idx += width
    if not param_specs:
        raise RuntimeError(
            "rolling compilation could not identify any streamed parameter arguments in the captured block graph"
        )

    lifecycle_pass = _RollingLifecyclePass(tuple((idx, *spec) for idx, spec in param_specs.items()))
    options = dict(cast(Mapping[str, object], kwargs.get("options") or {}))
    options["post_grad_custom_post_pass"] = _append_post_grad_pass(
        options.get("post_grad_custom_post_pass"),
        lifecycle_pass,
    )
    options["_pre_fusion_custom_pass"] = _append_pre_fusion_pass(
        options.get("_pre_fusion_custom_pass"),
        _RollingSchedulerPass(),
    )

    # Rollover is appended so existing Piper/user graph rewrites run first.
    with inductor_config.patch(options):
        backend = cast(Callable[..., Callable[..., object]], lookup_backend("inductor"))
        return backend(graph_module, list(example_inputs))


__all__ = [
    "RollingRuntime",
    "register_rolling_target",
    "rolling_inductor_backend",
    "rolling_storage_tensors",
    "unregister_rolling_target",
]
