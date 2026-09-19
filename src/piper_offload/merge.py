"""Permanent adapter merge into model weights.

Merges additive parameter deltas directly into existing model parameters and
materializes dense or prequantized parameter values for meta targets. Plain
floating-point targets support combined low-rank and full-rank deltas.
Quantized adapters own their factorized and dense encoding paths and may select
a format-specific kernel or a dequantize/requantize fallback. Mixed deltas and
scaled quantized values encode the target once.

Permanent and activation merge use the same parameter-delta and
parameter-value transforms. Permanent merge applies them to resident model
parameters; activation merge invokes them after individual parameter copies.
"""

import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .adapter import Adapter, AdapterTargetUpdates
from .module_names import resolve_parent_leaf
from .parameter_delta import ParameterDeltaTransform
from .parameter_transform import ParameterTransform
from .parameter_value import ParameterValueTransform
from .pin_manager import host_pin_manager
from .tensor_adapter_registry import (
    param_representation,
    param_tensor_id,
    select_adapter,
)
from .tensor_adapters import (
    PermanentUpdateValidationTensorAdapter,
    host_state_is_owned,
    independent_host_capture,
)

logger = logging.getLogger(__name__)

__all__ = ["merge_adapter"]


def _validate_permanent_update(adapter: object) -> None:
    if isinstance(adapter, PermanentUpdateValidationTensorAdapter):
        adapter.validate_permanent_update()


@dataclass(slots=True)
class _TargetGroup:
    target_key: str
    param: nn.Parameter
    updates: AdapterTargetUpdates = field(default_factory=AdapterTargetUpdates)


@dataclass(slots=True, frozen=True)
class _MergeOp:
    aliases: tuple[str, ...]
    param: nn.Parameter
    transform: ParameterTransform

    def validate(self) -> None:
        """Preflight this operation's parameter."""
        if isinstance(self.transform, ParameterValueTransform):
            if self.transform.requires_update:
                _validate_permanent_update(self.transform.backing.adapter)
        else:
            _validate_permanent_update(
                select_adapter(param_representation(self.param))
            )
        self.transform.validate_parameter(self.param)

    def apply(self, model: nn.Module) -> None:
        """Apply this operation's parameter update."""
        if isinstance(self.transform, ParameterValueTransform):
            self._install(model, self.transform.materialize())
            return
        param = self.param
        owned = _owned_copy(param)
        if owned is not None:
            self._install(model, owned)
            param = owned
        self.transform.apply_parameter(param)

    def _install(self, model: nn.Module, replacement: nn.Parameter) -> None:
        for alias in self.aliases:
            parent, leaf = resolve_parent_leaf(model, alias)
            if leaf not in parent._parameters:
                raise RuntimeError(f"Parameter {alias!r} disappeared during permanent merge.")
            parent._parameters[leaf] = replacement


def _owned_copy(param: nn.Parameter) -> nn.Parameter | None:
    """An independent copy of ``param`` if it lives in storage Piper must not write, else None.

    Merging writes the parameter in place. Piper never writes into a file
    mapping, because the pin manager may return its pages to the file, so a
    target that is a view into one is replaced by a copy the process owns.
    """
    if param.is_meta:
        return None
    representation = param_representation(param)
    adapter = select_adapter(representation)
    if host_state_is_owned(adapter, adapter.capture_host(representation)):
        return None
    with independent_host_capture():
        state = adapter.capture_host(representation)
    return adapter.cpu_param(state, requires_grad=param.requires_grad)


def merge_adapter(
    model: nn.Module,
    adapters: Sequence[tuple[Adapter, float]],
    *,
    stochastic_rounding: bool = True,
) -> int:
    """Merge one or more adapters into model parameters in-place.

    Returns the number of unique parameters that were modified. Exact-zero
    strengths are inactive and do not create merge operations. Merge reads
    immutable host backing, so the same adapter may also serve other
    merge or routed uses. All active target names and merge capabilities are
    validated before any parameter is modified. An adapter constructed with
    ``allow_partial_targets=True`` ignores targets absent from this model.
    Quantized targets use terminal-code stochastic rounding by default so
    sub-step additive updates are not systematically rounded away; pass
    ``stochastic_rounding=False`` for deterministic rounding. Parameter values
    populate frozen floating-point meta targets according to their strength
    policy. Every registered physical value retains its source representation;
    explicit non-unit scaling additionally requires dequantization and dense
    merge support. A populated meta target is replaced by one independent
    frozen CPU parameter, preserving any tied aliases of the original
    parameter. A target that is a view into a file mapping is likewise
    replaced by an independent parameter under every tied name before it is
    merged: Piper never writes into a file mapping, because the pin manager
    may return a mapping's pages to the file.
    """
    # Filtering here avoids target lookup, staging, validation, and
    # requantization for work that cannot modify a parameter.
    active_adapters: list[tuple[Adapter, float]] = []
    for adapter, strength in adapters:
        normalized = float(strength)
        if normalized != 0.0:
            active_adapters.append((adapter, normalized))
    return _merge_adapters(
        model,
        active_adapters,
        stochastic_rounding=stochastic_rounding,
    )


def _merge_adapters(
    model: nn.Module,
    adapters: Sequence[tuple[Adapter, float]],
    *,
    stochastic_rounding: bool,
) -> int:
    params_by_target = _collect_params_by_target(model)

    missing_targets = sorted(
        {
            target_key
            for adapter, _strength in adapters
            for target_key in adapter.targets
            if target_key not in params_by_target and not adapter.allow_partial_targets
        }
    )
    if missing_targets:
        sample = sorted(params_by_target)[:3]
        raise ValueError(
            f"Adapter targets are not parameters in the model: {missing_targets}. "
            "Adapter target keys must match the model's parameter names exactly. "
            f"Sample model parameter keys: {sample} ..."
        )

    merge_ops = _build_merge_ops(
        params_by_target,
        adapters,
        stochastic_rounding=stochastic_rounding,
    )
    applied_target_count = sum(
        target_key in params_by_target for adapter, _strength in adapters for target_key in adapter.targets
    )

    # Preflight every operation before applying any of them. This catches all
    # expected name, shape, and adapter-capability errors without leaving a
    # permanently half-merged model.
    # Validation and application both stage adapter sources onto a CUDA
    # target asynchronously, so the sources are leased, pageable, until the
    # device has synchronized; a failed step may already have enqueued copies.
    # If the synchronization itself fails the context is unusable and no
    # later call could retry it, so the lease closes with this function.
    devices = {op.param.device for op in merge_ops if op.param.device.type == "cuda"}
    sources = (tensor for op in merge_ops for tensor in op.transform.storage_tensors())
    with host_pin_manager.acquire(sources, pin=False) if devices else contextlib.nullcontext():
        try:
            for op in merge_ops:
                op.validate()
            for op in merge_ops:
                op.apply(model)
        finally:
            for device in devices:
                torch.cuda.synchronize(device)

    modified_tensor_ids = {param_tensor_id(op.param) for op in merge_ops}

    logger.info(
        "merge_adapter: merged %d unique parameters from %d adapter targets",
        len(modified_tensor_ids),
        applied_target_count,
    )
    return len(modified_tensor_ids)


def _collect_params_by_target(model: nn.Module) -> dict[str, nn.Parameter]:
    params_by_target: dict[str, nn.Parameter] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        params_by_target[name] = param
    return params_by_target


def _tie_key(param: nn.Parameter) -> tuple[Any, ...]:
    """Identity under which parameters sharing one backing are one merge target."""
    try:
        return param_tensor_id(param)
    except NotImplementedError:
        # Let transform validation produce the target-specific capability
        # error. Unsupported wrappers cannot participate in tied-storage
        # detection, so object identity is the conservative grouping key.
        return ("__unsupported_param__", id(param))


def _build_merge_ops(
    params_by_target: dict[str, nn.Parameter],
    adapters: Sequence[tuple[Adapter, float]],
    *,
    stochastic_rounding: bool,
) -> list[_MergeOp]:
    """Group adapter updates and reject ambiguous parameter ties."""
    groups_by_tensor_id: dict[tuple[Any, ...], _TargetGroup] = {}

    def target_group(target_key: str) -> _TargetGroup | None:
        param = params_by_target.get(target_key)
        if param is None:
            return None
        tensor_id = _tie_key(param)
        group = groups_by_tensor_id.get(tensor_id)
        if group is None:
            group = _TargetGroup(target_key, param)
            groups_by_tensor_id[tensor_id] = group
        elif group.target_key != target_key:
            raise ValueError(
                f"Adapter targets {group.target_key!r} and {target_key!r} "
                "resolve to the same tied parameter backing. Apply only one "
                "name for a tied parameter in a single merge_adapter() call; "
                "otherwise the same base would receive multiple logical updates."
            )
        return group

    for adapter, strength in adapters:
        for target_key, target in adapter.targets.items():
            group = target_group(target_key)
            if group is None:
                continue
            group.updates.add(target, strength, target_key=target_key)

    # A replacement parameter must reach every name sharing the target's
    # backing, including distinct wrapper objects over the same storage.
    names_by_tie: dict[tuple[Any, ...], list[str]] = {}
    if groups_by_tensor_id:
        for name, candidate in params_by_target.items():
            names_by_tie.setdefault(_tie_key(candidate), []).append(name)

    merge_ops: list[_MergeOp] = []
    for tensor_id, group in groups_by_tensor_id.items():
        target_key = group.target_key
        param = group.param
        aliases = tuple(names_by_tie[tensor_id])
        transform: ParameterTransform
        if group.updates.deltas:
            transform = ParameterDeltaTransform(
                group.updates.deltas,
                stochastic_rounding=stochastic_rounding,
                target_key=target_key,
            )
        else:
            assert group.updates.value is not None
            transform = ParameterValueTransform(
                group.updates.value,
                stochastic_rounding=stochastic_rounding,
                target_key=target_key,
            )
        merge_ops.append(
            _MergeOp(
                aliases,
                param,
                transform,
            )
        )

    return merge_ops
