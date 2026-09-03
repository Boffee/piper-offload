"""Permanent adapter merge into model weights.

Merges additive parameter deltas directly into existing model parameters and
materializes parameter values for meta targets. Plain floating-point targets
support combined low-rank and full-rank deltas. Quantized adapters own their
factorized and dense encoding paths and may select a format-specific kernel or
a dequantize/requantize fallback. Mixed deltas are staged as one full-rank
update so the quantized base is encoded once.

Permanent and activation merge use the same parameter-delta and
parameter-value transforms. Permanent merge applies them to resident model
parameters; activation merge invokes them after individual parameter copies.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from torch import nn

from .adapter import Adapter, AdapterTargetUpdates
from .module_names import resolve_parent_leaf
from .parameter_delta import ParameterDeltaTransform
from .parameter_transform import ParameterTransform
from .parameter_value import ParameterValueTransform
from .tensor_adapter_registry import param_tensor_id

logger = logging.getLogger(__name__)

__all__ = ["merge_adapter"]


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
        self.transform.validate_parameter(self.param)

    def apply(self, model: nn.Module) -> None:
        """Apply this operation's parameter update."""
        if isinstance(self.transform, ParameterValueTransform):
            materialized = self.transform.materialize()
            for alias in self.aliases:
                parent, leaf = resolve_parent_leaf(model, alias)
                if leaf not in parent._parameters:
                    raise RuntimeError(f"Parameter {alias!r} disappeared during permanent merge.")
                parent._parameters[leaf] = materialized
            return
        self.transform.apply_parameter(self.param)


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
    policy. A populated meta target is replaced by one frozen CPU parameter,
    preserving any tied aliases of the original parameter.
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
    for op in merge_ops:
        op.validate()

    for op in merge_ops:
        op.apply(model)

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
        try:
            tensor_id = param_tensor_id(param)
        except NotImplementedError:
            # Let transform validation produce the target-specific capability
            # error. Unsupported wrappers cannot participate in tied-storage
            # detection, so object identity is the conservative grouping key.
            tensor_id = ("__unsupported_param__", id(param))
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

    merge_ops: list[_MergeOp] = []
    for group in groups_by_tensor_id.values():
        target_key = group.target_key
        param = group.param
        aliases = (
            tuple(name for name, param in params_by_target.items() if param is group.param) if param.is_meta else ()
        )
        transform: ParameterTransform
        if group.updates.deltas:
            transform = ParameterDeltaTransform(
                group.updates.deltas,
                stochastic_rounding=stochastic_rounding,
                target_key=target_key,
            )
        else:
            assert group.updates.value is not None
            transform = ParameterValueTransform(group.updates.value)
        merge_ops.append(
            _MergeOp(
                aliases,
                param,
                transform,
            )
        )

    return merge_ops
