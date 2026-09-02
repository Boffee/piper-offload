"""Permanent LoRA merge into model weights.

Merges LoRA deltas directly into model parameters, supporting tensors
whose adapter exposes either dense in-place ``addmm_`` or a staged LoRA
merge. Quantized adapters own their encoding path and may select a
format-specific kernel or a dequantize/requantize fallback; requantized
merges are lossy but standard practice for permanent LoRA merges into
quantized bases.

Permanent and activation merge compose the same factor and dense parameter
transforms. Permanent merge applies them to resident model parameters;
activation merge invokes them after individual parameter copies.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn

from .dense_diff import DenseDiffTransform, ScaledDenseTarget
from .lora import (
    LoRA,
    LoRATransform,
    ScaledLoRAFactor,
)
from .module_names import resolve_parent_leaf, sibling_parameter_name
from .parameter_transform import ParameterTransformSequence
from .tensor_adapter_registry import param_tensor_id

logger = logging.getLogger(__name__)

__all__ = ["merge_lora"]


@dataclass(slots=True)
class _TargetGroup:
    target_key: str
    param: nn.Parameter
    factors: list[ScaledLoRAFactor]
    dense: list[ScaledDenseTarget]


@dataclass(slots=True, frozen=True)
class _MergeOp:
    aliases: tuple[str, ...]
    param: nn.Parameter
    bias: nn.Parameter | None
    transform: ParameterTransformSequence
    lora_transform: LoRATransform | None
    dense_transform: DenseDiffTransform | None

    def validate(self) -> None:
        """Preflight this operation's parameter and optional legacy bias."""
        self.transform.validate_parameter(self.param)
        if self.lora_transform is not None and self.lora_transform.has_bias:
            assert self.bias is not None
            self.lora_transform.validate_bias_target(self.bias)

    def apply(self, model: nn.Module) -> None:
        """Apply this operation's parameter and optional legacy-bias updates."""
        if self.param.is_meta:
            dense_transform = self.dense_transform
            if dense_transform is None:
                raise RuntimeError("A meta merge operation requires a dense transform")
            materialized = dense_transform.materialize_meta()
            for alias in self.aliases:
                parent, leaf = resolve_parent_leaf(model, alias)
                if leaf not in parent._parameters:
                    raise RuntimeError(
                        f"Parameter {alias!r} disappeared during permanent merge."
                    )
                parent._parameters[leaf] = materialized
            return
        self.transform.apply_parameter(self.param)
        if self.lora_transform is not None and self.lora_transform.has_bias:
            assert self.bias is not None
            self.lora_transform.apply_bias(self.bias)


def merge_lora(
    model: nn.Module,
    loras: Sequence[tuple[LoRA, float]],
    *,
    stochastic_rounding: bool = True,
) -> int:
    """Merge one or more LoRAs into model parameters in-place.

    Returns the number of unique parameters that were modified. Exact-zero
    strengths are inactive and do not create merge operations. Merge reads
    immutable host adapter backing, so the same LoRA may also serve other
    merge or routed uses. All active target names and merge capabilities are
    validated before any parameter is modified. A LoRA constructed with
    ``allow_partial_targets=True`` ignores targets absent from this model.
    Quantized targets use
    terminal-code stochastic rounding by default so sub-step LoRA updates are
    not systematically rounded away; pass ``stochastic_rounding=False`` for
    deterministic rounding. Full-shape dense targets use plain floating-point
    addition. A logical-zero meta target is replaced by one frozen CPU
    parameter, preserving any tied aliases of the original parameter.
    """
    # Filtering here avoids target lookup, staging, validation, and
    # requantization for work that cannot modify a parameter.
    active_loras = [(lora, strength) for lora, strength in loras if strength != 0.0]
    return _merge_loras(
        model,
        active_loras,
        stochastic_rounding=stochastic_rounding,
    )


def _merge_loras(
    model: nn.Module,
    loras: Sequence[tuple[LoRA, float]],
    *,
    stochastic_rounding: bool,
) -> int:
    params_by_target = _collect_params_by_target(model)

    missing_targets = sorted({
        target_key
        for lora, _strength in loras
        for target_key, _factor, _dense in lora._iter_targets()
        if target_key not in params_by_target and not lora.allow_partial_targets
    })
    if missing_targets:
        sample = sorted(params_by_target)[:3]
        raise ValueError(
            f"LoRA targets are not parameters in the model: {missing_targets}. "
            "LoRA target keys must match the model's parameter names exactly. "
            f"Sample model parameter keys: {sample} ..."
        )

    merge_ops = _build_merge_ops(
        params_by_target,
        loras,
        stochastic_rounding=stochastic_rounding,
    )
    applied_target_count = sum(
        target_key in params_by_target
        for lora, _strength in loras
        for target_key, _factor, _dense in lora._iter_targets()
    )

    # Preflight every operation before applying any of them. This catches all
    # expected name, shape, and adapter-capability errors without leaving a
    # permanently half-merged model.
    for op in merge_ops:
        op.validate()

    for op in merge_ops:
        op.apply(model)

    bias_count = sum(op.bias is not None for op in merge_ops)
    modified_tensor_ids = {param_tensor_id(op.param) for op in merge_ops}
    modified_tensor_ids.update(
        param_tensor_id(op.bias)
        for op in merge_ops
        if op.bias is not None
    )

    logger.info(
        "merge_lora: merged %d unique parameters (%d weights, %d biases) from %d LoRA targets",
        len(modified_tensor_ids),
        len(merge_ops),
        bias_count,
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
    loras: Sequence[tuple[LoRA, float]],
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
            group = _TargetGroup(target_key, param, [], [])
            groups_by_tensor_id[tensor_id] = group
        elif group.target_key != target_key:
            raise ValueError(
                f"Adapter targets {group.target_key!r} and {target_key!r} "
                "resolve to the same tied parameter backing. Apply only one "
                "name for a tied parameter in a single merge_lora() call; "
                "otherwise the same base would receive multiple logical updates."
            )
        return group

    for lora, strength in loras:
        for target_key, factor, dense in lora._iter_targets():
            group = target_group(target_key)
            if group is None:
                continue
            if factor is not None:
                group.factors.append(factor.scaled(strength))
            if dense is not None:
                group.dense.append(ScaledDenseTarget(dense, strength))

    merge_ops: list[_MergeOp] = []
    bias_owner_by_tensor_id: dict[tuple[Any, ...], str] = {}
    for group in groups_by_tensor_id.values():
        target_key = group.target_key
        param = group.param
        aliases = (
            tuple(
                name
                for name, param in params_by_target.items()
                if param is group.param
            )
            if param.is_meta
            else ()
        )
        lora_transform = (
            LoRATransform(
                group.factors,
                stochastic_rounding=stochastic_rounding,
                target_key=target_key,
            )
            if group.factors
            else None
        )
        dense_transform = (
            DenseDiffTransform(group.dense)
            if group.dense
            else None
        )
        transform = ParameterTransformSequence(
            *(candidate for candidate in (lora_transform, dense_transform) if candidate is not None)
        )
        bias: nn.Parameter | None = None
        if lora_transform is not None and lora_transform.has_bias:
            bias_key = sibling_parameter_name(target_key, "bias")
            bias = params_by_target.get(bias_key)
            if bias is None:
                raise ValueError(
                    f"Cannot merge legacy LoRA bias for {target_key!r}: "
                    f"the model has no base bias parameter {bias_key!r}. "
                    "Use routed LoRA for a bias-less base layer."
                )

            bias_tensor_id = param_tensor_id(bias)
            previous_owner = bias_owner_by_tensor_id.setdefault(
                bias_tensor_id,
                target_key,
            )
            if previous_owner != target_key:
                raise ValueError(
                    f"LoRA targets {previous_owner!r} and {target_key!r} "
                    "resolve to the same tied base-bias backing. Apply only "
                    "one logical target for a tied bias."
                )

        merge_ops.append(
            _MergeOp(
                aliases,
                param,
                bias,
                transform,
                lora_transform,
                dense_transform,
            )
        )

    return merge_ops
