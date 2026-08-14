"""Permanent LoRA merge into model weights.

Merges LoRA deltas directly into model parameters, supporting tensors
whose adapter exposes either dense in-place ``addmm_`` or a staged LoRA
merge. Quantized adapters own their encoding path and may select a
format-specific kernel or a dequantize/requantize fallback; requantized
merges are lossy but standard practice for permanent LoRA merges into
quantized bases.

This uses the same :class:`LoRATransform` as activation merge. Permanent merge
applies its complete weight-and-optional-bias operation to resident model
parameters; activation merge invokes its partial operations after individual
parameter copies.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn

from .lora import (
    LoRA,
    LoRATransform,
    ScaledLoRAFactor,
    _lora_bias_target_key,
)
from .tensor_adapter_registry import param_tensor_id

logger = logging.getLogger(__name__)

__all__ = ["merge_lora"]


type _FactorGroup = tuple[str, nn.Parameter, list[ScaledLoRAFactor]]


@dataclass(slots=True, frozen=True)
class _MergeOp:
    target_key: str
    weight: nn.Parameter
    bias: nn.Parameter | None
    transform: LoRATransform

    def validate(self) -> None:
        """Preflight this operation's weight and optional bias."""
        try:
            self.transform.validate_target(self.weight, self.bias)
        except ValueError as exc:
            raise ValueError(
                f"Cannot merge LoRA into {self.target_key!r}: {exc}",
            ) from exc

    def apply(self) -> None:
        """Apply this operation's weight and optional bias updates."""
        try:
            self.transform.apply(self.weight, self.bias)
        except ValueError as exc:
            raise ValueError(
                f"Cannot merge LoRA into {self.target_key!r}: {exc}",
            ) from exc


def merge_lora(
    model: nn.Module,
    loras: Sequence[tuple[LoRA, float]],
    *,
    stochastic_rounding: bool = True,
) -> int:
    """Merge one or more LoRAs into model parameters in-place.

    Returns the number of unique parameters that were modified. Exact-zero
    strengths are inactive and do not create merge operations. Merge reads
    immutable host factor backing, so the same LoRA may also serve other
    merge or routed uses. All active target names and merge capabilities are
    validated before any parameter is modified. Quantized targets use
    terminal-code stochastic rounding by default so sub-step LoRA updates are
    not systematically rounded away; pass ``stochastic_rounding=False`` for
    deterministic rounding. Dense targets always use their ordinary exact
    ``addmm_`` update.
    """
    if len({id(lora) for lora, _strength in loras}) != len(loras):
        raise ValueError("merge_lora() does not accept the same LoRA instance more than once")

    for lora, _strength in loras:
        if not isinstance(lora, LoRA):
            raise TypeError("merge_lora() expects LoRA instances")

    # Validate the request structure above before treating exact-zero
    # contributors as absent. Filtering here avoids target lookup, staging,
    # validation, and requantization for work that cannot modify a parameter.
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
        for target_key in lora.targets
        if target_key not in params_by_target
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

    # Preflight every operation before applying any of them. This catches all
    # expected name, shape, and adapter-capability errors without leaving a
    # permanently half-merged model.
    for op in merge_ops:
        op.validate()

    for op in merge_ops:
        op.apply()

    bias_count = sum(op.bias is not None for op in merge_ops)
    modified_tensor_ids = {param_tensor_id(op.weight) for op in merge_ops}
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
        sum(len(lora.targets) for lora, _ in loras),
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
    """Group factors, resolve targets, and reject ambiguous parameter ties."""
    factor_groups_by_tensor_id: dict[tuple[Any, ...], _FactorGroup] = {}
    for lora, strength in loras:
        for target_key, factor in lora.targets.items():
            weight = params_by_target[target_key]
            tensor_id = param_tensor_id(weight)
            group = factor_groups_by_tensor_id.get(tensor_id)
            if group is None:
                factors: list[ScaledLoRAFactor] = []
                group = (target_key, weight, factors)
                factor_groups_by_tensor_id[tensor_id] = group
            else:
                existing_target_key, _existing_weight, factors = group
                if existing_target_key != target_key:
                    raise ValueError(
                        f"LoRA targets {existing_target_key!r} and "
                        f"{target_key!r} resolve to the same tied parameter "
                        "backing. Apply only one name for a tied weight in a "
                        "single merge_lora() call; otherwise the same base "
                        "weight would receive multiple logical updates."
                    )
            factors.append(factor.scaled(strength))

    merge_ops: list[_MergeOp] = []
    bias_owner_by_tensor_id: dict[tuple[Any, ...], str] = {}
    for target_key, weight, factors in factor_groups_by_tensor_id.values():
        transform = LoRATransform(
            factors,
            stochastic_rounding=stochastic_rounding,
            target_key=target_key,
        )
        bias: nn.Parameter | None = None
        if transform.has_bias:
            bias_key = _lora_bias_target_key(target_key)
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

        merge_ops.append(_MergeOp(target_key, weight, bias, transform))

    return merge_ops
