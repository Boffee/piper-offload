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
from dataclasses import dataclass, field
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


@dataclass(slots=True, frozen=True)
class _MergeTarget:
    key: str
    param: nn.Parameter
    tensor_id: tuple[Any, ...]


@dataclass(slots=True)
class _MergeOp:
    weight: _MergeTarget
    factors: list[ScaledLoRAFactor]
    stochastic_rounding: bool
    bias: _MergeTarget | None = field(init=False, default=None)

    @property
    def transform(self) -> LoRATransform:
        return LoRATransform(
            self.factors,
            stochastic_rounding=self.stochastic_rounding,
            target_key=self.weight.key,
        )

    def resolve_bias(
        self,
        params_by_target: dict[str, nn.Parameter],
        bias_owner_by_tensor_id: dict[tuple[Any, ...], str],
    ) -> None:
        """Bind the optional base bias and reject tied-bias ambiguity."""
        if not self.transform.has_bias:
            return

        bias_key = _lora_bias_target_key(self.weight.key)
        bias_param = params_by_target.get(bias_key)
        if bias_param is None:
            raise ValueError(
                f"Cannot merge legacy LoRA bias for {self.weight.key!r}: "
                f"the model has no base bias parameter {bias_key!r}. "
                "Use routed LoRA for a bias-less base layer."
            )

        tensor_id = param_tensor_id(bias_param)
        previous_owner = bias_owner_by_tensor_id.setdefault(
            tensor_id,
            self.weight.key,
        )
        if previous_owner != self.weight.key:
            raise ValueError(
                f"LoRA targets {previous_owner!r} and {self.weight.key!r} "
                "resolve to the same tied base-bias backing. Apply only "
                "one logical target for a tied bias."
            )
        self.bias = _MergeTarget(bias_key, bias_param, tensor_id)

    def validate(self) -> None:
        """Preflight this operation's weight and optional bias."""
        bias_param = None if self.bias is None else self.bias.param
        try:
            self.transform.validate_target(self.weight.param, bias_param)
        except ValueError as exc:
            raise ValueError(
                f"Cannot merge LoRA into {self.weight.key!r}: {exc}",
            ) from exc

    def apply(self) -> None:
        """Apply this operation's weight and optional bias updates."""
        bias_param = None if self.bias is None else self.bias.param
        try:
            self.transform.apply(self.weight.param, bias_param)
        except ValueError as exc:
            raise ValueError(
                f"Cannot merge LoRA into {self.weight.key!r}: {exc}",
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

    merge_ops_by_tensor_id: dict[tuple[Any, ...], _MergeOp] = {}
    for lora, strength in loras:
        for target_key, factor in lora.targets.items():
            param = params_by_target[target_key]
            tensor_id = param_tensor_id(param)
            op = merge_ops_by_tensor_id.get(tensor_id)
            if op is None:
                op = _MergeOp(
                    _MergeTarget(target_key, param, tensor_id),
                    [],
                    stochastic_rounding,
                )
                merge_ops_by_tensor_id[tensor_id] = op
            elif op.weight.key != target_key:
                raise ValueError(
                    f"LoRA targets {op.weight.key!r} and {target_key!r} "
                    f"resolve to the same tied parameter backing. Apply "
                    f"only one name for a tied weight in a single "
                    f"merge_lora() call; otherwise the same base weight "
                    f"would receive multiple logical updates."
                )
            op.factors.append(factor.scaled(strength))

    merge_ops = list(merge_ops_by_tensor_id.values())
    bias_owner_by_tensor_id: dict[tuple[Any, ...], str] = {}
    for op in merge_ops:
        op.resolve_bias(params_by_target, bias_owner_by_tensor_id)

    # Preflight every operation before applying any of them. This catches all
    # expected name, shape, and adapter-capability errors without leaving a
    # permanently half-merged model.
    for op in merge_ops:
        op.validate()

    for op in merge_ops:
        op.apply()

    bias_count = sum(op.bias is not None for op in merge_ops)
    modified_tensor_ids = {op.weight.tensor_id for op in merge_ops}
    modified_tensor_ids.update(
        op.bias.tensor_id
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
