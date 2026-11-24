"""Permanent adapter merge into model weights.

Merges additive parameter deltas directly into existing model parameters and
materializes dense or prequantized parameter values for meta targets. Plain
floating-point targets support combined low-rank and full-rank deltas.
Quantized adapters own their factorized and dense encoding paths and may select
a format-specific kernel or a dequantize/requantize fallback. Mixed deltas and
scaled quantized values encode the target once.

Permanent and activation merge share source selection and arithmetic planning.
Permanent merge applies updates to resident model parameters; activation merge
invokes them after individual parameter copies.
"""

import contextlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ._parameter_update import ParameterUpdatePlan, merge_target_id, plan_merge_targets
from .adapter import Adapter, AdapterTargetUpdates, active_adapter_pairs, group_adapter_updates
from .host_module import install_parameters
from .pin_manager import host_pin_manager
from .tensor_adapter_registry import (
    param_representation,
    select_adapter,
)
from .tensor_adapters import (
    PermanentUpdateValidationTensorAdapter,
    clone_host_state,
    host_state_is_owned,
    view_host_state,
)

logger = logging.getLogger(__name__)

__all__ = ["merge_adapter"]


def _validate_permanent_update(adapter: object) -> None:
    if isinstance(adapter, PermanentUpdateValidationTensorAdapter):
        adapter.validate_permanent_update()


@dataclass(slots=True, frozen=True)
class _MergeOp:
    aliases: tuple[str, ...]
    param: nn.Parameter
    plan: ParameterUpdatePlan

    def prepare(self, params_by_name: Mapping[str, nn.Parameter]) -> nn.Parameter:
        """Validate the update and resolve its writable target before any model mutation."""
        trainable_aliases = [name for name in self.aliases if params_by_name[name].requires_grad]
        if trainable_aliases:
            raise ValueError(
                "Permanent adapter merge requires frozen parameters (requires_grad=False); "
                f"found trainable parameter(s): {', '.join(repr(name) for name in trainable_aliases)}."
            )
        if self.plan.update is not None:
            adapter = (
                self.plan.source.adapter
                if self.plan.source is not None
                else select_adapter(param_representation(self.param))
            )
            _validate_permanent_update(adapter)
        self.plan.validate_parameter(self.param)

        return self.plan.source.clone_cpu_param() if self.plan.source is not None else _writable_parameter(self.param)

    def apply(self, model: nn.Module, param: nn.Parameter) -> None:
        """Apply to the prepared target and install a replacement under all tied names."""
        if self.plan.update is not None:
            self.plan.update.apply_parameter(param)
        if param is not self.param:
            install_parameters(model, dict.fromkeys(self.aliases, param))


def _writable_parameter(param: nn.Parameter) -> nn.Parameter:
    """Keep an owned target in place; copy a borrowed representation before writing."""
    if param.device.type != "cpu":
        return param
    tensor = param_representation(param)
    adapter = select_adapter(tensor)
    state = view_host_state(adapter, tensor)
    if not host_state_is_owned(adapter, state):
        state = clone_host_state(adapter, tensor, state)
    target = adapter.cpu_param(state)
    if adapter.tensor_id(tensor) == adapter.tensor_id(param_representation(target)):
        return param
    return target


def merge_adapter(
    model: nn.Module,
    adapters: Sequence[tuple[Adapter, float]],
    *,
    stochastic_rounding: bool = True,
) -> int:
    """Merge one or more adapters into model parameters in-place.

    Returns the number of unique parameters that were modified. Exact-zero
    strengths are inactive. If no adapters are active, return zero before
    visiting model parameters. Merge requires frozen target parameters
    (``requires_grad=False``) across every tied alias and reads immutable host
    backing, so the same adapter may also serve other merge or routed uses.
    Target names and capabilities of planned merges are validated before any
    parameter is modified. An adapter constructed with
    ``allow_partial_targets=True`` ignores targets absent from this model
    and skips deltas to meta parameters without an active value provider.
    A strict delta on the same target still requires a source. Skipped
    targets stay meta and do not count as modified parameters.
    Quantized targets use terminal-code stochastic rounding by default so
    sub-step additive updates are not systematically rounded away; pass
    ``stochastic_rounding=False`` for deterministic rounding. Parameter values
    populate frozen floating-point meta targets according to their strength
    policy and can compose with deltas from other adapters on the same named
    parameter. Value scaling and all deltas combine before terminal encoding.
    Every registered physical value retains its source representation.
    LoRA-only compositions use factorized merge support at any value scale;
    value-only scaling and compositions with dense deltas require dense merge
    support. A populated meta target is replaced by one independent frozen CPU
    parameter, preserving any tied aliases of the original parameter.
    A target that is a view into a file mapping is likewise
    merged into an independent copy installed under every tied name: Piper
    never writes into a file mapping, because the pin manager
    may return a mapping's pages to the file.
    """
    active = active_adapter_pairs(adapters)
    if not active:
        return 0
    params_by_target = dict(model.named_parameters(remove_duplicate=False))
    updates = group_adapter_updates(active, params_by_target.keys())
    merge_ops = _build_merge_ops(params_by_target, updates, stochastic_rounding=stochastic_rounding)

    # Preflight every operation before applying any of them. This catches all
    # expected name, shape, and adapter-capability errors without leaving a
    # permanently half-merged model.
    # Validation and application both stage adapter sources onto a CUDA
    # target asynchronously, so the sources are leased, pageable, until the
    # device has synchronized; a failed step may already have enqueued copies.
    # If the synchronization itself fails the context is unusable and no
    # later call could retry it, so the lease closes with this function.
    devices = {op.param.device for op in merge_ops if op.param.device.type == "cuda"}
    sources = (
        tensor
        for op in merge_ops
        if op.plan.update is not None
        for tensor in op.plan.update.storage_tensors()
    )
    with host_pin_manager.acquire(sources, pin=False) if devices else contextlib.nullcontext():
        try:
            targets = [op.prepare(params_by_target) for op in merge_ops]
            for op, target in zip(merge_ops, targets, strict=True):
                op.apply(model, target)
        finally:
            for device in devices:
                torch.cuda.synchronize(device)

    logger.info("merge_adapter: merged %d unique parameters", len(merge_ops))
    return len(merge_ops)


def _build_merge_ops(
    params_by_target: dict[str, nn.Parameter],
    updates: dict[str, AdapterTargetUpdates],
    *,
    stochastic_rounding: bool,
) -> list[_MergeOp]:
    """Plan one operation per updated backing, preserving all its aliases."""
    targets = list(plan_merge_targets(params_by_target, updates, stochastic_rounding=stochastic_rounding))
    if not targets:
        return []

    # A replacement parameter must reach every name sharing the target's
    # backing, including distinct wrapper objects over the same storage.
    names_by_tie: dict[tuple[Any, ...], list[str]] = {}
    for name, param in params_by_target.items():
        names_by_tie.setdefault(merge_target_id(param), []).append(name)
    return [_MergeOp(tuple(names_by_tie[merge_target_id(param)]), param, plan) for _, param, plan in targets]
