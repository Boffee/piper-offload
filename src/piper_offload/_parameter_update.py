"""Shared source selection, validation, and arithmetic for parameter updates."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .host_param import HostParam
from .lora import (
    LoRATransform,
    _localize_materialized_weight_factors,
    _materialize_weight_factors,
    _MaterializedWeightFactor,
    _pack_materialized_weight_factors,
    _validate_factor_shapes,
    _validate_materialized_weight_factors,
)
from .seeding import derive_seed
from .tensor_adapter_registry import param_representation, param_tensor_id, select_adapter
from .tensor_adapters import (
    DenseMergeTargetValidationTensorAdapter,
    DenseMergeTensorAdapter,
    DenseMergeValidationTensorAdapter,
    LogicalShapeTensorAdapter,
    MergeLocalityTensorAdapter,
    RegularAdapter,
    adapter_name,
    transfer_,
    validate_broadcast_merge,
)

if TYPE_CHECKING:
    from .parameter_delta import ScaledParameterDelta
    from .parameter_value import ScaledParameterValue


def merge_target_id(param: nn.Parameter) -> tuple[Any, ...]:
    """Opaque tensor-adapter identity used for merge validation and installation."""
    try:
        return param_tensor_id(param)
    except NotImplementedError:
        # Leave unsupported representations to arithmetic validation. They
        # cannot participate in storage-based alias detection.
        return ("__unsupported_param__", id(param))


def validate_merge_target_aliases(
    params_by_name: Mapping[str, nn.Parameter],
    target_names: Iterable[str],
) -> None:
    """Require one requested name per backing across active adapter updates.

    Call after resolving absent names and before skipping unpopulated targets.
    Repeated contributions under the same name are allowed. Physical aliases
    use tensor-adapter storage identity; meta parameters use object identity.
    """
    names_by_identity: dict[tuple[Any, ...], str] = {}
    for name in target_names:
        identity = merge_target_id(params_by_name[name])
        previous = names_by_identity.setdefault(identity, name)
        if previous != name:
            raise ValueError(
                f"Tied parameter aliases {previous!r} and {name!r} "
                "cannot both be adapter targets; use one name per parameter."
            )


@dataclass(slots=True, frozen=True)
class _DenseMergePlan:
    """Prepared host views and target contract for repeated dense merges.

    Dense sources and factors already describe the target-local region and
    share immutable adapter backing. Their common ``update_shape`` can be
    smaller than the target's ``logical_shape`` when broadcasting.
    """

    adapter: DenseMergeTensorAdapter[Any, Any]
    dense_sources: tuple[tuple[torch.Tensor, float], ...]
    factors: tuple[_MaterializedWeightFactor, ...]
    logical_shape: tuple[int, ...]
    update_shape: tuple[int, ...]
    compute_dtype: torch.dtype


class ParameterUpdateTransform:
    """Repeated arithmetic on an already-loaded parameter source.

    Callers supply at least one delta or a non-unit finite ``source_scale``.
    Each delta has its own strength; source scaling applies only to the loaded
    weight. LoRA-only updates retain factorized merge at any source scale.
    Value-only scaling and updates containing dense deltas use dense merge.
    Both paths combine source scaling and updates before one terminal encoding.
    This operation owns validation, staging, and the repeated-merge seed, while
    callers own source selection, allocation, installation, and transfer leases.
    """

    __slots__ = (
        "_deltas",
        "_dense_plan",
        "_lora_factors",
        "_lora_transform",
        "_merge_index",
        "_source_scale",
        "_stochastic_rounding",
        "_target_key",
        "_uses_dense_merge",
    )

    def __init__(
        self,
        deltas: Sequence[ScaledParameterDelta],
        *,
        source_scale: float,
        stochastic_rounding: bool,
        target_key: str,
    ) -> None:
        self._source_scale = source_scale
        self._deltas = tuple(deltas)
        if stochastic_rounding and not target_key:
            raise ValueError(
                "Stochastic parameter updates require a non-empty target_key."
            )
        self._lora_factors = tuple(
            scaled.delta.lora.scaled(scaled.strength)
            for scaled in self._deltas
            if scaled.delta.lora is not None
        )
        self._uses_dense_merge = not self._lora_factors or any(
            scaled.delta.dense is not None for scaled in self._deltas
        )
        self._lora_transform = (
            None
            if self._uses_dense_merge
            else LoRATransform(
                self._lora_factors,
                beta=source_scale,
                stochastic_rounding=stochastic_rounding,
                target_key=target_key,
            )
        )
        self._stochastic_rounding = stochastic_rounding
        self._target_key = target_key
        self._merge_index = 0
        self._dense_plan: _DenseMergePlan | None = None

    def validate_parameter(self, param: nn.Parameter) -> None:
        """Validate every contribution before preparing repeated application."""
        self._dense_plan = None
        if not self._uses_dense_merge:
            # Factor-only application is completely owned by LoRATransform.
            lora_transform = self._lora_transform
            assert lora_transform is not None
            lora_transform.validate_parameter(param)
            return
        dense_sources = self._materialize_dense_sources()

        target = param_representation(param)
        if target.is_meta:
            raise ValueError(
                "Parameter updates require an existing plain floating-point target "
                "or structured target with dense merge support; "
                f"got {type(target).__name__} on {target.device}."
            )
        adapter = _select_dense_merge_adapter(target)
        shape = adapter.logical_shape(target)
        dtype = adapter.compute_dtype(target)
        for source, _strength in dense_sources:
            if torch.broadcast_shapes(source.shape, shape) != shape:
                raise ValueError(
                    "Dense parameter delta would expand the target: "
                    f"source shape is {tuple(source.shape)}, target shape is {shape}."
                )

        _validate_factor_shapes(self._lora_factors, shape)
        source_shapes = [tuple(source.shape) for source, _strength in dense_sources]
        source_shapes.extend(factor.produced_shape for factor in self._lora_factors)
        if any(source_shape != shape for source_shape in source_shapes):
            validate_broadcast_merge(adapter, target)

        rounding_seed = self._rounding_seed()
        requires_update_validation = isinstance(
            adapter,
            DenseMergeValidationTensorAdapter,
        )
        if isinstance(adapter, DenseMergeTargetValidationTensorAdapter):
            requires_update_validation = adapter.validate_dense_merge_target(
                target,
                rounding_seed=rounding_seed,
            )
        if requires_update_validation and not isinstance(
            adapter,
            DenseMergeValidationTensorAdapter,
        ):
            raise ValueError(
                f"{adapter_name(adapter)} requested staged dense-update validation "
                "without implementing validate_dense_merge()."
            )

        factors = _materialize_weight_factors(self._lora_factors)
        _validate_materialized_weight_factors(factors)
        local_shape = shape
        if isinstance(adapter, MergeLocalityTensorAdapter):
            local_shape, offsets = adapter.merge_local_shape_and_offsets(target)
            dense_sources = [
                (
                    _local_dense_view(source.expand(shape), offsets=offsets, local_shape=local_shape),
                    strength,
                )
                for source, strength in dense_sources
            ]
            if factors:
                factors = _localize_materialized_weight_factors(
                    factors,
                    out_range=(offsets[-2], local_shape[-2]),
                    in_range=(offsets[-1], local_shape[-1]),
                )

        update_shape = (
            tuple(torch.broadcast_shapes(
                *(source.shape for source, _strength in dense_sources),
                *((factor.b.shape[0], factor.a.shape[1]) for factor in factors),
            ))
            if dense_sources or factors
            else local_shape
        )

        plan = _DenseMergePlan(
            adapter=adapter,
            dense_sources=tuple(dense_sources),
            factors=tuple(factors),
            logical_shape=shape,
            update_shape=update_shape,
            compute_dtype=dtype,
        )
        if requires_update_validation:
            assert isinstance(adapter, DenseMergeValidationTensorAdapter)
            staged = self._stage_update_for_plan(param, plan)
            adapter.validate_dense_merge(
                target,
                staged,
                1.0,
                beta=self._source_scale,
                rounding_seed=rounding_seed,
            )
        self._dense_plan = plan

    def apply_parameter(self, param: nn.Parameter) -> None:
        """Stage all deltas, then scale and update the loaded parameter."""
        if not self._uses_dense_merge:
            lora_transform = self._lora_transform
            assert lora_transform is not None
            lora_transform.apply_parameter(param)
            return

        plan = self._dense_plan
        if plan is None:
            raise RuntimeError("Parameter update target must be validated before application.")
        dense_update = self._stage_update_for_plan(param, plan)
        plan.adapter.merge_dense_(
            param_representation(param),
            dense_update,
            1.0,
            beta=self._source_scale,
            rounding_seed=self._rounding_seed(),
        )
        self._merge_index += 1

    def storage_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return all physical host tensors read while staging the delta."""
        tensors: list[torch.Tensor] = []
        for scaled in self._deltas:
            delta = scaled.delta
            if delta.lora is not None:
                tensors.extend(delta.lora.a.storage_tensors())
                tensors.extend(delta.lora.b.storage_tensors())
            if delta.dense is not None:
                tensors.extend(delta.dense.storage_tensors())
        return tuple(tensors)

    def _materialize_dense_sources(self) -> list[tuple[torch.Tensor, float]]:
        sources: list[tuple[torch.Tensor, float]] = []
        for bound in self._deltas:
            backing = bound.delta.dense
            if backing is None:
                continue
            source = param_representation(backing.make_cpu_param())
            assert type(source) is torch.Tensor
            assert source.device.type == "cpu"
            sources.append((source, bound.strength))
        return sources

    def _stage_update_for_plan(
        self,
        param: nn.Parameter,
        plan: _DenseMergePlan,
    ) -> torch.Tensor:
        """Materialize one local logical update from a validated plan."""
        target = param_representation(param)
        try:
            current_adapter = select_adapter(target)
        except NotImplementedError as exc:
            raise RuntimeError(
                "Parameter update application target no longer has a "
                "registered adapter."
            ) from exc
        if (
            target.is_meta
            or current_adapter.compute_dtype(target) is not plan.compute_dtype
            or not isinstance(current_adapter, LogicalShapeTensorAdapter)
            or current_adapter.logical_shape(target) != plan.logical_shape
        ):
            raise RuntimeError(
                "Parameter update application requires a physical tensor "
                "matching the validated target representation."
            )

        update = torch.zeros(
            plan.update_shape,
            device=target.device,
            dtype=plan.compute_dtype,
        )
        for source, strength in plan.dense_sources:
            staged = torch.empty(source.shape, dtype=plan.compute_dtype, device=target.device)
            transfer_(staged, source, non_blocking=True)
            update.add_(staged, alpha=strength)
        self._accumulate_lora(update, plan)
        return update

    @staticmethod
    def _accumulate_lora(update: torch.Tensor, plan: _DenseMergePlan) -> None:
        """Accumulate all logical low-rank terms into one dense update."""
        if not plan.factors:
            return
        packed_a, packed_b = _pack_materialized_weight_factors(
            update,
            plan.factors,
            compute_dtype=plan.compute_dtype,
        )
        RegularAdapter.merge_lora_(update, packed_b, packed_a, 1.0)

    def _rounding_seed(self) -> int | None:
        if not self._stochastic_rounding:
            return None
        return derive_seed(self._target_key, self._merge_index)


def _select_dense_merge_adapter(
    target: torch.Tensor,
) -> DenseMergeTensorAdapter[Any, Any]:
    try:
        adapter = select_adapter(target)
    except NotImplementedError as exc:
        raise ValueError(
            f"Tensor type {type(target).__name__} has no registered tensor adapter. "
            "Merge requires a tensor adapter with dense merge support."
        ) from exc
    if not isinstance(adapter, DenseMergeTensorAdapter):
        raise ValueError(
            f"{adapter_name(adapter)} does not support dense parameter merge."
        )
    dtype = adapter.compute_dtype(target)
    if not dtype.is_floating_point:
        raise ValueError(
            "Dense parameter merge requires a floating-point compute dtype, "
            f"got {dtype}."
        )
    return adapter


def _local_dense_view(
    source: torch.Tensor,
    *,
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> torch.Tensor:
    """Keep a local view of the backing that transfer_ resolves under its lease."""
    local = source
    for dim, (offset, size) in enumerate(
        zip(offsets, local_shape, strict=True)
    ):
        local = local.narrow(dim, offset, size)
    return local


@dataclass(frozen=True, slots=True)
class ParameterUpdatePlan:
    """An optional replacement source and its repeated arithmetic.

    Validate against the destination before any model mutation. The application
    path owns materialization, transfer leases, and installation; this plan
    neither captures nor mutates host backing.
    """

    source: HostParam | None
    update: ParameterUpdateTransform | None

    def validate_parameter(self, param: nn.Parameter) -> None:
        if self.source is not None:
            _validate_value_target(self.source, param)
            param = self.source.make_cpu_param()
        if self.update is not None:
            self.update.validate_parameter(param)

    def apply_parameter(self, param: nn.Parameter) -> None:
        """Apply arithmetic, or check an exact replacement's loaded representation."""
        if self.update is not None:
            self.update.apply_parameter(param)
            return
        source = self.source
        assert source is not None
        target = param_representation(param)
        adapter = select_adapter(target)
        if (
            target.is_meta
            or adapter.compute_dtype(target) is not source.compute_dtype
            or not isinstance(adapter, LogicalShapeTensorAdapter)
            or adapter.logical_shape(target) != source.logical_shape
        ):
            raise RuntimeError(
                "Parameter value update requires physical storage matching "
                "the validated replacement source."
            )


def plan_parameter_update(
    deltas: Sequence[ScaledParameterDelta],
    *,
    value: ScaledParameterValue | None = None,
    stochastic_rounding: bool,
    target_key: str,
) -> ParameterUpdatePlan:
    """Select one source and combine all arithmetic before validation."""
    source = None if value is None else value.value.backing
    source_scale = 1.0 if value is None else value.effective_strength
    update = (
        ParameterUpdateTransform(
            deltas,
            source_scale=source_scale,
            stochastic_rounding=stochastic_rounding,
            target_key=target_key,
        )
        if deltas or source_scale != 1.0
        else None
    )
    return ParameterUpdatePlan(source, update)


def _validate_value_target(backing: HostParam, param: nn.Parameter) -> None:
    """Validate a replacement against its frozen meta destination slot."""
    target = param_representation(param)
    if (
        type(target) is not torch.Tensor
        or not target.is_meta
        or target.layout is not torch.strided
    ):
        raise ValueError(
            "Parameter values require a plain floating-point meta target; "
            f"got {type(target).__name__} on {target.device}."
        )
    if not target.is_floating_point():
        raise ValueError(
            "Parameter values require a floating-point meta target; "
            f"got {target.dtype}."
        )
    if param.requires_grad:
        raise ValueError(
            "Parameter values are inference-only and require requires_grad=False."
        )

    logical_shape = backing.logical_shape
    if logical_shape != tuple(target.shape):
        raise ValueError(
            "Parameter value shape mismatch: "
            f"source shape is {logical_shape}, "
            f"target shape is {tuple(target.shape)}."
        )
