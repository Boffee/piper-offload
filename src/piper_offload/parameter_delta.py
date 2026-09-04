"""Host-backed additive updates for existing model parameters."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import torch
from torch import nn

from .lora import (
    LoRAFactor,
    LoRATransform,
    ScaledLoRAFactor,
    _localize_materialized_weight_factors,
    _materialize_weight_factors,
    _MaterializedWeightFactor,
    _pack_materialized_weight_factors,
    _validate_factor_shapes,
    _validate_materialized_weight_factors,
)
from .pinned_param import PinnedParam
from .seeding import derive_seed
from .tensor_adapter_registry import param_representation, select_adapter
from .tensor_adapters import (
    DenseMergeTargetValidationTensorAdapter,
    DenseMergeTensorAdapter,
    DenseMergeValidationTensorAdapter,
    MergeLocalityTensorAdapter,
    adapter_name,
)

__all__ = [
    "ParameterDelta",
    "ParameterDeltaTransform",
    "ScaledParameterDelta",
]


def _validate_dense_tensor(source: torch.Tensor) -> None:
    if type(source) is not torch.Tensor:
        raise ValueError(f"Dense parameter deltas must be plain torch.Tensor values; got {type(source).__name__}.")
    if source.is_meta:
        raise ValueError("Dense parameter deltas must own physical values, not meta storage.")
    if not source.is_floating_point():
        raise ValueError(f"Dense parameter deltas must be floating-point; got {source.dtype}.")


def _capture_dense_tensor(
    source: torch.Tensor,
    *,
    dtype: torch.dtype | None,
    pin_memory: bool,
) -> PinnedParam:
    _validate_dense_tensor(source)
    tensor = source if dtype is None or source.dtype is dtype else source.to(dtype=dtype)
    return PinnedParam(
        nn.Parameter(tensor, requires_grad=False),
        pin_memory=pin_memory,
    )


@dataclass(slots=True, frozen=True)
class ParameterDelta:
    """One reusable additive update for an exact model parameter.

    ``lora`` stores an optional low-rank contribution and ``dense`` an
    optional full-rank contribution. At least one representation must be
    present. Strength is deliberately extrinsic: binding this resource to an
    activation produces ``strength * (lora.B @ lora.A + dense)`` without
    mutating or copying its host backing. Tensor payload numerical validity is
    the caller's responsibility.
    """

    lora: LoRAFactor | None = None
    dense: PinnedParam | None = None

    def __post_init__(self) -> None:
        if self.lora is None and self.dense is None:
            raise ValueError("ParameterDelta requires a LoRA factor, a dense delta, or both.")
        if self.lora is not None and not isinstance(self.lora, LoRAFactor):
            raise ValueError(f"ParameterDelta lora must be a LoRAFactor; got {type(self.lora).__name__}.")
        if self.dense is not None:
            if not isinstance(self.dense, PinnedParam):
                raise ValueError(f"ParameterDelta dense must be a PinnedParam; got {type(self.dense).__name__}.")
            _validate_dense_tensor(param_representation(self.dense.make_cpu_param()))

    @classmethod
    def from_tensors(
        cls,
        *,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        dense: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        pin_memory: bool = True,
    ) -> Self:
        """Validate and capture an optional LoRA pair and dense contribution."""
        if (a is None) != (b is None):
            raise ValueError("ParameterDelta requires both LoRA A and LoRA B when either factor is provided.")
        lora = (
            None
            if a is None or b is None
            else LoRAFactor.from_tensors(
                a,
                b,
                dtype=dtype,
                pin_memory=pin_memory,
            )
        )
        dense_backing = (
            None
            if dense is None
            else _capture_dense_tensor(
                dense,
                dtype=dtype,
                pin_memory=pin_memory,
            )
        )
        return cls(lora=lora, dense=dense_backing)

    @property
    def cache_bytes(self) -> int:
        """Host-backing bytes held by every representation of this delta."""
        lora_bytes = 0 if self.lora is None else self.lora.cache_bytes
        dense_bytes = 0 if self.dense is None else self.dense.cache_bytes
        return lora_bytes + dense_bytes

    def scaled(self, strength: float) -> ScaledParameterDelta:
        """Associate this reusable update with one application strength."""
        return ScaledParameterDelta(self, strength)


@dataclass(slots=True, frozen=True)
class ScaledParameterDelta:
    """A parameter delta bound to an extrinsic application strength."""

    delta: ParameterDelta
    strength: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.strength):
            raise ValueError(f"Parameter delta strength must be finite; got {self.strength}.")


@dataclass(slots=True, frozen=True)
class _DenseMergePlan:
    """Validated logical sources and target representation for dense merge."""

    adapter: DenseMergeTensorAdapter[Any, Any]
    dense_sources: tuple[tuple[torch.Tensor, float], ...]
    factors: tuple[_MaterializedWeightFactor, ...]
    logical_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    offsets: tuple[int, ...]
    compute_dtype: torch.dtype


class ParameterDeltaTransform:
    """Apply combined low-rank and dense updates to one existing parameter.

    Factor-only transforms delegate to :class:`LoRATransform`. When any dense
    term is present, every dense and low-rank contribution is staged into one
    full-rank logical update and delegated to the target adapter's dense merge
    capability. Quantized targets can therefore re-encode the base only once.
    """

    __slots__ = (
        "_deltas",
        "_dense_plan",
        "_has_dense",
        "_lora_factors",
        "_lora_transform",
        "_merge_index",
        "_stochastic_rounding",
        "_target_key",
    )

    def __init__(
        self,
        deltas: Sequence[ScaledParameterDelta],
        *,
        stochastic_rounding: bool = False,
        target_key: str = "",
    ) -> None:
        if not deltas:
            raise ValueError("ParameterDeltaTransform requires a delta.")
        self._deltas = tuple(deltas)
        self._has_dense = any(
            scaled.delta.dense is not None for scaled in self._deltas
        )
        if stochastic_rounding and not target_key:
            raise ValueError(
                "Stochastic ParameterDeltaTransform requires a non-empty target_key."
            )
        self._lora_factors = tuple(
            ScaledLoRAFactor(scaled.delta.lora, scaled.strength)
            for scaled in self._deltas
            if scaled.delta.lora is not None
        )
        self._lora_transform = (
            None
            if not self._lora_factors or self._has_dense
            else LoRATransform(
                self._lora_factors,
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
        if not self._has_dense:
            # Factor-only application is completely owned by LoRATransform.
            lora_transform = self._lora_transform
            assert lora_transform is not None
            lora_transform.validate_parameter(param)
            return
        dense_sources = self._materialize_dense_sources()
        assert dense_sources

        target = param_representation(param)
        if target.is_meta:
            raise ValueError(
                "Dense parameter deltas require an existing plain floating-point target "
                "or structured target with dense merge support; "
                f"got {type(target).__name__} on {target.device}."
            )
        adapter = _select_dense_merge_adapter(target)
        shape = adapter.logical_shape(target)
        dtype = adapter.compute_dtype(target)
        for source, _strength in dense_sources:
            if tuple(source.shape) != shape:
                raise ValueError(
                    "Dense parameter delta shape mismatch: "
                    f"source shape is {tuple(source.shape)}, target shape is {shape}."
                )

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

        if isinstance(adapter, MergeLocalityTensorAdapter):
            local_shape, offsets = adapter.merge_local_shape_and_offsets(target)
        else:
            offsets = tuple(0 for _ in shape)
            local_shape = shape

        _validate_factor_shapes(self._lora_factors, shape)
        factors = _materialize_weight_factors(self._lora_factors)
        _validate_materialized_weight_factors(factors)
        if factors:
            factors = _localize_materialized_weight_factors(
                factors,
                out_range=(offsets[0], local_shape[0]),
                in_range=(offsets[1], local_shape[1]),
            )

        plan = _DenseMergePlan(
            adapter=adapter,
            dense_sources=tuple(dense_sources),
            factors=tuple(factors),
            logical_shape=shape,
            local_shape=local_shape,
            offsets=offsets,
            compute_dtype=dtype,
        )
        if requires_update_validation:
            assert isinstance(adapter, DenseMergeValidationTensorAdapter)
            staged = self._stage_update_for_plan(param, plan)
            adapter.validate_dense_merge(
                target,
                staged,
                1.0,
                rounding_seed=rounding_seed,
            )
        self._dense_plan = plan

    def apply_parameter(self, param: nn.Parameter) -> None:
        """Stage the complete update, then mutate the base parameter once."""
        if not self._has_dense:
            lora_transform = self._lora_transform
            assert lora_transform is not None
            lora_transform.apply_parameter(param)
            return

        plan = self._dense_plan
        if plan is None:
            raise RuntimeError("Dense parameter delta target must be validated before application.")
        dense_update = self._stage_update_for_plan(param, plan)
        plan.adapter.merge_dense_(
            param_representation(param),
            dense_update,
            1.0,
            rounding_seed=self._rounding_seed(),
        )
        self._merge_index += 1

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
                "Dense parameter delta application target no longer has a "
                "registered adapter."
            ) from exc
        if (
            target.is_meta
            or type(current_adapter) is not type(plan.adapter)
            or current_adapter.compute_dtype(target) is not plan.compute_dtype
            or current_adapter.logical_shape(target) != plan.logical_shape
        ):
            raise RuntimeError(
                "Dense parameter delta application requires a physical tensor "
                "matching the validated target representation."
            )

        update = torch.zeros(
            plan.local_shape,
            device=target.device,
            dtype=plan.compute_dtype,
        )
        for source, strength in plan.dense_sources:
            local_source = _local_dense_view(
                source,
                offsets=plan.offsets,
                local_shape=plan.local_shape,
            )
            staged = local_source.to(
                device=target.device,
                dtype=plan.compute_dtype,
                non_blocking=True,
            )
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
            logical_shape=plan.local_shape,
            compute_dtype=plan.compute_dtype,
        )
        update.addmm_(packed_b, packed_a)

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
    local = source
    for dim, (offset, size) in enumerate(
        zip(offsets, local_shape, strict=True)
    ):
        local = local.narrow(dim, offset, size)
    return local.contiguous()
