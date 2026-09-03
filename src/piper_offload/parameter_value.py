"""Physical replacement values for frozen meta parameters."""

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from .pinned_param import PinnedParam
from .seeding import derive_seed
from .tensor_adapter_registry import param_representation, select_adapter
from .tensor_adapters import (
    DenseMergeTargetValidationTensorAdapter,
    DenseMergeTensorAdapter,
    DenseMergeValidationTensorAdapter,
    DequantizeTensorAdapter,
    RegularAdapter,
    TensorAdapter,
    adapter_name,
)

__all__ = [
    "ParameterValue",
    "ParameterValueTransform",
    "ScaledParameterValue",
]


def _select_value_adapter(
    source: torch.Tensor,
) -> TensorAdapter[Any, Any]:
    """Return the adapter after validating representation-level invariants."""
    if source.is_meta:
        raise ValueError("Parameter values must own physical values, not meta storage.")
    try:
        adapter = select_adapter(source)
    except NotImplementedError as exc:
        raise ValueError(
            f"Parameter value tensor type {type(source).__name__} has no "
            "registered tensor adapter."
        ) from exc
    compute_dtype = adapter.compute_dtype(source)
    if not compute_dtype.is_floating_point:
        raise ValueError(
            "Parameter values must have a floating-point compute dtype; "
            f"got {compute_dtype}."
        )
    return adapter


def _validate_value_representation(
    source: torch.Tensor,
) -> TensorAdapter[Any, Any]:
    """Fully validate one final physical value representation."""
    adapter = _select_value_adapter(source)
    if isinstance(adapter, RegularAdapter) and source.numel():
        finite_source = (
            source.float()
            if source.element_size() == 1
            else source
        )
        if not bool(torch.isfinite(finite_source).all()):
            raise ValueError("Parameter values must contain only finite values.")
    return adapter


@dataclass(slots=True, frozen=True)
class ParameterValue:
    """One authoritative physical replacement for an exact parameter name.

    The backing defines active dtype, layout, quantization metadata, and
    copied bytes. Adapter strength does not scale a complete value unless
    ``scale_with_strength=True`` is explicitly requested.
    """

    backing: PinnedParam
    scale_with_strength: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.backing, PinnedParam):
            raise ValueError(
                "ParameterValue backing must be a PinnedParam; "
                f"got {type(self.backing).__name__}."
            )
        if self.backing.requires_grad:
            raise ValueError("Parameter values are inference-only and must be frozen.")
        _validate_value_representation(
            param_representation(self.backing.make_cpu_param())
        )

    @classmethod
    def from_tensor(
        cls,
        source: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        pin_memory: bool = True,
        scale_with_strength: bool = False,
    ) -> ParameterValue:
        """Validate and capture one physical replacement representation."""
        if not isinstance(source, torch.Tensor):
            raise ValueError(
                "Parameter values must be tensor representations; "
                f"got {type(source).__name__}."
            )
        if issubclass(type(source), nn.Parameter) and source.requires_grad:
            raise ValueError("Parameter values are inference-only and must be frozen.")
        # Inspect only the source representation and dtype policy here. The
        # final converted and pinned representation is fully validated once by
        # ``ParameterValue.__post_init__``.
        adapter = _select_value_adapter(source)
        compute_dtype = adapter.compute_dtype(source)
        if isinstance(adapter, RegularAdapter):
            tensor = (
                source
                if dtype is None or source.dtype is dtype
                else source.to(dtype=dtype)
            )
        else:
            if dtype is not None and dtype is not compute_dtype:
                raise ValueError(
                    "A structured parameter value is already encoded for "
                    f"compute dtype {compute_dtype}; dtype={dtype} would discard "
                    "its representation. Prequantize the value with the desired "
                    "logical dtype instead."
                )
            tensor = source
        parameter = (
            cast(nn.Parameter, tensor)
            if issubclass(type(tensor), nn.Parameter)
            else nn.Parameter(tensor, requires_grad=False)
        )
        return cls(
            PinnedParam(parameter, pin_memory=pin_memory),
            scale_with_strength=scale_with_strength,
        )

    @property
    def cache_bytes(self) -> int:
        return self.backing.cache_bytes

    def scaled(self, strength: float) -> ScaledParameterValue:
        return ScaledParameterValue(self, strength)


@dataclass(slots=True, frozen=True)
class ScaledParameterValue:
    """A parameter value bound to one activation's adapter strength."""

    value: ParameterValue
    strength: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.strength):
            raise ValueError(
                f"Parameter value strength must be finite; got {self.strength}."
            )

    @property
    def effective_strength(self) -> float:
        """Strength applied to the complete value after policy resolution."""
        return self.strength if self.value.scale_with_strength else 1.0


@dataclass(slots=True, frozen=True)
class _ParameterValuePlan:
    """Validated replacement representation and optional scaling."""

    effective_strength: float
    target_layout: tuple[object, object]


class ParameterValueTransform:
    """Validate a replacement and optionally scale it after loading.

    Exact values have no update. Non-unit strength scaling dequantizes the
    freshly loaded representation and merges ``(strength - 1) * value`` back
    through the representation's dense-merge operation.
    """

    __slots__ = (
        "_merge_index",
        "_plan",
        "_stochastic_rounding",
        "_target_key",
        "_value",
    )

    def __init__(
        self,
        value: ScaledParameterValue,
        *,
        stochastic_rounding: bool = False,
        target_key: str = "",
    ) -> None:
        if (
            stochastic_rounding
            and value.effective_strength != 1.0
            and not target_key
        ):
            raise ValueError(
                "Stochastic ParameterValueTransform requires a non-empty target_key."
            )
        self._value = value
        self._stochastic_rounding = stochastic_rounding
        self._target_key = target_key
        self._merge_index = 0
        self._plan: _ParameterValuePlan | None = None

    @property
    def backing(self) -> PinnedParam:
        """Physical source that defines allocation and copied bytes."""
        return self._value.value.backing

    @property
    def requires_update(self) -> bool:
        """Whether the loaded replacement needs explicit strength scaling."""
        return self._value.effective_strength != 1.0

    def validate_parameter(self, param: nn.Parameter) -> None:
        """Validate the model slot and prepare optional repeated scaling."""
        self._plan = None
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

        backing = self.backing
        source = param_representation(backing.make_cpu_param())
        adapter = backing.adapter
        logical_shape = backing.logical_shape
        if logical_shape != tuple(target.shape):
            raise ValueError(
                "Parameter value shape mismatch: "
                f"source shape is {logical_shape}, "
                f"target shape is {tuple(target.shape)}."
            )

        effective_strength = self._value.effective_strength
        self._validate_scaling(source, adapter, effective_strength)
        self._plan = _ParameterValuePlan(
            effective_strength=effective_strength,
            target_layout=backing.target_layout,
        )

    def apply_parameter(self, param: nn.Parameter) -> None:
        """Apply optional scaling to an already-loaded replacement."""
        plan = self._require_plan()
        target = param_representation(param)
        try:
            adapter = select_adapter(target)
        except NotImplementedError as exc:
            raise RuntimeError(
                "Parameter value target no longer has a registered tensor adapter."
            ) from exc
        if (
            target.is_meta
            or PinnedParam.target_layout_for(param) != plan.target_layout
        ):
            raise RuntimeError(
                "Parameter value update requires physical storage matching "
                "the validated replacement source."
            )
        if plan.effective_strength == 1.0:
            return
        if not (
            isinstance(adapter, DenseMergeTensorAdapter)
            and isinstance(adapter, DequantizeTensorAdapter)
        ):
            raise RuntimeError(
                "Validated parameter value scaling capability disappeared."
            )
        dense = adapter.dequantize(target)
        if dense.numel() and not bool(torch.isfinite(dense).all()):
            raise ValueError("Parameter values must contain only finite values.")
        adapter.merge_dense_(
            target,
            dense,
            plan.effective_strength - 1.0,
            rounding_seed=self._rounding_seed(),
        )
        self._merge_index += 1

    def materialize(
        self,
        *,
        device: torch.device | None = None,
    ) -> nn.Parameter:
        """Clone the physical replacement and apply optional scaling."""
        self._require_plan()
        target_device = torch.device("cpu") if device is None else device
        backing = self.backing
        param = (
            backing.clone_cpu_param()
            if target_device.type == "cpu"
            else backing.materialize(target_device)
        )
        if self.requires_update:
            self.apply_parameter(param)
        return param

    def _require_plan(self) -> _ParameterValuePlan:
        plan = self._plan
        if plan is None:
            raise RuntimeError(
                "Parameter value target must be validated before application."
            )
        return plan

    def _validate_scaling(
        self,
        source: torch.Tensor,
        adapter: TensorAdapter[Any, Any],
        strength: float,
    ) -> None:
        if strength == 1.0:
            return
        if not (
            isinstance(adapter, DenseMergeTensorAdapter)
            and isinstance(adapter, DequantizeTensorAdapter)
        ):
            raise ValueError(
                f"{adapter_name(adapter)} cannot scale a parameter value: "
                "the representation requires both dense merge and dequantize support."
            )

        rounding_seed = self._rounding_seed()
        requires_update_validation = isinstance(
            adapter,
            DenseMergeValidationTensorAdapter,
        )
        if isinstance(adapter, DenseMergeTargetValidationTensorAdapter):
            requires_update_validation = adapter.validate_dense_merge_target(
                source,
                rounding_seed=rounding_seed,
            )

        dense = adapter.dequantize(source)
        if dense.numel() and not bool(torch.isfinite(dense).all()):
            raise ValueError("Parameter values must contain only finite values.")
        if requires_update_validation:
            if not isinstance(adapter, DenseMergeValidationTensorAdapter):
                raise ValueError(
                    f"{adapter_name(adapter)} requested staged dense-update "
                    "validation without implementing validate_dense_merge()."
                )
            adapter.validate_dense_merge(
                source,
                dense,
                strength - 1.0,
                rounding_seed=rounding_seed,
            )

    def _rounding_seed(self) -> int | None:
        if not self._stochastic_rounding:
            return None
        return derive_seed(self._target_key, self._merge_index)
