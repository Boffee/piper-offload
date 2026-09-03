"""Exact dense or quantized values that populate frozen meta parameters."""

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


def _validate_value_representation(
    source: torch.Tensor,
) -> TensorAdapter[Any, Any]:
    """Validate one physical value and return its registered adapter."""
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
    if isinstance(adapter, RegularAdapter) and torch.finfo(compute_dtype).bits == 8:
        raise ValueError(f"Parameter values do not support float8 sources; got {source.dtype}.")
    if not (
        isinstance(adapter, DenseMergeTensorAdapter)
        and isinstance(adapter, DequantizeTensorAdapter)
    ):
        raise ValueError(
            f"{adapter_name(adapter)} cannot be used as a parameter value: "
            "structured values require both dense merge and dequantize support."
        )
    if not isinstance(adapter, RegularAdapter) and isinstance(
        adapter,
        DenseMergeTargetValidationTensorAdapter,
    ):
        # Composing adapters such as DTensor advertise the outer capability
        # structurally, then validate that the concrete inner representation
        # supports it here.
        adapter.validate_dense_merge_target(source)
    return adapter


@dataclass(slots=True, frozen=True)
class ParameterValue:
    """One host-backed value for an exact model parameter name.

    Use :meth:`from_tensor` for standalone construction. Adapter resources
    use the same constructor after classifying their canonical state dict.
    ``scale_with_strength`` controls whether an active adapter's strength is
    applied when the complete value is materialized.
    """

    backing: PinnedParam
    scale_with_strength: bool = True

    def __post_init__(self) -> None:
        """Keep direct construction subject to the source-value invariant."""
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
        scale_with_strength: bool = True,
    ) -> ParameterValue:
        """Validate and capture one parameter value and its scaling policy."""
        if not isinstance(source, torch.Tensor):
            raise ValueError(
                "Parameter values must be tensor representations; "
                f"got {type(source).__name__}."
            )
        if issubclass(type(source), nn.Parameter) and source.requires_grad:
            raise ValueError("Parameter values are inference-only and must be frozen.")
        adapter = _validate_value_representation(source)
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
            PinnedParam(
                parameter,
                pin_memory=pin_memory,
            ),
            scale_with_strength=scale_with_strength,
        )

    @property
    def cache_bytes(self) -> int:
        """Host-backing bytes held by this value."""
        return self.backing.cache_bytes

    def scaled(self, strength: float) -> ScaledParameterValue:
        """Bind this value to an extrinsic materialization strength."""
        return ScaledParameterValue(self, strength)


@dataclass(slots=True, frozen=True)
class ScaledParameterValue:
    """A parameter value bound to an extrinsic adapter strength."""

    value: ParameterValue
    strength: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.strength):
            raise ValueError(f"Parameter value strength must be finite; got {self.strength}.")

    @property
    def materialization_strength(self) -> float:
        """Return the multiplier selected by the value's scaling policy."""
        return self.strength if self.value.scale_with_strength else 1.0


@dataclass(slots=True, frozen=True)
class _PlainParameterValuePlan:
    """Validated source and layout for one meta target."""

    source: torch.Tensor
    strength: float
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


@dataclass(slots=True, frozen=True)
class _StructuredParameterValuePlan:
    """Validated structured representation for one logical meta target."""

    adapter: TensorAdapter[Any, Any]
    strength: float
    logical_shape: tuple[int, ...]
    compute_dtype: torch.dtype


type _ParameterValuePlan = (
    _PlainParameterValuePlan | _StructuredParameterValuePlan
)


class ParameterValueTransform:
    """Populate one frozen floating-point meta parameter.

    Validation runs against the model's storage-free meta placeholder. During
    activation, :meth:`apply_parameter` fills the physical storage allocated
    for that placeholder. Permanent merge uses :meth:`materialize` to create a
    CPU parameter. Dense values preserve the placeholder's logical layout,
    which must be strided and non-overlapping with zero storage offset.
    Structured values instead retain their source adapter's complete storage
    representation and metadata.
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
        if stochastic_rounding and not target_key:
            raise ValueError(
                "Stochastic ParameterValueTransform requires a non-empty target_key."
            )
        self._value = value
        self._stochastic_rounding = stochastic_rounding
        self._target_key = target_key
        self._merge_index = 0
        self._plan: _ParameterValuePlan | None = None

    @property
    def materialization_backing(self) -> PinnedParam | None:
        """Backing that defines active storage for a structured value."""
        backing = self._value.value.backing
        return None if isinstance(backing.adapter, RegularAdapter) else backing

    def validate_parameter(self, param: nn.Parameter) -> None:
        """Validate a meta target and prepare repeated fills."""
        self._plan = None
        target = param_representation(param)
        if type(target) is not torch.Tensor or not target.is_meta:
            raise ValueError(
                "Parameter values require a plain floating-point meta target; "
                f"got {type(target).__name__} on {target.device}."
            )
        if not target.is_floating_point():
            raise ValueError(f"Parameter values require a floating-point meta target; got {target.dtype}.")
        if param.requires_grad:
            raise ValueError("Parameter values are inference-only and require requires_grad=False.")

        backing = self._value.value.backing
        source = param_representation(backing.make_cpu_param())
        adapter = backing.adapter
        logical_shape = backing.logical_shape
        if logical_shape != tuple(target.shape):
            raise ValueError(
                "Parameter value shape mismatch: "
                f"source shape is {logical_shape}, "
                f"target shape is {tuple(target.shape)}."
            )

        materialization_strength = self._value.materialization_strength
        if not isinstance(adapter, RegularAdapter):
            compute_dtype = adapter.compute_dtype(source)
            self._validate_structured_merge(
                source,
                adapter,
                materialization_strength,
            )
            self._plan = _StructuredParameterValuePlan(
                adapter,
                materialization_strength,
                logical_shape,
                compute_dtype,
            )
            return

        if torch.finfo(target.dtype).bits == 8:
            raise ValueError(f"Parameter values do not support float8 targets; got {target.dtype}.")
        _validate_target_layout(target)
        assert type(source) is torch.Tensor
        assert source.device.type == "cpu"
        _validate_target_range(
            source,
            target_dtype=target.dtype,
            strength=materialization_strength,
        )
        self._plan = _PlainParameterValuePlan(
            source,
            materialization_strength,
            tuple(target.shape),
            target.stride(),
            target.dtype,
        )

    def apply_parameter(self, param: nn.Parameter) -> None:
        """Fill active storage for a previously validated meta parameter."""
        plan = self._require_plan()
        if isinstance(plan, _StructuredParameterValuePlan):
            self._apply_structured(param, plan)
            return
        target = param_representation(param)
        if (
            type(target) is not torch.Tensor
            or target.is_meta
            or target.layout is not torch.strided
            or tuple(target.shape) != plan.shape
            or target.stride() != plan.stride
            or target.storage_offset() != 0
            or target.dtype is not plan.dtype
        ):
            raise RuntimeError(
                "Parameter value application requires physical storage matching the validated meta target."
            )
        self._fill(target, plan)

    def materialize(
        self,
        *,
        device: torch.device | None = None,
    ) -> nn.Parameter:
        """Materialize a validated value on ``device`` (CPU by default)."""
        plan = self._require_plan()
        target_device = torch.device("cpu") if device is None else device
        if isinstance(plan, _StructuredParameterValuePlan):
            backing = self._value.value.backing
            param = (
                backing.clone_cpu_param()
                if target_device.type == "cpu"
                else backing.materialize(target_device)
            )
            self._apply_structured(param, plan)
            return param
        param = nn.Parameter(
            torch.empty_strided(
                plan.shape,
                plan.stride,
                dtype=plan.dtype,
                device=target_device,
            ),
            requires_grad=False,
        )
        self._fill(param.data, plan)
        return param

    def _require_plan(self) -> _ParameterValuePlan:
        plan = self._plan
        if plan is None:
            raise RuntimeError("Parameter value target must be validated before application.")
        return plan

    @staticmethod
    def _fill(target: torch.Tensor, plan: _PlainParameterValuePlan) -> None:
        target.copy_(plan.source, non_blocking=True)
        if plan.strength != 1.0:
            target.mul_(plan.strength)

    def _validate_structured_merge(
        self,
        source: torch.Tensor,
        adapter: TensorAdapter[Any, Any],
        strength: float,
    ) -> None:
        """Preflight the optional scaling merge against source metadata."""
        if strength == 1.0:
            return
        assert isinstance(adapter, DenseMergeTensorAdapter)
        assert isinstance(adapter, DequantizeTensorAdapter)
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
        if requires_update_validation:
            if not isinstance(adapter, DenseMergeValidationTensorAdapter):
                raise ValueError(
                    f"{adapter_name(adapter)} requested staged dense-update "
                    "validation without implementing validate_dense_merge()."
                )
            dense = adapter.dequantize(source)
            adapter.validate_dense_merge(
                source,
                dense,
                strength - 1.0,
                rounding_seed=rounding_seed,
            )

    def _apply_structured(
        self,
        param: nn.Parameter,
        plan: _StructuredParameterValuePlan,
    ) -> None:
        target = param_representation(param)
        try:
            adapter = select_adapter(target)
        except NotImplementedError as exc:
            raise RuntimeError(
                "Structured parameter value application target no longer has "
                "a registered tensor adapter."
            ) from exc
        if (
            target.is_meta
            or type(adapter) is not type(plan.adapter)
            or adapter.compute_dtype(target) is not plan.compute_dtype
            or not isinstance(adapter, DenseMergeTensorAdapter)
            or not isinstance(adapter, DequantizeTensorAdapter)
            or adapter.logical_shape(target) != plan.logical_shape
        ):
            raise RuntimeError(
                "Structured parameter value application requires physical "
                "storage matching the validated value representation."
            )
        if plan.strength == 1.0:
            return
        dense = adapter.dequantize(target)
        if dense.numel() and not bool(torch.isfinite(dense).all()):
            raise ValueError("Parameter values must contain only finite values.")
        adapter.merge_dense_(
            target,
            dense,
            plan.strength - 1.0,
            rounding_seed=self._rounding_seed(),
        )
        self._merge_index += 1

    def _rounding_seed(self) -> int | None:
        if not self._stochastic_rounding:
            return None
        return derive_seed(self._target_key, self._merge_index)


def _validate_target_layout(target: torch.Tensor) -> None:
    """Require a layout that can represent and populate every source value."""
    if target.layout is not torch.strided:
        raise ValueError(
            "Parameter values require a strided meta target; "
            f"got layout {target.layout}."
        )
    if target.storage_offset() != 0:
        raise ValueError(
            "Parameter values require a meta target with storage_offset=0; "
            f"got {target.storage_offset()}."
        )
    if not _is_non_overlapping_and_dense(target.shape, target.stride()):
        raise ValueError(
            "Parameter values require a non-overlapping dense meta target; "
            f"got shape={tuple(target.shape)}, stride={target.stride()}."
        )


def _is_non_overlapping_and_dense(
    shape: torch.Size,
    stride: tuple[int, ...],
) -> bool:
    """Check dense strided layout without relying on private PyTorch APIs."""
    if any(size == 0 for size in shape):
        return True

    dimensions = sorted(
        (dimension_stride, size)
        for size, dimension_stride in zip(shape, stride, strict=True)
        if size > 1
    )
    expected_stride = 1
    for dimension_stride, size in dimensions:
        if dimension_stride != expected_stride:
            return False
        expected_stride *= size
    return True


def _validate_target_range(
    source: torch.Tensor,
    *,
    target_dtype: torch.dtype,
    strength: float,
) -> None:
    """Reject invalid source and scaled values before target mutation."""
    if source.numel() == 0:
        return

    minimum, maximum = torch.aminmax(source)
    minimum_value = minimum.item()
    maximum_value = maximum.item()
    if not math.isfinite(minimum_value) or not math.isfinite(maximum_value):
        raise ValueError("Parameter values must contain only finite values.")

    maximum_magnitude = max(abs(minimum_value), abs(maximum_value))
    target_maximum = torch.finfo(target_dtype).max
    if maximum_magnitude > target_maximum:
        raise ValueError(
            "Parameter value source exceeds the finite range of target dtype "
            f"{target_dtype}: maximum magnitude {maximum_magnitude}, "
            f"limit {target_maximum}."
        )

    scaled_maximum = maximum_magnitude * abs(strength)
    if not math.isfinite(scaled_maximum) or scaled_maximum > target_maximum:
        raise ValueError(
            "Scaled parameter value exceeds the finite range of target dtype "
            f"{target_dtype}: maximum magnitude {scaled_maximum}, "
            f"limit {target_maximum}."
        )
