"""Exact values that populate frozen floating-point meta parameters."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from .pinned_param import PinnedParam
from .tensor_adapter_registry import param_representation

__all__ = [
    "ParameterValue",
    "ParameterValueTransform",
    "ScaledParameterValue",
]


def _validate_value_tensor(source: torch.Tensor) -> None:
    if type(source) is not torch.Tensor:
        raise ValueError(f"Parameter values must be plain torch.Tensor values; got {type(source).__name__}.")
    if source.is_meta:
        raise ValueError("Parameter values must own physical values, not meta storage.")
    if not source.is_floating_point():
        raise ValueError(f"Parameter values must be floating-point; got {source.dtype}.")


@dataclass(slots=True, frozen=True)
class ParameterValue:
    """One host-backed value for an exact model parameter name.

    Use :meth:`from_tensor` for standalone construction. Adapter resources
    use the same constructor after classifying their canonical state dict.
    """

    backing: PinnedParam

    def __post_init__(self) -> None:
        """Keep direct construction subject to the source-value invariant."""
        _validate_value_tensor(param_representation(self.backing.make_cpu_param()))

    @classmethod
    def from_tensor(
        cls,
        source: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        pin_memory: bool = True,
    ) -> ParameterValue:
        """Validate and capture one unscaled parameter value."""
        _validate_value_tensor(source)
        tensor = source if dtype is None or source.dtype is dtype else source.to(dtype=dtype)
        return cls(
            PinnedParam(
                nn.Parameter(tensor, requires_grad=False),
                pin_memory=pin_memory,
            )
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
    """A parameter value bound to an extrinsic strength."""

    value: ParameterValue
    strength: float


@dataclass(slots=True, frozen=True)
class _ParameterValuePlan:
    """Validated source and layout for one meta target."""

    source: torch.Tensor
    strength: float
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


class ParameterValueTransform:
    """Populate one frozen floating-point meta parameter.

    Validation runs against the model's storage-free meta placeholder. During
    activation, :meth:`apply_parameter` fills the physical storage allocated
    for that placeholder. Permanent merge uses :meth:`materialize` to create a
    CPU parameter with the same logical layout. Targets must have a strided,
    non-overlapping dense layout with zero storage offset so every logical
    element has one independently writable physical location.
    """

    __slots__ = ("_plan", "_value")

    def __init__(self, value: ScaledParameterValue) -> None:
        self._value = value
        self._plan: _ParameterValuePlan | None = None

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
        _validate_target_layout(target)

        source = param_representation(self._value.value.backing.make_cpu_param())
        assert type(source) is torch.Tensor
        assert source.device.type == "cpu"
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(
                "Parameter value shape mismatch: "
                f"source shape is {tuple(source.shape)}, "
                f"target shape is {tuple(target.shape)}."
            )

        _validate_target_range(
            source,
            target_dtype=target.dtype,
            strength=self._value.strength,
        )
        prepared_source, prepared_strength = _prepare_source(
            source,
            target_dtype=target.dtype,
            strength=self._value.strength,
        )
        self._plan = _ParameterValuePlan(
            prepared_source,
            prepared_strength,
            tuple(target.shape),
            target.stride(),
            target.dtype,
        )

    def apply_parameter(self, param: nn.Parameter) -> None:
        """Fill active storage for a previously validated meta parameter."""
        plan = self._require_plan()
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
    def _fill(target: torch.Tensor, plan: _ParameterValuePlan) -> None:
        target.copy_(plan.source, non_blocking=True)
        if plan.strength != 1.0:
            target.mul_(plan.strength)


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
    if not math.isfinite(strength):
        raise ValueError(f"Parameter value strength must be finite; got {strength}.")
    if source.numel() == 0:
        return

    range_source = source.float() if torch.finfo(source.dtype).bits == 8 else source
    minimum, maximum = torch.aminmax(range_source)
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

    _validate_no_underflow(
        source,
        target_dtype=target_dtype,
        strength=strength,
    )


def _validate_no_underflow(
    source: torch.Tensor,
    *,
    target_dtype: torch.dtype,
    strength: float,
) -> None:
    """Reject nonzero values that target conversion or scaling erases."""
    converted = source.to(dtype=target_dtype)
    source_nonzero = _count_nonzero(source)
    converted_nonzero = _count_nonzero(converted)
    if converted_nonzero != source_nonzero:
        raise ValueError(
            "Parameter value source underflows to zero in target dtype "
            f"{target_dtype} before strength is applied."
        )

    if strength in {0.0, 1.0}:
        return
    if converted is source:
        converted = converted.clone()
    if torch.finfo(target_dtype).bits == 8:
        compute_source = converted.float()
        torch.mul(compute_source, strength, out=converted)
    else:
        converted.mul_(strength)
    if _count_nonzero(converted) != converted_nonzero:
        raise ValueError(
            "Scaled parameter value underflows to zero in target dtype "
            f"{target_dtype}."
        )


def _count_nonzero(source: torch.Tensor) -> int:
    """Count nonzero floating values, including CPU float8 tensors."""
    count_source = source.float() if torch.finfo(source.dtype).bits == 8 else source
    return int(torch.count_nonzero(count_source).item())


def _prepare_source(
    source: torch.Tensor,
    *,
    target_dtype: torch.dtype,
    strength: float,
) -> tuple[torch.Tensor, float]:
    """Prepare only scaled float8 targets, whose in-place multiply is unsupported."""
    if strength == 1.0 or torch.finfo(target_dtype).bits != 8:
        return source, strength

    scaled = torch.empty(
        tuple(source.shape),
        dtype=target_dtype,
        device="cpu",
        pin_memory=source.is_pinned(),
    )
    scaled.copy_(source)
    compute_source = scaled.float()
    torch.mul(compute_source, strength, out=scaled)
    return scaled, 1.0
