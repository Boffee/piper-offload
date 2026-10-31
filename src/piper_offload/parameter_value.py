"""Physical replacement values for frozen meta parameters."""

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from ._parameter_update import ParameterUpdatePlan, plan_parameter_update
from .host_param import HostParam
from .tensor_adapter_registry import param_representation, select_adapter
from .tensor_adapters import (
    RegularAdapter,
    TensorAdapter,
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


@dataclass(slots=True, frozen=True)
class ParameterValue:
    """One authoritative physical replacement for an exact parameter name.

    The backing defines active dtype, layout, quantization metadata, and
    copied bytes. Adapter strength does not scale a complete value unless
    ``scale_with_strength=True`` is explicitly requested. Payload numerical
    validity is the caller's responsibility; values are not scanned for NaN
    or infinity.
    """

    backing: HostParam
    scale_with_strength: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.backing, HostParam):
            raise ValueError(
                "ParameterValue backing must be a HostParam; "
                f"got {type(self.backing).__name__}."
            )
        if self.backing.requires_grad:
            raise ValueError("Parameter values are inference-only and must be frozen.")
        _select_value_adapter(
            param_representation(self.backing.make_cpu_param())
        )

    @classmethod
    def from_tensor(
        cls,
        source: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
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
        # final captured representation is validated once by
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
            HostParam(parameter),
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


class ParameterValueTransform:
    """Validate a replacement and optionally scale it after loading.

    Replacement validation and arithmetic use the same plan as activation
    and permanent merge. Exact values have no arithmetic update.
    """

    __slots__ = ("_plan", "_validated")

    def __init__(
        self,
        value: ScaledParameterValue,
        *,
        stochastic_rounding: bool = False,
        target_key: str = "",
    ) -> None:
        self._plan = plan_parameter_update(
            (),
            value=value,
            stochastic_rounding=stochastic_rounding,
            target_key=target_key,
        )
        self._validated = False

    @property
    def backing(self) -> HostParam:
        """Physical source that defines allocation and copied bytes."""
        source = self._plan.source
        assert source is not None
        return source

    @property
    def requires_update(self) -> bool:
        """Whether the loaded replacement needs explicit strength scaling."""
        return self._plan.update is not None

    def validate_parameter(self, param: nn.Parameter) -> None:
        """Validate the model slot and prepare optional repeated scaling."""
        self._validated = False
        self._plan.validate_parameter(param)
        self._validated = True

    def apply_parameter(self, param: nn.Parameter) -> None:
        """Apply optional scaling to an already-loaded replacement."""
        self._require_plan().apply_parameter(param)

    def storage_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return update-only storage; the load plan already owns the value."""
        return ()

    def materialize(
        self,
        *,
        device: torch.device | None = None,
    ) -> nn.Parameter:
        """Clone the physical replacement and apply optional scaling."""
        plan = self._require_plan()
        target_device = torch.device("cpu") if device is None else device
        backing = self.backing
        param = (
            backing.clone_cpu_param()
            if target_device.type == "cpu"
            else backing.materialize(target_device)
        )
        if plan.update is not None:
            plan.update.apply_parameter(param)
        return param

    def _require_plan(self) -> ParameterUpdatePlan:
        if not self._validated:
            raise RuntimeError(
                "Parameter value target must be validated before application."
            )
        return self._plan
