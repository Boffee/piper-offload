"""Host-backed additive updates for existing model parameters."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import nn

from ._parameter_update import ParameterUpdateTransform
from .host_param import HostParam
from .lora import LoRAFactor
from .tensor_adapter_registry import param_representation

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
) -> HostParam:
    _validate_dense_tensor(source)
    tensor = source if dtype is None or source.dtype is dtype else source.to(dtype=dtype)
    return HostParam(
        nn.Parameter(tensor, requires_grad=False),
    )


@dataclass(slots=True, frozen=True)
class ParameterDelta:
    """One reusable additive update for an exact model parameter.

    ``lora`` stores an optional low-rank contribution and ``dense`` an
    optional full-rank contribution. At least one representation must be
    present. Strength is deliberately extrinsic: binding this resource to an
    activation produces
    ``strength * (lora.scaling * lora.B @ lora.A + dense)`` without mutating or
    copying its host backing. Only the factors carry an intrinsic scaling; a
    dense contribution has no magnitude of its own. Tensor payload numerical validity is
    the caller's responsibility.

    Each contribution must broadcast to the existing target's logical shape
    using PyTorch rules, without expanding the target. Broadcasting requires
    the target tensor adapter's explicit support; complete parameter values
    have a separate exact-shape contract.
    """

    lora: LoRAFactor | None = None
    dense: HostParam | None = None

    def __post_init__(self) -> None:
        if self.lora is None and self.dense is None:
            raise ValueError("ParameterDelta requires a LoRA factor, a dense delta, or both.")
        if self.lora is not None and not isinstance(self.lora, LoRAFactor):
            raise ValueError(f"ParameterDelta lora must be a LoRAFactor; got {type(self.lora).__name__}.")
        if self.dense is not None:
            if not isinstance(self.dense, HostParam):
                raise ValueError(f"ParameterDelta dense must be a HostParam; got {type(self.dense).__name__}.")
            _validate_dense_tensor(param_representation(self.dense.make_cpu_param()))

    @classmethod
    def from_tensors(
        cls,
        *,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        dense: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        lora_alpha: float | None = None,
    ) -> Self:
        """Validate and capture an optional LoRA pair, its ``alpha``, and a dense contribution.

        ``lora_alpha`` belongs to the factors alone; a delta with no factors has
        no use for it.
        """
        if (a is None) != (b is None):
            raise ValueError("ParameterDelta requires both LoRA A and LoRA B when either factor is provided.")
        lora = (
            None
            if a is None or b is None
            else LoRAFactor.from_tensors(
                a,
                b,
                dtype=dtype,
                alpha=lora_alpha,
            )
        )
        dense_backing = (
            None
            if dense is None
            else _capture_dense_tensor(
                dense,
                dtype=dtype,
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


class ParameterDeltaTransform:
    """Apply additive low-rank and dense updates to an existing parameter.

    Each delta retains its own strength. Factor-only updates use factorized
    merge; dense and mixed updates combine before one terminal encoding.
    Source selection and optional value scaling belong to adapter composition.
    """

    __slots__ = ("_update",)

    def __init__(
        self,
        deltas: Sequence[ScaledParameterDelta],
        *,
        stochastic_rounding: bool = False,
        target_key: str = "",
    ) -> None:
        if not deltas:
            raise ValueError("ParameterDeltaTransform requires a delta.")
        self._update = ParameterUpdateTransform(
            deltas,
            source_scale=1.0,
            stochastic_rounding=stochastic_rounding,
            target_key=target_key,
        )

    def validate_parameter(self, param: nn.Parameter) -> None:
        """Validate every delta before preparing repeated application."""
        self._update.validate_parameter(param)

    def apply_parameter(self, param: nn.Parameter) -> None:
        """Add the prepared deltas to an already-loaded parameter."""
        self._update.apply_parameter(param)

    def storage_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return physical host tensors read while staging the deltas."""
        return self._update.storage_tensors()
