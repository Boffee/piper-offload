"""Host-backed additive updates for existing model parameters."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import nn

from .lora import LoRAFactor, LoRATransform, ScaledLoRAFactor
from .pinned_param import PinnedParam
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
    mutating or copying its host backing.
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
class _DenseDeltaPlan:
    """Validated dense sources and target representation for repeated merges."""

    sources: tuple[tuple[torch.Tensor, float], ...]
    shape: tuple[int, ...]
    dtype: torch.dtype


class ParameterDeltaTransform:
    """Apply combined low-rank and dense updates to one existing parameter.

    This first implementation supports dense terms on physical plain
    floating-point tensors. Factor-only transforms continue to delegate to
    :class:`LoRATransform`, including all existing quantized merge paths.
    """

    __slots__ = (
        "_deltas",
        "_dense_plan",
        "_lora_transform",
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
        factors = [
            ScaledLoRAFactor(scaled.delta.lora, scaled.strength)
            for scaled in self._deltas
            if scaled.delta.lora is not None
        ]
        self._lora_transform = (
            None
            if not factors
            else LoRATransform(
                factors,
                stochastic_rounding=stochastic_rounding,
                target_key=target_key,
            )
        )
        self._dense_plan: _DenseDeltaPlan | None = None

    @property
    def has_dense(self) -> bool:
        """Whether any bound update contains a full-rank contribution."""
        return any(scaled.delta.dense is not None for scaled in self._deltas)

    def validate_parameter(self, param: nn.Parameter) -> None:
        """Validate every contribution before preparing repeated application."""
        self._dense_plan = None
        lora_transform = self._lora_transform
        if lora_transform is not None:
            lora_transform.validate_parameter(param)

        dense_sources = self._materialize_dense_sources()
        if not dense_sources:
            # Factor-only application is completely owned by LoRATransform.
            return

        target = param_representation(param)
        if type(target) is not torch.Tensor or target.is_meta:
            raise ValueError(
                "Dense parameter deltas currently require an existing plain floating-point target; "
                f"got {type(target).__name__} on {target.device}."
            )
        if not target.is_floating_point():
            raise ValueError(f"Dense parameter deltas require a floating-point target; got {target.dtype}.")
        if torch.finfo(target.dtype).bits == 8:
            raise ValueError(f"Dense parameter deltas do not support float8 targets; got {target.dtype}.")

        shape = tuple(target.shape)
        for source, _strength in dense_sources:
            if tuple(source.shape) != shape:
                raise ValueError(
                    "Dense parameter delta shape mismatch: "
                    f"source shape is {tuple(source.shape)}, target shape is {shape}."
                )
            if source.numel() and not bool(torch.isfinite(source).all()):
                raise ValueError("Dense parameter deltas must contain only finite values.")

        self._dense_plan = _DenseDeltaPlan(
            tuple(dense_sources),
            shape,
            target.dtype,
        )

    def apply_parameter(self, param: nn.Parameter) -> None:
        """Stage the complete update, then mutate the base parameter once."""
        dense_update = self._stage_dense_update(param)
        lora_transform = self._lora_transform
        if dense_update is None:
            if lora_transform is not None:
                lora_transform.apply_parameter(param)
            return

        if lora_transform is not None:
            lora_transform.accumulate_parameter_update(dense_update)
        param_representation(param).add_(dense_update)

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

    def _stage_dense_update(self, param: nn.Parameter) -> torch.Tensor | None:
        plan = self._dense_plan
        if plan is None:
            if self.has_dense:
                raise RuntimeError("Dense parameter delta target must be validated before application.")
            return None

        target = param_representation(param)
        if (
            type(target) is not torch.Tensor
            or target.is_meta
            or tuple(target.shape) != plan.shape
            or target.dtype is not plan.dtype
        ):
            raise RuntimeError(
                "Dense parameter delta application requires a physical plain tensor matching the validated target."
            )

        update = torch.zeros_like(target)
        for source, strength in plan.sources:
            staged = source.to(
                device=target.device,
                dtype=target.dtype,
                non_blocking=True,
            )
            update.add_(staged, alpha=strength)
        return update
