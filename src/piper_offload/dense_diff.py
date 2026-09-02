"""Full-shape dense adapter diffs for plain floating-point parameters."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .pinned_param import PinnedParam
from .tensor_adapter_registry import param_representation


@dataclass(slots=True, frozen=True)
class ScaledDenseTarget:
    """One host-backed full-shape diff bound to an extrinsic strength."""

    diff: PinnedParam
    strength: float


@dataclass(slots=True, frozen=True)
class _DenseTargetPlan:
    """Validated dense diffs for one physical or meta target."""

    diffs: tuple[tuple[torch.Tensor, float], ...]
    is_meta: bool
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


class DenseDiffTransform:
    """Apply one or more full-shape diffs to a plain floating parameter.

    Validation prepares immutable CPU sources for repeated application after
    offload copies. A frozen meta parameter is treated as a storage-free zero:
    active storage is filled directly from the first diff rather than from an
    allocated zero base.
    """

    __slots__ = ("_plan", "_targets")

    def __init__(self, targets: Sequence[ScaledDenseTarget]) -> None:
        if not targets:
            raise ValueError("DenseDiffTransform requires a dense target")
        self._targets = tuple(targets)
        self._plan: _DenseTargetPlan | None = None

    def validate_parameter(self, param: nn.Parameter) -> None:
        """Validate ``param`` and prepare subsequent applications."""
        self._plan = None
        target = param_representation(param)
        if type(target) is not torch.Tensor:
            raise ValueError(
                f"Dense adapter targets require a plain floating-point base parameter; got {type(target).__name__}."
            )
        if not target.is_floating_point():
            raise ValueError(f"Dense adapter targets require a floating-point base parameter; got {target.dtype}.")
        if target.is_meta and param.requires_grad:
            raise ValueError("Meta dense targets are inference-only and require requires_grad=False.")

        diffs: list[tuple[torch.Tensor, float]] = []
        for dense_target in self._targets:
            source = param_representation(dense_target.diff.make_cpu_param())
            if type(source) is not torch.Tensor or source.device.type != "cpu":
                raise ValueError(
                    "Dense adapter diffs require plain CPU torch.Tensor backing; "
                    f"got {type(source).__name__} on {source.device}."
                )
            if not source.is_floating_point():
                raise ValueError(f"Dense adapter diffs must be floating-point; got {source.dtype}.")
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    "Dense adapter diff shape mismatch: "
                    f"diff shape is {tuple(source.shape)}, "
                    f"target shape is {tuple(target.shape)}."
                )
            diffs.append((source, dense_target.strength))

        self._plan = _DenseTargetPlan(
            tuple(diffs),
            is_meta=target.is_meta,
            shape=tuple(target.shape),
            stride=target.stride(),
            dtype=target.dtype,
        )

    def apply_parameter(self, param: nn.Parameter) -> None:
        """Apply this transform to a previously validated parameter."""
        plan = self._plan
        if plan is None:
            raise RuntimeError("Dense diff target must be validated before application.")
        self._apply(param_representation(param), plan)

    def materialize_meta(
        self,
        *,
        device: torch.device | None = None,
    ) -> nn.Parameter:
        """Materialize a previously validated meta target on ``device``."""
        plan = self._plan
        if plan is None or not plan.is_meta:
            raise RuntimeError("Dense diff materialization requires a validated meta target.")
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
        self._apply(param.data, plan)
        return param

    @staticmethod
    def _apply(target: torch.Tensor, plan: _DenseTargetPlan) -> None:
        """Apply validated diffs, filling a meta base directly."""
        diffs = iter(plan.diffs)
        if plan.is_meta:
            source, strength = next(diffs)
            target.copy_(source, non_blocking=True)
            if strength != 1.0:
                target.mul_(strength)

        for source, strength in diffs:
            staged = source.to(
                device=target.device,
                dtype=target.dtype,
                non_blocking=True,
            )
            target.add_(staged, alpha=strength)


__all__ = ["DenseDiffTransform", "ScaledDenseTarget"]
