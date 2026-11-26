"""Host-backed LoRA data and parameter merging.

The general mixed adapter resource lives in :mod:`piper_offload.adapter`;
this module owns the complete low-rank update family beneath it.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import torch
from torch import nn

from .host_param import HostParam
from .seeding import derive_seed
from .tensor_adapter_registry import param_representation, select_adapter
from .tensor_adapters import (
    LoRAMergeTensorAdapter,
    LoRAMergeValidationTensorAdapter,
    MergeLocalityTensorAdapter,
    adapter_name,
    transfer_,
    validate_broadcast_merge,
)

__all__ = [
    "LoRAFactor",
    "ScaledLoRAFactor",
]


def _validate_factor_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
) -> None:
    for name, tensor in (("LoRA A", a), ("LoRA B", b)):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor; got {type(tensor).__name__}.")
        if type(tensor) is not torch.Tensor:
            raise ValueError(
                f"{name} must be a plain torch.Tensor; got {type(tensor).__name__}."
            )
        if tensor.requires_grad:
            raise ValueError(f"{name} must be frozen (requires_grad=False).")
        if tensor.is_meta:
            raise ValueError(f"{name} must own physical values, not meta storage.")
    if not a.is_floating_point() or not b.is_floating_point():
        raise ValueError(f"LoRA factors must be floating-point; got A.dtype={a.dtype}, B.dtype={b.dtype}.")
    if a.dim() != 2 or b.dim() != 2 or a.shape[0] != b.shape[1]:
        raise ValueError(
            f"LoRA factor shape mismatch: A.shape={tuple(a.shape)}, "
            f"B.shape={tuple(b.shape)}. Expected A=(rank, in_dim), "
            "B=(out_dim, rank) with A.shape[0] == B.shape[1]."
        )


def _validate_factor_scaling(scaling: float) -> None:
    if not math.isfinite(scaling):
        raise ValueError(f"LoRA scaling must be finite; got {scaling}.")


def _capture_factor_tensor(
    source: torch.Tensor,
    *,
    dtype: torch.dtype | None,
) -> HostParam:
    tensor = source if dtype is None or source.dtype is dtype else source.to(dtype=dtype)
    return HostParam(
        nn.Parameter(tensor, requires_grad=False),
    )


@dataclass(slots=True, frozen=True)
class LoRAFactor:
    """A LoRA's host-backed tensors for one target weight.

    ``a`` is the ``(rank, in_dim)`` down-projection and ``b`` the
    ``(out_dim, rank)`` up-projection, each held as a :class:`HostParam`.
    :meth:`from_tensors` validates the stored pair before capture; direct
    construction enforces the same invariant. The match against a concrete
    target shape is checked separately, where the target is known.

    Merge targets accept a product that broadcasts to their logical shape
    when their tensor adapter supports it.

    Two scalars meet here and they are not the same thing. ``scale`` is
    *intrinsic*: the checkpoint's own ``alpha / rank`` magnitude, a property of
    these tensors. Strength is *extrinsic*, supplied when the adapter is bound
    to a target. Carrying ``scale`` here rather than folding it into the
    tensors keeps a mapped factor mapped, so it can still be pinned through an
    owned copy; folding is host arithmetic that would replace the mapping with
    an anonymous tensor.

    :meth:`scaled` binds the extrinsic strength without discarding the host
    representation, so the merge path can stage factors onto the target device.
    """

    a: HostParam
    b: HostParam
    scaling: float = 1.0

    def __post_init__(self) -> None:
        """Keep direct construction subject to the same data invariants."""
        _validate_factor_scaling(self.scaling)
        _validate_factor_tensors(
            param_representation(self.a.make_cpu_param()),
            param_representation(self.b.make_cpu_param()),
        )

    @classmethod
    def from_tensors(
        cls,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        alpha: float | None = None,
    ) -> Self:
        """Validate and capture one factor pair, dividing ``alpha`` by the rank it just validated."""
        _validate_factor_tensors(a, b)
        rank = a.shape[0]
        if alpha is not None and rank == 0:
            raise ValueError("A rank-0 LoRA factor has no rank for its alpha to scale.")
        return cls(
            _capture_factor_tensor(a, dtype=dtype),
            _capture_factor_tensor(b, dtype=dtype),
            1.0 if alpha is None else alpha / rank,
        )

    @property
    def cache_bytes(self) -> int:
        """Host-backing bytes held by this target's adapter tensors."""
        return self.a.cache_bytes + self.b.cache_bytes

    def scaled(self, strength: float) -> ScaledLoRAFactor:
        """Bind this pair to ``strength``, composing it with the factor's intrinsic scaling."""
        if not math.isfinite(strength):
            raise ValueError(f"LoRA strength must be finite; got {strength}.")
        return ScaledLoRAFactor(self, strength * self.scaling)


@dataclass(slots=True, frozen=True)
class ScaledLoRAFactor:
    """Host-backed LoRA tensors bound to an application ``strength``.

    The application-side carrier used by parameter updates. Keeping
    :class:`HostParam` retains the host storage backing and cache accounting
    across repeated loads. The contribution to the base weight is
    ``coefficient * (b @ a)``.

    One scalar, not two: :attr:`coefficient` is the caller's strength already
    composed with the factor's intrinsic scaling, so no application path has to
    remember to combine them. Build it through :meth:`LoRAFactor.scaled`,
    which does that composition, or :meth:`from_tensors` to construct one
    directly from source tensors.
    """

    factor: LoRAFactor
    coefficient: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient):
            raise ValueError(f"LoRA coefficient must be finite; got {self.coefficient}.")

    @classmethod
    def from_tensors(
        cls,
        a: torch.Tensor,
        b: torch.Tensor,
        strength: float,
    ) -> Self:
        """Capture unbound adapter tensors and bind them to ``strength``.

        These factors declare no alpha, so their scale is neutral and the
        composed coefficient is ``strength`` itself.
        """
        return cls(LoRAFactor.from_tensors(a, b), strength)

    @property
    def a(self) -> HostParam:
        return self.factor.a

    @property
    def b(self) -> HostParam:
        return self.factor.b

    @property
    def rank(self) -> int:
        return self.a.shape[0]

    @property
    def in_dim(self) -> int:
        return self.a.shape[1]

    @property
    def out_dim(self) -> int:
        return self.b.shape[0]

    @property
    def produced_shape(self) -> tuple[int, int]:
        """Shape of ``b @ a`` before any target-side broadcasting."""
        return (self.b.shape[0], self.a.shape[1])


@dataclass(slots=True, frozen=True)
class _MaterializedWeightFactor:
    """One scaled weight factor exposed as plain host tensors."""

    coefficient: float
    a: torch.Tensor
    b: torch.Tensor


def _stage(tensor: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A contiguous ``dtype`` copy of a host tensor on ``device``, read from its pinned copy if any."""
    staged = torch.empty(tensor.shape, dtype=dtype, device=device)
    transfer_(staged, tensor, non_blocking=True)
    return staged


def _materialize_weight_factors(
    factors: Sequence[ScaledLoRAFactor],
) -> list[_MaterializedWeightFactor]:
    """Expose scaled factors as plain host tensors."""
    return [
        _MaterializedWeightFactor(
            coefficient=factor.coefficient,
            a=param_representation(factor.a.make_cpu_param()),
            b=param_representation(factor.b.make_cpu_param()),
        )
        for factor in factors
    ]


def _localize_materialized_weight_factors(
    factors: Sequence[_MaterializedWeightFactor],
    *,
    out_range: tuple[int, int],
    in_range: tuple[int, int],
) -> list[_MaterializedWeightFactor]:
    """Slice factors to a local region, retaining broadcast singleton axes."""
    out_offset, out_size = out_range
    in_offset, in_size = in_range
    return [
        _MaterializedWeightFactor(
            coefficient=factor.coefficient,
            a=(
                factor.a.expand(-1, in_size) if factor.a.shape[1] == 1
                else factor.a.narrow(1, in_offset, in_size)
            ),
            b=(
                factor.b.expand(out_size, -1) if factor.b.shape[0] == 1
                else factor.b.narrow(0, out_offset, out_size)
            ),
        )
        for factor in factors
    ]


def _pack_materialized_weight_factors(
    data: torch.Tensor,
    factors: Sequence[_MaterializedWeightFactor],
    *,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack LoRAs at their common matrix shape, without repeating stacked heads.

    Host tensors copy directly into their destination slices through transfer_;
    singleton expansion happens during the copy. Factors pack across the rank
    dimension in the target's compute dtype. Each coefficient scales only its
    staged A slice, so the combined merge uses ``alpha=1``. The packed buffers
    die after the merge.
    """
    rows, cols = torch.broadcast_shapes(*[
        (factor.b.shape[0], factor.a.shape[1]) for factor in factors
    ])
    total_rank = sum(factor.a.shape[0] for factor in factors)
    a_packed = torch.empty(
        (total_rank, cols),
        device=data.device,
        dtype=compute_dtype,
    )
    b_packed = torch.empty(
        (rows, total_rank),
        device=data.device,
        dtype=compute_dtype,
    )

    rank_offset = 0
    for factor in factors:
        next_offset = rank_offset + factor.a.shape[0]
        a_slice = a_packed[rank_offset:next_offset]
        b_slice = b_packed[:, rank_offset:next_offset]
        transfer_(a_slice, factor.a, non_blocking=True)
        transfer_(b_slice, factor.b, non_blocking=True)
        if factor.coefficient != 1.0:
            # Scaling the contiguous A slice keeps B's strided destination
            # copy as the only non-contiguous operation for each factor.
            a_slice.mul_(factor.coefficient)
        rank_offset = next_offset

    return a_packed, b_packed


@dataclass(slots=True, frozen=True)
class _LoRAWeightPlan:
    """Validated, device-independent inputs for repeated weight merges."""

    adapter: LoRAMergeTensorAdapter[Any, Any]
    factors: tuple[_MaterializedWeightFactor, ...]
    compute_dtype: torch.dtype


class LoRATransform:
    """Apply low-rank factors to one weight.

    Holds references to Adapter-owned host tensors — no cloning or capture
    happens here. Multiple ordinary factors are packed into transient buffers
    and applied as one update with ``beta * weight`` before terminal encoding.
    ``beta`` must be finite and defaults to preserving the starting weight.
    Validation builds a reusable execution plan,
    keeping validation-only work out of repeated offload copy hooks.
    """

    __slots__ = (
        "_beta",
        "_factors",
        "_merge_index",
        "_stochastic_rounding",
        "_target_key",
        "_weight_plan",
    )

    def __init__(
        self,
        factors: Sequence[ScaledLoRAFactor],
        *,
        beta: float = 1.0,
        stochastic_rounding: bool = False,
        target_key: str = "",
    ) -> None:
        if not factors:
            raise ValueError("LoRATransform requires a factor")
        if not math.isfinite(beta):
            raise ValueError(f"LoRA base coefficient must be finite; got {beta}.")
        if stochastic_rounding and not target_key:
            raise ValueError("Stochastic LoRATransform requires a non-empty target_key.")
        self._factors = tuple(factors)
        self._beta = beta
        self._stochastic_rounding = stochastic_rounding
        self._target_key = target_key
        self._merge_index = 0
        self._weight_plan: _LoRAWeightPlan | None = None

    def _rounding_seed(self) -> int | None:
        if not self._stochastic_rounding:
            return None
        return derive_seed(self._target_key, self._merge_index)

    def validate_parameter(self, param: nn.Parameter) -> None:
        """Validate ``param`` and prepare subsequent applications."""
        self._weight_plan = None
        self._weight_plan = self._build_weight_plan(param)

    def _build_weight_plan(self, param: nn.Parameter) -> _LoRAWeightPlan:
        """Validate a weight target and return its reusable merge plan."""
        representation = param_representation(param)
        if representation.is_meta:
            raise ValueError(
                "Meta parameters can be populated only by parameter values; low-rank A/B factors are unsupported."
            )
        adapter = _select_lora_merge_adapter(representation)
        logical_shape = adapter.logical_shape(representation)
        compute_dtype = adapter.compute_dtype(representation)
        rounding_seed = self._rounding_seed()
        _validate_factor_shapes(
            self._factors,
            logical_shape,
        )
        if any(factor.produced_shape != logical_shape for factor in self._factors):
            validate_broadcast_merge(adapter, representation)
        factors = self._localize_weight_factors(
            representation,
            adapter,
            _materialize_weight_factors(self._factors),
        )
        if isinstance(adapter, LoRAMergeValidationTensorAdapter):
            adapter.validate_lora_merge(
                representation,
                rounding_seed=rounding_seed,
            )
        return _LoRAWeightPlan(
            adapter,
            tuple(factors),
            compute_dtype,
        )

    def apply_parameter(self, param: nn.Parameter) -> None:
        """Apply this transform to a previously validated weight target."""
        plan = self._weight_plan
        if plan is None:
            raise RuntimeError("LoRA weight target must be validated before application.")
        data = param_representation(param)
        b, a, alpha = self._stage_single_or_packed_update(
            data,
            plan.factors,
            compute_dtype=plan.compute_dtype,
        )
        plan.adapter.merge_lora_(
            data,
            b,
            a,
            alpha,
            beta=self._beta,
            rounding_seed=self._rounding_seed(),
        )
        self._merge_index += 1

    def storage_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the physical host tensors used to stage all factors."""
        return tuple(
            tensor
            for factor in self._factors
            for backing in (factor.a, factor.b)
            for tensor in backing.storage_tensors()
        )

    @staticmethod
    def _localize_weight_factors(
        data: torch.Tensor,
        adapter: LoRAMergeTensorAdapter[Any, Any],
        factors: Sequence[_MaterializedWeightFactor],
    ) -> list[_MaterializedWeightFactor]:
        """Slice host-backed factors to an adapter-local merge region."""
        if not isinstance(adapter, MergeLocalityTensorAdapter):
            return list(factors)

        local_shape, offsets = adapter.merge_local_shape_and_offsets(data)
        out_size, in_size = local_shape[-2:]
        out_offset, in_offset = offsets[-2:]
        return _localize_materialized_weight_factors(
            factors,
            out_range=(out_offset, out_size),
            in_range=(in_offset, in_size),
        )

    @staticmethod
    def _stage_single_or_packed_update(
        data: torch.Tensor,
        factors: Sequence[_MaterializedWeightFactor],
        *,
        compute_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        if len(factors) == 1:
            factor = factors[0]
            return (
                _stage(factor.b, device=data.device, dtype=compute_dtype),
                _stage(factor.a, device=data.device, dtype=compute_dtype),
                factor.coefficient,
            )

        a, b = _pack_materialized_weight_factors(
            data,
            factors,
            compute_dtype=compute_dtype,
        )
        return b, a, 1.0


def _validate_factor_shapes(
    factors: Sequence[ScaledLoRAFactor],
    target_shape: tuple[int, ...],
) -> None:
    # Per-pair validity (2-D, matching inner rank) is guaranteed by
    # ScaledLoRAFactor construction; only the match against this concrete
    # target shape is checked here.
    for factor in factors:
        if torch.broadcast_shapes(factor.produced_shape, target_shape) != target_shape:
            raise ValueError(
                "LoRA update would expand the target: "
                f"B@A produces {factor.produced_shape}, target shape is {target_shape}."
            )


def _select_lora_merge_adapter(
    data: torch.Tensor,
) -> LoRAMergeTensorAdapter[Any, Any]:
    try:
        adapter = select_adapter(data)
    except NotImplementedError as exc:
        raise ValueError(
            f"Tensor type {type(data).__name__} has no registered tensor adapter. "
            "Merge requires a tensor adapter with LoRA merge support."
        ) from exc

    if not isinstance(adapter, LoRAMergeTensorAdapter):
        raise ValueError(f"{adapter_name(adapter)} does not support LoRA merge.")

    compute_dtype = adapter.compute_dtype(data)
    if not compute_dtype.is_floating_point:
        raise ValueError(f"LoRA merge requires a floating-point compute dtype, got {compute_dtype}.")
    return adapter
