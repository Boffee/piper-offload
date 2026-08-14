"""LoRA types and per-weight merge / routed transforms.

:class:`LoRA` pairs and validates factor matrices from a flat safetensors state
dict at construction. Legacy PEFT ``lora_B.bias`` vectors are retained as an
optional third tensor on their factor pair. By default the resource owns pinned
copies; strict adoption mode instead adopts existing CPU allocations, including
mmap-backed storage. Merge and routed consumers may share either immutable host
backing.

Two application paths apply the resource's factors:

- :class:`LoRATransform` (merge mode) — represents the joint weight and
  optional bias update. Permanent merge applies it as one logical operation;
  block streaming invokes its partial weight and bias operations after each
  parameter's DMA. Weight updates delegate to the target tensor's adapter.
- routed mode (:func:`install_routed_residual_hook`) — a forward-PRE hook
  copies the target's factors from pinned CPU storage to the input device for
  that invocation; a forward-POST hook adds
  ``strength * ((x @ A.T) @ B.T + bias)`` to the layer's output and
  drops those device copies. The base weight is not touched in place.
  Restricted to ``nn.Linear``
  parents (other layer types raise); shared weight storage is allowed (the
  hook targets the matched module, not the weight bytes). Factors are cast to
  the layer's output dtype before the residual, so quantized bases work as
  long as the matched module exposes a compatible logical ``nn.Linear``
  shape. Formats whose logical shape differs from their packed storage shape
  still need a richer per-format LoRA layer.

:class:`~piper_offload.ModelOffloader` is the consumer-facing API; its
``activate(..., loras=..., lora_mode=...)`` receives the requested path once
the device is known. The merge path runs :class:`LoRATransform` partial
operations from activation-scoped post-copy hooks; the routed path lives as
forward hooks installed on activate and removed on deactivate.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, Self, runtime_checkable

import torch
from torch import nn

from .dtensor_adapter import DTensorAdapter
from .host_backing import (
    HostBacking,
    validate_host_backing,
)
from .pinned_param import PinnedParam
from .seeding import derive_seed
from .tensor_adapter_registry import param_representation, select_adapter
from .tensor_adapters import (
    LoRAMergeTensorAdapter,
    LoRAMergeValidationTensorAdapter,
    adapter_name,
)

__all__ = [
    "LoRA",
    "LoRAFactor",
    "LoRAMode",
    "LoRATransform",
    "ScaledLoRAFactor",
]

type LoRAMode = Literal["merge", "routed"]
type _RawLoRAFactor = tuple[float, torch.Tensor, torch.Tensor]


@runtime_checkable
class _FactorAwareLoRAMergeAdapter(Protocol):
    """Optional staging path for formats that transform individual factors.

    The ordinary packer folds each strength into an already-low-precision
    ``A`` slice. Formats whose stored-weight coordinates require another
    transform can instead stage every factor atomically, before packing loses
    the original strength boundaries.
    """

    @staticmethod
    def stage_lora_factors(
        target: torch.Tensor,
        factors: Sequence[_RawLoRAFactor],
        *,
        logical_shape: tuple[int, ...],
        compute_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, float] | None:
        """Return a prepared ``(B, A, strength)`` or defer to normal staging."""
        ...

    @staticmethod
    def validate_prepared_lora_merge(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Validate a prepared update without transforming it again."""
        ...

    @staticmethod
    def merge_prepared_lora_(
        target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        """Apply a prepared update without transforming it again."""
        ...


@dataclass(slots=True, frozen=True)
class LoRAFactor:
    """A LoRA's host-backed tensors for one target weight.

    ``a`` is the ``(rank, in_dim)`` down-projection and ``b`` the
    ``(out_dim, rank)`` up-projection, each held as a :class:`PinnedParam`.
    ``bias`` optionally carries a legacy PEFT ``lora_B.bias`` vector of
    length ``out_dim``. Modern A/B-only LoRAs leave it as ``None``.
    Strength is *not* part of the pair — it is extrinsic and supplied when the
    LoRA is bound to a target. Per-pair shape
    validity is checked before capture (in :func:`_validate_factor_pair`); the
    match against a concrete target shape is checked separately, where the
    target is known.

    :meth:`scaled` binds the extrinsic strength without discarding the pinned
    representation, so each application path can materialize the factors
    through their tensor adapters.
    """

    a: PinnedParam
    b: PinnedParam
    bias: PinnedParam | None = None

    @property
    def cache_bytes(self) -> int:
        """Host-backing bytes held by this target's adapter tensors."""
        bias_bytes = 0 if self.bias is None else self.bias.cache_bytes
        return self.a.cache_bytes + self.b.cache_bytes + bias_bytes

    def scaled(self, strength: float) -> ScaledLoRAFactor:
        """Bind this host-backed factor pair to ``strength``."""
        return ScaledLoRAFactor(self.a, self.b, strength, self.bias)


@dataclass(slots=True, frozen=True)
class ScaledLoRAFactor:
    """Host-backed LoRA tensors bound to an application ``strength``.

    The application-side carrier used by :class:`LoRATransform` and routed
    hooks. Keeping :class:`PinnedParam` rather than CPU tensor views preserves
    adapter-specific reconstruction metadata such as a ``DTensor``'s original
    device mesh. The contribution to the base weight is
    ``strength * (b @ a)``. When ``bias`` is present, the corresponding
    output contribution is ``strength * bias``.

    Use :meth:`from_tensors` when constructing a standalone transform from
    unpinned tensors. LoRA resources normally create this through
    :meth:`LoRAFactor.scaled` and reuse their existing host backing.
    """

    a: PinnedParam
    b: PinnedParam
    strength: float
    bias: PinnedParam | None = None

    def __post_init__(self) -> None:
        if (
            len(self.a.shape) != 2
            or len(self.b.shape) != 2
            or self.a.shape[0] != self.b.shape[1]
        ):
            raise ValueError(
                f"LoRA factor shape mismatch: A shape is {tuple(self.a.shape)}, B shape is {tuple(self.b.shape)}."
            )
        if self.bias is not None and (
            len(self.bias.shape) != 1
            or self.bias.shape[0] != self.b.shape[0]
        ):
            raise ValueError(
                "LoRA bias shape mismatch: "
                f"bias shape is {tuple(self.bias.shape)}, "
                f"B shape is {tuple(self.b.shape)}. Expected bias=(out_dim,)."
            )

    @classmethod
    def from_tensors(
        cls,
        a: torch.Tensor,
        b: torch.Tensor,
        strength: float,
        bias: torch.Tensor | None = None,
    ) -> Self:
        """Pin unbound adapter tensors and bind them to ``strength``."""
        return cls(
            PinnedParam(nn.Parameter(a, requires_grad=False)),
            PinnedParam(nn.Parameter(b, requires_grad=False)),
            strength,
            None
            if bias is None
            else PinnedParam(nn.Parameter(bias, requires_grad=False)),
        )

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
        """Shape of ``b @ a`` — the base-weight shape this factor targets."""
        return (self.b.shape[0], self.a.shape[1])


class LoRA:
    """Reusable immutable host-backed LoRA resource.

    Build once from a flat ``state_dict``: factor pairs are validated, cast to
    the optional storage ``dtype``, and pinned directly by default. Adopt mode
    retains compatible CPU factor storage without copying it. The resource
    retains the resulting factor tensors but not the raw ``state_dict``
    mapping.

    Satisfies :class:`~piper_offload.protocols.ResourceStore`, so it can be
    registered in :class:`~piper_offload.ResourceCache` for budget tracking and
    policy-driven eviction. Merge and routed consumers read the same immutable
    factor backing and may overlap.

    Strength is extrinsic — specify it when passing the resource to
    :meth:`ModelOffloader.activate` via ``lora_strengths``.

    ``state_dict`` keys must already be model parameter paths (``.lora_A`` /
    ``.lora_B`` suffixed). Any key remapping — e.g. stripping the
    ``diffusion_model.`` prefix on ComfyUI adapters — is the caller's
    responsibility, done in the factory that produces the state dict.
    """

    def __init__(self, targets: Mapping[str, LoRAFactor]) -> None:
        self._targets = MappingProxyType(dict(targets))
        self._cache_bytes = sum(factor.cache_bytes for factor in self._targets.values())

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        dtype: torch.dtype | None = None,
        host_backing: HostBacking = "pinned",
    ) -> Self:
        """Pair, validate, and build ``state_dict`` into a LoRA.

        ``dtype`` casts every factor before pinned capture. For routed mode,
        matching the model's compute dtype reduces storage and per-forward H2D
        traffic. Left as ``None``, factors keep their stored dtype. Merge mode
        casts at apply time regardless. ``host_backing="adopt"`` strictly
        adopts existing CPU storage and therefore rejects any ``dtype`` that
        would require conversion. Adopted factor storage must remain immutable
        for the resource's lifetime.
        """
        backing = validate_host_backing(host_backing)
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError(f"LoRA dtype must be floating-point, got {dtype}.")
        _validate_lora_state_dict(state_dict)
        if backing == "adopt":
            _validate_adopted_lora_dtype(state_dict, dtype=dtype)
        return cls(
            _build_lora_targets(
                state_dict,
                dtype=dtype,
                pin_memory=backing == "pinned",
            )
        )

    @property
    def targets(self) -> Mapping[str, LoRAFactor]:
        """Immutable target-weight to pinned-factor mapping."""
        return self._targets

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes


class LoRATransform:
    """LoRA factors applied to a base weight and optional base bias.

    Holds references to LoRA-owned host factor matrices — no cloning or
    pinning happens here. :meth:`apply` performs the complete logical update
    across the weight and optional bias. Multiple ordinary factors are packed
    into transient buffers and applied as one weight update. The explicit
    :meth:`apply_weight` and :meth:`apply_bias` partial operations support
    offload paths where those parameters are copied independently. Target
    :class:`~torch.nn.Parameter` objects are always preserved.

    Validation is an explicit phase: call :meth:`validate_target` before a
    joint :meth:`apply`, or the corresponding partial validation method before
    a partial application when the caller requires non-mutating preflight.
    """

    __slots__ = (
        "_factors",
        "_merge_index",
        "_stochastic_rounding",
        "_target_key",
    )

    def __init__(
        self,
        factors: list[ScaledLoRAFactor],
        *,
        stochastic_rounding: bool = False,
        target_key: str = "",
    ) -> None:
        if stochastic_rounding and not target_key:
            raise ValueError(
                "Stochastic LoRATransform requires a non-empty target_key."
            )
        self._factors = factors
        self._stochastic_rounding = stochastic_rounding
        self._target_key = target_key
        self._merge_index = 0

    def _rounding_seed(self) -> int | None:
        if not self._stochastic_rounding:
            return None
        return derive_seed(self._target_key, self._merge_index)

    def validate_target(
        self,
        weight: nn.Parameter,
        bias: nn.Parameter | None = None,
    ) -> None:
        """Preflight the complete weight and optional bias update.

        A transform containing adapter bias requires an explicit base ``bias``
        target. Modern weight-only transforms continue to accept only
        ``weight``.
        """
        self.validate_weight_target(weight)
        if not self.has_bias:
            return
        if bias is None:
            raise ValueError(
                "LoRA transform contains a bias but no base bias target was provided."
            )
        self.validate_bias_target(bias)

    def validate_weight_target(self, param: nn.Parameter) -> None:
        """Raise if ``param`` cannot receive this LoRA weight update.

        This is an optional preflight for callers that need validation to be
        separate from application.
        """
        representation = param_representation(param)
        adapter = _select_lora_merge_adapter(representation)
        logical_shape = adapter.logical_shape(representation)
        rounding_seed = self._rounding_seed()
        _validate_factor_shapes(
            self._factors,
            logical_shape,
        )
        factor_tensors, staging_shape = self._localize_factor_tensors(
            representation,
            adapter,
            self._factor_tensors(),
            logical_shape=logical_shape,
        )
        if not isinstance(adapter, LoRAMergeValidationTensorAdapter):
            self._validate_staged_factors(factor_tensors)
            return
        staged, prepared = self._stage_update_for_adapter(
            representation,
            adapter,
            factor_tensors,
            logical_shape=staging_shape,
            compute_dtype=adapter.compute_dtype(representation),
        )
        b, a, strength = staged
        _validate_lora_merge(
            adapter,
            representation,
            b,
            a,
            strength,
            prepared=prepared,
            rounding_seed=rounding_seed,
        )

    def apply(
        self,
        weight: nn.Parameter,
        bias: nn.Parameter | None = None,
    ) -> None:
        """Apply the complete update after any caller-required preflight."""
        has_bias = self.has_bias
        if has_bias and bias is None:
            raise ValueError(
                "LoRA transform contains a bias but no base bias target was provided."
            )
        self.apply_weight(weight)
        if has_bias and bias is not None:
            self.apply_bias(bias)

    def apply_weight(self, param: nn.Parameter) -> None:
        """Apply only this transform's weight update to ``param``."""
        # Operate on the representation tensor: ``param.data`` for plain and
        # wrapped-quant parameters, but the param itself for a Parameter
        # subclass whose ``.data`` is lossy (bitsandbytes Params4bit).
        representation = param_representation(param)
        adapter = _select_lora_merge_adapter(representation)
        logical_shape = adapter.logical_shape(representation)
        rounding_seed = self._rounding_seed()
        _validate_factor_shapes(self._factors, logical_shape)
        self._apply_merge(
            representation,
            adapter,
            logical_shape=logical_shape,
            compute_dtype=adapter.compute_dtype(representation),
            rounding_seed=rounding_seed,
        )
        self._merge_index += 1

    @property
    def has_bias(self) -> bool:
        """Whether any factor includes a legacy ``lora_B.bias`` vector."""
        return any(factor.bias is not None for factor in self._factors)

    def validate_bias_target(self, param: nn.Parameter) -> None:
        """Raise unless ``param`` can receive this transform's bias update."""
        if not self.has_bias:
            raise ValueError("LoRA bias merge requires at least one bias")

        target = param_representation(param)
        if type(target) is not torch.Tensor:
            raise ValueError(
                "LoRA bias merge requires a plain dense base bias; "
                f"got {type(target).__name__}."
            )
        if not target.is_floating_point():
            raise ValueError(
                "LoRA bias merge requires a floating-point base bias; "
                f"got {target.dtype}."
            )
        if target.dim() != 1:
            raise ValueError(
                "LoRA bias merge requires a rank-one base bias; "
                f"got shape {tuple(target.shape)}."
            )

        for factor in self._factors:
            if factor.bias is None:
                continue
            source = param_representation(factor.bias.make_cpu_param())
            if type(source) is not torch.Tensor or source.device.type != "cpu":
                raise ValueError(
                    "LoRA bias merge requires a plain CPU adapter bias; "
                    f"got {type(source).__name__} on {source.device}."
                )
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    "LoRA bias shape mismatch: "
                    f"adapter bias shape is {tuple(source.shape)}, "
                    f"base bias shape is {tuple(target.shape)}."
                )

    def apply_bias(self, param: nn.Parameter) -> None:
        """Apply only the bias update to a previously validated base bias."""
        target = param_representation(param)
        delta = torch.zeros_like(target)
        for factor in self._factors:
            if factor.bias is None:
                continue
            source = param_representation(factor.bias.make_cpu_param())
            staged = source.to(
                device=target.device,
                dtype=target.dtype,
                non_blocking=True,
            )
            delta.add_(staged, alpha=factor.strength)
        target.add_(delta)

    def _factor_tensors(
        self,
    ) -> list[tuple[ScaledLoRAFactor, torch.Tensor, torch.Tensor]]:
        return [
            (
                factor,
                param_representation(factor.a.make_cpu_param()),
                param_representation(factor.b.make_cpu_param()),
            )
            for factor in self._factors
        ]

    def _apply_merge(
        self,
        data: torch.Tensor,
        adapter: LoRAMergeTensorAdapter[Any, Any],
        *,
        logical_shape: tuple[int, ...],
        compute_dtype: torch.dtype,
        rounding_seed: int | None,
    ) -> None:
        """Stage one combined update and delegate it to the target adapter."""
        factor_tensors = self._factor_tensors()
        factor_tensors, staging_shape = self._localize_factor_tensors(
            data,
            adapter,
            factor_tensors,
            logical_shape=logical_shape,
        )
        staged, prepared = self._stage_update_for_adapter(
            data,
            adapter,
            factor_tensors,
            logical_shape=staging_shape,
            compute_dtype=compute_dtype,
        )
        b, a, strength = staged
        _validate_lora_merge(
            adapter,
            data,
            b,
            a,
            strength,
            prepared=prepared,
            rounding_seed=rounding_seed,
        )
        if prepared:
            assert isinstance(adapter, _FactorAwareLoRAMergeAdapter)
            adapter.merge_prepared_lora_(
                data,
                b,
                a,
                strength,
                rounding_seed=rounding_seed,
            )
        else:
            adapter.merge_lora_(
                data,
                b,
                a,
                strength,
                rounding_seed=rounding_seed,
            )

    @staticmethod
    def _localize_factor_tensors(
        data: torch.Tensor,
        adapter: LoRAMergeTensorAdapter[Any, Any],
        factor_tensors: Sequence[tuple[ScaledLoRAFactor, torch.Tensor, torch.Tensor]],
        *,
        logical_shape: tuple[int, ...],
    ) -> tuple[
        list[tuple[ScaledLoRAFactor, torch.Tensor, torch.Tensor]],
        tuple[int, ...],
    ]:
        """Slice host-backed factors to a DTensor target's local shard."""
        if not isinstance(adapter, DTensorAdapter):
            return list(factor_tensors), logical_shape

        (out_offset, out_size), (in_offset, in_size) = adapter.lora_factor_ranges(data)
        localized = [
            (
                factor,
                a.narrow(1, in_offset, in_size),
                b.narrow(0, out_offset, out_size),
            )
            for factor, a, b in factor_tensors
        ]
        return localized, (out_size, in_size)

    @classmethod
    def _stage_single_or_packed_update(
        cls,
        data: torch.Tensor,
        factor_tensors: Sequence[tuple[ScaledLoRAFactor, torch.Tensor, torch.Tensor]],
        *,
        logical_shape: tuple[int, ...],
        compute_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        cls._validate_staged_factors(factor_tensors)
        if len(factor_tensors) == 1:
            factor, a, b = factor_tensors[0]
            return (
                b.to(
                    device=data.device,
                    dtype=compute_dtype,
                    non_blocking=True,
                ).contiguous(),
                a.to(
                    device=data.device,
                    dtype=compute_dtype,
                    non_blocking=True,
                ).contiguous(),
                factor.strength,
            )

        a, b = cls._pack_factors(
            data,
            factor_tensors,
            logical_shape=logical_shape,
            compute_dtype=compute_dtype,
        )
        return b, a, 1.0

    @classmethod
    def _stage_update_for_adapter(
        cls,
        data: torch.Tensor,
        adapter: LoRAMergeTensorAdapter[Any, Any],
        factor_tensors: Sequence[tuple[ScaledLoRAFactor, torch.Tensor, torch.Tensor]],
        *,
        logical_shape: tuple[int, ...],
        compute_dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, float], bool]:
        """Let an adapter preserve factor boundaries before normal packing."""
        cls._validate_staged_factors(factor_tensors)
        if isinstance(adapter, _FactorAwareLoRAMergeAdapter):
            prepared = adapter.stage_lora_factors(
                data,
                tuple((factor.strength, a, b) for factor, a, b in factor_tensors),
                logical_shape=logical_shape,
                compute_dtype=compute_dtype,
            )
            if prepared is not None:
                return prepared, True
        return (
            cls._stage_single_or_packed_update(
                data,
                factor_tensors,
                logical_shape=logical_shape,
                compute_dtype=compute_dtype,
            ),
            False,
        )

    @classmethod
    def _validate_staged_factors(
        cls,
        factor_tensors: Sequence[tuple[ScaledLoRAFactor, torch.Tensor, torch.Tensor]],
    ) -> None:
        if cls._are_plain_cpu_factors(factor_tensors):
            return
        raise ValueError(
            "LoRA merge requires plain CPU torch.Tensor factors; wrapped factor representations are unsupported."
        )

    @staticmethod
    def _are_plain_cpu_factors(
        factor_tensors: Sequence[tuple[ScaledLoRAFactor, torch.Tensor, torch.Tensor]],
    ) -> bool:
        return all(
            type(a) is torch.Tensor and type(b) is torch.Tensor and a.device.type == "cpu" and b.device.type == "cpu"
            for _factor, a, b in factor_tensors
        )

    @staticmethod
    def _pack_factors(
        data: torch.Tensor,
        factor_tensors: Sequence[tuple[ScaledLoRAFactor, torch.Tensor, torch.Tensor]],
        *,
        logical_shape: tuple[int, ...],
        compute_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage several LoRAs as one update without per-factor device tensors.

        The temporary packed buffers contain exactly the combined factors for
        this weight and die after the merge. CPU factors copy directly into
        their destination slices, avoiding the extra individual device
        tensors that a target-side ``torch.cat`` would require.
        """
        total_rank = sum(factor.rank for factor, _a, _b in factor_tensors)
        a_packed = torch.empty(
            (total_rank, logical_shape[1]),
            device=data.device,
            dtype=compute_dtype,
        )
        b_packed = torch.empty(
            (logical_shape[0], total_rank),
            device=data.device,
            dtype=compute_dtype,
        )

        rank_offset = 0
        for factor, a, b in factor_tensors:
            next_offset = rank_offset + factor.rank
            a_slice = a_packed[rank_offset:next_offset]
            b_slice = b_packed[:, rank_offset:next_offset]
            a_slice.copy_(a, non_blocking=True)
            b_slice.copy_(b, non_blocking=True)
            if factor.strength != 1.0:
                # Scaling the contiguous A slice keeps B's strided destination
                # copy as the only non-contiguous operation for each factor.
                a_slice.mul_(factor.strength)
            rank_offset = next_offset

        return a_packed, b_packed


def _validate_factor_shapes(
    factors: Sequence[ScaledLoRAFactor],
    target_shape: tuple[int, ...],
) -> None:
    # Per-pair validity (2-D, matching inner rank) is guaranteed by
    # ScaledLoRAFactor construction; only the match against this concrete
    # target shape is checked here.
    for factor in factors:
        if factor.produced_shape != target_shape:
            raise ValueError(
                f"LoRA factor shape mismatch: B@A produces {factor.produced_shape}, target shape is {target_shape}."
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
        raise ValueError(f"{adapter_name(adapter)} does not support LoRA merge. Use routed LoRA for this tensor type.")

    compute_dtype = adapter.compute_dtype(data)
    if not compute_dtype.is_floating_point:
        raise ValueError(f"LoRA merge requires a floating-point compute dtype, got {compute_dtype}.")
    return adapter


def _validate_lora_merge(
    adapter: LoRAMergeTensorAdapter[Any, Any],
    target: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
    *,
    prepared: bool = False,
    rounding_seed: int | None = None,
) -> None:
    """Run adapter-specific staged-merge checks without mutation."""
    if prepared:
        assert isinstance(adapter, _FactorAwareLoRAMergeAdapter)
        adapter.validate_prepared_lora_merge(
            target,
            b,
            a,
            strength,
            rounding_seed=rounding_seed,
        )
    elif isinstance(adapter, LoRAMergeValidationTensorAdapter):
        adapter.validate_lora_merge(
            target,
            b,
            a,
            strength,
            rounding_seed=rounding_seed,
        )


@dataclass(slots=True, frozen=True)
class _StagedLoRAFactor:
    """Adapter tensors materialized for one forward invocation."""

    a: nn.Parameter
    b: nn.Parameter
    strength: float
    bias: nn.Parameter | None = None


def _routed_residual(
    x: torch.Tensor,
    factors: Sequence[_StagedLoRAFactor],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Routed contribution ``Σ strength_i · ((x @ A_i.T) @ B_i.T + bias_i)``.

    Strength scales the intermediate ``M·r`` projection (cheaper than scaling
    the ``M·out`` result, and keeps it extrinsic to the stored factors rather
    than baked into a buffer).
    """
    x_compute = x.to(dtype=output_dtype)
    total: torch.Tensor | None = None
    for factor in factors:
        # The PRE hook has already reconstructed the factors on their proper
        # device representation. A dtype-only cast preserves wrappers such as
        # DTensor and their device meshes.
        a = param_representation(factor.a).to(dtype=output_dtype)
        b = param_representation(factor.b).to(dtype=output_dtype)
        part = ((x_compute @ a.T) * factor.strength) @ b.T
        if factor.bias is not None:
            bias = param_representation(factor.bias).to(
                dtype=output_dtype,
            )
            part = part + bias * factor.strength
        total = part if total is None else total + part
    if total is None:
        raise ValueError("Routed LoRA residual requires at least one factor")
    return total


def _linear_input(
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    if inputs:
        x = inputs[0]
    else:
        try:
            x = kwargs["input"]
        except KeyError as exc:
            raise TypeError(
                "Routed LoRA expected the Linear input as either the first positional argument or the 'input' keyword"
            ) from exc
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Routed LoRA requires the Linear input to be a torch.Tensor; got {type(x).__name__}")
    return x


def _stage_routed_factors(
    factors: Sequence[ScaledLoRAFactor],
    x: torch.Tensor,
) -> tuple[_StagedLoRAFactor, ...]:
    """Materialize host factors on the invocation's input device."""
    return tuple(
        _StagedLoRAFactor(
            factor.a.materialize(x.device, non_blocking=True),
            factor.b.materialize(x.device, non_blocking=True),
            factor.strength,
            None
            if factor.bias is None
            else factor.bias.materialize(x.device, non_blocking=True),
        )
        for factor in factors
    )


def install_routed_residual_hook(
    parent: nn.Module,
    factors: Sequence[ScaledLoRAFactor],
) -> Callable[[], None]:
    """Stage host factors in a PRE hook and add their residual in POST.

    Returns an idempotent callable that removes both hooks. One hook pair
    covers every LoRA targeting this parent. Device copies are scoped to a
    single invocation and released after the residual is enqueued, so routed
    mode needs no adapter activation lifecycle or block scheduler.
    """
    if not factors:
        raise ValueError("Routed LoRA hook requires at least one factor")

    staged_factors: list[tuple[_StagedLoRAFactor, ...]] = []

    def pre_hook(
        _module: nn.Module,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        x = _linear_input(inputs, kwargs)
        staged_factors.append(_stage_routed_factors(factors, x))

    def post_hook(
        _module: nn.Module,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: object,
    ) -> object:
        # ``always_call=True`` also reaches this hook when the Linear or an
        # earlier hook raises. In that case there may be no staged entry or
        # Tensor output; only discard any completed staging work.
        staged = staged_factors.pop() if staged_factors else None
        if staged is None or not isinstance(output, torch.Tensor):
            return output
        x = _linear_input(inputs, kwargs)
        return output + _routed_residual(x, staged, output.dtype)

    pre_handle = parent.register_forward_pre_hook(
        pre_hook,
        with_kwargs=True,
    )
    try:
        post_handle = parent.register_forward_hook(
            post_hook,
            with_kwargs=True,
            always_call=True,
        )
    except BaseException:
        pre_handle.remove()
        raise

    removed = False

    def remove_hooks() -> None:
        nonlocal removed
        if removed:
            return
        post_handle.remove()
        pre_handle.remove()
        staged_factors.clear()
        removed = True

    return remove_hooks


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _build_lora_targets(
    state_dict: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype | None = None,
    pin_memory: bool = True,
) -> Mapping[str, LoRAFactor]:
    """Build each validated adapter target without a module hierarchy."""
    a_tensors, b_tensors, bias_tensors = _split_factor_tensors(state_dict)
    factors: dict[str, LoRAFactor] = {}
    for base, a_source in a_tensors.items():
        b_source = b_tensors[base]
        bias_source = bias_tensors.get(base)
        a_tensor = a_source if dtype is None or a_source.dtype is dtype else a_source.to(dtype=dtype)
        b_tensor = b_source if dtype is None or b_source.dtype is dtype else b_source.to(dtype=dtype)
        bias_tensor = (
            bias_source
            if bias_source is None or dtype is None or bias_source.dtype is dtype
            else bias_source.to(dtype=dtype)
        )
        factors[f"{base}.weight"] = LoRAFactor(
            a=PinnedParam(
                nn.Parameter(a_tensor, requires_grad=False),
                pin_memory=pin_memory,
            ),
            b=PinnedParam(
                nn.Parameter(b_tensor, requires_grad=False),
                pin_memory=pin_memory,
            ),
            bias=None
            if bias_tensor is None
            else PinnedParam(
                nn.Parameter(bias_tensor, requires_grad=False),
                pin_memory=pin_memory,
            ),
        )
    return factors


def _validate_adopted_lora_dtype(
    state_dict: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype | None,
) -> None:
    """Reject a requested conversion that would defeat source adoption."""
    if dtype is None:
        return
    a_tensors, b_tensors, bias_tensors = _split_factor_tensors(state_dict)
    tensors_by_suffix = (
        ("A.weight", a_tensors),
        ("B.weight", b_tensors),
        ("B.bias", bias_tensors),
    )
    incompatible = [
        f"{base}.lora_{suffix}"
        for suffix, tensors in tensors_by_suffix
        for base, tensor in tensors.items()
        if tensor.dtype is not dtype
    ]
    if incompatible:
        raise ValueError(
            "adopted LoRA host backing cannot convert factor dtype without "
            "copying and losing source/mmap backing. Remove dtype=, convert "
            "the source before loading, or use host_backing='pinned'. "
            f"Mismatched factors: {incompatible!r}."
        )


def _validate_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    """Check ``state_dict`` is a well-formed LoRA before it is built.

    Every target needs a paired ``lora_A`` / ``lora_B`` and each factor must be
    a 2-D floating-point matrix with a matching inner rank. A legacy
    ``lora_B.bias`` is optional, but must accompany a complete pair and match
    B's output dimension.
    """
    a_tensors, b_tensors, bias_tensors = _split_factor_tensors(state_dict)

    if not a_tensors and not b_tensors:
        raise ValueError("LoRA state_dict contains no factor pairs")

    a_only = set(a_tensors) - set(b_tensors)
    b_only = set(b_tensors) - set(a_tensors)
    if a_only or b_only:
        raise ValueError(
            f"Unpaired LoRA factors: A-only={sorted(a_only)}, "
            f"B-only={sorted(b_only)}. Each target needs both "
            f".lora_A.weight and .lora_B.weight."
        )

    bias_only = set(bias_tensors) - set(a_tensors)
    if bias_only:
        raise ValueError(
            "Unpaired LoRA biases: "
            f"{sorted(bias_only)}. Each .lora_B.bias must accompany "
            "a complete .lora_A.weight / .lora_B.weight pair."
        )

    for base_key, a in a_tensors.items():
        _validate_factor_pair(
            f"{base_key}.weight",
            a,
            b_tensors[base_key],
            bias_tensors.get(base_key),
        )


def _split_factor_tensors(
    state_dict: dict[str, torch.Tensor],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    a_tensors: dict[str, torch.Tensor] = {}
    b_tensors: dict[str, torch.Tensor] = {}
    bias_tensors: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.endswith(".lora_A.weight"):
            a_tensors[key[: -len(".lora_A.weight")]] = tensor
        elif key.endswith(".lora_B.weight"):
            b_tensors[key[: -len(".lora_B.weight")]] = tensor
        elif key.endswith(".lora_B.bias"):
            bias_tensors[key[: -len(".lora_B.bias")]] = tensor
    return a_tensors, b_tensors, bias_tensors


def _validate_factor_pair(
    target_key: str,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    if not a.is_floating_point() or not b.is_floating_point():
        raise ValueError(
            f"LoRA factors for {target_key!r}: must be floating-point; got A.dtype={a.dtype}, B.dtype={b.dtype}."
        )
    if a.dim() != 2 or b.dim() != 2 or a.shape[0] != b.shape[1]:
        raise ValueError(
            f"LoRA factor shape mismatch for {target_key!r}: "
            f"A.shape={tuple(a.shape)}, B.shape={tuple(b.shape)}. "
            f"Expected A=(rank, in_dim), B=(out_dim, rank) with "
            f"A.shape[0] == B.shape[1]."
        )
    if bias is None:
        return
    if not bias.is_floating_point():
        raise ValueError(
            f"LoRA bias for {target_key!r} must be floating-point; "
            f"got dtype={bias.dtype}."
        )
    if bias.dim() != 1 or bias.shape[0] != b.shape[0]:
        raise ValueError(
            f"LoRA bias shape mismatch for {target_key!r}: "
            f"bias.shape={tuple(bias.shape)}, B.shape={tuple(b.shape)}. "
            "Expected bias=(out_dim,) with bias.shape[0] == B.shape[0]."
        )


def _lora_bias_target_key(weight_target_key: str) -> str:
    """Return the sibling base-bias key for a canonical LoRA weight target."""
    if not weight_target_key.endswith(".weight"):
        raise ValueError(
            f"LoRA target {weight_target_key!r} does not end in '.weight'."
        )
    return f"{weight_target_key.removesuffix('.weight')}.bias"
