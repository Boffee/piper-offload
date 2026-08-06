"""LoRA types and per-weight merge / routed transforms.

:class:`LoRA` pairs and validates factor matrices from a flat safetensors state
dict at construction. By default it owns pinned copies; strict adoption mode
instead adopts existing CPU allocations, including mmap-backed storage. Merge
and routed consumers may share either immutable host backing.

Two application paths apply the resource's factors:

- :class:`LoRATransform` (merge mode) — applied to the GPU parameter
  after DMA; integrates with block streaming. Stages the factors and delegates
  the in-place update to the target tensor's adapter.
- routed mode (:func:`install_routed_residual_hook`) — a forward-PRE hook
  copies the target's factors from pinned CPU storage to the input device for
  that invocation; a forward-POST hook adds
  ``strength * (x @ A.T) @ B.T`` to the layer's output and drops those device
  copies. The base weight is not touched in place. Restricted to ``nn.Linear``
  parents (other layer types raise); shared weight storage is allowed (the
  hook targets the matched module, not the weight bytes). Factors are cast to
  the layer's output dtype before the residual, so quantized bases work as
  long as the matched module exposes a compatible logical ``nn.Linear``
  shape. Formats whose logical shape differs from their packed storage shape
  still need a richer per-format LoRA layer.

:class:`~piper_offload.ModelOffloader` is the consumer-facing API; its
``activate(..., loras=..., lora_mode=...)`` receives the requested path once
the device is known. The merge path runs
:class:`LoRATransform` from an activation-scoped post-copy hook; the
routed path lives as forward hooks installed on activate and removed
on deactivate.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Self

import torch
from torch import nn

from .dtensor_adapter import DTensorAdapter
from .host_backing import (
    HostBacking,
    validate_host_backing,
)
from .pinned_param import PinnedParam
from .tensor_adapter_registry import param_representation, select_adapter
from .tensor_adapters import LoRAMergeTensorAdapter, adapter_name

__all__ = [
    "LoRA",
    "LoRAFactor",
    "LoRAMode",
    "LoRATransform",
    "ScaledLoRAFactor",
]

type LoRAMode = Literal["merge", "routed"]


@dataclass(slots=True, frozen=True)
class LoRAFactor:
    """A LoRA's host-backed factor pair for one target weight.

    ``a`` is the ``(rank, in_dim)`` down-projection and ``b`` the
    ``(out_dim, rank)`` up-projection, each held as a :class:`PinnedParam`.
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

    @property
    def cache_bytes(self) -> int:
        """Host-backing bytes held by this factor pair."""
        return self.a.cache_bytes + self.b.cache_bytes

    def scaled(self, strength: float) -> ScaledLoRAFactor:
        """Bind this host-backed factor pair to ``strength``."""
        return ScaledLoRAFactor(self.a, self.b, strength)


@dataclass(slots=True, frozen=True)
class ScaledLoRAFactor:
    """A host-backed factor pair bound to an application ``strength``.

    The application-side carrier used by :class:`LoRATransform` and routed
    hooks. Keeping :class:`PinnedParam` rather than CPU tensor views preserves
    adapter-specific reconstruction metadata such as a ``DTensor``'s original
    device mesh. The contribution to the base weight is
    ``strength * (b @ a)``.

    Use :meth:`from_tensors` when constructing a standalone transform from
    unpinned tensors. LoRA resources normally create this through
    :meth:`LoRAFactor.scaled` and reuse their existing host backing.
    """

    a: PinnedParam
    b: PinnedParam
    strength: float

    def __post_init__(self) -> None:
        if (
            len(self.a.shape) != 2
            or len(self.b.shape) != 2
            or self.a.shape[0] != self.b.shape[1]
        ):
            raise ValueError(
                f"LoRA factor shape mismatch: A shape is {tuple(self.a.shape)}, "
                f"B shape is {tuple(self.b.shape)}."
            )

    @classmethod
    def from_tensors(
        cls,
        a: torch.Tensor,
        b: torch.Tensor,
        strength: float,
    ) -> Self:
        """Pin an unbound tensor pair and bind it to ``strength``."""
        return cls(
            PinnedParam(nn.Parameter(a, requires_grad=False)),
            PinnedParam(nn.Parameter(b, requires_grad=False)),
            strength,
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
        self._cache_bytes = sum(
            factor.cache_bytes for factor in self._targets.values()
        )

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
    """Per-weight LoRA factors applied to one base parameter.

    Holds references to LoRA-owned host factor matrices — no cloning or
    pinning happens here. :meth:`apply` copies factors to the target
    parameter's device and delegates the update to its tensor adapter. Multiple
    ordinary factors are packed into transient buffers and applied as one
    update. The target
    :class:`~torch.nn.Parameter` object is always preserved.
    """

    __slots__ = ("_factors",)

    def __init__(self, factors: list[ScaledLoRAFactor]) -> None:
        self._factors = factors

    def validate_target(self, param: nn.Parameter) -> None:
        """Raise if ``param`` cannot receive this LoRA merge.

        This is an optional preflight for callers that want an earlier
        error. :meth:`apply` uses the same validation path immediately
        before mutating the target parameter.
        """
        representation = param_representation(param)
        adapter = _select_lora_merge_adapter(representation)
        logical_shape = adapter.logical_shape(representation)
        _validate_factor_shapes(
            self._factors,
            logical_shape,
        )
        factor_tensors, _ = self._localize_factor_tensors(
            representation,
            adapter,
            self._factor_tensors(),
            logical_shape=logical_shape,
        )
        self._validate_staged_factors(factor_tensors)

    def apply(self, param: nn.Parameter) -> None:
        # Operate on the representation tensor: ``param.data`` for plain and
        # wrapped-quant parameters, but the param itself for a Parameter
        # subclass whose ``.data`` is lossy (bitsandbytes Params4bit).
        representation = param_representation(param)
        adapter = _select_lora_merge_adapter(representation)
        logical_shape = adapter.logical_shape(representation)
        _validate_factor_shapes(self._factors, logical_shape)
        self._apply_merge(
            representation,
            adapter,
            logical_shape=logical_shape,
            compute_dtype=adapter.compute_dtype(representation),
        )

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
    ) -> None:
        """Stage one combined update and delegate it to the target adapter."""
        factor_tensors = self._factor_tensors()
        factor_tensors, staging_shape = self._localize_factor_tensors(
            data,
            adapter,
            factor_tensors,
            logical_shape=logical_shape,
        )
        staged = self._stage_single_or_packed_update(
            data,
            factor_tensors,
            logical_shape=staging_shape,
            compute_dtype=compute_dtype,
        )
        b, a, strength = staged
        adapter.merge_lora_(
            data,
            b,
            a,
            strength,
        )

    @staticmethod
    def _localize_factor_tensors(
        data: torch.Tensor,
        adapter: LoRAMergeTensorAdapter[Any, Any],
        factor_tensors: Sequence[
            tuple[ScaledLoRAFactor, torch.Tensor, torch.Tensor]
        ],
        *,
        logical_shape: tuple[int, ...],
    ) -> tuple[
        list[tuple[ScaledLoRAFactor, torch.Tensor, torch.Tensor]],
        tuple[int, ...],
    ]:
        """Slice host-backed factors to a DTensor target's local shard."""
        if not isinstance(adapter, DTensorAdapter):
            return list(factor_tensors), logical_shape

        (out_offset, out_size), (in_offset, in_size) = (
            adapter.lora_factor_ranges(data)
        )
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
        factor_tensors: Sequence[
            tuple[ScaledLoRAFactor, torch.Tensor, torch.Tensor]
        ],
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
    def _validate_staged_factors(
        cls,
        factor_tensors: Sequence[
            tuple[ScaledLoRAFactor, torch.Tensor, torch.Tensor]
        ],
    ) -> None:
        if cls._are_plain_cpu_factors(factor_tensors):
            return
        raise ValueError(
            "LoRA merge requires plain CPU torch.Tensor "
            "factors; wrapped factor representations are unsupported."
        )

    @staticmethod
    def _are_plain_cpu_factors(
        factor_tensors: Sequence[
            tuple[ScaledLoRAFactor, torch.Tensor, torch.Tensor]
        ],
    ) -> bool:
        return all(
            type(a) is torch.Tensor
            and type(b) is torch.Tensor
            and a.device.type == "cpu"
            and b.device.type == "cpu"
            for _factor, a, b in factor_tensors
        )

    @staticmethod
    def _pack_factors(
        data: torch.Tensor,
        factor_tensors: Sequence[
            tuple[ScaledLoRAFactor, torch.Tensor, torch.Tensor]
        ],
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
                "LoRA factor shape mismatch: B@A produces "
                f"{factor.produced_shape}, target shape is {target_shape}."
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
        raise ValueError(
            f"{adapter_name(adapter)} does not support LoRA merge. "
            "Use routed LoRA for this tensor type."
        )

    compute_dtype = adapter.compute_dtype(data)
    if not compute_dtype.is_floating_point:
        raise ValueError(
            "LoRA merge requires a floating-point compute dtype, "
            f"got {compute_dtype}."
        )
    return adapter


@dataclass(slots=True, frozen=True)
class _StagedLoRAFactor:
    """Adapter-materialized factor pair owned for one forward invocation."""

    a: nn.Parameter
    b: nn.Parameter
    strength: float


def _routed_residual(
    x: torch.Tensor,
    factors: Sequence[_StagedLoRAFactor],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Routed contribution ``Σ strength_i · (x @ A_i.T) @ B_i.T``.

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
                "Routed LoRA expected the Linear input as either the first "
                "positional argument or the 'input' keyword"
            ) from exc
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            "Routed LoRA requires the Linear input to be a torch.Tensor; "
            f"got {type(x).__name__}"
        )
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
    """Build each validated factor pair without a module hierarchy."""
    a_tensors, b_tensors = _split_factor_tensors(state_dict)
    factors: dict[str, LoRAFactor] = {}
    for base, a_source in a_tensors.items():
        b_source = b_tensors[base]
        a_tensor = (
            a_source
            if dtype is None or a_source.dtype is dtype
            else a_source.to(dtype=dtype)
        )
        b_tensor = (
            b_source
            if dtype is None or b_source.dtype is dtype
            else b_source.to(dtype=dtype)
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
    a_tensors, b_tensors = _split_factor_tensors(state_dict)
    incompatible = [
        f"{base}.lora_{side}.weight"
        for side, tensors in (("A", a_tensors), ("B", b_tensors))
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
    a 2-D floating-point matrix with a matching inner rank.
    """
    a_tensors, b_tensors = _split_factor_tensors(state_dict)

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

    for base_key, a in a_tensors.items():
        _validate_factor_pair(f"{base_key}.weight", a, b_tensors[base_key])


def _split_factor_tensors(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    a_tensors: dict[str, torch.Tensor] = {}
    b_tensors: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.endswith(".lora_A.weight"):
            a_tensors[key[: -len(".lora_A.weight")]] = tensor
        elif key.endswith(".lora_B.weight"):
            b_tensors[key[: -len(".lora_B.weight")]] = tensor
    return a_tensors, b_tensors


def _validate_factor_pair(
    target_key: str,
    a: torch.Tensor,
    b: torch.Tensor,
) -> None:
    if not a.is_floating_point() or not b.is_floating_point():
        raise ValueError(
            f"LoRA factors for {target_key!r}: must be floating-point; "
            f"got A.dtype={a.dtype}, B.dtype={b.dtype}."
        )
    if a.dim() != 2 or b.dim() != 2 or a.shape[0] != b.shape[1]:
        raise ValueError(
            f"LoRA factor shape mismatch for {target_key!r}: "
            f"A.shape={tuple(a.shape)}, B.shape={tuple(b.shape)}. "
            f"Expected A=(rank, in_dim), B=(out_dim, rank) with "
            f"A.shape[0] == B.shape[1]."
        )
