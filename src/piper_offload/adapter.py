"""Reusable host-backed model adapter resources.

An :class:`Adapter` captures a canonical state dict containing low-rank LoRA
factors, full-rank parameter deltas, exact-name values for meta parameters,
or a combination. It owns
storage and target metadata only; merge and routed execution live in their
respective transform modules.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Self

import torch

from .lora import ScaledLoRAFactor
from .parameter_delta import ParameterDelta, ScaledParameterDelta
from .parameter_value import ParameterValue, ScaledParameterValue

__all__ = [
    "Adapter",
    "AdapterMode",
    "AdapterTarget",
]

type AdapterMode = Literal["merge", "routed"]

_LORA_A_SUFFIX = ".lora_A.weight"
_LORA_B_SUFFIX = ".lora_B.weight"
_DELTA_WEIGHT_SUFFIX = ".delta.weight"
_DELTA_BIAS_SUFFIX = ".delta.bias"

type AdapterTarget = ParameterDelta | ParameterValue


@dataclass(slots=True)
class AdapterTargetUpdates:
    """Package-internal active contributions for one parameter name."""

    deltas: list[ScaledParameterDelta] = field(default_factory=list)
    value: ScaledParameterValue | None = None

    def add(
        self,
        target: AdapterTarget,
        strength: float,
        *,
        target_key: str,
    ) -> None:
        """Bind one contribution, enforcing exclusive value ownership."""
        if isinstance(target, ParameterDelta):
            if self.value is not None:
                raise ValueError(f"Adapter target {target_key!r} cannot combine parameter deltas and a value.")
            self.deltas.append(target.scaled(strength))
            return

        if self.deltas:
            raise ValueError(f"Adapter target {target_key!r} cannot combine parameter deltas and a value.")
        if self.value is not None:
            raise ValueError(f"Adapter target {target_key!r} has multiple active parameter values.")
        self.value = target.scaled(strength)

    @property
    def factors(self) -> list[ScaledLoRAFactor]:
        """Low-rank contributions extracted for routed execution."""
        return [
            ScaledLoRAFactor(bound.delta.lora, bound.strength) for bound in self.deltas if bound.delta.lora is not None
        ]

    @property
    def has_dense(self) -> bool:
        """Whether any active delta contains a full-rank contribution."""
        return any(bound.delta.dense is not None for bound in self.deltas)


@dataclass(slots=True)
class _AdapterSources:
    """One parsed view of canonical adapter tensors."""

    a: dict[str, torch.Tensor]
    b: dict[str, torch.Tensor]
    deltas: dict[str, torch.Tensor]
    values: dict[str, torch.Tensor]


class Adapter:
    """Reusable immutable host-backed model adapter resource.

    Build once from a flat canonical ``state_dict``. Exact LoRA and delta
    suffixes identify low-rank factors and full-rank additive updates; every
    other key is the complete value for an exact-name meta parameter.
    Inputs are validated and copied to pageable CPU storage. The optional
    ``dtype`` casts dense inputs; a structured parameter value must already
    have that logical compute dtype because casting would discard its encoded
    representation. The resource retains the resulting tensors but not the
    raw input mapping.

    Satisfies :class:`~piper_offload.protocols.ResourceStore`, so it can be
    registered in :class:`~piper_offload.ResourceCache` for budget tracking and
    policy-driven eviction. Merge and routed consumers read the same immutable
    backing and may overlap. Parameter values are merge-only and populate
    only frozen floating-point meta parameters.

    Strength is extrinsic and supplied when the adapter is activated or
    permanently merged. It always scales parameter deltas. Complete parameter
    values remain unchanged by default and can explicitly opt into strength
    scaling.

    ``state_dict`` keys must already use model parameter paths. Any key
    remapping and removal of non-adapter metadata are the caller's
    responsibility.

    ``allow_partial_targets`` is an explicit application policy for adapters
    that intentionally span multiple separately loaded model components.
    When enabled, targets absent from a particular model are ignored. Present
    targets retain the same shape and capability validation as strict mode.
    """

    def __init__(
        self,
        targets: Mapping[str, AdapterTarget],
        *,
        allow_partial_targets: bool = False,
    ) -> None:
        captured_targets = dict(targets)
        if not captured_targets:
            raise ValueError("Adapter resource contains no targets")
        for target_key, target in captured_targets.items():
            if not isinstance(target_key, str) or not target_key:
                raise ValueError("Adapter target names must be non-empty strings")
            if not isinstance(target, (ParameterDelta, ParameterValue)):
                raise ValueError(
                    "Adapter targets must be ParameterDelta or ParameterValue instances; "
                    f"target {target_key!r} has {type(target).__name__}."
                )

        self._targets = MappingProxyType(captured_targets)
        self._cache_bytes = sum(target.cache_bytes for target in self._targets.values())
        self._allow_partial_targets = allow_partial_targets

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        dtype: torch.dtype | None = None,
        allow_partial_targets: bool = False,
        scale_parameter_values: bool = False,
    ) -> Self:
        """Validate and capture factor and/or parameter-value tensors.

        Keys ending in ``.lora_A.weight`` and ``.lora_B.weight`` form factor
        targets. Keys ending in ``.delta.weight`` or ``.delta.bias`` are
        full-rank additive updates targeting the corresponding model weight or
        bias. Every other key is an exact model parameter name whose tensor is
        the complete dense or supported prequantized value for a meta
        parameter. ``dtype`` casts dense inputs; structured values must already
        use it as their logical compute dtype.
        ``scale_parameter_values=True`` multiplies those complete values by
        an active adapter's strength; by default they remain unchanged.
        A zero-strength adapter remains inactive.
        Captured values own independent pageable CPU storage.
        """
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError(f"Adapter dtype must be floating-point, got {dtype}.")
        sources = _parse_adapter_state_dict(state_dict)
        targets = _build_adapter_targets(
            sources,
            dtype=dtype,
            scale_parameter_values=scale_parameter_values,
        )
        return cls(targets, allow_partial_targets=allow_partial_targets)

    @property
    def targets(self) -> Mapping[str, AdapterTarget]:
        """Immutable exact parameter-name to update mapping."""
        return self._targets

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes

    @property
    def allow_partial_targets(self) -> bool:
        """Whether application may ignore targets absent from a model."""
        return self._allow_partial_targets


def _parse_adapter_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> _AdapterSources:
    """Parse and validate canonical adapter sources in one pass."""
    sources = _AdapterSources({}, {}, {}, {})
    for key, tensor in state_dict.items():
        if key.endswith(_LORA_A_SUFFIX):
            base = key[: -len(_LORA_A_SUFFIX)]
            _validate_lora_base(base)
            sources.a[base] = tensor
        elif key.endswith(_LORA_B_SUFFIX):
            base = key[: -len(_LORA_B_SUFFIX)]
            _validate_lora_base(base)
            sources.b[base] = tensor
        elif key.endswith(_DELTA_WEIGHT_SUFFIX):
            base = key[: -len(_DELTA_WEIGHT_SUFFIX)]
            _validate_delta_base(base)
            sources.deltas[f"{base}.weight"] = tensor
        elif key.endswith(_DELTA_BIAS_SUFFIX):
            base = key[: -len(_DELTA_BIAS_SUFFIX)]
            _validate_delta_base(base)
            sources.deltas[f"{base}.bias"] = tensor
        else:
            if not key:
                raise ValueError("Parameter value target names must be non-empty")
            sources.values[key] = tensor

    _validate_factor_pairing(sources)
    return sources


def _validate_factor_pairing(sources: _AdapterSources) -> None:
    """Validate factor pairing in an already parsed state dict."""
    a_only = set(sources.a) - set(sources.b)
    b_only = set(sources.b) - set(sources.a)
    if a_only or b_only:
        raise ValueError(
            f"Unpaired LoRA factors: A-only={sorted(a_only)}, "
            f"B-only={sorted(b_only)}. Each target needs both "
            f".lora_A.weight and .lora_B.weight."
        )


def _validate_lora_base(base: str) -> None:
    if not base:
        raise ValueError("LoRA target names must be non-empty")


def _validate_delta_base(base: str) -> None:
    if not base:
        raise ValueError("Parameter delta target names must be non-empty")


def _build_adapter_targets(
    sources: _AdapterSources,
    *,
    dtype: torch.dtype | None = None,
    scale_parameter_values: bool = False,
) -> dict[str, AdapterTarget]:
    """Capture parsed sources into one target value per parameter name."""
    targets: dict[str, AdapterTarget] = {}
    factor_sources = {f"{base}.weight": base for base in sources.a}
    delta_targets = dict.fromkeys((*factor_sources, *sources.deltas))
    for target_key in delta_targets:
        base = factor_sources.get(target_key)
        targets[target_key] = ParameterDelta.from_tensors(
            a=None if base is None else sources.a[base],
            b=None if base is None else sources.b[base],
            dense=sources.deltas.get(target_key),
            dtype=dtype,
        )

    for target_key, source in sources.values.items():
        if target_key in targets:
            raise ValueError(f"Adapter target {target_key!r} cannot contain both a parameter delta and a value.")
        targets[target_key] = ParameterValue.from_tensor(
            source,
            dtype=dtype,
            scale_with_strength=scale_parameter_values,
        )
    return targets
