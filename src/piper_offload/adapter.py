"""Reusable host-backed model adapter resources.

An :class:`Adapter` captures a canonical state dict containing low-rank LoRA
factors, full-rank parameter deltas, exact-name values for meta parameters,
or a combination. It owns
storage and target metadata only; merge and routed execution live in their
respective transform modules.
"""

from collections.abc import Iterable, Mapping, Sequence, Set
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
type AdapterTarget = ParameterDelta | ParameterValue

type _InputPart = Literal["a", "b", "alpha", "dense", "value"]
type _TargetInputs = dict[_InputPart, torch.Tensor]

# Reserved input suffix, target suffix, and role within that parameter.
_INPUT_SUFFIXES: tuple[tuple[str, str, _InputPart], ...] = (
    (".lora_A.weight", ".weight", "a"),
    (".lora_B.weight", ".weight", "b"),
    (".alpha", ".weight", "alpha"),
    (".delta.weight", ".weight", "dense"),
    (".delta.bias", ".bias", "dense"),
)


@dataclass(slots=True)
class AdapterTargetUpdates:
    """Package-internal active contributions for one parameter name."""

    deltas: list[ScaledParameterDelta] = field(default_factory=list)
    value: ScaledParameterValue | None = None
    requires_base: bool = False

    def add(
        self,
        target: AdapterTarget,
        strength: float,
        *,
        target_key: str,
        allow_partial_targets: bool,
    ) -> None:
        """Bind one contribution, allowing only one authoritative value."""
        if isinstance(target, ParameterDelta):
            self.deltas.append(target.scaled(strength))
            self.requires_base |= not allow_partial_targets
            return

        if self.value is not None:
            raise ValueError(f"Adapter target {target_key!r} has multiple active parameter values.")
        self.value = target.scaled(strength)

    def should_skip_merge(self, param: torch.nn.Parameter) -> bool:
        """Skip only an unpopulated target whose deltas all allow partial targets.

        Call after collecting every active contribution. A supplied value
        makes the target available regardless of adapter ordering. Both merge
        paths validate target aliases before this check. A skipped target must
        create no device override or permanent merge operation and requires no
        staging or materialization; any strict delta still requires a base.
        """
        return param.is_meta and self.value is None and not self.requires_base

    @property
    def factors(self) -> list[ScaledLoRAFactor]:
        """Low-rank contributions extracted for routed execution."""
        return [
            bound.delta.lora.scaled(bound.strength) for bound in self.deltas if bound.delta.lora is not None
        ]

    @property
    def has_dense(self) -> bool:
        """Whether any active delta contains a full-rank contribution."""
        return any(bound.delta.dense is not None for bound in self.deltas)


class Adapter:
    """Reusable immutable host-backed model adapter resource.

    Build once from a flat canonical ``state_dict``. Exact LoRA and delta
    suffixes identify low-rank factors and full-rank additive updates; every
    other key is the complete value for an exact-name meta parameter.
    Inputs are validated and captured in pageable CPU storage. Compatible
    complete CPU allocations and non-empty views into non-resizable storage
    transfer to the resource; device tensors and incompatible CPU layouts are
    copied. The optional ``dtype`` casts dense inputs; a structured parameter
    value must already have that logical compute dtype because casting would
    discard its encoded representation. The resource retains the resulting
    tensors but not the raw input mapping.

    Satisfies :class:`~piper_offload.protocols.ResourceStore`, so it can be
    registered in :class:`~piper_offload.ResourceCache` for budget tracking and
    policy-driven eviction. Merge and routed consumers read the same immutable
    backing and may overlap. Parameter values are merge-only and populate
    only frozen floating-point meta parameters.

    Strength is extrinsic and supplied when the adapter is activated or
    permanently merged. It always scales parameter deltas. Complete parameter
    values remain unchanged by default and can explicitly opt into strength
    scaling. For one parameter, an adapter supplies either a value or a delta.
    Merge application can combine one value with deltas from other adapters,
    validating their arithmetic against the supplied representation.

    ``state_dict`` keys must already use model parameter paths. Any key
    remapping and removal of non-adapter metadata are the caller's
    responsibility.

    ``allow_partial_targets`` is an explicit application policy for adapters
    that intentionally span separately loaded or optional model components.
    When enabled, absent target names are ignored. Merge application also
    skips deltas to meta parameters without an active complete value, leaving
    them meta. Any strict delta on the same target still requires a source.
    Physical targets and active supplied values retain normal validation.
    Routed execution retains its existing target requirements.
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
        targets, and a matching ``.alpha`` scalar gives that target its
        intrinsic ``alpha / rank`` magnitude. A checkpoint that states its
        alpha out of band, such as in safetensors metadata, states it per
        factor here, the same way it states the factors themselves. Carrying
        the magnitude means a caller need not fold it into the factors: folding
        is host arithmetic that replaces a mapped checkpoint tensor with an
        anonymous one, which cannot then be pinned through an owned copy.

        Keys ending in ``.delta.weight`` or ``.delta.bias`` are
        full-rank additive updates targeting the corresponding model weight or
        bias. Every other key is an exact model parameter name whose tensor is
        the complete dense or supported prequantized value for a meta
        parameter. ``dtype`` casts dense inputs; structured values must already
        use it as their logical compute dtype.
        ``scale_parameter_values=True`` multiplies those complete values by
        an active adapter's strength; by default they remain unchanged.
        A zero-strength adapter remains inactive. Input keys for every target
        are checked for missing partners and conflicts before tensor capture.
        Compatible complete pageable CPU allocations and non-empty views into
        non-resizable storage transfer to the resulting resource. The caller
        must not mutate input tensors after construction. Device tensors and
        incompatible CPU layouts are copied.
        """
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError(f"Adapter dtype must be floating-point, got {dtype}.")
        inputs = _group_target_inputs(state_dict)
        targets = {
            name: _capture_target(parts, dtype=dtype, scale_parameter_values=scale_parameter_values)
            for name, parts in inputs.items()
        }
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
        """Whether to skip absent names and additive merges into unpopulated meta targets."""
        return self._allow_partial_targets


def active_adapter_pairs[T](pairs: Iterable[tuple[T, float]]) -> list[tuple[T, float]]:
    """Keep nonzero strengths in request order, including repeated adapters."""
    active: list[tuple[T, float]] = []
    for adapter, requested_strength in pairs:
        strength = float(requested_strength)
        if strength != 0.0:
            active.append((adapter, strength))
    return active


def resolve_adapter_strengths[T](adapters: Sequence[T], strengths: Sequence[float] | None) -> list[tuple[T, float]]:
    """Pair adapters or cache specs with their requested strengths before use."""
    if strengths is None:
        return [(adapter, 1.0) for adapter in adapters]
    return active_adapter_pairs(zip(adapters, strengths, strict=True))


def group_adapter_updates(
    adapters: Sequence[tuple[Adapter, float]],
    param_names: Set[str],
) -> dict[str, AdapterTargetUpdates]:
    """Collect contributions by exact name, omitting absent partial targets."""
    updates: dict[str, AdapterTargetUpdates] = {}
    missing: set[str] = set()
    for adapter, strength in adapters:
        for name, target in adapter.targets.items():
            if name not in param_names:
                if not adapter.allow_partial_targets:
                    missing.add(name)
                continue
            contributions = updates.setdefault(name, AdapterTargetUpdates())
            contributions.add(target, strength, target_key=name, allow_partial_targets=adapter.allow_partial_targets)
    if missing:
        raise ValueError(
            f"Adapter targets are not parameters in the model: {sorted(missing)}. "
            f"Target keys must match exactly. Sample model parameter keys: {sorted(param_names)[:3]}."
        )
    return updates


def _target_and_part(key: str) -> tuple[str, _InputPart]:
    for suffix, target_suffix, part in _INPUT_SUFFIXES:
        if key.endswith(suffix):
            base = key.removesuffix(suffix)
            if not base:
                raise ValueError(f"Adapter target names must be non-empty: {key!r}.")
            return base + target_suffix, part
    if not key:
        raise ValueError("Parameter value target names must be non-empty")
    return key, "value"


def _group_target_inputs(state_dict: Mapping[str, torch.Tensor]) -> dict[str, _TargetInputs]:
    groups: dict[str, _TargetInputs] = {}
    # Preserve Adapter.targets order: A-factor order, then remaining deltas, then values.
    order: dict[_InputPart, list[str]] = {"a": [], "dense": [], "value": []}
    for key, tensor in state_dict.items():
        name, part = _target_and_part(key)
        groups.setdefault(name, {})[part] = tensor
        if part in order:
            order[part].append(name)

    _validate_target_inputs(groups)
    return {name: groups[name] for names in order.values() for name in names}


def _validate_target_inputs(groups: dict[str, _TargetInputs]) -> None:
    """Check every target's input combination before capturing the first tensor."""
    a_only: list[str] = []
    b_only: list[str] = []
    orphaned: list[str] = []
    for name, parts in groups.items():
        if "a" in parts and "b" not in parts:
            a_only.append(name.removesuffix(".weight"))
        elif "b" in parts and "a" not in parts:
            b_only.append(name.removesuffix(".weight"))
        elif "alpha" in parts and "a" not in parts:
            orphaned.append(name.removesuffix(".weight"))
    if a_only or b_only:
        raise ValueError(
            f"Unpaired LoRA factors: A-only={sorted(a_only)}, B-only={sorted(b_only)}. "
            "Each target needs both .lora_A.weight and .lora_B.weight."
        )
    if orphaned:
        raise ValueError(
            f"LoRA alpha without a factor pair: {sorted(orphaned)}. Each .alpha needs "
            "its .lora_A.weight and .lora_B.weight."
        )

    for name, parts in groups.items():
        if "value" in parts and len(parts) > 1:
            raise ValueError(f"Adapter target {name!r} cannot contain both a parameter delta and a value.")
        alpha = parts.get("alpha")
        if alpha is not None and alpha.numel() != 1:
            base = name.removesuffix(".weight")
            raise ValueError(f"LoRA alpha for {base!r} must be a scalar; got shape {tuple(alpha.shape)}.")


def _capture_target(
    parts: _TargetInputs,
    *,
    dtype: torch.dtype | None,
    scale_parameter_values: bool,
) -> AdapterTarget:
    if "value" in parts:
        return ParameterValue.from_tensor(parts["value"], dtype=dtype, scale_with_strength=scale_parameter_values)
    alpha = parts.get("alpha")
    return ParameterDelta.from_tensors(
        a=parts.get("a"),
        b=parts.get("b"),
        dense=parts.get("dense"),
        dtype=dtype,
        lora_alpha=None if alpha is None else float(alpha.item()),
    )
