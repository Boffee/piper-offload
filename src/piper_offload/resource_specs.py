"""Standard :class:`ResourceSpec` implementations.

The cache itself remains resource-agnostic. These frozen dataclasses adapt
model, adapter, and ordinary-object factories to the structural resource-spec
protocol.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn

from .adapter import Adapter
from .model_offloader import ModelOffloader
from .model_store import ModelStore
from .offload_config import OffloadConfig, validate_offload_config
from .protocols import ResourceStore


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelSpec[M: nn.Module]:
    """Model resource built from one user model factory.

    ``factory`` runs once to construct the cached :class:`ModelOffloader`.
    Every lease reuses that same model runtime sequentially; overlapping uses
    are rejected by the offloader. ``config`` selects dynamic offload paths,
    ``vram_budget_bytes``, and optional compilation for the bound runtime. The factory
    transfers compatible pageable CPU storage to the cached runtime,
    preserving checkpoint-backed mappings.
    """

    key: str
    factory: Callable[[], M]
    estimated_cache_bytes: int = 0
    config: OffloadConfig = field(default_factory=OffloadConfig)

    def __post_init__(self) -> None:
        validate_offload_config(self.config)

    def build_store(self) -> ModelOffloader:
        """Build, capture, and bind the cached model runtime."""
        return ModelOffloader.from_module(self.factory(), self.config)

    def value(self, store: ResourceStore) -> ModelOffloader:
        """Return the leased model runtime."""
        return cast(ModelOffloader, store)


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelStoreSpec:
    """Reusable host-backing store resource built from a frozen model factory.

    ``factory`` runs once to build a flat :class:`ModelStore` whose host bytes
    can be bound across multiple target models or offload configurations via
    :meth:`ModelStore.bind`.
    """

    key: str
    factory: Callable[[], nn.Module]
    estimated_cache_bytes: int = 0

    def build_store(self) -> ModelStore:
        """Build and capture this reusable flat model store."""
        return ModelStore.from_module(self.factory())

    def value(self, store: ResourceStore) -> ModelStore:
        """Return the leased model store."""
        return cast(ModelStore, store)


@dataclass(frozen=True, kw_only=True, slots=True)
class AdapterSpec:
    """Adapter resource built from a state-dict factory.

    ``dtype`` is forwarded to :meth:`Adapter.from_state_dict`; matching the
    model's compute dtype avoids conversion work during merge staging.
    Compatible complete pageable CPU allocations and non-empty views into
    non-resizable storage transfer to the resource, so the caller must not
    mutate factory values after construction. The factory's reserved
    LoRA-suffixed entries form factor pairs; every other entry is an exact
    parameter-name physical value used to populate a frozen floating-point
    meta target.
    ``allow_partial_targets`` ignores absent target names and, during merge,
    skips deltas to meta parameters without an active complete value.
    See :class:`Adapter` for composition with strict adapters.
    ``scale_parameter_values`` opts complete values into adapter-strength
    scaling; by default active parameter values remain unchanged.
    """

    key: str
    factory: Callable[[], Mapping[str, torch.Tensor]]
    estimated_cache_bytes: int = 0
    dtype: torch.dtype | None = None
    allow_partial_targets: bool = False
    scale_parameter_values: bool = False

    def build_store(self) -> Adapter:
        """Build and capture this reusable adapter resource."""
        return Adapter.from_state_dict(
            self.factory(),
            dtype=self.dtype,
            allow_partial_targets=self.allow_partial_targets,
            scale_parameter_values=self.scale_parameter_values,
        )

    def value(self, store: ResourceStore) -> Adapter:
        """Return the leased adapter resource."""
        return cast(Adapter, store)


@dataclass(frozen=True, slots=True)
class _ObjectStore[T]:
    """Accounting wrapper for a plain Python object."""

    value: T
    cache_bytes: int


@dataclass(frozen=True, kw_only=True, slots=True)
class ObjectSpec[T]:
    """Resource spec for a tokenizer, processor, config, or other object.

    Every lease yields the same object instance. The default zero-byte reservation
    keeps ordinary heap objects outside the host-memory budget.
    """

    key: str
    factory: Callable[[], T]
    estimated_cache_bytes: int = 0

    def build_store(self) -> ResourceStore:
        """Build the accounting wrapper around the cached object."""
        return _ObjectStore(self.factory(), self.estimated_cache_bytes)

    def value(self, store: ResourceStore) -> T:
        """Return the object held by its accounting store."""
        return cast(_ObjectStore[T], store).value


__all__ = ["AdapterSpec", "ModelSpec", "ModelStoreSpec", "ObjectSpec"]
