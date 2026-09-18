"""Standard :class:`ResourceSpec` implementations.

The cache itself remains resource-agnostic. These frozen dataclasses adapt
model, adapter, and ordinary-object factories to the structural resource-spec
protocol.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from .adapter import Adapter
from .block_compile import BlockCompileConfig
from .block_mode import BlockMode
from .model_offloader import ModelOffloader
from .protocols import ResourceStore


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelSpec[M: nn.Module]:
    """Model resource built from one user model factory.

    ``factory`` runs once to construct the cached :class:`ModelOffloader`.
    Every lease reuses that same model runtime sequentially; overlapping uses
    are rejected by the offloader. ``block_compile`` is an opt-in construction
    policy for every group named by ``block_paths`` or
    ``transient_block_paths``. ``block_mode`` selects resident, whole-block
    streaming, compiled rolling, or automatic rolling-with-streaming-fallback
    execution for every block group. Transient block groups release their CUDA
    working sets after their final blocks.
    ``transient_paths`` gives named modules independent CUDA working sets
    scoped to their forwards. The factory transfers compatible pageable CPU
    storage to the cached runtime, preserving checkpoint-backed mappings.
    """

    key: str
    factory: Callable[[], M]
    estimated_cache_bytes: int = 0
    block_paths: tuple[str, ...] = ()
    transient_block_paths: tuple[str, ...] = ()
    include_block_trainables: bool = False
    block_mode: BlockMode = "streaming"
    block_compile: BlockCompileConfig | None = None
    transient_paths: tuple[str, ...] = ()

    def build_store(self) -> ModelOffloader:
        """Build, capture, and bind the cached model runtime."""
        return ModelOffloader.from_module(
            self.factory(),
            block_paths=self.block_paths,
            transient_block_paths=self.transient_block_paths,
            include_block_trainables=self.include_block_trainables,
            block_mode=self.block_mode,
            block_compile=self.block_compile,
            transient_paths=self.transient_paths,
        )

    def value(self, store: ResourceStore) -> ModelOffloader:
        """Return the leased model runtime."""
        return cast(ModelOffloader, store)


@dataclass(frozen=True, kw_only=True, slots=True)
class AdapterSpec:
    """Adapter resource built from a state-dict factory.

    ``dtype`` is forwarded to :meth:`Adapter.from_state_dict`; matching the
    model's compute dtype reduces routed per-forward transfer volume.
    Compatible complete pageable CPU allocations and non-empty views into
    non-resizable storage transfer to the resource, so the caller must not
    mutate factory values after construction. The factory's reserved
    LoRA-suffixed entries form factor pairs; every other entry is an exact
    parameter-name physical value used to populate a frozen floating-point
    meta target.
    ``allow_partial_targets`` opts the built resource into applying only the
    intersection of its targets and a model's parameters.
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

    Every lease yields the same object instance. The default zero-byte charge
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


__all__ = ["AdapterSpec", "ModelSpec", "ObjectSpec"]
