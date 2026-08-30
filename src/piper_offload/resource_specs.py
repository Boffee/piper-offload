"""Standard :class:`ResourceSpec` implementations.

The cache itself remains resource-agnostic. These frozen dataclasses adapt
model, LoRA, and ordinary-object factories to the structural resource-spec
protocol.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from .block_compile import BlockCompileConfig
from .host_backing import HostBacking
from .lora import LoRA
from .model_offloader import ModelOffloader
from .protocols import ResourceStore


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelSpec[M: nn.Module]:
    """Model resource built from one user model factory.

    ``factory`` runs once to construct the cached :class:`ModelOffloader`.
    Every lease reuses that same model runtime sequentially; overlapping uses
    are rejected by the offloader. ``block_compile`` is an opt-in construction
    policy for every streamed group named by ``block_paths``.
    ``transient_streaming`` scopes those groups' CUDA pools to model-forward
    execution. ``host_backing`` selects pinned copies (the default) or strict
    zero-copy adoption of existing CPU model backing.
    """

    key: str
    estimated_cache_bytes: int
    factory: Callable[[], M]
    block_paths: tuple[str, ...] = ()
    stream_trainable_weights: bool = False
    block_compile: BlockCompileConfig | None = None
    host_backing: HostBacking = "pinned"
    transient_streaming: bool = False

    def build_store(self) -> ModelOffloader:
        """Build, pin, and bind the cached model runtime."""
        return ModelOffloader.from_module(
            self.factory(),
            block_paths=self.block_paths,
            stream_trainable_weights=self.stream_trainable_weights,
            block_compile=self.block_compile,
            host_backing=self.host_backing,
            transient_streaming=self.transient_streaming,
        )

    def value(self, store: ResourceStore) -> ModelOffloader:
        """Return the leased model runtime."""
        return cast(ModelOffloader, store)


@dataclass(frozen=True, kw_only=True, slots=True)
class LoRASpec:
    """LoRA resource built from a state-dict factory.

    ``dtype`` and ``host_backing`` are forwarded to
    :meth:`LoRA.from_state_dict`; matching the model's compute dtype reduces
    routed per-forward transfer volume when using pinned backing. Adopted
    backing strictly retains compatible CPU factor tensors.
    """

    key: str
    estimated_cache_bytes: int
    factory: Callable[[], dict[str, torch.Tensor]]
    dtype: torch.dtype | None = None
    host_backing: HostBacking = "pinned"

    def build_store(self) -> LoRA:
        """Build and pin this reusable adapter resource."""
        return LoRA.from_state_dict(
            self.factory(),
            dtype=self.dtype,
            host_backing=self.host_backing,
        )

    def value(self, store: ResourceStore) -> LoRA:
        """Return the leased LoRA resource."""
        return cast(LoRA, store)


@dataclass(frozen=True, slots=True)
class _ObjectStore[T]:
    """Accounting wrapper for a plain Python object."""

    value: T
    cache_bytes: int


@dataclass(frozen=True, kw_only=True, slots=True)
class ObjectSpec[T]:
    """Resource spec for a tokenizer, processor, config, or other object.

    Every lease yields the same object instance. The default zero-byte charge
    keeps ordinary heap objects outside the pinned-host-memory budget.
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


__all__ = ["LoRASpec", "ModelSpec", "ObjectSpec"]
