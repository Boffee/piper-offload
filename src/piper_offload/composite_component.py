"""Composition of resident and block-streamed offload components."""

import contextlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import nn

from .block_compile import BlockCompileConfig
from .host_backing import HostBacking, validate_host_backing
from .module_names import buffer_names, parameter_names
from .pinned_component import PinnedComponent, PinnedComponentStore
from .pinned_module import PostCopyHook
from .streamed_component import StreamedComponent, StreamedComponentStore


class CompositeComponent:
    """One resident component plus zero or more streamed block groups."""

    def __init__(
        self,
        *,
        resident: PinnedComponent | None,
        streamed: Sequence[StreamedComponent],
    ) -> None:
        self._resident = resident
        self._streamed = tuple(streamed)
        self._teardown_stack: contextlib.ExitStack | None = None

    @property
    def resident(self) -> PinnedComponent | None:
        return self._resident

    @property
    def streamed(self) -> tuple[StreamedComponent, ...]:
        return self._streamed

    def _components(self) -> Iterator[PinnedComponent | StreamedComponent]:
        if self._resident is not None:
            yield self._resident
        yield from self._streamed

    @property
    def param_names(self) -> frozenset[str]:
        return frozenset(
            name
            for component in self._components()
            for name in component.param_names
        )

    @property
    def buffer_names(self) -> frozenset[str]:
        return frozenset(
            name
            for component in self._components()
            for name in component.buffer_names
        )

    def component_for_param_name(
        self,
        param_name: str,
    ) -> PinnedComponent | StreamedComponent:
        for component in self._components():
            if param_name in component.param_names:
                return component
        raise KeyError(
            f"param name {param_name!r} is not managed by this composite"
        )

    def register_post_copy_hook(
        self,
        name: str,
        hook: PostCopyHook,
    ) -> Callable[[], None]:
        return self.component_for_param_name(name).register_post_copy_hook(
            name,
            hook,
        )

    def activate(
        self,
        device: torch.device,
        *,
        compile_blocks: bool = True,
    ) -> None:
        if not isinstance(compile_blocks, bool):
            raise TypeError(
                "compile_blocks must be bool; "
                f"got {type(compile_blocks).__name__}."
            )
        if self._teardown_stack is not None:
            raise RuntimeError(
                "CompositeComponent.activate() called while already active; "
                "deactivate() first."
            )

        with contextlib.ExitStack() as stack:
            for component in self._components():
                stack.callback(component.deactivate)
                component.activate(device, compile_blocks=compile_blocks)
            self._teardown_stack = stack.pop_all()

    def deactivate(self) -> None:
        stack = self._teardown_stack
        self._teardown_stack = None
        if stack is not None:
            stack.close()

    @contextlib.contextmanager
    def optimizer_step(self) -> Iterator[None]:
        with contextlib.ExitStack() as stack:
            for component in self._components():
                stack.enter_context(component.optimizer_step())
            yield


@dataclass(frozen=True, slots=True)
class CompositeComponentStore:
    """Reusable stores for resident and block-streamed model state."""

    resident_store: PinnedComponentStore | None
    streamed_stores: tuple[StreamedComponentStore, ...]

    def __post_init__(self) -> None:
        if self.resident_store is None and not self.streamed_stores:
            raise ValueError(
                "Offloading requires at least one parameter, registered "
                "buffer, or streamed block to manage."
            )

    def _stores(
        self,
    ) -> Iterator[PinnedComponentStore | StreamedComponentStore]:
        if self.resident_store is not None:
            yield self.resident_store
        yield from self.streamed_stores

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        block_paths: Sequence[str] = (),
        stream_trainable_weights: bool = False,
        host_backing: HostBacking = "pinned",
    ) -> Self:
        backing = validate_host_backing(host_backing)
        streamed_stores = tuple(
            StreamedComponentStore.from_module(
                model,
                blocks_path=blocks_path,
                stream_trainable_weights=stream_trainable_weights,
                host_backing=backing,
            )
            for blocks_path in block_paths
        )
        streamed_params = {
            name for store in streamed_stores for name in store.param_names
        }
        streamed_buffers = {
            name for store in streamed_stores for name in store.buffer_names
        }
        resident_params = parameter_names(model) - streamed_params
        resident_buffers = buffer_names(model) - streamed_buffers
        resident_store = (
            PinnedComponentStore.from_module(
                model,
                include_param_names=resident_params,
                include_buffer_names=resident_buffers,
                host_backing=backing,
            )
            if resident_params or resident_buffers
            else None
        )
        return cls(
            resident_store=resident_store,
            streamed_stores=streamed_stores,
        )

    @property
    def cache_bytes(self) -> int:
        return sum(store.cache_bytes for store in self._stores())

    @property
    def has_trainables(self) -> bool:
        return any(store.has_trainables for store in self._stores())

    def bind(
        self,
        model: nn.Module,
        *,
        block_compile: BlockCompileConfig | None = None,
    ) -> CompositeComponent:
        resident = self.resident_store.bind(model) if self.resident_store else None
        streamed = tuple(
            store.bind(model, block_compile=block_compile)
            for store in self.streamed_stores
        )
        return CompositeComponent(
            resident=resident,
            streamed=streamed,
        )


__all__ = ["CompositeComponent", "CompositeComponentStore"]
