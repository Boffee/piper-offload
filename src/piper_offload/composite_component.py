"""Composition of resident, transient, and block-streamed components."""

import contextlib
from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import nn

from .block_compile import BlockCompileConfig
from .host_backing import HostBacking
from .module_names import buffer_names, parameter_names
from .pinned_component import PinnedComponent, PinnedComponentStore
from .pinned_module import PostCopyHook
from .streamed_component import StreamedComponent, StreamedComponentStore


class CompositeComponent:
    """Resident, transient-path, and streamed-block components."""

    def __init__(
        self,
        *,
        resident: PinnedComponent | None,
        streamed: Sequence[StreamedComponent],
        transient: Sequence[tuple[str, PinnedComponent]] = (),
        transient_streamed: Sequence[StreamedComponent] = (),
    ) -> None:
        self._resident = resident
        self._transient = tuple(transient)
        self._streamed = tuple(streamed)
        self._transient_streamed = tuple(transient_streamed)
        self._teardown_stack: contextlib.ExitStack | None = None

    @property
    def resident(self) -> PinnedComponent | None:
        return self._resident

    @property
    def streamed(self) -> tuple[StreamedComponent, ...]:
        return self._streamed

    @property
    def transient(self) -> tuple[tuple[str, PinnedComponent], ...]:
        return self._transient

    @property
    def transient_streamed(self) -> tuple[StreamedComponent, ...]:
        return self._transient_streamed

    def _components(self) -> Iterator[PinnedComponent | StreamedComponent]:
        if self._resident is not None:
            yield self._resident
        for _path, component in self._transient:
            yield component
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
    def optimizer_step(self) -> Generator[None]:
        with contextlib.ExitStack() as stack:
            for component in self._components():
                stack.enter_context(component.optimizer_step())
            yield


@dataclass(frozen=True, slots=True)
class CompositeComponentStore:
    """Reusable stores for resident, transient, and streamed model state."""

    resident_store: PinnedComponentStore | None
    streamed_stores: tuple[StreamedComponentStore, ...]
    transient_streamed_stores: tuple[StreamedComponentStore, ...] = ()
    transient_stores: tuple[tuple[str, PinnedComponentStore], ...] = ()

    def __post_init__(self) -> None:
        if (
            self.resident_store is None
            and not self.transient_stores
            and not self.streamed_stores
            and not self.transient_streamed_stores
        ):
            raise ValueError(
                "Offloading requires at least one parameter, registered "
                "buffer, or streamed block to manage."
            )

    def _stores(
        self,
    ) -> Iterator[PinnedComponentStore | StreamedComponentStore]:
        if self.resident_store is not None:
            yield self.resident_store
        for _path, store in self.transient_stores:
            yield store
        yield from self.streamed_stores
        yield from self.transient_streamed_stores

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        block_paths: Sequence[str] = (),
        transient_block_paths: Sequence[str] = (),
        transient_paths: Sequence[str] = (),
        stream_trainable_weights: bool = False,
        host_backing: HostBacking = "pinned",
    ) -> Self:
        persistent_paths = tuple(block_paths)
        transient_streamed_paths = tuple(transient_block_paths)
        overlap = set(persistent_paths) & set(transient_streamed_paths)
        if overlap:
            raise ValueError(
                "block_paths and transient_block_paths must be disjoint; "
                f"both contain {sorted(overlap)!r}."
            )
        for path in transient_streamed_paths:
            blocks = model.get_submodule(path)
            if isinstance(blocks, nn.ModuleList) and len(
                {id(block) for block in blocks}
            ) != len(blocks):
                raise ValueError(
                    "transient_block_paths does not support aliased block "
                    f"modules; {path!r} contains repeated module objects."
                )

        def make_streamed_store(blocks_path: str) -> StreamedComponentStore:
            return StreamedComponentStore.from_module(
                model,
                blocks_path=blocks_path,
                stream_trainable_weights=stream_trainable_weights,
                host_backing=host_backing,
            )

        streamed_stores = tuple(
            make_streamed_store(blocks_path)
            for blocks_path in persistent_paths
        )
        transient_streamed_stores = tuple(
            make_streamed_store(blocks_path)
            for blocks_path in transient_streamed_paths
        )
        all_streamed_stores = (*streamed_stores, *transient_streamed_stores)
        streamed_params = {
            name for store in all_streamed_stores for name in store.param_names
        }
        streamed_buffers = {
            name for store in all_streamed_stores for name in store.buffer_names
        }
        resident_params = parameter_names(model) - streamed_params
        resident_buffers = buffer_names(model) - streamed_buffers
        transient_stores: list[tuple[str, PinnedComponentStore]] = []
        for path in transient_paths:
            module = model.get_submodule(path)
            prefix = f"{path}." if path else ""
            selected_params = {
                f"{prefix}{name}" for name in parameter_names(module)
            } & resident_params
            selected_buffers = {
                f"{prefix}{name}" for name in buffer_names(module)
            } & resident_buffers
            if not selected_params and not selected_buffers:
                continue
            transient_stores.append(
                (
                    path,
                    PinnedComponentStore.from_module(
                        model,
                        include_param_names=selected_params,
                        include_buffer_names=selected_buffers,
                        host_backing=host_backing,
                    ),
                )
            )
            resident_params -= selected_params
            resident_buffers -= selected_buffers
        resident_store = (
            PinnedComponentStore.from_module(
                model,
                include_param_names=resident_params,
                include_buffer_names=resident_buffers,
                host_backing=host_backing,
            )
            if resident_params or resident_buffers
            else None
        )
        return cls(
            resident_store=resident_store,
            streamed_stores=streamed_stores,
            transient_streamed_stores=transient_streamed_stores,
            transient_stores=tuple(transient_stores),
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
        transient = tuple(
            (path, store.bind(model))
            for path, store in self.transient_stores
        )
        streamed = tuple(
            store.bind(
                model,
                block_compile=block_compile,
                wraparound=True,
            )
            for store in self.streamed_stores
        )
        transient_streamed = tuple(
            store.bind(
                model,
                block_compile=block_compile,
                wraparound=False,
            )
            for store in self.transient_streamed_stores
        )
        return CompositeComponent(
            resident=resident,
            streamed=(*streamed, *transient_streamed),
            transient=transient,
            transient_streamed=transient_streamed,
        )


__all__ = ["CompositeComponent", "CompositeComponentStore"]
