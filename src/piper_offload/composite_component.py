"""Composition of resident, transient, and block components."""

import contextlib
from collections.abc import Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import nn

from .block_compile import BlockCompileConfig
from .block_component import BlockComponent, BlockComponentStore
from .block_mode import BlockMode
from .host_component import HostComponent, HostComponentStore
from .host_module import ParameterOverride
from .module_names import buffer_names, parameter_names


class CompositeComponent:
    """Resident, transient-path, and block components."""

    def __init__(
        self,
        *,
        resident: HostComponent | None,
        blocks: Sequence[BlockComponent],
        transient: Sequence[tuple[str, HostComponent]] = (),
        transient_blocks: Sequence[BlockComponent] = (),
    ) -> None:
        self._resident = resident
        self._transient = tuple(transient)
        self._blocks = tuple(blocks)
        self._transient_blocks = tuple(transient_blocks)
        self._teardown_stack: contextlib.ExitStack | None = None

    @property
    def resident(self) -> HostComponent | None:
        return self._resident

    @property
    def blocks(self) -> tuple[BlockComponent, ...]:
        return self._blocks

    @property
    def transient(self) -> tuple[tuple[str, HostComponent], ...]:
        return self._transient

    @property
    def transient_blocks(self) -> tuple[BlockComponent, ...]:
        return self._transient_blocks

    def _components(self) -> Iterator[HostComponent | BlockComponent]:
        if self._resident is not None:
            yield self._resident
        for _path, component in self._transient:
            yield component
        yield from self._blocks

    @property
    def param_names(self) -> frozenset[str]:
        return frozenset(name for component in self._components() for name in component.param_names)

    @property
    def buffer_names(self) -> frozenset[str]:
        return frozenset(name for component in self._components() for name in component.buffer_names)

    def activate(
        self,
        device: torch.device,
        *,
        compile_blocks: bool = True,
        parameter_overrides: Mapping[str, ParameterOverride] | None = None,
    ) -> None:
        if self._teardown_stack is not None:
            raise RuntimeError("CompositeComponent.activate() called while already active; deactivate() first.")

        overrides = {} if parameter_overrides is None else dict(parameter_overrides)
        if overrides:
            unknown = sorted(set(overrides) - set(self.param_names))
            if unknown:
                raise ValueError(
                    f"Parameter overrides contain unmanaged names: {unknown!r}."
                )

        with contextlib.ExitStack() as stack:
            for component in self._components():
                stack.callback(component.deactivate)
                component.activate(
                    device,
                    compile_blocks=compile_blocks,
                    parameter_overrides=(
                        {
                            name: override
                            for name, override in overrides.items()
                            if name in component.param_names
                        }
                        if overrides
                        else None
                    ),
                )
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
    """Reusable stores for resident, transient, and block model state."""

    resident_store: HostComponentStore | None
    block_stores: tuple[BlockComponentStore, ...]
    transient_block_stores: tuple[BlockComponentStore, ...] = ()
    transient_stores: tuple[tuple[str, HostComponentStore], ...] = ()

    def __post_init__(self) -> None:
        if (
            self.resident_store is None
            and not self.transient_stores
            and not self.block_stores
            and not self.transient_block_stores
        ):
            raise ValueError("Offloading requires at least one parameter, registered buffer, or block to manage.")

    def _stores(
        self,
    ) -> Iterator[HostComponentStore | BlockComponentStore]:
        if self.resident_store is not None:
            yield self.resident_store
        for _path, store in self.transient_stores:
            yield store
        yield from self.block_stores
        yield from self.transient_block_stores

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        block_paths: Sequence[str] = (),
        transient_block_paths: Sequence[str] = (),
        transient_paths: Sequence[str] = (),
        include_block_trainables: bool = False,
    ) -> Self:
        persistent_paths = tuple(block_paths)
        transient_paths_with_blocks = tuple(transient_block_paths)
        overlap = set(persistent_paths) & set(transient_paths_with_blocks)
        if overlap:
            raise ValueError(
                f"block_paths and transient_block_paths must be disjoint; both contain {sorted(overlap)!r}."
            )
        for path in transient_paths_with_blocks:
            blocks = model.get_submodule(path)
            if isinstance(blocks, nn.ModuleList) and len({id(block) for block in blocks}) != len(blocks):
                raise ValueError(
                    "transient_block_paths does not support aliased block "
                    f"modules; {path!r} contains repeated module objects."
                )

        def make_block_store(blocks_path: str) -> BlockComponentStore:
            return BlockComponentStore.from_module(
                model,
                blocks_path=blocks_path,
                include_block_trainables=include_block_trainables,
            )

        block_stores = tuple(make_block_store(blocks_path) for blocks_path in persistent_paths)
        transient_block_stores = tuple(make_block_store(blocks_path) for blocks_path in transient_paths_with_blocks)
        all_block_stores = (*block_stores, *transient_block_stores)
        block_params = {name for store in all_block_stores for name in store.param_names}
        block_buffers = {name for store in all_block_stores for name in store.buffer_names}
        resident_params = parameter_names(model) - block_params
        resident_buffers = buffer_names(model) - block_buffers
        transient_stores: list[tuple[str, HostComponentStore]] = []
        for path in transient_paths:
            module = model.get_submodule(path)
            prefix = f"{path}." if path else ""
            selected_params = {f"{prefix}{name}" for name in parameter_names(module)} & resident_params
            selected_buffers = {f"{prefix}{name}" for name in buffer_names(module)} & resident_buffers
            if not selected_params and not selected_buffers:
                continue
            transient_stores.append(
                (
                    path,
                    HostComponentStore.from_module(
                        model,
                        include_param_names=selected_params,
                        include_buffer_names=selected_buffers,
                    ),
                )
            )
            resident_params -= selected_params
            resident_buffers -= selected_buffers
        resident_store = (
            HostComponentStore.from_module(
                model,
                include_param_names=resident_params,
                include_buffer_names=resident_buffers,
            )
            if resident_params or resident_buffers
            else None
        )
        return cls(
            resident_store=resident_store,
            block_stores=block_stores,
            transient_block_stores=transient_block_stores,
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
        block_mode: BlockMode = "streaming",
    ) -> CompositeComponent:
        resident = self.resident_store.bind(model) if self.resident_store else None
        transient = tuple((path, store.bind(model)) for path, store in self.transient_stores)
        blocks = tuple(
            store.bind(
                model,
                block_compile=block_compile,
                wraparound=True,
                block_mode=block_mode,
            )
            for store in self.block_stores
        )
        transient_blocks = tuple(
            store.bind(
                model,
                block_compile=block_compile,
                wraparound=False,
                block_mode=block_mode,
            )
            for store in self.transient_block_stores
        )
        return CompositeComponent(
            resident=resident,
            blocks=(*blocks, *transient_blocks),
            transient=transient,
            transient_blocks=transient_blocks,
        )


__all__ = ["CompositeComponent", "CompositeComponentStore"]
