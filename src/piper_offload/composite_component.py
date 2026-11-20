"""Composition of resident, transient, and block components."""

import contextlib
from collections.abc import Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Self

import torch
from torch import nn

from .block_compile import BlockCompileConfig
from .block_component import BlockComponent, BlockComponentStore
from .block_mode import BlockMode
from .host_component import HostComponent, HostComponentStore
from .host_module import HostModuleInstance, HostModuleStore, ParameterOverride
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
        self._active = False

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

    @cached_property
    def param_names(self) -> frozenset[str]:
        return frozenset(name for component in self._components() for name in component.param_names)

    @property
    def buffer_names(self) -> frozenset[str]:
        return frozenset(name for component in self._components() for name in component.buffer_names)

    @property
    def failed(self) -> bool:
        """Whether any component still holds device work from a prior session."""
        return any(component.failed for component in self._components())

    def activate(
        self,
        device: torch.device,
        *,
        compile_blocks: bool = True,
        parameter_overrides: Mapping[str, ParameterOverride] | None = None,
    ) -> None:
        """Activate every component in order.

        A failure leaves the components as far as they got; the caller's
        :meth:`deactivate` releases them, as it releases any ended session.
        """
        if self._active:
            raise RuntimeError("CompositeComponent.activate() called while already active; deactivate() first.")

        overrides = {} if parameter_overrides is None else dict(parameter_overrides)
        if overrides:
            unknown = sorted(overrides.keys() - self.param_names)
            if unknown:
                raise ValueError(
                    f"Parameter overrides contain unmanaged names: {unknown!r}."
                )

        self._active = True
        for component in self._components():
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

    def deactivate(self) -> None:
        """Deactivate every component in reverse order, continuing past failures.

        Component deactivation is idempotent, safe before activation, and
        retries whatever an earlier failed release kept, so this both ends
        a session and finishes releasing a failed one.
        """
        self._active = False
        with contextlib.ExitStack() as stack:
            for component in self._components():
                stack.callback(component.deactivate)

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
        return _capture_model(
            model,
            block_paths=block_paths,
            transient_block_paths=transient_block_paths,
            transient_paths=transient_paths,
            include_block_trainables=include_block_trainables,
        ).as_store(cls)

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


type _CapturedHost = tuple[HostComponentStore, HostModuleInstance]
type _CapturedBlocks = tuple[BlockComponentStore, tuple[HostModuleInstance, ...]]


@dataclass(slots=True)
class _ModelCapture:
    """Each store paired with the binding already installed on its source."""

    resident: _CapturedHost | None
    transient: list[tuple[str, _CapturedHost]]
    blocks: tuple[_CapturedBlocks, ...]
    transient_blocks: tuple[_CapturedBlocks, ...]

    def as_store[S: CompositeComponentStore](self, store_type: type[S]) -> S:
        return store_type(
            resident_store=self.resident[0] if self.resident else None,
            transient_stores=tuple((path, store) for path, (store, _) in self.transient),
            block_stores=tuple(store for store, _ in self.blocks),
            transient_block_stores=tuple(store for store, _ in self.transient_blocks),
        )

    def component(self, *, block_compile: BlockCompileConfig | None, block_mode: BlockMode) -> CompositeComponent:
        def block_component(group: _CapturedBlocks, *, wraparound: bool) -> BlockComponent:
            store, instances = group
            return store._component(
                instances, block_compile=block_compile, block_mode=block_mode, wraparound=wraparound,
            )

        blocks = tuple(block_component(group, wraparound=True) for group in self.blocks)
        transient_blocks = tuple(block_component(group, wraparound=False) for group in self.transient_blocks)
        return CompositeComponent(
            resident=HostComponent(self.resident[1]) if self.resident else None,
            blocks=(*blocks, *transient_blocks),
            transient=tuple((path, HostComponent(instance)) for path, (_, instance) in self.transient),
            transient_blocks=transient_blocks,
        )


def _capture_model(
    model: nn.Module,
    *,
    block_paths: Sequence[str],
    transient_block_paths: Sequence[str],
    transient_paths: Sequence[str],
    include_block_trainables: bool,
) -> _ModelCapture:
    overlap = set(block_paths) & set(transient_block_paths)
    if overlap:
        raise ValueError(
            f"block_paths and transient_block_paths must be disjoint; both contain {sorted(overlap)!r}."
        )
    for path in transient_block_paths:
        blocks = model.get_submodule(path)
        if isinstance(blocks, nn.ModuleList) and len({id(block) for block in blocks}) != len(blocks):
            raise ValueError(
                "transient_block_paths does not support aliased block "
                f"modules; {path!r} contains repeated module objects."
            )

    def capture_blocks(path: str) -> _CapturedBlocks:
        return BlockComponentStore._capture(
            model, blocks_path=path, include_block_trainables=include_block_trainables,
        )

    def capture_host(params: set[str], buffers: set[str]) -> _CapturedHost:
        store, instance = HostModuleStore.capture(model, include_param_names=params, include_buffer_names=buffers)
        return HostComponentStore(store), instance

    permanent = tuple(capture_blocks(path) for path in block_paths)
    transient_blocks = tuple(capture_blocks(path) for path in transient_block_paths)
    resident_params = parameter_names(model)
    resident_buffers = buffer_names(model)
    for store, _ in (*permanent, *transient_blocks):
        resident_params -= store.param_names
        resident_buffers -= store.buffer_names
    transient: list[tuple[str, _CapturedHost]] = []
    for path in transient_paths:
        module = model.get_submodule(path)
        prefix = f"{path}." if path else ""
        selected_params = {f"{prefix}{name}" for name in parameter_names(module)} & resident_params
        selected_buffers = {f"{prefix}{name}" for name in buffer_names(module)} & resident_buffers
        if not selected_params and not selected_buffers:
            continue
        transient.append((path, capture_host(selected_params, selected_buffers)))
        resident_params -= selected_params
        resident_buffers -= selected_buffers
    resident = capture_host(resident_params, resident_buffers) if resident_params or resident_buffers else None
    return _ModelCapture(resident, transient, permanent, transient_blocks)



__all__ = ["CompositeComponent", "CompositeComponentStore"]
