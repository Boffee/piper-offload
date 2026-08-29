"""Composition and scheduling of one model's offload components."""

import contextlib
import weakref
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Self, cast

import torch
from torch import nn

from .block_compile import BlockCompileConfig
from .host_backing import HostBacking, validate_host_backing
from .module_names import group_names, walk_attr_path
from .pinned_component import PinnedComponent, PinnedComponentStore
from .pinned_module import PostCopyHook
from .streamed_component import StreamedComponent, StreamedComponentStore
from .tensor_adapter_registry import buffer_tensor_id, param_tensor_id


def _is_within(path: str, parent: str) -> bool:
    return path == parent or path.startswith(f"{parent}.")


def _resolve_attr_state(
    model: nn.Module,
    paths: Sequence[str],
    *,
    argument: str,
) -> tuple[tuple[str, ...], set[str], set[str]]:
    """Resolve module paths and return their recursively-owned state names."""
    paths = tuple(paths)
    if len(set(paths)) != len(paths):
        raise ValueError(f"{argument} contains duplicate paths")

    params: set[str] = set()
    buffers: set[str] = set()
    for path in paths:
        if not isinstance(path, str) or not path:
            raise TypeError(f"{argument} paths must be non-empty strings")
        resolved = walk_attr_path(model, path)
        if not isinstance(resolved, nn.Module):
            raise TypeError(
                f"{argument} path {path!r} resolved to "
                f"{type(resolved).__name__}, expected nn.Module"
            )
        module = cast(nn.Module, resolved)
        params.update(
            f"{path}.{name}"
            for name, _param in module.named_parameters(remove_duplicate=False)
        )
        buffers.update(
            f"{path}.{name}"
            for name, _buffer in module.named_buffers(remove_duplicate=False)
        )
    return paths, params, buffers


def _validate_disjoint_scopes(
    prefix_attr: Sequence[str],
    suffix_attr: Sequence[str],
) -> None:
    for prefix in prefix_attr:
        for suffix in suffix_attr:
            if _is_within(prefix, suffix) or _is_within(suffix, prefix):
                raise ValueError(
                    "prefix_attr and suffix_attr must not overlap; "
                    f"got {prefix!r} and {suffix!r}"
                )


def _keep_scope_local_storage(
    prefix: set[str],
    suffix: set[str],
    groups: Sequence[Sequence[str]],
) -> None:
    """Keep aliases crossing a temporal scope in the resident remainder."""
    for names in groups:
        group = set(names)
        if group & prefix and not group <= prefix:
            prefix.difference_update(group)
        if group & suffix and not group <= suffix:
            suffix.difference_update(group)


class _BoundaryRuntime:
    """Move prefix/suffix pinned components around one central block span."""

    def __init__(
        self,
        model: nn.Module,
        *,
        prefix: PinnedComponent | None,
        suffix: PinnedComponent | None,
        first_block: nn.Module,
        last_block: nn.Module,
    ) -> None:
        self._model = model
        self._prefix = prefix
        self._suffix = suffix
        self._first_block = first_block
        self._last_block = last_block
        self._device: torch.device | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._executor: ThreadPoolExecutor | None = None
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._prefix_prefetch: Future[None] | None = None
        self._in_forward = False
        self._suffix_active = False

    def activate(self, device: torch.device) -> None:
        if self._device is not None:
            raise RuntimeError("boundary runtime is already active")
        if device.type != "cuda":
            raise ValueError(f"boundary runtime requires CUDA; got {device}")
        self._device = device
        if self._prefix is not None:
            self._executor = ThreadPoolExecutor(max_workers=1)
            self._prefetch_stream = torch.cuda.Stream(
                device=device,
                priority=-1,
            )
        runtime_ref = weakref.ref(self)

        def hook(method: str) -> Callable[..., None]:
            def call(*_args: object) -> None:
                runtime = runtime_ref()
                if runtime is not None:
                    getattr(runtime, method)()

            return call

        try:
            self._hooks.append(
                self._model.register_forward_pre_hook(
                    hook("_before_model"),
                    prepend=True,
                )
            )
            if self._prefix is not None:
                self._hooks.append(
                    self._first_block.register_forward_pre_hook(
                        hook("_before_blocks"),
                        prepend=True,
                    )
                )
            if self._suffix is not None:
                self._hooks.append(
                    self._last_block.register_forward_hook(hook("_after_blocks"))
                )
            self._hooks.append(
                self._model.register_forward_hook(hook("_after_model"))
            )
        except BaseException:
            self.deactivate()
            raise

    def _before_model(self) -> None:
        if self._in_forward:
            raise RuntimeError(
                "prefix/suffix runtime has an unfinished forward; reentrant "
                "calls and retries after failure require deactivation first"
            )
        self._in_forward = True
        if self._prefix is None:
            return
        device = self._device
        assert device is not None
        try:
            future = self._prefix_prefetch
            self._prefix_prefetch = None
            current_stream = torch.cuda.current_stream(device)
            if future is None:
                self._prefix._stage(device, current_stream)
            else:
                future.result()
            self._prefix._acquire(current_stream)
        except BaseException:
            self._prefix.deactivate()
            self._in_forward = False
            raise

    def _before_blocks(self) -> None:
        if not self._in_forward or self._prefix is None:
            return
        device = self._device
        assert device is not None
        self._prefix._release(torch.cuda.current_stream(device))

    def _after_blocks(self) -> None:
        if not self._in_forward or self._suffix is None or self._suffix_active:
            return
        device = self._device
        assert device is not None
        try:
            current_stream = torch.cuda.current_stream(device)
            self._suffix._stage(device, current_stream)
            self._suffix._acquire(current_stream)
            self._suffix_active = True
        except BaseException:
            self._suffix.deactivate()
            raise

    def _after_model(self) -> None:
        current_stream: torch.cuda.Stream | None = None
        if self._prefix is not None or self._suffix is not None:
            device = self._device
            assert device is not None
            current_stream = torch.cuda.current_stream(device)
        self._deactivate_scopes(current_stream)
        if self._prefix is not None:
            assert current_stream is not None
            event = torch.cuda.Event()
            event.record(current_stream)
            self._submit_prefix_prefetch(event)
        self._in_forward = False

    def _submit_prefix_prefetch(self, done: torch.cuda.Event) -> None:
        prefix = self._prefix
        executor = self._executor
        stream = self._prefetch_stream
        device = self._device
        assert prefix is not None
        assert executor is not None
        assert stream is not None
        assert device is not None

        def prefetch() -> None:
            # Delay target allocation until the model has released its peak
            # forward allocations, then overlap the copy with caller work.
            done.synchronize()
            prefix._stage(device, stream)

        self._prefix_prefetch = executor.submit(prefetch)

    def _deactivate_scopes(
        self,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        try:
            with contextlib.ExitStack() as stack:
                for component in (self._prefix, self._suffix):
                    if component is not None:
                        stack.callback(component._release, stream)
        finally:
            self._suffix_active = False

    def deactivate(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        first_error: BaseException | None = None
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)
        future = self._prefix_prefetch
        self._prefix_prefetch = None
        if future is not None:
            try:
                future.result()
            except BaseException as exc:
                first_error = exc
        self._prefetch_stream = None
        try:
            self._deactivate_scopes()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        finally:
            self._in_forward = False
            self._device = None
        if first_error is not None:
            raise first_error


class CompositeComponent:
    """Resident, boundary-scoped, and block-streamed model state."""

    def __init__(
        self,
        *,
        resident: PinnedComponent | None,
        prefix: PinnedComponent | None,
        suffix: PinnedComponent | None,
        streamed: Sequence[StreamedComponent],
        boundary: _BoundaryRuntime | None,
    ) -> None:
        self._resident = resident
        self._prefix = prefix
        self._suffix = suffix
        self._streamed = tuple(streamed)
        self._boundary = boundary
        self._teardown_stack: contextlib.ExitStack | None = None

    @property
    def resident(self) -> PinnedComponent | None:
        return self._resident

    @property
    def prefix(self) -> PinnedComponent | None:
        return self._prefix

    @property
    def suffix(self) -> PinnedComponent | None:
        return self._suffix

    @property
    def streamed(self) -> tuple[StreamedComponent, ...]:
        return self._streamed

    def _components(self) -> Iterator[PinnedComponent | StreamedComponent]:
        for component in (self._resident, self._prefix, self._suffix):
            if component is not None:
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
            if device.type != "cuda" or self._boundary is None:
                for component in self._components():
                    stack.callback(component.deactivate)
                    component.activate(device, compile_blocks=compile_blocks)
            else:
                if self._resident is not None:
                    stack.callback(self._resident.deactivate)
                    self._resident.activate(device)
                stack.callback(self._boundary.deactivate)
                self._boundary.activate(device)
                for component in self._streamed:
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
    """Reusable stores for resident, prefix, suffix, and streamed state."""

    resident_store: PinnedComponentStore | None
    prefix_store: PinnedComponentStore | None
    suffix_store: PinnedComponentStore | None
    streamed_stores: tuple[StreamedComponentStore, ...]
    boundary_streamed_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not any(
            (
                self.resident_store,
                self.prefix_store,
                self.suffix_store,
                self.streamed_stores,
            )
        ):
            raise ValueError(
                "Offloading requires at least one parameter, registered "
                "buffer, or streamed block to manage."
            )

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        blocks_attr: Sequence[str] = (),
        prefix_attr: Sequence[str] = (),
        suffix_attr: Sequence[str] = (),
        stream_trainable_weights: bool = False,
        host_backing: HostBacking = "pinned",
    ) -> Self:
        backing = validate_host_backing(host_backing)
        blocks_attr = tuple(blocks_attr)
        prefix_attr, prefix_params, prefix_buffers = _resolve_attr_state(
            model,
            prefix_attr,
            argument="prefix_attr",
        )
        suffix_attr, suffix_params, suffix_buffers = _resolve_attr_state(
            model,
            suffix_attr,
            argument="suffix_attr",
        )
        _validate_disjoint_scopes(
            prefix_attr,
            suffix_attr,
        )

        has_boundary_state = bool(prefix_attr or suffix_attr)
        if has_boundary_state and not blocks_attr:
            raise ValueError(
                "prefix_attr and suffix_attr require at least one blocks_attr path"
            )
        for scope in (*prefix_attr, *suffix_attr):
            for blocks_path in blocks_attr:
                if scope != blocks_path and _is_within(scope, blocks_path):
                    raise ValueError(
                        f"boundary path {scope!r} cannot be nested inside "
                        f"blocks_attr path {blocks_path!r}"
                    )
        boundary_streamed_indices = tuple(
            idx
            for idx, blocks_path in enumerate(blocks_attr)
            if not any(
                _is_within(blocks_path, scope)
                for scope in (*prefix_attr, *suffix_attr)
            )
        )
        if has_boundary_state and not boundary_streamed_indices:
            raise ValueError(
                "prefix_attr and suffix_attr leave no central blocks_attr group"
            )

        all_params_by_name = dict(model.named_parameters(remove_duplicate=False))
        all_buffers_by_name = dict(model.named_buffers(remove_duplicate=False))
        all_params = set(all_params_by_name)
        all_buffers = set(all_buffers_by_name)
        trainable_params = {
            name for name, param in all_params_by_name.items() if param.requires_grad
        }
        if has_boundary_state:
            param_ids = {
                name: param_tensor_id(param)
                for name, param in all_params_by_name.items()
            }
            buffer_ids = {
                name: buffer_tensor_id(buffer)
                for name, buffer in all_buffers_by_name.items()
            }
            param_groups = group_names(param_ids, param_ids.__getitem__)
            buffer_groups = group_names(buffer_ids, buffer_ids.__getitem__)
        else:
            param_groups = buffer_groups = ()
        # Construction may replace source wrappers incrementally. Keep only
        # names and storage-group metadata across the pinning phase so those
        # source allocations can be reclaimed promptly.
        del all_params_by_name, all_buffers_by_name

        streamed_stores = tuple(
            StreamedComponentStore.from_module(
                model,
                blocks_path=blocks_path,
                stream_trainable_weights=stream_trainable_weights,
                host_backing=backing,
            )
            for blocks_path in blocks_attr
        )
        streamed_params = {n for s in streamed_stores for n in s.param_names}
        streamed_buffers = {n for s in streamed_stores for n in s.buffer_names}

        prefix_params -= streamed_params | trainable_params
        suffix_params -= streamed_params | trainable_params
        prefix_buffers -= streamed_buffers
        suffix_buffers -= streamed_buffers
        _keep_scope_local_storage(prefix_params, suffix_params, param_groups)
        _keep_scope_local_storage(prefix_buffers, suffix_buffers, buffer_groups)

        resident_params = all_params - streamed_params - prefix_params - suffix_params
        resident_buffers = all_buffers - streamed_buffers - prefix_buffers - suffix_buffers

        def pinned_store(
            params: set[str],
            buffers: set[str],
        ) -> PinnedComponentStore | None:
            if not params and not buffers:
                return None
            return PinnedComponentStore.from_module(
                model,
                include_param_names=params,
                include_buffer_names=buffers,
                host_backing=backing,
            )

        return cls(
            resident_store=pinned_store(resident_params, resident_buffers),
            prefix_store=pinned_store(prefix_params, prefix_buffers),
            suffix_store=pinned_store(suffix_params, suffix_buffers),
            streamed_stores=streamed_stores,
            boundary_streamed_indices=(
                boundary_streamed_indices if has_boundary_state else ()
            ),
        )

    @property
    def cache_bytes(self) -> int:
        pinned = sum(
            store.cache_bytes
            for store in (
                self.resident_store,
                self.prefix_store,
                self.suffix_store,
            )
            if store is not None
        )
        return pinned + sum(store.cache_bytes for store in self.streamed_stores)

    @property
    def has_trainables(self) -> bool:
        pinned = any(
            store is not None and store.has_trainables
            for store in (
                self.resident_store,
                self.prefix_store,
                self.suffix_store,
            )
        )
        return pinned or any(store.has_trainables for store in self.streamed_stores)

    def bind(
        self,
        model: nn.Module,
        *,
        block_compile: BlockCompileConfig | None = None,
    ) -> CompositeComponent:
        resident = self.resident_store.bind(model) if self.resident_store else None
        prefix = self.prefix_store.bind(model) if self.prefix_store else None
        suffix = self.suffix_store.bind(model) if self.suffix_store else None
        streamed = tuple(
            store.bind(model, block_compile=block_compile)
            for store in self.streamed_stores
        )
        boundary = None
        if prefix is not None or suffix is not None:
            first = streamed[self.boundary_streamed_indices[0]].blocks[0]
            last = streamed[self.boundary_streamed_indices[-1]].blocks[-1]
            boundary = _BoundaryRuntime(
                model,
                prefix=prefix,
                suffix=suffix,
                first_block=first,
                last_block=last,
            )
        return CompositeComponent(
            resident=resident,
            prefix=prefix,
            suffix=suffix,
            streamed=streamed,
            boundary=boundary,
        )


__all__ = ["CompositeComponent", "CompositeComponentStore"]
