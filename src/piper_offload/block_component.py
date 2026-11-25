"""Internal block-residency primitive for memory-efficient inference.

A :class:`BlockComponent` manages a structurally compatible block list bound to
host backing from :class:`~piper_offload.model_store.ModelStore`, composed
internally inside :class:`~piper_offload.model_offloader.ModelOffloader`.
"""

import contextlib
import functools
import logging
import weakref
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from typing import cast

import torch
from torch import nn

from ._devices import canonical_device
from ._working_set import WorkingSetState
from .block_compile import BlockCompileConfig, _BlockCompileState
from .block_mode import BlockMode
from .block_runtime import BlockRuntime
from .host_module import (
    HostModuleInstance,
    HostModuleLoadPlan,
    ParameterOverride,
)
from .host_param import HostParam
from .module_names import walk_attr_path
from .pin_manager import PinLease, TransferLease, host_pin_manager
from .resident_runtime import ResidentBlockRuntime
from .rolling_runtime import RollingBlockRuntime, create_rolling_block_runtime
from .streaming_runtime import StreamingBlockRuntime

logger = logging.getLogger(__name__)


def _release_cuda_cache_on_drop(is_cuda: bool) -> None:
    # Process-wide PyTorch CUDA allocator cache is the only state the
    # refcount-based GC of a block component can't release on its own. Without
    # this, freed GPU pages stay held by the allocator until the
    # next allocation pressure event, which manifests as OOMs at
    # workload boundaries (e.g. successive workloads in one process).
    # ``empty_cache()`` is process-global (not per-device), so a single
    # bool is the right abstraction — capturing the device object would
    # imply per-device scoping that PyTorch doesn't actually provide.
    if not is_cuda:
        return
    # Finalizers can run at interpreter shutdown when CUDA is already torn
    # down, so suppress teardown-time noise.
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Block instances
# ---------------------------------------------------------------------------


def _param_target_layout(p: nn.Parameter) -> tuple[object, object]:
    """Per-parameter target-compatibility layout, computed pre-capture.

    Two params with equal values pool into structurally identical GPU
    targets, so a refill is a plain ``Tensor.copy_``; unequal values must
    never share a target, because ``copy_`` silently casts dtype and
    silently broadcasts compatible shapes, and wrapper metadata (qtype,
    axis, activation_qtype, group_size) is similarly invisible to it.

    This is the standalone form of the same value
    :attr:`HostParam.target_layout` supplies the same value to the block
    runtime's target-pool signature; this helper is not called on the
    streaming path. Kept as the package's documented
    way to compare two params' target compatibility directly (e.g. in
    adapter tests) without capturing either.

    :class:`HostParam` owns the tensor-adapter details because wrapper
    metadata is type-specific. The returned value intentionally excludes
    tensor identity so distinct blocks with the same layout share one
    pooled target.
    """
    return HostParam.target_layout_for(p)


def _build_param_name_index(
    instances: Sequence[HostModuleInstance],
    prefix: str | None,
    block_indices: Sequence[int],
) -> dict[str, tuple[int, str]]:
    # The external NAME uses the true block index so a sparse group's params
    # are addressed at their real path; the stored VALUE keeps the compact
    # position used to index ``_block_instances`` for the block runtime.
    index: dict[str, tuple[int, str]] = {}
    for compact_idx, instance in enumerate(instances):
        true_idx = block_indices[compact_idx]
        for local_name in instance.params:
            name = _block_tensor_name(prefix, true_idx, local_name)
            if name in index:
                raise ValueError(f"duplicate block parameter name {name!r}")
            index[name] = (compact_idx, local_name)
    return index


def _block_buffer_names(
    instances: Sequence[HostModuleInstance],
    prefix: str | None,
    block_indices: Sequence[int],
) -> frozenset[str]:
    names: set[str] = set()
    for block_idx, instance in zip(block_indices, instances, strict=True):
        for local_name in instance.buffers:
            name = _block_tensor_name(prefix, block_idx, local_name)
            if name in names:
                raise ValueError(f"duplicate block buffer name {name!r}")
            names.add(name)
    return frozenset(names)


def _block_tensor_name(
    prefix: str | None,
    block_idx: int,
    local_name: str,
) -> str:
    name = f"{block_idx}.{local_name}"
    return name if prefix is None else f"{prefix}.{name}"


def _resolve_blocks(module: nn.Module, blocks_path: str) -> list[nn.Module]:
    obj = walk_attr_path(module, blocks_path)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{blocks_path}', got {type(obj).__name__}")
    blocks = list(cast(nn.ModuleList, obj))
    if not blocks:
        raise ValueError(f"Block path {blocks_path!r} resolved to an empty list")
    return blocks


def _block_is_empty(block: nn.Module) -> bool:
    """Return whether ``block`` holds no parameters or buffers at any depth."""
    return next(block.parameters(), None) is None and next(block.buffers(), None) is None


# ---------------------------------------------------------------------------
# BlockComponent — internal block-residency primitive
# ---------------------------------------------------------------------------


class BlockComponent:
    """Internal CUDA/CPU residency runtime for one block group.

    Manages bound block instances backed by host parameters and buffers from
    :class:`~piper_offload.model_store.ModelStore`. By default it streams them
    to CUDA via forward-pre hooks on :meth:`activate`; resident mode loads every
    block once without scheduling hooks, and rolling compilation uses one shared
    parameter runtime. CPU activation runs directly over host-backed state.

    Lifecycle is uniform with :class:`HostComponent`: ``activate`` starts a
    device session and initially acquires its CUDA working set, while
    ``deactivate`` releases that working set, returns state to host CPU,
    removes hooks, and restores eager forwards. :meth:`release` and
    :meth:`acquire` cycle the CUDA working set without ending the session.

    **Calling deactivate() before dropping the binding is preferred**
    — it removes forward hooks and reverts GPU-resident blocks back to CPU.
    The hook closure uses ``weakref.ref(self)`` so dropping the binding
    without deactivate is non-fatal: the orphaned hooks remain installed on the
    model
    but no-op; resident blocks stay on GPU until the model itself is
    dropped. Forward calls through still-resident blocks work; calls
    through previously-evicted blocks find CPU tensors (slow but
    functional).

    Instances are usually created by binding a
    :class:`~piper_offload.model_store.ModelStore` or via
    :meth:`ModelOffloader.from_module`.

    Parameters
    ----------
    block_instances:
        The concrete bound block instances.
    name:
        Optional model path for the block list. When set,
        :attr:`param_names` and activation parameter overrides use names like
        ``"blocks.3.weight"``. When omitted, standalone components use
        ``"3.weight"``.
    block_indices:
        True block index for each instance, used to NAME its params/buffers
        (so a sparse group addresses ``"blocks.2.weight"`` not the compact
        ``"blocks.1.weight"``). The block runtime still uses the compact
        ``0..k-1`` position internally. Defaults to ``0..k-1`` (contiguous).
    block_compile:
        Optional forward-only compile policy. One lazy callable is retained per
        distinct block module object and compiler backend, installed during eligible
        activations and removed on deactivate. CPU uses ordinary Inductor on host state.
    wraparound:
        Whether the runtime prepares block 0 while executing the final block.
        :class:`ModelOffloader` disables this for transient streamed pools.
    block_mode:
        CUDA residency strategy. ``"resident"`` keeps every block loaded,
        ``"streaming"`` rotates whole blocks through an active/lookahead
        target pool, and ``"rolling"`` refills one shared target
        parameter-by-parameter through a compiled full graph. ``"auto"``
        selects rolling independently for each supported block group and
        otherwise uses streaming.
    """

    _pin_lease: PinLease | None
    _transfer: TransferLease

    def __init__(
        self,
        block_instances: Sequence[HostModuleInstance],
        *,
        name: str | None = None,
        block_indices: Sequence[int] | None = None,
        param_name_index: Mapping[str, tuple[int, str]] | None = None,
        buffer_names: Iterable[str] | None = None,
        block_compile: BlockCompileConfig | None = None,
        wraparound: bool = True,
        block_mode: BlockMode = "streaming",
    ) -> None:
        fullgraph = block_compile is not None and block_compile.fullgraph
        if block_mode == "rolling" and not fullgraph:
            raise ValueError("block_mode='rolling' requires BlockCompileConfig(fullgraph=True).")
        self._block_instances = list(block_instances)
        if block_indices is None:
            block_indices = range(len(self._block_instances))
        block_indices = list(block_indices)
        if len(block_indices) != len(self._block_instances):
            raise ValueError(
                "block_indices must have one index per managed block: "
                f"got {len(block_indices)} for "
                f"{len(self._block_instances)} blocks."
            )
        self._blocks = [instance.module for instance in self._block_instances]
        rolling = None
        if block_mode == "resident":
            eager_runtime: BlockRuntime = ResidentBlockRuntime(self._block_instances)
            self._block_mode: BlockMode = "resident"
        elif block_mode in ("streaming", "rolling", "auto"):
            # Check rolling support before constructing the streaming runtime.
            if block_mode != "streaming" and fullgraph:
                try:
                    rolling = create_rolling_block_runtime(self._block_instances, wraparound=wraparound)
                except NotImplementedError as exc:
                    if block_mode == "rolling":
                        raise
                    logger.info("block group %r does not support rolling; using streaming: %s", name, exc)
            eager_runtime = StreamingBlockRuntime(self._block_instances, wraparound=wraparound)
            self._block_mode = "rolling" if rolling is not None else "streaming"
        else:
            raise ValueError(f"unsupported block mode: {block_mode!r}")
        self._eager_runtime = eager_runtime
        self._runtime: BlockRuntime = rolling if rolling is not None else eager_runtime
        self._allow_rolling_fallback = block_mode == "auto"
        self._block_compile = _BlockCompileState.create(
            self._blocks,
            block_compile,
            backend=self._runtime.compile_backend,
        )
        self._inductor_compile: _BlockCompileState | None = (
            self._block_compile if self._runtime.compile_backend == "inductor" else None
        )
        self._active_device: torch.device | None = None
        self._active_runtime: BlockRuntime | None = None
        self._acquire_runtime: Callable[[], None] | None = None
        self._load_plans: tuple[HostModuleLoadPlan, ...] = ()
        # Streaming and rolling hold one lease per session; resident holds
        # one per transfer. The working set's state says how far acquiring
        # or releasing them got.
        self._pin_lease, self._transfer, self._working_set_state = None, TransferLease(), WorkingSetState.RELEASED
        self._param_name_to_block_param = (
            dict(param_name_index)
            if param_name_index is not None
            else _build_param_name_index(
                self._block_instances,
                name,
                block_indices,
            )
        )
        self._param_names = frozenset(self._param_name_to_block_param)
        self._buffer_names = (
            frozenset(buffer_names)
            if buffer_names is not None
            else _block_buffer_names(self._block_instances, name, block_indices)
        )

        # Auto-flush the CUDA allocator cache when the component is GC'd,
        # so callers don't need to remember an explicit empty_cache() at
        # workload boundaries. Captures only a bool (no self ref) so it
        # never blocks collection.
        weakref.finalize(
            self,
            _release_cuda_cache_on_drop,
            True,
        )

    @property
    def blocks(self) -> tuple[nn.Module, ...]:
        """The bound modules managed by this component, in order."""
        return tuple(self._blocks)

    @property
    def block_compile(self) -> BlockCompileConfig | None:
        """This component's optional block compilation policy."""
        return self._block_compile.config

    @property
    def block_mode(self) -> BlockMode:
        """Resolved CUDA block residency strategy."""
        return self._block_mode

    @property
    def managed_param_names_by_block(self) -> list[list[str]]:
        """Per-block managed parameter names."""
        return [list(instance.params) for instance in self._block_instances]

    @property
    def managed_buffer_names_by_block(self) -> list[list[str]]:
        """Per-block managed buffer names."""
        return [list(instance.buffers) for instance in self._block_instances]

    @property
    def param_names(self) -> frozenset[str]:
        """Externally addressable managed parameter names."""
        return self._param_names

    @property
    def buffer_names(self) -> frozenset[str]:
        """Externally addressable managed buffer names."""
        return self._buffer_names

    @property
    def failed(self) -> bool:
        """Whether a prior session left its working set unreleased.

        A runtime whose release failed stays acquired, keeping its targets,
        and the component keeps the pin leases its transfers read. A later
        :meth:`release` or :meth:`deactivate` retries.
        """
        return self._active_device is None and self._working_set_state is not WorkingSetState.RELEASED

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(
        self,
        device: torch.device,
        *,
        compile_blocks: bool = True,
        parameter_overrides: Mapping[str, ParameterOverride] | None = None,
    ) -> None:
        """Activate the block list on ``device``.

        CUDA activation selects the ordinary, all-resident, or compiled rolling
        block runtime, then installs optional compiled block forwards. CPU
        activation uses host-backed state with optional ordinary Inductor forwards.
        The composite's :meth:`activate` returns the model — this
        method returns ``None`` because the component doesn't own one.

        **Lifecycle is caller's responsibility.** Calling activate()
        twice without an intervening deactivate() raises before hooks or
        block pools are installed.

        **Activation failure semantics:** if activation fails midway,
        the component is left in an undefined partial state. Retrying
        activation on that component is unsupported; the caller's only
        supported cleanup path is :meth:`deactivate`, which idempotently
        tears down whatever was allocated."""
        # Hard-guard against the documented "don't activate twice"
        # case. Without this, a double-activate would double-install
        # forward-pre hooks and stack a second
        # target pool on top of an active one.
        if self._active_device is not None:
            raise RuntimeError(
                "BlockComponent.activate() called while already "
                "active. Deactivate first, or check for a leaked "
                "context manager."
            )
        if self.failed:
            # A new session would silently reuse the old targets and load plan.
            raise RuntimeError(
                "BlockComponent cannot activate after its prior CUDA session "
                "failed to finish releasing; deactivate() retries that "
                "release. Recreate the CUDA worker only if the retry fails too."
            )
        active_device = canonical_device(device)
        if active_device.type == "cpu":
            if parameter_overrides:
                raise ValueError(
                    "Parameter overrides require CUDA activation."
                )
            self._activate_cpu_resolved(compile_blocks=compile_blocks)
            return
        if active_device.type != "cuda":
            raise ValueError(f"BlockComponent.activate() supports CUDA or CPU; got {active_device}.")
        self._load_plans = self._resolve_load_plans(parameter_overrides)
        self._activate_cuda_resolved(
            active_device,
            compile_blocks=compile_blocks,
        )

    def _resolve_load_plans(
        self,
        overrides: Mapping[str, ParameterOverride] | None,
    ) -> tuple[HostModuleLoadPlan, ...]:
        peer_indices_by_module: dict[int, list[int]] = {}
        for idx, instance in enumerate(self._block_instances):
            peer_indices_by_module.setdefault(id(instance.module), []).append(idx)

        by_block: list[dict[str, ParameterOverride]] = [
            {} for _instance in self._block_instances
        ]
        if overrides is not None:
            for name, override in overrides.items():
                ref = self._param_name_to_block_param.get(name)
                if ref is None:
                    raise ValueError(
                        f"param name {name!r} is not owned by this block component"
                    )
                block_idx, local_name = ref
                for peer_idx in peer_indices_by_module[
                    id(self._block_instances[block_idx].module)
                ]:
                    existing = by_block[peer_idx].get(local_name)
                    if existing is not None and (
                        existing.source is not override.source
                        or existing.update is not override.update
                    ):
                        raise ValueError(
                            "Tied parameter aliases cannot resolve to different loads; "
                            f"conflict at {name!r}."
                        )
                    by_block[peer_idx][local_name] = override
        return tuple(
            instance.resolve_load_plan(local_overrides)
            for instance, local_overrides in zip(
                self._block_instances,
                by_block,
                strict=True,
            )
        )

    def _get_inductor_compile(self) -> _BlockCompileState:
        """Share ordinary compilation between CPU and non-rolling CUDA execution."""
        if self._inductor_compile is None:
            self._inductor_compile = _BlockCompileState.create(
                self._blocks,
                self._block_compile.config,
                backend="inductor",
            )
        return self._inductor_compile

    def _activate_cpu_resolved(self, *, compile_blocks: bool) -> None:
        self._active_device = torch.device("cpu")
        if compile_blocks:
            self._get_inductor_compile().install(True)

    def _activate_cuda_resolved(
        self,
        active_device: torch.device,
        *,
        compile_blocks: bool,
    ) -> None:
        runtime = self._eager_runtime
        acquire = functools.partial(runtime.acquire, active_device, self._load_plans)
        block_compile = self._block_compile
        if compile_blocks and isinstance(self._runtime, RollingBlockRuntime):
            try:
                acquire = self._runtime._prepare_acquire(active_device, self._load_plans)
            except NotImplementedError as exc:
                if not self._allow_rolling_fallback:
                    raise
                logger.info("block group does not support rolling for this activation; using streaming: %s", exc)
                block_compile = self._get_inductor_compile()
            else:
                runtime = self._runtime

        # Record the selected runtime before acquisition so deactivate() can
        # clean up a partially-created pool, stream, or hook set if activation
        # raises midway through its lifecycle.
        self._active_device = active_device
        self._active_runtime = runtime
        self._acquire_runtime = acquire
        self.acquire()
        block_compile.install(compile_blocks)

    def acquire(self) -> None:
        """Acquire this active session's CUDA working set.

        Idempotent when already acquired. CPU sessions have no CUDA working
        set and therefore need no acquisition. The activation session and any
        installed compiled forwards remain intact across release/acquire
        cycles. The component leases host storage before the selected runtime
        starts. Streaming and rolling ask the lease to register that storage
        under the global pin budget, since they read it every step; a resident
        upload leases it pageable, registers nothing, and releases the lease
        once the runtime has synchronized the upload. A failed acquisition
        releases its partial working set, so the session can acquire again,
        also after an acquire or release that an exception stopped partway.
        """
        active_device = self._active_device
        if active_device is None:
            raise RuntimeError("BlockComponent.acquire() requires an active session; call activate() first.")
        if active_device.type == "cpu":
            return
        acquire = self._acquire_runtime
        if acquire is None:
            raise RuntimeError("BlockComponent CUDA session has no selected runtime.")
        if self._working_set_state is WorkingSetState.ACQUIRED:
            return
        if self._working_set_state is WorkingSetState.CHANGING:
            self.release()
        sources = (tensor for plan in self._load_plans for tensor in plan.storage_tensors())
        self._working_set_state = WorkingSetState.CHANGING
        try:
            if self._block_mode == "resident":
                # A one-time upload: lease the sources pageable, for the upload
                # only; the runtime synchronizes it before returning.
                self._transfer.start(host_pin_manager, sources, active_device)
                acquire()
                self._transfer.close()  # the runtime synchronized the upload
            else:
                self._pin_lease = host_pin_manager.acquire(sources)
                acquire()
        except BaseException:
            self.release()
            raise
        self._working_set_state = WorkingSetState.ACQUIRED

    def _close_pin_lease(self) -> None:
        if self._pin_lease is not None:
            self._pin_lease.close()
            self._pin_lease = None

    def release(self) -> None:
        """Idempotently release this session's CUDA working set.

        Blocks return to host backing and runtime-owned CUDA targets, streams,
        prefetch work, and hooks are released. The component remains active,
        and compiled forwards remain installed, so :meth:`acquire` can prepare
        the same session for another traversal. Target retirement completes
        recorded CUDA work, so release is safe immediately after a forward.
        Host pin leases are released after the runtime confirms all copies
        have completed. Their registrations remain in the idle LRU for reuse
        until budget pressure, explicit clearing, or source disposal
        unregisters them. A runtime whose release fails stays acquired and
        keeps those leases; calling this again, also after the session has
        ended, retries it.
        """
        if self._working_set_state is WorkingSetState.RELEASED:
            return
        self._working_set_state = WorkingSetState.CHANGING
        runtimes = (self._runtime, self._eager_runtime)
        try:
            for runtime in runtimes:
                if runtime.acquired:
                    runtime.release()
        finally:
            # A runtime can raise after finishing its cleanup, as after a
            # failed prefetch; one that could not finish stays acquired.
            if not any(runtime.acquired for runtime in runtimes):
                # Every runtime synchronized before releasing its targets.
                self._close_pin_lease()
                self._transfer.close()
                self._working_set_state = WorkingSetState.RELEASED

    def deactivate(self) -> None:
        """Tear down active resources idempotently — safe to call
        before activate or multiple times. The selected runtime owns cleanup
        of any partial acquisition state. A release that fails still ends the
        session but keeps what it could not release (see :attr:`failed`);
        calling this again retries it. Drop the binding reference after
        deactivate to release host memory."""
        self._block_compile.restore()
        if self._inductor_compile is not None:
            self._inductor_compile.restore()
        try:
            self.release()
        finally:
            self._active_runtime = None
            self._acquire_runtime = None
            self._active_device = None
            self._load_plans = ()

    @contextlib.contextmanager
    def use(
        self,
        device: torch.device | str,
        *,
        compile_blocks: bool = True,
    ) -> Generator[None]:
        """Activate on ``device`` for the duration of the context."""
        self.activate(canonical_device(device), compile_blocks=compile_blocks)
        try:
            yield
        finally:
            self.deactivate()


__all__ = ["BlockComponent"]
