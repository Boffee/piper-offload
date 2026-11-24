"""Block-residency primitive for memory-efficient inference.

A :class:`BlockComponentStore` manages reusable CPU backing
storage for a single structurally compatible block list.
Binding that store to a compatible model creates a
:class:`BlockComponent` that either streams the resolved blocks to GPU on
demand or keeps every block resident. Ordinary streaming uses a reusable GPU
target pool and background prefetcher; rolling compilation selects a
single-target parameter runtime; resident mode loads every target once. On
CPU, the host-backed state is used directly without streaming.

Blocks in one list may be heterogeneously quantized (mixed dtypes or
quant formats on the same-named weights): the GPU target pool keys its
reusable targets by per-block layout signature, so each block streams
into a target matching its own format and no cross-format ``copy_`` ever
happens. Only block lists whose blocks differ in *structure* — different
parameter or buffer *names*, e.g. Flux's two block kinds — must split
into multiple :class:`BlockComponent` instances composed via
:class:`ModelOffloader`.

This is the sharp, low-level primitive. It does NOT manage:

- Non-block parts of the model (parent-module state, sibling
  modules) — caller derives :class:`HostComponent` include-name sets
  by excluding the block component's owned names.
- Shared storage with tensors outside the block list — caller
  must choose a valid composition; use whole-model
  :class:`HostComponent` if sharing must be preserved.

Most users want :class:`ModelOffloader` (the blessed safe API). Reach for
:class:`BlockComponentStore` / :class:`BlockComponent` directly only
when you need bespoke composition (e.g., multiple block lists like Flux's
``transformer_blocks`` + ``single_transformer_blocks``).
"""

import contextlib
import functools
import logging
import weakref
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import Self, cast

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
    HostModuleStore,
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
    axis, activation_qtype, quant_type) is similarly invisible to it.

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
    """A block with no parameters or buffers at any depth.

    Empty positions carry nothing to stream and are skipped by
    :meth:`BlockComponentStore.from_module` while later blocks retain their
    true externally-visible indices.
    """
    return next(block.parameters(), None) is None and next(block.buffers(), None) is None


@dataclass(frozen=True, slots=True)
class BlockComponentStore:
    """Reusable host backing storage for a block group.

    Built via :meth:`from_module`: the tensor source is the model's block list,
    resolved by ``blocks_path``.

    ``block_indices`` records which positions of the resolved block list this
    group actually occupies. :meth:`from_module` skips structurally-empty
    positions (no parameters or buffers) while keeping each retained block at
    its true externally-visible index.
    """

    _block_stores: tuple[HostModuleStore, ...]
    blocks_path: str
    block_indices: tuple[int, ...]

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        blocks_path: str,
    ) -> Self:
        """Resolve ``blocks_path`` on ``model`` and capture its managed blocks.

        Structurally-empty positions (no parameters or buffers) are skipped and
        dropped from :attr:`block_indices`. Requires frozen parameters
        (``requires_grad=False``). Surviving blocks must agree on parameter
        names and buffer names.
        """
        store, _ = cls._capture(model, blocks_path=blocks_path)
        return store

    @classmethod
    def _capture(
        cls, model: nn.Module, *, blocks_path: str,
    ) -> tuple[Self, tuple[HostModuleInstance, ...]]:
        # Validate every block before capture mutates the first. The schema
        # holds names only, so each block's original tensors can be reclaimed as
        # soon as its host backing is installed.
        blocks = _resolve_blocks(model, blocks_path)
        indices: list[int] = []
        params: set[str] = set()
        buffers: set[str] = set()
        for idx, block in enumerate(blocks):
            trainable = sorted(
                name
                for name, p in block.named_parameters(remove_duplicate=False)
                if p.requires_grad
            )
            if trainable:
                raise ValueError(
                    "Piper Offload is inference-only and requires frozen parameters "
                    f"(requires_grad=False); block {idx} at {blocks_path!r} has "
                    f"trainable parameter(s): {trainable!r}."
                )
            block_params = {name for name, _ in block.named_parameters(remove_duplicate=False)}
            block_buffers = {name for name, _ in block.named_buffers(remove_duplicate=False)}
            if not block_params and not block_buffers:
                continue
            if indices:
                if block_params != params:
                    difference = sorted(block_params ^ params)
                    raise ValueError(
                        f"Block {idx} parameter names differ from block {indices[0]}: {difference}."
                    )
                if block_buffers != buffers:
                    difference = sorted(block_buffers ^ buffers)
                    raise ValueError(f"Block {idx} buffer names differ from block {indices[0]}: {difference}.")
            else:
                params, buffers = block_params, block_buffers
            indices.append(idx)
        if not indices:
            raise ValueError(
                f"Block path {blocks_path!r} has no streamable blocks (every block is structurally empty)."
            )

        captured = [
            HostModuleStore.capture(blocks[idx], include_param_names=params, include_buffer_names=buffers)
            for idx in indices
        ]
        store = cls(tuple(store for store, _ in captured), blocks_path, tuple(indices))
        return store, tuple(instance for _, instance in captured)

    @property
    def managed_param_names_by_block(self) -> list[list[str]]:
        """Per-block managed parameter names."""
        return [list(store.params) for store in self._block_stores]

    @property
    def managed_buffer_names_by_block(self) -> list[list[str]]:
        """Per-block managed buffer names."""
        return [list(store.buffers) for store in self._block_stores]

    @property
    def param_names(self) -> frozenset[str]:
        """Externally addressable managed parameter names.

        Named by TRUE block index (:attr:`block_indices`), not the compact
        store position — so a sparse group's factors are advertised at their
        real path (``blocks.2...``) and the host-remainder subtraction in
        :meth:`CompositeComponentStore.from_module` lines up correctly.
        """
        names = {
            _block_tensor_name(self.blocks_path, true_idx, local_name)
            for true_idx, store in zip(
                self.block_indices,
                self._block_stores,
                strict=True,
            )
            for local_name in store.params
        }
        return frozenset(names)

    @property
    def buffer_names(self) -> frozenset[str]:
        """Externally addressable managed buffer names (true block indices)."""
        return frozenset(
            _block_tensor_name(self.blocks_path, block_idx, local_name)
            for block_idx, store in zip(self.block_indices, self._block_stores, strict=True)
            for local_name in store.buffers
        )

    @property
    def cache_bytes(self) -> int:
        return sum(store.cache_bytes for store in self._block_stores)

    def resolve_blocks(self, model: nn.Module) -> list[nn.Module]:
        """Resolve this store's blocks path on ``model``."""
        return _resolve_blocks(model, self.blocks_path)

    def _validate_bind(self, model: nn.Module) -> list[nn.Module]:
        last = self.block_indices[-1]
        bind_blocks = self.resolve_blocks(model)
        if last >= len(bind_blocks):
            raise ValueError(
                f"BlockComponentStore.bind() bind model has too few blocks "
                f"at {self.blocks_path!r}: needs index {last}, found "
                f"{len(bind_blocks)}."
            )
        occupied = set(self.block_indices)
        unmanaged = [pos for pos, block in enumerate(bind_blocks) if pos not in occupied and not _block_is_empty(block)]
        if unmanaged:
            raise ValueError(
                f"BlockComponentStore.bind() bind model has non-empty "
                f"block(s) {unmanaged} at {self.blocks_path!r} that this store "
                f"does not occupy (it occupies {list(self.block_indices)}); "
                "those blocks would never be moved or streamed. Bind a model "
                "whose only non-empty blocks are the occupied ones."
            )
        for store, idx in zip(self._block_stores, self.block_indices, strict=True):
            store._validate_bind(bind_blocks[idx])
        return bind_blocks

    def _bind_validated(
        self,
        bind_blocks: Sequence[nn.Module],
        *,
        block_compile: BlockCompileConfig | None = None,
        wraparound: bool = True,
        block_mode: BlockMode = "streaming",
    ) -> BlockComponent:
        instances = [
            store._install(bind_blocks[idx]) for store, idx in zip(self._block_stores, self.block_indices, strict=True)
        ]
        return self._component(instances, block_compile=block_compile, wraparound=wraparound, block_mode=block_mode)

    def bind(
        self,
        model: nn.Module,
        *,
        block_compile: BlockCompileConfig | None = None,
        wraparound: bool = True,
        block_mode: BlockMode = "streaming",
    ) -> BlockComponent:
        """Bind this store's per-block backing bytes to ``model``.

        Each instance owns and is installed onto its model block; loads repoint
        the instance's own module onto the filled target, and that block's
        forward triggers its streaming. ``model`` must have a block at every
        occupied :attr:`block_indices` position. Non-empty blocks outside those
        positions are rejected because they would be silently unmanaged.
        """
        bind_blocks = self._validate_bind(model)
        return self._bind_validated(
            bind_blocks,
            block_compile=block_compile,
            wraparound=wraparound,
            block_mode=block_mode,
        )

    def _component(
        self,
        instances: Sequence[HostModuleInstance],
        *,
        block_compile: BlockCompileConfig | None,
        wraparound: bool,
        block_mode: BlockMode,
    ) -> BlockComponent:
        return BlockComponent(
            instances,
            name=self.blocks_path,
            block_indices=self.block_indices,
            block_compile=block_compile,
            wraparound=wraparound,
            block_mode=block_mode,
        )


# ---------------------------------------------------------------------------
# BlockComponent — public block-residency primitive
# ---------------------------------------------------------------------------


class BlockComponent:
    """Manages a block list between CPU and CUDA.

    The sharp, low-level block-residency primitive. Manages bound block
    instances whose owned params and buffers are stored on CPU. By default it
    streams them to CUDA via forward-pre hooks on :meth:`activate`; with
    resident mode loads every block once and installs no scheduling hooks.
    CPU activation is pass-through over that host-backed state:
    no pool, no hooks, no copies.

    A :class:`BlockComponent` is a *component* meant to be composed
    inside a :class:`~piper_offload.model_offloader.ModelOffloader`.
    It deliberately is NOT a top-level model runtime (its
    :meth:`activate` returns ``None`` because it doesn't own the
    model). For top-level use, build a model runtime via
    :class:`~piper_offload.model_offloader.ModelOffloader`.

    Lifecycle is uniform with :class:`HostComponent`: store construction
    captures host storage (so ``cache_bytes`` is final at construction time, ready
    for :class:`~piper_offload.resource_cache.ResourceCache` admission), and
    ``activate`` starts a device session and initially acquires its CUDA working
    set, while ``deactivate`` releases that working set, returns state to host
    CPU, removes hooks, and restores eager forwards. :meth:`release` and
    :meth:`acquire` may cycle the CUDA working set without ending the session.
    Optional
    ``block_compile`` policy belongs to this bound runtime and installs lazy
    compiled forwards for eligible CPU and CUDA inference activations. Ordinary
    streaming owns one active block and one lookahead target, resident mode
    owns one target per block, and rolling compilation owns one shared
    parameter target. Streaming strategies wrap to block 0 by default;
    transient model scheduling disables that wraparound.
    There is no ``close()``; host memory in module state is freed when the
    caller drops the binding and model references.

    **Calling deactivate() before dropping the binding is preferred**
    — it removes the forward hooks (cleaner model state) and reverts
    GPU-resident blocks back to CPU. The hook closure uses
    ``weakref.ref(self)`` so dropping the binding without deactivate
    is non-fatal: the orphaned hooks remain installed on the model
    but no-op; resident blocks stay on GPU until the model itself is
    dropped. Forward calls through still-resident blocks work; calls
    through previously-evicted blocks find CPU tensors (slow but
    functional).

    Instances are usually created by binding a
    :class:`BlockComponentStore` to a compatible model.

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
        self._param_name_to_block_param = _build_param_name_index(
            self._block_instances,
            name,
            block_indices,
        )
        self._param_names = frozenset(self._param_name_to_block_param)
        self._buffer_names = _block_buffer_names(self._block_instances, name, block_indices)

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
                by_block[block_idx][local_name] = override
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


__all__ = [
    "BlockComponent",
    "BlockComponentStore",
]
