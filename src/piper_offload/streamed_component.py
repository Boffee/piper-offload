"""Block-streaming primitive for memory-efficient training and inference.

A :class:`StreamedComponentStore` manages reusable pinned CPU backing
storage for a single block list whose blocks share the same parameter
layout (names, shapes, dtypes, and any tensor-adapter wrapper metadata).
Binding that store to a compatible model creates a
:class:`StreamedComponent` that streams the resolved blocks to GPU on
demand. Ordinary streaming uses a reusable GPU target pool and background
prefetcher; rolling compilation selects a single-target parameter runtime.
On CPU, the host-backed pinned state is used directly without streaming.

Blocks in one list may be heterogeneously quantized (mixed dtypes or
quant formats on the same-named weights): the GPU target pool keys its
reusable targets by per-block layout signature, so each block streams
into a target matching its own format and no cross-format ``copy_`` ever
happens. Only block lists whose blocks differ in *structure* — different
parameter or buffer *names*, e.g. Flux's two block kinds — must split
into multiple :class:`StreamedComponent` instances composed via
:class:`ModelOffloader`.

In-block trainable params (LoRA adapters) flow through the same target
pool; pinned module instances branch on the source trainable flag to swap
``.data`` (preserves user Parameter identity for autograd / optimizer
state) instead of replacing the Parameter wrapper. Gradients live on GPU
during backward via PyTorch's native ``AccumulateGrad``; only ``.data``
is materialized around ``optimizer.step()`` via :meth:`optimizer_step`.

This is the sharp, low-level primitive. It does NOT manage:

- Non-block parts of the model (parent-module state, sibling
  modules) — caller derives :class:`PinnedComponent` include-name sets
  by excluding the streamer's owned names.
- Out-of-block trainable parameter movement — caller handles that
  alongside non-streamed parameters, usually with :class:`PinnedComponent`.
- Shared storage with tensors outside the streamed block list — caller
  must choose a valid composition; use whole-model
  :class:`PinnedComponent` if sharing must be preserved.
- Activation-checkpointing enforcement — required for in-block
  trainable streaming, but checked at the composer level.

Most users want :class:`ModelOffloader` (the blessed safe API). Reach for
:class:`StreamedComponentStore` / :class:`StreamedComponent` directly only
when you need bespoke composition (e.g., multiple block lists like Flux's
``transformer_blocks`` + ``single_transformer_blocks``).
"""

import contextlib
import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Self, cast

import torch
from torch import nn

from ._devices import canonical_device
from .block_compile import BlockCompileConfig, _BlockCompileState
from .host_backing import (
    HostBacking,
    validate_host_backing,
)
from .module_names import walk_attr_path
from .pinned_module import (
    PinnedModuleInstance,
    PinnedModuleStore,
    PostCopyHook,
)
from .pinned_param import PinnedParam
from .rolling_runtime import create_rolling_runtime
from .stream_config import DEFAULT_STREAM_CONFIG, StreamConfig
from .streaming_runtime import BlockStreamingRuntime, StreamingRuntime


def _stream_config_from_kwargs(kwargs: dict[str, object]) -> StreamConfig:
    """Extract the optional ``stream_config`` activation kwarg.

    :meth:`StreamedComponent.activate` accepts ``**kwargs`` to satisfy the
    open component lifecycle contract; the streamer is the one component
    that consumes a ``stream_config``. Absent (or ``None``) falls back to
    the default policy.
    """
    stream_config = kwargs.get("stream_config")
    if stream_config is None:
        return DEFAULT_STREAM_CONFIG
    if not isinstance(stream_config, StreamConfig):
        raise TypeError(f"stream_config must be a StreamConfig; got {type(stream_config).__name__}.")
    return stream_config


def _release_cuda_cache_on_drop(is_cuda: bool) -> None:
    # Process-wide PyTorch CUDA allocator cache is the only state the
    # refcount-based GC of a streamer can't release on its own. Without
    # this, freed pinned/GPU pages stay held by the allocator until the
    # next allocation pressure event, which manifests as OOMs at
    # workload boundaries (e.g. successive trainers in one process).
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
# Streamed block instances
# ---------------------------------------------------------------------------


def _param_target_layout(p: nn.Parameter) -> tuple[object, object]:
    """Per-parameter target-compatibility layout, computed pre-pin.

    Two params with equal values pool into structurally identical GPU
    targets, so a refill is a plain ``Tensor.copy_``; unequal values must
    never share a target, because ``copy_`` silently casts dtype and
    silently broadcasts compatible shapes, and wrapper metadata (qtype,
    axis, activation_qtype, quant_type) is similarly invisible to it.

    This is the standalone form of the same value
    :attr:`PinnedParam.target_layout` supplies the same value to the block
    runtime's target-pool signature; this helper is not called on the
    streaming path. Kept as the package's documented
    way to compare two params' target compatibility directly (e.g. in
    adapter tests) without pinning either.

    :class:`PinnedParam` owns the tensor-adapter details because wrapper
    metadata is type-specific. The returned value intentionally excludes
    tensor identity so distinct blocks with the same layout share one
    pooled target.
    """
    return PinnedParam.target_layout_for(p)


def _collect_streamed_schemas(
    blocks: list[nn.Module],
    stream_param_names: set[str] | None,
    stream_buffer_names: set[str] | None,
) -> tuple[list[dict[str, bool]], list[set[str]]]:
    """Snapshot the cross-block contract without retaining tensor objects.

    Pinned construction replaces each completed block's source wrappers so
    their pageable/file-backed storage can be released before the next block
    is copied. Keeping ``Parameter`` or buffer objects in this pre-validation
    snapshot would extend every source storage's lifetime until the complete
    block list finished pinning, producing an approximately 2x model-sized
    peak for structured tensor adapters.
    """
    block_param_schemas: list[dict[str, bool]] = []
    block_buffer_schemas: list[set[str]] = []

    for block in blocks:
        params, buffers = _select_streamed_schema(
            block,
            stream_param_names,
            stream_buffer_names,
        )
        block_param_schemas.append(params)
        block_buffer_schemas.append(buffers)

    return block_param_schemas, block_buffer_schemas


def _select_streamed_schema(
    block: nn.Module,
    stream_param_names: set[str] | None,
    stream_buffer_names: set[str] | None,
) -> tuple[dict[str, bool], set[str]]:
    params: dict[str, bool] = {}
    all_param_names: set[str] = set()
    for name, param in block.named_parameters(remove_duplicate=False):
        all_param_names.add(name)
        if stream_param_names is not None and name not in stream_param_names:
            continue
        params[name] = param.requires_grad
    _validate_streamed_names_known(stream_param_names, all_param_names)

    buffers: set[str] = set()
    all_buffer_names: set[str] = set()
    for name, _buffer in block.named_buffers(remove_duplicate=False):
        all_buffer_names.add(name)
        if stream_buffer_names is not None and name not in stream_buffer_names:
            continue
        buffers.add(name)
    _validate_streamed_names_known(stream_buffer_names, all_buffer_names)

    return params, buffers


def _validate_streamed_names_known(
    names: set[str] | None,
    known_names: set[str],
) -> None:
    if names is None:
        return
    missing = sorted(names - known_names)
    if missing:
        raise ValueError(f"StreamedComponent cannot select unknown block-local names: {_format_names(missing)}.")


def _format_names(names: Sequence[str]) -> str:
    return ", ".join(repr(name) for name in names)


def _check_block_requires_grad_consistent(
    block_param_schemas: Sequence[dict[str, bool]],
) -> None:
    """Reject blocks that disagree on ``requires_grad`` for a shared name.

    Per-block tensor *layouts* (shape, dtype, quant format, tying) may
    differ — the morphing target pool keys targets by layout signature.
    But mixing a trainable and a frozen weight under the same selected
    name in one streamed group is unsupported: trainable streaming swaps
    ``.data`` under an activation-checkpointing guard, so a half-trainable
    group has no coherent optimizer-step / checkpointing contract.
    Validated before pinning so a mismatch leaves the model unmutated.
    """
    if len(block_param_schemas) <= 1:
        return
    ref = block_param_schemas[0]
    for i in range(1, len(block_param_schemas)):
        for name, requires_grad in block_param_schemas[i].items():
            ref_requires_grad = ref.get(name)
            if ref_requires_grad is not None and requires_grad != ref_requires_grad:
                raise ValueError(
                    f"Block {i} param {name!r} requires_grad="
                    f"{requires_grad} differs from block 0 "
                    f"(requires_grad={ref_requires_grad}). All blocks "
                    "in a StreamedComponent group must agree on requires_grad "
                    "per parameter; quantization formats, shapes, and dtypes "
                    "may differ, but trainable and frozen weights cannot be "
                    "mixed in one streamed group."
                )


def _pin_block_module_stores(
    blocks: Sequence[nn.Module],
    *,
    stream_param_names: set[str] | None = None,
    stream_buffer_names: set[str] | None = None,
    pin_memory: bool = True,
) -> list[PinnedModuleStore]:
    """Collect, validate, and pin one :class:`PinnedModuleStore` per block.

    Pre-pin validation failures do not pin and do not mutate module
    parameters or buffers. Once pinning starts, :class:`PinnedParam` may use its
    low-peak ``Parameter.data`` repointing optimization; recovery from
    a pin-time failure is unsupported, matching :class:`PinnedComponent`.
    """
    # Walk each block to snapshot selected names and requires_grad flags
    # WITHOUT retaining the Parameter/buffer objects or pinning anything.
    # Cross-block name consistency is enforced upstream
    # (``_streamed_param_names_for_blocks`` / ``_streamed_buffer_names_for_blocks``);
    # per-block tensor *layouts* may differ (heterogeneous quantization),
    # since the streamer's morphing target pool keys reusable GPU targets
    # by per-block layout signature so blocks of different formats never
    # share a target.
    block_param_schemas, block_buffer_schemas = _collect_streamed_schemas(
        list(blocks),
        stream_param_names,
        stream_buffer_names,
    )
    _check_block_requires_grad_consistent(block_param_schemas)

    # Only lightweight name collections cross into pinning. Pin and install
    # each block before resolving the next block's live tensors so structured
    # source wrappers from completed blocks can be reclaimed immediately.
    param_names_by_block = [set(schema) for schema in block_param_schemas]
    buffer_names_by_block = block_buffer_schemas
    del block_param_schemas, block_buffer_schemas

    stores: list[PinnedModuleStore] = []
    for block, param_names, buffer_names in zip(
        blocks,
        param_names_by_block,
        buffer_names_by_block,
        strict=True,
    ):
        stores.append(
            PinnedModuleStore.from_module(
                block,
                include_param_names=param_names,
                include_buffer_names=buffer_names,
                pin_memory=pin_memory,
                install_backing=pin_memory,
            )
        )
    return stores


def _build_param_name_index(
    instances: Sequence[PinnedModuleInstance],
    prefix: str | None,
    block_indices: Sequence[int],
) -> dict[str, tuple[int, str]]:
    # The external NAME uses the true block index so a sparse group's params
    # are addressed at their real path; the stored VALUE keeps the compact
    # position used to index ``_block_instances`` for the streaming engine.
    index: dict[str, tuple[int, str]] = {}
    for compact_idx, instance in enumerate(instances):
        true_idx = block_indices[compact_idx]
        for local_name in instance.params:
            name = _streamed_param_name(prefix, true_idx, local_name)
            if name in index:
                raise ValueError(f"duplicate streamed parameter name {name!r}")
            index[name] = (compact_idx, local_name)
    return index


def _build_buffer_name_index(
    instances: Sequence[PinnedModuleInstance],
    prefix: str | None,
    block_indices: Sequence[int],
) -> dict[str, tuple[int, str]]:
    index: dict[str, tuple[int, str]] = {}
    for compact_idx, instance in enumerate(instances):
        true_idx = block_indices[compact_idx]
        for local_name in instance.buffers:
            name = _streamed_param_name(prefix, true_idx, local_name)
            if name in index:
                raise ValueError(f"duplicate streamed buffer name {name!r}")
            index[name] = (compact_idx, local_name)
    return index


def _streamed_param_name(
    prefix: str | None,
    block_idx: int,
    local_name: str,
) -> str:
    name = f"{block_idx}.{local_name}"
    return name if prefix is None else f"{prefix}.{name}"


def _streamed_log_label(name: str | None, block_count: int) -> str:
    if name is None:
        return f"StreamedComponent({block_count} blocks)"
    return f"StreamedComponent[{name}]"


def _resolve_blocks(module: nn.Module, blocks_path: str) -> list[nn.Module]:
    obj = walk_attr_path(module, blocks_path)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{blocks_path}', got {type(obj).__name__}")
    blocks = list(cast(nn.ModuleList, obj))
    if not blocks:
        raise ValueError(f"blocks_attr = {blocks_path!r} resolved to empty list")
    return blocks


def _streamed_param_names_for_blocks(
    blocks: Sequence[nn.Module],
    *,
    stream_trainables: bool,
) -> set[str]:
    param_names = _block_param_names(blocks[0], stream_trainables=stream_trainables)
    for i, block in enumerate(blocks[1:], start=1):
        if _block_param_names(block, stream_trainables=stream_trainables) != param_names:
            raise ValueError(
                f"Block {i} selected parameter names differ from block 0. "
                "All blocks in a StreamedComponent group must select the "
                "same parameter names (their shapes, dtypes, and quant "
                "formats may differ). Split structurally different block "
                "kinds across separate `blocks_attr=[...]` groups."
            )
    return param_names


def _streamed_buffer_names_for_blocks(blocks: Sequence[nn.Module]) -> set[str]:
    buffer_names = _block_buffer_names(blocks[0])
    for i, block in enumerate(blocks[1:], start=1):
        if _block_buffer_names(block) != buffer_names:
            raise ValueError(
                f"Block {i} selected buffer names differ from block 0. "
                "All blocks in a StreamedComponent group must select the "
                "same buffer names (their shapes, dtypes, and layouts may "
                "differ). Split structurally different block kinds across "
                "separate `blocks_attr=[...]` groups."
            )
    return buffer_names


def _block_param_names(
    block: nn.Module,
    *,
    stream_trainables: bool,
) -> set[str]:
    return {
        name
        for name, param in block.named_parameters(remove_duplicate=False)
        if stream_trainables or not param.requires_grad
    }


def _block_buffer_names(block: nn.Module) -> set[str]:
    return {name for name, _buffer in block.named_buffers(remove_duplicate=False)}


def _block_is_empty(block: nn.Module) -> bool:
    """A block with no parameters or buffers at any depth.

    Empty positions carry nothing to stream and are skipped by
    :meth:`StreamedComponentStore.from_module` while later blocks retain their
    true externally-visible indices.
    """
    return next(block.parameters(), None) is None and next(block.buffers(), None) is None


@dataclass(frozen=True, slots=True)
class StreamedComponentStore:
    """Reusable pinned backing storage for a streamed block group.

    Built via :meth:`from_module`: the streamed tensor source IS the
    model's block list, resolved by ``blocks_path``. Each block's forward-pre
    hook triggers the load of its own streamed instance.

    ``block_indices`` records which positions of the resolved block list this
    group actually occupies. :meth:`from_module` skips structurally-empty
    positions (no parameters or buffers) while keeping each retained block at
    its true externally-visible index.
    """

    _block_stores: tuple[PinnedModuleStore, ...]
    blocks_path: str
    block_indices: tuple[int, ...]

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        blocks_path: str,
        stream_trainable_weights: bool = False,
        host_backing: HostBacking = "pinned",
    ) -> Self:
        """Resolve ``blocks_path`` on ``model`` and pin its streamed blocks.

        Structurally-empty positions (no parameters or buffers) are skipped and
        dropped from :attr:`block_indices`. The surviving blocks must still
        agree on selected names (see
        :func:`_streamed_param_names_for_blocks`).
        """
        backing = validate_host_backing(host_backing)
        all_blocks = _resolve_blocks(model, blocks_path)
        kept = [(idx, block) for idx, block in enumerate(all_blocks) if not _block_is_empty(block)]
        if not kept:
            raise ValueError(
                f"blocks_attr = {blocks_path!r} has no streamable blocks (every block is structurally empty)."
            )
        block_indices = tuple(idx for idx, _ in kept)
        blocks = [block for _, block in kept]
        stream_param_names = _streamed_param_names_for_blocks(
            blocks,
            stream_trainables=stream_trainable_weights,
        )
        stream_buffer_names = _streamed_buffer_names_for_blocks(blocks)
        block_stores = _pin_block_module_stores(
            blocks,
            stream_param_names=stream_param_names,
            stream_buffer_names=stream_buffer_names,
            pin_memory=backing == "pinned",
        )
        return cls(
            _block_stores=tuple(block_stores),
            blocks_path=blocks_path,
            block_indices=block_indices,
        )

    @property
    def streamed_param_names_by_block(self) -> list[list[str]]:
        """Per-block streamed parameter names."""
        return [list(store.params) for store in self._block_stores]

    @property
    def streamed_buffer_names_by_block(self) -> list[list[str]]:
        """Per-block streamed buffer names."""
        return [list(store.buffers) for store in self._block_stores]

    @property
    def param_names(self) -> frozenset[str]:
        """Externally addressable streamed parameter names.

        Named by TRUE block index (:attr:`block_indices`), not the compact
        store position — so a sparse group's factors are advertised at their
        real path (``blocks.2...``) and the pinned-remainder subtraction in
        :meth:`CompositeComponentStore.from_module` lines up correctly.
        """
        names = {
            _streamed_param_name(self.blocks_path, true_idx, local_name)
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
        """Externally addressable streamed buffer names (true block indices)."""
        names = {
            _streamed_param_name(self.blocks_path, true_idx, local_name)
            for true_idx, store in zip(
                self.block_indices,
                self._block_stores,
                strict=True,
            )
            for local_name in store.buffers
        }
        return frozenset(names)

    @property
    def cache_bytes(self) -> int:
        return sum(store.cache_bytes for store in self._block_stores)

    @property
    def has_trainables(self) -> bool:
        return any(store.has_trainables for store in self._block_stores)

    def resolve_blocks(self, model: nn.Module) -> list[nn.Module]:
        """Resolve this store's blocks path on ``model``."""
        return _resolve_blocks(model, self.blocks_path)

    def bind(
        self,
        model: nn.Module,
        *,
        block_compile: BlockCompileConfig | None = None,
    ) -> StreamedComponent:
        """Bind this store's per-block backing bytes to ``model``.

        Each instance owns and is installed onto its model block; loads repoint
        the instance's own module onto the filled target, and that block's
        forward triggers its streaming. ``model`` must have a block at every
        occupied :attr:`block_indices` position. Non-empty blocks outside those
        positions are rejected because they would be silently unmanaged.
        """
        last = self.block_indices[-1]
        bind_blocks = self.resolve_blocks(model)
        if last >= len(bind_blocks):
            raise ValueError(
                f"StreamedComponentStore.bind() bind model has too few blocks "
                f"at {self.blocks_path!r}: needs index {last}, found "
                f"{len(bind_blocks)}."
            )
        occupied = set(self.block_indices)
        unmanaged = [pos for pos, block in enumerate(bind_blocks) if pos not in occupied and not _block_is_empty(block)]
        if unmanaged:
            raise ValueError(
                f"StreamedComponentStore.bind() bind model has non-empty "
                f"block(s) {unmanaged} at {self.blocks_path!r} that this store "
                f"does not occupy (it occupies {list(self.block_indices)}); "
                "those blocks would never be moved or streamed. Bind a model "
                "whose only non-empty blocks are the occupied ones."
            )
        instances = [
            store.bind(bind_blocks[idx]) for store, idx in zip(self._block_stores, self.block_indices, strict=True)
        ]
        return StreamedComponent(
            instances,
            name=self.blocks_path,
            block_indices=self.block_indices,
            block_compile=block_compile,
        )


# ---------------------------------------------------------------------------
# StreamedComponent — public block-streaming primitive
# ---------------------------------------------------------------------------


class StreamedComponent:
    """Streams a single block list between pinned CPU and CUDA.

    The sharp, low-level streaming primitive. Manages bound block
    instances whose owned params and buffers are pinned to CPU and
    streams them to CUDA via forward-pre hooks on :meth:`activate`.
    CPU activation is pass-through over that pinned host-backed state:
    no pool, no hooks, no copies. Frozen params use parameter
    replacement; trainable params keep Parameter identity and swap only
    ``.data``.
    Does not touch parent modules, sibling modules, or out-of-block
    trainable parameters — those are the composer's responsibility.

    A :class:`StreamedComponent` is a *component* meant to be composed
    inside a :class:`~piper_offload.model_offloader.ModelOffloader`.
    It deliberately is NOT a top-level model runtime (its
    :meth:`activate` returns ``None`` because it doesn't own the
    model). For top-level use, build a model runtime via
    :class:`~piper_offload.model_offloader.ModelOffloader`.

    Lifecycle is uniform with :class:`PinnedComponent`: store construction
    pins (so ``cache_bytes`` is final at construction time, ready
    for :class:`~piper_offload.resource_cache.ResourceCache` admission), and
    ``activate`` brings to CUDA or marks CPU active, ``deactivate`` returns state to
    pinned CPU, removes hooks, and restores eager forwards. Optional
    ``block_compile`` policy belongs to this bound runtime and installs lazy
    compiled forwards only for eligible CUDA inference activations. The
    residency/prefetch policy
    (``num_resident_blocks``, ``num_prefetch_blocks``, ``cyclic``) is supplied
    per activation via :class:`~piper_offload.stream_config.StreamConfig` passed
    to :meth:`activate` — a runtime concern, not part of the pinned backing.
    There is no ``close()``; pinned memory in module state is freed when the
    caller drops the binding and model references.

    **Calling deactivate() before dropping the binding is preferred**
    — it removes the forward hooks (cleaner model state) and reverts
    GPU-resident blocks back to pinned CPU. The hook closure uses
    ``weakref.ref(self)`` so dropping the binding without deactivate
    is non-fatal: the orphaned hooks remain installed on the model
    but no-op; resident blocks stay on GPU until the model itself is
    dropped. Forward calls through still-resident blocks work; calls
    through previously-evicted blocks find pinned-CPU tensors (slow but
    functional).

    Instances are usually created by binding a
    :class:`StreamedComponentStore` to a compatible model.

    Parameters
    ----------
    block_instances:
        The concrete bound block instances.
    name:
        Optional model path for the streamed block list. When set,
        :attr:`param_names` and name-based post-copy hook registration
        use names like ``"blocks.3.weight"``. When omitted, standalone
        streamers use ``"3.weight"``.
        Trainable streaming requires activation checkpointing on every
        block (the ``.data`` swap bypasses autograd's version-counter
        check). The streamer doesn't enforce that precondition itself —
        :class:`ModelOffloader` does.
    block_indices:
        True block index for each instance, used to NAME its params/buffers
        (so a sparse group addresses ``"blocks.2.weight"`` not the compact
        ``"blocks.1.weight"``). The streaming engine still uses the compact
        ``0..k-1`` position internally. Defaults to ``0..k-1`` (contiguous).
    block_compile:
        Optional forward-only compile policy. One lazy callable is retained per
        distinct block module object, installed during eligible CUDA
        activations, and removed on deactivate. CPU activation stays eager.
    """

    def __init__(
        self,
        block_instances: Sequence[PinnedModuleInstance],
        *,
        name: str | None = None,
        block_indices: Sequence[int] | None = None,
        block_compile: BlockCompileConfig | None = None,
    ) -> None:
        self._block_instances = list(block_instances)
        if block_indices is None:
            block_indices = range(len(self._block_instances))
        block_indices = list(block_indices)
        if len(block_indices) != len(self._block_instances):
            raise ValueError(
                "block_indices must have one index per streamed block: "
                f"got {len(block_indices)} for "
                f"{len(self._block_instances)} blocks."
            )
        self._blocks = [instance.module for instance in self._block_instances]
        self._log_label = _streamed_log_label(name, len(self._block_instances))
        self._block_runtime = BlockStreamingRuntime(
            self._block_instances,
            log_label=self._log_label,
        )
        self._rolling_runtime = create_rolling_runtime(
            self._block_instances,
            block_compile,
            log_label=self._log_label,
        )
        compile_runtime: StreamingRuntime = self._rolling_runtime or self._block_runtime
        self._block_compile = _BlockCompileState.create(
            self._blocks,
            block_compile,
            backend=compile_runtime.compile_backend,
        )
        self._active_device: torch.device | None = None
        self._active_runtime: StreamingRuntime | None = None
        self._param_name_to_block_param = _build_param_name_index(
            self._block_instances,
            name,
            block_indices,
        )
        self._buffer_name_to_block_buffer = _build_buffer_name_index(
            self._block_instances,
            name,
            block_indices,
        )
        self._param_names = frozenset(self._param_name_to_block_param)
        self._buffer_names = frozenset(self._buffer_name_to_block_buffer)
        self._cpu_optimizer_step_active = False

        # Auto-flush the CUDA allocator cache when the streamer is GC'd,
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
        """The bound modules whose forward drives streaming, in order."""
        return tuple(self._blocks)

    @property
    def block_compile(self) -> BlockCompileConfig | None:
        """This bound streamer's optional block compilation policy."""
        return self._block_compile.config

    @property
    def streamed_param_names_by_block(self) -> list[list[str]]:
        """Per-block streamed parameter names."""
        return [list(instance.params) for instance in self._block_instances]

    @property
    def streamed_buffer_names_by_block(self) -> list[list[str]]:
        """Per-block streamed buffer names."""
        return [list(instance.buffers) for instance in self._block_instances]

    @property
    def param_names(self) -> frozenset[str]:
        """Externally addressable streamed parameter names."""
        return self._param_names

    @property
    def buffer_names(self) -> frozenset[str]:
        """Externally addressable streamed buffer names."""
        return self._buffer_names

    @property
    def has_trainables(self) -> bool:
        return any(instance.has_trainables for instance in self._block_instances)

    def register_post_copy_hook(
        self,
        name: str,
        hook: PostCopyHook,
    ) -> Callable[[], None]:
        """Register a hook after this component copies ``name`` to GPU.

        Package-internal: used by :class:`ModelOffloader` for merge-mode
        LoRA. Returns a callable that unregisters the hook.
        """
        instance, name = self._resolve_param_name(name)
        return instance.register_post_copy_hook(name, hook)

    def _resolve_param_name(
        self,
        name: str,
    ) -> tuple[PinnedModuleInstance, str]:
        ref = self._param_name_to_block_param.get(name)
        if ref is None:
            raise ValueError(f"param name {name!r} is not owned by this streamer")
        block_idx, local_name = ref
        return self._resolve_block_param(block_idx, local_name)

    def _resolve_buffer_name(
        self,
        name: str,
    ) -> tuple[PinnedModuleInstance, str]:
        ref = self._buffer_name_to_block_buffer.get(name)
        if ref is None:
            raise ValueError(f"buffer name {name!r} is not owned by this streamer")
        block_idx, local_name = ref
        return self._resolve_block_buffer(block_idx, local_name)

    def _resolve_block_param(
        self,
        block_idx: int,
        name: str,
    ) -> tuple[PinnedModuleInstance, str]:
        if block_idx < 0 or block_idx >= len(self._block_instances):
            raise ValueError(f"streamed block index {block_idx} is out of range")
        instance = self._block_instances[block_idx]
        if name not in instance.params:
            raise ValueError(f"param name {name!r} is not owned by streamed block {block_idx}")
        return instance, name

    def _resolve_block_buffer(
        self,
        block_idx: int,
        name: str,
    ) -> tuple[PinnedModuleInstance, str]:
        if block_idx < 0 or block_idx >= len(self._block_instances):
            raise ValueError(f"streamed block index {block_idx} is out of range")
        instance = self._block_instances[block_idx]
        if name not in instance.buffers:
            raise ValueError(f"buffer name {name!r} is not owned by streamed block {block_idx}")
        return instance, name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self, device: torch.device, **kwargs: object) -> None:
        """Activate the block list on ``device``.

        CUDA activation selects either the resident-pool block runtime or the
        single-target rolling runtime, then installs optional compiled block
        forwards. CPU activation is pass-through over pinned host-backed state.
        The composite's :meth:`activate` returns the model — this
        method returns ``None`` because the streamer doesn't own one.

        **Lifecycle is caller's responsibility.** Calling activate()
        twice without an intervening deactivate() raises before hooks or
        block pools are installed.

        **Activation failure semantics:** if activation fails midway,
        the streamer is left in an undefined partial state. Retrying
        activation on that streamer is unsupported; the caller's only
        supported cleanup path is :meth:`deactivate`, which idempotently
        tears down whatever was allocated."""
        # Hard-guard against the documented "don't activate twice"
        # case. Without this, a double-activate would double-install
        # forward-pre hooks (silent grad doubling) and stack a second
        # target pool on top of an active one.
        if self._active_device is not None:
            raise RuntimeError(
                "StreamedComponent.activate() called while already "
                "active. Deactivate first, or check for a leaked "
                "context manager."
            )
        active_device = canonical_device(device)
        if active_device.type == "cpu":
            self._activate_cpu_resolved()
            return
        if active_device.type != "cuda":
            raise ValueError(f"StreamedComponent.activate() supports CUDA or CPU; got {active_device}.")
        active_block_compile = cast(
            BlockCompileConfig | None,
            kwargs.get("block_compile", self._block_compile.config),
        )
        self._activate_cuda_resolved(
            active_device,
            _stream_config_from_kwargs(kwargs),
            block_compile=active_block_compile,
        )

    def _activate_cpu_resolved(self) -> None:
        self._active_device = torch.device("cpu")

    def _activate_cuda_resolved(
        self,
        active_device: torch.device,
        stream_config: StreamConfig,
        *,
        block_compile: BlockCompileConfig | None,
    ) -> None:
        if block_compile is not None and block_compile.rolling != (self._rolling_runtime is not None):
            raise ValueError(
                "the activation block_compile policy cannot change rolling "
                "mode from the policy used to construct the streamer"
            )
        runtime: StreamingRuntime
        if block_compile is not None and block_compile.rolling:
            assert self._rolling_runtime is not None
            runtime = self._rolling_runtime
        else:
            runtime = self._block_runtime

        # Record the selected runtime before activation so deactivate() can
        # clean up a partially-created pool, stream, or hook set if activation
        # raises midway through its lifecycle.
        self._active_device = active_device
        self._active_runtime = runtime
        runtime.activate(active_device, stream_config)
        self._block_compile.install(block_compile)

    def deactivate(self) -> None:
        """Tear down active resources idempotently — safe to call
        before activate or multiple times. The selected runtime owns cleanup
        of any partial activation state. Drop the binding reference after
        deactivate to release pinned memory."""
        self._block_compile.restore()
        if self._active_device == torch.device("cpu"):
            self._active_device = None
            return

        runtime = self._active_runtime
        try:
            if runtime is not None:
                runtime.deactivate()
        finally:
            self._active_runtime = None
            self._active_device = None

    @contextlib.contextmanager
    def use(
        self,
        device: torch.device | str,
        **kwargs: object,
    ) -> Iterator[None]:
        """Activate on ``device`` for the duration of the context."""
        self.activate(canonical_device(device), **kwargs)
        try:
            yield
        finally:
            self.deactivate()

    # ------------------------------------------------------------------
    # Optimizer-step boundary
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def optimizer_step(self) -> Iterator[None]:
        """Materialize streamed trainables around an optimizer step."""
        if self._active_device == torch.device("cpu"):
            if self._cpu_optimizer_step_active:
                raise RuntimeError("StreamedComponent.optimizer_step() does not support reentrant entry.")
            self._cpu_optimizer_step_active = True
            try:
                yield
            finally:
                self._cpu_optimizer_step_active = False
            return

        runtime = self._active_runtime
        if runtime is None:
            raise RuntimeError(
                "StreamedComponent.optimizer_step() called on inactive "
                "streamer. Use it inside the offloader's context "
                "manager, between backward and the next forward."
            )
        with runtime.optimizer_step():
            yield

    @contextlib.contextmanager
    def gather_for_step(self) -> Iterator[None]:
        """Backward-compatible alias for :meth:`optimizer_step`."""
        with self.optimizer_step():
            yield


__all__ = [
    "StreamedComponent",
    "StreamedComponentStore",
]
