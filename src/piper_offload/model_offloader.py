"""Unified CUDA offload binding with optional adapter application.

Supports whole-model pinned bulk offload or block streaming, with optional
per-parameter adapter application.
"""

import contextlib
import threading
import weakref
from collections.abc import Callable, Sequence
from typing import Self

import torch
from torch import nn

from ._devices import canonical_device
from .adapter import Adapter, AdapterMode, AdapterTargetUpdates
from .block_compile import BlockCompileConfig
from .block_component import BlockComponent
from .block_mode import BlockMode
from .composite_component import CompositeComponent, CompositeComponentStore
from .host_backing import HostBacking
from .lora import install_routed_residual_hook
from .module_names import resolve_parent_leaf
from .parameter_delta import ParameterDeltaTransform
from .parameter_transform import ParameterTransform
from .parameter_value import ParameterValueTransform
from .pinned_component import PinnedComponent

type _ParameterUpdateMap = dict[str, AdapterTargetUpdates]
type _TransientComponent = PinnedComponent | BlockComponent
type _ForwardHook = Callable[
    [nn.Module, tuple[object, ...], object],
    object | None,
]


def _release_after_forward_hook(
    component: _TransientComponent,
    *,
    record_device: torch.device | None = None,
) -> _ForwardHook:
    component_ref = weakref.ref(component)

    def release(
        _module: nn.Module,
        _args: tuple[object, ...],
        _output: object,
    ) -> None:
        component = component_ref()
        if component is not None:
            if record_device is not None:
                assert isinstance(component, PinnedComponent)
                component.record_stream(
                    torch.cuda.current_stream(record_device),
                )
            component.release()

    return release


class ModelRuntimeInUseError(RuntimeError):
    """A model offloader already has an active use."""


__all__ = [
    "ModelOffloader",
    "ModelRuntimeInUseError",
]


class ModelOffloader:
    """Move a whole model or managed block groups between host RAM and
    CUDA, with optional adapter application and trainable-parameter support.

    Construct with :meth:`from_module`. One offloader owns one model and may
    be reused sequentially, but it cannot create model replicas or serve
    overlapping activations. Concurrent use fails immediately with
    :class:`ModelRuntimeInUseError`.

    ``block_mode`` selects resident, whole-block streaming, compiled rolling,
    or automatic rolling-with-streaming-fallback execution for groups named by
    ``block_paths`` and ``transient_block_paths``. Other state remains resident
    unless its module is selected by ``transient_paths``. Supplying
    ``block_compile`` opts declared block forwards into Inductor during CUDA
    inference. CPU activation is pass-through over the host-backed module state
    and remains eager.

    Composes resident and transient :class:`PinnedComponent`\\ s with one or
    more :class:`BlockComponent`\\ s internally. Adapter requests are supplied
    directly to :meth:`activate`; merge mode installs activation-scoped
    post-copy hooks so the merge fires immediately after each CPU->GPU weight
    copy. No separate merge binding is needed.

    Training
    --------
    Training through streamed blocks **requires activation
    checkpointing on each block** — wrap call sites in
    :func:`torch.utils.checkpoint.checkpoint`, or call
    ``model.gradient_checkpointing_enable()`` on a HuggingFace model.
    Without it, ``loss.backward()`` raises ``RuntimeError: ... has
    been modified by an inplace operation`` on the first target reuse.

    Why: autograd saves a reference to each ``Linear``'s weight
    tensor at forward time and records its version counter. Streaming
    is a sequence of in-place ``copy_`` writes into a fixed pool of
    GPU target tensors — every load bumps the target tensor's version,
    invalidating any previously-saved reference into that target.
    Checkpointing makes each block's internal forward run under
    ``no_grad`` (no internal tensors saved); when backward arrives,
    PyTorch re-runs the block's forward with grad enabled, building
    a fresh autograd graph whose saved references only live within
    that one block's recompute-then-backward window. Target reuse
    outside that window is then safe. Ensuring each streamed block that
    participates in training is checkpointed is the caller's
    responsibility — there is no auto-detection or guard.

    By default, trainable params are not streamed through the block
    residency pool. They are managed by :class:`PinnedComponent`, stay
    GPU-resident while the offloader binding is active on CUDA, and must be
    updated inside :meth:`optimizer_step` so CUDA updates are copied
    back to the pinned CPU cache. CPU activation leaves them in the
    host-backed module state.

    Configure ``include_block_trainables=True`` on :meth:`from_module` to
    stream in-block trainable parameter data through the CUDA block target
    pool. In that mode,
    :meth:`optimizer_step` is the optimizer boundary: it materializes
    streamed trainable ``.data`` on GPU while an arbitrary PyTorch
    optimizer updates it, then copies the updated data back to pinned
    CPU. CPU activation makes :meth:`optimizer_step` a guarded no-op.
    Gradients are not streamed; PyTorch owns ``param.grad`` normally.

    Parameters
    ----------
    model:
        The concrete model bound to the supplied composite.
    composite:
        Bound :class:`CompositeComponent` owning the model's pinned
        and block offload components.
    cache_bytes:
        Stable host-cache bytes owned by the bound components.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        composite: CompositeComponent,
        cache_bytes: int,
    ) -> None:
        self._model = model
        self._active_device: torch.device | None = None
        self._composite = composite
        self._cache_bytes = cache_bytes
        self._activation_lock = threading.Lock()
        self._adapter_hook_removers: list[Callable[[], None]] = []
        self._transient_hook_removers: list[Callable[[], None]] = []

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        block_paths: Sequence[str] = (),
        transient_block_paths: Sequence[str] = (),
        include_block_trainables: bool = False,
        block_mode: BlockMode = "streaming",
        block_compile: BlockCompileConfig | None = None,
        host_backing: HostBacking = "pinned",
        transient_paths: Sequence[str] = (),
    ) -> Self:
        """Clone and bind ``model`` as one reusable cached runtime.

        The intermediate component store exists only during construction.
        Bound component instances retain the host state afterward, so the
        model is never rebound on subsequent uses.
        ``block_compile`` applies one forward-only compile policy to every
        block group and is unused when no block group is declared. By default,
        ``block_mode`` selects how every declared block group becomes resident:
        all at once, one whole block ahead, parameter-by-parameter through a
        compiled rolling graph, or rolling with per-group streaming fallback.
        Groups named by ``block_paths`` retain their
        CUDA working sets for the activation. Groups named by
        ``transient_block_paths`` release after their final blocks and
        reacquire after the root model forward. These groups are inference-only
        and must not be traversed again later in the same root forward. Their
        block lists must contain distinct module objects because a module hook
        cannot distinguish occurrences of an aliased block.
        Each module named by ``transient_paths`` similarly owns a separate
        CUDA working set that releases after that module's forward.
        ``host_backing`` defaults to a pinned copy; ``"adopt"`` strictly
        retains frozen state already in CPU RAM and uses direct CUDA copies
        without an application-owned staging pool. It never silently
        materializes an incompatible source, and adoption failures occur
        before binding mutates model registries. Retained source tensors and
        writable mmap contents must remain immutable for the offloader's
        lifetime.
        """
        composite_store = CompositeComponentStore.from_module(
            model,
            block_paths=block_paths,
            transient_block_paths=transient_block_paths,
            transient_paths=transient_paths,
            include_block_trainables=include_block_trainables,
            host_backing=host_backing,
        )
        cache_bytes = composite_store.cache_bytes
        composite = composite_store.bind(
            model,
            block_compile=block_compile,
            block_mode=block_mode,
        )
        return cls(
            model,
            composite=composite,
            cache_bytes=cache_bytes,
        )

    # ------------------------------------------------------------------ API

    @staticmethod
    def _normalize_adapters(
        adapters: Sequence[Adapter],
        *,
        adapter_strengths: Sequence[float] | None = None,
    ) -> list[tuple[Adapter, float]]:
        adapter_list = list(adapters)
        if adapter_strengths is None:
            strength_list = [1.0] * len(adapter_list)
        else:
            strength_list = [float(strength) for strength in adapter_strengths]
        return [
            (adapter, strength)
            for adapter, strength in zip(
                adapter_list,
                strength_list,
                strict=True,
            )
            if strength != 0.0
        ]

    def _require_managed_target(self, target_key: str) -> str:
        """Validate that ``target_key`` names a parameter this offloader
        manages, returning it unchanged.

        Adapter target keys must match the model's own parameter paths
        exactly. Any key remapping (stripping a ``diffusion_model.``
        prefix, inserting a PEFT ``.base_layer.`` segment, …) is the
        caller's responsibility when building the adapter state dict.
        """
        if target_key not in self.param_names:
            sample = sorted(self.param_names)[:3]
            raise ValueError(
                f"Adapter target {target_key!r} is not managed by this "
                "ModelOffloader. Adapter target keys must match the model's "
                f"parameter names exactly. Sample managed keys: {sample} ..."
            )
        return target_key

    def _group_adapter_updates_by_param_name(
        self,
        adapters: Sequence[tuple[Adapter, float]],
    ) -> _ParameterUpdateMap:
        per_param: _ParameterUpdateMap = {}
        for adapter, strength in adapters:
            for target_key, target in adapter.targets.items():
                if adapter.allow_partial_targets and target_key not in self.param_names:
                    continue
                managed = self._require_managed_target(target_key)
                contributions = per_param.setdefault(managed, AdapterTargetUpdates())
                contributions.add(target, strength, target_key=target_key)
        return per_param

    def _register_merge_adapter_hooks(
        self,
        active_device: torch.device,
        updates: _ParameterUpdateMap,
        *,
        stochastic_rounding: bool = True,
    ) -> None:
        if active_device.type != "cuda":
            raise ValueError(
                "ModelOffloader merge mode requires CUDA activation; "
                f"got {active_device}. Use adapter_mode='routed' "
                "for CPU activation."
            )

        params_by_name = dict(self._model.named_parameters(remove_duplicate=False))
        for param_name, contributions in updates.items():
            transform: ParameterTransform
            if contributions.deltas:
                transform = ParameterDeltaTransform(
                    contributions.deltas,
                    stochastic_rounding=stochastic_rounding,
                    target_key=param_name,
                )
            else:
                assert contributions.value is not None
                transform = ParameterValueTransform(contributions.value)

            transform.validate_parameter(params_by_name[param_name])

            remove_hook = self._register_post_copy_hook(
                param_name,
                transform.apply_parameter,
            )
            self._adapter_hook_removers.append(remove_hook)

    def _register_routed_lora_hooks(
        self,
        updates: _ParameterUpdateMap,
    ) -> None:
        """Install one staged PRE/POST routed hook per target Linear.

        The PRE hook copies all LoRA factors for that target from immutable
        pinned backing to the invocation's input device. The POST hook applies
        their additive residual and releases the staged device tensors.
        """
        value_names = sorted(
            param_name for param_name, contributions in updates.items() if contributions.value is not None
        )
        if value_names:
            raise ValueError(
                "Routed LoRA mode does not support parameter values; "
                f"use adapter_mode='merge'. Parameter values: {value_names!r}."
            )

        dense_names = sorted(param_name for param_name, contributions in updates.items() if contributions.has_dense)
        if dense_names:
            raise ValueError(
                "Routed LoRA mode does not support dense parameter deltas; "
                f"use adapter_mode='merge'. Dense parameter deltas: {dense_names!r}."
            )

        for param_name, contributions in updates.items():
            parent, _leaf = resolve_parent_leaf(self._model, param_name)
            if not isinstance(parent, nn.Linear):
                raise ValueError(
                    f"Routed LoRA mode requires nn.Linear targets; "
                    f"target {param_name!r} has parent module of "
                    f"type {type(parent).__name__}. Use mode='merge' "
                    f"for non-Linear targets."
                )
            remove_hook = install_routed_residual_hook(
                parent,
                contributions.factors,
            )
            self._adapter_hook_removers.append(remove_hook)

    def _register_post_copy_hook(
        self,
        param_name: str,
        hook: Callable[[nn.Parameter], None],
    ) -> Callable[[], None]:
        return self._composite.register_post_copy_hook(param_name, hook)

    def register_post_copy_hook(
        self,
        param_name: str,
        hook: Callable[[nn.Parameter], None],
    ) -> Callable[[], None]:
        """Register a post-copy hook and return a callable that removes it."""
        return self._register_post_copy_hook(param_name, hook)

    def register_forward_hook(
        self,
        module_name: str,
        hook: Callable[
            [nn.Module, tuple[object, ...], object],
            object | None,
        ],
    ) -> Callable[[], None]:
        """Register a forward hook on a named module and return its remover.

        ``module_name`` uses the fully-qualified namespace from
        :meth:`torch.nn.Module.named_modules`; an empty name selects the model
        itself. The caller owns the hook lifetime.
        """
        handle = self._model.get_submodule(module_name).register_forward_hook(
            hook,
        )
        return handle.remove

    def _install_transient_hooks(self) -> None:
        active_device = self._active_device
        assert active_device is not None
        assert active_device.type == "cuda"
        components: list[_TransientComponent] = []
        for path, component in self._composite.transient:
            self._transient_hook_removers.append(
                self.register_forward_hook(
                    path,
                    _release_after_forward_hook(
                        component,
                        record_device=active_device,
                    ),
                )
            )
            components.append(component)

        for block_component in self._composite.transient_blocks:
            handle = block_component.blocks[-1].register_forward_hook(
                _release_after_forward_hook(block_component),
            )
            self._transient_hook_removers.append(handle.remove)
            components.append(block_component)

        component_refs = tuple(weakref.ref(component) for component in components)

        def reacquire_after_forward(
            _module: nn.Module,
            _args: tuple[object, ...],
            _output: object,
        ) -> None:
            # A model post-hook inherits the caller's inference-mode context.
            # Reusable targets must remain mutable because later prefetch
            # threads refill them outside that context.
            with torch.inference_mode(False):
                for component_ref in component_refs:
                    component = component_ref()
                    if component is not None:
                        component.acquire()

        self._transient_hook_removers.append(self.register_forward_hook("", reacquire_after_forward))

    def _clear_transient_hooks(self) -> None:
        remove_hooks = self._transient_hook_removers
        self._transient_hook_removers = []
        for remove_hook in reversed(remove_hooks):
            remove_hook()

    def _clear_active_adapter_hooks(self) -> None:
        remove_hooks = self._adapter_hook_removers
        self._adapter_hook_removers = []
        for remove_hook in reversed(remove_hooks):
            remove_hook()

    # ----------------------------------------------- ResourceBinding interface

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def value(self) -> nn.Module:
        return self._model

    @property
    def cache_bytes(self) -> int:
        """Stable host-backing bytes charged to :class:`ResourceCache`."""
        return self._cache_bytes

    @property
    def active_device(self) -> torch.device | None:
        """Currently active device, or ``None`` when inactive."""
        return self._active_device

    @property
    def param_names(self) -> frozenset[str]:
        """Parameter names managed by this offloader."""
        return self._composite.param_names

    @property
    def buffer_names(self) -> frozenset[str]:
        """Buffer names managed by this offloader."""
        return self._composite.buffer_names

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        if device is not None:
            return canonical_device(device)
        raise ValueError(
            "ModelOffloader.activate() requires a device; pass "
            "activate(device) or use this binding through "
            "ModelCache.use(..., device=...)"
        )

    def activate(
        self,
        device: torch.device | str | None = None,
        *,
        adapters: Sequence[Adapter] = (),
        adapter_strengths: Sequence[float] | None = None,
        adapter_mode: AdapterMode = "merge",
        stochastic_rounding: bool = True,
    ) -> None:
        """Make the owned model usable on ``device``.

        ``adapters`` and their optional ``adapter_strengths`` apply only to this
        activation. Exact-zero strengths are inactive and install no hooks.
        Adapters that allow partial targets apply only to parameters managed by
        this offloader; strict adapters reject any absent target.
        ``adapter_mode`` selects in-place merge hooks or routed LoRA residual
        hooks. Routed mode requires factor-only adapters.
        ``stochastic_rounding`` uses stochastic requantization for quantized
        merge targets by default; pass ``False`` for deterministic rounding.
        Parameter values are merge-only and populate frozen floating-point
        meta parameters; routed mode never requantizes. Such a meta target is
        materialized only while its parameter value is active.
        Because the offloader owns one model runtime, a
        second activation before :meth:`deactivate` raises
        :class:`ModelRuntimeInUseError` immediately.
        """
        active_device = self._resolve_device(device)
        if not self._activation_lock.acquire(blocking=False):
            raise ModelRuntimeInUseError(
                "ModelOffloader already has an active use; overlapping model activations are not supported"
            )
        self._active_device = active_device
        try:
            if adapter_mode not in ("merge", "routed"):
                raise ValueError(f"adapter_mode must be 'merge' or 'routed', got {adapter_mode!r}")
            active_adapters = self._normalize_adapters(
                adapters,
                adapter_strengths=adapter_strengths,
            )
            updates = self._group_adapter_updates_by_param_name(active_adapters) if active_adapters else {}
            # Adapter hooks are installed before the composite activates. Merge
            # hooks must be present for the first base-weight copy; routed
            # hooks do no work until a target Linear runs.
            if updates:
                if adapter_mode == "merge":
                    self._register_merge_adapter_hooks(
                        active_device,
                        updates,
                        stochastic_rounding=stochastic_rounding,
                    )
                else:
                    self._register_routed_lora_hooks(updates)
            schedule_transient = active_device.type == "cuda" and (
                bool(self._composite.transient) or bool(self._composite.transient_blocks)
            )
            activation_context = torch.inference_mode(False) if schedule_transient else contextlib.nullcontext()
            # Reusable transient targets must be mutable even if the caller
            # activates the offloader from inside inference mode.
            with activation_context:
                self._composite.activate(
                    active_device,
                    compile_blocks=not (updates and adapter_mode == "routed"),
                )
            if schedule_transient:
                self._install_transient_hooks()
        except BaseException:
            # Deactivation is idempotent over partial component and hook state.
            self.deactivate()
            raise

    def deactivate(self) -> None:
        if self._active_device is None:
            return
        # Scheduling hooks must stop before their components deactivate. Drain
        # asynchronous copies before removing adapter merge hooks. Cleanup and
        # activation-lock release still run if component teardown raises.
        try:
            self._clear_transient_hooks()
        finally:
            try:
                self._composite.deactivate()
            finally:
                try:
                    self._clear_active_adapter_hooks()
                finally:
                    self._active_device = None
                    self._activation_lock.release()

    def optimizer_step(self) -> contextlib.AbstractContextManager[None]:
        """Context manager wrapping the optimizer-step boundary for
        managed trainable weights.

        On CUDA activation, non-streamed trainables are already active
        through :class:`PinnedComponent`, while block-component trainables
        are materialized on enter after force-evicting loaded blocks.
        On exit, updated trainable bytes are copied back to their pinned
        CPU storage. On CPU activation, this is a guarded no-op.

        ``param.grad`` is unaffected throughout. On CUDA, it lives on
        GPU during backward via PyTorch's native ``AccumulateGrad`` and
        is read+modified by the optimizer in place. ``optimizer.zero_grad()``,
        ``clip_grad_norm_``, AMP's ``GradScaler.unscale_`` and other
        grad-walking tools work as in vanilla PyTorch — they don't need
        to be inside this context.

        Typical loop::

            loss.backward()
            with offload.optimizer_step():
                optimizer.step()
            optimizer.zero_grad()

        This context steps on the *GPU* for speed. To run the optimizer on
        *CPU* instead — keeping its state on the host — call
        ``optimizer.step()`` after :meth:`deactivate` without this context. On
        ``deactivate()`` every managed trainable has its ``.data``
        restored to pinned CPU storage *and* its ``.grad`` moved to CPU
        (:class:`PinnedComponent` and :class:`BlockComponent` alike), so
        the step runs on CPU and the in-place update is streamed to GPU on the
        next forward. Keep such trainables in fp32 so the update is a correct
        master-weight update::

            offload.activate("cuda")
            try:
                loss = model(x); loss.backward()
            finally:
                offload.deactivate()
            optimizer.step()        # runs on CPU; states stay on host
            optimizer.zero_grad()
        """
        return self._composite.optimizer_step()

    def gather_for_step(self) -> contextlib.AbstractContextManager[None]:
        """Backward-compatible alias for :meth:`optimizer_step`.

        The public API names the boundary after the operation that
        requires all streamed trainable weight data to be materialized: the
        optimizer step.
        """
        return self.optimizer_step()
