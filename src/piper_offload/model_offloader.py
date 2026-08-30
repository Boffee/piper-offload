"""Unified CUDA offload binding with optional LoRA application.

Supports whole-model pinned bulk offload or block streaming, with optional
per-weight LoRA application in both modes.
"""

import contextlib
import threading
import weakref
from collections.abc import Callable, Iterator, Sequence
from typing import Self

import torch
from torch import nn

from ._devices import canonical_device
from .block_compile import BlockCompileConfig
from .composite_component import CompositeComponent, CompositeComponentStore
from .host_backing import HostBacking, validate_host_backing
from .lora import (
    LoRA,
    LoRAMode,
    LoRATransform,
    ScaledLoRAFactor,
    install_routed_residual_hook,
)
from .module_names import resolve_parent_leaf, sibling_parameter_name
from .streamed_component import StreamedComponent

type _LoraParamMap = dict[str, list[ScaledLoRAFactor]]
type _ForwardHook = Callable[
    [nn.Module, tuple[object, ...], object],
    object | None,
]


def _release_after_forward_hook(
    streamer: StreamedComponent,
) -> _ForwardHook:
    streamer_ref = weakref.ref(streamer)

    def release(
        _module: nn.Module,
        _args: tuple[object, ...],
        _output: object,
    ) -> None:
        streamer = streamer_ref()
        if streamer is not None:
            streamer.release()

    return release


class ModelRuntimeInUseError(RuntimeError):
    """A model offloader already has an active use."""


__all__ = [
    "ModelOffloader",
    "ModelRuntimeInUseError",
]


class ModelOffloader:
    """Move a whole model or streamed block groups between host RAM and
    CUDA, with optional LoRA merge and trainable-parameter support.

    Construct with :meth:`from_module`. One offloader owns one model and may
    be reused sequentially, but it cannot create model replicas or serve
    overlapping activations. Concurrent use fails immediately with
    :class:`ModelRuntimeInUseError`.

    When ``block_paths`` is omitted, CUDA activation bulk-copies every
    managed parameter and buffer to CUDA. When it is set, CUDA activation
    streams the selected block groups while non-streamed state remains resident
    for the activation.
    Supplying ``block_compile`` at construction opts streamed block forwards
    into Inductor during CUDA inference. CPU activation is pass-through over
    the host-backed module state and remains eager.

    Composes a resident :class:`PinnedComponent`
    with one or more :class:`StreamedComponent`\\ s internally. LoRA requests
    are supplied directly to :meth:`activate`; merge mode installs
    activation-scoped post-copy hooks so the merge fires immediately after
    each CPU->GPU weight copy. No separate merge binding is needed.

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

    Configure ``stream_trainable_weights=True`` on :meth:`from_module` to
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
        and streamed offload components.
    cache_bytes:
        Stable host-cache bytes owned by the bound components.
    transient_streaming:
        Release each streamed CUDA pool after its final block and reacquire all
        pools after the root model forward. This opt-in policy is intended for
        inference and leaves CPU activation unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        composite: CompositeComponent,
        cache_bytes: int,
        transient_streaming: bool = False,
    ) -> None:
        self._model = model
        self._active_device: torch.device | None = None
        self._composite = composite
        self._cache_bytes = cache_bytes
        self._transient_streaming = transient_streaming
        self._activation_lock = threading.Lock()
        self._lora_hook_removers: list[Callable[[], None]] = []
        self._transient_hook_removers: list[Callable[[], None]] = []

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        block_paths: Sequence[str] = (),
        stream_trainable_weights: bool = False,
        block_compile: BlockCompileConfig | None = None,
        host_backing: HostBacking = "pinned",
        transient_streaming: bool = False,
    ) -> Self:
        """Clone and bind ``model`` as one reusable cached runtime.

        The intermediate component store exists only during construction.
        Bound component instances retain the host state afterward, so the
        model is never rebound on subsequent uses.
        ``block_compile`` applies one forward-only compile policy to every
        ``block_paths`` group and is unused when no block group is declared.
        ``transient_streaming=True`` scopes each streamed CUDA pool from
        activation or the preceding model completion through its final block;
        the root model post-forward hook reacquires released pools for the next
        call.
        ``host_backing`` defaults to a pinned copy; ``"adopt"`` strictly
        retains frozen state already in CPU RAM and uses direct CUDA copies
        without an application-owned staging pool. It never silently
        materializes an incompatible source, and adoption failures occur
        before binding mutates model registries. Retained source tensors and
        writable mmap contents must remain immutable for the offloader's
        lifetime.
        """
        backing = validate_host_backing(host_backing)
        if backing == "adopt":
            trainable_names = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
            if trainable_names:
                raise ValueError(
                    "adopted host backing is inference-only; freeze all "
                    "parameters before constructing the offloader. Trainable "
                    f"parameters: {trainable_names!r}."
                )
        composite_store = CompositeComponentStore.from_module(
            model,
            block_paths=block_paths,
            stream_trainable_weights=stream_trainable_weights,
            host_backing=backing,
        )
        cache_bytes = composite_store.cache_bytes
        composite = composite_store.bind(model, block_compile=block_compile)
        return cls(
            model,
            composite=composite,
            cache_bytes=cache_bytes,
            transient_streaming=transient_streaming,
        )

    # ------------------------------------------------------------------ API

    @staticmethod
    def _normalize_loras(
        loras: Sequence[LoRA],
        *,
        lora_strengths: Sequence[float] | None = None,
    ) -> list[tuple[LoRA, float]]:
        lora_list = list(loras)
        if lora_strengths is None:
            strength_list = [1.0] * len(lora_list)
        else:
            strength_list = [float(strength) for strength in lora_strengths]
        return [
            (lora, strength)
            for lora, strength in zip(lora_list, strength_list, strict=True)
            if strength != 0.0
        ]

    def _require_managed_target(self, target_key: str) -> str:
        """Validate that ``target_key`` names a parameter this offloader
        manages, returning it unchanged.

        LoRA target keys must match the model's own parameter paths
        exactly. Any key remapping (stripping a ``diffusion_model.``
        prefix, inserting a PEFT ``.base_layer.`` segment, …) is the
        caller's responsibility when building the LoRA state dict.
        """
        if target_key not in self.param_names:
            sample = sorted(self.param_names)[:3]
            raise ValueError(
                f"LoRA target {target_key!r} is not managed by this "
                "ModelOffloader. LoRA target keys must match the model's "
                f"parameter names exactly. Sample managed keys: {sample} ..."
            )
        return target_key

    def _group_lora_factors_by_param_name(
        self,
        loras: Sequence[tuple[LoRA, float]],
    ) -> _LoraParamMap:
        per_param: _LoraParamMap = {}
        for lora, strength in loras:
            for target_key, factor in lora.targets.items():
                managed = self._require_managed_target(target_key)
                per_param.setdefault(managed, []).append(factor.scaled(strength))
        return per_param

    def _register_merge_lora_hooks(
        self,
        active_device: torch.device,
        targets: _LoraParamMap,
        *,
        stochastic_rounding: bool = True,
    ) -> None:
        if active_device.type != "cuda":
            raise ValueError(
                "ModelOffloader merge mode requires CUDA activation; "
                f"got {active_device}. Use lora_mode='routed' "
                "for CPU activation."
            )

        params_by_name = dict(
            self._model.named_parameters(remove_duplicate=False)
        )
        for param_name, factors in targets.items():
            transform = LoRATransform(
                factors,
                stochastic_rounding=stochastic_rounding,
                target_key=param_name,
            )
            bias_name: str | None = None
            bias: nn.Parameter | None = None
            if transform.has_bias:
                bias_name = self._require_managed_target(
                    sibling_parameter_name(param_name, "bias"),
                )
                bias = params_by_name[bias_name]

            transform.validate_target(params_by_name[param_name], bias)

            remove_hook = self._register_post_copy_hook(
                param_name,
                transform.apply_weight,
            )
            self._lora_hook_removers.append(remove_hook)
            if bias_name is not None:
                remove_bias_hook = self._register_post_copy_hook(
                    bias_name,
                    transform.apply_bias,
                )
                self._lora_hook_removers.append(remove_bias_hook)

    def _register_routed_lora_hooks(
        self,
        targets: _LoraParamMap,
    ) -> None:
        """Install one staged PRE/POST routed hook per target Linear.

        The PRE hook copies all LoRA factors for that target from immutable
        pinned backing to the invocation's input device. The POST hook applies
        their additive residual and releases the staged device tensors.
        """
        for param_name, factors in targets.items():
            parent, _leaf = resolve_parent_leaf(self._model, param_name)
            if not isinstance(parent, nn.Linear):
                raise ValueError(
                    f"Routed LoRA mode requires nn.Linear targets; "
                    f"target {param_name!r} has parent module of "
                    f"type {type(parent).__name__}. Use mode='merge' "
                    f"for non-Linear targets."
                )
            remove_hook = install_routed_residual_hook(parent, factors)
            self._lora_hook_removers.append(remove_hook)

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

    def _install_transient_streaming_hooks(self) -> None:
        streamers = self._composite.streamed
        for streamer in streamers:
            handle = streamer.blocks[-1].register_forward_hook(
                _release_after_forward_hook(streamer),
            )
            self._transient_hook_removers.append(handle.remove)

        offloader_ref = weakref.ref(self)

        def reacquire_after_forward(
            _module: nn.Module,
            _args: tuple[object, ...],
            _output: object,
        ) -> None:
            offloader = offloader_ref()
            if offloader is not None:
                # A model post-hook inherits the caller's inference-mode
                # context. Reusable targets must remain mutable because later
                # prefetch threads refill them outside that context.
                with torch.inference_mode(False):
                    for streamer in offloader._composite.streamed:
                        streamer.acquire()

        self._transient_hook_removers.append(
            self.register_forward_hook("", reacquire_after_forward)
        )

    def _clear_transient_streaming_hooks(self) -> None:
        remove_hooks = self._transient_hook_removers
        self._transient_hook_removers = []
        for remove_hook in reversed(remove_hooks):
            remove_hook()

    def _clear_active_lora_hooks(self) -> None:
        remove_hooks = self._lora_hook_removers
        self._lora_hook_removers = []
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
    def transient_streaming(self) -> bool:
        """Whether streamed CUDA pools follow model-forward lifetimes."""
        return self._transient_streaming

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
        loras: Sequence[LoRA] = (),
        lora_strengths: Sequence[float] | None = None,
        lora_mode: LoRAMode = "merge",
        stochastic_rounding: bool = True,
    ) -> None:
        """Make the owned model usable on ``device``.

        ``loras`` and their optional ``lora_strengths`` apply only to this
        activation. Exact-zero strengths are inactive and install no hooks.
        ``lora_mode`` selects in-place merge hooks or routed residual hooks.
        ``stochastic_rounding`` uses stochastic requantization for quantized
        merge targets by default; pass ``False`` for deterministic rounding.
        Dense targets always use their ordinary exact ``addmm_``, and routed
        mode never requantizes. Because the offloader owns one model runtime, a
        second activation before :meth:`deactivate` raises
        :class:`ModelRuntimeInUseError` immediately.
        """
        active_device = self._resolve_device(device)
        if not self._activation_lock.acquire(blocking=False):
            raise ModelRuntimeInUseError(
                "ModelOffloader already has an active use; overlapping model "
                "activations are not supported"
            )
        self._active_device = active_device
        try:
            if lora_mode not in ("merge", "routed"):
                raise ValueError(
                    "lora_mode must be 'merge' or 'routed', "
                    f"got {lora_mode!r}"
                )
            active_loras = self._normalize_loras(
                loras,
                lora_strengths=lora_strengths,
            )
            # LoRA hooks are installed before the composite activates. Merge
            # hooks must be present for the first base-weight copy; routed
            # hooks do no work until a target Linear runs.
            if active_loras:
                targets = self._group_lora_factors_by_param_name(active_loras)
                if lora_mode == "merge":
                    self._register_merge_lora_hooks(
                        active_device,
                        targets,
                        stochastic_rounding=stochastic_rounding,
                    )
                else:
                    self._register_routed_lora_hooks(targets)
            schedule_streaming = (
                self._transient_streaming
                and bool(self._composite.streamed)
                and active_device.type == "cuda"
            )
            activation_context = (
                torch.inference_mode(False)
                if schedule_streaming
                else contextlib.nullcontext()
            )
            # Reusable transient targets must be mutable even if the caller
            # activates the offloader from inside inference mode.
            with activation_context:
                self._composite.activate(
                    active_device,
                    compile_blocks=not (
                        active_loras and lora_mode == "routed"
                    ),
                )
            if schedule_streaming:
                self._install_transient_streaming_hooks()
        except BaseException:
            # Deactivation is idempotent over partial component and hook state.
            self.deactivate()
            raise

    def deactivate(self) -> None:
        if self._active_device is None:
            return
        # Scheduling hooks must stop before their components deactivate. Drain
        # asynchronous copies before removing LoRA merge hooks. Cleanup and
        # activation-lock release still run if component teardown raises.
        try:
            self._clear_transient_streaming_hooks()
        finally:
            try:
                self._composite.deactivate()
            finally:
                try:
                    self._clear_active_lora_hooks()
                finally:
                    self._active_device = None
                    self._activation_lock.release()

    @contextlib.contextmanager
    def optimizer_step(self) -> Iterator[None]:
        """Context manager wrapping the optimizer-step boundary for
        managed trainable weights.

        On CUDA activation, non-streamed trainables are already active
        through :class:`PinnedComponent`, while streamed-component trainables
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
        (:class:`PinnedComponent` and :class:`StreamedComponent` alike), so
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
        with self._composite.optimizer_step():
            yield

    @contextlib.contextmanager
    def gather_for_step(self) -> Iterator[None]:
        """Backward-compatible alias for :meth:`optimizer_step`.

        The public API names the boundary after the operation that
        requires all streamed trainable weight data to be materialized: the
        optimizer step.
        """
        with self.optimizer_step():
            yield
