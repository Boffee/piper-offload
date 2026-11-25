"""Unified CUDA offload binding with optional adapter application.

Supports resident bulk offload and dynamic unit residency / streaming with
optional per-parameter adapter application.
"""

import threading
from collections.abc import Callable, Sequence
from typing import Self

import torch
from torch import nn

from ._devices import canonical_device
from ._parameter_update import plan_merge_targets
from .adapter import (
    Adapter,
    AdapterMode,
    AdapterTargetUpdates,
    group_adapter_updates,
    resolve_adapter_strengths,
)
from .host_module import ParameterOverride
from .lora import ScaledLoRAFactor, install_routed_residual_hook
from .model_partition import ModelPartition, _capture_and_bind, _ShellClaim, claim_shell
from .module_names import resolve_parent_leaf
from .offload_config import (
    _USE_CONFIG_BUDGET,
    OffloadConfig,
    _UseConfigBudget,
    resolve_offload_config,
    validate_vram_budget,
)
from .pin_manager import PinLease, host_pin_manager

type _ParameterUpdateMap = dict[str, AdapterTargetUpdates]


class ModelRuntimeInUseError(RuntimeError):
    """A model offloader already has an active use."""


class ModelRuntimeFailedError(RuntimeError):
    """Activation could not release what a prior CUDA session kept.

    Raised after a retry of that release failed too, with the retry's error
    as the cause. Recreate the CUDA worker; see the models guide.
    """


__all__ = [
    "ModelOffloader",
    "ModelRuntimeFailedError",
    "ModelRuntimeInUseError",
]


class ModelOffloader:
    """Move a whole model or managed offload units between host RAM and
    CUDA, with optional adapter application.

    Construct directly from a module with :meth:`from_module` or bind a reusable
    flat :class:`~piper_offload.model_store.ModelStore` with
    :meth:`~piper_offload.model_store.ModelStore.bind`. One offloader owns one
    bound model and may be reused sequentially, but it cannot create model
    replicas or serve overlapping activations. Concurrent use fails immediately
    with :class:`ModelRuntimeInUseError`.

    :class:`~piper_offload.offload_config.OffloadConfig` selects
    ``offload_paths``, ``vram_budget_bytes``, and optional per-unit Inductor
    compilation. Adapter requests are supplied directly to :meth:`activate`;
    merge mode builds activation-scoped parameter overrides so each update runs
    once when staged and stays cached across hits.

    Parameters
    ----------
    model:
        The concrete model bound to the supplied partition.
    partition:
        Bound internal :class:`ModelPartition` owning the model's host backing
        and :class:`~piper_offload.residency_engine.ResidencyEngine`.
    cache_bytes:
        Stable host-cache bytes owned by the bound store.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        partition: ModelPartition,
        cache_bytes: int,
        _shell_claim: _ShellClaim | None = None,
    ) -> None:
        self._shell_claim = claim_shell(model) if _shell_claim is None else _shell_claim
        self._model = model
        self._active_device: torch.device | None = None
        self._partition = partition
        self._engine = partition.engine
        self._cache_bytes = cache_bytes
        self._activation_lock = threading.Lock()
        self._routed_hook_removers: list[Callable[[], None]] = []
        self._routed_lease: PinLease | None = None

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        config: OffloadConfig | None = None,
    ) -> Self:
        """Capture ``model`` and bind it under ``config`` as one cached runtime.

        Capture installs host backing on ``model``. The runtime retains those
        bindings across activations. ``config`` configures ``offload_paths``,
        ``vram_budget_bytes``, and ``block_compile`` (defaulting to whole-model
        resident offload when omitted).
        Requires frozen parameters (``requires_grad=False``) and takes
        ownership of compatible pageable CPU backing for every managed
        parameter and buffer, including non-resizable checkpoint views, while
        preserving mappings and each adapter's physical representation. Other
        devices and incompatible views are normalized into CPU storage.
        """
        resolved = resolve_offload_config(config)
        store, partition, claim = _capture_and_bind(model, resolved)
        try:
            return cls(
                model,
                partition=partition,
                cache_bytes=store.cache_bytes,
                _shell_claim=claim,
            )
        except BaseException:
            claim.release()
            raise

    # ------------------------------------------------------------------ API

    def _build_merge_parameter_overrides(
        self,
        active_device: torch.device,
        updates: _ParameterUpdateMap,
        *,
        stochastic_rounding: bool = True,
    ) -> dict[str, ParameterOverride]:
        if active_device.type != "cuda":
            raise ValueError(
                "ModelOffloader merge mode requires CUDA activation; "
                f"got {active_device}. Use adapter_mode='routed' "
                "for CPU activation."
            )

        params_by_name = dict(self._model.named_parameters(remove_duplicate=False))
        targets = plan_merge_targets(params_by_name, updates, stochastic_rounding=stochastic_rounding)
        overrides: dict[str, ParameterOverride] = {}
        for param_name, param, plan in targets:
            plan.validate_parameter(param)
            overrides[param_name] = ParameterOverride(source=plan.source, update=plan.update)
        return overrides

    def _register_routed_lora_hooks(
        self,
        updates: _ParameterUpdateMap,
    ) -> None:
        """Resolve every routed target before leasing factors or installing hooks.

        The PRE hook copies all LoRA factors for that target from immutable
        host backing to the invocation's input device. The POST hook applies
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

        targets: list[tuple[nn.Linear, list[ScaledLoRAFactor]]] = []
        for param_name, contributions in updates.items():
            parent, _leaf = resolve_parent_leaf(self._model, param_name)
            if not isinstance(parent, nn.Linear):
                raise ValueError(
                    f"Routed LoRA mode requires nn.Linear targets; "
                    f"target {param_name!r} has parent module of "
                    f"type {type(parent).__name__}. Use mode='merge' "
                    f"for non-Linear targets."
                )
            targets.append((parent, contributions.factors))

        if self._active_device is not None and self._active_device.type == "cuda":
            # The hooks stage every factor again on each forward: hold the
            # factors' storage, pageable, for the session so a pinned copy of
            # it stays readable and is never evicted under a staging copy. A
            # CPU session stages synchronously and needs no lease.
            self._routed_lease = host_pin_manager.acquire(
                (
                    tensor
                    for _, factors in targets
                    for factor in factors
                    for host in (factor.a, factor.b)
                    for tensor in host.storage_tensors()
                ),
                pin=False,
            )
        for parent, factors in targets:
            remove_hook = install_routed_residual_hook(parent, factors)
            self._routed_hook_removers.append(remove_hook)

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

    def _clear_active_adapter_hooks(self) -> None:
        remove_hooks = self._routed_hook_removers
        self._routed_hook_removers = []
        for remove_hook in reversed(remove_hooks):
            remove_hook()
        # Routed staging can still be in flight only if the residency engine
        # failed to finish releasing; the lease then stays until a retry
        # releases it.
        if self._routed_lease is not None and not self._engine.failed:
            self._routed_lease.close()
            self._routed_lease = None

    # ----------------------------------------------- ResourceBinding interface

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def value(self) -> nn.Module:
        return self._model

    @property
    def cache_bytes(self) -> int:
        """Stable host-backing bytes reserved in the :class:`ResourceCache`."""
        return self._cache_bytes

    @property
    def active_device(self) -> torch.device | None:
        """Currently active device, or ``None`` when inactive."""
        return self._active_device

    @property
    def vram_budget_bytes(self) -> int | None:
        """VRAM budget for offloaded units in the active session, or configured default."""
        return self._engine.vram_budget_bytes

    @property
    def failed(self) -> bool:
        """Whether a prior session still holds CUDA, pin, or compile state it could not release.

        The next :meth:`deactivate` or :meth:`activate` retries the release.
        """
        return self._engine.failed

    @property
    def _retains_cache_lease(self) -> bool:
        """Keep the cached runtime alive until failed cleanup has finished."""
        return self.failed

    def _retry_release(self) -> None:
        """Retry releasing what a prior session kept; the caller holds the activation lock."""
        try:
            self._engine.deactivate()
        finally:
            self._clear_active_adapter_hooks()

    @property
    def param_names(self) -> frozenset[str]:
        """Parameter names managed by this offloader."""
        return self._partition.param_names

    @property
    def buffer_names(self) -> frozenset[str]:
        """Buffer names managed by this offloader."""
        return self._partition.buffer_names

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
        vram_budget_bytes: int | _UseConfigBudget | None = _USE_CONFIG_BUDGET,
    ) -> None:
        """Make the owned model usable on ``device``.

        ``vram_budget_bytes`` optionally overrides :attr:`OffloadConfig.vram_budget_bytes`
        for this session only (for example, to pass the free-VRAM budget
        available for offloaded unit weights in this run); omitting it uses the
        configured default on every activation.

        ``adapters`` and their optional ``adapter_strengths`` apply only to this
        activation. Exact-zero strengths are inactive and create no updates.
        Adapters that allow partial targets apply only to parameters managed by
        this offloader; strict adapters reject any absent target. In merge
        mode, partial adapters also skip deltas to meta parameters without
        an active value provider, unless another delta requires that source.
        ``adapter_mode`` selects in-place merge updates or routed LoRA
        residual hooks. Routed mode requires factor-only adapters.
        ``stochastic_rounding`` uses stochastic requantization for quantized
        merge targets by default; pass ``False`` for deterministic rounding.
        Parameter values are merge-only and populate frozen floating-point
        meta parameters from registered physical representations; routed mode
        never requantizes. Such a meta target is materialized only while its
        parameter value is active. Deltas from other adapters may modify the
        supplied value, subject to its representation's merge capabilities.
        Value scaling and deltas combine before terminal quantized encoding.
        Because the offloader owns one model runtime, a
        second activation before :meth:`deactivate` raises
        :class:`ModelRuntimeInUseError` immediately. If a prior session
        could not finish releasing (see :attr:`failed`), activation first
        retries that release; if it fails again, activation raises
        :class:`ModelRuntimeFailedError` before activating any component.
        """
        if not isinstance(vram_budget_bytes, _UseConfigBudget):
            validate_vram_budget(vram_budget_bytes)
        active_device = self._resolve_device(device)
        if not self._activation_lock.acquire(blocking=False):
            raise ModelRuntimeInUseError(
                "ModelOffloader already has an active use; overlapping model activations are not supported"
            )
        # The try starts as soon as the claim is taken, so an interrupt
        # anywhere after it still gives the claim back.
        try:
            if self.failed:
                try:
                    self._retry_release()
                except Exception as error:  # an interrupt stays an interrupt
                    raise ModelRuntimeFailedError(
                        "ModelOffloader cannot activate: releasing what its prior CUDA "
                        "session kept failed again. Recreate the CUDA worker."
                    ) from error
            self._active_device = active_device
            if adapter_mode not in ("merge", "routed"):
                raise ValueError(f"adapter_mode must be 'merge' or 'routed', got {adapter_mode!r}")
            active_adapters = resolve_adapter_strengths(adapters, adapter_strengths)
            updates = group_adapter_updates(active_adapters, self.param_names) if active_adapters else {}
            parameter_overrides: dict[str, ParameterOverride] | None = None
            if updates:
                if adapter_mode == "merge":
                    parameter_overrides = self._build_merge_parameter_overrides(
                        active_device,
                        updates,
                        stochastic_rounding=stochastic_rounding,
                    )
                else:
                    self._register_routed_lora_hooks(updates)
            resident_overrides, unit_overrides = self._partition.route_overrides(
                parameter_overrides
            )
            self._engine.activate(
                active_device,
                compile_blocks=not (updates and adapter_mode == "routed"),
                resident_overrides=resident_overrides,
                unit_overrides=unit_overrides,
                vram_budget_bytes=vram_budget_bytes,
            )
        except BaseException:
            if self._active_device is None:
                # Nothing was activated yet; only the claim is held.
                self._activation_lock.release()
            else:
                # Deactivation is idempotent over partial component and hook
                # state, and gives the claim back.
                self.deactivate()
            raise

    def deactivate(self) -> None:
        """End the active use and release its device working sets; idempotent.

        Cleanup continues past a failing component and then raises. What it
        could not release stays held and :attr:`failed` reports it; calling
        this again while inactive, or :meth:`activate`, retries.
        """
        if self._active_device is None:
            if self.failed and self._activation_lock.acquire(blocking=False):
                try:
                    self._retry_release()
                finally:
                    self._activation_lock.release()
            return
        # Engine synchronizes before routed hooks and their lease go. Cleanup
        # and the activation-lock release still run if engine teardown raises.
        try:
            self._engine.deactivate()
        finally:
            try:
                self._clear_active_adapter_hooks()
            finally:
                self._active_device = None
                self._activation_lock.release()
