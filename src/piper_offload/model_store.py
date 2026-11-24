"""Reusable topology-free host backing store for frozen models."""

from dataclasses import dataclass
from typing import Self

from torch import nn

from .composite_component import _bind_module_store, reserve_shell
from .host_module import HostModuleStore
from .model_offloader import ModelOffloader
from .offload_config import OffloadConfig, resolve_offload_config


@dataclass(frozen=True, slots=True)
class ModelStore:
    """Reusable frozen host storage captured once and bindable across shells.

    ``ModelStore`` holds flat name-keyed host parameters and buffers with no
    block-streaming topology or execution config baked in. Callers can bind
    one store to one or more structurally compatible models with independent
    :class:`~piper_offload.offload_config.OffloadConfig` instances via
    :meth:`bind`, sharing host backing (including checkpoint file mappings)
    without duplicating CPU memory.
    """

    _module_store: HostModuleStore

    @classmethod
    def from_module(cls, model: nn.Module) -> Self:
        """Capture frozen parameters and registered buffers from ``model``.

        Requires ``requires_grad=False`` on every parameter before mutating
        host state and transfers ownership of compatible CPU allocations while
        preserving checkpoint file mappings.
        """
        with reserve_shell(model):
            store = HostModuleStore.from_module(model)
        if not store.params and not store.buffers:
            raise ValueError(
                "Offloading requires at least one parameter or registered buffer to manage."
            )
        return cls(store)

    @property
    def param_names(self) -> frozenset[str]:
        """Managed parameter names in this store."""
        return frozenset(self._module_store.params)

    @property
    def buffer_names(self) -> frozenset[str]:
        """Managed buffer names in this store."""
        return frozenset(self._module_store.buffers)

    @property
    def cache_bytes(self) -> int:
        """Total host-backing bytes held by this store."""
        return self._module_store.cache_bytes

    def bind(
        self,
        model: nn.Module,
        config: OffloadConfig | None = None,
    ) -> ModelOffloader:
        """Validate ``model`` and bind this store under ``config``.

        Preflight validations check every managed parameter, buffer, offload
        path, cross-group tie rule, target slot conflict, and exclusive live
        shell ownership before installation mutates ``model``. If an unexpected
        error occurs partway through host installation, discard that target
        shell and bind a fresh one.

        ``model`` must not already be owned by a live :class:`ModelOffloader`.
        Extra unmanaged parameters anywhere on ``model`` (including inside
        occupied blocks) are permitted, ignored, and stay on the caller's
        device, whereas extra non-empty blocks inside a configured block list
        are rejected so no block silently remains unstreamed.
        """
        resolved = resolve_offload_config(config)
        composite, claim = _bind_module_store(self._module_store, model, resolved)
        try:
            return ModelOffloader(
                model,
                composite=composite,
                cache_bytes=self.cache_bytes,
                _shell_claim=claim,
            )
        except BaseException:
            claim.release()
            raise


__all__ = ["ModelStore"]
