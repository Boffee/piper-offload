"""Model-aware resource cache.

:class:`ResourceCache` owns resource registration, accounting, leases, and
explicit eviction. :class:`ModelCache` supplies shared host-memory policy at
construction and activation-scoped model use, including adapter dependency
leasing and device activation. The generic cache machinery remains unaware
of models and adapters.
"""

import contextlib
from collections.abc import Generator, Sequence
from typing import Any, cast, override

import torch
from torch import nn

from .adapter import Adapter, AdapterMode
from .host_memory import HostMemoryManager
from .model_offloader import ModelOffloader
from .protocols import ResourceSpec, ResourceStore
from .resource_cache import ResourceCache
from .resource_specs import AdapterSpec, ModelSpec


class ModelCache(ResourceCache):
    """Resource cache with model activation and adapter coordination.

    Inherits the resource-agnostic registry and lease API. Each model entry owns
    one :class:`ModelOffloader` and supports sequential reuse only; overlapping
    uses of the same entry fail regardless of which caller initiates them.
    Model and adapter stores are retained until explicit eviction. Their
    compatible CPU tensors preserve factory-supplied file mappings, so the OS
    controls pageable residency. One :class:`HostMemoryManager` supplies the
    shared pin budget and allocation handles for all models and adapters built
    from :class:`ModelSpec` and :class:`AdapterSpec`, including direct leases.
    Pass a manager to share that policy across caches or with manual captures;
    otherwise the cache creates its own. Custom specs own their construction.
    """

    def __init__(self, *, memory_manager: HostMemoryManager | None = None) -> None:
        super().__init__(max_cache_bytes=None)
        self._memory_manager = memory_manager if memory_manager is not None else HostMemoryManager()

    @property
    def memory_manager(self) -> HostMemoryManager:
        """Shared host-memory policy; configure its budget or inspect its stats."""
        return self._memory_manager

    @override
    def _build_store(self, spec: ResourceSpec[Any]) -> ResourceStore:
        if isinstance(spec, (ModelSpec, AdapterSpec)):
            return spec.build_store(memory_manager=self._memory_manager)
        return super()._build_store(spec)

    @contextlib.contextmanager
    def use[M: nn.Module](
        self,
        model: ModelSpec[M],
        *,
        device: torch.device | str,
        adapter_specs: Sequence[AdapterSpec] = (),
        adapter_strengths: Sequence[float] | None = None,
        adapter_mode: AdapterMode = "merge",
        stochastic_rounding: bool = True,
    ) -> Generator[M]:
        """Lease dependencies and activate a cached model runtime.

        ``adapter_strengths`` defaults to one for each adapter and, when
        supplied, must have the same length as ``adapter_specs``. Repeated resource keys
        contribute once per occurrence. Exact-zero strengths are inactive and
        their adapter resources are not leased. ``stochastic_rounding`` is
        forwarded to the model activation's merge path and defaults to
        stochastic requantization for quantized delta targets and scaled
        quantized parameter values; routed targets are unaffected.
        """
        specs = tuple(adapter_specs)
        strengths = None if adapter_strengths is None else tuple(adapter_strengths)
        # A zero-strength adapter is absent from this activation. Filter it
        # before leasing so its factory, cache admission, and host backing are
        # never needed merely to produce a no-op.
        if strengths is not None:
            active = tuple((spec, strength) for spec, strength in zip(specs, strengths, strict=True) if strength != 0.0)
            specs = tuple(spec for spec, _strength in active)
            strengths = tuple(strength for _spec, strength in active)

        # Dependencies are leased first, so admitting the model cannot evict a
        # adapter selected for this same runtime.
        with self.lease_many((*specs, model)) as resources:
            adapters = cast(tuple[Adapter, ...], resources[:-1])
            offloader = cast(ModelOffloader, resources[-1])
            offloader.activate(
                device,
                adapters=adapters,
                adapter_strengths=strengths,
                adapter_mode=adapter_mode,
                stochastic_rounding=stochastic_rounding,
            )
            try:
                yield cast(M, offloader.value)
            finally:
                offloader.deactivate()


__all__ = ["ModelCache"]
