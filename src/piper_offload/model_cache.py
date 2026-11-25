"""Model-aware resource cache.

:class:`ResourceCache` owns resource registration, accounting, leases, and
explicit eviction. :class:`ModelCache` starts unbounded and adds
activation-scoped model use, including adapter dependency leasing and device
activation, while keeping the generic cache machinery unaware of models and
adapters.
"""

import contextlib
from collections.abc import Generator, Sequence
from typing import cast

import torch
from torch import nn

from .adapter import AdapterMode, resolve_adapter_strengths
from .offload_config import _USE_CONFIG_BUDGET, _UseConfigBudget
from .resource_cache import ResourceCache
from .resource_specs import AdapterSpec, ModelSpec


class ModelCache(ResourceCache):
    """Resource cache with model activation and adapter coordination.

    Inherits the resource-agnostic registry and lease API. Each model entry owns
    one :class:`ModelOffloader` and supports sequential reuse only; overlapping
    uses of the same entry fail regardless of which caller initiates them.
    By default, stores are retained until explicit eviction. Set a finite
    budget with the inherited :meth:`ResourceCache.resize` to allow eviction
    of inactive stores. Compatible CPU tensors preserve factory-supplied file
    mappings. The OS controls pageable residency, and the independent host-pin
    budget controls locked pages.
    """

    def __init__(self) -> None:
        super().__init__(max_cache_bytes=None)

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
        vram_budget_bytes: int | _UseConfigBudget | None = _USE_CONFIG_BUDGET,
    ) -> Generator[M]:
        """Lease dependencies and activate a cached model runtime.

        ``vram_budget_bytes`` optionally overrides the model spec's configured
        ``OffloadConfig.vram_budget_bytes`` for this activation only; omitting
        it uses the configured default on every run.
        ``adapter_strengths`` defaults to one for each adapter and, when
        supplied, must have the same length as ``adapter_specs``. Repeated resource keys
        contribute once per occurrence. Exact-zero strengths are inactive and
        their adapter resources are not leased. ``stochastic_rounding`` is
        forwarded to the model activation's merge path and defaults to
        stochastic requantization for quantized delta targets and scaled
        quantized parameter values; routed targets are unaffected.
        """
        active = resolve_adapter_strengths(adapter_specs, adapter_strengths)
        # Hold active adapters before model admission can evict them.
        with (
            self.lease_many([spec for spec, _ in active]) as adapters,
            self.lease(model) as offloader,
        ):
            offloader.activate(
                device,
                adapters=adapters,
                adapter_strengths=None if adapter_strengths is None else tuple(strength for _, strength in active),
                adapter_mode=adapter_mode,
                stochastic_rounding=stochastic_rounding,
                vram_budget_bytes=vram_budget_bytes,
            )
            try:
                yield cast(M, offloader.value)
            finally:
                offloader.deactivate()


__all__ = ["ModelCache"]
