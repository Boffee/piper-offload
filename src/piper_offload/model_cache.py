"""Model-aware resource cache.

:class:`ResourceCache` owns resource admission, accounting, leases, and
eviction. :class:`ModelCache` specializes it with activation-scoped model use,
including LoRA dependency leasing and device activation, while keeping the
generic cache machinery unaware of models and adapters.
"""

import contextlib
from collections.abc import Iterator, Sequence
from typing import cast

import torch
from torch import nn

from .lora import LoRA, LoRAMode
from .model_offloader import ModelOffloader
from .resource_cache import ResourceCache
from .resource_specs import LoRASpec, ModelSpec
from .stream_config import StreamConfig


class ModelCache(ResourceCache):
    """Resource cache with model activation and LoRA coordination.

    Inherits the complete resource-agnostic cache API. Each model entry owns
    one :class:`ModelOffloader` and supports sequential reuse only; overlapping
    uses of the same entry fail regardless of which caller initiates them.
    """

    @contextlib.contextmanager
    def use[M: nn.Module](
        self,
        model: ModelSpec[M],
        *,
        device: torch.device | str,
        lora_specs: Sequence[LoRASpec] = (),
        lora_strengths: Sequence[float] | None = None,
        lora_mode: LoRAMode = "merge",
        stochastic_rounding: bool = True,
        stream_config: StreamConfig | None = None,
    ) -> Iterator[M]:
        """Lease dependencies and activate a cached model runtime.

        ``lora_strengths`` defaults to one for each LoRA and, when supplied,
        must have the same length as ``lora_specs``. A LoRA resource key may
        appear only once in a use. Exact-zero strengths are inactive and their
        LoRA resources are not leased. ``stochastic_rounding`` is forwarded
        to the model activation's merge path and defaults to stochastic
        requantization for quantized targets; dense and routed targets are
        unaffected.
        """
        specs = tuple(lora_specs)
        strengths = None if lora_strengths is None else tuple(lora_strengths)
        if strengths is not None and len(strengths) != len(specs):
            raise ValueError(
                "lora_strengths must have the same length as lora_specs"
            )
        if len({spec.key for spec in specs}) != len(specs):
            raise ValueError(
                "lora_specs must not contain the same LoRA resource key more than once"
            )

        # A zero-strength LoRA is absent from this activation. Filter it
        # before leasing so its factory, cache admission, and host backing are
        # never needed merely to produce a no-op.
        if strengths is not None:
            active = tuple(
                (spec, strength)
                for spec, strength in zip(specs, strengths, strict=True)
                if strength != 0.0
            )
            specs = tuple(spec for spec, _strength in active)
            strengths = tuple(strength for _spec, strength in active)

        # Dependencies are leased first, so admitting the model cannot evict a
        # LoRA selected for this same runtime.
        with self.lease_many((*specs, model)) as resources:
            loras = cast(tuple[LoRA, ...], resources[:-1])
            offloader = cast(ModelOffloader, resources[-1])
            config = stream_config if stream_config is not None else StreamConfig()
            offloader.activate(
                device,
                loras=loras,
                lora_strengths=strengths,
                lora_mode=lora_mode,
                stochastic_rounding=stochastic_rounding,
                stream_config=config,
            )
            try:
                yield cast(M, offloader.value)
            finally:
                offloader.deactivate()


__all__ = ["ModelCache"]
