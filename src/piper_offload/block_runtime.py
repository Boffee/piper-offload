"""Shared contract for CUDA block-residency runtimes."""

import contextlib
from collections.abc import Sequence
from typing import Protocol

import torch

from .block_compile import CompileBackend
from .pinned_module import PinnedModuleInstance, PinnedModuleLoadPlan


def validate_load_plans(
    instances: Sequence[PinnedModuleInstance],
    load_plans: Sequence[PinnedModuleLoadPlan],
) -> tuple[PinnedModuleLoadPlan, ...]:
    """Return plans after validating one positional plan per instance."""
    plans = tuple(load_plans)
    if len(plans) != len(instances) or any(
        plan.instance is not instance
        for plan, instance in zip(plans, instances, strict=True)
    ):
        raise ValueError(
            "block runtime requires one matching load plan per block"
        )
    return plans


class BlockRuntime(Protocol):
    """CUDA working-set lifecycle shared by block-residency strategies.

    Implementations allocate accelerator resources only during
    :meth:`acquire`. Acquisition may fail after partially initializing a
    runtime, so :meth:`release` must be safe for released and partial states
    and release every resource it can before propagating a cleanup error.
    """

    @property
    def acquired(self) -> bool: ...

    @property
    def compile_backend(self) -> CompileBackend:
        """The ``torch.compile`` backend required by this strategy."""
        ...

    def acquire(
        self,
        device: torch.device,
        load_plans: Sequence[PinnedModuleLoadPlan],
    ) -> None:
        """Allocate the CUDA working set and install execution hooks."""
        ...

    def release(self) -> None:
        """Idempotently release the CUDA working set from any state."""
        ...

    def optimizer_step(self) -> contextlib.AbstractContextManager[None]: ...


__all__ = ["BlockRuntime", "validate_load_plans"]
