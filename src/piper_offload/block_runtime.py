"""Shared contract for CUDA block-residency runtimes."""

import contextlib
from collections.abc import Iterator, Sequence
from typing import Protocol

import torch

from .block_compile import CompileBackend
from .host_module import HostModuleInstance, HostModuleLoadPlan


def validate_load_plans(
    instances: Sequence[HostModuleInstance],
    load_plans: Sequence[HostModuleLoadPlan],
) -> tuple[HostModuleLoadPlan, ...]:
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


def host_transfer_tensors(load_plans: Sequence[HostModuleLoadPlan]) -> Iterator[torch.Tensor]:
    """Enumerate resolved upload sources and trainable optimizer backing.

    Replacements supersede frozen model sources. Optimizer steps still gather
    and scatter the instance's own trainable backing, so retain that too.
    The pin manager deduplicates aliases and whole storage allocations.
    """
    for plan in load_plans:
        for load in plan.loads.values():
            yield from load.source.storage_tensors()
        for buffer in plan.instance.buffers.values():
            yield from buffer.storage_tensors()
        for host in plan.instance.params.values():
            if host.requires_grad:
                yield from host.storage_tensors()


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
        load_plans: Sequence[HostModuleLoadPlan],
    ) -> None:
        """Allocate the CUDA working set and install execution hooks."""
        ...

    def release(self) -> None:
        """Idempotently release the CUDA working set from any state."""
        ...

    def optimizer_step(self) -> contextlib.AbstractContextManager[None]: ...


__all__ = ["BlockRuntime", "host_transfer_tensors", "validate_load_plans"]
