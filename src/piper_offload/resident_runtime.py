"""CUDA execution strategy that keeps every block target resident."""

import contextlib
import logging
from collections.abc import Generator, Sequence

import torch

from .block_compile import CompileBackend
from .block_runtime import validate_load_plans
from .host_module import HostModuleInstance, HostModuleLoadPlan
from .target_lease import CudaTargetLease

logger = logging.getLogger(__name__)


class ResidentBlockRuntime:
    """Keep one CUDA target per block for the complete acquired session."""

    def __init__(
        self,
        instances: Sequence[HostModuleInstance],
    ) -> None:
        self._instances = tuple(instances)
        self._device: torch.device | None = None
        self._leases: list[CudaTargetLease] = []
        self._optimizer_step_active = False
        self._move_trainable_grads_to(torch.device("cpu"))

    @property
    def acquired(self) -> bool:
        return self._device is not None

    @property
    def compile_backend(self) -> CompileBackend:
        return "inductor"

    def acquire(
        self,
        device: torch.device,
        load_plans: Sequence[HostModuleLoadPlan],
    ) -> None:
        if self.acquired:
            raise RuntimeError("resident block runtime is already acquired")

        plans = validate_load_plans(self._instances, load_plans)

        self._device = device
        current_stream = torch.cuda.current_stream(device)
        try:
            self._move_trainable_grads_to(device)
            for instance, plan in zip(self._instances, plans, strict=True):
                lease = CudaTargetLease.allocate(
                    plan,
                    device,
                    allocation_stream=current_stream,
                )
                self._leases.append(lease)
                lease.stage(
                    plan,
                    current_stream,
                    non_blocking=True,
                )
                instance.install_target(lease.acquire(current_stream))
            current_stream.synchronize()
        except BaseException:
            self.release()
            raise

        logger.info(
            "resident block runtime acquired: all %d blocks resident",
            len(self._instances),
        )

    def release(self) -> None:
        if self._device is not None:
            # Leave acquired state intact on failure so the component keeps
            # its pageable lease while an upload or copy-back may still be live.
            torch.cuda.synchronize(self._device)
        # Every block returns to host storage before any target closes. A
        # failure keeps the remaining targets and leaves the runtime acquired,
        # so nothing reuses storage a block may still point at; restores are
        # idempotent, so a later release() retries from the start.
        for instance in self._instances:
            instance.install_host()
            instance.move_trainable_grads_to(torch.device("cpu"))
        while self._leases:
            self._leases[-1].close()
            self._leases.pop()
        self._device = None

    @contextlib.contextmanager
    def optimizer_step(self) -> Generator[None]:
        if not self.acquired:
            raise RuntimeError(
                "BlockComponent.optimizer_step() called while its CUDA "
                "working set is released. Acquire the component before "
                "entering the optimizer step."
            )
        if self._optimizer_step_active:
            raise RuntimeError(
                "BlockComponent.optimizer_step() does not support "
                "reentrant entry."
            )
        if not any(instance.has_trainables for instance in self._instances):
            yield
            return

        self._optimizer_step_active = True
        try:
            yield
        finally:
            try:
                self._copy_trainables_to_host()
            finally:
                self._optimizer_step_active = False

    def _copy_trainables_to_host(self) -> None:
        device = self._device
        if device is None:
            raise RuntimeError("resident block runtime is released")
        current_stream = torch.cuda.current_stream(device)
        for instance, lease in zip(
            self._instances,
            self._leases,
            strict=True,
        ):
            if instance.has_trainables:
                instance.copy_trainables_from_target(
                    lease.target,
                    non_blocking=True,
                )
        current_stream.synchronize()

    def _move_trainable_grads_to(self, device: torch.device) -> None:
        for instance in self._instances:
            instance.move_trainable_grads_to(device)


__all__ = ["ResidentBlockRuntime"]
