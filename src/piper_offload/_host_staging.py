"""Bounded Linux staging for pageable host-to-device copies.

Private writable checkpoint mappings must not be registered in place on
Linux: CUDA/HIP can request writable page pins even for a host-to-device copy,
breaking copy-on-write for every mapped page.  A small pair of reusable pinned
buffers preserves asynchronous DMA without retaining an anonymous copy of the
checkpoint.
"""

import logging
import sys
import threading

import torch

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 8 * 1024**2
_SLOT_COUNT = 2


class _LinuxHostStaging:
    """Process-wide ping-pong buffers for contiguous pageable sources."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots: tuple[torch.Tensor, ...] | None = None
        self._events: list[torch.cuda.Event | None] = [None] * _SLOT_COUNT
        self._disabled = False

    def _allocate(self) -> tuple[torch.Tensor, ...] | None:
        if self._disabled:
            return None
        if self._slots is None:
            try:
                self._slots = tuple(
                    torch.empty(
                        _CHUNK_BYTES,
                        dtype=torch.uint8,
                        device="cpu",
                        pin_memory=True,
                    )
                    for _ in range(_SLOT_COUNT)
                )
            except RuntimeError as error:
                # Pageable copies remain correct.  Remember the failure so a
                # constrained process does not retry the allocation per tensor.
                self._disabled = True
                logger.warning(
                    "Could not allocate the bounded Linux host staging window; "
                    "using pageable host-to-device copies: %s",
                    str(error),
                )
                return None
        return self._slots

    def reserve(self) -> bool:
        """Allocate the fixed window before other host pins consume capacity."""
        with self._lock:
            return self._allocate() is not None

    def copy(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        *,
        non_blocking: bool,
    ) -> bool:
        """Stage one compatible copy, returning False for the direct fallback."""
        if (
            self._disabled
            or source.device.type != "cpu"
            or destination.device.type != "cuda"
            or source.layout is not torch.strided
            or destination.layout is not torch.strided
            or source.dtype != destination.dtype
            or source.shape != destination.shape
            or not source.is_contiguous()
            or not destination.is_contiguous()
            or source.is_pinned()
        ):
            return False

        with self._lock:
            slots = self._allocate()
            if slots is None:
                return False
            stream = torch.cuda.current_stream(destination.device)
            source_flat = source.reshape(-1)
            destination_flat = destination.reshape(-1)
            elements_per_chunk = _CHUNK_BYTES // source.element_size()
            last_event: torch.cuda.Event | None = None
            with torch.cuda.stream(stream):
                for chunk_idx, start in enumerate(
                    range(0, source.numel(), elements_per_chunk)
                ):
                    slot_idx = chunk_idx % _SLOT_COUNT
                    prior = self._events[slot_idx]
                    if prior is not None:
                        prior.synchronize()
                    end = min(start + elements_per_chunk, source.numel())
                    staging = slots[slot_idx].view(source.dtype)[: end - start]
                    staging.copy_(source_flat[start:end])
                    destination_flat[start:end].copy_(
                        staging,
                        non_blocking=True,
                    )
                    event = torch.cuda.Event()
                    event.record(stream)
                    self._events[slot_idx] = event
                    last_event = event

            if not non_blocking and last_event is not None:
                last_event.synchronize()
            return True


_linux_host_staging = _LinuxHostStaging()


def reserve_linux_host_staging() -> bool:
    """Reserve the Linux window, or report that pageable fallback is required."""
    return sys.platform.startswith("linux") and _linux_host_staging.reserve()


def copy_host_to_device(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    non_blocking: bool,
) -> None:
    """Copy host bytes, using bounded pinned staging when useful on Linux."""
    if sys.platform.startswith("linux") and _linux_host_staging.copy(
        destination,
        source,
        non_blocking=non_blocking,
    ):
        return
    destination.copy_(source, non_blocking=non_blocking)


__all__ = ["copy_host_to_device", "reserve_linux_host_staging"]
