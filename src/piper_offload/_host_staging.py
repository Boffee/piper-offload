"""Bounded pinned staging for host-to-device copies.

Private writable checkpoint mappings must not be registered in place on
Linux: CUDA/HIP can request writable page pins even for a host-to-device copy,
breaking copy-on-write for every mapped page.  A small pair of reusable pinned
buffers preserves asynchronous DMA without retaining an anonymous copy of the
checkpoint.  The same window packs row-strided projected shards on every
platform.
"""

import logging
import sys
import threading
from collections.abc import Iterator

import torch

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 8 * 1024**2
_SLOT_COUNT = 2


class _HostStaging:
    """Process-wide ping-pong buffers for pageable and row-strided sources."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots: tuple[torch.Tensor, ...] | None = None
        self._events: list[torch.cuda.Event | None] = [None] * _SLOT_COUNT
        self._event_devices: list[torch.device | None] = [None] * _SLOT_COUNT
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
                    "Could not allocate the bounded host staging window; "
                    "using direct host-to-device copies: %s",
                    str(error),
                )
                return None
        return self._slots

    def reserve(self) -> bool:
        """Allocate the fixed window before other host pins consume capacity."""
        with self._lock:
            return self._allocate() is not None

    @staticmethod
    def _iter_regions(
        destination: torch.Tensor,
        source: torch.Tensor,
        elements_per_chunk: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield matching regions that each fit one staging slot."""
        if source.is_contiguous():
            source_flat = source.reshape(-1)
            destination_flat = destination.reshape(-1)
            for start in range(0, source.numel(), elements_per_chunk):
                end = min(start + elements_per_chunk, source.numel())
                yield source_flat[start:end], destination_flat[start:end]
            return

        rows, columns = source.shape
        if columns == 0:
            return
        if columns > elements_per_chunk:
            for row in range(rows):
                for start in range(0, columns, elements_per_chunk):
                    end = min(start + elements_per_chunk, columns)
                    yield source[row, start:end], destination[row, start:end]
            return
        rows_per_chunk = max(1, elements_per_chunk // columns)
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            yield source[start:end], destination[start:end]

    def copy(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        *,
        non_blocking: bool,
    ) -> bool:
        """Stage one compatible copy, returning False for the direct fallback."""
        source_is_contiguous = source.is_contiguous()
        stageable_row_strided = (
            source.ndim == 2
            and source.stride(1) == 1
            and source.stride(0) >= source.shape[1]
        )
        if (
            self._disabled
            or source.device.type != "cpu"
            or destination.device.type != "cuda"
            or source.layout is not torch.strided
            or destination.layout is not torch.strided
            or source.dtype != destination.dtype
            or source.shape != destination.shape
            or not (source_is_contiguous or stageable_row_strided)
            or not destination.is_contiguous()
            or (source.is_pinned() and source_is_contiguous)
        ):
            return False

        with self._lock:
            slots = self._allocate()
            if slots is None:
                return False
            stream = torch.cuda.current_stream(destination.device)
            elements_per_chunk = _CHUNK_BYTES // source.element_size()
            last_event: torch.cuda.Event | None = None
            with torch.cuda.stream(stream):
                for chunk_idx, (source_region, destination_region) in enumerate(
                    self._iter_regions(destination, source, elements_per_chunk)
                ):
                    slot_idx = chunk_idx % _SLOT_COUNT
                    event = self._events[slot_idx]
                    if event is not None:
                        event.synchronize()
                    staging = slots[slot_idx].view(source.dtype)[
                        : source_region.numel()
                    ].view(source_region.shape)
                    staging.copy_(source_region)
                    destination_region.copy_(
                        staging,
                        non_blocking=True,
                    )
                    if (
                        event is None
                        or self._event_devices[slot_idx] != destination.device
                    ):
                        event = torch.cuda.Event()
                        self._events[slot_idx] = event
                        self._event_devices[slot_idx] = destination.device
                    event.record(stream)
                    last_event = event

            if not non_blocking and last_event is not None:
                last_event.synchronize()
            return True


_host_staging = _HostStaging()


def reserve_host_staging() -> bool:
    """Reserve the staging window, or report that direct fallback is required."""
    return _host_staging.reserve()


def copy_host_to_device(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    non_blocking: bool,
) -> None:
    """Copy host bytes through bounded pinned staging when useful."""
    should_stage = sys.platform.startswith("linux") or not source.is_contiguous()
    if should_stage and _host_staging.copy(
        destination,
        source,
        non_blocking=non_blocking,
    ):
        return
    destination.copy_(source, non_blocking=non_blocking)


__all__ = ["copy_host_to_device", "reserve_host_staging"]
