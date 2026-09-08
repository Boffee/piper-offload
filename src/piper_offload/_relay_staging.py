"""Chunk views and bounded CUDA copy ordering for the host relay.

The caller owns the host allocation, its pin lease, and the collective lock.
CPU communication runs between generator yields. Closing a pipeline drains
both copy streams before the caller may release the lease or reuse storage.
"""

from collections import deque
from collections.abc import Generator
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class HostChunk:
    inputs: list[torch.Tensor]
    outputs: list[torch.Tensor]
    scratch: torch.Tensor | None = None


def chunk_ranges(length: int, capacity: int) -> Generator[tuple[int, int]]:
    # Empty tensors still issue one collective, preserving rank ordering.
    for start in range(0, max(length, 1), capacity):
        yield start, min(start + capacity, length)


def make_host_chunk(
    buffer: torch.Tensor, capacity: int, width: int, inputs: int, outputs: int, inplace: bool,
) -> HostChunk:
    used = inputs if inplace else inputs + outputs
    views = [buffer[i * capacity:i * capacity + width] for i in range(used)]
    scratch_start = used * capacity
    return HostChunk(
        views[:inputs], views[:outputs] if inplace else views[inputs:],
        buffer[scratch_start:] if scratch_start < buffer.numel() else None,
    )


class TransferPipeline:
    def __init__(self, device: torch.device, depth: int) -> None:
        self.device = device
        self.depth = depth
        self.download = torch.cuda.Stream(device=device)
        self.upload = torch.cuda.Stream(device=device)
        self.ready = [torch.cuda.Event() for _ in range(depth)]
        self.uploaded = [torch.cuda.Event() for _ in range(depth)]

    def chunks(
        self, sources: list[torch.Tensor], destinations: list[torch.Tensor],
        buffer: torch.Tensor, capacity: int, slots: int, inplace: bool,
    ) -> Generator[HostChunk]:
        pending: deque[tuple[int, int, int, HostChunk]] = deque()
        ranges = iter(chunk_ranges(destinations[0].numel(), capacity))
        region_bytes = slots * capacity

        def download(lane: int, start: int, end: int, *, reuse: bool) -> None:
            region = buffer[lane * region_bytes:(lane + 1) * region_bytes]
            chunk = make_host_chunk(region, capacity, end - start, len(sources), len(destinations), inplace)
            with torch.cuda.stream(self.download):
                if reuse:
                    # The previous upload must finish reading this region
                    # before a new download can overwrite it.
                    self.download.wait_event(self.uploaded[lane])
                for source, host in zip(sources, chunk.inputs, strict=True):
                    host.copy_(source[start:end], non_blocking=True)
                self.ready[lane].record(self.download)
            pending.append((lane, start, end, chunk))

        try:
            # Honor both source production and previous destination use on
            # the caller's stream, including receive-only ranks.
            current = torch.cuda.current_stream(self.device)
            self.download.wait_stream(current)
            self.upload.wait_stream(current)
            for lane in range(self.depth):
                span = next(ranges, None)
                if span is None:
                    break
                download(lane, *span, reuse=False)
            while pending:
                lane, start, end, chunk = pending.popleft()
                self.ready[lane].synchronize()
                yield chunk
                with torch.cuda.stream(self.upload):
                    for destination, host in zip(destinations, chunk.outputs, strict=True):
                        destination[start:end].copy_(host, non_blocking=True)
                    self.uploaded[lane].record(self.upload)
                span = next(ranges, None)
                if span is not None:
                    download(lane, *span, reuse=True)
        finally:
            # Also drain prefetched downloads and prior uploads on exceptions
            # or generator.close(). No transfer may outlive the outer pin lease.
            try:
                self.download.synchronize()
            finally:
                self.upload.synchronize()
