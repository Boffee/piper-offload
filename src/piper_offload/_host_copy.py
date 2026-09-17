"""Copy from anonymous storage when available, otherwise from source storage."""

from collections.abc import Callable, Mapping

import torch

from ._host_backing import HostBacking
from ._host_staging import copy_host_to_device as _copy_host_to_device

type TensorCopy = Callable[[torch.Tensor, torch.Tensor], None]


def copy_host_to_device(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    backings: Mapping[int, HostBacking],
    non_blocking: bool,
) -> None:
    """Copy using the source owner's handles and protect storage until completion."""
    if source.device.type != "cpu":
        raise ValueError("Host copies require a CPU source")
    try:
        backing = backings[source.untyped_storage()._cdata]
    except KeyError:
        raise ValueError("Copy source was not captured by this backing owner") from None
    # Source storage uses the existing runtime and pin leases. Other accelerators
    # need completion tracking before they can use evictable anonymous storage.
    if destination.device.type not in {"cpu", "cuda"} or not backing.has_anonymous_storage:
        _copy_host_to_device(destination, source, non_blocking=non_blocking)
        return
    stream = None
    if destination.device.type == "cuda":
        with torch.cuda.device(destination.device):
            if torch.cuda.is_current_stream_capturing():
                # A captured H2D node can read this pointer on every replay;
                # an ordinary copy event cannot protect that lifetime.
                # Reject before acquisition, which may query pending events.
                raise RuntimeError("Copies from anonymous CPU storage cannot be captured in CUDA graphs")
        stream = torch.cuda.current_stream(destination.device)
    lease = backing.acquire(source)
    try:
        _copy_host_to_device(destination, lease.tensor, non_blocking=non_blocking)
    finally:
        # Even a failing copy can have queued partial work. A marker on the
        # actual destination stream covers that work without blocking the CPU.
        if stream is None:
            lease.close()
        else:
            try:
                completion = stream.record_event()
            except BaseException:
                stream.synchronize()
                lease.close()
                raise
            lease.close(completion)


__all__ = ["TensorCopy", "copy_host_to_device"]
