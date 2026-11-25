"""Shared test configuration.

Piper Offload is a GPU offloading library — host memory and device
transfers are CUDA features — so most of the suite genuinely needs a GPU.
The CPU-runnable subset (registry/dispatch, specs, caching, dequant math,
layout signatures) is what a CPU-only gate (and CI runner) can cover.

Each test that needs a GPU says so with a skip mark, checked before its
setup runs. Nothing turns errors into skips: on a CPU-only host, an
unmarked test that reaches CUDA fails, so a missing mark, or a library
error that mentions CUDA, is reported instead of hidden.
"""

import contextlib
import sys
from collections.abc import Callable, Generator

import torch
from torch import nn

from piper_offload import ModelOffloader


class CallbackParameterTransform:
    """Small transform implementation for low-level load-plan tests."""

    def __init__(
        self,
        callback: Callable[[nn.Parameter], None],
    ) -> None:
        self._callback = callback

    def validate_parameter(self, _param: nn.Parameter) -> None:
        pass

    def apply_parameter(self, param: nn.Parameter) -> None:
        self._callback(param)

    def storage_tensors(self) -> tuple[torch.Tensor, ...]:
        return ()


class Interrupt(BaseException):
    """Like ``KeyboardInterrupt``: not an ``Exception``.

    ``except Exception`` does not catch it, and PyTorch skips ``always_call``
    forward hooks for it.
    """


def _windows_cuda_current_device_unavailable() -> int:
    """Keep GPU-less Windows tests out of PyTorch's native CUDA initializer."""
    raise RuntimeError("CUDA is unavailable on this Windows runner.")


if sys.platform == "win32" and not torch.cuda.is_available():
    # CUDA PyTorch wheels can terminate the process with a native access
    # violation when current_device() enters CUDA internals on GPU-less
    # Windows hosts. Host allocation is guarded in capture_host_tensor;
    # guard this remaining direct CUDA entry point for tests as well.
    torch.cuda.current_device = _windows_cuda_current_device_unavailable


@contextlib.contextmanager
def activated_model(
    offloader: ModelOffloader,
    device: torch.device | str,
    **kwargs: object,
) -> Generator[nn.Module]:
    """Test-only exception-safe scope for the low-level activation API."""
    offloader.activate(device, **kwargs)
    try:
        yield offloader.value
    finally:
        offloader.deactivate()


def residency_engine(offloader: object):
    """A ModelOffloader's ResidencyEngine (test-introspection helper)."""
    return offloader._engine  # type: ignore[attr-defined]


def resident_instance(offloader: object):
    """A ModelOffloader's resident HostModuleInstance, or None."""
    return offloader._engine.resident  # type: ignore[attr-defined]


def offloaded_units(offloader: object) -> list:
    """A ModelOffloader's offloaded unit HostModuleInstances."""
    return list(offloader._engine.units)  # type: ignore[attr-defined]
