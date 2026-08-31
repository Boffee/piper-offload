"""Shared test configuration.

Piper Offload is a GPU offloading library — pinned host memory and device
transfers are CUDA features — so most of the suite genuinely needs a GPU.
The CPU-runnable subset (registry/dispatch, specs, caching, dequant math,
layout signatures) is what a CPU-only gate (and CI runner) can cover.

To make ``pytest`` green on a CPU-only machine without hand-marking every
GPU test, treat PyTorch's CUDA-unavailable and missing-NVIDIA-driver runtime
errors as "needs a GPU" and report them as skipped rather than failed. On a
GPU box nothing is intercepted and the full suite runs. Tests that intend to
assert a CUDA error catch it themselves, so they are unaffected.
"""

import contextlib
import sys
from collections.abc import Generator

import pytest
import torch
from torch import nn

from piper_offload import ModelOffloader

_CUDA_UNAVAILABLE_ERROR_FRAGMENTS = ("CUDA", "NVIDIA driver")


def _windows_cuda_current_device_unavailable() -> int:
    """Keep GPU-less Windows tests out of PyTorch's native CUDA initializer."""
    raise RuntimeError("CUDA is unavailable on this Windows runner.")


if sys.platform == "win32" and not torch.cuda.is_available():
    # CUDA PyTorch wheels can terminate the process with a native access
    # violation when current_device() enters CUDA internals on GPU-less
    # Windows hosts. Pinned allocation is guarded in clone_to_pinned_cpu;
    # guard this remaining direct CUDA entry point for tests as well.
    torch.cuda.current_device = _windows_cuda_current_device_unavailable


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cuda: test requires a CUDA GPU")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None]:
    outcome = yield
    if torch.cuda.is_available() or outcome.excinfo is None:
        return
    exc = outcome.excinfo[1]
    if isinstance(exc, RuntimeError) and any(
        fragment in str(exc) for fragment in _CUDA_UNAVAILABLE_ERROR_FRAGMENTS
    ):
        outcome.force_exception(
            pytest.skip.Exception(f"needs a CUDA GPU: {exc}", _use_item_location=True)
        )


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


def streamed_components(offloader: object) -> list:
    """A ModelOffloader's streamed components (test-introspection helper)."""
    return list(offloader._composite.streamed)  # type: ignore[attr-defined]


def pinned_component(offloader: object):
    """A ModelOffloader's resident component, or None."""
    return offloader._composite.resident  # type: ignore[attr-defined]


def transient_components(offloader: object) -> list:
    """A ModelOffloader's ``(path, component)`` transient pairs."""
    return list(offloader._composite.transient)  # type: ignore[attr-defined]
