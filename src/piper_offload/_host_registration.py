"""In-place registration using the CUDA/HIP runtime already loaded by PyTorch.

PyTorch's cudart wrapper exposes registration but not GetLastError. A failed
registration leaves a thread-local runtime error that can poison the next
PyTorch kernel launch. Bind the native operations together so handled failures
are cleared without swallowing errors from earlier GPU work.
"""

import ctypes
from collections.abc import Callable
from ctypes.util import dllist
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Protocol, cast

import torch


class HostRegistrationBackend(Protocol):
    """Register complete byte ranges; never allocate or copy their contents."""

    def register(self, pointer: int, size: int) -> bool:
        """Return False for unavailable registration or exhausted capacity."""
        ...

    def unregister(self, pointer: int) -> None:
        """Release an owned registration, raising if it remains registered."""
        ...


class HostRegistrationError(RuntimeError):
    """An unexpected runtime error during registration or unregistration."""

    def __init__(self, operation: str, code: int) -> None:
        super().__init__(f"Host memory {operation} failed with CUDA/HIP error {code}")
        self.code = code


@dataclass(frozen=True, slots=True)
class _Runtime:
    library: ctypes.CDLL
    register: Callable[[int, int, int], int]
    unregister: Callable[[int], int]
    get_last_error: Callable[[], int]


def _load_runtime() -> _Runtime | None:
    if not torch.cuda.is_available():
        return None
    torch.cuda.init()
    hip = torch.version.hip is not None
    prefix = "hip" if hip else "cuda"
    libraries = [
        path for path in dllist()
        if Path(path).name.lower().startswith(
            ("libamdhip64.so", "amdhip64") if hip else ("libcudart.so", "cudart64_")
        )
    ]
    if len(libraries) != 1:
        raise RuntimeError(
            f"Expected one loaded {prefix} runtime library, found {len(libraries)}; "
            "cannot safely select the runtime that owns PyTorch's registrations"
        )
    library = ctypes.CDLL(libraries[0])
    register = getattr(library, f"{prefix}HostRegister")
    register.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    register.restype = ctypes.c_int
    unregister = getattr(library, f"{prefix}HostUnregister")
    unregister.argtypes = [ctypes.c_void_p]
    unregister.restype = ctypes.c_int
    get_last_error = getattr(library, f"{prefix}GetLastError")
    get_last_error.argtypes = []
    get_last_error.restype = ctypes.c_int
    return _Runtime(
        library,
        cast(Callable[[int, int, int], int], register),
        cast(Callable[[int], int], unregister),
        cast(Callable[[], int], get_last_error),
    )


class RuntimeHostRegistration:
    """Lazy CUDA/HIP backend with portable registration and error isolation.

    Portable registration (flag 1) permits use across device contexts. CUDA
    and HIP use error 2 for allocation failure and 801 for unsupported calls.
    Other errors propagate, including foreign registrations: their ownership
    and coverage cannot be assumed to match our requested storage.
    """

    @cached_property
    def _runtime(self) -> _Runtime | None:
        return _load_runtime()

    @staticmethod
    def _check_prior_error(runtime: _Runtime) -> None:
        code = runtime.get_last_error()
        if code:
            raise HostRegistrationError("prior runtime work", code)

    @staticmethod
    def _clear_failed_call(runtime: _Runtime, code: int) -> None:
        pending = runtime.get_last_error()
        if pending not in (0, code):
            raise HostRegistrationError("runtime work", pending)

    def register(self, pointer: int, size: int) -> bool:
        runtime = self._runtime
        if runtime is None:
            return False
        self._check_prior_error(runtime)
        code = runtime.register(pointer, size, 1)
        if code:
            self._clear_failed_call(runtime, code)
            if code in (2, 801):
                return False
            raise HostRegistrationError("registration", code)
        return True

    def unregister(self, pointer: int) -> None:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Cannot unregister without a CUDA/HIP runtime")
        self._check_prior_error(runtime)
        code = runtime.unregister(pointer)
        if code:
            self._clear_failed_call(runtime, code)
            raise HostRegistrationError("unregistration", code)
