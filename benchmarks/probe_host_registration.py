"""Report what native host registration does to a private file mapping here.

Prints how many anonymous (private, unshareable) bytes the process gains
when a private file mapping is registered. A gain equal to the mapping size
means registration forced copy-on-write, which is why such mappings are
copied into owned memory before pinning. Works on Linux and Windows.

    uv run python benchmarks/probe_host_registration.py
"""

import ctypes
import os
import sys
import tempfile
from pathlib import Path

import torch

from piper_offload._host_registration import RuntimeHostRegistration

SIZE = 64 * 1024**2


def anonymous_bytes() -> int:
    if sys.platform.startswith("linux"):
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
        raise RuntimeError("RssAnon missing from /proc/self/status")
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                *[(name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage", "PrivateUsage",
                )],
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        psapi = ctypes.WinDLL("psapi")  # type: ignore[attr-defined]
        kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()  # type: ignore[attr-defined]
        return int(counters.PrivateUsage)
    raise RuntimeError(f"unsupported platform {sys.platform}")


def main() -> None:
    backend = RuntimeHostRegistration()
    out = sys.stdout
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "probe.bin"
        path.write_bytes(os.urandom(SIZE))
        mapped = torch.from_file(str(path), shared=False, size=SIZE, dtype=torch.uint8)
        mapped.sum()  # fault the pages in as file cache
        pointer = mapped.data_ptr()
        before = anonymous_bytes()
        if not backend.register(pointer, SIZE):
            out.write("registration refused\n")
            return
        during = anonymous_bytes()
        target = mapped.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        ok = bool(torch.equal(target.cpu(), mapped))
        backend.unregister(pointer)
        after = anonymous_bytes()
        out.write(
            f"copy ok={ok}, anonymous bytes gained while pinned={during - before}, "
            f"after unregister={after - before} (mapping is {SIZE})\n"
        )
        del mapped


if __name__ == "__main__":
    main()
