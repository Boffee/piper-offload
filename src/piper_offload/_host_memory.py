"""Select the host memory component once at import; native calls remain lazy.

Pinning mechanisms are not probed at runtime. Linux copies use anonymous
mappings and Windows copies use ``VirtualAlloc`` regions. File fills use
positional reads where available, otherwise one handle per worker. Allocation
and reader contracts live in the selected platform module; copy ownership and
preparation are defined by :mod:`piper_offload._copy_memory`.
"""

import os
import sys

if sys.platform == "win32":
    from ._copy_memory_windows import Memory
    from ._host_memory_windows import available_memory
else:
    from ._host_memory_linux import Memory, available_memory

    # Preserve per-worker handles on systems without positional reads.
    if not hasattr(os, "preadv"):
        from ._host_memory_windows import Readers

        Memory.readers = Readers

__all__ = ["Memory", "available_memory"]
