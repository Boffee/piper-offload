"""Select host allocation, memory limits, and checkpoint readers once at import.

Native Windows calls remain lazy in ``_host_memory_windows``. The pin manager
owns leases and budgets; these modules supply only the OS mechanisms.
"""

import os
import sys

if sys.platform == "win32":
    from ._host_memory_windows import available_memory, new_region
else:
    from ._host_memory_linux import available_memory, new_region

# Preserve the per-worker handle fallback on systems without positional reads.
if hasattr(os, "preadv"):
    from ._host_memory_linux import Readers
else:
    from ._host_memory_windows import Readers

__all__ = ["Readers", "available_memory", "new_region"]
