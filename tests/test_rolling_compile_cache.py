"""Rolling graph disk reuse must preserve parameter lifetime and runtime ownership."""

import json
import os
import subprocess
import sys

import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _run_worker(tmp_path, mode, seed, *, reverse=False):
    result = tmp_path / f"result-{seed}.json"
    command = [sys.executable, "-m", "tests._rolling_cache_worker", str(result), mode, str(seed)]
    if reverse:
        command.append("--reverse")
    env = {
        **os.environ,
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "0",
        # These tiny graphs need no compile pool; its pass_fds path is POSIX-only.
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "OMP_NUM_THREADS": "2",
    }
    if mode.startswith("cpu"):
        env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(result.read_text(encoding="utf-8"))


@pytest.mark.parametrize("mode", [
    "cpu", "cpu-int8", pytest.param("cuda", marks=CUDA), pytest.param("cuda-int8", marks=CUDA),
])
def test_rolling_disk_cache_rebinds_runtime_after_reset_and_restart(tmp_path, mode):
    if mode.endswith("int8"):
        pytest.importorskip("piper_kernels.weights.convrot.int8")
    first = _run_worker(tmp_path, mode, seed=1)
    assert first == [
        {"hit": 0, "miss": 1, "bypass": 0},
        {"hit": 1, "miss": 0, "bypass": 0},
    ]
    # A fresh process uses different weights/allocations and an empty runtime registry.
    second = _run_worker(tmp_path, mode, seed=42)
    assert second == [
        {"hit": 1, "miss": 0, "bypass": 0},
        {"hit": 1, "miss": 0, "bypass": 0},
    ]
    if mode.startswith("cpu"):
        # Same arithmetic graph, different slot indices: the old callbacks would
        # target the wrong parameters if this mapping were omitted from the key.
        changed = _run_worker(tmp_path, mode, seed=7, reverse=True)
        assert changed == first
