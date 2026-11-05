"""Discarded real traces release inputs without waiting for cyclic GC."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch._dynamo.output_graph import DynamoTracerOutput

from piper_offload._dynamo import install_discarded_trace_cleanup


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
        ),
    ],
)
def test_discarded_trace_releases_real_input_without_gc(device):
    # Compiler logs/caches in this process must not become additional input owners.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_dynamo_cleanup_worker.py")), device],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        pytest.fail(result.stdout + result.stderr)


def test_discard_cleanup_without_output_graph():
    install_discarded_trace_cleanup()
    tracer = DynamoTracerOutput.__new__(DynamoTracerOutput)
    tracer.output_graph_for_cleanup = None
    tracer._cleanup_output_graph()
