"""Discarded real traces release inputs without waiting for cyclic GC."""

import gc
import logging
from weakref import ref

import pytest
import torch
from torch import nn
from torch._dynamo.exc import SkipFrame, TensorifyScalarRestartAnalysis
from torch._dynamo.output_graph import DynamoTracerOutput

from piper_offload._dynamo import install_discarded_trace_cleanup
from piper_offload.block_compile import BlockCompileConfig, _BlockCompileState


class _InputBlock(nn.Module):
    def forward(self, value):
        return value.sin() + 1


@pytest.fixture
def input_block():
    # SkipFrame marks code for future calls, even with another compiler backend.
    torch._dynamo.reset_code(_InputBlock.forward.__code__)
    # Pytest retains compiler diagnostic records, including exception call frames.
    # Their argument tuples are separate owners from the discarded trace under test.
    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield _InputBlock()
    finally:
        logging.disable(previous_logging_disable)
        torch._dynamo.reset_code(_InputBlock.forward.__code__)


@pytest.mark.parametrize("discard", ["restart", "skip"])
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
def test_discarded_trace_releases_real_input_without_gc(monkeypatch, discard, device, input_block):
    install_discarded_trace_cleanup()
    cleanup = DynamoTracerOutput._cleanup_output_graph
    # Repeated block setup must not stack process-wide wrappers.
    install_discarded_trace_cleanup()
    assert DynamoTracerOutput._cleanup_output_graph is cleanup
    discarded = []

    def observe_cleanup(self):
        graph = self.output_graph_for_cleanup
        original_scope = None if graph is None else graph.local_scope
        keys = None if original_scope is None else set(original_scope)
        cleanup(self)
        if graph is not None:
            assert set(original_scope) == keys  # never mutate caller-owned locals
            assert graph.local_scope == {}
            assert not graph.guards
            discarded.append(graph)  # keep the trace alive to test input ownership
        del original_scope

    monkeypatch.setattr(DynamoTracerOutput, "_cleanup_output_graph", observe_cleanup)
    compilations = 0

    def backend(graph, example_inputs):
        nonlocal compilations
        compilations += 1
        if compilations == 1:
            if discard == "restart":
                raise TensorifyScalarRestartAnalysis("exercise discarded trace")
            raise SkipFrame("exercise skipped trace")
        return graph.forward

    block = input_block
    state = _BlockCompileState.create(
        [block],
        BlockCompileConfig(fullgraph=discard != "skip"),
        backend=backend,
    )
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        state.install(True)
        with torch.inference_mode():
            value = torch.randn(2, 8, device=device)
            value_ref = ref(value)
            expected = value.sin() + 1
            torch.testing.assert_close(block(value), expected)
            assert discarded
            del value
            assert value_ref() is None
            expected_compilations = 2 if discard == "restart" else 1
            assert compilations == expected_compilations
            # Successful guards still work, and warm calls do not compile again.
            for rows in (2, 3):
                value = torch.randn(rows, 8, device=device)
                torch.testing.assert_close(block(value), value.sin() + 1)
            assert compilations == expected_compilations
    finally:
        state.restore()
        if was_enabled:
            gc.enable()


def test_discard_cleanup_without_output_graph():
    install_discarded_trace_cleanup()
    tracer = DynamoTracerOutput.__new__(DynamoTracerOutput)
    tracer.output_graph_for_cleanup = None
    tracer._cleanup_output_graph()
