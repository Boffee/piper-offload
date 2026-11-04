"""Discarded real traces release inputs without waiting for cyclic GC."""

import gc
import json
import types
from weakref import ref

import pytest
import torch
from torch import nn
from torch._dynamo.exc import SkipFrame, TensorifyScalarRestartAnalysis
from torch._dynamo.output_graph import DynamoTracerOutput

from piper_offload._dynamo import install_discarded_trace_cleanup
from piper_offload.block_compile import BlockCompileConfig, _BlockCompileState


def _describe_input_owners(value_ref):
    value = value_ref()
    result = []
    for owner in gc.get_referrers(value):
        row = {"type": type(owner).__module__ + "." + type(owner).__qualname__}
        if isinstance(owner, tuple):
            row["items"] = [type(item).__module__ + "." + type(item).__qualname__ for item in owner]
        if isinstance(owner, (dict, tuple)):
            if isinstance(owner, dict):
                row["input_names"] = [str(key) for key, item in owner.items() if item is value]
                row["keys"] = [str(key) for key in owner][:30]
            paths = []
            for parent in gc.get_referrers(owner):
                path = {"type": type(parent).__module__ + "." + type(parent).__qualname__}
                if isinstance(parent, types.FrameType):
                    path["frame"] = (parent.f_code.co_filename, parent.f_code.co_name, parent.f_lineno)
                elif isinstance(parent, dict):
                    path["names"] = [str(key) for key, item in parent.items() if item is owner]
                elif isinstance(parent, tuple):
                    path["items"] = [type(item).__module__ + "." + type(item).__qualname__ for item in parent]
                elif isinstance(parent, types.CellType):
                    path["cell_owners"] = [
                        type(item).__module__ + "." + type(item).__qualname__ for item in gc.get_referrers(parent)
                    ]
                paths.append(path)
            row["owners"] = paths
        elif isinstance(owner, types.FrameType):
            row["frame"] = (owner.f_code.co_filename, owner.f_code.co_name, owner.f_lineno)
        result.append(row)
    return result


class _InputBlock(nn.Module):
    def forward(self, value):
        return value.sin() + 1


@pytest.fixture
def input_block():
    # SkipFrame marks code for future calls, even with another compiler backend.
    torch._dynamo.reset_code(_InputBlock.forward.__code__)
    yield _InputBlock()
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
        # A retained callback frame must not own the input locals itself.
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

    state = _BlockCompileState.create(
        [input_block],
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
            torch.testing.assert_close(input_block(value), expected)
            assert discarded
            del value
            if value_ref() is not None:
                pytest.fail(json.dumps(_describe_input_owners(value_ref), indent=2))
            expected_compilations = 2 if discard == "restart" else 1
            assert compilations == expected_compilations
            # Successful guards still work, and warm calls do not compile again.
            for rows in (2, 3):
                value = torch.randn(rows, 8, device=device)
                torch.testing.assert_close(input_block(value), value.sin() + 1)
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
