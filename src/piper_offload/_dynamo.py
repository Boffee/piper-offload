"""Release real inputs when PyTorch discards a Dynamo tracing attempt.

PyTorch 2.14 clears ``OutputGraph.local_scope`` on successful compilation, but
leaves that owning reference in discarded restart/skip attempts. Their reference
cycles can retain an entire block input until a full garbage collection. Install
this compatibility fix only when Offload compilation is requested. It extends the
existing discard cleanup, after its consumers are finished, without collecting
garbage or changing successful traces, guards, or compiled execution.

This process-wide shim can be removed when the supported PyTorch version clears
these locals itself. Keep the private PyTorch dependency contained here.
"""

from functools import cache, wraps


@cache
def install_discarded_trace_cleanup() -> None:
    """Extend Dynamo's discarded-trace cleanup once, before compiling blocks."""
    from torch._dynamo.output_graph import DynamoTracerOutput  # noqa: PLC0415

    original_cleanup = DynamoTracerOutput._cleanup_output_graph

    @wraps(original_cleanup)
    def cleanup(self: DynamoTracerOutput) -> None:
        original_cleanup(self)
        if (graph := self.output_graph_for_cleanup) is not None:
            # Rebind the owning reference; the caller still owns its locals dict.
            graph.local_scope = {}

    DynamoTracerOutput._cleanup_output_graph = cleanup
