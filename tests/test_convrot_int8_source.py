"""Tests for the shared behavior of sources that activate as ConvRot INT8."""

import pytest

from piper_offload._convrot_int8_source import ConvRotInt8SourceAdapter
from piper_offload.gguf_adapter import GgufAdapter
from piper_offload.trellis_adapter import TrellisAdapter


@pytest.mark.parametrize(
    ("adapter", "source_name"),
    [(GgufAdapter, "packed GGUF"), (TrellisAdapter, "trellis-coded")],
)
def test_sources_reject_permanent_updates(
    adapter: type[ConvRotInt8SourceAdapter], source_name: str
) -> None:
    with pytest.raises(ValueError, match=f"Permanent updates to {source_name} parameters"):
        adapter.validate_permanent_update()
