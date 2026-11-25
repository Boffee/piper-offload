"""Tests for the shared behavior of sources that activate as ConvRot INT8."""

import inspect

import pytest
import torch

import piper_offload._convrot_int8_source as source_mod
import piper_offload._piper_convrot_int8 as piper_convrot_mod
from piper_offload._convrot_int8_source import (
    ConvRotInt8GpuTarget,
    ConvRotInt8SourceAdapter,
    _bucketed_scratch_bytes,
    stage_packed_tensors,
)
from piper_offload._transfer_priority import (
    _can_chunk,
    background_transfers,
    register_foreground_producer,
)
from piper_offload.gguf_adapter import GgufAdapter
from piper_offload.trellis_adapter import TrellisAdapter

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    ("adapter", "source_name"),
    [(GgufAdapter, "packed GGUF"), (TrellisAdapter, "trellis-coded")],
)
def test_sources_reject_permanent_updates(
    adapter: type[ConvRotInt8SourceAdapter], source_name: str
) -> None:
    with pytest.raises(ValueError, match=f"Permanent updates to {source_name} parameters"):
        adapter.validate_permanent_update()


def test_bucketed_scratch_bytes_bounds_rounding() -> None:
    assert _bucketed_scratch_bytes(0) == 0
    assert _bucketed_scratch_bytes(4096) == 4096
    two_mib = 2 * 1024 * 1024
    assert _bucketed_scratch_bytes(two_mib - 1) == two_mib - 1
    assert _bucketed_scratch_bytes(two_mib) == two_mib

    for exp in range(21, 29):
        exact = 1 << exp
        assert _bucketed_scratch_bytes(exact) == exact
        worst = exact + 1
        rounded = _bucketed_scratch_bytes(worst)
        assert rounded > worst
        assert rounded % (1024 * 1024) == 0
        assert (rounded - worst) * 8 <= max(worst, 8 * 1024 * 1024)
        if exp >= 23:
            assert (rounded - worst) * 8 < worst


def test_stage_packed_tensors_packs_aligned_views_via_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []
    real_transfer = source_mod.transfer_

    def _spy_transfer(
        dst: torch.Tensor, src: torch.Tensor, *, non_blocking: bool = False
    ) -> None:
        calls.append((dst, src, non_blocking))
        real_transfer(dst, src, non_blocking=non_blocking)

    monkeypatch.setattr(source_mod, "transfer_", _spy_transfer)

    first = torch.arange(13, dtype=torch.uint8)
    empty = torch.empty((0, 3), dtype=torch.float32)
    second = torch.tensor([[1.25, -2.5], [3.5, 4.0]], dtype=torch.float32)

    staged = stage_packed_tensors((first, empty, second), torch.device("cpu"), non_blocking=True)

    assert len(staged) == 3
    assert len(calls) == 3
    assert all(flag is True for _, _, flag in calls)
    assert staged[0].storage_offset() == 0
    assert staged[1].storage_offset() * staged[1].element_size() == 256
    assert staged[2].storage_offset() * staged[2].element_size() == 256
    assert staged[2].data_ptr() - staged[0].data_ptr() == 256
    assert all(view.untyped_storage()._cdata == staged[0].untyped_storage()._cdata for view in staged)
    assert all(view.is_contiguous() for view in staged)
    assert torch.equal(staged[0], first)
    assert staged[1].shape == (0, 3)
    assert staged[1].dtype is torch.float32
    assert torch.equal(staged[2], second)


def test_stage_packed_tensors_all_empty_and_rejects_non_contiguous() -> None:
    empty = stage_packed_tensors((torch.empty(0, dtype=torch.uint8),), torch.device("cpu"))
    assert empty[0].numel() == 0

    non_contig = torch.arange(16, dtype=torch.uint8).view(4, 4).t()
    with pytest.raises(ValueError, match="contiguous"):
        stage_packed_tensors((non_contig,), torch.device("cpu"))


def test_gpu_param_wraps_target_state() -> None:
    target = torch.zeros((4, 4), dtype=torch.float32)
    param = ConvRotInt8SourceAdapter.gpu_param(object(), ConvRotInt8GpuTarget(target=target))
    assert param.data.data_ptr() == target.data_ptr()
    assert not param.requires_grad


def test_supports_variable_row_trellis_caches_per_method_and_handles_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sig_calls = 0
    real_signature = inspect.signature

    def _spy_signature(obj: object) -> inspect.Signature:
        nonlocal sig_calls
        sig_calls += 1
        return real_signature(obj)  # type: ignore[arg-type]

    monkeypatch.setattr(piper_convrot_mod.inspect, "signature", _spy_signature)
    monkeypatch.setattr(piper_convrot_mod, "PIPER_CONVROT_AVAILABLE", False)
    assert not piper_convrot_mod.supports_variable_row_trellis()
    assert sig_calls == 0

    class _NoMethod:
        pass

    monkeypatch.setattr(piper_convrot_mod, "PIPER_CONVROT_AVAILABLE", True)
    monkeypatch.setattr(piper_convrot_mod, "ConvRotInt8Tensor", _NoMethod)
    assert not piper_convrot_mod.supports_variable_row_trellis()
    assert sig_calls == 0

    class _WithVariableRow:
        def copy_from_trellis_(
            self, bitstreams: object, block_scales: object, scale: object, row_widths: object = None
        ) -> None:
            pass

    monkeypatch.setattr(piper_convrot_mod, "ConvRotInt8Tensor", _WithVariableRow)
    assert piper_convrot_mod.supports_variable_row_trellis()
    assert piper_convrot_mod.supports_variable_row_trellis()
    assert sig_calls == 1

    class _UniformOnly:
        def copy_from_trellis_(
            self, bitstreams: object, block_scales: object, scale: object
        ) -> None:
            pass

    monkeypatch.setattr(piper_convrot_mod, "ConvRotInt8Tensor", _UniformOnly)
    assert not piper_convrot_mod.supports_variable_row_trellis()
    assert sig_calls == 2


@CUDA
def test_stage_packed_tensors_cuda_bucketing_and_background_chunking() -> None:
    nbytes = 2 * 1024 * 1024 + 512
    host = torch.randint(0, 256, (nbytes,), dtype=torch.uint8)
    with register_foreground_producer([torch.device("cuda")]), background_transfers():
        staged = stage_packed_tensors((host,), torch.device("cuda"), non_blocking=True)
        assert _can_chunk(staged[0], host)
        assert staged[0].untyped_storage().nbytes() >= 3 * 1024 * 1024
        torch.cuda.synchronize()
        assert torch.equal(staged[0].cpu(), host)
