"""Tests for ``piper_offload.host_param.HostParam``."""

from pathlib import Path

import pytest
import torch
from torch import nn

from piper_offload.host_buffer import HostBuffer
from piper_offload.host_param import HostParam
from piper_offload.tensor_adapters import (
    DequantRequantTensorAdapter,
    LoRAMergeTensorAdapter,
    TensorCopyIntoAdapter,
    capture_host_tensor,
)
from piper_offload.tensor_adapter_registry import select_adapter, tensor_id

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _unexpected_allocation(*args: object, **kwargs: object) -> None:
    raise AssertionError(f"unexpected allocation: {args!r} {kwargs!r}")


# ---------------------------------------------------------------------------
# HostParam basic correctness
# ---------------------------------------------------------------------------


class TestHostParam:
    def test_meta_parameter_has_no_host_storage(
        self,
    ) -> None:
        source = nn.Parameter(
            torch.empty_strided(
                (3, 4),
                (1, 3),
                dtype=torch.bfloat16,
                device="meta",
            ),
            requires_grad=False,
        )

        host = HostParam(source)
        restored = host.make_cpu_param()

        assert host.is_meta
        assert host.cache_bytes == 0
        assert host.compute_dtype is torch.bfloat16
        assert restored.is_meta
        assert restored.shape == source.shape
        assert restored.stride() == source.stride()
        assert not restored.requires_grad

    @pytest.mark.parametrize(
        ("dtype", "requires_grad", "message"),
        [
            (torch.int32, False, "floating-point"),
            (torch.float32, True, "requires_grad=False"),
        ],
    )
    def test_meta_parameter_rejects_unsupported_state(
        self,
        dtype: torch.dtype,
        requires_grad: bool,
        message: str,
    ) -> None:
        source = nn.Parameter(
            torch.empty(3, 4, dtype=dtype, device="meta"),
            requires_grad=requires_grad,
        )

        with pytest.raises(ValueError, match=message):
            HostParam(source)

    def test_meta_parameter_rejects_sparse_layout(self) -> None:
        source = nn.Parameter(
            torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.int64, device="meta"),
                torch.empty(0, device="meta"),
                (2, 2),
                device="meta",
                check_invariants=False,
            ),
            requires_grad=False,
        )

        with pytest.raises(ValueError, match="strided layout"):
            HostParam(source)

    def test_meta_parameter_cannot_materialize_without_parameter_value(self) -> None:
        source = nn.Parameter(
            torch.empty(3, 4, device="meta"),
            requires_grad=False,
        )
        host = HostParam(source)

        with pytest.raises(RuntimeError, match="active parameter value"):
            host.materialize(torch.device("cpu"))

    def test_meta_target_layout_includes_stride(self) -> None:
        contiguous = nn.Parameter(
            torch.empty(2, 3, device="meta"),
            requires_grad=False,
        )
        transposed = nn.Parameter(
            torch.empty(3, 2, device="meta").t(),
            requires_grad=False,
        )

        assert contiguous.shape == transposed.shape
        assert contiguous.stride() != transposed.stride()
        assert HostParam.target_layout_for(contiguous) != HostParam.target_layout_for(transposed)

    def test_meta_target_layout_includes_storage_offset(self) -> None:
        backing = torch.empty(12, device="meta")
        first = nn.Parameter(
            backing.as_strided((2, 3), (3, 1), 1),
            requires_grad=False,
        )
        second = nn.Parameter(
            backing.as_strided((2, 3), (3, 1), 2),
            requires_grad=False,
        )

        assert first.shape == second.shape
        assert first.stride() == second.stride()
        assert first.storage_offset() != second.storage_offset()
        assert HostParam.target_layout_for(first) != HostParam.target_layout_for(second)

    def test_capture_host_tensor_retains_complete_cpu_storage(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        monkeypatch.setattr(torch, "empty_like", _unexpected_allocation)

        host = capture_host_tensor(source)

        assert host.data_ptr() == source.data_ptr()
        assert not host.is_pinned()
        assert host.stride() == source.stride()
        torch.testing.assert_close(host, source)

    def test_capture_host_tensor_normalizes_partial_cpu_storage(self) -> None:
        source = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]

        host = capture_host_tensor(source)

        assert host.data_ptr() != source.data_ptr()
        assert host.untyped_storage().nbytes() == host.numel() * host.element_size()
        assert host.is_contiguous()
        torch.testing.assert_close(host, source)

    def test_capture_host_tensor_retains_mapped_split_view(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "split.bin"
        path.touch()
        mapped = torch.from_file(
            str(path),
            shared=True,
            size=24,
            dtype=torch.float32,
        )
        mapped.copy_(torch.arange(24, dtype=torch.float32))
        source = mapped[6:18].reshape(3, 4)
        monkeypatch.setattr(torch, "empty_like", _unexpected_allocation)

        host = capture_host_tensor(source)

        assert host.data_ptr() == source.data_ptr()
        assert host.untyped_storage().data_ptr() == mapped.untyped_storage().data_ptr()
        assert host.storage_offset() == 6
        assert not host.untyped_storage().resizable()
        torch.testing.assert_close(host, source)

    @CUDA
    def test_capture_host_tensor_preserves_cuda_source_stride(self) -> None:
        source = torch.arange(24, device="cuda").reshape(4, 6)[:, ::2]

        host = capture_host_tensor(source)

        assert host.stride() == source.stride()
        assert not host.is_pinned()
        torch.testing.assert_close(host, source.cpu())

    @CUDA
    def test_capture_host_tensor_normalizes_pinned_cpu_storage(self) -> None:
        source = torch.arange(24, dtype=torch.float32).pin_memory()

        host = capture_host_tensor(source)

        assert host.data_ptr() != source.data_ptr()
        assert not host.is_pinned()
        torch.testing.assert_close(host, source)

    def test_select_adapter_returns_adapter_for_plain_tensor(self) -> None:
        first = select_adapter(torch.randn(1))
        second = select_adapter(torch.randn(2))

        assert type(first) is type(second)

    def test_regular_tensor_id_includes_device(self) -> None:
        t = torch.randn(2, 3)

        assert tensor_id(t)[:2] == ("regular", t.device)

    def test_non_quanto_capture_and_load(self) -> None:
        p = nn.Parameter(torch.randn(8, 16, dtype=torch.bfloat16), requires_grad=False)
        host_param = HostParam(p)
        cpu_param = host_param.make_cpu_param()
        other_cpu_param = host_param.make_cpu_param()

        # make_cpu_param wraps a plain host tensor — no quanto subclass.
        assert type(cpu_param.data) is torch.Tensor
        assert not cpu_param.data.is_pinned()
        assert cpu_param.data.shape == p.shape
        assert host_param.shape == p.shape
        # CPU params are distinct wrappers over the same host buffer,
        # not a second clone — callers replace registry entries with this and rely on
        # the storage staying alive for the host parameter's lifetime.
        assert cpu_param is not other_cpu_param
        assert cpu_param.data.data_ptr() == host_param.host_state.data.data_ptr()
        assert other_cpu_param.data.data_ptr() == host_param.host_state.data.data_ptr()
        # Low-peak construction repoints the source Parameter to the
        # host backing without making HostParam own that wrapper.
        assert p.data.data_ptr() == host_param.host_state.data.data_ptr()

    @CUDA
    def test_allocate_copy_make_gpu_param_non_quanto(self) -> None:
        p = nn.Parameter(torch.randn(4, 8, dtype=torch.bfloat16), requires_grad=False)
        host_param = HostParam(p)
        gpu_state = host_param.allocate_gpu_storage(torch.device("cuda"))
        host_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu = host_param.make_gpu_param(gpu_state)
        assert gpu.is_cuda
        assert gpu.shape == p.shape
        torch.cuda.synchronize()
        assert torch.equal(gpu.cpu(), host_param.make_cpu_param().data)

    @CUDA
    def test_pool_pattern_allocate_and_copy(self) -> None:
        # Mirrors how HostModuleTarget uses HostParam: allocate GPU
        # storage once, then copy_to_gpu in place on each load.
        p = nn.Parameter(torch.randn(16, dtype=torch.bfloat16), requires_grad=False)
        host_param = HostParam(p)
        device = torch.device("cuda")
        gpu_state = host_param.allocate_gpu_storage(device)
        gpu_param = host_param.make_gpu_param(gpu_state)
        assert gpu_param.is_cuda
        # First copy
        host_param.copy_to_gpu(gpu_state, non_blocking=True)
        torch.cuda.synchronize()
        cpu_param = host_param.make_cpu_param()
        assert torch.equal(gpu_state.data.cpu(), cpu_param.data)
        # Mutate host source and re-copy — gpu state should track.
        new_vals = torch.randn(16, dtype=torch.bfloat16)
        cpu_param.data.copy_(new_vals)
        host_param.copy_to_gpu(gpu_state, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(gpu_state.data.cpu(), new_vals)
        # Stable storage — gpu_param wraps the same GPU bytes as gpu_state.
        # HostModuleTarget relies on this: build the Parameter wrapper once at target
        # construction, mutate underlying storage in place on each load.
        assert gpu_param.data_ptr() == gpu_state.data.data_ptr()

    def test_contiguous_format_forced(self) -> None:
        # A view of a transposed tensor is non-contiguous. clone() with
        # contiguous_format normalizes it; host data must be 1-D
        # contiguous so downstream callers can rely on it.
        base = torch.randn(8, 16, dtype=torch.bfloat16)
        non_contig = base.t()
        assert not non_contig.is_contiguous()
        p = nn.Parameter(non_contig, requires_grad=False)
        host_param = HostParam(p)
        cpu_param = host_param.make_cpu_param()
        assert cpu_param.data.is_contiguous()
        assert not cpu_param.data.is_pinned()

class TestHostBuffer:
    def test_target_layout_matches_host_tensor_layout(self) -> None:
        source = torch.randn(2, 3).t()

        host = HostBuffer.capture(source)

        assert not source.is_contiguous()
        assert host.tensor.is_contiguous()
        assert host.target_layout == HostBuffer.target_layout_for(
            host.tensor,
        )


# ---------------------------------------------------------------------------
# Quanto path — only nontrivial branch in HostParam
# ---------------------------------------------------------------------------


class TestHostParamQuanto:
    def test_quanto_tensor_id_includes_inner_devices(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )

        key = tensor_id(qt)
        assert key[1] == qt._data.device
        assert key[7] == qt._scale.device

    def test_quanto_adapter_conversion_and_merge_capabilities(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.optimizers import AbsmaxOptimizer
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )

        adapter = select_adapter(qt)
        assert isinstance(adapter, DequantRequantTensorAdapter)
        assert isinstance(adapter, LoRAMergeTensorAdapter)
        assert isinstance(adapter, TensorCopyIntoAdapter)
        dense = adapter.dequantize(qt)
        assert type(dense) is torch.Tensor
        assert dense.dtype == qt.dtype
        assert dense.shape == qt.shape

        expected_scale = AbsmaxOptimizer()(
            dense,
            qtype=quanto.qint8,
            axis=0,
        ).to(scale.dtype)
        expected = WeightQBytesTensor.quantize(
            dense,
            quanto.qint8,
            0,
            expected_scale,
            optimized=False,
        )
        requantized = adapter.requantize(dense, like=qt)
        assert isinstance(requantized, WeightQBytesTensor)
        assert requantized.qtype is quanto.qint8
        assert requantized.axis == 0
        assert tuple(requantized.size()) == (rows, cols)
        torch.testing.assert_close(requantized._data, expected._data)
        torch.testing.assert_close(requantized._scale, expected._scale)

        updated = dense + 1
        updated_scale = AbsmaxOptimizer()(
            updated,
            qtype=quanto.qint8,
            axis=0,
        ).to(scale.dtype)
        expected_updated = WeightQBytesTensor.quantize(
            updated,
            quanto.qint8,
            0,
            updated_scale,
            optimized=False,
        )
        updated_qt = adapter.requantize(updated, like=qt)
        original_scale_ptr = qt._scale.data_ptr()
        adapter.copy_into(updated_qt, target=qt)
        torch.testing.assert_close(qt._data, expected_updated._data)
        torch.testing.assert_close(qt._scale, expected_updated._scale)
        assert qt._scale.data_ptr() == original_scale_ptr

    def test_capture_decomposes_data_and_scale(self) -> None:
        # Quanto WeightQBytesTensor must be decomposed into _data + _scale
        # and the CPU wrapper reconstructed from the host tensors.
        # A naive tensor.clone() would silently dequantize via the dispatch
        # fallback — that bug is the reason host_param.py exists.
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )
        p = nn.Parameter(qt, requires_grad=False)
        host_param = HostParam(p)

        # make_cpu_param wraps a quanto tensor whose _data and _scale are
        # host, contiguous, and carry the original quant metadata.
        cpu_param = host_param.make_cpu_param()
        assert isinstance(cpu_param.data, WeightQBytesTensor)
        qt_host = cpu_param.data
        assert not qt_host._data.is_pinned()
        assert qt_host._data.is_contiguous()
        assert qt_host._data.dtype == torch.int8
        assert not qt_host._scale.is_pinned()
        assert qt_host.qtype is quanto.qint8
        assert qt_host.axis == 0
        assert tuple(qt_host.size()) == (rows, cols)
        assert qt_host.stride() == (cols, 1)
        assert getattr(qt_host, "activation_qtype", None) is None
        # Zero-copy: the wrapper's _data and _scale point at the same
        # host buffers the adapter host, not separate clones.
        assert qt_host._data.data_ptr() == host_param.host_state.data.data_ptr()
        assert qt_host._scale.data_ptr() == host_param.host_state.scale.data_ptr()

    @CUDA
    def test_allocate_copy_make_gpu_param_quanto_round_trip(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )
        p = nn.Parameter(qt, requires_grad=False)
        host_param = HostParam(p)

        gpu_state = host_param.allocate_gpu_storage(torch.device("cuda"))
        host_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = host_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()
        assert isinstance(gpu_param.data, WeightQBytesTensor)
        assert gpu_param.data._data.is_cuda
        assert gpu_param.data._scale.is_cuda
        qt_host = host_param.make_cpu_param().data
        assert torch.equal(gpu_param.data._data.cpu(), qt_host._data)
        assert torch.equal(gpu_param.data._scale.cpu(), qt_host._scale)


# ---------------------------------------------------------------------------
# requires_grad propagation through the wrapper builders
# ---------------------------------------------------------------------------


class TestRequiresGradPropagation:
    """The host parameter captures the source param's ``requires_grad`` at
    construction time and threads it through to ``make_cpu_param`` /
    ``gpu_param``. Frozen sources get historic ``requires_grad=False``
    wrappers; trainable sources get ``True`` wrappers so consumers
    that DO use the wrapper objects (rather than ``.data``-swapping)
    see the right autograd flag."""

    def test_frozen_source_yields_frozen_cpu_param(self) -> None:
        p = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        host_param = HostParam(p)
        assert host_param.requires_grad is False
        assert host_param.make_cpu_param().requires_grad is False

    def test_trainable_source_yields_trainable_cpu_param(self) -> None:
        p = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=True)
        host_param = HostParam(p)
        assert host_param.requires_grad is True
        assert host_param.make_cpu_param().requires_grad is True

    @CUDA
    def test_trainable_source_yields_trainable_gpu_param(self) -> None:
        p = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=True)
        host_param = HostParam(p)
        gpu_state = host_param.allocate_gpu_storage(torch.device("cuda"))
        gpu_param = host_param.make_gpu_param(gpu_state)
        assert gpu_param.requires_grad is True

    @CUDA
    def test_frozen_source_yields_frozen_gpu_param(self) -> None:
        p = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        host_param = HostParam(p)
        gpu_state = host_param.allocate_gpu_storage(torch.device("cuda"))
        gpu_param = host_param.make_gpu_param(gpu_state)
        assert gpu_param.requires_grad is False


# ---------------------------------------------------------------------------
# copy_to_cpu — D2H counterpart to copy_to_gpu
# ---------------------------------------------------------------------------


class TestCopyToCpu:
    """Symmetric D2H of GPU contents back into the host state.
    Used at the optimizer-step boundary in trainable streaming: GPU
    weights got updated in place by ``optimizer.step()``, scatter the
    update back to the host clone so the next H2D reads it."""

    @CUDA
    def test_regular_round_trip(self) -> None:
        p = nn.Parameter(torch.randn(16, dtype=torch.bfloat16), requires_grad=False)
        original = p.data.clone()
        host_param = HostParam(p)
        device = torch.device("cuda")
        gpu_state = host_param.allocate_gpu_storage(device)
        host_param.copy_to_gpu(gpu_state, non_blocking=True)
        torch.cuda.synchronize()

        # Mutate GPU side as if optimizer.step had run there.
        new_vals_gpu = torch.randn(16, dtype=torch.bfloat16, device=device)
        gpu_state.data.copy_(new_vals_gpu)
        torch.cuda.synchronize()

        # D2H should overwrite the host state with the new GPU contents.
        host_param.copy_to_cpu(gpu_state, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(host_param.host_state.data, new_vals_gpu.cpu())
        # The host state has actually changed from the original.
        assert not torch.equal(host_param.host_state.data, original)

    @CUDA
    def test_regular_host_tensor_identity_preserved(self) -> None:
        # The host buffer stays at the same address after D2H —
        # we're overwriting in place, not allocating a new tensor.
        # Callers that hold CPU Parameter wrapper references rely on this.
        p = nn.Parameter(torch.randn(16, dtype=torch.bfloat16), requires_grad=True)
        host_param = HostParam(p)
        original_ptr = host_param.host_state.data.data_ptr()
        gpu_state = host_param.allocate_gpu_storage(torch.device("cuda"))
        host_param.copy_to_gpu(gpu_state)
        host_param.copy_to_cpu(gpu_state)
        torch.cuda.synchronize()
        assert host_param.host_state.data.data_ptr() == original_ptr

    @CUDA
    def test_quanto_round_trip(self) -> None:
        # Quanto D2H must write back BOTH _data (int8) and _scale, and
        # the quant metadata on the host wrapper must be unchanged.
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )
        p = nn.Parameter(qt, requires_grad=False)
        host_param = HostParam(p)
        device = torch.device("cuda")
        gpu_state = host_param.allocate_gpu_storage(device)
        host_param.copy_to_gpu(gpu_state)
        torch.cuda.synchronize()

        # Mutate GPU-side _data and _scale in place.
        new_data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8, device=device)
        new_scale = torch.rand(rows, 1, dtype=torch.bfloat16, device=device)
        gpu_state.data.copy_(new_data)
        gpu_state.scale.copy_(new_scale)
        torch.cuda.synchronize()

        host_param.copy_to_cpu(gpu_state)
        torch.cuda.synchronize()
        assert torch.equal(host_param.host_state.data, new_data.cpu())
        assert torch.equal(host_param.host_state.scale, new_scale.cpu())
        # Metadata untouched — qtype, axis, etc. live on the host
        # state and are not part of the GPU representation.
        assert host_param.host_state.qtype is quanto.qint8
        assert host_param.host_state.axis == 0

    @CUDA
    def test_gguf_lacks_cpu_round_trip_capability(self) -> None:
        # GGUF stores packed source bytes on CPU but a different ConvRot
        # representation on GPU. D2H would require a reverse conversion,
        # which isn't implemented. Surface it as NotImplementedError, not a
        # silent corruption of the host packed bytes.
        gguf = pytest.importorskip("gguf")
        from tests._gguf_helpers import GGUFParameter

        # Build minimal GGUF state directly via the adapter — avoids
        # needing a real .gguf file to load. Q4_0 has the simplest
        # block layout and is broadly supported.
        qt_value = int(gguf.GGMLQuantizationType.Q4_0)
        # Q4_0 packs 32 fp16 weights into an 18-byte block (2 scale + 16 quants).
        packed = torch.zeros((1, 36), dtype=torch.uint8)
        gguf_t = GGUFParameter(packed, quant_type=qt_value)
        host_param = HostParam(gguf_t)
        gpu_state = host_param.allocate_gpu_storage(torch.device("cuda"))
        with pytest.raises(NotImplementedError, match="CPU round-trip"):
            host_param.copy_to_cpu(gpu_state)
