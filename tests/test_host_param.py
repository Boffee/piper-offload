"""Tests for ``piper_offload.host_param.HostParam``."""

from pathlib import Path

import pytest
import torch
from torch import nn

from piper_offload._dtensor import DTENSOR_AVAILABLE
from piper_offload.host_buffer import HostBuffer
from piper_offload.host_param import HostParam
from piper_offload.tensor_adapters import (
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

    def test_plain_capture_and_load(self) -> None:
        p = nn.Parameter(torch.randn(8, 16, dtype=torch.bfloat16), requires_grad=False)
        host_param = HostParam(p)
        cpu_param = host_param.make_cpu_param()
        other_cpu_param = host_param.make_cpu_param()

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
    def test_pool_pattern_allocate_and_copy(self) -> None:
        # Mirrors how HostModuleTarget uses HostParam: allocate GPU
        # storage once, then load different immutable host sources into it.
        p = nn.Parameter(torch.randn(4, 8, dtype=torch.bfloat16), requires_grad=False)
        host_param = HostParam(p)
        device = torch.device("cuda")
        gpu_state = host_param.allocate_gpu_storage(device)
        host_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = host_param.make_gpu_param(gpu_state)
        assert gpu_param.is_cuda
        assert gpu_param.shape == p.shape
        torch.cuda.synchronize()
        cpu_param = host_param.make_cpu_param()
        assert torch.equal(gpu_state.data.cpu(), cpu_param.data)
        assert torch.equal(gpu_param.cpu(), cpu_param.data)
        new_vals = torch.randn(4, 8, dtype=torch.bfloat16)
        next_host = HostParam(nn.Parameter(new_vals, requires_grad=False))
        next_host.copy_to_gpu(gpu_state, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(gpu_state.data.cpu(), new_vals)
        assert torch.equal(gpu_param.cpu(), new_vals)
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
# Frozen-only contract
# ---------------------------------------------------------------------------


class TestFrozenContract:
    """HostParam captures frozen parameters (``requires_grad=False``) only
    and yields frozen wrappers from ``make_cpu_param`` / ``make_gpu_param``."""

    def test_rejects_trainable_parameter(self) -> None:
        source = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=True)

        with pytest.raises(ValueError, match="requires_grad=False"):
            HostParam(source)

    def test_project_dtensor_rejects_trainable_target(self) -> None:
        source = HostParam(nn.Parameter(torch.randn(4, 4), requires_grad=False))
        target = nn.Parameter(torch.empty(4, 4), requires_grad=True)

        with pytest.raises(ValueError, match="requires_grad=False"):
            HostParam.project_dtensor(source, target)

    def test_frozen_source_yields_frozen_cpu_param(self) -> None:
        p = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        host_param = HostParam(p)
        assert host_param.make_cpu_param().requires_grad is False

    @CUDA
    def test_frozen_source_yields_frozen_gpu_param(self) -> None:
        p = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        host_param = HostParam(p)
        gpu_state = host_param.allocate_gpu_storage(torch.device("cuda"))
        gpu_param = host_param.make_gpu_param(gpu_state)
        assert gpu_param.requires_grad is False

    def test_frozen_view_into_mapping_retains_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "weights.bin"
        path.write_bytes(bytes(24 * 4))
        mapped = torch.from_file(str(path), shared=False, size=24, dtype=torch.float32)
        mapped.copy_(torch.arange(24, dtype=torch.float32))
        frozen = nn.Parameter(mapped[:12].reshape(3, 4), requires_grad=False)

        (frozen_host,) = HostParam(frozen).storage_tensors()

        assert frozen_host.untyped_storage().data_ptr() == mapped.untyped_storage().data_ptr()
        assert not frozen_host.untyped_storage().resizable()


class TestDTensorProjection:
    def test_project_dtensor_sets_projected_target_layout_and_supports_nd_shards(
        self,
    ) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        from torch.distributed.tensor import DTensor, Shard, init_device_mesh

        from piper_offload._dtensor import cpu_mesh_for
        from piper_offload.parallel import ParallelExecutor
        from piper_offload.tensor_adapters import RegularAdapter

        src_2d = torch.randn(4, 6, dtype=torch.bfloat16)
        src_3d = torch.arange(2 * 4 * 6, dtype=torch.bfloat16).reshape(2, 4, 6)
        host_2d = HostParam(nn.Parameter(src_2d, requires_grad=False))
        host_3d = HostParam(nn.Parameter(src_3d, requires_grad=False))
        ptr_2d = host_2d.storage_tensors()[0].untyped_storage().data_ptr()
        ptr_3d = host_3d.storage_tensors()[0].untyped_storage().data_ptr()

        def verify(rank: int) -> None:
            mesh = init_device_mesh("cpu", (2,))
            assert cpu_mesh_for(mesh) is mesh

            # Default float32 meta placeholder against bfloat16 source must record bfloat16 target_layout.
            meta_2d = DTensor.from_local(
                torch.empty((4, 3), dtype=torch.float32, device="meta"),
                mesh,
                [Shard(1)],
                shape=src_2d.shape,
                stride=src_2d.stride(),
                run_check=False,
            )
            proj_2d = HostParam.project_dtensor(host_2d, nn.Parameter(meta_2d, requires_grad=False))
            assert proj_2d.compute_dtype is torch.bfloat16
            assert proj_2d.target_layout[0] is type(select_adapter(meta_2d))
            inner_sig = proj_2d.target_layout[1][1]
            assert inner_sig == RegularAdapter.layout_signature(torch.empty((4, 3), dtype=torch.bfloat16))
            assert proj_2d.storage_tensors()[0].untyped_storage().data_ptr() == ptr_2d
            assert proj_2d.accepts_bind_target(nn.Parameter(meta_2d, requires_grad=False))
            assert proj_2d.accepts_bind_target(proj_2d.make_cpu_param())

            # 3-D Shard(1) and Shard(2) produce non-contiguous zero-copy views with exact data.
            for shard_dim, local_shape in ((1, (2, 2, 6)), (2, (2, 4, 3))):
                meta_3d = DTensor.from_local(
                    torch.empty(local_shape, dtype=torch.float32, device="meta"),
                    mesh,
                    [Shard(shard_dim)],
                    shape=src_3d.shape,
                    stride=src_3d.stride(),
                    run_check=False,
                )
                proj_3d = HostParam.project_dtensor(host_3d, nn.Parameter(meta_3d, requires_grad=False))
                (backing,) = proj_3d.storage_tensors()
                assert backing.untyped_storage().data_ptr() == ptr_3d
                assert backing.dtype is torch.bfloat16
                expected = src_3d.chunk(2, dim=shard_dim)[rank]
                torch.testing.assert_close(backing, expected, rtol=0, atol=0)

        with ParallelExecutor(["cpu", "cpu"]) as ex:
            ex.run(verify)

    def test_project_dtensor_quantized_replicate_and_rejections(self) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        torchao_quant = pytest.importorskip("torchao.quantization")
        from torch.distributed.tensor import DTensor, Replicate, Shard, init_device_mesh

        from piper_offload.dtensor_adapter import DTensorAdapter
        from piper_offload.parallel import ParallelExecutor
        from piper_offload.tensor_adapters import RegularAdapter

        linear = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16).eval().requires_grad_(False)
        torchao_quant.quantize_(linear, torchao_quant.Float8WeightOnlyConfig())
        q_host = HostParam(linear.weight)
        q_ptrs = tuple(t.untyped_storage().data_ptr() for t in q_host.storage_tensors())

        class _SubParam(nn.Parameter):
            pass

        class _ParamSubclassAdapter(RegularAdapter):
            @staticmethod
            def cpu_param(state):
                return _SubParam(state.data, requires_grad=False)

        plain_state = RegularAdapter.capture_host(torch.randn(4, 4))

        def verify(_rank: int) -> None:
            mesh = init_device_mesh("cpu", (2,))
            meta_rep = DTensor.from_local(
                torch.empty((16, 16), dtype=torch.float32, device="meta"),
                mesh,
                [Replicate()],
                run_check=False,
            )
            proj_q = HostParam.project_dtensor(q_host, nn.Parameter(meta_rep, requires_grad=False))
            assert proj_q.compute_dtype is torch.bfloat16
            assert proj_q.target_layout[1][0] is type(q_host.adapter)
            assert tuple(t.untyped_storage().data_ptr() for t in proj_q.storage_tensors()) == q_ptrs
            assert proj_q.accepts_bind_target(nn.Parameter(meta_rep, requires_grad=False))
            assert proj_q.accepts_bind_target(proj_q.make_cpu_param())

            meta_shard = DTensor.from_local(
                torch.empty((8, 16), dtype=torch.float32, device="meta"),
                mesh,
                [Shard(0)],
                shape=(16, 16),
                stride=(16, 1),
                run_check=False,
            )
            with pytest.raises(NotImplementedError, match="plain tensor sources only"):
                HostParam.project_dtensor(q_host, nn.Parameter(meta_shard, requires_grad=False))

            rep_4x4 = DTensor.from_local(
                torch.empty((4, 4), dtype=torch.float32, device="meta"),
                mesh,
                [Replicate()],
                run_check=False,
            )
            with pytest.raises(NotImplementedError, match="nn.Parameter subclass"):
                DTensorAdapter.project_host(_ParamSubclassAdapter(), plain_state, rep_4x4)

        with ParallelExecutor(["cpu", "cpu"]) as ex:
            ex.run(verify)
