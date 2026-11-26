"""Tests for ``piper_offload.host_param.HostParam``."""

from pathlib import Path
from typing import Any

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
        convrot_mod = pytest.importorskip("piper_kernels.weights.convrot.int8")
        from torch.distributed.tensor import DTensor, Replicate, Shard, init_device_mesh

        from piper_offload import TrellisParameter
        from piper_offload.dtensor_adapter import DTensorAdapter
        from piper_offload.parallel import ParallelExecutor
        from piper_offload.tensor_adapters import RegularAdapter

        convrot_w = convrot_mod.ConvRotInt8Tensor.from_hp(
            torch.randn(16, 32, dtype=torch.bfloat16),
            group_size=16,
        )
        q_host = HostParam(nn.Parameter(convrot_w, requires_grad=False))
        q_ptrs = tuple(t.untyped_storage().data_ptr() for t in q_host.storage_tensors())

        trellis_param = TrellisParameter(
            torch.zeros((16, 1, 16), dtype=torch.uint8),
            torch.ones((16, 1), dtype=torch.float8_e4m3fn),
            torch.ones((16, 1)),
            block_size=64,
            group_size=16,
        )
        trellis_host = HostParam(trellis_param)

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
                torch.empty((16, 32), dtype=torch.float32, device="meta"),
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
                torch.empty((8, 32), dtype=torch.float32, device="meta"),
                mesh,
                [Shard(0)],
                shape=(16, 32),
                stride=(32, 1),
                run_check=False,
            )
            proj_q_s0 = HostParam.project_dtensor(q_host, nn.Parameter(meta_shard, requires_grad=False))
            assert proj_q_s0.compute_dtype is torch.bfloat16
            assert proj_q_s0.target_layout[1][0] is type(q_host.adapter)
            assert tuple(t.untyped_storage().data_ptr() for t in proj_q_s0.storage_tensors()) == q_ptrs
            assert proj_q_s0.accepts_bind_target(nn.Parameter(meta_shard, requires_grad=False))
            assert proj_q_s0.accepts_bind_target(proj_q_s0.make_cpu_param())

            meta_shard1 = DTensor.from_local(
                torch.empty((16, 16), dtype=torch.float32, device="meta"),
                mesh,
                [Shard(1)],
                shape=(16, 32),
                stride=(32, 1),
                run_check=False,
            )
            proj_q_s1 = HostParam.project_dtensor(q_host, nn.Parameter(meta_shard1, requires_grad=False))
            assert proj_q_s1.compute_dtype is torch.bfloat16
            assert proj_q_s1.target_layout[1][0] is type(q_host.adapter)
            assert tuple(t.untyped_storage().data_ptr() for t in proj_q_s1.storage_tensors()) == q_ptrs
            assert proj_q_s1.accepts_bind_target(nn.Parameter(meta_shard1, requires_grad=False))
            assert proj_q_s1.accepts_bind_target(proj_q_s1.make_cpu_param())

            with pytest.raises(NotImplementedError, match="unsupported"):
                HostParam.project_dtensor(trellis_host, nn.Parameter(meta_shard, requires_grad=False))

            rep_4x4 = DTensor.from_local(
                torch.empty((4, 4), dtype=torch.float32, device="meta"),
                mesh,
                [Replicate()],
                run_check=False,
            )
            with pytest.raises(NotImplementedError, match="nn.Parameter subclass"):
                DTensorAdapter.project_host(_ParamSubclassAdapter(), plain_state, rep_4x4)

            shard_4x4 = DTensor.from_local(
                torch.empty((2, 4), dtype=torch.float32, device="meta"),
                mesh,
                [Shard(0)],
                shape=(4, 4),
                stride=(4, 1),
                run_check=False,
            )
            with pytest.raises(NotImplementedError, match="nn.Parameter subclass"):
                DTensorAdapter.project_host(_ParamSubclassAdapter(), plain_state, shard_4x4)

        with ParallelExecutor(["cpu", "cpu"]) as ex:
            ex.run(verify)

def _make_meta_shard_dtensor(
    mesh: object,
    global_shape: tuple[int, ...],
    local_shape: tuple[int, ...],
    placements: list[object],
) -> nn.Parameter:
    from torch.distributed.tensor import DTensor

    stride = tuple(
        1 if i == len(global_shape) - 1 else int(torch.tensor(global_shape[i + 1 :]).prod().item())
        for i in range(len(global_shape))
    )
    dt = DTensor.from_local(
        torch.empty(local_shape, device="meta"),
        mesh,
        placements,
        shape=global_shape,
        stride=stride,
        run_check=False,
    )
    return nn.Parameter(dt, requires_grad=False)


def _build_convrot_nvfp4_high_first(rows: int = 128, cols: int = 128, group_size: int = 64) -> torch.Tensor:
    nvfp4_mod = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    convrot_mod = pytest.importorskip("piper_kernels.weights.convrot.nvfp4")
    nv_w = torch.randn(rows, cols, dtype=torch.bfloat16)
    cr_raw = convrot_mod.ConvRotNVFP4Tensor.from_hp(
        nv_w,
        group_size=group_size,
        compute_per_tensor_scale=True,
        is_swizzled_scales=True,
        use_triton_kernel=False,
        act_quant_kwargs=nvfp4_mod.QuantizeTensorToNVFP4Kwargs(
            block_size=16, is_swizzled_scales=True, use_dynamic_per_tensor_scale=True, use_triton_kernel=False
        ),
    )
    return convrot_mod.ConvRotNVFP4Tensor(
        (((cr_raw.qdata & 0x0F) << 4) | (cr_raw.qdata >> 4)).to(torch.uint8),
        cr_raw.scale,
        cr_raw.block_size,
        cr_raw.orig_dtype,
        cr_raw.group_size,
        cr_raw.per_tensor_scale,
        torch.tensor(0.25, dtype=torch.float32),
        cr_raw.is_swizzled_scales,
        cr_raw.use_triton_kernel,
        cr_raw.act_quant_kwargs,
        high_first=True,
    )


def _reconstruct_projected_cpu(adapter_cls: Any, state: Any) -> torch.Tensor:
    slots: list[torch.Tensor | None] = []
    targets = state.target_shapes
    for idx, tensor in enumerate(state.storage):
        if tensor is None:
            slots.append(None)
            continue
        t_shape = None if targets is None else targets[idx]
        contiguous = tensor.contiguous()
        slots.append(contiguous if t_shape is None else contiguous.view(t_shape))
    return adapter_cls._reconstruct(tuple(slots), state.meta)


def _assert_unbound_projected_merge_rejected(projected_param: nn.Parameter) -> None:
    from piper_offload import Adapter, ParameterDelta, merge_adapter

    unbound = nn.Module()
    unbound.w = projected_param
    delta = Adapter(
        {"w": ParameterDelta.from_tensors(dense=torch.randn(128, 128, dtype=torch.bfloat16) * 0.02)}
    )
    with pytest.raises(ValueError, match="Permanent updates to projected quantized shards are unsupported"):
        merge_adapter(unbound, [(delta, 0.5)])


def _assert_nvfp4_and_convrot_rejections(piper_nv: Any, cr_mod: Any) -> None:
    from dataclasses import replace

    from piper_offload.piper_convrot_nvfp4_adapter import PiperConvRotNVFP4Adapter
    from piper_offload.piper_nvfp4_adapter import PiperNvfp4Adapter
    from piper_offload.structured_adapter import StructuredHost

    nv_swiz = PiperNvfp4Adapter.capture_host(
        piper_nv.PiperNVFP4Tensor.from_hp(torch.randn(256, 128, dtype=torch.bfloat16), is_swizzled_scales=True)
    )
    nv_unswiz = PiperNvfp4Adapter.capture_host(
        piper_nv.PiperNVFP4Tensor.from_hp(torch.randn(64, 64, dtype=torch.bfloat16), is_swizzled_scales=False)
    )
    with pytest.raises(ValueError, match="empty shard"):
        PiperNvfp4Adapter.project_shard(nv_swiz, offsets=(0, 0), local_shape=(0, 128))
    with pytest.raises(ValueError, match="block_size must be positive"):
        bad_bs = StructuredHost(
            storage=nv_unswiz.storage,
            meta=replace(nv_unswiz.meta, block_size=0),
            logical_shape=nv_unswiz.logical_shape,
        )
        PiperNvfp4Adapter.project_shard(bad_bs, offsets=(0, 0), local_shape=(64, 64))
    with pytest.raises(ValueError, match="even for packed FP4"):
        PiperNvfp4Adapter.project_shard(nv_unswiz, offsets=(0, 1), local_shape=(64, 15))
    bad_layout = StructuredHost(
        storage=(nv_swiz.storage[0].t(), nv_swiz.storage[1], nv_swiz.storage[2], nv_swiz.storage[3]),
        meta=nv_swiz.meta,
        logical_shape=nv_swiz.logical_shape,
    )
    with pytest.raises(ValueError, match="contiguous"):
        PiperNvfp4Adapter.project_shard(bad_layout, offsets=(0, 0), local_shape=(128, 128))
    with pytest.raises(ValueError, match="block_size"):
        PiperNvfp4Adapter.project_shard(nv_swiz, offsets=(0, 8), local_shape=(256, 8))
    with pytest.raises(ValueError, match=r"swizzled Shard\(0\)"):
        PiperNvfp4Adapter.project_shard(nv_swiz, offsets=(0, 0), local_shape=(64, 128))
    with pytest.raises(ValueError, match=r"swizzled Shard\(1\)"):
        PiperNvfp4Adapter.project_shard(nv_swiz, offsets=(0, 0), local_shape=(256, 32))

    b4 = piper_nv.PiperNVFP4Tensor.from_hp(torch.randn(4, 100, 128, dtype=torch.bfloat16), is_swizzled_scales=True)
    p4_state = PiperNvfp4Adapter.capture_host(piper_nv.PiperNVFP4Tensor(
        b4.qdata.view(2, 2, 100, 64), b4.scale.view(2, 2, *b4.scale.shape[1:]), b4.block_size, b4.orig_dtype,
        None, None, True, False, None,
    ))
    with pytest.raises(ValueError, match="only supports 2-D and 3-D"):
        PiperNvfp4Adapter.project_shard(p4_state, offsets=(1, 0, 0, 0), local_shape=(1, 2, 100, 128))

    for bad_pts in (torch.tensor([1.0]), torch.tensor([[1.0]]), torch.ones(4)):
        bad_nv = StructuredHost(
            storage=(b4.qdata, b4.scale, bad_pts, None), meta=p4_state.meta, logical_shape=(4, 100, 128)
        )
        with pytest.raises(ValueError, match="per_tensor_scale shape"):
            PiperNvfp4Adapter.project_shard(bad_nv, offsets=(0, 0, 0), local_shape=(2, 100, 128))

    bad_act = StructuredHost(
        storage=(nv_swiz.storage[0], nv_swiz.storage[1], None, torch.ones(2, 1, 1)),
        meta=nv_swiz.meta,
        logical_shape=nv_swiz.logical_shape,
    )
    with pytest.raises(ValueError, match="act_per_tensor_scale shape"):
        PiperNvfp4Adapter.project_shard(bad_act, offsets=(0, 0), local_shape=(128, 128))

    cr = PiperConvRotNVFP4Adapter.capture_host(
        cr_mod.ConvRotNVFP4Tensor.from_hp(
            torch.randn(256, 128, dtype=torch.bfloat16),
            group_size=64,
            is_swizzled_scales=False,
        )
    )
    for offsets, local_shape in (
        ((0, 16), (256, 32)),
        ((0, 0), (256, 32)),
        ((0, 32), (256, 64)),
    ):
        with pytest.raises(ValueError, match="rotation group_size"):
            PiperConvRotNVFP4Adapter.project_shard(cr, offsets=offsets, local_shape=local_shape)
    with pytest.raises(ValueError, match="group_size must be positive"):
        bad_cr = StructuredHost(storage=cr.storage, meta=replace(cr.meta, group_size=0), logical_shape=cr.logical_shape)
        PiperConvRotNVFP4Adapter.project_shard(bad_cr, offsets=(0, 0), local_shape=(256, 64))
    with pytest.raises(ValueError, match="only supports 2-D"):
        PiperConvRotNVFP4Adapter.project_shard(
            StructuredHost(storage=cr.storage, meta=cr.meta, logical_shape=(2, 128, 128)),
            offsets=(0, 0, 0),
            local_shape=(1, 128, 128),
        )
    with pytest.raises(ValueError, match="negative dimensions"):
        PiperNvfp4Adapter._finalize_projected_shard(
            nv_swiz.storage, nv_swiz.meta, nv_swiz.logical_shape, target_shapes=((-256, -64), None, None, None)
        )


class TestDTensorProjectionPacked:
    def test_project_dtensor_piper_nvfp4_and_convrot_nvfp4_shards(self) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        pytest.importorskip("numpy")
        piper_nv = pytest.importorskip("piper_kernels.weights.nvfp4")
        from torch.distributed.tensor import Shard, init_device_mesh

        from piper_offload import requires_activation
        from piper_offload.parallel import ParallelExecutor

        torch.manual_seed(7)
        nv_unswiz = piper_nv.PiperNVFP4Tensor.from_hp(
            torch.randn(64, 64, dtype=torch.bfloat16), is_swizzled_scales=False
        )
        nv_hp = HostParam(nv_unswiz)
        nv_ptrs = tuple(t.untyped_storage().data_ptr() for t in nv_hp.storage_tensors())

        nv_swiz = piper_nv.PiperNVFP4Tensor.from_hp(
            torch.randn(256, 256, dtype=torch.bfloat16), is_swizzled_scales=True
        )
        swiz_hp = HostParam(nv_swiz)
        swiz_ptrs = tuple(t.untyped_storage().data_ptr() for t in swiz_hp.storage_tensors())

        cr_hp = HostParam(_build_convrot_nvfp4_high_first())
        cr_ptrs = tuple(t.untyped_storage().data_ptr() for t in cr_hp.storage_tensors())

        def verify(rank: int) -> None:
            mesh = init_device_mesh("cpu", (2,))
            meta_s0 = _make_meta_shard_dtensor(mesh, (64, 64), (32, 64), [Shard(0)])
            proj_s0 = HostParam.project_dtensor(nv_hp, meta_s0).make_cpu_param()
            assert not requires_activation(proj_s0)
            s0_loc = proj_s0.to_local()
            assert tuple(t.untyped_storage().data_ptr() for t in (s0_loc.qdata, s0_loc.scale)) == nv_ptrs
            torch.testing.assert_close(
                s0_loc.dequantize(), nv_unswiz.dequantize().chunk(2, dim=0)[rank], rtol=0, atol=0
            )

            meta_s1 = _make_meta_shard_dtensor(mesh, (64, 64), (64, 32), [Shard(1)])
            proj_s1_hp = HostParam.project_dtensor(nv_hp, meta_s1)
            assert requires_activation(proj_s1_hp.make_cpu_param())
            assert tuple(t.untyped_storage().data_ptr() for t in proj_s1_hp.storage_tensors()) == nv_ptrs

            pm0 = _make_meta_shard_dtensor(mesh, (256, 256), (128, 256), [Shard(0)])
            p_s0 = HostParam.project_dtensor(swiz_hp, pm0).make_cpu_param()
            assert not requires_activation(p_s0)
            assert tuple(p_s0.to_local().scale.shape) == (32, 64)
            pm1 = _make_meta_shard_dtensor(mesh, (256, 256), (256, 128), [Shard(1)])
            p_s1_hp = HostParam.project_dtensor(swiz_hp, pm1)
            assert requires_activation(p_s1_hp.make_cpu_param())
            assert tuple(p_s1_hp.storage_tensors()[1].shape) == (2, 2, 512)
            assert p_s1_hp.host_state.inner_state.target_shapes == ((256, 64), (64, 32), None, None)
            assert tuple(t.untyped_storage().data_ptr() for t in p_s1_hp.storage_tensors()) == swiz_ptrs

            cm1 = _make_meta_shard_dtensor(mesh, (128, 128), (128, 64), [Shard(1)])
            cr_s1_hp = HostParam.project_dtensor(cr_hp, cm1)
            assert requires_activation(cr_s1_hp.make_cpu_param())
            assert tuple(t.untyped_storage().data_ptr() for t in cr_s1_hp.storage_tensors()) == cr_ptrs
            assert cr_s1_hp.host_state.inner_state.meta.high_first is True
            _assert_unbound_projected_merge_rejected(cr_s1_hp.make_cpu_param())

        with ParallelExecutor(["cpu", "cpu"]) as ex:
            ex.run(verify)

    def test_project_dtensor_nvfp4_3d_per_expert_and_high_first_shards(self) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        pytest.importorskip("numpy")
        piper_nv = pytest.importorskip("piper_kernels.weights.nvfp4")
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import Shard

        from piper_offload import requires_activation
        from piper_offload.parallel import ParallelExecutor

        torch.manual_seed(17)
        base_3d = piper_nv.PiperNVFP4Tensor.from_hp(
            torch.randn(4, 64, 64, dtype=torch.bfloat16),
            per_tensor_scale=torch.tensor(0.5, dtype=torch.float32),
            is_swizzled_scales=False,
            use_triton_kernel=False,
        )
        nv_3d = piper_nv.PiperNVFP4Tensor(
            base_3d.qdata,
            base_3d.scale,
            base_3d.block_size,
            base_3d.orig_dtype,
            torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32).view(4, 1, 1),
            torch.tensor(0.125, dtype=torch.float32),
            False,
            False,
            None,
            high_first=True,
        )
        hp_3d = HostParam(nv_3d)
        ptrs_3d = tuple(t.untyped_storage().data_ptr() for t in hp_3d.storage_tensors())

        nv_tail = piper_nv.PiperNVFP4Tensor.from_hp(
            torch.randn(192, 128, dtype=torch.bfloat16),
            per_tensor_scale=torch.tensor(0.5, dtype=torch.float32),
            is_swizzled_scales=True,
            use_triton_kernel=False,
        )
        hp_tail = HostParam(nv_tail)
        ptrs_tail = tuple(t.untyped_storage().data_ptr() for t in hp_tail.storage_tensors())

        def verify(_rank: int) -> None:
            mesh = DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names=("ep", "tp"))
            meta_3d = _make_meta_shard_dtensor(mesh, (4, 64, 64), (2, 64, 32), [Shard(0), Shard(2)])
            proj_3d = HostParam.project_dtensor(hp_3d, meta_3d)
            assert requires_activation(proj_3d.make_cpu_param())
            assert tuple(t.untyped_storage().data_ptr() for t in proj_3d.storage_tensors()) == ptrs_3d
            assert tuple(proj_3d.storage_tensors()[2].shape) == (2, 1, 1)
            assert proj_3d.host_state.inner_state.meta.high_first is True

            meta_tail = _make_meta_shard_dtensor(mesh["tp"], (192, 128), (192, 64), [Shard(1)])
            proj_tail = HostParam.project_dtensor(hp_tail, meta_tail)
            assert requires_activation(proj_tail.make_cpu_param())
            assert tuple(t.untyped_storage().data_ptr() for t in proj_tail.storage_tensors()) == ptrs_tail
            assert tuple(proj_tail.storage_tensors()[1].shape) == (2, 1, 512)
            assert proj_tail.host_state.inner_state.target_shapes == ((192, 32), (64, 16), (), None)

        with ParallelExecutor(["cpu"] * 4) as ex:
            ex.run(verify)

    def test_packed_swizzled_shard_projection_rejections(self) -> None:
        pytest.importorskip("numpy")
        piper_nv = pytest.importorskip("piper_kernels.weights.nvfp4")
        cr_mod = pytest.importorskip("piper_kernels.weights.convrot.nvfp4")
        torch.manual_seed(23)
        _assert_nvfp4_and_convrot_rejections(piper_nv, cr_mod)

    def test_swizzled_quadrant_and_3d_uneven_cpu_dequantize_parity(self) -> None:
        pytest.importorskip("numpy")
        piper_nv = pytest.importorskip("piper_kernels.weights.nvfp4")
        from piper_offload.piper_nvfp4_adapter import PiperNvfp4Adapter

        torch.manual_seed(29)
        p2 = piper_nv.PiperNVFP4Tensor.from_hp(
            torch.randn(192, 160, dtype=torch.bfloat16),
            is_swizzled_scales=True,
        )
        p2_state = PiperNvfp4Adapter.capture_host(p2)
        p2_ref = p2.dequantize()
        for off, loc in (((0, 0), (128, 128)), ((0, 128), (128, 32)), ((128, 0), (64, 128)), ((128, 128), (64, 32))):
            proj, _ = PiperNvfp4Adapter.project_shard(p2_state, offsets=off, local_shape=loc)
            rebuilt = _reconstruct_projected_cpu(PiperNvfp4Adapter, proj)
            ref = p2_ref[off[0] : off[0] + loc[0], off[1] : off[1] + loc[1]]
            torch.testing.assert_close(rebuilt.dequantize(), ref, rtol=0, atol=0)

        p3_raw = piper_nv.PiperNVFP4Tensor.from_hp(
            torch.randn(4, 192, 80, dtype=torch.bfloat16),
            per_tensor_scale=torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32).view(4, 1, 1),
            is_swizzled_scales=True,
        )
        p3 = piper_nv.PiperNVFP4Tensor(
            (((p3_raw.qdata & 0x0F) << 4) | (p3_raw.qdata >> 4)).to(torch.uint8),
            p3_raw.scale, p3_raw.block_size, p3_raw.orig_dtype, p3_raw.per_tensor_scale,
            torch.tensor(0.25, dtype=torch.float32), True, False, None, high_first=True,
        )
        p3_state = PiperNvfp4Adapter.capture_host(p3)
        p3_ref = p3.dequantize()
        for off, loc in (((2, 128, 64), (2, 64, 16)), ((1, 0, 0), (2, 128, 64))):
            proj, _ = PiperNvfp4Adapter.project_shard(p3_state, offsets=off, local_shape=loc)
            rebuilt = _reconstruct_projected_cpu(PiperNvfp4Adapter, proj)
            assert rebuilt.high_first is True
            assert tuple(rebuilt.per_tensor_scale.shape) == (loc[0], 1, 1)
            ref = p3_ref[off[0] : off[0] + loc[0], off[1] : off[1] + loc[1], off[2] : off[2] + loc[2]]
            torch.testing.assert_close(rebuilt.dequantize(), ref, rtol=0, atol=0)
