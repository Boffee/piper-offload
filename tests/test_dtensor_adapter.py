"""Tests for :class:`DTensorAdapter` (tensor-parallel weights).

A ``DTensor`` weight needs a ``DeviceMesh`` and a process group, so the whole
module requires CUDA and initializes a single-rank group. Single-rank means
``local == global`` (the N≥2 sharding traps — ``data_ptr()==0`` dedup collapse
and ``cache_bytes`` over-accounting — can't be reproduced here), so those are
covered by reasoning: the adapter keys identity/bytes off the *local* shard.
"""

import os
from typing import Any

import pytest
import torch
from torch import nn

import piper_offload.dtensor_adapter as dtensor_adapter_module
from piper_offload import (
    Adapter,
    LoRATransform,
    ModelOffloader,
    ModelStore,
    OffloadConfig,
    ParameterDelta,
    ScaledLoRAFactor,
    merge_adapter,
    requires_activation,
)
from piper_offload._dtensor import local_shard
from piper_offload.dtensor_adapter import DTensorAdapter
from piper_offload.host_module import HostModuleStore
from piper_offload.int8_adapter import Int8Adapter
from piper_offload.host_param import HostParam
from piper_offload.tensor_adapters import (
    DenseMergeTargetValidationTensorAdapter,
    DenseMergeTensorAdapter,
    DenseMergeValidationTensorAdapter,
    DequantizeTensorAdapter,
    DequantRequantTensorAdapter,
    LoRAMergeTensorAdapter,
    MergeLocalityTensorAdapter,
    RegularAdapter,
    TensorCopyIntoAdapter,
)
from piper_offload.tensor_adapter_registry import (
    param_representation,
    select_adapter,
    tensor_id,
)
from tests.conftest import activated_model, host_component

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DTensor tensor-parallel weights need a CUDA mesh + process group",
)


@pytest.fixture(scope="module")
def tp_mesh() -> Any:
    import torch.distributed as dist
    from torch.distributed.tensor import init_device_mesh

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29593")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    created = False
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)
        created = True
    torch.cuda.set_device(0)
    mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("tp",))
    yield mesh
    if created and dist.is_initialized():
        dist.destroy_process_group()


def _shard(dim: int = 0) -> Any:
    from torch.distributed.tensor import Shard

    return Shard(dim)


def _dtensor_weight(
    mesh: Any,
    *,
    rows: int = 16,
    cols: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    placement: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch.distributed.tensor import distribute_tensor

    full = torch.randn(rows, cols, dtype=dtype, device="cuda")
    dt = distribute_tensor(full, mesh, [placement or _shard(0)])
    return dt, full


def _is_dtensor(t: torch.Tensor) -> bool:
    from torch.distributed.tensor import DTensor

    return isinstance(t, DTensor)


class TestDTensorAdapter:
    def test_matches_and_dispatches(self, tp_mesh: Any) -> None:
        dt, _ = _dtensor_weight(tp_mesh)
        assert DTensorAdapter.matches(dt)
        assert isinstance(select_adapter(dt), DTensorAdapter)
        assert not DTensorAdapter.matches(torch.zeros(4, 4))

    def test_delegates_local_shard_to_inner_adapter(self, tp_mesh: Any) -> None:
        dt, _ = _dtensor_weight(tp_mesh)
        host_param = HostParam(nn.Parameter(dt, requires_grad=False))
        # A plain local shard is moved by the registry's RegularAdapter — the
        # DTensorAdapter only adds the distributed wrapper on top.
        assert isinstance(host_param.host_state.inner, RegularAdapter)
        assert isinstance(host_param.host_state.inner, LoRAMergeTensorAdapter)

    def test_storage_enumeration_reads_local_backing_without_rebuilding_dtensor(
        self,
        tp_mesh: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dt, full = _dtensor_weight(tp_mesh)
        host = HostParam(nn.Parameter(dt, requires_grad=False))

        def unexpected_rebuild(*args: object, **kwargs: object) -> None:
            raise AssertionError("storage enumeration must not rebuild DTensor")

        monkeypatch.setattr(dtensor_adapter_module, "rebuild_dtensor", unexpected_rebuild)
        (local,) = host.storage_tensors()

        assert type(local) is torch.Tensor
        assert local.device.type == "cpu"
        assert local is host.host_state.inner_state.data
        torch.testing.assert_close(local, full.cpu())

    def test_host_param_roundtrip_reconstructs_dtensor(self, tp_mesh: Any) -> None:
        dt, full = _dtensor_weight(tp_mesh)
        host_param = HostParam(nn.Parameter(dt, requires_grad=False))

        # Resting state: still a DTensor (type-stable), but on a CPU mesh so
        # the local shard stays on the host (no GPU memory held).
        cpu = host_param.make_cpu_param()
        assert _is_dtensor(cpu.data)
        assert cpu.data.device_mesh.device_type == "cpu"
        assert cpu.data.to_local().device.type == "cpu"
        assert cpu.data.placements == dt.placements
        assert torch.equal(cpu.data.to_local(), dt.to_local().cpu())

        # Resident state: the DTensor is reconstructed on the GPU.
        gpu_state = host_param.allocate_gpu_storage(torch.device("cuda"))
        host_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = host_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()

        assert _is_dtensor(gpu_param.data)
        assert gpu_param.data.to_local().is_cuda
        assert gpu_param.data.placements == dt.placements
        assert gpu_param.data.device_mesh == dt.device_mesh
        assert torch.equal(gpu_param.data.full_tensor(), full)

    def test_one_shot_materialize_reconstructs_cuda_mesh(self, tp_mesh: Any) -> None:
        dt, full = _dtensor_weight(tp_mesh)
        host_param = HostParam(nn.Parameter(dt, requires_grad=False))
        assert host_param.shape == dt.shape

        materialized = host_param.materialize(
            torch.device("cuda"),
            non_blocking=True,
        )
        torch.cuda.synchronize()

        tensor = param_representation(materialized)
        assert _is_dtensor(tensor)
        assert tensor.device_mesh == tp_mesh
        assert tensor.device_mesh.device_type == "cuda"
        assert tensor.to_local().is_cuda
        assert torch.equal(tensor.full_tensor(), full)

    def test_gpu_param_aliases_inner_storage_for_refill(self, tp_mesh: Any) -> None:
        # The pooled streaming path reuses one wrapper across loads, refilling
        # its buffers in place. from_local must alias the inner GPU storage
        # (not copy) so refills are visible through the wrapper.
        dt, _ = _dtensor_weight(tp_mesh)
        host_param = HostParam(nn.Parameter(dt, requires_grad=False))
        gpu_state = host_param.allocate_gpu_storage(torch.device("cuda"))
        host_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = host_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()

        assert gpu_param.data.to_local().data_ptr() == gpu_state.inner_gpu.data.data_ptr()

    def test_tensor_id_and_layout_track_placements(self, tp_mesh: Any) -> None:
        from torch.distributed.tensor import Replicate, distribute_tensor

        full = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")
        sharded = distribute_tensor(full, tp_mesh, [_shard(0)])
        replicated = distribute_tensor(full, tp_mesh, [Replicate()])

        assert DTensorAdapter.layout_signature(sharded) != DTensorAdapter.layout_signature(replicated)
        assert tensor_id(sharded) != tensor_id(replicated)
        assert tensor_id(sharded)[0] == "dtensor"

    def test_accepts_bind_target_relaxes_inner_dtype_and_checks_shape_and_placements(
        self, tp_mesh: Any,
    ) -> None:
        from torch.distributed.tensor import DTensor, Replicate

        from piper_offload._dtensor import cpu_mesh_for

        bf16 = _dtensor_weight(tp_mesh, dtype=torch.bfloat16)[0]
        fp32 = _dtensor_weight(tp_mesh, dtype=torch.float32)[0]
        assert DTensorAdapter.layout_signature(bf16) != DTensorAdapter.layout_signature(fp32)

        host = HostParam(nn.Parameter(bf16, requires_grad=False))
        cpu_mesh = cpu_mesh_for(bf16.device_mesh)
        # Plain dtype mismatch is accepted even on a CPU mesh.
        fp32_cpu = DTensor.from_local(
            fp32.to_local().cpu(),
            cpu_mesh,
            bf16.placements,
            run_check=False,
        )
        assert host.accepts_bind_target(nn.Parameter(fp32_cpu, requires_grad=False))

        # Global shape, placements, and local shard shape remain strict.
        different_global = DTensor.from_local(
            torch.empty((bf16.shape[0] + 4, *bf16.shape[1:]), device="meta"),
            cpu_mesh,
            bf16.placements,
            run_check=False,
        )
        assert not host.accepts_bind_target(nn.Parameter(different_global, requires_grad=False))

        different_place = DTensor.from_local(
            fp32.to_local().cpu(),
            fp32.device_mesh,
            [Replicate()],
            run_check=False,
        )
        assert not host.accepts_bind_target(nn.Parameter(different_place, requires_grad=False))
        local_shape = tuple(bf16.to_local().shape)
        mismatched_local = DTensor.from_local(
            torch.empty((max(1, local_shape[0] // 2), *local_shape[1:]), device="meta"),
            bf16.device_mesh,
            bf16.placements,
            shape=tuple(bf16.shape),
            stride=bf16.stride(),
            run_check=False,
        )
        assert not host.accepts_bind_target(nn.Parameter(mismatched_local, requires_grad=False))

    def test_tensor_id_keys_off_local_not_dtensor_dataptr(self, tp_mesh: Any) -> None:
        dt, _ = _dtensor_weight(tp_mesh)
        # The DTensor's own data_ptr is 0 — using it would collapse tied-weight
        # dedup. Two independently-allocated DTensors must get distinct ids.
        assert dt.data_ptr() == 0
        other, _ = _dtensor_weight(tp_mesh)
        assert tensor_id(dt) != tensor_id(other)
        assert tensor_id(dt) == tensor_id(dt)
        # The global shape is in the identity key so aliased local shards with
        # different global views are not deduped onto one HostParam.
        assert tuple(dt.shape) in tensor_id(dt)

    def test_cache_bytes_counts_local_shard(self, tp_mesh: Any) -> None:
        dt, _ = _dtensor_weight(tp_mesh, rows=16, cols=8)
        host_param = HostParam(nn.Parameter(dt, requires_grad=False))
        local = dt.to_local()
        # Local-shard bytes (== global only because world_size==1; on N ranks
        # this is the ~1/N local footprint, not the global numel).
        assert DTensorAdapter.cache_bytes(host_param.host_state) == (local.numel() * local.element_size())

    def test_compute_dtype_delegates_to_local(self, tp_mesh: Any) -> None:
        dt, _ = _dtensor_weight(tp_mesh, dtype=torch.bfloat16)
        assert DTensorAdapter.compute_dtype(dt) is torch.bfloat16

    def test_compute_dtype_via_host_param_property(self, tp_mesh: Any) -> None:
        # Regression: HostParam.compute_dtype feeds the bare local shard
        # (what cpu_param yields, not a DTensor) into the adapter; it must not
        # require a live DTensor. Every other adapter asserts this property.
        dt, _ = _dtensor_weight(tp_mesh, dtype=torch.bfloat16)
        host_param = HostParam(nn.Parameter(dt, requires_grad=False))
        assert host_param.compute_dtype is torch.bfloat16

    def test_direct_validation_localizes_global_factors_like_merge(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[torch.Tensor, torch.Tensor]] = []

        class RecordingValidator:
            @staticmethod
            def validate_lora_merge(
                _target: torch.Tensor,
                b: torch.Tensor,
                a: torch.Tensor,
                _alpha: float,
                *,
                beta: float = 1.0,
                rounding_seed: int | None = None,
            ) -> None:
                del rounding_seed
                assert beta == -0.5
                calls.append((b, a))

        local = torch.empty(3, 4, device="cuda")
        context = dtensor_adapter_module._DTensorLayoutContext(
            global_shape=(6, 8),
            local_shape=(3, 4),
            offsets=(2, 3),
            local=local,
            inner=RecordingValidator(),  # type: ignore[arg-type]
        )
        monkeypatch.setattr(
            dtensor_adapter_module,
            "_merge_context",
            lambda _target: context,
        )
        global_b = torch.arange(12, device="cuda").reshape(6, 2)
        global_a = torch.arange(16, device="cuda").reshape(2, 8)

        DTensorAdapter.validate_lora_merge(
            torch.empty(0, device="cuda"),
            global_b,
            global_a,
            0.5,
            beta=-0.5,
        )

        assert len(calls) == 1
        local_b, local_a = calls[0]
        assert local_b.is_contiguous()
        assert local_a.is_contiguous()
        torch.testing.assert_close(local_b, global_b.narrow(0, 2, 3))
        torch.testing.assert_close(local_a, global_a.narrow(1, 3, 4))

    def test_streamed_offloader_reconstructs_dtensor_blocks(self, tp_mesh: Any) -> None:
        # The real production path: bind a ModelOffloader over DTensor-weighted
        # blocks, activate (gpu_param rebuilds the DTensor), deactivate
        # (cpu_param restores the bare local shard).
        class Block(nn.Module):
            def __init__(self, w: torch.Tensor) -> None:
                super().__init__()
                self.weight = nn.Parameter(w, requires_grad=False)

            def forward(self) -> torch.Tensor:
                return self.weight

        class Net(nn.Module):
            def __init__(self, blocks: list[nn.Module]) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(blocks)

        net = Net([Block(_dtensor_weight(tp_mesh)[0]), Block(_dtensor_weight(tp_mesh)[0])])
        pw = ModelOffloader.from_module(net, OffloadConfig(block_paths=("blocks",)))
        try:
            # resting: a DTensor on a CPU mesh (local on the host)
            resting = net.blocks[0].weight.data
            assert _is_dtensor(resting)
            assert resting.device_mesh.device_type == "cpu"
            with activated_model(pw, "cuda"):
                for blk in net.blocks:
                    blk()
                    assert _is_dtensor(blk.weight.data)
                    assert blk.weight.data.device_mesh.device_type == "cuda"
                    assert blk.weight.data.to_local().is_cuda
            # back to resting (CPU mesh) after deactivate
            assert net.blocks[0].weight.data.device_mesh.device_type == "cpu"
        finally:
            pw.deactivate()

    def test_merge_lora_updates_plain_dtensor_local_shard(
        self,
        tp_mesh: Any,
    ) -> None:
        from torch.distributed.tensor import distribute_tensor

        in_dim = 8
        out_dim = 12
        rank = 3
        weight = torch.randn(out_dim, in_dim, device="cuda")
        a = torch.randn(rank, in_dim, device="cuda")
        b = torch.randn(out_dim, rank, device="cuda")
        strength = 0.4

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Linear(in_dim, out_dim, bias=False, device="meta")
                self.target.weight = nn.Parameter(
                    distribute_tensor(weight, tp_mesh, [_shard(0)]),
                    requires_grad=False,
                )

        model = Net()
        lora = Adapter.from_state_dict(
            {
                "target.lora_A.weight": a,
                "target.lora_B.weight": b,
            }
        )
        expected = weight + strength * (b @ a)
        offloader = ModelOffloader.from_module(model)

        # A fresh base copy precedes every activation-scoped merge, so repeated
        # activation must not accumulate the update.
        for _ in range(2):
            with activated_model(
                offloader,
                "cuda",
                adapters=[lora],
                adapter_strengths=[strength],
                adapter_mode="merge",
            ) as active:
                actual = active.target.weight.data
                assert _is_dtensor(actual)
                torch.testing.assert_close(actual.full_tensor(), expected)

    def test_permanent_merge_preflight_rejects_unsupported_placement(
        self,
        tp_mesh: Any,
    ) -> None:
        from torch.distributed.tensor import DTensor, Partial

        rows = 12
        cols = 8
        rank = 3
        first_local = torch.randn(rows, cols, device="cuda")
        second_local = torch.randn(rows, cols, device="cuda")

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.first = nn.Linear(cols, rows, bias=False, device="meta")
                self.first.weight = nn.Parameter(
                    DTensor.from_local(
                        first_local,
                        tp_mesh,
                        [_shard(0)],
                        run_check=False,
                    ),
                    requires_grad=False,
                )
                self.second = nn.Linear(cols, rows, bias=False, device="meta")
                self.second.weight = nn.Parameter(
                    DTensor.from_local(
                        second_local,
                        tp_mesh,
                        [Partial()],
                        run_check=False,
                    ),
                    requires_grad=False,
                )

        model = Net()
        lora = Adapter.from_state_dict(
            {
                "first.lora_A.weight": torch.randn(rank, cols),
                "first.lora_B.weight": torch.randn(rows, rank),
                "second.lora_A.weight": torch.randn(rank, cols),
                "second.lora_B.weight": torch.randn(rows, rank),
            }
        )
        first_before = first_local.clone()

        with pytest.raises(
            ValueError,
            match="supports only Replicate and contiguous Shard placements",
        ):
            merge_adapter(model, [(lora, 0.25)])

        # merge_adapter preflights every target before mutating any of them.
        torch.testing.assert_close(model.first.weight.to_local(), first_before)

    def test_permanent_merge_preflight_delegates_to_inner_quant_adapter(
        self,
        tp_mesh: Any,
    ) -> None:
        pytest.importorskip("torchao")
        from torch.distributed.tensor import DTensor
        from torchao.float8.inference import Float8MMConfig
        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            Float8Tensor,
        )

        rows, cols, rank = 32, 16, 3
        first_local = torch.randn(rows, cols, device="cuda")
        grouped = Float8Tensor.from_hp(
            torch.randn(cols, rows, device="cuda"),
            float8_dtype=torch.float8_e4m3fn,
            granularity=PerGroup(4),
            mm_config=Float8MMConfig(use_fast_accum=True),
        ).t()
        assert tuple(grouped.block_size) == (4, 1)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.first = nn.Linear(cols, rows, bias=False, device="meta")
                self.first.weight = nn.Parameter(
                    DTensor.from_local(
                        first_local,
                        tp_mesh,
                        [_shard(0)],
                        run_check=False,
                    ),
                    requires_grad=False,
                )
                self.second = nn.Linear(cols, rows, bias=False, device="meta")
                self.second.weight = nn.Parameter(
                    DTensor.from_local(
                        grouped,
                        tp_mesh,
                        [_shard(0)],
                        run_check=False,
                    ),
                    requires_grad=False,
                )

        model = Net()
        lora = Adapter.from_state_dict(
            {
                "first.lora_A.weight": torch.randn(rank, cols),
                "first.lora_B.weight": torch.randn(rows, rank),
                "second.lora_A.weight": torch.randn(rank, cols),
                "second.lora_B.weight": torch.randn(rows, rank),
            }
        )
        first_before = model.first.weight.to_local().clone()

        with pytest.raises(
            ValueError,
            match="transposed PerGroup.*routed LoRA",
        ):
            merge_adapter(model, [(lora, 0.25)])

        torch.testing.assert_close(model.first.weight.to_local(), first_before)

    def test_routed_lora_materializes_factors_on_cuda_mesh(
        self,
        tp_mesh: Any,
    ) -> None:
        from torch.distributed.tensor import Replicate, distribute_tensor

        in_dim = 8
        out_dim = 12
        rank = 3
        weight = torch.randn(out_dim, in_dim, device="cuda")
        a = torch.randn(rank, in_dim, device="cuda")
        b = torch.randn(out_dim, rank, device="cuda")
        x = torch.randn(4, in_dim, device="cuda")

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Linear(in_dim, out_dim, bias=False, device="meta")
                self.target.weight = nn.Parameter(
                    distribute_tensor(weight, tp_mesh, [Replicate()]),
                    requires_grad=False,
                )

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.target(inputs)

        model = Net()
        lora = Adapter.from_state_dict(
            {
                "target.lora_A.weight": distribute_tensor(
                    a,
                    tp_mesh,
                    [Replicate()],
                ),
                "target.lora_B.weight": distribute_tensor(
                    b,
                    tp_mesh,
                    [Replicate()],
                ),
            }
        )
        inputs = distribute_tensor(x, tp_mesh, [Replicate()])
        strength = 0.4
        expected = x @ weight.T + ((x @ a.T) * strength) @ b.T
        offloader = ModelOffloader.from_module(model)

        with activated_model(
            offloader,
            "cuda",
            adapters=[lora],
            adapter_strengths=[strength],
            adapter_mode="routed",
        ):
            actual = model(inputs)
            assert _is_dtensor(actual)
            assert actual.device_mesh == tp_mesh
            torch.testing.assert_close(actual.full_tensor(), expected)

    def test_advertises_merge_capabilities(
        self,
        tp_mesh: Any,
    ) -> None:
        # Frozen-inference scope: shard-local additive merge is available, while
        # full dequant/requant and copy_into stay hidden. Dense local-shard
        # exposure is available for parameter-value scaling.
        adapter = select_adapter(_dtensor_weight(tp_mesh)[0])
        assert isinstance(adapter, LoRAMergeTensorAdapter)
        assert isinstance(adapter, DenseMergeTensorAdapter)
        assert isinstance(adapter, DenseMergeTargetValidationTensorAdapter)
        assert isinstance(adapter, DenseMergeValidationTensorAdapter)
        assert isinstance(adapter, DequantizeTensorAdapter)
        assert isinstance(adapter, MergeLocalityTensorAdapter)
        assert not isinstance(adapter, DequantRequantTensorAdapter)
        assert not isinstance(adapter, TensorCopyIntoAdapter)

    @pytest.mark.parametrize("with_delta", [False, True])
    def test_parameter_value_preserves_dtensor_and_scales_local_shard(
        self,
        tp_mesh: Any,
        with_delta: bool,
    ) -> None:
        dt, full = _dtensor_weight(tp_mesh, dtype=torch.float32 if with_delta else torch.bfloat16)
        adapter = Adapter.from_state_dict(
            {"weight": dt},
            scale_parameter_values=True,
        )
        delta = Adapter({"weight": ParameterDelta.from_tensors(
            dense=torch.full(tuple(dt.shape), 0.25),
        )})
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.empty(dt.shape, device="meta"),
            requires_grad=False,
        )
        offloader = ModelOffloader.from_module(model)

        with activated_model(
            offloader,
            "cuda",
            adapters=[adapter, delta],
            adapter_strengths=[0.5, 1.0 if with_delta else 0.0],
            stochastic_rounding=False,
        ):
            actual = param_representation(model.weight)
            assert _is_dtensor(actual)
            assert actual.device_mesh == tp_mesh
            torch.testing.assert_close(
                actual.full_tensor(), full * 0.5 + (0.25 if with_delta else 0.0),
            )

        assert model.weight.is_meta

    @pytest.mark.parametrize("with_delta", [False, True])
    def test_permanent_parameter_value_scales_cpu_dtensor(
        self,
        tp_mesh: Any,
        with_delta: bool,
    ) -> None:
        dt, _ = _dtensor_weight(tp_mesh, dtype=torch.float32 if with_delta else torch.bfloat16)
        adapter = Adapter.from_state_dict(
            {"weight": dt},
            scale_parameter_values=True,
        )
        delta = Adapter({"weight": ParameterDelta.from_tensors(
            dense=torch.full(tuple(dt.shape), 0.25),
        )})
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.empty(dt.shape, device="meta"),
            requires_grad=False,
        )

        assert (
            merge_adapter(
                model,
                [(adapter, 0.5), (delta, 1.0 if with_delta else 0.0)],
                stochastic_rounding=False,
            )
            == 1
        )

        actual = param_representation(model.weight)
        assert _is_dtensor(actual)
        assert actual.device_mesh.device_type == "cpu"
        assert actual.placements == dt.placements
        torch.testing.assert_close(
            actual.to_local(),
            dt.to_local().cpu() * 0.5 + (0.25 if with_delta else 0.0),
        )

    def test_composes_with_quantized_local_shard(self, tp_mesh: Any) -> None:
        # The crown-jewel claim: one adapter composes with every quant adapter.
        # A DTensor wrapping a TorchAO Float8Tensor must reuse Float8Adapter
        # for the local shard with no DTensor-specific quant code.
        #
        # Build via from_local (wrap a per-rank shard) — the canonical TP
        # construction path frameworks use. distribute_tensor (split a full
        # tensor) hits a TorchAO Float8Tensor torch.chunk dispatch bug; the
        # adapter is agnostic to how the DTensor was created.
        pytest.importorskip("torchao")
        from torch.distributed.tensor import DTensor

        from piper_offload.float8_adapter import Float8Adapter

        try:
            from torchao.quantization import (
                Float8WeightOnlyConfig,
                quantize_,
            )

            layer = nn.Linear(8, 16, bias=False).to(torch.bfloat16).cuda()
            quantize_(layer, Float8WeightOnlyConfig())
            f8 = layer.weight.data  # a Float8Tensor
            dt = DTensor.from_local(f8, tp_mesh, [_shard(0)], run_check=False)
        except Exception as exc:  # env/version dependent
            pytest.skip(f"torchao Float8 DTensor unavailable: {exc}")

        assert isinstance(select_adapter(dt), DTensorAdapter)
        assert isinstance(select_adapter(dt.to_local()), Float8Adapter)

        host_param = HostParam(nn.Parameter(dt, requires_grad=False))
        assert isinstance(host_param.host_state.inner, Float8Adapter)

        qdata, scale = host_param.storage_tensors()
        assert type(qdata) is torch.Tensor and type(scale) is torch.Tensor
        assert qdata is host_param.host_state.inner_state.storage[0]
        assert scale is host_param.host_state.inner_state.storage[1]
        assert qdata.dtype == f8.qdata.dtype
        assert qdata.nbytes + scale.nbytes == host_param.cache_bytes

        gpu_state = host_param.allocate_gpu_storage(torch.device("cuda"))
        host_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = host_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()

        assert _is_dtensor(gpu_param.data)
        # the reconstructed local shard is still the Float8 quant subclass
        assert isinstance(select_adapter(gpu_param.data.to_local()), Float8Adapter)
        assert gpu_param.data.placements == dt.placements

        value_adapter = Adapter.from_state_dict(
            {"weight": dt},
            scale_parameter_values=True,
        )
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.empty(dt.shape, device="meta"),
            requires_grad=False,
        )
        offloader = ModelOffloader.from_module(model)
        dense_source = Float8Adapter.dequantize(dt.to_local())
        with activated_model(
            offloader,
            "cuda",
            adapters=[value_adapter],
            adapter_strengths=[0.5],
            stochastic_rounding=False,
        ):
            actual = param_representation(model.weight)
            assert _is_dtensor(actual)
            assert isinstance(select_adapter(actual.to_local()), Float8Adapter)
            torch.testing.assert_close(
                Float8Adapter.dequantize(actual.to_local()),
                dense_source * 0.5,
                rtol=0.15,
                atol=0.05,
            )

    @pytest.mark.parametrize("beta", [1.0, -0.5])
    def test_packed_int8_delegates_factor_aware_staging(
        self,
        tp_mesh: Any,
        beta: float,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The outer DTensor must preserve factor boundaries for INT8 p."""
        pytest.importorskip("torchao")
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from torchao.quantization.granularity import PerRow
        from torchao.quantization.quant_primitives import MappingType
        from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
            Int8Tensor,
        )

        torch.manual_seed(126)
        rows, cols = 4, 8
        pre_scale_value = 0.75
        pre_scale = torch.tensor(pre_scale_value, device="cuda")
        base = Int8Tensor.from_hp(
            torch.randn(rows, cols, device="cuda") * 0.05,
            PerRow(),
            MappingType.SYMMETRIC,
        )
        local = Int8Tensor(
            base.qdata.clone(),
            base.scale.clone(),
            list(base.block_size),
            base.dtype,
            zero_point=(None if base.zero_point is None else base.zero_point.clone()),
            act_pre_scale=pre_scale,
            act_quant_kwargs=base.act_quant_kwargs,
            reduce_range=base.reduce_range,
        )

        # TorchAO 0.18 lacks aten.view for Int8Tensor, while DTensor's public
        # from_local/to_local wrappers add a view solely for autograd aliasing.
        # This frozen-inference adapter test builds the identical DTensor spec
        # directly and bypasses only that unsupported no-op view.
        spec = DTensorSpec(
            tp_mesh,
            (_shard(0),),
            tensor_meta=TensorMeta(
                torch.Size(local.shape),
                local.stride(),
                local.dtype,
            ),
        )
        weight = DTensor(local, spec, requires_grad=False)
        original_to_local = DTensor.to_local

        def int8_to_local_without_view(
            tensor: DTensor,
            *args: object,
            **kwargs: object,
        ) -> torch.Tensor:
            if isinstance(tensor._local_tensor, Int8Tensor):
                return tensor._local_tensor
            return original_to_local(tensor, *args, **kwargs)

        monkeypatch.setattr(DTensor, "to_local", int8_to_local_without_view)

        a = torch.full((1, cols), 0.2)
        b = torch.full((rows, 1), 0.15)
        strength = 0.5
        factors = [ScaledLoRAFactor.from_tensors(a.clone(), b.clone(), strength) for _ in range(2)]
        param = nn.Parameter(weight, requires_grad=False)
        x = torch.full((3, cols), 0.1, device="cuda")
        routed_output = torch.nn.functional.linear(x, local) * beta
        for _ in factors:
            routed_output = routed_output + ((x @ a.to("cuda").T) * strength) @ b.to("cuda").T

        prepared_targets: list[torch.Tensor] = []
        original_prepared_merge = Int8Adapter.merge_prepared_lora_

        def record_prepared_merge(
            target: torch.Tensor,
            packed_b: torch.Tensor,
            packed_a: torch.Tensor,
            packed_alpha: float,
            *,
            beta: float = 1.0,
            rounding_seed: int | None = None,
        ) -> None:
            prepared_targets.append(target)
            assert torch.isfinite(packed_a).all()
            torch.testing.assert_close(
                packed_a,
                torch.full_like(
                    packed_a,
                    0.2 * strength / pre_scale_value,
                ),
            )
            original_prepared_merge(
                target,
                packed_b,
                packed_a,
                packed_alpha,
                beta=beta,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(
            Int8Adapter,
            "merge_prepared_lora_",
            staticmethod(record_prepared_merge),
        )

        pre_scale_ptr = local.act_pre_scale.data_ptr()
        transform = LoRATransform(factors, beta=beta)
        transform.validate_target(param)
        transform.apply(param)
        torch.cuda.synchronize()

        assert len(prepared_targets) == 1
        assert prepared_targets[0].qdata.data_ptr() == local.qdata.data_ptr()
        merged_local = param.data.to_local()
        assert merged_local.act_pre_scale.data_ptr() == pre_scale_ptr
        assert torch.isfinite(merged_local.dequantize(torch.float32)).all()
        merged_output = torch.nn.functional.linear(x, merged_local)
        torch.testing.assert_close(
            merged_output,
            routed_output,
            rtol=0.01,
            atol=0.05,
        )

    def test_merge_lora_delegates_to_quantized_local_adapter(
        self,
        tp_mesh: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("torchao")
        from torch.distributed.tensor import DTensor
        from torchao.quantization import Float8WeightOnlyConfig, quantize_

        from piper_offload.float8_adapter import Float8Adapter

        in_dim = 8
        out_dim = 16
        rank = 3
        layer = nn.Linear(
            in_dim,
            out_dim,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda",
        )
        try:
            quantize_(layer, Float8WeightOnlyConfig())
            local_weight = layer.weight.data
            weight = DTensor.from_local(
                local_weight,
                tp_mesh,
                [_shard(0)],
                run_check=False,
            )
        except Exception as exc:
            pytest.skip(f"torchao Float8 DTensor unavailable: {exc}")

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Linear(in_dim, out_dim, bias=False, device="meta")
                self.target.weight = nn.Parameter(weight, requires_grad=False)

        calls: list[tuple[type[torch.Tensor], tuple[int, ...], tuple[int, ...]]] = []
        original_merge = Float8Adapter.merge_lora_

        def record_local_merge(
            target: torch.Tensor,
            b: torch.Tensor,
            a: torch.Tensor,
            alpha: float,
            *,
            beta: float = 1.0,
            rounding_seed: int | None = None,
        ) -> None:
            calls.append((type(target), tuple(b.shape), tuple(a.shape)))
            original_merge(
                target,
                b,
                a,
                alpha,
                beta=beta,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(
            Float8Adapter,
            "merge_lora_",
            staticmethod(record_local_merge),
        )

        model = Net()
        lora = Adapter.from_state_dict(
            {
                "target.lora_A.weight": torch.randn(
                    rank,
                    in_dim,
                    dtype=torch.bfloat16,
                ),
                "target.lora_B.weight": torch.randn(
                    out_dim,
                    rank,
                    dtype=torch.bfloat16,
                ),
            }
        )
        offloader = ModelOffloader.from_module(model)

        with activated_model(
            offloader,
            "cuda",
            adapters=[lora],
            adapter_strengths=[0.25],
            adapter_mode="merge",
        ):
            assert calls == [
                (
                    type(local_weight),
                    (out_dim, rank),
                    (rank, in_dim),
                )
            ]

    def test_meta_placeholder_requires_dtensor_and_accepts_quantized_meta_shard(
        self, tp_mesh: Any,
    ) -> None:
        from torch.distributed.tensor import DTensor, Replicate

        convrot_mod = pytest.importorskip("piper_kernels.weights.convrot.int8")
        convrot_cls = convrot_mod.ConvRotInt8Tensor

        rows, cols = 16, 64
        q_local = convrot_cls.from_quantized(
            torch.randint(-127, 128, (rows, cols), dtype=torch.int8, device="cuda"),
            torch.rand((rows, 1), dtype=torch.float32, device="cuda") + 0.1,
            group_size=64,
            logical_dtype=torch.bfloat16,
        )
        weight = DTensor.from_local(q_local, tp_mesh, [_shard(0)], run_check=False)

        proto = nn.Module()
        proto.weight = nn.Parameter(weight, requires_grad=False)
        store = HostModuleStore.from_module(proto)

        # Reject plain non-DTensor meta placeholder against a DTensor store.
        plain_meta = nn.Module()
        plain_meta.weight = nn.Parameter(torch.empty((rows, cols), device="meta"), requires_grad=False)
        with pytest.raises(ValueError, match="layout mismatch"):
            store.bind(plain_meta)

        # Accept DTensor(meta) Shard(0) against an unsharded ConvRotInt8 store (zero-copy projection),
        # while still rejecting unsupported quantized formats such as Int8Tensor.
        non_dtensor_proto = nn.Module()
        non_dtensor_proto.weight = nn.Parameter(q_local, requires_grad=False)
        non_dtensor_store = HostModuleStore.from_module(non_dtensor_proto)
        dtensor_meta = nn.Module()
        dtensor_meta.weight = nn.Parameter(
            DTensor.from_local(
                torch.empty((rows, cols), dtype=torch.float32, device="meta"),
                tp_mesh,
                [_shard(0)],
                run_check=False,
            ),
            requires_grad=False,
        )
        non_dtensor_store.bind(dtensor_meta)
        assert isinstance(dtensor_meta.weight.data, DTensor) and isinstance(
            dtensor_meta.weight.data.to_local(), convrot_cls
        )
        assert (
            dtensor_meta.weight.data.to_local().qdata.untyped_storage().data_ptr()
            == non_dtensor_store.params["weight"].storage_tensors()[0].untyped_storage().data_ptr()
        )
        dtensor_meta_second = nn.Module()
        dtensor_meta_second.weight = nn.Parameter(
            DTensor.from_local(
                torch.empty((rows, cols), dtype=torch.float32, device="meta"),
                tp_mesh,
                [_shard(0)],
                run_check=False,
            ),
            requires_grad=False,
        )
        non_dtensor_store.bind(dtensor_meta_second)

        torchao_quant = pytest.importorskip("torchao.quantization")
        int8_layer = nn.Linear(cols, rows, bias=False, dtype=torch.bfloat16).eval().requires_grad_(False)
        torchao_quant.quantize_(int8_layer, torchao_quant.Int8WeightOnlyConfig())
        with pytest.raises(ValueError, match="layout mismatch"):
            HostModuleStore.from_module(int8_layer).bind(dtensor_meta)

        # Accept matching DTensor over plain strided meta local shard (with mismatched placeholder dtype),
        # and allow re-binding after initial bind.
        meta_skeleton = nn.Module()
        meta_dt = DTensor.from_local(
            torch.empty((rows, cols), dtype=torch.float32, device="meta"),
            tp_mesh,
            [_shard(0)],
            run_check=False,
        )
        meta_skeleton.weight = nn.Parameter(meta_dt, requires_grad=False)
        store.bind(meta_skeleton)
        store.bind(meta_skeleton)
        assert isinstance(meta_skeleton.weight.data, DTensor)
        assert meta_skeleton.weight.data.device_mesh.device_type == "cpu"
        assert isinstance(meta_skeleton.weight.data.to_local(), convrot_cls)

        # Reject mismatched placements.
        bad_place = nn.Module()
        bad_place.weight = nn.Parameter(
            DTensor.from_local(
                torch.empty((rows, cols), dtype=torch.float32, device="meta"),
                tp_mesh,
                [Replicate()],
                run_check=False,
            ),
            requires_grad=False,
        )
        with pytest.raises(ValueError, match="layout mismatch"):
            store.bind(bad_place)

        # Reject mismatched global shape.
        bad_global = nn.Module()
        bad_global.weight = nn.Parameter(
            DTensor.from_local(
                torch.empty((rows + 8, cols), dtype=torch.float32, device="meta"),
                tp_mesh,
                [_shard(0)],
                run_check=False,
            ),
            requires_grad=False,
        )
        with pytest.raises(ValueError, match="layout mismatch"):
            store.bind(bad_global)

        # Reject mismatched local shard shape.
        bad_local = nn.Module()
        bad_local.weight = nn.Parameter(
            DTensor.from_local(
                torch.empty((rows // 2, cols), dtype=torch.float32, device="meta"),
                tp_mesh,
                [_shard(0)],
                shape=(rows, cols),
                stride=(cols, 1),
                run_check=False,
            ),
            requires_grad=False,
        )
        with pytest.raises(ValueError, match="layout mismatch"):
            store.bind(bad_local)


class TestQuantizedShardProjection:
    """Zero-copy quantized DTensor Shard(dim) projection, activation, merge, and rejections."""

    def test_float8_shard_projection_granularities_and_cuda_lifecycle(
        self,
        tp_mesh: Any,
    ) -> None:
        del tp_mesh
        pytest.importorskip("torchao.quantization")
        from torchao.quantization.granularity import PerGroup, PerRow, PerTensor
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor

        from piper_offload.float8_adapter import Float8Adapter

        torch.manual_seed(0)
        w2 = torch.randn(32, 64, dtype=torch.bfloat16)
        w3 = torch.randn(4, 32, 64, dtype=torch.bfloat16)

        cases = [
            ("PerTensor", Float8Tensor.from_hp(w2, granularity=PerTensor()), (16, 32), (16, 32), (16, 32), True),
            ("PerRow", Float8Tensor.from_hp(w2, granularity=PerRow()), (16, 32), (16, 32), (1, 32), True),
            ("PerRow_dim0", Float8Tensor.from_hp(w2, granularity=PerRow(dim=0)), (16, 32), (16, 32), (16, 1), False),
            ("PerGroup32", Float8Tensor.from_hp(w2, granularity=PerGroup(32)), (16, 32), (16, 32), (1, 32), True),
            (
                "3D_PerGroup32",
                Float8Tensor.from_hp(w3, granularity=PerGroup(32)),
                (2, 16, 32),
                (2, 16, 32),
                (1, 1, 32),
                False,
            ),
            ("PerRow_1row_S1", Float8Tensor.from_hp(w2[:1], granularity=PerRow()), (0, 32), (1, 32), (1, 32), True),
        ]

        for label, src_tensor, offsets, local_shape, expected_block_size, supports_matmul in cases:
            host_state = Float8Adapter.capture_host(src_tensor)
            proj_state, layout_sig = Float8Adapter.project_shard(
                host_state,
                offsets=offsets,
                local_shape=local_shape,
            )
            assert proj_state.meta.block_size == expected_block_size, label
            q_host, s_host = host_state.storage
            q_proj, s_proj = proj_state.storage
            assert q_host is not None and s_host is not None
            assert q_proj is not None and s_proj is not None
            assert q_proj.untyped_storage().data_ptr() == q_host.untyped_storage().data_ptr(), label
            assert s_proj.untyped_storage().data_ptr() == s_host.untyped_storage().data_ptr(), label

            cpu_param = Float8Adapter.cpu_param(proj_state)
            assert cpu_param.data.device.type == "cpu"
            assert not cpu_param.is_meta
            assert requires_activation(cpu_param)
            assert Float8Adapter.accepts_bind_target(
                proj_state,
                cpu_param.data,
                layout_signature=layout_sig,
            )
            assert "ProjectedTorchaoShard" in repr(cpu_param.data)
            with pytest.raises(RuntimeError, match="requires CUDA activation"):
                cpu_param.data.clone()
            with (
                torch.inference_mode(),
                pytest.raises(RuntimeError, match="requires CUDA activation"),
            ):
                torch.zeros(local_shape, dtype=torch.bfloat16).view_as(cpu_param.data)

            gpu_state = Float8Adapter.alloc_gpu(proj_state, torch.device("cuda"))
            Float8Adapter.copy_to_gpu(proj_state, gpu_state, non_blocking=False)
            gpu_param = Float8Adapter.gpu_param(proj_state, gpu_state)
            assert gpu_param.data.qdata.is_contiguous(), label
            assert gpu_param.data.scale.is_contiguous(), label
            assert tuple(gpu_param.data.shape) == local_shape, label
            assert Float8Adapter.layout_signature(gpu_param.data) == layout_sig, label

            expected_dense = src_tensor.dequantize()
            for dim, (off, sz) in enumerate(zip(offsets, local_shape, strict=True)):
                expected_dense = expected_dense.narrow(dim, off, sz)
            torch.testing.assert_close(gpu_param.data.dequantize().cpu(), expected_dense, rtol=0, atol=0)

            if supports_matmul and len(local_shape) == 2:
                x = torch.randn(3, local_shape[1], dtype=torch.bfloat16, device="cuda")
                actual_y = torch.nn.functional.linear(x, gpu_param.data)
                expected_y = torch.nn.functional.linear(x, expected_dense.to("cuda"))
                torch.testing.assert_close(actual_y, expected_y, rtol=2e-2, atol=2e-2)

    def test_float8_model_store_projection_shard0_shard1_merge_and_rejections(
        self,
        tp_mesh: Any,
    ) -> None:
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
        from torchao.quantization.granularity import PerGroup, PerRow
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor

        torch.manual_seed(1)
        rows, cols = 16, 64
        w_s0 = Float8Tensor.from_hp(torch.randn(rows, cols, dtype=torch.bfloat16), granularity=PerRow())
        w_s1 = Float8Tensor.from_hp(torch.randn(rows, cols, dtype=torch.bfloat16), granularity=PerGroup(32))

        proto = nn.Module()
        proto.fc1 = nn.Linear(cols, rows, bias=False, dtype=torch.bfloat16)
        proto.fc2 = nn.Linear(cols, rows, bias=False, dtype=torch.bfloat16)
        proto.fc1.weight = nn.Parameter(w_s0, requires_grad=False)
        proto.fc2.weight = nn.Parameter(w_s1, requires_grad=False)
        proto.tied = proto.fc2.weight
        proto.requires_grad_(False)
        store = ModelStore.from_module(proto)
        fc1_qptr = store._module_store.params["fc1.weight"].storage_tensors()[0].untyped_storage().data_ptr()

        shell = nn.Module()
        shell.fc1 = nn.Linear(cols, rows, bias=False, device="meta")
        shell.fc2 = nn.Linear(cols, rows, bias=False, device="meta")
        parallelize_module(shell, tp_mesh, {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()})
        shell.tied = shell.fc2.weight
        shell.requires_grad_(False)

        offloader = store.bind(shell)
        assert isinstance(shell.fc1.weight.data, DTensor)
        assert not requires_activation(shell.fc1.weight)
        assert not requires_activation(shell.fc2.weight)
        assert shell.fc2.weight is shell.tied
        assert shell.fc1.weight.to_local().qdata.untyped_storage().data_ptr() == fc1_qptr

        lora_a_cpu = torch.randn(4, cols, dtype=torch.bfloat16) * 0.1
        lora_b_cpu = torch.randn(rows, 4, dtype=torch.bfloat16) * 0.1
        lora = Adapter.from_state_dict(
            {
                "fc1.lora_A.weight": torch.randn(4, cols, dtype=torch.bfloat16) * 0.1,
                "fc1.lora_B.weight": torch.randn(rows, 4, dtype=torch.bfloat16) * 0.1,
                "fc2.lora_A.weight": lora_a_cpu,
                "fc2.lora_B.weight": lora_b_cpu,
            }
        )
        x = torch.randn(2, cols, dtype=torch.bfloat16, device="cuda")
        with activated_model(
            offloader,
            "cuda",
            adapters=[lora],
            adapter_strengths=[0.5],
            adapter_mode="merge",
        ) as model:
            local_w2 = model.fc2.weight.to_local()
            assert local_w2.qdata.is_contiguous()
            torch.testing.assert_close(
                local_w2.dequantize(),
                w_s1.dequantize().to("cuda") + 0.5 * (lora_b_cpu.to("cuda") @ lora_a_cpu.to("cuda")),
                rtol=0.1,
                atol=0.1,
            )
            assert torch.isfinite(model.fc1(x)).all()
            assert torch.isfinite(model.fc2(x)).all()
        offloader.deactivate()

        delta = Adapter(
            {"fc2.weight": ParameterDelta.from_tensors(dense=torch.randn(rows, cols, dtype=torch.bfloat16) * 0.05)}
        )
        with activated_model(
            offloader,
            "cuda",
            adapters=[delta],
            adapter_strengths=[1.0],
            adapter_mode="merge",
        ) as model:
            assert torch.isfinite(model.fc2(x)).all()
        offloader.deactivate()

    def test_convrot_int8_shard_projection_adapter_roundtrip(self) -> None:
        convrot_mod = pytest.importorskip("piper_kernels.weights.convrot.int8")
        convrot_cls = convrot_mod.ConvRotInt8Tensor

        from piper_offload.piper_convrot_int8_adapter import PiperConvRotInt8Adapter

        torch.manual_seed(2)
        rows, cols = 16, 128
        for use_static in (False, True):
            act_scale = torch.tensor(0.35, dtype=torch.float32) if use_static else None
            w = convrot_cls.from_hp(
                torch.randn(rows, cols, dtype=torch.bfloat16),
                group_size=64,
                act_per_tensor_scale=act_scale,
            )
            host_state = PiperConvRotInt8Adapter.capture_host(w)
            proj_state, layout_sig = PiperConvRotInt8Adapter.project_shard(
                host_state,
                offsets=(8, 64),
                local_shape=(8, 64),
            )
            q_host, s_host, a_host = host_state.storage
            q_proj, s_proj, a_proj = proj_state.storage
            assert q_host is not None and s_host is not None
            assert q_proj is not None and s_proj is not None
            assert q_proj.untyped_storage().data_ptr() == q_host.untyped_storage().data_ptr()
            assert s_proj.untyped_storage().data_ptr() == s_host.untyped_storage().data_ptr()
            if use_static:
                assert a_proj is a_host

            cpu_param = PiperConvRotInt8Adapter.cpu_param(proj_state)
            assert requires_activation(cpu_param)
            gpu_state = PiperConvRotInt8Adapter.alloc_gpu(proj_state, torch.device("cuda"))
            PiperConvRotInt8Adapter.copy_to_gpu(proj_state, gpu_state, non_blocking=False)
            gpu_param = PiperConvRotInt8Adapter.gpu_param(proj_state, gpu_state)
            assert gpu_param.data.qdata.is_contiguous()
            assert gpu_param.data.scale.is_contiguous()
            assert PiperConvRotInt8Adapter.layout_signature(gpu_param.data) == layout_sig
            torch.testing.assert_close(
                gpu_param.data.dequantize().cpu(),
                w.dequantize().narrow(0, 8, 8).narrow(1, 64, 64),
                rtol=0,
                atol=0,
            )

    def test_convrot_int8_shard_projection_static_scale_and_cuda_lifecycle(
        self,
        tp_mesh: Any,
    ) -> None:
        from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module

        convrot_mod = pytest.importorskip("piper_kernels.weights.convrot.int8")
        convrot_cls = convrot_mod.ConvRotInt8Tensor

        torch.manual_seed(2)
        rows, cols = 16, 128
        w_full = convrot_cls.from_hp(
            torch.randn(rows, cols, dtype=torch.bfloat16),
            group_size=64,
            act_per_tensor_scale=torch.tensor(0.5, dtype=torch.float32),
        )
        proto = nn.Module()
        proto.fc = nn.Linear(cols, rows, bias=False, dtype=torch.bfloat16)
        proto.fc.weight = nn.Parameter(w_full, requires_grad=False)
        proto.requires_grad_(False)
        store = ModelStore.from_module(proto)

        shell = nn.Module()
        shell.fc = nn.Linear(cols, rows, bias=False, device="meta")
        parallelize_module(shell, tp_mesh, {"fc": RowwiseParallel()})
        shell.requires_grad_(False)
        offloader = store.bind(shell)
        comp = host_component(offloader)
        assert comp is not None
        resting_sig = DTensorAdapter.layout_signature(shell.fc.weight.data)
        cached_sig = comp._instance.params["fc.weight"].target_layout[1]

        lora = Adapter.from_state_dict(
            {
                "fc.lora_A.weight": torch.randn(4, cols, dtype=torch.bfloat16) * 0.1,
                "fc.lora_B.weight": torch.randn(rows, 4, dtype=torch.bfloat16) * 0.1,
            }
        )
        x = torch.randn(3, cols, dtype=torch.bfloat16, device="cuda")
        with activated_model(
            offloader,
            "cuda",
            adapters=[lora],
            adapter_strengths=[0.25],
            adapter_mode="merge",
        ) as model:
            active_sig = DTensorAdapter.layout_signature(model.fc.weight.data)
            assert active_sig[:4] == resting_sig[:4]
            assert active_sig[5] == resting_sig[5]
            assert active_sig == cached_sig
            assert model.fc.weight.to_local().qdata.is_contiguous()
            assert torch.isfinite(model.fc(x)).all()
        offloader.deactivate()

        delta = Adapter(
            {"fc.weight": ParameterDelta.from_tensors(dense=torch.randn(rows, cols, dtype=torch.bfloat16) * 0.05)}
        )
        with activated_model(
            offloader,
            "cuda",
            adapters=[delta],
            adapter_strengths=[1.0],
            adapter_mode="merge",
        ) as model:
            assert torch.isfinite(model.fc(x)).all()
        offloader.deactivate()

    def test_quantized_shard_projection_rejections_and_atomicity(
        self,
        tp_mesh: Any,
    ) -> None:
        from torch.distributed.tensor import DTensor, Replicate
        from torchao.quantization.granularity import PerGroup, PerRow, PerTensor
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            Float8Tensor,
            QuantizeTensorToFloat8Kwargs,
        )

        from piper_offload.float8_adapter import Float8Adapter
        from piper_offload.piper_convrot_int8_adapter import PiperConvRotInt8Adapter

        convrot_mod = pytest.importorskip("piper_kernels.weights.convrot.int8")
        convrot_cls = convrot_mod.ConvRotInt8Tensor

        f8_grp = Float8Adapter.capture_host(
            Float8Tensor.from_hp(torch.randn(16, 64, dtype=torch.bfloat16), granularity=PerGroup(32))
        )
        with pytest.raises(ValueError, match="must align to quantization block_size"):
            Float8Adapter.project_shard(f8_grp, offsets=(0, 16), local_shape=(16, 48))
        with pytest.raises(ValueError, match="non-empty local shards"):
            Float8Adapter.project_shard(f8_grp, offsets=(16, 0), local_shape=(0, 64))

        f8_t = Float8Tensor.from_hp(torch.randn(32, 64, dtype=torch.bfloat16), granularity=PerRow()).t()
        with pytest.raises(ValueError, match="contiguous, untransposed"):
            Float8Adapter.project_shard(
                Float8Adapter.capture_host(f8_t),
                offsets=(0, 0),
                local_shape=(32, 32),
            )

        cr8 = PiperConvRotInt8Adapter.capture_host(
            convrot_cls.from_hp(torch.randn(16, 128, dtype=torch.bfloat16), group_size=64)
        )
        with pytest.raises(ValueError, match="must align to rotation group_size"):
            PiperConvRotInt8Adapter.project_shard(cr8, offsets=(0, 32), local_shape=(16, 64))
        with pytest.raises(ValueError, match="non-empty local shards"):
            PiperConvRotInt8Adapter.project_shard(cr8, offsets=(16, 0), local_shape=(0, 128))

        cr8_conv3d = convrot_cls.from_quantized(
            torch.randint(-127, 128, (4, 3, 3, 3, 64), dtype=torch.int8),
            torch.rand((4, 1), dtype=torch.float32) + 0.1,
            group_size=64,
            logical_dtype=torch.bfloat16,
        )
        with pytest.raises(ValueError, match="2-D matrices only"):
            PiperConvRotInt8Adapter.project_shard(
                PiperConvRotInt8Adapter.capture_host(cr8_conv3d),
                offsets=(0, 0, 0, 0, 0),
                local_shape=(2, 64, 3, 3, 3),
            )

        proto_f8_mod = pytest.importorskip("torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor")
        static_w = proto_f8_mod.PrototypeFloat8Tensor.from_hp(
            torch.randn(16, 32, dtype=torch.bfloat16),
            granularity=PerTensor(),
            act_quant_kwargs=QuantizeTensorToFloat8Kwargs(granularity=PerTensor()),
            act_quant_scale=torch.tensor(0.5, dtype=torch.float32),
        )
        static_proto = nn.Module()
        static_proto.embed = nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16), requires_grad=False)
        static_proto.w = nn.Parameter(static_w, requires_grad=False)
        static_store = ModelStore.from_module(static_proto)

        for placement in (Replicate(), _shard(0)):
            shell = nn.Module()
            orig_embed = nn.Parameter(torch.empty((8, 8), device="meta"), requires_grad=False)
            orig_w = nn.Parameter(
                DTensor.from_local(torch.empty((16, 32), device="meta"), tp_mesh, [placement], run_check=False),
                requires_grad=False,
            )
            shell.embed = orig_embed
            shell.w = orig_w
            with pytest.raises(ValueError, match="layout mismatch"):
                static_store.bind(shell)
            assert shell.embed is orig_embed
            assert shell.w is orig_w

        f8_proto = nn.Module()
        f8_proto.w = nn.Parameter(
            Float8Tensor.from_hp(torch.randn(16, 64, dtype=torch.bfloat16), granularity=PerGroup(32)),
            requires_grad=False,
        )
        f8_store = ModelStore.from_module(f8_proto)

        cpu_plain_target = nn.Module()
        cpu_plain_target.w = nn.Parameter(
            DTensor.from_local(torch.zeros((16, 64), dtype=torch.bfloat16), tp_mesh, [_shard(0)], run_check=False),
            requires_grad=False,
        )
        with pytest.raises(ValueError, match="layout mismatch"):
            f8_store.bind(cpu_plain_target)

        wrong_block_local = Float8Tensor.from_hp(torch.randn(16, 64, dtype=torch.bfloat16), granularity=PerRow())
        wrong_phys_target = nn.Module()
        wrong_phys_target.w = nn.Parameter(
            DTensor.from_local(wrong_block_local, tp_mesh, [_shard(0)], run_check=False),
            requires_grad=False,
        )
        with pytest.raises(ValueError, match="layout mismatch"):
            f8_store.bind(wrong_phys_target)

    def test_resident_float8_activation_and_materialize_inside_inference_mode(
        self,
        tp_mesh: Any,
    ) -> None:
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            parallelize_module,
        )
        from torchao.quantization.granularity import PerRow
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor

        torch.manual_seed(3)
        rows, cols = 32, 64
        w1 = Float8Tensor.from_hp(torch.randn(rows, cols, dtype=torch.bfloat16), granularity=PerRow())
        w2 = Float8Tensor.from_hp(torch.randn(cols, rows, dtype=torch.bfloat16), granularity=PerRow())
        proto = nn.Module()
        proto.fc1 = nn.Linear(cols, rows, bias=False, dtype=torch.bfloat16)
        proto.fc2 = nn.Linear(rows, cols, bias=False, dtype=torch.bfloat16)
        proto.fc1.weight = nn.Parameter(w1, requires_grad=False)
        proto.fc2.weight = nn.Parameter(w2, requires_grad=False)
        proto.requires_grad_(False)
        store = ModelStore.from_module(proto)

        shell = nn.Module()
        shell.fc1 = nn.Linear(cols, rows, bias=False, device="meta")
        shell.fc2 = nn.Linear(rows, cols, bias=False, device="meta")
        parallelize_module(shell, tp_mesh, {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()})
        shell.requires_grad_(False)
        offloader = store.bind(shell)
        comp = host_component(offloader)
        assert comp is not None
        hp1 = comp._instance.params["fc1.weight"]
        hp2 = comp._instance.params["fc2.weight"]

        lora = Adapter.from_state_dict(
            {
                "fc1.lora_A.weight": torch.randn(4, cols, dtype=torch.bfloat16) * 0.1,
                "fc1.lora_B.weight": torch.randn(rows, 4, dtype=torch.bfloat16) * 0.1,
                "fc2.lora_A.weight": torch.randn(4, rows, dtype=torch.bfloat16) * 0.1,
                "fc2.lora_B.weight": torch.randn(cols, 4, dtype=torch.bfloat16) * 0.1,
            }
        )
        x = torch.randn(2, cols, dtype=torch.bfloat16, device="cuda")
        with torch.inference_mode():
            with activated_model(offloader, "cuda") as model:
                assert not model.fc1.weight.is_inference()
                assert not model.fc2.weight.is_inference()
                assert not local_shard(model.fc1.weight).is_inference()
                assert not local_shard(model.fc2.weight).is_inference()
                assert torch.isfinite(model.fc2(model.fc1(x))).all()
            with activated_model(
                offloader,
                "cuda",
                adapters=[lora],
                adapter_strengths=[0.5],
                adapter_mode="merge",
            ) as model:
                assert not model.fc1.weight.is_inference()
                assert not model.fc2.weight.is_inference()
                assert not local_shard(model.fc1.weight).is_inference()
                assert not local_shard(model.fc2.weight).is_inference()
                assert torch.isfinite(model.fc2(model.fc1(x))).all()
            mat1 = hp1.materialize(torch.device("cuda"))
            mat2 = hp2.materialize(torch.device("cuda"))
            assert not mat1.is_inference()
            assert not mat2.is_inference()
            assert not local_shard(mat1).is_inference()
            assert not local_shard(mat2).is_inference()

def _verify_shard1_streaming_lin(
    tp_mesh: Any,
    w0: torch.Tensor,
    w1: torch.Tensor,
    global_shape: tuple[int, int],
    merge_obj: Adapter,
) -> None:
    from torch.distributed.tensor import DTensor, Shard

    from piper_offload import host_pin_manager

    rows, cols = global_shape
    mod = nn.Module()
    mod.blocks = nn.ModuleList([
        nn.Linear(cols, rows, bias=False, dtype=torch.bfloat16).eval().requires_grad_(False),
        nn.Linear(cols, rows, bias=False, dtype=torch.bfloat16).eval().requires_grad_(False),
    ])
    mod.blocks[0].weight = nn.Parameter(w0, requires_grad=False)
    mod.blocks[1].weight = nn.Parameter(w1, requires_grad=False)
    store = ModelStore.from_module(mod)
    shell = nn.Module()
    shell.blocks = nn.ModuleList([
        nn.Linear(cols, rows, bias=False, device="meta"),
        nn.Linear(cols, rows, bias=False, device="meta"),
    ])
    for blk in shell.blocks:
        blk.weight = nn.Parameter(
            DTensor.from_local(
                torch.empty(global_shape, device="meta"),
                tp_mesh,
                [Shard(1)],
                shape=global_shape,
                stride=(cols, 1),
                run_check=False,
            ),
            requires_grad=False,
        )
    off = store.bind(
        shell,
        OffloadConfig(block_paths=("blocks",), block_mode="streaming"),
    )
    try:
        x = torch.randn(2, cols, device="cuda", dtype=torch.bfloat16)
        x_dt = DTensor.from_local(x, tp_mesh, [Shard(1)], run_check=False)
        with activated_model(ModelOffloader.from_module(mod), "cuda") as ref_active, torch.inference_mode():
            exp0 = ref_active.blocks[0](x)
        with activated_model(off, "cuda") as active:
            y0 = active.blocks[0](x_dt).to_local()
            y1 = active.blocks[1](x_dt).to_local()
            torch.testing.assert_close(y0, exp0, rtol=4e-2, atol=4e-2)
            assert torch.isfinite(y1).all()
        with activated_model(
            off, "cuda", adapters=[merge_obj], adapter_strengths=[0.5], adapter_mode="merge"
        ) as active:
            y0_m = active.blocks[0](x_dt).to_local()
            assert torch.isfinite(y0_m).all() and not torch.allclose(y0_m, y0, atol=1e-2)
            assert torch.isfinite(active.blocks[1](x_dt).to_local()).all()
    finally:
        off.deactivate()
        host_pin_manager.clear()


class TestPackedAndSwizzledShardProjection:
    def test_mx_shard_projection_unswizzled_swizzled_and_cuda_lifecycle(self, tp_mesh: Any) -> None:
        from torchao.prototype.mx_formats.mx_tensor import MXTensor, QuantizeTensorToMXKwargs
        from torchao.quantization.quantize_.common import KernelPreference

        from piper_offload import host_pin_manager
        from piper_offload.mx_adapter import MxAdapter

        host_pin_manager.clear()
        torch.manual_seed(31)
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        elem_dtype = fp4_dtype if fp4_dtype is not None else torch.float8_e4m3fn
        act_kw = QuantizeTensorToMXKwargs(
            elem_dtype=elem_dtype, block_size=32, is_swizzled_scales=True, kernel_preference=KernelPreference.EMULATED
        )
        w = torch.randn(192, 160, dtype=torch.bfloat16)
        mx_swiz = MXTensor.to_mx(
            w,
            elem_dtype,
            block_size=32,
            kernel_preference=KernelPreference.EMULATED,
            act_quant_kwargs=act_kw,
            is_swizzled_scales=True,
        )
        state = MxAdapter.capture_host(mx_swiz)
        proj_col, _ = MxAdapter.project_shard(state, offsets=(0, 128), local_shape=(192, 32))
        assert proj_col.target_shapes == ((192, 16 if fp4_dtype is not None else 32), (64, 16))
        assert tuple(proj_col.storage[1].shape) == (2, 1, 512)
        assert proj_col.storage[1].untyped_storage().data_ptr() == state.storage[1].untyped_storage().data_ptr()

        gpu = MxAdapter.alloc_gpu(proj_col, torch.device("cuda"))
        try:
            with host_pin_manager.acquire(MxAdapter.storage_tensors(proj_col), pin=True):
                MxAdapter.copy_to_gpu(proj_col, gpu, non_blocking=True)
            torch.cuda.synchronize()
            mx_gpu = MxAdapter.gpu_param(proj_col, gpu).data
            ref_tail = MXTensor.to_mx(
                w[:, 128:160].contiguous(),
                elem_dtype,
                block_size=32,
                kernel_preference=KernelPreference.EMULATED,
                is_swizzled_scales=True,
            )
            assert tuple(mx_gpu.scale.shape) == (64, 16)
            assert torch.equal(mx_gpu.scale.cpu().view(torch.uint8), ref_tail.scale.view(torch.uint8))
            assert torch.equal(mx_gpu.qdata.cpu().view(torch.uint8), ref_tail.qdata.view(torch.uint8))
        finally:
            host_pin_manager.clear()

        u_kw = QuantizeTensorToMXKwargs(
            elem_dtype=torch.float8_e4m3fn, block_size=32, kernel_preference=KernelPreference.EMULATED
        )
        u_mx = MXTensor.to_mx(
            torch.randn(128, 128, dtype=torch.bfloat16),
            torch.float8_e4m3fn,
            block_size=32,
            kernel_preference=KernelPreference.EMULATED,
            act_quant_kwargs=u_kw,
            is_swizzled_scales=False,
        )
        u_state = MxAdapter.capture_host(u_mx)
        u_proj, _ = MxAdapter.project_shard(u_state, offsets=(0, 64), local_shape=(128, 64))
        assert u_proj.storage[0].untyped_storage().data_ptr() == u_state.storage[0].untyped_storage().data_ptr()

        lin_w0 = MXTensor.to_mx(
            torch.randn(128, 128, dtype=torch.bfloat16),
            elem_dtype,
            block_size=32,
            kernel_preference=KernelPreference.EMULATED,
            act_quant_kwargs=act_kw,
            is_swizzled_scales=True,
        )
        lin_w1 = MXTensor.to_mx(
            torch.randn(128, 128, dtype=torch.bfloat16),
            elem_dtype,
            block_size=32,
            kernel_preference=KernelPreference.EMULATED,
            act_quant_kwargs=act_kw,
            is_swizzled_scales=True,
        )
        lora = Adapter.from_state_dict({
            "blocks.0.lora_A.weight": torch.randn(4, 128, dtype=torch.bfloat16) * 0.05,
            "blocks.0.lora_B.weight": torch.randn(128, 4, dtype=torch.bfloat16) * 0.05,
        })
        _verify_shard1_streaming_lin(tp_mesh, lin_w0, lin_w1, (128, 128), lora)

    def test_nvfp4_and_piper_nvfp4_shard_projection_and_cuda_lifecycle(self, tp_mesh: Any) -> None:
        from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        from piper_offload import host_pin_manager
        from piper_offload.nvfp4_adapter import Nvfp4Adapter

        host_pin_manager.clear()
        torch.manual_seed(37)
        w2d = torch.randn(192, 160, dtype=torch.bfloat16)
        gs2d = torch.tensor(0.5, dtype=torch.float32)
        nv2d = NVFP4Tensor.to_nvfp4(w2d, per_tensor_scale=gs2d, is_swizzled_scales=True, use_triton_kernel=False)
        proj2d, _ = Nvfp4Adapter.project_shard(
            Nvfp4Adapter.capture_host(nv2d), offsets=(0, 128), local_shape=(192, 32)
        )
        assert proj2d.target_shapes == ((192, 16), (64, 16), (), None)
        gpu2d = Nvfp4Adapter.alloc_gpu(proj2d, torch.device("cuda"))
        Nvfp4Adapter.copy_to_gpu(proj2d, gpu2d, non_blocking=False)
        nv2d_gpu = Nvfp4Adapter.gpu_param(proj2d, gpu2d).data
        ref2d = NVFP4Tensor.to_nvfp4(
            w2d[:, 128:160].contiguous(),
            per_tensor_scale=gs2d,
            is_swizzled_scales=True,
            use_triton_kernel=False,
        )
        assert torch.equal(nv2d_gpu.scale.cpu().view(torch.uint8), ref2d.scale.view(torch.uint8))
        assert torch.equal(nv2d_gpu.qdata.cpu().view(torch.uint8), ref2d.qdata.view(torch.uint8))

        base3d = NVFP4Tensor.to_nvfp4(
            torch.randn(4, 256, 128, dtype=torch.bfloat16),
            per_tensor_scale=gs2d,
            is_swizzled_scales=True,
            use_triton_kernel=False,
        )
        p3d = PiperNVFP4Tensor(
            base3d.qdata,
            base3d.scale,
            base3d.block_size,
            base3d.orig_dtype,
            torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32).view(4, 1, 1),
            torch.tensor(0.25, dtype=torch.float32),
            True,
            False,
            None,
            high_first=True,
        )
        proj3d, _ = Nvfp4Adapter.project_shard(
            Nvfp4Adapter.capture_host(p3d), offsets=(2, 0, 64), local_shape=(2, 256, 64)
        )
        assert proj3d.target_shapes == ((2, 256, 32), (2, 64, 16), (2, 1, 1), ())
        gpu3d = Nvfp4Adapter.alloc_gpu(proj3d, torch.device("cuda"))
        try:
            with host_pin_manager.acquire(Nvfp4Adapter.storage_tensors(proj3d), pin=True):
                Nvfp4Adapter.copy_to_gpu(proj3d, gpu3d, non_blocking=True)
            torch.cuda.synchronize()
            rebuilt = Nvfp4Adapter.gpu_param(proj3d, gpu3d).data
            assert rebuilt.high_first is True and tuple(rebuilt.scale.shape) == (2, 64, 16)
            torch.testing.assert_close(rebuilt.dequantize().cpu(), p3d.dequantize()[2:4, :, 64:128], rtol=0, atol=0)
        finally:
            host_pin_manager.clear()

        nv_b0 = NVFP4Tensor.to_nvfp4(
            torch.randn(256, 128, dtype=torch.bfloat16),
            per_tensor_scale=gs2d,
            is_swizzled_scales=True,
            use_triton_kernel=False,
        )
        nv_b1 = NVFP4Tensor.to_nvfp4(
            torch.randn(256, 128, dtype=torch.bfloat16),
            per_tensor_scale=gs2d,
            is_swizzled_scales=True,
            use_triton_kernel=False,
        )
        dense = Adapter(
            {"blocks.0.weight": ParameterDelta.from_tensors(dense=torch.randn(256, 128, dtype=torch.bfloat16) * 0.02)}
        )
        _verify_shard1_streaming_lin(
            tp_mesh, PiperNVFP4Tensor.from_torchao(nv_b0), PiperNVFP4Tensor.from_torchao(nv_b1), (256, 128), dense
        )

    def test_convrot_nvfp4_shard_projection_and_cuda_lifecycle(self, tp_mesh: Any) -> None:
        from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        from piper_offload import host_pin_manager
        from piper_offload.piper_convrot_nvfp4_adapter import PiperConvRotNVFP4Adapter

        host_pin_manager.clear()
        torch.manual_seed(41)
        cr_unswiz = ConvRotNVFP4Tensor.from_torchao(
            NVFP4Tensor.to_nvfp4(
                torch.randn(128, 128, dtype=torch.bfloat16),
                per_tensor_scale=torch.tensor(0.5, dtype=torch.float32),
                is_swizzled_scales=False,
                use_triton_kernel=False,
            ),
            group_size=64,
        )
        u_state = PiperConvRotNVFP4Adapter.capture_host(cr_unswiz)
        u_s0, _ = PiperConvRotNVFP4Adapter.project_shard(u_state, offsets=(17, 0), local_shape=(19, 128))
        assert not requires_activation(PiperConvRotNVFP4Adapter.cpu_param(u_s0))
        u_s1, _ = PiperConvRotNVFP4Adapter.project_shard(u_state, offsets=(0, 64), local_shape=(128, 64))
        u_gpu = PiperConvRotNVFP4Adapter.alloc_gpu(u_s1, torch.device("cuda"))
        with host_pin_manager.acquire(PiperConvRotNVFP4Adapter.storage_tensors(u_s1), pin=True):
            PiperConvRotNVFP4Adapter.copy_to_gpu(u_s1, u_gpu, non_blocking=True)
        torch.cuda.synchronize()
        rebuilt_u = PiperConvRotNVFP4Adapter.gpu_param(u_s1, u_gpu).data
        u_ref = cr_unswiz.dequantize()[:, 64:128]
        torch.testing.assert_close(rebuilt_u.dequantize().cpu(), u_ref, rtol=0, atol=0)
        lora_b = torch.randn(128, 4, device="cuda", dtype=torch.bfloat16) * 0.05
        lora_a = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16) * 0.05
        PiperConvRotNVFP4Adapter.merge_lora_(rebuilt_u, lora_b, lora_a, 0.5)
        u_merged = rebuilt_u.dequantize().cpu()
        assert torch.isfinite(u_merged).all() and not torch.allclose(u_merged, u_ref, atol=1e-2)
        torch.testing.assert_close(u_merged, u_ref + 0.5 * (lora_b @ lora_a).cpu(), rtol=3e-1, atol=4e-1)
        host_pin_manager.clear()

        sw_b0 = NVFP4Tensor.to_nvfp4(
            torch.randn(256, 128, dtype=torch.bfloat16),
            per_tensor_scale=torch.tensor(0.5, dtype=torch.float32),
            is_swizzled_scales=True,
            use_triton_kernel=False,
        )
        sw_b1 = NVFP4Tensor.to_nvfp4(
            torch.randn(256, 128, dtype=torch.bfloat16),
            per_tensor_scale=torch.tensor(0.5, dtype=torch.float32),
            is_swizzled_scales=True,
            use_triton_kernel=False,
        )
        dense = Adapter(
            {"blocks.0.weight": ParameterDelta.from_tensors(dense=torch.randn(256, 128, dtype=torch.bfloat16) * 0.1)}
        )
        _verify_shard1_streaming_lin(
            tp_mesh,
            ConvRotNVFP4Tensor.from_torchao(sw_b0, group_size=64),
            ConvRotNVFP4Tensor.from_torchao(sw_b1, group_size=64),
            (256, 128),
            dense,
        )

    def test_mapped_checkpoint_swizzled_pin_manager_cuda_copy(self, tmp_path: Any) -> None:
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        from piper_offload import MappedCheckpoint, host_pin_manager
        from piper_offload.mx_adapter import MxAdapter
        from piper_offload.nvfp4_adapter import Nvfp4Adapter
        from tests.test_model_store import _write_safetensors

        host_pin_manager.clear()
        torch.manual_seed(57)
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        mx_dt = fp4_dtype if fp4_dtype is not None else torch.float8_e4m3fn
        mx_w = MXTensor.to_mx(
            torch.randn(256, 256, dtype=torch.bfloat16), mx_dt, block_size=32, is_swizzled_scales=True
        )
        nv_w = NVFP4Tensor.to_nvfp4(
            torch.randn(256, 128, dtype=torch.bfloat16),
            per_tensor_scale=torch.tensor(0.5, dtype=torch.float32),
            is_swizzled_scales=True,
            use_triton_kernel=False,
        )
        ckpt_path = tmp_path / "swiz.safetensors"
        _write_safetensors(
            ckpt_path,
            {
                "pad": torch.ones(17, dtype=torch.uint8),
                "mx_q": mx_w.qdata,
                "mx_s": mx_w.scale,
                "nv_q": nv_w.qdata,
                "nv_s": nv_w.scale,
            },
        )
        ckpt = MappedCheckpoint(ckpt_path)
        try:
            mx_q = ckpt.get_tensor("mx_q")
            mx_mapped = MXTensor(
                mx_q, ckpt.get_tensor("mx_s"), mx_w.elem_dtype, mx_w.block_size, mx_w.orig_dtype,
                mx_w.kernel_preference, mx_w.act_quant_kwargs, True,
            )
            nv_mapped = NVFP4Tensor(
                ckpt.get_tensor("nv_q"), ckpt.get_tensor("nv_s"), nv_w.block_size, nv_w.orig_dtype,
                torch.tensor(0.5, dtype=torch.float32), None, True, False, None,
            )
            mx_proj, _ = MxAdapter.project_shard(
                MxAdapter.capture_host(mx_mapped), offsets=(0, 128), local_shape=(256, 128)
            )
            nv_proj, _ = Nvfp4Adapter.project_shard(
                Nvfp4Adapter.capture_host(nv_mapped), offsets=(0, 64), local_shape=(256, 64)
            )
            assert mx_proj.storage[1].storage_offset() == 512 and tuple(mx_proj.storage[1].shape) == (2, 1, 512)
            assert nv_proj.storage[1].storage_offset() == 512 and tuple(nv_proj.storage[1].shape) == (2, 1, 512)
            mx_gpu = MxAdapter.alloc_gpu(mx_proj, torch.device("cuda"))
            nv_gpu = Nvfp4Adapter.alloc_gpu(nv_proj, torch.device("cuda"))
            with host_pin_manager.acquire(
                (*MxAdapter.storage_tensors(mx_proj), *Nvfp4Adapter.storage_tensors(nv_proj)), pin=True
            ):
                MxAdapter.copy_to_gpu(mx_proj, mx_gpu, non_blocking=True)
                Nvfp4Adapter.copy_to_gpu(nv_proj, nv_gpu, non_blocking=True)
            torch.cuda.synchronize()
            mx_rec = MxAdapter.gpu_param(mx_proj, mx_gpu).data
            nv_rec = Nvfp4Adapter.gpu_param(nv_proj, nv_gpu).data
            torch.testing.assert_close(mx_rec.dequantize().cpu(), mx_w.dequantize()[:, 128:256], rtol=0, atol=0)
            torch.testing.assert_close(nv_rec.dequantize().cpu(), nv_w.dequantize()[:, 64:128], rtol=0, atol=0)
        finally:
            host_pin_manager.clear()
            ckpt.close()
