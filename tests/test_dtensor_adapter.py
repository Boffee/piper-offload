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
    LoRA,
    LoRATransform,
    ModelOffloader,
    ScaledLoRAFactor,
    merge_lora,
)
from piper_offload.dtensor_adapter import DTensorAdapter
from piper_offload.int8_adapter import Int8Adapter
from piper_offload.pinned_param import PinnedParam
from piper_offload.tensor_adapters import (
    CpuRoundTripTensorAdapter,
    DequantRequantTensorAdapter,
    LoRAMergeTensorAdapter,
    ParameterDataSwapTensorAdapter,
    RegularAdapter,
    TensorCopyIntoAdapter,
)
from piper_offload.tensor_adapter_registry import (
    param_representation,
    select_adapter,
    tensor_id,
)
from tests.conftest import activated_model

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
        pinned_param = PinnedParam(nn.Parameter(dt, requires_grad=False))
        # A plain local shard is moved by the registry's RegularAdapter — the
        # DTensorAdapter only adds the distributed wrapper on top.
        assert isinstance(pinned_param.pinned_state.inner, RegularAdapter)
        assert isinstance(pinned_param.pinned_state.inner, LoRAMergeTensorAdapter)

    def test_host_adoption_rejects_cuda_local_shard(self, tp_mesh: Any) -> None:
        dt, _ = _dtensor_weight(tp_mesh)

        with pytest.raises(NotImplementedError, match="host adoption"):
            PinnedParam(
                nn.Parameter(dt, requires_grad=False),
                pin_memory=False,
            )

    def test_pinned_param_roundtrip_reconstructs_dtensor(self, tp_mesh: Any) -> None:
        dt, full = _dtensor_weight(tp_mesh)
        pinned_param = PinnedParam(nn.Parameter(dt, requires_grad=False))

        # Resting state: still a DTensor (type-stable), but on a CPU mesh so
        # the local shard stays on the host (no GPU memory held).
        cpu = pinned_param.make_cpu_param()
        assert _is_dtensor(cpu.data)
        assert cpu.data.device_mesh.device_type == "cpu"
        assert cpu.data.to_local().device.type == "cpu"
        assert cpu.data.placements == dt.placements
        assert torch.equal(cpu.data.to_local(), dt.to_local().cpu())

        # Resident state: the DTensor is reconstructed on the GPU.
        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()

        assert _is_dtensor(gpu_param.data)
        assert gpu_param.data.to_local().is_cuda
        assert gpu_param.data.placements == dt.placements
        assert gpu_param.data.device_mesh == dt.device_mesh
        assert torch.equal(gpu_param.data.full_tensor(), full)

    def test_one_shot_materialize_reconstructs_cuda_mesh(self, tp_mesh: Any) -> None:
        dt, full = _dtensor_weight(tp_mesh)
        pinned_param = PinnedParam(nn.Parameter(dt, requires_grad=False))
        assert pinned_param.shape == dt.shape

        materialized = pinned_param.materialize(
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
        pinned_param = PinnedParam(nn.Parameter(dt, requires_grad=False))
        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
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

    def test_bind_layout_relaxes_inner_dtype_keeps_global_shape(self, tp_mesh: Any) -> None:
        bf16 = _dtensor_weight(tp_mesh, dtype=torch.bfloat16)[0]
        fp32 = _dtensor_weight(tp_mesh, dtype=torch.float32)[0]

        # bind_layout delegates to the inner adapter's bind signature, which
        # drops dtype (for meta-skeleton binding) — so the two compare equal;
        # the strict layout_signature keeps dtype, so they differ.
        assert DTensorAdapter.layout_signature(bf16) != DTensorAdapter.layout_signature(fp32)
        assert DTensorAdapter.bind_layout_signature(bf16) == DTensorAdapter.bind_layout_signature(fp32)
        # Both keys carry the GLOBAL shape (gpu_param replays it; uneven shards
        # are not pinned by the local shape alone).
        assert tuple(bf16.shape) in DTensorAdapter.layout_signature(bf16)
        assert tuple(bf16.shape) in DTensorAdapter.bind_layout_signature(bf16)

    def test_tensor_id_keys_off_local_not_dtensor_dataptr(self, tp_mesh: Any) -> None:
        dt, _ = _dtensor_weight(tp_mesh)
        # The DTensor's own data_ptr is 0 — using it would collapse tied-weight
        # dedup. Two independently-allocated DTensors must get distinct ids.
        assert dt.data_ptr() == 0
        other, _ = _dtensor_weight(tp_mesh)
        assert tensor_id(dt) != tensor_id(other)
        assert tensor_id(dt) == tensor_id(dt)
        # The global shape is in the identity key so aliased local shards with
        # different global views are not deduped onto one PinnedParam.
        assert tuple(dt.shape) in tensor_id(dt)

    def test_cache_bytes_counts_local_shard(self, tp_mesh: Any) -> None:
        dt, _ = _dtensor_weight(tp_mesh, rows=16, cols=8)
        pinned_param = PinnedParam(nn.Parameter(dt, requires_grad=False))
        local = dt.to_local()
        # Local-shard bytes (== global only because world_size==1; on N ranks
        # this is the ~1/N local footprint, not the global numel).
        assert DTensorAdapter.cache_bytes(pinned_param.pinned_state) == (local.numel() * local.element_size())

    def test_compute_dtype_delegates_to_local(self, tp_mesh: Any) -> None:
        dt, _ = _dtensor_weight(tp_mesh, dtype=torch.bfloat16)
        assert DTensorAdapter.compute_dtype(dt) is torch.bfloat16

    def test_compute_dtype_via_pinned_param_property(self, tp_mesh: Any) -> None:
        # Regression: PinnedParam.compute_dtype feeds the bare local shard
        # (what cpu_param yields, not a DTensor) into the adapter; it must not
        # require a live DTensor. Every other adapter asserts this property.
        dt, _ = _dtensor_weight(tp_mesh, dtype=torch.bfloat16)
        pinned_param = PinnedParam(nn.Parameter(dt, requires_grad=False))
        assert pinned_param.compute_dtype is torch.bfloat16

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
                _strength: float,
                *,
                rounding_seed: int | None = None,
            ) -> None:
                del rounding_seed
                calls.append((b, a))

        local = torch.empty(3, 4, device="cuda")
        context = dtensor_adapter_module._DTensorMergeContext(
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
        pw = ModelOffloader.from_module(net, block_paths=["blocks"])
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
        lora = LoRA.from_state_dict(
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
                loras=[lora],
                lora_strengths=[strength],
                lora_mode="merge",
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
        lora = LoRA.from_state_dict(
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
            merge_lora(model, [(lora, 0.25)])

        # merge_lora preflights every target before mutating any of them.
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
        lora = LoRA.from_state_dict(
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
            merge_lora(model, [(lora, 0.25)])

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
        lora = LoRA.from_state_dict(
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
            loras=[lora],
            lora_strengths=[strength],
            lora_mode="routed",
        ):
            actual = model(inputs)
            assert _is_dtensor(actual)
            assert actual.device_mesh == tp_mesh
            torch.testing.assert_close(actual.full_tensor(), expected)

    def test_advertises_merge_but_not_training_capabilities(
        self,
        tp_mesh: Any,
    ) -> None:
        # Frozen-inference scope: local LoRA merge is available, but CPU
        # round-trip, dequant/requant, copy_into, and trainable .data swap stay
        # hidden even when the inner adapter has those capabilities.
        adapter = select_adapter(_dtensor_weight(tp_mesh)[0])
        assert isinstance(adapter, LoRAMergeTensorAdapter)
        assert not isinstance(adapter, CpuRoundTripTensorAdapter)
        assert not isinstance(adapter, DequantRequantTensorAdapter)
        assert not isinstance(adapter, TensorCopyIntoAdapter)
        assert not isinstance(adapter, ParameterDataSwapTensorAdapter)

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

        pinned_param = PinnedParam(nn.Parameter(dt, requires_grad=False))
        assert isinstance(pinned_param.pinned_state.inner, Float8Adapter)

        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()

        assert _is_dtensor(gpu_param.data)
        # the reconstructed local shard is still the Float8 quant subclass
        assert isinstance(select_adapter(gpu_param.data.to_local()), Float8Adapter)
        assert gpu_param.data.placements == dt.placements

    def test_packed_int8_extreme_strength_delegates_factor_aware_staging(
        self,
        tp_mesh: Any,
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
        pre_scale = torch.tensor(1e30, device="cuda")
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

        a = torch.full((1, cols), 1e10)
        b = torch.full((rows, 1), 1e-10)
        strength = 1e30
        factors = [ScaledLoRAFactor.from_tensors(a.clone(), b.clone(), strength) for _ in range(2)]
        param = nn.Parameter(weight, requires_grad=False)
        x = torch.full((3, cols), 1e-30, device="cuda")
        routed_output = torch.nn.functional.linear(x, local)
        for _ in factors:
            routed_output = routed_output + ((x @ a.to("cuda").T) * strength) @ b.to("cuda").T

        prepared_targets: list[torch.Tensor] = []
        original_prepared_merge = Int8Adapter.merge_prepared_lora_

        def record_prepared_merge(
            target: torch.Tensor,
            packed_b: torch.Tensor,
            packed_a: torch.Tensor,
            packed_strength: float,
            *,
            rounding_seed: int | None = None,
        ) -> None:
            prepared_targets.append(target)
            assert torch.isfinite(packed_a).all()
            torch.testing.assert_close(
                packed_a,
                torch.full_like(packed_a, 1e10),
            )
            original_prepared_merge(
                target,
                packed_b,
                packed_a,
                packed_strength,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(
            Int8Adapter,
            "merge_prepared_lora_",
            staticmethod(record_prepared_merge),
        )

        pre_scale_ptr = local.act_pre_scale.data_ptr()
        transform = LoRATransform(factors)
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
            strength: float,
            *,
            rounding_seed: int | None = None,
        ) -> None:
            calls.append((type(target), tuple(b.shape), tuple(a.shape)))
            original_merge(
                target,
                b,
                a,
                strength,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(
            Float8Adapter,
            "merge_lora_",
            staticmethod(record_local_merge),
        )

        model = Net()
        lora = LoRA.from_state_dict(
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
            loras=[lora],
            lora_strengths=[0.25],
            lora_mode="merge",
        ):
            assert calls == [
                (
                    type(local_weight),
                    (out_dim, rank),
                    (rank, in_dim),
                )
            ]
