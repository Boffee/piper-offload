"""Tests for DTensor Adapter shard localization."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.tensor import Partial, Replicate, Shard

import piper_offload.dtensor_adapter as dtensor_adapter_module
import piper_offload.lora as lora_module
from piper_offload import (
    LoRATransform,
    ParameterDelta,
    ParameterDeltaTransform,
    ScaledLoRAFactor,
)
from piper_offload.dtensor_adapter import DTensorAdapter, _local_shape_and_offsets
from piper_offload import derive_seed
from piper_offload.tensor_adapters import RegularAdapter

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _run_two_rank_merge(rank: int, world_size: int, init_file: str) -> None:
    from torch.distributed.tensor import distribute_tensor, init_device_mesh

    store = dist.FileStore(init_file, world_size)
    dist.init_process_group(
        "gloo",
        store=store,
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
        weight = torch.arange(11 * 7, dtype=torch.float32).reshape(11, 7) / 10
        a = torch.arange(3 * 7, dtype=torch.float32).reshape(3, 7) / 20
        b = torch.arange(11 * 3, dtype=torch.float32).reshape(11, 3) / 30
        strength = 0.25
        expected = weight + strength * (b @ a)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, strength)])

        for shard_dim in (0, 1):
            target = distribute_tensor(weight.clone(), mesh, [Shard(shard_dim)])
            param = nn.Parameter(target, requires_grad=False)
            transform.validate_target(param)
            transform.apply(param)
            torch.testing.assert_close(target.full_tensor(), expected)

            direct_target = distribute_tensor(
                weight.clone(),
                mesh,
                [Shard(shard_dim)],
            )
            DTensorAdapter.merge_lora_(direct_target, b, a, strength)
            torch.testing.assert_close(direct_target.full_tensor(), expected)

            dense = torch.arange(
                weight.numel(),
                dtype=weight.dtype,
            ).reshape_as(weight).div(50)
            dense_target = distribute_tensor(
                weight.clone(),
                mesh,
                [Shard(shard_dim)],
            )
            dense_transform = ParameterDeltaTransform(
                [
                    ParameterDelta.from_tensors(
                        a=a,
                        b=b,
                        dense=dense,
                        pin_memory=False,
                    ).scaled(strength)
                ]
            )
            dense_param = nn.Parameter(dense_target, requires_grad=False)
            dense_transform.validate_parameter(dense_param)
            dense_transform.apply_parameter(dense_param)
            torch.testing.assert_close(
                dense_target.full_tensor(),
                expected + strength * dense,
            )

        empty_cases = (
            (torch.arange(2, dtype=torch.float32).reshape(1, 2), 0),
            (torch.arange(2, dtype=torch.float32).reshape(2, 1), 1),
        )
        for empty_weight, shard_dim in empty_cases:
            empty_a = torch.arange(
                2 * empty_weight.shape[1],
                dtype=torch.float32,
            ).reshape(2, empty_weight.shape[1])
            empty_b = torch.arange(
                empty_weight.shape[0] * 2,
                dtype=torch.float32,
            ).reshape(empty_weight.shape[0], 2)
            empty_transform = LoRATransform([ScaledLoRAFactor.from_tensors(empty_a, empty_b, strength)])
            empty_target = distribute_tensor(
                empty_weight.clone(),
                mesh,
                [Shard(shard_dim)],
            )

            empty_param = nn.Parameter(empty_target, requires_grad=False)
            empty_transform.validate_target(empty_param)
            empty_transform.apply(empty_param)

            torch.testing.assert_close(
                empty_target.full_tensor(),
                empty_weight + strength * (empty_b @ empty_a),
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("coordinate", "expected_shape", "expected_offsets"),
    [
        ((0,), (3, 7), (0, 0)),
        ((1,), (3, 7), (3, 0)),
        ((2,), (3, 7), (6, 0)),
        ((3,), (2, 7), (9, 0)),
    ],
)
def test_uneven_output_shards(
    coordinate: tuple[int, ...],
    expected_shape: tuple[int, ...],
    expected_offsets: tuple[int, ...],
) -> None:
    assert _local_shape_and_offsets(
        (11, 7),
        (4,),
        coordinate,
        (Shard(0),),
    ) == (expected_shape, expected_offsets)


def test_input_shard_localizes_weight_columns() -> None:
    assert _local_shape_and_offsets(
        (8, 11),
        (4,),
        (3,),
        (Shard(1),),
    ) == ((8, 2), (0, 9))


def test_two_dimensional_mesh_localizes_rows_and_columns() -> None:
    assert _local_shape_and_offsets(
        (5, 7),
        (2, 3),
        (1, 2),
        (Shard(0), Shard(1)),
    ) == ((2, 1), (3, 6))


def test_replicated_mesh_dimension_does_not_change_local_shape() -> None:
    assert _local_shape_and_offsets(
        (5, 7),
        (2, 3),
        (1, 2),
        (Replicate(), Shard(1)),
    ) == ((5, 1), (0, 6))


def test_repeated_sharding_of_one_tensor_dimension() -> None:
    assert _local_shape_and_offsets(
        (11, 7),
        (2, 2),
        (1, 1),
        (Shard(0), Shard(0)),
    ) == ((2, 7), (9, 0))


def test_empty_repeated_shard_uses_pytorch_global_end_offset() -> None:
    assert _local_shape_and_offsets(
        (1, 2),
        (3, 2),
        (0, 1),
        (Shard(1), Shard(1)),
    ) == ((1, 0), (0, 2))


@pytest.mark.parametrize("factor_count", [1, 2])
def test_transform_localizes_factors_before_staging(
    monkeypatch: pytest.MonkeyPatch,
    factor_count: int,
) -> None:
    target = nn.Parameter(torch.zeros(4, 3), requires_grad=False)
    context = dtensor_adapter_module._DTensorLayoutContext(
        global_shape=(6, 8),
        local_shape=(4, 3),
        offsets=(2, 5),
        local=target.data,
        inner=RegularAdapter(),
    )
    monkeypatch.setattr(
        dtensor_adapter_module,
        "_merge_context",
        lambda _target: context,
    )
    monkeypatch.setattr(
        dtensor_adapter_module,
        "_layout_context",
        lambda _target: context,
    )
    monkeypatch.setattr(
        DTensorAdapter,
        "logical_shape",
        staticmethod(lambda _target: context.global_shape),
    )
    monkeypatch.setattr(
        lora_module,
        "_select_lora_merge_adapter",
        lambda _target: DTensorAdapter(),
    )

    staged_inputs: list[tuple[list[tuple[tuple[int, ...], tuple[int, ...]]], tuple[int, ...]]] = []
    original_stage = LoRATransform._stage_single_or_packed_update.__func__

    def record_stage(
        cls: type[LoRATransform],
        data: torch.Tensor,
        factors: list[lora_module._MaterializedWeightFactor],
        *,
        logical_shape: tuple[int, ...],
        compute_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        staged_inputs.append(
            (
                [(tuple(factor.a.shape), tuple(factor.b.shape)) for factor in factors],
                logical_shape,
            )
        )
        return original_stage(
            cls,
            data,
            factors,
            logical_shape=logical_shape,
            compute_dtype=compute_dtype,
        )

    monkeypatch.setattr(
        LoRATransform,
        "_stage_single_or_packed_update",
        classmethod(record_stage),
    )

    factor_inputs = [
        (
            torch.randn(rank, 8),
            torch.randn(6, rank),
            0.25 * (rank - 1),
        )
        for rank in range(2, 2 + factor_count)
    ]
    transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, strength) for a, b, strength in factor_inputs])

    transform.validate_target(target)
    transform.apply(target)

    expected_staging = (
        [((rank, 3), (4, rank)) for rank in range(2, 2 + factor_count)],
        (4, 3),
    )
    assert staged_inputs == [
        expected_staging,  # validation
        expected_staging,  # application
    ]
    expected = torch.zeros_like(target)
    for a, b, strength in factor_inputs:
        expected.addmm_(b[2:6], a[:, 5:8], alpha=strength)
    torch.testing.assert_close(target, expected)


def test_stochastic_merge_forwards_seed_to_local_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = nn.Parameter(torch.zeros(4, 3), requires_grad=False)
    context = dtensor_adapter_module._DTensorLayoutContext(
        global_shape=(6, 8),
        local_shape=(4, 3),
        offsets=(2, 5),
        local=target.data,
        inner=RegularAdapter(),
    )
    monkeypatch.setattr(
        dtensor_adapter_module,
        "_merge_context",
        lambda _target: context,
    )
    monkeypatch.setattr(
        dtensor_adapter_module,
        "_layout_context",
        lambda _target: context,
    )
    monkeypatch.setattr(
        DTensorAdapter,
        "logical_shape",
        staticmethod(lambda _target: context.global_shape),
    )
    monkeypatch.setattr(
        lora_module,
        "_select_lora_merge_adapter",
        lambda _target: DTensorAdapter(),
    )

    captured: list[int] = []
    original_merge = RegularAdapter.merge_lora_

    def tracked_merge(
        local_target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        assert rounding_seed is not None
        captured.append(rounding_seed)
        original_merge(
            local_target,
            b,
            a,
            strength,
            rounding_seed=rounding_seed,
        )

    monkeypatch.setattr(
        RegularAdapter,
        "merge_lora_",
        staticmethod(tracked_merge),
    )
    transform = LoRATransform(
        [
            ScaledLoRAFactor.from_tensors(
                torch.randn(2, 8),
                torch.randn(6, 2),
                0.25,
            )
        ],
        stochastic_rounding=True,
        target_key="sharded.weight",
    )

    transform.validate_target(target)
    transform.apply(target)

    expected_seed = dtensor_adapter_module._localize_rounding_seed(
        derive_seed("sharded.weight", 0),
        context.offsets,
    )
    assert captured == [expected_seed]


def test_dtensor_rounding_seed_decorrelates_shards_and_aligns_replicas() -> None:
    seed = derive_seed("sharded.weight", 0)
    replica = dtensor_adapter_module._localize_rounding_seed(seed, (0, 0))

    assert replica == dtensor_adapter_module._localize_rounding_seed(seed, (0, 0))
    assert replica != dtensor_adapter_module._localize_rounding_seed(seed, (2, 0))
    assert replica != dtensor_adapter_module._localize_rounding_seed(seed, (0, 5))
    assert dtensor_adapter_module._localize_rounding_seed(None, (2, 5)) is None


def test_adapter_owned_merge_receives_contiguous_local_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = dtensor_adapter_module._DTensorLayoutContext(
        global_shape=(6, 8),
        local_shape=(6, 4),
        offsets=(0, 4),
        local=torch.empty(6, 4),
        inner=RegularAdapter(),
    )
    monkeypatch.setattr(
        dtensor_adapter_module,
        "_merge_context",
        lambda _target: context,
    )

    received: list[tuple[tuple[int, ...], tuple[int, ...], bool, bool]] = []

    def record_merge(
        _target: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        _strength: float,
        *,
        rounding_seed: int | None = None,
    ) -> None:
        del rounding_seed
        received.append(
            (
                tuple(b.shape),
                tuple(a.shape),
                b.is_contiguous(),
                a.is_contiguous(),
            )
        )

    monkeypatch.setattr(
        RegularAdapter,
        "merge_lora_",
        staticmethod(record_merge),
    )

    DTensorAdapter.merge_lora_(
        torch.empty(0),
        torch.randn(6, 3),
        torch.randn(3, 8)[:, 4:],
        0.25,
    )

    assert received == [((6, 3), (3, 4), True, True)]


def test_partial_placement_has_actionable_error() -> None:
    with pytest.raises(
        ValueError,
        match="supports only Replicate and contiguous Shard placements",
    ):
        _local_shape_and_offsets(
            (8, 8),
            (2,),
            (0,),
            (Partial(),),
        )


def test_mesh_metadata_lengths_must_match() -> None:
    with pytest.raises(ValueError, match="one placement and coordinate"):
        _local_shape_and_offsets(
            (8, 8),
            (2, 2),
            (0,),
            (Shard(0), Shard(1)),
        )


@CUDA
def test_two_rank_row_column_and_empty_shard_merge(tmp_path: Path) -> None:
    init_file = tmp_path / "dtensor-lora-init"
    mp.spawn(
        _run_two_rank_merge,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
    )
