"""CPU-only tests for DTensor LoRA shard localization."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import Partial, Replicate, Shard

import torch_offload.dtensor_adapter as dtensor_adapter_module
from torch_offload.dtensor_adapter import DTensorAdapter, _local_shape_and_offsets
from torch_offload.float8_adapter import Float8Adapter


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

        for shard_dim in (0, 1):
            target = distribute_tensor(weight.clone(), mesh, [Shard(shard_dim)])
            DTensorAdapter.merge_lora_(target, b, a, strength)
            torch.testing.assert_close(target.full_tensor(), expected)
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


def test_adapter_owned_merge_receives_contiguous_factor_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inner = Float8Adapter()
    context = dtensor_adapter_module._DTensorMergeContext(
        global_shape=(6, 8),
        local_shape=(6, 4),
        offsets=(0, 4),
        local=torch.empty(6, 4),
        inner=inner,
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
    ) -> None:
        received.append(
            (
                tuple(b.shape),
                tuple(a.shape),
                b.is_contiguous(),
                a.is_contiguous(),
            )
        )

    monkeypatch.setattr(
        Float8Adapter,
        "merge_lora_",
        staticmethod(record_merge),
    )

    DTensorAdapter.merge_lora_(
        torch.empty(0),
        torch.randn(6, 3),
        torch.randn(3, 8),
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


def test_two_rank_row_and_column_shard_merge(tmp_path: Path) -> None:
    init_file = tmp_path / "dtensor-lora-init"
    mp.spawn(
        _run_two_rank_merge,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
    )
