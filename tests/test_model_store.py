"""Tests for ``ModelStore``, ``ModelStoreSpec``, and ``OffloadConfig``."""

from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
import struct
import threading
import tracemalloc
from typing import Any, cast
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from piper_offload import (
    Adapter,
    BlockCompileConfig,
    ModelCache,
    ModelOffloader,
    ModelSpec,
    ModelStore,
    ModelStoreSpec,
    OffloadConfig,
    ResourceCache,
    ResourceStore,
)
from piper_offload._dtensor import DTENSOR_AVAILABLE
from piper_offload._parameter_update import group_adapter_updates
from piper_offload.host_module import HostModuleInstance, ParameterOverride
from tests.conftest import activated_model, offloaded_units, residency_engine, resident_instance

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _TwoStageBlockModel(nn.Module):
    def __init__(self, *, num_blocks: int = 2, width: int = 8, meta: bool = False) -> None:
        super().__init__()
        device = torch.device("meta") if meta else torch.device("cpu")
        self.embed = nn.Linear(width, width, bias=False, device=device)
        self.blocks = nn.ModuleList(
            [nn.Linear(width, width, bias=False, device=device) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(width, width, bias=False, device=device)
        self.register_buffer("scale", torch.empty(width, device=device) if meta else torch.ones(width))
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = torch.relu(block(x))
        return self.head(x) * self.scale


def _snapshot_tensors(module: nn.Module) -> list[tuple[str, nn.Parameter | torch.Tensor, int]]:
    snap: list[tuple[str, nn.Parameter | torch.Tensor, int]] = []
    for name, param in module.named_parameters(remove_duplicate=False):
        snap.append((f"param:{name}", param, param.untyped_storage().data_ptr()))
    for name, buf in module.named_buffers(remove_duplicate=False):
        snap.append((f"buffer:{name}", buf, buf.untyped_storage().data_ptr()))
    return snap


def _assert_unmutated(
    module: nn.Module,
    before: list[tuple[str, nn.Parameter | torch.Tensor, int]],
) -> None:
    after = _snapshot_tensors(module)
    assert len(after) == len(before)
    for (name_a, obj_a, ptr_a), (name_b, obj_b, ptr_b) in zip(before, after, strict=True):
        assert name_a == name_b
        assert obj_a is obj_b
        assert ptr_a == ptr_b


class TestOffloadConfigValidation:
    def test_accepts_list_or_tuple_and_normalizes_to_tuple(self) -> None:
        cfg = OffloadConfig(offload_paths=["embed", "blocks"])
        assert cfg.offload_paths == ("embed", "blocks")

    def test_rejects_non_sequence_and_non_str_paths(self) -> None:
        with pytest.raises(TypeError, match="offload_paths must be a sequence"):
            OffloadConfig(offload_paths=cast(Any, "blocks"))

        with pytest.raises(TypeError, match="offload_paths entries must be str"):
            OffloadConfig(offload_paths=cast(Any, (1,)))

        for bad in ("", " ", "a..b", " a", "a "):
            with pytest.raises(ValueError, match="non-empty dotted attribute paths"):
                OffloadConfig(offload_paths=(bad,))

    def test_rejects_duplicate_paths(self) -> None:
        with pytest.raises(ValueError, match="Duplicate path"):
            OffloadConfig(offload_paths=("blocks", "blocks"))

    def test_rejects_invalid_vram_budget_bytes(self) -> None:
        for bad in (True, "1024", 1.5):
            with pytest.raises(TypeError, match="vram_budget_bytes must be an int or None"):
                OffloadConfig(vram_budget_bytes=cast(Any, bad))

        with pytest.raises(ValueError, match="vram_budget_bytes must be >= 0 or None"):
            OffloadConfig(vram_budget_bytes=-1)

        assert OffloadConfig(vram_budget_bytes=0).vram_budget_bytes == 0
        assert OffloadConfig(vram_budget_bytes=None).vram_budget_bytes is None

    def test_rejects_invalid_block_compile_type(self) -> None:
        with pytest.raises(TypeError, match="block_compile must be a BlockCompileConfig or None"):
            OffloadConfig(block_compile=cast(Any, {"fullgraph": True}))

    def test_entrypoints_reject_non_offload_config_and_resolve_none(self) -> None:
        model = _TwoStageBlockModel()
        store = ModelStore.from_module(model)

        default_direct = ModelOffloader.from_module(_TwoStageBlockModel(), None)
        default_bound = store.bind(_TwoStageBlockModel(), None)
        assert not offloaded_units(default_direct)
        assert not offloaded_units(default_bound)

        with pytest.raises(TypeError, match="OffloadConfig"):
            ModelOffloader.from_module(_TwoStageBlockModel(), config=cast(Any, {"offload_paths": ("blocks",)}))

        with pytest.raises(TypeError, match="OffloadConfig"):
            store.bind(_TwoStageBlockModel(), config=cast(Any, "streaming"))

        with pytest.raises(TypeError, match="OffloadConfig"):
            ModelSpec(
                key="bad",
                factory=_TwoStageBlockModel,
                config=cast(Any, object()),
            )

        with pytest.raises(TypeError, match="OffloadConfig"):
            ModelSpec(
                key="bad-none",
                factory=_TwoStageBlockModel,
                config=cast(Any, None),
            )


class TestModelStoreCaptureAndBind:
    def test_conformance_and_capture_metadata(self) -> None:
        model = _TwoStageBlockModel(num_blocks=2, width=8)
        expected_bytes = sum(t.nbytes for t in model.state_dict().values())

        store = ModelStore.from_module(model)
        assert isinstance(store, ResourceStore)
        assert store.param_names == frozenset(
            {"embed.weight", "blocks.0.weight", "blocks.1.weight", "head.weight"}
        )
        assert store.buffer_names == frozenset({"scale"})
        assert store.cache_bytes == expected_bytes

    def test_from_module_rejects_empty_and_trainable_without_partial_mutation(self) -> None:
        with pytest.raises(ValueError, match="at least one parameter or registered buffer"):
            ModelStore.from_module(nn.Module())

        with pytest.raises(ValueError, match="at least one parameter"):
            ModelOffloader.from_module(nn.Module())

        model = _TwoStageBlockModel(num_blocks=2, width=8)
        model.head.weight.requires_grad_(True)
        before = _snapshot_tensors(model)

        with pytest.raises(ValueError, match="frozen"):
            ModelStore.from_module(model)

        _assert_unmutated(model, before)

    def test_one_store_binds_multiple_shells_with_distinct_configs_on_cpu(self) -> None:
        torch.manual_seed(0)
        prototype = _TwoStageBlockModel(num_blocks=2, width=8)
        reference = copy.deepcopy(prototype)
        x = torch.randn(3, 8)
        with torch.inference_mode():
            expected = reference(x)

        store = ModelStore.from_module(prototype)
        configs = (
            OffloadConfig(),
            OffloadConfig(offload_paths=("blocks",)),
            OffloadConfig(
                offload_paths=("embed", "blocks"),
                vram_budget_bytes=2 * 8 * 8 * 4,
            ),
            OffloadConfig(
                offload_paths=("blocks", "head"),
                vram_budget_bytes=1024 * 1024 * 1024,
            ),
        )
        offloaders = [store.bind(_TwoStageBlockModel(num_blocks=2, width=8), cfg) for cfg in configs]
        try:
            assert resident_instance(offloaders[0]) is not None
            assert not offloaded_units(offloaders[0])
            assert residency_engine(offloaders[1]).unit_paths == ("blocks.0", "blocks.1")
            assert residency_engine(offloaders[2]).unit_paths == ("embed", "blocks.0", "blocks.1")
            assert residency_engine(offloaders[3]).unit_paths == ("blocks.0", "blocks.1", "head")

            proto_ptrs = {name: p.untyped_storage().data_ptr() for name, p in prototype.named_parameters()}
            proto_buf_ptr = prototype.scale.untyped_storage().data_ptr()
            for offloader in offloaders:
                shell = offloader.value
                for name, param in shell.named_parameters():
                    assert param.untyped_storage().data_ptr() == proto_ptrs[name]
                    assert param is not prototype.get_parameter(name)
                assert shell.scale.untyped_storage().data_ptr() == proto_buf_ptr
                with activated_model(offloader, "cpu"), torch.inference_mode():
                    actual = shell(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            for offloader in offloaders:
                offloader.deactivate()

    @CUDA
    def test_one_store_binds_multiple_shells_with_cuda_forward_parity(self) -> None:
        torch.manual_seed(1)
        prototype = _TwoStageBlockModel(num_blocks=2, width=8)
        reference = copy.deepcopy(prototype).cuda()
        x = torch.randn(3, 8, device="cuda")
        with torch.inference_mode():
            expected = reference(x).cpu()

        store = ModelStore.from_module(prototype)
        offloaders = [
            store.bind(_TwoStageBlockModel(num_blocks=2, width=8), OffloadConfig()),
            store.bind(
                _TwoStageBlockModel(num_blocks=2, width=8),
                OffloadConfig(offload_paths=("blocks",)),
            ),
            store.bind(
                _TwoStageBlockModel(num_blocks=2, width=8),
                OffloadConfig(
                    offload_paths=("embed", "blocks", "head"),
                    vram_budget_bytes=2 * 8 * 8 * 4,
                ),
            ),
        ]
        for offloader in offloaders:
            with activated_model(offloader, "cuda"), torch.inference_mode():
                actual = offloader.value(x).cpu()
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_bind_populates_meta_skeleton_and_ignores_extra_trainable_params(self) -> None:
        torch.manual_seed(2)
        prototype = _TwoStageBlockModel(num_blocks=2, width=8)
        store = ModelStore.from_module(prototype)

        class ShellWithExtras(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(8, 8, bias=False, device="meta")
                self.blocks = nn.ModuleList(
                    [nn.Linear(8, 8, bias=False, device="meta") for _ in range(2)]
                )
                for block in self.blocks:
                    block.extra_trainable = nn.Parameter(torch.randn(8))
                self.head = nn.Linear(8, 8, bias=False, device="meta")
                self.register_buffer("scale", torch.empty(8, device="meta"))
                self.outer_trainable = nn.Parameter(torch.randn(8))
                for name, param in self.named_parameters():
                    if "extra_trainable" not in name and name != "outer_trainable":
                        param.requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.embed(x)
                for block in self.blocks:
                    x = torch.relu(block(x)) + block.extra_trainable
                return self.head(x) * self.scale + self.outer_trainable

        shell = ShellWithExtras()
        outer_param = shell.outer_trainable
        outer_ptr = outer_param.untyped_storage().data_ptr()
        block0_extra = shell.blocks[0].extra_trainable
        block0_ptr = block0_extra.untyped_storage().data_ptr()

        offloader = store.bind(shell, OffloadConfig(offload_paths=("blocks",)))
        try:
            assert all(not p.is_meta for p in shell.parameters())
            assert not shell.scale.is_meta
            assert shell.outer_trainable is outer_param
            assert shell.outer_trainable.requires_grad
            assert shell.outer_trainable.untyped_storage().data_ptr() == outer_ptr
            assert shell.blocks[0].extra_trainable is block0_extra
            assert shell.blocks[0].extra_trainable.requires_grad
            assert shell.blocks[0].extra_trainable.untyped_storage().data_ptr() == block0_ptr
            assert "outer_trainable" not in offloader.param_names
            assert "blocks.0.extra_trainable" not in offloader.param_names
            with activated_model(offloader, "cpu"):
                out = shell(torch.randn(2, 8))
            assert out.shape == (2, 8)
        finally:
            offloader.deactivate()

    def test_bind_quantized_store_onto_meta_skeleton(self) -> None:
        convrot_mod = pytest.importorskip("piper_kernels.weights.convrot.int8")
        convrot_cls = convrot_mod.ConvRotInt8Tensor

        proto = nn.Module()
        proto.blocks = nn.ModuleList([nn.Linear(64, 64, bias=False, dtype=torch.bfloat16) for _ in range(2)])
        for block in proto.blocks:
            block.weight = nn.Parameter(
                convrot_cls.from_hp(block.weight.detach(), group_size=64),
                requires_grad=False,
            )

        store = ModelStore.from_module(proto)
        skeleton = nn.Module()
        skeleton.blocks = nn.ModuleList([nn.Linear(64, 64, bias=False, device="meta") for _ in range(2)])
        skeleton.requires_grad_(False)

        offloader = store.bind(skeleton, OffloadConfig(offload_paths=("blocks",)))
        for idx, block in enumerate(skeleton.blocks):
            assert isinstance(block.weight.data, convrot_cls)
            assert block.weight.data.qdata.data_ptr() == proto.blocks[idx].weight.data.qdata.data_ptr()
        if torch.cuda.is_available():
            x = torch.randn(2, 64, dtype=torch.bfloat16, device="cuda")
            with activated_model(offloader, "cuda"), torch.inference_mode():
                out = skeleton.blocks[1](skeleton.blocks[0](x))
            assert out.shape == (2, 64)


class TestAtomicRejectionAndAliasing:
    def test_atomic_rejection_on_bind_errors(self) -> None:
        prototype = _TwoStageBlockModel(num_blocks=2, width=8)
        store = ModelStore.from_module(prototype)

        # Trainable managed param
        bad_trainable = _TwoStageBlockModel(num_blocks=2, width=8)
        bad_trainable.blocks[1].weight.requires_grad_(True)
        snap = _snapshot_tensors(bad_trainable)
        with pytest.raises(ValueError, match="frozen"):
            store.bind(bad_trainable, OffloadConfig(offload_paths=("blocks",)))
        _assert_unmutated(bad_trainable, snap)

        # Layout mismatch
        bad_shape = _TwoStageBlockModel(num_blocks=2, width=8)
        bad_shape.blocks[1] = nn.Linear(8, 16, bias=False).requires_grad_(False)
        snap = _snapshot_tensors(bad_shape)
        with pytest.raises(ValueError, match="layout mismatch"):
            store.bind(bad_shape, OffloadConfig(offload_paths=("blocks",)))
        _assert_unmutated(bad_shape, snap)

        # Unmanaged non-empty block
        bad_unmanaged = _TwoStageBlockModel(num_blocks=3, width=8)
        snap = _snapshot_tensors(bad_unmanaged)
        with pytest.raises(ValueError, match="does not occupy"):
            store.bind(bad_unmanaged, OffloadConfig(offload_paths=("blocks",)))
        _assert_unmutated(bad_unmanaged, snap)

        # Overlapping offload_paths leaves non-contiguous and CUDA weights unmutated
        overlap_cfg = OffloadConfig(offload_paths=("blocks", "blocks.0"))
        target_overlap = _TwoStageBlockModel(num_blocks=2, width=8)
        target_overlap.blocks[0].weight = nn.Parameter(
            torch.randn(8, 8).t(),
            requires_grad=False,
        )
        assert not target_overlap.blocks[0].weight.is_contiguous()
        snap_direct = _snapshot_tensors(target_overlap)
        with pytest.raises(ValueError, match="Overlapping offload_paths"):
            ModelOffloader.from_module(target_overlap, overlap_cfg)
        _assert_unmutated(target_overlap, snap_direct)
        assert not target_overlap.blocks[0].weight.is_contiguous()

        snap_bind = _snapshot_tensors(target_overlap)
        with pytest.raises(ValueError, match="Overlapping offload_paths"):
            store.bind(target_overlap, overlap_cfg)
        _assert_unmutated(target_overlap, snap_bind)

        if torch.cuda.is_available():
            cuda_target = _TwoStageBlockModel(num_blocks=2, width=8).cuda()
            snap_cuda = _snapshot_tensors(cuda_target)
            with pytest.raises(ValueError, match="Overlapping offload_paths"):
                ModelOffloader.from_module(cuda_target, overlap_cfg)
            _assert_unmutated(cuda_target, snap_cuda)
            assert cuda_target.blocks[0].weight.is_cuda

    @pytest.mark.parametrize(
        "tie_kind",
        [
            "resident_block",
            "block_block",
            "block_other_block",
            "standalone_block",
            "standalone_resident",
            "standalone_standalone",
        ],
    )
    def test_cross_group_param_ties_rejected_atomically(self, tie_kind: str) -> None:
        def make_tied() -> nn.Module:
            m = _TwoStageBlockModel(num_blocks=2, width=8)
            if tie_kind == "block_other_block":
                m.other_blocks = nn.ModuleList([nn.Linear(8, 8, bias=False).requires_grad_(False)])
                m.other_blocks[0].weight = m.blocks[0].weight
            elif tie_kind == "resident_block":
                m.head.weight = m.blocks[0].weight
            elif tie_kind == "block_block":
                m.blocks[1].weight = m.blocks[0].weight
            elif tie_kind == "standalone_block":
                m.embed.weight = m.blocks[0].weight
            else:
                m.head.weight = m.embed.weight
            return m

        if tie_kind == "block_other_block":
            cfg = OffloadConfig(offload_paths=("blocks", "other_blocks"))
        elif tie_kind in {"standalone_block", "standalone_resident"}:
            cfg = OffloadConfig(offload_paths=("blocks", "embed"))
        elif tie_kind == "standalone_standalone":
            cfg = OffloadConfig(offload_paths=("blocks", "embed", "head"))
        else:
            cfg = OffloadConfig(offload_paths=("blocks",))

        direct = make_tied()
        snap_direct = _snapshot_tensors(direct)
        with pytest.raises(ValueError, match="Parameter storage shared across offload boundaries"):
            ModelOffloader.from_module(direct, cfg)
        _assert_unmutated(direct, snap_direct)

        proto = make_tied()
        store = ModelStore.from_module(proto)
        target = make_tied()
        snap_target = _snapshot_tensors(target)
        with pytest.raises(ValueError, match="Parameter storage shared across offload boundaries"):
            store.bind(target, cfg)
        _assert_unmutated(target, snap_target)

    def test_cross_group_buffer_only_slot_aliases_rejected_atomically(self) -> None:
        def assert_rejected(
            make_model: Any,
            cfg: OffloadConfig,
            match: str,
        ) -> None:
            direct = make_model()
            snap_direct = _snapshot_tensors(direct)
            with pytest.raises(ValueError, match=match):
                ModelOffloader.from_module(direct, cfg)
            _assert_unmutated(direct, snap_direct)

            store = ModelStore.from_module(make_model())
            target = make_model()
            snap_target = _snapshot_tensors(target)
            with pytest.raises(ValueError, match=match):
                store.bind(target, cfg)
            _assert_unmutated(target, snap_target)

        def make_buf_only_block_model() -> nn.Module:
            m = nn.Module()
            buf_block = nn.Module()
            buf_block.register_buffer("table", torch.randn(4))
            m.blocks = nn.ModuleList([buf_block])
            m.alias_blocks = m.blocks
            return m

        def make_nested_resident_alias_model() -> nn.Module:
            m = nn.Module()
            sub = nn.Module()
            sub.register_buffer("table", torch.randn(4))
            m.outer = nn.Sequential(sub)
            m.other = sub
            return m

        def make_nested_offload_overlap_model() -> nn.Module:
            m = nn.Module()
            sub = nn.Module()
            sub.register_buffer("table", torch.randn(4))
            m.a = nn.Sequential(sub)
            m.b = nn.Sequential(sub)
            return m

        assert_rejected(
            make_buf_only_block_model,
            OffloadConfig(offload_paths=("blocks", "alias_blocks")),
            "Overlapping offload_paths",
        )
        assert_rejected(
            make_nested_resident_alias_model,
            OffloadConfig(offload_paths=("outer",)),
            "Buffer module slot shared across offload boundaries",
        )
        assert_rejected(
            make_nested_offload_overlap_model,
            OffloadConfig(offload_paths=("a", "b")),
            "Overlapping offload_paths",
        )

    def test_rejects_distinct_store_entries_on_aliased_target_slot_without_mutation(self) -> None:
        # Pair of distinct weights vs shared param module slot
        proto = nn.Module()
        proto.left = nn.Linear(4, 4, bias=False)
        proto.right = nn.Linear(4, 4, bias=False)
        proto.register_buffer("buf_a", torch.randn(4))
        proto.register_buffer("buf_b", torch.randn(4))
        proto.requires_grad_(False)
        store = ModelStore.from_module(proto)

        shared = nn.Linear(4, 4, bias=False).requires_grad_(False)
        target_param_alias = nn.Module()
        target_param_alias.left = shared
        target_param_alias.right = shared
        target_param_alias.register_buffer("buf_a", torch.zeros(4))
        target_param_alias.register_buffer("buf_b", torch.zeros(4))
        snap = _snapshot_tensors(target_param_alias)
        with pytest.raises(ValueError, match="distinct store params"):
            store.bind(target_param_alias)
        _assert_unmutated(target_param_alias, snap)

        # Pair of distinct blocks bound onto ModuleList([shared, shared])
        block_proto = nn.Module()
        block_proto.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]).requires_grad_(False)
        block_store = ModelStore.from_module(block_proto)
        shared_block = nn.Linear(4, 4, bias=False).requires_grad_(False)
        target_blocks = nn.Module()
        target_blocks.blocks = nn.ModuleList([shared_block, shared_block])
        snap_blocks = _snapshot_tensors(target_blocks)
        with pytest.raises(ValueError, match="distinct store params"):
            block_store.bind(target_blocks, OffloadConfig(offload_paths=("blocks",)))
        _assert_unmutated(target_blocks, snap_blocks)

        # Buffer slot conflict
        buf_proto = nn.Module()
        buf_proto.left = nn.Module()
        buf_proto.right = nn.Module()
        buf_proto.left.register_buffer("table", torch.ones(4))
        buf_proto.right.register_buffer("table", torch.zeros(4))
        buf_store = ModelStore.from_module(buf_proto)

        shared_sub = nn.Module()
        shared_sub.register_buffer("table", torch.zeros(4))
        target_buf_alias = nn.Module()
        target_buf_alias.left = shared_sub
        target_buf_alias.right = shared_sub
        snap_buf = _snapshot_tensors(target_buf_alias)
        with pytest.raises(ValueError, match="distinct store buffers"):
            buf_store.bind(target_buf_alias)
        _assert_unmutated(target_buf_alias, snap_buf)

    def test_allows_aliased_block_modules_root_block_alias_and_shared_buffers(self) -> None:
        torch.manual_seed(3)

        class AliasedBlockModel(nn.Module):
            def __init__(self, shared_buf: torch.Tensor | None = None) -> None:
                super().__init__()
                shared_block = nn.Linear(8, 8, bias=False)
                other_block = nn.Linear(8, 8, bias=False)
                table = torch.randn(8) if shared_buf is None else shared_buf
                shared_block.register_buffer("table", table)
                other_block.register_buffer("table", table)
                self.blocks = nn.ModuleList([shared_block, shared_block, other_block])
                self.first_block = shared_block
                self.requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = block(x) * block.table
                return self.first_block(x)

        ref = AliasedBlockModel()
        x = torch.randn(2, 8)
        with torch.inference_mode():
            expected = ref(x)

        direct = copy.deepcopy(ref)
        cfg = OffloadConfig(offload_paths=("blocks",))
        direct_offloader = ModelOffloader.from_module(direct, cfg)
        assert direct.blocks[0] is direct.blocks[1] is direct.first_block
        assert resident_instance(direct_offloader) is None
        with activated_model(direct_offloader, "cpu"), torch.inference_mode():
            torch.testing.assert_close(direct(x), expected)

        store = ModelStore.from_module(copy.deepcopy(ref))
        shell = AliasedBlockModel()
        bound_offloader = store.bind(shell, cfg)
        assert shell.blocks[0] is shell.blocks[1] is shell.first_block
        assert resident_instance(bound_offloader) is None
        with activated_model(bound_offloader, "cpu"), torch.inference_mode():
            torch.testing.assert_close(shell(x), expected)
        if torch.cuda.is_available():
            with activated_model(bound_offloader, "cuda"), torch.inference_mode():
                actual = shell(x.cuda()).cpu()
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


class TestModelStoreSpec:
    def test_cache_accounting_eviction_and_shared_leasing(self) -> None:
        builds = {"a": 0, "b": 0}

        def build_a() -> nn.Module:
            builds["a"] += 1
            return _TwoStageBlockModel(num_blocks=2, width=8)

        def build_b() -> nn.Module:
            builds["b"] += 1
            return _TwoStageBlockModel(num_blocks=2, width=8)

        sample = _TwoStageBlockModel(num_blocks=2, width=8)
        one_bytes = sum(t.nbytes for t in sample.state_dict().values())

        cache = ResourceCache(max_cache_bytes=one_bytes)
        spec_a = ModelStoreSpec(key="store:a", factory=build_a, estimated_cache_bytes=one_bytes)
        spec_b = ModelStoreSpec(key="store:b", factory=build_b, estimated_cache_bytes=one_bytes)

        with cache.lease(spec_a) as store_a_first:
            assert isinstance(store_a_first, ModelStore)
            assert store_a_first.cache_bytes == one_bytes
            with cache.lease(spec_a) as store_a_second:
                assert store_a_second is store_a_first
                assert cache.info("store:a").lease_count == 2
                first = store_a_first.bind(_TwoStageBlockModel(num_blocks=2, width=8), OffloadConfig())
                second = store_a_second.bind(
                    _TwoStageBlockModel(num_blocks=2, width=8),
                    OffloadConfig(offload_paths=("blocks",)),
                )
                try:
                    assert first.value.embed.weight.untyped_storage().data_ptr() == (
                        second.value.embed.weight.untyped_storage().data_ptr()
                    )
                finally:
                    first.deactivate()
                    second.deactivate()

        assert builds["a"] == 1
        assert cache.used_cache_bytes == one_bytes

        # Exceeding max_cache_bytes evicts cold store:a when admitting store:b.
        with cache.lease(spec_b) as store_b:
            assert store_b.cache_bytes == one_bytes
            assert not cache.info("store:a").cached
            assert cache.info("store:b").cached

        assert builds["b"] == 1

    def test_model_cache_resize_eviction_and_binding(self) -> None:
        sample = _TwoStageBlockModel(num_blocks=2, width=8)
        one_bytes = sum(t.nbytes for t in sample.state_dict().values())

        cache = ModelCache()
        spec_a = ModelStoreSpec(
            key="model-store-a",
            factory=lambda: _TwoStageBlockModel(num_blocks=2, width=8),
            estimated_cache_bytes=one_bytes,
        )
        spec_b = ModelStoreSpec(
            key="model-store-b",
            factory=lambda: _TwoStageBlockModel(num_blocks=2, width=8),
            estimated_cache_bytes=one_bytes,
        )
        with cache.lease(spec_a) as store:
            assert isinstance(store, ModelStore)
            offloader = store.bind(
                _TwoStageBlockModel(num_blocks=2, width=8),
                OffloadConfig(offload_paths=("blocks",)),
            )
            with activated_model(offloader, "cpu"), torch.inference_mode():
                out = offloader.value(torch.randn(2, 8))
            assert out.shape == (2, 8)

        cache.resize(one_bytes)
        with cache.lease(spec_b) as store_b:
            assert store_b.cache_bytes == one_bytes
            assert not cache.info("model-store-a").cached
            assert cache.info("model-store-b").cached


def _plan_lora_merge(
    offloader: ModelOffloader,
    adapter: Adapter,
) -> dict[str, ParameterOverride]:
    updates = group_adapter_updates([(adapter, 1.0)], offloader.param_names)
    return offloader._build_parameter_overrides(torch.device("cuda"), updates)


class _UnsupportedTensor(torch.Tensor):
    pass


class _EmptyShardDTensorModel(nn.Module):
    def __init__(self, mesh: Any) -> None:
        super().__init__()
        from torch.distributed.tensor import Shard, distribute_tensor

        def make_block() -> nn.Module:
            block = nn.Module()
            block.weight = nn.Parameter(
                distribute_tensor(torch.ones(1, 4, dtype=torch.float32), mesh, [Shard(0)]),
                requires_grad=False,
            )
            block.register_buffer(
                "table",
                distribute_tensor(torch.ones(1, 4, dtype=torch.float32), mesh, [Shard(0)]),
            )
            return block

        self.blocks = nn.ModuleList([make_block(), make_block()])
        self.head = make_block()
        self.requires_grad_(False)


def _dtensor_empty_shard_worker(rank: int, init_file: str) -> None:
    from torch.distributed.tensor import init_device_mesh

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        cfg = OffloadConfig(offload_paths=("blocks",))

        direct = _EmptyShardDTensorModel(mesh)
        if rank == 1:
            assert direct.blocks[0].weight.to_local().numel() == 0
            assert direct.blocks[1].weight.to_local().numel() == 0
            assert direct.blocks[0].table.to_local().numel() == 0
            assert direct.blocks[1].table.to_local().numel() == 0
        offloader = ModelOffloader.from_module(direct, cfg)
        assert len(offloader.param_names) == 3
        offloader.deactivate()
        del offloader

        proto = _EmptyShardDTensorModel(mesh)
        target = _EmptyShardDTensorModel(mesh)
        store = ModelStore.from_module(proto)
        bound = store.bind(target, cfg)
        assert len(bound.param_names) == 3
        bound.deactivate()
        del bound

        # A genuine cross-group alias of the same empty-shard Parameter still fails on both ranks.
        tied_direct = _EmptyShardDTensorModel(mesh)
        tied_direct.head.weight = tied_direct.blocks[0].weight
        with pytest.raises(ValueError, match="Parameter storage shared across offload boundaries"):
            ModelOffloader.from_module(tied_direct, cfg)

        tied_proto = _EmptyShardDTensorModel(mesh)
        tied_proto.head.weight = tied_proto.blocks[0].weight
        tied_store = ModelStore.from_module(tied_proto)
        tied_target = _EmptyShardDTensorModel(mesh)
        with pytest.raises(ValueError, match="Parameter storage shared across offload boundaries"):
            tied_store.bind(tied_target, cfg)

        # Distinct wrappers over the same DTensor preserve aliasing on BOTH ranks (even when local shard is empty).
        from torch.distributed.tensor import Shard, distribute_tensor

        shared_dt = distribute_tensor(torch.ones(1, 4, dtype=torch.float32), mesh, [Shard(0)])
        flat_shared = nn.Module()
        flat_shared.a = nn.Parameter(shared_dt, requires_grad=False)
        flat_shared.b = nn.Parameter(shared_dt, requires_grad=False)
        flat_shared.register_buffer("ba", shared_dt.detach())
        flat_shared.register_buffer("bb", shared_dt.detach())
        shared_store = ModelStore.from_module(flat_shared)
        assert shared_store._module_store.params["a"] is shared_store._module_store.params["b"]
        assert shared_store._module_store.buffers["ba"] is shared_store._module_store.buffers["bb"]

        distinct_wrapper_blocks = _EmptyShardDTensorModel(mesh)
        distinct_wrapper_blocks.blocks[1].weight = nn.Parameter(
            distinct_wrapper_blocks.blocks[0].weight.data,
            requires_grad=False,
        )
        with pytest.raises(ValueError, match="Parameter storage shared across offload boundaries"):
            ModelOffloader.from_module(distinct_wrapper_blocks, cfg)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestReviewRegressions:
    def test_incremental_buffer_capture_releases_previous_buffers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from piper_offload import host_module

        real_capture = host_module.HostBuffer.capture
        live_counts: list[int] = []
        watched_refs: list[weakref.ref[torch.Tensor]] = []

        def wrapped_capture(buf: torch.Tensor) -> Any:
            live_counts.append(sum(ref() is not None for ref in watched_refs))
            return real_capture(buf)

        monkeypatch.setattr(host_module.HostBuffer, "capture", wrapped_capture)

        def make_buffer_model() -> tuple[nn.Module, list[weakref.ref[torch.Tensor]]]:
            model = nn.Module()
            blocks = [nn.Module() for _ in range(6)]
            refs: list[weakref.ref[torch.Tensor]] = []
            for idx, block in enumerate(blocks):
                non_contig = (
                    torch.arange(16, dtype=torch.float32) + float(idx)
                ).reshape(4, 4).T
                assert not non_contig.is_contiguous()
                block.register_buffer("table", non_contig)
                refs.append(weakref.ref(non_contig))
            model.blocks = nn.ModuleList(blocks)
            return model, refs

        direct, direct_refs = make_buffer_model()
        watched_refs = direct_refs
        offloader = ModelOffloader.from_module(direct, OffloadConfig(offload_paths=("blocks",)))
        assert live_counts == [6, 5, 4, 3, 2, 1]
        offloader.deactivate()

        live_counts.clear()
        store_model, store_refs = make_buffer_model()
        watched_refs = store_refs
        store = ModelStore.from_module(store_model)
        assert live_counts == [6, 5, 4, 3, 2, 1]
        assert store.cache_bytes > 0

    def test_preflight_inspects_param_representation_before_capture(self) -> None:
        cfg = OffloadConfig(
            offload_paths=("blocks",),
            block_compile=BlockCompileConfig(fullgraph=True),
        )

        # Unregistered tensor subclass wrapped inside nn.Parameter.
        unreg = nn.Module()
        b0 = nn.Module()
        b0.weight = nn.Parameter(torch.randn(4, 4).T, requires_grad=False)
        assert not b0.weight.is_contiguous()
        b1 = nn.Module()
        raw_sub = cast(
            _UnsupportedTensor,
            torch.Tensor._make_subclass(
                _UnsupportedTensor,
                torch.randn(4, 4),
                False,
            ),
        )
        b1.weight = nn.Parameter(raw_sub, requires_grad=False)
        unreg.blocks = nn.ModuleList([b0, b1])
        snap_unreg = _snapshot_tensors(unreg)
        with pytest.raises(NotImplementedError, match="register_adapter"):
            ModelOffloader.from_module(unreg, cfg)
        _assert_unmutated(unreg, snap_unreg)
        assert not unreg.blocks[0].weight.is_contiguous()

    @pytest.mark.skipif(
        not (DTENSOR_AVAILABLE and dist.is_gloo_available()),
        reason="PyTorch DTensor and Gloo backend required",
    )
    def test_empty_local_shard_dtensor_two_rank_capture_and_bind(self, tmp_path: Path) -> None:
        init_file = tmp_path / "dtensor-empty-shard-init"
        mp.spawn(
            _dtensor_empty_shard_worker,
            args=(str(init_file),),
            nprocs=2,
            join=True,
        )

    def test_conj_and_neg_buffers_preserve_distinct_values(self) -> None:
        class _ViewBufferBlock(nn.Module):
            def __init__(self, c: torch.Tensor, r: torch.Tensor) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
                self.register_buffer("c", c)
                self.register_buffer("r", r)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return x.to(self.c.dtype) * self.c, x * self.r

        def make_view_buffer_model() -> nn.Module:
            model = nn.Module()
            base_complex = torch.tensor([1j])
            base_real = torch.tensor([3.0])
            b0 = _ViewBufferBlock(base_complex, base_real)
            b1 = _ViewBufferBlock(base_complex.conj(), torch._neg_view(base_real))
            assert b1.c.is_conj()
            assert b1.r.is_neg()
            model.blocks = nn.ModuleList([b0, b1])
            return model

        cfg = OffloadConfig(offload_paths=("blocks",))
        direct = make_view_buffer_model()
        offloader = ModelOffloader.from_module(direct, cfg)
        try:
            assert torch.equal(direct.blocks[0].c, torch.tensor([1j]))
            assert torch.equal(direct.blocks[1].c, torch.tensor([-1j]))
            assert torch.equal(direct.blocks[0].r, torch.tensor([3.0]))
            assert torch.equal(direct.blocks[1].r, torch.tensor([-3.0]))
        finally:
            offloader.deactivate()

        proto = make_view_buffer_model()
        store = ModelStore.from_module(proto)
        assert store._module_store.buffers["blocks.0.c"] is not store._module_store.buffers["blocks.1.c"]
        assert store._module_store.buffers["blocks.0.r"] is not store._module_store.buffers["blocks.1.r"]
        shell = nn.Module()
        shell.blocks = nn.ModuleList(
            [
                _ViewBufferBlock(
                    torch.zeros(1, dtype=torch.complex64),
                    torch.zeros(1, dtype=torch.float32),
                )
                for _ in range(2)
            ]
        )
        shell.requires_grad_(False)

        bound = store.bind(shell, cfg)
        try:
            assert torch.equal(shell.blocks[0].c, torch.tensor([1j]))
            assert torch.equal(shell.blocks[1].c, torch.tensor([-1j]))
            assert torch.equal(shell.blocks[0].r, torch.tensor([3.0]))
            assert torch.equal(shell.blocks[1].r, torch.tensor([-3.0]))
            if torch.cuda.is_available():
                with activated_model(bound, "cuda"), torch.inference_mode():
                    x_cuda = torch.ones(1, device="cuda")
                    out0_c, out0_r = shell.blocks[0](x_cuda)
                    out1_c, out1_r = shell.blocks[1](x_cuda)
                assert torch.equal(out0_c.cpu(), torch.tensor([1j]))
                assert torch.equal(out1_c.cpu(), torch.tensor([-1j]))
                assert torch.equal(out0_r.cpu(), torch.tensor([3.0]))
                assert torch.equal(out1_r.cpu(), torch.tensor([-3.0]))
        finally:
            bound.deactivate()

    def test_root_block_alias_targets_merge_and_conflicts(self) -> None:
        torch.manual_seed(4)

        class AliasedRootModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                shared = nn.Linear(8, 8, bias=False)
                other = nn.Linear(8, 8, bias=False)
                self.blocks = nn.ModuleList([shared, other])
                self.first_block = shared
                self.requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = torch.relu(block(x))
                return self.first_block(x)

        ref = AliasedRootModel()
        lora_a = torch.randn(2, 8)
        lora_b = torch.randn(8, 2)
        adapter = Adapter.from_state_dict(
            {
                "first_block.lora_A.weight": lora_a,
                "first_block.lora_B.weight": lora_b,
            },
            allow_partial_targets=False,
        )
        other_alias_adapter = Adapter.from_state_dict(
            {
                "blocks.0.lora_A.weight": lora_a * 2.0,
                "blocks.0.lora_B.weight": lora_b,
            },
            allow_partial_targets=False,
        )
        conflict_adapter = Adapter.from_state_dict(
            {
                "first_block.lora_A.weight": lora_a,
                "first_block.lora_B.weight": lora_b,
                "blocks.0.lora_A.weight": lora_a * 2.0,
                "blocks.0.lora_B.weight": lora_b,
            },
            allow_partial_targets=False,
        )

        x = torch.randn(3, 8)
        merged_ref = copy.deepcopy(ref)
        with torch.no_grad():
            cast(nn.Linear, merged_ref.blocks[0]).weight.add_(lora_b @ lora_a)
        with torch.inference_mode():
            expected = merged_ref(x)

        cfg = OffloadConfig(offload_paths=("blocks",))
        store = ModelStore.from_module(copy.deepcopy(ref))
        candidates = [
            ModelOffloader.from_module(copy.deepcopy(ref), cfg),
            store.bind(copy.deepcopy(ref), cfg),
        ]
        try:
            for offloader in candidates:
                assert "first_block.weight" in offloader.param_names
                assert "blocks.0.weight" in offloader.param_names

                if torch.cuda.is_available():
                    with (
                        activated_model(
                            offloader,
                            "cuda",
                            adapters=[adapter],
                        ),
                        torch.inference_mode(),
                    ):
                        actual_merged = offloader.value(x.cuda()).cpu()
                    torch.testing.assert_close(actual_merged, expected, atol=1e-5, rtol=1e-5)

                overrides = _plan_lora_merge(offloader, adapter)
                _res, unit_ovrs = offloader._partition.route_overrides(overrides)
                plans = [
                    u.resolve_load_plan(ovr)
                    for u, ovr in zip(offloaded_units(offloader), unit_ovrs, strict=True)
                ]
                assert plans[0].loads["weight"].update is not None
                assert plans[1].loads["weight"].update is None

                with pytest.raises(ValueError, match="Tied parameter aliases"):
                    _plan_lora_merge(offloader, conflict_adapter)

                other_overrides = _plan_lora_merge(offloader, other_alias_adapter)
                with pytest.raises(ValueError, match="Tied parameter aliases"):
                    offloader._partition.route_overrides({**overrides, **other_overrides})

                if torch.cuda.is_available():
                    with (
                        activated_model(offloader, "cuda", adapters=[adapter]),
                        torch.inference_mode(),
                    ):
                        actual_cuda = offloader.value(x.cuda()).cpu()
                    torch.testing.assert_close(actual_cuda, expected, atol=1e-5, rtol=1e-5)
        finally:
            for offloader in candidates:
                offloader.deactivate()

    @pytest.mark.parametrize("vram_budget_bytes", [None, 1024 * 1024])
    @pytest.mark.parametrize("target_prefix", ["blocks.0", "blocks.1"])
    def test_aliased_block_module_propagates_merge_and_rejects_conflicts(
        self,
        vram_budget_bytes: int | None,
        target_prefix: str,
    ) -> None:
        torch.manual_seed(5)

        class AliasedBlockListModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                shared = nn.Linear(8, 8, bias=False)
                other = nn.Linear(8, 8, bias=False)
                self.blocks = nn.ModuleList([shared, shared, other])
                self.requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = torch.relu(block(x))
                return x

        ref = AliasedBlockListModel()
        lora_a = torch.randn(2, 8)
        lora_b = torch.randn(8, 2)
        adapter = Adapter.from_state_dict(
            {
                f"{target_prefix}.lora_A.weight": lora_a,
                f"{target_prefix}.lora_B.weight": lora_b,
            },
            allow_partial_targets=False,
        )
        peer_prefix = "blocks.1" if target_prefix == "blocks.0" else "blocks.0"
        peer_adapter = Adapter.from_state_dict(
            {
                f"{peer_prefix}.lora_A.weight": lora_a * 2.0,
                f"{peer_prefix}.lora_B.weight": lora_b,
            },
            allow_partial_targets=False,
        )
        conflict_adapter = Adapter.from_state_dict(
            {
                "blocks.0.lora_A.weight": lora_a,
                "blocks.0.lora_B.weight": lora_b,
                "blocks.1.lora_A.weight": lora_a * 2.0,
                "blocks.1.lora_B.weight": lora_b,
            },
            allow_partial_targets=False,
        )

        x = torch.randn(3, 8)
        merged_ref = copy.deepcopy(ref)
        with torch.no_grad():
            cast(nn.Linear, merged_ref.blocks[0]).weight.add_(lora_b @ lora_a)
        with torch.inference_mode():
            expected = merged_ref(x)

        cfg = OffloadConfig(offload_paths=("blocks",), vram_budget_bytes=vram_budget_bytes)
        store = ModelStore.from_module(copy.deepcopy(ref))
        candidates = [
            ModelOffloader.from_module(copy.deepcopy(ref), cfg),
            store.bind(copy.deepcopy(ref), cfg),
        ]
        try:
            for offloader in candidates:
                overrides = _plan_lora_merge(offloader, adapter)
                _res, unit_ovrs = offloader._partition.route_overrides(overrides)
                plans = [
                    u.resolve_load_plan(ovr)
                    for u, ovr in zip(offloaded_units(offloader), unit_ovrs, strict=True)
                ]
                assert len(plans) == 2
                assert plans[0].loads["weight"].update is not None
                assert plans[1].loads["weight"].update is None

                with pytest.raises(ValueError, match="Tied parameter aliases"):
                    _plan_lora_merge(offloader, conflict_adapter)

                peer_overrides = _plan_lora_merge(offloader, peer_adapter)
                with pytest.raises(ValueError, match="Tied parameter aliases"):
                    offloader._partition.route_overrides({**overrides, **peer_overrides})

                if torch.cuda.is_available():
                    with (
                        activated_model(offloader, "cuda", adapters=[adapter]),
                        torch.inference_mode(),
                    ):
                        actual_cuda = offloader.value(x.cuda()).cpu()
                    torch.testing.assert_close(actual_cuda, expected, atol=1e-5, rtol=1e-5)
                    with pytest.raises(ValueError, match="Tied parameter aliases"):
                        offloader.activate("cuda", adapters=[conflict_adapter])
        finally:
            for offloader in candidates:
                offloader.deactivate()

    def test_swapped_per_block_parameter_order_parity_between_direct_and_bind(self) -> None:
        def make_swapped_order_model() -> nn.Module:
            model = nn.Module()
            b0 = nn.Module()
            b0.a = nn.Parameter(torch.randn(4, 4), requires_grad=False)
            b0.b = nn.Parameter(torch.randn(4, 8), requires_grad=False)
            b1 = nn.Module()
            b1.b = nn.Parameter(torch.randn(4, 8), requires_grad=False)
            b1.a = nn.Parameter(torch.randn(4, 4), requires_grad=False)
            model.blocks = nn.ModuleList([b0, b1])
            return model

        cfg = OffloadConfig(
            offload_paths=("blocks",),
            block_compile=BlockCompileConfig(fullgraph=True),
        )
        store = ModelStore.from_module(make_swapped_order_model())
        direct = ModelOffloader.from_module(make_swapped_order_model(), cfg)
        bound = store.bind(make_swapped_order_model(), cfg)
        try:
            assert [set(u.params) for u in offloaded_units(direct)] == [
                {"a", "b"},
                {"a", "b"},
            ]
            assert [set(u.params) for u in offloaded_units(bound)] == [
                {"a", "b"},
                {"a", "b"},
            ]
        finally:
            direct.deactivate()
            bound.deactivate()

    def test_property_aliased_block_paths_whole_and_slice(self) -> None:
        torch.manual_seed(6)

        class PropertyWholeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(2)])
                self.requires_grad_(False)

            @property
            def blocks(self) -> nn.ModuleList:
                return self.layers

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = torch.relu(block(x))
                return x

        class PropertySlicedModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(3)])
                self.requires_grad_(False)

            @property
            def blocks(self) -> nn.ModuleList:
                return self.layers[1:]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.relu(self.layers[0](x))
                for block in self.blocks:
                    x = torch.relu(block(x))
                return x

        x = torch.randn(3, 8)
        cfg = OffloadConfig(offload_paths=("blocks",))

        # Full alias: @property blocks -> self.layers
        whole_ref = PropertyWholeModel()
        with torch.inference_mode():
            whole_expected = whole_ref(x)
        whole_store = ModelStore.from_module(copy.deepcopy(whole_ref))
        whole_offloaders = [
            ModelOffloader.from_module(copy.deepcopy(whole_ref), cfg),
            whole_store.bind(copy.deepcopy(whole_ref), cfg),
        ]
        try:
            for offloader in whole_offloaders:
                assert resident_instance(offloader) is None
                assert offloader.param_names == frozenset({"layers.0.weight", "layers.1.weight"})
                with activated_model(offloader, "cpu"), torch.inference_mode():
                    torch.testing.assert_close(offloader.value(x), whole_expected, atol=0, rtol=0)
        finally:
            for offloader in whole_offloaders:
                offloader.deactivate()

        # Sliced property alias: @property blocks -> self.layers[1:]
        slice_ref = PropertySlicedModel()
        lora_a = torch.randn(2, 8)
        lora_b = torch.randn(8, 2)
        slice_adapter = Adapter.from_state_dict(
            {
                "layers.1.lora_A.weight": lora_a,
                "layers.1.lora_B.weight": lora_b,
            },
            allow_partial_targets=False,
        )
        merged_slice_ref = copy.deepcopy(slice_ref)
        with torch.no_grad():
            cast(nn.Linear, merged_slice_ref.layers[1]).weight.add_(lora_b @ lora_a)
        with torch.inference_mode():
            slice_expected = slice_ref(x)
            slice_lora_expected = merged_slice_ref(x)

        slice_store = ModelStore.from_module(copy.deepcopy(slice_ref))
        slice_offloaders = [
            ModelOffloader.from_module(copy.deepcopy(slice_ref), cfg),
            slice_store.bind(copy.deepcopy(slice_ref), cfg),
        ]
        try:
            for offloader in slice_offloaders:
                res = resident_instance(offloader)
                assert res is not None
                assert set(res.params) == {"layers.0.weight"}
                assert "layers.1.weight" in offloader.param_names
                with activated_model(offloader, "cpu"), torch.inference_mode():
                    torch.testing.assert_close(offloader.value(x), slice_expected, atol=0, rtol=0)
                if torch.cuda.is_available():
                    with (
                        activated_model(
                            offloader,
                            "cuda",
                            adapters=[slice_adapter],
                        ),
                        torch.inference_mode(),
                    ):
                        torch.testing.assert_close(
                            offloader.value(x.cuda()).cpu(),
                            slice_lora_expected,
                            atol=1e-5,
                            rtol=1e-5,
                        )
        finally:
            for offloader in slice_offloaders:
                offloader.deactivate()

    def test_bind_noncontiguous_buffer_onto_fresh_cpu_and_meta_shells(self) -> None:
        torch.manual_seed(7)

        class NonContiguousBufferModel(nn.Module):
            def __init__(self, *, meta: bool = False) -> None:
                super().__init__()
                device = torch.device("meta") if meta else torch.device("cpu")
                self.proj = nn.Linear(3, 4, bias=False, device=device)
                buf = (
                    torch.empty(3, 4, device=device).T
                    if meta
                    else torch.arange(12, dtype=torch.float32).reshape(3, 4).T
                )
                assert not buf.is_contiguous()
                self.register_buffer("table", buf)
                self.requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.proj(x) * self.table.reshape(-1)[:4]

        ref = NonContiguousBufferModel()
        x = torch.randn(2, 3)
        with torch.inference_mode():
            expected = ref(x)

        store = ModelStore.from_module(copy.deepcopy(ref))
        for fresh in (
            NonContiguousBufferModel(meta=False),
            NonContiguousBufferModel(meta=True),
        ):
            offloader = store.bind(fresh)
            try:
                assert not fresh.table.is_meta
                assert fresh.table.is_contiguous()
                assert torch.equal(fresh.table, ref.table)
                with activated_model(offloader, "cpu"), torch.inference_mode():
                    torch.testing.assert_close(fresh(x), expected, atol=0, rtol=0)
            finally:
                offloader.deactivate()

    def test_exclusive_live_shell_ownership(self) -> None:
        model = _TwoStageBlockModel(num_blocks=2, width=8)
        cfg = OffloadConfig(offload_paths=("blocks",))
        store = ModelStore.from_module(_TwoStageBlockModel(num_blocks=2, width=8))
        offloader: ModelOffloader | None = ModelOffloader.from_module(model, cfg)
        assert offloader is not None

        snap = _snapshot_tensors(model)
        with pytest.raises(ValueError, match="already bound to a live ModelOffloader"):
            ModelOffloader.from_module(model, cfg)
        _assert_unmutated(model, snap)

        with pytest.raises(ValueError, match="already bound to a live ModelOffloader"):
            ModelStore.from_module(model)
        _assert_unmutated(model, snap)

        with pytest.raises(ValueError, match="already bound to a live ModelOffloader"):
            store.bind(model, cfg)
        _assert_unmutated(model, snap)

        with activated_model(offloader, "cpu"):
            with pytest.raises(ValueError, match="already bound to a live ModelOffloader"):
                store.bind(model, cfg)
            with pytest.raises(ValueError, match="already bound to a live ModelOffloader"):
                ModelStore.from_module(model)

        sub_snap = _snapshot_tensors(model.embed)
        with pytest.raises(ValueError, match="already bound to a live ModelOffloader"):
            ModelOffloader.from_module(model.embed)
        with pytest.raises(ValueError, match="already bound to a live ModelOffloader"):
            ModelStore.from_module(model.embed)
        _assert_unmutated(model.embed, sub_snap)

        sub_store = ModelStore.from_module(nn.Linear(8, 8, bias=False).requires_grad_(False))
        with pytest.raises(ValueError, match="already bound to a live ModelOffloader"):
            sub_store.bind(model.blocks[0])

        wrapper = nn.Module()
        wrapper.inner = model
        with pytest.raises(ValueError, match="already bound to a live ModelOffloader"):
            ModelOffloader.from_module(wrapper)

        # ModelStore.from_module releases its temporary claim on return so bind(same_model) succeeds once.
        reusable = _TwoStageBlockModel(num_blocks=2, width=8)
        reusable_store = ModelStore.from_module(reusable)
        reusable_offloader = reusable_store.bind(reusable, cfg)
        with pytest.raises(ValueError, match="already bound to a live ModelOffloader"):
            reusable_store.bind(reusable, cfg)
        reusable_offloader.deactivate()

        # Dropping the live offloader releases exclusive ownership.
        offloader.deactivate()
        offloader = None
        gc.collect()
        ModelStore.from_module(model)
        rebound = store.bind(model, cfg)
        rebound.deactivate()

    @staticmethod
    def _pause_first_install(
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[threading.Event, threading.Event]:
        real_install = HostModuleInstance.install_host
        entered = threading.Event()
        proceed = threading.Event()
        paused = False
        pause_lock = threading.Lock()

        def paused_install(instance: HostModuleInstance) -> None:
            nonlocal paused
            should_pause = False
            with pause_lock:
                if not paused:
                    paused = True
                    should_pause = True
            if should_pause:
                entered.set()
                assert proceed.wait(timeout=5)
            real_install(instance)

        monkeypatch.setattr(HostModuleInstance, "install_host", paused_install)
        return entered, proceed

    def test_concurrent_bind_race_rejects_loser_before_mutating_winner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def filled_store(value: float) -> ModelStore:
            proto = nn.Linear(4, 4, bias=False).requires_grad_(False)
            with torch.no_grad():
                proto.weight.fill_(value)
            return ModelStore.from_module(proto)

        store_one, store_two = filled_store(1.0), filled_store(2.0)
        shell = nn.Linear(4, 4, bias=False, device="meta").requires_grad_(False)
        winner_entered, winner_proceed = self._pause_first_install(monkeypatch)
        winner_result: list[ModelOffloader] = []
        winner_error: list[BaseException] = []

        def run_winner() -> None:
            try:
                winner_result.append(store_one.bind(shell))
            except BaseException as exc:
                winner_error.append(exc)

        t_winner = threading.Thread(target=run_winner)
        t_winner.start()
        assert winner_entered.wait(timeout=5)
        try:
            with pytest.raises(ValueError, match="Deactivate and drop"):
                store_two.bind(shell)
        finally:
            winner_proceed.set()
            t_winner.join(timeout=5)

        assert not winner_error
        (offloader,) = winner_result
        try:
            with activated_model(offloader, "cpu"), torch.inference_mode():
                out = shell(torch.ones(1, 4))
            torch.testing.assert_close(out, torch.full((1, 4), 4.0), atol=0, rtol=0)
        finally:
            offloader.deactivate()

    def test_custom_eq_and_hash_sibling_modules_do_not_bypass_ownership(self) -> None:
        class EqualContainer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(4, 4, bias=False)

            def __eq__(self, other: object) -> bool:
                return isinstance(other, EqualContainer)

            def __hash__(self) -> int:
                return 42

        root = nn.Module()
        root.a = EqualContainer()
        root.b = EqualContainer()
        root.a.requires_grad_(False)
        root.b.requires_grad_(False)
        offloader = ModelOffloader.from_module(root)
        try:
            assert offloader.param_names == frozenset({"a.lin.weight", "b.lin.weight"})
            child_store = ModelStore.from_module(EqualContainer().requires_grad_(False))
            snap_b = _snapshot_tensors(root.b)
            with pytest.raises(ValueError, match="already bound"):
                child_store.bind(root.b)
            _assert_unmutated(root.b, snap_b)

            # Equal but disjoint shells remain independently bindable.
            shell_x = EqualContainer().requires_grad_(False)
            shell_y = EqualContainer().requires_grad_(False)
            off_x = child_store.bind(shell_x)
            off_y = child_store.bind(shell_y)
            off_x.deactivate()
            off_y.deactivate()
        finally:
            offloader.deactivate()

    def test_unregistered_property_block_rejected_atomically(self) -> None:
        class UnregisteredPropModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.registered = nn.Linear(4, 4, bias=False)
                self._unregistered = [nn.Linear(4, 4, bias=False)]
                self.requires_grad_(False)

            @property
            def blocks(self) -> nn.ModuleList:
                return nn.ModuleList([self.registered, self._unregistered[0]])

        direct = UnregisteredPropModel()
        snap = _snapshot_tensors(direct)
        with pytest.raises(ValueError, match="not registered on the root model"):
            ModelOffloader.from_module(direct, OffloadConfig(offload_paths=("blocks",)))
        _assert_unmutated(direct, snap)

        store = ModelStore.from_module(UnregisteredPropModel())
        target = UnregisteredPropModel()
        snap_target = _snapshot_tensors(target)
        with pytest.raises(ValueError, match="does not occupy"):
            store.bind(target, OffloadConfig(offload_paths=("blocks",)))
        _assert_unmutated(target, snap_target)

    def test_bind_ignores_extra_shell_only_block_submodule_alias(self) -> None:
        torch.manual_seed(8)

        class ProjBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(4, 4, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(self.proj(x))

        class OrdinaryModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([ProjBlock(), ProjBlock()])
                self.requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = block(x)
                return x

        class SlicedPropModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([ProjBlock() for _ in range(3)])
                self.requires_grad_(False)

            @property
            def blocks(self) -> nn.ModuleList:
                return self.layers[1:]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.layers[0](x)
                for block in self.blocks:
                    x = block(x)
                return x

        x = torch.randn(2, 4)
        cfg = OffloadConfig(offload_paths=("blocks",))

        ord_proto = OrdinaryModel()
        with torch.inference_mode():
            ord_expected = ord_proto(x)
        ord_store = ModelStore.from_module(copy.deepcopy(ord_proto))
        ord_shell = OrdinaryModel()
        ord_shell.blocks[0].extra = ord_shell.blocks[0].proj
        ord_offloader = ord_store.bind(ord_shell, cfg)
        try:
            assert "blocks.0.extra.weight" not in ord_offloader.param_names
            assert "blocks.0.proj.weight" in ord_offloader.param_names
            with activated_model(ord_offloader, "cpu"), torch.inference_mode():
                torch.testing.assert_close(ord_shell(x), ord_expected, atol=0, rtol=0)
        finally:
            ord_offloader.deactivate()

        prop_proto = SlicedPropModel()
        with torch.inference_mode():
            prop_expected = prop_proto(x)
        prop_store = ModelStore.from_module(copy.deepcopy(prop_proto))
        prop_shell = SlicedPropModel()
        prop_shell.blocks[0].extra = prop_shell.blocks[0].proj
        prop_offloader = prop_store.bind(prop_shell, cfg)
        try:
            assert "layers.1.extra.weight" not in prop_offloader.param_names
            assert "layers.1.proj.weight" in prop_offloader.param_names
            with activated_model(prop_offloader, "cpu"), torch.inference_mode():
                torch.testing.assert_close(prop_shell(x), prop_expected, atol=0, rtol=0)
        finally:
            prop_offloader.deactivate()

    def test_repeated_block_aliases_scale_linearly_in_memory(self) -> None:
        def make_repeated() -> nn.Module:
            shared = nn.Module()
            for i in range(12):
                setattr(shared, f"p{i}", nn.Parameter(torch.zeros(2, 2), requires_grad=False))
            model = nn.Module()
            model.blocks = nn.ModuleList([shared] * 256)
            return model

        cfg = OffloadConfig(offload_paths=("blocks",))
        direct = make_repeated()
        tracemalloc.start()
        try:
            offloader = ModelOffloader.from_module(direct, cfg)
            _cur, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        try:
            assert peak < 4 * 1024 * 1024
            assert len(offloader.param_names) == 256 * 12
            index = offloader._partition._unit_param_index
            assert index["blocks.0.p0"] == (0, "p0")
            assert index["blocks.255.p11"] == (0, "p11")
        finally:
            offloader.deactivate()

        store = ModelStore.from_module(make_repeated())
        bound = store.bind(make_repeated(), cfg)
        try:
            assert len(bound.param_names) == 256 * 12
            index_bound = bound._partition._unit_param_index
            assert index_bound["blocks.0.p0"] == (0, "p0")
            assert index_bound["blocks.255.p11"] == (0, "p11")
        finally:
            bound.deactivate()


def _write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    import piper_offload.checkpoint as checkpoint_module

    tags = {dtype: tag for tag, dtype in checkpoint_module._DTYPES.items()}
    header: dict[str, object] = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        raw = (
            tensor.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
            if tensor.numel()
            else b""
        )
        header[name] = {
            "dtype": tags[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload += raw
    encoded = json.dumps(header).encode()
    encoded += b" " * (-(8 + len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(payload))


class _TPSequenceBlock(nn.Module):
    def __init__(self, dim: int, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2, bias=True, device=device, dtype=dtype)
        self.fc2 = nn.Linear(dim * 2, dim, bias=True, device=device, dtype=dtype)
        self.norm = nn.LayerNorm(dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc2(torch.relu(self.fc1(x)))) + x


class _TPSequenceModel(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 8,
        num_blocks: int = 2,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        dev = torch.device(device)
        self.embed = nn.Linear(dim, dim, bias=False, device=dev, dtype=dtype)
        self.blocks = nn.ModuleList(
            [_TPSequenceBlock(dim, device=dev, dtype=dtype) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(dim, dim, bias=False, device=dev, dtype=dtype)
        self.register_buffer("scale", torch.ones(dim, device=dev, dtype=dtype))
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x) * self.scale
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def _parallelize_tp_sequence(model: nn.Module, mesh: Any) -> nn.Module:
    from torch.distributed.tensor import Shard
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        SequenceParallel,
        parallelize_module,
    )

    plan: dict[str, Any] = {}
    for idx in range(len(cast(nn.ModuleList, model.blocks))):
        plan[f"blocks.{idx}.fc1"] = ColwiseParallel(input_layouts=Shard(1))
        plan[f"blocks.{idx}.fc2"] = RowwiseParallel(output_layouts=Shard(1))
        plan[f"blocks.{idx}.norm"] = SequenceParallel(sequence_dim=1, use_local_output=True)
    # PyTorch's SequenceParallel wraps replicated norm params without preserving requires_grad=False.
    return parallelize_module(model, mesh, plan).requires_grad_(False)


def _make_meta_dtensor_param(
    mesh: Any,
    placements: list[Any],
    *,
    global_shape: tuple[int, ...] = (4, 4),
    local_shape: tuple[int, ...] | None = None,
) -> nn.Parameter:
    from torch.distributed.tensor import DTensor, Shard

    resolved_local = local_shape
    if resolved_local is None:
        if len(placements) == 1 and placements[0] == Shard(0):
            resolved_local = (global_shape[0] // 2, *global_shape[1:])
        elif len(placements) == 1 and placements[0] == Shard(1):
            resolved_local = (global_shape[0], global_shape[1] // 2, *global_shape[2:])
        else:
            resolved_local = global_shape
    stride = (global_shape[1], 1) if len(global_shape) == 2 else (1,)
    return nn.Parameter(
        DTensor.from_local(
            torch.empty(resolved_local, device="meta"),
            mesh,
            placements,
            shape=global_shape,
            stride=stride,
            run_check=False,
        ),
        requires_grad=False,
    )


def _load_tp_model_from_mmap_checkpoint(tmp_path: Path, ref: _TPSequenceModel) -> tuple[Any, ModelStore]:
    from piper_offload import MappedCheckpoint

    ckpt_path = tmp_path / "tp_model.safetensors"
    _write_safetensors(ckpt_path, {k: v.clone() for k, v in ref.state_dict().items()})
    ckpt = MappedCheckpoint(ckpt_path)
    try:
        with torch.device("meta"):
            proto = _TPSequenceModel(dim=8, num_blocks=2, device="meta", dtype=torch.bfloat16)
        for name, _p in list(proto.named_parameters()):
            parent = proto.get_submodule(name.rpartition(".")[0]) if "." in name else proto
            parent.register_parameter(
                name.rpartition(".")[2],
                nn.Parameter(ckpt.get_tensor(name), requires_grad=False),
            )
        proto.register_buffer("scale", ckpt.get_tensor("scale").clone())
        store = ModelStore.from_module(proto)
    except BaseException:
        ckpt.close()
        raise
    return ckpt, store


def _build_valid_tied_shell(mesh: Any) -> nn.Module:
    from torch.distributed.tensor import Shard

    m = nn.Module()
    m.a = _make_meta_dtensor_param(mesh, [Shard(0)])
    m.b = m.a
    m.blocks = nn.ModuleList([nn.Module()])
    m.blocks[0].w = _make_meta_dtensor_param(mesh, [Shard(0)])
    m.register_buffer("buf", torch.empty(4, device="meta"))
    return m


def _assert_2d_quantized_cuda_staging(
    *,
    nv_hp: Any,
    cr_hp: Any,
    resting_sigs: tuple[tuple[Any, ...], tuple[Any, ...]],
    nv_global: Any,
    cr_global: Any,
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
    cuda_lock: threading.Lock,
) -> None:
    cuda_dev = torch.device("cuda:0")
    with cuda_lock:
        nv_gpu_state = cast(Any, nv_hp.allocate_gpu_storage(cuda_dev))
        cr_gpu_state = cast(Any, cr_hp.allocate_gpu_storage(cuda_dev))
        nv_hp.copy_to_gpu(nv_gpu_state, non_blocking=False)
        cr_hp.copy_to_gpu(cr_gpu_state, non_blocking=False)
        nv_host_state = cast(Any, nv_hp.host_state)
        cr_host_state = cast(Any, cr_hp.host_state)
        nv_local = cast(Any, nv_host_state.inner.gpu_param(nv_host_state.inner_state, nv_gpu_state.inner_gpu).data)
        cr_local = cast(Any, cr_host_state.inner.gpu_param(cr_host_state.inner_state, cr_gpu_state.inner_gpu).data)
        assert nv_host_state.inner.layout_signature(nv_local) == resting_sigs[0][1]
        assert cr_host_state.inner.layout_signature(cr_local) == resting_sigs[1][1]
        assert nv_local.qdata.is_cuda and nv_local.qdata.is_contiguous() and nv_local.scale.is_contiguous()
        assert cr_local.qdata.is_cuda and cr_local.qdata.is_contiguous() and cr_local.scale.is_contiguous()
        nv_ref = nv_global.dequantize().narrow(0, offsets[0], local_shape[0]).narrow(1, offsets[1], local_shape[1])
        cr_ref = (
            cr_global.to(cuda_dev)
            .dequantize()
            .narrow(0, offsets[0], local_shape[0])
            .narrow(1, offsets[1], local_shape[1])
            .cpu()
        )
        torch.testing.assert_close(nv_local.dequantize().cpu(), nv_ref, rtol=0, atol=0)
        torch.testing.assert_close(cr_local.dequantize().cpu(), cr_ref, rtol=0, atol=0)


def _assert_2d_quantized_rank(
    *,
    store: ModelStore,
    nv_global: Any,
    cr_global: Any,
    nv_ptrs: tuple[int, ...],
    cr_ptrs: tuple[int, ...],
    delta: Adapter,
    cuda_lock: threading.Lock,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard

    from piper_offload import merge_adapter, requires_activation
    from piper_offload.dtensor_adapter import DTensorAdapter, _local_shape_and_offsets
    from piper_offload.host_param import HostParam

    mesh = DeviceMesh("cpu", [[0, 1], [2, 3]])
    coord = tuple(mesh.get_coordinate() or ())
    placements = (Shard(0), Shard(1))
    local_shape, offsets = _local_shape_and_offsets((16, 64), tuple(mesh.shape), coord, placements)

    shell = nn.Module()
    for name in ("nv", "cr"):
        meta_dt = DTensor.from_local(
            torch.empty(local_shape, dtype=torch.float32, device="meta"),
            mesh,
            list(placements),
            shape=(16, 64),
            stride=(64, 1),
            run_check=False,
        )
        shell.register_parameter(name, nn.Parameter(meta_dt, requires_grad=False))

    offloader = store.bind(shell)
    try:
        nv_param = cast(DTensor, shell.get_parameter("nv").data)
        cr_param = cast(DTensor, shell.get_parameter("cr").data)
        assert requires_activation(nv_param) and requires_activation(cr_param)
        with pytest.raises(RuntimeError, match="requires CUDA activation"):
            nv_param.to_local().clone()
        with pytest.raises(ValueError, match="already bound to a live ModelOffloader"):
            merge_adapter(shell, [(delta, 0.5)])
        with pytest.raises(ValueError, match="cannot be captured directly"):
            HostParam(shell.get_parameter("nv"))
        res = resident_instance(offloader)
        assert res is not None
        nv_hp = res.params["nv"]
        cr_hp = res.params["cr"]
        assert tuple(t.untyped_storage().data_ptr() for t in nv_hp.storage_tensors()) == nv_ptrs
        assert tuple(t.untyped_storage().data_ptr() for t in cr_hp.storage_tensors()) == cr_ptrs
        assert nv_hp.cache_bytes == (8 * 16) + (8 * 2)
        assert cr_hp.cache_bytes == (8 * 32) + (8 * 4)

        resting_sigs = (
            DTensorAdapter.layout_signature(nv_param),
            DTensorAdapter.layout_signature(cr_param),
        )
        with activated_model(offloader, "cpu") as active_cpu:
            assert requires_activation(cast(DTensor, active_cpu.get_parameter("nv").data))
            active_cr_sig = DTensorAdapter.layout_signature(cast(DTensor, active_cpu.get_parameter("cr").data))
            assert active_cr_sig == resting_sigs[1]

        if torch.cuda.is_available():
            _assert_2d_quantized_cuda_staging(
                nv_hp=nv_hp,
                cr_hp=cr_hp,
                resting_sigs=resting_sigs,
                nv_global=nv_global,
                cr_global=cr_global,
                offsets=offsets,
                local_shape=local_shape,
                cuda_lock=cuda_lock,
            )
        return offloader.cache_bytes, local_shape, offsets
    finally:
        offloader.deactivate()


class TestModelStoreDTensorProjection:
    def test_bind_projects_parallelize_module_meta_shells_and_shares_mmap(
        self, tmp_path: Path,
    ) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        from torch.distributed.tensor import DTensor, Replicate, Shard, init_device_mesh

        from piper_offload.parallel import ParallelExecutor

        torch.manual_seed(11)
        ref = _TPSequenceModel(dim=8, num_blocks=2, device="cpu", dtype=torch.bfloat16)
        ckpt, store = _load_tp_model_from_mmap_checkpoint(tmp_path, ref)
        try:
            store_ptrs = {
                name: tuple(t.untyped_storage().data_ptr() for t in hp.storage_tensors())
                for name, hp in store._module_store.params.items()
            }
            x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
            with torch.inference_mode():
                expected = ref(x)

            cfg = OffloadConfig(offload_paths=("blocks",))
            bound_offloaders: list[ModelOffloader | None] = [None, None]

            def setup_worker(rank: int) -> tuple[torch.Tensor, int]:
                mesh = init_device_mesh("cpu", (2,))
                with torch.device("meta"):
                    shell = _TPSequenceModel(dim=8, num_blocks=2, device="meta", dtype=torch.float32)
                _parallelize_tp_sequence(shell, mesh)
                offloader = store.bind(shell, cfg)
                bound_offloaders[rank] = offloader

                for name, param in shell.named_parameters():
                    raw = param.to_local() if isinstance(param.data, DTensor) else param.data
                    assert (raw.untyped_storage().data_ptr(),) == store_ptrs[name]
                    assert raw.dtype is torch.bfloat16

                assert offloader.cache_bytes < store.cache_bytes
                assert shell.blocks[0].fc1.weight.placements == (Shard(0),)
                assert shell.blocks[0].fc2.weight.placements == (Shard(1),)
                assert shell.blocks[0].fc2.bias.placements == (Replicate(),)
                assert shell.blocks[0].norm.weight.placements == (Replicate(),)
                assert type(shell.embed.weight.data) is torch.Tensor
                fc1_hp = offloaded_units(offloader)[0].params["fc1.weight"]
                assert fc1_hp.compute_dtype is torch.bfloat16

                offloader.activate("cpu")
                with torch.inference_mode():
                    out = offloader.value(x.chunk(2, dim=1)[rank])
                return out.clone(), offloader.cache_bytes

            with ParallelExecutor(["cpu", "cpu"]) as ex:
                results = ex.run(setup_worker)
                for r in range(2):
                    torch.testing.assert_close(results[r][0], expected.chunk(2, dim=1)[r], rtol=1e-2, atol=1e-2)
                    cast(ModelOffloader, bound_offloaders[r]).deactivate()

                def rerun(rank: int) -> torch.Tensor:
                    off = cast(ModelOffloader, bound_offloaders[rank])
                    with activated_model(off, "cpu"), torch.inference_mode():
                        return off.value(x.chunk(2, dim=1)[rank]).clone()

                outs2 = ex.run(rerun)
                for r in range(2):
                    torch.testing.assert_close(outs2[r], expected.chunk(2, dim=1)[r], rtol=1e-2, atol=1e-2)

                ex.run(lambda rank: cast(ModelOffloader, bound_offloaders[rank]).activate("cpu"))

            for off in bound_offloaders:
                cast(ModelOffloader, off).deactivate()
        finally:
            ckpt.close()

    def test_bind_dtensor_2d_mesh_uneven_and_empty_shards(self) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import DTensor, Replicate, Shard

        from piper_offload.dtensor_adapter import _local_shape_and_offsets
        from piper_offload.parallel import ParallelExecutor

        class _MixedShardModel(nn.Module):
            def __init__(self, *, device: str = "cpu", dtype: torch.dtype = torch.bfloat16) -> None:
                super().__init__()
                self.w_rep_s0 = nn.Parameter(torch.empty((6, 4), device=device, dtype=dtype), requires_grad=False)
                self.w_s0_s1 = nn.Parameter(torch.empty((4, 6), device=device, dtype=dtype), requires_grad=False)
                self.w_uneven = nn.Parameter(torch.empty((5, 4), device=device, dtype=dtype), requires_grad=False)
                self.w_empty = nn.Parameter(torch.empty((1, 4), device=device, dtype=dtype), requires_grad=False)
                self.b_rep = nn.Parameter(torch.empty((4,), device=device, dtype=dtype), requires_grad=False)

        torch.manual_seed(17)
        proto = _MixedShardModel()
        for p in proto.parameters():
            p.copy_(torch.arange(p.numel(), dtype=torch.bfloat16).reshape(p.shape) * 0.01)
        full_tensors = {n: p.detach().clone() for n, p in proto.named_parameters()}
        store = ModelStore.from_module(proto)
        store_ptrs = {
            n: hp.storage_tensors()[0].untyped_storage().data_ptr()
            for n, hp in store._module_store.params.items()
        }
        specs = {
            "w_rep_s0": (Replicate(), Shard(0)),
            "w_s0_s1": (Shard(0), Shard(1)),
            "w_uneven": (Shard(0), Replicate()),
            "w_empty": (Shard(0), Replicate()),
            "b_rep": (Replicate(), Replicate()),
        }

        def verify(_rank: int) -> dict[str, tuple[int, tuple[int, ...]]]:
            mesh = DeviceMesh("cpu", [[0, 1], [2, 3]])
            coord = tuple(mesh.get_coordinate() or ())
            shell = _MixedShardModel(device="meta", dtype=torch.float32)
            for name, placements in specs.items():
                global_t = full_tensors[name]
                local_shape, _ = _local_shape_and_offsets(
                    tuple(global_t.shape), tuple(mesh.shape), coord, placements,
                )
                meta_dt = DTensor.from_local(
                    torch.empty(local_shape, device="meta"),
                    mesh,
                    list(placements),
                    shape=global_t.shape,
                    stride=global_t.stride(),
                    run_check=False,
                )
                shell.register_parameter(name, nn.Parameter(meta_dt, requires_grad=False))

            offloader = store.bind(shell)
            try:
                summary: dict[str, tuple[int, tuple[int, ...]]] = {}
                with activated_model(offloader, "cpu"):
                    for name, placements in specs.items():
                        param = offloader.model.get_parameter(name)
                        local = param.to_local()
                        assert local.untyped_storage().data_ptr() == store_ptrs[name]
                        local_shape, offsets = _local_shape_and_offsets(
                            tuple(full_tensors[name].shape), tuple(mesh.shape), coord, placements,
                        )
                        expected = full_tensors[name]
                        for dim, (off, sz) in enumerate(zip(offsets, local_shape, strict=True)):
                            if off != 0 or sz != expected.shape[dim]:
                                expected = expected.narrow(dim, off, sz)
                        torch.testing.assert_close(local, expected, rtol=0, atol=0)
                        summary[name] = (local.numel(), tuple(local.shape))
                return summary
            finally:
                offloader.deactivate()

        with ParallelExecutor(["cpu"] * 4) as ex:
            results = ex.run(verify)
        assert results[0]["w_empty"] == (4, (1, 4))
        assert results[2]["w_empty"] == (0, (0, 4))
        assert results[0]["w_uneven"] == (12, (3, 4))
        assert results[2]["w_uneven"] == (8, (2, 4))

    def test_bind_dtensor_tied_weights_and_group_ties(self) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        from torch.distributed.tensor import Shard, init_device_mesh

        from piper_offload.parallel import ParallelExecutor

        shared = nn.Parameter(torch.randn(4, 4), requires_grad=False)
        proto = nn.Module()
        proto.a = shared
        proto.b = shared
        proto.blocks = nn.ModuleList([nn.Module()])
        proto.blocks[0].w = shared
        store = ModelStore.from_module(proto)
        ptr = store._module_store.params["a"].storage_tensors()[0].untyped_storage().data_ptr()

        def verify(_rank: int) -> None:
            mesh = init_device_mesh("cpu", (2,))
            same = nn.Module()
            same.a = _make_meta_dtensor_param(mesh, [Shard(0)])
            same.b = _make_meta_dtensor_param(mesh, [Shard(0)])
            same.blocks = nn.ModuleList([nn.Module()])
            same.blocks[0].w = _make_meta_dtensor_param(mesh, [Shard(0)])
            offloader = store.bind(same)
            assert same.a is same.b
            offloader.deactivate()

            conflict = nn.Module()
            conflict.a = _make_meta_dtensor_param(mesh, [Shard(0)])
            conflict.b = _make_meta_dtensor_param(mesh, [Shard(0)])
            conflict.blocks = nn.ModuleList([nn.Module()])
            conflict.blocks[0].w = _make_meta_dtensor_param(mesh, [Shard(0)])
            snap = _snapshot_tensors(conflict)
            with pytest.raises(ValueError, match="Parameter storage shared across offload boundaries"):
                store.bind(conflict, OffloadConfig(offload_paths=("blocks",)))
            _assert_unmutated(conflict, snap)

            # Differing placements project independently over 1x storage and pass across groups.
            cross = nn.Module()
            cross.a = _make_meta_dtensor_param(mesh, [Shard(1)])
            cross.b = _make_meta_dtensor_param(mesh, [Shard(1)])
            cross.blocks = nn.ModuleList([nn.Module()])
            cross.blocks[0].w = _make_meta_dtensor_param(mesh, [Shard(0)])
            diff_offloader = store.bind(cross, OffloadConfig(offload_paths=("blocks",)))
            assert cross.a is cross.b
            assert cross.a is not cross.blocks[0].w
            assert cross.a.to_local().untyped_storage().data_ptr() == ptr
            assert cross.blocks[0].w.to_local().untyped_storage().data_ptr() == ptr
            diff_offloader.deactivate()

        with ParallelExecutor(["cpu", "cpu"]) as ex:
            ex.run(verify)

    def test_bind_dtensor_preflight_atomicity_and_mesh_safety(self) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import DTensor, Partial, Shard, init_device_mesh

        from piper_offload._dtensor import cpu_mesh_for
        from piper_offload.parallel import ParallelExecutor

        shared = nn.Parameter(torch.randn(4, 4), requires_grad=False)
        proto = nn.Module()
        proto.a, proto.b = shared, shared
        proto.blocks = nn.ModuleList([nn.Module()])
        proto.blocks[0].w = nn.Parameter(torch.randn(4, 4), requires_grad=False)
        proto.register_buffer("buf", torch.ones(4))
        store = ModelStore.from_module(proto)

        def assert_rejected(
            shell: nn.Module,
            exc_type: type[Exception],
            match: str,
            cfg: OffloadConfig | None = None,
        ) -> None:
            snap = _snapshot_tensors(shell)
            with pytest.raises(exc_type, match=match):
                store.bind(shell, cfg)
            _assert_unmutated(shell, snap)

        def verify(_rank: int) -> None:
            mesh = init_device_mesh("cpu", (2,))

            m_shape = _build_valid_tied_shell(mesh)
            m_shape.a = _make_meta_dtensor_param(mesh, [Shard(0)], global_shape=(6, 4), local_shape=(3, 4))
            m_shape.b = m_shape.a
            assert_rejected(m_shape, ValueError, "layout mismatch")

            m_partial = _build_valid_tied_shell(mesh)
            m_partial.a = _make_meta_dtensor_param(mesh, [Partial()], local_shape=(4, 4))
            m_partial.b = m_partial.a
            assert_rejected(m_partial, ValueError, "layout mismatch")

            m_buf = _build_valid_tied_shell(mesh)
            buf_dt = _make_meta_dtensor_param(mesh, [Shard(0)], global_shape=(4,), local_shape=(2,)).data
            m_buf.register_buffer("buf", buf_dt)
            assert_rejected(m_buf, ValueError, "Buffer 'buf' layout mismatch")

            uninit = DeviceMesh("cpu", [0, 1], mesh_dim_names=("_piper_resting_0",), _init_backend=False)
            m_uninit = _build_valid_tied_shell(mesh)
            m_uninit.a = _make_meta_dtensor_param(uninit, [Shard(0)])
            m_uninit.b = m_uninit.a
            assert_rejected(m_uninit, ValueError, "layout mismatch")

            m_alias = _build_valid_tied_shell(mesh)
            m_alias.b = _make_meta_dtensor_param(mesh, [Shard(0)], local_shape=(1, 4))
            assert_rejected(m_alias, ValueError, "layout mismatch")

            assert cpu_mesh_for(mesh) is mesh
            _ = nn.Parameter(
                DTensor.from_local(torch.empty(2, 4), uninit, [Shard(0)], shape=(4, 4), stride=(4, 1), run_check=False),
                requires_grad=False,
            )
            live = nn.Parameter(
                DTensor.from_local(torch.ones(2, 4), mesh, [Shard(0)], shape=(4, 4), stride=(4, 1), run_check=False),
                requires_grad=False,
            )
            torch.testing.assert_close(live.full_tensor(), torch.ones(4, 4), rtol=0, atol=0)

        with ParallelExecutor(["cpu", "cpu"]) as ex:
            ex.run(verify)

    def test_bind_dtensor_quantized_replicate_and_shard_projection(self) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        convrot_mod = pytest.importorskip("piper_kernels.weights.convrot.int8")
        from torch.distributed.tensor import DTensor, Replicate, Shard, init_device_mesh

        from piper_offload import TrellisParameter, requires_activation
        from piper_offload.parallel import ParallelExecutor

        q_mod = nn.Linear(32, 16, bias=False, dtype=torch.bfloat16).eval().requires_grad_(False)
        q_mod.weight = nn.Parameter(
            convrot_mod.ConvRotInt8Tensor.from_hp(q_mod.weight.data, group_size=16),
            requires_grad=False,
        )
        q_store = ModelStore.from_module(q_mod)
        q_host = q_store._module_store.params["weight"].storage_tensors()
        q_ptrs = tuple(t.untyped_storage().data_ptr() for t in q_host)

        trellis_mod = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16).eval().requires_grad_(False)
        trellis_mod.weight = TrellisParameter(
            torch.zeros((16, 1, 16), dtype=torch.uint8),
            torch.ones((16, 1), dtype=torch.float8_e4m3fn),
            torch.ones((16, 1)),
            block_size=64,
            group_size=16,
        )
        trellis_store = ModelStore.from_module(trellis_mod)

        def verify(_rank: int) -> None:
            mesh = init_device_mesh("cpu", (2,))
            rep_shell = nn.Linear(32, 16, bias=False, device="meta")
            rep_shell.weight = _make_meta_dtensor_param(mesh, [Replicate()], global_shape=(16, 32))
            rep_offloader = q_store.bind(rep_shell)
            assert rep_offloader.model.weight.placements == (Replicate(),)
            assert not requires_activation(rep_offloader.model.weight)
            rep_offloader.deactivate()

            s0_shell = nn.Linear(32, 16, bias=False, device="meta")
            s0_shell.weight = _make_meta_dtensor_param(mesh, [Shard(0)], global_shape=(16, 32), local_shape=(8, 32))
            s0_offloader = q_store.bind(s0_shell)
            assert s0_offloader.model.weight.placements == (Shard(0),)
            assert not requires_activation(s0_offloader.model.weight)
            s0_local = s0_offloader.model.weight.to_local()
            assert (s0_local.qdata.untyped_storage().data_ptr(), s0_local.scale.untyped_storage().data_ptr()) == q_ptrs
            s0_offloader.deactivate()

            s1_shell = nn.Linear(32, 16, bias=False, device="meta")
            s1_shell.weight = _make_meta_dtensor_param(mesh, [Shard(1)], global_shape=(16, 32), local_shape=(16, 16))
            s1_offloader = q_store.bind(s1_shell)
            assert isinstance(s1_offloader.model.weight.data, DTensor)
            assert requires_activation(s1_offloader.model.weight)
            s1_offloader.deactivate()

            trellis_shell = nn.Linear(64, 16, bias=False, device="meta")
            trellis_shell.weight = _make_meta_dtensor_param(
                mesh, [Shard(0)], global_shape=(16, 64), local_shape=(8, 64)
            )
            snap = _snapshot_tensors(trellis_shell)
            with pytest.raises(ValueError, match="layout mismatch"):
                trellis_store.bind(trellis_shell)
            _assert_unmutated(trellis_shell, snap)

        with ParallelExecutor(["cpu", "cpu"]) as ex:
            ex.run(verify)

    def test_bind_dtensor_dedups_sliced_submeshes_and_preserves_source_stride(self) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        from torch.distributed.tensor import DTensor, Shard, init_device_mesh

        from piper_offload.parallel import ParallelExecutor

        shared = nn.Parameter(torch.randn(4, 6), requires_grad=False)
        proto = nn.Module()
        proto.a, proto.b = shared, shared
        proto.blocks = nn.ModuleList([nn.Module()])
        proto.blocks[0].w = shared
        store = ModelStore.from_module(proto)

        def verify(_rank: int) -> None:
            mesh_2d = init_device_mesh("cpu", (1, 2), mesh_dim_names=("dp", "tp"))
            tp_first, tp_second = mesh_2d["tp"], mesh_2d["tp"]
            assert tp_first is not tp_second

            shell = nn.Module()
            shell.a = nn.Parameter(
                DTensor.from_local(
                    torch.empty((6, 2), device="meta").t(),
                    tp_first,
                    [Shard(0)],
                    shape=(4, 6),
                    stride=(1, 4),
                    run_check=False,
                ),
                requires_grad=False,
            )
            shell.b = _make_meta_dtensor_param(tp_second, [Shard(0)], global_shape=(4, 6), local_shape=(2, 6))
            shell.blocks = nn.ModuleList([nn.Module()])
            shell.blocks[0].w = _make_meta_dtensor_param(
                mesh_2d["tp"], [Shard(0)], global_shape=(4, 6), local_shape=(2, 6),
            )
            offloader = store.bind(shell)
            assert shell.a is shell.b
            assert shell.a.stride() == (6, 1)
            assert tuple(shell.a.view(24).shape) == (24,)
            offloader.deactivate()

            # Confirm cross-group rejection on fresh meta wrappers over distinct mesh_2d["tp"] slices.
            conflict = nn.Module()
            conflict.a = _make_meta_dtensor_param(tp_first, [Shard(0)], global_shape=(4, 6), local_shape=(2, 6))
            conflict.b = _make_meta_dtensor_param(tp_second, [Shard(0)], global_shape=(4, 6), local_shape=(2, 6))
            conflict.blocks = nn.ModuleList([nn.Module()])
            conflict.blocks[0].w = _make_meta_dtensor_param(
                mesh_2d["tp"], [Shard(0)], global_shape=(4, 6), local_shape=(2, 6),
            )
            snap = _snapshot_tensors(conflict)
            with pytest.raises(ValueError, match="Parameter storage shared across offload boundaries"):
                store.bind(conflict, OffloadConfig(offload_paths=("blocks",)))
            _assert_unmutated(conflict, snap)

        with ParallelExecutor(["cpu", "cpu"]) as ex:
            ex.run(verify)

    def test_bind_dtensor_quantized_2d_mesh_zero_copy_and_per_rank_bytes(self) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        pytest.importorskip("numpy")
        piper_nv = pytest.importorskip("piper_kernels.weights.nvfp4")
        convrot_mod = pytest.importorskip("piper_kernels.weights.convrot.int8")

        from piper_offload import ParameterDelta
        from piper_offload.parallel import ParallelExecutor

        torch.manual_seed(11)
        nv_global = piper_nv.PiperNVFP4Tensor.from_hp(
            torch.randn(16, 64, dtype=torch.bfloat16), is_swizzled_scales=False
        )
        cr_global = convrot_mod.ConvRotInt8Tensor.from_hp(torch.randn(16, 64, dtype=torch.bfloat16), group_size=16)

        proto = nn.Module()
        proto.nv = nn.Parameter(nv_global, requires_grad=False)
        proto.cr = nn.Parameter(cr_global, requires_grad=False)
        proto.requires_grad_(False)
        store = ModelStore.from_module(proto)
        nv_ptrs = tuple(t.untyped_storage().data_ptr() for t in store._module_store.params["nv"].storage_tensors())
        cr_ptrs = tuple(t.untyped_storage().data_ptr() for t in store._module_store.params["cr"].storage_tensors())
        cuda_lock = threading.Lock()
        delta = Adapter(
            {
                "nv": ParameterDelta.from_tensors(dense=torch.randn(16, 64, dtype=torch.bfloat16) * 0.05),
                "cr": ParameterDelta.from_tensors(dense=torch.randn(16, 64, dtype=torch.bfloat16) * 0.05),
            }
        )

        with ParallelExecutor(["cpu"] * 4) as ex:
            results = ex.run(
                lambda _rank: _assert_2d_quantized_rank(
                    store=store,
                    nv_global=nv_global,
                    cr_global=cr_global,
                    nv_ptrs=nv_ptrs,
                    cr_ptrs=cr_ptrs,
                    delta=delta,
                    cuda_lock=cuda_lock,
                )
            )
        assert all(bytes_count < store.cache_bytes for bytes_count, _, _ in results)
        assert {offsets for _, _, offsets in results} == {(0, 0), (0, 32), (8, 0), (8, 32)}

    def test_bind_dtensor_preflight_atomic_on_unwrappable_quant_replicate_and_cpu_mesh_post_close(self) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        convrot_mod = pytest.importorskip("piper_kernels.weights.convrot.int8")
        from types import SimpleNamespace

        from torch.distributed.tensor import Replicate, init_device_mesh

        from piper_offload._dtensor import cpu_mesh_for
        from piper_offload.parallel import ParallelExecutor

        proto = nn.Module()
        proto.embed = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
        proto.blocks = nn.ModuleList([nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)])
        proto.eval().requires_grad_(False)
        proto.blocks[0].weight = nn.Parameter(
            convrot_mod.ConvRotInt8Tensor.from_hp(proto.blocks[0].weight.data, group_size=16),
            requires_grad=False,
        )
        store = ModelStore.from_module(proto)

        def verify(_rank: int) -> None:
            mesh = init_device_mesh("cpu", (2,))
            with torch.device("meta"):
                shell = nn.Module()
                shell.embed = nn.Linear(16, 16, bias=False)
                shell.blocks = nn.ModuleList([nn.Linear(16, 16, bias=False)])
                shell.eval().requires_grad_(False)
            # Mismatched global shape on Replicate must fail preflight atomically.
            shell.blocks[0].weight = _make_meta_dtensor_param(mesh, [Replicate()], global_shape=(8, 16))
            snap = _snapshot_tensors(shell)
            with pytest.raises(ValueError, match="layout mismatch"):
                store.bind(shell, OffloadConfig(offload_paths=("blocks",)))
            _assert_unmutated(shell, snap)
            assert shell.embed.weight.is_meta

        with ParallelExecutor(["cpu", "cpu"]) as ex:
            captured_mesh = ex.run(lambda _r: init_device_mesh("cpu", (2,)))[0]
            ex.run(verify)

        # cpu_mesh_for handles post-close non-CPU mesh descriptors when _rank is preserved.
        fake_cuda = SimpleNamespace(
            device_type="cuda",
            mesh=captured_mesh.mesh,
            ndim=captured_mesh.ndim,
            _rank=getattr(captured_mesh, "_rank", 0),
        )
        assert not dist.is_initialized()
        resting = cpu_mesh_for(fake_cuda)
        assert resting.device_type == "cpu"

def _build_2d_packed_swizzled_fixtures(
    piper_nv: Any,
    cr_mod: Any,
) -> tuple[ModelStore, ModelStore, dict[str, Any], dict[str, tuple[int, ...]], Adapter]:
    from piper_offload import ParameterDelta

    nv_global = piper_nv.PiperNVFP4Tensor.from_hp(
        torch.randn(512, 256, dtype=torch.bfloat16),
        per_tensor_scale=torch.tensor(0.5, dtype=torch.float32),
        is_swizzled_scales=True,
        use_triton_kernel=False,
    )
    cr_raw = cr_mod.ConvRotNVFP4Tensor.from_torchao(
        piper_nv.PiperNVFP4Tensor.from_hp(
            torch.randn(512, 256, dtype=torch.bfloat16),
            per_tensor_scale=torch.tensor(0.5, dtype=torch.float32),
            is_swizzled_scales=True,
            use_triton_kernel=False,
        ),
        group_size=64,
    )
    cr_global = cr_mod.ConvRotNVFP4Tensor(
        (((cr_raw.qdata & 0x0F) << 4) | (cr_raw.qdata >> 4)).to(torch.uint8),
        cr_raw.scale,
        cr_raw.block_size,
        cr_raw.orig_dtype,
        cr_raw.group_size,
        cr_raw.per_tensor_scale,
        None,
        True,
        False,
        None,
        high_first=True,
    )
    proto = nn.Module()
    proto.nv = nn.Parameter(nv_global, requires_grad=False)
    proto.cr = nn.Parameter(cr_global, requires_grad=False)
    proto.requires_grad_(False)
    store = ModelStore.from_module(proto)
    ptrs = {
        k: tuple(t.untyped_storage().data_ptr() for t in store._module_store.params[k].storage_tensors())
        for k in ("nv", "cr")
    }
    bad_proto = nn.Module()
    bad_proto.nv = nn.Parameter(
        piper_nv.PiperNVFP4Tensor.from_hp(
            torch.randn(256, 64, dtype=torch.bfloat16), is_swizzled_scales=True
        ),
        requires_grad=False,
    )
    bad_store = ModelStore.from_module(bad_proto)
    delta = Adapter({"nv": ParameterDelta.from_tensors(dense=torch.randn(512, 256, dtype=torch.bfloat16) * 0.02)})
    return store, bad_store, {"nv": nv_global, "cr": cr_global}, ptrs, delta


def _assert_cpu_projected_dequantize(
    hp: Any,
    glb: Any,
    offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> None:
    hs = cast(Any, hp.host_state)
    inner_state = hs.inner_state
    slots: list[torch.Tensor | None] = []
    targets = inner_state.target_shapes
    for idx, t in enumerate(inner_state.storage):
        if t is None:
            slots.append(None)
            continue
        ts = None if targets is None else targets[idx]
        c = t.contiguous()
        slots.append(c if ts is None else c.view(ts))
    rebuilt = hs.inner._reconstruct(tuple(slots), inner_state.meta)
    ref = glb.dequantize().narrow(0, offsets[0], local_shape[0]).narrow(1, offsets[1], local_shape[1])
    torch.testing.assert_close(rebuilt.dequantize(), ref, rtol=0, atol=0)


def _assert_2d_packed_swizzled_rank(
    *,
    store: ModelStore,
    bad_store: ModelStore,
    globals_map: dict[str, Any],
    ptrs: dict[str, tuple[int, ...]],
    delta: Adapter,
    cuda_lock: threading.Lock,
) -> tuple[int, tuple[int, ...]]:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard

    from piper_offload import merge_adapter, requires_activation
    from piper_offload.dtensor_adapter import _local_shape_and_offsets
    from piper_offload.host_param import HostParam

    mesh = DeviceMesh("cpu", [[0, 1], [2, 3]])
    coord = tuple(mesh.get_coordinate() or ())
    placements = (Shard(0), Shard(1))
    local_shape, offsets = _local_shape_and_offsets((512, 256), tuple(mesh.shape), coord, placements)
    shell = nn.Module()
    for name in ("nv", "cr"):
        shell.register_parameter(
            name,
            nn.Parameter(
                _make_meta_dtensor_param(mesh, list(placements), global_shape=(512, 256), local_shape=local_shape).data,
                requires_grad=False,
            ),
        )
    offloader = store.bind(shell)
    try:
        resident = resident_instance(offloader)
        assert resident is not None
        rank_hps = resident.params
        expected_bytes = sum(
            sum(t.numel() * t.element_size() for t in rank_hps[k].storage_tensors()) for k in ("nv", "cr")
        )
        assert offloader.cache_bytes == expected_bytes < store.cache_bytes
        for name, glb in globals_map.items():
            assert requires_activation(cast(DTensor, shell.get_parameter(name).data))
            hp = rank_hps[name]
            assert tuple(t.untyped_storage().data_ptr() for t in hp.storage_tensors()) == ptrs[name]
            _assert_cpu_projected_dequantize(hp, glb, offsets, local_shape)
            with pytest.raises(ValueError, match="cannot be captured directly"):
                HostParam(shell.get_parameter(name))
        with pytest.raises(ValueError, match="already bound to a live ModelOffloader"):
            merge_adapter(shell, [(delta, 0.5)])
        if torch.cuda.is_available():
            with cuda_lock:
                dev = torch.device("cuda:0")
                for name, glb in globals_map.items():
                    hp = rank_hps[name]
                    hs = cast(Any, hp.host_state)
                    gs = cast(Any, hp.allocate_gpu_storage(dev))
                    hp.copy_to_gpu(gs, non_blocking=False)
                    loc = cast(Any, hs.inner.gpu_param(hs.inner_state, gs.inner_gpu).data)
                    assert loc.scale.is_cuda and loc.scale.is_contiguous()
                    ref = glb.dequantize().narrow(0, offsets[0], local_shape[0]).narrow(1, offsets[1], local_shape[1])
                    torch.testing.assert_close(loc.dequantize().cpu(), ref, rtol=0, atol=0)
    finally:
        offloader.deactivate()

    bad_shell = nn.Module()
    bad_shell.nv = _make_meta_dtensor_param(mesh, list(placements), global_shape=(256, 64), local_shape=(128, 32))
    snap = _snapshot_tensors(bad_shell)
    with pytest.raises(ValueError, match="layout mismatch"):
        bad_store.bind(bad_shell)
    _assert_unmutated(bad_shell, snap)
    return offloader.cache_bytes, offsets


def _assert_tied_and_canonical_guard_on_rank(
    tied_store: ModelStore,
    canon_store: ModelStore,
    canon_before: tuple[torch.Tensor, torch.Tensor],
    nv_4d_store: ModelStore,
) -> None:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard

    from piper_offload import ParameterDelta, merge_adapter

    mesh = DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names=("ep", "tp"))
    tp = mesh["tp"]
    tied_shell = nn.Module()
    tied_shell.a = _make_meta_dtensor_param(tp, [Shard(1)], global_shape=(256, 256), local_shape=(256, 128))
    tied_shell.b = tied_shell.a
    tied_off = tied_store.bind(tied_shell)
    try:
        tied_resident = resident_instance(tied_off)
        assert tied_resident is not None
        a_hp = tied_resident.params["a"]
        assert tied_off.cache_bytes == a_hp.cache_bytes < tied_store.cache_bytes
    finally:
        tied_off.deactivate()

    canon_shell = nn.Module()
    canon_shell.nv = _make_meta_dtensor_param(tp, [Shard(0)], global_shape=(256, 128), local_shape=(128, 128))
    canon_off = canon_store.bind(canon_shell)
    try:
        d = Adapter({"nv": ParameterDelta.from_tensors(dense=torch.randn(256, 128, dtype=torch.bfloat16) * 0.1)})
        with pytest.raises(ValueError, match="already bound to a live ModelOffloader"):
            merge_adapter(canon_shell, [(d, 1.0)])
        store_slots = canon_store._module_store.params["nv"].storage_tensors()
        assert torch.equal(store_slots[0], canon_before[0])
        assert torch.equal(store_slots[2], canon_before[1])
    finally:
        canon_off.deactivate()

    sh = nn.Module()
    sh.w = nn.Parameter(
        DTensor.from_local(
            torch.empty((1, 2, 100, 128), device="meta"),
            tp,
            [Shard(0)],
            shape=(2, 2, 100, 128),
            stride=(25600, 12800, 128, 1),
            run_check=False,
        ),
        requires_grad=False,
    )
    snap = _snapshot_tensors(sh)
    with pytest.raises(ValueError, match="layout mismatch"):
        nv_4d_store.bind(sh)
    _assert_unmutated(sh, snap)


class TestModelStorePackedSwizzledDTensorProjection:
    def test_bind_dtensor_packed_swizzled_2d_mesh_zero_copy_and_rejections(self) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        pytest.importorskip("numpy")
        piper_nv = pytest.importorskip("piper_kernels.weights.nvfp4")
        cr_mod = pytest.importorskip("piper_kernels.weights.convrot.nvfp4")
        from piper_offload.parallel import ParallelExecutor

        torch.manual_seed(19)
        store, bad_store, globals_map, ptrs, delta = _build_2d_packed_swizzled_fixtures(piper_nv, cr_mod)
        cuda_lock = threading.Lock()
        with ParallelExecutor(["cpu"] * 4) as ex:
            results = ex.run(
                lambda _rank: _assert_2d_packed_swizzled_rank(
                    store=store,
                    bad_store=bad_store,
                    globals_map=globals_map,
                    ptrs=ptrs,
                    delta=delta,
                    cuda_lock=cuda_lock,
                )
            )
        assert {offsets for _, offsets in results} == {(0, 0), (0, 128), (256, 0), (256, 128)}

    def test_bind_dtensor_tied_canonical_guard_and_4d_swizzle_rejection(self) -> None:
        if not DTENSOR_AVAILABLE:
            pytest.skip("DTensor required")
        pytest.importorskip("numpy")
        piper_nv = pytest.importorskip("piper_kernels.weights.nvfp4")
        from piper_offload.parallel import ParallelExecutor

        torch.manual_seed(33)
        tied_proto = nn.Module()
        tied_proto.a = nn.Parameter(
            piper_nv.PiperNVFP4Tensor.from_hp(
                torch.randn(256, 256, dtype=torch.bfloat16), is_swizzled_scales=True
            ),
            requires_grad=False,
        )
        tied_proto.b = tied_proto.a
        tied_store = ModelStore.from_module(tied_proto)

        canon_proto = nn.Module()
        canon_proto.nv = nn.Parameter(
            piper_nv.PiperNVFP4Tensor.from_hp(
                torch.randn(256, 128, dtype=torch.bfloat16),
                per_tensor_scale=torch.tensor(0.5, dtype=torch.float32),
                is_swizzled_scales=True,
                use_triton_kernel=False,
            ),
            requires_grad=False,
        )
        canon_store = ModelStore.from_module(canon_proto)
        c_slots = canon_store._module_store.params["nv"].storage_tensors()
        canon_before = (c_slots[0].clone(), c_slots[2].clone())

        base3d = piper_nv.PiperNVFP4Tensor.from_hp(
            torch.randn(4, 100, 128, dtype=torch.bfloat16), is_swizzled_scales=True, use_triton_kernel=False
        )
        bad_4d = piper_nv.PiperNVFP4Tensor(
            base3d.qdata.view(2, 2, 100, 64),
            base3d.scale.view(2, 2, *base3d.scale.shape[1:]),
            base3d.block_size,
            base3d.orig_dtype,
            None,
            None,
            True,
            False,
            None,
        )
        nv_4d_proto = nn.Module()
        nv_4d_proto.w = nn.Parameter(bad_4d, requires_grad=False)
        nv_4d_store = ModelStore.from_module(nv_4d_proto)

        with ParallelExecutor(["cpu"] * 4) as ex:
            ex.run(
                lambda _r: _assert_tied_and_canonical_guard_on_rank(
                    tied_store, canon_store, canon_before, nv_4d_store
                )
            )
