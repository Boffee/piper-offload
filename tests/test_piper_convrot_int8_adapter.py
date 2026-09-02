"""Tests for Piper ``ConvRotInt8Tensor`` offload support."""

import subprocess
import sys
import weakref
from typing import Any

import pytest
import torch
from torch import nn

from piper_offload import (
    Adapter,
    LoRATransform,
    ModelOffloader,
    ScaledLoRAFactor,
    derive_seed,
    merge_adapter,
)
from piper_offload.piper_convrot_int8_adapter import PiperConvRotInt8Adapter
from piper_offload.dtensor_adapter import DTensorAdapter
from piper_offload.pinned_module import PinnedModuleStore
from piper_offload.pinned_param import PinnedParam
from piper_offload.block_component import (
    _param_target_layout,
    _pin_block_module_stores,
)
from piper_offload.tensor_adapter_registry import select_adapter, tensor_id
from piper_offload.tensor_adapters import (
    CpuRoundTripTensorAdapter,
    DequantRequantTensorAdapter,
    LoRAMergeTensorAdapter,
    LoRAMergeValidationTensorAdapter,
    ParameterDataSwapTensorAdapter,
    TensorCopyIntoAdapter,
)
from tests.conftest import activated_model

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _convrot_cls() -> type:
    module = pytest.importorskip("piper_kernels.linear.convrot")
    return module.ConvRotInt8Tensor


def _make_convrot(
    *,
    rows: int = 8,
    cols: int = 64,
    group_size: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    qdata: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    convrot_cls = _convrot_cls()
    if qdata is None:
        qdata = torch.randint(
            -127,
            128,
            (rows, cols),
            dtype=torch.int8,
            device=device,
        )
    if scale is None:
        scale = torch.rand(
            rows,
            1,
            dtype=torch.float32,
            device=device,
        )
    return convrot_cls.from_quantized(
        qdata,
        scale,
        group_size=group_size,
        logical_dtype=dtype,
    )


def test_package_import_does_not_require_piper_kernels() -> None:
    script = (
        "import builtins\n"
        "real_import = builtins.__import__\n"
        "def import_without_piper("
        "name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if name == 'piper_kernels' or name.startswith('piper_kernels.'):\n"
        "        raise ModuleNotFoundError("
        "'No module named piper_kernels', name='piper_kernels')\n"
        "    return real_import(name, globals, locals, fromlist, level)\n"
        "builtins.__import__ = import_without_piper\n"
        "import torch\n"
        "import piper_offload\n"
        "from piper_offload.piper_convrot_int8_adapter import "
        "PiperConvRotInt8Adapter\n"
        "assert not PiperConvRotInt8Adapter.matches(torch.zeros(1))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


class TestPiperConvRotInt8Adapter:
    def test_matches_and_dispatches_convrot_only(self) -> None:
        convrot = _make_convrot()

        assert PiperConvRotInt8Adapter.matches(convrot)
        assert not PiperConvRotInt8Adapter.matches(torch.zeros(8, 64))
        assert isinstance(select_adapter(convrot), PiperConvRotInt8Adapter)

    def test_validates_mutated_public_storage(self) -> None:
        convrot: Any = _make_convrot()
        convrot.scale = convrot.scale.to(torch.float16)

        with pytest.raises(ValueError, match="scale must be float32"):
            PiperConvRotInt8Adapter.matches(convrot)

    def test_meta_layout_preserves_representation(self) -> None:
        convrot = _make_convrot(device="meta")
        signature = PiperConvRotInt8Adapter.layout_signature(convrot)

        assert convrot.device.type == "meta"
        assert signature[0] == (8, 64)
        assert signature[1] is torch.bfloat16
        assert signature[-2] == (
            "qdata",
            ((8, 64), torch.int8, (64, 1)),
        )
        assert signature[-1] == (
            "scale",
            ((8, 1), torch.float32, (1, 1)),
        )

    def test_pin_preserves_storage_metadata_shape_and_cache_bytes(self) -> None:
        convrot_cls = _convrot_cls()
        source = _make_convrot(dtype=torch.float16)
        pinned_param = PinnedParam(nn.Parameter(source, requires_grad=False))
        pinned = pinned_param.make_cpu_param().data

        assert isinstance(pinned, convrot_cls)
        assert pinned.qdata.is_pinned()
        assert pinned.scale.is_pinned()
        assert pinned.qdata.data_ptr() == pinned_param.pinned_state.storage[0].data_ptr()
        assert pinned.scale.data_ptr() == pinned_param.pinned_state.storage[1].data_ptr()
        assert pinned.group_size == 64
        assert pinned.dtype is torch.float16
        assert tuple(pinned.shape) == (8, 64)
        assert pinned_param.adapter.logical_shape(pinned) == (8, 64)
        assert pinned_param.compute_dtype is torch.float16
        assert pinned_param.cache_bytes == source.qdata.nbytes + source.scale.nbytes
        assert torch.equal(pinned.qdata, source.qdata)
        assert torch.equal(pinned.scale, source.scale)

    def test_streamed_prevalidation_does_not_retain_source_wrappers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    _make_convrot(),
                    requires_grad=False,
                )

        blocks = [Block(), Block()]
        first_source_ref = weakref.ref(blocks[0].weight)
        original_from_module = PinnedModuleStore.from_module
        calls = 0

        @classmethod
        def tracked_from_module(
            cls: type[PinnedModuleStore],
            module: nn.Module,
            **kwargs: object,
        ) -> PinnedModuleStore:
            nonlocal calls
            if calls == 1:
                # The first block has already installed its pinned wrapper.
                # Pre-validation must not keep the replaced source wrapper
                # (and therefore its original qdata/scale storage) alive.
                assert first_source_ref() is None
            calls += 1
            return original_from_module(module, **kwargs)

        monkeypatch.setattr(
            PinnedModuleStore,
            "from_module",
            tracked_from_module,
        )

        stores = _pin_block_module_stores(blocks)

        assert len(stores) == 2
        assert calls == 2
        assert first_source_ref() is None

    def test_adopted_backing_retains_storage_and_metadata(self) -> None:
        source = _make_convrot(dtype=torch.float16)
        pageable_param = PinnedParam(
            nn.Parameter(source, requires_grad=False),
            pin_memory=False,
        )
        pageable = pageable_param.make_cpu_param().data

        assert not pageable.qdata.is_pinned()
        assert not pageable.scale.is_pinned()
        assert pageable.qdata.data_ptr() == source.qdata.data_ptr()
        assert pageable.scale.data_ptr() == source.scale.data_ptr()
        assert pageable.group_size == source.group_size
        assert pageable.dtype is source.dtype
        assert torch.equal(pageable.qdata, source.qdata)
        assert torch.equal(pageable.scale, source.scale)

    def test_tensor_id_tracks_storage_group_size_and_logical_dtype(self) -> None:
        qdata = torch.randint(-127, 128, (8, 64), dtype=torch.int8)
        scale = torch.rand(8, 1)
        group_64 = _make_convrot(qdata=qdata, scale=scale, group_size=64)
        same_storage = _make_convrot(qdata=qdata, scale=scale, group_size=64)
        group_16 = _make_convrot(qdata=qdata, scale=scale, group_size=16)
        float16 = _make_convrot(
            qdata=qdata,
            scale=scale,
            group_size=64,
            dtype=torch.float16,
        )

        key = tensor_id(group_64)
        assert key[0] == "piper-kernels-convrot-int8"
        assert key == tensor_id(same_storage)
        assert key != tensor_id(group_16)
        assert key != tensor_id(float16)

    def test_target_layout_ignores_storage_identity_but_tracks_metadata(self) -> None:
        first = nn.Parameter(_make_convrot(), requires_grad=False)
        second = nn.Parameter(_make_convrot(), requires_grad=False)
        different_group = nn.Parameter(
            _make_convrot(group_size=16),
            requires_grad=False,
        )

        assert _param_target_layout(first) == _param_target_layout(second)
        assert _param_target_layout(first) != _param_target_layout(different_group)

    def test_tied_wrappers_sharing_storage_are_deduplicated(self) -> None:
        qdata = torch.randint(-127, 128, (8, 64), dtype=torch.int8)
        scale = torch.rand(8, 1)
        module = nn.Module()
        module.register_parameter(
            "first",
            nn.Parameter(
                _make_convrot(qdata=qdata, scale=scale),
                requires_grad=False,
            ),
        )
        module.register_parameter(
            "second",
            nn.Parameter(
                _make_convrot(qdata=qdata, scale=scale),
                requires_grad=False,
            ),
        )

        store = PinnedModuleStore.from_module(module)

        assert store.params["first"] is store.params["second"]
        assert module.first is module.second

    def test_dtensor_selection_stays_outermost(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        convrot = _make_convrot()
        monkeypatch.setattr(
            DTensorAdapter,
            "matches",
            staticmethod(lambda tensor: tensor is convrot),
        )

        assert isinstance(select_adapter(convrot), DTensorAdapter)

    def test_lora_transform_uses_piper_addmm_in_place(self) -> None:
        convrot_cls = _convrot_cls()
        rows, cols, rank = 8, 64, 4
        convrot = convrot_cls.from_hp(
            torch.randn(rows, cols, dtype=torch.bfloat16),
            group_size=64,
        )
        param = nn.Parameter(convrot, requires_grad=False)
        a = torch.randn(rank, cols)
        b = torch.randn(rows, rank)
        transform = LoRATransform(
            [ScaledLoRAFactor.from_tensors(a, b, 0.5)]
        )
        original_param = param
        qdata_ptr = param.data.qdata.data_ptr()
        scale_ptr = param.data.scale.data_ptr()

        expected = convrot.clone()
        expected.addmm_(
            b.to(expected.dtype),
            a.to(expected.dtype),
            alpha=0.5,
        )

        transform.validate_target(param)
        transform.apply(param)

        assert param is original_param
        assert param.data.qdata.data_ptr() == qdata_ptr
        assert param.data.scale.data_ptr() == scale_ptr
        assert torch.equal(param.data.qdata, expected.qdata)
        assert torch.equal(param.data.scale, expected.scale)

    def test_merge_lora_merges_convrot_weight(self) -> None:
        convrot_cls = _convrot_cls()
        model = nn.Module()
        model.lin = nn.Linear(64, 8, bias=False, dtype=torch.bfloat16)
        model.lin.weight = nn.Parameter(
            convrot_cls.from_hp(
                torch.zeros(8, 64, dtype=torch.bfloat16),
                group_size=64,
            ),
            requires_grad=False,
        )
        original_qdata = model.lin.weight.data.qdata.clone()
        lora = Adapter.from_state_dict(
            state_dict={
                "lin.lora_A.weight": torch.ones(4, 64),
                "lin.lora_B.weight": torch.ones(8, 4),
            }
        )

        merged = merge_adapter(model, [(lora, 1.0)])

        assert merged == 1
        assert isinstance(model.lin.weight.data, convrot_cls)
        assert not torch.equal(model.lin.weight.data.qdata, original_qdata)

    def test_stochastic_merge_forwards_seed_to_piper_addmm(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        convrot_cls = _convrot_cls()
        rounding_seeds: list[int | None] = []
        original_addmm = convrot_cls.addmm_

        def recording_addmm(
            target: torch.Tensor,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            *,
            beta: float = 1,
            alpha: float = 1,
            rounding_seed: int | None = None,
        ) -> torch.Tensor:
            rounding_seeds.append(rounding_seed)
            return original_addmm(
                target,
                mat1,
                mat2,
                beta=beta,
                alpha=alpha,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(convrot_cls, "addmm_", recording_addmm)
        model = nn.Module()
        model.lin = nn.Linear(64, 8, bias=False, dtype=torch.bfloat16)
        model.lin.weight = nn.Parameter(
            convrot_cls.from_hp(
                torch.zeros(8, 64, dtype=torch.bfloat16),
                group_size=64,
            ),
            requires_grad=False,
        )
        lora = Adapter.from_state_dict(
            state_dict={
                "lin.lora_A.weight": torch.ones(4, 64),
                "lin.lora_B.weight": torch.ones(8, 4),
            }
        )

        merged = merge_adapter(
            model,
            [(lora, 1.0)],
            stochastic_rounding=True,
        )

        assert merged == 1
        assert rounding_seeds == [derive_seed("lin.weight", 0)]

    def test_stochastic_merge_replays(self) -> None:
        convrot_cls = _convrot_cls()
        generator = torch.Generator().manual_seed(123)
        weight = torch.randn(
            8,
            64,
            dtype=torch.bfloat16,
            generator=generator,
        )
        state_dict = {
            "lin.lora_A.weight": torch.randn(8, 64, generator=generator),
            "lin.lora_B.weight": torch.randn(8, 8, generator=generator),
        }

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            model = nn.Module()
            model.lin = nn.Linear(
                64,
                8,
                bias=False,
                dtype=torch.bfloat16,
            )
            model.lin.weight = nn.Parameter(
                convrot_cls.from_hp(weight.clone(), group_size=64),
                requires_grad=False,
            )
            lora = Adapter.from_state_dict(
                state_dict={
                    key: value.clone()
                    for key, value in state_dict.items()
                }
            )
            assert (
                merge_adapter(
                    model,
                    [(lora, 0.125)],
                    stochastic_rounding=True,
                )
                == 1
            )
            return (
                model.lin.weight.data.qdata.clone(),
                model.lin.weight.data.scale.clone(),
            )

        first = run()
        replay = run()
        assert torch.equal(first[0], replay[0])
        assert torch.equal(first[1], replay[1])

    def test_advertises_merge_but_not_training_capabilities(self) -> None:
        adapter = PiperConvRotInt8Adapter()

        assert not isinstance(adapter, CpuRoundTripTensorAdapter)
        assert not isinstance(adapter, DequantRequantTensorAdapter)
        assert not isinstance(adapter, TensorCopyIntoAdapter)
        assert isinstance(adapter, LoRAMergeTensorAdapter)
        assert isinstance(adapter, LoRAMergeValidationTensorAdapter)
        assert not isinstance(adapter, ParameterDataSwapTensorAdapter)

        pinned_param = PinnedParam(
            nn.Parameter(_make_convrot(), requires_grad=True),
        )
        state = pinned_param.allocate_gpu_storage(torch.device("cpu"))
        with pytest.raises(NotImplementedError, match="CPU round-trip"):
            pinned_param.copy_to_cpu(state)
        with pytest.raises(NotImplementedError, match="Parameter.data-swap"):
            pinned_param.validate_parameter_data_swap_target()

    @CUDA
    def test_allocate_copy_and_reconstruct_gpu_wrapper(self) -> None:
        convrot_cls = _convrot_cls()
        source = _make_convrot(dtype=torch.float32)
        pinned_param = PinnedParam(nn.Parameter(source, requires_grad=False))

        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        torch.cuda.synchronize()

        assert isinstance(gpu_param.data, convrot_cls)
        assert gpu_param.data.qdata.is_cuda
        assert gpu_param.data.scale.is_cuda
        assert gpu_param.data.group_size == 64
        assert gpu_param.data.dtype is torch.float32
        assert torch.equal(gpu_param.data.qdata.cpu(), source.qdata)
        assert torch.equal(gpu_param.data.scale.cpu(), source.scale)

    @CUDA
    def test_model_offloader_cuda_forward_preserves_quantized_result(self) -> None:
        source = _make_convrot(device="cuda")
        layer = nn.Linear(
            64,
            8,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda",
        )
        layer.weight = nn.Parameter(source, requires_grad=False)
        inputs = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
        expected = layer(inputs)
        offloader = ModelOffloader.from_module(layer)

        try:
            with activated_model(offloader, "cuda") as active:
                actual = active(inputs)
                assert isinstance(active.weight.data, _convrot_cls())
                assert active.weight.data.qdata.is_cuda
                torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected)
        finally:
            offloader.deactivate()

    @CUDA
    def test_streamed_merge_requantizes_on_activate(self) -> None:
        convrot_cls = _convrot_cls()

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.Linear(
                            64,
                            64,
                            bias=False,
                            dtype=torch.bfloat16,
                        ),
                        nn.Linear(
                            64,
                            64,
                            bias=False,
                            dtype=torch.bfloat16,
                        ),
                    ]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = block(x)
                return x

        model = M()
        for block in model.blocks:
            block.weight = nn.Parameter(
                convrot_cls.from_hp(
                    torch.randn(64, 64, dtype=torch.bfloat16),
                    group_size=64,
                ),
                requires_grad=False,
            )
        convrot = model.blocks[0].weight.data
        rank = 4
        a = torch.randn(rank, 64)
        b = torch.randn(64, rank)
        lora = Adapter.from_state_dict(
            state_dict={
                "blocks.0.lora_A.weight": a,
                "blocks.0.lora_B.weight": b,
            }
        )

        convrot_cuda = convrot_cls(
            convrot.qdata.cuda(),
            convrot.scale.cuda(),
            convrot.group_size,
            convrot.dtype,
        )
        expected = convrot_cuda.clone()
        expected.addmm_(
            b.cuda().to(expected.dtype),
            a.cuda().to(expected.dtype),
            alpha=0.5,
        )

        offloader = ModelOffloader.from_module(
            model,
            block_paths=["blocks"],
        )
        try:
            inputs = torch.randn(
                4,
                64,
                dtype=torch.bfloat16,
                device="cuda",
            )
            with activated_model(
                offloader,
                "cuda",
                adapters=[lora],
                adapter_strengths=[0.5],
                adapter_mode="merge",
                stochastic_rounding=False,
            ) as active:
                merged = active.blocks[0].weight.data
                assert isinstance(merged, convrot_cls)
                assert torch.equal(merged.qdata, expected.qdata)
                assert torch.equal(merged.scale, expected.scale)
                output = active(inputs)
                torch.cuda.synchronize()
            assert output.shape == (4, 64)
        finally:
            offloader.deactivate()

    @CUDA
    def test_activation_defaults_to_stochastic_merge_and_releases_lock(
        self,
    ) -> None:
        convrot_cls = _convrot_cls()

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.blocks[0](x)

        model = M()
        model.blocks[0].weight = nn.Parameter(
            convrot_cls.from_hp(
                torch.randn(64, 64, dtype=torch.bfloat16),
                group_size=64,
            ),
            requires_grad=False,
        )
        lora = Adapter.from_state_dict(
            state_dict={
                "blocks.0.lora_A.weight": torch.randn(4, 64),
                "blocks.0.lora_B.weight": torch.randn(64, 4),
            }
        )
        offloader = ModelOffloader.from_module(model, block_paths=["blocks"])

        samples: list[torch.Tensor] = []
        for _ in range(2):
            with activated_model(
                offloader,
                "cuda",
                adapters=[lora],
            ) as active:
                samples.append(
                    active.blocks[0].weight.data.qdata.cpu().clone()
                )
                output = active(
                    torch.randn(
                        2,
                        64,
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                )
                assert output.shape == (2, 64)

        assert torch.equal(samples[0], samples[1])
        assert offloader.active_device is None
        assert offloader._adapter_hook_removers == []
