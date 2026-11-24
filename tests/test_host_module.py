"""Tests for name-based host module store/instance primitives."""

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
from torch import nn

from piper_offload import (
    GgufParameter,
    ModelOffloader,
    ModelStore,
    OffloadConfig,
    TrellisParameter,
    register_adapter,
)
from piper_offload._dtensor import DTENSOR_AVAILABLE
from piper_offload.dtensor_adapter import DTensorAdapter
from piper_offload.host_buffer import HostBuffer
from piper_offload.host_module import (
    ParameterOverride,
    HostBufferTarget,
    HostModuleInstance,
    HostModuleStore,
    HostModuleTarget,
    HostParamTarget,
)
from piper_offload.host_param import HostParam
from piper_offload.tensor_adapters import is_plain_meta_placeholder, matches_layout_signature
from tests._gguf_helpers import GGUFParameter, packed_q4_0
from tests.conftest import CallbackParameterTransform, activated_model


class _FakeHostParam:
    def __init__(self, target_data: torch.Tensor) -> None:
        self.allocated = 0
        self.copied = 0
        self.is_meta = False
        self.logical_shape = tuple(target_data.shape)
        self.target_data = target_data
        # Stable CPU backing aliased by every make_cpu_param() call,
        # mirroring real adapters (whose CPU wrapper aliases shared host
        # storage). A fresh wrapper object each call, but the same storage.
        self._cpu_data = torch.empty_like(target_data)
        self.target_layout = HostParam.target_layout_for(
            nn.Parameter(target_data, requires_grad=False),
        )

    @property
    def cache_bytes(self) -> int:
        return self.target_data.numel() * self.target_data.element_size()

    def make_cpu_param(self) -> nn.Parameter:
        return nn.Parameter(self._cpu_data, requires_grad=False)

    def allocate_gpu_storage(self, device: torch.device) -> object:
        self.allocated += 1
        return {"device": device}

    def make_gpu_param(self, target_state: object) -> nn.Parameter:
        del target_state
        return nn.Parameter(self.target_data.clone(), requires_grad=False)

    def copy_to_gpu(
        self,
        target_state: object,
        *,
        non_blocking: bool = False,
    ) -> None:
        del target_state, non_blocking
        self.copied += 1

    def accepts_bind_target(self, param: nn.Parameter) -> bool:
        return (
            not param.requires_grad
            and type(param) is nn.Parameter
            and type(param.data) is torch.Tensor
            and param.layout is torch.strided
            and tuple(param.shape) == self.logical_shape
        )


class TestHostModuleStore:
    def test_maps_module_aliases_to_shared_param_backing(self) -> None:
        module = nn.Module()
        shared = nn.Linear(2, 2, bias=False)
        shared.weight.requires_grad_(False)
        module.left = shared
        module.right = shared

        store = HostModuleStore.from_module(module)

        assert set(store.params) == {"left.weight", "right.weight"}
        assert store.params["left.weight"] is store.params["right.weight"]
        assert module.left.weight is module.right.weight
        assert module.left.weight.data_ptr() == (
            store.params["left.weight"].make_cpu_param().data_ptr()
        )
        assert store.cache_bytes == store.params["left.weight"].cache_bytes

    def test_shares_distinct_params_with_same_storage(
        self,
    ) -> None:
        module = nn.Module()
        shared = torch.randn(2, 2)
        module.a = nn.Parameter(shared, requires_grad=False)
        module.b = nn.Parameter(shared, requires_grad=False)

        store = HostModuleStore.from_module(module)

        assert set(store.params) == {"a", "b"}
        assert store.params["a"] is store.params["b"]
        assert module.a is module.b
        assert module.a.data_ptr() == store.params["a"].make_cpu_param().data_ptr()

    def test_rejects_trainable_parameter_before_mutating_earlier_params(self) -> None:
        module = nn.Module()
        module.first = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        module.second = nn.Parameter(torch.randn(2, 2), requires_grad=True)
        orig_first = module.first
        orig_ptr = module.first.untyped_storage().data_ptr()

        with pytest.raises(ValueError, match="frozen"):
            HostModuleStore.from_module(module)

        assert module.first is orig_first
        assert module.first.untyped_storage().data_ptr() == orig_ptr
        assert module.second.requires_grad is True

    def test_keeps_distinct_param_backings_separate(self) -> None:
        module = nn.Module()
        module.left = nn.Linear(2, 2, bias=False)
        module.right = nn.Linear(2, 2, bias=False)
        module.left.weight.requires_grad_(False)
        module.right.weight.requires_grad_(False)

        store = HostModuleStore.from_module(module)

        assert set(store.params) == {"left.weight", "right.weight"}
        assert store.params["left.weight"] is not store.params["right.weight"]
        assert store.cache_bytes == (
            store.params["left.weight"].cache_bytes
            + store.params["right.weight"].cache_bytes
        )

    def test_maps_buffer_aliases_to_shared_backing(self) -> None:
        module = nn.Module()
        shared = torch.tensor([1.0, 2.0])
        module.register_buffer("running", shared)
        module.register_buffer("running_alias", shared)

        store = HostModuleStore.from_module(module)

        assert store.params == {}
        assert set(store.buffers) == {"running", "running_alias"}
        assert store.buffers["running"] is store.buffers["running_alias"]
        assert module.running is module.running_alias
        assert module.running.data_ptr() == store.buffers["running"].tensor.data_ptr()
        assert store.cache_bytes == store.buffers["running"].cache_bytes

    def test_can_exclude_buffers(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        module.register_buffer("running", torch.randn(2))

        store = HostModuleStore.from_module(module, include_buffer_names=set())

        assert set(store.params) == {"weight"}
        assert store.buffers == {}
        assert store.cache_bytes == store.params["weight"].cache_bytes

    @pytest.mark.parametrize("shared_storage", [False, True], ids=["distinct", "shared"])
    def test_can_include_params_by_name(self, shared_storage: bool) -> None:
        module = nn.Module()
        data = torch.randn(2, 2)
        module.keep = nn.Parameter(data, requires_grad=False)
        module.skip = nn.Parameter(
            data if shared_storage else data.clone(), requires_grad=False
        )
        skipped_param = module.skip

        store = HostModuleStore.from_module(
            module,
            include_param_names={"keep"},
        )

        assert set(store.params) == {"keep"}
        assert module.keep.data_ptr() == store.params["keep"].make_cpu_param().data_ptr()
        assert module.skip is skipped_param

    def test_empty_include_name_sets_pin_nothing(self) -> None:
        module = nn.Module()
        param = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        buffer = torch.randn(2)
        module.weight = param
        module.register_buffer("running", buffer)

        store = HostModuleStore.from_module(
            module,
            include_param_names=set(),
            include_buffer_names=set(),
        )

        assert store.params == {}
        assert store.buffers == {}
        assert module.weight is param
        assert module.running is buffer

    def test_rejects_unknown_param_include_names(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)

        with pytest.raises(ValueError, match="unknown names: 'missing'"):
            HostModuleStore.from_module(
                module,
                include_param_names={"missing"},
            )

    @pytest.mark.parametrize("shared_storage", [False, True], ids=["distinct", "shared"])
    def test_can_include_buffers_by_name(self, shared_storage: bool) -> None:
        module = nn.Module()
        data = torch.randn(2)
        skipped_buffer = data if shared_storage else data.clone()
        module.register_buffer("keep", data)
        module.register_buffer("skip", skipped_buffer)

        store = HostModuleStore.from_module(
            module,
            include_buffer_names={"keep"},
        )

        assert store.params == {}
        assert set(store.buffers) == {"keep"}
        assert module.keep is store.buffers["keep"].tensor
        assert module.skip is skipped_buffer

    def test_rejects_unknown_buffer_include_names(self) -> None:
        module = nn.Module()
        module.register_buffer("running", torch.randn(2))

        with pytest.raises(ValueError, match="unknown names: 'missing'"):
            HostModuleStore.from_module(
                module,
                include_buffer_names={"missing"},
            )


class TestHostModuleInstance:
    def test_allocate_target_dedupes_alias_backings(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_empty_like = torch.empty_like

        def fake_empty_like(
            tensor: torch.Tensor,
            **kwargs: object,
        ) -> torch.Tensor:
            device = kwargs.get("device")
            if device is not None and torch.device(device).type == "cuda":
                kwargs["device"] = "cpu"
            return original_empty_like(tensor, **kwargs)

        monkeypatch.setattr(torch, "empty_like", fake_empty_like)
        host_param = _FakeHostParam(torch.empty(2, 2))
        host_buffer = HostBuffer.capture(torch.randn(2))
        store = HostModuleStore(
            params={
                "left.weight": cast(HostParam, host_param),
                "right.weight": cast(HostParam, host_param),
            },
            buffers={
                "running": host_buffer,
                "running_alias": host_buffer,
            },
        )
        instance = HostModuleInstance(
            module=nn.Module(),
            params=store.params,
            buffers=store.buffers,
        )

        target = instance.resolve_load_plan().allocate_target(torch.device("cuda"))

        assert host_param.allocated == 1
        assert target.param_targets["left.weight"] is target.param_targets["right.weight"]
        assert target.buffer_targets["running"] is target.buffer_targets["running_alias"]

    def test_allocate_target_rejects_non_cuda_device(self) -> None:
        instance = HostModuleInstance(
            module=nn.Module(),
            params={},
            buffers={},
        )

        with pytest.raises(ValueError, match="requires a CUDA device"):
            instance.resolve_load_plan().allocate_target(torch.device("cpu"))

    def test_parameter_override_resolves_meta_replacement_source(self) -> None:
        base = HostParam(
            nn.Parameter(torch.empty(2, device="meta"), requires_grad=False),
        )
        source = HostParam(
            nn.Parameter(torch.ones(2), requires_grad=False),
        )
        instance = HostModuleInstance(
            module=nn.Module(),
            params={"weight": base},
            buffers={},
        )

        assert instance.resolve_load_plan().loads == {}
        with pytest.raises(ValueError, match="physical replacement source"):
            instance.resolve_load_plan(
                {
                    "weight": ParameterOverride(
                        update=CallbackParameterTransform(lambda _param: None)
                    )
                }
            )

        plan = instance.resolve_load_plan(
            {"weight": ParameterOverride(source=source)}
        )
        assert plan.loads["weight"].source is source
        assert plan.loads["weight"].update is None

    def test_load_to_target_copies_and_updates_once_for_aliases(self) -> None:
        module = nn.Module()
        shared = nn.Linear(2, 2, bias=False)
        shared.weight.requires_grad_(False)
        module.left = shared
        module.right = shared
        host = _FakeHostParam(torch.ones(2, 2))
        store = HostModuleStore(
            params={
                "left.weight": cast(HostParam, host),
                "right.weight": cast(HostParam, host),
            },
            buffers={},
        )
        instance = HostModuleInstance(
            module=module,
            params=store.params,
            buffers=store.buffers,
        )
        update_calls: list[nn.Parameter] = []

        def update(param: nn.Parameter) -> None:
            update_calls.append(param)
            param.data.add_(1)

        transform = CallbackParameterTransform(update)
        plan = instance.resolve_load_plan(
            {"left.weight": ParameterOverride(update=transform)}
        )
        target = plan.allocate_target(torch.device("cuda"))
        plan.load_to_target(target)

        target_param = target.param_targets["left.weight"].param
        assert host.copied == 1
        assert update_calls == [target_param]
        assert module.left.weight is target_param
        assert module.right.weight is target_param
        torch.testing.assert_close(target_param, torch.full((2, 2), 2.0))

    def test_copy_to_target_does_not_mutate_module_registries(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.zeros(2), requires_grad=False)
        original = module.weight
        host = _FakeHostParam(torch.ones(2))
        instance = HostModuleInstance(
            module=module,
            params={"weight": cast(HostParam, host)},
            buffers={},
        )
        update_calls: list[nn.Parameter] = []
        plan = instance.resolve_load_plan(
            {
                "weight": ParameterOverride(
                    update=CallbackParameterTransform(update_calls.append)
                )
            }
        )
        target = plan.allocate_target(torch.device("cuda"))

        plan.copy_to_target(target)

        assert module.weight is original
        assert host.copied == 1
        assert update_calls == [target.param_targets["weight"].param]

    def test_base_load_plan_is_immutable_and_loads_host_values(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.zeros(2), requires_grad=False)
        host = _FakeHostParam(torch.ones(2))
        store = HostModuleStore(
            params={"weight": cast(HostParam, host)},
            buffers={},
        )
        instance = HostModuleInstance(
            module=module,
            params=store.params,
            buffers=store.buffers,
        )
        plan = instance.resolve_load_plan()
        with pytest.raises(AttributeError):
            plan.instance = instance  # type: ignore[misc]
        with pytest.raises(TypeError):
            plan.loads["weight"] = object()  # type: ignore[index, assignment]

        target = plan.allocate_target(torch.device("cuda"))
        plan.load_to_target(target)

        target_param = target.param_targets["weight"].param
        assert plan.loads["weight"].source is host
        assert plan.loads["weight"].update is None
        assert module.weight is target_param
        torch.testing.assert_close(target_param, torch.ones(2))

    def test_load_to_target_copies_buffers_and_preserves_persistence(self) -> None:
        prototype = nn.Module()
        shared = torch.tensor([1.0, 2.0])
        prototype.register_buffer("running", shared)
        prototype.register_buffer("running_alias", shared)
        store = HostModuleStore.from_module(prototype)
        module = nn.Module()
        module.register_buffer("running", torch.zeros(2), persistent=False)
        module.register_buffer("running_alias", module.running, persistent=False)
        instance = store.bind(module)
        assert module.running is store.buffers["running"].tensor
        assert module.running_alias is module.running
        assert "running" in module._non_persistent_buffers_set
        assert "running_alias" in module._non_persistent_buffers_set

        target_tensor = torch.empty_like(store.buffers["running"].tensor)
        buffer_target = HostBufferTarget(target_tensor)
        target = HostModuleTarget(
            param_targets={},
            buffer_targets={
                "running": buffer_target,
                "running_alias": buffer_target,
            },
        )

        instance.resolve_load_plan().load_to_target(target)

        torch.testing.assert_close(target_tensor, store.buffers["running"].tensor)
        assert module.running is target_tensor
        assert module.running_alias is target_tensor
        assert "running" in module._non_persistent_buffers_set
        assert "running_alias" in module._non_persistent_buffers_set

    def test_load_to_target_rejects_unknown_param_targets_before_copying(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.zeros(2), requires_grad=False)
        original = module.weight
        host = _FakeHostParam(torch.ones(2))
        store = HostModuleStore(
            params={"weight": cast(HostParam, host)},
            buffers={},
        )
        instance = HostModuleInstance(
            module=module,
            params=store.params,
            buffers=store.buffers,
        )
        target = HostModuleTarget(
            param_targets={
                "extra": HostParamTarget(
                    object(),
                    nn.Parameter(torch.empty(2)),
                    cast(HostParam, host),
                ),
            },
            buffer_targets={},
        )

        with pytest.raises(ValueError, match="entries outside the store.*'extra'"):
            instance.resolve_load_plan().load_to_target(target)

        assert host.copied == 0
        assert module.weight is original

    def test_load_to_target_rejects_unknown_buffer_targets_before_copying(self) -> None:
        module = nn.Module()
        module.register_buffer("running", torch.zeros(2))
        original = module.running
        host = HostBuffer.capture(torch.ones(2))
        store = HostModuleStore(
            params={},
            buffers={"running": host},
        )
        instance = HostModuleInstance(
            module=module,
            params=store.params,
            buffers=store.buffers,
        )
        target = HostModuleTarget(
            param_targets={},
            buffer_targets={
                "running": HostBufferTarget(torch.empty_like(host.tensor)),
                "extra": HostBufferTarget(torch.empty(2)),
            },
        )

        with pytest.raises(ValueError, match="entries outside the store.*'extra'"):
            instance.resolve_load_plan().load_to_target(target)

        assert module.running is original
        assert "extra" not in module._buffers

    def test_bind_rejects_trainable_managed_param_without_mutating_earlier_params(self) -> None:
        proto = nn.Module()
        proto.first = nn.Parameter(torch.ones(2), requires_grad=False)
        proto.second = nn.Parameter(torch.zeros(2), requires_grad=False)
        store = HostModuleStore.from_module(proto)

        target = nn.Module()
        target.first = nn.Parameter(torch.full((2,), 5.0), requires_grad=False)
        target.second = nn.Parameter(torch.zeros(2), requires_grad=True)
        orig_first = target.first
        orig_ptr = target.first.untyped_storage().data_ptr()

        with pytest.raises(ValueError, match="must be frozen"):
            store.bind(target)

        assert target.first is orig_first
        assert target.first.untyped_storage().data_ptr() == orig_ptr

    def test_bind_ignores_extra_unmanaged_trainable_params(self) -> None:
        proto = nn.Module()
        proto.weight = nn.Parameter(torch.ones(2), requires_grad=False)
        store = HostModuleStore.from_module(proto)

        target = nn.Module()
        target.weight = nn.Parameter(torch.zeros(2), requires_grad=False)
        target.extra_lora = nn.Parameter(torch.full((2,), 3.0), requires_grad=True)

        store.bind(target)

        torch.testing.assert_close(target.weight, torch.ones(2))
        assert target.extra_lora.requires_grad is True
        torch.testing.assert_close(target.extra_lora, torch.full((2,), 3.0))

    def test_binds_same_store_to_multiple_modules(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        prototype.register_buffer("running", torch.randn(2))
        store = HostModuleStore.from_module(prototype)

        instance = store.bind(prototype)
        assert not hasattr(instance, "parent")
        assert not hasattr(instance, "leaf")
        assert not hasattr(instance, "store")

        second_module = nn.Module()
        second_module.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        second_module.register_buffer("running", torch.randn(2))
        store.bind(second_module)

        host = store.params["weight"]
        # Each bind materializes its own CPU wrapper (deduped per host
        # within a bind), installed onto the bound module. The instance no
        # longer caches them, so read the installed wrappers off the modules.
        first_cpu = prototype.weight
        second_cpu = second_module.weight

        assert first_cpu is not second_cpu
        assert first_cpu.data_ptr() == host.make_cpu_param().data_ptr()
        assert second_cpu.data_ptr() == host.make_cpu_param().data_ptr()
        assert prototype.running is store.buffers["running"].tensor
        assert second_module.running is store.buffers["running"].tensor

    def test_restores_tied_params_with_one_cpu_wrapper(self) -> None:
        prototype = nn.Module()
        shared = nn.Linear(2, 2, bias=False)
        shared.weight.requires_grad_(False)
        prototype.left = shared
        prototype.right = shared
        store = HostModuleStore.from_module(prototype)

        target = nn.Module()
        target_shared = nn.Linear(2, 2, bias=False)
        target_shared.weight.requires_grad_(False)
        target.left = target_shared
        target.right = target_shared
        store.bind(target)

        host = store.params["left.weight"]
        # Tied names share one materialized CPU wrapper, installed under both
        # leaves. Read it off the module (the instance no longer caches it).
        cpu_param = target.left.weight

        assert store.params["left.weight"] is store.params["right.weight"]
        assert target.left.weight is cpu_param
        assert target.right.weight is cpu_param
        assert cpu_param.data_ptr() == host.make_cpu_param().data_ptr()

    def test_non_contiguous_buffer_can_bind_after_store_restore(self) -> None:
        module = nn.Module()
        source = torch.randn(2, 3).t()
        module.register_buffer("table", source)

        store = HostModuleStore.from_module(module)
        store.bind(module)

        assert not source.is_contiguous()
        assert store.buffers["table"].tensor.is_contiguous()
        assert module.table is store.buffers["table"].tensor

    def test_rejects_missing_param_name(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = HostModuleStore.from_module(prototype)

        with pytest.raises(ValueError, match="missing host names.*weight"):
            store.bind(nn.Module())

    def test_rejects_missing_buffer_name(self) -> None:
        prototype = nn.Module()
        prototype.register_buffer("running", torch.randn(2))
        store = HostModuleStore.from_module(prototype)

        with pytest.raises(ValueError, match="missing host names.*running"):
            store.bind(nn.Module())

    def test_rejects_trainable_bind_target(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = HostModuleStore.from_module(prototype)

        target = nn.Module()
        target.weight = nn.Parameter(torch.randn(2, 2), requires_grad=True)

        with pytest.raises(ValueError, match="must be frozen"):
            store.bind(target)

    def test_rejects_param_layout_mismatch(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = HostModuleStore.from_module(prototype)

        target = nn.Module()
        target.weight = nn.Parameter(torch.randn(3, 2), requires_grad=False)

        with pytest.raises(ValueError, match="Param 'weight' layout mismatch"):
            store.bind(target)

    def test_rejects_meta_param_stride_mismatch(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(
            torch.empty_strided((2, 3), (1, 2), device="meta"),
            requires_grad=False,
        )
        store = HostModuleStore.from_module(prototype)

        target = nn.Module()
        target.weight = nn.Parameter(
            torch.empty(2, 3, device="meta"),
            requires_grad=False,
        )

        with pytest.raises(ValueError, match="meta layout mismatch.*stride"):
            store.bind(target)

        assert target.weight.stride() == (3, 1)

    def test_bind_accepts_matching_meta_param_stride(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(
            torch.empty_strided((2, 3), (1, 2), device="meta"),
            requires_grad=False,
        )
        store = HostModuleStore.from_module(prototype)

        target = nn.Module()
        target.weight = nn.Parameter(
            torch.empty_strided((2, 3), (1, 2), device="meta"),
            requires_grad=False,
        )
        store.bind(target)

        assert target.weight.is_meta
        assert target.weight.stride() == (1, 2)

    def test_rejects_meta_param_storage_offset_mismatch(self) -> None:
        prototype_backing = torch.empty(12, device="meta")
        prototype = nn.Module()
        prototype.weight = nn.Parameter(
            prototype_backing.as_strided((2, 3), (3, 1), 1),
            requires_grad=False,
        )
        store = HostModuleStore.from_module(prototype)

        target_backing = torch.empty(12, device="meta")
        target = nn.Module()
        target.weight = nn.Parameter(
            target_backing.as_strided((2, 3), (3, 1), 2),
            requires_grad=False,
        )

        with pytest.raises(ValueError, match="meta layout mismatch.*storage_offset"):
            store.bind(target)

        assert target.weight.storage_offset() == 2

    def test_rejects_sparse_meta_bind_target(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(
            torch.empty(2, 2, device="meta"),
            requires_grad=False,
        )
        store = HostModuleStore.from_module(prototype)

        target = nn.Module()
        target.weight = nn.Parameter(
            torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.int64, device="meta"),
                torch.empty(0, device="meta"),
                (2, 2),
                device="meta",
                check_invariants=False,
            ),
            requires_grad=False,
        )

        with pytest.raises(ValueError, match="plain strided meta tensor"):
            store.bind(target)

    def test_rejects_buffer_layout_mismatch(self) -> None:
        prototype = nn.Module()
        prototype.register_buffer("running", torch.randn(2))
        store = HostModuleStore.from_module(prototype)

        target = nn.Module()
        target.register_buffer("running", torch.randn(3))

        with pytest.raises(ValueError, match="Buffer 'running' layout mismatch"):
            store.bind(target)

    def test_bind_allows_param_dtype_mismatch(self) -> None:
        # A config-built skeleton's placeholder dtype (e.g. fp32 default)
        # need not match the store: binding replaces the placeholder with
        # store-backed storage, so the store's dtype wins.
        prototype = nn.Module()
        prototype.weight = nn.Parameter(
            torch.randn(2, 2, dtype=torch.bfloat16), requires_grad=False
        )
        store = HostModuleStore.from_module(prototype)

        target = nn.Module()
        target.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store.bind(target)

        assert target.weight.dtype == torch.bfloat16

    def test_bind_allows_meta_skeleton_dtype_mismatch(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(
            torch.randn(2, 2, dtype=torch.bfloat16), requires_grad=False
        )
        store = HostModuleStore.from_module(prototype)

        with torch.device("meta"):
            skeleton = nn.Module()
            skeleton.weight = nn.Parameter(torch.empty(2, 2), requires_grad=False)
        store.bind(skeleton)

        assert skeleton.weight.dtype == torch.bfloat16
        assert skeleton.weight.device.type == "cpu"

    def test_bind_allows_buffer_dtype_mismatch(self) -> None:
        prototype = nn.Module()
        prototype.register_buffer("running", torch.randn(2, dtype=torch.bfloat16))
        store = HostModuleStore.from_module(prototype)

        target = nn.Module()
        target.register_buffer("running", torch.randn(2))
        store.bind(target)

        assert target.running.dtype == torch.bfloat16

    def test_bind_allows_param_sharing_mismatch(self) -> None:
        prototype = nn.Module()
        shared = nn.Linear(2, 2, bias=False)
        shared.weight.requires_grad_(False)
        prototype.left = shared
        prototype.right = shared
        store = HostModuleStore.from_module(prototype)

        target = nn.Module()
        target.left = nn.Linear(2, 2, bias=False)
        target.right = nn.Linear(2, 2, bias=False)
        target.left.weight.requires_grad_(False)
        target.right.weight.requires_grad_(False)

        store.bind(target)

        assert target.left.weight is target.right.weight

    def test_bind_allows_untracked_param_sharing(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = HostModuleStore.from_module(prototype)

        target = nn.Module()
        shared = torch.randn(2, 2)
        target.weight = nn.Parameter(shared, requires_grad=False)
        target.extra = nn.Parameter(shared, requires_grad=False)
        extra = target.extra

        store.bind(target)

        assert target.extra is extra
        assert target.weight is not target.extra

    def test_bind_allows_buffer_sharing_mismatch(self) -> None:
        prototype = nn.Module()
        shared = torch.randn(2)
        prototype.register_buffer("running", shared)
        prototype.register_buffer("running_alias", shared)
        store = HostModuleStore.from_module(prototype)

        target = nn.Module()
        target.register_buffer("running", torch.randn(2))
        target.register_buffer("running_alias", torch.randn(2))

        store.bind(target)

        assert target.running is target.running_alias


class _PackedWrapper(torch.Tensor):
    _packed: torch.Tensor
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls: type[_PackedWrapper],
        packed: torch.Tensor,
        *,
        requires_grad: bool = False,
    ) -> _PackedWrapper:
        logical = (int(packed.shape[0]), int(packed.shape[1]) // 2)
        tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            logical,
            dtype=torch.bfloat16,
            device=packed.device,
            requires_grad=requires_grad,
        )
        tensor._packed = packed
        return tensor

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        del types
        aten = torch.ops.aten
        src = cast(_PackedWrapper, args[0])
        if func in (aten.detach.default, aten.alias.default, aten.clone.default):
            out = src._packed if func is aten.alias.default else func(src._packed, **(kwargs or {}))
            return cls(out, requires_grad=src.requires_grad)
        if func in (aten.view.default, aten._unsafe_view.default) and tuple(args[1]) == tuple(src.shape):
            return cls(src._packed, requires_grad=src.requires_grad)
        raise NotImplementedError(func)


class _PackedStrictTensor(_PackedWrapper):
    pass


class _PackedMetaBindTensor(_PackedWrapper):
    pass


class _CollidingStrictTensor(_PackedWrapper):
    pass


class _UnregisteredSubTensor(_PackedWrapper):
    pass


class _MetaParamSubclass(nn.Parameter):
    pass


class _PackedStrictAdapter:
    _tensor_cls: type[_PackedWrapper] = _PackedStrictTensor
    _tag: str = "packed-strict"

    @classmethod
    def matches(cls, t: torch.Tensor) -> bool:
        return type(t) is cls._tensor_cls

    @classmethod
    def tensor_id(cls, t: torch.Tensor) -> tuple[object, ...]:
        packed = cast(_PackedWrapper, t)._packed
        return (cls._tag, packed.data_ptr(), tuple(packed.shape), packed.dtype)

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple[object, ...]:
        packed = cast(_PackedWrapper, t)._packed
        return (tuple(packed.shape), packed.dtype)

    @staticmethod
    def capture_host(t: torch.Tensor) -> torch.Tensor:
        return cast(_PackedWrapper, t)._packed.detach().clone()

    @staticmethod
    def storage_tensors(state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (state,)

    @classmethod
    def cpu_param(cls, state: torch.Tensor) -> nn.Parameter:
        return nn.Parameter(cls._tensor_cls(state, requires_grad=False), requires_grad=False)

    @staticmethod
    def alloc_gpu(state: torch.Tensor, device: torch.device) -> torch.Tensor:
        return torch.empty_like(state, device=device)

    @classmethod
    def gpu_param(cls, host: torch.Tensor, gpu_state: torch.Tensor) -> nn.Parameter:
        del host
        return nn.Parameter(cls._tensor_cls(gpu_state, requires_grad=False), requires_grad=False)

    @staticmethod
    def copy_to_gpu(src: torch.Tensor, dst: torch.Tensor, *, non_blocking: bool = False) -> None:
        dst.copy_(src, non_blocking=non_blocking)

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        return t.dtype

    @staticmethod
    def cache_bytes(state: torch.Tensor) -> int:
        return state.nbytes

    @staticmethod
    def logical_shape(t: torch.Tensor) -> tuple[int, ...]:
        return tuple(t.shape)


class _PackedMetaBindAdapter(_PackedStrictAdapter):
    _tensor_cls = _PackedMetaBindTensor
    _tag = "packed-meta-bind"

    @classmethod
    def accepts_bind_target(
        cls,
        state: torch.Tensor,
        target: torch.Tensor,
        *,
        layout_signature: object,
    ) -> bool:
        logical_shape = (int(state.shape[0]), int(state.shape[1]) // 2)
        return (
            is_plain_meta_placeholder(target, logical_shape)
            or matches_layout_signature(cls, layout_signature, target)
        )


class _CollidingStrictAdapter(_PackedStrictAdapter):
    _tensor_cls = _CollidingStrictTensor
    _tag = "colliding-strict"

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple[object, ...]:
        # Intentionally mirrors RegularAdapter.layout_signature(t) to verify
        # DTensorAdapter tags its inner adapter type and cannot collide.
        return (tuple(t.shape), t.dtype, t.layout)


def _wrap_packed(cls: type[_PackedWrapper], data: torch.Tensor, *, requires_grad: bool = False) -> torch.Tensor:
    return cls(data, requires_grad=requires_grad)


def _make_sparse_meta(shape: tuple[int, ...], *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.empty((len(shape), 0), dtype=torch.int64, device="meta"),
        torch.empty(0, dtype=dtype, device="meta"),
        shape,
        dtype=dtype,
        device="meta",
        check_invariants=False,
    )


def _make_trellis_weight(seed: int = 0) -> TrellisParameter:
    generator = torch.Generator().manual_seed(seed)
    rows, blocks, num_bytes = 64, 1, 143
    bitstreams = torch.randint(0, 256, (rows, blocks, num_bytes), dtype=torch.uint8, generator=generator)
    codes = torch.randint(0x20, 0x40, (rows, blocks), dtype=torch.uint8, generator=generator)
    scale = torch.rand(rows, 1, generator=generator) * 1e-3 + 1e-4
    return TrellisParameter(bitstreams, codes.view(torch.float8_e4m3fn), scale, block_size=256, group_size=256)


class TestQuantizedMetaPlaceholderBind:
    def test_binds_opt_in_packed_adapter_onto_plain_meta_skeleton_rebinds_and_restores_ties(self) -> None:
        remove = register_adapter(_PackedMetaBindAdapter)
        try:
            packed = torch.arange(32, dtype=torch.int8).reshape(4, 8)
            proto = nn.Module()
            proto.left = nn.Module()
            proto.right = nn.Module()
            shared_weight = nn.Parameter(_wrap_packed(_PackedMetaBindTensor, packed), requires_grad=False)
            proto.left.weight = shared_weight
            proto.right.weight = shared_weight
            module_store = HostModuleStore.from_module(proto)
            model_store = ModelStore.from_module(proto)

            def make_untied_skeleton() -> nn.Module:
                sk = nn.Module()
                sk.left = nn.Module()
                sk.right = nn.Module()
                sk.left.weight = nn.Parameter(
                    torch.empty((4, 4), dtype=torch.float32, device="meta"), requires_grad=False
                )
                sk.right.weight = nn.Parameter(
                    torch.empty((4, 4), dtype=torch.float32, device="meta"), requires_grad=False
                )
                return sk

            sk1 = make_untied_skeleton()
            module_store.bind(sk1)
            assert sk1.left.weight is sk1.right.weight
            assert type(sk1.left.weight.data) is _PackedMetaBindTensor
            assert torch.equal(cast(_PackedWrapper, sk1.left.weight.data)._packed, packed)
            first_ptr = cast(_PackedWrapper, sk1.left.weight.data)._packed.data_ptr()

            # Re-binding after initial bind succeeds and shares the store's backing tensor.
            module_store.bind(sk1)
            assert sk1.left.weight is sk1.right.weight
            assert cast(_PackedWrapper, sk1.left.weight.data)._packed.data_ptr() == first_ptr

            sk2 = make_untied_skeleton()
            model_store.bind(sk2)
            assert sk2.left.weight is sk2.right.weight
            assert type(sk2.left.weight.data) is _PackedMetaBindTensor
            assert torch.equal(cast(_PackedWrapper, sk2.left.weight.data)._packed, packed)
        finally:
            remove()

    def test_rejects_strict_logical_shape_adapter_without_bind_opt_in(self) -> None:
        remove = register_adapter(_PackedStrictAdapter)
        try:
            packed = torch.arange(32, dtype=torch.int8).reshape(4, 8)
            proto = nn.Module()
            proto.weight = nn.Parameter(_wrap_packed(_PackedStrictTensor, packed), requires_grad=False)
            store = HostModuleStore.from_module(proto)

            for bad_shape in ((4, 4), (4, 8)):
                sk = nn.Module()
                sk.weight = nn.Parameter(torch.empty(bad_shape, device="meta"), requires_grad=False)
                with pytest.raises(ValueError, match="layout mismatch"):
                    store.bind(sk)

            exact = nn.Module()
            exact.weight = nn.Parameter(
                _wrap_packed(_PackedStrictTensor, torch.zeros((4, 8), dtype=torch.int8)),
                requires_grad=False,
            )
            store.bind(exact)
            assert torch.equal(cast(_PackedWrapper, exact.weight.data)._packed, packed)
        finally:
            remove()

    def test_rejects_mismatched_non_meta_subclass_trainable_and_partial_targets(self) -> None:
        remove = register_adapter(_PackedMetaBindAdapter)
        try:
            packed = torch.arange(32, dtype=torch.int8).reshape(4, 8)
            proto = nn.Module()
            proto.weight = nn.Parameter(_wrap_packed(_PackedMetaBindTensor, packed), requires_grad=False)
            store = HostModuleStore.from_module(proto)
            frozen_host = store.params["weight"]

            wrong_shape = nn.Module()
            wrong_shape.weight = nn.Parameter(torch.empty((8, 4), device="meta"), requires_grad=False)
            with pytest.raises(ValueError, match="layout mismatch"):
                store.bind(wrong_shape)

            sparse_meta = nn.Module()
            sparse_meta.weight = nn.Parameter(_make_sparse_meta((4, 4)), requires_grad=False)
            with pytest.raises(ValueError, match="layout mismatch"):
                store.bind(sparse_meta)

            materialized_plain = nn.Module()
            materialized_plain.weight = nn.Parameter(torch.zeros((4, 4), dtype=torch.bfloat16), requires_grad=False)
            with pytest.raises(ValueError, match="layout mismatch"):
                store.bind(materialized_plain)

            subclass_meta = nn.Module()
            subclass_meta.weight = _MetaParamSubclass(torch.empty((4, 4), device="meta"), requires_grad=False)
            with pytest.raises(ValueError, match="layout mismatch"):
                store.bind(subclass_meta)

            trainable_meta = nn.Module()
            trainable_meta.weight = nn.Parameter(torch.empty((4, 4), device="meta"), requires_grad=True)
            assert not frozen_host.accepts_bind_target(trainable_meta.weight)
            with pytest.raises(ValueError, match="frozen"):
                store.bind(trainable_meta)

            trainable_packed = nn.Module()
            trainable_packed.weight = nn.Parameter(
                _wrap_packed(_PackedMetaBindTensor, packed, requires_grad=True),
                requires_grad=True,
            )
            with pytest.raises(ValueError, match="requires_grad=False"):
                HostParam(trainable_packed.weight)
            with pytest.raises(ValueError, match="frozen"):
                store.bind(trainable_packed)

            # Multi-parameter rejection must not partially mutate earlier valid params.
            pair_proto = nn.Module()
            pair_proto.a = nn.Parameter(_wrap_packed(_PackedMetaBindTensor, packed), requires_grad=False)
            pair_proto.b = nn.Parameter(_wrap_packed(_PackedMetaBindTensor, packed), requires_grad=False)
            pair_store = HostModuleStore.from_module(pair_proto)

            pair_target = nn.Module()
            pair_target.a = nn.Parameter(torch.empty((4, 4), device="meta"), requires_grad=False)
            pair_target.b = nn.Parameter(torch.empty((8, 4), device="meta"), requires_grad=False)
            with pytest.raises(ValueError, match="Param 'b' layout mismatch"):
                pair_store.bind(pair_target)
            assert pair_target.a.is_meta
        finally:
            remove()

    def test_regular_store_rejects_sparse_meta_target(self) -> None:
        proto = nn.Module()
        proto.weight = nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16), requires_grad=False)
        store = HostModuleStore.from_module(proto)

        for sparse_dtype in (torch.bfloat16, torch.float32):
            sparse = nn.Module()
            sparse.weight = nn.Parameter(
                _make_sparse_meta((4, 4), dtype=sparse_dtype),
                requires_grad=False,
            )
            with pytest.raises(ValueError, match="layout mismatch"):
                store.bind(sparse)

        dense_meta = nn.Module()
        dense_meta.weight = nn.Parameter(torch.empty((4, 4), dtype=torch.float32, device="meta"), requires_grad=False)
        store.bind(dense_meta)
        assert dense_meta.weight.dtype == torch.bfloat16
        assert torch.equal(dense_meta.weight.data, proto.weight.data)

    def test_binds_gguf_and_trellis_onto_plain_meta_skeleton(self) -> None:
        packed = packed_q4_0(8, 64)
        for cls in (GgufParameter, GGUFParameter):
            gguf_proto = nn.Module()
            gguf_proto.weight = cls(packed, requires_grad=False, quant_type=2)
            gguf_store = HostModuleStore.from_module(gguf_proto)
            gguf_skeleton = nn.Linear(64, 8, bias=False, dtype=torch.float32, device="meta")
            gguf_skeleton.weight.requires_grad_(False)
            gguf_store.bind(gguf_skeleton)
            assert isinstance(gguf_skeleton.weight, cls)
            assert torch.equal(gguf_skeleton.weight.as_tensor().detach(), packed.view(torch.uint8))

        trellis = _make_trellis_weight()
        trellis_proto = nn.Module()
        trellis_proto.weight = trellis
        trellis_store = HostModuleStore.from_module(trellis_proto)
        trellis_skeleton = nn.Linear(256, 64, bias=False, dtype=torch.float32, device="meta")
        trellis_skeleton.weight.requires_grad_(False)
        trellis_store.bind(trellis_skeleton)
        assert isinstance(trellis_skeleton.weight, TrellisParameter)
        assert torch.equal(trellis_skeleton.weight.bitstreams, trellis.bitstreams)
        assert torch.equal(
            trellis_skeleton.weight.block_scales.view(torch.uint8),
            trellis.block_scales.view(torch.uint8),
        )
        assert torch.equal(trellis_skeleton.weight.scale, trellis.scale)

    def test_meta_host_param_accepts_bind_target_checks_stride_offset_dtype_and_layout(self) -> None:
        base = torch.empty(12, dtype=torch.bfloat16, device="meta")
        meta_param = nn.Parameter(base.as_strided((2, 3), (3, 1), 1), requires_grad=False)
        host = HostParam(meta_param)

        def strided_meta(
            stride: tuple[int, int] = (3, 1),
            offset: int = 1,
            *,
            dtype: torch.dtype = torch.bfloat16,
            requires_grad: bool = False,
        ) -> nn.Parameter:
            t = torch.empty(12, dtype=dtype, device="meta").as_strided((2, 3), stride, offset)
            return nn.Parameter(t, requires_grad=requires_grad)

        matching = strided_meta()
        assert host.accepts_bind_target(matching)

        non_matching = (
            strided_meta(stride=(1, 2)),
            strided_meta(offset=2),
            strided_meta(dtype=torch.float32),
            nn.Parameter(_make_sparse_meta((2, 3), dtype=torch.bfloat16), requires_grad=False),
            nn.Parameter(torch.zeros((2, 3), dtype=torch.bfloat16), requires_grad=False),
            strided_meta(requires_grad=True),
        )
        for bad in non_matching:
            assert not host.accepts_bind_target(bad)

    def test_rejects_meta_gguf_and_trellis_wrappers_cleanly(self) -> None:
        packed = packed_q4_0(8, 64)
        for cls in (GgufParameter, GGUFParameter):
            gguf_proto = nn.Module()
            gguf_proto.weight = cls(packed, requires_grad=False, quant_type=2)
            gguf_store = HostModuleStore.from_module(gguf_proto)

            meta_mod = nn.Module()
            meta_mod.weight = cls(torch.empty_like(packed, device="meta"), requires_grad=False, quant_type=2)
            assert not gguf_store.params["weight"].accepts_bind_target(meta_mod.weight)
            with pytest.raises(ValueError, match="layout mismatch"):
                gguf_store.bind(meta_mod)

        trellis_proto = nn.Module()
        trellis_proto.weight = _make_trellis_weight()
        trellis_store = HostModuleStore.from_module(trellis_proto)

        meta_trellis_mod = nn.Module()
        meta_trellis_mod.weight = TrellisParameter(
            torch.empty((64, 1, 143), dtype=torch.uint8, device="meta"),
            torch.empty((64, 1), dtype=torch.float8_e4m3fn, device="meta"),
            torch.empty((64, 1), dtype=torch.float32, device="meta"),
            block_size=256,
            group_size=256,
        )
        assert not trellis_store.params["weight"].accepts_bind_target(meta_trellis_mod.weight)
        with pytest.raises(ValueError, match="layout mismatch"):
            trellis_store.bind(meta_trellis_mod)

    @staticmethod
    @contextmanager
    def _cpu_dtensor_mesh():
        if not dist.is_gloo_available() or not DTENSOR_AVAILABLE:
            pytest.skip("CPU Gloo and torch.distributed.tensor required")
        assert not dist.is_initialized()

        from torch.distributed.tensor import init_device_mesh

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            dist.init_process_group(
                "gloo",
                store=dist.FileStore(str(Path(tmpdir) / "store"), 1),
                rank=0,
                world_size=1,
            )
            try:
                yield init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
            finally:
                dist.destroy_process_group()

    def test_dtensor_cpu_mesh_binds_packed_adapter_and_rejects_mismatches(self) -> None:
        from torch.distributed.tensor import DTensor, Replicate, Shard, init_device_mesh

        with self._cpu_dtensor_mesh() as mesh:
            rem_meta = register_adapter(_PackedMetaBindAdapter)
            try:
                packed = torch.arange(32, dtype=torch.int8).reshape(4, 8)
                dt_packed = DTensor.from_local(
                    _wrap_packed(_PackedMetaBindTensor, packed), mesh, [Shard(0)], run_check=False
                )
                proto = nn.Module()
                proto.weight = nn.Parameter(dt_packed, requires_grad=False)
                store = HostModuleStore.from_module(proto)

                sk = nn.Module()
                sk.weight = nn.Parameter(
                    DTensor.from_local(
                        torch.empty((4, 4), dtype=torch.float32, device="meta"),
                        mesh,
                        [Shard(0)],
                        run_check=False,
                    ),
                    requires_grad=False,
                )
                store.bind(sk)
                store.bind(sk)
                assert isinstance(sk.weight.data, DTensor)
                bound_local = sk.weight.data.to_local()
                assert type(bound_local) is _PackedMetaBindTensor
                assert torch.equal(cast(_PackedWrapper, bound_local)._packed, packed)

                mesh_2d = init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp", "tp"))
                bad_targets: list[torch.Tensor] = [
                    torch.empty((4, 4), device="meta"),
                    DTensor.from_local(torch.zeros((4, 4), dtype=torch.bfloat16), mesh, [Shard(0)], run_check=False),
                    DTensor.from_local(torch.empty((4, 4), device="meta"), mesh, [Replicate()], run_check=False),
                    DTensor.from_local(torch.empty((8, 4), device="meta"), mesh, [Shard(0)], run_check=False),
                    DTensor.from_local(
                        torch.empty((2, 4), device="meta"),
                        mesh,
                        [Shard(0)],
                        shape=(4, 4),
                        stride=(4, 1),
                        run_check=False,
                    ),
                    DTensor.from_local(
                        torch.empty((4, 4), device="meta"), mesh_2d, [Replicate(), Shard(0)], run_check=False
                    ),
                ]
                for bad in bad_targets:
                    m = nn.Module()
                    m.weight = nn.Parameter(bad, requires_grad=False)
                    with pytest.raises(ValueError, match="layout mismatch"):
                        store.bind(m)

                # Non-DTensor quantized store rejects DTensor(meta).
                plain_proto = nn.Module()
                plain_proto.weight = nn.Parameter(
                    _wrap_packed(_PackedMetaBindTensor, packed), requires_grad=False
                )
                plain_store = HostModuleStore.from_module(plain_proto)
                dt_meta_only = nn.Module()
                dt_meta_only.weight = nn.Parameter(
                    DTensor.from_local(torch.empty((4, 4), device="meta"), mesh, [Shard(0)], run_check=False),
                    requires_grad=False,
                )
                with pytest.raises(ValueError, match="layout mismatch"):
                    plain_store.bind(dt_meta_only)
            finally:
                rem_meta()

    def test_dtensor_cpu_mesh_enforces_strict_and_regular_bind_policies(self) -> None:
        from torch.distributed.tensor import DTensor, Shard

        with self._cpu_dtensor_mesh() as mesh:
            rem_strict = register_adapter(_PackedStrictAdapter)
            try:
                packed = torch.arange(32, dtype=torch.int8).reshape(4, 8)
                strict_proto = nn.Module()
                strict_proto.weight = nn.Parameter(
                    DTensor.from_local(
                        _wrap_packed(_PackedStrictTensor, packed), mesh, [Shard(0)], run_check=False
                    ),
                    requires_grad=False,
                )
                strict_store = HostModuleStore.from_module(strict_proto)
                for bad_local_shape in ((4, 4), (4, 8)):
                    bad_strict = nn.Module()
                    bad_strict.weight = nn.Parameter(
                        DTensor.from_local(
                            torch.empty(bad_local_shape, device="meta"), mesh, [Shard(0)], run_check=False
                        ),
                        requires_grad=False,
                    )
                    with pytest.raises(ValueError, match="layout mismatch"):
                        strict_store.bind(bad_strict)
                good_strict = nn.Module()
                good_strict.weight = nn.Parameter(
                    DTensor.from_local(
                        _wrap_packed(_PackedStrictTensor, torch.zeros_like(packed)),
                        mesh,
                        [Shard(0)],
                        run_check=False,
                    ),
                    requires_grad=False,
                )
                strict_store.bind(good_strict)
                assert torch.equal(cast(_PackedWrapper, good_strict.weight.data.to_local())._packed, packed)

                # DTensor over RegularAdapter accepts dtype-mismatched meta.
                reg_proto = nn.Module()
                reg_weights = torch.randn(4, 4, dtype=torch.bfloat16)
                reg_proto.weight = nn.Parameter(
                    DTensor.from_local(reg_weights, mesh, [Shard(0)], run_check=False),
                    requires_grad=False,
                )
                reg_store = HostModuleStore.from_module(reg_proto)
                reg_meta = nn.Module()
                reg_meta.weight = nn.Parameter(
                    DTensor.from_local(
                        torch.empty((4, 4), dtype=torch.float32, device="meta"), mesh, [Shard(0)], run_check=False
                    ),
                    requires_grad=False,
                )
                reg_store.bind(reg_meta)
                assert torch.equal(reg_meta.weight.data.to_local(), reg_weights)

                # DTensor over RegularAdapter rejects mismatched local shard shape or subclass wrapper.
                for bad_reg_local in (
                    torch.empty((2, 4), device="meta"),
                    _wrap_packed(_PackedStrictTensor, packed),
                ):
                    kwargs: dict[str, Any] = (
                        {"shape": (4, 4), "stride": (4, 1)} if tuple(bad_reg_local.shape) != (4, 4) else {}
                    )
                    bad_reg = nn.Module()
                    bad_reg.weight = nn.Parameter(
                        DTensor.from_local(bad_reg_local, mesh, [Shard(0)], run_check=False, **kwargs),
                        requires_grad=False,
                    )
                    with pytest.raises(ValueError, match="layout mismatch"):
                        reg_store.bind(bad_reg)
            finally:
                rem_strict()

    def test_dtensor_cpu_mesh_isolates_inner_signature_and_rejects_unregistered_subclasses(self) -> None:
        from torch.distributed.tensor import DTensor, Shard

        with self._cpu_dtensor_mesh() as mesh:
            rem_colliding = register_adapter(_CollidingStrictAdapter)
            rem_meta = register_adapter(_PackedMetaBindAdapter)
            try:
                packed = torch.arange(32, dtype=torch.int8).reshape(4, 8)
                colliding_dt = DTensor.from_local(
                    _wrap_packed(_CollidingStrictTensor, packed), mesh, [Shard(0)], run_check=False
                )
                plain_bf16_dt = DTensor.from_local(
                    torch.zeros((4, 4), dtype=torch.bfloat16), mesh, [Shard(0)], run_check=False
                )
                assert DTensorAdapter.layout_signature(colliding_dt) != DTensorAdapter.layout_signature(plain_bf16_dt)

                # Strict external adapter inside DTensor must not collide with RegularAdapter even if
                # its layout_signature tuple matches RegularAdapter.
                colliding_proto = nn.Module()
                colliding_proto.weight = nn.Parameter(colliding_dt, requires_grad=False)
                colliding_store = HostModuleStore.from_module(colliding_proto)
                for bad_local in (
                    torch.empty((4, 4), dtype=torch.bfloat16, device="meta"),
                    torch.zeros((4, 4), dtype=torch.bfloat16),
                ):
                    m = nn.Module()
                    m.weight = nn.Parameter(
                        DTensor.from_local(bad_local, mesh, [Shard(0)], run_check=False),
                        requires_grad=False,
                    )
                    with pytest.raises(ValueError, match="layout mismatch"):
                        colliding_store.bind(m)

                # DTensor wrapping an unregistered subclass must return False / ValueError("layout mismatch")
                # rather than crashing with NotImplementedError from select_adapter(local).
                unreg_param = nn.Parameter(
                    DTensor.from_local(_wrap_packed(_UnregisteredSubTensor, packed), mesh, [Shard(0)], run_check=False),
                    requires_grad=False,
                )
                reg_proto = nn.Module()
                reg_proto.weight = nn.Parameter(plain_bf16_dt, requires_grad=False)
                reg_store = HostModuleStore.from_module(reg_proto)
                meta_proto = nn.Module()
                meta_proto.weight = nn.Parameter(
                    DTensor.from_local(_wrap_packed(_PackedMetaBindTensor, packed), mesh, [Shard(0)], run_check=False),
                    requires_grad=False,
                )
                meta_store = HostModuleStore.from_module(meta_proto)

                for store in (reg_store, meta_store):
                    assert not store.params["weight"].accepts_bind_target(unreg_param)
                    unreg_mod = nn.Module()
                    unreg_mod.weight = unreg_param
                    with pytest.raises(ValueError, match="layout mismatch"):
                        store.bind(unreg_mod)
            finally:
                rem_meta()
                rem_colliding()

    @staticmethod
    def _build_convrot_and_meta_nets(convrot_cls: Any):
        def make_convrot(seed: int) -> tuple[Any, torch.Tensor, torch.Tensor]:
            gen = torch.Generator().manual_seed(seed)
            q = torch.randint(-127, 128, (64, 64), dtype=torch.int8, generator=gen)
            s = torch.rand((64, 1), dtype=torch.float32, generator=gen) + 0.1
            return convrot_cls.from_quantized(q, s, group_size=64, logical_dtype=torch.bfloat16), q, s

        weights = [make_convrot(seed) for seed in (0, 1, 2)]
        proto = nn.Module()
        proto.head = nn.Linear(64, 64, bias=False, device="meta")
        proto.head.weight = nn.Parameter(weights[0][0], requires_grad=False)
        proto.blocks = nn.ModuleList([nn.Linear(64, 64, bias=False, device="meta") for _ in range(2)])
        proto.blocks[0].weight = nn.Parameter(weights[1][0], requires_grad=False)
        proto.blocks[1].weight = nn.Parameter(weights[2][0], requires_grad=False)
        proto.requires_grad_(False)

        def make_meta_skeleton() -> nn.Module:
            with torch.device("meta"):
                net = nn.Module()
                net.head = nn.Linear(64, 64, bias=False, dtype=torch.float32)
                net.blocks = nn.ModuleList([nn.Linear(64, 64, bias=False, dtype=torch.float32) for _ in range(2)])
                net.requires_grad_(False)
                return net

        return proto, weights, make_meta_skeleton

    def test_binds_convrot_model_store_onto_meta_skeleton(self) -> None:
        convrot_mod = pytest.importorskip("piper_kernels.weights.convrot.int8")
        convrot_cls = convrot_mod.ConvRotInt8Tensor
        proto, weights, make_meta_skeleton = self._build_convrot_and_meta_nets(convrot_cls)
        (_, head_q, head_s), (_, b0_q, b0_s), (_, b1_q, b1_s) = weights

        head_store = ModelStore.from_module(proto.head)
        sk_head = nn.Linear(64, 64, bias=False, dtype=torch.float32, device="meta")
        sk_head.weight.requires_grad_(False)
        head_store.bind(sk_head)
        assert isinstance(sk_head.weight.data, convrot_cls)
        assert torch.equal(sk_head.weight.data.qdata, head_q)
        assert torch.equal(sk_head.weight.data.scale, head_s)

        model_store = ModelStore.from_module(proto)
        sk1 = make_meta_skeleton()
        model_store.bind(sk1, OffloadConfig(block_paths=("blocks",)))
        for idx, (exp_q, exp_s) in enumerate(((b0_q, b0_s), (b1_q, b1_s))):
            w = sk1.blocks[idx].weight.data
            assert isinstance(w, convrot_cls)
            assert torch.equal(w.qdata, exp_q)
            assert torch.equal(w.scale, exp_s)

        sk2 = make_meta_skeleton()
        model_store.bind(sk2, OffloadConfig(block_paths=("blocks",), block_mode="streaming"))
        assert isinstance(sk2.head.weight.data, convrot_cls)
        assert torch.equal(sk2.head.weight.data.qdata, head_q)
        for idx, (exp_q, exp_s) in enumerate(((b0_q, b0_s), (b1_q, b1_s))):
            w = sk2.blocks[idx].weight.data
            assert isinstance(w, convrot_cls)
            assert torch.equal(w.qdata, exp_q)
            assert torch.equal(w.scale, exp_s)

    def test_convrot_meta_bound_offloader_forward_parity(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA/HIP device required")
        convrot_mod = pytest.importorskip("piper_kernels.weights.convrot.int8")
        proto, _, make_meta_skeleton = self._build_convrot_and_meta_nets(convrot_mod.ConvRotInt8Tensor)

        model_store = ModelStore.from_module(proto)
        skeleton = make_meta_skeleton()
        offloader = model_store.bind(
            skeleton,
            OffloadConfig(block_paths=("blocks",), block_mode="streaming"),
        )

        x = torch.randn(2, 64, dtype=torch.bfloat16, device="cuda")
        with activated_model(offloader, "cuda"):
            actual = skeleton.blocks[1](skeleton.blocks[0](skeleton.head(x)))
        proto_off = ModelOffloader.from_module(
            proto,
            config=OffloadConfig(block_paths=("blocks",), block_mode="resident"),
        )
        with activated_model(proto_off, "cuda"):
            expected = proto.blocks[1](proto.blocks[0](proto.head(x)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

