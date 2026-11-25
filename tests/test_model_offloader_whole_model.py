"""Tests for ``piper_offload.model_offloader.ModelOffloader``."""

from typing import Any, cast

import pytest
import torch
from torch import nn

from piper_offload import (
    ModelOffloader,
    ModelStore,
    ResourceBinding,
)
from piper_offload.residency_engine import ResidencyEngine
from piper_offload.host_module import HostModuleStore, ParameterOverride

from tests.conftest import CallbackParameterTransform, activated_model, resident_instance

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_simple_model() -> nn.Module:
    """Two-layer Linear, frozen, CPU."""
    m = nn.Sequential(nn.Linear(8, 16, bias=False), nn.Linear(16, 8, bias=False))
    for p in m.parameters():
        p.requires_grad = False
    return m


def _unique_host_param_count(pw: ModelOffloader) -> int:
    resident = resident_instance(pw)
    assert resident is not None
    return len({id(host) for host in resident.params.values()})


def _unique_host_buffer_count(pw: ModelOffloader) -> int:
    resident = resident_instance(pw)
    assert resident is not None
    return len({id(host) for host in resident.buffers.values()})


# ---------------------------------------------------------------------------
# ResourceBinding protocol conformance
# ---------------------------------------------------------------------------


class TestResourceBindingConformance:
    def test_isinstance_runtime_check(self) -> None:
        pw = ModelOffloader.from_module(_make_simple_model())
        try:
            assert isinstance(pw, ResourceBinding)
            assert callable(pw.activate)
            assert callable(pw.deactivate)
            assert pw.cache_bytes > 0
        finally:
            pw.deactivate()

    def test_engine_is_not_top_level_binding(self) -> None:
        model = _make_simple_model()
        offloader = ModelOffloader.from_module(model)
        try:
            assert not isinstance(offloader._engine, ResourceBinding)
        finally:
            offloader.deactivate()

    def test_offloader_constructor_requires_bound_partition(self) -> None:
        with pytest.raises(TypeError, match="partition"):
            cast(Any, ModelOffloader)(_make_simple_model())


class TestResidencyEngineResident:
    def test_bind_allows_empty_noop_engine(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        module.register_buffer("running", torch.randn(2))

        store = HostModuleStore.from_module(
            module,
            include_param_names=set(),
            include_buffer_names=set(),
        )

        assert store.cache_bytes == 0
        assert not store.params
        assert not store.buffers

        engine = ResidencyEngine(
            resident=store.bind(module),
            units=(),
            unit_paths=(),
        )
        try:
            assert engine.resident is not None
            assert not engine.resident.params
            assert not engine.resident.buffers
            engine.activate(torch.device("cpu"))
        finally:
            engine.deactivate()

    def test_cuda_use_hook_tracks_the_forward_stream(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class Lease:
            def __init__(self) -> None:
                self.used: list[object] = []

            def record_stream(self, stream: object) -> None:
                self.used.append(stream)

        model = _make_simple_model()
        engine = ResidencyEngine(
            resident=HostModuleStore.from_module(model).bind(model),
            units=(),
            unit_paths=(),
        )
        lease = Lease()
        forward_stream = object()
        engine._resident_lease = cast(Any, lease)
        engine._active_device = torch.device("cuda")
        engine._install_resident_use_hook()
        monkeypatch.setattr(
            torch.cuda,
            "current_stream",
            lambda _device: forward_stream,
        )
        try:
            model(torch.randn(1, 8))
            assert lease.used == [forward_stream]
        finally:
            engine._remove_resident_use_hook()
            engine._resident_lease = None
            engine._active_device = None
            engine.deactivate()

    def test_binds_existing_store_to_multiple_offloaders(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        prototype.register_buffer("running", torch.randn(2))
        store = ModelStore.from_module(prototype)

        first = store.bind(prototype)
        second_model = nn.Module()
        second_model.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        second_model.register_buffer("running", torch.randn(2))
        second = store.bind(second_model)

        try:
            first_resident = resident_instance(first)
            second_resident = resident_instance(second)
            assert first_resident is not None
            assert second_resident is not None
            assert store.param_names == frozenset({"weight"})
            assert first.param_names == store.param_names
            assert second.param_names == store.param_names
            assert store.buffer_names == frozenset({"running"})
            assert first.buffer_names == store.buffer_names
            assert second.buffer_names == store.buffer_names

            assert prototype.weight.data_ptr() == second_model.weight.data_ptr()
            assert prototype.weight is not second_model.weight
            assert prototype.running is second_model.running

            first.activate(torch.device("cpu"))
            second.activate(torch.device("cpu"))
        finally:
            first.deactivate()
            second.deactivate()

    def test_bind_propagates_module_mismatch_errors(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = ModelStore.from_module(prototype)

        target = nn.Module()
        target.weight = nn.Parameter(torch.randn(3, 2), requires_grad=False)

        with pytest.raises(ValueError, match="Param 'weight' layout mismatch"):
            store.bind(target)

    def test_cpu_release_is_noop(self) -> None:
        model = _make_simple_model()
        engine = ResidencyEngine(
            resident=HostModuleStore.from_module(model).bind(model),
            units=(),
            unit_paths=(),
        )
        try:
            engine.activate(torch.device("cpu"))
            engine.release()

            assert engine.active_device == torch.device("cpu")
            assert engine._resident_lease is None
            assert all(not param.is_pinned() for param in model.parameters())
        finally:
            engine.deactivate()

    def test_double_activate_requires_deactivate(self) -> None:
        model = _make_simple_model()
        engine = ResidencyEngine(
            resident=HostModuleStore.from_module(model).bind(model),
            units=(),
            unit_paths=(),
        )
        try:
            engine.activate(torch.device("cpu"))
            engine.release()
            with pytest.raises(RuntimeError, match="already active"):
                engine.activate(torch.device("cpu"))
        finally:
            engine.deactivate()

    @CUDA
    def test_cuda_working_set_can_deactivate_and_reactivate(self) -> None:
        torch.manual_seed(42)
        eager = _make_simple_model()
        model = _make_simple_model()
        model.load_state_dict(eager.state_dict())
        value = torch.randn(2, 8)
        with torch.inference_mode():
            expected = eager(value)

        engine = ResidencyEngine(
            resident=HostModuleStore.from_module(model).bind(model),
            units=(),
            unit_paths=(),
        )
        copied: list[torch.device] = []
        overrides = {
            "0.weight": ParameterOverride(
                update=CallbackParameterTransform(
                    lambda param: copied.append(param.device)
                )
            )
        }
        try:
            engine.activate(torch.device("cuda"), resident_overrides=overrides)
            assert engine._resident_lease is not None
            assert [device.type for device in copied] == ["cuda"]
            assert model._forward_pre_hooks
            assert all(param.is_cuda for param in model.parameters())

            engine.release()
            engine.release()
            assert engine.active_device is not None
            assert engine.active_device.type == "cuda"
            assert engine._resident_lease is None
            assert not model._forward_pre_hooks
            assert all(not param.is_pinned() for param in model.parameters())
            engine.deactivate()

            engine.activate(torch.device("cuda"), resident_overrides=overrides)
            assert engine._resident_lease is not None
            assert [device.type for device in copied] == ["cuda", "cuda"]
            assert model._forward_pre_hooks
            with torch.inference_mode():
                actual = model(value.cuda()).cpu()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected)
        finally:
            engine.deactivate()

        assert engine.active_device is None
        assert engine._resident_lease is None
        assert all(not param.is_pinned() for param in model.parameters())


# ---------------------------------------------------------------------------
# Frozen contract
# ---------------------------------------------------------------------------


class TestFrozenContract:
    def test_from_module_rejects_trainable_parameter_without_mutating_earlier(self) -> None:
        m = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 2, bias=False))
        m[0].weight.requires_grad_(False)
        first_storage_ptr = m[0].weight.untyped_storage().data_ptr()

        with pytest.raises(ValueError, match="requires_grad=False"):
            ModelOffloader.from_module(m)

        assert m[0].weight.untyped_storage().data_ptr() == first_storage_ptr
        assert m[1].weight.requires_grad is True


# ---------------------------------------------------------------------------
# Lifecycle: activate / deactivate
# ---------------------------------------------------------------------------


class TestLifecycle:
    @CUDA
    def test_activate_returns_model_on_gpu(self) -> None:
        m = _make_simple_model()
        pw = ModelOffloader.from_module(m)
        try:
            pw.activate("cuda")
            assert pw.model is m
            for p in m.parameters():
                assert p.is_cuda
            pw.deactivate()
            for p in m.parameters():
                assert not p.is_cuda
                assert not p.is_pinned()
        finally:
            pw.deactivate()

    def test_cpu_reactivation_preserves_model_and_host_storage(self) -> None:
        m = _make_simple_model()
        pw = ModelOffloader.from_module(m)
        value = torch.randn(2, 8)
        expected = m(value)
        # CPU activation creates fresh wrappers over the captured host storage.
        host_ptrs = [p.data_ptr() for p in m.parameters()]
        try:
            for _ in range(3):
                pw.activate(device="cpu")
                assert pw.value is m
                assert [p.data_ptr() for p in m.parameters()] == host_ptrs
                for p in m.parameters():
                    assert p.device == torch.device("cpu")
                    assert not p.is_pinned()
                torch.testing.assert_close(pw.value(value), expected, rtol=0, atol=0)
                pw.deactivate()
                assert pw.active_device is None
                assert [p.data_ptr() for p in m.parameters()] == host_ptrs
                for p in m.parameters():
                    assert not p.is_pinned()
        finally:
            pw.deactivate()

    def test_deactivate_when_not_active_is_noop(self) -> None:
        pw = ModelOffloader.from_module(_make_simple_model())
        try:
            pw.deactivate()
            pw.deactivate()
        finally:
            pw.deactivate()

    def test_double_activate_raises(self) -> None:
        pw = ModelOffloader.from_module(_make_simple_model())
        try:
            pw.activate("cpu")
            with pytest.raises(RuntimeError, match=r"already.*active"):
                pw.activate("cpu")
        finally:
            pw.deactivate()

    def test_activate_without_any_device_raises(self) -> None:
        pw = ModelOffloader.from_module(_make_simple_model())
        try:
            with pytest.raises(ValueError, match="requires a device"):
                pw.activate()
        finally:
            pw.deactivate()


# ---------------------------------------------------------------------------
# Cleanup: drop the strategy + model refs to free host
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_drop_strategy_and_model_frees_host(self) -> None:
        # The "drop refs to free host" contract. Strategies don't have
        # a destructive close(); host tensors live in module state
        # and are freed when the caller drops the model reference (and
        # the strategy reference, which is the only other holder).
        import gc
        import weakref

        m = _make_simple_model()
        pw = ModelOffloader.from_module(m)
        module_param_ref = weakref.ref(m[0]._parameters["weight"])
        pw.deactivate()
        assert module_param_ref() is not None  # still alive via model state
        del m, pw
        gc.collect()
        assert module_param_ref() is None  # GC freed it


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_rejects_empty_model(self) -> None:
        # Frozen but with no params or buffers — nothing to manage.
        class Empty(nn.Module):
            pass
        m = Empty()
        with pytest.raises(ValueError, match="at least one parameter"):
            ModelOffloader.from_module(m)

    def test_accepts_buffer_only_module(self) -> None:
        # A module with only registered buffers (no frozen params) is a
        # legitimate target — common for things like RoPE position tables
        # or sinusoidal embeddings. ModelOffloader should capture the buffers
        # and behave as a no-op for params.
        class BufferOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.randn(8, 4))

        m = BufferOnly()
        pw = ModelOffloader.from_module(m)
        try:
            assert pw.cache_bytes == 8 * 4 * 4  # float32
            assert not m.table.is_pinned()
        finally:
            pw.deactivate()

    def test_constructor_does_not_call_module_to(self) -> None:
        class Guarded(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4), requires_grad=False)

            def to(self, *args, **kwargs):
                raise AssertionError("constructor must capture directly")

        m = Guarded()
        pw = ModelOffloader.from_module(m)
        try:
            assert not m.weight.is_pinned()
        finally:
            pw.deactivate()


# ---------------------------------------------------------------------------
# Tied-weight dedup — the V1 model-agnostic claim
# ---------------------------------------------------------------------------


class TestTiedWeightDedup:
    def _make_tied_model(self) -> tuple[nn.Module, nn.Embedding, nn.Linear]:
        """Embed + linear head with tied weights (the standard HF
        tie_weights() pattern: same Parameter under two names)."""
        embed = nn.Embedding(32, 16)
        head = nn.Linear(16, 32, bias=False)
        head.weight = embed.weight  # same Parameter object
        m = nn.Module()
        m.embed = embed
        m.head = head
        for p in m.parameters():
            p.requires_grad = False
        return m, embed, head

    def _make_distinct_param_tied_model(self) -> tuple[nn.Module, nn.Parameter, nn.Parameter]:
        """Two distinct Parameter objects sharing the same storage."""
        shared = torch.randn(8, 16, dtype=torch.bfloat16)
        a = nn.Parameter(shared, requires_grad=False)
        b = nn.Parameter(shared, requires_grad=False)  # distinct Parameter, same storage
        assert a is not b
        assert a.data.data_ptr() == b.data.data_ptr()
        m = nn.Module()
        m.a = a
        m.b = b
        return m, a, b

    def test_same_parameter_under_two_names_dedupes(self) -> None:
        m, embed, head = self._make_tied_model()
        pw = ModelOffloader.from_module(m)
        try:
            # Exactly one unique host parameter for the tied weight.
            assert _unique_host_param_count(pw) == 1
            assert pw.param_names == {"embed.weight", "head.weight"}
            assert pw.cache_bytes == 32 * 16 * 4  # Count the tied float32 storage once.
            # After construction, both names reference the same Parameter
            # object, preserving tying at the strongest level.
            assert m.embed._parameters["weight"] is m.head._parameters["weight"]
            host = resident_instance(pw).params["embed.weight"]
            # The materialized CPU wrapper is installed onto the module (the
            # instance no longer caches it); both tied leaves share it.
            cpu_param = m.embed.weight
            assert m.head.weight is cpu_param
            assert cpu_param.data_ptr() == host.make_cpu_param().data_ptr()
        finally:
            pw.deactivate()

    def test_distinct_params_sharing_storage_dedupe(self) -> None:
        m, _, _ = self._make_distinct_param_tied_model()
        pw = ModelOffloader.from_module(m)
        try:
            assert _unique_host_param_count(pw) == 1
            assert pw.param_names == {"a", "b"}
            # Both module entries now reference the same Parameter object.
            assert m._parameters["a"] is m._parameters["b"]
        finally:
            pw.deactivate()

    def test_zero_sized_same_parameter_under_two_names_dedupes(self) -> None:
        p = nn.Parameter(torch.empty(0), requires_grad=False)
        m = nn.Module()
        m.a = p
        m.b = p

        pw = ModelOffloader.from_module(m)
        try:
            assert _unique_host_param_count(pw) == 1
            assert m._parameters["a"] is m._parameters["b"]
        finally:
            pw.deactivate()

    @CUDA
    def test_tied_params_share_gpu_storage_on_activate(self) -> None:
        m, embed, head = self._make_tied_model()
        pw = ModelOffloader.from_module(m)
        try:
            with activated_model(pw, "cuda"):
                assert embed.weight.is_cuda
                assert head.weight.is_cuda
                assert embed.weight.data.data_ptr() == head.weight.data.data_ptr()
                # Stronger: same Parameter object.
                assert m.embed._parameters["weight"] is m.head._parameters["weight"]
        finally:
            pw.deactivate()

    @CUDA
    def test_distinct_tied_params_share_gpu_storage_on_activate(self) -> None:
        m, a, b = self._make_distinct_param_tied_model()
        pw = ModelOffloader.from_module(m)
        try:
            with activated_model(pw, "cuda"):
                # Registry identity comparison; the local `a` / `b` refs are
                # the now-orphaned originals (replaced in module registries).
                assert m._parameters["a"].is_cuda
                assert m._parameters["b"].is_cuda
                assert m._parameters["a"].data.data_ptr() == m._parameters["b"].data.data_ptr()
                assert m._parameters["a"] is m._parameters["b"]
        finally:
            pw.deactivate()

    @CUDA
    def test_cuda_origin_tied_params_dedupe_before_cpu_pin(self) -> None:
        shared = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
        m = nn.Module()
        m.a = nn.Parameter(shared, requires_grad=False)
        m.b = nn.Parameter(shared, requires_grad=False)

        pw = ModelOffloader.from_module(m)
        try:
            assert _unique_host_param_count(pw) == 1
            assert m._parameters["a"] is m._parameters["b"]
            assert not m.a.is_pinned()
        finally:
            pw.deactivate()


# ---------------------------------------------------------------------------
# Edge cases caught in review
# ---------------------------------------------------------------------------


class TestSharedSubmoduleAlias:
    def test_same_module_under_two_attrs(self) -> None:
        # Valid PyTorch model: m.a is m.b is the same nn.Linear instance.
        # named_modules() default removes the duplicate, but
        # named_parameters(remove_duplicate=False) walks both — the parent
        # lookup must use remove_duplicate=False too or KeyError.
        shared = nn.Linear(4, 4, bias=False)
        for p in shared.parameters():
            p.requires_grad = False
        m = nn.Module()
        m.a = shared
        m.b = shared
        pw = ModelOffloader.from_module(m)
        try:
            assert _unique_host_param_count(pw) == 1
            assert pw.param_names == {"a.weight", "b.weight"}
            assert m.a._parameters["weight"] is m.b._parameters["weight"]
        finally:
            pw.deactivate()

    def test_aliased_buffer(self) -> None:
        # Two distinct submodules sharing the same buffer tensor. The
        # default named_buffers() walks only one path; using
        # remove_duplicate=False plus tensor-id dedup ensures both
        # paths get host consistently.
        shared_buf = torch.randn(4)

        class Inner(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.register_buffer("buf", b)
                self.weight = nn.Parameter(torch.randn(2), requires_grad=False)

        m = nn.Module()
        m.a = Inner(shared_buf)
        m.b = Inner(shared_buf)
        pw = ModelOffloader.from_module(m)
        try:
            # One host buffer backing covers both alias paths.
            assert _unique_host_buffer_count(pw) == 1
            assert pw.buffer_names == {"a.buf", "b.buf"}
            host_buffer = resident_instance(pw).buffers["a.buf"]
            assert not host_buffer.tensor.is_pinned()
            # Both module entries reference the SAME host tensor.
            assert m.a.buf is host_buffer.tensor
            assert m.b.buf is host_buffer.tensor
        finally:
            pw.deactivate()

    def test_zero_sized_aliased_buffer(self) -> None:
        shared_buf = torch.empty(0)

        class Inner(nn.Module):
            def __init__(self, b: torch.Tensor):
                super().__init__()
                self.register_buffer("buf", b)
                self.weight = nn.Parameter(torch.randn(2), requires_grad=False)

        m = nn.Module()
        m.a = Inner(shared_buf)
        m.b = Inner(shared_buf)
        pw = ModelOffloader.from_module(m)
        try:
            assert _unique_host_buffer_count(pw) == 1
            host_buffer = resident_instance(pw).buffers["a.buf"]
            assert m.a.buf is host_buffer.tensor
            assert m.b.buf is host_buffer.tensor
        finally:
            pw.deactivate()

    @CUDA
    def test_cuda_origin_aliased_buffer_dedupes_before_cpu_pin(self) -> None:
        shared_buf = torch.randn(4, device="cuda")

        class Inner(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.register_buffer("buf", b)
                self.weight = nn.Parameter(
                    torch.randn(2, device="cuda"),
                    requires_grad=False,
                )

        m = nn.Module()
        m.a = Inner(shared_buf)
        m.b = Inner(shared_buf)

        pw = ModelOffloader.from_module(m)
        try:
            assert _unique_host_buffer_count(pw) == 1
            host_buffer = resident_instance(pw).buffers["a.buf"]
            assert m.a.buf is host_buffer.tensor
            assert m.b.buf is host_buffer.tensor
            assert not m.a.buf.is_pinned()
        finally:
            pw.deactivate()


class TestMixedTrainableFrozenTied:
    def test_raises_when_tied_group_has_trainable_param(self) -> None:
        shared = torch.randn(8, dtype=torch.bfloat16)
        a = nn.Parameter(shared, requires_grad=True)
        b = nn.Parameter(shared, requires_grad=False)
        m = nn.Module()
        m.a = a
        m.b = b
        with pytest.raises(ValueError, match="requires_grad=False"):
            ModelOffloader.from_module(m)

class TestZeroSizedParams:
    def test_zero_sized_params_do_not_collapse(self) -> None:
        # Empty tensors all share data_ptr()==0; they must not dedupe
        # into one host param binding.
        m = nn.Module()
        m.a = nn.Parameter(torch.empty(0), requires_grad=False)
        m.b = nn.Parameter(torch.empty(0), requires_grad=False)
        # Need at least one non-empty frozen param so the constructor doesn't
        # reject the model. The empties should each be their own entry.
        m.c = nn.Parameter(torch.randn(4), requires_grad=False)
        pw = ModelOffloader.from_module(m)
        try:
            # 3 entries: a, b, c — empties did not collapse.
            assert _unique_host_param_count(pw) == 3
        finally:
            pw.deactivate()


# ---------------------------------------------------------------------------
# cache_bytes accounting
# ---------------------------------------------------------------------------


class TestCacheBytes:
    def test_cache_bytes_includes_buffers(self) -> None:
        m = nn.Module()
        m.weight = nn.Parameter(torch.randn(4, 4), requires_grad=False)
        m.register_buffer("running", torch.randn(8))

        store = ModelOffloader.from_module(m)

        expected = (4 * 4 * 4) + (8 * 4)
        assert store.cache_bytes == expected
