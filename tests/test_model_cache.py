"""Cache-scoped host memory ownership and shared registration budgets."""

import pytest
import torch
from torch import nn

from piper_offload import AdapterSpec, HostMemoryManager, ModelCache, ModelSpec, ObjectSpec
from tests._block_compile_helpers import _BlockModel
from tests._host_memory_helpers import RecordingBackend
from tests.conftest import block_components, host_component


def _model():
    model = nn.Linear(8, 4).requires_grad_(False)
    model.register_buffer("scale", torch.ones(4))
    return model


def _hosts(offloader):
    instance = host_component(offloader)._instance
    return (*instance.params.values(), *instance.buffers.values())


def test_registered_models_and_adapters_share_cache_manager_after_rebuild():
    memory = HostMemoryManager(max_pinned_bytes=0)
    cache = ModelCache(memory_manager=memory)
    model_spec = ModelSpec(key="model", factory=_model)
    adapter_spec = AdapterSpec(key="adapter", factory=lambda: {
        "projection.lora_A.weight": torch.randn(2, 8),
        "projection.lora_B.weight": torch.randn(4, 2),
        "projection.delta.weight": torch.randn(4, 8),
        "replacement.weight": torch.randn(4, 8),
    })
    cache.register(model_spec)
    cache.register(adapter_spec)
    previous = None

    for _ in range(2):
        with cache.lease_many(["model", "adapter"]) as (offloader, adapter):
            assert offloader is not previous
            delta = adapter.targets["projection.weight"]
            value = adapter.targets["replacement.weight"]
            hosts = (*_hosts(offloader), delta.lora.a, delta.lora.b, delta.dense, value.backing)
            assert len(hosts) == 7
            assert all(host.memory_manager is memory for host in hosts)
            assert all(
                handle.memory_manager is memory
                for host in hosts for handle in host.backing_handles()
            )
            previous = offloader
        cache.clear()


@pytest.mark.parametrize("shared", [False, True])
def test_manager_scope_is_per_cache_unless_explicitly_shared(shared):
    first = ModelCache()
    second = ModelCache(memory_manager=first.memory_manager) if shared else ModelCache()
    spec = ModelSpec(key="model", factory=_model)
    with first.lease(spec) as first_model, second.lease(spec) as second_model:
        first_host = _hosts(first_model)[0]
        second_host = _hosts(second_model)[0]
        assert first_host.memory_manager is first.memory_manager
        assert second_host.memory_manager is second.memory_manager
        assert (first_host.memory_manager is second_host.memory_manager) == shared
        first.memory_manager.max_pinned_bytes = 0
        assert second.memory_manager.max_pinned_bytes == (0 if shared else None)
    first.clear()
    second.clear()


def test_object_specs_keep_their_own_construction():
    cache = ModelCache()
    value = object()
    spec = ObjectSpec(key="tokenizer", factory=lambda: value)
    with cache.lease(spec) as cached:
        assert cached is value
    cache.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device required")
def test_cache_activations_share_pin_budget_and_evict_only_idle_weights():
    backend = RecordingBackend()
    memory = HostMemoryManager(64 * 1024**2, backend=backend)
    cache = ModelCache(memory_manager=memory)
    inputs = torch.randn(2, 64)
    expected = {}

    def build_model(key, num_blocks):
        model = _BlockModel(num_blocks=num_blocks, width=64)
        expected[key] = model(inputs)
        return model

    first = ModelSpec(
        key="first", factory=lambda: build_model("first", 1), block_paths=("blocks",),
    )
    second = ModelSpec(
        key="second", factory=lambda: build_model("second", 2), block_paths=("blocks",),
    )
    adapter = AdapterSpec(key="adapter", factory=lambda: {
        "blocks.0.proj.lora_A.weight": torch.randn(2, 64),
        "blocks.0.proj.lora_B.weight": torch.zeros(64, 2),
    })

    try:
        with torch.no_grad():
            with cache.use(first, device="cuda", adapter_specs=[adapter]) as model:
                torch.testing.assert_close(model(inputs.cuda()).cpu(), expected["first"])
                assert memory.stats.active_leases == 1
                assert len(backend.registrations) == 3  # Model weight and both LoRA factors.
                budget = memory.stats.pinned_bytes
                assert budget > 0
                memory.max_pinned_bytes = budget

                with cache.use(second, device="cuda") as other:
                    assert memory.stats.active_leases == 2
                    assert memory.stats.pinned_bytes <= budget
                    assert not backend.unregistrations
                    with cache.lease(second) as offloader:
                        lease = block_components(offloader)[0]._pin_leases[0]
                        assert lease.pageable_bytes > 0
                    torch.testing.assert_close(other(inputs.cuda()).cpu(), expected["second"])

            assert memory.stats.active_leases == 0
            with cache.use(second, device="cuda") as other:
                assert backend.unregistrations
                assert 0 < memory.stats.pinned_bytes <= budget
                with cache.lease(second) as offloader:
                    assert block_components(offloader)[0]._pin_leases[0].registered_bytes > 0
                torch.testing.assert_close(other(inputs.cuda()).cpu(), expected["second"])
    finally:
        cache.clear()
        memory.clear()
    assert memory.stats.active_leases == 0
    assert memory.stats.pinned_bytes == 0
