"""Tests for activation-scoped Adapter application through ``ModelOffloader``.

Covers Adapter construction validation, activation matching, lifecycle
(activate/deactivate), Adapter switching, and forward-output correctness
against a manually-merged baseline.

Routed activation and merge preflight run on CPU. Tests that load merged
weights or exercise CUDA streaming require a GPU.
"""

from collections.abc import Sequence
from typing import get_protocol_members
from weakref import ref

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

import piper_offload.lora as lora_impl

from piper_offload import (
    BlockCompileConfig,
    Adapter,
    MappedCheckpoint,
    file_slice,
    AdapterTarget,
    LoRAFactor,
    AdapterMode,
    LoRATransform,
    ModelCache,
    ResourceCache,
    ModelOffloader,
    ModelSpec,
    ParameterDelta,
    ParameterDeltaTransform,
    ParameterValue,
    ParameterValueTransform,
    ResourceNotRegisteredError,
    AdapterSpec,
    ScaledLoRAFactor,
    ScaledParameterValue,
    derive_seed,
    merge_adapter,
    register_adapter,
)
from piper_offload.adapter import group_adapter_updates, resolve_adapter_strengths
from piper_offload.int4_tile_adapter import Int4TilePackedAdapter
from piper_offload.host_module import ParameterOverride, HostModuleInstance
from piper_offload.host_param import HostParam
from piper_offload.protocols import (
    ResourceBinding,
    ResourceStore,
)
from piper_offload.tensor_adapters import (
    DenseMergeTensorAdapter,
    DequantRequantTensorAdapter,
    LoRAMergeTensorAdapter,
    RegularAdapter,
    TensorCopyIntoAdapter,
)

from tests.conftest import (
    Interrupt,
    activated_model,
    block_components,
    host_component,
    transient_components,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_factor(target: AdapterTarget) -> LoRAFactor:
    assert isinstance(target, ParameterDelta)
    assert target.lora is not None
    return target.lora


def _factor_tensors(
    target: AdapterTarget,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a host :class:`LoRAFactor`'s ``(a, b)`` as CPU tensors."""
    factor = _require_factor(target)
    return factor.a.make_cpu_param().data, factor.b.make_cpu_param().data


def _make_bf16_model(
    num_blocks: int = 4,
    dim: int = 16,
    *,
    attn_bias: bool = False,
) -> nn.Module:
    """Tiny block-streaming-shaped model with bf16 frozen params."""

    class Block(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.attn = nn.Linear(dim, dim, bias=attn_bias)
            self.ff = nn.Linear(dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.ff(self.attn(x))

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(dim, dim, bias=False)
            self.transformer_blocks = nn.ModuleList([Block(dim) for _ in range(num_blocks)])
            self.head = nn.Linear(dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            for blk in self.transformer_blocks:
                x = blk(x)
            return self.head(x)

    m = M()
    m = m.to(torch.bfloat16)
    for p in m.parameters():
        p.requires_grad = False
    return m


def _make_tied_non_block_model(
    num_blocks: int = 2,
    dim: int = 16,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    class Block(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.attn = nn.Linear(dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.attn(x)

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(dim, dim, bias=False)
            self.transformer_blocks = nn.ModuleList([Block(dim) for _ in range(num_blocks)])
            self.head = nn.Linear(dim, dim, bias=False)
            self.head.weight = self.embed.weight

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            for blk in self.transformer_blocks:
                x = blk(x)
            return self.head(x)

    model = M().to(dtype)
    for p in model.parameters():
        p.requires_grad = False
    return model


def _make_lora_sd(
    num_blocks: int,
    dim: int,
    rank: int = 4,
    seed: int = 0,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Build a flat safetensors-style state dict targeting attn.weight."""
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    for b in range(num_blocks):
        base = f"{prefix}transformer_blocks.{b}.attn"
        sd[f"{base}.lora_A.weight"] = torch.randn(
            rank,
            dim,
            generator=g,
            dtype=torch.float32,
        )
        sd[f"{base}.lora_B.weight"] = torch.randn(
            dim,
            rank,
            generator=g,
            dtype=torch.float32,
        )
    return sd


def _make_lora(
    num_blocks: int,
    dim: int,
    rank: int = 4,
    seed: int = 0,
    prefix: str = "",
) -> Adapter:
    """Build a Adapter targeting attn.weight across all blocks."""
    sd = _make_lora_sd(
        num_blocks,
        dim,
        rank=rank,
        seed=seed,
        prefix=prefix,
    )
    return Adapter.from_state_dict(state_dict=sd)


def _expected_merged_weight(
    base: torch.Tensor,
    loras: list[tuple[Adapter, float]],
    block_idx: int,
    qual: str,
) -> torch.Tensor:
    """Compute the target weight by summing all Adapter deltas onto the base."""
    out = base.clone()
    target_name = f"transformer_blocks.{block_idx}.{qual}"
    for lora, strength in loras:
        factors = lora.targets.get(target_name)
        if factors is None:
            continue
        a, b = _factor_tensors(factors)
        out.addmm_(
            b.to(device=out.device, dtype=out.dtype),
            a.to(device=out.device, dtype=out.dtype),
            alpha=strength,
        )
    return out


def _expected_routed_output(
    model: nn.Module,
    x: torch.Tensor,
    loras: list[tuple[Adapter, float]],
) -> torch.Tensor:
    """Manual routed baseline using F.linear to bypass installed hooks."""
    h = F.linear(x, model.embed.weight.to(x.device))
    for i, blk in enumerate(model.transformer_blocks):
        base_bias = blk.attn.bias
        base_attn = F.linear(
            h,
            blk.attn.weight.to(h.device),
            None if base_bias is None else base_bias.to(h.device),
        )
        target_name = f"transformer_blocks.{i}.attn.weight"
        a_parts = []
        b_parts = []
        for lora, strength in loras:
            factors = lora.targets.get(target_name)
            if factors is None:
                continue
            a, b = _factor_tensors(factors)
            a_parts.append(a.to(device=h.device, dtype=h.dtype))
            b_part = b.to(device=h.device, dtype=h.dtype).clone()
            b_part.mul_(strength)
            b_parts.append(b_part)
        if a_parts:
            a_fused = torch.cat(a_parts, dim=0)
            b_fused = torch.cat(b_parts, dim=1)
            base_attn = base_attn + (h @ a_fused.T) @ b_fused.T
        h = F.linear(base_attn, blk.ff.weight.to(h.device))
    return F.linear(h, model.head.weight.to(h.device))


def _make_strategy(model: nn.Module) -> ModelOffloader:
    """Shorthand for constructing the strategy."""
    return ModelOffloader.from_module(model, block_paths=["transformer_blocks"])


def _plan_lora_merge(offloader: ModelOffloader, adapter: Adapter) -> dict[str, ParameterOverride]:
    """Run merge preflight on host weights, before CUDA allocation."""
    updates = group_adapter_updates([(adapter, 1.0)], offloader.param_names)
    return offloader._build_merge_parameter_overrides(torch.device("cuda"), updates)


# ---------------------------------------------------------------------------
# Adapter construction validation
# ---------------------------------------------------------------------------


class TestLoRAConstruction:
    def test_rejects_empty_state_dict(self) -> None:
        with pytest.raises(ValueError, match="contains no targets"):
            Adapter.from_state_dict(state_dict={})

    def test_unpaired_a_factor(self) -> None:
        sd = {"transformer_blocks.0.attn.lora_A.weight": torch.randn(4, 16)}
        with pytest.raises(ValueError, match="Unpaired"):
            Adapter.from_state_dict(state_dict=sd)

    def test_unpaired_b_factor(self) -> None:
        sd = {"transformer_blocks.0.attn.lora_B.weight": torch.randn(16, 4)}
        with pytest.raises(ValueError, match="Unpaired"):
            Adapter.from_state_dict(state_dict=sd)

    def test_rejects_non_floating_factor_dtype(self) -> None:
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.zeros(4, 16, dtype=torch.int32),
            "transformer_blocks.0.attn.lora_B.weight": torch.zeros(16, 4, dtype=torch.int32),
        }
        with pytest.raises(ValueError, match="floating-point"):
            Adapter.from_state_dict(state_dict=sd)

    @pytest.mark.parametrize(
        "key",
        [
            "target.lora_A.weight",
            "target.lora_B.weight",
        ],
    )
    def test_rejects_meta_factor_sources(self, key: str) -> None:
        sd = {
            "target.lora_A.weight": torch.randn(1, 3),
            "target.lora_B.weight": torch.randn(2, 1),
        }
        sd[key] = torch.empty_like(sd[key], device="meta")

        with pytest.raises(ValueError, match="physical values"):
            Adapter.from_state_dict(sd)

    def test_rejects_non_tensor_factor_with_value_error(self) -> None:
        sd: dict[str, torch.Tensor] = {
            "target.lora_A.weight": torch.randn(1, 3),
            "target.lora_B.weight": torch.randn(2, 1),
        }
        sd["target.lora_A.weight"] = object()  # type: ignore[assignment]

        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            Adapter.from_state_dict(sd)

    def test_rejects_empty_factor_target_name(self) -> None:
        with pytest.raises(ValueError, match="target names must be non-empty"):
            Adapter.from_state_dict(
                {
                    ".lora_A.weight": torch.randn(1, 3),
                    ".lora_B.weight": torch.randn(2, 1),
                }
            )

    def test_rejects_rank_mismatch(self) -> None:
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(16, 8),
        }
        with pytest.raises(ValueError, match="shape mismatch"):
            Adapter.from_state_dict(state_dict=sd)

    def test_rejects_non_2d_factor(self) -> None:
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(4),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(16, 4),
        }
        with pytest.raises(ValueError, match="shape mismatch"):
            Adapter.from_state_dict(state_dict=sd)

    def test_construction_preserves_host_factors_and_resource_contract(self) -> None:
        lora = _make_lora(4, 16, rank=4)
        assert isinstance(lora, ResourceStore)
        assert not isinstance(lora, ResourceBinding)
        assert not isinstance(lora, nn.Module)
        assert not hasattr(lora, "activate")
        assert not hasattr(lora, "deactivate")
        assert lora.cache_bytes == 4 * (4 * 16 + 16 * 4) * 4
        for factor in lora.targets.values():
            a, b = _factor_tensors(factor)
            assert not a.is_pinned()
            assert not b.is_pinned()
        targets = lora.targets
        assert lora.targets is targets
        with pytest.raises(TypeError):
            targets["other.weight"] = next(iter(targets.values()))  # type: ignore[index]

    def test_factor_host_params_build_instance_without_repin(self) -> None:
        """Host factors can be consumed without cloning or recapturing."""
        lora = _make_lora(1, 16, rank=4)
        factor = _require_factor(lora.targets["transformer_blocks.0.attn.weight"])
        assert isinstance(factor.a, HostParam)
        assert isinstance(factor.b, HostParam)

        holder = nn.Module()
        holder.register_parameter("a", factor.a.make_cpu_param())
        holder.register_parameter("b", factor.b.make_cpu_param())
        instance = HostModuleInstance(
            module=holder,
            params={"a": factor.a, "b": factor.b},
            buffers={},
        )
        # Consumers retain the exact host objects.
        assert instance.params["a"] is factor.a
        assert instance.params["b"] is factor.b

    def test_keys_used_verbatim(self) -> None:
        # Keys are used as-is — no built-in remapping. A prefixed key stays
        # prefixed; stripping it (e.g. ComfyUI's ``diffusion_model.``) is the
        # caller's job in the factory that produces the state dict.
        lora = _make_lora(1, 16, prefix="diffusion_model.")
        assert "diffusion_model.transformer_blocks.0.attn.weight" in lora.targets
        assert "transformer_blocks.0.attn.weight" not in lora.targets

    def test_from_state_dict_dtype_casts_factors_at_build(self) -> None:
        # Routed callers pass the model's compute dtype to reduce cache bytes
        # and per-forward transfer volume.
        sd = _make_lora_sd(num_blocks=2, dim=16, seed=1)  # fp32 factors
        store = Adapter.from_state_dict(state_dict=sd, dtype=torch.bfloat16)
        for factor in store.targets.values():
            assert all(tensor.dtype == torch.bfloat16 for tensor in _factor_tensors(factor))
        # Default keeps the stored dtype.
        kept = Adapter.from_state_dict(state_dict=sd)
        for factor in kept.targets.values():
            assert all(tensor.dtype == torch.float32 for tensor in _factor_tensors(factor))

    @pytest.mark.parametrize("target_key", ["", 1])
    def test_direct_constructor_rejects_invalid_target_names(
        self,
        target_key: object,
    ) -> None:
        value = ParameterValue.from_tensor(torch.randn(2, 2))

        with pytest.raises(ValueError, match="non-empty strings"):
            Adapter({target_key: value})  # type: ignore[dict-item]

    def test_direct_constructor_rejects_invalid_target_type(self) -> None:
        with pytest.raises(ValueError, match="ParameterDelta or ParameterValue"):
            Adapter({"target.weight": object()})  # type: ignore[dict-item]

    def test_parameter_delta_resource_coalesces_lora_and_dense_weight(self) -> None:
        a = torch.randn(2, 4)
        b = torch.randn(3, 2)
        dense = torch.randn(3, 4)

        adapter = Adapter.from_state_dict(
            {
                "target.lora_A.weight": a,
                "target.lora_B.weight": b,
                "target.delta.weight": dense,
            },
        )

        assert tuple(adapter.targets) == ("target.weight",)
        delta = adapter.targets["target.weight"]
        assert isinstance(delta, ParameterDelta)
        assert delta.lora is not None
        assert delta.dense is not None
        for backing, source in ((delta.lora.a, a), (delta.lora.b, b), (delta.dense, dense)):
            captured = backing.make_cpu_param()
            assert captured.data_ptr() == source.data_ptr()
            torch.testing.assert_close(captured, source)
        assert delta.cache_bytes == a.nbytes + b.nbytes + dense.nbytes

    def test_parameter_delta_uses_final_component_as_exact_parameter_leaf(self) -> None:
        weight = torch.randn(3, 4)
        bias = torch.randn(3)

        adapter = Adapter.from_state_dict(
            {
                "target.delta.weight": weight,
                "target.delta.bias": bias,
            }
        )

        assert tuple(adapter.targets) == ("target.weight", "target.bias")
        assert all(isinstance(target, ParameterDelta) for target in adapter.targets.values())

    @pytest.mark.parametrize("key", [".delta.weight", ".delta.bias"])
    def test_rejects_empty_delta_target_name(self, key: str) -> None:
        with pytest.raises(ValueError, match="target names must be non-empty"):
            Adapter.from_state_dict({key: torch.randn(2)})

    def test_noncanonical_delta_leaf_remains_an_exact_parameter_value(self) -> None:
        source = torch.randn(2)

        adapter = Adapter.from_state_dict({"target.delta.scale": source})

        assert tuple(adapter.targets) == ("target.delta.scale",)
        assert isinstance(adapter.targets["target.delta.scale"], ParameterValue)

    def test_parameter_delta_and_value_cannot_share_target(self) -> None:
        with pytest.raises(ValueError, match="cannot contain both a parameter delta and a value"):
            Adapter.from_state_dict(
                {
                    "target.delta.weight": torch.randn(2, 2),
                    "target.weight": torch.randn(2, 2),
                }
            )

    @pytest.mark.parametrize(
        ("source", "message"),
        [
            (torch.ones(2, 2, dtype=torch.int32), "floating-point"),
            (torch.empty(2, 2, device="meta"), "physical values"),
        ],
    )
    def test_rejects_invalid_dense_parameter_delta_source(
        self,
        source: torch.Tensor,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            Adapter.from_state_dict({"target.delta.weight": source})

    def test_from_state_dict_rejects_non_floating_dtype(self) -> None:
        # A non-floating dtype would slip past the float-factor validation
        # (which runs on the original tensors) and silently produce int factors.
        sd = _make_lora_sd(num_blocks=1, dim=16, seed=1)
        with pytest.raises(ValueError, match="floating-point"):
            Adapter.from_state_dict(state_dict=sd, dtype=torch.int8)

    def test_parameter_value_resource_uses_exact_parameter_names(self) -> None:
        weight = torch.randn(3, 4)
        bias = torch.randn(3)

        lora = Adapter.from_state_dict(
            {
                "target.weight": weight,
                "target.bias": bias,
            }
        )

        assert tuple(lora.targets) == (
            "target.weight",
            "target.bias",
        )
        assert all(isinstance(target, ParameterValue) for target in lora.targets.values())
        value = lora.targets["target.weight"]
        assert isinstance(value, ParameterValue)
        scaled = value.scaled(0.25)
        assert isinstance(scaled, ScaledParameterValue)
        assert scaled.value is value
        assert scaled.strength == 0.25
        assert scaled.effective_strength == 1.0
        assert not value.scale_with_strength
        assert lora.cache_bytes == weight.nbytes + bias.nbytes
        with pytest.raises(TypeError):
            lora.targets["other"] = next(iter(lora.targets.values()))  # type: ignore[index]

    def test_parameter_value_resource_can_scale_with_adapter_strength(self) -> None:
        source = torch.randn(3, 4)
        adapter = Adapter.from_state_dict(
            {"target.weight": source},
            scale_parameter_values=True,
        )

        value = adapter.targets["target.weight"]
        assert isinstance(value, ParameterValue)
        assert value.scale_with_strength
        scaled = value.scaled(-0.25)
        assert scaled.strength == -0.25
        assert scaled.effective_strength == -0.25

        direct = ParameterValue.from_tensor(
            source,
            scale_with_strength=True,
        )
        assert direct.scale_with_strength

    def test_factor_and_parameter_value_cannot_share_target(self) -> None:
        sd = _make_lora_sd(num_blocks=1, dim=4, rank=2)
        diff = torch.randn(4, 4)

        with pytest.raises(ValueError, match="cannot contain both"):
            Adapter.from_state_dict(
                {
                    **sd,
                    "transformer_blocks.0.attn.weight": diff,
                }
            )

    @pytest.mark.parametrize(
        ("diff", "message"),
        [
            (torch.ones(2, 2, dtype=torch.int32), "floating-point"),
            (torch.empty(2, 2, device="meta"), "physical values"),
        ],
    )
    def test_rejects_invalid_parameter_value_source(
        self,
        diff: torch.Tensor,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            Adapter.from_state_dict({"target.weight": diff})

    def test_parameter_value_payload_finiteness_is_caller_owned(self) -> None:
        nan_value = Adapter.from_state_dict(
            {"target.weight": torch.tensor([float("nan")])},
        ).targets["target.weight"]
        assert isinstance(nan_value, ParameterValue)
        assert torch.isnan(nan_value.backing.make_cpu_param()).all()

        infinite_value = Adapter.from_state_dict(
            {"target.weight": torch.tensor([1.0e10])},
            dtype=torch.float16,
        ).targets["target.weight"]
        assert isinstance(infinite_value, ParameterValue)
        assert torch.isinf(infinite_value.backing.make_cpu_param()).all()

    def test_dense_delta_payload_finiteness_is_caller_owned(self) -> None:
        model = nn.Module()
        model.target = nn.Linear(2, 2, bias=False)
        model.requires_grad_(False)
        delta = Adapter.from_state_dict(
            {"target.delta.weight": torch.full((2, 2), float("nan"))},
        )

        assert merge_adapter(model, [(delta, 1.0)]) == 1
        assert torch.isnan(model.target.weight).all()


# ---------------------------------------------------------------------------
# Parameter delta semantics
# ---------------------------------------------------------------------------


class TestParameterDelta:
    def test_requires_at_least_one_delta_representation(self) -> None:
        with pytest.raises(ValueError, match="requires a LoRA factor, a dense delta, or both"):
            ParameterDelta.from_tensors()

    @pytest.mark.parametrize("strength", [float("nan"), float("inf"), float("-inf")])
    def test_scaled_adapter_representations_reject_non_finite_strength(self, strength: float) -> None:
        delta = ParameterDelta.from_tensors(
            a=torch.randn(1, 3),
            b=torch.randn(2, 1),
        )
        value = ParameterValue.from_tensor(torch.randn(2, 3))
        factor = LoRAFactor.from_tensors(
            torch.randn(1, 3),
            torch.randn(2, 1),
        )

        for representation in (delta, value, factor):
            with pytest.raises(ValueError, match="strength must be finite"):
                representation.scaled(strength)

    @pytest.mark.parametrize("mode", ["merge", "routed"])
    def test_adapter_modes_reject_non_finite_strength(self, mode: AdapterMode) -> None:
        model = nn.Module()
        model.target = nn.Linear(3, 2, bias=False)
        model.requires_grad_(False)
        before = model.target.weight.detach().clone()
        adapter = Adapter.from_state_dict(
            {
                "target.lora_A.weight": torch.randn(1, 3),
                "target.lora_B.weight": torch.randn(2, 1),
            }
        )

        if mode == "merge":
            with pytest.raises(ValueError, match="strength must be finite"):
                merge_adapter(model, [(adapter, float("inf"))])
        else:
            offloader = ModelOffloader.from_module(model)
            with pytest.raises(ValueError, match="strength must be finite"):
                offloader.activate(
                    "cpu",
                    adapters=[adapter],
                    adapter_strengths=[float("inf")],
                    adapter_mode="routed",
                )
            assert offloader.active_device is None
            assert offloader._routed_hook_removers == []

        torch.testing.assert_close(model.target.weight, before)

    def test_permanent_merge_combines_lora_and_dense_under_one_strength(self) -> None:
        model = nn.Module()
        model.target = nn.Linear(4, 3, bias=True)
        model.requires_grad_(False)
        base_weight = model.target.weight.detach().clone()
        base_bias = model.target.bias.detach().clone()
        a = torch.randn(2, 4)
        b = torch.randn(3, 2)
        dense_weight = torch.randn(3, 4)
        dense_bias = torch.randn(3)
        strength = -0.25
        adapter = Adapter.from_state_dict(
            {
                "target.lora_A.weight": a,
                "target.lora_B.weight": b,
                "target.delta.weight": dense_weight,
                "target.delta.bias": dense_bias,
            }
        )

        assert merge_adapter(model, [(adapter, strength)]) == 2

        torch.testing.assert_close(
            model.target.weight,
            base_weight + strength * (b @ a + dense_weight),
        )
        torch.testing.assert_close(
            model.target.bias,
            base_bias + strength * dense_bias,
        )

    @pytest.mark.parametrize(
        ("dtype", "base_value"),
        [(torch.float16, 2048.0), (torch.bfloat16, 256.0)],
    )
    @pytest.mark.parametrize("shape", [(1, 1), (2, 3, 1, 1)])
    def test_combined_delta_accumulates_before_low_precision_base_update(
        self,
        dtype: torch.dtype,
        base_value: float,
        shape: tuple[int, ...],
    ) -> None:
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.full(shape, base_value, dtype=dtype),
            requires_grad=False,
        )
        before = model.weight.detach().clone()
        adapter = Adapter(
            {
                "weight": ParameterDelta.from_tensors(
                    a=torch.ones(1, 1, dtype=dtype),
                    b=torch.ones(1, 1, dtype=dtype),
                    dense=-torch.ones(1, 1, dtype=dtype),
                )
            }
        )

        assert merge_adapter(model, [(adapter, 1.0)]) == 1
        torch.testing.assert_close(model.weight, before, rtol=0.0, atol=0.0)

    @CUDA
    def test_activation_merge_combines_lora_and_dense_then_restores_base(self) -> None:
        model = nn.Module()
        model.target = nn.Linear(4, 3, bias=True)
        model.requires_grad_(False)
        base_weight = model.target.weight.detach().clone()
        base_bias = model.target.bias.detach().clone()
        a = torch.randn(2, 4)
        b = torch.randn(3, 2)
        dense_weight = torch.randn(3, 4)
        dense_bias = torch.randn(3)
        strength = 0.375
        adapter = Adapter.from_state_dict(
            {
                "target.lora_A.weight": a,
                "target.lora_B.weight": b,
                "target.delta.weight": dense_weight,
                "target.delta.bias": dense_bias,
            }
        )
        x = torch.randn(2, 4)
        expected = F.linear(
            x,
            base_weight + strength * (b @ a + dense_weight),
            base_bias + strength * dense_bias,
        )
        offloader = ModelOffloader.from_module(model)

        offloader.activate(
            "cuda",
            adapters=[adapter],
            adapter_strengths=[strength],
            adapter_mode="merge",
        )
        try:
            actual = model.target(x.cuda()).cpu()
        finally:
            offloader.deactivate()

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(model.target.weight, base_weight)
        torch.testing.assert_close(model.target.bias, base_bias)

    def test_multiple_parameter_deltas_keep_independent_strengths(self) -> None:
        model = nn.Module()
        model.target = nn.Linear(3, 2, bias=False)
        model.requires_grad_(False)
        base = model.target.weight.detach().clone()
        first_dense = torch.randn_like(base)
        second_dense = torch.randn_like(base)
        a = torch.randn(1, 3)
        b = torch.randn(2, 1)
        first = Adapter.from_state_dict({"target.delta.weight": first_dense})
        second = Adapter.from_state_dict(
            {
                "target.lora_A.weight": a,
                "target.lora_B.weight": b,
                "target.delta.weight": second_dense,
            }
        )

        assert merge_adapter(model, [(first, 0.5), (second, 1.25)]) == 1

        torch.testing.assert_close(
            model.target.weight,
            base + 0.5 * first_dense + 1.25 * (b @ a + second_dense),
        )

    def test_plain_dense_merge_delegates_once_to_adapter_capability(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = nn.Linear(3, 2, bias=False)
        model.requires_grad_(False)
        before = model.weight.detach().clone()
        dense = torch.randn_like(before)
        adapter = Adapter(
            {
                "weight": ParameterDelta.from_tensors(
                    dense=dense,
                )
            }
        )
        calls: list[tuple[torch.Tensor, float, int | None]] = []
        original = RegularAdapter.merge_dense_

        def tracked_merge(
            target: torch.Tensor,
            update: torch.Tensor,
            alpha: float,
            *,
            beta: float = 1.0,
            rounding_seed: int | None = None,
        ) -> None:
            calls.append((update.detach().clone(), alpha, rounding_seed))
            original(
                target,
                update,
                alpha,
                beta=beta,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(
            RegularAdapter,
            "merge_dense_",
            staticmethod(tracked_merge),
        )

        assert merge_adapter(model, [(adapter, 0.25)]) == 1

        assert len(calls) == 1
        staged, strength, seed = calls[0]
        torch.testing.assert_close(staged, dense * 0.25)
        assert strength == 1.0
        assert seed == derive_seed("weight", 0)
        torch.testing.assert_close(model.weight, before + dense * 0.25)

    @pytest.mark.parametrize("with_lora", [False, True])
    def test_quantized_dense_merge_supports_dense_only_and_mixed_updates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        with_lora: bool,
    ) -> None:
        from piper_offload.int8_adapter import Int8Adapter
        from tests.test_int8_adapter import _make_affine_int8

        rows, cols, rank = 3, 4, 2
        qt = _make_affine_int8(torch.randn(rows, cols))
        model = nn.Module()
        model.target = nn.Module()
        model.target.weight = nn.Parameter(qt, requires_grad=False)
        before = qt.dequantize()
        dense = torch.randn(rows, cols)
        strength = -0.375
        state_dict = {"target.delta.weight": dense}
        low_rank = torch.zeros_like(dense)
        if with_lora:
            a = torch.randn(rank, cols)
            b = torch.randn(rows, rank)
            state_dict.update(
                {
                    "target.lora_A.weight": a,
                    "target.lora_B.weight": b,
                }
            )
            low_rank = b @ a
        adapter = Adapter.from_state_dict(state_dict)
        assert isinstance(Int8Adapter(), DenseMergeTensorAdapter)
        original_data_ptr = model.target.weight.data.qdata.data_ptr()
        original_scale_ptr = model.target.weight.data.scale.data_ptr()
        calls = 0
        original_dense_merge = Int8Adapter.merge_dense_

        def tracked_dense_merge(
            target: torch.Tensor,
            update: torch.Tensor,
            staged_alpha: float,
            *,
            beta: float = 1.0,
            rounding_seed: int | None = None,
        ) -> None:
            nonlocal calls
            calls += 1
            original_dense_merge(
                target,
                update,
                staged_alpha,
                beta=beta,
                rounding_seed=rounding_seed,
            )

        def unexpected_lora_merge(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("mixed update used the factorized merge path")

        monkeypatch.setattr(
            Int8Adapter,
            "merge_dense_",
            staticmethod(tracked_dense_merge),
        )
        monkeypatch.setattr(
            Int8Adapter,
            "merge_lora_",
            staticmethod(unexpected_lora_merge),
        )

        assert (
            merge_adapter(
                model,
                [(adapter, strength)],
                stochastic_rounding=False,
            )
            == 1
        )

        expected = Int8Adapter.requantize(
            before + strength * (dense + low_rank),
            like=qt,
        )
        assert calls == 1
        assert model.target.weight.data.qdata.data_ptr() == original_data_ptr
        assert model.target.weight.data.scale.data_ptr() == original_scale_ptr
        torch.testing.assert_close(model.target.weight.data.qdata, expected.qdata)
        torch.testing.assert_close(model.target.weight.data.scale, expected.scale)

    def test_zero_strength_dense_delta_is_inactive(self) -> None:
        model = nn.Linear(3, 2, bias=False)
        model.requires_grad_(False)
        base = model.weight.detach().clone()
        adapter = Adapter(
            {
                "weight": ParameterDelta.from_tensors(
                    dense=torch.randn_like(base),
                )
            }
        )

        assert merge_adapter(model, [(adapter, 0.0)]) == 0
        torch.testing.assert_close(model.weight, base)

    def test_dense_delta_rejects_float8_target_before_mutation(self) -> None:
        model = nn.Module()
        model.first = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        model.second = nn.Parameter(
            torch.zeros(2, 2, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        first_before = model.first.detach().clone()
        second_before = model.second.detach().clone()
        adapter = Adapter(
            {
                "first": ParameterDelta.from_tensors(dense=torch.ones(2, 2)),
                "second": ParameterDelta.from_tensors(dense=torch.ones(2, 2)),
            }
        )

        with pytest.raises(ValueError, match="do not support float8 targets"):
            merge_adapter(model, [(adapter, 1.0)])

        torch.testing.assert_close(model.first, first_before)
        assert torch.equal(model.second, second_before)

    def test_dense_delta_requires_broadcastable_shape(self) -> None:
        model = nn.Linear(3, 2, bias=False)
        model.requires_grad_(False)
        base = model.weight.detach().clone()
        adapter = Adapter(
            {
                "weight": ParameterDelta.from_tensors(
                    dense=torch.randn(3, 2),
                )
            }
        )

        with pytest.raises(RuntimeError, match="broadcast"):
            merge_adapter(model, [(adapter, 1.0)])
        torch.testing.assert_close(model.weight, base)

    def test_dense_delta_rejects_meta_target(self) -> None:
        with torch.device("meta"):
            model = nn.Linear(3, 2, bias=False)
        model.requires_grad_(False)
        adapter = Adapter(
            {
                "weight": ParameterDelta.from_tensors(
                    dense=torch.randn(2, 3),
                )
            }
        )

        with pytest.raises(ValueError, match="existing plain floating-point target"):
            merge_adapter(model, [(adapter, 1.0)])
        assert model.weight.is_meta

    def test_value_with_dense_delta_still_rejects_physical_base(self) -> None:
        model = nn.Linear(3, 2, bias=False)
        model.requires_grad_(False)
        base = model.weight.detach().clone()
        delta = Adapter(
            {
                "weight": ParameterDelta.from_tensors(
                    dense=torch.randn_like(base),
                )
            }
        )
        value = Adapter.from_state_dict({"weight": torch.randn_like(base)})

        with pytest.raises(ValueError, match="plain floating-point meta target"):
            merge_adapter(model, [(delta, 1.0), (value, 1.0)])
        torch.testing.assert_close(model.weight, base)

    def test_dense_delta_rejects_routed_mode(self) -> None:
        model = nn.Linear(3, 2, bias=False)
        model.requires_grad_(False)
        offloader = ModelOffloader.from_module(model)
        adapter = Adapter(
            {
                "weight": ParameterDelta.from_tensors(
                    dense=torch.randn(2, 3),
                )
            }
        )

        with pytest.raises(ValueError, match="does not support dense parameter deltas"):
            offloader.activate("cpu", adapters=[adapter], adapter_mode="routed")

        assert offloader.active_device is None
        assert offloader._routed_hook_removers == []

    def test_delta_transform_validates_and_reapplies_to_existing_weight(self) -> None:
        target = nn.Parameter(torch.full((2, 3), 3.0), requires_grad=False)
        delta = ParameterDelta.from_tensors(
            dense=torch.ones_like(target),
        )
        transform = ParameterDeltaTransform([delta.scaled(0.5)])

        with pytest.raises(RuntimeError, match="must be validated"):
            transform.apply_parameter(target)

        transform.validate_parameter(target)
        for expected in (3.5, 4.0):
            transform.apply_parameter(target)
            torch.testing.assert_close(target, torch.full_like(target, expected))
        sources = transform.storage_tensors()
        assert len(sources) == 1
        torch.testing.assert_close(sources[0], torch.ones_like(target))

    def test_delta_transform_requires_an_additive_contribution(self) -> None:
        with pytest.raises(ValueError, match="requires a delta"):
            ParameterDeltaTransform([])


# ---------------------------------------------------------------------------
# Adapter request validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("strengths", "expected"),
    [
        (None, [("a", 1.0), ("b", 1.0), ("a", 1.0)]),
        ([0.25, -0.5, 0.75], [("a", 0.25), ("b", -0.5), ("a", 0.75)]),
        ([0.0, 0.25, -0.0], [("b", 0.25)]),
    ],
)
def test_adapter_strengths_preserve_active_occurrences(strengths, expected) -> None:
    assert resolve_adapter_strengths(["a", "b", "a"], strengths) == expected


class TestActivationLoraValidation:
    def test_strength_mismatch_releases_activation_claim(self) -> None:
        model = nn.Linear(1, 1)
        offloader = ModelOffloader.from_module(model)
        adapter = Adapter.from_state_dict({"weight": torch.ones(1, 1)})
        with pytest.raises(ValueError, match="shorter"):
            offloader.activate("cpu", adapters=[adapter], adapter_strengths=[])
        with activated_model(offloader, "cpu") as active:
            assert active is model

    def test_zero_strength_merge_activation_installs_no_hooks(self) -> None:
        m = _make_bf16_model().to(torch.float32)
        s = _make_strategy(m)
        lora = _make_lora(4, 16)
        x = torch.randn(2, 16)
        expected = m(x)

        s.activate(
            "cpu",
            adapters=[lora],
            adapter_strengths=[0.0],
            adapter_mode="merge",
            stochastic_rounding=True,
        )
        try:
            assert s._routed_hook_removers == []
            torch.testing.assert_close(m(x), expected)
        finally:
            s.deactivate()

    def test_invalid_lora_mode_releases_activation_claim(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        with pytest.raises(ValueError, match="adapter_mode"):
            s.activate("cpu", adapter_mode="invalid")  # type: ignore[arg-type]

        with activated_model(s, "cpu") as active:
            assert active is m

    def test_routed_mode_ignores_stochastic_rounding(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16)

        with activated_model(
            s,
            "cpu",
            adapters=[lora],
            adapter_mode="routed",
            stochastic_rounding=True,
        ) as active:
            assert active is m

    def test_target_shape_mismatch_is_rejected_before_hooks(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(8, 4),
        }
        with pytest.raises(RuntimeError, match="broadcast"):
            s.activate("cuda:0", adapters=[Adapter.from_state_dict(sd)])
        assert s.active_device is None

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_merge_plan_matches_exact_block_and_resident_targets(self, dtype: torch.dtype) -> None:
        m = _make_bf16_model().to(dtype)
        sd = _make_lora_sd(4, 16)
        sd["embed.lora_A.weight"] = torch.randn(4, 16)
        sd["embed.lora_B.weight"] = torch.randn(16, 4)
        lora = Adapter.from_state_dict(sd)
        overrides = _plan_lora_merge(_make_strategy(m), lora)
        assert overrides.keys() == lora.targets.keys()
        assert all(override.update is not None for override in overrides.values())

    def test_merge_load_plans_apply_one_update_to_both_aliases(self) -> None:
        model = _make_tied_non_block_model(num_blocks=1, dtype=torch.bfloat16)
        block = model.transformer_blocks[0]
        block.alias = block.attn
        offloader = _make_strategy(model)
        delta = ParameterDelta.from_tensors(a=torch.ones(1, 16), b=torch.ones(16, 1))

        resident_overrides = _plan_lora_merge(offloader, Adapter({"head.weight": delta}))
        resident = host_component(offloader)
        resident_plan = resident._instance.resolve_load_plan(resident_overrides)
        block_overrides = _plan_lora_merge(offloader, Adapter({"transformer_blocks.0.alias.weight": delta}))
        blocks, = block_components(offloader)
        block_plan, = blocks._resolve_load_plans(block_overrides)
        for plan, target, alias in (
            (resident_plan, "head.weight", "embed.weight"),
            (block_plan, "alias.weight", "attn.weight"),
        ):
            update = plan.loads[target].update
            assert update is not None
            assert plan.loads[alias].update is update

    @CUDA
    @pytest.mark.parametrize("target", ["embed", "head"])
    def test_merge_through_either_tied_name_updates_both(self, target: str) -> None:
        m = _make_tied_non_block_model(dtype=torch.bfloat16)
        base = m.embed.weight.detach().clone()
        lora = Adapter.from_state_dict({
            f"{target}.lora_A.weight": torch.ones(1, 16),
            f"{target}.lora_B.weight": torch.ones(16, 1),
        })
        with activated_model(_make_strategy(m), "cuda", adapters=[lora]):
            assert m.head.weight is m.embed.weight
            torch.testing.assert_close(m.embed.weight.cpu(), base + 1)
        torch.testing.assert_close(m.embed.weight, base)

    def test_rejects_duplicate_tied_alias_targets(self) -> None:
        m = _make_tied_non_block_model(dtype=torch.bfloat16)
        s = _make_strategy(m)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
            "head.lora_A.weight": torch.randn(4, 16),
            "head.lora_B.weight": torch.randn(16, 4),
        }
        base = m.embed.weight.detach().clone()
        with pytest.raises(ValueError, match="Tied parameter aliases"):
            s.activate("cuda:0", adapters=[Adapter.from_state_dict(sd)])
        assert s.active_device is None
        torch.testing.assert_close(m.embed.weight, base)
        assert m.embed.weight is m.head.weight

    @CUDA
    def test_streamed_block_shared_submodule_alias_target_matched(self) -> None:
        m = _make_bf16_model(num_blocks=2)
        for block in m.transformer_blocks:
            block.ff = block.attn
        base = m.transformer_blocks[0].attn.weight.detach().clone()
        lora = Adapter.from_state_dict({
            "transformer_blocks.0.ff.lora_A.weight": torch.ones(1, 16),
            "transformer_blocks.0.ff.lora_B.weight": torch.ones(16, 1),
        })
        with activated_model(_make_strategy(m), "cuda", adapters=[lora]):
            block = m.transformer_blocks[0]
            assert block.attn.weight is block.ff.weight
            torch.testing.assert_close(block.attn.weight.cpu(), base + 1)

    def test_prefixed_keys_rejected(self) -> None:
        # Keys are verbatim, so a ``diffusion_model.``-prefixed adapter does not
        # match the model's params — the caller must strip it before building.
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16, prefix="diffusion_model.")
        with pytest.raises(ValueError, match="not parameters in the model"):
            s.activate("cpu", adapters=[lora])
        assert s.active_device is None

    def test_partial_targets_apply_intersection(self) -> None:
        m = _make_bf16_model(num_blocks=2).float()
        s = _make_strategy(m)
        sd = _make_lora_sd(num_blocks=1, dim=16)
        sd.update(_make_lora_sd(num_blocks=1, dim=16, prefix="missing."))
        lora = Adapter.from_state_dict(sd, allow_partial_targets=True)
        assert set(_plan_lora_merge(s, lora)) == {"transformer_blocks.0.attn.weight"}

        x = torch.randn(2, 16)
        expected = _expected_routed_output(m, x, [(lora, 1.0)])
        with activated_model(s, "cpu", adapters=[lora], adapter_mode="routed"):
            torch.testing.assert_close(m(x), expected)

    @pytest.mark.parametrize("mode", ["merge", "routed"])
    def test_partial_zero_overlap_is_activation_noop(self, mode: AdapterMode) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16)
        s = _make_strategy(m)
        lora = Adapter.from_state_dict(
            _make_lora_sd(num_blocks=1, dim=16, prefix="missing."),
            allow_partial_targets=True,
        )
        value = torch.randn(2, 16, dtype=torch.bfloat16)

        expected = m(value)
        s.activate("cpu", adapters=[lora], adapter_mode=mode)
        try:
            torch.testing.assert_close(m(value), expected)
        finally:
            s.deactivate()

    def test_target_keys_are_not_canonicalized(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {
            "transformer_blocks.0.attn.base_layer.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.attn.base_layer.lora_B.weight": torch.randn(16, 4),
        }
        with pytest.raises(ValueError, match="not parameters in the model"):
            s.activate("cpu", adapters=[Adapter.from_state_dict(sd)])
        assert s.active_device is None

    def test_merge_mode_activation_rejects_cpu(self) -> None:
        s = _make_strategy(_make_bf16_model())
        with pytest.raises(ValueError, match="merge mode requires CUDA"):
            s.activate("cpu", adapters=[_make_lora(4, 16)])

    @CUDA
    def test_merge_updates_weights_without_routed_hooks(self) -> None:
        m = _make_bf16_model(num_blocks=1).float()
        lora = _make_lora(1, 16)
        attn = m.transformer_blocks[0].attn
        expected = _expected_merged_weight(attn.weight, [(lora, 1.0)], 0, "attn.weight")
        with activated_model(_make_strategy(m), "cuda", adapters=[lora]):
            assert not attn._forward_pre_hooks
            assert not attn._forward_hooks
            torch.testing.assert_close(attn.weight.cpu(), expected)

    def test_accepts_int8_target_in_merge_and_routed_modes(self) -> None:
        from tests.test_int8_adapter import _make_affine_int8

        m = _make_bf16_model()
        weight = _make_affine_int8(torch.randn(16, 16, dtype=torch.bfloat16))
        m.embed.weight = nn.Parameter(weight, requires_grad=False)

        s = _make_strategy(m)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
        }
        adapter = Adapter.from_state_dict(sd)
        assert _plan_lora_merge(s, adapter)["embed.weight"].update is not None
        x = torch.eye(16, dtype=torch.bfloat16)
        a, b = _factor_tensors(adapter.targets["embed.weight"])
        expected = m.embed(x) + (x @ a.to(x).T) @ b.to(x).T
        with activated_model(s, "cpu", adapters=[adapter], adapter_mode="routed"):
            torch.testing.assert_close(m.embed(x), expected)

    def test_merge_mode_rejects_non_floating_dtype_before_hooks(self) -> None:
        m = _make_bf16_model()
        m.embed.weight = nn.Parameter(
            torch.zeros(16, 16, dtype=torch.int32),
            requires_grad=False,
        )
        s = _make_strategy(m)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
        }
        with pytest.raises(ValueError, match="floating-point compute dtype"):
            s.activate("cuda:0", adapters=[Adapter.from_state_dict(sd)])
        assert s.active_device is None

    def test_accepts_fp16_base(self) -> None:
        m = _make_bf16_model().to(torch.float16)
        for p in m.parameters():
            p.requires_grad = False
        s = _make_strategy(m)
        assert "embed.weight" in s.param_names

    def test_routed_mode_cpu_activation_uses_hooks(self) -> None:
        m = _make_bf16_model(num_blocks=2).to(torch.float32)
        for p in m.parameters():
            p.requires_grad = False
        lora = _make_lora(2, 16, seed=9)
        s = _make_strategy(m)
        x = torch.randn(2, 16)
        base = m.transformer_blocks[0].attn.weight.detach().clone()
        expected = _expected_routed_output(m, x, [(lora, 0.75)])
        with activated_model(s, "cpu", adapters=[lora], adapter_strengths=[0.75], adapter_mode="routed"):
            torch.testing.assert_close(m(x), expected, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(m.transformer_blocks[0].attn.weight, base, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @CUDA
    def test_reactivation_switches_adapters_and_restores_base(self) -> None:
        model = _make_bf16_model(num_blocks=4, dim=16)
        base_embed = model.embed.weight.detach().clone()
        base_block = model.transformer_blocks[0].attn.weight.detach().clone()
        adapters = []
        for seed in (1, 2):
            state = _make_lora_sd(num_blocks=4, dim=16, seed=seed)
            generator = torch.Generator().manual_seed(303 + seed)
            state["embed.lora_A.weight"] = torch.randn(4, 16, generator=generator)
            state["embed.lora_B.weight"] = torch.randn(16, 4, generator=generator)
            adapters.append(Adapter.from_state_dict(state))

        strategy = _make_strategy(model)
        previous_merged = None
        x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
        # First use, adapter replacement, and return to the pristine base.
        for loras in ([], [adapters[0]], [adapters[1]], []):
            with activated_model(strategy, "cuda", adapters=loras):
                assert model.embed.weight.is_cuda
                assert model.head.weight.is_cuda
                embed = model.embed.weight.detach().cpu()
                block = model.transformer_blocks[0].attn.weight.detach().cpu()
                if loras:
                    assert not torch.equal(embed, base_embed)
                    assert not torch.equal(block, base_block)
                    if previous_merged is not None:
                        assert not torch.equal(block, previous_merged)
                    previous_merged = block
                else:
                    torch.testing.assert_close(embed, base_embed, rtol=0, atol=0)
                    torch.testing.assert_close(block, base_block, rtol=0, atol=0)
                # Traverse every block, including the initial base-only use.
                output = x
                for component in model.transformer_blocks:
                    output = component(output)
                torch.cuda.synchronize()
                if not loras:
                    torch.testing.assert_close(
                        model.transformer_blocks[0].attn.weight.detach().cpu(),
                        base_block, rtol=0, atol=0,
                    )
            assert strategy.active_device is None
            for parameter in model.parameters():
                assert parameter.device == torch.device("cpu")
            # Streamed blocks may retain idle registrations; resident weights do not.
            assert not model.embed.weight.is_pinned()
            assert not model.head.weight.is_pinned()


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


class TestMergeCorrectness:
    @CUDA
    @pytest.mark.parametrize("mode", ["merge", "routed"])
    def test_transient_path_preserves_lora_on_reacquire(
        self,
        mode: AdapterMode,
    ) -> None:
        model = _make_bf16_model(num_blocks=2, dim=16)
        lora = Adapter.from_state_dict(
            {
                "embed.lora_A.weight": torch.randn(4, 16),
                "embed.lora_B.weight": torch.randn(16, 4),
            }
        )
        offloader = ModelOffloader.from_module(
            model,
            transient_paths=["embed"],
        )
        component = dict(transient_components(offloader))["embed"]
        with activated_model(offloader, "cuda", adapters=[lora], adapter_strengths=[0.5], adapter_mode=mode):
            value = torch.randn(
                2,
                16,
                dtype=torch.bfloat16,
                device="cuda",
            )
            with torch.inference_mode():
                first = model(value).clone()
                second = model(value).clone()
            torch.cuda.synchronize()
            assert component._lease is not None

        torch.testing.assert_close(second, first, rtol=0, atol=0)

    @CUDA
    @pytest.mark.parametrize("mode", ["merge", "routed"])
    def test_transient_block_path_preserves_lora_on_reacquire(
        self,
        mode: AdapterMode,
    ) -> None:
        model = _make_bf16_model(num_blocks=2, dim=16)
        lora = _make_lora(num_blocks=2, dim=16, seed=9)
        offloader = ModelOffloader.from_module(
            model,
            transient_block_paths=["transformer_blocks"],
        )
        with activated_model(offloader, "cuda", adapters=[lora], adapter_strengths=[0.5], adapter_mode=mode):
            value = torch.randn(
                2,
                16,
                dtype=torch.bfloat16,
                device="cuda",
            )
            with torch.inference_mode():
                first = model(value).clone()
                second = model(value).clone()
            torch.cuda.synchronize()

        torch.testing.assert_close(second, first, rtol=0, atol=0)

    @CUDA
    @pytest.mark.parametrize("streamed", [False, True])
    def test_dense_bias_delta_merge_resident_and_streamed(
        self,
        streamed: bool,
    ) -> None:
        m = _make_bf16_model(
            num_blocks=2,
            dim=16,
            attn_bias=True,
        )
        base_biases = [block.attn.bias.detach().clone() for block in m.transformer_blocks]

        def make_adapter(seed: int) -> Adapter:
            state_dict = _make_lora_sd(
                num_blocks=2,
                dim=16,
                seed=seed,
            )
            generator = torch.Generator().manual_seed(seed + 100)
            for block_idx in range(2):
                state_dict[f"transformer_blocks.{block_idx}.attn.delta.bias"] = torch.randn(
                    16,
                    generator=generator,
                )
            return Adapter.from_state_dict(state_dict)

        loras = [
            (make_adapter(10), 0.5),
            (make_adapter(20), -0.25),
        ]
        strategy = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"] if streamed else [],
        )
        adapters, strengths = zip(*loras, strict=True)
        with activated_model(strategy, "cuda", adapters=adapters, adapter_strengths=strengths):
            streamer = block_components(strategy)[0] if streamed else None
            for block_idx, block in enumerate(m.transformer_blocks):
                if streamer is not None:
                    streamer._runtime._before_block_forward(block_idx)
                actual = block.attn.bias
                assert actual is not None
                expected = base_biases[block_idx].to(actual.device)
                delta = torch.zeros_like(expected)
                target_name = f"transformer_blocks.{block_idx}.attn.bias"
                for lora, strength in loras:
                    target = lora.targets[target_name]
                    assert isinstance(target, ParameterDelta)
                    assert target.dense is not None
                    bias = target.dense.make_cpu_param().data
                    delta.add_(
                        bias.to(device=actual.device, dtype=actual.dtype),
                        alpha=strength,
                    )
                torch.testing.assert_close(actual, expected + delta)

    @CUDA
    def test_merged_weights_match_manual_baseline(self) -> None:
        m = _make_bf16_model(num_blocks=4, dim=16)
        captured_base = {i: m.transformer_blocks[i].attn.weight.detach().clone() for i in range(4)}

        loras = [
            (_make_lora(num_blocks=4, dim=16, seed=10), 0.5),
            (_make_lora(num_blocks=4, dim=16, seed=20), 0.25),
        ]
        s = _make_strategy(m)
        adapters, strengths = zip(*loras, strict=True)
        with activated_model(s, "cuda", adapters=adapters, adapter_strengths=strengths):
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            for i in range(4):
                _ = m.transformer_blocks[i](torch.randn(2, 16, dtype=torch.bfloat16, device="cuda"))
                torch.cuda.synchronize()
                expected = _expected_merged_weight(
                    captured_base[i].to("cuda"),
                    loras,
                    i,
                    "attn.weight",
                )
                actual = m.transformer_blocks[i].attn.weight.detach()
                assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), (
                    f"block {i} merged weight mismatch:\n"
                    f"  expected: {expected.flatten()[:4]}\n"
                    f"  actual:   {actual.flatten()[:4]}"
                )

    @CUDA
    def test_non_block_lora_merges_correctly(self) -> None:
        """Adapter targeting embed (non-block) should be merged at activate."""
        m = _make_bf16_model(num_blocks=4, dim=16)
        captured_embed = m.embed.weight.detach().clone()

        g = torch.Generator().manual_seed(99)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16, generator=g, dtype=torch.float32),
            "embed.lora_B.weight": torch.randn(16, 4, generator=g, dtype=torch.float32),
        }
        lora = Adapter.from_state_dict(state_dict=sd)
        s = _make_strategy(m)
        with activated_model(s, "cuda", adapters=[lora], adapter_strengths=[0.5]):
            factor = lora.targets["embed.weight"]
            a, b = _factor_tensors(factor)
            expected = (captured_embed + 0.5 * (b.to(torch.bfloat16) @ a.to(torch.bfloat16))).to("cuda")
            actual = m.embed.weight.detach()
            assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), (
                f"non-block merge mismatch:\n  expected: {expected.flatten()[:4]}\n  actual:   {actual.flatten()[:4]}"
            )

    @CUDA
    def test_base_layer_named_target_merges_with_exact_keys(self) -> None:
        """A model whose weight lives at a nested ``.base_layer.`` path (e.g.
        PEFT-wrapped) merges when the Adapter keys that exact path. The offloader
        does no key remapping — the caller builds keys that match the model."""

        class PEFTBlock(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.attn = nn.Module()
                self.attn.base_layer = nn.Linear(dim, dim, bias=False)
                self.ff = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.ff(self.attn.base_layer(x))

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer_blocks = nn.ModuleList([PEFTBlock(16) for _ in range(4)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for blk in self.transformer_blocks:
                    x = blk(x)
                return x

        m = M().to(torch.bfloat16)
        m.requires_grad_(False)

        captured_base = {i: m.transformer_blocks[i].attn.base_layer.weight.detach().clone() for i in range(4)}

        # Keys match the model's real paths, ``.base_layer.`` and all.
        g = torch.Generator().manual_seed(42)
        sd: dict[str, torch.Tensor] = {}
        for b in range(4):
            base = f"transformer_blocks.{b}.attn.base_layer"
            sd[f"{base}.lora_A.weight"] = torch.randn(4, 16, generator=g)
            sd[f"{base}.lora_B.weight"] = torch.randn(16, 4, generator=g)
        lora = Adapter.from_state_dict(state_dict=sd)

        s = _make_strategy(m)
        with activated_model(s, "cuda", adapters=[lora], adapter_strengths=[0.7]):
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            m(x)
            torch.cuda.synchronize()
            for i in range(4):
                _ = m.transformer_blocks[i](torch.randn(2, 16, dtype=torch.bfloat16, device="cuda"))
                torch.cuda.synchronize()
                expected = _expected_merged_weight(
                    captured_base[i].to("cuda"),
                    [(lora, 0.7)],
                    i,
                    "attn.base_layer.weight",
                )
                actual = m.transformer_blocks[i].attn.base_layer.weight.detach()
                assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), f"block {i} base_layer merge mismatch"


class TestLoRATransform:
    @pytest.mark.parametrize("beta", [float("nan"), float("inf"), -float("inf")])
    def test_rejects_non_finite_base_coefficient(self, beta: float) -> None:
        factor = ScaledLoRAFactor.from_tensors(torch.ones(1, 3), torch.ones(2, 1), 1.0)
        with pytest.raises(ValueError, match="base coefficient must be finite"):
            LoRATransform([factor], beta=beta)

    def test_stochastic_rounding_requires_target_key(self) -> None:
        factor = ScaledLoRAFactor.from_tensors(
            torch.randn(2, 8),
            torch.randn(4, 2),
            0.5,
        )

        with pytest.raises(ValueError, match="non-empty target_key"):
            LoRATransform([factor], stochastic_rounding=True)

    def test_validate_target_accepts_regular_tensor_without_mutation(self) -> None:
        param = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        before = param.detach().clone()
        a = torch.randn(2, 8)
        b = torch.randn(4, 2)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.5)])

        transform.validate_target(param)

        torch.testing.assert_close(param, before)

    def test_validate_target_rejects_shape_mismatch(self) -> None:
        param = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        a = torch.randn(2, 8)
        b = torch.randn(3, 2)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.5)])

        with pytest.raises(RuntimeError, match="broadcast"):
            transform.validate_target(param)

    def test_weight_application_requires_validation(self) -> None:
        param = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        transform = LoRATransform(
            [
                ScaledLoRAFactor.from_tensors(
                    torch.randn(2, 8),
                    torch.randn(4, 2),
                    0.5,
                )
            ]
        )

        with pytest.raises(RuntimeError, match="validated before application"):
            transform.apply_weight(param)

    def test_weight_application_reuses_validation_plan(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        param = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        before = param.detach().clone()
        a = torch.randn(2, 8)
        b = torch.randn(4, 2)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.5)])
        transform.validate_weight_target(param)

        def fail_validation(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("application repeated validation")

        monkeypatch.setattr(
            lora_impl,
            "_select_lora_merge_adapter",
            fail_validation,
        )
        monkeypatch.setattr(
            lora_impl,
            "_validate_factor_shapes",
            fail_validation,
        )
        monkeypatch.setattr(
            lora_impl,
            "_validate_lora_merge",
            fail_validation,
        )

        transform.apply_weight(param)

        expected = before.clone()
        expected.addmm_(b, a, alpha=0.5)
        torch.testing.assert_close(param, expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_regular_transform_mutates_param_in_place(
        self,
        dtype: torch.dtype,
    ) -> None:
        param = nn.Parameter(torch.randn(4, 8, dtype=dtype), requires_grad=False)
        before = param.detach().clone()
        a = torch.randn(2, 8, dtype=dtype)
        b = torch.randn(4, 2, dtype=dtype)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.5)])

        transform.validate_target(param)
        transform.apply(param)

        expected = before.clone()
        expected.addmm_(b, a, alpha=0.5)
        torch.testing.assert_close(param, expected)

    @CUDA
    def test_multiple_cuda_factors_use_one_packed_merge(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        param = nn.Parameter(
            torch.randn(12, 16, device="cuda"),
            requires_grad=False,
        )
        before = param.detach().clone()
        factor_inputs = [
            (torch.randn(2, 16), torch.randn(12, 2), 0.75),
            (torch.randn(3, 16), torch.randn(12, 3), -0.25),
            (torch.randn(1, 16), torch.randn(12, 1), 1.0),
        ]
        factors = [ScaledLoRAFactor.from_tensors(a, b, strength) for a, b, strength in factor_inputs]
        transform = LoRATransform(factors)
        merge_calls = 0
        original = RegularAdapter.merge_lora_

        def tracked_merge(
            target: torch.Tensor,
            b: torch.Tensor,
            a: torch.Tensor,
            alpha: float,
            *,
            beta: float = 1.0,
            rounding_seed: int | None = None,
        ) -> None:
            nonlocal merge_calls
            merge_calls += 1
            original(
                target,
                b,
                a,
                alpha,
                beta=beta,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(
            RegularAdapter,
            "merge_lora_",
            staticmethod(tracked_merge),
        )

        transform.validate_target(param)
        transform.apply(param)

        expected = before.clone()
        for a, b, strength in factor_inputs:
            expected.addmm_(
                b.cuda(),
                a.cuda(),
                alpha=strength,
            )
        assert merge_calls == 1
        torch.testing.assert_close(param, expected, rtol=2e-5, atol=2e-5)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=CUDA),
        ],
    )
    @pytest.mark.parametrize("factor_count", [1, 2])
    def test_adapter_staging_uses_factor_shape_and_compute_dtype(
        self,
        device: str,
        factor_count: int,
    ) -> None:
        rows, cols = 12, 16
        factors = [
            ScaledLoRAFactor.from_tensors(
                torch.randn(rank, cols),
                torch.randn(rows, rank),
                0.5,
            )
            for rank in range(2, 2 + factor_count)
        ]
        transform = LoRATransform(factors)
        packed_representation = torch.empty(
            (rows * cols // 2, 1),
            device=device,
            dtype=torch.uint8,
        )

        staged = transform._stage_single_or_packed_update(
            packed_representation,
            transform._materialize_weight_factors(),
            compute_dtype=torch.float16,
        )

        b, a, _strength = staged
        total_rank = sum(factor.rank for factor in factors)
        assert b.shape == (rows, total_rank)
        assert a.shape == (total_rank, cols)
        assert b.device.type == device
        assert a.device.type == device
        assert b.dtype is torch.float16
        assert a.dtype is torch.float16

    def test_conversion_and_copy_capabilities_do_not_enable_merge(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class ConversionOnlyAdapter(Int4TilePackedAdapter):
            @staticmethod
            def dequantize(t: torch.Tensor) -> torch.Tensor:
                return t

            @staticmethod
            def requantize(
                t: torch.Tensor,
                *,
                like: torch.Tensor,
                rounding_seed: int | None = None,
            ) -> torch.Tensor:
                del like, rounding_seed
                return t

            @staticmethod
            def copy_into(
                src: torch.Tensor,
                *,
                target: torch.Tensor,
            ) -> None:
                target.copy_(src)

        adapter = ConversionOnlyAdapter()
        assert isinstance(adapter, DequantRequantTensorAdapter)
        assert isinstance(adapter, TensorCopyIntoAdapter)
        assert not isinstance(adapter, LoRAMergeTensorAdapter)
        monkeypatch.setattr(
            lora_impl,
            "select_adapter",
            lambda _data: adapter,
        )

        param = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        transform = LoRATransform(
            [
                ScaledLoRAFactor.from_tensors(
                    torch.randn(2, 8),
                    torch.randn(4, 2),
                    0.5,
                )
            ]
        )

        with pytest.raises(ValueError, match="does not support LoRA merge"):
            transform.validate_target(param)

    def test_non_floating_compute_dtype_raises_on_validation(self) -> None:
        param = nn.Parameter(torch.zeros(4, 8, dtype=torch.int32), requires_grad=False)
        a = torch.randn(2, 8)
        b = torch.randn(4, 2)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.5)])

        with pytest.raises(ValueError, match="floating-point compute dtype"):
            transform.validate_target(param)

class TestParameterValueActivation:
    def test_routed_mode_rejects_parameter_values_before_activation(self) -> None:
        model = _make_bf16_model(num_blocks=1, dim=4).to(torch.float32)
        offloader = _make_strategy(model)
        lora = Adapter.from_state_dict(
            {
                "transformer_blocks.0.attn.weight": torch.randn(4, 4),
            }
        )

        with pytest.raises(ValueError, match="does not support parameter values"):
            offloader.activate("cpu", adapters=[lora], adapter_mode="routed")

        assert offloader.active_device is None
        assert offloader._routed_hook_removers == []

    @CUDA
    def test_parameter_value_rejects_physical_base(self) -> None:
        model = _make_bf16_model(num_blocks=1, dim=4).to(torch.float32)
        offloader = _make_strategy(model)
        base = model.transformer_blocks[0].attn.weight.detach().clone()
        value = torch.randn_like(base)
        lora = Adapter.from_state_dict(
            {
                "transformer_blocks.0.attn.weight": value,
            }
        )

        with pytest.raises(ValueError, match="floating-point meta target"):
            offloader.activate(
                "cuda",
                adapters=[lora],
                adapter_strengths=[0.25],
            )

        torch.testing.assert_close(
            model.transformer_blocks[0].attn.weight,
            base,
        )

    @CUDA
    def test_meta_parameter_is_storage_free_until_parameter_value_is_active(self) -> None:
        with torch.device("meta"):
            model = nn.Linear(3, 2, bias=False)
        model.requires_grad_(False)
        offloader = ModelOffloader.from_module(model)

        assert offloader.cache_bytes == 0
        offloader.activate("cuda")
        try:
            assert model.weight.is_meta
            resident = offloader._composite.resident
            assert resident is not None
            assert resident._lease is not None
            assert resident._lease.target.param_targets == {}
        finally:
            offloader.deactivate()

        value = torch.randn(2, 3)
        lora = Adapter.from_state_dict({"weight": value})
        offloader.activate(
            "cuda",
            adapters=[lora],
            adapter_strengths=[-0.5],
        )
        try:
            assert model.weight.device.type == "cuda"
            torch.testing.assert_close(model.weight.cpu(), value)
        finally:
            offloader.deactivate()

        assert model.weight.is_meta

    @CUDA
    def test_parameter_value_does_not_scale_by_default(self) -> None:
        with torch.device("meta"):
            model = nn.Linear(3, 2, bias=False)
        model.requires_grad_(False)
        offloader = ModelOffloader.from_module(model)
        value = torch.randn(2, 3)
        adapter = Adapter.from_state_dict({"weight": value})

        offloader.activate(
            "cuda",
            adapters=[adapter],
            adapter_strengths=[-0.5],
        )
        try:
            torch.testing.assert_close(model.weight.cpu(), value)
        finally:
            offloader.deactivate()

        assert model.weight.is_meta

    @CUDA
    @pytest.mark.parametrize("block_mode", ["resident", "streaming", "rolling", "auto"])
    def test_parameter_values_work_in_every_block_mode(
        self,
        block_mode: str,
    ) -> None:
        dim = 4

        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.empty(dim, dim),
                    requires_grad=False,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.linear(x, self.weight)

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([Block(), Block(), Block()])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = block(x)
                return x

        with torch.device("meta"):
            model = M()
        compile_config = (
            BlockCompileConfig(dynamic=False, fullgraph=True) if block_mode in {"rolling", "auto"} else None
        )
        offloader = ModelOffloader.from_module(
            model,
            block_paths=("blocks",),
            block_mode=block_mode,  # type: ignore[arg-type]
            block_compile=compile_config,
        )
        values = {f"blocks.{idx}.weight": torch.eye(dim) * (idx + 1) for idx in range(3)}
        lora = Adapter.from_state_dict(values)
        x = torch.ones(2, dim, device="cuda")

        assert offloader.cache_bytes == 0
        offloader.activate("cuda", adapters=[lora])
        try:
            torch.testing.assert_close(model(x).cpu(), x.cpu() * 6)
            component = block_components(offloader)[0]
            if component.block_mode == "rolling":
                runtime = component._active_runtime
                assert runtime is not None
                assert runtime._lease is not None
                assert len(runtime._lease.target.param_targets) == 1
        finally:
            offloader.deactivate()

        assert all(param.is_meta for param in model.parameters())

    @CUDA
    @pytest.mark.parametrize("block_mode", ["resident", "streaming", "rolling", "auto"])
    def test_inactive_meta_blocks_allocate_no_extra_slots(
        self,
        block_mode: str,
    ) -> None:
        dim = 4

        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.empty(dim, dim),
                    requires_grad=False,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.linear(x, self.weight)

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([Block(), Block(), Block()])

        with torch.device("meta"):
            model = M()
        compile_config = (
            BlockCompileConfig(dynamic=False, fullgraph=True) if block_mode in {"rolling", "auto"} else None
        )
        offloader = ModelOffloader.from_module(
            model,
            block_paths=("blocks",),
            block_mode=block_mode,  # type: ignore[arg-type]
            block_compile=compile_config,
        )
        value = torch.eye(dim)
        lora = Adapter.from_state_dict({"blocks.1.weight": value})
        x = torch.randn(2, dim, device="cuda")

        offloader.activate("cuda", adapters=[lora])
        try:
            torch.testing.assert_close(model.blocks[1](x).cpu(), x.cpu())
            component = block_components(offloader)[0]
            if component.block_mode == "rolling":
                # Rolling installs its one already-required shared slot on the
                # homogeneous block group. Inactive contents are unspecified,
                # but they consume no additional parameter storage.
                assert all(param.device.type == "cuda" for param in model.parameters())
                runtime = component._active_runtime
                assert runtime is not None
                assert runtime._lease is not None
                assert len(runtime._lease.target.param_targets) == 1
            else:
                assert model.blocks[0].weight.is_meta
                assert model.blocks[1].weight.device.type == "cuda"
                assert model.blocks[2].weight.is_meta
        finally:
            offloader.deactivate()

        assert all(param.is_meta for param in model.parameters())


class TestPermanentMerge:
    def test_parameter_value_rejects_physical_parameter(self) -> None:
        model = nn.Linear(3, 2, bias=False)
        model.requires_grad_(False)
        weight = model.weight
        base = model.weight.detach().clone()
        value = Adapter.from_state_dict({"weight": torch.randn_like(base)})

        with pytest.raises(ValueError, match="floating-point meta target"):
            merge_adapter(model, [(value, 0.25)])
        assert model.weight is weight
        torch.testing.assert_close(model.weight, base)

    def test_parameter_value_source_owns_dtype_and_range(self) -> None:
        source = torch.tensor([65520.0], dtype=torch.float32)
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.empty(source.shape, device="meta", dtype=torch.float16),
            requires_grad=False,
        )
        value = Adapter.from_state_dict({"weight": source})

        assert merge_adapter(model, [(value, 0.5)]) == 1
        assert model.weight.dtype is torch.float32
        torch.testing.assert_close(model.weight, source)

    def test_parameter_value_explicit_scaling_uses_source_compute_dtype(self) -> None:
        source = torch.tensor([40_000.0], dtype=torch.float32)
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.empty(source.shape, device="meta", dtype=torch.float16),
            requires_grad=False,
        )
        value = Adapter.from_state_dict(
            {"weight": source},
            scale_parameter_values=True,
        )

        assert merge_adapter(model, [(value, 2.0)]) == 1
        assert model.weight.dtype is torch.float32
        torch.testing.assert_close(model.weight, source * 2.0)

    def test_parameter_value_ignores_strength_for_range_and_materialization(self) -> None:
        source = torch.tensor([40_000.0], dtype=torch.float32)
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.empty(source.shape, device="meta", dtype=torch.float16),
            requires_grad=False,
        )
        value = Adapter.from_state_dict(
            {"weight": source},
        )

        assert merge_adapter(model, [(value, 0.0)]) == 0
        assert model.weight.is_meta
        assert merge_adapter(model, [(value, 2.0)]) == 1
        assert model.weight.dtype is torch.float32
        torch.testing.assert_close(model.weight, source)

    @pytest.mark.parametrize(
        "stride",
        [
            (0, 1),
            (1, 1),
            (3, 1),
        ],
        ids=["zero-stride-overlap", "nonzero-stride-overlap", "gapped"],
    )
    def test_parameter_value_ignores_placeholder_stride(
        self,
        stride: tuple[int, int],
    ) -> None:
        source = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.empty_strided((2, 2), stride, device="meta"),
            requires_grad=False,
        )
        value = Adapter.from_state_dict({"weight": source})

        assert merge_adapter(model, [(value, 1.0)]) == 1
        assert model.weight.is_contiguous()
        torch.testing.assert_close(model.weight, source)

    def test_parameter_value_ignores_placeholder_storage_offset(self) -> None:
        source = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        backing = torch.empty(8, device="meta")
        model = nn.Module()
        model.weight = nn.Parameter(
            backing.as_strided((2, 2), (2, 1), 1),
            requires_grad=False,
        )
        value = Adapter.from_state_dict({"weight": source})

        assert merge_adapter(model, [(value, 1.0)]) == 1
        assert model.weight.storage_offset() == 0
        torch.testing.assert_close(model.weight, source)

    def test_parameter_value_rejects_sparse_meta_layout(self) -> None:
        indices = torch.empty((2, 0), dtype=torch.int64, device="meta")
        values = torch.empty(0, device="meta")
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.sparse_coo_tensor(
                indices,
                values,
                (2, 2),
                device="meta",
                check_invariants=False,
            ),
            requires_grad=False,
        )
        value = Adapter.from_state_dict({"weight": torch.empty(2, 2)})

        with pytest.raises(ValueError, match="plain floating-point meta target"):
            merge_adapter(model, [(value, 1.0)])

        assert model.weight.is_meta
        assert model.weight.layout is torch.sparse_coo

    def test_parameter_value_rejects_physical_target_with_wrong_layout(self) -> None:
        source = torch.arange(6, dtype=torch.float32).view(2, 3)
        value = ParameterValue.from_tensor(source)
        transform = ParameterValueTransform(value.scaled(1.0))
        meta_target = nn.Parameter(
            torch.empty_strided((2, 3), (1, 2), device="meta"),
            requires_grad=False,
        )
        transform.validate_parameter(meta_target)

        wrong_dtype = nn.Parameter(
            torch.empty(2, 3, dtype=torch.float16),
            requires_grad=False,
        )
        with pytest.raises(RuntimeError, match="matching.*replacement source"):
            transform.apply_parameter(wrong_dtype)

        materialized = transform.materialize()
        assert materialized.is_contiguous()
        torch.testing.assert_close(materialized, source)

    def test_parameter_value_accepts_float8_source(self) -> None:
        source = torch.tensor([1.0], dtype=torch.float8_e4m3fn)
        value = ParameterValue.from_tensor(source)
        transform = ParameterValueTransform(value.scaled(0.5))
        transform.validate_parameter(
            nn.Parameter(torch.empty(1, device="meta"), requires_grad=False)
        )

        materialized = transform.materialize()
        assert materialized.dtype is torch.float8_e4m3fn
        assert torch.equal(materialized, source)

    def test_parameter_value_ignores_float8_placeholder_dtype(self) -> None:
        source = torch.tensor([1.0], dtype=torch.float32)
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.empty(source.shape, device="meta", dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        value = Adapter.from_state_dict({"weight": source})

        assert merge_adapter(model, [(value, 0.5)]) == 1
        assert model.weight.dtype is torch.float32
        torch.testing.assert_close(model.weight, source)

    def test_value_with_lora_still_rejects_physical_base(self) -> None:
        model = nn.Module()
        model.target = nn.Linear(3, 2, bias=False)
        model.requires_grad_(False)
        base = model.target.weight.detach().clone()
        a = torch.randn(1, 3)
        b = torch.randn(2, 1)
        lora = Adapter.from_state_dict(
            {
                "target.lora_A.weight": a,
                "target.lora_B.weight": b,
            }
        )
        value = Adapter.from_state_dict({"target.weight": torch.randn_like(base)})

        with pytest.raises(ValueError, match="plain floating-point meta target"):
            merge_adapter(model, [(lora, 0.5), (value, 0.5)])
        torch.testing.assert_close(model.target.weight, base)

    def test_multiple_parameter_values_cannot_share_target(self) -> None:
        with torch.device("meta"):
            model = nn.Linear(3, 2, bias=False)
        model.requires_grad_(False)
        first = Adapter.from_state_dict({"weight": torch.randn(2, 3)})
        second = Adapter.from_state_dict({"weight": torch.randn(2, 3)})

        with pytest.raises(ValueError, match="multiple active parameter values"):
            merge_adapter(model, [(first, 0.25), (second, -0.5)])
        assert model.weight.is_meta

    def test_meta_parameter_permanent_merge_materializes_cpu_aliases(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                shared = nn.Parameter(
                    torch.empty(2, 3, device="meta"),
                    requires_grad=False,
                )
                self.left = nn.Module()
                self.right = nn.Module()
                self.left.weight = shared
                self.right.weight = shared

        model = M()
        value = torch.randn(2, 3)
        lora = Adapter.from_state_dict({"left.weight": value})

        assert merge_adapter(model, [(lora, -0.25)]) == 1
        assert model.left.weight is model.right.weight
        assert model.left.weight.device.type == "cpu"
        assert not model.left.weight.requires_grad
        torch.testing.assert_close(model.left.weight, value)

    def test_low_rank_factor_cannot_materialize_meta_parameter(self) -> None:
        with torch.device("meta"):
            model = nn.Module()
            model.target = nn.Linear(3, 2, bias=False)
        model.requires_grad_(False)
        lora = Adapter.from_state_dict(
            {
                "target.lora_A.weight": torch.randn(1, 3),
                "target.lora_B.weight": torch.randn(2, 1),
            }
        )

        with pytest.raises(ValueError, match="only by parameter values"):
            merge_adapter(model, [(lora, 1.0)])
        assert model.target.weight.is_meta

    def test_parameter_value_rejects_tensor_subclass_before_mutation(self) -> None:
        class UnknownTensor(torch.Tensor):
            pass

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                wrapped = torch.Tensor._make_subclass(
                    UnknownTensor,
                    torch.randn(2, 3),
                    False,
                )
                self.weight = nn.Parameter(wrapped, requires_grad=False)

        model = M()
        before = model.weight.detach().clone()
        lora = Adapter.from_state_dict({"weight": torch.randn(2, 3)})

        with pytest.raises(ValueError, match="plain floating-point meta target"):
            merge_adapter(model, [(lora, 1.0)])
        torch.testing.assert_close(model.weight, before)

    @pytest.mark.parametrize("strengths", [(), (0.0,), (-0.0,)])
    def test_inactive_merge_skips_model_parameters(self, strengths, monkeypatch: pytest.MonkeyPatch) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16).to(torch.float32)
        before = {name: param.detach().clone() for name, param in m.named_parameters()}
        # Inactive LoRAs do not even require their target names to exist.
        lora = _make_lora(
            num_blocks=2,
            dim=16,
            prefix="missing.",
        )

        with monkeypatch.context() as patch:
            patch.setattr(m, "named_parameters", lambda **_: pytest.fail("inactive merge walked the model"))
            assert merge_adapter(m, [(lora, strength) for strength in strengths]) == 0
        for name, param in m.named_parameters():
            assert torch.equal(param, before[name])

    def test_can_share_lora_with_an_active_routed_use(self) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16).to(torch.float32)
        routed_model = _make_bf16_model(num_blocks=2, dim=16).to(torch.float32)
        routed = ModelOffloader.from_module(routed_model)
        lora = _make_lora(num_blocks=2, dim=16)
        before = m.transformer_blocks[0].attn.weight.detach().clone()
        expected = _expected_merged_weight(
            before,
            [(lora, 1.0)],
            0,
            "attn.weight",
        )

        routed.activate(
            "cpu",
            adapters=[lora],
            adapter_mode="routed",
        )
        try:
            assert merge_adapter(m, [(lora, 1.0)]) == 2
            assert routed_model(torch.randn(2, 16)).shape == (2, 16)
        finally:
            routed.deactivate()

        torch.testing.assert_close(m.transformer_blocks[0].attn.weight, expected)

    def test_rejects_unknown_targets_without_mutation(self) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16).to(torch.float32)
        before = {name: param.detach().clone() for name, param in m.named_parameters()}
        lora = _make_lora(
            num_blocks=2,
            dim=16,
            prefix="missing.",
        )

        with pytest.raises(ValueError, match="not parameters in the model"):
            merge_adapter(m, [(lora, 1.0)])

        for name, param in m.named_parameters():
            torch.testing.assert_close(param, before[name])

    def test_partial_targets_merge_intersection(self) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16).to(torch.float32)
        untouched = m.transformer_blocks[1].attn.weight.detach().clone()
        sd = _make_lora_sd(num_blocks=1, dim=16)
        sd.update(_make_lora_sd(num_blocks=1, dim=16, prefix="missing."))
        lora = Adapter.from_state_dict(sd, allow_partial_targets=True)
        before = m.transformer_blocks[0].attn.weight.detach().clone()
        expected = _expected_merged_weight(
            before,
            [(lora, 1.0)],
            0,
            "attn.weight",
        )

        assert merge_adapter(m, [(lora, 1.0)]) == 1

        torch.testing.assert_close(m.transformer_blocks[0].attn.weight, expected)
        torch.testing.assert_close(m.transformer_blocks[1].attn.weight, untouched)

    def test_partial_targets_permanent_merge_allows_zero_overlap(self) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16).to(torch.float32)
        before = {name: param.detach().clone() for name, param in m.named_parameters()}
        lora = Adapter.from_state_dict(
            _make_lora_sd(num_blocks=1, dim=16, prefix="missing."),
            allow_partial_targets=True,
        )

        assert merge_adapter(m, [(lora, 1.0)]) == 0
        for name, param in m.named_parameters():
            torch.testing.assert_close(param, before[name])

    def test_multiple_loras_on_one_target_count_one_parameter(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Linear(3, 3, bias=False)

        m = M()
        m.requires_grad_(False)
        base = m.target.weight.detach().clone()

        def make_lora(seed: int) -> Adapter:
            g = torch.Generator().manual_seed(seed)
            return Adapter.from_state_dict(
                {
                    "target.lora_A.weight": torch.randn(1, 3, generator=g),
                    "target.lora_B.weight": torch.randn(3, 1, generator=g),
                }
            )

        first = make_lora(1)
        second = make_lora(2)
        expected = base.clone()
        for lora in (first, second):
            a, b = _factor_tensors(lora.targets["target.weight"])
            expected.addmm_(b, a)

        assert merge_adapter(m, [(first, 1.0), (second, 1.0)]) == 1
        torch.testing.assert_close(m.target.weight, expected)

    def test_duplicate_lora_instances_each_contribute(self) -> None:
        model = _make_bf16_model(num_blocks=2, dim=16).to(torch.float32)
        lora = _make_lora(num_blocks=2, dim=16)
        before = model.transformer_blocks[0].attn.weight.detach().clone()
        contributions = [(lora, 0.25), (lora, 0.75)]
        expected = _expected_merged_weight(
            before,
            contributions,
            0,
            "attn.weight",
        )

        assert merge_adapter(model, contributions) == 2
        torch.testing.assert_close(
            model.transformer_blocks[0].attn.weight,
            expected,
        )

    def test_shape_preflight_prevents_partial_merge(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.first = nn.Linear(3, 3, bias=False)
                self.second = nn.Linear(3, 3, bias=False)

        m = M()
        m.requires_grad_(False)
        first_before = m.first.weight.detach().clone()
        second_before = m.second.weight.detach().clone()
        lora = Adapter.from_state_dict(
            {
                "first.lora_A.weight": torch.randn(1, 3),
                "first.lora_B.weight": torch.randn(3, 1),
                "second.lora_A.weight": torch.randn(1, 3),
                "second.lora_B.weight": torch.randn(2, 1),
            }
        )

        with pytest.raises(RuntimeError, match="broadcast"):
            merge_adapter(m, [(lora, 1.0)])

        torch.testing.assert_close(m.first.weight, first_before)
        torch.testing.assert_close(m.second.weight, second_before)

    def test_compute_dtype_preflight_prevents_partial_merge(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.first = nn.Linear(3, 3, bias=False)
                self.second = nn.Linear(3, 3, bias=False)
                self.second.weight = nn.Parameter(
                    torch.zeros(3, 3, dtype=torch.int32),
                    requires_grad=False,
                )

        m = M()
        m.requires_grad_(False)
        first_before = m.first.weight.detach().clone()
        second_before = m.second.weight.detach().clone()
        lora = Adapter.from_state_dict(
            {
                "first.lora_A.weight": torch.randn(1, 3),
                "first.lora_B.weight": torch.randn(3, 1),
                "second.lora_A.weight": torch.randn(1, 3),
                "second.lora_B.weight": torch.randn(3, 1),
            }
        )

        with pytest.raises(
            ValueError,
            match="floating-point compute dtype",
        ):
            merge_adapter(m, [(lora, 1.0)])

        torch.testing.assert_close(m.first.weight, first_before)
        torch.testing.assert_close(m.second.weight, second_before)

    def test_dense_bias_delta_merges_separately_from_quantized_weight(self) -> None:
        from piper_offload.int8_adapter import Int8Adapter
        from tests.test_int8_adapter import _make_affine_int8

        rows, cols, rank = 4, 8, 2
        weight = _make_affine_int8(torch.randn(rows, cols))
        model = nn.Linear(cols, rows).requires_grad_(False)
        model.weight = nn.Parameter(weight, requires_grad=False)
        bias = model.bias
        bias_before = bias.detach().clone()
        bias_delta = torch.randn(rows)
        a = torch.randn(rank, cols)
        b = torch.randn(rows, rank)
        adapter = Adapter({
            "weight": ParameterDelta.from_tensors(a=a, b=b),
            "bias": ParameterDelta.from_tensors(dense=bias_delta),
        })
        strength = 0.5
        expected_weight = Int8Adapter.requantize(
            weight.dequantize().addmm(b, a, alpha=strength),
            like=weight,
        )

        assert merge_adapter(model, [(adapter, strength)], stochastic_rounding=False) == 2

        torch.testing.assert_close(model.weight.data.qdata, expected_weight.qdata)
        torch.testing.assert_close(model.weight.data.scale, expected_weight.scale)
        assert model.bias is bias
        assert type(model.bias.data) is torch.Tensor
        torch.testing.assert_close(model.bias, bias_before + strength * bias_delta)

    def test_multiple_loras_requantize_shared_target_once(self) -> None:
        from tests.test_int8_adapter import _make_affine_int8

        rows, cols = 2, 2
        base = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        weight = _make_affine_int8(base)
        model = nn.Linear(cols, rows, bias=False)
        model.weight = nn.Parameter(weight, requires_grad=False)
        lora = Adapter({
            "weight": ParameterDelta.from_tensors(
                a=torch.tensor([[0.0, 1.0]]),
                b=torch.full((rows, 1), 0.003),
            ),
        })

        assert merge_adapter(model, [(lora, 1.0), (lora, 1.0)], stochastic_rounding=False) == 1
        # The first column fixes the row scale near 1 / 127.5. Each delta
        # alone rounds to zero. Accumulating both before encoding yields one.
        torch.testing.assert_close(model.weight.data.qdata[:, 1], torch.ones(rows, dtype=torch.int8))

    def test_tied_alias_target_merges_shared_storage(self) -> None:
        m = _make_tied_non_block_model(dtype=torch.float32)
        base = m.embed.weight.detach().clone()
        sd = {
            "head.lora_A.weight": torch.randn(4, 16),
            "head.lora_B.weight": torch.randn(16, 4),
        }
        lora = Adapter.from_state_dict(state_dict=sd)
        factor = lora.targets["head.weight"]
        a, b = _factor_tensors(factor)

        merged = merge_adapter(m, [(lora, 0.25)])

        expected = base.clone()
        expected.addmm_(
            b.to(dtype=expected.dtype),
            a.to(dtype=expected.dtype),
            alpha=0.25,
        )
        assert merged == 1
        torch.testing.assert_close(m.embed.weight, expected)
        torch.testing.assert_close(m.head.weight, expected)
        assert m.head.weight is m.embed.weight

    def test_duplicate_tied_alias_targets_raise_before_mutation(self) -> None:
        m = _make_tied_non_block_model(dtype=torch.float32)
        before = m.embed.weight.detach().clone()
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
            "head.lora_A.weight": torch.randn(4, 16),
            "head.lora_B.weight": torch.randn(16, 4),
        }

        with pytest.raises(ValueError, match="Tied parameter aliases"):
            merge_adapter(m, [(Adapter.from_state_dict(state_dict=sd), 1.0)])

        torch.testing.assert_close(m.embed.weight, before)
        assert m.head.weight is m.embed.weight

    def test_unmatched_unsupported_tensor_subclass_is_ignored(self) -> None:
        class UnknownTensor(torch.Tensor):
            pass

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Linear(16, 16, bias=False)
                self.other = nn.Linear(16, 16, bias=False)
                wrapped = torch.Tensor._make_subclass(
                    UnknownTensor,
                    torch.randn(16, 16),
                    False,
                )
                self.other.weight = nn.Parameter(wrapped, requires_grad=False)

        m = M()
        before = m.target.weight.detach().clone()
        sd = {
            "target.lora_A.weight": torch.randn(4, 16),
            "target.lora_B.weight": torch.randn(16, 4),
        }
        lora = Adapter.from_state_dict(state_dict=sd)
        factor = lora.targets["target.weight"]
        a, b = _factor_tensors(factor)

        merged = merge_adapter(m, [(lora, 0.5)])

        expected = before.clone()
        expected.addmm_(b, a, alpha=0.5)
        assert merged == 1
        torch.testing.assert_close(m.target.weight, expected)


# ---------------------------------------------------------------------------
# Cleanup invariants
# ---------------------------------------------------------------------------


class TestRoutedMode:
    """Routed-mode Adapter: forward hook on the parent layer, base
    weight untouched. Math: y = base(x) + alpha * B * A * x.
    """

    def test_routed_accepts_linear_input_by_keyword(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Linear(3, 3, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.target(input=x)

        m = M()
        m.requires_grad_(False)
        lora = Adapter.from_state_dict(
            {
                "target.lora_A.weight": torch.randn(1, 3),
                "target.lora_B.weight": torch.randn(3, 1),
            }
        )
        factor = lora.targets["target.weight"]
        a, b = _factor_tensors(factor)
        x = torch.randn(2, 3)
        strength = 0.5
        offloader = ModelOffloader.from_module(m)

        offloader.activate(
            "cpu",
            adapters=[lora],
            adapter_strengths=[strength],
            adapter_mode="routed",
        )
        try:
            actual = m(x)
            expected = F.linear(x, m.target.weight)
            expected += ((x @ a.T) * strength) @ b.T
            torch.testing.assert_close(actual, expected)
        finally:
            offloader.deactivate()

    def test_routed_interrupted_forward_staging_is_dropped_by_next_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        m = nn.Sequential(nn.Linear(3, 3, bias=False)).requires_grad_(False)
        lora = Adapter.from_state_dict(
            {
                "0.lora_A.weight": torch.randn(1, 3),
                "0.lora_B.weight": torch.randn(3, 1),
            }
        )
        staged: list[ref[nn.Parameter]] = []
        stage = lora_impl._stage_routed_factors

        def recording_stage(
            factors: Sequence[lora_impl.ScaledLoRAFactor],
            x: torch.Tensor,
        ) -> tuple[lora_impl._StagedLoRAFactor, ...]:
            result = stage(factors, x)
            staged.append(ref(result[0].a))
            return result

        def interrupt(_module: nn.Module, _args: tuple[object, ...]) -> None:
            raise Interrupt

        monkeypatch.setattr(lora_impl, "_stage_routed_factors", recording_stage)
        offloader = ModelOffloader.from_module(m)
        x = torch.randn(2, 3)
        with activated_model(offloader, "cpu", adapters=[lora], adapter_mode="routed"):
            with torch.no_grad():
                expected = m(x)
                handle = m[0].register_forward_pre_hook(interrupt)
                try:
                    with pytest.raises(Interrupt):
                        m(x)
                finally:
                    handle.remove()
                actual = m(x)

            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert len(staged) == 3
            assert staged[1]() is None

    @CUDA
    def test_routed_forward_and_teardown_match_manual_baselines(self) -> None:
        # Routed mode: base weight stays exactly as constructed; the
        # Adapter contribution rides as a forward-hook addition.
        with torch.random.fork_rng():
            torch.manual_seed(0)
            m = _make_bf16_model(num_blocks=3, dim=16)
        loras = [
            (_make_lora(num_blocks=3, dim=16, seed=11), 0.5),
            (_make_lora(num_blocks=3, dim=16, seed=22), 0.25),
        ]
        base_snapshots = [m.transformer_blocks[i].attn.weight.detach().clone() for i in range(3)]

        s = _make_strategy(m)
        adapters, strengths = zip(*loras, strict=True)
        with activated_model(s, "cuda", adapters=adapters, adapter_strengths=strengths, adapter_mode="routed"):
            generator = torch.Generator(device="cuda").manual_seed(0)
            x = torch.randn(
                2,
                16,
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            actual = m(x)
            torch.cuda.synchronize()
            expected = _expected_routed_output(m, x, loras)
            expected_base = _expected_routed_output(m, x, [])
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1), (
                f"routed forward mismatch:\n  expected: {expected.flatten()[:4]}\n  actual:   {actual.flatten()[:4]}"
            )

        # Base weights on CPU snapshots must equal the model's current
        # (post-deactivate) base weights — routed mode didn't mutate.
        for i in range(3):
            assert torch.equal(
                m.transformer_blocks[i].attn.weight.detach(),
                base_snapshots[i],
            ), f"routed mode mutated block {i} base weight"

        with activated_model(s, "cuda", adapter_mode="routed"):
            base_only = m(x)
            torch.testing.assert_close(base_only, expected_base, rtol=0, atol=0)
            assert not torch.allclose(actual, base_only, rtol=0.001, atol=0.001)

    @CUDA
    def test_routed_mixed_ranks(self) -> None:
        # Multiple LoRAs targeting the same weight at different ranks
        # must produce the same forward output as the per-adapter route
        # math: each (A_i, B_i) is applied independently and summed,
        # output = base(x) + sum_i strength_i * (x @ A_i.T) @ B_i.T.
        m = _make_bf16_model(num_blocks=2, dim=16)
        # Two LoRAs targeting the same blocks with different ranks.
        lora_r4 = _make_lora(num_blocks=2, dim=16, rank=4, seed=101)
        lora_r8 = _make_lora(num_blocks=2, dim=16, rank=8, seed=202)
        loras = [(lora_r4, 0.6), (lora_r8, 0.3)]

        s = _make_strategy(m)
        adapters, strengths = zip(*loras, strict=True)
        with activated_model(s, "cuda", adapters=adapters, adapter_strengths=strengths, adapter_mode="routed"):
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            actual = m(x)
            torch.cuda.synchronize()
            expected = _expected_routed_output(m, x, loras)
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1), (
                f"mixed-rank routed output mismatch:\n"
                f"  expected: {expected.flatten()[:4]}\n"
                f"  actual:   {actual.flatten()[:4]}"
            )

    @pytest.mark.parametrize("invalid_block", [0, 1])
    def test_routed_rejection_restores_base_model(self, invalid_block: int) -> None:
        class LinearLike(nn.Module):
            def __init__(self, weight: nn.Parameter) -> None:
                super().__init__()
                self.weight = weight

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.linear(x, self.weight)

        model = _make_bf16_model(num_blocks=2).float()
        block = model.transformer_blocks[invalid_block]
        block.attn = LinearLike(block.attn.weight)
        offloader = _make_strategy(model)
        lora = _make_lora(num_blocks=2, dim=16)
        x = torch.randn(2, 16)
        expected = model(x)

        # Rejecting block 1 exercises rollback after block 0 installed its hooks.
        with pytest.raises(ValueError, match=r"Routed LoRA mode requires nn\.Linear"):
            offloader.activate("cpu", adapters=[lora], adapter_mode="routed")

        assert offloader.active_device is None
        for block in model.transformer_blocks:
            assert not block.attn._forward_pre_hooks
            assert not block.attn._forward_hooks
        with activated_model(offloader, "cpu"):
            torch.testing.assert_close(model(x), expected)
        assert _plan_lora_merge(offloader, lora).keys() == lora.targets.keys()

    @CUDA
    def test_routed_single_adapter(self) -> None:
        # Single-adapter case: one (A, B) pair, no summation. Forward
        # output must still match the manual baseline.
        m = _make_bf16_model(num_blocks=2, dim=16)
        lora = _make_lora(num_blocks=2, dim=16, seed=33)

        s = _make_strategy(m)
        with activated_model(s, "cuda", adapters=[lora], adapter_strengths=[0.7], adapter_mode="routed"):
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            actual = m(x)
            torch.cuda.synchronize()
            expected = _expected_routed_output(m, x, [(lora, 0.7)])
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1)

    @pytest.mark.parametrize("target", ["embed", "head"])
    def test_routed_tied_weight_target_uses_exact_parent(self, target: str) -> None:
        model = _make_tied_non_block_model()
        offloader = _make_strategy(model)
        x = torch.eye(16)
        base = model.embed(x)
        lora = Adapter.from_state_dict({
            f"{target}.lora_A.weight": torch.ones(1, 16),
            f"{target}.lora_B.weight": torch.ones(16, 1),
        })
        with activated_model(offloader, "cpu", adapters=[lora], adapter_mode="routed"):
            assert model.embed.weight is model.head.weight
            for name in ("embed", "head"):
                expected = base + 1 if name == target else base
                torch.testing.assert_close(model.get_submodule(name)(x), expected)

    @CUDA
    def test_routed_with_bias_linear(self) -> None:
        # Routed math: hook adds alpha * B * A * x to the layer's output, which
        # already includes any bias from the base layer. Bias-having
        # Linears must produce the same output as a manual baseline
        # that goes through F.linear with the bias.
        m = _make_bf16_model(num_blocks=2, dim=16, attn_bias=True)

        lora = _make_lora(num_blocks=2, dim=16, seed=55)
        strength = 0.5
        s = ModelOffloader.from_module(
            m,
            block_paths=["transformer_blocks"],
        )
        with activated_model(s, "cuda", adapters=[lora], adapter_strengths=[strength], adapter_mode="routed"):
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            actual = m(x)
            torch.cuda.synchronize()
            # Manual baseline: walk layers via F.linear (bypasses hooks)
            # with bias included on attn.
            h = F.linear(x, m.embed.weight.to(x.device))
            for i, blk in enumerate(m.transformer_blocks):
                base_attn = F.linear(
                    h,
                    blk.attn.weight.to(h.device),
                    blk.attn.bias.to(h.device),
                )
                factor = lora.targets[f"transformer_blocks.{i}.attn.weight"]
                a, b = _factor_tensors(factor)
                a_dev = a.to(device=h.device, dtype=h.dtype)
                b_dev = b.to(device=h.device, dtype=h.dtype)
                attn_out = base_attn + strength * (h @ a_dev.T @ b_dev.T)
                h = F.linear(attn_out, blk.ff.weight.to(h.device))
            expected = F.linear(h, m.head.weight.to(h.device))
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1)

class TestRoutedStaging:
    """Routed LoRA stages host factors in target PRE hooks."""

    def test_multi_lora_stacked_hooks_compose_additively(self) -> None:
        # One block (so the two adapters don't feed each other across blocks)
        # and a fully-linear tail, so Adapter residuals superpose:
        # y(L1+L2) == y(L1) + y(L2) - y0. Two LoRAs on one target are grouped
        # into one hook pair and retain their independent strengths.
        torch.manual_seed(0)
        m = _make_bf16_model(num_blocks=1, dim=16).to(torch.float32)
        x = torch.randn(3, 16)
        lora1 = _make_lora(num_blocks=1, dim=16, seed=1)
        lora2 = _make_lora(num_blocks=1, dim=16, seed=2)
        s = ModelOffloader.from_module(m, block_paths=["transformer_blocks"])

        def routed_forward(adapters: list[Adapter], strengths: list[float]) -> torch.Tensor:
            with activated_model(s, "cpu", adapters=adapters, adapter_strengths=strengths, adapter_mode="routed"):
                return m(x)

        y0 = routed_forward([], [])
        y1 = routed_forward([lora1], [0.7])
        y2 = routed_forward([lora2], [1.3])
        y12 = routed_forward([lora1, lora2], [0.7, 1.3])
        torch.testing.assert_close(
            y12,
            y1 + y2 - y0,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_pre_hook_stages_grouped_factors_each_forward(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        m = _make_bf16_model(num_blocks=1, dim=16).to(torch.float32)
        loras = [
            (_make_lora(num_blocks=1, dim=16, seed=1), 0.7),
            (_make_lora(num_blocks=1, dim=16, seed=2), 1.3),
        ]
        calls: list[tuple[int, torch.device]] = []
        original = lora_impl._stage_routed_factors

        def record_stage(
            factors: Sequence[ScaledLoRAFactor],
            x: torch.Tensor,
        ) -> tuple[lora_impl._StagedLoRAFactor, ...]:
            calls.append((len(factors), x.device))
            return original(factors, x)

        monkeypatch.setattr(lora_impl, "_stage_routed_factors", record_stage)
        s = ModelOffloader.from_module(m)
        s.activate(
            "cpu",
            adapters=[lora for lora, _strength in loras],
            adapter_strengths=[strength for _lora, strength in loras],
            adapter_mode="routed",
        )
        try:
            target = m.transformer_blocks[0].attn
            assert len(target._forward_pre_hooks) == 1
            assert len(target._forward_hooks) == 1
            x = torch.randn(2, 16)
            m(x)
            m(x)
            assert calls == [
                (2, torch.device("cpu")),
                (2, torch.device("cpu")),
            ]
        finally:
            s.deactivate()

    def test_failed_linear_forward_discards_staged_factors(self) -> None:
        class FailingLinear(nn.Linear):
            fail = True

            def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
                if self.fail:
                    raise RuntimeError("linear failed")
                return super().forward(input_tensor)

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = FailingLinear(3, 3, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.target(x)

        m = M()
        m.requires_grad_(False)
        lora = Adapter.from_state_dict(
            {
                "target.lora_A.weight": torch.randn(1, 3),
                "target.lora_B.weight": torch.randn(3, 1),
            }
        )
        s = ModelOffloader.from_module(m)
        s.activate("cpu", adapters=[lora], adapter_mode="routed")
        try:
            assert len(s._routed_hook_removers) == 1
            assert callable(s._routed_hook_removers[0])
            with pytest.raises(RuntimeError, match="linear failed"):
                m(torch.randn(2, 3))

            m.target.fail = False
            assert m(torch.randn(2, 3)).shape == (2, 3)
        finally:
            s.deactivate()

    def test_routed_nested_base_layer_path(self) -> None:
        # A weight nested under ``.base_layer.`` (e.g. PEFT-wrapped). Routed
        # resolves the hook onto that exact nested Linear.
        class PEFTBlock(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.attn = nn.Module()
                self.attn.base_layer = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.attn.base_layer(x)

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer_blocks = nn.ModuleList([PEFTBlock(16) for _ in range(2)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for blk in self.transformer_blocks:
                    x = blk(x)
                return x

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        base_w = [m.transformer_blocks[b].attn.base_layer.weight.detach().clone() for b in range(2)]
        # Keys match the model's real ``.base_layer.`` paths exactly.
        g = torch.Generator().manual_seed(3)
        sd: dict[str, torch.Tensor] = {}
        for b in range(2):
            base = f"transformer_blocks.{b}.attn.base_layer"
            sd[f"{base}.lora_A.weight"] = torch.randn(4, 16, generator=g)
            sd[f"{base}.lora_B.weight"] = torch.randn(16, 4, generator=g)
        lora = Adapter.from_state_dict(state_dict=sd)
        assert set(lora.targets) == {
            "transformer_blocks.0.attn.base_layer.weight",
            "transformer_blocks.1.attn.base_layer.weight",
        }

        s = ModelOffloader.from_module(m, block_paths=["transformer_blocks"])
        x = torch.randn(2, 16)
        with activated_model(s, "cpu", adapters=[lora], adapter_strengths=[0.5], adapter_mode="routed"):
            actual = m(x).clone()
            # The hook lands on the matched ``.base_layer`` Linear.
            assert len(m.transformer_blocks[0].attn.base_layer._forward_hooks) == 1

        h = x
        for b in range(2):
            out = F.linear(h, base_w[b])
            a = sd[f"transformer_blocks.{b}.attn.base_layer.lora_A.weight"]
            bb = sd[f"transformer_blocks.{b}.attn.base_layer.lora_B.weight"]
            h = out + 0.5 * (h @ a.T) @ bb.T
        torch.testing.assert_close(actual, h, rtol=1e-4, atol=1e-4)

    def test_deactivate_removes_staging_hooks(self) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16).float()
        lora = _make_lora(num_blocks=2, dim=16, seed=4)
        s = ModelOffloader.from_module(m, block_paths=["transformer_blocks"])

        x = torch.randn(2, 16)
        base = m(x)
        expected = _expected_routed_output(m, x, [(lora, 1.0)])
        for _ in range(2):
            with activated_model(s, "cpu", adapters=[lora], adapter_mode="routed"):
                torch.testing.assert_close(m(x), expected, rtol=1e-5, atol=1e-5)
            for block in m.transformer_blocks:
                assert not block.attn._forward_pre_hooks
                assert not block.attn._forward_hooks
            torch.testing.assert_close(m(x), base, rtol=0, atol=0)

    @CUDA
    def test_routed_staging_matches_merge_with_streamed_model(self) -> None:
        # Per-target factor staging must compose with a streamed base model and
        # match merge mode bit-close.
        m = _make_bf16_model(num_blocks=2, dim=16)
        lora = Adapter.from_state_dict(
            state_dict=_make_lora_sd(num_blocks=2, dim=16, seed=7),
        )
        x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
        s = ModelOffloader.from_module(m, block_paths=["transformer_blocks"])
        with activated_model(s, "cuda", adapters=[lora], adapter_strengths=[0.5]):
            merged = m(x).clone()
            torch.cuda.synchronize()

        with activated_model(s, "cuda", adapters=[lora], adapter_strengths=[0.5], adapter_mode="routed"):
            routed = m(x).clone()
            torch.cuda.synchronize()

        # Routed residual vs merge (weight-bake) differ only
        # by bf16 rounding of the separate residual GEMMs.
        torch.testing.assert_close(routed, merged, rtol=0.1, atol=0.1)

    def test_unknown_routed_target_releases_activation_claim(self) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16)
        sd = {
            "transformer_blocks.0.absent.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.absent.lora_B.weight": torch.randn(16, 4),
        }
        bad = Adapter.from_state_dict(state_dict=sd)
        s = ModelOffloader.from_module(m, block_paths=["transformer_blocks"])
        x = torch.randn(2, 16, dtype=torch.bfloat16)
        base = m(x)
        with pytest.raises(ValueError, match="not parameters in the model"):
            s.activate("cpu", adapters=[bad], adapter_mode="routed")
        assert s.active_device is None
        with activated_model(s, "cpu"):
            torch.testing.assert_close(m(x), base)


class TestDeactivateCleanupInvariants:
    @CUDA
    @pytest.mark.parametrize("mode", ["merge", "routed"])
    def test_cleanup_runs_even_when_streamer_deactivate_raises(
        self, mode: AdapterMode, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        component, = block_components(s)
        x = torch.ones(1, 16, dtype=torch.bfloat16)
        base = m(x)

        def fail_release() -> None:
            raise RuntimeError("streamer cleanup failed")

        s.activate("cuda", adapters=[_make_lora(4, 16)], adapter_mode=mode)
        try:
            with monkeypatch.context() as patch:
                patch.setattr(component, "release", fail_release)
                with pytest.raises(RuntimeError, match="streamer cleanup failed"):
                    s.deactivate()
            assert s.active_device is None
            assert s.failed
            for block in m.transformer_blocks:
                assert not block.attn._forward_pre_hooks
                assert not block.attn._forward_hooks
            with activated_model(s, "cpu"):
                torch.testing.assert_close(m(x), base)
        finally:
            s.deactivate()


# ---------------------------------------------------------------------------
# Unified Adapter resource (ResourceCache integration)
# ---------------------------------------------------------------------------


class TestLoRAResource:
    def test_lora_through_resource_cache(self) -> None:
        sd = _make_lora_sd(num_blocks=2, dim=8, rank=2)
        cache = ResourceCache(10**9)
        spec = AdapterSpec(
            key="lora:test",
            estimated_cache_bytes=1000,
            factory=lambda: sd,
        )
        with cache.lease(spec) as lora:
            assert isinstance(lora, Adapter)
            assert lora.cache_bytes > 0
            assert len(lora.targets) == 2
            assert not lora.allow_partial_targets
        with cache.lease("lora:test") as lora2:
            assert lora2 is lora

    def test_parameter_value_spec_builds_cached_resource(self) -> None:
        value = torch.randn(2, 3)
        spec = AdapterSpec(
            key="adapter:value",
            estimated_cache_bytes=value.nbytes,
            factory=lambda: {"target.weight": value},
        )

        lora = spec.build_store()

        assert tuple(lora.targets) == ("target.weight",)
        target = lora.targets["target.weight"]
        assert isinstance(target, ParameterValue)
        assert not target.scale_with_strength
        assert lora.cache_bytes == value.nbytes

    def test_lora_spec_propagates_partial_target_policy(self) -> None:
        spec = AdapterSpec(
            key="lora:partial",
            estimated_cache_bytes=1000,
            factory=lambda: _make_lora_sd(num_blocks=1, dim=8, rank=2),
            allow_partial_targets=True,
        )

        assert spec.build_store().allow_partial_targets

    def test_model_cache_applies_loras_and_holds_leases(
        self,
    ) -> None:
        sd = _make_lora_sd(num_blocks=2, dim=8, rank=2, seed=17)
        expected_lora = Adapter.from_state_dict(sd)
        factory_calls = {"lora": 0}

        def lora_factory() -> dict[str, torch.Tensor]:
            factory_calls["lora"] += 1
            return sd

        cache = ModelCache()
        lora_spec = AdapterSpec(
            key="lora:style",
            estimated_cache_bytes=1000,
            factory=lora_factory,
        )
        model_spec = ModelSpec(
            key="model",
            estimated_cache_bytes=10**6,
            factory=lambda: _make_bf16_model(num_blocks=2, dim=8).to(torch.float32),
        )

        x = torch.randn(2, 8)
        with cache.use(
            model_spec,
            device="cpu",
            adapter_specs=[lora_spec],
            adapter_strengths=[0.5],
            adapter_mode="routed",
        ) as model:
            assert cache.info("lora:style").lease_count == 1
            assert cache.info("model").lease_count == 1
            actual = model(x)
            expected = _expected_routed_output(model, x, [(expected_lora, 0.5)])
            assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)

        assert cache.info("lora:style").lease_count == 0
        assert cache.info("model").lease_count == 0
        assert factory_calls["lora"] == 1

        with cache.use(
            model_spec,
            device="cpu",
            adapter_specs=[lora_spec],
            adapter_mode="routed",
        ) as model:
            assert cache.info("lora:style").lease_count == 1
            actual = model(x)
            expected = _expected_routed_output(model, x, [(expected_lora, 1.0)])
            assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
        assert factory_calls["lora"] == 1

    def test_one_cached_partial_lora_serves_disjoint_models(self) -> None:
        class PartModel(nn.Module):
            def __init__(self, target: str) -> None:
                super().__init__()
                self.target = target
                self.add_module(target, nn.Linear(3, 3, bias=False))
                self.requires_grad_(False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.get_submodule(self.target)(x)

        state = {
            f"{target}.lora_A.weight": torch.full((1, 3), value) for target, value in (("first", 0.25), ("second", 0.5))
        }
        state.update(
            {
                f"{target}.lora_B.weight": torch.full((3, 1), value)
                for target, value in (("first", 0.75), ("second", 1.25))
            }
        )
        factory_calls = 0

        def lora_factory() -> dict[str, torch.Tensor]:
            nonlocal factory_calls
            factory_calls += 1
            return state

        lora_spec = AdapterSpec(
            key="lora:shared-parts",
            estimated_cache_bytes=1000,
            factory=lora_factory,
            allow_partial_targets=True,
        )
        first_spec = ModelSpec(
            key="model:first-part",
            estimated_cache_bytes=1000,
            factory=lambda: PartModel("first"),
        )
        second_spec = ModelSpec(
            key="model:second-part",
            estimated_cache_bytes=1000,
            factory=lambda: PartModel("second"),
        )
        cache = ModelCache()
        value = torch.randn(2, 3)

        for model_spec, target in (
            (first_spec, "first"),
            (second_spec, "second"),
        ):
            with cache.use(
                model_spec,
                device="cpu",
                adapter_specs=[lora_spec],
                adapter_mode="routed",
            ) as model:
                layer = model.get_submodule(target)
                assert isinstance(layer, nn.Linear)
                base = F.linear(value, layer.weight)
                a = state[f"{target}.lora_A.weight"]
                b = state[f"{target}.lora_B.weight"]
                expected = base + (value @ a.T) @ b.T
                torch.testing.assert_close(model(value), expected)

        assert factory_calls == 1

    @pytest.mark.parametrize(
        ("stochastic_rounding", "expected"),
        [(None, True), (False, False)],
    )
    def test_model_cache_forwards_default_strengths_and_requested_rounding(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stochastic_rounding: bool | None,
        expected: bool,
    ) -> None:
        cache = ModelCache()
        model_spec = ModelSpec(
            key="model:rounding-forward",
            estimated_cache_bytes=10**6,
            factory=lambda: _make_bf16_model(num_blocks=2, dim=8),
        )
        captured: list[bool] = []
        original_activate = ModelOffloader.activate

        def tracked_activate(
            offloader: ModelOffloader,
            *args: object,
            **kwargs: object,
        ) -> None:
            assert kwargs.get("adapter_strengths") is None
            captured.append(bool(kwargs.get("stochastic_rounding")))
            original_activate(offloader, *args, **kwargs)

        monkeypatch.setattr(ModelOffloader, "activate", tracked_activate)
        use_kwargs = {} if stochastic_rounding is None else {"stochastic_rounding": stochastic_rounding}
        with cache.use(model_spec, device="cpu", **use_kwargs):
            pass

        assert captured == [expected]

    @pytest.mark.parametrize("zero", [0.0, -0.0])
    def test_model_cache_does_not_lease_zero_strength_lora(
        self,
        zero: float,
    ) -> None:
        factory_calls = {"lora": 0, "model": 0}

        def lora_factory() -> dict[str, torch.Tensor]:
            factory_calls["lora"] += 1
            return _make_lora_sd(num_blocks=2, dim=8, rank=2)

        def model_factory() -> nn.Module:
            factory_calls["model"] += 1
            return _make_bf16_model(num_blocks=2, dim=8).to(torch.float32)

        cache = ModelCache()
        lora_spec = AdapterSpec(
            key="lora:inactive",
            estimated_cache_bytes=1000,
            factory=lora_factory,
        )
        model_spec = ModelSpec(
            key="model:zero-strength",
            estimated_cache_bytes=10**6,
            factory=model_factory,
        )

        with cache.use(
            model_spec,
            device="cpu",
            adapter_specs=[lora_spec],
            adapter_strengths=[zero],
            adapter_mode="merge",
        ) as model:
            assert model(torch.randn(2, 8)).shape == (2, 8)

        assert factory_calls == {"lora": 0, "model": 1}
        with pytest.raises(ResourceNotRegisteredError):
            cache.info("lora:inactive")

    def test_cached_lora_can_overlap_across_model_runtimes(self) -> None:
        cache = ModelCache()
        lora_spec = AdapterSpec(
            key="lora:shared",
            estimated_cache_bytes=1000,
            factory=lambda: _make_lora_sd(num_blocks=2, dim=8, rank=2),
        )
        first_spec = ModelSpec(
            key="model:first",
            estimated_cache_bytes=10**6,
            factory=lambda: _make_bf16_model(2, 8).to(torch.float32),
        )
        second_spec = ModelSpec(
            key="model:second",
            estimated_cache_bytes=10**6,
            factory=lambda: _make_bf16_model(2, 8).to(torch.float32),
        )
        x = torch.randn(2, 8)

        with cache.use(
            first_spec,
            device="cpu",
            adapter_specs=[lora_spec],
            adapter_mode="routed",
        ) as first:
            with cache.use(
                second_spec,
                device="cpu",
                adapter_specs=[lora_spec],
                adapter_mode="routed",
            ) as second:
                assert cache.info("lora:shared").lease_count == 2
                assert first(x).shape == (2, 8)
                assert second(x).shape == (2, 8)
            assert cache.info("lora:shared").lease_count == 1

    def test_model_cache_rejects_strength_mismatch_before_admission(self) -> None:
        factory_calls = {"lora": 0, "model": 0}

        def lora_factory() -> dict[str, torch.Tensor]:
            factory_calls["lora"] += 1
            return _make_lora_sd(num_blocks=2, dim=8, rank=2)

        def model_factory() -> nn.Module:
            factory_calls["model"] += 1
            return _make_bf16_model(num_blocks=2, dim=8)

        lora_spec = AdapterSpec(
            key="lora:style",
            estimated_cache_bytes=1000,
            factory=lora_factory,
        )
        model_spec = ModelSpec(
            key="model",
            estimated_cache_bytes=10**6,
            factory=model_factory,
        )
        cache = ModelCache()

        with pytest.raises(ValueError, match="shorter"):
            with cache.use(
                model_spec,
                device="cpu",
                adapter_specs=[lora_spec],
                adapter_strengths=[],
            ):
                pass

        assert factory_calls == {"lora": 0, "model": 0}
        with pytest.raises(ResourceNotRegisteredError):
            cache.info("lora:style")
        with pytest.raises(ResourceNotRegisteredError):
            cache.info("model")

    def test_model_cache_applies_duplicate_lora_contributions(self) -> None:
        factory_calls = {"lora": 0, "model": 0}
        state_dict = _make_lora_sd(num_blocks=2, dim=8, rank=2)
        expected_lora = Adapter.from_state_dict(state_dict)

        def lora_factory() -> dict[str, torch.Tensor]:
            factory_calls["lora"] += 1
            return state_dict

        def model_factory() -> nn.Module:
            factory_calls["model"] += 1
            return _make_bf16_model(num_blocks=2, dim=8).to(torch.float32)

        lora_spec = AdapterSpec(
            key="lora:style",
            estimated_cache_bytes=1000,
            factory=lora_factory,
        )
        model_spec = ModelSpec(
            key="model",
            estimated_cache_bytes=10**6,
            factory=model_factory,
        )
        cache = ModelCache()
        value = torch.randn(2, 8)
        disabled = AdapterSpec(key="disabled", factory=lambda: pytest.fail("disabled adapter was built"))

        with cache.use(
            model_spec,
            device="cpu",
            adapter_specs=[lora_spec, disabled, lora_spec],
            adapter_strengths=[0.25, 0.0, 0.75],
            adapter_mode="routed",
        ) as model:
            actual = model(value)
            expected = _expected_routed_output(
                model,
                value,
                [(expected_lora, 0.25), (expected_lora, 0.75)],
            )
            torch.testing.assert_close(actual, expected)

        assert factory_calls == {"lora": 1, "model": 1}
        assert cache.info("lora:style").lease_count == 0

    def test_lora_leased_during_resource_cache_miss(self) -> None:
        # The LoRAs acquired by a model-cache use are leased before the
        # model store builds, so a model cache-miss must fail admission
        # rather than evict a lora it is about to be applied with.
        lora_spec = AdapterSpec(
            key="lora:style",
            estimated_cache_bytes=1000,
            factory=lambda: _make_lora_sd(num_blocks=2, dim=8, rank=2),
        )
        model_spec = ModelSpec(
            key="model",
            estimated_cache_bytes=10_000,
            factory=lambda: _make_bf16_model(num_blocks=2, dim=8).to(torch.float32),
        )

        cache = ModelCache()
        with cache.use(
            model_spec,
            device="cpu",
            adapter_specs=[lora_spec],
            adapter_mode="routed",
        ):
            pass

        assert cache.info("lora:style").cached
        assert cache.info("lora:style").lease_count == 0
        assert cache.info("model").cached


# ---------------------------------------------------------------------------


class TestMergeIntoMappedModel:
    @pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
    def test_mapped_target_is_copied_out_before_the_merge(self, tmp_path, shape) -> None:
        base = torch.randn(shape)
        path = tmp_path / "weights.bin"
        path.write_bytes(base.numpy().tobytes())
        mapped = torch.from_file(str(path), shared=False, size=base.numel(), dtype=torch.float32).reshape(shape)
        model = nn.Module()
        model.target = nn.Linear(4, 3, bias=False)
        model.tied = nn.Linear(4, 3, bias=False)
        model.target.weight = nn.Parameter(mapped, requires_grad=False)
        model.tied.weight = model.target.weight
        dense = torch.randn(3, 4)
        adapter = Adapter.from_state_dict({"target.delta.weight": dense})

        assert merge_adapter(model, [(adapter, 0.5)]) == 1

        weight = model.target.weight
        assert model.tied.weight is weight
        assert weight.untyped_storage().resizable()
        assert weight.data_ptr() != mapped.data_ptr()
        assert not weight.requires_grad
        torch.testing.assert_close(weight, base + 0.5 * dense)
        # The mapping itself was never written.
        torch.testing.assert_close(mapped, base)

    @pytest.mark.parametrize("transpose", [False, True])
    def test_distinct_parameter_objects_over_one_mapped_view_stay_tied(self, tmp_path, transpose) -> None:
        base = torch.randn(3, 4)
        path = tmp_path / "weights.bin"
        path.write_bytes(base.numpy().tobytes())
        mapped = torch.from_file(str(path), shared=False, size=12, dtype=torch.float32).reshape(3, 4)
        if transpose:
            mapped, base = mapped.T, base.T
        model = nn.Module()
        model.target = nn.Linear(base.shape[1], base.shape[0], bias=False)
        model.tied = nn.Linear(base.shape[1], base.shape[0], bias=False)
        model.target.weight = nn.Parameter(mapped, requires_grad=False)
        model.tied.weight = nn.Parameter(mapped, requires_grad=False)
        assert model.tied.weight is not model.target.weight
        dense = torch.randn_like(base)
        adapter = Adapter.from_state_dict({"target.delta.weight": dense})

        assert merge_adapter(model, [(adapter, 0.5)]) == 1

        assert model.tied.weight is model.target.weight
        assert model.target.weight.untyped_storage().resizable()
        torch.testing.assert_close(model.target.weight, base + 0.5 * dense)
        torch.testing.assert_close(mapped, base)

    def test_replacing_mixed_trainability_aliases_leaves_optimizer_attached(self) -> None:
        mapped = torch.frombuffer(bytearray(24), dtype=torch.float32).reshape(2, 3)
        model = nn.Sequential(nn.Linear(3, 2, bias=False), nn.Linear(3, 2, bias=False))
        model[0].weight = nn.Parameter(mapped, requires_grad=False)
        trainable = model[1].weight = nn.Parameter(mapped)
        optimizer = torch.optim.SGD([trainable], lr=0.25)
        delta = Adapter.from_state_dict({"0.delta.weight": torch.ones_like(mapped)})

        with pytest.raises(ValueError, match="mixed requires_grad"):
            merge_adapter(model, [(delta, 1.0)])
        assert model[1].weight is optimizer.param_groups[0]["params"][0]
        assert not model[0].weight.requires_grad
        torch.testing.assert_close(mapped, torch.zeros_like(mapped))

    def test_external_adapter_copy_failure_precedes_all_writes(self, request) -> None:
        class ViewOnlyAdapter(RegularAdapter):
            capture_host = staticmethod(RegularAdapter.capture_host_view)

        model = nn.Sequential(nn.Linear(3, 2, bias=False), nn.Linear(3, 2, bias=False)).requires_grad_(False)
        first = model[0].weight.detach().clone()
        borrowed = torch.frombuffer(bytearray(24), dtype=torch.float32).reshape(2, 3)
        model[1].weight = nn.Parameter(borrowed, requires_grad=False)
        delta = Adapter.from_state_dict({f"{i}.delta.weight": torch.ones(2, 3) for i in range(2)})
        request.addfinalizer(register_adapter(ViewOnlyAdapter))

        with pytest.raises(ValueError, match="owned CPU storage"):
            merge_adapter(model, [(delta, 1.0)])
        torch.testing.assert_close(model[0].weight, first)
        torch.testing.assert_close(borrowed, torch.zeros_like(borrowed))

    @pytest.mark.parametrize("external_buffer", [True, False], ids=["borrowed", "owned"])
    def test_custom_value_copy_must_be_independent(self, request, external_buffer) -> None:
        class ViewOnlyAdapter(RegularAdapter):
            capture_host = staticmethod(RegularAdapter.capture_host_view)

        request.addfinalizer(register_adapter(ViewOnlyAdapter))
        source = torch.frombuffer(bytearray(24), dtype=torch.float32) if external_buffer else torch.empty(6)
        source = source.reshape(2, 3).fill_(2.0)
        model = nn.Linear(3, 2, bias=False, device="meta").requires_grad_(False)
        value = Adapter.from_state_dict({"weight": source}, scale_parameter_values=True)

        with pytest.raises(ValueError, match="independent, owned CPU storage"):
            merge_adapter(model, [(value, 2.0)])
        assert model.weight.is_meta
        torch.testing.assert_close(source, torch.full_like(source, 2.0))

    @pytest.mark.parametrize("always_copy", [False, True], ids=["view", "copy"])
    def test_registered_dense_adapter_preserves_optimizer_parameter(self, request, always_copy) -> None:
        # Implement the public dense-merge protocol by delegation, as a third-party adapter can.
        dense_methods = {
            name: staticmethod(getattr(RegularAdapter, name)) for name in get_protocol_members(DenseMergeTensorAdapter)
        }
        if always_copy:
            dense_methods["capture_host"] = staticmethod(lambda t: RegularAdapter.capture_host(t.clone()))
        request.addfinalizer(register_adapter(type("ExternalDenseAdapter", (), dense_methods)))
        model = nn.Sequential(nn.Linear(2, 3, bias=False))
        weight = model[0].weight = nn.Parameter(torch.zeros(2, 3).T)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        delta = Adapter.from_state_dict({"0.delta.weight": torch.ones_like(weight)})

        merge_adapter(model, [(delta, 1.0)])
        assert model[0].weight is weight
        weight.grad = torch.ones_like(weight)
        optimizer.step()
        torch.testing.assert_close(model[0].weight, torch.full_like(weight, 0.75))

    @pytest.mark.parametrize("mapped", [True, False], ids=["mapped", "owned"])
    def test_wrapper_merge_uses_physical_payload(self, request, tmp_path, mapped) -> None:
        class Packed(nn.Parameter):
            def __new__(cls, payload, requires_grad=False):
                wrapper = super().__new__(cls, torch.zeros_like(payload), requires_grad)
                wrapper.payload = payload
                return wrapper

        class PackedAdapter(RegularAdapter):
            @staticmethod
            def matches(t):
                return isinstance(t, Packed)

            @staticmethod
            def tensor_id(t):
                return RegularAdapter.tensor_id(t.payload)

            @staticmethod
            def capture_host(t):
                return RegularAdapter.capture_host(t.payload)

            @staticmethod
            def cpu_param(state, *, requires_grad=False):
                return Packed(state.data, requires_grad)

            @staticmethod
            def merge_dense_(target, *args, **kwargs):
                RegularAdapter.merge_dense_(target.payload, *args, **kwargs)

        path = tmp_path / "packed"
        path.write_bytes(bytes(24))
        data = torch.from_file(str(path), shared=True, size=6) if mapped else torch.zeros(6)
        # Normal capture would make the transposed payload contiguous in both cases.
        weight = Packed(data.reshape(2, 3).T)
        model = nn.Sequential(nn.Linear(2, 3, bias=False))
        model[0].weight = weight
        delta = Adapter.from_state_dict({"0.delta.weight": torch.ones(3, 2)})
        request.addfinalizer(register_adapter(PackedAdapter))

        merge_adapter(model, [(delta, 1.0)])
        assert (model[0].weight is weight) == (not mapped)
        torch.testing.assert_close(model[0].weight.payload, torch.ones(3, 2))
        if mapped:
            assert path.read_bytes() == bytes(24)


class TestIntrinsicLoRAScale:
    """``alpha / rank`` carried on the factor instead of folded into its tensors."""

    RANK = 4

    @classmethod
    def _state(cls, alpha: float | None = None) -> dict[str, torch.Tensor]:
        sd = {
            "target.lora_A.weight": torch.randn(cls.RANK, 16),
            "target.lora_B.weight": torch.randn(16, cls.RANK),
        }
        if alpha is not None:
            sd["target.alpha"] = torch.tensor(alpha)
        return sd

    def test_alpha_key_sets_the_scale_and_is_not_a_parameter_value(self) -> None:
        adapter = Adapter.from_state_dict(state_dict=self._state(alpha=8.0))
        assert set(adapter.targets) == {"target.weight"}
        assert _require_factor(adapter.targets["target.weight"]).scaling == pytest.approx(2.0)

    def test_absent_alpha_leaves_the_scaling_neutral(self) -> None:
        adapter = Adapter.from_state_dict(state_dict=self._state())
        assert _require_factor(adapter.targets["target.weight"]).scaling == 1.0

    def test_each_factor_carries_its_own_alpha(self) -> None:
        """Alpha is per factor, like the factors themselves, not one value for the file."""
        sd = {
            "first.lora_A.weight": torch.randn(4, 16),
            "first.lora_B.weight": torch.randn(16, 4),
            "first.alpha": torch.tensor(8.0),
            "second.lora_A.weight": torch.randn(2, 16),
            "second.lora_B.weight": torch.randn(16, 2),
            "second.alpha": torch.tensor(1.0),
        }
        targets = Adapter.from_state_dict(state_dict=sd).targets
        assert _require_factor(targets["first.weight"]).scaling == pytest.approx(2.0)
        assert _require_factor(targets["second.weight"]).scaling == pytest.approx(0.5)

    def test_strength_composes_with_the_intrinsic_scale(self) -> None:
        factor = _require_factor(
            Adapter.from_state_dict(state_dict=self._state(alpha=8.0)).targets["target.weight"]
        )
        assert factor.scaling == pytest.approx(2.0)
        assert factor.scaled(3.0).coefficient == pytest.approx(6.0)

    @pytest.mark.parametrize("strength", [1.0, 0.5, 2.0])
    def test_a_merge_matches_pre_scaled_factors(self, strength: float) -> None:
        """Through a real merge, an intrinsic scale equals the fold it replaces."""
        sd = self._state(alpha=8.0)
        scaling = 8.0 / self.RANK
        folded = {
            "target.lora_A.weight": sd["target.lora_A.weight"] * scaling,
            "target.lora_B.weight": sd["target.lora_B.weight"].clone(),
        }

        def merged(state: dict[str, torch.Tensor]) -> torch.Tensor:
            factor = _require_factor(Adapter.from_state_dict(state_dict=state).targets["target.weight"])
            param = nn.Parameter(torch.zeros(16, 16), requires_grad=False)
            transform = LoRATransform([factor.scaled(strength)])
            transform.validate_target(param)
            transform.apply_weight(param)
            return param.detach()

        torch.testing.assert_close(merged(sd), merged(folded))

    def test_alpha_without_a_factor_pair_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="alpha without a factor pair"):
            Adapter.from_state_dict(state_dict={"orphan.alpha": torch.tensor(8.0)})

    def test_a_non_scalar_alpha_is_rejected(self) -> None:
        sd = self._state()
        sd["target.alpha"] = torch.ones(2)
        with pytest.raises(ValueError, match="must be a scalar"):
            Adapter.from_state_dict(state_dict=sd)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_an_unusable_alpha_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="LoRA scaling must be"):
            Adapter.from_state_dict(state_dict=self._state(alpha=bad))

    def test_a_rank_zero_factor_with_an_alpha_is_rejected(self) -> None:
        sd = {
            "target.lora_A.weight": torch.zeros(0, 16),
            "target.lora_B.weight": torch.zeros(16, 0),
            "target.alpha": torch.tensor(8.0),
        }
        with pytest.raises(ValueError, match="rank-0 LoRA factor"):
            Adapter.from_state_dict(state_dict=sd)

    def test_a_mapped_factor_stays_mapped(self, tmp_path) -> None:
        """Why the scale is intrinsic: folding alpha here would replace the mapping."""
        path = tmp_path / "lora.safetensors"
        save_file(
            {
                "target.lora_A.weight": torch.randn(4, 32),
                "target.lora_B.weight": torch.randn(32, 4),
            },
            str(path),
            metadata={"alpha": "8"},
        )
        reader = MappedCheckpoint(path)
        keys = reader.keys()
        state = {key: reader.get_tensor(key) for key in keys}
        assert all(file_slice(tensor) is not None for tensor in state.values())

        # A checkpoint-wide alpha reaches here as a per-factor entry, the way
        # the caller already canonicalizes factor names. The mapped tensors are
        # never touched, which is what keeps them mapped.
        state["target.alpha"] = torch.tensor(float((reader.metadata() or {})["alpha"]))
        adapter = Adapter.from_state_dict(state_dict=state)
        factor = _require_factor(adapter.targets["target.weight"])
        assert factor.scaling == pytest.approx(2.0)
        a, b = _factor_tensors(adapter.targets["target.weight"])
        assert file_slice(a) is not None
        assert file_slice(b) is not None
        torch.testing.assert_close(a, state["target.lora_A.weight"], atol=0, rtol=0)
