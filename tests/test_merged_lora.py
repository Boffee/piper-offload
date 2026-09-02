"""Tests for activation-scoped Adapter application through ``ModelOffloader``.

Covers Adapter construction validation, activation matching, lifecycle
(activate/deactivate), Adapter switching, and forward-output correctness
against a manually-merged baseline.

Most lifecycle tests run on CPU (the merge math is device-agnostic);
CUDA-only tests gate on availability.
"""

from array import array
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import piper_offload.lora as lora_impl
import piper_offload.quanto_adapter as quanto_adapter_impl

from piper_offload import (
    BlockCompileConfig,
    Adapter,
    AdapterTarget,
    LoRAFactor,
    AdapterMode,
    LoRATransform,
    ModelCache,
    ResourceCache,
    ModelOffloader,
    ModelSpec,
    ParameterValue,
    ParameterValueTransform,
    ResourceNotRegisteredError,
    ResourceTooLargeError,
    PinnedComponent,
    AdapterSpec,
    ScaledLoRAFactor,
    ScaledParameterValue,
    BlockComponent,
    derive_seed,
    merge_adapter,
)
from piper_offload.gguf_adapter import GgufAdapter
from piper_offload.pinned_module import PinnedModuleInstance
from piper_offload.pinned_param import PinnedParam
from piper_offload.quanto_adapter import QuantoAdapter
from piper_offload.protocols import (
    ResourceBinding,
    ResourceStore,
)
from piper_offload.tensor_adapters import (
    DequantRequantTensorAdapter,
    LoRAMergeTensorAdapter,
    RegularAdapter,
    TensorCopyIntoAdapter,
)

from tests.conftest import (
    activated_model,
    block_components,
    transient_components,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_factor(target: AdapterTarget) -> LoRAFactor:
    assert isinstance(target, LoRAFactor)
    return target


def _factor_tensors(
    target: AdapterTarget,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a pinned :class:`LoRAFactor`'s ``(a, b)`` as CPU tensors."""
    factor = _require_factor(target)
    return factor.a.make_cpu_param().data, factor.b.make_cpu_param().data


def _factor_bias(target: AdapterTarget) -> torch.Tensor | None:
    """Materialize a legacy bias, when present, as a CPU tensor."""
    factor = _require_factor(target)
    if factor.bias is None:
        return None
    return factor.bias.make_cpu_param().data


def _quanto_absmax_oracle(
    dense: torch.Tensor,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    """Quantize with Quanto's optimizer/operator, independent of the adapter."""
    from optimum.quanto.tensor.optimizers import AbsmaxOptimizer
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

    canonical = like if type(like) is WeightQBytesTensor else like.weight_qbytes_tensor()
    axis = canonical.axis
    optimizer_axis = -1 if axis == dense.ndim - 1 else axis
    scale = AbsmaxOptimizer()(dense, qtype=canonical.qtype, axis=optimizer_axis)
    scale = scale.to(dtype=canonical._scale.dtype).reshape(canonical._scale.shape)
    zero = scale == 0
    eps = torch.finfo(torch.float32).eps
    scale = torch.where(zero, torch.full_like(scale, eps), scale)
    return WeightQBytesTensor.quantize(
        dense,
        canonical.qtype,
        optimizer_axis,
        scale,
        getattr(canonical, "activation_qtype", None),
        optimized=False,
    )


def _make_model_offloader(
    model: nn.Module,
    *,
    block_paths: Sequence[str] = (),
    transient_block_paths: Sequence[str] = (),
    include_block_trainables: bool = False,
    transient_paths: Sequence[str] = (),
) -> ModelOffloader:
    return ModelOffloader.from_module(
        model,
        block_paths=block_paths,
        transient_block_paths=transient_block_paths,
        include_block_trainables=include_block_trainables,
        transient_paths=transient_paths,
    )


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
    *,
    legacy_bias: bool = False,
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
        if legacy_bias:
            sd[f"{base}.lora_B.bias"] = torch.randn(
                dim,
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
    *,
    legacy_bias: bool = False,
) -> Adapter:
    """Build a Adapter targeting attn.weight across all blocks."""
    sd = _make_lora_sd(
        num_blocks,
        dim,
        rank=rank,
        seed=seed,
        prefix=prefix,
        legacy_bias=legacy_bias,
    )
    return Adapter.from_state_dict(state_dict=sd)


def _request_loras(
    strategy: ModelOffloader,
    loras: Sequence[tuple[Adapter, float]],
    *,
    mode: AdapterMode = "merge",
) -> None:
    normalized = strategy._normalize_adapters(
        [lora for lora, _strength in loras],
        adapter_strengths=[strength for _lora, strength in loras],
    )
    _LORA_REQUESTS[strategy] = (normalized, mode)


_LORA_REQUESTS: dict[
    ModelOffloader,
    tuple[list[tuple[Adapter, float]], AdapterMode],
] = {}


def _activate(
    strategy: ModelOffloader,
    device: torch.device | str,
) -> None:
    loras, mode = _LORA_REQUESTS.pop(strategy, ([], "merge"))
    strategy.activate(
        device,
        adapters=[lora for lora, _strength in loras],
        adapter_strengths=[strength for _lora, strength in loras],
        adapter_mode=mode,
    )


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
            bias = _factor_bias(factors)
            if bias is not None:
                base_attn = (
                    base_attn
                    + bias.to(
                        device=h.device,
                        dtype=h.dtype,
                    )
                    * strength
                )
        if a_parts:
            a_fused = torch.cat(a_parts, dim=0)
            b_fused = torch.cat(b_parts, dim=1)
            base_attn = base_attn + (h @ a_fused.T) @ b_fused.T
        h = F.linear(base_attn, blk.ff.weight.to(h.device))
    return F.linear(h, model.head.weight.to(h.device))


def _make_strategy(model: nn.Module) -> ModelOffloader:
    """Shorthand for constructing the strategy."""
    return _make_model_offloader(model, block_paths=["transformer_blocks"])


def _has_post_copy_hook(strategy: ModelOffloader, target_key: str) -> bool:
    """Check whether a merge hook is installed for the given target."""
    if target_key not in strategy.param_names:
        return False
    param_name = target_key
    component = strategy._composite.component_for_param_name(param_name)
    if isinstance(component, PinnedComponent):
        instance = component._instance
        return instance.post_copy_hook_key(param_name) in instance._post_copy_hooks
    if isinstance(component, BlockComponent):
        instance, local = component._resolve_param_name(param_name)
        return instance.post_copy_hook_key(local) in instance._post_copy_hooks
    return False


def _activate_loras_for_test(
    strategy: ModelOffloader,
) -> int:
    loras, mode = _LORA_REQUESTS.pop(strategy, ([], "merge"))
    if mode == "merge":
        targets = strategy._group_adapter_updates_by_param_name(loras)
        try:
            strategy._register_merge_adapter_hooks(torch.device("cuda"), targets)
        except BaseException:
            strategy._clear_active_adapter_hooks()
            raise
        return len(targets)
    targets = strategy._group_adapter_updates_by_param_name(loras)
    before = len(strategy._adapter_hook_removers)
    try:
        strategy._register_routed_lora_hooks(targets)
        return len(strategy._adapter_hook_removers) - before
    finally:
        strategy._clear_active_adapter_hooks()


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
            "target.lora_B.bias",
        ],
    )
    def test_rejects_meta_factor_sources(self, key: str) -> None:
        sd = {
            "target.lora_A.weight": torch.randn(1, 3),
            "target.lora_B.weight": torch.randn(2, 1),
            "target.lora_B.bias": torch.randn(2),
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

    def test_accepts_optional_legacy_bias(self) -> None:
        sd = _make_lora_sd(
            num_blocks=1,
            dim=16,
            rank=4,
            legacy_bias=True,
        )

        lora = Adapter.from_state_dict(state_dict=sd)
        factor = lora.targets["transformer_blocks.0.attn.weight"]
        bias = _factor_bias(factor)

        assert bias is not None
        assert tuple(bias.shape) == (16,)
        assert bias.is_pinned()
        assert lora.cache_bytes == sum(tensor.nbytes for tensor in sd.values())

    def test_rejects_unpaired_legacy_bias(self) -> None:
        sd = _make_lora_sd(num_blocks=1, dim=16, rank=4)
        sd["other.lora_B.bias"] = torch.randn(16)

        with pytest.raises(ValueError, match="Unpaired LoRA biases"):
            Adapter.from_state_dict(state_dict=sd)

    @pytest.mark.parametrize(
        "bias",
        [
            torch.zeros(16, dtype=torch.int32),
            torch.randn(2, 8),
            torch.randn(8),
        ],
    )
    def test_rejects_invalid_legacy_bias(
        self,
        bias: torch.Tensor,
    ) -> None:
        sd = _make_lora_sd(num_blocks=1, dim=16, rank=4)
        sd["transformer_blocks.0.attn.lora_B.bias"] = bias

        with pytest.raises(ValueError, match="LoRA bias"):
            Adapter.from_state_dict(state_dict=sd)

    def test_legacy_bias_obeys_dtype_and_adoption_policy(self) -> None:
        sd = _make_lora_sd(
            num_blocks=1,
            dim=16,
            rank=4,
            legacy_bias=True,
        )
        bias_source = sd["transformer_blocks.0.attn.lora_B.bias"]

        adopted = Adapter.from_state_dict(
            state_dict=sd,
            dtype=torch.float32,
            host_backing="adopt",
        )
        adopted_bias = _factor_bias(
            adopted.targets["transformer_blocks.0.attn.weight"],
        )
        assert adopted_bias is not None
        assert adopted_bias.data_ptr() == bias_source.data_ptr()

        cast = Adapter.from_state_dict(state_dict=sd, dtype=torch.bfloat16)
        cast_bias = _factor_bias(
            cast.targets["transformer_blocks.0.attn.weight"],
        )
        assert cast_bias is not None
        assert cast_bias.dtype is torch.bfloat16

    def test_factors_are_pinned(self) -> None:
        lora = _make_lora(4, 16)
        for factor in lora.targets.values():
            a, b = _factor_tensors(factor)
            assert a.is_pinned()
            assert b.is_pinned()

    def test_adopted_factors_retain_source_storage(self) -> None:
        sd = _make_lora_sd(num_blocks=1, dim=16, rank=4)
        a_source = sd["transformer_blocks.0.attn.lora_A.weight"]
        b_source = sd["transformer_blocks.0.attn.lora_B.weight"]

        lora = Adapter.from_state_dict(
            state_dict=sd,
            dtype=torch.float32,
            host_backing="adopt",
        )
        a, b = _factor_tensors(lora.targets["transformer_blocks.0.attn.weight"])

        assert not a.is_pinned()
        assert not b.is_pinned()
        assert a.data_ptr() == a_source.data_ptr()
        assert b.data_ptr() == b_source.data_ptr()

    def test_adopted_factors_preserve_mmap_storage(
        self,
        tmp_path: Path,
    ) -> None:
        rank, dim = 2, 4
        factor_elements = rank * dim
        mapped_elements = factor_elements * 2 + 7
        path = tmp_path / "lora.bin"
        path.write_bytes(array("f", map(float, range(mapped_elements))).tobytes())
        mapped = torch.from_file(
            str(path),
            shared=False,
            size=mapped_elements,
            dtype=torch.float32,
        )
        a_source = mapped[:factor_elements].view(rank, dim)
        b_source = mapped[factor_elements : factor_elements * 2].view(dim, rank)

        lora = Adapter.from_state_dict(
            state_dict={
                "target.lora_A.weight": a_source,
                "target.lora_B.weight": b_source,
            },
            host_backing="adopt",
        )
        a, b = _factor_tensors(lora.targets["target.weight"])

        assert a.data_ptr() == a_source.data_ptr()
        assert b.data_ptr() == b_source.data_ptr()
        assert lora.cache_bytes == a_source.nbytes + b_source.nbytes

    def test_adoption_rejects_dtype_conversion(self) -> None:
        sd = _make_lora_sd(num_blocks=1, dim=16, rank=4)

        with pytest.raises(ValueError, match="cannot convert adapter tensor dtype"):
            Adapter.from_state_dict(
                state_dict=sd,
                dtype=torch.bfloat16,
                host_backing="adopt",
            )

    def test_rejects_invalid_host_backing(self) -> None:
        sd = _make_lora_sd(num_blocks=1, dim=16, rank=4)

        with pytest.raises(ValueError, match="host_backing"):
            Adapter.from_state_dict(
                state_dict=sd,
                host_backing="invalid",
            )

    def test_adoption_rejects_non_contiguous_factor(self) -> None:
        sd = {
            "target.lora_A.weight": torch.randn(16, 4).t(),
            "target.lora_B.weight": torch.randn(16, 4),
        }

        with pytest.raises(ValueError, match="non-contiguous"):
            Adapter.from_state_dict(
                state_dict=sd,
                host_backing="adopt",
            )

    def test_factor_pinned_params_build_instance_without_repin(self) -> None:
        """Pinned factors can be consumed without cloning or re-pinning."""
        lora = _make_lora(1, 16, rank=4)
        factor = _require_factor(lora.targets["transformer_blocks.0.attn.weight"])
        assert isinstance(factor.a, PinnedParam)
        assert isinstance(factor.b, PinnedParam)

        holder = nn.Module()
        holder.register_parameter("a", factor.a.make_cpu_param())
        holder.register_parameter("b", factor.b.make_cpu_param())
        instance = PinnedModuleInstance(
            module=holder,
            params={"a": factor.a, "b": factor.b},
            buffers={},
        )
        # Consumers retain the exact pinned objects.
        assert instance.params["a"] is factor.a
        assert instance.params["b"] is factor.b

    def test_cache_bytes(self) -> None:
        lora = _make_lora(4, 16, rank=4)
        expected = 4 * (4 * 16 + 16 * 4) * 4  # 4 blocks * 2 factors * float32
        assert lora.cache_bytes == expected

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

    def test_targets_are_cached_and_immutable(self) -> None:
        lora = _make_lora(1, 16)
        targets = lora.targets
        assert lora.targets is targets
        with pytest.raises(TypeError):
            targets["other.weight"] = next(iter(targets.values()))  # type: ignore[index]

    @pytest.mark.parametrize("target_key", ["", 1])
    def test_direct_constructor_rejects_invalid_target_names(
        self,
        target_key: object,
    ) -> None:
        value = ParameterValue.from_tensor(torch.randn(2, 2), pin_memory=False)

        with pytest.raises(ValueError, match="non-empty strings"):
            Adapter({target_key: value})  # type: ignore[dict-item]

    def test_direct_constructor_rejects_invalid_target_type(self) -> None:
        with pytest.raises(ValueError, match="LoRAFactor or ParameterValue"):
            Adapter({"target.weight": object()})  # type: ignore[dict-item]

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
        assert lora.cache_bytes == weight.nbytes + bias.nbytes
        with pytest.raises(TypeError):
            lora.targets["other"] = next(iter(lora.targets.values()))  # type: ignore[index]

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

    def test_parameter_values_obey_dtype_and_adoption_policy(self) -> None:
        source = torch.randn(3, 4)
        adopted = Adapter.from_state_dict(
            {"target.weight": source},
            host_backing="adopt",
        )
        adopted_value = adopted.targets["target.weight"]
        assert isinstance(adopted_value, ParameterValue)
        adopted_tensor = adopted_value.backing.make_cpu_param().data
        assert adopted_tensor.data_ptr() == source.data_ptr()

        cast = Adapter.from_state_dict(
            {"target.weight": source},
            dtype=torch.bfloat16,
        )
        cast_value = cast.targets["target.weight"]
        assert isinstance(cast_value, ParameterValue)
        assert cast_value.backing.compute_dtype is torch.bfloat16

        with pytest.raises(ValueError, match="cannot convert adapter tensor dtype"):
            Adapter.from_state_dict(
                {"target.weight": source},
                dtype=torch.bfloat16,
                host_backing="adopt",
            )


# ---------------------------------------------------------------------------
# Adapter request validation
# ---------------------------------------------------------------------------


class TestActivationLoraValidation:
    def test_lora_strengths_default_to_one(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16)
        assert s._normalize_adapters([lora]) == [(lora, 1.0)]

    def test_accepts_lora_strengths(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16)
        assert s._normalize_adapters([lora], adapter_strengths=[0.25]) == [
            (lora, 0.25),
        ]

    def test_lora_strengths_do_not_silently_truncate(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        with pytest.raises(ValueError, match="shorter"):
            s._normalize_adapters(
                [_make_lora(4, 16)],
                adapter_strengths=[],
            )

    @pytest.mark.parametrize("zero", [0.0, -0.0])
    def test_zero_strengths_are_inactive(self, zero: float) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        inactive = _make_lora(4, 16, seed=1)
        active = _make_lora(4, 16, seed=2)

        assert s._normalize_adapters(
            [inactive, active],
            adapter_strengths=[zero, 0.25],
        ) == [(active, 0.25)]

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
            assert s._adapter_hook_removers == []
            torch.testing.assert_close(m(x), expected)
        finally:
            s.deactivate()

    def test_duplicate_lora_instances_keep_each_strength(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16)
        assert s._normalize_adapters(
            [lora, lora],
            adapter_strengths=[0.25, 0.75],
        ) == [(lora, 0.25), (lora, 0.75)]

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
        _request_loras(s, [(Adapter.from_state_dict(state_dict=sd), 1.0)])
        with pytest.raises(ValueError, match="B@A produces"):
            _activate_loras_for_test(s)
        assert not _has_post_copy_hook(
            s,
            "transformer_blocks.0.attn.weight",
        )

    def test_accepts_fp32_lora_target(self) -> None:
        m = _make_bf16_model().to(torch.float32)
        for p in m.parameters():
            p.requires_grad = False
        s = _make_strategy(m)
        _request_loras(s, [(_make_lora(4, 16), 1.0)])
        assert not _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")

    def test_non_block_targets_matched(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
        }
        _request_loras(s, [(Adapter.from_state_dict(state_dict=sd), 1.0)])
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "embed.weight")

    def test_legacy_bias_registers_separate_base_bias_hook(self) -> None:
        m = _make_bf16_model(num_blocks=2, attn_bias=True)
        s = _make_strategy(m)
        lora = _make_lora(
            num_blocks=2,
            dim=16,
            legacy_bias=True,
        )

        _request_loras(s, [(lora, 0.5)], mode="merge")
        _activate_loras_for_test(s)

        for block_idx in range(2):
            assert _has_post_copy_hook(
                s,
                f"transformer_blocks.{block_idx}.attn.weight",
            )
            assert _has_post_copy_hook(
                s,
                f"transformer_blocks.{block_idx}.attn.bias",
            )

    def test_legacy_bias_merge_rejects_biasless_base_before_hooks(self) -> None:
        m = _make_bf16_model(num_blocks=2, attn_bias=False)
        s = _make_strategy(m)
        lora = _make_lora(
            num_blocks=2,
            dim=16,
            legacy_bias=True,
        )

        _request_loras(s, [(lora, 0.5)], mode="merge")
        with pytest.raises(ValueError, match="attn.bias.*is not managed"):
            _activate_loras_for_test(s)

        assert s._adapter_hook_removers == []
        assert not _has_post_copy_hook(
            s,
            "transformer_blocks.0.attn.weight",
        )

    def test_legacy_bias_merge_rejects_invalid_base_bias_before_hooks(
        self,
    ) -> None:
        m = _make_bf16_model(num_blocks=1, attn_bias=True)
        m.transformer_blocks[0].attn.bias = nn.Parameter(
            torch.randn(1, 16, dtype=torch.bfloat16),
            requires_grad=False,
        )
        s = _make_strategy(m)
        lora = _make_lora(
            num_blocks=1,
            dim=16,
            legacy_bias=True,
        )

        _request_loras(s, [(lora, 0.5)], mode="merge")
        with pytest.raises(ValueError, match="rank-one base bias"):
            _activate_loras_for_test(s)

        assert s._adapter_hook_removers == []
        assert not _has_post_copy_hook(
            s,
            "transformer_blocks.0.attn.weight",
        )
        assert not _has_post_copy_hook(
            s,
            "transformer_blocks.0.attn.bias",
        )

    def test_non_block_tied_alias_target_matched(self) -> None:
        m = _make_tied_non_block_model(dtype=torch.bfloat16)
        s = _make_strategy(m)
        sd = {
            "head.lora_A.weight": torch.randn(4, 16),
            "head.lora_B.weight": torch.randn(16, 4),
        }
        _request_loras(s, [(Adapter.from_state_dict(state_dict=sd), 1.0)], mode="merge")
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "head.weight")
        assert _has_post_copy_hook(s, "embed.weight")

    def test_rejects_duplicate_tied_alias_targets(self) -> None:
        m = _make_tied_non_block_model(dtype=torch.bfloat16)
        s = _make_strategy(m)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
            "head.lora_A.weight": torch.randn(4, 16),
            "head.lora_B.weight": torch.randn(16, 4),
        }
        _request_loras(s, [(Adapter.from_state_dict(state_dict=sd), 1.0)], mode="merge")
        with pytest.raises(RuntimeError, match="shared adapter targets"):
            _activate_loras_for_test(s)
        assert not _has_post_copy_hook(s, "embed.weight")
        assert not _has_post_copy_hook(s, "head.weight")

    def test_streamed_block_shared_submodule_alias_target_matched(self) -> None:
        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Linear(16, 16, bias=False)
                self.b = self.a

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.b(self.a(x))

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer_blocks = nn.ModuleList([Block(), Block()])

        m = M().to(torch.bfloat16)
        m.requires_grad_(False)
        s = _make_strategy(m)
        sd = {
            "transformer_blocks.0.b.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.b.lora_B.weight": torch.randn(16, 4),
        }
        _request_loras(s, [(Adapter.from_state_dict(state_dict=sd), 1.0)], mode="merge")
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "transformer_blocks.0.b.weight")
        assert _has_post_copy_hook(s, "transformer_blocks.0.a.weight")

    def test_exact_keys_match(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16)
        _request_loras(s, [(lora, 1.0)])
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")

    def test_prefixed_keys_rejected(self) -> None:
        # Keys are verbatim, so a ``diffusion_model.``-prefixed adapter does not
        # match the model's params — the caller must strip it before building.
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16, prefix="diffusion_model.")
        _request_loras(s, [(lora, 1.0)])
        with pytest.raises(ValueError, match="Adapter target .* is not managed"):
            _activate_loras_for_test(s)
        assert not _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")

    @pytest.mark.parametrize("mode", ["merge", "routed"])
    def test_partial_targets_apply_intersection(self, mode: AdapterMode) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16)
        s = _make_strategy(m)
        sd = _make_lora_sd(num_blocks=1, dim=16)
        sd.update(_make_lora_sd(num_blocks=1, dim=16, prefix="missing."))
        lora = Adapter.from_state_dict(sd, allow_partial_targets=True)

        _request_loras(s, [(lora, 1.0)], mode=mode)

        assert _activate_loras_for_test(s) == 1
        if mode == "merge":
            assert _has_post_copy_hook(
                s,
                "transformer_blocks.0.attn.weight",
            )

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
        _request_loras(s, [(Adapter.from_state_dict(state_dict=sd), 1.0)])

        with pytest.raises(ValueError, match="Adapter target .* is not managed"):
            _activate_loras_for_test(s)
        assert not _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")
        assert not _has_post_copy_hook(
            s,
            "transformer_blocks.0.attn.base_layer.weight",
        )

    def test_merge_mode_activation_rejects_cpu(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16)
        _request_loras(s, [(lora, 1.0)], mode="merge")
        with pytest.raises(ValueError, match="merge mode requires CUDA"):
            _activate(s, "cpu")

    def test_clear_active_adapter_hooks_clears_previous_merge_hooks(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        _request_loras(s, [(_make_lora(4, 16, rank=4), 1.0)])
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")
        s._clear_active_adapter_hooks()
        assert not _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")

    def test_accepts_quanto_target_in_merge_mode(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        m = _make_bf16_model()
        rows = cols = 16
        data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16)
        qt = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            data,
            scale,
            None,
        )
        m.embed.weight = nn.Parameter(qt, requires_grad=False)

        s = _make_strategy(m)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
        }
        _request_loras(s, [(Adapter.from_state_dict(state_dict=sd), 1.0)], mode="merge")
        assert _activate_loras_for_test(s) == 1
        assert _has_post_copy_hook(s, "embed.weight")

        # routed mode must still accept it.
        _request_loras(s, [(Adapter.from_state_dict(state_dict=sd), 1.0)], mode="routed")
        route_count = _activate_loras_for_test(s)
        assert route_count == 1

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
        _request_loras(s, [(Adapter.from_state_dict(state_dict=sd), 1.0)], mode="merge")
        with pytest.raises(ValueError, match="floating-point compute dtype"):
            _activate_loras_for_test(s)
        assert not _has_post_copy_hook(s, "embed.weight")

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
        loras = [(_make_lora(2, 16, seed=9), 0.75)]
        s = _make_strategy(m)
        _request_loras(s, loras, mode="routed")

        x = torch.randn(2, 16)
        _activate(s, "cpu")
        try:
            actual = m(x)
            expected = _expected_routed_output(m, x, loras)
            assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
            assert not _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")
        finally:
            s.deactivate()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @CUDA
    def test_activate_runs_components(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        _request_loras(s, [(_make_lora(4, 16), 1.0)])
        try:
            _activate(s, "cuda")
            assert m.embed.weight.is_cuda
            assert m.head.weight.is_cuda
        finally:
            s.deactivate()

    @CUDA
    def test_deactivate_returns_to_pinned(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        _request_loras(s, [(_make_lora(4, 16), 1.0)])
        _activate(s, "cuda")
        s.deactivate()
        assert m.embed.weight.is_pinned()
        assert m.head.weight.is_pinned()

    @CUDA
    def test_reactivation_with_different_loras(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        _request_loras(s, [(_make_lora(4, 16, seed=1), 1.0)])
        _activate(s, "cuda")
        s.deactivate()
        _request_loras(s, [(_make_lora(4, 16, seed=2), 1.0)])
        _activate(s, "cuda")
        s.deactivate()
        assert m.embed.weight.is_pinned()

    @CUDA
    def test_base_only_reactivation_does_not_reuse_previous_merge_hooks(self) -> None:
        m = _make_bf16_model(num_blocks=4, dim=16)
        base_embed = m.embed.weight.detach().clone()
        base_block = m.transformer_blocks[0].attn.weight.detach().clone()

        sd = _make_lora_sd(num_blocks=4, dim=16, seed=3)
        g = torch.Generator().manual_seed(303)
        sd["embed.lora_A.weight"] = torch.randn(
            4,
            16,
            generator=g,
            dtype=torch.float32,
        )
        sd["embed.lora_B.weight"] = torch.randn(
            16,
            4,
            generator=g,
            dtype=torch.float32,
        )
        s = _make_strategy(m)
        _request_loras(s, [(Adapter.from_state_dict(state_dict=sd), 1.0)], mode="merge")
        _activate(s, "cuda")
        s.deactivate()

        _request_loras(s, [])
        _activate(s, "cuda")
        try:
            torch.cuda.synchronize()
            torch.testing.assert_close(
                m.embed.weight.detach().cpu(),
                base_embed,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                m.transformer_blocks[0].attn.weight.detach().cpu(),
                base_block,
                rtol=0.0,
                atol=0.0,
            )
        finally:
            s.deactivate()

    @CUDA
    def test_activate_with_no_loras_runs_base_only(self) -> None:
        m = _make_bf16_model()
        captured = m.transformer_blocks[0].attn.weight.detach().clone()
        s = _make_strategy(m)
        _activate(s, "cuda")
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            actual = m.transformer_blocks[0].attn.weight.detach()
            assert torch.allclose(
                actual.cpu(),
                captured,
                rtol=0.0,
                atol=0.0,
            ), "no LoRAs must leave base weights unmodified"
        finally:
            s.deactivate()


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
        offloader = _make_model_offloader(
            model,
            transient_paths=["embed"],
        )
        component = dict(transient_components(offloader))["embed"]
        _request_loras(offloader, [(lora, 0.5)], mode=mode)
        _activate(offloader, "cuda")
        try:
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
        finally:
            offloader.deactivate()

        torch.testing.assert_close(second, first, rtol=0, atol=0)

    @CUDA
    @pytest.mark.parametrize("mode", ["merge", "routed"])
    def test_transient_block_path_preserves_lora_on_reacquire(
        self,
        mode: AdapterMode,
    ) -> None:
        model = _make_bf16_model(num_blocks=2, dim=16)
        lora = _make_lora(num_blocks=2, dim=16, seed=9)
        offloader = _make_model_offloader(
            model,
            transient_block_paths=["transformer_blocks"],
        )
        _request_loras(offloader, [(lora, 0.5)], mode=mode)
        _activate(offloader, "cuda")
        try:
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
        finally:
            offloader.deactivate()

        torch.testing.assert_close(second, first, rtol=0, atol=0)

    @CUDA
    @pytest.mark.parametrize("streamed", [False, True])
    def test_legacy_bias_merge_resident_and_streamed(
        self,
        streamed: bool,
    ) -> None:
        m = _make_bf16_model(
            num_blocks=2,
            dim=16,
            attn_bias=True,
        )
        base_biases = [block.attn.bias.detach().clone() for block in m.transformer_blocks]
        loras = [
            (
                _make_lora(
                    num_blocks=2,
                    dim=16,
                    seed=10,
                    legacy_bias=True,
                ),
                0.5,
            ),
            (
                _make_lora(
                    num_blocks=2,
                    dim=16,
                    seed=20,
                    legacy_bias=True,
                ),
                -0.25,
            ),
        ]
        strategy = _make_model_offloader(
            m,
            block_paths=["transformer_blocks"] if streamed else [],
        )
        _request_loras(strategy, loras, mode="merge")
        _activate(strategy, "cuda")
        try:
            streamer = block_components(strategy)[0] if streamed else None
            for block_idx, block in enumerate(m.transformer_blocks):
                if streamer is not None:
                    streamer._runtime._before_block_forward(block_idx)
                actual = block.attn.bias
                assert actual is not None
                expected = base_biases[block_idx].to(actual.device)
                delta = torch.zeros_like(expected)
                target_name = f"transformer_blocks.{block_idx}.attn.weight"
                for lora, strength in loras:
                    bias = _factor_bias(lora.targets[target_name])
                    assert bias is not None
                    delta.add_(
                        bias.to(device=actual.device, dtype=actual.dtype),
                        alpha=strength,
                    )
                torch.testing.assert_close(actual, expected + delta)
        finally:
            strategy.deactivate()

    @CUDA
    @pytest.mark.parametrize("mode", ["merge", "routed"])
    def test_adopted_lora_matches_pinned_lora(
        self,
        mode: AdapterMode,
    ) -> None:
        torch.manual_seed(42)
        pinned_model = _make_bf16_model(num_blocks=2, dim=16)
        pageable_model = _make_bf16_model(num_blocks=2, dim=16)
        pageable_model.load_state_dict(pinned_model.state_dict())
        sd = _make_lora_sd(num_blocks=2, dim=16, rank=4, seed=7)
        pinned_lora = Adapter.from_state_dict(state_dict=sd)
        pageable_lora = Adapter.from_state_dict(
            state_dict=sd,
            host_backing="adopt",
        )
        x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")

        def run(model: nn.Module, lora: Adapter) -> torch.Tensor:
            strategy = _make_strategy(model)
            _request_loras(strategy, [(lora, 0.75)], mode=mode)
            _activate(strategy, "cuda")
            try:
                output = model(x)
                torch.cuda.synchronize()
                return output.detach().cpu()
            finally:
                strategy.deactivate()

        torch.testing.assert_close(
            run(pageable_model, pageable_lora),
            run(pinned_model, pinned_lora),
        )

    @CUDA
    def test_merged_weights_match_manual_baseline(self) -> None:
        m = _make_bf16_model(num_blocks=4, dim=16)
        captured_base = {i: m.transformer_blocks[i].attn.weight.detach().clone() for i in range(4)}

        loras = [
            (_make_lora(num_blocks=4, dim=16, seed=10), 0.5),
            (_make_lora(num_blocks=4, dim=16, seed=20), 0.25),
        ]
        s = _make_strategy(m)
        _request_loras(s, loras)
        _activate(s, "cuda")
        try:
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
        finally:
            s.deactivate()

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
        _request_loras(s, [(lora, 0.5)])
        _activate(s, "cuda")
        try:
            factor = lora.targets["embed.weight"]
            a, b = _factor_tensors(factor)
            expected = (captured_embed + 0.5 * (b.to(torch.bfloat16) @ a.to(torch.bfloat16))).to("cuda")
            actual = m.embed.weight.detach()
            assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), (
                f"non-block merge mismatch:\n  expected: {expected.flatten()[:4]}\n  actual:   {actual.flatten()[:4]}"
            )
        finally:
            s.deactivate()

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
        _request_loras(s, [(lora, 0.7)])
        _activate(s, "cuda")
        try:
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
        finally:
            s.deactivate()


class TestLoRATransform:
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

        with pytest.raises(ValueError, match="B@A produces"):
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

    def test_joint_transform_applies_weight_and_bias(self) -> None:
        weight = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        bias = nn.Parameter(torch.randn(4), requires_grad=False)
        weight_before = weight.detach().clone()
        bias_before = bias.detach().clone()
        a = torch.randn(2, 8)
        b = torch.randn(4, 2)
        adapter_bias = torch.randn(4)
        strength = -0.75
        transform = LoRATransform(
            [
                ScaledLoRAFactor.from_tensors(
                    a,
                    b,
                    strength,
                    bias=adapter_bias,
                )
            ]
        )

        transform.validate_target(weight, bias)
        transform.apply(weight, bias)

        expected_weight = weight_before.clone()
        expected_weight.addmm_(b, a, alpha=strength)
        torch.testing.assert_close(weight, expected_weight)
        torch.testing.assert_close(
            bias,
            bias_before + adapter_bias * strength,
        )

    def test_joint_transform_requires_bias_before_weight_mutation(self) -> None:
        weight = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        weight_before = weight.detach().clone()
        transform = LoRATransform(
            [
                ScaledLoRAFactor.from_tensors(
                    torch.randn(2, 8),
                    torch.randn(4, 2),
                    0.5,
                    bias=torch.randn(4),
                )
            ]
        )

        with pytest.raises(ValueError, match="no base bias target"):
            transform.apply(weight)

        torch.testing.assert_close(weight, weight_before)

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
            strength: float,
            *,
            rounding_seed: int | None = None,
        ) -> None:
            nonlocal merge_calls
            merge_calls += 1
            original(
                target,
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
    def test_adapter_staging_uses_logical_shape_and_compute_dtype(
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
            logical_shape=(rows, cols),
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
        class ConversionOnlyAdapter(GgufAdapter):
            @staticmethod
            def dequantize(t: torch.Tensor) -> torch.Tensor:
                return t

            @staticmethod
            def requantize(
                t: torch.Tensor,
                *,
                like: torch.Tensor,
            ) -> torch.Tensor:
                del like
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

    def test_quanto_transform_delegates_cpu_merge_in_place(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols, rank = 4, 8, 2
        data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25)
        qt = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            data,
            scale,
            None,
        )
        param = nn.Parameter(qt, requires_grad=False)
        a = torch.randn(rank, cols)
        b = torch.randn(rows, rank)
        transform = LoRATransform([ScaledLoRAFactor.from_tensors(a, b, 0.5)])
        original_param = param
        original_packed_ptr = param.data._data.data_ptr()
        original_scale_ptr = param.data._scale.data_ptr()
        expected_dense = qt.dequantize()
        merge_calls: list[tuple[str, tuple[int, ...], tuple[int, ...], float]] = []
        original_merge = QuantoAdapter.merge_lora_

        def tracked_merge(
            target: torch.Tensor,
            staged_b: torch.Tensor,
            staged_a: torch.Tensor,
            strength: float,
            *,
            rounding_seed: int | None = None,
        ) -> None:
            merge_calls.append(
                (
                    target.device.type,
                    tuple(staged_b.shape),
                    tuple(staged_a.shape),
                    strength,
                )
            )
            original_merge(
                target,
                staged_b,
                staged_a,
                strength,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(
            QuantoAdapter,
            "merge_lora_",
            staticmethod(tracked_merge),
        )

        transform.validate_target(param)
        transform.apply(param)

        expected_dense.addmm_(
            b.to(expected_dense.dtype),
            a.to(expected_dense.dtype),
            alpha=0.5,
        )
        expected = _quanto_absmax_oracle(expected_dense, like=qt)
        assert param is original_param
        assert param.data._data.data_ptr() == original_packed_ptr
        assert param.data._scale.data_ptr() == original_scale_ptr
        assert isinstance(param.data, WeightQBytesTensor)
        assert merge_calls == [("cpu", (rows, rank), (rank, cols), 0.5)]
        torch.testing.assert_close(param.data._data, expected._data)
        torch.testing.assert_close(param.data._scale, expected._scale)

    def test_quanto_validation_reads_shape_without_dequantizing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        from piper_offload.quanto_adapter import QuantoAdapter

        rows, cols, rank = 4, 8, 2
        qt = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            torch.randint(-32, 32, (rows, cols), dtype=torch.int8),
            torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25),
            None,
        )
        transform = LoRATransform(
            [
                ScaledLoRAFactor.from_tensors(
                    torch.randn(rank, cols),
                    torch.randn(rows, rank),
                    0.5,
                )
            ]
        )

        def fail_dequantize(_tensor: torch.Tensor) -> torch.Tensor:
            raise AssertionError("validation should not dequantize")

        monkeypatch.setattr(
            QuantoAdapter,
            "dequantize",
            staticmethod(fail_dequantize),
        )

        transform.validate_target(nn.Parameter(qt, requires_grad=False))

    @pytest.mark.parametrize("axis", [0, -1, 1, None])
    @pytest.mark.parametrize("shape", [(0, 4), (4, 0)])
    @pytest.mark.parametrize(
        "qtype_name",
        ["qint8", "qfloat8_e4m3fn", "qfloat8_e5m2"],
    )
    def test_quanto_empty_merge_is_scale_preserving_noop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        axis: int | None,
        shape: tuple[int, int],
        qtype_name: str,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = shape
        qtype = getattr(quanto, qtype_name)
        scale_shape = (rows, 1) if axis == 0 else ((1, cols) if axis in (-1, 1) else ())
        scale_numel = rows if axis == 0 else (cols if axis in (-1, 1) else 1)
        scale = torch.arange(scale_numel, dtype=torch.float32).reshape(scale_shape)
        assert scale.numel() == 0 or torch.any(scale == 0)
        data = torch.empty(shape, dtype=qtype.dtype)
        qt = WeightQBytesTensor(
            qtype,
            axis,
            shape,
            data.stride(),
            data,
            scale,
            quanto.qint8,
        )
        b = torch.randn(rows, 2)
        a = torch.randn(2, cols)
        data_before = qt._data.clone()
        scale_before = qt._scale.clone()
        data_ptr = qt._data.data_ptr()
        scale_ptr = qt._scale.data_ptr()
        generic_calls = 0
        original_generic = quanto_adapter_impl._torch_merge_quanto_lora

        def tracked_generic(
            target: torch.Tensor,
            staged_b: torch.Tensor,
            staged_a: torch.Tensor,
            strength: float,
            *,
            rounding_seed: int | None = None,
        ) -> torch.Tensor | None:
            nonlocal generic_calls
            generic_calls += 1
            return original_generic(
                target,
                staged_b,
                staged_a,
                strength,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(
            quanto_adapter_impl,
            "_torch_merge_quanto_lora",
            tracked_generic,
        )

        QuantoAdapter.validate_lora_merge(qt, b, a, 0.5)
        QuantoAdapter.merge_lora_(qt, b, a, 0.5)

        assert generic_calls == 1
        assert qt._data.data_ptr() == data_ptr
        assert qt._scale.data_ptr() == scale_ptr
        assert qt.qtype is qtype
        assert qt.activation_qtype is quanto.qint8
        torch.testing.assert_close(qt._data, data_before)
        torch.testing.assert_close(qt._scale, scale_before)

    @pytest.mark.parametrize(
        ("axis", "shape", "scale_shape"),
        [
            (0, (0, 4), (1, 4)),
            (-1, (4, 0), (4, 1)),
            (1, (4, 0), (4, 1)),
            (None, (0, 4), (0,)),
        ],
    )
    def test_quanto_empty_validation_rejects_malformed_scale_layout(
        self,
        axis: int | None,
        shape: tuple[int, int],
        scale_shape: tuple[int, ...],
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = shape
        data = torch.empty(shape, dtype=torch.int8)
        qt = WeightQBytesTensor(
            quanto.qint8,
            axis,
            shape,
            data.stride(),
            data,
            torch.empty(scale_shape),
            None,
        )
        b = torch.randn(rows, 2)
        a = torch.randn(2, cols)

        with pytest.raises(ValueError, match="Quanto LoRA merge expects"):
            QuantoAdapter.validate_lora_merge(qt, b, a, 0.5)

    def test_quanto_empty_requantize_rejects_malformed_reference(
        self,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 0
        data = torch.empty(rows, cols, dtype=torch.int8)
        malformed = WeightQBytesTensor(
            quanto.qint8,
            0,
            (rows, cols),
            data.stride(),
            data,
            torch.ones(1, rows),
            None,
        )

        with pytest.raises(ValueError, match="scale storage does not match"):
            QuantoAdapter.requantize(torch.empty(rows, cols), like=malformed)

        unsupported_qtype = WeightQBytesTensor(
            quanto.qint4,
            0,
            (rows, cols),
            data.stride(),
            data,
            torch.ones(rows, 1),
            None,
        )
        with pytest.raises(ValueError, match="8-bit qtype"):
            QuantoAdapter.requantize(
                torch.empty(rows, cols),
                like=unsupported_qtype,
            )

    def test_quanto_zero_rank_is_rejected_before_merge(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 6
        data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1).add_(0.25)
        qt = WeightQBytesTensor(
            quanto.qint8,
            0,
            (rows, cols),
            data.stride(),
            data,
            scale,
            None,
        )
        b = torch.empty(rows, 0)
        a = torch.empty(0, cols)
        data_before = data.clone()
        scale_before = scale.clone()

        def fail_merge(*_args: object) -> torch.Tensor:
            raise AssertionError("zero-rank Quanto factors reached merge math")

        monkeypatch.setattr(
            quanto_adapter_impl,
            "_torch_merge_quanto_lora",
            fail_merge,
        )
        monkeypatch.setattr(
            quanto_adapter_impl,
            "_triton_merge_quanto_qint8_lora",
            fail_merge,
        )

        with pytest.raises(ValueError, match="positive LoRA rank"):
            QuantoAdapter.validate_lora_merge(qt, b, a, 1.0)

        torch.testing.assert_close(qt._data, data_before)
        torch.testing.assert_close(qt._scale, scale_before)

    @CUDA
    def test_quanto_empty_cuda_layout_falls_back_from_triton(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        pytest.importorskip("triton")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols, rank = 4, 0, 2
        data = torch.empty(rows, cols, device="cuda", dtype=torch.int8)
        scale = torch.tensor(
            [[0.0], [0.25], [0.5], [1.0]],
            device="cuda",
        )
        qt = WeightQBytesTensor(
            quanto.qint8,
            0,
            (rows, cols),
            data.stride(),
            data,
            scale,
            None,
        )
        scale_before = scale.clone()
        generic_calls = 0
        original_generic = quanto_adapter_impl._torch_merge_quanto_lora

        def fail_triton(*_args: object) -> tuple[torch.Tensor, torch.Tensor]:
            raise AssertionError("empty Quanto layout reached Triton")

        def tracked_generic(
            target: torch.Tensor,
            b: torch.Tensor,
            a: torch.Tensor,
            strength: float,
            *,
            rounding_seed: int | None = None,
        ) -> torch.Tensor | None:
            nonlocal generic_calls
            generic_calls += 1
            return original_generic(
                target,
                b,
                a,
                strength,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(
            quanto_adapter_impl,
            "_triton_merge_quanto_qint8_lora",
            fail_triton,
        )
        monkeypatch.setattr(
            quanto_adapter_impl,
            "_torch_merge_quanto_lora",
            tracked_generic,
        )

        QuantoAdapter.merge_lora_(
            qt,
            torch.randn(rows, rank, device="cuda"),
            torch.randn(rank, cols, device="cuda"),
            0.5,
        )

        assert generic_calls == 1
        torch.testing.assert_close(qt._scale, scale_before)

    @pytest.mark.parametrize("axis", [0, -1, 1, None])
    @pytest.mark.parametrize(
        "qtype_name",
        ["qint8", "qfloat8_e4m3fn", "qfloat8_e5m2"],
    )
    def test_quanto_generic_requantize_matches_absmax_oracle_and_repairs_zero(
        self,
        axis: int | None,
        qtype_name: str,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 6
        qtype = getattr(quanto, qtype_name)
        scale_shape = (rows, 1) if axis == 0 else ((1, cols) if axis in (-1, 1) else ())
        like = WeightQBytesTensor(
            qtype,
            axis,
            (rows, cols),
            (cols, 1),
            torch.zeros(rows, cols, dtype=qtype.dtype),
            torch.full(scale_shape, 1.0 / qtype.qmax),
            None,
        )
        dense = torch.linspace(-9.0, 12.0, rows * cols).reshape(rows, cols)
        if axis == 0:
            dense[0].zero_()
        elif axis in (-1, 1):
            dense[:, 0].zero_()

        expected = _quanto_absmax_oracle(dense, like=like)
        actual = QuantoAdapter.requantize(dense, like=like)

        assert actual.qtype is qtype
        assert actual.axis == axis
        assert actual._scale.max() > like._scale.max()
        torch.testing.assert_close(actual._scale, expected._scale)
        torch.testing.assert_close(actual._data.float(), expected._data.float())

        # Quanto's raw AbsmaxOptimizer produces scale zero for an exact-zero
        # block; qfloat8's subsequent 0/0 quantization produces NaNs. The
        # adapter deliberately floors only those scales and stores zero codes.
        all_zero = QuantoAdapter.requantize(torch.zeros_like(dense), like=like)
        assert torch.all(all_zero._scale == torch.finfo(torch.float32).eps)
        assert torch.count_nonzero(all_zero._data.float()).item() == 0
        assert torch.isfinite(all_zero.dequantize()).all()

    @pytest.mark.parametrize(
        "qtype_name",
        ["qfloat8_e4m3fn", "qfloat8_e5m2"],
    )
    @pytest.mark.parametrize("axis", [0, -1, 1, None])
    def test_quanto_generic_merge_recovers_real_zero_scale_qfloat8(
        self,
        qtype_name: str,
        axis: int | None,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.optimizers import AbsmaxOptimizer
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols, rank = 4, 6, 2
        qtype = getattr(quanto, qtype_name)
        base = torch.linspace(-3.0, 4.0, rows * cols).reshape(rows, cols)
        if axis == 0:
            base[0].zero_()
        elif axis in (-1, 1):
            base[:, 0].zero_()
        else:
            base.zero_()
        optimizer_axis = -1 if axis == base.ndim - 1 else axis
        scale = AbsmaxOptimizer()(base, qtype=qtype, axis=optimizer_axis)
        quantized = WeightQBytesTensor.quantize(
            base,
            qtype,
            optimizer_axis,
            scale,
            None,
            optimized=False,
        )
        qt = (
            WeightQBytesTensor(
                qtype,
                axis,
                quantized.size(),
                quantized.stride(),
                quantized._data,
                quantized._scale,
                quantized.activation_qtype,
            )
            if axis != optimizer_axis
            else quantized
        )
        assert torch.isnan(qt._data.float()).any()
        safe_base = QuantoAdapter.dequantize(qt)
        assert torch.isfinite(safe_base).all()
        b = torch.ones(rows, rank)
        a = torch.ones(rank, cols)
        expected = _quanto_absmax_oracle(
            safe_base.addmm(b, a),
            like=qt,
        )

        QuantoAdapter.merge_lora_(qt, b, a, 1.0)

        torch.testing.assert_close(qt._scale, expected._scale)
        torch.testing.assert_close(qt._data.float(), expected._data.float())
        assert torch.isfinite(qt.dequantize()).all()

    @CUDA
    @pytest.mark.parametrize("axis", [0, -1, 1, None])
    @pytest.mark.parametrize(
        "dtype",
        [torch.float16, torch.bfloat16, torch.float32],
    )
    def test_triton_quanto_qint8_matches_absmax_oracle(
        self,
        monkeypatch: pytest.MonkeyPatch,
        axis: int | None,
        dtype: torch.dtype,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        pytest.importorskip("triton")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        torch.manual_seed(41)
        rows, cols, rank = 70, 130, 7
        scale_shape = (rows, 1) if axis == 0 else ((1, cols) if axis in (-1, 1) else ())
        data = torch.randint(
            -32,
            33,
            (rows, cols),
            device="cuda",
            dtype=torch.int8,
        )
        scale = torch.rand(
            scale_shape,
            device="cuda",
            dtype=dtype,
        ).add_(0.25)
        qt = WeightQBytesTensor.create(
            quanto.qint8,
            axis,
            (rows, cols),
            (cols, 1),
            data,
            scale,
            None,
        )
        a = torch.randn(
            cols,
            rank,
            device="cuda",
            dtype=dtype,
        ).t()
        b = torch.randn(
            rank,
            rows,
            device="cuda",
            dtype=dtype,
        ).t()
        assert not a.is_contiguous()
        assert not b.is_contiguous()
        expected_dense = qt.dequantize()
        expected_dense.addmm_(b, a, alpha=0.375)
        expected = _quanto_absmax_oracle(expected_dense, like=qt)
        data_ptr = qt._data.data_ptr()
        scale_ptr = qt._scale.data_ptr()

        def fail_fallback(
            _target: torch.Tensor,
            _b: torch.Tensor,
            _a: torch.Tensor,
            _strength: float,
        ) -> torch.Tensor:
            raise AssertionError("supported CUDA Quanto qint8 must use Triton")

        monkeypatch.setattr(
            quanto_adapter_impl,
            "_torch_merge_quanto_lora",
            fail_fallback,
        )
        QuantoAdapter.merge_lora_(qt, b, a, 0.375)
        torch.cuda.synchronize()

        assert qt._data.data_ptr() == data_ptr
        assert qt._scale.data_ptr() == scale_ptr
        assert qt.qtype is quanto.qint8
        assert qt.axis == axis
        torch.testing.assert_close(
            qt._scale,
            expected._scale,
            rtol=0.02,
            atol=torch.finfo(dtype).eps,
        )
        difference = (qt._data.to(torch.int16) - expected._data.to(torch.int16)).abs()
        max_qbyte_error = 2 if dtype is torch.bfloat16 else 1
        assert difference.max().item() <= max_qbyte_error

    @CUDA
    def test_quanto_qint8_falls_back_when_triton_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols, rank = 16, 24, 5
        scale = torch.rand(
            rows,
            1,
            device="cuda",
            dtype=torch.bfloat16,
        ).add_(0.25)
        qt = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            torch.randint(
                -32,
                33,
                (rows, cols),
                device="cuda",
                dtype=torch.int8,
            ),
            scale,
            None,
        )
        a = torch.randn(rank, cols, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(rows, rank, device="cuda", dtype=torch.bfloat16)
        expected_dense = qt.dequantize()
        expected_dense.addmm_(b, a, alpha=-0.25)
        expected = _quanto_absmax_oracle(expected_dense, like=qt)

        monkeypatch.setattr(
            quanto_adapter_impl,
            "_triton_merge_quanto_qint8_lora",
            None,
        )
        QuantoAdapter.merge_lora_(qt, b, a, -0.25)

        assert torch.equal(qt._data, expected._data)
        assert torch.equal(qt._scale, expected._scale)

    @CUDA
    @pytest.mark.parametrize("axis", [0, -1, 1, None])
    @pytest.mark.parametrize(
        "qtype_name",
        ["qfloat8_e4m3fn", "qfloat8_e5m2"],
    )
    @pytest.mark.parametrize(
        "dtype",
        [torch.float16, torch.bfloat16, torch.float32],
    )
    def test_triton_quanto_qfloat8_matches_absmax_oracle(
        self,
        monkeypatch: pytest.MonkeyPatch,
        axis: int | None,
        qtype_name: str,
        dtype: torch.dtype,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        pytest.importorskip("triton")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols, rank = 70, 130, 7
        qtype = getattr(quanto, qtype_name)
        scale_shape = (rows, 1) if axis == 0 else ((1, cols) if axis in (-1, 1) else ())
        qt = WeightQBytesTensor.create(
            qtype,
            axis,
            (rows, cols),
            (cols, 1),
            torch.randn(rows, cols, device="cuda").to(qtype.dtype),
            torch.rand(
                scale_shape,
                device="cuda",
                dtype=dtype,
            ).add_(0.25),
            quanto.qint8,
        )
        a = torch.randn(rank, cols, device="cuda", dtype=dtype)
        b = torch.randn(rows, rank, device="cuda", dtype=dtype)
        expected_dense = qt.dequantize()
        expected_dense.addmm_(b, a, alpha=0.5)
        expected = _quanto_absmax_oracle(expected_dense, like=qt)
        data_ptr = qt._data.data_ptr()
        scale_ptr = qt._scale.data_ptr()

        def fail_fallback(
            _target: torch.Tensor,
            _b: torch.Tensor,
            _a: torch.Tensor,
            _strength: float,
        ) -> torch.Tensor:
            raise AssertionError("supported CUDA Quanto qfloat8 must use Triton")

        monkeypatch.setattr(
            quanto_adapter_impl,
            "_torch_merge_quanto_lora",
            fail_fallback,
        )
        QuantoAdapter.merge_lora_(qt, b, a, 0.5)
        torch.cuda.synchronize()

        assert qt._data.data_ptr() == data_ptr
        assert qt._scale.data_ptr() == scale_ptr
        assert qt.qtype is qtype
        assert qt.axis == axis
        assert qt.activation_qtype is quanto.qint8
        torch.testing.assert_close(
            qt._scale,
            expected._scale,
            rtol=0.02,
            atol=torch.finfo(dtype).eps,
        )
        differing_codes = torch.count_nonzero(qt._data.view(torch.uint8) != expected._data.view(torch.uint8)).item()
        # BF16 tiled accumulation can move values across an adjacent FP8
        # quantization boundary more often than FP16/FP32, but the affected
        # fraction must remain small and the reconstructed weight must still
        # match the independent Quanto round trip below.
        assert differing_codes <= qt._data.numel() // 20 + 1
        torch.testing.assert_close(
            qt.dequantize().float(),
            expected.dequantize().float(),
            rtol=0.3 if qtype.dtype is torch.float8_e5m2 else 0.13,
            atol=0.15 if qtype.dtype is torch.float8_e5m2 else 0.05,
        )

    @CUDA
    @pytest.mark.parametrize(
        "qtype_name",
        ["qfloat8_e4m3fn", "qfloat8_e5m2"],
    )
    @pytest.mark.parametrize("nonzero_update", [False, True])
    def test_triton_quanto_qfloat8_recovers_real_zero_scale(
        self,
        monkeypatch: pytest.MonkeyPatch,
        qtype_name: str,
        nonzero_update: bool,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        pytest.importorskip("triton")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols, rank = 8, 12, 3
        qtype = getattr(quanto, qtype_name)
        base = torch.zeros(rows, cols, device="cuda", dtype=torch.bfloat16)
        scale = torch.zeros(rows, 1, device="cuda", dtype=torch.bfloat16)
        qt = WeightQBytesTensor.quantize(
            base,
            qtype,
            0,
            scale,
            None,
            optimized=False,
        )
        assert torch.isnan(qt._data.float()).all()
        data_ptr = qt._data.data_ptr()
        scale_ptr = qt._scale.data_ptr()
        b = torch.full(
            (rows, rank),
            float(nonzero_update),
            device="cuda",
            dtype=torch.bfloat16,
        )
        a = torch.ones(rank, cols, device="cuda", dtype=torch.bfloat16)
        expected = _quanto_absmax_oracle(b @ a, like=qt)

        def fail_fallback(*_args: object) -> torch.Tensor:
            raise AssertionError("supported CUDA Quanto qfloat8 must use Triton")

        monkeypatch.setattr(
            quanto_adapter_impl,
            "_torch_merge_quanto_lora",
            fail_fallback,
        )
        QuantoAdapter.merge_lora_(
            qt,
            b,
            a,
            1.0,
        )
        torch.cuda.synchronize()

        assert qt._data.data_ptr() == data_ptr
        assert qt._scale.data_ptr() == scale_ptr
        torch.testing.assert_close(qt._scale, expected._scale)
        torch.testing.assert_close(qt._data.float(), expected._data.float())
        assert torch.isfinite(qt.dequantize()).all()

    @CUDA
    @pytest.mark.parametrize("fallback", ["unavailable", "unsupported-storage"])
    def test_quanto_qfloat8_uses_generic_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fallback: str,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols, rank = 8, 12, 3
        qtype = quanto.qfloat8 if fallback == "unavailable" else quanto.qfloat8_e4m3fnuz
        qt = WeightQBytesTensor.create(
            qtype,
            0,
            (rows, cols),
            (cols, 1),
            torch.randn(rows, cols, device="cuda").to(qtype.dtype),
            torch.rand(
                rows,
                1,
                device="cuda",
                dtype=torch.bfloat16,
            ).add_(0.25),
            None,
        )
        a = torch.randn(rank, cols, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(rows, rank, device="cuda", dtype=torch.bfloat16)
        expected_dense = qt.dequantize()
        expected_dense.addmm_(b, a, alpha=-0.25)
        expected = _quanto_absmax_oracle(expected_dense, like=qt)

        if fallback == "unavailable":
            monkeypatch.setattr(
                quanto_adapter_impl,
                "_triton_merge_quanto_qfloat8_lora",
                None,
            )
        else:

            def fail_triton(*_args: object) -> torch.Tensor:
                raise AssertionError("unsupported qfloat8 storage reached Triton")

            monkeypatch.setattr(
                quanto_adapter_impl,
                "_triton_merge_quanto_qfloat8_lora",
                fail_triton,
            )
        QuantoAdapter.merge_lora_(qt, b, a, -0.25)

        assert torch.equal(qt._data, expected._data)
        assert torch.equal(qt._scale, expected._scale)

    @CUDA
    def test_triton_quanto_qint8_repairs_zero_scale_and_tracks_range_growth(
        self,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        pytest.importorskip("triton")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols, rank = 18, 23, 3
        scale = torch.zeros(
            rows,
            1,
            device="cuda",
            dtype=torch.float16,
        )
        zero = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            torch.zeros(
                rows,
                cols,
                device="cuda",
                dtype=torch.int8,
            ),
            scale.clone(),
            None,
        )
        zero_scale_ptr = zero._scale.data_ptr()
        zero_data_ptr = zero._data.data_ptr()
        zero_b = torch.zeros(
            rows,
            rank,
            device="cuda",
            dtype=torch.float16,
        )
        zero_a = torch.zeros(
            rank,
            cols,
            device="cuda",
            dtype=torch.float16,
        )
        QuantoAdapter.merge_lora_(zero, zero_b, zero_a, 1.0)
        assert torch.count_nonzero(zero._data).item() == 0
        assert zero._data.data_ptr() == zero_data_ptr
        assert zero._scale.data_ptr() == zero_scale_ptr
        assert torch.all(zero._scale == torch.finfo(torch.float32).eps)
        assert torch.isfinite(zero.dequantize()).all()

        grown = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            torch.zeros(
                rows,
                cols,
                device="cuda",
                dtype=torch.int8,
            ),
            scale.clone(),
            None,
        )
        b = torch.full(
            (rows, rank),
            20.0,
            device="cuda",
            dtype=torch.float16,
        )
        b[rows // 2 :] *= -1
        a = torch.full(
            (rank, cols),
            20.0,
            device="cuda",
            dtype=torch.float16,
        )
        expected_dense = grown.dequantize()
        expected_dense.addmm_(b, a)
        expected = _quanto_absmax_oracle(expected_dense, like=grown)
        grown_scale_ptr = grown._scale.data_ptr()
        grown_data_ptr = grown._data.data_ptr()

        QuantoAdapter.merge_lora_(grown, b, a, 1.0)

        assert grown._data.data_ptr() == grown_data_ptr
        assert grown._scale.data_ptr() == grown_scale_ptr
        assert torch.all(grown._scale > 1.0)
        torch.testing.assert_close(grown._scale, expected._scale)
        assert (grown._data.to(torch.int16) - expected._data.to(torch.int16)).abs().max().item() <= 1
        assert torch.isfinite(grown.dequantize()).all()

    @CUDA
    def test_non_block_quanto_merge_requantizes_on_activate(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        m = _make_bf16_model(num_blocks=1, dim=16)
        rows = cols = 16
        rank = 4
        data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25)
        qt = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            data,
            scale,
            None,
        )
        m.embed.weight = nn.Parameter(qt, requires_grad=False)
        sd = {
            "embed.lora_A.weight": torch.randn(rank, cols),
            "embed.lora_B.weight": torch.randn(rows, rank),
        }
        lora = Adapter.from_state_dict(state_dict=sd)
        factor = lora.targets["embed.weight"]
        a, b = _factor_tensors(factor)
        # Compute the reference on CUDA, matching the device the offloader
        # merges on. A CPU reference flips occasional int8 elements at
        # quantization bucket edges (CPU vs CUDA round-to-nearest), and the
        # comparison is exact — so the device must match to be deterministic.
        qt_cuda = qt.cuda()
        expected_dense = qt_cuda.dequantize()
        expected_dense.addmm_(
            b.cuda().to(expected_dense.dtype),
            a.cuda().to(expected_dense.dtype),
            alpha=0.5,
        )
        expected = _quanto_absmax_oracle(expected_dense, like=qt_cuda)

        s = _make_strategy(m)
        _request_loras(s, [(lora, 0.5)], mode="merge")
        _activate(s, "cuda")
        try:
            merged_qt = m.embed.weight.data
            assert isinstance(merged_qt, WeightQBytesTensor)
            difference = (merged_qt._data.to(torch.int16) - expected._data.to(torch.int16)).abs()
            assert difference.max().item() <= 2
            torch.testing.assert_close(
                merged_qt._scale,
                expected._scale,
                rtol=0.02,
                atol=torch.finfo(torch.float32).eps,
            )
        finally:
            s.deactivate()

    @CUDA
    def test_streamed_quanto_merge_requantizes_pool_param_in_place(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        m = _make_bf16_model(num_blocks=2, dim=16)
        rows = cols = 16
        rank = 4
        original_qt: WeightQBytesTensor | None = None
        for block in m.transformer_blocks:
            data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
            scale = torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25)
            qt = WeightQBytesTensor.create(
                quanto.qint8,
                0,
                (rows, cols),
                (cols, 1),
                data,
                scale,
                None,
            )
            if original_qt is None:
                original_qt = qt
            block.attn.weight = nn.Parameter(qt, requires_grad=False)
        assert original_qt is not None

        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(rank, cols),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(rows, rank),
        }
        lora = Adapter.from_state_dict(state_dict=sd)
        factor = lora.targets["transformer_blocks.0.attn.weight"]
        a, b = _factor_tensors(factor)
        original_cuda = original_qt.cuda()
        expected_dense = original_cuda.dequantize()
        expected_dense.addmm_(
            b.cuda().to(expected_dense.dtype),
            a.cuda().to(expected_dense.dtype),
            alpha=0.5,
        )
        expected = _quanto_absmax_oracle(expected_dense, like=original_cuda)

        s = _make_strategy(m)
        _request_loras(s, [(lora, 0.5)], mode="merge")
        _activate(s, "cuda")
        try:
            merged_qt = m.transformer_blocks[0].attn.weight.data
            assert isinstance(merged_qt, WeightQBytesTensor)
            difference = (merged_qt._data.to(torch.int16) - expected._data.to(torch.int16)).abs()
            assert difference.max().item() <= 2
            torch.testing.assert_close(
                merged_qt._scale,
                expected._scale,
                rtol=0.02,
                atol=torch.finfo(torch.float32).eps,
            )
        finally:
            s.deactivate()


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
        assert offloader._adapter_hook_removers == []

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
            torch.testing.assert_close(model.weight.cpu(), value * -0.5)
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

    @pytest.mark.parametrize("strength", [1.0, 0.5])
    def test_parameter_value_rejects_source_outside_target_range(
        self,
        strength: float,
    ) -> None:
        source = torch.tensor([65520.0], dtype=torch.float32)
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.empty(source.shape, device="meta", dtype=torch.float16),
            requires_grad=False,
        )
        value = Adapter.from_state_dict({"weight": source})

        with pytest.raises(ValueError, match="source exceeds the finite range"):
            merge_adapter(model, [(value, strength)])
        assert model.weight.is_meta

    def test_parameter_value_rejects_scaled_value_outside_target_range(self) -> None:
        source = torch.tensor([40_000.0], dtype=torch.float32)
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.empty(source.shape, device="meta", dtype=torch.float16),
            requires_grad=False,
        )
        value = Adapter.from_state_dict({"weight": source})

        with pytest.raises(ValueError, match="Scaled parameter value exceeds"):
            merge_adapter(model, [(value, 2.0)])
        assert model.weight.is_meta

    @pytest.mark.parametrize("strength", [1.0, 1_000.0])
    def test_parameter_value_rejects_source_underflow_before_strength(
        self,
        strength: float,
    ) -> None:
        source = torch.tensor([1e-8], dtype=torch.float32)
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.empty(source.shape, device="meta", dtype=torch.float16),
            requires_grad=False,
        )
        value = Adapter.from_state_dict({"weight": source})

        with pytest.raises(ValueError, match="source underflows to zero"):
            merge_adapter(model, [(value, strength)])
        assert model.weight.is_meta

    def test_parameter_value_rejects_scaled_underflow(self) -> None:
        source = torch.tensor([1e-4], dtype=torch.float32)
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.empty(source.shape, device="meta", dtype=torch.float16),
            requires_grad=False,
        )
        value = Adapter.from_state_dict({"weight": source})

        with pytest.raises(ValueError, match="Scaled parameter value underflows"):
            merge_adapter(model, [(value, 1e-4)])
        assert model.weight.is_meta

    @pytest.mark.parametrize(
        "stride",
        [
            (0, 1),
            (1, 1),
            (3, 1),
        ],
        ids=["zero-stride-overlap", "nonzero-stride-overlap", "gapped"],
    )
    def test_parameter_value_rejects_unsupported_meta_layout(
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

        with pytest.raises(ValueError, match="non-overlapping dense meta target"):
            merge_adapter(model, [(value, 1.0)])

        assert model.weight.is_meta
        assert model.weight.stride() == stride

    def test_parameter_value_layout_preflight_prevents_partial_merge(self) -> None:
        model = nn.Module()
        model.base = nn.Linear(2, 2, bias=False)
        model.value = nn.Parameter(
            torch.empty_strided((2, 2), (0, 1), device="meta"),
            requires_grad=False,
        )
        model.requires_grad_(False)
        base_before = model.base.weight.detach().clone()
        adapter = Adapter.from_state_dict(
            {
                "base.lora_A.weight": torch.ones(1, 2),
                "base.lora_B.weight": torch.ones(2, 1),
                "value": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            }
        )

        with pytest.raises(ValueError, match="non-overlapping dense meta target"):
            merge_adapter(model, [(adapter, 1.0)])

        torch.testing.assert_close(model.base.weight, base_before)
        assert model.value.is_meta

    def test_parameter_value_rejects_nonzero_meta_storage_offset(self) -> None:
        source = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        backing = torch.empty(8, device="meta")
        model = nn.Module()
        model.weight = nn.Parameter(
            backing.as_strided((2, 2), (2, 1), 1),
            requires_grad=False,
        )
        value = Adapter.from_state_dict({"weight": source})

        with pytest.raises(ValueError, match="storage_offset=0"):
            merge_adapter(model, [(value, 1.0)])

        assert model.weight.is_meta
        assert model.weight.storage_offset() == 1

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

        with pytest.raises(ValueError, match="strided meta target"):
            merge_adapter(model, [(value, 1.0)])

        assert model.weight.is_meta
        assert model.weight.layout is torch.sparse_coo

    @pytest.mark.parametrize(
        ("shape", "stride"),
        [
            ((1, 2), (0, 1)),
            ((0, 2), (0, 1)),
            ((), ()),
        ],
        ids=["size-one", "empty", "scalar"],
    )
    def test_parameter_value_accepts_degenerate_dense_layout(
        self,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
    ) -> None:
        source = torch.zeros(shape)
        value = ParameterValue.from_tensor(source, pin_memory=False)
        transform = ParameterValueTransform(value.scaled(1.0))
        target = nn.Parameter(
            torch.empty_strided(shape, stride, device="meta"),
            requires_grad=False,
        )

        transform.validate_parameter(target)
        materialized = transform.materialize()

        assert materialized.shape == source.shape
        assert materialized.stride() == stride
        torch.testing.assert_close(materialized, source)

    def test_parameter_value_rejects_physical_target_with_wrong_stride(self) -> None:
        source = torch.arange(6, dtype=torch.float32).view(2, 3)
        value = ParameterValue.from_tensor(source, pin_memory=False)
        transform = ParameterValueTransform(value.scaled(1.0))
        meta_target = nn.Parameter(
            torch.empty_strided((2, 3), (1, 2), device="meta"),
            requires_grad=False,
        )
        transform.validate_parameter(meta_target)

        wrong_stride = nn.Parameter(torch.empty(2, 3), requires_grad=False)
        with pytest.raises(RuntimeError, match="matching the validated meta target"):
            transform.apply_parameter(wrong_stride)

        wrong_offset = nn.Parameter(
            torch.empty(7).as_strided((2, 3), (3, 1), 1),
            requires_grad=False,
        )
        with pytest.raises(RuntimeError, match="matching the validated meta target"):
            transform.apply_parameter(wrong_offset)

        materialized = transform.materialize()
        assert materialized.stride() == (1, 2)
        torch.testing.assert_close(materialized, source)

    def test_parameter_value_scales_float8_target(self) -> None:
        source = torch.tensor([1.0], dtype=torch.float32)
        model = nn.Module()
        model.weight = nn.Parameter(
            torch.empty(source.shape, device="meta", dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        value = Adapter.from_state_dict({"weight": source})

        assert merge_adapter(model, [(value, 0.5)]) == 1
        assert torch.equal(model.weight, torch.tensor([0.5], dtype=torch.float8_e4m3fn))

    def test_factor_and_parameter_value_cannot_share_target_across_adapters(self) -> None:
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

        with pytest.raises(ValueError, match="cannot combine"):
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
        torch.testing.assert_close(model.left.weight, value * -0.25)

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

    def test_legacy_lora_bias_cannot_materialize_meta_parameter(self) -> None:
        model = nn.Module()
        model.target = nn.Linear(3, 2, bias=False)
        model.target.bias = nn.Parameter(
            torch.empty(2, device="meta"),
            requires_grad=False,
        )
        model.requires_grad_(False)
        weight_before = model.target.weight.detach().clone()
        lora = Adapter.from_state_dict(
            {
                "target.lora_A.weight": torch.randn(1, 3),
                "target.lora_B.weight": torch.randn(2, 1),
                "target.lora_B.bias": torch.randn(2),
            }
        )

        with pytest.raises(ValueError, match="only by parameter values"):
            merge_adapter(model, [(lora, 1.0)])
        torch.testing.assert_close(model.target.weight, weight_before)
        assert model.target.bias.is_meta

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

    @pytest.mark.parametrize("zero", [0.0, -0.0])
    def test_zero_strength_is_absent(self, zero: float) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16).to(torch.float32)
        before = {name: param.detach().clone() for name, param in m.named_parameters()}
        # Inactive LoRAs do not even require their target names to exist.
        lora = _make_lora(
            num_blocks=2,
            dim=16,
            prefix="missing.",
        )

        assert (
            merge_adapter(
                m,
                [(lora, zero)],
                stochastic_rounding=True,
            )
            == 0
        )
        for name, param in m.named_parameters():
            assert torch.equal(param, before[name])

    def test_can_share_lora_with_an_active_routed_use(self) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16).to(torch.float32)
        routed_model = _make_bf16_model(num_blocks=2, dim=16).to(torch.float32)
        routed = _make_model_offloader(routed_model)
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

    def test_legacy_bias_merges_with_same_adapter_strength(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Linear(3, 4, bias=True)

        m = M()
        m.requires_grad_(False)
        base_weight = m.target.weight.detach().clone()
        base_bias = m.target.bias.detach().clone()
        bias_param = m.target.bias

        def make_lora(seed: int, *, with_bias: bool) -> Adapter:
            generator = torch.Generator().manual_seed(seed)
            state_dict = {
                "target.lora_A.weight": torch.randn(
                    2,
                    3,
                    generator=generator,
                ),
                "target.lora_B.weight": torch.randn(
                    4,
                    2,
                    generator=generator,
                ),
            }
            if with_bias:
                state_dict["target.lora_B.bias"] = torch.randn(
                    4,
                    generator=generator,
                )
            return Adapter.from_state_dict(state_dict)

        loras = [
            (make_lora(1, with_bias=True), 0.5),
            (make_lora(2, with_bias=False), 1.25),
            (make_lora(3, with_bias=True), -0.75),
        ]
        expected_weight = base_weight.clone()
        expected_bias = base_bias.clone()
        bias_delta = torch.zeros_like(expected_bias)
        for lora, strength in loras:
            factor = lora.targets["target.weight"]
            a, b = _factor_tensors(factor)
            expected_weight.addmm_(b, a, alpha=strength)
            bias = _factor_bias(factor)
            if bias is not None:
                bias_delta.add_(bias, alpha=strength)
        expected_bias.add_(bias_delta)

        assert merge_adapter(m, loras) == 2
        torch.testing.assert_close(m.target.weight, expected_weight)
        torch.testing.assert_close(m.target.bias, expected_bias)
        assert m.target.bias is bias_param

    def test_legacy_bias_merge_rejects_biasless_base_without_mutation(
        self,
    ) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Linear(3, 4, bias=False)

        m = M()
        m.requires_grad_(False)
        before = m.target.weight.detach().clone()
        lora = Adapter.from_state_dict(
            {
                "target.lora_A.weight": torch.randn(2, 3),
                "target.lora_B.weight": torch.randn(4, 2),
                "target.lora_B.bias": torch.randn(4),
            },
        )

        with pytest.raises(ValueError, match="no base bias parameter"):
            merge_adapter(m, [(lora, 0.5)])

        torch.testing.assert_close(m.target.weight, before)

    def test_legacy_bias_shape_preflight_prevents_weight_mutation(
        self,
    ) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Module()
                self.target.weight = nn.Parameter(torch.randn(4, 3))
                self.target.bias = nn.Parameter(torch.randn(3))

        m = M()
        m.requires_grad_(False)
        weight_before = m.target.weight.detach().clone()
        bias_before = m.target.bias.detach().clone()
        lora = Adapter.from_state_dict(
            {
                "target.lora_A.weight": torch.randn(2, 3),
                "target.lora_B.weight": torch.randn(4, 2),
                "target.lora_B.bias": torch.randn(4),
            },
        )

        with pytest.raises(ValueError, match="base bias shape"):
            merge_adapter(m, [(lora, 0.5)])

        torch.testing.assert_close(m.target.weight, weight_before)
        torch.testing.assert_close(m.target.bias, bias_before)

    def test_legacy_bias_rejects_tied_base_bias_without_mutation(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.first = nn.Linear(3, 4, bias=True)
                self.second = nn.Linear(3, 4, bias=False)
                self.second.bias = self.first.bias

        m = M()
        m.requires_grad_(False)
        first_weight_before = m.first.weight.detach().clone()
        second_weight_before = m.second.weight.detach().clone()
        bias_before = m.first.bias.detach().clone()
        lora = Adapter.from_state_dict(
            {
                "first.lora_A.weight": torch.randn(2, 3),
                "first.lora_B.weight": torch.randn(4, 2),
                "first.lora_B.bias": torch.randn(4),
                "second.lora_A.weight": torch.randn(2, 3),
                "second.lora_B.weight": torch.randn(4, 2),
                "second.lora_B.bias": torch.randn(4),
            },
        )

        with pytest.raises(ValueError, match="same tied base-bias backing"):
            merge_adapter(m, [(lora, 0.5)])

        torch.testing.assert_close(m.first.weight, first_weight_before)
        torch.testing.assert_close(m.second.weight, second_weight_before)
        torch.testing.assert_close(m.first.bias, bias_before)

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

        with pytest.raises(ValueError, match="LoRA factor shape mismatch"):
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

    def test_quanto_target_defaults_to_stochastic_dequant_requant(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        class M(nn.Module):
            def __init__(self, weight: torch.Tensor) -> None:
                super().__init__()
                self.target = nn.Linear(8, 4, bias=False)
                self.target.weight = nn.Parameter(weight, requires_grad=False)

        rows, cols, rank = 4, 8, 2
        data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25)
        qt = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            data,
            scale,
            None,
        )
        m = M(qt)
        original_param = m.target.weight
        original_packed_ptr = original_param.data._data.data_ptr()
        original_scale_ptr = original_param.data._scale.data_ptr()
        sd = {
            "target.lora_A.weight": torch.randn(rank, cols),
            "target.lora_B.weight": torch.randn(rows, rank),
        }
        lora = Adapter.from_state_dict(state_dict=sd)
        factor = lora.targets["target.weight"]
        a, b = _factor_tensors(factor)

        expected_dense = qt.dequantize()
        expected_dense.addmm_(
            b.to(expected_dense.dtype),
            a.to(expected_dense.dtype),
            alpha=0.5,
        )
        expected = _quanto_absmax_oracle(expected_dense, like=qt)
        rounding_seeds: list[int | None] = []
        original_merge = QuantoAdapter.merge_lora_

        def tracked_merge(
            target: torch.Tensor,
            staged_b: torch.Tensor,
            staged_a: torch.Tensor,
            strength: float,
            *,
            rounding_seed: int | None = None,
        ) -> None:
            rounding_seeds.append(rounding_seed)
            original_merge(
                target,
                staged_b,
                staged_a,
                strength,
                rounding_seed=rounding_seed,
            )

        monkeypatch.setattr(
            QuantoAdapter,
            "merge_lora_",
            staticmethod(tracked_merge),
        )

        merged = merge_adapter(m, [(lora, 0.5)])

        assert merged == 1
        assert rounding_seeds == [derive_seed("target.weight", 0)]
        assert m.target.weight is original_param
        merged_qt = m.target.weight.data
        assert isinstance(merged_qt, WeightQBytesTensor)
        assert merged_qt._data.data_ptr() == original_packed_ptr
        assert merged_qt._scale.data_ptr() == original_scale_ptr
        assert merged_qt.qtype is quanto.qint8
        assert merged_qt.axis == 0
        assert tuple(merged_qt.size()) == (rows, cols)
        code_difference = (merged_qt._data.to(torch.int16) - expected._data.to(torch.int16)).abs()
        assert code_difference.max().item() <= 1
        torch.testing.assert_close(merged_qt._scale, expected._scale)

    def test_legacy_bias_merges_separately_from_quantized_weight(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols, rank = 4, 8, 2
        data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25)
        quantized = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            data,
            scale,
            None,
        )

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Linear(cols, rows, bias=True)
                self.target.weight = nn.Parameter(
                    quantized,
                    requires_grad=False,
                )

        m = M()
        m.requires_grad_(False)
        original_bias = m.target.bias
        bias_before = m.target.bias.detach().clone()
        lora = Adapter.from_state_dict(
            {
                "target.lora_A.weight": torch.randn(rank, cols),
                "target.lora_B.weight": torch.randn(rows, rank),
                "target.lora_B.bias": torch.randn(rows),
            },
        )
        factor = lora.targets["target.weight"]
        a, b = _factor_tensors(factor)
        bias = _factor_bias(factor)
        assert bias is not None
        strength = 0.5
        expected_dense = quantized.dequantize()
        expected_dense.addmm_(
            b.to(expected_dense.dtype),
            a.to(expected_dense.dtype),
            alpha=strength,
        )
        expected_weight = _quanto_absmax_oracle(
            expected_dense,
            like=quantized,
        )

        assert merge_adapter(m, [(lora, strength)]) == 2

        merged_weight = m.target.weight.data
        assert isinstance(merged_weight, WeightQBytesTensor)
        code_difference = (merged_weight._data.to(torch.int16) - expected_weight._data.to(torch.int16)).abs()
        assert code_difference.max().item() <= 1
        assert m.target.bias is original_bias
        assert type(m.target.bias.data) is torch.Tensor
        torch.testing.assert_close(
            m.target.bias,
            bias_before + strength * bias.to(bias_before.dtype),
        )

    def test_permanent_quanto_merge_supports_empty_second_target(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        class M(nn.Module):
            def __init__(
                self,
                first_weight: torch.Tensor,
                empty_weight: torch.Tensor,
            ) -> None:
                super().__init__()
                self.first = nn.Module()
                self.empty = nn.Module()
                self.first.weight = nn.Parameter(
                    first_weight,
                    requires_grad=False,
                )
                self.empty.weight = nn.Parameter(
                    empty_weight,
                    requires_grad=False,
                )

        rows, cols, rank = 3, 4, 2
        first_data = torch.zeros(rows, cols, dtype=torch.int8)
        first_scale = torch.ones(rows, 1)
        first_qt = WeightQBytesTensor(
            quanto.qint8,
            0,
            (rows, cols),
            first_data.stride(),
            first_data,
            first_scale,
            None,
        )
        empty_data = torch.empty(rows, 0, dtype=torch.int8)
        empty_scale = torch.tensor([[0.0], [0.25], [1.0]])
        empty_qt = WeightQBytesTensor(
            quanto.qint8,
            0,
            (rows, 0),
            empty_data.stride(),
            empty_data,
            empty_scale,
            quanto.qint8,
        )
        model = M(first_qt, empty_qt)
        lora = Adapter.from_state_dict(
            {
                "first.lora_A.weight": torch.ones(rank, cols),
                "first.lora_B.weight": torch.ones(rows, rank),
                "empty.lora_A.weight": torch.empty(rank, 0),
                "empty.lora_B.weight": torch.ones(rows, rank),
            }
        )
        expected_dense = first_qt.dequantize().addmm(
            torch.ones(rows, rank),
            torch.ones(rank, cols),
        )
        expected_first = _quanto_absmax_oracle(
            expected_dense,
            like=first_qt,
        )
        first_data_ptr = model.first.weight.data._data.data_ptr()
        first_scale_ptr = model.first.weight.data._scale.data_ptr()
        empty_data_ptr = model.empty.weight.data._data.data_ptr()
        empty_scale_ptr = model.empty.weight.data._scale.data_ptr()
        empty_scale_before = model.empty.weight.data._scale.clone()

        merged = merge_adapter(model, [(lora, 1.0)])

        assert merged == 2
        assert model.first.weight.data._data.data_ptr() == first_data_ptr
        assert model.first.weight.data._scale.data_ptr() == first_scale_ptr
        torch.testing.assert_close(
            model.first.weight.data._data,
            expected_first._data,
        )
        torch.testing.assert_close(
            model.first.weight.data._scale,
            expected_first._scale,
        )
        assert model.empty.weight.data._data.data_ptr() == empty_data_ptr
        assert model.empty.weight.data._scale.data_ptr() == empty_scale_ptr
        assert model.empty.weight.data.activation_qtype is quanto.qint8
        assert model.empty.weight.data.numel() == 0
        torch.testing.assert_close(
            model.empty.weight.data._scale,
            empty_scale_before,
        )

    def test_permanent_quanto_zero_rank_preflight_prevents_partial_merge(
        self,
    ) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        class M(nn.Module):
            def __init__(
                self,
                first_weight: torch.Tensor,
                second_weight: torch.Tensor,
            ) -> None:
                super().__init__()
                self.first = nn.Module()
                self.second = nn.Module()
                self.first.weight = nn.Parameter(
                    first_weight,
                    requires_grad=False,
                )
                self.second.weight = nn.Parameter(
                    second_weight,
                    requires_grad=False,
                )

        rows, cols, rank = 3, 4, 2

        def make_weight() -> torch.Tensor:
            data = torch.zeros(rows, cols, dtype=torch.int8)
            return WeightQBytesTensor(
                quanto.qint8,
                0,
                (rows, cols),
                data.stride(),
                data,
                torch.ones(rows, 1),
                None,
            )

        model = M(make_weight(), make_weight())
        lora = Adapter.from_state_dict(
            {
                "first.lora_A.weight": torch.ones(rank, cols),
                "first.lora_B.weight": torch.ones(rows, rank),
                "second.lora_A.weight": torch.empty(0, cols),
                "second.lora_B.weight": torch.empty(rows, 0),
            }
        )
        first_data_before = model.first.weight.data._data.clone()
        first_scale_before = model.first.weight.data._scale.clone()
        second_data_before = model.second.weight.data._data.clone()
        second_scale_before = model.second.weight.data._scale.clone()

        with pytest.raises(
            ValueError,
            match="positive LoRA rank",
        ):
            merge_adapter(model, [(lora, 1.0)])

        torch.testing.assert_close(
            model.first.weight.data._data,
            first_data_before,
        )
        torch.testing.assert_close(
            model.first.weight.data._scale,
            first_scale_before,
        )
        torch.testing.assert_close(
            model.second.weight.data._data,
            second_data_before,
        )
        torch.testing.assert_close(
            model.second.weight.data._scale,
            second_scale_before,
        )

    def test_multiple_loras_requantize_shared_target_once(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        class M(nn.Module):
            def __init__(self, weight: torch.Tensor) -> None:
                super().__init__()
                self.target = nn.Linear(2, 2, bias=False)
                self.target.weight = nn.Parameter(weight, requires_grad=False)

        data = torch.zeros((2, 2), dtype=torch.int8)
        scale = torch.ones((2, 1), dtype=torch.float32)
        qt = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (2, 2),
            (2, 1),
            data,
            scale,
            None,
        )

        def make_lora() -> Adapter:
            return Adapter.from_state_dict(
                {
                    "target.lora_A.weight": torch.ones(1, 2),
                    "target.lora_B.weight": torch.full((2, 1), 0.6),
                }
            )

        m = M(qt)
        first = make_lora()
        second = make_lora()

        assert merge_adapter(m, [(first, 1.0), (second, 1.0)]) == 1
        # Both 0.6 deltas are accumulated in dense space before the one
        # absmax requantization. Quantizing after each contribution would add
        # an avoidable intermediate loss and choose two different grids.
        torch.testing.assert_close(
            m.target.weight.data.dequantize(),
            torch.full((2, 2), 1.2),
        )

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

        with pytest.raises(ValueError, match="same tied parameter backing"):
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

    def _expected_routed_output(
        self,
        model: nn.Module,
        x: torch.Tensor,
        loras: list[tuple[Adapter, float]],
    ) -> torch.Tensor:
        """Manual baseline: walk the block list using F.linear so we
        bypass any forward hooks installed on the layers (otherwise
        the expected calculation would also include the hook output
        and we'd be comparing the hook against itself)."""
        return _expected_routed_output(model, x, loras)

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
        offloader = _make_model_offloader(m)

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

    def test_legacy_bias_routes_on_biasless_base(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Linear(3, 4, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.target(x)

        m = M()
        m.requires_grad_(False)
        lora = Adapter.from_state_dict(
            {
                "target.lora_A.weight": torch.randn(2, 3),
                "target.lora_B.weight": torch.randn(4, 2),
                "target.lora_B.bias": torch.randn(4),
            },
        )
        factor = lora.targets["target.weight"]
        a, b = _factor_tensors(factor)
        bias = _factor_bias(factor)
        assert bias is not None
        strength = -0.75
        x = torch.randn(5, 3)
        expected = F.linear(x, m.target.weight)
        expected = expected + strength * (x @ a.T @ b.T + bias)
        offloader = _make_model_offloader(m)

        offloader.activate(
            "cpu",
            adapters=[lora],
            adapter_strengths=[strength],
            adapter_mode="routed",
        )
        try:
            torch.testing.assert_close(m(x), expected)
        finally:
            offloader.deactivate()

    @CUDA
    def test_routed_forward_matches_manual_baseline(self) -> None:
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
        _request_loras(s, loras, mode="routed")
        _activate(s, "cuda")
        try:
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
            expected = self._expected_routed_output(m, x, loras)
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1), (
                f"routed forward mismatch:\n  expected: {expected.flatten()[:4]}\n  actual:   {actual.flatten()[:4]}"
            )
        finally:
            s.deactivate()

        # Base weights on CPU snapshots must equal the model's current
        # (post-deactivate) base weights — routed mode didn't mutate.
        for i in range(3):
            assert torch.equal(
                m.transformer_blocks[i].attn.weight.detach(),
                base_snapshots[i],
            ), f"routed mode mutated block {i} base weight"

    @CUDA
    def test_routed_clears_on_deactivate(self) -> None:
        # Hooks installed on activate must be removed on deactivate so
        # subsequent base-only forward sees the unaugmented model.
        m = _make_bf16_model(num_blocks=3, dim=16)
        s = _make_strategy(m)
        _request_loras(s, [(_make_lora(3, 16, seed=7), 1.0)], mode="routed")
        _activate(s, "cuda")
        x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
        with_lora = m(x).detach().clone()
        torch.cuda.synchronize()
        s.deactivate()

        # Re-activate without LoRAs; output should differ from with_lora
        # (the hooks should be gone).
        _request_loras(s, [], mode="routed")
        _activate(s, "cuda")
        try:
            base_only = m(x)
            torch.cuda.synchronize()
            assert not torch.allclose(with_lora, base_only, rtol=0.001, atol=0.001), (
                "deactivate did not remove routed hooks; base-only output still reflects Adapter contribution"
            )
        finally:
            s.deactivate()

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
        _request_loras(s, loras, mode="routed")
        _activate(s, "cuda")
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            actual = m(x)
            torch.cuda.synchronize()
            expected = self._expected_routed_output(m, x, loras)
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1), (
                f"mixed-rank routed output mismatch:\n"
                f"  expected: {expected.flatten()[:4]}\n"
                f"  actual:   {actual.flatten()[:4]}"
            )
        finally:
            s.deactivate()

    def test_routed_with_non_linear_target_raises(self) -> None:
        # Routed math assumes y = x @ W.T + bias. If a target's parent
        # is not nn.Linear, routed activation must reject so we don't
        # silently install a hook against an incompatible forward.
        class LinearLike(nn.Module):
            """nn.Linear-shaped weight but not an nn.Linear instance."""

            def __init__(self, dim: int) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(dim, dim, dtype=torch.bfloat16),
                    requires_grad=False,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.weight.T

        class Block(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.attn = LinearLike(dim)
                self.ff = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.ff(self.attn(x))

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer_blocks = nn.ModuleList([Block(16) for _ in range(2)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for blk in self.transformer_blocks:
                    x = blk(x)
                return x

        model = M().to(torch.bfloat16)
        for p in model.parameters():
            p.requires_grad = False

        s = _make_model_offloader(
            model,
            block_paths=["transformer_blocks"],
        )
        # Build a Adapter targeting attn.weight (LinearLike, not nn.Linear).
        lora = _make_lora(num_blocks=2, dim=16, seed=3)
        _request_loras(s, [(lora, 1.0)], mode="routed")
        with pytest.raises(ValueError, match=r"Routed LoRA mode requires nn\.Linear"):
            _activate_loras_for_test(s)
        # Merge mode should still work — it doesn't care about parent type.
        _request_loras(s, [(lora, 1.0)], mode="merge")
        _activate_loras_for_test(s)

    def test_routed_partial_failure_leaves_no_hooks(self) -> None:
        # Mid-loop route rejection (e.g., one non-Linear target after
        # some valid Linear targets) must NOT leave half-built active
        # hooks.
        class LinearLike(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(dim, dim, dtype=torch.bfloat16),
                    requires_grad=False,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.weight.T

        class Block(nn.Module):
            def __init__(self, dim: int, normal_attn: bool) -> None:
                super().__init__()
                self.attn = nn.Linear(dim, dim, bias=False) if normal_attn else LinearLike(dim)
                self.ff = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.ff(self.attn(x))

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Block 0: normal Linear (passes routed validation).
                # Block 1: LinearLike (fails routed validation).
                self.transformer_blocks = nn.ModuleList([Block(16, normal_attn=True), Block(16, normal_attn=False)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for blk in self.transformer_blocks:
                    x = blk(x)
                return x

        model = M().to(torch.bfloat16)
        for p in model.parameters():
            p.requires_grad = False

        s = _make_model_offloader(
            model,
            block_paths=["transformer_blocks"],
        )
        lora = _make_lora(num_blocks=2, dim=16, seed=99)
        _request_loras(s, [(lora, 1.0)], mode="routed")
        with pytest.raises(ValueError, match=r"Routed LoRA mode requires nn\.Linear"):
            _activate_loras_for_test(s)

    @CUDA
    def test_routed_single_adapter(self) -> None:
        # Single-adapter case: one (A, B) pair, no summation. Forward
        # output must still match the manual baseline.
        m = _make_bf16_model(num_blocks=2, dim=16)
        loras = [(_make_lora(num_blocks=2, dim=16, seed=33), 0.7)]

        s = _make_strategy(m)
        _request_loras(s, loras, mode="routed")
        _activate(s, "cuda")
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            actual = m(x)
            torch.cuda.synchronize()
            expected = self._expected_routed_output(m, x, loras)
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1)
        finally:
            s.deactivate()

    @pytest.mark.parametrize("target", ["embed", "head"])
    def test_routed_tied_weight_target_uses_exact_parent(self, target: str) -> None:
        # Standard tied embed/head pattern: one Parameter aliased at
        # multiple names. Routed mode is name-centric and does not
        # mutate the shared storage, so it should hook only the exact
        # parent module named by the Adapter target.
        model = _make_tied_non_block_model(dtype=torch.bfloat16)

        s = _make_model_offloader(
            model,
            block_paths=["transformer_blocks"],
        )
        # Build a Adapter that targets either alias of the tied weight.
        sd = {
            f"{target}.lora_A.weight": torch.randn(4, 16),
            f"{target}.lora_B.weight": torch.randn(16, 4),
        }
        lora = Adapter.from_state_dict(state_dict=sd)
        _request_loras(s, [(lora, 1.0)], mode="routed")
        active_loras, _mode = _LORA_REQUESTS.pop(s)
        targets = s._group_adapter_updates_by_param_name(active_loras)
        s._register_routed_lora_hooks(targets)
        try:
            assert len(model.embed._forward_hooks) == (1 if target == "embed" else 0)
            assert len(model.head._forward_hooks) == (1 if target == "head" else 0)
            assert len(model.embed._forward_pre_hooks) == (1 if target == "embed" else 0)
            assert len(model.head._forward_pre_hooks) == (1 if target == "head" else 0)
        finally:
            s._clear_active_adapter_hooks()

        # Merge mode also matches by name; it mutates the copied backing,
        # so the normal shared-storage effect is preserved.
        _request_loras(s, [(lora, 1.0)], mode="merge")
        _activate_loras_for_test(s)

    @CUDA
    def test_routed_with_bias_linear(self) -> None:
        # Routed math: hook adds alpha * B * A * x to the layer's output, which
        # already includes any bias from the base layer. Bias-having
        # Linears must produce the same output as a manual baseline
        # that goes through F.linear with the bias.
        class Block(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.attn = nn.Linear(dim, dim, bias=True)
                self.ff = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.ff(self.attn(x))

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(16, 16, bias=False)
                self.transformer_blocks = nn.ModuleList([Block(16) for _ in range(2)])
                self.head = nn.Linear(16, 16, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.embed(x)
                for blk in self.transformer_blocks:
                    h = blk(h)
                return self.head(h)

        m = M().to(torch.bfloat16)
        for p in m.parameters():
            p.requires_grad = False

        loras = [(_make_lora(num_blocks=2, dim=16, seed=55), 0.5)]
        s = _make_model_offloader(
            m,
            block_paths=["transformer_blocks"],
        )
        _request_loras(s, loras, mode="routed")
        _activate(s, "cuda")
        try:
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
                lora = loras[0][0]
                strength = loras[0][1]
                factor = lora.targets[f"transformer_blocks.{i}.attn.weight"]
                a, b = _factor_tensors(factor)
                a_dev = a.to(device=h.device, dtype=h.dtype)
                b_dev = b.to(device=h.device, dtype=h.dtype)
                attn_out = base_attn + strength * (h @ a_dev.T @ b_dev.T)
                h = F.linear(attn_out, blk.ff.weight.to(h.device))
            expected = F.linear(h, m.head.weight.to(h.device))
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1)
        finally:
            s.deactivate()


class TestRoutedStaging:
    """Routed LoRA stages pinned factors in target PRE hooks."""

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
        s = _make_model_offloader(m, block_paths=["transformer_blocks"])

        def routed_forward(
            loras: list[tuple[Adapter, float]],
        ) -> torch.Tensor:
            if loras:
                _request_loras(s, loras, mode="routed")
            _activate(s, torch.device("cpu"))
            try:
                return m(x).clone()
            finally:
                s.deactivate()

        y0 = routed_forward([])
        y1 = routed_forward([(lora1, 0.7)])
        y2 = routed_forward([(lora2, 1.3)])
        y12 = routed_forward([(lora1, 0.7), (lora2, 1.3)])
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
        s = _make_model_offloader(m)
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
        s = _make_model_offloader(m)
        s.activate("cpu", adapters=[lora], adapter_mode="routed")
        try:
            assert len(s._adapter_hook_removers) == 1
            assert callable(s._adapter_hook_removers[0])
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

        s = _make_model_offloader(m, block_paths=["transformer_blocks"])
        _request_loras(s, [(lora, 0.5)], mode="routed")
        x = torch.randn(2, 16)
        _activate(s, torch.device("cpu"))
        try:
            actual = m(x).clone()
            # The hook lands on the matched ``.base_layer`` Linear.
            assert len(m.transformer_blocks[0].attn.base_layer._forward_hooks) == 1
        finally:
            s.deactivate()

        h = x
        for b in range(2):
            out = F.linear(h, base_w[b])
            a = sd[f"transformer_blocks.{b}.attn.base_layer.lora_A.weight"]
            bb = sd[f"transformer_blocks.{b}.attn.base_layer.lora_B.weight"]
            h = out + 0.5 * (h @ a.T) @ bb.T
        torch.testing.assert_close(actual, h, rtol=1e-4, atol=1e-4)

    def test_deactivate_removes_staging_hooks(self) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16)
        lora = _make_lora(num_blocks=2, dim=16, seed=4)
        s = _make_model_offloader(m, block_paths=["transformer_blocks"])

        _request_loras(s, [(lora, 1.0)], mode="routed")
        _activate(s, torch.device("cpu"))
        # One paired-hook remover per target.
        assert len(s._adapter_hook_removers) == 2

        s.deactivate()
        assert s._adapter_hook_removers == []

        # A new activation-scoped request works after teardown.
        _request_loras(s, [(lora, 1.0)], mode="routed")
        _activate(s, torch.device("cpu"))
        try:
            assert len(s._adapter_hook_removers) == 2
        finally:
            s.deactivate()

    @CUDA
    def test_routed_staging_matches_merge_with_streamed_model(self) -> None:
        # Per-target factor staging must compose with a streamed base model and
        # match merge mode bit-close.
        m = _make_bf16_model(num_blocks=2, dim=16)
        lora = Adapter.from_state_dict(
            state_dict=_make_lora_sd(num_blocks=2, dim=16, seed=7),
        )
        x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
        s = _make_model_offloader(m, block_paths=["transformer_blocks"])
        _request_loras(s, [(lora, 0.5)], mode="merge")
        _activate(s, "cuda")
        try:
            merged = m(x).clone()
            torch.cuda.synchronize()
        finally:
            s.deactivate()

        _request_loras(s, [(lora, 0.5)], mode="routed")
        _activate(s, "cuda")
        try:
            routed = m(x).clone()
            torch.cuda.synchronize()
        finally:
            s.deactivate()

        # Routed residual vs merge (weight-bake) differ only
        # by bf16 rounding of the separate residual GEMMs.
        torch.testing.assert_close(routed, merged, rtol=0.1, atol=0.1)

    def test_activate_rolls_back_composite_when_routed_registration_fails(
        self,
    ) -> None:
        # If route registration raises, activation must release its claim and
        # leave no hooks. Proven by a clean re-activation.
        m = _make_bf16_model(num_blocks=2, dim=16)
        sd = {
            "transformer_blocks.0.absent.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.absent.lora_B.weight": torch.randn(16, 4),
        }
        bad = Adapter.from_state_dict(state_dict=sd)
        s = _make_model_offloader(m, block_paths=["transformer_blocks"])
        _request_loras(s, [(bad, 1.0)], mode="routed")

        with pytest.raises(ValueError, match="not managed"):
            _activate(s, torch.device("cpu"))

        assert s.active_device is None
        assert s._adapter_hook_removers == []
        # Composite was deactivated -> a fresh activation succeeds and runs.
        _activate(s, torch.device("cpu"))
        try:
            out = m(torch.randn(2, 16, dtype=torch.bfloat16))
            assert out.shape == (2, 16)
        finally:
            s.deactivate()


class TestDeactivateCleanupInvariants:
    @CUDA
    def test_cleanup_runs_even_when_streamer_deactivate_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        _request_loras(s, [(_make_lora(4, 16), 1.0)])

        def streamer_boom() -> None:
            raise RuntimeError("streamer cleanup failed")

        monkeypatch.setattr(block_components(s)[0], "deactivate", streamer_boom)
        _activate(s, "cuda")

        with pytest.raises(RuntimeError):
            s.deactivate()


# ---------------------------------------------------------------------------
# Cache budget
# ---------------------------------------------------------------------------


class TestCacheBytes:
    def test_lora_cache_bytes_reports_factor_size(self) -> None:
        lora = _make_lora(num_blocks=4, dim=16, rank=4)
        assert lora.cache_bytes > 0


# ---------------------------------------------------------------------------
# Unified Adapter resource (ResourceCache integration)
# ---------------------------------------------------------------------------


class TestLoRAResource:
    def test_lora_is_immutable_cached_resource(self) -> None:
        lora = _make_lora(num_blocks=2, dim=8, rank=2)
        assert isinstance(lora, ResourceStore)
        assert not isinstance(lora, ResourceBinding)
        assert not isinstance(lora, nn.Module)
        assert not hasattr(lora, "activate")
        assert not hasattr(lora, "deactivate")

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
            host_backing="adopt",
        )

        lora = spec.build_store()

        assert tuple(lora.targets) == ("target.weight",)
        assert isinstance(lora.targets["target.weight"], ParameterValue)
        assert lora.cache_bytes == value.nbytes

    def test_lora_spec_propagates_adopted_host_backing(self) -> None:
        sd = _make_lora_sd(num_blocks=1, dim=8, rank=2)
        spec = AdapterSpec(
            key="lora:adopted",
            estimated_cache_bytes=1000,
            factory=lambda: sd,
            host_backing="adopt",
        )

        lora = spec.build_store()
        factor = lora.targets["transformer_blocks.0.attn.weight"]
        a, b = _factor_tensors(factor)

        assert not a.is_pinned()
        assert not b.is_pinned()
        assert a.data_ptr() == sd["transformer_blocks.0.attn.lora_A.weight"].data_ptr()
        assert b.data_ptr() == sd["transformer_blocks.0.attn.lora_B.weight"].data_ptr()

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

        cache = ModelCache(10**9)
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
        cache = ModelCache(10**9)
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
    def test_model_cache_stochastic_rounding_default_and_opt_out(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stochastic_rounding: bool | None,
        expected: bool,
    ) -> None:
        cache = ModelCache(10**9)
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

        cache = ModelCache(10**9)
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
        cache = ModelCache(10**9)
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
        cache = ModelCache(10**9)

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
        cache = ModelCache(10**9)
        value = torch.randn(2, 8)

        with cache.use(
            model_spec,
            device="cpu",
            adapter_specs=[lora_spec, lora_spec],
            adapter_strengths=[0.25, 0.75],
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

        cache = ModelCache(10_000)
        with pytest.raises(ResourceTooLargeError):
            with cache.use(
                model_spec,
                device="cpu",
                adapter_specs=[lora_spec],
                adapter_mode="routed",
            ):
                pass

        assert cache.info("lora:style").cached
        assert cache.info("lora:style").lease_count == 0


# ---------------------------------------------------------------------------
