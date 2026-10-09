# Applying adapters

An `Adapter` is reusable host state containing LoRA factors, full-rank additive
deltas, complete parameter values, or a combination. Apply it for one model use
through `ModelCache`, or permanently with `merge_adapter()`.

This guide covers targeting and application. See [models](models.md) for cache
lifetime, [formats](formats.md) for quantized capabilities, and the
[README](../README.md) for an overview.

## Apply an adapter for one use

This complete example creates a small frozen model and applies a rank-two LoRA
on CUDA. Replace the factories with your model and checkpoint loaders in an
application.

```python
from collections import OrderedDict

import torch
from torch import nn

from piper_offload import AdapterSpec, ModelCache, ModelSpec


def make_model():
    return nn.Sequential(OrderedDict(
        projection=nn.Linear(4, 3, bias=False),
    )).requires_grad_(False)


def make_adapter_state():
    return {
        "projection.lora_A.weight": torch.randn(2, 4) * 0.01,
        "projection.lora_B.weight": torch.randn(3, 2) * 0.01,
        "projection.alpha": torch.tensor(2.0),
    }


cache = ModelCache()
model_spec = ModelSpec(key="projection", factory=make_model)
adapter_spec = AdapterSpec(key="example-lora", factory=make_adapter_state)

with cache.use(
    model_spec,
    device="cuda",
    adapter_specs=[adapter_spec],
    adapter_strengths=[0.8],
) as model:
    with torch.no_grad():
        output = model(torch.randn(1, 4, device="cuda"))
```

The cache retains both resources for reuse. The adapter request belongs to this
activation; loading pristine host weights clears an earlier merge. Factories
transfer ownership of compatible CPU tensors, including mapped views. Do not
mutate those tensors after capture. See [memory](memory.md) for checkpoint
loading and pinning.

`adapter_strengths` defaults to one per adapter. When supplied, its length must
match `adapter_specs`. Entries are ordered contributions: repeating an adapter
applies it again at that occurrence's strength. Zero strength disables an entry;
`ModelCache` does not even build or lease its adapter resource.

## Target names and update types

State-dict keys must already use the model's parameter paths. Piper does not
strip checkpoint prefixes, insert PEFT `.base_layer` segments, or parse
ComfyUI `.diff` / `.diff_b` conventions. Remap those keys and remove checkpoint
metadata before constructing the adapter.

| State-dict entry | Meaning |
|---|---|
| `projection.lora_A.weight` and `projection.lora_B.weight` | Paired factors targeting `projection.weight`; shapes are `(rank, in_features)` and `(out_features, rank)` |
| `projection.alpha` | Optional scalar giving that pair an intrinsic `alpha / rank` scale; requires the pair |
| `projection.delta.weight` | Full-rank additive update to the existing `projection.weight` |
| `projection.delta.bias` | Additive update to the existing `projection.bias` |
| Any other exact parameter name | Complete value for a frozen plain floating-point **meta** parameter |

For one target, the additive update is
`strength * (scaling * B @ A + dense)`, using whichever terms are present.
`scaling` is `alpha / rank` when supplied, otherwise `1`. Dense deltas have no
intrinsic scale. Keeping alpha separate also preserves mapped factor storage
that host-side arithmetic would replace.

Unknown targets raise by default. Set `AdapterSpec(allow_partial_targets=True)`
when one checkpoint intentionally spans separately loaded model components.
Only matching targets apply, and an empty intersection is a valid no-op.
Matching targets still require compatible shapes and capabilities.

For activation merge and permanent `merge_adapter()`, use one target name per
tied parameter across all active adapters. The update reaches every alias;
targeting multiple names for that shared parameter raises.

`Adapter.from_state_dict()` uses the same parsing and options outside the cache.
For parameter names that themselves end in reserved suffixes, construct
`Adapter(targets=...)` with explicit `ParameterDelta` or `ParameterValue`
objects. Their contracts live in [adapter.py](../src/piper_offload/adapter.py),
[parameter_delta.py](../src/piper_offload/parameter_delta.py), and
[parameter_value.py](../src/piper_offload/parameter_value.py).

## Choose merge or routed LoRA

| Mode | Behavior | Supported updates |
|---|---|---|
| `"merge"` (default) | Applies updates after each host-to-device load | LoRA, dense deltas, and meta parameter values, subject to format capabilities |
| `"routed"` | Adds a LoRA residual to each targeted linear's output | LoRA factors only |

Activation-scoped merge requires CUDA. Routed LoRA works with CPU or CUDA
activation and leaves the base weight untouched. It requires an `nn.Linear`
parent with a compatible logical weight shape and compute dtype. Factors are
staged for that target's invocation and released afterwards; several LoRAs on
one target share a hook pair and contribute independently. Tied weights are
handled by hooking only the named parent.

Use `adapter_mode="routed"` in the example above when avoiding quantized
re-encoding matters more than the extra forward work, or the format cannot
merge. Routed adapters are inference-only: factors are frozen and do not receive
gradients. They cannot apply dense deltas, create a missing bias, or populate
meta parameters. Active routed adapters also bypass block compilation; see
[streaming](streaming.md).

Quantized merges are lossy. Piper combines contributions to a target before one
re-encoding, including LoRA terms when a dense delta is present. Stochastic
rounding is enabled by default to avoid systematically losing sub-step updates;
pass `stochastic_rounding=False` to `cache.use()` for round-to-nearest instead.
Sampling does not consume PyTorch's global RNG. Repeated streamed merges use
fresh deterministic samples derived from the target and merge count; matching
seeds do not promise identical bytes across different backend implementations.
Routed mode does not requantize and ignores this option.

## Populate optional meta parameters

Complete values let a model declare an optional weight without allocating a
base tensor. For example, a factory can create
`nn.Linear(4, 3, bias=False, device="meta").requires_grad_(False)` and an
`AdapterSpec` can supply `{"weight": torch.randn(3, 4)}`. Select that adapter
in a CUDA merge activation to materialize the weight.

The target must be a frozen, plain floating-point meta parameter of the same
logical shape. It supplies the name, shape, and alias group; the value supplies
the active dtype and representation. Values cannot replace existing physical
model parameters, combine with deltas on the same target, or compete with
another active value. LoRA factors alone cannot materialize a meta target.

An unpopulated placeholder consumes no host backing and receives no storage of
its own. The caller must avoid executing any path that needs it. Deactivation
restores the meta parameter. All block modes support values; inactive rolling
blocks may reference another block's reusable slot, whose contents are
unspecified.

Values remain unchanged at any nonzero adapter strength by default. Opt into
scaling with `AdapterSpec(scale_parameter_values=True)`, or per target with
`ParameterValue.from_tensor(..., scale_with_strength=True)`. Zero-strength
adapters remain inactive under either policy.

A value can use any supported physical representation with a floating compute
dtype and compatible logical shape, including tile-packed INT4. Exact copying
does not require additive merge support or a dequantize/requantize pass.
Explicit non-unit scaling additionally requires dequantization and dense merge
capabilities and re-encodes a quantized value once. `dtype=` casts dense adapter
inputs; prequantized values must already have that compute dtype. Requantize
them before capture if a different compute dtype is needed.

Callers own numerical validity: Piper checks representation structure, shape,
dtype, and capabilities, but does not scan tensor payloads for NaN or infinity.
Scalar strengths are finite-checked.

## Make an application permanent

Use `merge_adapter()` on a model before handing it to a cache when the adapter
should become part of its host weights. Continuing with the factories above:

```python
from piper_offload import Adapter, merge_adapter

model = make_model()
adapter = Adapter.from_state_dict(make_adapter_state())
modified = merge_adapter(model, [(adapter, 0.8)])
```

The return value counts unique modified parameters. The operation validates
active names, shapes, and capabilities before mutation, but it is not
reversible. Plain floating-point weights use in-place arithmetic. Mapped
targets are first replaced with owned copies under every tied name; Piper
never writes into a checkpoint mapping. Meta targets become independent frozen
CPU parameters with their aliases preserved.

Permanent merge accepts the same strength, partial-target, and rounding
policies. Quantized support follows the [format matrix](formats.md), except
that packed GGUF sources can only be merged during activation. See
[merge.py](../src/piper_offload/merge.py) for the function contract and
[lora.py](../src/piper_offload/lora.py) for factor and routing details.
