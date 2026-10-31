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

## Broadcast additive updates

Activation merge and permanent `merge_adapter()` use standard PyTorch
broadcasting for additive updates to plain floating-point parameters, excluding
plain float8. Each dense delta and each LoRA product `B @ A` must broadcast to
the existing target shape: `torch.broadcast_shapes(update_shape, target_shape)`
must equal `target_shape`. Incompatible shapes propagate PyTorch's `RuntimeError`;
broadcasts that would expand the target raise `ValueError`. Both fail during
preflight, before any permanent merge mutates weights.

Dimensions align from the right; matching sizes, singleton expansion, and
missing leading dimensions follow PyTorch's rules. There is no extra
restriction on matrix or feature dimensions. For example:

| Update shape | Target shape | Result |
|---|---|---|
| `[out, in]` | `[heads, out, in]` | Shared update for every head |
| `[1, in]` | `[heads, out, in]` | Shared update for every head and output row |
| `[out]` | `[heads, out]` | Shared bias delta |
| `[]` | Any shape | Scalar dense delta |
| `[out, in]` with `out > 1` | `[1, in]` | Rejected: would expand the target |

LoRA factors remain 2D with compatible rank dimensions. Different contributions
to one target may have different broadcastable shapes, including mixed LoRA and
dense updates. Factors and dense sources remain immutable; callers need not
repeat or pre-scale checkpoint tensors. Shared updates are staged at their
common shape, without repeating them across target-only dimensions.

Complete parameter values require **exact full-shape matching**. Routed LoRA
retains its exact linear-weight shape contract. Built-in quantized and DTensor
merge paths currently require exact full-shape updates; custom tensor adapters
can explicitly opt into broadcast merges. See [formats](formats.md).

### Accumulation and rounding

Sources are staged in the target's compute dtype. Plain merge arithmetic keeps
that dtype even inside an autocast context. A single LoRA with an exact 2D
target uses `addmm_` with its combined strength and intrinsic scale. Multiple
LoRAs pack their factors across the rank dimension, scaling each staged `A`
slice by its coefficient before one matrix multiplication. Singleton factor
dimensions expand only as needed to form that common matrix.

When the target shape differs from the staged product shape, the plain tensor
adapter computes `B @ A` in the compute dtype and then adds it with broadcasting.
The product therefore rounds before the addition, unlike the fused exact-2D
`addmm_` path. With any dense contribution, dense terms accumulate in request
order into a compute-dtype update, followed by the packed LoRA contribution;
that combined update is added to the base once. PyTorch kernels determine
internal accumulation precision; staging does not promote low-precision values
to a higher dtype.

These intermediate roundings mean broadcast merges need not be bit-identical
to separate per-head `addmm_` calls. Merging low-precision heads before averaging
also need not match adding a shared delta after averaging. Quantized rounding
is described below.

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
model parameters or compete with another active value.

A value from one adapter can supply the base for LoRA and dense deltas from
other active adapters targeting the same parameter name:

```text
active parameter = value_scale * supplied_value
                 + sum(strength_j * (scaling_j * B_j @ A_j + dense_j))
```

Each delta keeps its own adapter strength and intrinsic LoRA scaling. Those
strengths do not scale the supplied value. The order of adapters does not
choose the value owner; multiple active values are an error. Continue using
one name for a tied parameter across all adapters. Within one adapter, a
parameter entry remains either a complete value or an additive delta; supplying
both for that parameter is rejected.

Updates are validated against the supplied representation before application.
The same [broadcasting rules](#broadcast-additive-updates) apply to these deltas;
the supplied value itself must still match the complete target shape.
An active delta with no active value cannot materialize a meta parameter,
including when the value's adapter has zero strength.

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
LoRA-only contributions retain the factorized merge path, including when the
supplied value is scaled. When dense deltas are present, Piper stages one dense
additive update. Both paths pass the value scale separately to the tensor adapter.
The adapter computes the scaled base plus that update before one terminal
encoding; it never constructs a cancelling `(value_scale - 1) * value` term.
Value-only scaling requires dense merge support and uses a zero additive update.
Arithmetic follows the representation's compute dtype and backend accumulation;
adapter ordering does not promise bit-identical sums.

`dtype=` casts dense adapter inputs; prequantized values must already have that
compute dtype. Requantize them before capture if a different compute dtype is
needed.

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

Permanent merge accepts the same cross-adapter composition, strength,
partial-target, and rounding policies. Quantized support follows the [format matrix](formats.md), except
that packed GGUF sources can only be merged during activation. See
[merge.py](../src/piper_offload/merge.py) for the function contract and
[lora.py](../src/piper_offload/lora.py) for factor and routing details.
