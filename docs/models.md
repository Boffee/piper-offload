# Models and resources

Use `ModelCache` to retain models between calls and activate them for compute.
It owns model and adapter lifecycles; inputs, outputs, and application scheduling
remain yours. For a complete first example, see the [README](../README.md).

## Caching and activation

A `ModelSpec` identifies a model by key and supplies a factory that builds a
fresh module. The first use constructs and captures the model; later uses
reuse it until eviction. Configure the model, including `requires_grad`, in
the factory. Passing a new spec with an existing key does not replace the
registered factory; use `register(spec, replace=True)` when replacing an
inactive entry intentionally.

This example switches between two cached stages and keeps them available for
later calls:

```python
import torch
from torch import nn
from piper_offload import ModelCache, ModelSpec

cache = ModelCache()
encoder = ModelSpec(
    key="encoder",
    factory=lambda: nn.Linear(8, 16).eval().requires_grad_(False),
)
decoder = ModelSpec(
    key="decoder",
    factory=lambda: nn.Linear(16, 4).eval().requires_grad_(False),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs = torch.randn(1, 8, device=device)
with cache.use(encoder, device=device) as model, torch.inference_mode():
    hidden = model(inputs)
with cache.use(decoder, device=device) as model, torch.inference_mode():
    output = model(hidden)

del model
```

Each entry owns one `ModelOffloader` and one model instance. Overlapping
activation of that runtime raises `ModelRuntimeInUseError`, even from different
threads or callers. Concurrent replicas need fresh models and distinct keys.
Their checkpoint mappings may still share physical OS pages.

`ModelCache.use()` leases adapters before the model, activates the offloader,
yields its model, then deactivates it and releases leases in reverse order.
Use the yielded model inside that context. Retaining a Python reference after
exit does not keep it active and can prevent its host storage from being freed.

To configure block residency, pass `block_paths`, `block_mode`, and related
options to `ModelSpec`; see [streaming](streaming.md). To attach cached updates,
pass `adapter_specs` to `use()`; see [adapters](adapters.md).

## Ownership and persistent state

Capture takes ownership of compatible CPU storage, retaining checkpoint
mappings where possible. It may replace parameter and buffer registry entries
or repoint plain parameter data immediately. Factories should build their own
models rather than return externally retained instances. The precise
[capture rules](memory.md#preserve-checkpoint-backing) describe when copies
are required.

Frozen host bytes are immutable while captured. Trainable parameters keep
their parameter identity and use owned storage; file mappings are never
written. See [memory](memory.md) for checkpoint loading and storage lifetimes.

CUDA buffer mutations are discarded at deactivation. CPU activation uses
host-backed buffers directly, so mutations there have ordinary PyTorch
semantics. Models with persistent CUDA buffer state, such as training-time
BatchNorm statistics or a registered KV cache, do not fit this lifecycle.
Wrap the model before DDP/FSDP, whose own parameter-storage management can
conflict with capture. See [streaming](streaming.md#train-streamed-blocks) for training and
optimizer synchronization.

## Releasing and inspecting resources

`ModelCache` uses an unbounded resource cache. `used_cache_bytes` reports
logical backing bytes, and `info(key)` reports construction and lease state.
These are accounting values, not measurements of resident RAM or GPU usage.
The [pin budget](memory.md) independently controls registered host memory.

```python
info = cache.info("encoder")
backing_bytes = cache.used_cache_bytes
cache.evict("encoder")
freed = cache.evict_bytes(2 * 1024**3)
cache.clear()
```

The snippet uses the cache registered above. `evict()` releases an unleased
store by key. `evict_bytes()` releases whole inactive entries, so the reported
bytes can exceed the request; if too few bytes are eligible, it releases those
available and returns the smaller count. `clear()` evicts all stores, but
raises if any entry is leased. Factories remain registered after eviction,
allowing later uses to rebuild. `unregister()` also removes a registration.

There is no model `close()` method. End its use, evict its store, and drop any
escaped model or resource references to release host backing.

## Manual lifecycle

Use `ModelOffloader` directly when the application owns activation. This
example is independent of the cache examples:

```python
import torch
from torch import nn
from piper_offload import ModelOffloader

model = nn.Linear(8, 4).eval().requires_grad_(False)
offloader = ModelOffloader.from_module(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

offloader.activate(device)
try:
    with torch.inference_mode():
        output = offloader.value(torch.randn(1, 8, device=device))
finally:
    offloader.deactivate()

del offloader, model
```

Access the bound model while it is active. Successful activations require
matching deactivation; CUDA training also needs the
[`optimizer_step()` context](streaming.md#train-streamed-blocks) to copy updated weights
back before deactivation when the optimizer runs on CUDA.

If construction fails after capture begins, discard the partially captured
model and rebuild from a fresh instance. `ModelOffloader.activate()` cleans
up partial activation and releases its activation claim on failure, allowing
a later corrected activation to retry. This rollback is separate from
construction failure recovery.

For frozen CPU models destined to stay on Apple MPS, `MpsWeights` provides a
separate constructor-time materializer. It retains no CPU cache, does not
preserve tied aliases, and leaves the model on MPS when deactivated. See its
[contract](../src/piper_offload/mps_weights.py).

## Other resources and finite budgets

`ObjectSpec` caches ordinary objects such as tokenizers. A resource lease
holds the stored value without activating a device runtime:

```python
from piper_offload import ObjectSpec, ResourceCache

resources = ResourceCache(max_cache_bytes=1024)
spec = ObjectSpec(key="vocabulary", factory=lambda: {"hello": 0})
with resources.lease(spec) as vocabulary:
    token = vocabulary["hello"]
resources.clear()
```

Objects account for zero bytes by default; `estimated_cache_bytes` supplies
their accounting size. `ResourceCache` supports arbitrary specs and an optional
finite budget. It evicts inactive stores using LRU by default; leases protect
stores from eviction. It serializes construction and metadata changes, then
releases its lock while callers hold leases.

For a finite cache, `resize(bytes)` changes the budget. Growing preserves
entries; shrinking evicts eligible entries. If leased entries prevent the new
limit from being met, it raises without changing the budget or evicting stores.
Custom specs and eviction policies are covered in [development](development.md).

## Errors and API details

| Exception | Meaning |
|---|---|
| `ModelRuntimeInUseError` | The model runtime is already active. |
| `ResourceTooLargeError` | Finite-budget admission cannot fit the store; `required`, `used`, and `limit` describe the request. |
| `ResourceLeasedError` | A mutation would release an actively leased resource. |
| `ResourceCachedError` | `unregister(..., evict=False)` targets a built store. |
| `DuplicateResourceKeyError` | Explicit registration repeats a key without `replace=True`. |
| `ResourceNotRegisteredError` | A key lookup has no registered spec. |
| `EvictionPolicyError` | A custom policy returned invalid or insufficient victims. |

Signatures and per-method contracts are in
[`ModelCache`](../src/piper_offload/model_cache.py),
[`ModelSpec` and other specs](../src/piper_offload/resource_specs.py),
[`ModelOffloader`](../src/piper_offload/model_offloader.py), and
[`ResourceCache`](../src/piper_offload/resource_cache.py).
