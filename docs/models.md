# Models and resources

Use `ModelCache` to retain models between calls and activate them for compute.
It owns model and adapter lifecycles; inputs, outputs, and application scheduling
remain yours. For a complete first example, see the [README](../README.md).

## Caching and activation

A `ModelSpec` identifies a model by key and supplies a factory that builds a
fresh frozen module (`requires_grad=False`). The first use constructs and captures the model; later uses
reuse it until eviction. Passing a new spec with an existing key does not replace the
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

To configure block residency, pass `config=OffloadConfig(...)` to
`ModelSpec`; see [streaming](streaming.md). To attach cached updates, pass
`adapter_specs` to `use()`; see [adapters](adapters.md).

## Ownership and persistent state

Capture takes ownership of compatible CPU storage, retaining checkpoint
mappings where possible. It may replace parameter and buffer registry entries
or repoint plain parameter data immediately. Factories should build their own
models rather than return externally retained instances. The precise
[capture rules](memory.md#preserve-checkpoint-backing) describe when copies
are required.

Frozen host bytes are immutable while captured. File mappings are never written.
See [memory](memory.md) for checkpoint loading and storage lifetimes.

CUDA buffer mutations are discarded at deactivation. CPU activation uses
host-backed buffers directly, so mutations there have ordinary PyTorch
semantics. Models with persistent CUDA buffer state, such as RNN/SSM recurrent
state or a registered KV cache, do not fit this lifecycle.

## Releasing and inspecting resources

A new `ModelCache` retains every store until you evict it. To bound it, call
`cache.resize(limit_in_bytes)`. It evicts the least recently used inactive
stores when you lower the limit or another store needs room. A lease protects
its store. If leased stores alone exceed the requested limit, resizing raises
`ResourceTooLargeError` and preserves the old budget and all entries.

`used_cache_bytes` counts logical backing bytes. Use `info(key)` to see whether
a store is cached or leased. The [pin budget](memory.md#set-the-pin-budget)
controls registered host memory separately.

If model cleanup fails with device work outstanding, the cache retains the
model and reports a lease after the usage context exits. Keep the cache alive.
The next `cache.use()` of that model retries cleanup. To retry cleanup alone,
call `offloader.deactivate()` inside `with cache.lease(model_spec) as offloader:`.
Successful cleanup makes the model eligible for eviction when the caller releases
its lease.

```python
info = cache.info("encoder")
backing_bytes = cache.used_cache_bytes
cache.resize(2 * 1024**3)  # Keep up to 2 GiB of cached backing.
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
matching deactivation.

Preflight validation failures (such as trainable parameters, incompatible group
paths, or an already-owned shell) leave the model untouched. If construction
fails after capture or installation begins, discard the partially modified
model and build a fresh one. If activation fails, the offloader rolls back the
partial activation. Correct the cause before retrying.

If cleanup cannot finish releasing a CUDA session, `offloader.failed` is true.
Keep the offloader alive: device work may still use its host storage. Call
`deactivate()` to retry cleanup, or let the next `activate()` retry it before
starting another session. If that activation raises `ModelRuntimeFailedError`,
keep the offloader referenced until you recreate the CUDA worker.

For frozen CPU models destined to stay on Apple MPS, `MpsWeights` provides a
separate constructor-time materializer. It retains no CPU cache, does not
preserve tied aliases, and leaves the model on MPS when deactivated. See its
[contract](../src/piper_offload/mps_weights.py).

## Share backing across models

Capture flat host backing once with `ModelStore.from_module(model)` and bind
separate fresh shells or `OffloadConfig` instances via `store.bind(shell,
config=...)`. Each `ModelOffloader` owns its bound shell exclusively; binding
another offloader onto a shell with a live binding is rejected. Preflight
validation rejections leave the target shell untouched, whereas a failure mid-way
through installation requires discarding that shell and binding a fresh one. Extra
unmanaged parameters on the target shell (including inside occupied blocks)
are ignored and stay on the caller-selected device, while extra non-empty blocks
inside configured block lists are rejected:

```python
import torch
from torch import nn
from piper_offload import ModelStore, OffloadConfig

prototype = nn.Linear(8, 4).eval().requires_grad_(False)
store = ModelStore.from_module(prototype)
second_shell = nn.Linear(8, 4, device="meta").eval().requires_grad_(False)

offloader = store.bind(second_shell, OffloadConfig())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
offloader.activate(device)
try:
    with torch.inference_mode():
        output = offloader.value(torch.randn(1, 8, device=device))
finally:
    offloader.deactivate()
```

Use `ModelStoreSpec` when caching one reusable `ModelStore` under a `ModelCache`
or `ResourceCache` byte budget. Because an escaped `ModelOffloader` binding keeps
host backing alive even if the store is later evicted, keep `with cache.lease(spec)
as store:` open around the bound shells' use and cleanup when enforcing a cache
budget.

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

Objects count as zero bytes unless their spec supplies `estimated_cache_bytes`.
`ResourceCache(max_cache_bytes=...)` starts with a finite budget and follows
the same [eviction and resize rules](#releasing-and-inspecting-resources).
For custom specs and eviction policies, see [Development](development.md#extending-resources).

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
[`ModelStore`](../src/piper_offload/model_store.py),
[`OffloadConfig`](../src/piper_offload/offload_config.py),
[`ModelSpec` and other specs](../src/piper_offload/resource_specs.py),
[`ModelOffloader`](../src/piper_offload/model_offloader.py), and
[`ResourceCache`](../src/piper_offload/resource_cache.py).
