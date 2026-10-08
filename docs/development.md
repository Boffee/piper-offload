# Development

Piper Offload separates reusable host state from device execution. Most changes
belong to resource caching, activation scheduling, or tensor representation.
Understanding those boundaries makes it easier to extend the library without
duplicating policy. Start with [AGENTS.md](../AGENTS.md) for engineering guidance
and [models](models.md) for the public lifecycle.

## Setup and validation

Use the Python and PyTorch versions in [pyproject.toml](../pyproject.toml).
`uv sync` installs development dependencies, including optional tensor backends.

```bash
uv run pytest tests -q
uv run ruff check .
uv run pyright
```

GPU tests skip without suitable hardware. Run the relevant GPU tests when
changing transfers, scheduling, compilation, or quantized operations; a CPU-only
pass cannot validate those paths. CI checks Linux and Windows. Benchmarks under
[`benchmarks`](../benchmarks) exercise performance-sensitive paths; report the
hardware and configuration when using their results.

See [VERSIONING.md](../VERSIONING.md) for compatibility and release policy.
Piper Engine consumes pinned releases, so changes to public behavior need
matching documentation and release notes.

## Writing tests

Each case should protect a distinct behavior or failure. Extend existing coverage
where possible, and remove superseded cases and helpers while preserving the
regressions they catch.

- **Test the owning contract.** Keep detailed cases with the responsible component
  and use integration tests to verify composition. Assert results and ownership
  invariants, including storage identity and release ordering where relevant.
  Keep expected results independent of the implementation.
- **Use the appropriate boundary.** Test policy on CPU with small real objects or
  fake registration backends. Use real CUDA transfers, compilation, and processes
  when their behavior determines correctness. Skip unavailable hardware or
  dependencies before expensive setup.
- **Extend shared cases deliberately.** New formats should join applicable
  contract tests, with focused cases for distinct layouts, arithmetic, or runtime
  behavior. Parametrize meaningful boundaries and interactions; avoid multiplying
  unrelated dimensions. Assert expected capabilities so losing one fails a test.
- **Keep helpers narrow.** Share repeated mechanics while keeping important inputs
  and assertions visible. Split helpers that accumulate flags, format branches,
  or conditional expectations. Some repeated setup is preferable to a generic
  test framework; see Google's [test-maintainability guidance](https://abseil.io/resources/swe-book/html/ch12.html).
- **Preserve isolation and ownership.** Create fresh mutable models, caches, and
  managers. Share immutable inputs only where ownership permits it: captured
  storage cannot become another test's scratch space. Pair acquisition with
  cleanup after setup or assertion failures; see [pytest fixture guidance](https://docs.pytest.org/en/stable/how-to/fixtures.html#safe-teardowns).
- **Minimize work without weakening the case.** Use tiny inputs that retain the
  relevant page crossings, aliases, partial tiles, or block reuse. Use events and
  bounded waits for concurrency. Scope garbage collection and compiler resets to
  tests that need them. Batch related process cases only when state can be reset
  and failures remain identifiable.

Measure additions involving compilation, process startup, broad fixtures, or
large parameter matrices, including setup and teardown:

```bash
uv run pytest tests --collect-only -q
uv run pytest tests -q --durations=20
```

Record hardware, dependency versions, and compiler-cache conditions. Test bounded
work through allocation, transfer, or registration counts; keep latency and
throughput measurements in `benchmarks/`. Targeted runs help during development;
complete the validation required by the change before finishing.

## Architecture

| Part | Responsibility | Main source |
|---|---|---|
| Resource cache | Specs, stores, accounting, resource leases, eviction | [resource_cache.py](../src/piper_offload/resource_cache.py), [resource_specs.py](../src/piper_offload/resource_specs.py) |
| Model cache | Lease adapters and a model, then activate the model for a use | [model_cache.py](../src/piper_offload/model_cache.py) |
| Model offloader | One model's host capture, components, hooks, activation and deactivation | [model_offloader.py](../src/piper_offload/model_offloader.py) |
| Components | Own host state and device working sets for whole modules or block lists | [host_component.py](../src/piper_offload/host_component.py), [block_component.py](../src/piper_offload/block_component.py) |
| Host parameters and tensor adapters | Capture, account for, transfer, and reconstruct each tensor representation | [host_param.py](../src/piper_offload/host_param.py), [tensor_adapters.py](../src/piper_offload/tensor_adapters.py) |
| Model adapters | Immutable LoRA, delta, and value resources applied through load plans or hooks | [adapter.py](../src/piper_offload/adapter.py), [merge.py](../src/piper_offload/merge.py) |
| Pin manager and memory components | Transfer leases, registration, copies, platform allocation and retention | [memory guide](memory.md#implementation-notes) |

A *resource* is built from a *spec* into a *store*. A *binding* adds an active
device lifecycle. `ModelOffloader` is both a store and a binding; `Adapter` is
immutable host backing without an activation lifecycle. A resource lease
protects a store from eviction. A pin lease independently protects the storage
used by a transfer.

An activation is a *session*. Its *load plans* combine base weights with any
requested adapter updates. *Host state* is the CPU representation held by a
`HostParam` and interpreted by its *tensor adapter*. Capitalized `Adapter`
means a model-update resource, rather than that per-format protocol.

`ModelOffloader` composes a host component for non-streamed state, host
components for transient modules, and block components for declared block
lists. Components have `activate()` and `deactivate()` but no model-like
`value`. Their CUDA working sets can be released and reacquired during an
activation. `register_forward_hook()` on the offloader installs a native
PyTorch hook by qualified module name and returns a caller-owned remover.

Capture finalizes a store's `cache_bytes`. Activation acquires device state;
deactivation releases it while retaining captured host state. The offloader
cleans up partial activation on failure. Lower-level component callers own
their cleanup; see each component's docstring. Construction failures require
discarding the partially captured model, as described in [models](models.md).

## Extending resources

The [resource protocols](../src/piper_offload/protocols.py) are structural;
implementing them does not require inheritance:

- `ResourceSpec` supplies `key`, `estimated_cache_bytes`, `build_store()`,
  and `value(store)`.
- `ResourceStore` owns backing state and reports `cache_bytes`.
- `ResourceBinding` adds `value`, `activate()`, and `deactivate()` where a
  resource needs an active lifecycle.

Use `ObjectSpec` for ordinary cached objects. A custom spec and store are
useful when construction or byte accounting needs different behavior. Leasing
a binding does not activate it; the caller or a coordinating cache does that.

An `EvictionPolicy` chooses victims from inactive candidates. The cache still
owns validation, admission, accounting, and release. Policy methods run under
the cache lock. `choose_victims()` returns unique candidate keys accounting
for at least `context.bytes_to_free`; invalid selections raise
`EvictionPolicyError` before eviction. See the built-in `LRUEvictionPolicy`
in [resource_cache.py](../src/piper_offload/resource_cache.py).

## Extending tensor formats

Implement the stateless [TensorAdapter](../src/piper_offload/tensor_adapters.py)
protocol and call [`register_adapter()`](../src/piper_offload/tensor_adapter_registry.py)
during application startup, before constructing host resources. Registration
returns an idempotent removal callable. Dispatch checks DTensor first, external
adapters newest-first next, then built-in adapters. DTensor delegates its local
shard through the same registry.

The base protocol covers capture, physical storage enumeration, device copy,
wrapper reconstruction, byte accounting, compute dtype, identity, and block
layout. `storage_tensors()` returns existing plain CPU tensors, including
tensor-valued metadata, while preserving views and shared allocations. It
does not reconstruct wrappers or make copies.

Every physical host-to-device copy goes through `transfer_()` so checkpoint
copies and leases are honored. Copying directly from captured tensors can
bypass the very owned copy that the pin manager prepared. The full lease and
registration contracts are in [memory](memory.md).

Additional operations are separate capability protocols: CPU round-trip,
trainable parameter data swaps, representation-preserving copy, conversion,
and factorized or dense merge. Advertising conversion or copy does not imply
merge support. The [format matrix](formats.md) describes built-in capabilities.

Merge validation distinguishes target-only constraints from constraints that
need the staged update. Permanent merge validates requested operations before
mutation. Composing adapters can expose local shapes and global offsets through
`MergeLocalityTensorAdapter`. Merge and validation implementations accept
`rounding_seed: int | None = None`, even when they only support deterministic
rounding; `None` retains deterministic behavior. `derive_seed()` provides
reproducible substreams when needed.

## Runtime and compilation

The streaming, rolling, and resident runtimes share host state but schedule
different device working sets. Their user-visible boundaries are documented
in [streaming](streaming.md). Host capture and runtime changes must preserve
which transfers request pinning; a lease holds storage and does not itself
choose the pinning policy.

Rolling compilation tracks the first and last reader of each parameter across
all physical inputs of structured weights. After the last reader launches,
a CUDA event allows the copy stream to refill that storage for the next block;
the next block waits at its first reader. The rollover pass runs after user
post-grad passes so it sees their rewritten graph. Scheduler ordering preserves
the ordinary compute kernels and autotuning identity without describing frozen
parameters as graph mutations.

See [block_compile.py](../src/piper_offload/block_compile.py) and
[rolling_compile.py](../src/piper_offload/rolling_compile.py) for compiler
boundaries, and [the rolling benchmark](../benchmarks/benchmark_rolling_compile.py)
for output, latency, and allocator-residency comparisons. Compiler failures
propagate: retrying a partly executed graph eagerly could repeat mutations.
Compiler artifact caches and workspace are outside resource byte accounting;
model eviction does not reset the process-wide compiler cache.

## Experimental DTensor execution

DTensor weight capture composes with tensor adapters; see [formats](formats.md).
Two separate execution experiments sit outside ordinary model caching:

- [`communication.py`](../src/piper_offload/communication.py) registers a blocking
  host-relay process group. Gloo or shared host allocations transport payloads
  between processes, using the pin manager for host staging.
- [`sequential.py`](../src/piper_offload/sequential.py) runs two ranks in one
  process on a shared GPU and compute stream, yielding between ranks at
  collective boundaries so temporary buffers can be reused.

Their module and API docstrings contain usage, supported collectives, and
failure recovery. Neither facility coordinates ordinary offloader scheduling.
