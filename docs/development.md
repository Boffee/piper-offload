# Development

Start with [AGENTS.md](../AGENTS.md) for design and bug-fix decisions. Use the
[architecture map](#architecture) to find the owner of a change and the
[models guide](models.md) to trace the lifecycle a caller sees.

## Setup and validation

For documentation-only changes, check the edited examples, symbols, local
links, and heading anchors. Run any example whose executable steps changed.

For code changes, install the locked development environment with
`uv sync --locked --dev`. Use a focused pytest target while developing, then
run the full checks before finishing.

```bash
uv run pytest tests -q
uv run ruff check .
uv run pyright
```

To enable the optional local Ruff, Pyright, and CPU test hooks, run
`uv run pre-commit install`.
[CI](../.github/workflows/ci.yml) runs the portable suite on Linux and Windows
with CUDA hidden. Tests that need a GPU must mark that requirement explicitly.
Run relevant tests on the affected hardware for transfers, scheduling,
compilation, or quantized operations. For relay changes that depend on rank
coordination, exercise the affected transport with multiple processes.

For performance claims, compare the same workload and configuration on the
same hardware, checking outputs before comparing latency and allocator residency.

## Writing tests

Keep detailed cases with the component that owns the rule. Use an integration
test when the failure depends on how components work together.

- **Check the consequence.** For a checkpoint with an invalid tensor range,
  verify rejection before a tensor can expose bytes outside the mapping. If
  CUDA cleanup leaves work outstanding, check that the cached model stays
  protected from eviction until cleanup succeeds.
- **Choose the smallest faithful setup.** Test policy on CPU using small real
  objects or a fake registration backend. Use real transfers, compilers, or
  processes when correctness depends on them. Keep the alias, page crossing,
  partial tile, or block reuse that makes the case meaningful. Skip unavailable
  hardware and dependencies before expensive setup.
- **Extend existing coverage.** Add formats to the shared tests for their
  declared capabilities. Use focused cases for distinct layouts or arithmetic
  and independent expected results. When consolidating tests, preserve each
  distinct scenario and its assertions. Parametrize inputs that exercise
  different behavior. Split helpers once flags or branches obscure what a case tests.
- **Make timing and ownership explicit.** Coordinate concurrency with events
  and bounded waits. Give each test fresh mutable models, caches, and managers.
  Share inputs only while their storage can stay immutable. Arrange cleanup
  to run even if setup or an assertion fails. Scope garbage collection and
  compiler resets to cases that need them.

Measure costly additions such as compilation, process startup, broad fixtures,
or large parameter matrices. Include setup and teardown.

```bash
uv run pytest tests --collect-only -q
uv run pytest tests -q --durations=20
```

CI retains JUnit timings for 14 days in `test-results-linux` and
`test-results-windows`, including after test failures. Compare repeated runs
with matching hardware, dependencies, inputs, and compiler-cache state.
Batch process cases only if each starts from known state and failures remain
identifiable. When testing a bound on work, assert the relevant allocation,
transfer, or registration count.
Keep latency and throughput measurements in [benchmarks](../benchmarks).

## Architecture

| Part | Responsibility | Main source |
|---|---|---|
| Resource cache | Specs, stores, accounting, resource leases, eviction | [resource_cache.py](../src/piper_offload/resource_cache.py), [resource_specs.py](../src/piper_offload/resource_specs.py) |
| Checkpoint reader | Validate file headers, map tensor storage read-only, and record its file slice | [checkpoint.py](../src/piper_offload/checkpoint.py) |
| Model cache | Lease adapters and a model, then activate the model for a use | [model_cache.py](../src/piper_offload/model_cache.py) |
| Model store and offload config | Capture flat frozen host backing once and configure/bind offload topologies | [model_store.py](../src/piper_offload/model_store.py), [offload_config.py](../src/piper_offload/offload_config.py) |
| Model offloader | Coordinate shell ownership, activation, and dynamic VRAM controls | [model_offloader.py](../src/piper_offload/model_offloader.py) |
| Model partition | Partition resident state and offloaded units with cross-boundary alias validation | [model_partition.py](../src/piper_offload/model_partition.py) |
| Elastic VRAM pool & residency engine | Capacity-bounded GPU slot reuse (Belady MIN), trace lookahead, and hook lifecycle | [vram_pool.py](../src/piper_offload/vram_pool.py), [residency_engine.py](../src/piper_offload/residency_engine.py) |
| Expert slab pool | Global multi-group fixed-drawer cache, dumb-padded transfers, W-TinyLFU / bypass waves, and frontier reclaim | [expert_slab_pool.py](../src/piper_offload/expert_slab_pool.py) |
| KV chunk pool | VRAM-only descending fixed-drawer KV-chunk manager (`produce`/`read`/`discard`) and `EMPTY`-only frontier reclaim | [kv_chunk_pool.py](../src/piper_offload/kv_chunk_pool.py) |
| Shared slab pool helpers | Internal CUDA `StreamLease` / `_WaveReleaseTracker` lifecycle helpers shared by Layer 2 pools | [_slab_pool_common.py](../src/piper_offload/_slab_pool_common.py) |
| Shared Layer 1 substrate | Multi-reader/writer `StreamLease`, shared `DeviceTransferContext`, `VramBudget` coordinator, and single-`uint8` `Uint8DualArena` (`SlotSchema`, `LowSlotLayout`/`HighSlotLayout`, peer frontier reclaimers) + `TypedSidePool` | [stream_lease.py](../src/piper_offload/stream_lease.py), [device_transfer.py](../src/piper_offload/device_transfer.py), [vram_budget.py](../src/piper_offload/vram_budget.py), [dual_arena.py](../src/piper_offload/dual_arena.py) |
| Host parameters and tensor adapters | Capture, account for, transfer, and reconstruct each tensor representation | [host_param.py](../src/piper_offload/host_param.py), [tensor_adapters.py](../src/piper_offload/tensor_adapters.py) |
| Model adapters | Immutable LoRA, delta, and value resources merged during weight loading and permanent merge | [adapter.py](../src/piper_offload/adapter.py), [merge.py](../src/piper_offload/merge.py) |
| Parameter updates | Select merge targets and sources, validate them, and plan arithmetic for activation and permanent merge | [_parameter_update.py](../src/piper_offload/_parameter_update.py), [lora.py](../src/piper_offload/lora.py) |
| Pin manager | Transfer leases, Tier 2a/2b slab management, Direct I/O checkpoint fill, lazy bounce streaming, and budget accounting | [pin_manager.py](../src/piper_offload/pin_manager.py) |
| Transfer priority | Background host-to-device copies wait while foreground transfers move data | [_transfer_priority.py](../src/piper_offload/_transfer_priority.py) |
| Host arena & direct I/O | Page-aligned direct file reader, Tier 2a/2b slab arena, and pinned bounce ring staging | [_direct_io.py](../src/piper_offload/_direct_io.py), [host_arena.py](../src/piper_offload/host_arena.py) |
| Relay process group | Coordinate collectives, point-to-point operations, and host staging across ranks | [communication.py](../src/piper_offload/communication.py) |
| In-process multi-rank executors | Coordinate concurrent or serialized multi-rank threads, runtime isolation, and local transports | [sequential.py](../src/piper_offload/sequential.py), [parallel.py](../src/piper_offload/parallel.py), [_sequential_runtime.py](../src/piper_offload/_sequential_runtime.py), [_parallel_collectives.py](../src/piper_offload/_parallel_collectives.py) |
| Platform host memory | Physical and cgroup memory budgets and aligned OS region allocation | [_host_memory.py](../src/piper_offload/_host_memory.py), [_host_memory_linux.py](../src/piper_offload/_host_memory_linux.py), [_host_memory_windows.py](../src/piper_offload/_host_memory_windows.py) |
| Block compilation | Compile declared offloaded unit forwards | [block_compile.py](../src/piper_offload/block_compile.py) |

A resource *spec* builds a *store*. Capture establishes its `cache_bytes`.
A resource lease protects that store from eviction. A pin lease protects
storage used by a transfer. The runtime decides whether to request registration
under the [transfer pinning policy](memory.md#set-host-memory-budgets).

A *binding* adds a device lifecycle to a store. `ModelOffloader` implements
both. `ModelPartition` splits non-offloaded resident weights from offloaded units,
and `ResidencyEngine` orchestrates their CUDA lifecycle and `ElasticVramPool`
slots during an activation *session*. Deactivation releases device state and
retains captured host state.

An `Adapter` is an immutable model-update resource. A *tensor adapter*
interprets the CPU representation (*host state*) held by a `HostParam` and
owns format-specific arithmetic. Activation load plans and permanent merge
share parameter-update planning. Each path owns its allocations, leases, and
installation of results.

The offloader cleans up partial activation. If construction fails past preflight
checks, discard the partially modified model. See
[manual lifecycle](models.md#manual-lifecycle) for the public recovery paths.

When a log record or object retains a failure past the `except` block, store
`str(error)`. Keeping the exception also keeps its traceback, including caller
frames and any tensors they reference.

### Scheduling hooks

Unit offloading runs from module forward hooks. A hook
must work even if an earlier hook was skipped or interrupted.

- A forward that raises skips ordinary post-hooks. `always_call` handles
  `Exception`. A `KeyboardInterrupt` or cancellation outside that hierarchy
  can skip even those hooks.
- Callers can invoke a submodule directly, without the root model's hooks.
- Models skip, reorder, or loop units dynamically.

Act on the state actually installed.

1. Mark a multi-step transition in progress before the first step. Commit it
   only after the last. A call that finds an unfinished transition releases
   first. Idempotent release steps finish an interrupted release or roll back
   an interrupted acquire. The engine records this in
   [`WorkingSetState`](../src/piper_offload/_working_set.py). Unit switches
   in `ResidencyEngine` follow the same rule.
2. Re-establish the working set where it is used. Replace call-scoped state
   on the next call so interrupted forwards cannot accumulate it.
3. Hand storage back only once its last user has finished. If detaching a
   module or waiting for a writer fails, retain ownership so cleanup can retry.
   If a wait is interrupted, finish waiting for the worker before propagating
   the interrupt. The next forward can then safely reuse its target.

Add new modes and hook-driven features to the existing
[forward interruption](../tests/test_forward_interruption.py) and
[release failure](../tests/test_release_failure.py) cases. They interrupt a
forward or fail a release step and exercise recovery on the next call.

## Documentation ownership

Update the explanation where its reader will look for it. Revise examples
in place, remove superseded instructions, and link to the owner from elsewhere.

| Reader's question | Where to answer it |
|---|---|
| Does Offload fit my workload? How do I start? | The README covers installation, limitations, a working example, and a map of the guides. |
| How do these APIs work together? | Topic guides explain caller workflows and the ownership, lifetime, device, format, or numerical constraints that affect use. |
| How do I call or change this symbol? | Public docstrings lead with behavior, inputs, results, and constraints. Internal docstrings own layouts, algorithms, lock ordering, and local assumptions. Wrappers can refer to the underlying contract. |
| How do I run this tool or benchmark? | Tool docstrings and command help own invocation details. |
| How do I contribute? | `AGENTS.md` gives the decision rules. This guide covers architecture, cross-component invariants, extension, testing, and review. |
| Can a release change this behavior? What shipped? | [VERSIONING.md](../VERSIONING.md) defines compatibility and release policy. [CHANGELOG.md](../CHANGELOG.md) records caller-visible changes. |

## Extending resources

Implement the [resource protocols](../src/piper_offload/protocols.py)
structurally by supplying the members the resource needs.

- `ResourceSpec` supplies `key`, `estimated_cache_bytes`, `build_store()`,
  and `value(store)`.
- `ResourceStore` owns backing state and reports `cache_bytes`.
- `ResourceBinding` adds `value`, `activate()`, and `deactivate()` where a
  resource needs an active lifecycle.

Use `ObjectSpec` for ordinary cached objects. Define a custom spec and store
when construction or byte accounting needs different behavior. Leasing a
binding keeps its store alive. Activation is a separate step owned by the
caller or coordinating cache.

An `EvictionPolicy` chooses victims from inactive candidates. The cache still
owns validation, admission, accounting, and release. For locking and selection
rules, see the policy protocol and its built-in `LRUEvictionPolicy`
implementation in [resource_cache.py](../src/piper_offload/resource_cache.py).

## Extending tensor formats

1. Implement the stateless [TensorAdapter](../src/piper_offload/tensor_adapters.py)
   protocol. Its methods cover capture, physical storage, device copy, wrapper
   reconstruction, byte accounting, compute dtype, identity, and block layout.
   Route every physical host-to-device copy through `transfer_()` so the transfer
   uses the storage the pin manager prepared. See [custom transfers](memory.md#custom-transfers).
2. Implement the capabilities the format supports. Bind-target relaxation
   (`BindTargetTensorAdapter`), decoded GPU target sizing
   (`GpuTargetBytesTensorAdapter`, required when active GPU target allocation
   bytes differ from resting host `cache_bytes` so `vram_budget_bytes` stays
   accurate), caller-managed arena binding (`ArenaBindableTensorAdapter` via
   `BufferSpec` and `GpuAllocator`), zero-copy stacked-axis slicing
   (`SliceableTensorAdapter`), and merge are separate protocols.
   For factorized or dense merge, the merge and preflight
   docstrings specify rounding, coefficients, broadcasting, locality, and
   storage preservation. See the [caller contract](adapters.md#broadcast-additive-updates)
   for how updates compose.
3. For a built-in format, add the adapter class to `_BUILTIN_ADAPTERS` in the
   [registry](../src/piper_offload/tensor_adapter_registry.py). Add the format
   to the shared tests for its capabilities and update the [format matrix](formats.md).
   For an external format, call `register_adapter()` with the adapter class at
   application startup, before constructing host resources.

For experimental DTensor execution, see [formats](formats.md#dtensor-and-custom-formats)
and the module contracts linked there.

## Reviews

For a substantial change, ask an independent reviewer to start from the caller's
contract and trace the paths affected by the diff. Choose the questions that
apply.

- **Can the input make Offload do unsafe or unnecessary work?** For a checkpoint
  reader, trace a file-supplied length or offset to the first read or tensor
  view that uses it. Check where its bounds are established. The GGUF reader,
  for example, skips tokenizer metadata it never needs to decode.
- **When can storage change hands?** Follow one interruption through cleanup
  and the next use. For relay staging, identify what proves a copy is complete
  before another transfer gets the buffer. Check the relevant failure case in
  the [relay tests](../tests/test_communication.py).
- **Does the change simplify correct use?** Identify the caller that needs
  each new option or precondition. Look for an existing owner that can enforce
  the rule once. Check any restriction on supported use against the
  [compatibility policy](../VERSIONING.md#compatibility-policy).

For a behavior bug, follow the [bug-fix workflow](../AGENTS.md#fixing-bugs).
For a documentation finding, show where the guidance could lead a reader to
make the wrong decision and what they should do there.

## Commits and pull requests

Use an imperative subject of at most 72 characters for commits and PR titles,
such as "Preserve mapped weights during host capture". Omit type prefixes
such as `feat:`. Follow the commit subject with a blank line and a body
wrapped at 72 columns explaining why. Include the tests and lint fixes for
the change. Keep generated outputs, model caches, and secrets out of commits.

A pull request opens with the problem, who encounters it, and under what
conditions. Explain the cause of a bug, the design you chose, and what it
replaces.

- **Caller impact.** Identify changes to the Python API, lifecycle, ownership,
  pinning, formats, dependencies, or numerical results. Describe any migration
  for callers, including Piper Engine when it updates its pin.
- **Validation.** Give the commands run and their outcomes. Name affected
  paths you could not exercise, such as Windows GPU or distinct-GPU relay
  transfers. Include hardware, workload, and configuration for performance claims.

Link related issues and PRs. For a caller-visible change, add an entry under
**Unreleased** in `CHANGELOG.md`. Use Added, Changed, or Fixed, reusing
existing headings. Prefix breaking changes with `Breaking:`. If an unreleased
feature changes again, revise its entry. Describe what callers will receive
and any action they must take. Keep implementation and validation details in
the PR. Documentation, tests, and internal refactors need no changelog entry.
