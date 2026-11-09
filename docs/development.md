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

GPU tests carry skip marks for missing hardware; an unmarked test that needs
CUDA fails on a CPU-only host instead of skipping. Run the relevant GPU tests
when changing transfers, scheduling, compilation, or quantized operations; a
CPU-only pass cannot validate those paths. `pre-commit install` runs ruff,
pyright, and the CPU test suite on commits that change code. CI checks Linux
and Windows. Benchmarks under [`benchmarks`](../benchmarks) exercise
performance-sensitive paths. The
[rolling benchmark](../benchmarks/benchmark_rolling_compile.py) compares
outputs, latency, and allocator residency. Report the hardware and
configuration when using benchmark results.

See [VERSIONING.md](../VERSIONING.md) for compatibility and release policy.
Piper Engine consumes pinned releases, so changes to public behavior need
matching documentation and release notes.

## Writing tests

Each case should protect a distinct behavior or failure. Extend existing coverage
where possible, and remove superseded cases and helpers. Preserve distinct
scenarios and assertions when consolidating; matching line and branch coverage
alone does not prove test equivalence.

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

CI retains JUnit timings, including setup and teardown, for 14 days in the
`test-results-linux` and `test-results-windows` artifacts, also after test failures.

Record hardware, dependency versions, and compiler-cache conditions. Compare
repeated before/after runs with matching inputs and cache state; fewer cases alone
do not imply a faster suite. Test bounded work through allocation, transfer, or
registration counts; keep latency and throughput measurements in `benchmarks/`.
Targeted runs help during development; complete the validation required by the
change before finishing.

## Architecture

| Part | Responsibility | Main source |
|---|---|---|
| Resource cache | Specs, stores, accounting, resource leases, eviction | [resource_cache.py](../src/piper_offload/resource_cache.py), [resource_specs.py](../src/piper_offload/resource_specs.py) |
| Model cache | Lease adapters and a model, then activate the model for a use | [model_cache.py](../src/piper_offload/model_cache.py) |
| Model offloader | One model's host capture, components, hooks, activation and deactivation | [model_offloader.py](../src/piper_offload/model_offloader.py) |
| Components | Own host state and device working sets for whole modules or block lists | [host_component.py](../src/piper_offload/host_component.py), [block_component.py](../src/piper_offload/block_component.py) |
| Host parameters and tensor adapters | Capture, account for, transfer, and reconstruct each tensor representation | [host_param.py](../src/piper_offload/host_param.py), [tensor_adapters.py](../src/piper_offload/tensor_adapters.py) |
| Model adapters | Immutable LoRA, delta, and value resources applied through load plans or hooks | [adapter.py](../src/piper_offload/adapter.py), [merge.py](../src/piper_offload/merge.py) |
| Parameter updates | Shared source selection, validation, and arithmetic for activation and permanent merge | [_parameter_update.py](../src/piper_offload/_parameter_update.py), [lora.py](../src/piper_offload/lora.py) |
| Pin manager | Transfer leases, page accounting, registration order, and the shared lock | [pin_manager.py](../src/piper_offload/pin_manager.py) |
| Memory components | Owned checkpoint copies, platform allocation, and Windows retention | [_copy_memory.py](../src/piper_offload/_copy_memory.py), [_copy_memory_windows.py](../src/piper_offload/_copy_memory_windows.py), [_host_memory.py](../src/piper_offload/_host_memory.py) |
| Block compilation | Compile declared block forwards and schedule rolling transfers | [block_compile.py](../src/piper_offload/block_compile.py), [rolling_compile.py](../src/piper_offload/rolling_compile.py) |

A *resource* is built from a *spec* into a *store*. A *binding* adds an active
device lifecycle. `ModelOffloader` is both a store and a binding; `Adapter` is
immutable host backing without an activation lifecycle. A resource lease
protects a store from eviction. A pin lease independently protects the storage
used by a transfer; runtimes choose whether to request registration. Host
capture and runtime changes must preserve the [transfer pinning policy](memory.md#set-the-pin-budget).

An activation is a *session*. Its *load plans* combine base weights with any
requested adapter updates. Activation and permanent merge share parameter-update
planning. Application paths own allocation, leases, and installation; tensor
adapters own representation-specific arithmetic.

*Host state* is the CPU representation held by a
`HostParam` and interpreted by its *tensor adapter*. Capitalized `Adapter`
means a model-update resource, rather than that per-format protocol.

`ModelOffloader` composes a host component for non-streamed state, host
components for transient modules, and block components for declared block
lists. Components own device working sets that can be released and reacquired
during an activation; their docstrings state the lower-level lifecycle contracts.

Capture finalizes a store's `cache_bytes`. Activation acquires device state;
deactivation releases it while retaining captured host state. The offloader
cleans up partial activation on failure. Lower-level component callers own
their cleanup; see each component's docstring. Construction failures require
discarding the partially captured model, as described in [models](models.md).

A failure kept after its `except` block ends, in a log record or on an object,
is kept as text: pass `str(error)`, not the exception. An exception's traceback
keeps every frame on its call stack alive, callers included, so a logging
handler that buffers records would also keep the tensors and copies those
frames reference.

### Scheduling hooks

Transient release, block streaming and rolling refill, and routed LoRA staging
run from module forward hooks. Those hooks do not reliably run in pairs:

- A post-hook is skipped when the forward raises. `always_call` hooks still
  run for an `Exception`, but not for `KeyboardInterrupt` or other
  `BaseException` errors such as task cancellation.
- Callers can invoke a submodule directly, without the root model's hooks.
- Models skip or reorder blocks, for example for skip-layer guidance.

So a hook acts on the state that is actually installed instead of assuming
the previous hook ran, and the next call repairs whatever an interrupted one
left:

- A multi-step transition is marked in progress before its first step and
  committed only after its last, so an exception anywhere, including an
  interrupt, leaves it marked. A call that finds it in progress releases
  first; release steps are idempotent, so that finishes an interrupted release
  and rolls back an interrupted acquire. Components record this as their
  [`WorkingSetState`](../src/piper_offload/_working_set.py), and the streaming and
  rolling runtimes mark their block switches and refills the same way. Use the
  same pattern for any new multi-step transition.
- A working set is re-established where it is used, not only where the
  previous call would have restored it.
- State scoped to one call is replaced by the next call, so interrupted calls
  cannot accumulate it.
- Storage changes hands only after the step that frees it returns: a worker
  has finished writing it, or no module still points at it. A `finally` or
  `except` that runs because that step failed or was interrupted keeps the
  storage where it is, so cleanup can retry; leaking it is recoverable,
  reusing it is not. Waiting for a worker therefore ends only once the worker
  has finished, even when an interrupt ends the wait.

[test_forward_interruption.py](../tests/test_forward_interruption.py) runs each
block mode and hook-driven feature through these interruptions, and
[test_release_failure.py](../tests/test_release_failure.py) fails each release
step in turn; add new modes and features to their configurations.

## Documentation ownership

Each explanation has one owner, and other places link to it rather than restate it.

- **The README** is the public entry point: installation, a first working example,
  fit and limitations, and a task-oriented map of the guides.
- **Topic guides** in `docs/` are written for callers. They explain a subsystem's
  concepts, invariants, and how its APIs work together. Include ownership,
  lifetime, device, format, and numerical constraints when they change valid
  usage or how a caller interprets a result.
- **Module, class, and function docstrings** own per-symbol contracts and
  everything only a contributor to that code needs: tensor layouts, algorithms,
  scheduling, lock ordering, and internal assumptions. A topic guide has no
  implementation-notes section. Link to the owning code for those details.
  Public API docstrings lead with behavior and state the inputs, results, and
  constraints needed for a valid call; reuse existing contracts rather than
  repeat them at every wrapper.
- **Tool docstrings and command help** own specialized procedures and benchmark
  invocation details.
- **This guide** owns the architecture map, extension workflow, and conventions
  and invariants that span subsystems. `AGENTS.md` supplies concise repository
  instructions and links here for details.
- **VERSIONING.md** owns compatibility and release policy. `CHANGELOG.md` owns
  release history; see [commits and pull requests](#commits-and-pull-requests).

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
owns validation, admission, accounting, and release. The policy protocol in
[resource_cache.py](../src/piper_offload/resource_cache.py) states the locking
and victim-selection contracts; `LRUEvictionPolicy` is the built-in example.

## Extending tensor formats

Implement the stateless [TensorAdapter](../src/piper_offload/tensor_adapters.py)
protocol and call [`register_adapter()`](../src/piper_offload/tensor_adapter_registry.py)
during application startup, before constructing host resources. Registration
and dispatch contracts belong to the registry's docstrings.

The base protocol covers capture, physical storage enumeration, device copy,
wrapper reconstruction, byte accounting, compute dtype, identity, and block
layout. Its method docstrings define the representation and ownership contracts.

Every physical host-to-device copy goes through `transfer_()` so checkpoint
copies and leases are honored. Copying directly from captured tensors can
bypass the very owned copy that the pin manager prepared. The full lease and
registration contracts are in [memory](memory.md).

Additional operations are separate capability protocols: CPU round-trip,
trainable parameter data swaps, representation-preserving copy, conversion,
and factorized or dense merge. Advertising conversion or copy does not imply
merge support. The [format matrix](formats.md) describes built-in capabilities.

For merge support, implement the applicable merge and preflight protocols in
[tensor_adapters.py](../src/piper_offload/tensor_adapters.py). Their docstrings
own coefficient, rounding, broadcasting, locality, and storage-preservation
requirements. The caller-visible [shape and rounding contract](adapters.md#broadcast-additive-updates)
explains how applications compose updates. Add the format to the applicable
shared contract tests and document its supported capabilities in the format matrix.

For experimental DTensor execution, see [formats](formats.md#dtensor-and-custom-formats)
and the module contracts linked there.

## Commits and pull requests

Commit messages use the standard Git format: an imperative subject of at most
72 characters, such as "Preserve mapped weights during host capture", then a
blank line and a body wrapped at 72 columns that explains why. Subjects take no
type prefix such as `feat:`; the changelog classifies changes. PR titles follow
the same subject rule and should be suitable as squash-merge commit subjects.
Commits stay focused and include the tests and lint fixes for their change.
Do not commit generated outputs, model caches, or local secrets.

A pull request description leads with the problem: what goes wrong, for whom,
and how often. Then it covers:

- **Change:** what the PR does, why this design, and what it removes or supersedes.
- **Compatibility:** whether the Python API, resource or activation lifecycle,
  ownership and pinning behavior, supported formats or dependencies, or numerical
  results change. State any caller migration, including effects on Piper Engine's
  use of pinned releases; see [VERSIONING.md](../VERSIONING.md).
- **Validation:** the commands run and their results, and what remains
  unvalidated, such as GPU or Windows paths. Performance claims include the
  hardware, workload, and configuration used for measurements.

Link related issues and PRs. Add a `CHANGELOG.md` entry under **Unreleased**
when a caller can observe the change: in the Python API, lifecycle, memory
behavior, supported configurations, or results. Documentation, tests, and
internal refactors need none. Use Added, Changed, or Fixed headings as applicable,
reusing existing headings, and prefix a breaking change with "Breaking:".
State what changes for the caller and what they must do; implementation,
benchmark narratives, and test details belong in the PR.
