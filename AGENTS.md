# Piper Offload

Piper Offload caches PyTorch models and adapters in host memory and moves weights
to a device for execution. Start with the [README](README.md) for supported
workflows and the [architecture map](docs/development.md#architecture) to find
the code that owns a behavior.

## Engineering

- Start with the caller's problem. Choose the simplest design that meets a
  concrete need. Before adding a guard, flag, fallback, or dependency, ask
  whether removing a path, reading less data, or narrowing an interface would
  solve the problem.
- Give each invariant one owner. Validate untrusted data at the boundary that
  interprets it. Check caller preconditions where a violation would be silent
  or unsafe. Pass validated values inward using types Pyright can check.
  Explain loose types needed at library boundaries.
- Before changing a parser or a resource lifetime, state what must remain true
  on success and failure. For an asynchronous copy, identify who owns the
  storage and what proves the last reader or writer has finished.
- Judge simplicity across the implementation and its callers. Consolidate
  paths when that removes duplicated decisions. Preserve unrelated behavior,
  including which weights get pinned. If narrowing a contract would exclude a
  documented use, follow the [versioning policy](VERSIONING.md). Piper Engine
  consumes pinned releases.
- Prioritize security, memory safety, and maintainability. Use low-level FFI,
  raw pointers, or platform-specific syscalls only when measured workloads
  justify their complexity.
- Update the owning explanation and its examples when behavior changes. See
  [documentation ownership](docs/development.md#documentation-ownership).

## Fixing bugs

1. Reproduce the reported failure through a supported call. Identify who is
   affected and under what conditions. If it does not reproduce, report what
   you tried and what you observed.
2. Trace the failure to the component that owns the contract. Check sibling
   paths and earlier fixes for the same mistake. Fix an isolated error there.
   If the design lets it recur, change the design so one owner enforces the rule.
3. Verify that the regression test fails for the reported reason before the
   fix and passes afterward. Describe the cause and the caller-visible result
   in the pull request.

## Contracts to preserve

- [Models](docs/models.md). Models must be frozen (`requires_grad=False`) before
  capture. Factories transfer compatible host storage, and captured bytes stay
  immutable. Activation is exclusive. Construction and activation failures have
  different recovery paths.
- [Streaming](docs/streaming.md). Preserve aliases within each ownership
  boundary (host state or one streamed block). Consult the guide for
  compilation and transient lifetimes. [Scheduling hooks](docs/development.md#scheduling-hooks)
  must recover from interrupted, skipped, and direct calls.
- [Adapters](docs/adapters.md) and [formats](docs/formats.md). Model-update
  resources are immutable. Merge, routed execution, and tensor-format
  capabilities have separate contracts.
- [Memory](docs/memory.md). Keep mappings read-only and write to owned storage.
  Asynchronous transfers hold pin leases. Tensor adapters copy through
  `transfer_()`. Only repeated transfers request pinning. Memory component
  docstrings specify Windows reclaim and lock ordering.
- [Relay communication](src/piper_offload/communication.py). Preserve collective
  issue order, point-to-point progress, and buffer lifetimes through completion.
  A failed shared exchange can leave partial outputs and requires a new process group.

## Validation and review

Use [setup and validation](docs/development.md#setup-and-validation) for the
checks that apply to code or documentation. Report any affected hardware paths
you could not exercise. Follow [writing tests](docs/development.md#writing-tests)
when changing coverage.

Give substantial changes an [independent review](docs/development.md#reviews)
focused on the contracts at risk. Follow the
[commit, pull request, and changelog conventions](docs/development.md#commits-and-pull-requests).
