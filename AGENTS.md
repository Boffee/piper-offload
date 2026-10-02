# piper-offload

Piper Offload caches PyTorch model and adapter state in host memory and manages
whole-model uploads or block streaming. Piper Engine consumes pinned releases.

Start with [README.md](README.md) for the public entry point and topic map.
[Development](docs/development.md) covers architecture, extension points, and
validation. API docstrings describe per-symbol contracts; the topic guides
explain how those APIs work together.

## Engineering

- Lead with the problem: what goes wrong, for whom, how often, and how badly.
  Design for production needs and foreseeable uses. Hypothetical problems
  usually need a note or a contract before they need a mechanism.
- Minimize how much a reader must understand to make a correct change.
  Make ownership and invariants clear, and hide implementation details behind
  useful interfaces. Consider the whole design: broader changes, including
  to internal contracts, are welcome when they reduce special cases, coupling,
  or duplicated knowledge. Remove what they supersede. Preserve unrelated
  behavior, including which weights are pinned; make intended behavior changes
  explicit.
- State caller contracts in docs and docstrings. Runtime checks belong at
  boundaries where violations would be silent or unsafe. Avoid checking the
  same contract redundantly or adding defensive machinery for misuse the
  contract already forbids.
- Use strong types and let pyright prove what it can. Contain loose types at
  boundaries and explain why they are needed.
- Treat documentation as a coherent explanation of the current library.
  Rework the organization and examples when the concepts change, consolidate
  overlap, and remove obsolete material. Prefer one authoritative explanation
  with links from other contexts. Release history belongs in `CHANGELOG.md`.

## Contracts to preserve

Read the relevant guide before changing its behavior:

- [Models](docs/models.md): factories transfer compatible host storage;
  frozen bytes remain immutable; set `requires_grad` before capture. Runtime
  activation is exclusive, and construction and activation failures have
  different recovery paths.
- [Streaming](docs/streaming.md): shared storage is preserved within host
  state or one streamed block, not across their ownership boundaries.
  Training, transient lifetimes, and compilation have distinct contracts.
- [Adapters](docs/adapters.md) and [formats](docs/formats.md): update resources
  are immutable; merge, routed execution, and tensor-format capabilities are
  separate concepts.
- [Memory](docs/memory.md): mappings stay read-only; writes use owned storage.
  Asynchronous transfers hold pin leases and tensor adapters copy through
  `transfer_()`. Leases protect storage; only repeated transfers request
  pinning. The guide also covers Windows reclaim ordering and lock ownership.

Use the code's terminology consistently; see
[development](docs/development.md#architecture) for vocabulary.
A resource lease and a pin lease protect different things; an `Adapter`
resource and a tensor adapter serve different purposes.

## Validation

- Tests: `uv run pytest tests -q`. GPU tests skip without suitable hardware.
- Lint and types: `uv run ruff check .` and `uv run pyright`.
- Reviews should explain the failure, its impact, and where it occurs. Judge
  the resulting design, including the documentation. A finding about forbidden
  misuse calls for a clearer contract; a new capability needs a foreseeable use.

See [VERSIONING.md](VERSIONING.md) for compatibility and release policy.
