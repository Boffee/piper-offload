# Piper Offload

Keep reusable PyTorch models in host memory and move their weights onto a GPU
when needed. Piper Offload manages whole-model uploads when a model fits in
VRAM, and block streaming when it does not. Models and adapters stay cached
between uses, including compatible checkpoint mappings.

Use it for workloads that run several models in sequence or repeatedly reuse
large models. `ModelCache` handles caching and activation; `ModelOffloader`
provides direct control over one model's lifecycle.

## Installation

Requires Python 3.14 or newer and PyTorch 2.14.

```bash
pip install piper-offload
```

The base dependencies are `torch` and `piper-kernels`. Optional integrations
are available through the `bnb`, `quanto`, `gguf`, and `torchao` extras;
`triton` adds accelerated kernels, and `all` includes every integration.
See [supported formats](docs/formats.md) for backend requirements.

## Quick start

This example runs on CUDA when available and on CPU otherwise. The factory
builds a fresh model; the cache retains it for subsequent uses.

```python
import torch
from torch import nn
from piper_offload import ModelCache, ModelSpec


def build_model():
    return nn.Sequential(
        nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4),
    ).eval().requires_grad_(False)


cache = ModelCache()
spec = ModelSpec(key="example", factory=build_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for _ in range(2):
    inputs = torch.randn(1, 8, device=device)
    with cache.use(spec, device=device) as model, torch.inference_mode():
        output = model(inputs)

print(output.shape)  # torch.Size([1, 4])
del model  # release the reference returned by the last use
cache.clear()
```

Each `use()` activates the model and deactivates it on exit. Place inputs on
the compute device yourself. For a model larger than VRAM, select its block
lists with `ModelSpec(..., block_paths=("transformer_blocks",))`; see
[streaming](docs/streaming.md) for the required model structure.

## Fit and limitations

- Each cached model has one runtime and supports sequential use. Concurrent
  replicas need separately constructed models under distinct keys.
- `ModelCache` retains host state until explicit eviction. Pinned memory has
  a separate budget; cache accounting is not a limit on process RAM.
- Factories transfer ownership of compatible CPU storage. Set `requires_grad`
  before capture and do not mutate factory-produced tensors afterwards.
- Buffer changes made on CUDA are discarded at deactivation. Models that
  need persistent buffer state across calls require a different arrangement.
- Compilation supports declared block forwards. Streamed training requires
  activation checkpointing and explicit handling of optimizer updates.

## Documentation

Both application developers and coding agents can start with the relevant
topic. These documents describe the checked-out revision; use the matching
Git tag when integrating a pinned release.

| Task | Read |
|---|---|
| Cache models, manage activation, inspect or release resources | [Models](docs/models.md) |
| Run models larger than VRAM, compile blocks, or train streamed weights | [Streaming](docs/streaming.md) |
| Apply LoRA, parameter deltas, or parameter values | [Adapters](docs/adapters.md) |
| Configure checkpoint backing, pinning, and Windows copy retention | [Memory](docs/memory.md) |
| Check tensor formats, optional dependencies, or DTensor support | [Formats](docs/formats.md) |
| Understand the implementation or contribute changes | [Development](docs/development.md) |

Contributor instructions are in [AGENTS.md](AGENTS.md). See
[versioning and releases](VERSIONING.md) for compatibility policy and
[the changelog](CHANGELOG.md) for release history.

Licensed under the [Apache License, Version 2.0](LICENSE).
