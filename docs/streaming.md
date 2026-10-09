# Block streaming

Block streaming runs models whose weights exceed GPU memory by loading blocks
as they execute. It trades transfer time for a smaller GPU working set. Start
with the [README](../README.md) for an overview and [models](models.md) for
capture, caching, and activation.

## Select block groups

Set `ModelSpec.block_paths` to dotted paths resolving to nonempty
`nn.ModuleList` objects. Each list becomes an independent group. The model's
forward calls its blocks normally, letting Piper's hooks load their weights.
Everything outside those groups stays resident for the activation by default.

This small model shows the required structure; the same setup applies to a
larger model with, for example, a block list at `model.transformer.blocks`:

```python
import torch
from torch import nn
from piper_offload import ModelCache, ModelSpec


class BlockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(8, 8) for _ in range(3))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def build_model():
    return BlockModel().eval().requires_grad_(False)


cache = ModelCache()
spec = ModelSpec(key="streaming-example", factory=build_model, block_paths=("blocks",))
with cache.use(spec, device="cuda") as model, torch.inference_mode():
    output = model(torch.randn(1, 8, device="cuda"))

del model
cache.clear()
```

Enter `cache.use()` before `torch.inference_mode()` so background copies can
refill the streamed targets. Keep inference mode around forward computation.

For multiple lists, use paths such as
`block_paths=("transformer_blocks", "single_transformer_blocks")`. The same
block settings are accepted by `ModelOffloader.from_module()` when you manage
activation directly.

Blocks within a group must share parameter and buffer names and trainability
structure. Ordinary streaming supports differing
shapes, dtypes, quantization formats, alias topology, and buffer layouts within
that structure. Shared storage is preserved within host state, including tied
embeddings and output heads, or within one streamed block. Sharing between host
and streamed state, between streamed blocks, or between block groups is
unsupported. Use whole-model offloading when those ties must be preserved.
For custom grouping, see
[`BlockComponentStore`](../src/piper_offload/block_component.py).

The GPU must fit the resident state, block targets, activations, and temporary
workspace together. Ordinary streaming keeps an active block and a lookahead
target, overlapping computation with the next whole-block copy. Heterogeneous
groups may also retain reusable targets for distinct layouts, so two targets
are not a universal memory bound. Host pinning is configured separately; see
[memory](memory.md).

With no block paths, activation uploads the whole model. CPU activation uses
the captured CPU state directly: no block target pool, streaming hooks, or
weight copies. It can still use block compilation as described below.

## Choose a block mode

`block_mode` applies to every ordinary and transient block group.

| Mode | CUDA behavior | When to use it |
|---|---|---|
| `"streaming"` (default) | Copies whole blocks through an active/lookahead pool. | General streaming, including heterogeneous groups and training. |
| `"resident"` | Loads all block targets at activation. | All weights fit, but per-block compilation or transient group release is useful. |
| `"rolling"` | Refills one shared target parameter by parameter during compiled computation. | Supported inference workloads needing a smaller weight working set. |
| `"auto"` | Selects rolling per compatible group with full-graph compilation; otherwise uses streaming. | Try rolling while allowing unsupported groups to stream normally. |

Resident mode creates no prefetch thread, private copy stream, or block
scheduling hooks. Streaming handles direction changes and traversal wraparound.
Rolling can refill in the foreground for skipped or out-of-order blocks.

### Compile block forwards

Compilation is optional and inference-only. Add one `BlockCompileConfig` to
the spec for all declared groups. Using the factory above:

```python
from piper_offload import BlockCompileConfig

spec = ModelSpec(
    key="compiled-example",
    factory=build_model,
    block_paths=("blocks",),
    block_compile=BlockCompileConfig(),
)
```

Pass this spec to `cache.use()` as above. These examples use distinct keys
because changing a spec does not replace an already registered cache entry.

Only each distinct block's `forward` is compiled. Module calls and streaming
pre-hooks remain eager, so weight loading finishes before compiled computation.
Original forwards are restored on deactivation; later activations can reuse
cached compiled callables. Without a compile configuration, execution is eager.
External whole-model `torch.compile(model)` or `model.compile()`, and compilation
outside declared block groups, are unsupported.

The backend is Inductor. Defaults are `dynamic=True` and `fullgraph=False`;
`options` accepts Inductor settings and compiler extensions, with a separate
copy of the mapping for each block. There is no `mode` argument. See
[`BlockCompileConfig`](../src/piper_offload/block_compile.py) for the controls.
CPU compilation operates on existing host weights and requires a working C++
toolchain; on Windows, use an x64 MSVC developer shell.

Merge-mode adapters work with compilation. An activation with any routed LoRA
runs all declared blocks eagerly because its child-module hooks stage factors
inside the block forward. Selecting routed mode without an adapter does not
disable compilation. See [adapters](adapters.md).

Compiler errors propagate. Piper does not retry a failed compiled invocation
eagerly: earlier graph segments may already have executed, and retrying could
repeat mutations. PyTorch's graph-break and recompilation behavior still
applies. Compilation with autograd is unsupported.

### Rolling requirements

Rolling requires full-graph compilation:

```python
spec = ModelSpec(
    key="rolling-example",
    factory=build_model,
    block_paths=("blocks",),
    block_mode="auto",  # use "rolling" to require it
    block_compile=BlockCompileConfig(fullgraph=True),
)
```

As a parameter's last compiled reader launches, a private stream can refill
its storage from the next block. That block waits for each parameter immediately
before its first reader. The shared block target, resident state, activations,
and workspace must still fit together.

Rolling is experimental. Groups require distinct block modules, identical
parameter names/order and layouts, frozen managed parameters, no tied managed
parameters, no streamed buffers, and no zero-sized parameter slots. Supported
representations include regular dense, TorchAO-family, Quanto, GGUF, and Piper
ConvRot INT8/NVFP4. Bitsandbytes, DTensor, and unreviewed external tensor
adapters use ordinary streaming in auto mode; explicit rolling rejects
unsupported groups. Activation-specific parameter overrides must also be
compatible. See [formats](formats.md) for format and merge restrictions.

Auto falls back for unsupported groups or activation load plans; it does not
recover from compiler failures. Routed-LoRA activations use eager streaming
instead of rolling. On CPU, both modes use ordinary compilation without CUDA
rolling.

The [rolling benchmark](../benchmarks/benchmark_rolling_compile.py) compares
latency, CUDA memory, and output equivalence against compiled whole-block
streaming. Validate your model after changing PyTorch or compiler extensions.
Implementation pointers are in [development](development.md).

## Release working sets during a forward

For inference, transient paths release state before the rest of the model
finishes:

```python
spec = ModelSpec(
    key="transient-example",
    factory=build_model,
    transient_block_paths=("blocks",),
)
```

For models with large non-streamed modules, use
`transient_paths=("input_embedder", "output_head")` with paths matching those
modules. A `transient_paths` module owns its non-streamed state recursively and
releases it after its own successful forward. A `transient_block_paths` group releases
its working set after its final block. The selected block mode still applies.
Ordinary `block_paths` retain their working sets throughout the activation;
the same path cannot appear in both block-path lists.

A successful root-model forward reacquires released components for the next
invocation. A skipped component stays acquired. Transient streaming and
rolling stop at the last block rather than preloading block zero before release.

Choose boundaries after which that state is no longer used in the current
root forward. A transient block group cannot be traversed again later in that
call, and its module objects must be distinct. Paths must own disjoint state,
including shared-storage aliases. `ModuleList` and `ModuleDict` entries in
`transient_paths` are not expanded into child boundaries: the selected module's
own forward is the release point. Piper does not infer repeated calls,
functional parameter access, cross-component aliases, or autograd lifetimes.
These are caller contracts. CPU activation installs no transient scheduling.

## Train streamed blocks

Use eager execution for training. **Every streamed block participating in
training must use activation checkpointing.** Enable your model's checkpointing
support or wrap each block call with `torch.utils.checkpoint.checkpoint`.
Verify that it covers every selected block; Piper does not detect omissions.

Streaming reuses GPU weight storage. Without checkpointing, backward can find
overwritten weights and raise a version-counter error. With streamed trainable
parameters, the `.data` swap can instead silently produce incorrect gradients.
Checkpointing reconstructs each block's graph when its weights are resident
again for backward.

By default, trainable parameters remain GPU-resident for the activation even
when their frozen block weights stream. Set `requires_grad` before capture.
To stream in-block trainable weight data too, pass
`include_block_trainables=True`.

```python
from piper_offload import ModelOffloader

# model is a fresh training model with trainable parameters already selected.
# Its checkpointing support must cover all selected blocks.
model.gradient_checkpointing_enable()
model.train()
offload = ModelOffloader.from_module(
    model,
    block_paths=["transformer_blocks"],
    include_block_trainables=True,
)
optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)

offload.activate("cuda")
try:
    for batch in loader:  # batch tensors are on CUDA
        loss = offload.value(**batch).loss
        loss.backward()
        with offload.optimizer_step():
            optimizer.step()
        optimizer.zero_grad()
finally:
    offload.deactivate()
```

Wrap CUDA optimizer updates in `optimizer_step()` whether trainables are
streamed or resident. It gathers streamed trainable data onto GPU for the
update and copies updated bytes back to host storage afterward. Those gathered
weights, gradients, and optimizer state must fit; streaming weights does not
stream gradients or optimizer state. Gradient clipping, AMP unscaling, and
`zero_grad()` need not be inside the context.

For an optimizer that runs on CPU, deactivate after backward, then call
`optimizer.step()` outside the context. Deactivation returns managed trainable
data and gradients to CPU, so optimizer state can remain there. Use fp32
trainables for master-weight updates. See
[`ModelOffloader.optimizer_step`](../src/piper_offload/model_offloader.py)
for this lifecycle. During CPU activation, the context does no data movement.
