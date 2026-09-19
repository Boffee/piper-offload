# H3 checkpoint pinning diagnosis (#117)

This is the historical release-on-deactivation experiment preserved in
[the reference snapshot](../REFERENCE.md). Raw reports are archived under
[`results/117-linux-release`](results/117-linux-release).

Measured 2026-09-18 on the Ryzen Threadripper 9960X / RTX 5090 workstation,
Linux 7.0.0-31, NVIDIA driver 615.71.09, PyTorch 2.14.0+cu130. The other CUDA
worker was idle during these measurements.

Checkpoint: `minimax_h3_fl2va_int8_convrot.safetensors`, 34,038,892,334 bytes.
Excluding `token_refiner.`, `condition_proj.`, and `context_embedder.` as Piper's
H3 transformer loader does leaves **32,442,261,598 bytes** (32.44 GB / 30.21 GiB),
mostly INT8 weights, with BF16 parameters and quantization metadata.

## Actual Piper activation

Constructed `MiniMaxH3Transformer` through its existing automatic loader and
`ModelSpec.build_store()`, with `block_mode="streaming"` and block compilation
disabled. Model loading/capture and CUDA runtime initialization were outside
the activation timer. This measures loading the actual GPU working set; it
does not time a denoising forward or video generation.

| Configuration | First activation in process | Subsequent activations | Release |
| --- | ---: | ---: | ---: |
| Zero pin budget | 0.036 s | 0.033, 0.033 s | 0.0035 s |
| Current #117 implementation | 8.192 s | **4.564, 4.572 s** | 1.043–1.055 s |
| Experimental parallel private-page preparation, 16 workers | 5.820 s | **1.620, 1.614 s** | 1.070–1.080 s |
| Same experiment, resetting read-populated mappings first | **1.823 s** | **1.612, 1.612 s** | 1.056–1.098 s |

The normal path spent 4.520–4.527 s in native registration on subsequent
activations, with 765 registration calls. The experimental path spent
0.844–0.854 s preparing private pages and 0.712–0.716 s registering them.
Both paths discarded file-backed private pages on release, leaving about
2.6 MiB of partial boundary pages. Anonymous storage retains its existing LRU.

The first activation was slower when preparation started from the loader's
read-populated mappings. An additional experiment first dropped those existing
page-table entries with `MADV_DONTNEED`, keeping the backing file in cache, then
prepared writable mappings in parallel. This removed most of the initial
penalty too. All these activation trials had warm file cache; reading an
uncached checkpoint still adds disk time. The zero-budget path initially uploads only the streaming
working set; its 33 ms activation does not include a complete model traversal.

## Where the time goes

Direct testing through the real pin manager covered all 1,016 selected raw
checkpoint tensors. Loading their mappings took 0.076 s. The first explicit
page read took 5.22 s and read 32.34 GB from disk; it finished **before** the
registration timings below. Subsequent registration trials performed effectively
no disk I/O (0–8 KiB in the repeated discard trials).

| Operation | Measured time |
| --- | ---: |
| Register warm file-backed pages after discarding private copies | 4.567–4.620 s |
| Native CUDA registration within that measurement | 4.556–4.609 s |
| Register again while retaining the private copies | 0.659–0.676 s |
| Transfer all selected bytes to a reused 64 MiB GPU buffer, pinned | 0.630–0.632 s |
| Same transfer, pageable | 1.094 s |
| Unregister and discard private pages | 1.180–1.203 s |

The current native registration path requests writable pages. For a private
checkpoint mapping, Linux must create private copies of roughly **7.9 million
4 KiB pages** before pinning them. `smaps` confirmed the creation of 32.44 GB of
anonymous private pages. Keeping those copies for the comparison removed most
of the registration cost, but also kept the 32.44 GB private allocation. The
manager's Python bookkeeping accounts for only about 10 ms.

The transfer measurements copy every selected byte and compare the first and
last byte of every transfer chunk after GPU synchronization. All samples
matched. This is a transfer check, not a full inference-output comparison.

## Experimental optimization

`MADV_POPULATE_WRITE` can prepare the same private mappings in parallel before
registration, without changing their bytes. The experiment merges page ranges,
uses 64 MiB chunks, then invokes the existing serial registration and release
path. Two trials per worker count gave these preparation-plus-registration
times on the real checkpoint:

| Workers | Total setup |
| --- | ---: |
| 1 | 6.238 s first trial, 4.897 s second trial |
| 4 | 2.300–2.315 s |
| 8 | 1.719–1.761 s |
| 16 | 1.594–1.641 s |

The actual model activation experiment above confirms the improvement survives
Piper's load plans and quantized tensor wrappers. This optimization is currently
**benchmark-only**. Production integration must prepare only admitted mappings
and retain the existing ownership and failure cleanup guarantees; blindly
preparing an entire model before budget admission would create unnecessary
private memory.

The mapping-reset experiment also requires exclusive control over the affected
pages: resetting an overlapping live registration could invalidate an in-flight
GPU transfer. It is safe in this diagnostic because no prior benchmark lease is
active; production integration must account for shared pages before doing it.

CUDA also exposes a read-only registration flag, gated by a device attribute.
This machine reported `cudaDevAttrHostRegisterReadOnlySupported = 0`, and a
small real mapped-tensor test with the flag returned `cudaErrorNotSupported`
(801). It therefore cannot remove the private-copy requirement here.
See [NVIDIA's host-registration documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html).

## Reproduce the checkpoint measurements

Requires NumPy and safetensors in the environment. Checkpoint contents are
unchanged. No system-wide file cache is flushed. The comparison temporarily
retains private pages only in the benchmark process and discards them afterward.

```sh
.venv/bin/python benchmarks/benchmark_checkpoint_pinning.py \
  ~/git/apps/ComfyUI/models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors \
  --exclude-prefix token_refiner. \
  --exclude-prefix condition_proj. \
  --exclude-prefix context_embedder. \
  --output /tmp/h3-pinning.json
```

For the experimental path, add `--populate-workers 16 --policies discard`.
Add `--reset-mapping` to test the removal of the initial read-populated mappings.
The report separates page reading, native registration, private-page preparation,
full-byte transfers, native unregistration, and discard/warm advice.
