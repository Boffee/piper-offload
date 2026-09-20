# Owned checkpoint copies on H3

Measurements for [#112](https://github.com/Boffee/piper-offload/issues/112): what the
owned-copy pin path costs on a real checkpoint, and how that compares with the
release-at-last-lease design archived in
[the reference branch](https://github.com/Boffee/piper-offload/blob/2199d7e9e2426e01bff12c8ff5812524906ebe6c/benchmarks/h3_pinning_results.md).

## Conditions

| | |
| --- | --- |
| Date | 2026-09-20 |
| Release | v0.10.0rc4 |
| CPU / RAM | AMD Ryzen Threadripper 9960X, 48 logical cores, 125 GiB |
| GPU | RTX 5090, 32 GiB, driver 615.71.09, no other compute processes |
| Storage | NVMe, ext4 |
| Checkpoint | `minimax_h3_fl2va_refpdd100_vsa025_fused_allint8.safetensors`, 32.8 GB on disk |
| Selected | 1170 tensors, 32.02 GiB, excluding what the Engine's H3 loader drops |
| Pin budget | 62.6 GiB (the default, half of RAM) except where stated |
| Load average | ~1.0 at start, no competing GPU work |

**No measurement here includes model computation.** Nothing is built, compiled or
denoised; these are host pinning and host-to-device transfer only. Times are a
single run, so treat sub-100 ms differences as noise.

## Phases

From `benchmark_checkpoint_copies.py`. `fill` is reading the file into the owned
copies, `register` is the native `cudaHostRegister` calls, and `disk` is the
kernel's own read counter for this process.

| Phase | Wall | fill | register | unregister | native calls | Disk read | RSS | Pinned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| First pin, cold file | 7.813 s | 6.762 s | 1.038 s | — | 1170 | 34.39 GB | 32.9 GiB | 32.0 GiB |
| Pinned transfer of every byte | 0.674 s | — | — | — | 0 | 0.06 GB | 33.0 GiB | 32.0 GiB |
| **Cached reacquire** | **0.003 s** | — | — | — | **0** | **0.00 GB** | 33.0 GiB | 32.0 GiB |
| Evict (`clear()`) | 2.085 s | — | — | 0.484 s | 0 | 0.00 GB | **1.0 GiB** | 0.0 GiB |
| Refill, partly cached | 3.295 s | 2.343 s | 0.939 s | — | 1170 | 7.96 GB | 33.0 GiB | 32.0 GiB |
| Pageable transfer of every byte | 4.463 s | — | — | — | 0 | 6.73 GB | 32.9 GiB | 0.0 GiB |

## Refill under a known page-cache state

The phase benchmark's refill runs on whatever the previous phase left cached, so
it reads some disk and is neither warm nor cold. `benchmark_checkpoint_rotation.py`
bounds it from both sides.

| Phase | Wall | Disk read | Pinned | Owned copies | RSS |
| --- | --- | --- | --- | --- | --- |
| Refill, file fully resident | **1.712 s** | 0.00 GB | 32.0 GiB | 32.0 GiB | 32.9 GiB |
| Refill, file fully evicted | 6.446 s | 34.39 GB | 32.0 GiB | 32.0 GiB | 32.9 GiB |

A fully warm refill is **1.71 s**; the same refill from disk is **6.45 s**. The
3.30 s in the phase table sits between them because 7.96 GB of the file had been
evicted by then.

## Rotation over a budget that holds one checkpoint

Two 32.8 GB H3 checkpoints, one manager, a 40 GiB budget, both files fully
resident in the page cache first. Admitting either must evict the other in LRU
order.

| Activation | Wall | Disk read | Pinned | RSS |
| --- | --- | --- | --- | --- |
| A, round 0 | 7.257 s | 34.38 GB | 32.0 GiB | 32.9 GiB |
| B, round 0 | 9.106 s | 33.15 GB | 40.0 GiB | 40.9 GiB |
| A, round 1 | 7.380 s | 25.39 GB | 40.0 GiB | 40.9 GiB |
| B, round 1 | 7.179 s | 23.88 GB | 39.8 GiB | 40.7 GiB |

Pinned bytes sit at the 40 GiB cap rather than dropping to one checkpoint:
admission evicts only as much idle LRU as it needs, so roughly 8 GiB of the
previous checkpoint survives each switch and its disk reads fall from 34.4 GB to
23.9 GB. Each switch still costs a near-full refill, 7–9 s, against 3 ms for a
cache hit that fits.

**A budget smaller than the set being rotated turns every switch into a refill.**
Size it to hold what is rotated between, or accept a first-pin cost per switch.
Note the reads here come from the page cache, not the disk — the kernel counts
them because the copies are filled with positional reads. A rotation whose files
do not fit in RAM would be slower again.

## Against the reference

| | Reference (release at last lease) | Owned copies (now) |
| --- | --- | --- |
| Repeated activation | 4.56 s | **0.003 s** while registered |
| Repeated activation, experimental parallel page prep | 1.61 s | 1.712 s after eviction, file warm |
| Releasing the model | ~1.05 s | 2.085 s for `clear()` of 32 GiB |

Repeated activation is the number this design set out to move, and retaining the
registration removes it: **4.56 s to 3 ms**, with no file read and no native
register call. When the copy has been evicted, refilling it from a warm page
cache costs 1.71 s, which matches the reference's experimental parallel
private-page preparation, so retention is the win here rather than a faster fill.

Eviction is slower than the reference's release (2.09 s against ~1.05 s) because
it does more: 1170 `cudaHostUnregister` calls, 0.48 s of them, and then actually
frees 32 GiB, which RSS confirms by dropping from 32.9 GiB to 1.0 GiB.

## What the acceptance checks show

- **Cache hits do not refill or re-register.** The cached reacquire made 0 native
  calls and read 0 bytes, in 3 ms.
- **Eviction releases real memory.** RSS falls from 32.9 GiB to 1.0 GiB.
- **Idle copies stay cached without pressure.** Under the default budget the
  reacquire found all 32 GiB still registered.
- **Pinned transfers are worth the pin.** 32.02 GiB moved in 0.674 s (47.5 GiB/s)
  pinned, against 4.463 s (7.2 GiB/s) pageable — 6.6x. Both verified the first
  and last byte of every 64 MiB chunk.
- **Nothing silently fell back.** The first pin reported 0 pageable bytes, so the
  whole selection was admitted.
- **Owned allocations are the whole pinned set.** `copy_bytes` equals
  `pinned_bytes` at 32.0 GiB: all of it is owned copies, none pinned in place.

## Not covered here

- **Windows commitment.** These are Linux `VmRSS` and the kernel's read counter.
  Windows working set and commit charge are not measured; CI covers Windows
  correctness, not its memory behaviour.
- **HIP.** CUDA only. #112's platform validation still wants a HIP run.
- **Model computation.** No forward pass, so none of these are end-to-end
  activation times.

## Reproducing

```
python benchmarks/benchmark_checkpoint_copies.py <checkpoint> --device cuda:0 --output phases.json
python benchmarks/benchmark_checkpoint_rotation.py <checkpoint-a> <checkpoint-b> --budget-gib 40 --output rotation.json
```
