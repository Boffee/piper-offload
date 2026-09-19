# Linux mapped-checkpoint rotation (#117)

This is the historical release-on-deactivation experiment preserved in
[the reference snapshot](../REFERENCE.md). Raw reports are archived under
[`results/117-linux-release`](results/117-linux-release).

Follow-up: [measurements on the actual H3 INT8 checkpoint](h3_pinning_results.md)
confirm the registration cost and demonstrate a faster experimental preparation
path. The figures below are the original synthetic-model measurements.

Measured on 2026-09-18 with an RTX 5090, Linux 7.0.0-31, PyTorch
2.14.0+cu130, and safetensors 0.8.0. Baseline: `f284a4f`.

Two independent 256 MiB models stay loaded through the existing
`safetensors.safe_open` loader. Each has four float32 linear blocks. The pin
budget is 1 GiB, so it can accommodate both models. After zero-budget compiler
and GPU warmup, the benchmark alternates the models three times, verifies
outputs exactly against each model's zero-budget GPU reference, and samples
`/proc/self/smaps` plus process RSS. No extra leases cover these measurements;
the regression tests separately check overlapping leases.

```sh
.venv/bin/python benchmarks/benchmark_mapped_rotation.py \
  --width 4096 --blocks 4 --repeats 3 \
  --modes streaming rolling resident host --check-release \
  --output /tmp/piper-117-rotation.json
```

For the baseline comparison, run the same script with the baseline source on
`PYTHONPATH` and omit `--check-release`.

| Mode | Median activate, before → after | Median deactivate, before → after | Maximum retained pins, before → after |
| --- | ---: | ---: | ---: |
| Streaming | 1.88 → 41.83 ms | 1.19 → 9.90 ms | 512 MiB → 0 |
| Rolling | 1.69 → 42.87 ms | 1.19 → 10.92 ms | 512 MiB → 0 |
| Resident blocks | 9.38 → 53.01 ms | 0.10 → 0.12 ms | 0 → 0 |
| Whole-model host component | 9.14 → 52.58 ms | 0.08 → 0.09 ms | 0 → 0 |

All outputs matched exactly. After release, the two checkpoints retained at
most **40 KiB of private pages** with the change, comprising partial boundary
pages that cannot be safely discarded while neighboring storage may be
registered. Baseline streaming and rolling retained about **512 MiB** of private
checkpoint pages. Baseline resident and host execution did not register their
uploads and retained no private checkpoint pages. With the change, their upload
leases close during activation, so pins are already zero before forward.

Peak sampled process RSS after release was 1,361.6 MiB for streaming and
1,386.7–1,386.8 MiB for the other modes, compared with 1,619.8 and
1,643.7–1,643.8 MiB respectively in the baseline run. RSS also contains runtime,
allocator, and reclaimable file pages; it is not a count of pinned memory.

Another CUDA workload was active during both runs. These timings quantify the
local cost of repeated registration and discard, not isolated throughput.
The release policy deliberately trades cached-registration speed for
reclaimability. Resident and host uploads also pay registration costs they
previously avoided. Larger production-model switching measurements and ROCm
verification remain in the parent issue #112.
