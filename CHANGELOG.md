# Changelog

All notable changes to Piper Offload are documented here. Versions follow the policy in
[VERSIONING.md](VERSIONING.md).

## [Unreleased]

### Added

- Add opt-out parameter-value strength scaling. Set
  `scale_parameter_values=False` on `Adapter.from_state_dict()` or
  `AdapterSpec` to materialize exact-name values unchanged for every active
  adapter strength, or set `scale_with_strength=False` on an individual
  `ParameterValue`. Default scaling and zero-strength inactivity are
  unchanged.

## [0.9.0rc1] - 2026-09-02

### Added

- Add the adapter-owned `merge_dense_` capability alongside `merge_lora_`.
  Dense-only and mixed dense + LoRA parameter deltas now stage one full-rank
  logical update and delegate it through this capability, allowing supported
  quantized targets to requantize exactly once. Quanto, bitsandbytes 4-bit and
  8-bit, TorchAO scaled/static FP8, INT8, MX, NVFP4, Piper ConvRot INT8/NVFP4,
  plain tensors, and their supported DTensor compositions opt in; GGUF and
  TorchAO INT4 tile-packed remain unsupported.
  Target-only and staged-update validation are separate capabilities, and
  composing adapters can expose one shared locality contract for both merge
  representations.
- Add dense specializations to every built-in Triton quantized LoRA merge
  kernel. Standard CUDA layouts now use the same dequantize/update/requantize
  pipeline for factorized and full-rank updates, with reference fallbacks for
  supported nonstandard layouts and stochastic nested bitsandbytes 4-bit
  scales.
- Delegate ConvRot INT8 and NVFP4 dense and mixed updates to Piper Kernels'
  public in-place `add_`, including stochastic-rounding seeds, and require
  `piper-kernels[convrot]>=0.7.0rc1` for the optional integration.

## [0.8.0] - 2026-09-02

### Added

- Add `ParameterDelta` and `ParameterDeltaTransform` for combined low-rank and
  full-rank additive updates to existing plain floating-point parameters.
  `Adapter.from_state_dict()` recognizes the exact `.delta.weight` and
  `.delta.bias` suffixes; a dense weight delta composes with that parameter's
  LoRA factors under one adapter strength.

### Changed

- Change `AdapterTarget` to `ParameterDelta | ParameterValue`. LoRA factors
  are now the optional low-rank representation within a parameter delta;
  exact-name parameter values retain their meta-only replacement semantics.
  Routed mode remains factor-only, and dense deltas for quantized targets are
  intentionally deferred.
- Simplify parameter-value conversion to use ordinary target-dtype rounding,
  including underflow to zero, and narrow parameter values to conventional
  floating-point tensors rather than plain float8 storage.
- Reject non-finite strengths when scaled adapter representations are created,
  so merge and routed execution enforce the same invariant.
- Reject dense parameter deltas for plain float8 targets during preflight;
  full-rank updates currently require conventional floating-point storage.
- Accumulate combined LoRA and dense terms into one update tensor before
  mutating a low-precision base parameter, avoiding an intermediate base
  rounding step.

### Removed

- Remove legacy `lora_B.bias` storage and execution. Represent an additive
  bias update with the canonical `.delta.bias` suffix; routed mode remains
  factor-only.

## [0.7.0] - 2026-09-02

### Added

- Treat every non-factor entry passed to `Adapter.from_state_dict()` as an
  exact-name `ParameterValue`. These merge-only values populate storage-free
  model parameters represented by frozen plain floating-point meta tensors.
  Values may be scaled during materialization and work across
  resident, streaming, rolling, and automatic block modes and permanent
  merges. Targets must use non-overlapping dense strided layouts with zero
  storage offset so validation can reject unsupported materialization before
  mutating the model.

### Changed

- Separate low-rank and meta-parameter execution into `LoRATransform` and
  `ParameterValueTransform`, both following the shared `ParameterTransform`
  protocol.
- Generalize the cached mixed update resource as `Adapter`, with one immutable
  `AdapterTarget` union per exact parameter name. Each target is directly a
  `LoRAFactor` or `ParameterValue`; the two are mutually exclusive for one
  target. Model-cache and activation APIs use the corresponding generalized
  adapter terminology.

### Removed

- Remove the resource-level `LoRA`, `LoRAMode`, `LoRASpec`, and `merge_lora`
  APIs. Use `Adapter`, `AdapterMode`, `AdapterSpec`, and `merge_adapter`
  instead.

## [0.6.1] - 2026-09-01

### Added

- Add `block_mode="auto"` to select rolling independently for supported block
  groups and fall back to whole-block streaming for other groups.
- Add dedicated `ConvRotNVFP4Tensor` pinning, movement, rolling, identity, and
  LoRA integration. Reconstruction preserves the rotation group and all
  canonical NVFP4 metadata; merge-mode LoRA delegates factors and the optional
  stochastic-rounding seed to Piper Kernels' in-place `addmm_` operation.

### Changed

- Allow the ConvRot optional backend to use Piper Kernels 0.6 while retaining
  compatibility with the 0.3 through 0.5 series. ConvRot NVFP4 movement is
  available with 0.6.0; merge-mode LoRA requires the kernel-owned `addmm_`
  introduced in 0.6.1 and otherwise fails closed with an upgrade message.

## [0.6.0] - 2026-08-31

### Changed

- Rename `StreamedComponent` to `BlockComponent` and standardize the resident,
  streaming, and rolling block runtime names.
- Replace `stream_blocks` and `BlockCompileConfig.rolling` with a single
  `block_mode` setting for resident, whole-block streaming, or rolling
  execution. The selected mode applies to ordinary and transient block paths;
  transience controls working-set lifetime only.
- Rename `stream_trainable_weights` to `include_block_trainables` so the option
  describes block ownership independently of residency mode.

## [0.5.1] - 2026-08-31

### Added

- Add opt-in partial LoRA target application. A LoRA built with
  `allow_partial_targets=True` applies the intersection of its targets and a
  model's parameters in activation-scoped merge, routed, and permanent merge
  paths; strict target validation remains the default.

### Changed

- Allow the ConvRot optional backend to use Piper Kernels 0.5 while retaining
  compatibility with the 0.3 and 0.4 series.

### Fixed

- Preserve compatible concrete NVFP4 subclasses across pinned-host and device
  reconstruction, layout identity, requantization, and streamed LoRA merging.

## [0.5.0] - 2026-08-30

### Added

- Add explicit `BlockComponent.acquire()` and `release()` operations that
  can cycle a block streamer's CUDA working set without ending its activation
  session. Activation still acquires immediately, preserving existing model
  behavior.
- Add the same acquire/release lifecycle to `PinnedComponent`, allowing its
  bulk CUDA working set to cycle independently from its activation session.
- Add `ModelOffloader.register_forward_hook()` for caller-owned native PyTorch
  hooks addressed by fully-qualified module name.
- Add `transient_block_paths` to `ModelOffloader` for streamed CUDA pools that
  release after their final blocks and reacquire after the root model forward
  without runtime-specific coordination or a redundant block-0 wraparound
  refill. Ordinary `block_paths` pools remain resident.
- Add `transient_paths` for named modules whose independent CUDA working sets
  release after their forwards and reacquire after the root model forward.

### Changed

- Allow `ModelOffloader`, `ModelCache`, and `merge_lora()` to apply the same
  LoRA more than once. Each occurrence contributes its supplied strength.

### Removed

- Remove the `prefix_paths` and `suffix_paths` model selectors and their
  boundary-scoped CUDA runtime. Non-streamed model state is resident for the
  activation again.

## [0.4.1] - 2026-08-29

### Changed

- Allow the ConvRot optional backend to use Piper Kernels 0.4 while retaining
  compatibility with the 0.3 series.

## [0.4.0] - 2026-08-29

### Added

- Allow callers to select `prefix_paths` and `suffix_paths` module paths whose
  frozen non-block state is loaded only before or after the central streamed
  block span. Successful forwards asynchronously prefetch the next prefix
  after the model-done event, reducing peak CUDA residency and overlapping
  inter-step work while preserving pinned/adopted backing, quantized adapters,
  LoRA merge hooks, and compiled block execution.

### Changed

- Rename the public streamed-model selector from `blocks_attr` to
  `block_paths`; the new boundary selectors use the matching `prefix_paths`
  and `suffix_paths` names.

## [0.3.0] - 2026-08-28

### Added

- Allow `ResourceCache` and `ModelCache` budgets to be resized at runtime via
  `resize()` or `max_cache_bytes` assignment, with policy-driven eviction and
  atomic failure when leased entries prevent a shrink.
- Add experimental `BlockCompileConfig(rolling=True)` inference for homogeneous
  dense, TorchAO-family, Quanto, GGUF, and Piper ConvRot INT8 blocks. Inductor
  inserts per-parameter refills after final graph use and waits at first use so
  resident and prefetched state share one GPU target without a whole-block
  readiness stall; the mode composes after Piper Kernels graph passes, models
  lifecycle callbacks as non-mutating ordered effects with late scheduler-only
  dependencies to preserve compute kernel autotuning without forced reader
  materialization, and preserves supported merge-mode LoRA hooks.

### Changed

- Make streamed scheduling strategy-owned: ordinary streaming uses one active
  block plus one asynchronous lookahead target, while rolling compilation uses
  one shared parameter target. Remove the public `StreamConfig` residency,
  prefetch-depth, and cyclic-traversal knobs; traversal wraparound is handled
  internally.

## [0.2.5] - 2026-08-17

### Added

- Allow `BlockCompileConfig` to forward a copied Inductor `options` mapping to
  every streamed block's `torch.compile` call, including Piper Kernels'
  `convrot_int8_compile_options()` graph-pass configuration.

### Changed

- Update Piper ConvRot INT8 integration for Piper Kernels 0.3: use the new
  `piper_kernels.linear.convrot` package and `from_quantized` factory.

## [0.2.4] - 2026-08-14

### Added

- Support optional legacy PEFT `<module>.lora_B.bias` vectors in cached LoRA
  resources, activation-scoped resident and block-streamed merge, permanent
  merge, and routed residuals. Modern A/B-only LoRAs retain their existing
  path; merge requires an existing plain dense base bias, while routed mode
  remains valid for bias-less `nn.Linear` targets.

### Changed

- Use stochastic rounding by default when merging LoRA updates into quantized
  weights so sub-step updates are not systematically rounded away. Dense
  merges remain exact, routed LoRA is unaffected, and callers can pass
  `stochastic_rounding=False` for deterministic requantization.

## [0.2.3] - 2026-08-13

### Fixed

- Reduce peak host memory while pinning streamed structured-tensor models by
  retaining only lightweight pre-validation metadata, releasing replaced
  source wrappers block by block, and allocating final pinned destinations
  directly without intermediate pageable clones.

## [0.2.2] - 2026-08-10

### Added

- Add a composable `triton` acceleration extra that selects `triton-windows`
  on Windows and upstream `triton` on Linux. Individual quantization extras and
  Piper ConvRot remain portable without it, while `all` includes the accelerated
  backends. Exercise the Windows selection and portable test suite on Windows CI.

### Fixed

- Resolve TorchAO from its portable PyPI wheel on Windows, where PyTorch's CUDA
  13.0 index does not publish a compatible TorchAO wheel.

## [0.2.1] - 2026-08-10

### Added

- Forward Piper Offload's reproducible per-target stochastic-rounding seed to
  Piper ConvRot INT8 `addmm_` LoRA merges.

## [0.2.0] - 2026-08-09

### Added

- Add boolean opt-in stochastic rounding for quantized LoRA merges across Quanto, bitsandbytes,
  TorchAO FP8/INT8/MX/NVFP4, and DTensor-composed weights. Automatic per-target, per-merge seed
  selection remains internal to LoRA merge; deterministic rounding remains the default and Piper ConvRot
  INT8 reports stochastic merge as unsupported. Each quantized adapter composes both modes
  through its existing requantization method: deterministic representation construction followed
  by optional stochastic terminal-code recoding. Standard CUDA layouts perform that terminal
  selection in their existing Triton kernels; nested bitsandbytes 4-bit scales retain the reference path.
- Add public `derive_seed(*parts)` as the canonical stable unsigned 64-bit seed derivation
  utility for Piper Offload and downstream adapters.
- Treat exact-zero LoRA strengths as inactive before target lookup, factor staging, hook
  installation, or cache leasing.

### Fixed

- Recompute data-dependent Quanto qint8/qfloat8 weight scales after LoRA merges in both
  generic and CUDA paths, including safe exact-zero blocks.
- Merge TorchAO INT8 LoRA factors in the stored SmoothQuant/AWQ weight coordinates while
  preserving activation pre-scales and calibration metadata.
- Repair exact-zero TorchAO FP8 PerGroup blocks in the generic requantization path.
- Preflight adapter-specific quantized merge constraints before mutating any model weight.

## [0.1.0] - 2026-08-03

### Added

- Policy-driven resource and model caches with lease-aware LRU eviction and reusable model,
  LoRA, and general-object stores.
- Whole-model pinned-host offloading and asynchronous block streaming with configurable
  residency, prefetching, optional block compilation, and trainable-weight support.
- Activation-scoped merged and routed LoRA application, including format-specific CUDA merge
  kernels and tensor-parallel DTensor composition.
- Tensor adapters for dense, bitsandbytes, GGUF, optimum-quanto, Piper ConvRot INT8, and TorchAO
  scaled-FP8, INT8, MX, NVFP4, and INT4 tile-packed weights.
- Public tensor-adapter registration for downstream tensor subclasses.
- Python 3.14, PyTorch 2.13, TorchAO 0.18, Apache-2.0 licensing, and the Piper Offload package
  identity.

[Unreleased]: https://github.com/Boffee/piper-offload/compare/v0.9.0rc1...HEAD
[0.9.0rc1]: https://github.com/Boffee/piper-offload/compare/v0.8.0...v0.9.0rc1
[0.8.0]: https://github.com/Boffee/piper-offload/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Boffee/piper-offload/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/Boffee/piper-offload/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/Boffee/piper-offload/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/Boffee/piper-offload/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Boffee/piper-offload/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/Boffee/piper-offload/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Boffee/piper-offload/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Boffee/piper-offload/compare/v0.2.5...v0.3.0
[0.2.5]: https://github.com/Boffee/piper-offload/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/Boffee/piper-offload/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/Boffee/piper-offload/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/Boffee/piper-offload/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Boffee/piper-offload/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Boffee/piper-offload/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Boffee/piper-offload/releases/tag/v0.1.0
