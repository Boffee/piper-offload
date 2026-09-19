# Reference snapshot: Linux release-on-deactivation experiment

This branch preserves the original #117 implementation and the measurements
that motivated revising #112. It is a comparison point, not the implementation
of the revised cross-platform pinned-copy cache.

- Branch: `reference/117-linux-release-pinning`
- Original baseline: `f284a4f`
- Implementation snapshot: `f8e1aaa`
- Measurements: 2026-09-18, Linux / CUDA / RTX 5090

## Preserved implementation

Linux file-backed registrations unregister and discard private interior pages
when their last lease closes. Anonymous registrations retain the idle LRU.
Resident uploads, host-component uploads, and optimizer copy-back hold host
leases through transfer completion, including failure cleanup. The default pin
budget is half of physical RAM, rounded down to OS pages.

The transfer protection, storage enumeration, failure handling, and budget work
remain useful for the revised design. Mandatory release and discard at the last
lease close is the policy being replaced; do not merge this snapshot wholesale
as the retained-cache implementation.

## Evidence and reproduction

- [Two-model rotation](benchmarks/mapped_rotation_results.md): synthetic models,
  exact output comparison, retained pins, and mapped private memory.
- [H3 checkpoint diagnosis](benchmarks/h3_pinning_results.md): actual Engine
  activation, registration, release, full-byte transfer timing, and experimental
  parallel private-page preparation.
- [Raw benchmark samples](benchmarks/results/117-linux-release): saved JSON
  reports from the original runs, including verification runs.
- [Validation records](benchmarks/results/117-linux-release/validation): prior
  CPU and CUDA test logs and baseline/reference type-check diagnostics.

The reports contain commands for the checked-in benchmark CLIs. The Engine
activation samples were collected using a separate diagnostic harness against
the existing MiniMax H3 loader; that harness is not part of the benchmark CLI.
No complete denoising or video-generation benchmark was performed. Checkpoint
transfer verification samples the first and last byte of every transfer chunk.
Parallel page preparation is benchmark-only and bypasses production admission
concerns; it is not integrated into the pin manager.

Validation of the implementation before this snapshot: **910 passed, 667
skipped** with CUDA hidden; **709 passed** in the selected CUDA regression
suite. Ruff on all changed Python files and `git diff --check` passed again
when preparing this branch. Pyright still reports **12 errors**, identical to
the baseline diagnostics after normalizing checkout paths; there are no new
type-check diagnostics. No Windows or ROCm hardware validation was performed.

## Revised direction tracked in the issues

Use read-only resting mappings on both platforms, with owned copies filled by
positional reads (`pread` on Linux, `ReadFile` on Windows). Retain populated
registrations after deactivation under a finite budget. Evict only idle copies,
unregistering and freeing their allocations, with explicit trimming before
known large pageable allocations. Transfer leases protect active copies.

Windows offer/reclaim becomes an optional later tier, not a prerequisite for
the initial implementation. The shared copy path has not yet been benchmarked;
these results establish the cost of repeated registration and discard, not
that positional reads outperform retained mmap registrations.

Issue ownership: #112 is the parent; #117 covers transfer leasing, budgeting,
and trimming; #118 covers the reader and Engine migration; #119 covers retained
copies and transfer-source selection on both operating systems.
