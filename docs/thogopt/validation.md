# Validation record — 2026-09-02

Implementation: fixed public Chebyshev DEPTH thogopt. Baseline commit: `e36fe2330d849ccee93514fb5349a21d6b3bea21`. Branch: `instra_weight_inspector`.

## Verified

- 500 updates of synthetic gradient replay against PyTorch AdamW in FP64 and FP32, including small denominators.
- Independent SciPy constrained least-squares checks with normalized tiny and sharply varying targets.
- P < L projected-AdamW oracle, accumulation before squaring, physical norm clipping, zero-beta/zero-gradient/absent-gradient semantics, nonfinite rollback and exact restart.
- 100 complete model updates against AdamW in FP64 and FP32, and 100 FP32 SGD updates. The reference uses matching physical parameter groups: THOG excludes embeddings from decay, whereas the generic DENSE defaults include them.
- Both retained and fast-discard materialization modes, ordinary uncompressed parameters, checkpoint compatibility, and explicit reset forks.
- FP32 persistent moments for direct FP16 ordinary parameters, and CPU GradScaler growth/scale-state restart with an unchanged training trajectory.
- Actual public CLI fresh/resume/reset-fork execution, persisted sampled references, full matrix capture and shell argument forwarding.
- Captured values, constrained-projection update error, all six matrix families, matrix windows, exact stored precision, signed differences, retention cleanup and passive monitoring.
- 22 JavaScript regression suites, including the new jsdom event-level history UI checks. These cover group creation, run switching, latest requests, precision, signed difference, bounded matrix DOM and inspection actions in standard/maximized card states.

The numerical model fixture preserves FP64 normalization weights. Production FP32 and mixed-precision normalization behavior is unchanged. FP64 model comparison uses atol=1e-10, rtol=1e-9; FP32 model comparison uses atol=5e-6, rtol=5e-4. The synthetic optimizer replay uses FP32 atol=2e-6, rtol=2e-5 and FP64 atol=1e-10, rtol=1e-9. The independent normalized constrained-fit fixture uses 1e-9. These bounds are recorded in the tests.

## CPU benchmark

PyTorch 2.8.0+cpu; one CPU thread; L=8, P=3, D=16, batch 2, sequence 8, three warmup updates and 20 measured updates. Synthetic tokens. No telemetry capture. This compares overhead, not matched-loss convergence.

| Model / optimizer | Tokens/s | Total seconds | Optimizer seconds | All-parameter moment bytes |
|---|---:|---:|---:|---:|
| dense / adamw | 765.7 | 0.4179 | 0.0602 | 215,296 |
| thog2_sheet / adamw | 701.0 | 0.4565 | 0.0439 | 92,416 |
| thog2_sheet / thogopt | 474.9 | 0.6738 | 0.2743 | 116,992 |

thogopt stores 98,304 moment bytes for covered matrix families versus 196,608 for dense AdamW: **50% less**. The same run requires 98,304 bytes of raw-gradient host staging. Moment savings do not imply the same reduction in total training memory.
History fitting took 0.2073s, raw-gradient staging 0.0167s and preparation/clipping 0.0185s over the measured updates. Fitting is the dominant added optimizer cost here. CPU throughput is lower for thogopt in this small benchmark.

## Boundaries of this verification

- No GPU is available; CUDA/AMP correctness, NCCL, peak GPU memory and representative GPU throughput remain unverified.
- Two-rank CPU Gloo startup was attempted twice, including loopback. Both attempts failed before the worker tests at Gloo TCP device.cc:186 with “Operation not permitted”. The shipped DDP fixture remains available for a training machine.
- The browser binary download timed out. The DOM and JavaScript regression checks passed, but they do not establish rendered Firefox/Chromium layout or real-browser Plotly interaction.
- Long OpenWebText runs, production history-budget sweeps and convergence effectiveness have not been measured. No claim of improved model quality or production speed is made.
- The existing test_optimizer_wrapper.py has five failures on the untouched baseline and the implementation tree: missing legacy smoke script, outdated wrapper length/help assertions and outdated console spacing expectations. These unrelated tests were not rewritten.
- Runtime process RSS includes non-optimizer allocations; the benchmark labels it accordingly. History/read-side diagnostics use declared FP64 evaluation, which is not a bitwise replay of GPU rounding.

## Delivery and baseline tag

The preserved local tag is `pre_thogopt_20260902` at the baseline commit above. The GitHub connector exposes branch updates and Git-object creation but no tag-creation operation; publishing that tag remains blocked by the connector capability. No shell-based GitHub write was used. The final download stanza supplies the tag command for the user to run locally.

See the sibling Python and JavaScript result files for the final verification counts.
