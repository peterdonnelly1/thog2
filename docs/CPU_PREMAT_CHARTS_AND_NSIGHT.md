# CPU PREMAT charts and Nsight

PREMAT Recapitulation now captures every original-forward microstep at a sampled
optimiser update by default. Titles and history entries use one-based labels,
for example `Premat Recapitulation - Microstep 3 of Step 11`. CSV exports retain
the zero-based `micro_step` identity alongside `optimizer_update`.

Capture still follows `--log-interval`, including the first and final updates.
Use `--log-interval 1` to capture every update. The live store and browser history
retain the newest 128 microsteps. Existing databases migrate without losing
their stored snapshots; legacy snapshots represent microstep 1.

The optional `--premat_instra_microsteps first` restores first-microstep capture
at each sampled update. `all` is the default. Runner exposes the choice in
PREMAT and CPU Premat, and resume/fork can override it without changing model
compatibility.

## Runner controls and defaults

CPU Premat starts with the materialisation provider and the CPU-specific worker,
batch, staging, memory, deadline and shadow-validation controls. Shared PREMAT
targets and instrumentation remain available there too. Pure CPU controls have
moved out of PREMAT.

NSIGHT includes the Processing profiler, capture update, metric sampling
frequency, full-step timing enablement and full-step capture selector. Selecting
NSYS, NCU or a pair updates the visible effective values and supplies these
defaults when those fields are blank:

| Parameter | Effective value |
| --- | --- |
| `--premat_processing_logging` | `enabled` (required by the selected mode) |
| `--premat_processing_profiler` | `nsys` or `ncu` (required by each execution) |
| `--premat_processing_logging_capture_frequency_hz` | `10000` (NSYS sampling) |
| `--premat_processing_logging_capture_update` | `10`, bounded by the shortest run's final update |
| `--premat_instra__full_step_timing_capture_and_chart` | `enable` |
| `--premat_instra__full_step_timing_capture_and_chart_capture` | `step N`, matching the Processing capture update |
| `--ncu-probe-layer` | `0` for NCU |
| `--premat_instra` | `enabled` when PREMAT is enabled |
| `--premat_logging` | `enabled` when PREMAT is enabled |
| `--premat_retain_detailed_premat_history` | `true` when PREMAT is enabled |
| `--premat_instra_microsteps` | `all` when PREMAT is enabled |

A pair runs NSYS first, then NCU, with a common Runner pairing identity. The
Recipe shows `nsys` for the first execution and explains the second execution's
`ncu` setting. Explicit user settings are retained except for the Processing
logging and profiler values required by the chosen mode. Runner reports capture
selectors outside the run length and inconsistent full-step settings before
launch. CPU NSYS full-step and Processing capture updates must match.

CPU NSYS with full-step timing captures all forwards, backwards and the
optimiser at update N. Other Nsight Processing scopes capture the first forward
microstep. PREMAT Recapitulation has its own sampling cadence. NCU replays
kernels, so NCU elapsed times are unsuitable for measuring training speed.

## CPU NCU export and pairing

CPU materialisation has no GPU PREMAT materialisation kernels. NCU export
therefore requires semantically identified MAIN consume kernels and retains
their measured launch resources. GPU MAIN/PREMAT co-residency is explicitly
`N/A: CPU materialisation`; the exporter does not invent GPU PREMAT evidence.
GPU materialisation still requires both MAIN and PREMAT launches.

Instra accepts that MAIN-only CPU evidence for pairing, preferring Runner's
pair identity and retaining host/grid scope. Legacy runs still use encoded
configuration matching. The paired NSYS view includes MAIN resources, pair
provenance and the NCU files in paired downloads. Header-only or missing NCU
resource evidence remains an invalid companion. The NSIGHT table column shows
`NSYS`, `NCU` or `-` immediately before Updated, including saved-layout migration.

Processing operations expand to include all CPU workers and both D2H and H2D
lanes. Resource maximisation survives same-tab refresh and chart redraw.
Browsers without WebGL use SVG resource traces.

## Verification after updating

Restart Instra after pulling this branch; the asset version is changed so the
updated scripts and styles load. Reopen an existing CPU NSYS/NCU pair, then
check its paired eye, MAIN resources, N/A compatibility and downloads. For a
fresh pair, confirm the visible capture defaults in NSIGHT before launching.

Automated checks cover SQLite migration and microstep cursors, asynchronous
sampling, resume overrides, Runner expansion and CLI parsing, CPU and GPU export
contracts, forward-only pairing, paired archives, existing dashboard behavior,
and real Chromium/Firefox interactions and refresh soaks. An actual Nsight run
on CUDA hardware is still the final hardware smoke check.

The focused 34-module pass recorded 613 passes and 201 skips. One legacy CLI
alias compatibility test was deselected after its unchanged baseline failure
was confirmed. The final storage/migration pass recorded 64 passes, and all
eight JavaScript regression suites passed. Chromium and Firefox each passed
the Runner controls, microstep history/export, and CPU chart flows, including
a 60-second refresh soak with unchanged DOM size.

The broader non-distributed pass recorded 2,225 passes, 220 skips and 87 reported
failures. Rerunning those failure cases on the original branch and this change
produced the same failure set. Distributed subprocess checks timed out in this
environment, including an explicit loopback probe; CUDA/Nsight hardware smoke
testing remains outstanding.
