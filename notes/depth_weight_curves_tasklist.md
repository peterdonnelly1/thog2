# THOG depth-weight curves + observational probes task list

Base branch: `chaos_bump_sampling`
Implementation branch: `depth_weight_curves_and_observational_probes`
Base commit: `33f89809354cf9075b71498c0549dfb8d4b4ad9f`
Validated code commit: `87266d942b3eb62207f58de9bc723a2c31f43a98`

- [x] Add DEBUG>2 continuous depth-weight W&B charts under group `depth`.
  - [x] Track three model regions: one attention-Q head, MLP up, MLP down.
  - [x] Configurable scalar weights per matrix.
  - [x] Deterministic coordinates fixed for a run.
  - [x] Optional same coordinates across runs.
  - [x] Configurable latest vs accumulate time mode.
  - [x] Configurable bounded history length.
  - [x] Configurable chart logging interval.
  - [x] Evaluate at dense arbitrary depth coordinates, not only active layers.
  - [x] Bound accumulated W&B rows by dropping oldest complete snapshots first.
- [x] Existing `PLASTIC sampled coefficients by capacity layer sample number` only when DEBUG>9.
  - [x] Below DEBUG 10, skip coefficient sampling work rather than merely hiding the chart.
- [x] Permit layer-count probes in fixed THOG DEPTH runs even when PLASTIC/growth/count-learning are disabled.
  - [x] Explicit cadence opt-in; default cadence remains `None`, so ordinary runs pay no probe cost.
  - [x] Probes remain observational: no layer-count commit when learning disabled.
  - [x] Probe radius / token count / cadence respected and checkpoint-persistent.
  - [x] Candidate set is full radius `L-R .. L+R`, lower bounded at one layer.
  - [x] Growth-side candidates execute real additional logical layers from the continuous DEPTH field.
  - [x] Fixed-run probe vector is visible in both W&B and the ordinary console row.
  - [x] Preserve existing learned-count behavior.
- [x] Darken green used for right-side negative probe delta loss only.
- [x] Add regression tests for config, deterministic selection, arbitrary-depth curves, W&B row bounds, DEBUG gates, observational probes, console visibility, persistence and colour.
- [x] Run targeted tests.
  - [x] Final focused CPU gate: `71 passed, 9 subtests passed`.
- [x] Run broader regression suite.
  - [x] Exact-head versus `chaos_bump_sampling` comparison across 205 CPU test files.
  - [x] Result: `head_passed=173`, `head_nonpass=32`, `new_regressions=0`.
- [x] Review changed source files against THOG comment-marker rules.
- [x] Finalize progress log and report branch/commit/run guidance.

Scope note: observational growth beyond the configured fixed layer count is implemented and regression-tested for the public Chebyshev DEPTH trajectory. It is not a claim that arbitrary non-DEPTH compact topologies can synthesize additional layers with the same executor.
