# PLASTIC same-batch operator visibility implementation log

Parent implementation: `docs/PLASTIC_SAME_BATCH_ALL_PROBES_IMPLEMENTATION_LOG.md`
Branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`
PR: #33
Date: 2026-08-09

## Trigger

A real scruffy run at head `a9e95f571731e54895d838b06a7c2e87c1e5bb09` used the public `--plastic__layer_count__same_batch_all_probes` flag successfully, but neither startup summary stated that the mode was active and probe rows did not expose a fixed-batch identity. The observed provenance P1,P1-2,P1-3,P1-4 then P5 was consistent with strict non-overlapping windows but was not sufficient operator proof.

The user also clarified that, under same-batch mode, visible probe numbering should reset to P1 whenever a fresh evidence batch/window begins. A global P5/P6/... sequence has no useful operator meaning once the evidence batch changes.

## Governing display semantics

When same-batch mode is enabled:

1. Compact startup reports `same_batch=true` on the `plastic fine:` row.
2. Detailed startup reports `plastic__layer_count__same_batch_all_probes: true`.
3. Each probe row reports the actual same-batch window and batch identity as `same_batch W<window_id>:<ordinal>/<window_size> B=<digest8>`.
4. Operator-facing P sequence/provenance is window-local and restarts at P1 for every fresh batch.
5. A monotonically increasing global probe sequence remains available only in audit state as `probe_global_sequence` / `probe_global_provenance`; it is not the console provenance.
6. False/default mode retains the established display/runtime path and does not emit a same-batch per-probe marker.

## Implementation

### `sheet/plastic_depth_same_batch_visibility_patch.py`

New additive overlay. It:

- rewrites enabled-mode audit `probe_window_provenance` to local `1..ordinal`;
- preserves prior global provenance separately;
- maps console `plastic_probe_sequence` and `plastic_probe_provenance` to the local window ordinal/provenance;
- exposes window ID, ordinal, size and persisted batch digest to the progress formatter;
- appends the compact same-batch marker to probe rows;
- injects the resolved same-batch Boolean into the detailed PLASTIC startup block.

Initial commit: `b99186fad0ae5633b89bd3e96bf523d8f0ac4a9f`.

The overlay is imported after the v0.53 same-batch runtime overlay in `sheet/plastic_depth_console_postfix_patch.py` so operator display wins last without replacing selector behavior. Integration commit: `a67ed03ac82e5de182645019c6201da375f3a222`.

### Shell startup visibility

`plastic_depth_lookahead_wrapper_options.sh` now retains the explicitly requested same-batch Boolean in `THOG2_PLASTIC_LAYER_COUNT__SAME_BATCH_ALL_PROBES`, exports it, validates it, and still routes the actual public Python flag after the wrapper delimiter. Commit: `e655708fec7d4a951897ab79a7c2c361cd00d966`.

`train_OWT.sh` appends `same_batch=true|false` to the compact `plastic fine:` startup row. Commit: `11f4d8833502235f865e8e987bd984219eb06caf`.

### Tests

`tests/test_plastic_depth_same_batch_visibility.py` checks:

- a real same-batch W=2 runtime produces audit windows 1,1,2;
- local audit provenance is `(1,)`, `(1,2)`, `(1,)`;
- internal global sequence remains `1,2,3` with global provenance retained separately;
- rendered probe rows show P1/P2 then reset to P1 and expose matching W/batch markers;
- detailed startup prints the resolved same-batch Boolean;
- a public `train_OWT.sh` dry-run with the public flag prints compact `same_batch=true`.

The first version of this test incorrectly invoked `_prepare_console_progress_payload` on `SharedTrainer`. That method belongs to the Stage6 console layer. The classifier correctly identified this as the sole new branch failure. The test was corrected to use `SharedTrainer` only for actual window/audit runtime state and invoke the Stage6 formatter at its proper boundary. Production code was not changed to satisfy that test defect. Correction commit: `e40748ebe5cd0467f25cde4d916e3b7cee28111c`.

The visibility production module was added to the workflow's explicit `py_compile` list in `c26ab54e55de0419257fd2323cc56b919233d5a8`. The public compact-header regression was added in `d7838fab7f9a5522a8249a8cafd477087bf072a0`.

## Validation history

At visibility head `11f4d8833502235f865e8e987bd984219eb06caf`:

- exact disabled-equivalence workflow passed;
- same-batch-enabled two-rank DDP workflow passed;
- the broad branch suite had the recorded inherited failures plus one new failure in `test_same_batch_visible_probe_number_resets_with_fresh_batch`;
- root cause was the test-boundary mistake above, not production behavior.

Final validation after `e40748e` / `d7838fa` is pending at the time this log entry is written. Do not mark the visibility follow-on complete until the classifier reports zero new branch-only failures and the disabled/DDP gates remain green.

## External GPU gate

The corrected same-batch CUDA smoke remains the final real-GPU gate:

```bash
cd ~/git/thog2
git fetch origin PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git checkout PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git reset --hard origin/PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
bash tools/run_plastic_coarse_fine_gpu_smoke.sh
```
