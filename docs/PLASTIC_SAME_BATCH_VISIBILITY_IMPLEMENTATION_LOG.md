# PLASTIC same-batch operator visibility implementation log

Parent implementation: `docs/PLASTIC_SAME_BATCH_ALL_PROBES_IMPLEMENTATION_LOG.md`
Branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`
PR: #33
Date: 2026-08-09
Validated code head: `a55e2c5d53b9ce8540c426ab28a2b824e03bcd42`

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

The visibility test went through three fixture corrections. None required a production change:

1. It first called a Stage6 console method on `SharedTrainer`; corrected by separating real runtime/audit state from formatter testing (`e40748e`).
2. The tiny public-wrapper dry-run inherited the real 1024-token probe default despite an 8-token toy microbatch; corrected by explicitly requesting 8 sampled tokens (`5c6a882`).
3. The synthetic formatter row expected provenance text without supplying the directional-summary context that v0.541 requires before rendering provenance; the synthetic assertion was narrowed while local provenance remains directly asserted in the real audit (`a55e2c5`).

The visibility production module was added to the workflow's explicit `py_compile` list in `c26ab54e55de0419257fd2323cc56b919233d5a8`.

## Final validation

Validated code head: `a55e2c5d53b9ce8540c426ab28a2b824e03bcd42`.

- Regression workflow `31292791606`: SUCCESS. Explicit compile and shell syntax passed; broad CPU suite contained only recorded inherited failures; classifier reported zero new branch-only failures.
- Disabled-equivalence workflow `31292791593`: SUCCESS.
- Same-batch-enabled two-rank DDP workflow `31292791598`: SUCCESS.

The visibility/local-P follow-on is therefore CPU/DDP validated. The corrected real-CUDA smoke remains the final external gate.

## External GPU gate

```bash
cd ~/git/thog2
git fetch origin PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git checkout PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git reset --hard origin/PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
bash tools/run_plastic_coarse_fine_gpu_smoke.sh
```
