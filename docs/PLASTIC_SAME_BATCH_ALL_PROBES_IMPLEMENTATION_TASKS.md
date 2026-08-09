# PLASTIC same_batch_all_probes implementation tasks

Governing specification: `docs/THOG2_PLASTIC_Requirements_Specification_v0.53.txt`
Branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`
PR: #33
Started: 2026-08-09
Implementation code head validated: `da43638b2266ed7122618400299e768af3edb52d`

## Status

Overall: IMPLEMENTED; CPU/DDP VALIDATED; REAL CUDA SMOKE OUTSTANDING

## Task list

- [x] Confirm governing branch/spec and preserve existing disabled path.
- [x] Create takeover-safe task list and implementation log before runtime edits.
- [x] Map current FINE probe call chain, batch ownership, decision-history ownership, checkpoint serialization, DDP and console provenance.
- [x] Add public `plastic__layer_count__same_batch_all_probes`, default `false`, help exposure, enabled-mode persistent identity and checkpoint compatibility.
- [x] Add explicit same-batch probe-window runtime state with strict non-overlapping `W`-probe lifecycle.
- [x] Add one fresh cached probe batch per window, including stable sampled-token selection.
- [x] Decouple decision-loss evidence from ordinary training microbatches in same-batch mode.
- [x] Enforce no-grad/no-backward semantics for decision-loss evidence while preserving distinct resource/timing probes.
- [x] Enforce complete-window-only decision/commit and retire the window after both CHANGE and STAY.
- [x] Invalidate partial same-batch windows after a relevant active-count/joint-state mismatch.
- [x] Preserve/extend P provenance and structured window/batch diagnostics.
- [x] Persist/reconstruct partial-window batch/evidence/provenance for exact checkpoint resume.
- [x] Synchronize same-batch window semantics under DDP.
- [x] Add focused unit/regression tests, including disabled-path regression and final overlay ownership.
- [x] Run available CPU validation / CI.
- [x] Update PR implementation summary and finalize this task list/log.
- [ ] Run the external real-CUDA smoke before relying on this mode for a serious GPU run.

## Validation at code head `da43638b2266ed7122618400299e768af3edb52d`

- `Validate PLASTIC COARSE FINE regression` run `31289904430`: SUCCESS.
  - explicit `py_compile`, including `sheet/plastic_depth_same_batch_all_probes_patch.py`: passed;
  - shell syntax: passed;
  - full CPU suite completed;
  - untouched-`PLASTIC_DEPTH` inherited-failure classifier: passed with no new branch-only failures.
- `Validate PLASTIC disabled equivalence` run `31289904395`: SUCCESS, exact disabled equivalence.
- `Validate PLASTIC COARSE FINE DDP` run `31289904420`: SUCCESS with `same_batch_all_probes=true`; two ranks agreed on the fixed-batch/window audit.

## External CUDA gate

Run:

```bash
bash tools/run_plastic_coarse_fine_gpu_smoke.sh
```

The CPU/DDP suite does not validate real CUDA allocator/OOM interaction or performance. In particular, same-batch decision evidence is intentionally no-grad and therefore cheaper in activation/gradient memory than the authoritative training update. A live GPU smoke is required before treating upward-candidate allocator behavior as production-proven.

## Non-negotiable semantics

1. `--plastic__layer_count__same_batch_all_probes` is the public option; short prose name is `same_batch_all_probes`.
2. Default `false` preserves current behaviour.
3. `true` means one fixed probe batch per complete decision window and strictly non-overlapping windows.
4. The next window always gets a fresh probe batch, including after STAY.
5. Ordinary training continues on normal fresh training batches; the cached probe batch is evidence-only.
6. No layer-count decision/commit before the full configured probe window is complete.
7. One completed-window decision may still move multiple layers, bounded independently by `plastic__layer_count__max_allowable_layer_change`.
8. Decision-loss probes are no-grad and perform no backward/optimizer mutation; resource probes remain a separate class.
9. Existing P provenance remains visible and is window-local.
10. Partial-window exact resume reproduces the same batch, sampled-token selection, evidence and next decision.

## THOG source-history discipline

Do not delete inherited nanoGPT/THOG lines. Preserve superseded lines as comments where required. New or replaced larger blocks use `# vvv THOG` / `# ^^^ THOG`; single-line edits use the project trailing `# <<< THOG ...` convention at column 156 where practical.
