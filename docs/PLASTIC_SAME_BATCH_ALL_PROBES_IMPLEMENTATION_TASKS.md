# PLASTIC same_batch_all_probes implementation tasks

Governing specification: `docs/THOG2_PLASTIC_Requirements_Specification_v0.53.txt`
Branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`
PR: #33
Started: 2026-08-09

## Status

Overall: IN PROGRESS

## Task list

- [x] Confirm governing branch/spec and preserve existing disabled path.
- [x] Create takeover-safe task list and implementation log before runtime edits.
- [ ] Map current FINE probe call chain, batch ownership, decision-history ownership, checkpoint serialization, DDP and console provenance.
- [ ] Add canonical configuration field `plastic__layer_count__same_batch_all_probes`, default `false`, validation, identity/checkpoint metadata and startup/help exposure.
- [ ] Add explicit same-batch probe-window runtime state with strict non-overlapping `W`-probe lifecycle.
- [ ] Add one fresh cached probe batch per window, including stable sampled-token selection.
- [ ] Decouple decision-loss evidence from ordinary training microbatches in same-batch mode.
- [ ] Enforce no-grad/no-backward semantics for decision-loss evidence while preserving distinct resource/timing probes.
- [ ] Enforce complete-window-only decision/commit and retire the window after both CHANGE and STAY.
- [ ] Invalidate partial same-batch windows after any relevant architecture-state change.
- [ ] Preserve/extend P provenance and structured window/batch diagnostics.
- [ ] Persist/reconstruct partial-window batch/evidence/provenance for exact checkpoint resume.
- [ ] Synchronize same-batch window semantics under DDP.
- [ ] Add focused unit/regression tests, including disabled-path regression and source-history markers.
- [ ] Run available CPU validation / CI and record any external GPU-only gate.
- [ ] Update PR implementation summary and finalize this task list/log.

## Non-negotiable semantics

1. `--plastic__layer_count__same_batch_all_probes` is the public option; short prose name is `same_batch_all_probes`.
2. Default `false` preserves current behaviour.
3. `true` means one fixed probe batch per complete decision window and strictly non-overlapping windows.
4. The next window always gets a fresh probe batch, including after STAY.
5. Ordinary training must continue on its normal fresh training batches; the cached probe batch is evidence-only.
6. No layer-count decision/commit before the full configured probe window is complete.
7. One completed-window decision may still move multiple layers, bounded independently by `plastic__layer_count__max_allowable_layer_change`.
8. Decision-loss probes are no-grad and perform no backward/optimizer mutation; resource probes remain a separate class.
9. Existing P provenance remains visible and becomes window-local.
10. Partial-window exact resume must reproduce the same batch, sampled-token selection, evidence and next decision.

## THOG source-history discipline

Do not delete inherited nanoGPT/THOG lines. Preserve superseded lines as comments where required. New or replaced larger blocks use `# vvv THOG` / `# ^^^ THOG`; single-line edits use the project trailing `# <<< THOG ...` convention at column 156 where practical.
