# PLASTIC same_batch_all_probes implementation tasks

Governing specification: `docs/THOG2_PLASTIC_Requirements_Specification_v0.53.txt`
Branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`
PR: #33
Started: 2026-08-09
Implementation code head validated: `da43638b2266ed7122618400299e768af3edb52d`
Corrected CUDA-smoke test head: `e5bd20485ac1bb463b2eca0b5e244932e5f6ae3f`

## Status

Overall: IMPLEMENTED; CPU/DDP VALIDATED; FIRST REAL CUDA ATTEMPT REACHED FINE/TRAINING BUT EXPOSED A STALE SMOKE ASSERTION; CORRECTED REAL CUDA RERUN OUTSTANDING

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
- [x] Run the first external real-CUDA smoke attempt on scruffy; it reached COARSE, fresh FINE construction and one successful optimizer update, then the test failed because its inherited `warmup_updates=1` correctly suppressed the first FINE probe while the stale assertion expected an audit row.
- [x] Correct the CUDA smoke so it disables warmup for the probe check, explicitly enables same-batch mode, completes a four-probe fixed-batch window and validates non-overlap/provenance/audit semantics.
- [ ] Rerun the corrected external real-CUDA smoke before relying on this mode for a serious GPU run.

## Validation at code head `da43638b2266ed7122618400299e768af3edb52d`

- `Validate PLASTIC COARSE FINE regression` run `31289904430`: SUCCESS.
  - explicit `py_compile`, including `sheet/plastic_depth_same_batch_all_probes_patch.py`: passed;
  - shell syntax: passed;
  - full CPU suite completed;
  - untouched-`PLASTIC_DEPTH` inherited-failure classifier: passed with no new branch-only failures.
- `Validate PLASTIC disabled equivalence` run `31289904395`: SUCCESS, exact disabled equivalence.
- `Validate PLASTIC COARSE FINE DDP` run `31289904420`: SUCCESS with `same_batch_all_probes=true`; two ranks agreed on the fixed-batch/window audit.

## External CUDA gate

First attempt on scruffy at branch head `a23636f090c3cb0e21f9eb30f41dec10d29dbb49`:

- CUDA device: NVIDIA GeForce RTX 4090 Laptop GPU;
- PyTorch 2.12.1+cu126 / CUDA runtime 12.6;
- COARSE trial and fresh FINE construction completed;
- first authoritative FINE optimizer update completed without being skipped;
- failure was `AttributeError: SharedTrainer has no attribute plastic_depth_count_audit` in the smoke assertion, because the fixture inherited `warmup_updates=1` and the warmup guard correctly created no probe/audit on update 1.

This was a smoke-test defect, not an observed CUDA allocator/OOM failure.

Corrected smoke commit: `e5bd20485ac1bb463b2eca0b5e244932e5f6ae3f`.

The corrected smoke now explicitly exercises `same_batch_all_probes=true` for a complete four-probe window and checks:

- four successful real CUDA optimizer updates;
- active layer count unchanged before the full window decision;
- one fixed `probe_batch_digest` across all four probes;
- window ordinals 1,2,3,4;
- provenance `(P1)`, `(P1,2)`, `(P1,2,3)`, `(P1,2,3,4)`;
- final STAY retirement and cleared window histories;
- full-radius candidate counts and audit replay;
- nonzero CUDA peak allocation.

Rerun:

```bash
cd ~/git/thog2
git fetch origin PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git checkout PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git reset --hard origin/PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
bash tools/run_plastic_coarse_fine_gpu_smoke.sh
```

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