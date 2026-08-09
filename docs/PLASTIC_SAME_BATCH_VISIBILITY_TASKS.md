# PLASTIC same-batch operator visibility tasks

Parent implementation: `docs/PLASTIC_SAME_BATCH_ALL_PROBES_IMPLEMENTATION_LOG.md`
Branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`
PR: #33
Started: 2026-08-09

## Status

IMPLEMENTED; FINAL CI REVALIDATION IN PROGRESS.

## Task list

- [x] Diagnose user report that enabled same-batch mode was not visible in startup output.
- [x] Confirm the observed P1..P4 then P5 provenance reset was consistent with strict non-overlapping windows, while noting this was inference rather than acceptable operator proof.
- [x] Add explicit compact startup visibility: `plastic fine: ... same_batch=true|false`.
- [x] Add explicit detailed startup visibility: `plastic__layer_count__same_batch_all_probes: true|false`.
- [x] Add per-probe fixed-batch identity marker: `same_batch W<window>:<ordinal>/<size> B=<digest8>`.
- [x] Make operator-facing P numbering window/batch-local: every fresh batch starts at P1.
- [x] Retain monotonically increasing global probe sequence/provenance only as internal audit fields for durability/debugging.
- [x] Preserve false/default runtime behavior.
- [x] Add focused runtime/audit/console tests for P reset and batch identity.
- [x] Add an end-to-end public-wrapper dry-run test requiring compact `same_batch=true` startup output.
- [x] Add `sheet/plastic_depth_same_batch_visibility_patch.py` to explicit CI `py_compile` validation.
- [x] Correct the first visibility test, which mistakenly called a Stage6 console method on `SharedTrainer`; production code was not changed for that test defect.
- [ ] Require final regression classifier to report zero new branch-only failures on the corrected test head.
- [ ] Require exact disabled-equivalence and same-batch-enabled DDP workflows to remain green on the corrected test head.
- [ ] Rerun the corrected real-CUDA same-batch smoke externally before relying on a serious long GPU run.

## Expected operator sequence for W=4

```text
P   1 ... (P1)       ... same_batch W1:1/4 B=aaaaaaaa
P   2 ... (P1,2)     ... same_batch W1:2/4 B=aaaaaaaa
P   3 ... (P1,2,3)   ... same_batch W1:3/4 B=aaaaaaaa
P   4 ... (P1,2,3,4) ... same_batch W1:4/4 B=aaaaaaaa

P   1 ... (P1)       ... same_batch W2:1/4 B=bbbbbbbb
```

`B` must remain identical within a window. A fresh window acquires a fresh evidence batch and starts visible P numbering at 1.

## Relevant commits

- `b99186fad0ae5633b89bd3e96bf523d8f0ac4a9f` — visibility/local-provenance overlay.
- `a67ed03ac82e5de182645019c6201da375f3a222` — install visibility overlay last.
- `855434d8a2ffeebe5005eac89610d030aa171417` — initial visibility tests.
- `e655708fec7d4a951897ab79a7c2c361cd00d966` — wrapper remembers requested same-batch mode.
- `11f4d8833502235f865e8e987bd984219eb06caf` — compact header shows same-batch mode.
- `e40748ebe5cd0467f25cde4d916e3b7cee28111c` — correct Stage6-boundary visibility test.
- `c26ab54e55de0419257fd2323cc56b919233d5a8` — explicit visibility-module compile gate.
- `d7838fab7f9a5522a8249a8cafd477087bf072a0` — public-wrapper compact-header regression.
