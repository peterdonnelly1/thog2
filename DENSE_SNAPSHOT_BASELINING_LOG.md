# DENSE Snapshot Baselining v0.2 - Implementation Log

## 2026-08-30 - Start

- Created the implementation branch from promoted `master` commit `b9b09b1e125633e451bae60d9d2f9ea3f0721987`; renamed it to `initialisation_baselining` before publication at the user's request.
- Extracted and reviewed the v0.2 specification.
- Located the pre-optimizer construction boundary in `SharedTrainer.__init__`.
- Confirmed that pure DEPTH stores matrices as depth coefficients, keeps LayerNorm and bias conventional by default, and exposes the production QR-stabilised Chebyshev basis through `DepthTrajectory.depth_basis`.
- Confirmed the DENSE repeated-family layout and explicit split required to map the combined attention Q/K/V projection.
- Chosen implementation direction: a dedicated versioned snapshot module prepares and validates all tensors before committing any model mutation; the trainer invokes it after ordinary model initialisation and before DDP, optimizer, or batch-source construction.

## Decisions

- Snapshot files will be individual immutable files under repository-root `dense_baseline_snapshots/`.
- The payload will store unique parameter tensors once and record all alias names, including tied token-embedding/output-head storage.
- The compatibility hash will describe physical source structure; P and target representation will be validated separately and excluded from source compatibility.
- B and C will call the same float32 Chebyshev mapping implementation and production materialisation boundary.
- Snapshot action controls will not be replayed when resuming a checkpoint; accepted snapshot provenance will be persisted separately.

## Test record

- Installed the CPU-capable project test dependencies in the transient sandpit only.
- `tests/test_dense_snapshot_baselining.py`: 22 passed.  Coverage includes payload minimality and aliases, filename resolution, atomic no-overwrite publication, integrity rejection, RNG non-consumption/restoration, physical compatibility and dtype conversion, P=L reconstruction, reduced-P shared B/C mapping, exact direct-copy families, rank/non-finite rollback, CLI conflicts, non-replayable actions, and full SharedTrainer A/B/C construction/checkpoint provenance.
- Python compilation, Bash syntax, and `git diff --check` passed.
- Existing targeted regression batch: 81 passed and 8 failed initially. Two failures were attributable to this branch and were fixed (mock parser compatibility and optional metadata on a patched trainer). The remaining six failures reproduce existing post-INSTRA test/source drift in geometry report wording and outer-wrapper source-layout expectations; none exercise DENSE snapshot code. The two affected regressions plus the focused suite were rerun: 23 passed.
- Full repository suite after installing the repository-declared Plotly/TensorBoard extras: 1,246 passed, 24 skipped, 66 failed, and 452 subtests passed in 233.28 seconds. Representative failures were rerun against a detached worktree at the exact baseline commit `b9b09b1e`; they fail there identically. The broad failures divide into promoted-baseline test/source drift (outer wrapper now delegates to `train_OWT_core.sh`, renamed console/chart/artifact contracts, and pre-existing configuration tests) and sandbox-denied Gloo socket creation for DDP tests. No broad-suite failure enters `sheet/dense_snapshot.py` or uses either new CLI action.
- Public wrapper dry runs for A Normal DENSE, B Compressor-baselined DENSE, and C Compact Run all resolved successfully using underscore spellings normalized by `train_OWT.sh`. B and C emitted the pure DEPTH Chebyshev mapping and omitted residual-initialisation arguments; A retained ordinary DENSE residual initialisation.

## Implementation checkpoint

- Added `sheet/dense_snapshot.py` as the single snapshot, compatibility, mapping, and diagnostics implementation.
- Added fresh-run CLI controls and wrapper normalization/validation.
- Integrated the action immediately after ordinary model construction and before parameter reporting, DDP, optimizer, batch source, or forward execution.
- Added snapshot provenance to parameter reports, lifecycle manifests, and checkpoints while omitting action controls from persisted run/trainer configuration.
- Added `dense_baseline_snapshots/` to `.gitignore`; runtime files remain immutable individual files under that repository-root directory.

## Final verification and publication

- Renamed the unpushed implementation branch to `initialisation_baselining` as requested; the superseded branch name was never published.
- Fresh final pass: 22 focused tests passed in 3.10 seconds; new implementation/test lint passed; Python compilation, Bash syntax, and `git diff --check` passed.
- Reviewed the complete integration delta and staged only the snapshot implementation, its tests, its CLI/trainer/checkpoint integrations, `.gitignore`, and these two handover files.
- Published the reviewed commit to `origin/initialisation_baselining` through the ChatGPT GitHub connector.
