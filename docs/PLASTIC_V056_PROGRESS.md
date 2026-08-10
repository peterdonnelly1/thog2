# PLASTIC v0.56 — progress log

## 2026-08-10

### 0–18% — scope, specification and governing-document recovery

- Starting branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`.
- Recovered the v0.55 governing specification and issued v0.56 as a +0.01 revision.
- v0.56 incorporates the already-implemented post-v0.55-document changes: full feasible-radius TSK, asymmetric growth-side discount and W&B-only rolling probe curves.
- Recovered and rendered `THOG2_PLASTIC_Requirements_Specification_v0.56.docx`; final render is 8 pages and was visually reviewed page-by-page.
- Added repository text copy `docs/THOG2_PLASTIC_Requirements_Specification_v0.56.txt`.

### 18–45% — objective-neutral controller and operator surface

- Removed the artificial TSK requirement for `relative_training_wall_time`.
- Decoupled `plastic__layer_count_objective` from `plastic__layer_count_decision_algorithm`.
- Objective-neutral public TSK names are now:
  - `theil_sen_kendall_LRA`
  - `sen_kendall__tau__stratified`
- Retired wall-time-prefixed TSK names fail clearly rather than mutating the selected objective.
- Added canonical lower-is-better objective-score extraction for TSK while preserving the special wall-time-equivalent-seconds conversion only for `relative_training_wall_time`.
- Preserved hard feasibility exclusion for `memory_budget`.
- Added one-shot bold/bright-yellow highlighting of the authoritative new numeric `layers` value on the committed transition row only.
- Expanded canonical registry and wrapper help with the 4 × 3 compatibility matrix, base `directional_coherence` semantics, Sen/Kendall interpretation, fixed |tau|=0.5 acceptance rule, adjacent gate and objective-neutral growth-side discount wording.

### 45–76% — focused cross-product and exclusivity testing

- Added representative validation for all 12 objective × decision-algorithm combinations.
- Added fail-fast sentinels proving TSK does not execute legacy robust median/MAD/sigma/change_z/score_z or directional-coherence calculations.
- Added inverse coverage proving `directional_coherence` retains its legacy robust-z machinery.
- Added TSK console reconstruction tests that fail if `score_evidence` is traversed to rebuild score_z.
- Added transition-highlight tests for change, STET, first row, validation row and subsequent row.
- Added help-registry uniqueness and migration tests.
- First focused run exposed three fixture-prerequisite errors for `memory_budget` and one real argparse-help layering defect; these were corrected.
- Focused v0.56 gate then passed.

### 76–97% — broad regression and audit ownership

- Broad regression exposed four obsolete pre-v0.56 contract tests: old wrapper ownership, old wall-time-prefixed TSK name, old TSK-to-directional bootstrap fallback and old retirement wording. These were migrated to the v0.56 contract.
- Strengthened bootstrap regression: TSK bootstrap evidence must STET and must not invoke legacy directional/robust-z fallback.
- Added a real `SharedTrainer` exclusivity integration test using `lowest_loss + sen_kendall__tau__stratified` with legacy robust-z primitives replaced by fail-fast sentinels.
- That integration test exposed a real commit-wrapper ordering defect: generic audit replay occurred before the outer TSK commit wrapper attached its TSK decision data.
- Fixed audit ownership by capturing `sen_kendall_report` in the durable audit row before immediate replay; independent TSK replay now prefers that dedicated field and falls back only for older audit rows.
- The fix preserves `directional_report is None` for current TSK operation and therefore does not reintroduce a fake legacy directional-decision event.

### 97–100% — final regression closure

- Expanded focused gate caught the real trainer/audit path and all previously escaped migration tests.
- A subsequent broad run found only two stale help fixtures: old “economic evidence” wording and retired `wall_time__...` wrapper names. Production output was already correct; the fixtures were migrated and added to the focused gate.
- Final validated production/test head: `a71c7ccf65753baf3e44503715742782d81c00d8`.
- Final CPU run `31360177428`:
  - `1042 passed`
  - `15 skipped`
  - `50 failed`, all exactly the recorded inherited baseline
  - `branch_only_failures.txt` empty
  - baseline classifier succeeded
- Final DDP run `31360177417`: success.
- Final PLASTIC-disabled equivalence run `31360177412`: success with exact fingerprint equivalence.
- Compile and shell syntax: success.
- No inherited-failure baseline entries were added for v0.56.

### Pickup

```bash
cd ~/git/thog2
git fetch origin PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git checkout PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git reset --hard origin/PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
chmod +x *.sh
git log -1 --oneline
```
