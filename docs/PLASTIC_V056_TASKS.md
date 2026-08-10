# PLASTIC v0.56 — task log

Branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`

## Governing scope

Implement the v0.56 controller refinement immediately after the completed v0.55/full-radius/growth-discount package.

- [x] Decouple `plastic__layer_count_objective` from `plastic__layer_count_decision_algorithm`.
- [x] Permit the complete 4 × 3 matrix:
  - objectives: `lowest_loss`, `layer_efficiency`, `relative_training_wall_time`, `memory_budget`;
  - decision algorithms: `directional_coherence`, `theil_sen_kendall_LRA`, `sen_kendall__tau__stratified`.
- [x] Retire the wall-time-prefixed TSK names; algorithm names do not silently choose or change the objective.
- [x] Make decision algorithms genuinely mutually exclusive:
  - TSK modes do not execute/update/calculate legacy median/MAD/sigma/change_z/score_z or legacy directional-coherence machinery;
  - `directional_coherence` retains its existing robust-z machinery and does not execute TSK machinery.
- [x] Make TSK consume the selected objective's canonical lower-is-better score:
  - `lowest_loss`: probe loss;
  - `layer_efficiency`: existing objective score;
  - `relative_training_wall_time`: existing wall-time-equivalent economic score in seconds;
  - `memory_budget`: exclude infeasible candidates; use probe loss for feasible candidates.
- [x] Preserve full feasible-radius TSK, ±1 commits, exact adjacent gate and the existing asymmetric `plastic__layer_count_decision_algorithm__growth_side_discount`.
- [x] Expand help/registry documentation for both decision families, including the agreed Sen/Kendall interpretation table and objective/decision compatibility matrix.
- [x] Highlight the numeric `layers` value in bold/bright yellow exactly once on the optimizer-progress row where a committed layer-count transition becomes authoritative.
- [x] Keep W&B rolling probe curves unchanged; no TensorBoard additions.
- [x] Update requirements specification to v0.56 DOCX and repository TXT copy.
- [x] Add focused tests plus proportionate full CPU, disabled-equivalence and two-rank DDP regression gates.
- [x] Preserve strict TSK independent audit replay without reintroducing a legacy directional-decision event.
- [x] Record validation evidence and pickup command.

## Validation evidence

Validated production/test head: `a71c7ccf65753baf3e44503715742782d81c00d8`.

- Focused PLASTIC v0.56 regression: success, including the real `SharedTrainer` TSK exclusivity/audit path and wrapper/help migration checks.
- Full CPU run `31360177428`: `1042 passed`, `15 skipped`, `50` recorded inherited failures; `branch_only_failures.txt` empty and the baseline classifier succeeded.
- Two-rank DDP run `31360177417`: success, including lifecycle/FINE-probe rank agreement.
- PLASTIC-disabled equivalence run `31360177412`: success with exact fingerprint equivalence.
- Production-module compile and public shell-syntax gates: success.
- v0.56 DOCX rendered to 8 pages and visually reviewed page-by-page with no clipping, overlap or broken layout.

## Pickup

```bash
cd ~/git/thog2
git fetch origin PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git checkout PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git reset --hard origin/PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
chmod +x *.sh
git log -1 --oneline
```

## Non-goals retained

- Auto was not implemented.
- Objective definitions were not changed merely to make TSK fit them.
- The user-facing Kendall threshold was not reintroduced.
- TSK commits remain one adjacent step.
- CUDA allocator-reserve guards and same-batch audit replay remain enforced.
