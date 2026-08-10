# PLASTIC v0.56 — task log

Branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`

## Governing scope

Implement the v0.56 controller refinement immediately after the completed v0.55/full-radius/growth-discount package.

1. Decouple `plastic__layer_count_objective` from `plastic__layer_count_decision_algorithm`.
2. Permit the complete 4 × 3 matrix:
   - objectives: `lowest_loss`, `layer_efficiency`, `relative_training_wall_time`, `memory_budget`;
   - decision algorithms: `directional_coherence`, `theil_sen_kendall_LRA`, `sen_kendall__tau__stratified`.
3. Retire the wall-time-prefixed TSK names; do not let an algorithm name silently choose or change the objective.
4. Make decision algorithms genuinely mutually exclusive:
   - TSK modes must not execute/update/calculate legacy median/MAD/sigma/change_z/score_z or legacy directional-coherence machinery;
   - `directional_coherence` retains its existing robust-z machinery and must not execute TSK machinery.
5. Make TSK consume the selected objective's canonical lower-is-better score:
   - `lowest_loss`: probe loss;
   - `layer_efficiency`: existing objective score;
   - `relative_training_wall_time`: existing wall-time-equivalent economic score in seconds;
   - `memory_budget`: exclude infeasible candidates; use probe loss for feasible candidates.
6. Preserve full feasible-radius TSK, ±1 commits, exact adjacent gate and the existing asymmetric `plastic__layer_count_decision_algorithm__growth_side_discount`.
7. Expand help/registry documentation for both decision families, including the agreed Sen/Kendall interpretation table and objective/decision compatibility matrix.
8. Highlight the numeric `layers` value in bold/bright yellow exactly once on the optimizer-progress row where a committed layer-count transition becomes authoritative.
9. Keep W&B rolling probe curves unchanged unless regression work reveals a direct compatibility defect; no TensorBoard additions.
10. Update requirements specification to v0.56 DOCX and repository TXT copy.
11. Add focused tests plus proportionate full CPU, disabled-equivalence and two-rank DDP regression gates.
12. Record final branch head and pickup command.

## Non-goals

- Do not implement Auto.
- Do not change objective definitions merely to make TSK fit them.
- Do not reintroduce a user-tunable Kendall threshold.
- Do not add multi-layer TSK commits.
- Do not weaken CUDA allocator-reserve guards or same-batch audit replay.
