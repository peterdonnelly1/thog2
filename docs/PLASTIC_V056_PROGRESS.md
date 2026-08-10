# PLASTIC v0.56 — progress log

## 2026-08-10

### 0–10% — scope and governing-document recovery

- Starting branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`.
- Recovered current governing specification from `/mnt/data/THOG2_PLASTIC_Requirements_Specification_v0.55.{docx,txt}`.
- v0.56 is a +0.01 revision over v0.55.
- v0.56 will fold in already-implemented post-v0.55-document changes: full feasible-radius TSK, asymmetric growth-side discount, and W&B-only rolling probe curves.
- Confirmed current v0.55 implementation artificially requires `relative_training_wall_time` for TSK; this restriction is being removed.
- Confirmed current v0.55 selector dispatch is TSK-authoritative, but retained legacy wrappers can still execute directional/z-oriented machinery under TSK and the console path can reconstruct score_z before stripping it. v0.56 will make that machinery structurally inactive under TSK.
- Confirmed the common base candidate scorer already exposes lower-is-better scalar scores for `lowest_loss`, `layer_efficiency`, and feasible `memory_budget` candidates; relative wall time retains its existing equivalent-time economic conversion.

### Current next action

Write and render the v0.56 specification, commit its text copy under `docs/`, then implement controller/objective decoupling and exclusivity before console/help work.
