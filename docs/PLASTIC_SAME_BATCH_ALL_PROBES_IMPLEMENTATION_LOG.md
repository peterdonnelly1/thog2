# PLASTIC same_batch_all_probes implementation log

Governing specification: `docs/THOG2_PLASTIC_Requirements_Specification_v0.53.txt`
Branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`
PR: #33
Started: 2026-08-09

## Takeover summary

Implement Version 0.53 as an additive FINE probe-framework mode. The current branch already implements v0.521/v0.52 FINE full-radius probing, full-window readiness, directional coherence, probe provenance, token subsampling, audit/replay and COARSE/FINE lifecycle. Do not disturb the established path when the new flag is false.

The key architectural change is that the current inline FINE decision evidence is derived from a training microbatch. That cannot simply be cached across probe events because it would cause repeated training on the same data. In same-batch mode, decision-loss evidence must therefore be evaluated on a separate cached probe batch under no-grad, while the authoritative training update continues on its ordinary fresh microbatches.

## Chronological log

### 2026-08-09 - initialization

- Confirmed active implementation branch `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION` and draft PR #33.
- Confirmed governing text spec v0.53 is committed at branch head ancestry.
- Confirmed user requested frequent progress heartbeats and takeover-safe task/log files.
- Created `docs/PLASTIC_SAME_BATCH_ALL_PROBES_IMPLEMENTATION_TASKS.md` and this log before modifying runtime code.
- Local `git clone` is unavailable in the current execution environment because DNS/network access is blocked. Repository inspection/writes will use the connected GitHub app; validation will use repository CI where practical and static/focused reasoning otherwise.
- Earlier source inspection established:
  - `sheet/trainer_step.py::_plastic_depth_candidate_loss()` is explicitly `torch.no_grad()` but belongs to the superseded external controller path.
  - current FINE inline probing is initiated from the first training microbatch via `plastic_depth_probe_request`; the same forward participates in the training backward path.
  - legacy timing/memory resource probes deliberately execute backward but zero gradients and do not optimizer-step; they are not decision-loss evidence and must remain distinct.
  - v0.52 decision readiness already requires the complete configured history window.

## Planned implementation shape

Prefer an additive final overlay/module imported after existing v0.521/v0.52 patches, plus minimal canonical dataclass/config/checkpoint plumbing. Keep the old inline path intact when `same_batch_all_probes=false`.

For `true`, expected state machine:

1. First eligible probe of new window acquires and caches a fresh dedicated probe batch and deterministic sampled-token indices.
2. Every scheduled probe event performs a no-grad decision-evidence forward against that cached batch for all feasible candidate counts.
3. The ordinary training update still executes on its normal current training microbatch chain.
4. Evidence/provenance accumulates only within the active window.
5. No commit before exactly W valid probes.
6. At W, run one decision; commit bounded movement or STAY; record audit; retire batch/evidence/provenance.
7. Next eligible probe starts a new window with a fresh probe batch.
8. Any joint architecture-state change invalidates a partial window.
9. Checkpoint/resume reconstructs the exact partial window and cached batch/token selection.

## Decisions still to verify in source before coding

- Exact current `SheetGPT.forward(... plastic_depth_probe_request=...)` implementation and whether candidate losses can be obtained through a dedicated no-grad helper without mutating authoritative active-count state.
- Exact trainer-state/checkpoint serialization shape for `plastic_depth_probe_histories`, audit and transient runtime state.
- Existing batch-source state format and best deterministic way to checkpoint the fixed probe batch: raw tensors vs source indices/state.
- Existing console provenance owner and audit event names.
- CLI/help generation path for canonical dataclass fields.
- Existing test/CI workflow entry points that can validate branch-only changes without CUDA.

## Validation log

No runtime code changed yet. Validation pending.
