# THOG2 Dynamic Prematerialisation - Work Log

## 2026-09-08 - Initialization

- Created GitHub branch `THOG2_Dynamic_Prematerialisation_Enhancement` from `master` commit `38763e41f31ae3f16f1f9a5609a0a7a63a686fe8`.
- Confirmed authoritative documents are fully readable:
  - requirements v0.2: 1,027 extracted lines;
  - implementation plan v0.1: 776 extracted lines.
- Confirmed GitHub permissions include read and write access to `peterdonnelly1/thog2`.
- Current local sandpit began empty; source is being grounded from the named GitHub branch through the GitHub connector.
- Implementation strategy follows the four plan stages. The disabled THOG path is treated as a hard regression boundary.

## Decisions and invariants

- Fused attention remains the default.
- Prematerialisation remains default-disabled.
- Exactly one dedicated CUDA premat stream is permitted per active model device.
- Candidate queue order is strict; a deferred head candidate cannot be bypassed.
- v1 lookahead stops at l+1.
- A candidate already MATERIALISING at its deadline is awaited, never duplicated or cancelled.
- A candidate still UNAVAILABLE at its deadline falls back immediately to authoritative main-stream materialisation.
- Current-peak and global-buffer guards remain separate process-level and device-level calculations.
- UI telemetry is observational and must not influence scheduler decisions.
- Existing nanoGPT lines are preserved/commented and all new THOG changes use the repository marker convention.

## Test record

No tests run yet.

## Known blockers or deferred work

None at initialization.
