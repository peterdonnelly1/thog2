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

- Baseline test attempt: `python -m pytest ...` failed before collection because the sandpit runtime has neither `pytest` nor `torch` installed.
- Static Python validation: all repository Python files compile successfully with `python -m py_compile`.
- Shell validation: `bash -n train_OWT_core.sh` passes.
- Added `tests/test_premat.py` for pure admission guards, allocator-cache accounting, option exclusivity, retired-option diagnostics, and fused/unfused CPU numerical equivalence.

## 2026-09-08 - Core implementation checkpoint

- Added `sheet/premat.py`: deterministic admission engine, lifecycle model, strict two-layer candidate window, one-in-flight scheduler, one lazy persistent CUDA stream, completion events, main-stream deadlines, `record_stream` ownership, event history, and aggregate telemetry.
- Fused execution routes QKV/O/UP/DOWN through the scheduler. QKV admission accounts for the packed output plus separately materialised Q/K/V operation peak.
- Unfused execution routes QK and later V as distinct deadlines, uses explicit causal score/softmax/value application, and includes the larger foreground activation envelope.
- Activation-checkpoint forward uses one outer pass; backward recomputation creates fresh segment-local state and cannot reuse stale tensors/events.
- `--premat enabled` forces effective early discard. The shell accepts `-E` but reports it as ignored under premat.
- Retired `--plastic__layer_count__memory_budget_gib`; supplying it now fails with `--premat_gpu_memory_buffer_gb` as the replacement. PLASTIC `memory_budget` now derives capacity minus the shared global buffer at runtime.
- Deliberately rejects premat with HYPERBLOCK/non-DEPTH, torch.compile, and PLASTIC inline-probe replay in v1 rather than silently entering an unsupported execution path.

## Known blockers or deferred work

- CUDA correctness, stream-overlap, peak-envelope, and profiler gates require a torch/CUDA host and remain unexecuted in this sandpit.
