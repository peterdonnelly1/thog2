# INSTRA August 30 Enhancements - Implementation Log

## 2026-08-30 - Start

- Confirmed a clean `initialisation_baselining` worktree at remote commit `ef10005`.
- Located the dashboard's established patch-stack architecture, current table/Workspace/Overview implementations, and existing regression suites.
- Created this handover log and the accompanying top-level task list before implementation.

## Scope decisions

- Preserve existing run data and API contracts; derive new presentation from the already-recorded configuration and run metadata where possible.
- Keep `latest step` run-relative in multi-run Workspace overlays; explicit numeric ranges remain shared ranges.
- Add narrow regression coverage rather than refactoring the historical dashboard patch stack.

## Implementation

- Added compact `weight curves` and `DENSE snapshot` rows to the pre-training CLI header. The summary reads the effective instrumentation environment populated by the layered parser, so it reports the user's actual capture window and cadence rather than dataclass defaults.
- Began recording the exact launched Python command in new-run telemetry configuration. Overview displays that command; older runs are labelled honestly as predating command capture.
- Added a persistent drag handle to the Run NAME header. Resizing changes the table width with NAME, leaving the bounded columns unchanged; double-click resets to 390 px.
- Relocated `STEPS` after semantic preset column `p` in both the header and every rendered run row.
- Changed Workspace `latest step` from a shared numeric intersection to latest-only per-run requests, with a second client-side per-run maximum filter for safety.
- Fixed the colour-picker debounce race by capturing the run ID before the picker can close. Workspace colour changes now clear the merged depth cache and refresh all visible curves; reset uses the same path.
- Stacked Overview Summary above Config, made Summary/Config/Artifact Outputs collapsible, and preserved search text, focus/caret, scroll position, and disclosure state across periodic live refreshes.

## Verification

- Passed focused suites: `instra_aug30_enhancements_regression`, `instra_weight_request_router_regression`, `instra_weight_stability_regression`, and `instra_weight_runtime_loader_regression`.
- Passed related suites: dashboard consistency, dashboard performance, range interaction, Workspace overlay, v0.58 smoke, and weight coupling.
- Passed JavaScript syntax, Python byte-compilation, `bash -n train_OWT_core.sh`, `git diff --check`, and an isolated effective-environment execution of the CLI header formatter.
- The scratch container has neither PyTorch nor pytest, so a full Python dashboard process and pytest collection could not be launched here. No production dependency was added; the project test module is ready for its normal PyTorch environment.

## Delivery

- Final branch: `initialisation_baselining`.
- Publication method: ChatGPT GitHub connector, preserving the remote branch history.

## Live B-run correction

- The first B launch showed that `-I wandb` selects the lifecycle runner, whose resolved payload did not yet include `console_header`; the shell consequently raised `KeyError` before training.
- Added the same effective header summary to the lifecycle payload.
- Made both shell lookups defensive so missing optional presentation metadata can no longer prevent a run from starting.
