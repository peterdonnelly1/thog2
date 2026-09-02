# Instra September 2 work log

## Scope and recovery

All ten user asks are tracked in INSTRA_SEP02_TASKS.md (duplicate item 6 becomes 6a/6b).
Fresh isolated checkout at /workspace/scratch/7ef32b232cbc/thog2 cloned from prior delivered local git history. Parent verified through GitHub connector; unrelated dirty files in previous scratch are untouched. No AGENTS.md found in repository.

## Decisions

Keep existing training/materialization code unchanged unless a demonstrated logging fault requires a narrowly scoped repair. Preserve working dashboard paths and add targeted regression tests for actual failures. No subagents requested or used.
For Workspace difference use per-layer max(weight)-min(weight) across all selected runs at matching step/coupling; nonnegative, zero exactly when equal, no unstable percent denominator near zero. Show blank when fewer than two values or a selected run is missing; explain in column tooltip.

## Progress

10%: Confirmed GitHub instra_weight_inspector head 1f8995063750910894007fb3a23fcce5a45ac4fd; inspecting current inspector, metric state, data fetch paths, and run columns.

55%: All feature code implemented. Confirmed missing-train cause in dashboard_performance_patch.js: iterates mutable chart_titles including local_metric_* then clears plots absent from app.figures.depth. Restricted owner to depth_weight_chart_names. Added train/val sentinel plots to production-owner regression; suite changes also cover new expected group persistence and inspector comparison schema.
Browser skill attempted full fixture at localhost:8766; tool reports ERR_BLOCKED_BY_CLIENT. No alternate browser/bypass attempted. Browser clipboard, pixel layout remain unverified; production DOM handler tests exercise shortcuts, pointer selection, normal/maximized visibility, virtualization.
Difference columns use scientific notation for tiny nonzero spreads rather than rounding them to false equality. CSV exports all current inspector rows/columns with raw numeric precision and run/step/coupling/layer headers.

## Verification and delivery preparation

- All 20 tests/*regression.js programs pass; all production dashboard JavaScript passes node --check; git diff --check passes.
- Strengthened weights-refresh regression fails against parent 1f89950 with "weight refresh erased a train/val chart" and passes with the restriction to the six weight charts.
- Inspector tests run actual production pointer/key/copy/download handlers. Checked standard icon, maximized-only Select all, Ctrl+A/C, three-run spreads, missing values, tiny nonzero spreads, raw CSV precision, quote/newline headers, filename and Blob URL cleanup, selection preservation, both scroll axes, 1000 x 144 x 2-run virtualization (432 columns including differences), and context cleanup.
- Column tests verify lifecycle optimizer metadata, momentum only for SGD/Nesterov/RMSprop, missing metadata, integer LR codes 90/9 and 1000, zero min LR, and explicit lower-case c/f despite uppercase table CSS.
- Metric tests verify independent train/val z cycling, selected front run surviving refresh, source figures untouched, single-point visibility, and Runs-mode hiding. Group-state tests now verify persistence across Runs changes and Workspace membership changes, separately by mode.
- No Python, training, optimizer execution, materialization, or snapshot capture code changed. Existing synthetic weights and heatmap expansion behavior is retained; pending train group now inherits the same expansion state as the real group.
- Browser fixture expanded with train/val histories and optimizer/LR metadata for manual acceptance. Browser access remains blocked (ERR_BLOCKED_BY_CLIENT), so no real-browser/Firefox clipboard or pixel-layout pass is claimed.

Review complete. Publish the reviewed files through connector blobs/tree/commit and a non-forced update of instra_weight_inspector from 1f8995063750910894007fb3a23fcce5a45ac4fd. Verify the new ref and tree before reporting success. The final assistant handoff gives the actual commit SHA; use git pull --ff-only, restart Instra, then Ctrl+Shift+R.
