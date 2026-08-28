# INSTRA August 27 Work Log

## 2026-08-27

- Started from clean local tree equivalent to published head `b978347a715d86ef280e844a37e2ef730e838e5e`.
- Reviewed the supplied screenshot and DENSE runstring.
- Screenshot state: current optimizer step 65; configured weight capture window 3000-5000; no snapshots should exist yet. Header copy currently mixes a valid future capture window with stale `data available -` presentation.
- Runstring state: logging interval `-l 10`; weight capture interval 10; start 3000; end 5000; history length 100. The reported first-curve latency must therefore be investigated independently of this screenshot, whose run had not reached capture start.
- Retrieved prior INSTRA constraints: retained snapshots remain visible after capture completion; Workspace overlap uses retained ranges; Runs and Workspace line-segment settings were previously independent, but the new requirement changes initial Workspace behaviour to per-run inheritance until an explicit Workspace override.
- Added storage-level minimum update fields for heatmap probes and weight snapshots; the run table now shows `P_s/P_e/C_s/C_e` and sorts by the relevant end step.
- Added optional per-run step gradients with an exact run-colour midpoint, light earliest curve and dark latest curve.
- Workspace overlap now refreshes every visible run's status, merges the selected run's fresher status, and can derive retained bounds from displayed traces while the catalog catches up.
- Workspace range errors are context-scoped and clear on return to Runs. Legacy retained-range copy is neutralised; the final header shows only the configured capture window.
- Loading state now persists across empty/stale family responses when status reports retained snapshots; an authoritative empty requested range can still report no records.
- Workspace line-segment display inherits each run's Runs preference until a Workspace group/chart value is explicitly saved.
- Maximized weight charts retain their original live action controls, positioned in the now-visible Weights group header rather than cloned.
- The delayed train chart had two contributors: a pending train group opening was not transferred to the real W&B group, and discovery polled only every 2.5 seconds. Explicit pending-group state now transfers and discovery polls every second; chart payloads remain lazy for collapsed groups.
- `history_length` is not globally redundant: it is the safety bound for open-ended accumulation. It is redundant for a finite start/end window because lifecycle code auto-sizes retention to the full inclusive cadence. CLI help now states that distinction.
- Local verification: all dashboard JavaScript regressions pass; 198 non-browser focused Python tests plus 9 subtests pass; an additional 52 storage/dashboard/W&B tests pass; Python and JavaScript compilation pass.
- Local Firefox reached driver startup but the workspace cancelled its driver/network approval. The real Firefox cases remain mandatory in hosted CI before release.

## 2026-08-28 takeover

- Resumed from the clean sandpit handoff. The six local unpublished commits were patch-equivalent to commits already published remotely; rebasing skipped those duplicates and aligned the sandpit to remote head `3d505e395fa149f077df033e9020ae76677d4dc1`.
- The remote line included four post-handoff fixes for Firefox loading/gradient assertions, retained curves during cache refresh, Workspace redraw timing, and remounting weight controls.
- Hosted run `33066483802` failed the focused job at `3d505e3`; broad CPU comparison was consequently skipped. Existing CI exposed only an exit-code annotation, so the focused pytest step now emits its failure tail as a GitHub check annotation for deterministic remote diagnosis.
- Reverified the non-browser portion at `3d505e3`: 198 tests plus 9 subtests passed, all JavaScript dashboard regressions passed, and Python/JavaScript compilation passed.
- Firefox 154.0.1 and geckodriver 0.37.1 were obtained in the sandbox, but the execution environment blocks even the local WebDriver TCP connection. Real Firefox acceptance therefore remains a hosted-CI gate.
- Published diagnostic commit `5457ca9a951b147458c7b5674ca2f1fc04171a09`. Hosted run `33136269263` passed setup, compilation, and 207 focused tests plus 9 subtests, but failed `test_real_firefox_workspace_intersection_and_step_windows` at the final light/base/dark Workspace gradient assertion; broad CI was correctly skipped. Added state-rich diagnostics to that assertion before changing production code.
