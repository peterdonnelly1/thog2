# INSTRA September 1 Regression Repair - Log

## Report and initial state

User reports slower interaction, initial values doing nothing/no selected colour, and a loading stall after magnification. Screenshots show Workspace with three runs, controls at 0-0 while multiple steps remain plotted, then five charts loading and one incorrectly sized chart.

Sandpit starts at published e53eb34 on initialisation_baselining. Existing modification to docs/PLASTIC_V056_PROGRESS.md is unrelated and must be preserved. Investigate request/range/render ownership before editing; do not reimplement the completed enhancements.

## Findings and repairs

1. Performance: removed a redundant second 2-second poller; the original timer now resolves the current refresh wrapper at tick time. Memoized raw per-run maxima and stopped interpreting a deliberately bounded overlap as stale merely because a run extends beyond it. Gradient redraw now has one owner instead of invoking two full redraw paths. No downsampling or removal of recorded data.
2. Header: z is inserted immediately after show overlapping range, with margin-left: 12px. Both views share that header. Cycling and gradient redraw use the current authoritative figure rather than a cached mount from another range.
3. Initial values: range-aware Workspace and family cache signatures; selected custom 0..0 survives unavailable overlap; aria-pressed reflects initial-values/step-1 selection. Empty requested-range metadata is retained through the Workspace merge, normal fetch, direct family fetch and cache reuse. A completed empty zero range clears old curves and reports 'No recorded initial weights (step 0) in this view.' It does not invent missing snapshots.
4. Loading/races: the refresh revision includes the selected view; obsolete in-flight responses are discarded and the latest range retried. Failed requests remain retryable. Deferred hidden charts now record their actual rendering completion and reconcile loading placeholders; pending jobs from an old run/range are discarded. Reopening Weights now routes its result through the depth family, correcting a coefficients/depth naming mismatch; that path also rejects obsolete range responses.

## Verification

- All 17 tests/*regression.js programs passed, including the new production refresh race test and strengthened cache, empty-range, group-reopen, deferred-render, initial-selection and existing z/Overview tests.
- Cache stress: 10,000 unchanged refreshes cause no extra depth fetches. Deferred-render stress: 1,000 hidden updates retain only the newest job per chart. These are deterministic tests, not measured interactive speed benchmarks.
- All dashboard JavaScript files, all regression programs and the optional browser script pass node --check. Three dependency-light September 1 Python unittest cases pass. git diff --check passes.
- Added an optional full production-stack Firefox/Plotly fixture script. Browser launch reached Firefox, but content subprocess creation failed with user-namespace EPERM before dashboard navigation. Therefore no real-browser pass or screenshot reproduction is claimed. The restriction was not bypassed. GPU/training tests are not relevant to these frontend-only repairs and were not run.
- The unrelated docs/PLASTIC_V056_PROGRESS.md file retains SHA-256 a09a758d89be32684bca340c270bb557deed9921a150d6138e77d25cc800e0fa and is excluded from publication.

## Publication / handoff

Reviewed parent: e53eb348ee2e863e873646f533cc011eaa5402b0. Publish only the reviewed dashboard files, regression tests and these two top-level handoff files using GitHub connector blobs/tree/commit and non-forced update of initialisation_baselining. Verify the remote commit before reporting delivery. The delivery commit contains this log; its SHA is in the final response. Pull --ff-only, restart Instra, then hard-refresh the browser.
