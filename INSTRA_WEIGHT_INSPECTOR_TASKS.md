# Instra weight inspection tasks

Base: 63e5b97 on initialisation_baselining. Worktree: instra_weight_inspector.

- [x] Reproduce and fix latest-step filtering, including unequal per-run final steps.
- [x] Add fullscreen-only inspector of raw recorded chart values (step rows, layer columns).
- [x] Interleave Workspace runs within each layer and retain run colours.
- [x] Add persistent Weights group precision, default 4 decimal places.
- [x] Add two-axis scrolling, rectangular selection, clipboard TSV and back navigation.
- [x] Verify regressions, record the blocked browser check, and commit for review.

Existing logs retain selected coupling trajectories, not complete layer matrices. Inspector uses exact recorded executed-layer values; does not synthesize missing data. Full matrix recording is outside this implementation.

Validation: 19 Node regression programs pass; full JavaScript syntax checks and git diff --check pass. Browser navigation was rejected by approval review; browser acceptance remains unverified. See INSTRA_WEIGHT_INSPECTOR_LOG.md.
