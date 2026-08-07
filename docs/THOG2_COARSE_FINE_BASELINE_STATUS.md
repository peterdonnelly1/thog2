# THOG2 PLASTIC COARSE/FINE — Regression Baseline

## Comparison basis

- Untouched base branch: `PLASTIC_DEPTH`
- Base commit: `311decd512829b94116df19815e463e544cdf23a`
- Working branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`
- Comparison environment: GitHub-hosted Ubuntu CPU runner with the same Python, PyTorch and pytest installation used by the branch-wide regression workflow.

## Result

An initial complete current-branch run reported 49 distinct failing pytest nodes. Those exact 49 nodes were then executed against the untouched base commit:

- 47 failed on the untouched base and are inherited baseline failures.
- 2 passed on the untouched base and were therefore genuine branch regressions.

The two branch regressions were:

1. `tests/test_modified_fast_discard_false.py::test_stage4_update_is_equivalent_and_releases_materializations`
2. `tests/test_plastic_depth_cuda.py::test_successful_reserve_is_released_by_update_cleanup`

Both now pass together with the focused full-radius OOM/materialisation matrix. The CUDA issue was caused by routing a one-candidate upward probe through the new multi-candidate suffix API; the established adjacent N+1 path is now preserved exactly.

The canonical inherited-node list is:

- `docs/THOG2_COARSE_FINE_INHERITED_FAILURES.txt`

The broad regression workflow now runs the complete CPU suite, records all failures, and fails only when the observed set contains a node not present in that exact untouched-base list. It does not relabel branch regressions as inherited.

## Final v0.52 closure

The final v0.52 cleanup migrated 20 stale pre-v0.52 assertions to the implemented contract: exact `plastic__` CLI names, removal of the retired min-probes control, full-window directional evidence, ambiguous-direction HOLD semantics, `Δ` console labelling, current artifact naming, and the `99999` non-finite skip default.

Complete CPU regression workflow run `31149244593` then reported:

- 50 failed
- 862 passed
- 15 skipped
- 452 subtests passed
- 0 branch-only failures after comparison with the untouched-base inherited-node list

The 50 observed failures are all recorded inherited baseline failures. `branch_only_failures_after_v052_migration.txt` was empty. No test deselection was used in this final v0.52 closure run.

## Evidence runs

- Initial branch-wide failure evidence: workflow run `31067179754`, job `92507232907`.
- Exact 49-node untouched-base comparison: workflow run `31067676342`, job `92508759157`.
- Focused OOM/materialisation regression closure: workflow run `31068062566`, job `92509962488`.
- Final v0.52 migrated-contract and complete-suite closure: workflow run `31149244593`, job `92775267759`.

The earlier baseline comparison temporarily deselected one separately documented console-minor assertion after prior evidence established that it already failed on `PLASTIC_DEPTH`. The final v0.52 closure run above contains no deselection.
