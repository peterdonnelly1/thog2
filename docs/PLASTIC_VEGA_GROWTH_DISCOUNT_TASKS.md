# PLASTIC Vega + growth-side discount — task log

Branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`

## Goal

Implement the agreed PLASTIC FINE enhancements without touching TensorBoard or Auto:

1. W&B-only rolling spaghetti curves for raw `probe_Δloss`, with separate shrink/interpolation and growth/extrapolation charts.
2. Add `--plastic__layer_count_decision_algorithm__growth_side_discount`.
3. Admit every feasible probe `L-r ... L ... L+r` to the v0.55 Sen/Kendall decision fit.
4. Apply the growth-side discount only to beneficial right-side economic evidence before Sen/Kendall; adverse right-side evidence remains undiscounted.
5. Preserve the existing `plastic__wall_time_equivalent_time_gain_discount` semantics unchanged.

## Tasks

- [x] Confirm current branch/head and inspect W&B telemetry, parser/config, TSK selector and existing extrapolation control.
- [x] Confirm W&B table/multi-line chart support and 10,000-row table limit from current W&B documentation.
- [x] Add first-class run/training config field, CLI parser entry, validation, persistence/resume identity and startup reporting for `plastic__layer_count_decision_algorithm__growth_side_discount`.
- [x] Update v0.55 stratified and LRA TSK inputs to use the full feasible radius.
- [x] Apply growth-side discount to beneficial positive-offset equivalent-time deltas only.
- [x] Preserve raw probe loss values unchanged for console and telemetry.
- [x] Add W&B-only rolling 300-probe shrink and growth spaghetti charts; no TensorBoard output.
- [x] Keep W&B chart tables below 10,000 rows by dropping oldest complete probe lines and preserve higher precision than console formatting.
- [x] Update help/registry surface and wrapper-facing discoverability.
- [x] Add focused controller/config/telemetry/audit tests.
- [x] Run compile/syntax/full CPU regression, two-rank DDP validation and disabled-equivalence validation; no branch-only CPU failures remain.
- [x] Record pickup command; exact final branch-tip SHA is reported by the completion turn after the final workflow trigger.

## Validation evidence

- Branch-clean CPU regression candidate: `6f6c5e34a03dce0c1aab94c34b7cfb57dc3e9d8b`.
- CPU regression run `31354437986`: `1013 passed`, `15 skipped`, `50` recorded inherited failures, `branch_only_failures.txt` empty; baseline classifier succeeded.
- DDP run `31354437987`: success, including two-rank lifecycle/FINE-probe agreement.
- Disabled-equivalence run `31354437978`: success, including exact fingerprint equivalence.

## Pickup

```bash
cd ~/git/thog2
git fetch origin PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git checkout PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git reset --hard origin/PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
chmod +x *.sh
git log -1 --oneline
```

## Non-goals

- Do not implement `Auto`.
- Do not add TensorBoard versions of the Vega curves.
- Do not alter CUDA growth-headroom behaviour except where tests reveal a direct regression from this package. The rejected-growth reserve-release correction made during regression closure is exactly such a direct regression fix.
