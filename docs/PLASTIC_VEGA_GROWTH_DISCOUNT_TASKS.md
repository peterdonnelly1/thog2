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
- [ ] Add first-class run/training config field, CLI parser entry, validation, persistence/resume identity and startup reporting for `plastic__layer_count_decision_algorithm__growth_side_discount`.
- [ ] Update v0.55 stratified and LRA TSK inputs to use the full feasible radius.
- [ ] Apply growth-side discount to beneficial positive-offset equivalent-time deltas only.
- [ ] Preserve raw probe loss values unchanged for console and telemetry.
- [ ] Add W&B-only rolling 300-probe shrink and growth spaghetti charts; no TensorBoard output.
- [ ] Keep W&B chart tables below 10,000 rows and preserve higher precision than console formatting.
- [ ] Update help/registry surface and wrapper-facing discoverability.
- [ ] Add focused controller/config/telemetry tests.
- [ ] Run compile/syntax/focused CPU tests and inspect GitHub CI classification.
- [ ] Record final branch head and pickup command.

## Non-goals

- Do not implement `Auto`.
- Do not add TensorBoard versions of the Vega curves.
- Do not alter CUDA growth-headroom behaviour except where tests reveal a direct regression from this package.
