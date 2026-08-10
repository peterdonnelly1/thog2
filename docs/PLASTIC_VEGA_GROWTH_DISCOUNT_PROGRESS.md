# PLASTIC Vega + growth-side discount — progress log

## 2026-08-10

### 0–20%: inspection and design pinning

- Starting branch head before this package: `b1ae60da5fdaa12f4b4129e8d817921625adcf75`.
- Confirmed W&B custom/multi-line charts use `wandb.Table`; W&B documents a 10,000-row maximum per table.
- Chosen rolling window: 300 probes.
- Chosen W&B presentation: two independent multi-line charts, shrink/interpolation and growth/extrapolation, each with one line per probe and independent autoscaling.
- TensorBoard must receive no spaghetti-chart tables or figures.
- Raw probe deltas remain the source for charting; do not chart economic-score-adjusted values.
- New knob is first-class configuration: `plastic__layer_count_decision_algorithm__growth_side_discount` / CLI `--plastic__layer_count_decision_algorithm__growth_side_discount`.
- Agreed semantics: full feasible radius participates in v0.55 TSK; for positive offsets only, a beneficial economic delta is attenuated by the discount before Sen/Kendall. Harmful positive-offset evidence is unchanged. Left/interpolated evidence is unchanged.
- Existing `plastic__wall_time_equivalent_time_gain_discount` remains separate and unchanged.
- Existing legacy directional-coherence extrapolation control remains intact for that legacy algorithm; do not repurpose it.

### Current next action

Implement config/CLI/persistence and v0.55 decision transformation, then focused tests before touching telemetry.
