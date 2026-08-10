# PLASTIC Vega + growth-side discount — progress log

## 2026-08-10

### 0–20%: inspection and design pinning

- Starting branch head before this package: `b1ae60da5fdaa12f4b4129e8d817921625adcf75`.
- Confirmed W&B custom/multi-line charts use `wandb.Table`; W&B documents a 10,000-row maximum per table.
- Chosen rolling window: 300 probes.
- Chosen W&B presentation: two independent multi-line charts, shrink/interpolation and growth/extrapolation, each with one line per probe and independent autoscaling.
- TensorBoard receives no spaghetti-chart tables or figures.
- Raw probe deltas remain the source for charting; economic-score-adjusted values are not charted.
- New knob is first-class configuration: `plastic__layer_count_decision_algorithm__growth_side_discount` / CLI `--plastic__layer_count_decision_algorithm__growth_side_discount`.
- Full feasible radius participates in v0.55 TSK; for positive offsets only, a beneficial economic delta is attenuated by the discount before Sen/Kendall. Harmful positive-offset evidence is unchanged. Left/interpolated evidence is unchanged.
- Existing `plastic__wall_time_equivalent_time_gain_discount` remains separate and unchanged.
- Existing legacy directional-coherence extrapolation control remains intact for that legacy algorithm.

### 20–70%: implementation

- Added the growth-side discount to parser/config validation, persistent config/resume restoration, startup reporting, wrapper routing and help/registry discoverability.
- Updated both v0.55 Sen/Kendall paths to use the complete feasible `L-r ... L ... L+r` decision landscape while keeping committed count movement at exactly ±1.
- Applied growth-side discount only to beneficial positive-offset economic evidence; raw `probe_Δloss` is untouched.
- Added W&B-only rolling shrink/interpolation and growth/extrapolation multi-line charts over the most recent 300 probes.
- Bounded W&B table data below 10,000 rows by dropping oldest complete probe lines, retaining newest evidence and avoiding partial historical lines.
- Preserved same-batch window-local provenance and strict independent audit replay. Framework holds record and verify the raw pre-hold v0.55 selection before replaying STAY.

### 70–95%: regression closure

- First full CPU regression run `31353576368` exposed 13 branch-only failures beyond the recorded untouched-`PLASTIC_DEPTH` baseline.
- Two were production defects and were corrected:
  - rejected CUDA growth preflight now releases the reserve object actually returned by the headroom allocator instead of relying on a context side effect;
  - compact validation-loss numeric width now matches training-loss numeric width.
- The remaining failures were obsolete fixtures/assertions from superseded console geometry, stacked resume wrappers, canonical audit reason spelling, or shrink-side row ordering; those tests were updated without adding anything to the inherited-failure baseline.

### 95–100%: validation

- Candidate head `6f6c5e34a03dce0c1aab94c34b7cfb57dc3e9d8b` passed production-module compilation and public shell syntax.
- Full CPU regression run `31354437986`: `1013 passed`, `15 skipped`, `50 failed` where all 50 are recorded inherited failures; `branch_only_failures.txt` is empty and the baseline classifier succeeded.
- Two-rank DDP run `31354437987`: success.
- Disabled-equivalence run `31354437978`: success with exact fingerprint equivalence.
- Task and progress logs closed. A final workflow trigger follows these documentation commits so the reported branch tip includes the closure records.

### Pickup

```bash
cd ~/git/thog2
git fetch origin PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git checkout PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git reset --hard origin/PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
chmod +x *.sh
git log -1 --oneline
```
