# THOG depth-weight curves + observational probes task list

Base branch: `chaos_bump_sampling`
Implementation branch: `depth_weight_curves_and_observational_probes`
Base commit: `33f89809354cf9075b71498c0549dfb8d4b4ad9f`

- [ ] Add DEBUG>2 continuous depth-weight W&B charts under group `depth`.
  - [ ] Track three model regions: one attention-Q head, MLP up, MLP down.
  - [ ] Configurable scalar weights per matrix.
  - [ ] Deterministic coordinates fixed for a run.
  - [ ] Optional same coordinates across runs.
  - [ ] Configurable latest vs accumulate time mode.
  - [ ] Configurable bounded history length.
  - [ ] Configurable chart logging interval.
  - [ ] Evaluate at dense arbitrary depth coordinates, not only active layers.
- [ ] Existing `PLASTIC sampled coefficients by capacity layer sample number` only when DEBUG>9.
- [ ] Permit layer-count probes in any THOG DEPTH run even when PLASTIC/growth/count-learning are disabled.
  - [ ] Probes remain observational: no layer-count commit when learning disabled.
  - [ ] Probe radius / token count / cadence respected.
  - [ ] Preserve existing learned-count behavior.
- [ ] Darken green used for right-side negative probe delta loss.
- [ ] Add regression tests for config, deterministic selection, arbitrary-depth curves, DEBUG gates, observational probes, and colour.
- [ ] Run targeted tests.
- [ ] Run broader regression suite.
- [ ] Review changed files against THOG comment-marker rules.
- [ ] Finalize progress log and report branch/commit/run guidance.
