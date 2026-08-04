# PLASTIC DEPTH implementation log

Last updated: 2026-08-04 14:20 AEST

## Repository state

- Repository: `peterdonnelly1/thog2`
- Working branch: `PLASTIC_DEPTH`
- Pull request: #29, draft, base `HYPERBLOCK_LOOP_ENHANCEMENTS`
- Baseline head: `97612234648848e0666b009bdc15ee6aaa2f2560`
- Requirements target: PLASTIC DEPTH specification v0.3
- Implementation plan: THOG2 PLASTIC DEPTH Implementation and Testing Plan v0.1

## Baseline validation

Focused command:

```bash
pytest -q \
  tests/test_plastic_depth.py \
  tests/test_plastic_depth_basis_cache.py \
  tests/test_plastic_depth_interfaces.py \
  tests/test_sheet_stage6_trainer.py \
  tests/test_sheet_stage6_wandb.py \
  tests/test_console_progress_layout.py \
  tests/test_console_progress_pretty_rows.py
```

Result: 48 passed, 2 failed.

The two failures are pre-existing and unrelated to PLASTIC DEPTH. They expect explicit RGB yellow (`\033[1;38;2;255;255;0m`) while the directly imported console formatter uses palette-dependent bright yellow (`\033[1;93m`). Do not treat them as regressions from this work.

## Non-negotiable invariants

- `plastic__max_permitted_layers` is only a count ceiling/allocation capacity.
- Learned-count mode owns N active points plus one derived N+1 probe point.
- Inactive capacity slots cannot influence active coordinates or output.
- Count movement is exactly N-1, N or N+1.
- Add/subtract chart changes must preserve the generated field before the next optimiser update.
- Stay performs no controller-induced remap; ordinary geometry gradients remain active.
- PyTorch AdamW remains the optimiser. Re-expressed coefficient parameter state is reset rather than reimplementing AdamW.
- Existing non-plastic paths must remain unchanged.

## Phase checklist

- [x] P0: baseline head and focused tests recorded
- [x] P0: durable implementation log created
- [ ] P1: uniform `plastic__` configuration/UI names and `_L_dyn_` artifact identity
- [ ] P2: pure float64 Chebyshev affine re-expression kernel and property tests
- [ ] P3: active-prefix geometry replacing the phantom maximum lattice
- [ ] P4: atomic model-family re-gauge with verification diagnostics
- [ ] P5: targeted stock-AdamW state reset on committed re-gauge
- [ ] P6: shared inline N-1/N/N+1 first-microstep probe
- [ ] P7: robust MAD threshold, five-update brake, objectives and universal VRAM reserve
- [ ] P8: checkpoint schema, sampled-value console diagnostic and W&B fields
- [ ] P9: staged/full CPU regressions and hosted CI
- [ ] P10: GPU smoke commands and final handoff
- [ ] Update specification with as-built deviations

## Current exact task

Implement P2 first: a standalone float64 Chebyshev coefficient change-of-chart kernel plus exhaustive CPU property tests. Do not modify runtime count transitions until this gate passes.

## Work-in-progress files

None yet.

## Known unresolved decisions

- Old v0.1 PLASTIC DEPTH checkpoint migration may be ambiguous. Prefer an explicit rejection over silent reinterpretation unless a deterministic conversion is proven.
- Repeated re-gauges must be stress-tested for accumulated stored-dtype error.
- Count-dependent residual-depth scaling remains a separate architectural limitation and is not part of the gauge transform.

## Takeover instructions

1. Read this file and the v0.3 requirements specification.
2. Fetch `PLASTIC_DEPTH` and confirm the current branch head.
3. Run the focused tests above and compare failures with the recorded baseline.
4. Continue only the single task named under **Current exact task**.
5. After each phase, update this file with commits, tests, failures and the next exact task.
