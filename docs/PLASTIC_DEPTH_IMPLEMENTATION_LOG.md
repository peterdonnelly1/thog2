# PLASTIC DEPTH implementation log

Last updated: 2026-08-04 14:32 AEST

## Repository state

- Repository: `peterdonnelly1/thog2`
- Working branch: `PLASTIC_DEPTH`
- Pull request: #29, draft, base `HYPERBLOCK_LOOP_ENHANCEMENTS`
- Baseline head: `97612234648848e0666b009bdc15ee6aaa2f2560`
- Latest completed phase commit: `cc787d1d72ab5a8d618e0a7a21486e8cdb539fbf`
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

## Completed phase evidence

### P2: pure gauge transform

Commits:

- `7eb74f4458d0ca81d4f5e0ab30c6c932308dfb6a` — gauge-preserving Chebyshev chart transform
- `cc787d1d72ab5a8d618e0a7a21486e8cdb539fbf` — property and stored-dtype tests

Implementation:

- `sheet/plastic_depth_gauge.py`
- Exact affine composition is built in ordinary Chebyshev coefficient space by recurrence.
- The result is converted into the fixed QR-stabilised THOG coefficient coordinates.
- No Vandermonde/interpolation approximation is used.

Tests:

```bash
pytest -q tests/test_plastic_depth_gauge.py
# 33 passed

pytest -q \
  tests/test_plastic_depth_gauge.py \
  tests/test_plastic_depth.py \
  tests/test_plastic_depth_basis_cache.py \
  tests/test_plastic_depth_interfaces.py \
  tests/test_sheet_stage6_trainer.py \
  tests/test_sheet_stage6_wandb.py
# 72 passed
```

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
- [x] P2: pure float64 Chebyshev affine re-expression kernel and property tests
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

Implement P3: replace maximum-lattice geometry with active-prefix gaps plus one derived N+1 probe point. Prove that inactive storage slots cannot affect active coordinates, basis rows or outputs. Do not yet connect count transitions to coefficient re-expression.

## Work-in-progress files

- `sheet/plastic_depth.py`
- `tests/test_plastic_depth.py`
- likely interface tests that currently assert evenly distributed maximum-lattice ranks

## Known unresolved decisions

- Runtime re-gauge cannot mutate coefficient parameters before backward through a graph that used the old parameterisation. The likely safe implementation is to select the count for the current optimiser update, train using the frozen old chart, and atomically commit/re-gauge after `optimizer.step()` for the next update. This needs explicit integration tests and an as-built specification note.
- Old v0.1 PLASTIC DEPTH checkpoint migration may be ambiguous. Prefer an explicit rejection over silent reinterpretation unless a deterministic conversion is proven.
- Repeated re-gauges must be stress-tested for accumulated stored-dtype error.
- Count-dependent residual-depth scaling remains a separate architectural limitation and is not part of the gauge transform.

## Takeover instructions

1. Read this file and the v0.3 requirements specification.
2. Fetch `PLASTIC_DEPTH` and confirm the current branch head.
3. Run the focused tests above and compare failures with the recorded baseline.
4. Continue only the single task named under **Current exact task**.
5. After each phase, update this file with commits, tests, failures and the next exact task.
