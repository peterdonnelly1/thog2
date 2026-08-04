# PLASTIC DEPTH implementation log

Last updated: 2026-08-04 16:08 AEST

## Repository state

- Repository: `peterdonnelly1/thog2`
- Working branch: `PLASTIC_DEPTH`
- Pull request: #29, draft, base `HYPERBLOCK_LOOP_ENHANCEMENTS`
- Baseline head: `97612234648848e0666b009bdc15ee6aaa2f2560`
- Local implementation commit: `dedf942` (`Implement adaptive PLASTIC DEPTH controller and gauge transitions`)
- Requirements target: PLASTIC DEPTH specification v0.3
- Implementation plan: THOG2 PLASTIC DEPTH Implementation and Testing Plan v0.1

## Baseline validation

Focused baseline result: 48 passed, 2 unrelated pre-existing ANSI-yellow console failures. Do not treat those two failures as PLASTIC regressions.

## Completed implementation

- Exact float64 affine Chebyshev change-of-chart kernel; no interpolation/Vandermonde approximation.
- Active-prefix geometry with N active points plus one derived N+1 point; inactive capacity cannot influence active coordinates, basis rows, gradients or output.
- Fixed QR reference count independent of maximum capacity:
  - learned: `max(depth_order, initial_active_count + 1)`
  - fixed: `max(depth_order, fixed_active_count)`
- Shared first-microstep N-1/N/N+1 transformer chain with detached candidate heads and one selected grad-bearing head.
- Selected count used for all remaining accumulation microsteps.
- Add/subtract commit and exact coefficient re-expression after stock AdamW steps.
- Stock AdamW moment migration:
  - first moment: inverse-transpose covector transform;
  - diagonal second moment: squared-transform approximation;
  - logged full-state reset fallback for non-finite/singular/ill-conditioned transforms.
- Newly activated conventional and geometry slots are initialised/reset before use.
- Paired MAD significance controller, per-count/per-direction history, minimum observations, lambda threshold and five-update brake.
- Online affine count-to-update-time model over timing EMAs; no extra timed probes.
- Universal CUDA allocator reserve plus upward precheck and rank-synchronised OOM fallback.
- Exact public `plastic__...` CLI/config names; internal sampling seed is not exposed.
- `_L_dyn_` only for learned count; fixed-count artifacts retain numeric L.
- Transition-only sampled scalar UI, two decimal places/scientific notation when required.
- Startup report and scalar W&B/TensorBoard diagnostics.
- Ambiguous v0.1 phantom-lattice checkpoints are rejected explicitly.
- GPU smoke script updated for fixed, lowest-loss, relative-wall-time and memory-budget modes.

## Current test evidence

```bash
pytest -q \
  tests/test_plastic_depth.py \
  tests/test_plastic_depth_gauge.py \
  tests/test_plastic_depth_basis_cache.py \
  tests/test_plastic_depth_interfaces.py \
  tests/test_sheet_stage6_trainer.py \
  tests/test_sheet_stage6_wandb.py \
  tests/test_console_model_summary.py \
  tests/test_console_progress_layout.py::ConsoleProgressLayoutTests::test_sampled_values_are_emitted_once_after_a_plastic_transition
# 98 passed
```

Static checks:

```bash
git diff --check
python -m compileall -q sheet run_thog2_owt.py run_thog2_owt_core.py
bash -n train_OWT.sh train_OWT_core.sh plastic_depth_gpu_smokes.sh
```

All passed at local commit `dedf942`.

## Phase checklist

- [x] P0: baseline and durable log
- [x] P1: uniform public names and artifact identity
- [x] P2: exact gauge kernel and property tests
- [x] P3: active-prefix geometry and capacity invariance
- [x] P4: atomic model-family re-gauge and diagnostics
- [x] P5: stock-AdamW moment migration with fallback
- [x] P6: shared inline N-1/N/N+1 probe
- [x] P7: MAD gate, brake, objectives, timing model and VRAM reserve
- [x] P8: checkpoint, console, startup and telemetry surfaces
- [ ] P9: broad/full CPU regressions and hosted CI
- [ ] P10: GPU smoke handoff and final download stanza
- [ ] Update requirements specification with as-built details and limitations

## Current exact task

Run broad/full CPU regression suites. Classify every failure against the recorded baseline, fix only genuine regressions, then push the implementation to `PLASTIC_DEPTH` and run hosted CI.

## Known issues and limitations to retain in the as-built specification

- AdamW diagonal second-moment migration is necessarily approximate because diagonal state cannot represent covariance induced by coefficient mixing. Stock PyTorch AdamW is not reimplemented.
- Extremely ill-conditioned gauge transforms use a logged moment-state reset fallback.
- Upward OOM is recoverable only before backward collectives; later selected-head/backward OOM remains fatal.
- Re-gauge preparation temporarily holds transformed coefficient copies and may create a memory spike.
- Regional `torch.compile` is not supported by inline learned-count probing.
- Count-dependent residual-addition scaling remains a separate architecture question.
- v0.1 globally normalised phantom-lattice checkpoints cannot be converted safely and are rejected.

## Takeover instructions

1. Fetch `PLASTIC_DEPTH` and read this file plus the v0.3 requirements specification.
2. Confirm whether local/remote includes implementation commit `dedf942` or its pushed GitHub equivalent.
3. Run the focused command above first.
4. Continue only the **Current exact task**.
5. Update this file after every stable phase with commit IDs, commands, results and the next exact task.
