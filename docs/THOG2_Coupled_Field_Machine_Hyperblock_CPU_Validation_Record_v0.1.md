# THOG2 Coupled Field Machine HYPERBLOCK

**CPU Validation and Regression Record**  
**Version 0.1 — 1 August 2026**

## Validated revision

- Branch: `COUPLED_FIELD_MACHINE_HYPERBLOCk`
- Validated head: `9a76da9db9f97927b273099274a4f475604c205a`
- Exact comparison base: `MODIFIED_FAST_DISCARD_FALSE_CASE` at `c38ef70c850a55a2f12556c72ee64b259bb1fbae`
- Pull request: `#26`
- GitHub Actions run: `30697296117`
- Workflow artifact: `8817742711`
- Evidence digest: `sha256:4fdc36d31718ccf5ece5a1f56eeddb6d77985a8fb113ba2b82ba346259b632c5`

The workflow used Ubuntu 24.04, CPython 3.11.15, PyTorch 2.13.0 and pytest 9.1.1.

## Dedicated HYPERBLOCK CPU suite

The dedicated job completed successfully.

```text
49 passed, 1 skipped in 11.53s
```

The skipped test was the CUDA-only bfloat16 one-update smoke because the GitHub runner had no CUDA device. CPU bfloat16 one-update tests ran in both retained-materialisation and ephemeral-materialisation modes.

The job also passed:

```text
python -m compileall -q sheet tests run_thog2_owt_core.py
bash -n train_OWT.sh train_OWT_core.sh
```

## Legacy regression comparison

Every legacy `tests/test_*.py` file outside the HYPERBLOCK suite was run in an isolated process on the HYPERBLOCK head. Any non-passing head file was rerun against the exact base commit in the same runner and dependency environment.

```text
files checked:          125
head passing files:     106
head non-passing files: 19
new regressions:        0
```

All 19 non-passing head files failed on the same test nodes as the base, with the same failure/error counts. They are therefore existing base or environment failures, not HYPERBLOCK regressions:

```text
tests/test_basis_family_plugin_registry.py
tests/test_console_progress_pretty_rows.py
tests/test_direct_thog_mlp_application.py
tests/test_geometry_registry_phase1.py
tests/test_jpeg_like_v1_wrappers.py
tests/test_karitane_restart_launcher.py
tests/test_layer_dropout_wrappers.py
tests/test_optimizer_wrapper.py
tests/test_picton_cli_wrapper_contract.py
tests/test_picton_wrapper_defaults_and_nonfinite_policy.py
tests/test_sheet_stage3_compatibility.py
tests/test_sheet_stage6_runner_scripts.py
tests/test_stage8_mlp_channel_order_and_wrapper_loops.py
tests/test_stage8_training_instrumentation_selector.py
tests/test_stage8_wrapper_shell_syntax.py
tests/test_stage8_wrappers_and_run_config.py
tests/test_train_owt_public_wrapper.py
tests/test_vectorise_per_head_materialisation.py
tests/test_wrapper_learning_rate_and_batch_grids.py
```

Notably, the retained-materialisation regression files all passed on the head in this environment:

```text
tests/test_modified_fast_discard_false.py
tests/test_modified_fast_discard_false_distributed.py
tests/test_modified_fast_discard_false_optimizer_layout.py
```

## Source-history audit

The automated no-silent-deletion audit checked all existing Python and shell files modified by HYPERBLOCK integration:

```text
run_thog2_owt_core.py
sheet/model.py
sheet/run_config.py
sheet/training_config.py
sheet/training_model_factory.py
train_OWT_core.sh
```

Result:

```text
violation count: 0
```

## CPU conclusion

The validated revision satisfies the CPU correctness and regression requirements for the non-breathing HYPERBLOCK implementation:

- dedicated mathematical and integration tests pass;
- all legacy test files have been compared against the exact branch base;
- no new legacy regression was found;
- modified nanoGPT/THOG source history is preserved under the project rule.

This conclusion is limited to CPU correctness and compatibility. It does not establish CUDA correctness, GPU memory behaviour, materialisation speed, training quality, convergence, or superiority over dense or DEPTH controls.

## Remaining empirical gate

Before merge into `master`, run a small CUDA smoke on the intended local hardware, covering at least:

1. bfloat16 on scruffy with both retained and ephemeral materialisation;
2. forward, backward and one optimizer update;
3. checkpoint save and resume;
4. peak VRAM and materialisation timing;
5. confirmation that retained tensors are released after projection;
6. a short loss-decrease sanity run before any long experiment.
