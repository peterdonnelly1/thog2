# THOG2 Stage 4b Test Plan

Stage 4b removes the temporary Stage 4 selector-aware model factory and makes the main training model factory responsible for forwarding compact selector fields into `SheetGPTConfig`.

## Scope

Allowed implementation work:

- update `sheet.training_model_factory.build_training_model(...)` so curve `TrainingConfig` constructs a `TrainingSheetGPT` backed by `CurveTrajectory`
- route `Stage4Trainer` back through `sheet.training_model_factory`
- delete the temporary `sheet/stage4_training_model_factory.py` module
- keep `TrainingConfig.model_arguments()` behaviour stable if direct edit remains blocked

Explicitly out of scope:

- new materialization geometry
- changing CHEBY_CURVE coefficient layout
- changing checkpoint schema
- changing legacy `legacy_sheet_col` behaviour

## Verbose test naming

Run this test with verbose unittest output:

`python -m unittest -v tests.test_stage4b_training_factory_cleanup`

The test method names are intentionally descriptive so the cleanup checks are visible as they run.

## Required tests

### Main factory selector forwarding tests

Verify:

- `sheet.training_model_factory.build_training_model(curve_config)` returns a `TrainingSheetGPT`
- the returned model uses `CurveTrajectory`
- the returned model identity remains curve
- the returned model has no legacy packed `attention_input_weight` coefficient

### Legacy preservation tests

Verify:

- `build_training_model(legacy_config)` still returns a `TrainingSheetGPT`
- the returned model still uses legacy `SheetTrajectory`
- legacy packed `attention_input_weight` coefficient still exists

### Stage4Trainer factory cleanup tests

Verify:

- `Stage4Trainer(curve_config, ...)` uses `CurveTrajectory`
- `Stage4Trainer(legacy_config, ...)` uses legacy `SheetTrajectory`
- checkpoint round-trip still works after routing back through the main factory

### Temporary module removal test

Verify:

- `sheet.stage4_training_model_factory` is no longer importable

## Commands

`python -m unittest -v tests.test_stage4b_training_factory_cleanup`

`python -m unittest -v tests.test_stage4_curve_materialization`

`python -m unittest tests.test_stage3b_model_semantic_attention`

`python -m unittest tests.test_stage3_semantic_materialization`

`python -m unittest tests.test_stage2_basis_kernel`

`python -m unittest tests.test_stage1_compact_identity`

`python -m unittest tests.test_stage0_legacy_sheet_col_baseline`

`python -m unittest discover tests`

## Exit gate

Stage 4b is complete when the main factory constructs curve and legacy models correctly, the temporary factory is gone, Stage 4 still passes, and full discovery remains green.
