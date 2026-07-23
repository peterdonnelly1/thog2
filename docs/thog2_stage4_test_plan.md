# THOG2 Stage 4 Test Plan

Stage 4 implements CHEBY_CURVE: depth-only compact materialization for repeated semantic matrix families. It must preserve the legacy path and must not implement MLP_BLOCK, HEAD_AWARE_BLOCK, BLOCK, or DCT.

## Scope

Allowed implementation work:

- accept `geometry_preset="curve"` for `model_type="thog2_sheet"`
- route `SheetGPTConfig` and `TrainingConfig.model_arguments()` through resolved compact selectors
- add a curve trajectory for repeated semantic matrix families:
  - `attention_query_weight`
  - `attention_key_weight`
  - `attention_value_weight`
  - `attention_output_weight`
  - `mlp_expansion_weight`
  - `mlp_contraction_weight`
- use coefficient layout `C[output_row, input_col, depth_order]`
- materialize by depth contraction only: `W_l[o, i] = sum_p C[o, i, p] * D[l, p]`
- keep layernorm and bias families on the current legacy vector sheet path
- keep packed QKV only as a boundary tensor reconstructed from semantic Q/K/V curve matrices
- include curve identity metadata in checkpoints and compatibility signatures

Explicitly out of scope:

- DCT basis
- MLP_BLOCK
- HEAD_AWARE_BLOCK
- BLOCK preset
- row/column 2D block basis
- separate vector/bias curve implementation
- checkpoint conversion from legacy to curve
- performance optimization

## Descriptive test naming

Stage 4 tests intentionally use long method names and should be run with verbose unittest output:

`python -m unittest -v tests.test_stage4_curve_materialization`

That makes the major behavioural checks visible as they run.

## Required tests

### Config and identity tests

Verify:

- `TrainingConfig(model_type="thog2_sheet", geometry_preset="curve")` is accepted
- explicit `attention_geometry="curve", mlp_geometry="curve"` is accepted
- `basis_family="chebyshev"` is accepted
- `basis_family="dct"` is still rejected
- `geometry_preset="mlp_block"` is still rejected
- `geometry_preset="block"` is still rejected
- compact identity reports `geometry_preset="curve"`, `attention_geometry="curve"`, `mlp_geometry="curve"`, `basis_family="chebyshev"`, and `materialization_version="curve_v1"`
- `TrainingConfig.model_arguments()` passes the selector fields into `SheetGPTConfig`

### Curve coefficient shape tests

For a small fixture config, verify coefficient shapes exactly:

- Q/K/V: `(n_embd, n_embd, depth_order)`
- attention output: `(n_embd, n_embd, depth_order)`
- MLP expansion: `(4*n_embd, n_embd, depth_order)`
- MLP contraction: `(n_embd, 4*n_embd, depth_order)`

Verify there is no packed `attention_input_weight` coefficient parameter in curve mode.

### Direct materialization algebra tests

With deterministic hand-filled coefficients, verify for sampled entries:

- semantic matrix materialization equals manual depth contraction
- `direct_value(...)` equals the corresponding materialized matrix entry
- packed QKV boundary equals concatenated semantic Q/K/V matrices
- packed QKV bias still comes from the legacy vector sheet path

### Parameter-count tests

Verify:

- curve matrix coefficient count equals `depth_order * sum(output_rows * input_cols)` over the six semantic matrix families
- curve matrix coefficient count is larger than legacy SHEET_COL for the Stage 0 tiny fixture, as expected for depth-only curve
- total persistent parameter report equals curve matrix coefficients plus legacy vector coefficients plus conventional non-sheet parameters
- optimizer groups include curve coefficients exactly once
- legacy unused packed matrix coefficients are not present in curve mode

### Model execution tests

For a tiny model:

- forward logits have correct shape
- loss is finite
- backward succeeds
- curve Q/K/V coefficients receive gradients
- legacy vector coefficients still receive gradients where used
- compact state violations remain empty

### Checkpoint compatibility tests

Verify:

- curve checkpoint payload records `compact_identity.materialization_version == "curve_v1"`
- loading a curve checkpoint as curve succeeds
- loading a curve checkpoint as legacy `legacy_sheet_col` fails
- loading a legacy checkpoint as curve fails

### Regression tests

After Stage 4, run Stage 0, Stage 1, Stage 2, Stage 3, and Stage 3b tests. Legacy mode must remain unchanged.

## Commands

`python -m unittest -v tests.test_stage4_curve_materialization`

`python -m unittest tests.test_stage3b_model_semantic_attention`

`python -m unittest tests.test_stage3_semantic_materialization`

`python -m unittest tests.test_stage2_basis_kernel`

`python -m unittest tests.test_stage1_compact_identity`

`python -m unittest tests.test_stage0_legacy_sheet_col_baseline`

`python -m unittest discover tests`

## Exit gate

Stage 4 is complete when CHEBY_CURVE trains and checkpoints in tiny tests, all older-stage tests still pass, and full discovery remains green.
