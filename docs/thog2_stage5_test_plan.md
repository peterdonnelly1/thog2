# THOG2 Stage 5 Test Plan

Stage 5 implements CHEBY_MLP_BLOCK: attention remains CHEBY_CURVE while MLP repeated matrices use a depth x output-basis x input-basis block materialization.

## Scope

Allowed implementation work:

- accept `geometry_preset="mlp_block"` for `model_type="thog2_sheet"`
- accept explicit `attention_geometry="curve", mlp_geometry="mlp_block"`
- keep Q/K/V and attention-output matrices on CHEBY_CURVE
- move only these MLP matrices to block materialization:
  - `mlp_expansion_weight`
  - `mlp_contraction_weight`
- use coefficient layout `C[depth_order, output_order, input_order]` for MLP block matrices
- materialize by `W_l = output_basis @ (sum_p D[l,p] * C[p]) @ input_basis.T`
- keep all vectors and biases on the legacy vector sheet path
- keep packed QKV only as an attention compatibility boundary
- record `materialization_version="mlp_block_v1"`

Explicitly out of scope:

- HEAD_AWARE_BLOCK attention
- full BLOCK preset
- DCT basis
- token embeddings, layernorms, biases, lm_head compaction changes
- checkpoint conversion between legacy/curve/mlp_block
- performance optimization

## Verbose test naming

Run this test with verbose unittest output:

`python -m unittest -v tests.test_stage5_mlp_block_materialization`

The method names are intentionally descriptive so the important Stage 5 gates are visible while they run.

## Required tests

### Config and identity tests

Verify:

- `TrainingConfig(model_type="thog2_sheet", geometry_preset="mlp_block")` is accepted
- explicit `attention_geometry="curve", mlp_geometry="mlp_block"` is accepted
- `basis_family="chebyshev"` is accepted
- `basis_family="dct"` is still rejected
- `geometry_preset="block"` is still rejected
- compact identity reports `geometry_preset="mlp_block"`, `attention_geometry="curve"`, `mlp_geometry="mlp_block"`, `basis_family="chebyshev"`, and `materialization_version="mlp_block_v1"`

### Coefficient-shape tests

For the Stage 0 tiny fixture shape, verify:

- Q/K/V and attention-output remain curve coefficients shaped `(n_embd, n_embd, depth_order)`
- MLP expansion has block coefficients shaped `(depth_order, output_order(4*n_embd), input_order(n_embd))`
- MLP contraction has block coefficients shaped `(depth_order, output_order(n_embd), input_order(4*n_embd))`
- no packed `attention_input_weight` coefficient exists
- no curve-style MLP coefficient shape remains in MLP_BLOCK mode

### Manual algebra tests

With deterministic hand-filled coefficients, verify:

- MLP expansion materialization equals manual depth/output/input basis contraction
- MLP contraction materialization equals manual depth/output/input basis contraction
- direct-value lookup equals the materialized matrix entry
- Q/K/V packed boundary still equals concatenated attention curve semantic matrices

### Parameter-count tests

Verify:

- attention matrix coefficient count remains curve-style
- MLP matrix coefficient count equals block basis formula
- MLP_BLOCK matrix coefficient count is smaller than CHEBY_CURVE for the same tiny config
- total persistent parameter count is matrix coefficients + legacy vector coefficients + conventional parameters
- optimizer groups contain MLP block coefficients exactly once

### Model execution tests

For a tiny model:

- forward logits have correct shape
- loss is finite
- backward succeeds
- gradients reach attention curve coefficients
- gradients reach MLP block coefficients
- gradients still reach legacy vector coefficients where used
- compact-state violations remain empty

### Checkpoint compatibility tests

Verify:

- MLP_BLOCK checkpoint payload records `compact_identity.materialization_version == "mlp_block_v1"`
- loading an MLP_BLOCK checkpoint as MLP_BLOCK succeeds
- loading MLP_BLOCK as CURVE fails
- loading CURVE as MLP_BLOCK fails
- loading LEGACY as MLP_BLOCK fails

### Regression tests

Run Stage 0, Stage 1, Stage 2, Stage 3, Stage 3b, Stage 4, and Stage 4b tests after Stage 5.

## Commands

`python -m unittest -v tests.test_stage5_mlp_block_materialization`

`python -m unittest -v tests.test_stage4b_training_factory_cleanup`

`python -m unittest -v tests.test_stage4_curve_materialization`

`python -m unittest tests.test_stage3b_model_semantic_attention`

`python -m unittest tests.test_stage3_semantic_materialization`

`python -m unittest tests.test_stage2_basis_kernel`

`python -m unittest tests.test_stage1_compact_identity`

`python -m unittest tests.test_stage0_legacy_sheet_col_baseline`

`python -m unittest discover tests`

## Exit gate

Stage 5 is complete when CHEBY_MLP_BLOCK trains and checkpoints in tiny tests, older stages still pass, and full discovery remains green.
