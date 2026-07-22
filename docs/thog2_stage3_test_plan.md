# THOG2 Stage 3 Test Plan

Stage 3 introduces the semantic materialization seam. It must preserve the current legacy CHEBY_SHEET_COL coefficient layout and materialized tensors exactly.

## Scope

Stage 3 covers semantic family naming, attention-head partition metadata, and a compatibility materializer over the existing legacy trajectory.

Allowed implementation work:

- define semantic repeated-weight family names:
  - `attention_query_weight`
  - `attention_key_weight`
  - `attention_value_weight`
  - `attention_output_weight`
  - `mlp_expansion_weight`
  - `mlp_contraction_weight`
- define semantic attention bias names for the legacy packed bias split:
  - `attention_query_bias`
  - `attention_key_bias`
  - `attention_value_bias`
- introduce a materialization adapter for the current legacy `SHEET_COL` trajectory
- expose Q/K/V by slicing the existing `attention_input_weight` rows
- expose Q/K/V bias by slicing the existing `attention_input_bias` vector
- expose MLP and attention-output families without changing their current materialization
- expose explicit head partition metadata for Q/K/V and attention-output input columns
- provide boundary pack helpers for the current nanoGPT-style packed QKV tensors

Explicitly out of scope:

- new coefficient layout
- CURVE, MLP_BLOCK, HEAD_AWARE_BLOCK, BLOCK, or DCT materialization
- separate persistent coefficients for Q/K/V
- any parameter-count change
- checkpoint conversion
- replacing the legacy trajectory internals

## Required tests

### Semantic family registry tests

Verify:

- semantic family names are complete and stable
- Q/K/V semantic families map to row slices of legacy `attention_input_weight`
- Q/K/V bias families map to slices of legacy `attention_input_bias`
- MLP and attention-output semantic names map directly to the legacy family names
- unknown semantic family names fail with useful errors

### Shape and head metadata tests

For a small fixture model, verify:

- query/key/value weight shapes are `(n_embd, n_embd)`
- attention-output weight shape is `(n_embd, n_embd)`
- MLP expansion shape is `(4*n_embd, n_embd)`
- MLP contraction shape is `(n_embd, 4*n_embd)`
- query/key/value bias shapes are `(n_embd,)`
- head_dim is `n_embd // n_head`
- Q/K/V head row ranges match the Stage 0 fixture
- attention-output input-head column ranges match the Stage 0 fixture

### Legacy materialization equivalence tests

Verify:

- semantic Q/K/V weights concatenate exactly to the legacy `attention_input_weight`
- semantic Q/K/V biases concatenate exactly to the legacy `attention_input_bias`
- semantic attention-output, MLP expansion, and MLP contraction match their legacy tensors exactly
- selected Stage 0 fixture first values, sums, and norms remain unchanged
- `direct_value(...)` through the semantic adapter matches the legacy direct value for sampled entries

### Boundary helper tests

Verify:

- `packed_attention_input_weight(...)` returns the exact legacy packed weight
- `packed_attention_input_bias(...)` returns the exact legacy packed bias
- helper names make the packing boundary explicit; packing is not hidden in semantic-family definitions

### Forward/backward preservation tests

Construct two tiny models with the same seed and config:

- the existing model path
- a reference path using the semantic materializer to reproduce the packed QKV tensors

Verify:

- forward logits match for a fixed input
- loss matches for fixed targets
- backward succeeds
- optimizer parameter coverage remains unchanged

If the full model is not yet routed through the materializer in this stage, the test must still prove that the adapter reproduces the tensors consumed by the existing model path exactly.

### Negative validation tests

Verify useful failures for:

- unknown semantic family
- invalid layer index
- requesting vector materialization from a matrix family
- requesting matrix materialization from a bias family
- invalid direct-value row or column

## Commands

`python -m unittest tests.test_stage3_semantic_materialization`

`python -m unittest tests.test_stage2_basis_kernel`

`python -m unittest tests.test_stage1_compact_identity`

`python -m unittest tests.test_stage0_legacy_sheet_col_baseline`

`python -m unittest discover tests`

## Exit gate

Stage 3 is complete when semantic materialization tests pass, Stage 0/1/2 tests still pass, and full discovery passes. Materialized legacy tensors and parameter counts must remain unchanged.
