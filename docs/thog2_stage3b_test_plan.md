# THOG2 Stage 3b Test Plan

Stage 3b routes the actual `SheetGPT` attention path through the Stage 3 semantic materialization seam. It must preserve the legacy CHEBY_SHEET_COL behaviour exactly.

## Scope

Stage 3b covers model-path adoption of the semantic materializer only.

Allowed implementation work:

- construct a `LegacySheetColMaterializer` inside `SheetGPT`
- replace direct packed QKV materialization in `SheetGPT._attention`
- use the semantic materializer's explicit packed-boundary helpers:
  - `reconstructed_attention_input_weight(layer_index)`
  - `reconstructed_attention_input_bias(layer_index)` when bias is enabled
- keep attention-output and MLP materialization unchanged for now
- expose enough evidence that the model path is actually using the materializer

Explicitly out of scope:

- new coefficient layout
- CURVE, MLP_BLOCK, HEAD_AWARE_BLOCK, BLOCK, or DCT materialization
- separate Q/K/V coefficients
- changing optimizer groups or parameter counts
- checkpoint conversion
- performance optimization

## Required tests

### Model wiring tests

Verify:

- `SheetGPT` owns a `semantic_materializer` attribute
- that attribute is a `LegacySheetColMaterializer`
- it wraps the model's actual `trajectory` object, not a copy
- the materializer's reconstructed QKV tensors exactly match the legacy packed trajectory tensors

### Attention-path usage tests

Use a spy/wrapper materializer to verify `_attention(...)` calls:

- `reconstructed_attention_input_weight(layer_index)` exactly once
- `reconstructed_attention_input_bias(layer_index)` exactly once when bias is enabled
- no direct `trajectory.materialize("attention_input_weight", ...)` call from `_attention`
- no direct `trajectory.materialize_vector("attention_input_bias", ...)` call from `_attention`

For `bias=False`, verify:

- QKV weight still comes from the semantic materializer
- QKV bias is not requested

### Behaviour equivalence tests

Construct two models with identical config and copied state:

- current semantic-routed model
- reference model that computes attention using the legacy packed trajectory tensors directly

Verify for a fixed input:

- `_attention(...)` output matches exactly or within float32 roundoff
- full forward logits match
- full loss matches
- backward succeeds
- gradients exist for trajectory coefficients used by attention

### Optimizer and report invariance tests

Verify:

- total persistent parameters unchanged
- sheet coefficient count unchanged
- optimizer parameter group names unchanged
- no new trainable parameters are added by the semantic materializer
- compact-state violation check remains empty

### Existing-stage regression tests

Run Stage 0, Stage 1, Stage 2, and Stage 3 tests after Stage 3b.

## Commands

`python -m unittest tests.test_stage3b_model_semantic_attention`

`python -m unittest tests.test_stage3_semantic_materialization`

`python -m unittest tests.test_stage2_basis_kernel`

`python -m unittest tests.test_stage1_compact_identity`

`python -m unittest tests.test_stage0_legacy_sheet_col_baseline`

`python -m unittest discover tests`

## Exit gate

Stage 3b is complete when the actual model attention path demonstrably uses the semantic materializer, all equivalence tests pass, and full discovery remains green.
