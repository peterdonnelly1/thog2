# THOG2 Stage 1 Test Plan

Stage 1 adds the compact-geometry identity surface and metadata plumbing. It must not change current materialization behaviour.

## Scope

Stage 1 covers config identity, preset expansion, metadata visibility, checkpoint identity, and resume mismatch rejection.

Stage 1 must not implement CURVE, MLP_BLOCK, HEAD_AWARE_BLOCK, BLOCK, DCT, or any new materialization path. The only accepted resolved geometry in Stage 1 is the current legacy CHEBY_SHEET_COL path, exposed through the new identity surface as a legacy compatibility preset.

## Identity surface under test

New enum-like values:

- `geometry_preset`: `legacy_sheet_col`, `curve`, `mlp_block`, `block`, or unset.
- `attention_geometry`: `legacy_sheet_col`, `curve`, `head_aware_block`, or unset.
- `mlp_geometry`: `legacy_sheet_col`, `curve`, `mlp_block`, or unset.
- `basis_family`: `chebyshev`, `dct`, or unset.

Stage 1 accepts only the current legacy materialization identity:

- requested preset unset or `legacy_sheet_col`
- resolved attention geometry `legacy_sheet_col`
- resolved MLP geometry `legacy_sheet_col`
- resolved basis family `chebyshev`

The future preset names should parse and expand in pure unit tests, but model construction must reject them until their materialization stages exist.

## Required tests

### Preset expansion tests

Verify a single central function expands presets deterministically:

- unset compact request resolves to legacy CHEBY_SHEET_COL compatibility identity.
- `legacy_sheet_col` resolves to attention `legacy_sheet_col`, MLP `legacy_sheet_col`, basis `chebyshev`.
- `curve` resolves to attention `curve`, MLP `curve`.
- `mlp_block` resolves to attention `curve`, MLP `mlp_block`.
- `block` resolves to attention `head_aware_block`, MLP `mlp_block`.

### Explicit override tests

Verify explicit module selectors override preset defaults only when the combination is coherent:

- `geometry_preset=curve`, `mlp_geometry=mlp_block` resolves as attention `curve`, MLP `mlp_block`.
- `geometry_preset=mlp_block`, `attention_geometry=head_aware_block` resolves as attention `head_aware_block`, MLP `mlp_block`.
- `basis_family=dct` is retained in the resolved identity but rejected by model construction in Stage 1.

### Invalid combination tests

Verify invalid values are rejected with useful errors:

- unknown preset
- unknown attention geometry
- unknown MLP geometry
- unknown basis family
- incompatible attention geometry under legacy-only model construction
- incompatible MLP geometry under legacy-only model construction
- unsupported basis family under legacy-only model construction

DENSE config paths must reject compact geometry fields unless they are unset or explicitly conventional. If DENSE support is not in a central config object yet, Stage 1 must add a pure validator and test it.

### Metadata visibility tests

Verify model/config metadata exposes:

- requested geometry preset
- requested module overrides
- resolved attention geometry
- resolved MLP geometry
- basis family
- basis version
- compact materialization version
- model dimensions
- head metadata: `n_head`, `head_dim`, and Q/K/V role ranges
- resolved orders: depth order and base row order

Verify metadata appears in:

- `SheetGPTConfig.to_dict()`
- `SheetGPT.parameter_report()` or equivalent model metadata report
- training/checkpoint payload metadata

### Checkpoint metadata match tests

Save a tiny checkpoint and resume/load with matching metadata. Verify resume succeeds and preserves the compact identity fields exactly.

### Checkpoint metadata mismatch tests

Mutate one compact-identity field at a time in a saved payload and verify resume/load hard-fails:

- missing compact identity metadata
- changed geometry preset
- changed attention geometry
- changed MLP geometry
- changed basis family
- changed basis version
- changed `n_head`
- changed `n_embd`
- changed `depth_order`
- changed `base_row_order`

The failure should happen before any state dict reinterpretation.

### Materialization invariance tests

Reuse Stage 0 fixture values and verify that constructing the current model through the new legacy identity produces exactly the same selected materialized tensors.

## Commands

`python -m unittest tests.test_stage1_compact_identity`

`python -m unittest tests.test_stage0_legacy_sheet_col_baseline`

`python -m unittest discover tests`

## Exit gate

Stage 1 is complete when all new identity/metadata tests pass, Stage 0 still passes, and full test discovery passes. No test should require CURVE, MLP_BLOCK, HEAD_AWARE_BLOCK, BLOCK, or DCT materialization.
