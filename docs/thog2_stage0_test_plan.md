# THOG2 Stage 0 Test Plan

Stage 0 is a guard rail stage. It must not refactor geometry code. It captures the current legacy CHEBY_SHEET_COL behaviour before later stages move the implementation behind new abstractions.

The tests should catch accidental changes to coefficient shapes, parameter counts, basis identity, current head packing assumptions, and deterministic materialization values.

## Scope

The scope is the current compact Sheet path only.

Stage 0 should add fixture data and tests. It should not add CURVE, MLP_BLOCK, HEAD_AWARE_BLOCK, BLOCK, or DCT behaviour.

## Fixture

Add `tests/fixtures/stage0_legacy_sheet_col_fixture.json`.

The fixture uses a tiny deterministic config: `n_layer=4`, `n_head=2`, `n_embd=16`, `depth_order=3`, `base_row_order=8`, and float32 runtime bases.

The fixture records family shapes, coefficient shapes, parameter counts, basis hashes, head packing ranges, and selected materialized tensor values from a fixed seed.

## Required tests

Family and parameter audit:

- Verify the current family list and stable order.
- Verify `output_rows`, `row_width`, `row_order`, `coefficient_shape`, `sheet_parameters`, and `dense_equivalent_parameters` for each family.
- Verify public `parameter_report()` totals.

Basis audit:

- Verify the current CHEBY basis version.
- Verify deterministic basis hashes for the tiny fixture's depth basis and distinct row bases.
- Verify fixed basis buffers are non-persistent.

Materialization audit:

- Verify selected materialized tensors for layer 2 under the fixed seed.
- Compare shapes, first flattened values, sums, and norms.
- Verify `direct_value()` agrees with `materialize()` for sampled matrix coordinates.

Attention packing audit:

- Verify the fused attention input shape and Q/K/V row ranges.
- Verify per-head row ranges inside each Q/K/V role.
- Verify attention-output input-head column ranges.
- Verify legacy SHEET_COL compacts across attention-output input-head boundaries but does not compact across Q/K/V output rows.

## Commands

`python -m unittest tests.test_stage0_legacy_sheet_col_baseline`

`python -m unittest discover tests`

## Exit gate

Stage 0 is complete when the new Stage 0 test passes and existing test discovery still passes.

Any later refactor that changes legacy CHEBY_SHEET_COL behaviour must either preserve this fixture or deliberately replace it with an explained compatibility-breaking decision.
