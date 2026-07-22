# THOG2 Stage 2 Test Plan

Stage 2 introduces a basis-kernel interface while preserving the current CHEBY materialization exactly.

## Scope

Stage 2 covers basis construction abstraction only. It must not change family geometry, coefficient layout, initialization, materialization algebra, checkpoint schema semantics, or training behaviour.

Allowed implementation work:

- introduce a `BasisKernel` interface
- implement the current Chebyshev first-kind plus deterministic QR basis as the `chebyshev` kernel
- add a basis-kernel registry and resolver
- make cache keys include basis family and kernel version
- route `BasisCache`, `BasisOwner`, and `SheetTrajectory` through the kernel seam
- keep existing public helper functions as compatibility wrappers

Explicitly out of scope:

- DCT implementation
- CURVE, MLP_BLOCK, HEAD_AWARE_BLOCK, or BLOCK materialization
- semantic family refactor
- parameter-count changes
- checkpoint conversion
- basis persistence in checkpoints

## Required tests

### Kernel registry tests

Verify:

- `get_basis_kernel("chebyshev")` returns the CHEBY kernel
- aliases used by Stage 1 identity surface resolve deterministically
- unknown basis family fails with a useful `ValueError`
- the CHEBY kernel exposes stable `basis_family`, `basis_version`, and coordinate policy metadata
- requesting `dct` fails in Stage 2 because DCT is identity-only until its own implementation stage

### CHEBY equivalence tests

For several `(sample_count, order)` pairs, verify:

- `build_stabilized_basis(...)` still equals `get_basis_kernel("chebyshev").build(...)`
- existing Stage 1 basis hashes still match for the Stage 0 fixture shapes
- the one-point axis remains centred at 0.0
- coordinate endpoints, recurrence values, deterministic QR signs, and orthonormality tolerances remain unchanged

### Cache-key tests

Verify:

- repeated requests for the same family/version/sample/order/dtype/device return the identical cached tensor object
- changing runtime dtype changes the cache entry
- changing device changes the cache key
- changing basis family changes the cache key
- changing basis version changes the cache key
- the cache length reflects those distinctions

### BasisOwner non-persistence tests

Verify:

- registered bases are non-trainable
- registered bases appear as module buffers
- registered bases do not appear in `state_dict()`
- no trainable parameters are created by `BasisOwner`
- the owner can pass basis family and version to the cache

### SheetTrajectory integration tests

Verify a tiny legacy CHEBY trajectory:

- uses the CHEBY kernel internally
- still builds the same distinct row bases as before
- `persistent_basis_keys()` remains empty
- selected materialized tensors still match the Stage 0 fixture exactly
- `direct_value(...)` still matches `materialize(...)`
- family reports and parameter counts are unchanged

### Negative validation tests

Verify useful failures for:

- non-positive sample count
- non-positive order
- order greater than sample count
- non-floating dtype
- unknown basis family
- empty basis version
- duplicate `BasisOwner.add_basis(...)` name

## Commands

`python -m unittest tests.test_stage2_basis_kernel`

`python -m unittest tests.test_stage1_compact_identity`

`python -m unittest tests.test_stage0_legacy_sheet_col_baseline`

`python -m unittest discover tests`

## Exit gate

Stage 2 is complete when the new basis-kernel tests pass, Stage 0 and Stage 1 still pass, and full discovery passes. The Stage 0 materialization fixture must remain unchanged.
