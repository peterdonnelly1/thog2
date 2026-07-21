# THOG2 Balanced Haar Basis — Test Plan v0.1

## 1. Purpose

This stage adds a balanced discrete Haar basis family to the existing fixed-basis registry.

Haar shall reuse the existing geometry and order controls unchanged. `P`, `Q`, `J`, `O`, `X`, and `Y` remain the retained basis orders for their respective compressed axes.

The implementation must not add Haar-specific branches to geometry, materialisation, caching, checkpointing, or training code.

## 2. Basis contract

The basis family shall use:

- family: `haar`;
- version: `haar_balanced_binary_orthonormal_v1`;
- artifact tag: `HAAR`;
- aliases: `balanced_haar` and `haar_balanced`;
- deterministic balanced binary interval partitioning;
- breadth-first, coarse-to-fine column ordering;
- closed-form orthonormal scaling;
- arbitrary positive sample counts, including non-powers of two;
- prefix stability across retained orders.

Column zero shall be the normalized constant vector. Every later column shall be the zero-mean, unit-norm contrast associated with one balanced binary interval split.

For an interval with `left_count`, `right_count`, and `total_count = left_count + right_count`, the contrast values shall be:

- left interval: `sqrt(right_count / (left_count * total_count))`;
- right interval: `-sqrt(left_count / (right_count * total_count))`;
- outside the interval: zero.

## 3. Required numerical tests

- Closed-form expected matrices for sample counts 1, 2, and 4, with exact structural zeros and rational entries and at most one float64 ULP for irrational entries.
- Full-basis orthonormality for odd and even sample counts.
- Full-basis orthonormality for representative THOG axis lengths, including 144.
- Deterministic repeated construction.
- Prefix stability: a lower-order basis must equal the leading columns of every higher-order basis for the same sample count.
- Correct output shape, dtype, device, finite values, and `requires_grad=False`.
- Rejection of zero or negative sample counts, zero or negative orders, orders exceeding sample count, non-floating runtime dtypes, and wrong basis versions.

## 4. Registry and identity tests

- `haar`, `balanced_haar`, `haar_balanced`, and the version string shall normalize to the canonical family.
- The registry shall report the correct version and artifact tag.
- Python CLI choices shall include `haar` without a Haar-specific CLI branch.
- Run artifact fragments shall use `HAAR`.
- Existing Chebyshev and DCT identities and artifact labels shall remain unchanged.
- Checkpoints shall record the Haar family and version.
- Cross-family checkpoint loads between Haar, Chebyshev, and DCT shall fail.

## 5. Geometry and training tests

A tiny finite training update shall run for Haar with every existing compact geometry preset:

- `legacy_sheet_col`;
- `depth`;
- `mlp_block`;
- `head_aware_block`;
- `full_block`.

No geometry or materialiser source file may contain a Haar-specific branch.

## 6. Wrapper tests

- Scruffy and Dreedle wrappers shall accept `-B haar` through registry validation.
- Dry-run artifact naming shall contain `HAAR`.
- Invalid unregistered basis names shall fail before training begins.
- Existing shell-syntax tests shall pass.

## 7. Regression execution

Run focused tests covering:

```bash
python -m unittest \
  tests.test_haar_basis_kernel \
  tests.test_haar_training_and_checkpoint \
  tests.test_basis_family_plugin_registry \
  tests.test_stage2_basis_kernel \
  tests.test_stage7_dct_basis_kernel \
  tests.test_stage7_dct_training_and_checkpoint \
  tests.test_stage8_wrapper_shell_syntax \
  tests.test_stage8_wrappers_and_run_config
```

Then run:

```bash
python -m unittest discover tests
```

CUDA-only tests may skip where CUDA is unavailable. Any other failure is blocking.

## 8. Acceptance criteria

The stage is accepted only when:

1. all focused and full regression tests pass;
2. Haar is orthonormal and prefix-stable for arbitrary tested lengths;
3. all existing geometries train with Haar unchanged;
4. checkpoint identity and cross-family rejection work correctly;
5. existing Chebyshev and DCT numerical behaviour remains unchanged;
6. `P/Q/J/O/X/Y` require no reinterpretation or new configuration fields;
7. the production implementation consists of one Haar family module plus one explicit registry entry.
