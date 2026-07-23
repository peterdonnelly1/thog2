# THOG2 Lapped Cosine Basis Test Plan v0.2

## 1. Purpose

Validate the `lapped_cosine` basis as a deterministic, orthonormal, locally supported basis-family plugin without changing the behaviour of the existing Chebyshev, DCT, Haar, geometry, training, checkpoint, or wrapper paths.

## 2. Initial contract

- Canonical family: `lapped_cosine`
- Artifact tag: `LAPPED_COSINE`
- Version: `lapped_cosine_dc_preserving_orthonormal_v1`
- Transform: weighted balanced-Haar coarse block means plus block-local orthonormal DCT-II detail atoms, followed by a DC-preserving orthogonal cosine-domain boundary prefilter
- Boundary policy: finite, non-circular; the first and last axis regions never couple
- Coefficient ordering: exact global DC first, then balanced block-mean contrasts, then local cosine mode first across blocks
- Locality control: `lapped_cosine_window_length`, default `36`
- Overlap control: `lapped_cosine_overlap_fraction`, default and initially supported value `0.5`
- No global QR or other global re-orthogonalisation

## 3. Basis-kernel tests

1. Full bases are square and orthonormal for odd, even, small, and THOG-relevant axis lengths, including 144, 768, and 3072.
2. Truncated bases match the corresponding full-basis prefixes to float64 roundoff tolerance; repeated construction at the same order remains bitwise deterministic.
3. Every retained column has unit norm and retained columns are mutually orthogonal.
4. The first column is exactly the normalized global constant for every axis length and locality setting.
5. Construction is deterministic across repeated CPU calls and runtime casts.
6. Local cosine detail atoms have bounded contiguous support consistent with the configured locality scale.
7. No atom couples the first and last axis regions except the intentional global DC/coarse backbone.
8. Prefix ordering gives every block its mean degree of freedom before allocating local detail modes, then allocates each detail mode evenly across eligible blocks.
9. The full basis reconstructs arbitrary vectors to numerical tolerance.
10. Invalid window lengths, overlap values, versions, dtypes, and dimensions fail explicitly.
11. Non-lapped basis kernels reject lapped-cosine-specific options rather than silently consuming them.

## 4. Initialization and integration tests

1. Existing first-column initialization assumptions remain valid: depth matrices initialize across every logical layer rather than only the first local region.
2. Layer-normalization weights materialize as exact ones and biases as exact zeros at initialization.
3. Standard and residual matrix families retain their intended materialized initialization scale.
4. The basis plugin registry exposes the canonical family, aliases, version, metadata, and artifact tag.
5. Python CLI and primary wrappers expose `lapped_cosine` and both controls.
6. Wrapper dry runs propagate the controls and include them in run identity.
7. Training configuration, model configuration, basis cache keys, checkpoint compatibility signatures, manifests, and telemetry preserve both controls.
8. Changing either control changes compact identity and prevents an incompatible resume.
9. A tiny CPU depth-model forward/backward/update completes with finite loss and gradients.
10. Materialised values and direct-value access agree.
11. Checkpoint save/load round-trips the basis family and controls.

## 5. Regression tests

1. Existing Chebyshev, DCT, and Haar basis matrices remain byte-for-byte unchanged for representative dimensions and orders.
2. Existing basis aliases, versions, artifact tags, and registry order remain unchanged ahead of the appended family.
3. Existing dense and compact default configurations produce unchanged compatibility signatures except for explicitly versioned schema additions required by this feature.
4. Existing wrapper grids continue to expand basis, order, learning-rate, and batch axes correctly.
5. Full repository Python regression suite passes.
6. All top-level shell scripts pass `bash -n`.
7. Python sources pass `compileall`.

## 6. Acceptance criteria

The feature is accepted only if:

- exact orthonormality, global-DC, initialization, and locality tests pass;
- the full CPU regression suite passes;
- no existing basis or geometry regression changes unexpectedly;
- the CI source snapshot and test log are retained;
- the implementation is committed only after those checks pass.
