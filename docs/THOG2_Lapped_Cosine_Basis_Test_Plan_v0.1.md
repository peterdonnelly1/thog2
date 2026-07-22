# THOG2 Lapped Cosine Basis Test Plan v0.1

## 1. Purpose

Validate the `lapped_cosine` basis as a deterministic, orthonormal, locally supported basis-family plugin without changing the behaviour of the existing Chebyshev, DCT, Haar, geometry, training, checkpoint, or wrapper paths.

## 2. Initial contract

- Canonical family: `lapped_cosine`
- Artifact tag: `LAPPED_COSINE`
- Version: `lapped_cosine_balanced_orthonormal_v1`
- Transform: balanced block-local orthonormal DCT-IV atoms followed by orthogonal sine/cosine boundary rotations
- Boundary policy: finite, non-circular; the first and last axis regions never couple
- Coefficient ordering: local mode first, balanced across all blocks before advancing to the next local mode
- Locality control: `lapped_cosine_window_length`, default `36`
- Overlap control: `lapped_cosine_overlap_fraction`, default and initially supported value `0.5`
- No global QR or other global re-orthogonalisation

## 3. Basis-kernel tests

1. Full bases are square and orthonormal for odd, even, small, and THOG-relevant axis lengths, including 144, 768, and 3072.
2. Truncated bases are exact prefixes of the corresponding full basis.
3. Every retained column has unit norm and retained columns are mutually orthogonal.
4. Construction is deterministic across repeated CPU calls and runtime casts.
5. Every atom has bounded contiguous support consistent with the configured locality scale.
6. No atom couples the first and last axis regions.
7. Prefix ordering allocates modes across blocks evenly; no block receives a second mode before every eligible block receives its first.
8. The full basis reconstructs arbitrary vectors to numerical tolerance.
9. Invalid window lengths, overlap values, versions, dtypes, and dimensions fail explicitly.
10. Non-lapped basis kernels reject lapped-cosine-specific options rather than silently consuming them.

## 4. Integration tests

1. The basis plugin registry exposes the canonical family, aliases, version, metadata, and artifact tag.
2. Python CLI and primary wrappers expose `lapped_cosine` and both controls.
3. Wrapper dry runs propagate the controls and include them in run identity.
4. Training configuration, model configuration, basis cache keys, checkpoint compatibility signatures, manifests, and telemetry preserve both controls.
5. Changing either control changes compact identity and prevents an incompatible resume.
6. A tiny CPU depth-model forward/backward/update completes with finite loss and gradients.
7. Materialised values and direct-value access agree.
8. Checkpoint save/load round-trips the basis family and controls.

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

- the exact orthonormality and locality tests pass;
- the full CPU regression suite passes;
- no existing basis or geometry regression changes unexpectedly;
- the CI source snapshot and test log are retained;
- the implementation is committed only after those checks pass.
