# DENSE Snapshot Baselining v0.2 - Implementation Tasks

Branch: `initialisation_baselining`

Baseline: `master` at `b9b09b1e125633e451bae60d9d2f9ea3f0721987`

## Status

- [x] Extract the v0.2 requirements and identify the model-construction boundary.
- [x] Confirm pure DEPTH uses the existing QR-stabilised Chebyshev basis and explicit Q/K/V families.
- [x] Create the implementation branch from the promoted `master`.
- [x] Implement the versioned immutable DENSE snapshot payload, manifests, hashes, atomic writer, and exact-path loader.
- [x] Preserve CPU and CUDA RNG state across snapshot capture and restore the paired post-initialisation boundary for consuming runs.
- [x] Implement physical-source compatibility validation independently of dense versus DEPTH storage representation.
- [x] Implement one Chebyshev coefficient-fitting/materialisation path for B Compressor-baselined DENSE and C Compact Run.
- [x] Direct-copy embeddings, final normalization, LayerNorms, biases, and other families outside the active DEPTH matrix plan.
- [x] Add CLI options and reject save/load, resume/fork, residual-init, incomplete, non-Chebyshev, compressed-vector, and HYPERBLOCK conflicts before model construction.
- [x] Emit and persist lifecycle role, snapshot identity, mapping fingerprint, numerical diagnostics, and step-zero manifest metadata.
- [x] Add focused unit and integration tests covering v0.2 acceptance cases.
- [x] Run focused DENSE snapshot tests.
- [x] Run existing DENSE, DEPTH, geometry, optimizer, instrumentation, resume, and fork regression tests; classify baseline and environment failures.
- [x] Review the final diff, update this task list and the implementation log, commit, and push to `origin/initialisation_baselining`.

## Scope guard

v1 is Chebyshev-only and pure DEPTH-only. DCT, Haar, lapped cosine, nonlinear, multi-axis, block, HYPERBLOCK, and compressed LayerNorm/bias support are deliberately excluded.
