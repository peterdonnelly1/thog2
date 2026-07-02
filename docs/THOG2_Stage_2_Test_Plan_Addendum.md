# THOG2 Stage 2 Test Plan Addendum

## Added test S2-19 — Epsilon approximation contract

**Reason added:** During Stage 2 planning, the question arose whether Chebyshev Sheet can demonstrate approximation of genuine sampled weights below an arbitrary error threshold `epsilon`. The original Test Plan established saturated sampled completeness but did not explicitly separate that exact finite-dimensional result from lower-order approximation claims.

**Method:**

1. Generate an arbitrary finite sampled sheet and project it through saturated orthonormal bases with `P=L` and `Q=C`.
2. Require reconstruction below a strict floating-point maximum-error epsilon.
3. Generate a lower-order smooth sheet directly inside a selected tensor-product basis span.
4. Project it through a larger but still sub-saturated basis and require reconstruction below epsilon.
5. Project an arbitrary sampled sheet through two nested sub-saturated geometries and require the higher-capacity projection not to increase Frobenius residual beyond numerical tolerance.
6. Confirm that a deliberately low-order projection of arbitrary sampled weights is not falsely reported as satisfying a small epsilon.

**Acceptance:**

- Saturated arbitrary sampled reconstruction is within the fixed epsilon.
- A sheet lying in the selected lower-order span is within the fixed epsilon.
- Projection residual does not increase when the tested tensor-product span is enlarged.
- The test and documentation make no universal low-order epsilon claim for arbitrary weight sheets.

**Status:** PASS in Stage 2 accepted workflow run `28563464826`.

This addendum preserves the original S2-04 saturated-completeness test and adds a distinct test rather than reinterpreting or weakening it.
