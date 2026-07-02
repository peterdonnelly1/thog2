# THOG2 Stage 2 Reference Model Record

**Stage:** 2 — Compact state and reference model  
**Branch:** `feature/sheet-stage-2-reference-model`  
**Status:** Provisional; acceptance evidence pending CI execution.

## Scope implemented

- one compact coefficient tensor per generated transformer-block family;
- shared non-persistent depth and row bases;
- one-layer materialisation through depth mixing and row-basis multiplication;
- nanoGPT-compatible matrix initialization with shared initial depth profiles;
- exact LayerNorm-one and bias-zero sheet initialization;
- semantic AdamW decay grouping;
- functional attention, MLP, residual, LayerNorm, embedding, and tied-head execution;
- sequential CPU SheetGPT reference forward path;
- next-token cross-entropy as the sole loss;
- compact-state and parameter reporting;
- sampled-sheet fitting, projection, reconstruction, and epsilon measurement;
- planned tests S2-01 through S2-18 plus added test S2-19 for the epsilon approximation contract.

## Epsilon approximation contract

For a sampled sheet `S`, discrete orthonormal depth basis `B_d`, and row basis `B_r`, the fitted coefficients are the orthogonal projection coefficients:

```text
A = transpose(B_d) @ S @ B_r
```

and the reconstructed sheet is:

```text
S_hat = B_d @ A @ transpose(B_r)
```

This gives three materially different claims:

1. When `P=L` and `Q=C`, the bases are saturated and every finite sampled sheet can be reconstructed to floating-point epsilon.
2. When a sheet already belongs to the selected lower-order tensor-product span, it can also be reconstructed to floating-point epsilon.
3. For arbitrary genuine dense weights with `P<L` or `Q<C`, no universal small-epsilon guarantee exists. The projection is the best sampled Frobenius-norm approximation, but its residual must be measured unless smoothness or analyticity assumptions are supplied.

A later empirical study can project trained dense block weights family by family and report residual versus `P` and `Q`, followed by logit or validation-loss sensitivity. That would test whether genuine learned weights are compressible by the sheet basis; it would not be a theorem that all trained weights must be.

## Explicit exclusions

Stage 2 does not integrate the shared trainer, checkpoint save/resume, activation checkpointing, compact-checkpoint inference, GPU target-scale execution, or DDP.

No original nanoGPT runtime source file is modified.
