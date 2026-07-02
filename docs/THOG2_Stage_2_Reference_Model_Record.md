# THOG2 Stage 2 Reference Model Record

**Stage:** 2 — Compact state and reference model  
**Branch:** `feature/sheet-stage-2-reference-model`  
**Accepted implementation head:** `67cab5b32eab2ce8ddadb1980f10ef345ffb7d68`  
**Accepted Stage 2 workflow run:** `28563464826`  
**Accepted Stage 1 regression run:** `28563464822`  
**Status:** All Stage 2 exit gates satisfied; completion pending merge of PR #4.

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

No original nanoGPT runtime source file was modified.

## Accepted commands

```text
python tests/run_sheet_stage1_tests.py --evidence /tmp/stage1_math_regression.json
python tests/run_sheet_stage2_tests.py --evidence evidence/stage2_reference_model.json
```

Accepted results:

```text
Stage 1 regression tests: 15 passed
Stage 2 tests:            19 passed
Stage 2 failures:          0
Stage 2 errors:            0
Stage 2 skipped:           0
```

The Stage 2 suite covers compact coefficient shapes, direct materialisation, direct point evaluation, saturated completeness, conventional generated shapes, family isolation, coefficient-gradient reference checks, initialization structure and statistics, optimizer coverage and semantic decay, persistent-state guards, forward/backward/update behavior, a repeated CPU learning smoke, dense nanoGPT regression, and the epsilon approximation contract.

## Reference-model evidence

The committed tiny reference geometry is:

```text
L=3
D=16
heads=4
P=3
Q_D=8
block_size=8
vocab_size=32
bias=true
```

Its parameter report is:

```text
persistent parameters:                 5,592
sheet coefficients:                    4,920
conventional non-sheet parameters:       672
dense-equivalent repeated parameters:  9,840
dense-equivalent total parameters:    10,512
matrix sheet coefficients:             4,608
matrix dense-equivalent parameters:    9,216
```

The state dictionary contained 17 keys. It contained all required compact coefficient families, no persistent fixed basis, no persistent dense logical block stack, and no `transformer.h` block list.

The full machine-readable evidence is committed at `evidence/stage2_reference_model.json`.

## Epsilon approximation contract

For a sampled sheet `S`, discrete orthonormal depth basis `B_d`, and row basis `B_r`, the fitted coefficients are the orthogonal projection coefficients:

```text
A = transpose(B_d) @ S @ B_r
```

and the reconstructed sheet is:

```text
S_hat = B_d @ A @ transpose(B_r)
```

This supports three materially different claims.

1. **Saturated sampled completeness.** When `P=L` and `Q=C`, every finite sampled sheet can be reconstructed to floating-point epsilon. Stage 2 demonstrates this for arbitrary random sampled weights.
2. **Exact lower-order reconstruction within the span.** A sheet already belonging to the selected lower-order tensor-product span is reconstructed to floating-point epsilon, even when `P<L` and `Q<C`.
3. **Best lower-order approximation without a universal guarantee.** For arbitrary genuine dense weights with `P<L` or `Q<C`, the reconstruction is the orthogonal best approximation in sampled Frobenius norm, but there is no universal guarantee that the residual is below a chosen epsilon.

A genuine continuous sheet can receive stronger approximation bounds only after assumptions are made about its regularity. For example, sufficiently smooth sheets admit improving polynomial approximation, while analytic sheets can have rapidly decreasing Chebyshev truncation error. Those are conditional mathematical statements; they do not prove that an independently trained transformer weight stack has the required regularity.

The appropriate empirical test for genuine trained weights is therefore:

1. train or load a dense model;
2. arrange each repeated matrix row as an `L x C` sampled sheet;
3. compute the orthogonal projection for selected `P` and `Q`;
4. report maximum, RMS, and relative Frobenius residuals by family;
5. increase `P` and `Q` until a chosen epsilon is reached or the required capacity becomes unattractive;
6. separately measure the effect of projected weights on logits and validation loss.

This would demonstrate whether genuine learned weights are compressible by Chebyshev Sheet. It would not claim that all possible weight sheets are compressible at low order.

## Initialization result

Only the constant depth mode is populated at initialization. Consequently:

- all generated matrix and vector profiles begin equal across logical layers;
- higher depth modes are exact zero but immediately trainable;
- attention input and MLP expansion target generated standard deviation `0.02`;
- attention output and MLP contraction retain the nanoGPT residual scaling `0.02 / sqrt(2L)`;
- repeated LayerNorm scales generate ones;
- repeated biases generate zeros;
- no complete dense logical block stack is allocated for initialization.

## Optimizer result

Matrix-family coefficient tensors are weight-decayed. LayerNorm and bias coefficient tensors are not weight-decayed. Conventional embeddings retain decay and the final LayerNorm remains non-decayed. Every trainable parameter appears in exactly one deterministic optimizer group.

## Failures and corrections

1. The Stage 2 model and all Stage 2 tests passed on their first execution.
2. The permanent Stage 1 workflow initially rejected the Stage 2 PR because its Stage 1 branch-scope guard ran on every later-stage PR. The workflow was corrected so the Stage 1 scope guard applies only to Stage 1 implementation or documentation branches, while Stage 1 compilation and regression tests continue to run on every PR.
3. An accidental temporary documentation placeholder was briefly committed to `master` before the Stage 2 branch was created. It was removed immediately in the next commit. No runtime source or accepted repository content was changed by that mistake.
4. The Stage 1 implementation defects previously found—missing failure evidence, worker import setup, missing NumPy test dependency, and the misplaced raw-recurrence rank restriction—were already corrected before Stage 2 began and remain covered by regression tests.

No numerical tolerance or acceptance rule was weakened after observing a failure.

## Explicit exclusions

Stage 2 does not integrate the shared trainer, checkpoint save/resume, activation checkpointing, compact-checkpoint inference, GPU target-scale execution, or DDP.

## Exit-gate disposition

| Gate | Result |
|---|---|
| Compact coefficient tensor per active family | PASS |
| Direct and pointwise materialisation agreement | PASS |
| Saturated sampled-sheet completeness | PASS |
| Finite non-zero gradients to all active families | PASS |
| Initialization structure and statistics | PASS |
| Semantic optimizer coverage | PASS |
| No persistent dense logical block stack | PASS |
| Tiny CPU forward, backward, and update | PASS |
| Repeated CPU learning smoke | PASS |
| Dense nanoGPT regression | PASS |
| Parameter reporting agrees with explicit state | PASS |
| Added epsilon approximation contract test | PASS |
| Stage 1 mathematical regression | PASS |

Stage 2 becomes complete when PR #4 is merged to `master`. Stage 3 has not begun.
