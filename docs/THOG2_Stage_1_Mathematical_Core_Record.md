# THOG2 Stage 1 Mathematical Core Record

**Stage:** 1 — Mathematical core  
**Branch:** `feature/sheet-stage-1-core`  
**Accepted base:** `5b45f70e7682b1f1b4eb4d358f559783092ecedd`  
**Accepted test head:** `537bf30e6b7257d87e7fb0fc45ee589053179c1c`  
**Accepted workflow run:** `28559341422`  
**Status:** All Stage 1 exit gates satisfied; completion pending merge of PR #2.

## Scope implemented

- normalized depth and fixed left-to-right row coordinates;
- the documented one-point coordinate convention of `0.0`;
- first-kind Chebyshev recurrence without explicit monomial powers;
- deterministic reduced-QR stabilization with positive `R` diagonal;
- deterministic basis reconstruction and hashing;
- basis caching by geometry, basis version, device, and runtime dtype;
- non-persistent, non-trainable basis ownership;
- proportional row-order derivation;
- packed attention, attention output, MLP expansion, MLP contraction, LayerNorm, and optional bias family geometry;
- analytical sheet and dense-equivalent parameter counts;
- the complete S1-01 through S1-15 CPU test inventory;
- high-order calibration through row order 1024.

## Explicit exclusions

Stage 1 does not implement coefficient state, weight materialisation, initialization, SheetGPT, optimizer integration, training integration, activation checkpointing, checkpoint lifecycle, inference, GPU execution, or DDP.

No original nanoGPT runtime source file was modified.

## Accepted tests

The accepted workflow executed the exact command:

```text
python tests/run_sheet_stage1_tests.py --evidence evidence/stage1_math_calibration.json
```

Result:

```text
tests run: 15
failures:  0
errors:    0
skipped:   0
```

All planned tests S1-01 through S1-15 passed on CPU.

## High-order calibration

| Width | Row order | Construction time | Float64 orthogonality max error | Float32 orthogonality max error | Observed RSS increase | Result |
|---:|---:|---:|---:|---:|---:|---|
| 768 | 128 | 0.044 s | 5.77e-15 | 1.55e-6 | 55,372 KiB | PASS |
| 1536 | 256 | 0.072 s | 5.77e-15 | 2.98e-6 | 65,516 KiB | PASS |
| 3072 | 512 | 0.112 s | 5.55e-15 | 2.98e-6 | 118,460 KiB | PASS |
| 3072 | 1024 | 0.237 s | 5.55e-15 | 2.98e-6 | 177,948 KiB | PASS |

For the required width-3072, order-1024 construction:

- all raw and stabilized values were finite;
- the QR rank proxy passed;
- the minimum absolute `R` diagonal was positive;
- estimated peak tensor memory was 92,299,264 bytes in float64;
- elapsed time and memory were far below the fixed acceptance limits;
- no tolerance was relaxed after observing a failure.

The full machine-readable evidence is committed at `evidence/stage1_math_calibration.json`.

## Failures and corrections

Failures were retained rather than silently replaced.

1. The first workflow run failed before useful evidence was written. The harness was corrected to checkpoint evidence before and after each calibration geometry and to preserve worker diagnostics.
2. The next diagnostic run exposed two environment defects: subprocess workers could not import the repository-local `sheet` package, and the minimal CI environment lacked NumPy for deterministic tensor hashing. Repository-root import setup and the missing test dependency were added.
3. A subsequent run completed all four high-order calibrations but exposed one genuine API defect: `chebyshev_first_kind_basis()` incorrectly rejected raw recurrence evaluation when the requested term count exceeded the sample count. That rank restriction belongs to QR stabilization only. It was removed from the raw recurrence helper and retained in `build_stabilized_basis()`.
4. The corrected accepted run passed all calibration and unit-test gates.

## Parameter-count cross-check

For `L=144`, `d=768`, `P=16`, and `Q_d=128`, the four large matrix families produce:

```text
sheet coefficients:          18,874,368
dense-equivalent parameters: 1,019,215,872
```

The analytical helpers agree exactly with explicit products of the planned coefficient shapes.

## Exit-gate disposition

| Gate | Result |
|---|---|
| Every mathematical test passes on CPU | PASS |
| Deterministic repeated basis construction | PASS |
| Finite, full-rank Q=1024 construction | PASS |
| Operationally acceptable Q=1024 time and memory | PASS |
| Analytical counts agree with tensor-shape products | PASS |
| No incomplete Stage 1 behavior used by model or training code | PASS |
| No nanoGPT runtime source changed | PASS |

Stage 1 becomes complete when PR #2 is merged to `master`. Stage 2 has not begun.
