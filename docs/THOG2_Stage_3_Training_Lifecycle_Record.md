# THOG2 Stage 3 Training Lifecycle Record

**Stage:** 3 — Training lifecycle  
**Implementation branch:** `stage3-training-lifecycle`  
**Pull request:** #6  
**Accepted CPU workflow:** `28570132565`  
**Accepted CPU merge-candidate SHA:** `4c6477af7ab93d1c7135ae6c05730998832f88fd`  
**Status:** CPU lifecycle accepted. Mandatory CUDA test S3-11 remains unsatisfied because the hosted runner has no CUDA device.

## Implemented scope

- explicit `dense` and `thog2_sheet` model selection;
- one shared accumulation, scheduler, clipping, evaluation, logging, and checkpoint lifecycle;
- exact completed-optimizer-update counting;
- independent model-construction and data-sampling RNG streams;
- deterministic train and validation batch traces;
- compact Sheet checkpoint save, reload, and resume;
- deterministic optimizer-group reconstruction and optimizer-state restoration;
- checkpointed geometry, basis version, row-order scaling rule, counters, data RNG, process RNG, and parameter report;
- direct incompatibility diagnostics;
- compatible legacy dense nanoGPT checkpoint loading;
- startup persistent, sheet, conventional, and dense-equivalent parameter reporting;
- CPU float32 and bfloat16 reference paths;
- a bounded real-CUDA S3-11 script covering training, validation, save, resume, and another completed update;
- a new `train_thog2.py` entry point without modifying original nanoGPT runtime files.

## Accepted CPU evidence

```text
Stage 1 regression: PASS
Stage 2 regression: PASS
Stage 3 CPU tests:  13 passed
Failures:            0
Errors:              0
Skipped:             0
```

Machine-readable evidence:

- `evidence/stage3_cpu_lifecycle.json`
- `evidence/stage3_gpu_smoke.json`

## Failures and corrections

1. Initial connector write failures left an isolated partial branch; `master` was never affected. The retry completed the implementation through PR #6.
2. The first complete lifecycle run exposed a real CPU-bfloat16 defect: autocast converted generated LayerNorm affine tensors to bfloat16, while CPU LayerNorm requires float32 affine state. `TrainingSheetGPT` now materialises generated LayerNorm weight and bias with autocast disabled and float32 precision. S3-12 detects this regression.
3. The next run passed all 13 CPU tests, but the workflow incorrectly treated the deliberate no-CUDA exit status as failure and omitted the GPU evidence file. The CUDA smoke now records `satisfied: false` and returns normally when CUDA is absent. Actual CUDA execution still fails normally on any training, validation, checkpoint, or resume error.

No numerical tolerance or acceptance condition was weakened.

## CUDA gate

S3-11 is not satisfied by CPU emulation. Run on a CUDA host:

```text
python tests/run_sheet_stage3_gpu_smoke.py --evidence evidence/stage3_gpu_smoke.json
```

Stage 3 must not be described as fully complete or merged as accepted until that file reports `"satisfied": true`. The PR is otherwise ready for review and GPU validation.
