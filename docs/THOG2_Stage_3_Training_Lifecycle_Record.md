# THOG2 Stage 3 Training Lifecycle Record

**Stage:** 3 — Training lifecycle  
**Implementation branch:** `stage3-training-lifecycle`  
**Pull request:** #6  
**Accepted CPU workflow:** `28576160629`  
**Accepted CPU merge-candidate SHA:** `406bb21b06f729cfe89194566e57a95d5cbe1a0d`  
**Accepted CUDA host:** `scruffy`  
**Accepted CUDA device:** `NVIDIA GeForce RTX 4090 Laptop GPU`  
**Status:** CPU lifecycle accepted and mandatory CUDA test S3-11 satisfied.

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
- a bounded real-CUDA S3-11 check covering training, validation, save, resume, and another completed update;
- a new `train_thog2.py` entry point without modifying original nanoGPT runtime files;
- a sourceable uv environment script at `docs/THOG2_Stage_3_Setup_Environment.sh`.

## Dedicated environment setup

The environment script:

- must be sourced so environment changes apply to the current Bash shell;
- deactivates any current Python virtual environment;
- repeatedly runs `conda deactivate` until Conda is inactive;
- creates or reuses the dedicated environment `~/.venvs/thog2`;
- creates it with `uv venv --prompt thog2`, producing the prompt `(thog2)` after activation;
- installs NumPy from PyPI and PyTorch from the configurable CUDA wheel index;
- activates the environment and runs a real CUDA matrix-multiplication check;
- leaves the environment active for diagnosis if CUDA verification fails;
- defaults uv HTTP requests to a 600-second timeout and five retries for large CUDA wheels.

Normal invocation:

```text
source docs/THOG2_Stage_3_Setup_Environment.sh
```

Clean rebuild:

```text
source docs/THOG2_Stage_3_Setup_Environment.sh --recreate
```

The setup contract is covered by `tests/test_sheet_stage3_environment.py`, including Bash syntax, mandatory sourcing, the Conda-deactivation command, the fixed environment path, and the `thog2` prompt setting.

## Accepted CPU evidence

```text
Stage 1 regression: PASS
Stage 2 regression: PASS
Stage 3 CPU tests:  14 passed
Failures:            0
Errors:              0
Skipped:             0
```

## Accepted CUDA evidence

The mandatory S3-11 hardware check completed successfully on `scruffy`:

```text
Device:                 NVIDIA GeForce RTX 4090 Laptop GPU
Precision:              bfloat16
Completed updates:      3
Checkpoint bytes:       310183
Latest training loss:   4.17333984375
Latest validation loss: 4.14013671875
Satisfied:              true
```

The check exercised:

1. CUDA model construction;
2. mixed-precision training;
3. validation;
4. compact checkpoint save;
5. checkpoint reload and optimizer restoration;
6. resumed CUDA training to the third completed update.

Machine-readable evidence:

- `evidence/stage3_cpu_lifecycle.json`
- `evidence/stage3_gpu_smoke.json`

## Failures and corrections

1. Initial connector write failures left an isolated partial branch; `master` was never affected. The retry completed the implementation through PR #6.
2. The first complete lifecycle run exposed a real CPU-bfloat16 defect: autocast converted generated LayerNorm affine tensors to bfloat16, while CPU LayerNorm requires float32 affine state. `TrainingSheetGPT` now materialises generated LayerNorm weight and bias with autocast disabled and float32 precision. S3-12 detects this regression.
3. The next run passed all original 13 CPU tests, but the workflow incorrectly treated the deliberate no-CUDA exit status as failure and omitted the GPU evidence file. The CUDA check now records `satisfied: false` and returns normally when CUDA is absent. Actual CUDA execution still fails normally on any training, validation, checkpoint, or resume error.
4. The first environment-script draft checked for the wrong literal activation value. uv stores `VIRTUAL_ENV_PROMPT="thog2"` and the activation script adds parentheses to the displayed shell prompt. The corrected script and its contract test now enforce the actual uv behavior.
5. Direct execution of the original test-path CUDA script could not import the repository package because Python set `sys.path[0]` to `tests/`. The accepted entry point is now the repository-root `run_thog2_stage3_cuda_check.py`, which imports the packaged runtime check normally.

No numerical tolerance or acceptance condition was weakened.

## Completion decision

All Stage 3 CPU requirements and the mandatory real-CUDA S3-11 hardware gate are satisfied. PR #6 is ready to merge as the accepted Stage 3 implementation.
