# THOG2 Stage 4 Ephemeral Execution and Inference Record

**Stage:** 4  
**Branch:** `stage4-checkpointed-ephemeral-execution`  
**Pull request:** #7  
**CPU workflow:** `28621636854`  
**CPU-tested head:** `8d5614462afa7ec2bbf613a8bf459d01425919d9`

## Implemented scope

Stage 4 adds segmented non-reentrant checkpoint execution with preserved random-number state, final partial-segment handling, evaluation bypass, and an execution-only resume setting. Generated Sheet weights continue to be materialised per logical layer and are not stored as a repeated dense model.

Compact inference reconstructs model-only state from the Stage 3 checkpoint format, disables checkpoint segmentation, and exposes deterministic generation controls. Prompt handling supports inline text, text files, dataset `meta.pkl` tokenizers, GPT-2 tokenization, sample count, continuation length, temperature, top-k, seed, device, and dtype.

The diagnostic surface reports coefficient order RMS and energy fractions, parameter derivative norms, and sampled generated-weight statistics without modifying model state. Run names and output directories identify dense versus Sheet architecture and include the Sheet orders.

Original nanoGPT runtime files remain unchanged: `model.py`, `train.py`, `sample.py`, and `configurator.py`.

## CPU evidence

```text
Stage 1 tests: 15 passed
Stage 2 tests: 19 passed
Stage 3 tests: 14 passed
Stage 4 tests: 22 passed
Total CPU unit tests: 70 passed
Failures: 0
Errors: 0
Skipped: 0
```

Stage 1 additionally runs its four planned large-basis calibration geometries.

Machine-readable CPU evidence is stored in `evidence/stage4_cpu_ephemeral.json`.

## Initial S4-07 hardware result

The RTX 4090 Laptop GPU comparison passed:

```text
reference peak allocated:    159,021,056 bytes
checkpointed peak allocated:  48,757,248 bytes
reduction:                   110,263,808 bytes
checkpoint/reference ratio:          0.3066
loss delta:                          0.0
checkpoint segments:                 12
```

The measured peak allocated-memory reduction was approximately 69.3%. The recorded result is in `evidence/stage4_gpu_memory.json`.

## Expanded CUDA acceptance

The initial memory comparison is necessary but not sufficient. Stage 4 now has a separate seven-test CUDA suite covering:

1. forward, loss, and coefficient-gradient equivalence for segment sizes 1, 2, 3, and 7;
2. dropout RNG equivalence under checkpoint recomputation;
3. three-update bfloat16 state and metric equivalence;
4. float16 checkpointed training;
5. resume with a changed execution segment size;
6. model-only compact-checkpoint inference on CUDA;
7. the peak allocated-memory reduction gate.

Run:

```text
python tests/run_sheet_stage4_gpu_tests.py \
  --evidence evidence/stage4_gpu_acceptance.json

cat evidence/stage4_gpu_acceptance.json
```

Stage 4 is accepted only when the expanded evidence reports seven tests run, zero failures, zero errors, zero skips, and `"satisfied": true`. PR #7 remains draft until then.
