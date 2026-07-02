# THOG2 Stage 4 Ephemeral Execution and Inference Record

**Stage:** 4  
**Branch:** `stage4-checkpointed-ephemeral-execution`  
**Pull request:** #7  
**CPU workflow:** `28588002980`  
**CPU merge-candidate:** `054947abf726039346b8fe4deefd5e7b655b14a4`

## Implemented scope

Stage 4 adds segmented non-reentrant checkpoint execution with preserved random-number state, final partial-segment handling, evaluation bypass, and an execution-only resume setting. Generated Sheet weights continue to be materialised per logical layer and are not stored as a repeated dense model.

Compact inference reconstructs from the Stage 3 checkpoint format, disables checkpoint segmentation, and exposes deterministic generation controls. Prompt handling supports inline text, text files, and UTF-8 byte tokens.

The diagnostic surface reports coefficient order summaries, parameter update-vector norms, and sampled generated-weight statistics without modifying model state. Run names identify dense versus Sheet architecture and include the Sheet orders.

Original nanoGPT runtime files remain unchanged: `model.py`, `train.py`, `sample.py`, and `configurator.py`.

## CPU evidence

```text
Stage 1 regression: PASS
Stage 2 regression: PASS
Stage 3 regression: PASS
Stage 4 tests: 16 passed
Failures: 0
Errors: 0
Skipped: 0
```

Machine-readable CPU evidence is stored in `evidence/stage4_cpu_ephemeral.json`.

Stage 4 remains pending until S4-07 passes on the accepted host.
