# THOG2 Stage 6 W&B and Artifact Naming Record

## Status

This record describes the pre-pilot instrumentation and artifact layout for the final THOG2 Chebyshev Sheet Stage 6 controlled comparison.

The implementation and local rehearsals are complete. The actual L72 OpenWebText GPU pilot, scientific classification, final evidence refresh, and Stage 6 merge remain pending.

## Architecture prefixes

THOG-family architecture prefixes are:

- `DENSE` — dense control in the original THOG repository;
- `THOG` — one-dimensional THOG basis model;
- `VERMEER` — THOG Vermeer correction model;
- `SHEET` — two-dimensional Chebyshev Sheet model.

THOG2 supports only:

- `DENSE2` — THOG2 dense control;
- `SHEET` — THOG2 two-dimensional Chebyshev Sheet.

`DENSE2` is intentionally distinct from THOG's `DENSE` mode. `SHEET__` uses two underscores after its five-character prefix so the following host field aligns with the six-character `DENSE2` prefix.

## Artifact naming

Each Stage 6 run receives one architecture-first name derived before training and locked inside `protocol.json`.

Representative forms are:

```text
DENSE2_scruffy_STAGE6__OWT__L72_H12_D768_C256__...
SHEET__scruffy_STAGE6__OWT__L72_H12_D768_C256_P16_Q128_Q4D512__...
```

The full name records:

- architecture prefix;
- host label;
- comparison run name;
- dataset;
- L/H/D/C logical geometry;
- P/Q/Q4D orders for Sheet;
- global batch, gradient accumulation, and world size;
- update, evaluation, logging, and warmup controls;
- learning-rate schedule, weight decay, Adam betas, and gradient clipping;
- model and data seeds;
- dtype and activation-checkpoint segment size;
- source commit prefix;
- experiment suffix.

Names are limited to 240 characters.

## Derived files

For each run, the manifest records:

```text
checkpoints/<artifact_name>/ckpt.pt
checkpoints/<artifact_name>/result.json
results/<artifact_name>.json
logs/<artifact_name>.log
```

The worker-local result remains beside the checkpoint. The orchestrator atomically mirrors it to the canonical architecture-named result file. Restart and analysis validate the protocol digest before accepting either copy.

## W&B ownership

The scientific THOG2 worker remains the authoritative producer of local evidence. W&B is owned by the existing THOG controlled-comparison adapter, extended for THOG2 Stage 6 and merged into THOG `master` at:

```text
213f144162d2cf76d51e7338ba8434c4e8dcb64c
```

The single-owner launcher is:

```text
thog/thog2_stage6_launcher.py
```

It:

1. imports the THOG2 worker in a fresh process;
2. neutralizes the worker's provisional telemetry hook;
3. observes the existing Stage 6 progress seam outside synchronized training and evaluation timers;
4. creates exactly one W&B run per architecture;
5. restores all monkeypatched process state;
6. reads final resource and Sheet diagnostics only after the authoritative local result exists;
7. uses THOG's resilient online-to-offline W&B initialization and bounded finish behavior.

The THOG2 convenience launcher is:

```text
run_thog2_stage6_instrumented.sh
```

It locks the required THOG adapter commit, preserves restart-at-run-boundary behavior, writes architecture-named logs, validates each result, and finally invokes ordinary THOG2 analysis.

## W&B organization

One W&B run is created per architecture under one locked comparison group.

Run names equal the full artifact names. Job types are:

- `dense2`;
- `sheet_q64`;
- `sheet_q128`;
- `sheet_q256`.

The primary axes are:

- `optimizer_update`;
- `iter` as a compatibility alias;
- `tokens_seen`;
- `clean_training_seconds`.

The W&B configuration contains hashes, counts, geometry, controls, artifact identity, and parameter reports. It intentionally excludes local filesystem paths and repository remote URLs.

## Logged metrics

At Stage 6 reporting updates:

- training loss;
- learning rate;
- gradient norm;
- consumed tokens;
- clean cumulative training seconds.

At aligned evaluations:

- train-split evaluation loss and perplexity;
- validation loss and perplexity.

At completion:

- persistent and dense-equivalent parameter counts;
- checkpoint size;
- clean tokens per second;
- peak allocated and reserved device memory;
- Sheet coefficient RMS;
- high depth-order and row-order energy fractions;
- compact-state violation count.

## Timing integrity

W&B communication is not inside the synchronized `train_one_update()` timer or evaluation timer. Therefore `clean_training_seconds` excludes telemetry overhead.

No `wandb.watch()`, graph capture, parameter histograms, automatic checkpoint upload, or microstep logging is enabled.

## Source of truth

W&B is visualization and operational telemetry only. Acceptance is based on the local Stage 6 artifacts:

- `protocol.json`;
- per-run worker results;
- architecture-named result mirrors;
- checkpoints;
- logs;
- `pilot_status.json`;
- analysis JSON, CSV, Markdown, and SVG files.

A W&B communication failure must not alter model state, batch traces, update count, or scientific acceptance evidence.

## Verification completed before the GPU pilot

- THOG2 Stage 6 CPU/control suite: 25 tests passed locally;
- existing THOG controlled-comparison W&B suite: 9 tests passed;
- THOG2 adapter mapping suite: 4 tests passed;
- two-update single-run adapter rehearsal passed;
- four-run matched DENSE2/Q64/Q128/Q256 launcher rehearsal passed;
- rehearsal verified one telemetry run per architecture, update/evaluation axes, final resource metrics, named checkpoints/results/logs, restart validation, and final analysis generation.

These rehearsals are implementation evidence only. They are not substitutes for the locked OpenWebText GPU pilot.
