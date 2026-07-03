# THOG2 Chebyshev Sheet As-Built Record

Version 1.0 — candidate final pending final regression and Stage 6 merge

## 1. Status

The initial THOG2 Chebyshev Sheet implementation is functionally complete through the Stage 6 controlled GPU pilot.

The locked pilot completed all four runs and passed its scientific control checks. The recorded classification is:

**viable for further study**.

The implementation stage is not yet closed. Closure still requires final full regression on the post-evidence Stage 6 branch head, acceptance of PR #10, and merge to `master`.

## 2. Repository and baseline

- repository: `peterdonnelly1/thog2`;
- local convention: `~/git/thog2`;
- default branch: `master`;
- accepted Stage 5 baseline: `ed26a681d2a91ddfb47b312ed0c6ef6391fea071`;
- Stage 6 pilot source commit: `2071746ab18f4182010797241d2b29adb5a7a305`;
- Stage 6 pull request: #10;
- architecture designation: THOG2 Chebyshev Sheet;
- original nanoGPT dense path: retained and independently selectable.

The implementation is contained in the `sheet/` package and THOG2-specific launch paths. Original nanoGPT runtime files were not replaced by the sheet implementation.

## 3. Implemented architecture

### 3.1 Compact state

Each repeated transformer family is represented by a compact coefficient tensor with shape:

`[output_rows, depth_order, row_order]`.

The stored trainable state includes separate families for:

- attention input weight;
- attention output weight;
- MLP expansion weight;
- MLP contraction weight;
- two LayerNorm weights;
- two LayerNorm biases;
- attention input and output biases;
- MLP expansion and contraction biases.

The full logical dense block stack is not persisted.

### 3.2 Fixed bases

Depth and row coordinates are fixed on `[-1, 1]`. First-kind Chebyshev samples are stabilized with deterministic reduced QR and deterministic column signs.

Fixed bases are reconstructed from geometry and basis version. They are not required in compact checkpoint state.

Accepted basis version:

`chebyshev_first_kind_qr_v1`

Accepted row-order scaling rule:

`proportional_ceil_v1`

### 3.3 Materialisation

For logical layer `l`, one family is materialized by:

1. mixing the coefficient tensor against the selected depth-basis row;
2. multiplying the resulting output-row coefficients by the transpose of the family row basis.

Logical layers execute sequentially. Generated weights are ephemeral and are not cached as a persistent logical stack.

### 3.4 Initialization

Matrix families are initialized through coefficient space to reproduce the intended generated-weight scale at initialization.

- attention input and MLP expansion use standard nanoGPT-like scale;
- attention output and MLP contraction use residual scale `0.02 / sqrt(2L)`;
- LayerNorm weights generate exact ones at initialization;
- biases generate exact zeros at initialization.

### 3.5 Objective

The sole training objective is ordinary next-token cross-entropy. No auxiliary sheet, smoothness, reconstruction, or coefficient loss is used.

## 4. Source map

### 4.1 Mathematical and geometry core

- `sheet/coordinates.py` — normalized fixed coordinates;
- `sheet/basis.py` — Chebyshev recurrence, QR stabilization, signs, cache, basis ownership;
- `sheet/geometry.py` — family geometry, proportional row orders, analytical counts;
- `sheet/materialize.py` — direct and factored materialisation helpers.

### 4.2 Compact model

- `sheet/trajectory.py` — compact coefficient families, initialization, semantic metadata, materialisation;
- `sheet/model.py` — reference functional SheetGPT;
- `sheet/training_model.py` — training model with segmented activation checkpointing;
- `sheet/training_model_factory.py` — explicit dense or sheet construction and parameter reports.

### 4.3 Training lifecycle

- `sheet/training_config.py` — architecture and execution configuration;
- `sheet/trainer.py` and trainer mixins — shared dense/sheet lifecycle;
- `sheet/batch_source.py` — deterministic global batch generation, rank sharding, and Stage 6 trace evidence;
- `sheet/checkpoints.py` and checkpoint trainer mixins — compact save, validation, resume, RNG state;
- `sheet/distributed.py` — rank identity, process-group setup, collectives;
- `train_thog2.py` — explicit command-line training entry point.

### 4.4 Ephemeral execution and inference

- `sheet/checkpointing.py` — indexed segmented checkpoint execution;
- `sheet/memory.py` — lifecycle memory samples;
- `sheet/stage4_trainer.py` — memory-instrumented trainer;
- `sheet/inference.py` and THOG2 inference launchers — direct compact-checkpoint inference;
- Stage 4 CUDA tests — reference/checkpoint equivalence, RNG, dtype, memory, resume, and inference evidence.

### 4.5 Stage 5 target and DDP paths

- `sheet/stage5_target.py` — principal and reduced target configurations;
- `sheet/stage5_profile.py` — basis, materialisation, component, and recomputation profiling;
- Stage 5 DDP workers/tests — two-rank deterministic construction, gradient, update, state, resume, boundary, communication, and failure propagation;
- Stage 5 GPU gate — retained Stage 4 CUDA cases plus target L144 cases.

### 4.6 Stage 6 controlled pilot paths

- `sheet/stage6_protocol.py` — locked budget, run matrix, control signatures, memory feasibility, protocol digest;
- `sheet/stage6_diagnostics.py` — detached coefficient, gradient, order-energy, and generated-weight diagnostics;
- `sheet/stage6_trainer.py` — clean timing and run-evidence capture;
- `sheet/stage6_analysis.py` — control validation, aligned comparisons, resource tables, CSV and SVG outputs;
- `run_thog2_stage6_one.py` — one isolated run process;
- `run_thog2_stage6_pilot.py` — dataset fingerprinting, immutable manifest, four-run orchestration, logs, and analysis;
- `docs/THOG2_Stage_6_Scientific_Conclusion.md` — reviewed scientific classification;
- `evidence/stage6_pilot_acceptance.json` — committed summarized pilot evidence and artifact hashes.

## 5. Configuration semantics

`model_type` is explicit:

- `dense` selects the retained nanoGPT GPT;
- `thog2_sheet` selects Chebyshev Sheet.

Sheet geometry controls:

- `depth_order` — depth-axis basis order P;
- `base_row_order` — Q for width-d rows;
- wider rows derive proportional orders under the accepted scaling rule;
- `checkpoint_segment_size` — execution-only logical-layer segment size.

The sheet architecture remains non-default. Callers must select it explicitly.

## 6. Checkpoint semantics

Compact checkpoints contain:

- schema version;
- model type and model arguments;
- compatibility signature;
- basis and row-scaling versions;
- compact model state;
- optimizer state and parameter-group name order;
- trainer state and completed updates;
- resolved training configuration;
- deterministic batch-source state;
- RNG state;
- parameter report;
- distributed execution metadata.

Resume validates model geometry before optimizer state is restored. Execution-only controls may be overridden only through the allowed override set. World-size changes are rejected because they would invalidate deterministic rank sharding and batch-source state.

## 7. DDP semantics

The configured `batch_size` is global. It must be divisible by world size. Each rank receives an equal disjoint shard of the same deterministically generated global batch.

Non-final accumulation microsteps use DDP `no_sync()`. Final microsteps synchronize gradients. Loss and finite-state decisions are collective. Compact checkpoints are written by rank zero followed by a barrier.

Accepted automated distributed evidence uses two CPU/Gloo ranks. Multi-GPU NCCL throughput and scaling have not been measured.

## 8. Accepted target-scale evidence

Stage 5 exercised:

- L144/H12/D768/C256;
- P16;
- Q128 for width-d matrix families;
- Q512 for the width-4d MLP contraction family;
- bfloat16 principal execution;
- segmented checkpoint size 4;
- four completed updates;
- direct validation and compact checkpoint writing.

Observed Stage 5 summary:

- persistent parameters: 57,732,608;
- dense-equivalent total parameters: 1,059,485,184;
- peak allocated CUDA memory: 1,250,801,664 bytes;
- training loss: 10.9993 to 9.2430;
- validation loss: 11.0020 to 9.0338;
- compact-state violations: none;
- aggregate CUDA cases: 16/16 passed.

This is bounded functional evidence, not a dense quality comparison.

## 9. Stage 6 matched-pilot deviation

The governing implementation plan specifies L144/D768 for the first pilot where hardware permits.

A matched L144 dense control is not feasible on the available 16 GiB GPU under the current FP32-parameter AdamW path. Its minimum parameter, gradient, and two-moment storage is approximately 15.788 GiB before activations and runtime overhead.

The locked matched pilot therefore used L72/H12/D768/C256/P16 with dense, Q64, Q128, and Q256 runs. This is an implementation-plan hardware fallback, not an architectural change. The scientific conclusion is limited to the tested L72 geometry and 1,024,000-token budget.

## 10. Stage 6 controlled pilot evidence

### 10.1 Identity and controls

- protocol SHA-256: `1a7c66a02e5480c862f9c13d7cc3231eafa3b54c688c08093f5744bc6c16d490`;
- pilot source commit: `2071746ab18f4182010797241d2b29adb5a7a305`;
- dataset train sampled SHA-256: `e1daf49a9ac3e8d7c0b070bd4bee3a784b1a648f35d955d278bf4e129d6816c7`;
- dataset validation sampled SHA-256: `c2a5937e2418846130e3b8ae0c854a18631e8ce069ff8d339874965d5dc19458`;
- training trace SHA-256: `66c6663db2820cdc2648efbd8ac15ee1082b80800b5ded34694c08a30d1638b0`;
- validation trace SHA-256: `b0973c3b94bcf1d8662770861ad0a3cc69f4f2a42b53775bf08351e38d18cef6`;
- completed updates: 250 per run;
- consumed tokens: 1,024,000 per run;
- evaluation updates: 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250.

All runs matched the protocol, completed-update count, consumed-token count, training batch trace, validation sample, and evaluation schedule.

### 10.2 Resource and loss summary

| Run | Persistent parameters | Reduction vs dense | Peak allocated GiB | Reduction vs dense | Tokens/s | Final validation loss |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 549,158,400 | 0.00% | 10.069 | 0.00% | 3,693.7 | 6.843831 |
| Sheet Q64 | 48,282,112 | 91.21% | 1.060 | 89.47% | 1,232.4 | 6.470158 |
| Sheet Q128 | 57,732,608 | 89.49% | 1.266 | 87.42% | 1,207.7 | 6.454906 |
| Sheet Q256 | 76,633,600 | 86.05% | 1.669 | 83.42% | 1,026.6 | 6.478035 |

Checkpoint sizes were:

- dense: 6,591,126,257 bytes;
- Q64: 579,588,857 bytes;
- Q128: 692,994,873 bytes;
- Q256: 919,806,841 bytes.

### 10.3 Equal-update and equal-time interpretation

At equal completed updates and consumed tokens, all three Sheet runs finished below dense validation loss. Q128 was numerically best, but only 0.015252 below Q64. Q256 was larger, slower, and worse than Q64.

Raw Sheet throughput was 27.8% to 33.4% of dense. At approximately the dense endpoint clean time, dense retained a modest validation-loss advantage. The Sheet runs nevertheless first passed the dense final loss at update 150 after approximately 1.81x, 1.83x, and 2.16x the dense final training time for Q64, Q128, and Q256 respectively.

The tested equal-loss penalty was therefore materially smaller than the raw 3.0x to 3.6x throughput penalty, although the validation grid is coarse and noisy.

### 10.4 Capacity and diagnostic interpretation

All recorded coefficient families had nonzero fraction 1.0. No compact-state stop condition occurred.

The Q64/Q128/Q256 result does not show a monotonic benefit from more row coefficients. Q64 is the tested Pareto choice. Lower Q values and replicated longer runs are the appropriate next capacity study.

## 11. Scientific classification

The accepted classification is:

**viable for further study**.

The result is not classified as `viable_only_at_weak_compression` because the strongest tested compression, Q64, achieved essentially the same quality as Q128 and Q256 while retaining only 8.792% of dense persistent parameters.

The result is not classified as superior. It covers one short, single-seed L72 pilot and the current implementation remains slower at equal tokens and approximately equal wall-clock time.

## 12. Verification record

Accepted through Stage 5:

- mathematical basis and geometry tests through Q1024;
- direct materialisation and saturated-basis tests;
- initialization and optimizer grouping;
- CPU forward, backward, optimization, checkpoint, resume, and inference;
- activation-checkpoint equivalence and RNG preservation;
- compact-state source and checkpoint guards;
- seven Stage 4 CUDA cases;
- deterministic two-rank DDP tests including injected failures;
- nine Stage 5 target GPU cases;
- complete accepted CPU regression through Stage 5.

Stage 6 verification includes:

- principal dense memory feasibility;
- locked matched controls and token budget;
- unique run identities and output paths;
- deterministic training and validation trace digests;
- coefficient-order and generated-weight diagnostics;
- pilot evidence schema;
- mismatch rejection and aligned comparison tables;
- 25/25 Stage 6 CPU/control tests before the pilot;
- four successful controlled GPU runs;
- validated equal-update, equal-time, resource, and coefficient-utilization analysis;
- explicit scientific classification and committed summarized evidence.

Final full regression on the post-evidence branch head remains pending.

## 13. Known limitations

- No matched L144 dense pilot on the available 16 GiB GPU.
- No multi-GPU NCCL performance or memory-scaling evidence.
- No custom CUDA or Triton materialisation kernel.
- No compile-specific optimized path.
- Generated matrices are reconstructed independently per logical layer and family.
- Fixed row coordinates may be a representational limitation.
- The pilot covers one dataset snapshot, one logical geometry, one training budget, and one model/data seed.
- A single pilot cannot establish broad scaling behavior or superiority.
- Large raw checkpoints and logs remain external artifacts; committed evidence records their summaries and hashes.

## 14. Explicitly deferred successor work

The following are not part of this implementation:

- learned row coordinates;
- learned row permutations;
- coordinate warping;
- multiple sheets per row;
- Vermeer integration;
- auxiliary objectives;
- three-dimensional Chebyshev fields;
- custom fused kernels;
- automatic dense-to-sheet conversion.

Any such work requires a successor specification or controlled amendment.

## 15. Remaining closure gates

Before Stage 6 is merged and this record becomes final:

- run the complete regression on the post-evidence branch head;
- record the final regression evidence and accepted Stage 6 head;
- mark PR #10 ready for review;
- accept and merge PR #10;
- record the resulting Stage 6 merge commit.
