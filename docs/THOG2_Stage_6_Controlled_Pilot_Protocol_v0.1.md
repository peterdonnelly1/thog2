# THOG2 Stage 6 Controlled Pilot Protocol

Version 0.1

## 1. Status and authority

This protocol was locked before scientific training. It implements Stage 6 of the accepted THOG2 Chebyshev Sheet Staging Plan and the S6-01 through S6-15 controls in the accepted Test Plan.

Stage 6 is the final planned stage of the initial Chebyshev Sheet implementation.

The controlled pilot governed by this document completed successfully under protocol SHA-256 `1a7c66a02e5480c862f9c13d7cc3231eafa3b54c688c08093f5744bc6c16d490`. The reviewed classification is `viable_for_further_study`. Detailed results are recorded in `docs/THOG2_Stage_6_Scientific_Conclusion.md` and `evidence/stage6_pilot_acceptance.json`.

## 2. Scientific question

The pilot asks whether a fixed-coordinate bivariate Chebyshev sheet with depth order 16 and materially compressed row order can learn a language-model objective usefully relative to a dense transformer at the same logical geometry, data trace, validation sample, optimization schedule, completed updates, and consumed tokens.

The pilot does not ask whether this implementation is globally optimal, whether it matches larger dense models, or whether successor architectures such as learned coordinates, row permutations, Vermeer, auxiliary losses, or three-dimensional fields would perform better.

## 3. Hardware-driven geometry decision

The accepted Stage 5 sheet-only target geometry was L144/H12/D768/C256/P16/Q128/Q512.

A matched L144 dense control is not credible on the available 16 GiB RTX 4090 Laptop GPU under the current FP32-parameter AdamW path. The exact dense parameter count is 1,059,485,184. FP32 parameters, gradients, and two FP32 Adam moment tensors require at least 16,951,762,944 bytes, approximately 15.788 GiB, before activations, temporary tensors, CUDA context, allocator fragmentation, or evaluation logits.

The controlled matched pilot therefore uses L72/H12/D768/C256. This preserves the width, head geometry, context, fixed-coordinate row hypothesis, and meaningful P16 depth compression while providing credible memory headroom for the dense control.

The Stage 5 L144 result remains valid target-scale sheet execution evidence. Stage 6 shall not represent the L72 matched pilot as an L144 dense comparison.

## 4. Locked run matrix

The pilot shall run all four configurations in separate fresh processes:

| Selector | Architecture | Logical geometry | Depth order | Base row order | 4d row order | Checkpoint segment |
|---|---|---|---:|---:|---:|---:|
| dense | Dense GPT | L72/H12/D768/C256 | n/a | n/a | n/a | 0 |
| q64 | Chebyshev Sheet | L72/H12/D768/C256 | 16 | 64 | 256 | 4 |
| q128 | Chebyshev Sheet | L72/H12/D768/C256 | 16 | 128 | 512 | 4 |
| q256 | Chebyshev Sheet | L72/H12/D768/C256 | 16 | 256 | 1024 | 4 |

The Q64 and Q256 bracket is run up front rather than selected after observing Q128. This costs additional compute but avoids post-hoc selective expansion and provides a directly interpretable capacity trend.

## 5. Locked optimization and data budget

All runs use:

- dataset and tokenizer: identical dataset directory and metadata fingerprint;
- dtype: bfloat16 autocast with FP32 persistent parameters and optimizer state;
- global microbatch size: 1;
- gradient accumulation steps: 16;
- context length: 256;
- tokens per completed update: 4,096;
- completed updates: 250;
- consumed tokens per run: 1,024,000;
- optimizer: AdamW;
- learning rate: 6e-4;
- minimum learning rate: 6e-5;
- warmup: 10 completed updates;
- cosine decay endpoint: 250 completed updates;
- weight decay: 0.1;
- beta1: 0.9;
- beta2: 0.95;
- gradient clipping: 1.0;
- dropout: 0.0;
- model seed: 6101;
- data seed: 6102;
- evaluation interval: 25 completed updates;
- evaluation batches per split: 20;
- checkpoint interval during run: disabled;
- final checkpoint: required.

The deterministic batch source records the complete global training start-index trace and validation start-index trace. Analysis rejects the comparison unless their digests match across all four runs.

## 6. Timing definitions

Clean training time is the sum of synchronized optimizer-update durations. It excludes:

- model and optimizer construction;
- evaluation;
- checkpoint writing;
- analysis;
- process startup.

Evaluation time, checkpoint time, total wall time, and setup information are recorded separately.

Throughput is consumed training tokens divided by clean training time.

## 7. Required evidence

Each run must record:

- complete resolved configuration;
- dataset fingerprint and token counts;
- protocol digest;
- exact batch and validation trace digests and indices;
- per-update loss, learning rate, gradient norm, duration, cumulative training time, cumulative wall time, and consumed tokens;
- validation losses at aligned updates;
- persistent and dense-equivalent parameter counts;
- peak allocated and reserved CUDA memory;
- checkpoint size;
- coefficient and gradient diagnostics for sheet runs;
- depth-order and row-order energy fractions;
- selected generated-weight statistics;
- compact-state violations.

## 8. Control rejection rules

The analysis must reject the comparison if any run differs in:

- logical L/H/D/C geometry;
- tokenizer or dataset fingerprint;
- model seed or data seed;
- training batch trace;
- validation trace;
- batch size or accumulation;
- optimizer schedule;
- completed updates;
- consumed tokens;
- evaluation update positions.

Architecture-required differences in model type, sheet orders, and checkpoint segmentation are permitted and recorded.

## 9. Stop conditions

A run stops and preserves evidence if it encounters:

- non-finite training or validation loss;
- non-finite gradients or optimizer state;
- out-of-memory failure;
- checkpoint corruption;
- compact-state violation;
- invalid comparison controls;
- reproducible distributed divergence, if a distributed pilot is attempted separately.

No successor architecture is introduced inside this stage.

## 10. Analysis and classification

The analyzer produces:

- validation loss versus completed updates and consumed tokens;
- validation loss versus clean training time;
- resource and throughput comparison;
- persistent-parameter reduction relative to dense;
- Q64/Q128/Q256 capacity trend;
- coefficient order-utilization summaries;
- SVG curves, CSV tables, JSON evidence, and a Markdown summary.

The analyzer does not automatically choose the scientific conclusion. After review, the project must record exactly one outcome:

1. viable for further study;
2. viable only at weak compression;
3. inconclusive;
4. not viable under the tested design.

A stable model with a substantial resource reduction and modest quality penalty may be called promising, not superior. A requirement for Q near full row width is evidence against useful rowwise compression. A negative or inconclusive scientific result does not invalidate a functionally complete implementation.

The completed pilot was classified as `viable_for_further_study`. This is based on stable completion, validated controls, 86.05% to 91.21% persistent-parameter reduction, 83.42% to 89.47% peak allocated memory reduction, and lower final validation loss for every Sheet run at equal updates and tokens. The classification explicitly retains the observed 3.0x to 3.6x raw throughput penalty and the single-seed L72 scope limitation.

## 11. Execution sequence

First prepare and inspect the immutable manifest:

```bash
python run_thog2_stage6_pilot.py \
  --dataset-dir data/openwebtext \
  --out-dir runs/stage6_pilot_v1 \
  --prepare-only
```

Then execute exactly that manifest:

```bash
python run_thog2_stage6_pilot.py \
  --manifest runs/stage6_pilot_v1/protocol.json
```

The output root must not already exist when the protocol is prepared. This prevents accidental collision or silent reuse of previous artifacts.

## 12. Acceptance boundary

Stage 6 is not complete merely because these scripts and controls pass CPU CI. Completion requires:

- all four controlled GPU runs or a recorded stop-condition outcome;
- validated aligned analysis;
- final full regression;
- explicit scientific classification;
- completed as-built document;
- committed summarized evidence;
- accepted Stage 6 pull request.

The GPU pilot, aligned analysis, scientific classification, and summarized evidence are complete. Final regression and PR acceptance remain pending.
