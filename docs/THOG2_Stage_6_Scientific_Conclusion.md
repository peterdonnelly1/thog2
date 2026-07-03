# THOG2 Stage 6 Scientific Conclusion

## Classification

**Viable for further study.**

This is the strongest classification justified by the locked Stage 6 pilot. It is not a claim of general superiority over dense transformers.

## Accepted pilot identity

- protocol SHA-256: `1a7c66a02e5480c862f9c13d7cc3231eafa3b54c688c08093f5744bc6c16d490`;
- pilot source commit: `2071746ab18f4182010797241d2b29adb5a7a305`;
- geometry: L72/H12/D768/C256;
- Sheet depth order: P16;
- run matrix: dense, Q64/Q256, Q128/Q512, Q256/Q1024;
- completed optimizer updates per run: 250;
- consumed tokens per run: 1,024,000;
- model seed: 6101;
- data seed: 6102;
- device: NVIDIA GeForce RTX 4090 Laptop GPU;
- execution: single GPU, bfloat16 autocast, FP32 persistent parameters and AdamW state.

All four runs completed successfully. The analyzer verified the same protocol, completed updates, consumed tokens, training batch trace, validation sample, and evaluation updates across all runs.

## Principal results

| Run | Persistent parameters | Reduction vs dense | Peak allocated GiB | Reduction vs dense | Tokens/s | Relative throughput | Final validation loss | Delta vs dense |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 549,158,400 | 0.00% | 10.069 | 0.00% | 3,693.7 | 100.0% | 6.843831 | 0.000000 |
| Sheet Q64 | 48,282,112 | 91.21% | 1.060 | 89.47% | 1,232.4 | 33.4% | 6.470158 | -0.373672 |
| Sheet Q128 | 57,732,608 | 89.49% | 1.266 | 87.42% | 1,207.7 | 32.7% | 6.454906 | -0.388924 |
| Sheet Q256 | 76,633,600 | 86.05% | 1.669 | 83.42% | 1,026.6 | 27.8% | 6.478035 | -0.365796 |

Checkpoint sizes fell from 6,591,126,257 bytes for dense to 579,588,857 bytes, 692,994,873 bytes, and 919,806,841 bytes for Q64, Q128, and Q256 respectively.

## Equal-update conclusion

At 250 completed updates and 1,024,000 consumed tokens, every Sheet configuration achieved lower validation loss than dense.

This is a meaningful positive result because it holds at the strongest tested compression, Q64. The result therefore does not fit `viable_only_at_weak_compression`.

Q128 produced the lowest final loss, but its advantage over Q64 was only 0.015252. Q256 was larger, slower, and worse than Q64. The tested capacity curve is therefore not monotonic and does not justify preferring Q256.

Within the tested set, Q64 is the practical Pareto choice.

## Equal-time conclusion

The current Sheet implementation is substantially slower per token:

- Q64: 2.997x dense training time per token;
- Q128: 3.058x;
- Q256: 3.598x.

At approximately the dense endpoint time of 277 seconds, the nearest Sheet evaluations were roughly 6.981 to 7.071 versus dense at 6.844. Dense therefore retained a modest quality advantage at approximately equal clean training time near that endpoint.

However, the Sheet models learned more per update. They first passed the dense final loss at update 150:

- Q64: 501.9 seconds, 1.81x dense final training time;
- Q128: 508.5 seconds, 1.83x;
- Q256: 598.3 seconds, 2.16x.

The result is therefore not “90% less memory for a full 3x equal-loss penalty.” The observed equal-loss penalty to reach the dense endpoint was materially smaller than the raw throughput penalty, although the evaluation grid is coarse and the curves are noisy.

## Resource conclusion

The pilot demonstrates a real memory and persistent-state reduction, not an accounting artifact:

- 86.05% to 91.21% fewer persistent parameters;
- 83.42% to 89.47% lower peak allocated CUDA memory;
- 86.04% to 91.21% smaller checkpoints.

These reductions are large enough to be useful for fitting substantially deeper or wider logical models on constrained hardware.

## Sheet diagnostics

All recorded coefficient families had nonzero fraction 1.0. No compact-state stop condition occurred.

High row-order energy was generally material for matrix families, often near one quarter of total row-order energy. The higher-order coefficients are therefore not simply dead parameters. This observation does not prove that Q64 is saturated, because orthonormal coefficient energy can remain broadly distributed without a quality gain from larger Q.

The lack of a monotonic Q64/Q128/Q256 quality trend means the next capacity study should include lower orders, particularly Q16 and Q32, rather than assuming that Q128 is required.

## Why the result is not classified as superior

The pilot has important limits:

- one dataset snapshot;
- one L72 geometry;
- one model seed and one data seed;
- only 250 optimizer updates;
- coarse 25-update validation intervals;
- no matched L144 dense run;
- no multi-GPU NCCL performance evidence;
- no optimized fused materialisation kernel;
- validation loss is visibly noisy late in training.

The result establishes that the fixed-coordinate two-dimensional Chebyshev Sheet is a viable research direction under the tested design. It does not establish broad scaling behavior, final convergence quality, or wall-clock superiority.

## Immediate successor experiments

The highest-value next experiments are:

1. replicate dense, Q32, Q64, and Q128 across several seeds;
2. extend the training budget far enough to compare sustained convergence rather than the first million tokens;
3. profile and optimize materialisation, especially the Q256/Q1024 path;
4. test whether Q16 or Q32 preserves the Q64 result;
5. repeat at larger logical depth where dense execution is memory-limited or impossible;
6. retain the same trace-controlled equal-update and equal-time reporting.

## Closure state

The scientific pilot gate is satisfied. Stage 6 still requires final regression on the accepted branch head, final as-built completion, PR acceptance, and merge before the implementation stage is closed.
