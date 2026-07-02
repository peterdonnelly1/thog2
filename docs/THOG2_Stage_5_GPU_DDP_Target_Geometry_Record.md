# THOG2 Stage 5 GPU, DDP, and Target Geometry Record

## 1. Status

Stage 5 is accepted.

Accepted evidence:

- CPU and two-rank Gloo DDP: 11/11 passed;
- inherited Stage 1-4 CPU regression: 70/70 passed;
- aggregate CUDA gate: 16/16 passed;
- failures: 0;
- errors: 0;
- skips: 0.

The aggregate CUDA evidence is recorded in `evidence/stage5_gpu_acceptance.json`.

## 2. Baseline and accepted feature head

Stage 4 merged baseline:

`a75e913f40133d64eb936354581ffcc64da26d3e`

Feature head exercised by the target GPU run:

`d666bd527873e88b3729ae3e42bbfdc7ad44a3de`

Pull request:

`#9 - Stage 5: GPU, DDP, and target geometry`

## 3. Implemented scope

The shared trainer supports single-process execution and `torchrun` DDP without changing compact sheet semantics.

Implemented behavior includes:

- explicit rank, local-rank, world-size, backend, and rank-local device resolution;
- NCCL for CUDA DDP and Gloo for CPU DDP;
- identical parameter-registration and optimizer-group signatures across ranks;
- global batch-size semantics with deterministic, disjoint equal-size rank shards;
- DDP `no_sync()` during non-final gradient-accumulation microsteps;
- collective loss and gradient finite-value decisions;
- rank-averaged loss reporting;
- rank-zero compact checkpoint writes followed by a barrier;
- distributed checkpoint resume with deterministic data-generator restoration;
- explicit rejection of resume under a changed world size;
- per-rank identity reports and communication timing evidence;
- target-geometry, dtype, materialisation, basis, component, recomputation, memory, checkpoint, and stability instrumentation.

No original nanoGPT runtime file was modified.

## 4. Principal geometry and parameter accounting

Accepted principal configuration:

- logical layers: 144;
- attention heads: 12;
- embedding width: 768;
- context length: 256;
- depth order: 16;
- base row order: 128;
- MLP-contraction row order: 512.

Observed parameter accounting:

- persistent parameters: 57,732,608;
- sheet coefficients: 18,900,992;
- matrix sheet coefficients: 18,874,368;
- dense-equivalent repeated parameters: 1,020,653,568;
- dense-equivalent total parameters: 1,059,485,184.

The persistent representation is approximately 18.35 times smaller than the dense-equivalent total, a 94.55% reduction in parameter count.

## 5. CPU and distributed acceptance

Stage 5 adds deterministic and adversarial distributed tests rather than merely launch smokes. Coverage includes:

- principal-geometry and analytical-count lock;
- deterministic global-batch rank partitioning;
- rejection of non-divisible global batches;
- rejection of world-size changes on resume;
- two-rank construction identity;
- pre-optimizer averaged-gradient equivalence;
- bounded post-Adam state and fixed-probe functional equivalence;
- exact model and optimizer synchronization across ranks;
- distributed compact-checkpoint resume;
- uneven final checkpoint-segment handling;
- injected rank-structure disagreement and rank-local non-finite failure propagation;
- aggregate-GPU module-launch regression.

Accepted CPU total: 82 tests.

## 6. Target CUDA acceptance

Environment:

- GPU: NVIDIA GeForce RTX 4090 Laptop GPU;
- PyTorch: 2.12.1+cu126;
- CUDA runtime: 12.6;
- principal dtype: bfloat16;
- bfloat16 support: true.

The aggregate gate reran seven Stage 4 CUDA regressions and nine Stage 5 target cases.

Observed principal-run results:

- model construction: 0.982 seconds;
- first optimizer update: 0.474 seconds;
- four training losses: 10.9993, 10.4758, 9.6960, 9.2430;
- validation loss: 11.0020 to 9.0338;
- peak allocated CUDA memory: 1,250,801,664 bytes, approximately 1.165 GiB;
- peak reserved CUDA memory: 1,417,674,752 bytes, approximately 1.320 GiB;
- compact checkpoint size: 692,841,629 bytes, approximately 0.645 GiB;
- checkpoint recomputation time ratio: 1.520;
- checkpointed/reference loss delta: 0.0;
- compact-state violations: none.

All float32, bfloat16, and float16 reduced-geometry update tests passed. The uneven 17-layer, segment-size-4 boundary produced five checkpoint segments and passed.

## 7. Test sufficiency conclusion

The earlier stages were not light by raw test count. The material gaps were hardware breadth and distributed correctness. Those gaps are now addressed by:

- 82 CPU tests across Stages 1-5;
- exact and bounded distributed invariants;
- injected distributed failure cases;
- seven retained Stage 4 CUDA regressions;
- nine Stage 5 target CUDA cases;
- explicit compact-state and memory evidence.

Stage 5 meets its acceptance criteria.

## 8. Explicit limitation

The available target machine has one RTX 4090 Laptop GPU. Stage 5 proves:

- deterministic two-rank DDP correctness under CPU/Gloo;
- target-scale single-GPU CUDA execution and profiling;
- retained Stage 4 CUDA behavior after the shared-trainer changes.

It does not prove multi-GPU NCCL throughput, scaling efficiency, or multi-GPU peak-memory behavior. Those claims shall not be made without a genuine multi-GPU run.
