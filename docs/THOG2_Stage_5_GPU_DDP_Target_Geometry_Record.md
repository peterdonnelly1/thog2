# THOG2 Stage 5 GPU, DDP, and Target Geometry Record

## 1. Status

Stage 5 CPU and two-rank distributed correctness is accepted.

The target CUDA gate remains pending on the RTX 4090 Laptop GPU. Stage 5 shall not be merged until the aggregate CUDA evidence reports `satisfied: true`.

## 2. Baseline and implementation head

Stage 4 merged baseline:

`a75e913f40133d64eb936354581ffcc64da26d3e`

CPU/DDP accepted implementation head:

`e1a90b329777573209174c04379a7b3353fd692c`

Pull request:

`#9 - Stage 5: GPU, DDP, and target geometry`

## 3. Implemented scope

The shared trainer now supports single-process execution and `torchrun` DDP without changing compact sheet semantics.

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

## 4. Principal geometry

The Stage 5 principal configuration is locked as follows:

- logical layers: 144;
- attention heads: 12;
- embedding width: 768;
- context length: 256;
- depth order: 16;
- base row order: 128;
- MLP-contraction row order: 512.

The four principal matrix families have:

- sheet coefficients: 18,874,368;
- dense-equivalent repeated parameters: 1,019,215,872.

These values are checked analytically in CPU CI and reconstructed on the target GPU gate.

## 5. Test sufficiency audit

The earlier implementation was not light by raw count: Stages 1 through 4 already contained 70 CPU tests, four large-basis calibrations, and seven accepted Stage 4 CUDA tests.

The weakness was narrower: the first Stage 4 hardware command initially exercised only one CUDA memory case, and no DDP implementation existed before Stage 5. Stage 4 was subsequently strengthened to seven CUDA cases. Stage 5 adds deterministic and adversarial distributed coverage rather than merely adding launch smokes.

### 5.1 CPU and DDP acceptance

Stage 5 adds 11 tests:

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
- injected rank-structure disagreement and rank-local non-finite failure propagation.

Accepted Stage 5 CPU/DDP result:

- tests run: 11;
- failures: 0;
- errors: 0;
- skips: 0;
- backend: Gloo;
- world size: 2.

The complete CPU regression total on the accepted implementation head is 81 tests:

- Stage 1: 15;
- Stage 2: 19;
- Stage 3: 14;
- Stage 4: 22;
- Stage 5: 11.

### 5.2 Numerical interpretation

Exact rank-to-rank model and optimizer synchronization is required and tested.

Single-process and DDP execution use different floating reduction orders. The acceptance test therefore distinguishes:

1. pre-optimizer averaged gradients, which must agree tightly;
2. rank-to-rank DDP states, which must agree exactly;
3. post-Adam single-process/DDP states and fixed-probe logits, which must remain within explicit bounded numerical error.

This avoids both false exactness and unbounded tolerance relaxation.

### 5.3 Aggregate CUDA gate

The aggregate Stage 5 hardware gate contains 16 cases:

- seven Stage 4 CUDA regression cases;
- nine Stage 5 target-geometry and profiling cases.

The Stage 5 component covers:

- principal L144/D768/H12/C256/P16/Q128/Q512 construction;
- principal checkpointed forward, backward, and first optimizer update;
- four-update bounded principal run with validation and compact checkpoint save;
- phase-resolved CUDA memory accounting;
- P16, Q128, and Q512 basis-construction profiling;
- selected-layer materialisation profiling;
- layer-normalization, attention, and MLP component timing;
- checkpoint recomputation cost and loss equivalence;
- float32, bfloat16, and float16 reduced-geometry updates;
- uneven CUDA checkpoint-segment boundary handling;
- finite-state and compact-state stability checks.

Acceptance requires:

```json
{
  "tests_run": 16,
  "failure_count": 0,
  "error_count": 0,
  "skipped_count": 0,
  "successful": true
}
```

and top-level:

```json
{
  "satisfied": true
}
```

## 6. Target GPU command

The local branch must first be made identical to the remote feature branch. This deliberately discards tracked local evidence changes from earlier runs.

```bash
cd ~/git/thog2

git switch stage5-gpu-ddp-target-geometry
git fetch origin
git reset --hard origin/stage5-gpu-ddp-target-geometry

python tests/run_sheet_stage5_gpu_gate.py \
  --evidence evidence/stage5_gpu_acceptance.json

cat evidence/stage5_gpu_acceptance.json
```

## 7. Explicit limitation

The available target machine has one RTX 4090 Laptop GPU. Stage 5 therefore proves:

- deterministic two-rank DDP correctness under CPU/Gloo;
- target-scale single-GPU CUDA execution and profiling;
- retained Stage 4 CUDA behavior after the shared-trainer changes.

It does not prove multi-GPU NCCL throughput, scaling efficiency, or multi-GPU peak-memory behavior. Those claims shall not be made without a genuine multi-GPU run.
