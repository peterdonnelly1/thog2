# PLASTIC DEPTH implementation log

Last updated: 2026-08-05 08:45 AEST

## Critical correction retained

The earlier log incorrectly described an unpushed local implementation as completed. Commit `dedf942` did not exist. Work resumed from verified GitHub source; only phases explicitly marked complete below are implemented and pushed.

## Repository state

- Repository: `peterdonnelly1/thog2`
- Working branch: `PLASTIC_DEPTH`
- Pull request: #29, draft
- Base: `HYPERBLOCK_LOOP_ENHANCEMENTS`
- Current verified branch head before this log update: `7c2ac900372240f3411532c7479983c909cfdcf4`
- Requirements target: PLASTIC DEPTH specification v0.3
- Implementation plan: THOG2 PLASTIC DEPTH Implementation and Testing Plan v0.1

## Completed revised phases

- [x] Requirements specification v0.3
- [x] Implementation and testing plan v0.1
- [x] Actual branch/source discrepancy identified and corrected
- [x] Exact recurrence-built float64 affine Chebyshev re-expression kernel and tests
- [x] Active-prefix geometry with N active positions plus one derived N+1 probe position
- [x] Maximum permitted layers reduced to count/allocation capacity only
- [x] Inactive capacity proven unable to affect active coordinates, logits or gradients
- [x] Fixed QR reference count made independent of maximum capacity
- [x] Atomic model-wide add/subtract chart re-gauge
- [x] Every generated coefficient family transformed and field-verified before commit
- [x] Geometry and coefficient version checks prevent stale prepared transitions from committing
- [x] Gauge preparation remains non-mutating; count/coefficients change only in the commit operation
- [x] Stock PyTorch AdamW coefficient-state migration without reimplementing AdamW
- [x] First moments transformed with the inverse-transpose covector rule
- [x] Diagonal second moments and optional AMSGrad maxima transformed with the squared-covector approximation
- [x] Step counters retained and included in stale-state detection
- [x] Non-finite and ill-conditioned migrations select a logged zero-moment reset fallback before model mutation
- [x] Shared first-microstep N-1/N/N+1 transformer chain
- [x] Candidate heads detached/no-grad on common sampled token positions
- [x] One selected grad-bearing head and selected prefix reused for remaining accumulation microsteps
- [x] Persistent count and gauge remain unchanged through forward/backward and stock AdamW step
- [x] Atomic model/gauge/AdamW transition commits only after the successful optimizer step
- [x] Activation-checkpoint and retained-materialisation paths covered directly
- [x] Paired per-count/per-direction inline score histories collected on every successful optimizer update
- [x] Robust MAD significance gate with configurable window, minimum observations, lambda and positive zero-MAD floor
- [x] Deterministic standardized-improvement selection with exact ties preferring the lower count
- [x] Five-update count-change brake replacing the obsolete periodic hold controller
- [x] Evidence and count-change spacing checkpointed only after successful AdamW/model transition commit
- [x] Failed/non-finite updates discard transient evidence without mutating persistent controller state
- [x] Universal CUDA allocator reserve acquired before every learned-count upward probe
- [x] Reserve acquisition feasibility synchronized across ranks before model execution
- [x] Failed reserve preflight removes only N+1 while preserving N and N-1
- [x] Safety reserve released immediately before the single adjacent N+1 layer attempt
- [x] CUDA OOM in the N+1 layer or detached probe head becomes an infeasible candidate rather than a failed update
- [x] N+1 feasibility synchronized across ranks; any-rank failure discards N+1 everywhere
- [x] Exact non-CUDA shared-prefix path remains unchanged
- [x] New persisted PLASTIC identity surfaces use exact canonical `plastic__...` control names
- [x] Internal `plastic__sampling_seed` remains unexposed and absent from public identity
- [x] Retired short PLASTIC identity aliases remain semantically checkpoint-compatible
- [x] `_L_dyn_` is emitted only for learned-count PLASTIC runs
- [x] Fixed-count PLASTIC and disabled runs retain numeric `_L_<count>_` identity
- [x] Descriptor post-processing accepts both numeric and dynamic PLASTIC layer identity
- [x] Successful count transitions sample generated `attention_query_weight[0,0]` across the committed active chart
- [x] Sampled values remain transient console state and never enter metrics, telemetry or checkpoints
- [x] Transition optimizer rows are forced without an extra standalone line
- [x] Samples are emitted once only and never repeated on later training or validation rows
- [x] UI formatting uses two decimals normally and scientific notation when magnitude requires it

## Validation evidence

### Active-prefix checkpoint `b7231c2`

- focused PLASTIC/gauge/interface/cache tests: 69 passed
- affected CPU regression tests: 160 passed
- parameterised subtests: 200 passed
- two-rank CPU DDP: zero model-state and optimiser-state divergence

### Atomic gauge-transition checkpoint `de57d61f`

The gated applier verified patch SHA, compile/shell syntax, focused and affected tests, and two-rank CPU DDP before commit.

### AdamW-state checkpoint `e0d0a3fb`

- 103 focused/affected CPU tests
- 10 direct AdamW-state tests
- two-rank CPU DDP with zero model-state and optimiser-state divergence

### Shared inline-probe checkpoint `855ea5f2`

The gated applier verified the complete five-file patch and ran before committing:

- Python compile and shell syntax
- 113 focused PLASTIC/trainer/checkpoint/W&B tests
- 160 affected CPU regression tests
- 200 parameterised subtests
- 10 direct inline-probe tests
- two-rank CPU DDP with zero model-state and optimiser-state divergence

Direct coverage proves first-microstep-only probing, shared maximum-prefix execution, deterministic sampled positions, selected-prefix reuse, post-step-only count commit, retained-materialisation support and scalar-only metrics.

### MAD gate and update-brake takeover phase

The original staged transport payload at `6b0b759c` was corrupt after the first six files. The phase was recovered from its intact controller/config fragments, completed against the authoritative task, validated locally, then reassembled as an exact SHA-256-gated recovery patch. Hosted runners repeated the focused, affected, DDP and source-history gates before committing `f334a024`.

- 122 focused PLASTIC/gauge/optimizer/inline/controller/interface/trainer/checkpoint tests passed
- 141 affected CPU regression tests passed
- 200 parameterised subtests passed
- two-rank CPU DDP passed with zero model-state and optimizer-state divergence
- Python compile, shell syntax and `git diff --check` passed
- THOG source-history audit passed with zero violations
- direct integration proves failed updates do not checkpoint evidence and count changes at updates 1 and 6 under the default five-update brake

Commit `f334a024` is the clean MAD/brake phase commit. Ordinary hosted CI on descendant `1b93a5be` passed focused CPU, affected regressions, two-rank DDP, source-history audit and exact head-versus-base comparison.

### CUDA reserve and recoverable upward-probe phase

The phase introduces `plastic__cuda_allocator_reserve_gib`, default 0.5 GiB. It is persistent execution configuration but deliberately does not yet alter the public artifact-name fragment. The reserve is CUDA-only; CPU execution follows the established inline-probe code path exactly.

- 132 focused PLASTIC/gauge/optimizer/inline/controller/CUDA/interface/trainer/checkpoint tests passed
- 141 affected CPU regression tests passed
- 200 parameterised subtests passed
- two-rank CPU DDP passed with zero model-state and optimizer-state divergence
- direct tests cover reserve allocation/OOM, distributed preflight rejection, local upward OOM, successful-local/failed-remote rejection, cleanup, and successful N+1 gradient equivalence
- Python compile, shell syntax and `git diff --check` passed
- THOG source-history audit passed with zero violations

Hosted recovery runners repeated all phase gates and published clean commit `1c90e500`. Ordinary hosted CI on descendant `2eb3e644` passed focused CPU, affected regressions, two-rank DDP, source-history audit and exact head-versus-base comparison.

### Canonical public identity and dynamic layer naming phase

- 138 focused PLASTIC/gauge/optimizer/inline/controller/CUDA/interface/trainer/checkpoint/identity tests passed
- 141 affected CPU regression tests passed
- 200 parameterised subtests passed
- two-rank CPU DDP passed with zero model-state and optimizer-state divergence
- Python compile, shell syntax and `git diff --check` passed
- THOG source-history audit passed with zero violations
- direct identity tests prove fixed PLASTIC remains numeric, learned PLASTIC uses `L_dyn`, disabled runs are unchanged, internal sampling seed is absent, and retired checkpoint aliases compare semantically

Hosted recovery runners repeated all phase gates and published clean commit `5b7ae45c`. Ordinary hosted CI on descendant `e29b6491` passed focused CPU, affected regressions, two-rank DDP, source-history audit and exact head-versus-base comparison.

### Transition-only sampled-value diagnostic phase

- 139 focused PLASTIC/gauge/optimizer/inline/controller/CUDA/interface/trainer/checkpoint/identity/console tests passed
- 141 affected CPU regression tests passed
- 200 parameterised subtests passed
- two-rank CPU DDP passed with zero model-state and optimizer-state divergence
- direct integration proves samples are computed after the atomic chart transition and match direct semantic materialisation exactly
- direct console tests prove a non-interval transition forces one ordinary optimizer row, consumes samples once, formats compactly, and does not repeat them on validation
- metrics and checkpoint surfaces remain unchanged
- Python compile, shell syntax and `git diff --check` passed
- THOG source-history audit passed with zero violations

Hosted recovery runners repeated all phase gates and published clean commit `612e28f7`. This log-only descendant triggers ordinary hosted head-versus-base validation for the published tree.

### Checkpoint-format and contradictory-metadata phase

The checkpoint-format phase completed in three commits. Commit `31ccee2a` added the explicit active-prefix/gauge format discriminator and broad v0.1 rejection coverage. Commit `e2acf6f2` hardened resume-control preflight and enabled/canonical-identity consistency. Commit `7c2ac900` rejected the remaining contradictory disabled-trainer/PLASTIC-metadata case before model construction or state loading.

Validation evidence:

- local direct checkpoint-format tests: 13 passed after the final contradiction guard
- local focused PLASTIC suite: 126 passed after the final contradiction guard
- hosted contradiction gate: patch SHA verified, Python compile passed, shell syntax passed, focused PLASTIC tests passed, checkpoint/resume regressions passed, two-rank CPU DDP passed
- ordinary hosted CI at `e2acf6f2`: focused CPU passed, affected regressions passed, two-rank DDP passed, source-history audit passed, exact head-versus-base comparison passed

Direct coverage proves canonical and retired-short v0.3 checkpoints still resume, explicit v0.1/versionless/unknown/inconsistent format payloads are rejected before state restoration, compact inference is guarded before model construction, CLI resume-control preflight is guarded before `TrainingConfig` reconstruction, and disabled trainer state cannot carry PLASTIC metadata into model loading.

## Remaining revised implementation

- [x] MAD significance gate and five-update count brake
- [x] Universal VRAM reserve and recoverable upward-probe fallback
- [x] Uniform public `plastic__` controls and `_L_dyn_` identity
- [x] Sampled-value transition diagnostic
- [x] Checkpoint versioning and rejection of ambiguous v0.1 geometry
- [x] Broad head-versus-base regression and final hosted CI
- [x] GPU smoke handoff
- [x] As-built specification and final download stanza

## Current exact task

Implementation is complete through hosted CPU/DDP validation. The remaining project-hardware task is to run `plastic_depth_gpu_smokes.sh` on the intended CUDA machine with the project OWT dataset.

## Known issues to carry forward

- AdamW diagonal second-moment state cannot exactly represent covariance introduced by coefficient mixing.
- Re-gauge preparation may temporarily duplicate coefficient storage.
- Count-dependent residual-addition scaling remains a separate architecture question.
- Selected N+1 head/backward OOM after feasibility selection remains fatal; only the detached upward probe is recoverable.

## Takeover instructions

1. Treat this file as authoritative.
2. Fetch `PLASTIC_DEPTH` and verify `7c2ac900372240f3411532c7479983c909cfdcf4` or a documented descendant.
3. Run `plastic_depth_gpu_smokes.sh` on the target CUDA host before merge.
4. Inspect all four smoke cases: fixed random geometry, lowest-loss count learning, relative wall-time count learning and memory-budget count learning.
5. Keep this log updated if additional changes are made.
