# PLASTIC DEPTH implementation log

Last updated: 2026-08-04 23:28 AEST

## Critical correction retained

The earlier log incorrectly described an unpushed local implementation as completed. Commit `dedf942` did not exist. Work resumed from verified GitHub source; only phases explicitly marked complete below are implemented and pushed.

## Repository state

- Repository: `peterdonnelly1/thog2`
- Working branch: `PLASTIC_DEPTH`
- Pull request: #29, draft
- Base: `HYPERBLOCK_LOOP_ENHANCEMENTS`
- Current verified branch head before this log update: `855ea5f209a0554c4b2a128dcc0914e81d2d5ba6`
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

The original staged transport payload at `6b0b759c` was corrupt after the first six files. The phase was recovered from its intact controller/config fragments, completed against the authoritative task, and validated locally before direct commit.

- 122 focused PLASTIC/gauge/optimizer/inline/controller/interface/trainer/checkpoint tests passed
- 141 affected CPU regression tests passed
- 200 parameterised subtests passed
- two-rank CPU DDP passed with zero model-state and optimizer-state divergence
- Python compile, shell syntax and `git diff --check` passed
- THOG source-history audit passed with zero violations
- direct integration proves failed updates do not checkpoint evidence and count changes at updates 1 and 6 under the default five-update brake

Hosted head-versus-base CI remains the authoritative broad regression gate for the committed phase.

## Remaining revised implementation

- [x] MAD significance gate and five-update count brake
- [ ] Universal VRAM reserve and recoverable upward-probe fallback
- [ ] Uniform public `plastic__` controls and `_L_dyn_` identity
- [ ] Sampled-value transition diagnostic
- [ ] Checkpoint versioning and rejection of ambiguous v0.1 geometry
- [ ] Broad head-versus-base regression and final hosted CI
- [ ] GPU smoke handoff
- [ ] As-built specification and final download stanza

## Current exact task

Implement the universal CUDA allocator reserve and recoverable upward-probe fallback. Reserve the configured safety margin before learned-count probing, preflight N+1 feasibility without adding a separate timed model probe, synchronize the feasibility decision across ranks, and treat an upward-probe allocation failure as an infeasible N+1 candidate rather than a fatal update. Preserve N and N-1 evaluation and the exact non-CUDA path. Do not yet change public artifact identity, sampled-value UI, or checkpoint-version rejection semantics.

## Known issues to carry forward

- AdamW diagonal second-moment state cannot exactly represent covariance introduced by coefficient mixing.
- Re-gauge preparation may temporarily duplicate coefficient storage.
- Count-dependent residual-addition scaling remains a separate architecture question.
- Old globally normalised phantom-lattice checkpoints are geometrically ambiguous.
- Upward-probe OOM recovery and DDP feasibility remain pending.

## Takeover instructions

1. Treat this file as authoritative.
2. Fetch `PLASTIC_DEPTH` and verify `855ea5f2` or a documented descendant.
3. Run the focused PLASTIC/gauge/optimizer/inline/interface/cache command before new work.
4. Continue only the current exact task.
5. Record each tested and pushed phase here before proceeding.
