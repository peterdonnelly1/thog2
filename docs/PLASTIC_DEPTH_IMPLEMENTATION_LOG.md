# PLASTIC DEPTH implementation log

Last updated: 2026-08-04 21:57 AEST

## Critical correction retained

The earlier log incorrectly described an unpushed local implementation as completed. Commit `dedf942` did not exist. Work resumed from verified GitHub source; only phases explicitly marked complete below are implemented and pushed.

## Repository state

- Repository: `peterdonnelly1/thog2`
- Working branch: `PLASTIC_DEPTH`
- Pull request: #29, draft
- Base: `HYPERBLOCK_LOOP_ENHANCEMENTS`
- Current verified branch head before this log update: `e0d0a3fb80ced2b2c636c3bc3fb80d7ceb778282`
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

## Validation evidence

### Active-prefix checkpoint `b7231c2`

- focused PLASTIC/gauge/interface/cache tests: 69 passed
- affected CPU regression tests: 160 passed
- parameterised subtests: 200 passed
- two-rank CPU DDP: zero model-state and optimiser-state divergence

### Atomic gauge-transition checkpoint `de57d61f`

The gated applier verified patch SHA, compile/shell syntax, focused and affected tests, and two-rank CPU DDP before commit.

### AdamW-state checkpoint `e0d0a3fb`

The gated applier verified the patch SHA and ran before committing:

- Python compile and shell syntax
- 103 focused/affected CPU tests
- 10 direct AdamW-state tests within that set
- two-rank CPU DDP with zero model-state and optimiser-state divergence

Direct coverage includes transformed first/second moments, AMSGrad, retained steps, uninitialised state, unaffected-state identity, stale-state aborts, non-AdamW rejection, reset fallback and a subsequent unmodified stock AdamW step.

This log commit retriggers ordinary hosted validation for the exact post-AdamW branch head.

## Remaining revised implementation

- [ ] Shared inline N-1/N/N+1 first-microstep probe
- [ ] MAD significance gate and five-update count brake
- [ ] Universal VRAM reserve and recoverable upward-probe fallback
- [ ] Uniform public `plastic__` controls and `_L_dyn_` identity
- [ ] Sampled-value transition diagnostic
- [ ] Checkpoint versioning and rejection of ambiguous v0.1 geometry
- [ ] Broad head-versus-base regression and final hosted CI
- [ ] GPU smoke handoff
- [ ] As-built specification and final download stanza

## Current exact task

Replace the external separate-forward layer-count controller with a shared first-microstep transformer chain that evaluates exactly N-1, N and N+1 depth checkpoints on the same examples and sampled token positions. Candidate heads are detached/no-grad; after selection, run one normal grad-bearing head from the selected hidden state and backpropagate through only the selected prefix. Use the selected count for all remaining accumulation microsteps. Do not yet add the MAD gate or count brake beyond the minimum plumbing required for deterministic selection.

## Known issues to carry forward

- AdamW diagonal second-moment state cannot exactly represent covariance introduced by coefficient mixing.
- Re-gauge preparation may temporarily duplicate coefficient storage.
- Count-dependent residual-addition scaling remains a separate architecture question.
- Old globally normalised phantom-lattice checkpoints are geometrically ambiguous.
- Upward-probe OOM recovery and DDP feasibility are separate from the core shared-chain implementation and remain pending.

## Takeover instructions

1. Treat this file as authoritative.
2. Fetch `PLASTIC_DEPTH` and verify `e0d0a3fb` or a documented descendant.
3. Run the focused PLASTIC/gauge/optimizer/interface/cache command before new work.
4. Continue only the current exact task.
5. Record each tested and pushed phase here before proceeding.
