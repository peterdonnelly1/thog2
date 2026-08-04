# PLASTIC DEPTH implementation log

Last updated: 2026-08-04 19:02 AEST

## Critical correction retained

The earlier log incorrectly described an unpushed local implementation as completed. Commit `dedf942` did not exist. Work has since resumed from verified GitHub source; only phases explicitly marked complete below are implemented and pushed.

## Repository state

- Repository: `peterdonnelly1/thog2`
- Working branch: `PLASTIC_DEPTH`
- Pull request: #29, draft
- Base: `HYPERBLOCK_LOOP_ENHANCEMENTS`
- Current verified branch head: `b7231c2b74313fc510bcf5a8700ab356265c006b`
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

## Active-prefix validation at `b7231c2`

- payload SHA verification and `git apply --check`: passed
- Python compile and shell syntax: passed
- focused PLASTIC/gauge/interface/cache tests: 69 passed
- affected CPU regression tests: 160 passed
- parameterised subtests: 200 passed
- two-rank CPU DDP: zero model-state and optimiser-state divergence
- temporary transport workflows and payloads removed after successful commit

## Remaining revised implementation

- [ ] Atomic add/subtract gauge transition preserving generated weights
- [ ] Stock AdamW state migration or explicit reset policy
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

Implement an atomic model-wide add/subtract chart re-gauge using the existing exact affine Chebyshev kernel. Verify all generated parameter families before commit. Do not enable runtime count transitions until this phase and its optimiser-state policy pass.

## Known issues to carry forward

- AdamW diagonal second-moment state cannot exactly represent covariance introduced by coefficient mixing.
- Re-gauge preparation may temporarily duplicate coefficient storage.
- Count-dependent residual-addition scaling remains a separate architecture question.
- Old globally normalised phantom-lattice checkpoints are geometrically ambiguous.

## Takeover instructions

1. Treat this file as authoritative.
2. Fetch `PLASTIC_DEPTH` and verify head `b7231c2` or a documented descendant.
3. Run the focused command before new work.
4. Continue only the current exact task.
5. Record each tested and pushed phase here before proceeding.
