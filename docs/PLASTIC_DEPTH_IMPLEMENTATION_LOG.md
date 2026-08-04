# PLASTIC DEPTH implementation log

Last updated: 2026-08-04 21:42 AEST

## Critical correction retained

The earlier log incorrectly described an unpushed local implementation as completed. Commit `dedf942` did not exist. Work resumed from verified GitHub source; only phases explicitly marked complete below are implemented and pushed.

## Repository state

- Repository: `peterdonnelly1/thog2`
- Working branch: `PLASTIC_DEPTH`
- Pull request: #29, draft
- Base: `HYPERBLOCK_LOOP_ENHANCEMENTS`
- Current verified branch head before this log update: `de57d61ff6961ae2f330c07618f5d212ea24ab1a`
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

## Validation evidence

### Active-prefix checkpoint `b7231c2`

- payload SHA verification and `git apply --check`: passed
- Python compile and shell syntax: passed
- focused PLASTIC/gauge/interface/cache tests: 69 passed
- affected CPU regression tests: 160 passed
- parameterised subtests: 200 passed
- two-rank CPU DDP: zero model-state and optimiser-state divergence

### Atomic gauge-transition checkpoint `de57d61f`

The gated applier that produced the commit ran, before committing:

- patch SHA verification and `git apply --check`
- Python compile and shell syntax
- focused PLASTIC/gauge/interface/cache tests
- affected DEPTH and trainer regressions
- two-rank CPU DDP check

Ordinary branch CI is being retriggered by this log commit so the exact post-transition head receives a normal validation artifact.

## Remaining revised implementation

- [ ] Stock AdamW state migration with logged reset fallback
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

Integrate the atomic model transition with stock PyTorch AdamW state. Transform coefficient first moments with the inverse-transpose covector rule, approximate diagonal second moments with the elementwise-squared inverse-transpose rule, retain step counters, and use a logged affected-state reset fallback for non-finite, singular, ill-conditioned or verification-failing migration. Do not reimplement AdamW and do not enable runtime count transitions until this policy passes direct tests.

## Known issues to carry forward

- AdamW diagonal second-moment state cannot exactly represent covariance introduced by coefficient mixing.
- Re-gauge preparation may temporarily duplicate coefficient storage.
- Count-dependent residual-addition scaling remains a separate architecture question.
- Old globally normalised phantom-lattice checkpoints are geometrically ambiguous.
- A failed optimiser-state migration must not roll back or approximate the already verified coefficient transform silently; it must select and log the defined reset fallback before atomic commit.

## Takeover instructions

1. Treat this file as authoritative.
2. Fetch `PLASTIC_DEPTH` and verify `de57d61f` or a documented descendant.
3. Run the focused PLASTIC/gauge/interface/cache command before new work.
4. Continue only the current exact task.
5. Record each tested and pushed phase here before proceeding.
