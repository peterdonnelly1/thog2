# THOG2 Coefficient Salience Measurement Specification v0.1

## 1. Purpose

This enhancement measures whether the coefficient directions used by THOG2 are materially unequal in their importance to model loss.

The motivating question is deliberately narrow:

> Are some learned coefficient directions much more valuable than others, strongly enough that future THOG2 variants should allocate representational capacity to them unequally?

This stage is diagnostic only. It must establish whether a useful salience signal exists before any salience-aware compression, optimization, phase adjustment, or learned basis rotation is attempted.

## 2. Scope

The measurement applies to any THOG2 compressor whose learned representation exposes a discrete coefficient/order axis. The implementation must not assume that low order is more important than high order, or that coefficient order has any particular frequency interpretation.

Initial validation shall cover the existing Chebyshev and DCT paths. The mechanism should remain basis-agnostic where the coefficient layout permits it.

A **measurement scope** is a selected set of compatible coefficient banks. A coefficient-order ablation acts on order `k` across every coefficient bank in that scope.

The default scope should be one geometry/registry target at a time. A broader aggregate scope may also be supported to ask whether an order is important across a whole compressor configuration.

This specification does not introduce new trainable parameters and does not alter training behaviour.

## 3. Primary measurement: counterfactual loss impact

Coefficient salience shall be defined primarily by direct counterfactual effect on held-out loss, not by coefficient magnitude.

For a trained checkpoint with baseline validation loss `L0`, temporarily zero coefficient order `k` in the selected scope and evaluate the identical validation examples:

`delta_loss_zero[k] = L(coefficient_order_k = 0) - L0`

Interpretation:

- positive `delta_loss_zero[k]`: the trained model depends on that coefficient order;
- approximately zero: removal has little measurable effect on the evaluated loss;
- negative: removing the order improves the evaluated loss and must be reported as such, not clipped away.

The signed value is the authoritative measurement. Ranking by coefficient magnitude alone is explicitly out of scope because a large coefficient need not be loss-important and a small coefficient can be highly sensitive.

Zero-ablation is intentionally not normalized by coefficient magnitude. The question being asked is the practical one: **how much does the trained model suffer if this learned coefficient direction is removed?**

## 4. Secondary diagnostics

For each measured order, the analyzer shall also report inexpensive descriptive quantities that may explain or predict the direct ablation result:

- coefficient RMS;
- gradient RMS on a fixed calibration/evaluation sample;
- first-order removal proxy, based on the elementwise product of coefficient and gradient, aggregated consistently within the scope.

These are diagnostics, not substitutes for the ablation measurement. Their usefulness is itself empirical: later work may use them as cheap online salience estimators only if they correlate well with direct loss impact.

No optimizer step may occur while collecting gradient diagnostics.

## 5. Measurement protocol

The measurement shall be implemented as an offline, non-mutating analysis path rather than inserted into the normal training loop.

For each run:

1. Load a specified checkpoint and resolve the selected coefficient banks and their order axis through the existing THOG2 geometry/compressor structure.
2. Put the model in evaluation mode and freeze training behaviour.
3. Select one deterministic validation sample set and partition it into fixed evaluation shards. Four shards is the default for the first implementation.
4. Evaluate the unmodified checkpoint on every shard to establish the shard baselines.
5. For each coefficient order `k`:
   - temporarily zero that order in the selected scope;
   - evaluate the exact same shards;
   - record the signed loss change for every shard;
   - restore the original coefficient values exactly before continuing.
6. Collect the secondary coefficient/gradient diagnostics without updating model state.
7. Emit machine-readable results and a concise human-readable summary.

All coefficient orders must be evaluated against identical data. The analyzer must never advance to different validation examples merely because it is testing a different order.

A salience run must leave the checkpoint and in-memory model state unchanged when it finishes.

If coefficient banks in a requested scope do not share a compatible order definition, the analyzer must reject the scope rather than silently reinterpret or truncate it.

## 6. Aggregation and concentration

For each order, report the mean signed loss change across evaluation shards and its variation across shards.

To characterize how concentrated the useful coefficient space is, define a non-negative value only for the concentration calculation:

`s[k] = max(mean_delta_loss_zero[k], 0)`

The original signed measurements remain the source data.

When `sum(s) > 0`, report the effective salience dimension:

`d_eff = (sum_k s[k])^2 / sum_k s[k]^2`

Interpretation:

- `d_eff` near the total order count `P` indicates broadly distributed salience;
- `d_eff << P` indicates that loss dependence is concentrated into substantially fewer coefficient orders.

Also report the fraction of total positive salience carried by the most salient quartile of orders.

The report must preserve natural order as well as a second ranking by measured salience. It must not assume that salience decreases monotonically with coefficient order.

## 7. Reproducibility and controls

A salience result is useful only if the ranking is larger than evaluation noise and reasonably stable.

The analyzer shall therefore:

- use a fixed checkpoint, validation token set, shard partition, precision, and random seed;
- use evaluation mode for every baseline and ablation;
- record all run metadata needed to reproduce the measurement;
- preserve per-shard results rather than only the aggregate mean;
- report rank stability between independent halves of the evaluation shards, using Spearman rank correlation when enough orders are present.

A strong salience signal for follow-on work is provisionally defined as:

- `d_eff / P <= 0.5`; and
- coefficient-order ranking stability of approximately `rho >= 0.8` across the independent evaluation halves.

These are engineering go/no-go thresholds, not claims of statistical significance. Results close to the thresholds should be treated as inconclusive and measured with more validation tokens before any architectural conclusion is drawn.

If the spectrum is nearly flat or the ranking is unstable, salience-based capacity allocation should not be pursued on the evidence of that run.

## 8. Required outputs

Each salience run shall produce a structured result containing, at minimum:

- checkpoint/artifact identity;
- git commit and complete geometry/run descriptor;
- compressor and measurement scope;
- coefficient order count `P`;
- validation token count and shard definition;
- per-order coefficient count;
- coefficient RMS;
- gradient RMS;
- first-order removal proxy;
- baseline loss;
- ablated loss;
- signed `delta_loss_zero`;
- per-shard loss changes;
- salience rank;
- `d_eff`, `d_eff / P`, top-quartile salience fraction, and rank-stability statistic.

The primary machine-readable format should be simple enough for direct analysis and plotting without parsing console logs. CSV plus a small metadata JSON is sufficient.

The console summary should show the baseline, the salience spectrum by natural order, the same orders ranked by salience, and the concentration/reproducibility statistics.

## 9. Verification requirements

The implementation is acceptable only when:

1. Repeated analysis of the same checkpoint and fixed evaluation sample produces the same results within the numerical tolerance expected for the selected backend and precision.
2. Baseline loss matches the existing THOG2 evaluation path on the same examples.
3. Ablating an order changes only the intended coefficient slice and restoration is exact.
4. No optimizer state, model parameter, checkpoint, or training artifact is modified.
5. A tiny CPU configuration can exercise the complete measurement path.
6. At least one real Chebyshev checkpoint is analyzed end-to-end before any salience-aware enhancement is designed.

## 10. Explicit non-goals

This stage shall not implement:

- salience-weighted learning rates or optimizer behaviour;
- coefficient pruning or unequal persistent-state allocation;
- phase-shifted harmonic bases;
- learned coefficient-space rotations;
- automatic replacement of the existing compressor order selection.

Those are possible later enhancements. Their justification depends on the result of this measurement.

## 11. Decision produced by this stage

The output of this work is not a better compressor. It is a defensible answer to one question:

> Does THOG2's learned coefficient space contain stable, strongly unequal loss-relevant directions that are worth exploiting?

If yes, the next enhancement should target salience-based capacity allocation. If no, the salience branch should stop rather than adding machinery to a signal that is not there.
