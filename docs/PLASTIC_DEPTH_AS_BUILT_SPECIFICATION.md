# PLASTIC DEPTH as-built specification

Status: as built on `PLASTIC_DEPTH` after checkpoint-format and contradictory-metadata hardening.
Base branch: `HYPERBLOCK_LOOP_ENHANCEMENTS`.
Primary PR: #29.

## 1. Purpose

PLASTIC DEPTH makes the DEPTH axis adaptive without changing the persistent coefficient tensor capacity. The model keeps a fixed maximum layer capacity, but executes only an active prefix of sampled notional layer positions. In learned-count mode, the active layer count can move between adjacent counts after a guarded probe and an atomic gauge transition.

The implementation is deliberately conservative. It preserves disabled-path behaviour, preserves existing non-PLASTIC checkpoints, accepts semantically compatible v0.3 PLASTIC checkpoints, and rejects old or ambiguous PLASTIC geometry instead of silently approximating it.

## 2. Public controls

All new public PLASTIC controls use the `plastic__` prefix. The retired short aliases are not used for new identity surfaces.

Core controls:

- `plastic__enabled`: enables PLASTIC DEPTH.
- `plastic__layers_to_sample`: fixed active layer count when count learning is disabled.
- `plastic__do_learn_layer_count`: enables learned active-count control.
- `plastic__initial_layer_count`: initial active count for learned-count mode.
- `plastic__max_permitted_layers`: maximum allocation/count capacity.
- `plastic__layer_sampling_initialisation`: initial active lattice placement.
- `plastic__layer_count_objective`: count-selection objective.
- `plastic__layer_count_update_brake`: minimum successful-update spacing between count changes.
- `plastic__layer_count_probe__window_size_as_number_of_probes`: rolling paired-score window length.
- Full-window readiness: no separate minimum-probes knob; count movement is ineligible until the complete configured probe-history window is present.
- `plastic__layer_count_extrapolation_weight`: right/up directional-credibility and distance discount; default 0.8.
- `plastic__layer_count_probe_noise_lambda`: robust MAD significance multiplier.
- `plastic__layer_count_cost_weight`: cost penalty for count objectives that use it.
- `plastic__layer_memory_budget_gib`: memory-budget objective input.
- `plastic__cuda_allocator_reserve_gib`: CUDA upward-probe safety reserve.
- `plastic__geometry_learning_rate_multiplier`: sampling-geometry optimizer multiplier.
- `plastic__freeze_geometry_during_warmup`: disables geometry updates during warmup when set.

Internal `plastic__sampling_seed` remains internal and is absent from public identity.

## 3. Public layer identity

Fixed-count PLASTIC and disabled runs retain numeric layer identity: `_L_<count>_`.

Learned-count PLASTIC runs emit `_L_dyn_`, because the number of executed layers can change during training. `L_dyn` applies only when layer-count learning is enabled.

## 4. Coordinate semantics

The public DEPTH ruler is the notional 1-100 layer axis. Internally, coordinates are mapped into the existing Chebyshev coordinate system.

The active-prefix rule is:

- execute exactly `N` active coordinates;
- materialise only the active prefix for normal execution;
- derive at most one adjacent `N + 1` probe coordinate when learned-count probing needs it;
- treat maximum permitted layers as capacity only, not as a source of active coordinate spacing.

Inactive capacity must not affect active coordinates, logits, optimizer gradients, or count decisions. Tests explicitly mutate inactive capacity and verify active outputs and gradients are unchanged.

## 5. Chebyshev re-expression and gauge transitions

PLASTIC DEPTH uses exact float64 affine Chebyshev re-expression for chart changes. Gauge transitions are prepared non-mutatingly, checked for stale geometry/coefficient versions, then committed atomically.

The atomic transition rule is:

1. Prepare the candidate chart and coefficient transforms without mutating model state.
2. Verify every generated coefficient family and optimizer state migration.
3. Complete the successful stock AdamW optimizer step.
4. Commit model gauge, active count, transformed coefficients, and migrated optimizer state together.
5. Discard prepared state if any check fails.

Existing nanoGPT/source-history lines are preserved under THOG markers rather than deleted.

## 6. Optimizer state migration

Stock PyTorch AdamW remains the optimizer. PLASTIC DEPTH migrates optimizer state across gauge transitions:

- first moments use the inverse-transpose covector rule;
- diagonal second moments and optional AMSGrad maxima use a squared-covector approximation;
- AdamW step counters are preserved;
- non-finite or ill-conditioned migrations choose a logged zero-moment reset fallback before model mutation.

The diagonal second-moment approximation is a known limitation: it cannot exactly represent covariance induced by coefficient mixing.

## 7. Learned-count probing

Learned-count mode evaluates adjacent active counts with a shared first-microstep transformer chain over `N - 1`, `N`, and `N + 1` where feasible.

Probe rules:

- candidate heads are detached and no-grad;
- candidate scores use common sampled token positions;
- only the selected head becomes grad-bearing;
- the selected prefix is reused for remaining accumulation microsteps;
- persistent count and gauge do not change during forward/backward or the stock AdamW step;
- evidence is checkpointed only after a successful optimizer/model transition commit.

Failed or non-finite updates discard transient evidence and leave persistent controller state unchanged.

## 8. MAD gate and update brake

The count controller records paired per-count/per-direction inline score histories. Count changes require robust standardized improvement under a MAD-based significance gate.

Tie-breaking is deterministic and prefers the lower count. The five-update brake replaces the obsolete periodic hold controller, preventing too-frequent count changes even when the signal is favourable.

## 9. CUDA reserve and upward-probe recovery

Before an upward learned-count probe, CUDA execution attempts to acquire the configured allocator reserve. Reserve feasibility is synchronized across ranks before model execution.

Failure modes:

- failed reserve preflight removes only `N + 1` and preserves `N` and `N - 1`;
- the safety reserve is released immediately before the single adjacent `N + 1` layer attempt;
- CUDA OOM in the detached `N + 1` probe becomes an infeasible candidate rather than a failed update;
- N+1 feasibility is synchronized across ranks;
- any-rank failure discards `N + 1` everywhere.

Selected `N + 1` head/backward OOM after feasibility selection remains fatal. Only the detached upward probe is recoverable.

## 10. Diagnostics

Runtime diagnostics include active/full coordinates, lattice movement, separate coefficient/lattice gradient norms, timing, memory, and count decisions.

On a successful count transition, the console samples generated `attention_query_weight[0,0]` across the committed active chart. These sampled values are transient console diagnostics only. They do not enter metrics, telemetry, or checkpoints.

Sample formatting uses two decimal places normally and scientific notation when magnitude requires it. Transition rows are forced into the ordinary optimizer-row stream without an extra standalone line, and samples are consumed once only.

## 11. Checkpoint format and compatibility

New enabled PLASTIC checkpoints write:

`plastic_depth_checkpoint_format_version = "plastic_depth_active_prefix_gauge_v1"`

Accepted:

- disabled/non-PLASTIC legacy checkpoints;
- canonical v0.3 PLASTIC checkpoints;
- retired short-key v0.3 PLASTIC identity aliases when they normalize to the same semantics.

Rejected before model construction, optimizer restoration, compact inference loading, or CLI resume-control `TrainingConfig` reconstruction:

- explicit `plastic_depth_v0_1` phantom-lattice checkpoints;
- enabled PLASTIC checkpoints with no trustworthy version discriminator;
- unknown format discriminators;
- format/identity version mismatches;
- enabled trainer state whose canonical compact identity says `plastic__enabled = False`;
- disabled trainer state that still carries PLASTIC compact identity or PLASTIC format metadata.

There is no approximate conversion path from v0.1/global phantom-lattice geometry to the active-prefix gauge-preserving geometry.

## 12. Validation summary

Hosted and local validation covered:

- Python compilation and shell syntax;
- GPU smoke-script preflight;
- focused PLASTIC CPU tests;
- checkpoint-format compatibility tests;
- affected CPU regression tests;
- exact head-versus-base comparison against legacy tests;
- THOG source-history audit;
- two-rank CPU DDP synchronization with zero model-state and optimizer-state divergence.

The remaining validation requiring project hardware is execution of `plastic_depth_gpu_smokes.sh` on the intended CUDA machine and OWT dataset.

## 13. GPU smoke handoff

Use this on the project machine. Do not run `git clean`.

```bash
cd ~/git/thog2

git fetch origin

git switch PLASTIC_DEPTH
git pull --ff-only origin PLASTIC_DEPTH

git status --short

export THOG2_OWT_DATA_DIR=/path/to/openwebtext
export THOG2_PYTHON=.venv/bin/python
export PLASTIC_SMOKE_BACKEND=flash2
export PLASTIC_SMOKE_DTYPE=bfloat16

./plastic_depth_gpu_smokes.sh
```

The smoke script covers fixed random geometry, lowest-loss count learning, relative wall-time count learning, and memory-budget count learning.

## 14. Known limitations

- AdamW diagonal second moments cannot exactly represent covariance from coefficient mixing.
- Re-gauge preparation can temporarily duplicate coefficient storage.
- Count-dependent residual-addition scaling is a separate architecture question.
- Selected upward head/backward OOM after feasibility selection is still fatal.
