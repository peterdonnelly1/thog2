# THOG2 PLASTIC Requirements Specification

Version 1.3
Date: 2026-08-11
Status: Normative sampling-chaos-bump specification

## 1. Version scope

Version 1.3 adds the sampling-coordinate form of `chaos_bump` to the PLASTIC training system. It does not specify or implement `chaos_bump__depth_change__`.

The feature is a bounded, reversible excursion in the active PLASTIC sampling geometry. It is intended to test whether temporarily training the shared compact coefficient field at nearby depth coordinates can dislodge an early-stabilised optimisation trajectory.

The feature shall be disabled by default and shall leave the established training path, model state, optimiser state, checkpoint contents and numerical results unchanged when disabled.

## 2. Governing behaviour

A sampling chaos bump shall:

1. begin only at an optimiser-update boundary;
2. preserve the active layer count present at bump start;
3. construct one randomly rattled set of active sampling coordinates;
4. hold that one rattled geometry fixed for the complete bump;
5. continue ordinary coefficient and non-geometry parameter training;
6. prevent persistent sampling-geometry learning throughout the bump;
7. prevent all dynamic-depth probes, decisions and layer-count changes throughout the bump; and
8. remove the override at expiry, returning exactly to the persistent geometry that existed at bump start.

The persistent sampling parameters and their optimiser moments shall not be copied, replaced, reset or otherwise modified by bump entry or exit.

## 3. Public controls

| Control | Type | Default | Requirement |
|---|---:|---:|---|
| `chaos_bump__sampling__enabled` | bool | `false` | Enables sampling-coordinate chaos bumps. It requires `plastic__enabled=true`. |
| `chaos_bump__sampling__initial_lockout__steps` | integer | `16` | Number of successfully completed FINE updates before the first bump may start. |
| `chaos_bump__sampling__maximum_bumps` | integer | `1` | Maximum number of bumps in one logical run. Must be at least one when enabled. |
| `chaos_bump__sampling__interlude__min_steps` | integer | `128` | Minimum successfully completed non-bump updates between consecutive bumps. |
| `chaos_bump__sampling__interlude__max_steps` | integer | `256` | Maximum successfully completed non-bump updates between consecutive bumps. |
| `chaos_bump__sampling__duration__min_steps` | integer | `16` | Inclusive minimum duration in successfully completed optimiser updates. |
| `chaos_bump__sampling__duration__max_steps` | integer | `256` | Deterministic upper cap on bump duration. |
| `chaos_bump__sampling__duration__max_fraction_of_elapsed_steps` | float | `0.05` | Makes the random duration ceiling grow with elapsed successful training, subject to the minimum and deterministic cap. |
| `chaos_bump__sampling__max_movement_fraction_of_local_gap` | float | `0.10` | Maximum fraction of the currently available neighbouring gap through which one active coordinate may move. |

All step-count controls shall reject booleans. Lockout may be zero. Interlude minima and maxima shall be non-negative. Duration bounds shall be positive. Every minimum shall be less than or equal to its corresponding maximum. The duration fraction shall be finite and positive. The movement fraction shall be finite and lie in `(0, 1]`.

The public CLI shall use the exact names above, with `--` prefixes and `--no-chaos_bump__sampling__enabled` as the explicit Boolean disable form. Single-underscore hierarchy spellings and hyphenated aliases shall not be accepted.

## 4. Eligibility and recurrence

The feature operates only during FINE training. COARSE trials shall not start or advance a sampling bump.

If `plastic__freeze_geometry_during_warmup=true`, the first bump shall additionally remain locked out until ordinary warmup has completed. The first eligible update is therefore:

```text
max(chaos_bump__sampling__initial_lockout__steps, warmup_updates) + 1
```

The first bump shall start on that update exactly. It is not randomly delayed.

After a bump ends, the next interlude shall be drawn uniformly from the inclusive integer interval:

```text
[chaos_bump__sampling__interlude__min_steps,
 chaos_bump__sampling__interlude__max_steps]
```

The next bump shall start only after that many successfully completed non-bump updates. Failed or skipped non-finite attempts shall not consume lockout, interlude or duration.

No new bump shall start after `chaos_bump__sampling__maximum_bumps` bumps have started.

## 5. Duration

For a bump starting on successful update number `s`, its inclusive random upper duration bound shall be:

```text
duration_high(s) = min(
    chaos_bump__sampling__duration__max_steps,
    max(
        chaos_bump__sampling__duration__min_steps,
        floor(
            chaos_bump__sampling__duration__max_fraction_of_elapsed_steps * s
        )
    )
)
```

The realised duration shall be drawn uniformly from the inclusive integer interval:

```text
[chaos_bump__sampling__duration__min_steps, duration_high(s)]
```

One draw is made at bump start. The duration is not redrawn after restart.

## 6. Coordinate rattle

Let the persistent active public coordinates at bump start be `x[0:N]`. Construction shall use the following rules:

1. Copy `x[0:N]` as the immutable base geometry for the bump.
2. Randomly permute the movable coordinate identities.
3. Visit each movable identity once in that order.
4. Uniformly select one currently feasible direction, left or right.
5. Draw a movement fraction uniformly from `[0, max_movement_fraction_of_local_gap]`.
6. Apply that fraction to the currently available gap in the selected direction.
7. Recalculate available gaps after every movement.
8. Preserve coordinate identity, strict ordering, public depth bounds and the lattice's minimum separation.

A coordinate anchored at public depth `1` shall remain fixed. A coordinate anchored at public depth `100` shall remain fixed. For a learned active prefix that ends before public depth `100`, the boundary at `100` is available to the last active coordinate but is not itself moved.

For a one-coordinate fixed-count lattice at public depth `50.5`, the coordinate may move toward either public boundary. A configuration that has no movable coordinate is valid; its bump shall be recorded as a zero-movement bump rather than changing layer count or persistent geometry.

Each coordinate receives its own direction and magnitude draw. Coordinates shall never swap identities.

## 7. Execution override and optimisation isolation

The rattle shall be installed as a detached forward-pass coordinate override. The persistent `raw_intervals` tensor shall remain the authoritative checkpointed geometry.

While the override is active:

- every forward and activation-checkpoint recomputation shall use the same rattled active coordinates;
- inactive capacity coordinates shall remain unchanged and inert;
- `raw_intervals` shall receive no gradient;
- no geometry optimiser update or weight decay shall affect `raw_intervals`;
- all non-geometry trainable parameters shall follow the ordinary optimiser path;
- the active layer count shall equal the count captured at bump start;
- no dynamic-depth candidate forward, probe history update, decision or gauge transition shall occur.

The implementation shall assert the active count at every bump update and fail loudly if another path changes it.

## 8. Exit and controller reset

After the final successful update of a bump:

1. clear the execution override;
2. verify that persistent public coordinates still equal the captured base geometry;
3. clear stale dynamic-depth probe evidence;
4. prevent new dynamic-depth probing for the configured `plastic__layer_count_update_brake` interval; and
5. schedule the next interlude if another bump is permitted.

Ordinary geometry learning may resume on the next eligible non-bump update. The persistent geometry is not rewound because it was never changed during the bump.

## 9. Determinism, DDP and checkpoint/resume

Random draws shall come from an isolated deterministic stream derived from `model_seed` and the bump number. The stream shall not consume or perturb model, dropout, batch-source or global PyTorch RNG state.

Every DDP rank shall resolve the same start, duration, interlude and rattled coordinates. The implementation shall verify distributed equality before executing the first bumped forward.

Checkpoint state shall preserve or deterministically reconstruct:

- scheduler version;
- number of bumps started;
- active/inactive state;
- active bump number;
- start and final update numbers;
- realised duration;
- next scheduled start;
- captured active-layer count;
- base coordinates;
- rattled coordinates;
- last bump-end update and post-bump probe lockout; and
- deterministic draw identity.

Resuming inside a bump shall reinstall the exact stored rattled coordinates before evaluation or training. It shall not redraw the bump or restart its duration. Forking shall inherit the checkpointed bump state under the existing material-configuration rules.

## 10. Diagnostics

The startup report shall show every public control when sampling chaos bump is enabled.

Bump start shall record:

- bump number;
- start and final update numbers;
- realised duration;
- active-layer count;
- base and rattled coordinates;
- per-coordinate signed and absolute movements; and
- mean and maximum absolute movement.

Every ordinary progress row during a bump shall carry a concise terminal marker containing bump number and progress within the realised duration. The bump-entry and bump-exit updates shall force progress rows even when they fall outside the normal logging cadence.

Bump exit shall record successful duration, restored persistent coordinates, next scheduled start and post-bump probe-lockout boundary.

Machine-readable update metrics and telemetry shall expose active state, bump number, bump step, duration, mean movement and maximum movement. Diagnostics shall use full-precision coordinates; ordinary console coordinate formatting may remain compact.

## 11. Acceptance criteria

`CB-S-001` Disabled-path state dictionaries, optimiser groups, progress output and numerical results are unchanged.

`CB-S-002` A forced bump starts on the specified first eligible update and lasts exactly the sampled number of successful updates.

`CB-S-003` The rattled coordinates are ordered, bounded, identity-preserving and individually limited by the configured local-gap fraction under the sequential algorithm.

`CB-S-004` The same rattled coordinates are used for every microbatch and activation-checkpoint recomputation in one bump.

`CB-S-005` Persistent sampling coordinates, `raw_intervals` and their optimiser moments are bitwise unchanged from bump entry to bump exit.

`CB-S-006` Active layer count is constant and no probe or count-decision event occurs during a bump.

`CB-S-007` Clearing the override restores the exact pre-bump persistent coordinates.

`CB-S-008` Checkpoint/resume inside a bump reproduces uninterrupted training state, schedule and next update behaviour.

`CB-S-009` DDP ranks execute identical bump state and coordinates.

`CB-S-010` Warmup lockout, recurrence interludes, maximum bump count, post-bump brake and non-finite retry semantics are covered by focused tests.

`CB-S-011` Existing PLASTIC fixed-count, dynamic-depth, COARSE/FINE, checkpoint, CPU and available GPU/DDP regression suites continue to pass.

## 12. Source-change discipline

All changes shall follow the THOG source-marking rules. Inherited nanoGPT lines shall be commented rather than deleted. Small changes shall carry a trailing `# <<< THOG` marker at column 156 with a clarifying explanation; larger inserted or replaced blocks shall be enclosed by `# vvv THOG` and `# ^^^ THOG`.
