# THOG2 Recurrence Generators Requirements Specification v0.2

This specification supersedes v0.1 for implementation decisions on `RECURRENCE_GENERATORS_ENHANCEMENT`.

## 1. Purpose

THOG2 shall support **Recurrence Generators (RGs)** as compact materialisation families alongside the existing fixed-basis compressors.

The common contract is:

`small learned persistent state -> deterministic differentiable materialiser -> many ephemeral operational weights`

For an RG, the compact learned state parameterises a recurrent dynamical law rather than coefficients of a fixed basis.

The first RG is **BQRG — Bounded Quadratic Recurrent Generator**.

BQRG is the first plugin, not the architecture. Adding later RGs must not require copying or modifying BQRG-specific code.

## 2. Architectural invariants

1. The compact learned state is native persistent model state.
2. Generated operational weights are derived ephemeral state and must not be checkpointed.
3. Materialisation must remain differentiable end-to-end under ordinary PyTorch autograd.
4. No auxiliary neural network is introduced to generate weights.
5. Existing Chebyshev, DCT, Haar, lapped-cosine, JPEG-like and DENSE paths must remain unchanged unless an RG is explicitly selected.
6. Unsupported geometry, width, version or generator options must fail before training rather than silently falling back.
7. RG materialisation runs on the model parameter device. BQRG v1 must not silently migrate materialisation to CPU.

## 3. Registry model

### 3.1 Geometry-facing compressor registry

The existing `COMPRESSOR_REGISTRY` remains the public geometry-facing catalogue.

BQRG must be selectable through the same systematic geometry mechanism as other compressors:

`--select-depth --option DEPTH.compressor=bqrg`

The geometry layer must obtain RG capabilities from registry metadata rather than BQRG-specific conditionals.

### 3.2 Dedicated recurrence-generator registry

A dedicated `RECURRENCE_GENERATOR_REGISTRY` shall own recurrence-generator definitions.

Each definition shall expose at least:

- canonical family name;
- aliases;
- version;
- artifact/display tag;
- accepted persistent widths;
- supported targets;
- declared `generator_*` option names;
- differentiable materialisation callable;
- initialization callable;
- output-rescaling callable;
- human-readable description.

Registration must reject collisions in family names, aliases, versions and artifact tags.

A second RG should normally require one new implementation module, one registry definition and generator-local tests, with no BQRG edits.

### 3.3 Fixed-basis registry remains fixed-basis only

BQRG must **not** be inserted into `BASIS_REGISTRY` or advertised as a mathematical basis.

Compatibility plumbing may allow existing internal fields such as `basis_family` / `basis_version` to carry RG identity where avoiding a repo-wide rename is useful, but fixed-basis construction APIs must reject RGs because no fixed basis matrix exists.

## 4. Structured configuration

No new short/getopts letters are allocated to RGs.

Generator-specific options use the archetype:

`TARGET.generator_xxx=VALUE`

Examples for possible future RGs include:

`--option DEPTH.generator_state_dimension=4`

`--option DEPTH.generator_burn_in=2`

Rules:

- `generator_*` is accepted only for a registered RG;
- the selected RG must explicitly declare each supported option;
- unknown options fail with a clear error;
- fixed-basis compressors reject `generator_*` options;
- resolved values are stored in the resolved geometry plan and canonical run metadata.

BQRG v1 exposes no generator-specific tunable options.

## 5. BQRG v1 identity and scope

Canonical family: `bqrg`

Canonical version: `bqrg_v1`

Artifact/display tag: `BQRG`

Persistent width: exactly `P = 16`

Supported target: **DEPTH only**.

For the first implementation, BQRG is permitted only on the public pure-DEPTH trajectory path. It must be rejected for:

- non-DEPTH CURVE selectors;
- SHEET or SHEET_SET geometry;
- private legacy SHEET_COL fallback paths;
- BLOCK/JPEG/helper trajectories that instantiate DEPTH internally rather than selecting public DEPTH;
- any persistent width other than 16.

This restriction is deliberate. A later RG or later BQRG version may widen its capability only through explicit registry metadata and tests.

## 6. BQRG v1 persistent parameterisation

For each scalar DEPTH trajectory, the 16 learned persistent values are interpreted as:

`[x0, y0, a0..a5, b0..b5, mu, lambda]`

where:

- `x0`, `y0` are the initial two-state recurrent coordinates;
- `a0..a5` define the quadratic update for `x`;
- `b0..b5` define the quadratic update for `y`;
- `mu` is the output offset;
- `lambda` is the raw positive-scale parameter.

These 16 values replace the semantic role of the 16 Chebyshev coefficients for a `P16` comparison; no additional learned basis or generator parameters are introduced.

## 7. BQRG v1 recurrence

For state `(x_n, y_n)`, define:

`q_n = [1, x_n, y_n, x_n^2, x_n*y_n, y_n^2]`

Then:

`x_(n+1) = tanh(a . q_n)`

`y_(n+1) = tanh(b . q_n)`

The `tanh` state clamp is part of `bqrg_v1` and bounds the updated state coordinates to `(-1, 1)`.

Output is:

`scale = softplus(lambda)`

`w_n = mu + scale * x_n`

The first weight uses the initial state. Each subsequent weight follows one recurrence update.

The recurrence is deterministic for fixed persistent state, dtype and execution semantics.

## 8. Initialization

BQRG initialization must:

- use the existing model RNG/seed;
- avoid immediate `tanh` saturation;
- avoid exact collapse of both recurrent coordinates;
- initialize generated matrix weights near the intended nanoGPT/THOG scale;
- reproduce exact zero bias trajectories for zero-initialized bias families;
- reproduce exact one trajectories for LayerNorm weights when those vectors participate in DEPTH compression;
- require no dense reference model or fitted target trajectory.

Initialization behavior is versioned as part of `bqrg_v1` reproducibility.

## 9. Residual initialization scaling

Existing fixed-basis residual initialization may rescale all coefficients because materialisation is linear in those coefficients.

An RG must instead own an explicit output-rescaling operation.

For BQRG, multiplying generated weights by a positive factor must rescale only the output mapping (`mu` and the positive `softplus(lambda)` scale) and must leave the recurrent dynamics parameters unchanged.

Generic trainer/model-factory code must dispatch through the RG definition rather than contain `if bqrg` logic.

## 10. Differentiability and numerical behavior

Materialisation must not detach recurrent state, convert learned tensors to Python scalars, mutate persistent parameters, or register generated weights as trainable parameters.

Loss gradients must flow through every recurrence step into the 16 persistent parameters.

BQRG relies on:

- bounded recurrent state through `tanh`;
- positive output scale through `softplus`;
- the existing trainer non-finite update policy and gradient clipping.

BQRG must not add hidden generator-specific gradient clipping or optimizer recovery.

Hot-path materialisation must avoid unconditional device synchronization solely for diagnostic finite checks. Dedicated tests and the existing training non-finite policy provide the primary guards.

## 11. Device and execution policy

BQRG materialisation occurs on the same device as its persistent parameters.

There is no automatic CPU fallback in v1. Silent CPU fallback would introduce device transfers into the differentiable hot path and could make training catastrophically slow.

If an explicit alternate materialisation backend is later useful, it must be a declared execution option with benchmarks and tests rather than an emergency fallback hidden inside BQRG.

BQRG must remain compatible with eager execution and should be graph-friendly for `torch.compile`. Performance is empirical and must be measured separately from correctness.

## 12. Materialisation cost and memory

A logical length-`L` BQRG trajectory requires `L-1` recurrent state transitions to generate sequentially from its initial state.

The implementation must preserve THOG's ephemeral-memory objective: it must not materialise all layers' complete dense matrices simultaneously merely to avoid recurrence work.

The first correctness implementation may use stateless indexed materialisation, recomputing a trajectory prefix when a later layer is requested. This is correct but may perform more recurrence work than the theoretical `L-1` minimum over a whole forward pass.

If profiling shows this cost is material, later optimization should preserve correctness under activation checkpointing, layer dropout and compilation before introducing recurrent-state caching or segment-aware generation.

## 13. Geometry help and discoverability

`--print-geometry-registry` must include a dedicated recurrence-generator subsection showing at least:

- generator name;
- version;
- persistent widths;
- supported targets;
- declared `generator_*` options;
- description.

BQRG must also appear in the top-level compressor registry as a registered compressor while remaining absent from the fixed-basis registry.

`--explain-geometry` for BQRG must resolve the compressor as `bqrg@bqrg_v1`, report `P=16`, and show an implemented DEPTH materializer.

## 14. Run and checkpoint identity

A BQRG systematic run descriptor must identify the compressor as `bqrg`, e.g. the normal `G0_bqrg` geometry slot.

Canonical configuration/checkpoint metadata must preserve at least:

- RG family;
- version;
- persistent width;
- resolved geometry plan;
- declared generator options when present.

The authoritative checkpoint invariant remains:

> Checkpoint the model's native persistent representation, whatever form each parameter family uses. Never checkpoint derived materialisations.

For BQRG this means saving the 16 learned values per trajectory and the configuration required to interpret them, never generated layer weights or recurrent intermediate states.

Resume must reject incompatible family/version/width identity.

## 15. Verification requirements

The BQRG implementation is not acceptable until tests cover:

1. BQRG is present in `RECURRENCE_GENERATOR_REGISTRY` and `COMPRESSOR_REGISTRY` but absent from `BASIS_REGISTRY`.
2. `DEPTH.compressor=bqrg` resolves successfully at P16.
3. P15/P17 and other invalid persistent widths are rejected.
4. BQRG is rejected outside public DEPTH.
5. Unknown `generator_*` options are rejected.
6. Indexed materialisation matches full-sequence materialisation.
7. Materialisation is deterministic and differentiable with finite representative gradients.
8. Output rescaling changes generated amplitude without altering recurrent dynamics.
9. A tiny CPU `DepthTrajectory` materialises and backpropagates.
10. A tiny complete CPU THOG2 model performs forward and backward with BQRG.
11. BQRG owns no persistent fixed basis buffer and generated weights do not enter checkpoint state.
12. Existing fixed-basis DEPTH tests remain unchanged and pass.
13. Geometry help lists BQRG and its DEPTH-only capability.
14. Representative generated lengths including 32, 64 and 768 are checked for finite forward behavior.

GPU smoke testing must then verify actual memory behavior, training stability and materialisation cost on the normal THOG execution path.

## 16. First experiment

The first scientific comparison should keep the persistent DEPTH width equal:

`Chebyshev P16` versus `BQRG P16`

with the same model geometry, optimizer, batch/tokens, initialization policy intent and training budget.

The first questions are deliberately basic:

- does BQRG train at all;
- are gradients stable;
- what is the materialisation runtime penalty;
- does `torch.compile` materially reduce it;
- does BQRG reach competitive loss for the same persistent DoF;
- do learned trajectories exhibit nontrivial recurrence behavior rather than collapsing to near-constant sequences.

No broader RG capability should be added until this DEPTH-only experiment is understood.
