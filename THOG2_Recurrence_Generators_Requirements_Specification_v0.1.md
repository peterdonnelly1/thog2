# THOG2 Recurrence Generators Requirements Specification v0.1

## 1. Purpose

This enhancement introduces **Recurrence Generators (RGs)** as a new THOG2 compact materialisation family.

The central idea is:

> Persist a small learned parameter vector, and deterministically materialise a much larger ephemeral weight sequence by repeatedly applying a compact differentiable recurrence.

This is intended to test a broader class of compact representations than the existing fixed basis families without introducing a second neural network or persisting derived dense weights.

The first recurrence generator shall be **BQRG — Bounded Quadratic Recurrent Generator**.

BQRG is explicitly the first plugin, not the architecture. The enhancement must be designed so that additional recurrence generators can be added with minimal changes outside the new generator module and registry entry.

The primary research question is:

> Can a small learned nonlinear dynamical law generate useful transformer weight trajectories more efficiently than a fixed basis expansion with the same persistent degree-of-freedom budget?

## 2. Architectural principles

The enhancement shall preserve THOG2's core persistent/ephemeral distinction:

`compact learned state -> deterministic materialisation -> ephemeral operational weights`

For a recurrence generator:

`P learned scalars -> recurrence generator -> N operational weights`

The `P` learned scalars are native persistent model state. The `N` generated weights are derived execution state.

The implementation shall satisfy the following principles:

1. **No dense fallback state.** A recurrence-generated compact family must not persist the materialised weight sequence merely to make training, checkpointing, resume, or analysis convenient.
2. **End-to-end differentiability.** Language-model loss must backpropagate through the complete recurrence materialisation into the compact learned parameters using ordinary autograd.
3. **No auxiliary neural generator.** BQRG itself is a mathematical recurrent materialiser, not a hypernetwork.
4. **Plugin architecture.** BQRG-specific behaviour must not be spread through geometry, checkpoint, CLI, trainer, or model code where a generic RG interface can carry it instead.
5. **Fail explicitly.** Unsupported geometry types, invalid persistent widths, unknown generator options, incompatible versions, and non-finite materialisations must fail with clear errors rather than silently falling back to another compressor.
6. **Preserve existing paths.** Chebyshev, DCT, Haar, lapped cosine, JPEG-like, DENSE, resume/fork, activation checkpointing, torch compilation, and existing geometry selection must retain their current behaviour unless an RG is explicitly selected.

## 3. Terminology

### 3.1 Compressor

A **compressor** is the public geometry-level materialisation family selected by THOG2, for example `chebyshev`, `dct`, `haar`, `lapped_cosine`, `jpeg_like`, or `bqrg`.

The existing `COMPRESSOR_REGISTRY` remains the authoritative geometry-facing catalogue of compressor capabilities.

### 3.2 Fixed basis compressor

A fixed basis compressor materialises from learned coefficients through a predetermined basis, for example:

`w = B c`

where `B` is fixed/reproducible and `c` is learned persistent state.

### 3.3 Recurrence Generator (RG)

A **Recurrence Generator** is a compressor whose learned persistent state parameterises a recurrent dynamical law and whose materialiser iterates that law to produce the operational weight sequence.

In general:

`state_(n+1) = F_theta(state_n)`

`w_n = H_theta(state_n)`

where all learned quantities required by `F`, `H`, and the initial state are contained in the compact persistent parameter vector.

### 3.4 Persistent width

For RGs, the existing geometry `order` / `P` plumbing shall represent the number of learned persistent scalar degrees of freedom per compact trajectory unless a later generator explicitly defines a compatible extension.

For BQRG v1, the persistent width is exactly 16.

The term `order` therefore remains a software/configuration field for compatibility, but for RGs it must not be documented as polynomial, spectral, or frequency order.

## 4. Registry architecture

### 4.1 Same top-level compressor registry

Recurrence generators shall participate in the same geometry-facing compressor selection mechanism as Chebyshev, DCT, Haar, and lapped cosine.

A user shall select BQRG using the ordinary geometry form:

`--option DEPTH.compressor=bqrg`

or the equivalent selected CURVE target.

`COMPRESSOR_REGISTRY` shall expose BQRG capability metadata alongside the existing compressor families.

Geometry resolution must not require special-case checks such as `if compressor == "bqrg"` when a registry capability/property can express the distinction.

### 4.2 Dedicated recurrence-generator registry

A dedicated `RECURRENCE_GENERATOR_REGISTRY` shall own recurrence-generator definitions.

The top-level compressor registry shall consume this registry in the same spirit that it currently consumes registered fixed basis families.

The RG registry shall provide, at minimum, per-generator metadata for:

- canonical family name;
- aliases;
- canonical version;
- artifact/descriptor tag;
- supported geometry element types;
- accepted persistent width or widths;
- generator-specific option schema;
- materialisation implementation;
- human-readable description;
- capability flags needed by runtime materialisers.

Registration must detect collisions in canonical names, aliases, versions, and artifact tags.

Adding a second RG should normally require:

1. one new generator implementation module;
2. one registry definition/registration entry;
3. generator-local tests;
4. no edits to BQRG code;
5. no edits to generic geometry option parsing other than where a genuinely new generic capability is introduced.

### 4.3 Generator protocol

The implementation shall define a generic recurrence-generator protocol or definition object rather than making BQRG itself the interface.

Conceptually the protocol must support operations equivalent to:

- normalize/validate generator version;
- validate persistent width;
- validate declared `generator_*` configuration options;
- materialise a one-dimensional sequence of requested length from compact learned state;
- report machine-readable metadata;
- report help text/capabilities.

The protocol must permit future RGs whose state dimension, recurrence order, nonlinearity, or recurrence family differs from BQRG.

## 5. Geometry capability model

BQRG v1 shall support **CURVE** materialisation only.

That includes:

- the universal `DEPTH` curve;
- any selected one-dimensional registered geometry whose implied type is `CURVE`, provided the generic materialiser path supports that selector.

BQRG v1 shall not flatten a two-dimensional SHEET or SHEET_SET into an arbitrary one-dimensional sequence merely to claim support.

Attempting to select BQRG for an unsupported SHEET or SHEET_SET must fail during geometry resolution/capability validation before training begins.

Future RGs may advertise additional geometry capabilities through the RG/compressor registries without changing BQRG.

## 6. BQRG v1 definition

### 6.1 Identity

Canonical family name:

`bqrg`

Public display name:

`BQRG`

Meaning:

**Bounded Quadratic Recurrent Generator**

Canonical initial version:

`bqrg_v1`

BQRG shall have a distinct artifact/descriptor identity so checkpoints and runs cannot be confused with fixed-basis runs having the same persistent width.

### 6.2 Persistent state

BQRG v1 uses exactly 16 learned scalars per compact trajectory.

Let the compact parameter vector be:

`c = [x0, y0, a0..a5, b0..b5, mu, lambda]`

The fields have the following semantics:

- `x0`, `y0`: initial two-dimensional recurrent state;
- `a0..a5`: coefficients of the quadratic update for the first state coordinate;
- `b0..b5`: coefficients of the quadratic update for the second state coordinate;
- `mu`: output offset;
- `lambda`: learned output-scale parameter.

These 16 scalars are the only learned persistent state required to materialise that BQRG trajectory.

BQRG v1 must reject a requested compact width other than 16 with a clear validation error.

### 6.3 State update

For recurrent state `(x_n, y_n)`, define the quadratic feature vector:

`q_n = [1, x_n, y_n, x_n^2, x_n*y_n, y_n^2]`

Then:

`x_(n+1) = tanh(a . q_n)`

`y_(n+1) = tanh(b . q_n)`

where `a = [a0..a5]` and `b = [b0..b5]`.

The `tanh` is part of the BQRG v1 definition, not an optional implementation detail. It bounds each recurrent state coordinate to `(-1, 1)` after an update while preserving differentiability.

### 6.4 Output mapping

BQRG v1 shall generate one operational scalar from each recurrent state:

`scale = softplus(lambda) + epsilon`

`w_n = mu + scale * x_n`

where `epsilon` is a small fixed positive numerical constant defined by the implementation/version.

The use of a positive softplus scale avoids the exponential blow-up risk of `exp(lambda)` while keeping the output scale differentiable and learned.

The first output is produced from the initial state `(x0, y0)`. Subsequent outputs are produced after successive recurrence updates until the requested materialisation length is reached.

For requested length `N`, materialisation returns exactly:

`[w_0, w_1, ..., w_(N-1)]`

### 6.5 Expressivity motivation

BQRG is motivated by repeated nonlinear composition rather than by polynomial approximation in the sequence index.

Although each state update is only quadratic before the bounded nonlinearity, repeated iteration applies the same learned nonlinear law many times and can represent qualitatively richer deterministic behaviour than evaluating a fixed low-degree polynomial of position.

The expected useful behaviours include, but are not limited to, smooth evolution, oscillatory behaviour, multiple time scales, periodic behaviour, nonlinear transitions, and other structured deterministic trajectories.

This is a hypothesis to test. The requirements do not assume that BQRG will outperform a fixed basis or train stably at useful sequence lengths.

## 7. Differentiability and training contract

BQRG materialisation must be differentiable with respect to all 16 persistent parameters.

The implementation must not:

- detach recurrent state;
- convert learned state to Python scalars in the differentiable path;
- use non-differentiable selection to construct the recurrence;
- mutate persistent state during materialisation;
- hide generated weights inside registered trainable parameters.

A normal transformer loss must propagate gradients through all generated operational weights and through the recurrence back to the compact state.

Autograd tests shall demonstrate finite gradients for every BQRG parameter in representative non-degenerate cases.

## 8. Numerical stability requirements

BQRG deliberately introduces long chains of recurrent differentiation. Bounded forward state does not by itself guarantee bounded or useful gradients.

The implementation shall therefore:

1. keep the recurrent state bounded through the defined `tanh` update;
2. use a numerically stable positive output-scale mapping;
3. detect non-finite materialised weights before they enter downstream computation where practical;
4. integrate with THOG2's existing non-finite update policy rather than inventing a hidden optimizer recovery path;
5. provide tests across representative sequence lengths, including at least 32, 64, and 768 outputs;
6. report/diagnose gradient explosion or collapse during dedicated tests rather than silently clipping inside the BQRG materialiser;
7. define an initialization policy that avoids immediate `tanh` saturation, immediate collapse to a constant trajectory, and gross mismatch with the expected transformer weight scale.

Gradient clipping at the trainer level may continue to operate normally. BQRG must not introduce generator-specific gradient clipping unless a later version explicitly specifies it.

## 9. Initialization requirements

BQRG requires a generator-aware compact-state initializer.

The initializer shall:

- be deterministic under the existing model seed;
- initialize recurrent coefficients in a regime that keeps early trajectories finite and non-saturated;
- avoid exact symmetry that makes one recurrent coordinate permanently irrelevant;
- initialize `mu` and output scale so generated weights have a variance compatible with the existing THOG/nanoGPT initialization intent for that parameter family;
- avoid requiring a pre-existing dense model or fitting a dense target trajectory.

The exact numerical initialization scheme belongs in the implementation plan and tests, but it must be versioned as part of BQRG behaviour where changing it would materially alter reproducibility.

## 10. Structured options

No new short/getopts letters shall be consumed for recurrence generators.

Public RG configuration shall use the existing structured geometry option mechanism.

The naming archetype is:

`TARGET.generator_xxx=VALUE`

For example, a future generator might declare options such as:

`--option DEPTH.generator_state_dimension=4`

`--option DEPTH.generator_burn_in=2`

`--option MLP_UP.MLP_D_MODEL.generator_boundary_policy=...`

These examples define the namespace convention, not required BQRG v1 controls.

### 10.1 Option parser extension

The geometry option parser shall accept registry-declared properties beginning with `generator_`.

Validation rules:

- `generator_*` options are legal only when the selected compressor is a registered recurrence generator;
- the selected RG definition must declare the specific option name and validation rule;
- an unknown `generator_*` option must fail with a message listing valid options for that generator;
- fixed-basis compressors must reject `generator_*` options rather than ignore them;
- RG-specific option parsing/normalization must live in the generic RG registry/protocol plus generator definition, not in a growing central `if/elif` chain.

### 10.2 BQRG v1 options

BQRG v1 should initially expose **no required tunable generator options** beyond ordinary compressor identity/version and the existing compact-width/order setting.

Its two-dimensional state, quadratic feature set, `tanh` state clamp, softplus output scale, and parameter interpretation are part of `bqrg_v1`.

If experimentation later requires configurable behaviour, new settings must use the `generator_*` namespace and must be represented in run identity/checkpoint metadata.

## 11. Geometry help and discoverability

`--print-geometry-registry` shall gain a dedicated **recurrence generators** subsection.

For each registered RG, the help output shall show at minimum:

- canonical name;
- aliases, if any;
- version;
- display/artifact tag;
- supported geometry types;
- accepted persistent widths;
- declared `generator_*` options and defaults, if any;
- short description.

The existing compressor section shall also show that BQRG is a registered compressor and distinguish its implementation kind from fixed-basis compressors.

The geometry help must make the plugin boundary visible enough that a user can answer:

- which RGs exist;
- where they may be used;
- how many persistent values they require;
- which structured options they accept.

`--explain-geometry` for a selected BQRG geometry shall report:

- compressor `bqrg`;
- compressor/generator version;
- persistent width 16;
- resolved generator options;
- geometry type and generated axis length;
- materialiser identity.

## 12. Run naming, metadata, and checkpoint identity

BQRG runs must be unambiguously identifiable from descriptors and checkpoint metadata.

At minimum, canonical metadata shall record:

- compressor family `bqrg`;
- generator kind `recurrence` or equivalent registry type;
- generator version `bqrg_v1`;
- persistent width;
- resolved `generator_*` options;
- materialisation/version identity required for deterministic reconstruction.

The run descriptor shall include a concise BQRG identity consistent with existing basis/compressor naming conventions.

Future RGs must receive distinct canonical identities rather than being encoded as parameter variations of `bqrg` unless they are genuinely compatible versions of the same generator definition.

## 13. Checkpoint and resume contract

The existing THOG2 invariant remains authoritative:

> Checkpoint the model's native persistent representation, whatever form each parameter family uses. Never checkpoint derived materialisations.

For BQRG this means:

- save the 16 learned compact scalars per BQRG trajectory;
- save generator/compressor identity and configuration needed to reconstruct materialisation;
- do not save the generated weight sequence as model state;
- do not save recurrent intermediate states that are deterministically regenerated from the compact state during materialisation;
- do not save fixed constants or deterministic generator code artifacts.

Resume must reconstruct the same BQRG materialisation from checkpointed persistent state and metadata.

A normal resume must reject an incompatible generator family, version, persistent width, or material configuration rather than silently reinterpret compact state.

Existing checkpoints from pre-RG branches must remain loadable on their existing compressor paths.

## 14. Materialisation and execution integration

The RG subsystem shall integrate at the compressor/materialiser boundary rather than inside transformer layer semantics.

The generic caller should provide:

- compact persistent tensor/state;
- requested generated length/shape for the compressed CURVE axis;
- resolved RG definition/version/options;
- runtime dtype/device context.

The RG returns the materialised operational trajectory in the shape expected by the existing geometry/materialisation pipeline.

BQRG shall not know whether a trajectory belongs to Q, K, V, attention output, MLP expansion, MLP contraction, DEPTH, or another registered one-dimensional axis. Those semantics belong to the geometry layer.

The implementation should permit efficient batched materialisation of many independent BQRG trajectories sharing the same recurrence definition. A Python loop per individual scalar trajectory is not an acceptable final GPU design.

A loop over recurrence steps may be acceptable initially, because recurrence is sequential in the generated axis, but all independent trajectories at a step must be vectorized where practical.

## 15. Torch compilation and activation checkpointing

BQRG must coexist with the current torch compilation and activation-checkpointing paths.

Requirements:

- eager execution is the reference implementation;
- regional torch compilation must not change BQRG numerics beyond normal backend tolerance;
- whole-model compilation, where supported by existing THOG2, must either work or fail explicitly with a documented capability boundary;
- recurrence materialisation must not introduce hidden graph breaks on the supported compiled path without a measured reason;
- activation recomputation must regenerate deterministic BQRG weights from the same compact state.

Performance optimization shall follow correctness. The first implementation must not replace a clear differentiable recurrence with opaque fused code before reference equivalence exists.

## 16. Verification requirements

The enhancement is acceptable only when all of the following are covered.

### 16.1 Registry/plugin tests

1. BQRG appears in `RECURRENCE_GENERATOR_REGISTRY`.
2. BQRG appears in the geometry-facing `COMPRESSOR_REGISTRY`.
3. Duplicate names/aliases/versions/tags are rejected.
4. A minimal test-only second RG can be registered without editing BQRG logic, demonstrating the plugin contract.
5. Registry metadata is deterministic and serializable.

### 16.2 Geometry/CLI tests

1. `--option DEPTH.compressor=bqrg` resolves as a CURVE compressor.
2. BQRG on an unsupported SHEET/SHEET_SET is rejected before training.
3. BQRG rejects persistent width other than 16.
4. Unknown `generator_*` options fail clearly.
5. Fixed-basis compressors reject `generator_*` options.
6. `--print-geometry-registry` includes the recurrence-generator subsection and BQRG metadata.
7. `--explain-geometry` reports resolved BQRG identity and width.

### 16.3 Mathematical/materialisation tests

1. A hand-computed short BQRG sequence matches implementation output.
2. State values after update remain in `(-1, 1)` for finite input parameters.
3. Requested output length is exact for lengths including 1, 2, 32, 64, and 768.
4. Batched materialisation matches independent reference materialisation.
5. Materialisation is deterministic for identical compact state/configuration.
6. Materialised output remains finite for defined representative initialization/test ranges.

### 16.4 Gradient tests

1. `torch.autograd.grad` reaches all 16 compact parameters in a non-degenerate test.
2. Analytical autograd gradients agree with finite-difference checks on a short sequence within suitable tolerance.
3. Backpropagation through representative lengths does not produce non-finite gradients under the canonical initializer.
4. Materialisation does not create trainable dense parameters.

### 16.5 Checkpoint tests

1. BQRG compact state round-trips through save/load exactly within dtype semantics.
2. Generated operational weights are absent from compact checkpoint state.
3. Materialising before checkpoint save does not add generated weights or recurrent intermediate states to saved state.
4. Loaded BQRG state reproduces the pre-save materialisation.
5. Existing fixed-basis and DENSE checkpoint invariants remain unchanged.

### 16.6 Training tests

1. A tiny CPU model completes forward, backward, optimizer step, checkpoint, reload, and another step using BQRG.
2. A tiny BQRG run can reduce loss on a deterministic toy task, proving trainability rather than only differentiability.
3. Existing basis/compressor CPU regression tests continue to pass.
4. At least one real GPU smoke run completes using BQRG before broader benchmarking.

## 17. Benchmark requirements

The first meaningful BQRG benchmark shall compare it with an existing fixed-basis compressor at the same persistent scalar budget wherever the geometry permits a fair comparison.

The primary initial comparison is intended to be:

`BQRG P16` versus `Chebyshev P16`

on otherwise identical model/training configuration.

Record at least:

- persistent parameter count;
- materialised operational weight count;
- training loss trajectory;
- validation loss;
- training throughput;
- materialisation cost;
- peak memory where practical;
- gradient/non-finite incidents;
- checkpoint size.

The experiment must not claim BQRG superiority merely because it can generate complicated sequences. Utility is determined by trained model quality, stability, and cost.

## 18. Explicit non-goals for the first stage

The first BQRG stage shall not attempt to implement:

- neural/hypernetwork generators;
- automatic search over recurrence equations;
- symbolic regression/genetic programming;
- learned switching among multiple RG families;
- SHEET/SHEET_SET flattening;
- stochastic/random sequence generators;
- generator-specific optimizer algorithms;
- custom CUDA kernels before the reference path is validated;
- higher-dimensional or higher-degree BQRG variants merely to improve the first result.

These may become later experiments, but the first stage must establish a clean generic RG architecture and one credible generator.

## 19. Extension criteria for future RGs

A future recurrence generator should be addable when it provides a materially different compact dynamical family, for example a different state transition class, bounded rational recurrence, controlled oscillatory recurrence, state-space recurrence, or other deterministic iterative law.

Each new RG must define its own:

- family/version identity;
- persistent-width semantics;
- recurrence/materialisation rule;
- stability constraints;
- initializer;
- supported geometry types;
- `generator_*` option schema;
- tests.

The generic RG and geometry infrastructure must not assume that future generators use:

- two state variables;
- quadratic updates;
- `tanh`;
- 16 persistent scalars;
- one particular output transform.

Those are BQRG properties only.

## 20. Success criterion for this enhancement

This enhancement succeeds architecturally if THOG2 can treat BQRG as an ordinary selectable compressor while keeping recurrence-generator internals modular and making a second RG straightforward to add.

It succeeds scientifically only if BQRG can be trained end-to-end and produces useful weight trajectories at a competitive compact-state budget.

The important outcome is not that BQRG must win. It is that THOG2 gains a clean experimental framework for asking:

> How much useful model structure can be generated by repeatedly applying a tiny learned dynamical law instead of expanding learned coefficients in a predetermined basis?
