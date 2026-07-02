THOG2

Chebyshev Sheet Staging Plan

Version 0.1

|                              |                                                                            |
|------------------------------|----------------------------------------------------------------------------|
| **System**                   | thog2 enhancement of nanoGPT                                               |
| **Architecture designation** | Chebyshev Sheet                                                            |
| **Governing specification**  | THOG2 Chebyshev Sheet Requirements Specification v0.1                      |
| **Implementation plan**      | THOG2 Chebyshev Sheet Implementation Plan v0.1                             |
| **Status**                   | Draft staging plan; repository implementation not commenced                |
| **Date**                     | 2 July 2026                                                                |
| **Planned repository**       | thog2                                                                      |
| **Repository interaction**   | None; this document is authored outside GitHub pending repository creation |

Non-normative note: This plan is deliberately external to the existing
thog repository. It is intended to be added to thog2 after that
repository is created and access is provided. No stage in this document
authorises changes to the existing Vermeer work.

# 1. Purpose and authority

**1.1 **This document divides implementation of the Chebyshev Sheet
architecture into the minimum sensible sequence of independently
reviewable and testable stages.

**1.2 **The governing functional requirements are contained in THOG2
Chebyshev Sheet Requirements Specification v0.1.

**1.3 **The planned technical design is contained in THOG2 Chebyshev
Sheet Implementation Plan v0.1.

**1.4 **Where this staging plan conflicts with the governing
requirements specification, the requirements specification shall
prevail.

**1.5 **Where this staging plan conflicts with the implementation plan
on sequencing only, this staging plan shall prevail until a versioned
amendment resolves the conflict.

**1.6 **This document defines stage scope, prerequisites, required
evidence, exit gates, stop conditions, branch boundaries, and merge
boundaries.

**1.7 **This document does not define the complete test inventory or
numerical tolerances; those shall be governed by the later Chebyshev
Sheet Test Plan.

**1.8 **A stage shall not be treated as complete merely because its code
imports, a forward pass starts, or a short training command does not
crash.

# 2. Staging objectives

**2.1 **Each stage shall leave the repository executable, testable, and
suitable for review.

**2.2 **Each stage shall establish one coherent capability and shall
avoid including speculative work from later stages.

**2.3 **Mathematical correctness shall be established before model
integration.

**2.4 **CPU reference correctness shall be established before activation
checkpointing, GPU performance work, or DDP work.

**2.5 **The dense nanoGPT path shall remain operational after every
merged stage.

**2.6 **The sheet architecture shall remain non-default until Stage 6
acceptance is complete.

**2.7 **Scientific quality claims shall be deferred until Stage 6 and
shall not be used to waive an earlier correctness gate.

**2.8 **No stage shall modify or depend at runtime on the existing thog
repository.

**2.9 **No stage shall disturb completion or merge of the separate
Vermeer enhancement.

# 3. Branch, commit, and pull-request strategy

**3.1 **Stage 0 shall begin only after the thog2 repository has been
created as a clean nanoGPT clone and repository access has been
provided.

**3.2 **Each stage shall use a fresh branch created from the current
accepted default branch after the preceding stage has been merged.

**3.3 **Stages shall execute sequentially. Parallel implementation
branches shall not be used for dependent stages.

**3.4 **The provisional branch naming convention shall be shown below.

> docs/sheet-stage-0-baseline feature/sheet-stage-1-core
> feature/sheet-stage-2-reference-model
> feature/sheet-stage-3-training-lifecycle
> feature/sheet-stage-4-ephemeral-inference
> feature/sheet-stage-5-gpu-ddp experiment/sheet-stage-6-pilot

**3.5 **Final branch names may be shortened after repository inspection,
but stage numbering and scope shall remain unambiguous.

**3.6 **Each stage shall normally produce one pull request to the
default branch.

**3.7 **Working commits may be used within a branch, but the pull
request history shall remain intelligible and shall not contain
unrelated experiments or generated artifacts.

**3.8 **A stage pull request shall identify its governing stage, list
changed files, state exclusions, report tests, record unresolved risks,
and state whether all exit gates are satisfied.

**3.9 **The next stage branch shall not be created from an unmerged
predecessor unless an explicit recovery decision is recorded.

**3.10 **After merge, the stage branch should be deleted and the
accepted merge commit shall become the baseline for the next stage.

**3.11 **All modifications to original nanoGPT source shall retain
superseded lines as commented source where required and shall use the
established THOG source markers.

# 4. Stage summary

|           |                                   |                                                                                                        |                                                                                         |
|-----------|-----------------------------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| **Stage** | **Name**                          | **Primary scope**                                                                                      | **Principal exit gate**                                                                 |
| 0         | Baseline and documentation        | Capture pristine nanoGPT baseline, environment, dense controls, and add governing documents.           | Clean reproducible baseline and documentation-only merge.                               |
| 1         | Mathematical core                 | Coordinates, bivariate Chebyshev bases, QR stabilization, geometry, Q scaling, counts.                 | All mathematical tests pass through row order 1024.                                     |
| 2         | Compact state and reference model | Coefficient state, materialisation, initialization, optimizer groups, CPU SheetGPT.                    | Reference forward/backward and short CPU training pass without dense block persistence. |
| 3         | Training lifecycle                | Shared trainer, architecture selection, checkpoints, resume, deterministic controls, single-GPU smoke. | Dense and sheet save/resume and exact-update smokes pass.                               |
| 4         | Ephemeral execution and inference | Segmented checkpointing, memory telemetry, compact inference, diagnostics, naming.                     | Reference/checkpointed equivalence and compact-checkpoint text generation pass.         |
| 5         | GPU, DDP, and target geometry     | DDP, bounded-memory execution, target L144/D768/P16/Q128 geometry, profiling.                          | No deadlock or state divergence; target geometry completes accepted bounded run.        |
| 6         | Controlled pilot and as-built     | Matched dense-versus-sheet pilot, row-capacity sweep decision, final regression, as-built.             | Evidence package complete; scientific conclusion stated with limits.                    |

**4.1 **The seven stages above are the minimum sensible grouping.
Combining additional stages would mix distinct failure domains;
splitting them further would create merge boundaries without
independently useful capability.

**4.2 **Internal checkpoints may be used within a stage, but they shall
not be represented as accepted stages unless this document is amended.

# 5. Stage 0 - Baseline and documentation

## 5.1 Objective

**5.1.1 **Stage 0 shall establish a trustworthy thog2 baseline before
any Chebyshev Sheet runtime code is added.

## 5.2 Entry criteria

**5.2.1 **The thog2 GitHub repository exists and is accessible.

**5.2.2 **The repository is confirmed to be a fresh clone of the
selected nanoGPT baseline.

**5.2.3 **The selected upstream commit and default branch are
identified.

## 5.3 Scope

**5.3.1 **Record the upstream repository URL and exact baseline commit
SHA.

**5.3.2 **Record the Python, PyTorch, CUDA, driver, operating-system,
and GPU environment used for baseline evidence.

**5.3.3 **Run and record a CPU import or minimal-forward smoke for the
dense model.

**5.3.4 **Run and record a short single-GPU dense training smoke where
hardware is available.

**5.3.5 **Add the requirements specification, implementation plan,
staging plan, and later test plan as documentation-only changes.

**5.3.6 **Add or verify ignore rules for datasets, checkpoints, logs,
generated text, W&B state, and local environment files.

**5.3.7 **Do not add sheet runtime code in Stage 0.

## 5.4 Required evidence

**5.4.1 **A baseline environment record shall be committed or attached
to the pull request.

**5.4.2 **Dense smoke commands and their outcomes shall be recorded
exactly.

**5.4.3 **Git status shall be clean after the accepted baseline
commands, excluding documented ignored artifacts.

## 5.5 Exit gate

**5.5.1 **The default branch reproduces the selected nanoGPT baseline
behavior.

**5.5.2 **The governing documents are available in thog2 without
modifying runtime behavior.

**5.5.3 **The Stage 0 pull request is merged before Stage 1 begins.

## 5.6 Stop conditions

**5.6.1 **Stop if the repository is not a clean or identifiable nanoGPT
baseline.

**5.6.2 **Stop if the dense baseline cannot complete its agreed smoke
tests.

**5.6.3 **Stop if unreviewed code from thog or Vermeer has already been
mixed into the repository.

# 6. Stage 1 - Mathematical core

## 6.1 Objective

**6.1.1 **Stage 1 shall establish the universal, model-independent
mathematics of the depth-row Chebyshev sheet.

## 6.2 Entry criteria

**6.2.1 **Stage 0 is merged and the default branch is clean.

**6.2.2 **The later Test Plan defines or confirms basis tolerances
before Stage 1 acceptance.

## 6.3 Scope

**6.3.1 **Implement normalized depth coordinates and fixed left-to-right
within-row coordinates.

**6.3.2 **Implement first-kind Chebyshev recurrence without explicit
monomial powers.

**6.3.3 **Implement deterministic reduced-QR stabilization of sampled
basis columns.

**6.3.4 **Implement deterministic QR column-sign normalization.

**6.3.5 **Implement row-order derivation from model width and Q_d.

**6.3.6 **Implement family geometry for attention input, attention
output, MLP expansion, MLP contraction, and selected repeated vector
families.

**6.3.7 **Implement analytical persistent-parameter and dense-equivalent
count helpers.

**6.3.8 **Implement basis caching by geometry, basis version, device,
and runtime dtype where useful.

**6.3.9 **Do not implement SheetGPT, training integration, activation
checkpointing, or inference in Stage 1.

## 6.4 Required tests and evidence

**6.4.1 **Test coordinate endpoints, monotonicity, and expected sample
count.

**6.4.2 **Test recurrence values against a trusted direct reference on
small orders.

**6.4.3 **Test basis shape, finiteness, rank, orthonormality,
deterministic reconstruction, and deterministic signs.

**6.4.4 **Test Q scaling for widths d and 4d, including the principal
values 128 and 512 at d equal to 768.

**6.4.5 **Test row orders through at least Q equal to 1024 for numerical
validity and practical construction cost.

**6.4.6 **Cross-check analytical parameter counts against explicit
tensor-shape products.

## 6.5 Exit gate

**6.5.1 **Every mathematical test passes on CPU.

**6.5.2 **Repeated construction of an identical basis is bitwise
identical where the chosen dtype and platform permit, or agrees within a
documented deterministic tolerance otherwise.

**6.5.3 **Basis construction through Q equal to 1024 is finite and
operationally acceptable.

**6.5.4 **No model or training code depends on incomplete Stage 1
behavior.

## 6.6 Stop conditions

**6.6.1 **Stop if the basis loses rank or produces non-finite values at
required orders.

**6.6.2 **Stop if deterministic sign normalization cannot make
checkpoint reconstruction stable.

**6.6.3 **Stop if the Q equal to 512 or Q equal to 1024 basis is too
costly to construct or store for the intended runtime, pending an
implementation-plan amendment.

**6.6.4 **Stop if parameter-count formulas and instantiated shapes
disagree.

# 7. Stage 2 - Compact state and reference model

## 7.1 Objective

**7.1.1 **Stage 2 shall produce the first complete correctness-first
Chebyshev Sheet language model on CPU.

## 7.2 Entry criteria

**7.2.1 **Stage 1 is merged and all mathematical tests pass on the
accepted default branch.

## 7.3 Scope

**7.3.1 **Implement one compact coefficient tensor per generated tensor
family with shape R_f by P by Q_f.

**7.3.2 **Implement family metadata, semantic decay classification,
basis ownership, and state-schema rules.

**7.3.3 **Implement one-layer materialisation by depth-axis mixing
followed by multiplication with the transposed row basis.

**7.3.4 **Implement initialization that produces the intended
nanoGPT-like generated-weight statistics without allocating a dense
logical block stack.

**7.3.5 **Implement semantic optimizer grouping for matrix, LayerNorm,
and bias families used by the reference model.

**7.3.6 **Implement the functional sheet transformer block and SheetGPT
reference model.

**7.3.7 **Execute logical layers sequentially without activation
checkpointing.

**7.3.8 **Retain the dense GPT implementation and its existing behavior.

**7.3.9 **Use ordinary next-token cross-entropy as the sole model loss.

**7.3.10 **Do not integrate checkpoint save/resume, DDP, compact
inference, or target-scale GPU execution in Stage 2.

## 7.4 Required tests and evidence

**7.4.1 **Compare materialisation against direct bivariate evaluation on
small exact geometries.

**7.4.2 **Use saturated small geometries with P equal to L and Q_f equal
to C_f to demonstrate sampled-sheet completeness within tolerance.

**7.4.3 **Verify that generated matrices have conventional nanoGPT
shapes.

**7.4.4 **Verify finite non-zero gradients to every active coefficient
family.

**7.4.5 **Verify initialization mean, standard deviation, RMS, endpoint
behavior, shared-depth starting behavior, exact LayerNorm ones, and
exact bias zeros where applicable.

**7.4.6 **Verify that every trainable parameter appears in exactly one
optimizer group.

**7.4.7 **Verify that no persistent dense logical block stack appears in
named parameters or state_dict.

**7.4.8 **Run a tiny CPU forward/backward test and a short CPU
optimization test showing finite loss and at least one completed
parameter update.

**7.4.9 **Run the dense nanoGPT regression tests affected by shared
configuration or model-factory changes.

## 7.5 Exit gate

**7.5.1 **A tiny SheetGPT completes forward, backward, optimizer step,
and repeated training updates on CPU.

**7.5.2 **All compact-state, materialisation, initialization, gradient,
optimizer, state-schema, and dense-regression tests pass.

**7.5.3 **Persistent and dense-equivalent parameter reporting agrees
with analytical expectations.

## 7.6 Stop conditions

**7.6.1 **Stop if the saturated-basis reconstruction test fails
materially.

**7.6.2 **Stop if gradients fail to reach any required coefficient
family.

**7.6.3 **Stop if initialization produces pathological outliers or
unstable loss before the initialization policy is corrected and
documented.

**7.6.4 **Stop if the implementation requires persistent dense layer
weights to obtain a functional forward pass.

**7.6.5 **Stop if the dense model path changes numerically outside
agreed regression tolerance.

# 8. Stage 3 - Training lifecycle

## 8.1 Objective

**8.1.1 **Stage 3 shall make the reference sheet model trainable,
checkpointable, resumable, and comparable through the shared nanoGPT
training lifecycle.

## 8.2 Entry criteria

**8.2.1 **Stage 2 is merged and the CPU reference model is accepted.

## 8.3 Scope

**8.3.1 **Add explicit dense and thog2_sheet model selection to the
shared trainer.

**8.3.2 **Preserve identical data loading, gradient accumulation, mixed
precision, scheduler, clipping, evaluation, and logging semantics where
architecture does not require a difference.

**8.3.3 **Implement exact completed-optimizer-update semantics.

**8.3.4 **Separate data-sampling RNG from model-construction RNG for
controlled comparisons.

**8.3.5 **Implement sheet checkpoint save, load, resume, compatibility
validation, and deterministic optimizer-group reconstruction.

**8.3.6 **Record model type, sheet geometry, basis version, Q scaling
rule, coefficient state, optimizer state, update count, and training
arguments.

**8.3.7 **Implement startup reporting for persistent parameters, sheet
coefficients, conventional parameters, and dense-equivalent parameters.

**8.3.8 **Add a small single-GPU reference-training path without
activation checkpointing.

**8.3.9 **Do not add DDP or target-scale L144/D768 execution in Stage 3.

## 8.4 Required tests and evidence

**8.4.1 **Run dense and sheet trainer smokes using the same shared
training loop.

**8.4.2 **Verify checkpoint round-trip equality for model state and
fixed-batch outputs.

**8.4.3 **Verify resume restores optimizer state, completed update
count, learning-rate schedule position, and deterministic group
ordering.

**8.4.4 **Verify incompatible model geometry or basis version fails with
a direct diagnostic rather than silent reinterpretation.

**8.4.5 **Verify two matched runs consume the same batch-index trace
when configured with the same data seed.

**8.4.6 **Run a short single-GPU sheet training smoke with finite
forward, backward, optimizer state allocation, validation, checkpoint,
and resume.

**8.4.7 **Run dense-path regression tests after trainer modifications.

## 8.5 Exit gate

**8.5.1 **Dense and sheet models both train through the shared trainer.

**8.5.2 **A compact sheet checkpoint saves, reloads, resumes, and
reproduces the accepted fixed-batch output within tolerance.

**8.5.3 **The single-GPU reference smoke completes without activation
checkpointing or DDP.

## 8.6 Stop conditions

**8.6.1 **Stop if checkpoint state includes a persistent dense logical
block stack.

**8.6.2 **Stop if resume changes optimizer-group membership or
learning-rate schedule position unexpectedly.

**8.6.3 **Stop if architecture selection changes dense training behavior
outside agreed tolerance.

**8.6.4 **Stop if matched data traces cannot be made independent of
model-construction RNG consumption.

# 9. Stage 4 - Ephemeral execution and inference

## 9.1 Objective

**9.1.1 **Stage 4 shall establish memory-controlled training execution
and direct inference from compact checkpoints.

## 9.2 Entry criteria

**9.2.1 **Stage 3 is merged and reference checkpoint save/resume is
accepted.

## 9.3 Scope

**9.3.1 **Implement segmented indexed activation checkpointing for the
logical-layer stack.

**9.3.2 **Use non-reentrant checkpointing and preserve RNG state unless
a documented baseline constraint requires another approach.

**9.3.3 **Regenerate required generated weights during backward
recomputation rather than retaining a dense layer stack.

**9.3.4 **Make checkpoint segment size configurable, including a
disabled reference mode.

**9.3.5 **Implement compact-checkpoint inference and checkpoint-driven
model reconstruction.

**9.3.6 **Implement stable architecture-first run naming containing L,
d, P, and Q_d.

**9.3.7 **Implement detached console and W&B diagnostics for coefficient
RMS, gradients, order utilization, generated-weight statistics, and
memory phases.

**9.3.8 **Bypass training-only activation checkpointing during
validation and inference.

**9.3.9 **Do not add DDP in Stage 4.

## 9.4 Required tests and evidence

**9.4.1 **Compare reference and checkpointed logits, losses, coefficient
gradients, and completed optimizer updates under deterministic
conditions.

**9.4.2 **Verify dropout agreement under preserved RNG state.

**9.4.3 **Verify memory phase telemetry for model construction,
optimizer allocation, first backward allocation, and steady-state
training.

**9.4.4 **Verify evaluation and inference execute without checkpoint
recomputation.

**9.4.5 **Verify a compact sheet checkpoint generates text directly
without dense conversion.

**9.4.6 **Verify run names and artifact prefixes distinguish dense and
sheet architectures and distinct P/Q configurations.

**9.4.7 **Verify diagnostics are finite, detached, and do not perturb
gradients or training RNG streams.

## 9.5 Exit gate

**9.5.1 **Reference and checkpointed execution agree within the Test
Plan tolerance.

**9.5.2 **Memory telemetry demonstrates that no persistent full logical
block stack is retained.

**9.5.3 **A compact checkpoint generates text using the normal inference
entry point.

**9.5.4 **All Stage 4 regressions pass in both reference and
checkpointed modes.

## 9.6 Stop conditions

**9.6.1 **Stop if checkpointed execution changes gradients or optimizer
updates outside tolerance.

**9.6.2 **Stop if RNG preservation cannot reproduce reference dropout
behavior.

**9.6.3 **Stop if memory grows approximately as a persisted L-layer
dense weight stack.

**9.6.4 **Stop if inference requires materialising and retaining the
complete dense model.

# 10. Stage 5 - GPU, DDP, and target geometry

## 10.1 Objective

**10.1.1 **Stage 5 shall demonstrate operational training at
representative and principal geometries, including distributed
execution.

## 10.2 Entry criteria

**10.2.1 **Stage 4 is merged and reference/checkpointed equivalence is
accepted.

**10.2.2 **The target hardware and DDP launch environment are
identified.

## 10.3 Scope

**10.3.1 **Implement DDP integration without changing compact model
semantics.

**10.3.2 **Verify deterministic parameter registration and
optimizer-group construction on all ranks.

**10.3.3 **Add per-rank memory and failure diagnostics sufficient to
identify imbalance or deadlock.

**10.3.4 **Run progressive GPU geometries before the principal target
geometry.

**10.3.5 **Exercise the principal L144/D768/P16/Q128 geometry, including
Q equal to 512 for 4d-wide rows.

**10.3.6 **Use the agreed head count and context length, initially H12
and context 256 unless the governing documents are amended.

**10.3.7 **Profile basis construction, layer materialisation, forward,
backward, optimizer, communication, and checkpoint recomputation costs.

**10.3.8 **Optimize only measured bottlenecks and retain the tested
reference path.

## 10.4 Required tests and evidence

**10.4.1 **Run a single-GPU reference smoke and a single-GPU
checkpointed smoke on a representative medium geometry.

**10.4.2 **Run a multi-process DDP smoke and verify no deadlock,
unused-parameter failure, rank divergence, or state-schema mismatch.

**10.4.3 **Compare single-GPU and DDP completed-update behavior on a
deterministic small fixture where practical.

**10.4.4 **Run the principal target geometry for a documented bounded
update count sufficient to allocate optimizer state and reach
steady-state memory.

**10.4.5 **Record peak allocated and reserved VRAM by rank and by
lifecycle phase.

**10.4.6 **Record update time, throughput, materialisation cost, and
checkpoint-recomputation cost.

**10.4.7 **Verify finite loss and gradients and at least a short-term
downward training-loss trend at the principal geometry.

## 10.5 Exit gate

**10.5.1 **Single-GPU and DDP smokes pass.

**10.5.2 **The principal geometry completes its accepted bounded run
without out-of-memory failure, deadlock, non-finite state, or persistent
dense-weight retention.

**10.5.3 **Performance and memory evidence is sufficient to design the
controlled pilot.

**10.5.4 **The complete regression suite passes on the proposed Stage 5
merge commit.

## 10.6 Stop conditions

**10.6.1 **Stop if any rank constructs a different parameter set or
optimizer grouping.

**10.6.2 **Stop if DDP produces reproducible state divergence or
deadlock attributable to the sheet architecture.

**10.6.3 **Stop if principal-geometry memory is unbounded or defeats the
compact-state objective.

**10.6.4 **Stop if basis or materialisation cost dominates so severely
that a measured implementation revision is required before scientific
training.

**10.6.5 **Stop before Stage 6 if the model cannot complete finite
updates or show any learning signal at representative settings.

# 11. Stage 6 - Controlled pilot and as-built

## 11.1 Objective

**11.1.1 **Stage 6 shall conduct the first controlled scientific
evaluation and close the initial implementation with an as-built record.

## 11.2 Entry criteria

**11.2.1 **Stage 5 is merged and all functional acceptance gates through
target geometry are satisfied.

**11.2.2 **The fixed training-batch trace or equivalent deterministic
sampler, fixed validation sample, and formal run budget are agreed.

## 11.3 Scope

**11.3.1 **Run matched dense and principal sheet configurations using
equal logical geometry, dataset, context length, batch trace, validation
sample, optimizer schedule, completed updates, and consumed tokens
except for architecture-required differences.

**11.3.2 **Use P equal to 16 and principal Q_d equal to 128 with Q_4d
equal to 512 for the main sheet run.

**11.3.3 **If the principal sheet run is stable but materially
capacity-limited, run the planned Q_d bracket of 64, 128, and 256 with
proportional 4d orders before rejecting the rowwise-sheet hypothesis.

**11.3.4 **Compare validation loss against completed updates, consumed
tokens, and wall-clock time.

**11.3.5 **Report throughput, persistent parameters, dense-equivalent
parameters, peak VRAM, coefficient-order utilization, and
generated-weight diagnostics.

**11.3.6 **Run the final complete regression suite on the accepted
implementation commit.

**11.3.7 **Write an as-built document identifying actual paths,
configuration semantics, deviations, known limitations, accepted tests,
and unresolved scientific questions.

**11.3.8 **Update governing documents through versioned amendments where
actual architecture differs materially from v0.1.

## 11.4 Decision rules

**11.4.1 **A low-Q failure accompanied by clear improvement as Q
increases shall be treated as capacity evidence rather than immediate
falsification.

**11.4.2 **A model requiring Q_d close to full row width for useful
learning shall be treated as evidence against useful rowwise
compression.

**11.4.3 **A stable model with a modest quality penalty but major
parameter or VRAM reduction shall be recorded as promising rather than
declared superior.

**11.4.4 **A result shall not be called positive if it depends on an
uncontrolled data trace, validation sample, token budget, or optimizer
difference.

**11.4.5 **Failure of the first pilot shall not justify unplanned
learned coordinates, row permutations, Vermeer, auxiliary losses, or 3D
volumes inside the same stage.

## 11.5 Exit gate

**11.5.1 **All controlled runs and their exact configurations are
recorded.

**11.5.2 **The final regression suite passes.

**11.5.3 **The as-built document and any required specification
amendments are complete.

**11.5.4 **The project records one of the following explicit outcomes:
viable for further study, viable only at weak compression, inconclusive,
or not viable under the tested design.

## 11.6 Stop conditions

**11.6.1 **Stop the pilot if non-finite loss, recurrent checkpoint
corruption, reproducible DDP divergence, or invalid comparison controls
occur.

**11.6.2 **Stop expansion of the sweep if the principal model is
operationally unstable and the failure is not attributable to row
capacity.

**11.6.3 **Stop architectural expansion after the agreed Q sweep and
produce a written conclusion before introducing a successor design.

# 12. Cross-stage test and evidence rules

**12.1 **Every stage shall add the tests required to prove its own new
capability.

**12.2 **Every stage shall run all earlier-stage tests unless the Test
Plan defines a justified narrower pre-merge subset followed by full CI.

**12.3 **Tests shall distinguish mathematical unit tests, CPU model
tests, single-GPU tests, DDP tests, source guards, and scientific runs.

**12.4 **A scientific run shall not substitute for a deterministic
correctness test.

**12.5 **Test outputs shall identify the exact commit, branch, command,
environment, model geometry, seed, and result.

**12.6 **Large logs and checkpoints shall remain outside Git unless
selected as small fixtures.

**12.7 **Each stage pull request shall include an evidence table of the
following form.

|                               |                                                                    |                                                    |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------|
| **Evidence class**            | **Minimum record**                                                 | **Acceptance decision**                            |
| Unit and integration tests    | Command, pass count, skipped count, failures, runtime.             | All mandatory tests pass.                          |
| GPU or DDP smoke              | Hardware, launch command, geometry, updates, peak memory, outcome. | Completes without architecture-attributable fault. |
| State and checkpoint evidence | State keys/counts, round-trip or resume comparison.                | Compact state is complete and reproducible.        |
| Regression evidence           | Dense commands and comparison tolerance.                           | Dense behavior remains accepted.                   |
| Scientific evidence           | Controlled configuration and measured metrics.                     | Used only in Stage 6 decisions.                    |

# 13. Change control and recovery

**13.1 **A failed stage gate shall be corrected on the same stage branch
before merge unless the stage is abandoned.

**13.2 **A merged regression discovered later shall be fixed before
dependent stage work continues.

**13.3 **If the default branch becomes non-functional, the preferred
recovery shall be a corrective pull request or revert to the last
accepted stage commit, not unrecorded local repair.

**13.4 **Any change to the fixed horizontal coordinate, independent
row-sheet design, sole cross-entropy objective, or bivariate Chebyshev
architecture shall require a requirements amendment.

**13.5 **Any change to stage boundaries, stage ordering, or merge
strategy shall require a staging-plan amendment.

**13.6 **A mathematically equivalent basis-stabilization change may be
handled through an implementation-plan amendment if the represented span
and checkpoint semantics remain unchanged.

**13.7 **Custom kernels, fused materialisation, or compilation-specific
paths shall not be introduced before Stage 5 profiling demonstrates a
material need.

**13.8 **The tested reference implementation shall remain available
until any optimized path has equivalent coverage.

# 14. Stage completion reports

**14.1 **Each stage completion report shall state the accepted commit
and pull-request number.

**14.2 **Each report shall list files added, files modified, and any
original nanoGPT lines superseded but retained as comments.

**14.3 **Each report shall identify every configuration control added or
changed.

**14.4 **Each report shall list tests run and provide exact outcomes.

**14.5 **Each report shall state which exit gates were satisfied and
which stop conditions were considered.

**14.6 **Each report shall identify deviations from the requirements
specification, implementation plan, Test Plan, or this staging plan.

**14.7 **Each report shall state what is intentionally deferred to the
next stage.

**14.8 **No report shall claim work from a later stage merely because
partial scaffolding exists.

# 15. Overall completion criteria

**15.1 **The initial Chebyshev Sheet implementation shall be considered
complete only after Stage 6 acceptance.

**15.2 **Completion shall require a preserved dense path, compact sheet
state, reference and checkpointed execution, training, checkpoint
save/resume, direct compact inference, DDP, target-geometry evidence,
complete regression success, and as-built documentation.

**15.3 **Completion shall not imply that the architecture matches or
exceeds dense nanoGPT quality.

**15.4 **A scientifically negative Stage 6 outcome may still accompany a
functionally complete implementation.

**15.5 **Further architectures, including learned horizontal geometry or
three-dimensional Chebyshev fields, shall be treated as successor work
rather than hidden extensions of this staging plan.

# 16. Immediate next actions

**16.1 **Complete the current Vermeer enhancement without interaction
from thog2 work.

**16.2 **Create the thog2 GitHub repository as a fresh nanoGPT clone.

**16.3 **Provide repository access and identify the intended upstream
baseline commit if it differs from the repository default.

**16.4 **Prepare the separate THOG2 Chebyshev Sheet Test Plan.

**16.5 **Add the requirements specification, implementation plan,
staging plan, and Test Plan in Stage 0.

**16.6 **Do not commence Stage 1 until Stage 0 is merged and the dense
baseline evidence is accepted.
