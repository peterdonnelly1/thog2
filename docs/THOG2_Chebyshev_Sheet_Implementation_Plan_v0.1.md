THOG2

Chebyshev Sheet Implementation Plan

Version 0.1

|                              |                                                                            |
|------------------------------|----------------------------------------------------------------------------|
| **System**                   | thog2 enhancement of nanoGPT                                               |
| **Architecture designation** | Chebyshev Sheet                                                            |
| **Governing specification**  | THOG2 Chebyshev Sheet Requirements Specification v0.1                      |
| **Status**                   | Draft implementation plan; repository implementation not commenced         |
| **Date**                     | 2 July 2026                                                                |
| **Planned repository**       | thog2                                                                      |
| **Repository baseline**      | Fresh clone of nanoGPT                                                     |
| **Repository interaction**   | None; this document is authored outside GitHub pending repository creation |

Non-normative note: This plan deliberately does not modify the existing
thog repository or the future thog2 repository. It is intended to be
version-controlled in thog2 after the repository is created and access
is provided.

# 1. Purpose and authority

**1.1 **This document defines the planned implementation of the
Chebyshev Sheet architecture in the future thog2 repository.

**1.2 **The governing functional requirements are contained in THOG2
Chebyshev Sheet Requirements Specification v0.1.

**1.3 **Where this plan conflicts with the governing requirements
specification, the requirements specification shall prevail until a
versioned amendment resolves the conflict.

**1.4 **This plan specifies implementation structure, sequencing, design
decisions, validation gates, and expected artifacts.

**1.5 **This plan does not itself authorise changes to the existing thog
repository.

**1.6 **The implementation shall begin only after the thog2 repository
exists and its baseline state has been captured.

**1.7 **The implementation shall preserve a working dense nanoGPT path
throughout development.

**1.8 **Scientific superiority shall not be treated as an implementation
acceptance criterion.

# 2. Implementation principles

**2.1 **Correctness shall precede optimization.

**2.2 **The first complete path shall use ordinary PyTorch tensor
operations and autograd without custom CUDA kernels.

**2.3 **The implementation shall generate one logical layer at a time
and shall not persist the complete sampled logical block stack.

**2.4 **Every generated matrix family shall use one universal bivariate
Chebyshev software abstraction parameterised by shape and initialization
policy.

**2.5 **Matrix output rows shall remain distinct sheets; output-row and
within-row coordinates shall not be flattened together.

**2.6 **The horizontal coordinate shall initially be the fixed
left-to-right within-row index normalized to the closed interval from -1
to 1.

**2.7 **The implementation shall use next-token cross-entropy only and
shall not add auxiliary losses.

**2.8 **The dense and sheet models should share the same training loop
so that controlled comparisons are not confounded by duplicated trainer
behavior.

**2.9 **The dense model implementation itself shall remain structurally
independent of sheet parameters.

**2.10 **All changes to original nanoGPT source shall follow the project
source-preservation and THOG-marker conventions.

**2.11 **No existing nanoGPT source line shall be deleted when it is
superseded; the original line shall remain commented where required by
the project convention.

# 3. Repository and branch strategy

**3.1 **The initial thog2 repository shall be a clean clone of the
selected nanoGPT baseline commit.

**3.2 **The baseline commit SHA, upstream URL, Python environment,
PyTorch version, CUDA version, GPU model, and a dense smoke result shall
be recorded before architecture work begins.

**3.3 **The default branch shall remain a reviewable baseline until the
implementation branch is created.

**3.4 **Implementation shall occur on a dedicated feature branch whose
final name shall be selected after repository inspection.

**3.5 **A separate staging plan shall define exact commit and
pull-request boundaries.

**3.6 **Each implementation work package shall leave the branch
executable and testable.

**3.7 **Generated datasets, checkpoints, logs, and run artifacts shall
not be committed unless explicitly selected as small test fixtures.

**3.8 **Documentation created before repository availability shall be
added in a documentation-only commit before or alongside the first
implementation work package.

# 4. Planned source structure

**4.1 **Final paths shall be confirmed against the actual nanoGPT clone
before the first code change.

**4.2 **The preferred source layout is shown below.

|                        |                                                |                                                                        |
|------------------------|------------------------------------------------|------------------------------------------------------------------------|
| **Path**               | **Purpose**                                    | **Implementation note**                                                |
| model.py               | Retained dense nanoGPT model                   | Modify only where a shared factory or configuration hook is necessary. |
| train.py               | Shared dense and sheet training loop           | Add architecture selection while preserving dense behavior.            |
| sample.py              | Checkpoint-driven inference                    | Instantiate dense or sheet model from checkpoint metadata.             |
| sheet/\_\_init\_\_.py  | Sheet package boundary                         | No runtime dependency on the existing thog repository.                 |
| sheet/basis.py         | Coordinate and basis construction              | Chebyshev recurrence, QR orthogonalisation, rank and condition checks. |
| sheet/geometry.py      | Family shapes and Q scaling                    | Single source of truth for R_f, C_f, and derived Q_f.                  |
| sheet/trajectory.py    | Compact coefficient state and materialisation  | One depth basis, shared row-basis cache, family-specific coefficients. |
| sheet/model.py         | Sheet GPT and functional block execution       | Uses generated weights with functional PyTorch operations.             |
| sheet/checkpointing.py | Segmented indexed execution                    | Non-reentrant activation checkpointing and RNG preservation.           |
| sheet/metrics.py       | Detached diagnostics and parameter accounting  | No influence on gradients.                                             |
| sheet/run_naming.py    | Stable architecture-first names                | Include L, d, P, and Q_d.                                              |
| tests/                 | Unit, integration, GPU, DDP, and source guards | Detailed coverage shall be governed by the later test plan.            |

**4.3 **If repository inspection shows that a smaller or clearer layout
is preferable, the implementation may consolidate modules provided
responsibilities and tests remain separable.

**4.4 **The implementation shall not copy the existing thog package
wholesale.

**4.5 **Reusable infrastructure from thog may be ported selectively only
after its behavior and dependencies are reviewed.

# 5. Configuration and model selection

**5.1 **The shared training configuration shall include a model_type
value supporting at least dense and thog2_sheet.

**5.2 **The sheet model configuration shall include L, d, attention-head
count, context length, bias, dropout, P, and Q_d.

**5.3 **The principal initial geometry shall use L equal to 144, d equal
to 768, P equal to 16, and Q_d equal to 128.

**5.4 **For a row width C_f, the default row basis count shall be
derived as follows.

> Q_f = min(C_f, ceil(C_f \* Q_d / d))

**5.5 **For d equal to 768 and Q_d equal to 128, rows of width 3072
shall therefore use Q_f equal to 512.

**5.6 **Configuration validation shall reject non-positive dimensions, P
greater than L, Q_d greater than d, incompatible head geometry, and any
derived Q_f outside the valid range.

**5.7 **Checkpoint model arguments shall record the basis-construction
version and row-order scaling rule so that future changes cannot
silently reinterpret old state.

**5.8 **Activation-checkpoint settings shall be treated as execution
controls rather than mathematical model state where compatibility
permits.

**5.9 **The first implementation shall not expose learned row
coordinates, row permutations, coordinate warping, Vermeer, or auxiliary
losses.

# 6. Basis construction

**6.1 **The basis implementation shall construct raw first-kind
Chebyshev values by recurrence rather than by unstable explicit powers.

**6.2 **Layer and within-row indices shall initially map linearly and
monotonically to coordinates in the closed interval from -1 to 1.

**6.3 **The raw sampled Chebyshev columns shall be converted to a
deterministic discrete orthonormal basis by reduced QR decomposition.

**6.4 **Column signs after QR shall be normalised deterministically so
that repeated construction under identical geometry yields identical
bases.

**6.5 **The QR-based basis shall preserve the discrete span of the
selected Chebyshev terms.

**6.6 **Basis construction shall occur in a numerically conservative
dtype, initially float64 on CPU unless measured cost or compatibility
requires a documented alternative.

**6.7 **The runtime basis buffer may be cast to float32 after
construction and shall be converted to the operation dtype only at the
materialisation boundary where necessary.

**6.8 **The implementation shall verify finiteness, expected shape, full
column rank, approximate column orthonormality, and deterministic
reproduction.

**6.9 **Distinct row bases shall be cached by the tuple of row width,
row order, basis version, device, and runtime dtype.

**6.10 **The depth basis shall be shared by all generated families in
one model instance.

**6.11 **Basis buffers shall be non-persistent so that checkpoints store
model coefficients rather than reproducible fixed tables.

**6.12 **The implementation shall measure basis-construction time and
peak temporary memory for Q_f values up to at least 1024 before
accepting the QR strategy for the full sweep.

**6.13 **If reduced QR proves operationally unacceptable, any
replacement shall preserve the same sampled span, deterministic
behavior, and conditioning tests and shall be documented before use.

# 7. Compact coefficient state

**7.1 **The sheet trajectory module shall own one learned coefficient
tensor for each generated tensor family.

**7.2 **For a matrix family with R_f output rows, the coefficient shape
shall be R_f by P by Q_f.

**7.3 **The attention input projection shall use one packed family with
coefficient shape 3d by P by Q_d unless later profiling justifies a
split implementation that is mathematically equivalent.

**7.4 **The attention output projection shall use coefficient shape d by
P by Q_d.

**7.5 **The MLP expansion projection shall use coefficient shape 4d by P
by Q_d.

**7.6 **The MLP contraction projection shall use coefficient shape d by
P by Q_4d.

**7.7 **Repeated vector families represented by sheets shall use one
output row and a row width equal to the vector length.

**7.8 **Family metadata shall record semantic type, output rows, row
width, row order, initialization scale, and weight-decay classification.

**7.9 **The implementation shall derive parameter counts directly from
coefficient tensor shapes and shall cross-check them against analytical
formulas.

|                             |          |               |                                   |                       |
|-----------------------------|----------|---------------|-----------------------------------|-----------------------|
| **Family**                  | **Rows** | **Row width** | **Coefficient shape at P16/Q128** | **Coefficient count** |
| Attention input projection  | 2304     | 768           | \[2304, 16, 128\]                 | 4,718,592             |
| Attention output projection | 768      | 768           | \[768, 16, 128\]                  | 1,572,864             |
| MLP expansion               | 3072     | 768           | \[3072, 16, 128\]                 | 6,291,456             |
| MLP contraction             | 768      | 3072          | \[768, 16, 512\]                  | 6,291,456             |
| Large-matrix total          | \-       | \-            | \-                                | 18,874,368            |

**7.10 **No generated L-by-R_f-by-C_f tensor shall be registered as a
parameter or persistent buffer.

# 8. Layer materialisation

**8.1 **The reference materialisation path shall evaluate one family for
one logical layer from the compact coefficient tensor and fixed basis
rows.

**8.2 **For family f and logical layer l, the implementation shall first
mix the depth axis.

> mixed_f = einsum("p,rpq-\>rq", depth_basis\[l, :\], coefficients_f)

**8.3 **The implementation shall then generate the conventional matrix
for that logical layer.

> weight_f = mixed_f @ transpose(row_basis_f)

**8.4 **The resulting weight_f shall have shape R_f by C_f and shall be
numerically equivalent to evaluating every row sheet at the selected
depth.

**8.5 **The implementation shall materialise only families required by
the current logical block.

**8.6 **Generated tensors shall be ephemeral and shall remain connected
to the compact coefficient graph for autograd.

**8.7 **The reference implementation may use einsum, tensordot, or
equivalent matrix operations, but the chosen expression shall have an
explicit shape contract and unit tests.

**8.8 **A batched or fused implementation may replace the reference path
only after numerical and gradient equivalence are established.

**8.9 **The materialisation API shall return conventional tensor shapes
so that attention and MLP calculations remain recognisable and testable.

# 9. Initialization

**9.1 **Initialization shall begin with all coefficient tensors set to
exact zero.

**9.2 **The first depth mode shall then be initialized so that generated
matrix profiles are initially shared across logical depth while higher
depth modes remain available to learn immediately.

**9.3 **For a discrete orthonormal depth basis and row basis, the
initial coefficient standard deviation for a target generated-weight
standard deviation s_f shall initially be derived from the average
basis-row energy.

> coefficient_std_f = s_f \* sqrt(L \* C_f / Q_f)

**9.4 **The formula in 9.3 assumes that only the constant depth mode is
active and that deterministic basis signs make the first depth column a
positive constant.

**9.5 **The attention input and MLP expansion families shall initially
target the conventional nanoGPT standard deviation of 0.02 unless
repository inspection identifies a baseline-specific value.

**9.6 **The attention output and MLP contraction families shall retain
nanoGPT residual-projection scaling, initially 0.02 divided by the
square root of 2L.

**9.7 **Layer-normalisation scale sheets, where used, shall generate
exact ones at all logical sample positions at initialization.

**9.8 **Bias sheets, where used, shall generate exact zeros at
initialization.

**9.9 **Initialization tests shall measure generated-weight mean, RMS,
standard deviation, endpoint behavior, and layer-to-layer equality under
the shared-depth initial mode.

**9.10 **The acceptable generated-weight distribution tolerance shall be
fixed in the test plan after a small numerical study of basis leverage
across supported Q_f values.

**9.11 **If endpoint leverage produces unacceptable initialization
outliers, the remedy shall be documented and shall preserve the
governing basis-span requirements.

**9.12 **The implementation shall not allocate a complete dense logical
block stack merely to initialize compact coefficients.

# 10. Sheet model integration

**10.1 **The sheet model shall retain nanoGPT token embeddings,
positional embeddings, final layer normalization, tied language-model
head, causal attention semantics, residual connections, dropout, and
cross-entropy behavior.

**10.2 **The persistent ModuleList of independent dense transformer
blocks shall be replaced in the sheet model by one compact sheet
trajectory module and an indexed logical-layer execution loop.

**10.3 **The dense GPT class shall remain available and shall not
allocate sheet coefficient tensors.

**10.4 **The sheet logical block shall use functional PyTorch operations
with generated tensors rather than constructing persistent nn.Linear
modules for each logical layer.

**10.5 **Attention shall continue to use the retained scaled-dot-product
attention path where supported by the nanoGPT baseline.

**10.6 **The packed query-key-value output shall be split into q, k, and
v activations exactly as in the dense baseline.

**10.7 **The sheet model forward method shall return logits and
next-token cross-entropy in the same interface shape as the dense model.

**10.8 **Reference execution shall initially run logical layers
sequentially without activation checkpointing so that correctness
failures remain observable.

**10.9 **Model startup reporting shall distinguish persistent
parameters, sheet coefficients, conventional non-sheet parameters, and
dense-equivalent parameters.

**10.10 **Model construction shall verify that no dense logical block
stack exists in named parameters or state_dict.

# 11. Optimizer and training integration

**11.1 **The shared training loop shall select the dense or sheet model
from model_type while retaining identical data loading, scheduler,
gradient accumulation, clipping, mixed precision, evaluation, and
logging semantics.

**11.2 **The initial sheet implementation shall use the same base
learning-rate schedule as the matched dense run unless a controlled
experiment explicitly changes it.

**11.3 **Matrix-family sheet coefficients shall be assigned to
weight-decayed optimizer groups.

**11.4 **Layer-normalisation and bias sheet coefficients shall be
assigned to non-decayed optimizer groups according to semantic family
metadata rather than tensor dimensionality.

**11.5 **Every trainable parameter shall appear in exactly one optimizer
group.

**11.6 **Optimizer group ordering and metadata shall be deterministic to
permit checkpoint resume tests.

**11.7 **The optimizer shall update compact coefficients directly;
generated weights shall never be optimizer parameters.

**11.8 **Completed optimizer updates shall use exact update-count
semantics independent of gradient-accumulation microsteps.

**11.9 **The data-sampling RNG shall be separated from
model-construction RNG before controlled dense-versus-sheet runs are
accepted.

**11.10 **The first complete trainer shall support single-GPU execution
before DDP work begins.

# 12. Ephemeral execution and memory control

**12.1 **After reference correctness is established, the model shall add
segmented activation-checkpointed execution for the indexed
logical-layer stack.

**12.2 **Checkpointing shall use the non-reentrant PyTorch path unless
baseline compatibility forces a documented alternative.

**12.3 **Checkpointing shall preserve RNG state so that dropout behavior
agrees with reference execution under deterministic tests.

**12.4 **The checkpoint segment size shall be configurable, including a
disabled reference value.

**12.5 **During backward recomputation, the required layer weights shall
be regenerated from compact coefficients rather than retrieved from a
stored dense stack.

**12.6 **Evaluation and inference shall bypass training-only activation
checkpoint recomputation.

**12.7 **Memory telemetry shall distinguish model construction,
optimizer allocation, first backward state allocation, and steady-state
training peaks.

**12.8 **The implementation shall verify that peak memory does not grow
in proportion to persisting all L logical matrices.

**12.9 **Performance optimization shall first target avoidable repeated
basis conversion and materialisation overhead before considering custom
kernels.

**12.10 **Any caching of generated weights across logical layers shall
be prohibited in the first memory-acceptance path.

# 13. Checkpoint, resume, and inference

**13.1 **Sheet checkpoints shall identify model_type as thog2_sheet.

**13.2 **Checkpoint model arguments shall include all sheet geometry,
basis version, bias state, and model dimensions required for exact
reconstruction.

**13.3 **Fixed basis buffers shall be regenerated from configuration and
shall not be required in persistent checkpoint state.

**13.4 **Checkpoint save shall include compact model state, optimizer
state, completed update count, best and latest validation state,
scheduler state where applicable, and required RNG state.

**13.5 **Resume shall validate requested model geometry before loading
optimizer state.

**13.6 **A geometry mismatch shall fail with a direct message rather
than attempting automatic conversion.

**13.7 **Execution-only activation-checkpoint settings may be changed on
resume if model state is unaffected.

**13.8 **sample.py shall reconstruct dense or sheet models from
checkpoint metadata without a user-side conversion step.

**13.9 **Sheet inference shall load compact coefficient state and shall
not create or save a dense checkpoint.

**13.10 **Inference controls shall retain prompt, sample count,
continuation length, temperature, top-k, and seed support from the
baseline workflow.

# 14. Telemetry and diagnostics

**14.1 **The implementation shall report training and validation
cross-entropy, completed updates, consumed tokens, throughput, elapsed
time, and learning rate.

**14.2 **The implementation shall report persistent parameters,
dense-equivalent parameters, sheet coefficient parameters, and
per-family counts.

**14.3 **The implementation shall report peak allocated and reserved GPU
memory at documented phases.

**14.4 **Coefficient RMS and gradient norm shall be reported by family
at logging cadence.

**14.5 **Depth-order energy shall be calculated from coefficient energy
aggregated over output rows and row orders.

**14.6 **Row-order energy shall be calculated from coefficient energy
aggregated over output rows and depth orders.

**14.7 **A documented high-order energy fraction shall expose whether
capacity accumulates near either configured order limit.

**14.8 **Sampled generated-weight mean, RMS, standard deviation, and
maximum absolute value shall be reported for selected layers and
families.

**14.9 **Optional finite-difference roughness shall compare adjacent
trained layer samples and adjacent trained within-row samples without
assigning semantics to off-grid coordinates.

**14.10 **Basis condition and orthonormality diagnostics shall be
recorded at model startup rather than every update.

**14.11 **All diagnostic calculations shall be detached and shall not
change gradients, optimizer state, or random-number streams used by
training.

# 15. Implementation verification

**15.1 **A separate test plan shall define the complete test inventory,
tolerances, fixtures, and execution matrix.

**15.2 **Each work package shall add or update tests before its
completion gate is accepted.

**15.3 **Core mathematical tests shall cover basis values, discrete
span, rank, orthonormality, deterministic signs, shape validation, and
row-order derivation.

**15.4 **Materialisation tests shall compare the implementation against
direct matrix evaluation on small exact geometries.

**15.5 **A saturated small geometry with P equal to L and Q_f equal to
C_f shall reproduce arbitrary sampled row sheets within numerical
tolerance.

**15.6 **Gradient tests shall prove finite non-zero gradients to every
used coefficient family.

**15.7 **Initialization tests shall verify family-specific
generated-weight statistics and exact LayerNorm-one and bias-zero
behavior where applicable.

**15.8 **Reference and checkpointed execution shall be compared under
deterministic conditions.

**15.9 **Checkpoint round-trip, resume, direct compact inference,
dense-path regression, single-GPU smoke, and DDP smoke shall be
mandatory before implementation completion.

**15.10 **Source-guard tests shall verify the absence of persistent
dense logical block parameters and compliance with required THOG source
markers in modified nanoGPT files.

# 16. Implementation work packages

**16.1 **The following work packages define the intended technical
sequence. The later staging plan may group or split them into different
commits or pull requests.

## 16.2 Work packages 0-4

|        |                              |                                                                                                         |                                                                                 |
|--------|------------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| **WP** | **Name**                     | **Primary work**                                                                                        | **Completion gate**                                                             |
| 0      | Baseline capture             | Clone, record commit/environment, run dense CPU and GPU smokes, freeze comparison evidence.             | Clean baseline; reproducible dense smoke; no sheet code.                        |
| 1      | Basis and geometry core      | Implement coordinates, Chebyshev recurrence, QR basis, Q scaling, family geometry, analytical counts.   | Mathematical and validation tests pass through Q=1024.                          |
| 2      | Compact trajectory state     | Implement coefficient containers, basis sharing, family metadata, materialisation API.                  | Shape, direct-equation, saturated-basis, state-schema, and gradient tests pass. |
| 3      | Reference sheet model        | Implement functional sheet block and SheetGPT; retain dense GPT; sequential reference forward/backward. | CPU reference model trains for representative updates; dense regressions pass.  |
| 4      | Initialization and optimizer | Implement family initialization, semantic decay grouping, counts, startup reports.                      | Generated statistics and optimizer coverage tests pass.                         |

## 16.3 Work packages 5-9

|        |                                  |                                                                                                  |                                                                                 |
|--------|----------------------------------|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| **WP** | **Name**                         | **Primary work**                                                                                 | **Completion gate**                                                             |
| 5      | Shared trainer and checkpoints   | Add model_type selection, save/resume metadata, exact updates, deterministic data controls.      | Dense and sheet single-GPU smoke, checkpoint round-trip, resume pass.           |
| 6      | Checkpointed ephemeral execution | Port indexed segmented checkpointing, RNG preservation, memory phase telemetry.                  | Reference equivalence and bounded-memory GPU tests pass.                        |
| 7      | Inference and diagnostics        | Checkpoint-driven sample path, run naming, W&B/console telemetry, coefficient-order diagnostics. | Compact checkpoint generates text; metrics finite; naming unambiguous.          |
| 8      | DDP and target geometry          | DDP integration, target L144/D768/P16/Q128 smoke, Q512 row basis, performance profiling.         | No deadlock/divergence; target geometry completes bounded update sequence.      |
| 9      | Controlled pilot and as-built    | Matched dense/sheet pilot, document deviations, final regression, as-built documentation.        | Implementation acceptance package complete; scientific claims remain qualified. |

**16.4 **No work package shall be considered complete solely because its
code imports or a single smoke run starts.

**16.5 **A failed gate shall be resolved in the same work package or
documented as a blocking defect before later work proceeds.

# 17. Controlled pilot configuration

**17.1 **The first scientific pilot shall not commence until functional
acceptance through work package 8 is complete.

**17.2 **The principal sheet pilot shall use P equal to 16, Q_d equal to
128, and Q_4d equal to 512 at the selected L144/D768 logical geometry
where hardware permits.

**17.3 **The first row-capacity sweep should bracket the principal
configuration with Q_d values 64 and 256 and corresponding 4d values 256
and 1024.

**17.4 **Dense and sheet runs shall use the same dataset, logical
geometry, context length, batch trace, validation sample, optimizer
schedule, completed update count, and consumed token count except for
architecture-required differences.

**17.5 **The pilot shall compare validation loss against updates,
tokens, and wall-clock time.

**17.6 **The pilot shall also report throughput, peak memory, persistent
parameters, dense-equivalent parameters, and coefficient-order
utilization.

**17.7 **A low-order failure shall trigger the planned capacity sweep
before the sheet hypothesis is rejected.

**17.8 **A configuration requiring Q_d close to full row width to train
usefully shall be treated as evidence against useful rowwise
compression.

**17.9 **Pilot outcomes shall not be generalized beyond the tested
geometry, dataset, and training budget without further evidence.

# 18. Deliverables and completion criteria

**18.1 **The implementation deliverable shall include source code,
tests, configuration controls, checkpoint and inference paths,
telemetry, and user-facing run instructions.

**18.2 **The repository shall include the requirements specification,
this implementation plan, the later test plan, the later staging plan,
and an as-built document.

**18.3 **The dense nanoGPT path shall remain operational and
independently selectable.

**18.4 **The sheet model shall store compact coefficient state and shall
not persist the full logical block stack.

**18.5 **Reference and checkpointed sheet execution shall be
operational.

**18.6 **Single-GPU and DDP training, checkpoint save, resume, and
direct compact-checkpoint inference shall be operational.

**18.7 **The complete regression suite shall pass on the accepted
implementation commit.

**18.8 **Target-geometry evidence shall demonstrate bounded memory and
finite training behavior for a documented update count.

**18.9 **All implementation deviations from the governing requirements
and this plan shall be recorded in the as-built document and, where
architectural, in a specification amendment.

**18.10 **Implementation completion shall not claim that Chebyshev Sheet
matches or exceeds dense nanoGPT quality.

# 19. Deferred decisions and change control

**19.1 **Learned row coordinates, learned permutations, coordinate
warping, multiple sheets per row, Vermeer, and three-dimensional
Chebyshev volumes are deferred.

**19.2 **Changing the fixed horizontal coordinate hypothesis shall
require a versioned requirements amendment rather than an unrecorded
implementation shortcut.

**19.3 **Changing the sole cross-entropy objective shall require a
versioned requirements amendment.

**19.4 **Changing from independent output-row sheets to shared row
coefficients or a flattened mega-sheet shall require a versioned
requirements amendment.

**19.5 **Replacing QR-orthogonalised sampled Chebyshev bases with a
mathematically equivalent stable basis implementation may be handled as
an implementation-plan amendment if the represented span and
configuration semantics remain unchanged.

**19.6 **Custom CUDA or Triton kernels shall be considered only after
profiling identifies materialisation as a dominant cost in a correct
end-to-end implementation.

**19.7 **Any optimization shall preserve the reference path for
correctness comparison until the optimized path has equivalent test
coverage.

**19.8 **The implementation plan shall be revised after repository
inspection if baseline nanoGPT structure or current PyTorch interfaces
materially differ from the assumptions recorded here.

# 20. Immediate next actions

**20.1 **Complete and merge the ongoing Vermeer enhancement without
interaction from thog2 work.

**20.2 **Create the thog2 GitHub repository as a fresh nanoGPT clone.

**20.3 **Provide repository access and identify the intended upstream
baseline commit if it is not the repository default.

**20.4 **Add the requirements specification and this implementation plan
to thog2 in a documentation-only change.

**20.5 **Prepare the separate Chebyshev Sheet Test Plan.

**20.6 **Prepare the separate Chebyshev Sheet Staging Plan with minimum
sensible implementation stages and explicit stop conditions.

**20.7 **Commence work package 0 only after repository state and access
are confirmed.
