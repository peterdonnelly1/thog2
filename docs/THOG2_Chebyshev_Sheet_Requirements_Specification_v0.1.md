THOG2

Chebyshev Sheet Requirements Specification

Version 0.1

|                                 |                                                                |
|---------------------------------|----------------------------------------------------------------|
| **System**                      | thog2 enhancement of nanoGPT                                   |
| **Architecture designation**    | Chebyshev Sheet                                                |
| **Checkpoint model identifier** | thog2_sheet                                                    |
| **Status**                      | Draft requirements specification; implementation not commenced |
| **Date**                        | 2 July 2026                                                    |
| **Planned repository**          | thog2                                                          |
| **Repository baseline**         | Fresh clone of nanoGPT                                         |
| **Relationship to THOGv1**      | Independent successor experiment; no implementation dependency |

Non-normative note: This document is being authored outside the existing
THOG repository so that completion of the Vermeer enhancement is not
disturbed. It is intended to be added to the future thog2 repository
under normal version control after that repository is created.

# 1. Purpose

**1.1 **This document specifies a transformer architecture named
Chebyshev Sheet for implementation in the planned thog2 project.

**1.2 **Chebyshev Sheet shall replace independently persisted per-layer
transformer-block matrices with compact learned bivariate Chebyshev
surfaces from which logical layer weights are generated.

**1.3 **The architecture shall test whether a transformer trained from
scratch can organise its internal representations so that matrix rows
are adequately expressible as smooth functions of logical depth and
within-row weight position.

**1.4 **The primary objective shall be substantial reduction of
persistent transformer-block parameters while retaining useful
next-token prediction quality.

**1.5 **The architecture shall use ordinary next-token cross-entropy as
its primary and, in the first implementation, sole model-training
objective.

**1.6 **The architecture shall not require a pre-trained dense teacher,
dense-weight reconstruction target, knowledge distillation, latent-space
mean-squared error, or post-training conversion phase.

**1.7 **Functional implementation acceptance shall be separated from
scientific evidence that the architecture is competitive with a dense
transformer or THOGv1.

# 2. Scope and project independence

**2.1 **The implementation target shall be a new Git and GitHub project
named thog2.

**2.2 **The initial thog2 codebase shall be a fresh clone of nanoGPT
rather than a branch of the existing thog repository.

**2.3 **The existing THOGv1 and Vermeer specifications, implementation,
tests, and experimental controls may be used as design references.

**2.4 **thog2 shall not import runtime code from the existing thog
repository.

**2.5 **Completion, testing, or merging of Vermeer shall not be a
prerequisite for commencing thog2 after the thog2 repository is
available.

**2.6 **The dense nanoGPT execution path shall remain present as an
internal control and regression baseline.

**2.7 **Where existing nanoGPT source lines are superseded, they shall
remain present as commented source in accordance with the project
modification convention.

**2.8 **New or replaced nanoGPT source shall use the established THOG
source markers exactly as required by the project conventions.

# 3. Normative language and terminology

**3.1 **The word shall denotes a mandatory requirement.

**3.2 **The word should denotes a preferred requirement that may be
departed from only for a documented reason.

**3.3 **The word may denotes a permitted option.

**3.4 **Logical layer shall mean an ordered transformer block position
in the model depth.

**3.5 **Tensor family shall mean a corresponding transformer-block
parameter type repeated at every logical layer, such as the attention
input projection weight.

**3.6 **Matrix row shall mean one output-coordinate row of a
conventional two-dimensional learned weight matrix.

**3.7 **Depth-row weight sheet shall mean a continuous bivariate
function whose vertical coordinate is logical depth, whose horizontal
coordinate is within-row weight position, and whose function value is a
scalar weight value.

**3.8 **Sheet height shall mean the scalar weight value returned by a
depth-row weight sheet at a specified depth and row coordinate.

**3.9 **Depthwise weight trajectory shall mean a vertical slice through
a depth-row weight sheet at one fixed within-row position.

**3.10 **Rowwise weight profile shall mean a horizontal slice through a
depth-row weight sheet at one fixed logical depth.

**3.11 **Interlayer sample shall mean evaluation of a sheet at a depth
coordinate that does not correspond to a trained logical layer.

**3.12 **Interweight sample shall mean evaluation of a sheet at a
horizontal coordinate that does not correspond to a trained discrete
weight position.

**3.13 **Persistent parameter shall mean a trainable value stored in
model state and checkpoints.

**3.14 **Generated weight shall mean an ephemeral scalar or tensor value
obtained by evaluating learned sheet coefficients at a logical sample
coordinate.

**3.15 **Basis terms shall mean the number of included Chebyshev basis
functions. A basis containing P terms has maximum polynomial degree P
minus one.

# 4. Conceptual model

**4.1 **The primary learned transformer-block objects shall be depth-row
weight sheets rather than independent logical-layer weight matrices.

**4.2 **The vertical sheet axis shall run through logical layers in
their execution order.

**4.3 **The horizontal sheet axis shall run left to right through scalar
weight positions within one conventional matrix row.

**4.4 **The value of the sheet at one vertical and horizontal coordinate
shall be one scalar weight value.

**4.5 **Holding the horizontal coordinate fixed and moving vertically
shall produce one depthwise weight trajectory.

**4.6 **Holding the vertical coordinate fixed and moving horizontally
shall produce one rowwise weight profile for one logical layer.

**4.7 **Logical transformer weights shall be treated as samples from the
continuous sheet rather than as the primary persistent learned objects.

**4.8 **The first implementation shall train and use only the discrete
layer and within-row sample locations required by the configured
transformer geometry.

**4.9 **Mathematical continuity between trained sample locations shall
not be interpreted as evidence that interlayer or interweight samples
have useful model semantics.

**4.10 **Sequence position shall not be a sheet coordinate for
transformer-block weight generation.

**4.11 **The same generated matrix shall continue to be applied to every
sequence position in the conventional transformer operation.

**4.12 **The architecture shall preserve the conventional common
activation representation consumed by the attention, MLP,
layer-normalisation, and residual operations of a logical layer.

**4.13 **The architecture may rely on training to organise
latent-coordinate usage so that the imposed fixed left-to-right row
geometry becomes functionally useful.

**4.14 **The primary scientific question shall be whether the resulting
constrained function class is sufficiently expressive and trainable, not
whether it reproduces the weights of an independently trained dense
model.

# 5. Logical transformer geometry

**5.1 **Let L denote the number of logical transformer layers.

**5.2 **Let d denote the model embedding width, also called d_model or
n_embd.

**5.3 **Let R_f and C_f denote respectively the output-row count and
within-row width of matrix tensor family f.

**5.4 **The first implementation shall preserve nanoGPT transformer
operations, residual paths, attention semantics, MLP semantics, masking,
dropout, embeddings, and language-model output semantics except where
weight storage and materialisation are explicitly replaced.

**5.5 **For the standard nanoGPT block, the attention input projection
matrix shall have R_f equal to 3d and C_f equal to d.

**5.6 **For the standard nanoGPT block, the attention output projection
matrix shall have R_f equal to d and C_f equal to d.

**5.7 **For the standard nanoGPT block, the MLP expansion matrix shall
have R_f equal to 4d and C_f equal to d.

**5.8 **For the standard nanoGPT block, the MLP contraction matrix shall
have R_f equal to d and C_f equal to 4d.

**5.9 **The packed query-key-value implementation may remain packed as
one 3d-by-d tensor family.

**5.10 **The implementation shall not require a separate mathematical
sheet theory for query, key, value, attention output, MLP expansion, or
MLP contraction weights.

**5.11 **Token embeddings, positional embeddings, final layer
normalisation, and the tied language-model head may remain conventional
persistent parameters because they are not repeated as transformer-block
tensor families over logical depth.

# 6. Universal Chebyshev sheet parameterisation

**6.1 **Every generated matrix tensor family shall use the same
tensor-product bivariate Chebyshev parameterisation.

**6.2 **Let P denote the number of depthwise Chebyshev basis terms.

**6.3 **Let Q_f denote the number of rowwise Chebyshev basis terms used
for tensor family f.

**6.4 **Let B_depth be the fixed depth basis matrix with shape L by P.

**6.5 **Let B_row_f be the fixed row basis matrix with shape C_f by Q_f.

**6.6 **Let A_f be the learned coefficient tensor for matrix family f
with shape R_f by P by Q_f.

**6.7 **For output row r of tensor family f, the complete sampled sheet
shall be defined by the matrix product:

> S_f_r = B_depth @ A_f\[r, :, :\] @ transpose(B_row_f)

**6.8 **S_f_r shall have shape L by C_f.

**6.9 **Row r of logical layer l shall be the l-th sampled row of S_f_r.

> W_f\[l, r, :\] = S_f_r\[l, :\]

**6.10 **Each output row shall have its own learned P-by-Q_f coefficient
table.

**6.11 **All output rows of the same width shall use the same fixed
basis-generation rule but shall not share their learned coefficient
table unless a later separately specified experiment introduces such
sharing.

**6.12 **A conventional three-dimensional logical weight collection with
shape L by R_f by C_f shall therefore be represented as a stack of R_f
independent two-dimensional sheets.

**6.13 **The implementation shall not flatten the output-row and
within-row dimensions into one combined mega-row in the first version.

**6.14 **A vector tensor family repeated over depth shall use the same
parameterisation with R_f equal to one.

**6.15 **The fixed basis matrices shall not be trainable model
parameters.

**6.16 **Generated per-layer dense tensors shall not be persistent model
parameters.

# 7. Coordinates and Chebyshev bases

**7.1 **Each logical layer index shall map deterministically and
monotonically to a normalized depth coordinate in the closed interval
from -1 to 1.

**7.2 **Each within-row scalar index shall map deterministically and
monotonically from left to right to a normalized row coordinate in the
closed interval from -1 to 1.

**7.3 **The first implementation shall use Chebyshev polynomials of the
first kind for both axes.

**7.4 **The depth basis shall contain terms T_0 through T\_(P-1).

**7.5 **The row basis for family f shall contain terms T_0 through
T\_(Q_f-1).

**7.6 **P shall be at least one and shall not exceed L.

**7.7 **Q_f shall be at least one and shall not exceed C_f.

**7.8 **The discrete basis matrices shall have full column rank for
every supported configuration.

**7.9 **The basis construction and any normalization or
orthogonalisation shall preserve the discrete span of the selected
Chebyshev terms.

**7.10 **The implementation shall use numerically stable basis
evaluation suitable for rowwise orders of at least 512 terms.

**7.11 **The same depth basis shall be used by every generated
transformer-block tensor family with the same L and P.

**7.12 **The same row basis may be cached and reused by every tensor
family having the same C_f and Q_f.

**7.13 **The first implementation shall use fixed row coordinates rather
than learned weight addresses, learned permutations, or family-specific
coordinate warping.

**7.14 **Any future learned-coordinate variant shall require a separate
specification because it changes the architectural hypothesis being
tested.

# 8. Tensor-family coverage

**8.1 **The attention input projection weight shall be generated by
Chebyshev sheets.

**8.2 **The attention output projection weight shall be generated by
Chebyshev sheets.

**8.3 **The MLP expansion weight shall be generated by Chebyshev sheets.

**8.4 **The MLP contraction weight shall be generated by Chebyshev
sheets.

**8.5 **Layer-normalisation scale vectors repeated over depth shall use
the universal sheet mechanism with one sheet per vector family unless an
implementation-stage measurement justifies retaining them
conventionally.

**8.6 **Layer-normalisation bias vectors repeated over depth shall use
the universal sheet mechanism when bias is enabled unless an
implementation-stage measurement justifies retaining them
conventionally.

**8.7 **Linear bias vectors repeated over depth shall use the universal
sheet mechanism when bias is enabled unless an implementation-stage
measurement justifies retaining them conventionally.

**8.8 **Any exception under 8.5 through 8.7 shall be documented, shall
preserve the matrix-sheet experiment unchanged, and shall not be
represented as a different sheet geometry.

**8.9 **All supported families shall be generated through one common
mathematical and software abstraction parameterised by shape,
coefficient state, and initialization policy.

**8.10 **Family-specific learned coefficient tensors shall remain
independent because the represented operations have different functions
and shapes.

# 9. Default basis configuration and scaling rule

**9.1 **The principal target geometry shall use L equal to 144 and d
equal to 768.

**9.2 **The principal depth basis shall use P equal to 16 terms.

**9.3 **The principal row basis for a row of width d equal to 768 shall
use Q_d equal to 128 terms.

**9.4 **For a row width C_f, the default derived row basis count shall
be:

> Q_f = min(C_f, ceil(C_f \* Q_d / d))

**9.5 **For d equal to 768, Q_d equal to 128, and C_f equal to 3072, the
derived Q_f shall be 512.

**9.6 **The principal rowwise configuration therefore represents
approximately six discrete within-row positions per rowwise basis term.

**9.7 **The principal depthwise configuration represents nine logical
layers per depthwise basis term.

**9.8 **P and Q_d shall be configurable independently.

**9.9 **The implementation shall reject configurations in which a
derived Q_f exceeds C_f unless the value is explicitly capped by the
rule in 9.4.

**9.10 **The first scientific sweep should include Q_d values 64, 128,
and 256 at P equal to 16, giving corresponding 4d row orders 256, 512,
and 1024.

**9.11 **The default values in this section shall be treated as initial
experimental settings rather than claims of optimality.

# 10. Parameter-count requirements

**10.1 **For a generated matrix family with R_f rows, depth order P, and
row order Q_f, the persistent sheet coefficient count shall be R_f times
P times Q_f.

**10.2 **For a generated vector family, the persistent sheet coefficient
count shall be P times Q_f.

**10.3 **Fixed basis matrices shall not be included in trainable
parameter counts.

**10.4 **Generated logical-layer weights shall not be included in
persistent parameter counts.

**10.5 **The implementation shall report persistent sheet parameters
separately from conventional non-sheet parameters.

**10.6 **The implementation shall report the dense-equivalent logical
parameter count for the same L, d, and bias configuration.

**10.7 **The implementation shall report parameter counts by tensor
family.

**10.8 **The implementation shall report compression ratios relative to
both the matched dense block matrices and, where calculated, a
depth-only THOG representation with the same P.

**10.9 **For L equal to 144, d equal to 768, P equal to 16, Q_d equal to
128, and Q_4d equal to 512, the four large matrix families shall contain
18,874,368 persistent sheet coefficients in total.

**10.10 **The corresponding dense-equivalent large matrix stack shall
contain 1,019,215,872 scalar weights.

**10.11 **The corresponding depth-only THOG representation with 16 terms
per scalar weight would contain 113,246,208 large-matrix coefficients.

**10.12 **The values in 10.9 through 10.11 exclude embeddings,
positional embeddings, final layer normalisation, biases, block
layer-normalisation vectors, and the tied language-model head.

|                             |          |               |                        |
|-----------------------------|----------|---------------|------------------------|
| **Large matrix family**     | **Rows** | **Row width** | **Sheet coefficients** |
| Attention input projection  | 2304     | 768           | 4,718,592              |
| Attention output projection | 768      | 768           | 1,572,864              |
| MLP expansion               | 3072     | 768           | 6,291,456              |
| MLP contraction             | 768      | 3072          | 6,291,456              |
| Total                       | \-       | \-            | 18,874,368             |

Non-normative note: The table is an exact reference calculation for the
principal L144/D768/P16/Q128 geometry. It is not the whole-model
parameter count.

# 11. Weight materialisation and execution

**11.1 **The implementation shall provide a correctness-first reference
execution path.

**11.2 **The implementation shall generate the tensor families required
by one logical layer from compact sheet coefficients and fixed basis
rows.

**11.3 **The implementation shall not persist the full sampled
L-by-R_f-by-C_f logical weight collection.

**11.4 **The implementation shall release generated logical-layer
weights when they are no longer required by forward or backward
computation.

**11.5 **The implementation shall support activation-checkpointed
ephemeral execution in which required generated weights are recreated
during backward computation.

**11.6 **Checkpointed execution shall preserve random-number-generator
state where required for dropout equivalence.

**11.7 **Evaluation and inference shall not perform unnecessary
activation-checkpoint recomputation.

**11.8 **The materialisation path shall preserve gradient flow from
cross-entropy loss through generated weights to every used sheet
coefficient tensor.

**11.9 **Generated tensors shall use the correct device and computation
dtype for the surrounding model operation.

**11.10 **Fixed bases may be cached as non-persistent buffers when doing
so reduces repeated work without changing model state semantics.

**11.11 **The implementation may batch or fuse row-sheet evaluation
provided that the mathematical result and gradient are equivalent to the
reference equation.

**11.12 **Custom CUDA kernels shall not be required for the first
functionally complete implementation.

# 12. Initialization

**12.1 **Sheet initialization shall produce generated matrix weights
with family-specific mean and scale compatible with nanoGPT
initialization objectives.

**12.2 **Attention output and MLP contraction projections shall retain
the depth-dependent residual scaling intent of nanoGPT initialization.

**12.3 **Generated layer-normalisation scale vectors shall initialize to
exact ones where they are represented by sheets.

**12.4 **Generated bias vectors shall initialize to exact zeros where
bias is enabled and represented by sheets.

**12.5 **Initialization shall not require temporary persistence of a
complete dense logical block stack.

**12.6 **Initialization shall be deterministic under a fixed
model-construction seed.

**12.7 **Initialization shall avoid coefficient scaling that causes
generated weight variance to grow with P or Q_f.

**12.8 **The implementation plan shall derive and test the coefficient
variance needed to meet 12.1 and 12.7.

**12.9 **A newly initialized compressed sheet model shall not be
required to match the exact weights or outputs of a separately
initialized dense nanoGPT model.

# 13. Training and optimisation

**13.1 **Chebyshev Sheet shall train end to end from scratch using
next-token cross-entropy.

**13.2 **The first implementation shall not add auxiliary weight
reconstruction, logit matching, ranking, KL-divergence, smoothness,
latent-MSE, or distillation losses.

**13.3 **The polynomial basis constraint itself shall provide the
initial smoothness and capacity restriction.

**13.4 **The optimizer shall update the compact sheet coefficients
directly.

**13.5 **The default optimizer shall remain compatible with nanoGPT
AdamW training.

**13.6 **Weight decay treatment shall preserve the conventional
distinction between matrix-like weights and layer-normalisation or bias
parameters, even when all are stored as sheet coefficients.

**13.7 **Gradient clipping, learning-rate warm-up, and learning-rate
decay shall remain configurable through the training path.

**13.8 **The architecture shall support single-process GPU training.

**13.9 **The architecture shall support DistributedDataParallel
training.

**13.10 **The architecture shall support the mixed-precision modes
supported by the retained nanoGPT training path.

**13.11 **Completed optimizer updates shall be counted exactly and
consistently across dense and sheet runs.

**13.12 **The data-sampling random-number stream used for controlled
comparisons shall be independent of model-construction random-number
consumption.

# 14. Checkpoint, resume, and inference

**14.1 **A Chebyshev Sheet checkpoint shall identify its model type as
thog2_sheet.

**14.2 **A checkpoint shall store compact sheet coefficients and
conventional non-sheet model parameters.

**14.3 **A checkpoint shall not store a persistent full dense logical
block stack.

**14.4 **A checkpoint shall store all model arguments needed to
reconstruct L, d, P, Q_d, width-scaling rules, bias state, and
execution-compatible configuration.

**14.5 **Fixed basis matrices should be regenerated from checkpoint
configuration rather than stored as persistent state.

**14.6 **Resume shall restore model state, optimizer state, update
count, scheduler state, and random state required by the retained
training workflow.

**14.7 **Resume shall fail clearly when requested sheet geometry is
incompatible with the checkpoint geometry.

**14.8 **Execution-only activation-checkpoint settings may be changed on
resume where doing so does not alter model state.

**14.9 **A compact Chebyshev Sheet checkpoint shall be loadable directly
for text generation without conversion to a dense checkpoint.

**14.10 **Inference shall reconstruct the model from checkpoint metadata
and load compact coefficient state.

**14.11 **The dense nanoGPT path shall continue to load compatible dense
nanoGPT checkpoints independently of sheet checkpoints.

# 15. Configuration, naming, and user controls

**15.1 **The codebase shall expose at least two architecture selections:
dense and thog2_sheet.

**15.2 **The dense architecture shall remain available without
allocating sheet parameters.

**15.3 **The sheet architecture shall expose a configurable depth basis
term count P.

**15.4 **The sheet architecture shall expose a configurable base row
basis term count Q_d for rows of width d.

**15.5 **The sheet architecture shall derive other row basis counts by
the documented proportional width rule unless an explicit validated
override is provided.

**15.6 **Run names shall identify the architecture, L, d, P, and Q_d
unambiguously.

**15.7 **Checkpoint metadata and experiment telemetry shall record the
derived Q_f value for every generated tensor-family width.

**15.8 **Startup output shall report persistent parameters,
dense-equivalent parameters, sheet coefficient parameters, and
per-family sheet geometry.

**15.9 **Invalid geometry shall be rejected before model training
begins.

**15.10 **The command-line and wrapper design shall be specified in the
later implementation plan rather than assumed from the existing thog
wrapper.

# 16. Telemetry and diagnostics

**16.1 **Training telemetry shall include training loss and validation
loss using the retained next-token cross-entropy definition.

**16.2 **Telemetry shall include completed optimizer updates, consumed
training tokens, elapsed training time, and throughput.

**16.3 **Telemetry shall include peak allocated and peak reserved GPU
memory at documented phases.

**16.4 **Telemetry shall include coefficient RMS and gradient norm by
generated tensor family.

**16.5 **Telemetry shall include sampled generated-weight RMS, standard
deviation, and maximum absolute value by tensor family.

**16.6 **Telemetry shall include depth-basis energy by order or a
documented aggregate that reveals whether high depthwise terms are used.

**16.7 **Telemetry shall include row-basis energy by order or a
documented aggregate that reveals whether high rowwise terms are used.

**16.8 **Telemetry shall include a high-order energy fraction for each
axis using a documented cutoff.

**16.9 **Telemetry should include sampled vertical and horizontal
finite-difference roughness measures on generated sheets.

**16.10 **Telemetry should include numerical-conditioning diagnostics
for each distinct basis matrix used by a run.

**16.11 **Expensive diagnostics shall run at a configurable logging
cadence rather than on every micro-step.

**16.12 **Telemetry computations shall be detached and shall not alter
model gradients or optimizer state.

# 17. Functional acceptance requirements

**17.1 **The retained dense nanoGPT path shall pass its pre-existing
functional tests after sheet support is added.

**17.2 **For every supported matrix family, generated tensor shapes
shall match the corresponding dense nanoGPT tensor shapes exactly.

**17.3 **A reference materialisation test shall match direct evaluation
of B_depth, A_f, and B_row_f within the documented numerical tolerance.

**17.4 **A constant coefficient configuration shall generate the
expected constant sheet exactly within floating-point tolerance.

**17.5 **At P equal to L and Q_f equal to C_f for a small test geometry,
the parameterisation shall be capable of reproducing an arbitrary
sampled L-by-C_f row sheet within documented numerical tolerance.

**17.6 **Every trainable sheet coefficient tensor used by a
representative forward pass shall receive a finite gradient during
backward computation.

**17.7 **Reference and activation-checkpointed sheet execution shall
agree within a documented tolerance under deterministic conditions.

**17.8 **Checkpoint save and reload shall reproduce model state and
deterministic outputs within a documented tolerance.

**17.9 **Resume shall continue training without loss of optimizer-group
or scheduler semantics.

**17.10 **Single-process and DDP smoke training shall complete without
deadlock, missing gradients, or rank-dependent model-state divergence
attributable to sheet generation.

**17.11 **Compact-checkpoint inference shall generate text without first
materialising or saving a dense model checkpoint.

**17.12 **The implementation shall demonstrate that the complete dense
logical block stack is absent from persistent model state and checkpoint
state.

**17.13 **The complete regression suite defined by the later test plan
shall pass before implementation acceptance.

# 18. Scientific evaluation requirements

**18.1 **Functional acceptance shall not be interpreted as evidence that
the horizontal smoothness hypothesis is correct.

**18.2 **The first controlled scientific comparison shall include a
matched dense nanoGPT model and at least one thog2_sheet model with the
same logical L, d, attention-head count, context length, dataset, and
training-token budget.

**18.3 **The principal sheet configuration shall use P equal to 16 and
Q_d equal to 128 for d equal to 768.

**18.4 **The first row-capacity sweep should include Q_d equal to 64,
128, and 256, with proportionally scaled values for 4d-wide rows.

**18.5 **Compared runs shall consume an identical or provably equivalent
training-batch trace.

**18.6 **Compared runs shall use a fixed shared validation sample and a
materially adequate validation size.

**18.7 **The primary quality comparison shall use validation
cross-entropy at equal completed optimizer updates and equal consumed
training tokens.

**18.8 **The comparison shall also report validation loss against
wall-clock time, throughput, peak memory, and persistent parameters.

**18.9 **Existing THOGv1 results may be used as an external reference
but shall not create a runtime or repository dependency for thog2.

**18.10 **A promising result shall require stable optimization and
reproducible quality consistent with the claimed compression advantage.

**18.11 **Failure of a low-Q_d configuration shall not by itself falsify
the sheet concept unless higher-capacity configurations show that useful
performance requires Q_d approaching the full row width.

# 19. Explicit non-goals for the first implementation

**19.1 **The first implementation shall not integrate into or modify the
existing thog repository.

**19.2 **The first implementation shall not incorporate Vermeer or
another layer-specific residual correction.

**19.3 **The first implementation shall not reconstruct or imitate a
separately trained dense model.

**19.4 **The first implementation shall not use a teacher model or
knowledge distillation.

**19.5 **The first implementation shall not use learned within-row
addresses, learned row permutations, or coordinate warping.

**19.6 **The first implementation shall not use a three-dimensional
depth-by-output-by-input Chebyshev volume.

**19.7 **The first implementation shall not flatten a complete matrix
into one sheet spanning both output rows and input columns.

**19.8 **The first implementation shall not claim that interweight
samples correspond to meaningful intermediate latent coordinates.

**19.9 **The first implementation shall not change model depth or width
at inference by resampling the continuous sheet.

**19.10 **The first implementation shall not require custom fused
kernels before correctness, training, checkpoint, and inference paths
are complete.

**19.11 **The first implementation shall not claim mechanistic
interpretability from sheet coefficients or basis orders alone.

# 20. Known risks and open questions

**20.1 **The fixed left-to-right within-row coordinate may impose a
geometry that the model cannot exploit effectively.

**20.2 **Training may fail to organise latent coordinates into a form
compatible with rowwise smoothness.

**20.3 **The chosen Q_d may be too small to represent useful rowwise
profiles or unnecessarily large to deliver the intended compression.

**20.4 **High rowwise Chebyshev orders may create correlated gradients
or numerical-conditioning problems even when basis values remain
bounded.

**20.5 **Sheet materialisation may reduce parameter memory while adding
enough computation to harm throughput materially.

**20.6 **Optimizer state for sheet coefficients and conventional
embeddings may remain a major memory cost even when logical block
weights are highly compressed.

**20.7 **A single proportional Q_f rule may not be equally suitable for
all tensor families despite using one universal mathematical mechanism.

**20.8 **Representing layer-normalisation and bias vectors as sheets may
add complexity with negligible parameter benefit.

**20.9 **A fresh nanoGPT fork may diverge from improvements already made
in thog unless useful infrastructure is deliberately ported after
review.

**20.10 **Successful approximation of sampled weight sheets shall not
guarantee good language modelling; end-to-end controlled training
evidence is required.

**20.11 **Any response to these risks that changes the core
architectural hypothesis shall require a versioned specification
amendment rather than an undocumented implementation workaround.

# 21. Required follow-on documents and change control

**21.1 **A separate implementation plan shall define source files,
classes, configuration fields, initialization derivation, optimizer
integration, checkpoint schema, wrappers, telemetry, and staged
delivery.

**21.2 **A separate test plan shall define mathematical, unit,
integration, GPU, DDP, checkpoint, inference, regression, performance,
and controlled-comparison tests.

**21.3 **A separate staging plan shall divide implementation into the
minimum sensible independently testable stages.

**21.4 **An as-built document shall be produced after implementation and
shall identify every material departure from this specification.

**21.5 **This specification shall remain outside the existing thog
repository until the user authorises its addition to the future thog2
repository.

**21.6 **When added to thog2, the specification shall be committed as a
versioned project document rather than replacing its external source
without traceability.

**21.7 **Changes to the requirements shall increment the document
version and identify the modified requirement identifiers.

Non-normative note: End of normative requirements. The later
implementation plan may make engineering choices only within the
constraints above or through an explicit requirements revision.
