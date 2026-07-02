THOG2

Chebyshev Sheet Test Plan

Version 0.1

|                                |                                                                             |
|--------------------------------|-----------------------------------------------------------------------------|
| **System**                     | thog2 enhancement of nanoGPT                                                |
| **Architecture designation**   | Chebyshev Sheet                                                             |
| **Requirements specification** | THOG2 Chebyshev Sheet Requirements Specification v0.1                       |
| **Implementation plan**        | THOG2 Chebyshev Sheet Implementation Plan v0.1                              |
| **Staging plan**               | THOG2 Chebyshev Sheet Staging Plan v0.1                                     |
| **Status**                     | Controlled draft; planned tests may be revised or extended during execution |
| **Date**                       | 2 July 2026                                                                 |
| **Planned repository**         | thog2                                                                       |
| **Repository interaction**     | None; this document is authored outside GitHub pending repository creation  |

Non-normative note: This is a controlled but deliberately revisable
plan. Actual implementation and test execution may expose missing cases,
invalid assumptions, or better test methods. Tests may be added or
corrected through the change-control rules in Section 2; failures shall
not be hidden by silently weakening the plan.

# 1. Purpose and authority

**1.1 **This document defines the planned verification and validation of
the THOG2 Chebyshev Sheet architecture by implementation stage.

**1.2 **The governing functional requirements are contained in THOG2
Chebyshev Sheet Requirements Specification v0.1.

**1.3 **The planned technical design is contained in THOG2 Chebyshev
Sheet Implementation Plan v0.1.

**1.4 **Stage boundaries and merge gates are contained in THOG2
Chebyshev Sheet Staging Plan v0.1.

**1.5 **Where this Test Plan conflicts with the requirements
specification, the requirements specification shall prevail.

**1.6 **Where this Test Plan conflicts with the staging plan on the
minimum evidence required for a stage gate, the stricter requirement
shall apply until a versioned amendment resolves the conflict.

**1.7 **This document defines planned test purpose, execution class,
evidence, and acceptance logic; exact source paths and command strings
may be finalized after repository inspection.

**1.8 **Passing this plan shall establish functional implementation and
controlled evidence. It shall not establish that the Chebyshev Sheet
architecture is scientifically superior to dense nanoGPT.

# 2. Controlled evolution of the Test Plan

**2.1 **This Test Plan shall be treated as a controlled living document
rather than an immutable prediction of every test needed before
implementation exists.

**2.2 **Tests may be added at any time when implementation work, code
review, profiling, a defect, or a scientific result exposes a previously
untested risk.

**2.3 **A planned test may be refined when its original method is
ambiguous, technically invalid, redundant, or unable to test the stated
requirement.

**2.4 **A planned test shall not be deleted, disabled, skipped, or
weakened merely because the implementation fails it.

**2.5 **Where a test is found to be invalid, the stage record shall
identify the invalid assumption, retain the original test ID as
superseded, and identify the replacement test or the reason no
replacement is required.

**2.6 **Numerical tolerances may be tightened or revised after
calibration, but each change shall record the previous value, new value,
evidence, and whether the change makes acceptance easier or harder.

**2.7 **A tolerance shall not be relaxed after observing a failure
unless numerical analysis or independent reference evidence shows that
the earlier tolerance was unjustified.

**2.8 **Every reproducible architecture-attributable defect should
receive a regression test before the defect is closed, unless a written
reason shows that an automated test is impractical.

**2.9 **New tests added during a stage shall be included in that stage
completion report and in the next version or addendum of this Test Plan.

**2.10 **Test IDs shall remain stable. New tests should receive the next
unused ID within their stage rather than renumbering earlier tests.

**2.11 **A test-plan revision shall not retroactively convert a recorded
failure into a pass without preserving the original result and
rationale.

**2.12 **The accepted as-built document shall distinguish tests planned
in v0.1, tests added during implementation, tests revised, tests
superseded, and tests deferred.

# 3. Test principles and classifications

**3.1 **Deterministic tests shall be preferred for mathematical
correctness, state schema, shape contracts, optimizer grouping,
checkpoint compatibility, and dense-path regression.

**3.2 **Statistical tests shall be used only where the property is
intrinsically statistical, including initialization distributions and
stochastic training behavior.

**3.3 **Scientific training runs shall not substitute for deterministic
unit or integration tests.

**3.4 **A no-crash smoke test shall establish only operational
reachability and shall not be treated as proof of numerical correctness.

**3.5 **The tested reference implementation shall remain the comparison
oracle for later checkpointed, compiled, fused, or otherwise optimized
implementations.

**3.6 **Negative tests shall verify that invalid configurations and
incompatible checkpoints fail directly rather than being silently
reinterpreted.

**3.7 **Dense nanoGPT regression tests shall be run after every stage
that touches shared source, configuration, training, checkpoint, or
inference code.

|                          |                                                  |                                                                            |
|--------------------------|--------------------------------------------------|----------------------------------------------------------------------------|
| **Test class**           | **Typical execution**                            | **Required evidence**                                                      |
| Mathematical unit        | CPU, normally float64 and float32 variants       | Inputs, expected relation, tolerance, pass/fail, runtime where material.   |
| Model unit / integration | CPU reference model; tiny deterministic geometry | Shapes, values, gradients, state keys, optimizer coverage.                 |
| Single-GPU smoke         | One GPU, bounded updates                         | Command, hardware, dtype, geometry, updates, loss, memory, outcome.        |
| DDP smoke                | At least two ranks where available               | Launch command, rank count, synchronization checks, completion evidence.   |
| Source guard             | Static or structural repository inspection       | No prohibited state, source-marker compliance, path and pattern checked.   |
| Scientific evaluation    | Controlled matched runs                          | Exact configuration, batch trace, validation sample, metrics, limitations. |

# 4. Test environments, fixtures, and evidence

**4.1 **The CPU reference environment shall be capable of float64 basis
construction and deterministic small-model tests.

**4.2 **The primary GPU environment shall record GPU model, driver,
CUDA, PyTorch, operating system, compiler setting, and supported
mixed-precision dtypes.

**4.3 **DDP evidence shall record rank count, backend, device mapping,
launch method, and relevant environment variables.

**4.4 **Small deterministic fixtures shall use fixed seeds and tiny
geometries chosen so that direct evaluation is inexpensive and
inspectable.

**4.5 **Dataset-dependent tests shall identify the dataset snapshot or
preparation command and shall avoid committing large data files.

**4.6 **Checkpoint fixtures committed to the repository shall be small,
stable, and justified; large checkpoints shall remain external
artifacts.

**4.7 **Each recorded test execution shall identify the exact commit,
branch, command, environment, seed, model geometry, dtype, and result.

**4.8 **A stage pull request shall report mandatory tests, optional
diagnostics, skipped tests, skip reasons, failures, and reruns.

**4.9 **A rerun after failure shall not erase the initial failure; the
report shall identify the cause and corrective change.

# 5. Numerical tolerance framework

**5.1 **Tests with exact discrete expectations shall use exact equality,
including shapes, counts, configuration derivation, exact zero
initialization, exact LayerNorm-one generation, exact bias-zero
generation, and state-key presence or absence.

**5.2 **Float64 mathematical tests shall initially use strict tolerances
appropriate to QR and matrix multiplication on the accepted CPU
platform.

**5.3 **Float32 reference tests shall use looser but still diagnostic
tolerances and shall not inherit mixed-precision tolerances by default.

**5.4 **Mixed-precision equivalence shall be evaluated against a float32
reference and shall use operation-specific tolerances fixed after Stage
3 or Stage 4 calibration.

**5.5 **Initialization distribution tests shall compare measured
statistics with statistics predicted from the actual sampled basis
leverage rather than using an arbitrary fixed percentage alone.

**5.6 **Reference-versus-checkpointed tests shall compare both outputs
and gradients under preserved RNG state.

**5.7 **Scientific validation loss shall be compared at equal completed
optimizer updates and equal consumed tokens; numerical equality is not
expected between architectures.

|                                     |                                                                                  |                                                                                  |
|-------------------------------------|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Property**                        | **Initial rule**                                                                 | **Revision rule**                                                                |
| Exact structural properties         | Exact equality.                                                                  | May not be relaxed unless the governing requirement changes.                     |
| Float64 basis and direct evaluation | Use a strict allclose threshold established by Stage 1 calibration.              | Record platform study and preserve span/rank requirements.                       |
| Float32 materialisation             | Use a strict reference allclose threshold established from small exact cases.    | Change only with independent error analysis.                                     |
| Mixed precision                     | Provisional until GPU calibration.                                               | Fix separately by dtype and operation before stage acceptance.                   |
| Initialization statistics           | Expected-value and leverage-aware statistical bounds.                            | Revise sample size or confidence method, not merely widen after failure.         |
| Training loss                       | Finite and directionally improving for smokes; controlled comparison for pilots. | Budget and decision rules may be amended before runs, not after seeing outcomes. |

# 6. Stage 0 tests - Baseline and documentation

## 6.1 Stage objective

**6.1.1 **Stage 0 tests shall establish that thog2 begins from an
identifiable, reproducible nanoGPT baseline and that documentation
changes do not alter runtime behavior.

## 6.2 Planned tests

|        |                                 |                                                                                                                  |                                                                   |
|--------|---------------------------------|------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| **ID** | **Test**                        | **Method / evidence**                                                                                            | **Acceptance**                                                    |
| S0-01  | Upstream identity               | Record upstream URL, default branch, and exact commit SHA; compare repository tree with the selected baseline.   | Baseline is identifiable and unexplained differences are absent.  |
| S0-02  | Clean repository state          | Run status before and after baseline commands; inspect ignore rules for generated artifacts.                     | Working tree is clean except documented ignored artifacts.        |
| S0-03  | Dense CPU import and forward    | Import model, instantiate a tiny dense configuration, and run a deterministic forward pass.                      | Construction and forward complete with finite logits and loss.    |
| S0-04  | Dense backward and update       | Run one tiny backward and optimizer update on CPU or the baseline-supported device.                              | Gradients and updated parameters are finite.                      |
| S0-05  | Dense single-GPU smoke          | Run the agreed short baseline training command when GPU access exists.                                           | Expected updates, validation, and checkpoint behavior complete.   |
| S0-06  | Documentation-only change guard | Compare runtime source tree before and after adding governing documents.                                         | No runtime source file changes in Stage 0.                        |
| S0-07  | ODT document open/render        | Open or headlessly render each governing ODT.                                                                    | Documents are readable, complete, and free of visible corruption. |
| S0-08  | Artifact hygiene                | Create representative log/checkpoint/output paths and verify ignore behavior without committing large artifacts. | Repository remains clean and required fixtures remain trackable.  |

## 6.3 Stage acceptance

**6.3.1 **All mandatory Stage 0 tests shall pass before mathematical
implementation begins.

**6.3.2 **S0-05 may be deferred only where GPU access is genuinely
unavailable and the staging decision records that limitation.

# 7. Stage 1 tests - Mathematical core

## 7.1 Stage objective

**7.1.1 **Stage 1 tests shall establish the numerical validity,
determinism, shape contracts, and practical constructibility of the
depth-row Chebyshev bases and geometry helpers.

## 7.2 Planned tests

|        |                                |                                                                                                     |                                                                                                       |
|--------|--------------------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| **ID** | **Test**                       | **Method / evidence**                                                                               | **Acceptance**                                                                                        |
| S1-01  | Coordinate endpoints           | Construct depth and row coordinates for representative lengths including 1, 2, 144, 768, and 3072.  | Correct count; endpoints are -1 and +1 where length exceeds 1; single-point convention is documented. |
| S1-02  | Coordinate monotonicity        | Check strict left-to-right and bottom-to-top ordering for lengths greater than one.                 | Coordinates are finite, ordered, and deterministic.                                                   |
| S1-03  | Chebyshev recurrence reference | Compare recurrence-generated terms with a trusted direct recurrence or analytic small-order values. | Agreement within calibrated float64 tolerance.                                                        |
| S1-04  | Raw basis shape and finiteness | Construct representative L x P and C x Q raw bases.                                                 | Expected shape; no NaN or infinity.                                                                   |
| S1-05  | QR orthonormality              | Check transpose(Q) @ Q against identity in construction and runtime dtypes.                         | Within calibrated dtype-specific tolerance.                                                           |
| S1-06  | Full column rank               | Measure matrix rank before and after QR for all supported geometries.                               | Rank equals requested order.                                                                          |
| S1-07  | Discrete-span preservation     | Project random vectors through raw and stabilized bases or compare projection matrices.             | The stabilized basis spans the same sampled column space within tolerance.                            |

## 7.2 Planned tests (continued)

|        |                             |                                                                                                                                      |                                                                                                                      |
|--------|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| **ID** | **Test**                    | **Method / evidence**                                                                                                                | **Acceptance**                                                                                                       |
| S1-08  | Deterministic column signs  | Reconstruct identical bases repeatedly and across process restarts.                                                                  | Basis is bitwise identical where supported, otherwise numerically identical under the documented deterministic rule. |
| S1-09  | Basis cache identity        | Request repeated bases with identical and differing cache keys.                                                                      | Identical keys reuse compatible basis state; differing geometry, version, device, or dtype cannot collide.           |
| S1-10  | Row-order derivation        | Evaluate Q_f = min(C_f, ceil(C_f \* Q_d / d)) across normal and boundary geometries.                                                 | Expected values, including 128 for width 768 and 512 for width 3072 at d=768/Q_d=128.                                |
| S1-11  | Configuration rejection     | Supply P\>L, Q_d\>d, non-positive dimensions, invalid head geometry, and invalid derived orders.                                     | Each invalid case fails directly with a specific diagnostic.                                                         |
| S1-12  | High-order construction     | Construct row bases at Q=128, 256, 512, and at least 1024 with representative widths.                                                | Finite, full-rank, deterministic, and operationally acceptable time/memory.                                          |
| S1-13  | Family geometry             | Derive rows, row widths, and Q_f for packed attention input, attention output, MLP expansion, MLP contraction, and selected vectors. | Matches architecture formulas for all families.                                                                      |
| S1-14  | Analytical parameter counts | Compare formulas with explicit products of planned coefficient shapes.                                                               | Exact agreement for tiny, principal, and boundary geometries.                                                        |
| S1-15  | Non-persistent basis schema | Instantiate basis owner and inspect parameters and state_dict.                                                                       | Fixed bases are not trainable and are absent from persistent checkpoint state.                                       |

## 7.3 Calibration study

**7.3.1 **Stage 1 shall record construction error, runtime, and
temporary memory by Q value and dtype before fixing final mathematical
tolerances.

**7.3.2 **Any replacement for reduced QR shall rerun every Stage 1
mathematical test and demonstrate equivalent sampled span and
deterministic reconstruction.

## 7.4 Stage acceptance

**7.4.1 **Every mandatory Stage 1 test shall pass on CPU, including Q
equal to 1024 construction, before model-state implementation begins.

# 8. Stage 2 tests - Compact state and reference model

## 8.1 Stage objective

**8.1.1 **Stage 2 tests shall prove that compact coefficients generate
conventional transformer weights correctly, train by ordinary
cross-entropy, and do not persist a dense logical block stack.

## 8.2 Planned tests

|        |                                |                                                                                                       |                                                                                                        |
|--------|--------------------------------|-------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **ID** | **Test**                       | **Method / evidence**                                                                                 | **Acceptance**                                                                                         |
| S2-01  | Coefficient tensor shapes      | Instantiate all families at tiny and principal geometries and inspect trainable shapes.               | Each family has shape R_f x P x Q_f and exact expected count.                                          |
| S2-02  | One-layer materialisation      | Compare implementation output with explicit depth mixing followed by row-basis matrix multiplication. | Generated values and shapes agree within calibrated reference tolerance.                               |
| S2-03  | Direct point evaluation        | For tiny geometries, evaluate selected sheet points directly from basis values and coefficients.      | Selected scalar weights agree with materialised matrices.                                              |
| S2-04  | Saturated sampled completeness | Use P=L and Q_f=C_f on small geometries; solve or assign coefficients for arbitrary sampled sheets.   | Arbitrary sampled sheet is reconstructed within tolerance.                                             |
| S2-05  | Conventional matrix shapes     | Materialise each family for one layer.                                                                | Packed attention, output projection, MLP expansion, and contraction shapes match nanoGPT expectations. |
| S2-06  | Family isolation               | Perturb one coefficient family while holding others fixed.                                            | Only the intended generated family changes.                                                            |
| S2-07  | Gradient reachability          | Backpropagate a deterministic scalar loss through each active family.                                 | Every required coefficient tensor receives finite non-zero gradients.                                  |
| S2-08  | Gradient reference             | Compare coefficient gradients with a small direct dense-sheet reference computation.                  | Gradients agree within calibrated tolerance.                                                           |

## 8.2 Planned tests (continued)

|        |                                |                                                                                                                          |                                                                                                                     |
|--------|--------------------------------|--------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| **ID** | **Test**                       | **Method / evidence**                                                                                                    | **Acceptance**                                                                                                      |
| S2-09  | Initialization exact structure | Inspect zeroed higher depth modes, LayerNorm-one generation, and bias-zero generation.                                   | Exact required values and mode structure.                                                                           |
| S2-10  | Initialization statistics      | Sample generated matrices across families, layers, rows, and endpoints; compare with leverage-aware expected statistics. | Means, RMS, standard deviations, and endpoint behavior satisfy calibrated bounds.                                   |
| S2-11  | Shared-depth initialization    | Materialise several layers immediately after initialization.                                                             | Generated matrix families intended to be depth-shared are equal across layers within exact or documented tolerance. |
| S2-12  | Optimizer coverage             | Enumerate trainable parameter IDs across semantic optimizer groups.                                                      | Every trainable parameter appears exactly once; no fixed basis appears.                                             |
| S2-13  | Weight-decay classification    | Inspect matrix, LayerNorm, and bias family group assignments.                                                            | Assignments follow semantic policy rather than accidental tensor rank.                                              |
| S2-14  | Compact state guard            | Inspect named_parameters, buffers, and state_dict; scan for L x R_f x C_f persistent tensors.                            | No persistent dense logical block stack exists.                                                                     |
| S2-15  | Tiny model forward             | Run deterministic SheetGPT forward with and without targets.                                                             | Logit and loss shapes are correct and finite.                                                                       |
| S2-16  | Tiny model backward/update     | Run backward, clipping where configured, and one optimizer step.                                                         | Finite gradients, state, and updated coefficients.                                                                  |
| S2-17  | Short CPU learning smoke       | Train a tiny model for several completed updates on a fixed tiny batch or fixture.                                       | Loss remains finite and demonstrates a learning signal rather than immediate divergence.                            |
| S2-18  | Dense path regression          | Run affected dense model tests and compare fixed-seed tiny outputs to Stage 0 evidence where applicable.                 | Dense behavior remains accepted.                                                                                    |

## 8.3 Stage acceptance

**8.3.1 **All Stage 2 compact-state, direct-equation, saturated-basis,
gradient, initialization, optimizer, model, and dense-regression tests
shall pass.

**8.3.2 **A short CPU training smoke shall complete without allocating
persistent dense block weights.

# 9. Stage 3 tests - Training lifecycle

## 9.1 Stage objective

**9.1.1 **Stage 3 tests shall establish shared trainer integration,
deterministic controls, compact checkpoints, exact completed-update
semantics, and single-GPU reference training.

## 9.2 Planned tests

|        |                            |                                                                                                                |                                                                                                                                                    |
|--------|----------------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| **ID** | **Test**                   | **Method / evidence**                                                                                          | **Acceptance**                                                                                                                                     |
| S3-01  | Model selection            | Select dense and thog2_sheet through supported configuration paths.                                            | Correct model is constructed; invalid names fail directly.                                                                                         |
| S3-02  | Shared-loop parity         | Instrument common data, accumulation, scheduler, clipping, evaluation, and logging paths for both model types. | No architecture-specific divergence exists except documented requirements.                                                                         |
| S3-03  | Completed-update semantics | Test boundaries around update zero, gradient accumulation, maximum updates, evaluation cadence, and resume.    | Reported update count equals completed optimizer steps exactly.                                                                                    |
| S3-04  | Data RNG separation        | Construct models with different RNG consumption but the same data seed; record batch indices.                  | Batch traces are identical.                                                                                                                        |
| S3-05  | Checkpoint schema          | Inspect saved checkpoint keys and model arguments.                                                             | Compact coefficients, basis version, scaling rule, optimizer state, update state, and required RNG state are present; dense block stack is absent. |
| S3-06  | Checkpoint round trip      | Save and reload a fixed model; compare state and fixed-batch outputs.                                          | State matches and outputs agree within reference tolerance.                                                                                        |
| S3-07  | Resume optimizer state     | Train, save, resume, and compare optimizer moments, group order, learning rate, and next update.               | Resume continues from the same optimization state.                                                                                                 |

## 9.2 Planned tests (continued)

|        |                                   |                                                                                                      |                                                                                                         |
|--------|-----------------------------------|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **ID** | **Test**                          | **Method / evidence**                                                                                | **Acceptance**                                                                                          |
| S3-08  | Resume update and scheduler       | Check warmup, decay position, best/latest validation state, and completed updates across resume.     | All lifecycle counters and scheduler values are restored correctly.                                     |
| S3-09  | Incompatible checkpoint rejection | Alter model type, L, d, P, Q_d, basis version, or scaling rule before load.                          | Load fails directly with a diagnostic naming the incompatibility.                                       |
| S3-10  | Legacy dense checkpoint behavior  | Load baseline-supported dense checkpoints through the dense path.                                    | Existing dense behavior remains supported or any deliberate incompatibility is documented before merge. |
| S3-11  | Single-GPU reference smoke        | Run bounded sheet training without activation checkpointing, including validation, save, and resume. | Finite loss and gradients; expected updates; checkpoint and resume complete.                            |
| S3-12  | Mixed-precision reference smoke   | Run supported float32 and at least one supported mixed-precision dtype at a small geometry.          | No architecture-attributable non-finite values; dtype-specific evidence recorded.                       |
| S3-13  | Parameter reporting               | Compare startup counts with analytical counts and state_dict totals.                                 | Persistent, sheet, conventional, and dense-equivalent counts reconcile exactly.                         |
| S3-14  | Dense trainer regression          | Run dense training smoke and relevant lifecycle tests after trainer changes.                         | Dense path remains accepted.                                                                            |

## 9.3 Stage acceptance

**9.3.1 **Dense and sheet models shall both train through the shared
trainer.

**9.3.2 **Compact checkpoint save, load, and resume shall pass all state
and output checks.

**9.3.3 **At least one single-GPU sheet reference smoke shall complete
before Stage 4.

# 10. Stage 4 tests - Ephemeral execution and inference

## 10.1 Stage objective

**10.1.1 **Stage 4 tests shall establish numerical equivalence of
checkpointed ephemeral execution, bounded materialisation lifetime, and
direct text generation from compact checkpoints.

## 10.2 Planned tests

|        |                                  |                                                                                                                           |                                                                                                 |
|--------|----------------------------------|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| **ID** | **Test**                         | **Method / evidence**                                                                                                     | **Acceptance**                                                                                  |
| S4-01  | Reference/checkpointed forward   | Run identical fixed input, model state, RNG, and dtype with checkpointing off and on.                                     | Logits and loss agree within calibrated dtype-specific tolerance.                               |
| S4-02  | Reference/checkpointed gradients | Compare coefficient gradients after the same backward pass.                                                               | Required gradients agree within calibrated tolerance.                                           |
| S4-03  | Dropout RNG preservation         | Enable dropout and compare reference versus checkpointed runs under fixed RNG state.                                      | Outputs and gradients satisfy equivalence; no double-advance defect.                            |
| S4-04  | Segment-size coverage            | Test disabled, one-layer, several-layer, exact-divisor, and non-divisor segment sizes.                                    | All valid settings execute correctly; invalid settings fail directly.                           |
| S4-05  | Evaluation bypass                | Instrument evaluation and inference execution.                                                                            | Training-only recomputation is not used where unnecessary.                                      |
| S4-06  | Ephemeral lifetime guard         | Inspect retained graph/state and memory phases around logical-layer execution.                                            | Generated dense matrices are not persisted across the full stack or stored in checkpoint state. |
| S4-07  | Memory phase telemetry           | Record model construction, optimizer allocation, first optimizer state, steady update, evaluation, and checkpoint phases. | Metrics are finite, labeled, and internally consistent.                                         |

## 10.2 Planned tests (continued)

|        |                                  |                                                                                                           |                                                                                          |
|--------|----------------------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **ID** | **Test**                         | **Method / evidence**                                                                                     | **Acceptance**                                                                           |
| S4-08  | Compact inference reconstruction | Load a compact sheet checkpoint using only recorded model arguments and coefficient state.                | Correct SheetGPT geometry and state are reconstructed without dense conversion.          |
| S4-09  | Deterministic generation         | Generate with fixed prompt, seed, temperature, and top-k twice.                                           | Output is reproducible under deterministic settings.                                     |
| S4-10  | Inference control coverage       | Exercise inline prompt, prompt file, sample count, continuation length, temperature, top-k, and seed.     | Each supported control is honored and invalid values fail clearly.                       |
| S4-11  | Diagnostic finiteness            | Collect coefficient order RMS, gradient norms, generated-weight statistics, and endpoint/interior ratios. | All required diagnostics are finite and do not alter gradients.                          |
| S4-12  | Naming and artifact identity     | Create dense and sheet run names and output paths across representative geometries.                       | Architecture and sheet orders are unambiguous; collisions are absent.                    |
| S4-13  | Checkpoint size/schema guard     | Compare compact checkpoint contents with expected coefficient counts.                                     | No generated per-layer dense matrices are stored; size is consistent with compact state. |
| S4-14  | Dense inference regression       | Run existing dense text-generation path after shared inference changes.                                   | Dense inference remains accepted.                                                        |

## 10.3 Stage acceptance

**10.3.1 **Reference and checkpointed outputs and gradients shall agree
within fixed tolerances for every required dtype and segment case.

**10.3.2 **A compact sheet checkpoint shall generate text directly
without dense conversion.

**10.3.3 **Memory evidence shall show that dense logical-layer weights
remain ephemeral.

# 11. Stage 5 tests - GPU, DDP, and target geometry

## 11.1 Stage objective

**11.1.1 **Stage 5 tests shall establish target-scale numerical
stability, bounded memory, DDP correctness, and sufficient performance
evidence for the controlled pilot.

## 11.2 Planned tests

|        |                                 |                                                                                                                         |                                                                                                         |
|--------|---------------------------------|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **ID** | **Test**                        | **Method / evidence**                                                                                                   | **Acceptance**                                                                                          |
| S5-01  | Principal geometry construction | Construct L144/d768/P16/Q_d128/Q_4d512 model and optimizer on the target GPU.                                           | Construction completes with reconciled counts and finite state.                                         |
| S5-02  | Principal forward/backward      | Run one full update at the target geometry using the accepted execution path.                                           | Finite logits, loss, gradients, and optimizer state.                                                    |
| S5-03  | Bounded target smoke            | Run the agreed number of completed updates with validation and checkpoint.                                              | No OOM, non-finite state, or retained dense stack; loss shows a learning signal.                        |
| S5-04  | GPU memory accounting           | Record allocated and reserved memory by phase and rank.                                                                 | Peak values are finite, reproducible enough for planning, and consistent with compact-state objectives. |
| S5-05  | Basis construction profiling    | Measure construction time and temporary memory for depth, Q128, Q512, and optional Q1024 bases.                         | Cost is acceptable or a documented implementation revision is triggered.                                |
| S5-06  | Materialisation profiling       | Measure time spent in basis mixing, generated matrix construction, attention/MLP compute, and checkpoint recomputation. | Profile is complete enough to identify actual bottlenecks; no unsupported performance claim is made.    |
| S5-07  | Supported dtype matrix          | Run principal or reduced geometry in every required GPU dtype.                                                          | Required dtypes complete without architecture-attributable non-finite behavior.                         |

## 11.2 Planned tests (continued)

|        |                                    |                                                                                                                   |                                                                                          |
|--------|------------------------------------|-------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **ID** | **Test**                           | **Method / evidence**                                                                                             | **Acceptance**                                                                           |
| S5-08  | DDP construction identity          | Construct on at least two ranks and compare parameter names, shapes, group order, and initial synchronized state. | All ranks construct identical trainable structure.                                       |
| S5-09  | DDP update smoke                   | Run bounded multi-rank training including accumulation, validation boundary, and checkpoint behavior.             | No deadlock; synchronized updates complete; rank-zero duties are correct.                |
| S5-10  | DDP state consistency              | Compare selected coefficients and optimizer metadata across ranks after deterministic updates.                    | State remains synchronized within expected distributed numerical behavior.               |
| S5-11  | DDP resume                         | Resume a multi-rank checkpoint and continue updates.                                                              | Group reconstruction, counters, scheduler, and synchronized training continue correctly. |
| S5-12  | Uneven segment and boundary stress | Exercise layer counts or segment sizes that create a shorter final checkpoint segment.                            | Correct execution and gradients at the boundary.                                         |
| S5-13  | Longer stability smoke             | Run enough updates to allocate steady-state optimizer memory and expose delayed numerical faults.                 | No recurrent non-finite values, leakage, or checkpoint corruption.                       |
| S5-14  | Full regression suite              | Run all mandatory tests from Stages 0-5 on the proposed merge commit, using environment-appropriate partitioning. | All mandatory tests pass; skips are explicitly justified.                                |

## 11.3 Stage acceptance

**11.3.1 **The principal geometry shall complete the accepted bounded
single-GPU run.

**11.3.2 **At least one accepted multi-rank DDP run shall complete
without deadlock or state divergence attributable to the architecture.

**11.3.3 **Memory and performance evidence shall be sufficient to set
the Stage 6 pilot configuration and budget.

# 12. Stage 6 tests - Controlled pilot and as-built

## 12.1 Stage objective

**12.1.1 **Stage 6 tests shall produce the first controlled scientific
evidence for the rowwise-sheet hypothesis and close the initial
implementation with a complete as-built record.

## 12.2 Control validation before training

|        |                                     |                                                                                                           |                                                                                             |
|--------|-------------------------------------|-----------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| **ID** | **Test**                            | **Method / evidence**                                                                                     | **Acceptance**                                                                              |
| S6-01  | Matched logical geometry            | Compare dense and sheet model configuration excluding architecture-required parameterization differences. | L, d, heads, context, dataset, tokenizer, and other agreed logical settings match.          |
| S6-02  | Matched batch trace                 | Record and compare training batch indices or equivalent deterministic sampler trace.                      | Compared runs consume the same examples in the same order.                                  |
| S6-03  | Matched validation sample           | Freeze and identify the validation indices used by all compared runs.                                     | Validation sample is identical and materially sized for the pilot.                          |
| S6-04  | Matched optimization budget         | Compare batch size, accumulation, optimizer, scheduler, clipping, updates, and consumed tokens.           | All non-architectural controls match or differences are explicitly justified before launch. |
| S6-05  | Run identity and artifact isolation | Dry-run naming and output paths for dense, Q64, Q128, and Q256 configurations.                            | No collision; exact configuration is encoded or recorded.                                   |

## 12.3 Pilot execution and analysis

|        |                         |                                                                                                                  |                                                                                                                          |
|--------|-------------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **ID** | **Test**                | **Method / evidence**                                                                                            | **Acceptance**                                                                                                           |
| S6-06  | Principal Q128 pilot    | Run P16/Q_d128/Q_4d512 against the matched dense control.                                                        | Run completes or stops under a predefined failure rule; all metrics and artifacts are preserved.                         |
| S6-07  | Row-capacity bracket    | If justified by the principal result, run Q_d64 and Q_d256 with proportional 4d orders.                          | Capacity trend is measurable or the reason the sweep was stopped is recorded.                                            |
| S6-08  | Equal-update comparison | Plot and tabulate validation loss at equal completed optimizer updates and consumed tokens.                      | Comparison uses aligned points and identifies missing or invalid intervals.                                              |
| S6-09  | Equal-time comparison   | Compare validation loss against clean wall-clock training time.                                                  | Timing definition excludes or separately reports setup, evaluation, and checkpoint overhead.                             |
| S6-10  | Resource comparison     | Compare persistent parameters, optimizer state, peak allocated/reserved VRAM, throughput, and checkpoint size.   | Metrics use consistent definitions and hardware.                                                                         |
| S6-11  | Coefficient utilization | Analyze RMS or energy by depth and row order, gradient norms, and endpoint/interior generated-weight statistics. | Analysis identifies unused capacity, high-order dependence, collapse, or instability without claiming mechanistic proof. |

## 12.3 Pilot execution and analysis (continued)

|        |                                      |                                                                                          |                                                                                                                                                     |
|--------|--------------------------------------|------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| **ID** | **Test**                             | **Method / evidence**                                                                    | **Acceptance**                                                                                                                                      |
| S6-12  | Repeatability decision               | Assess whether observed results require an additional seed or rerun before a conclusion. | Decision and budget are recorded before selective rerunning.                                                                                        |
| S6-13  | Final regression                     | Run the complete accepted regression suite on the final implementation commit.           | All mandatory functional tests pass.                                                                                                                |
| S6-14  | As-built traceability                | Map requirements and stage gates to implemented paths and accepted tests.                | Every implemented requirement has evidence or a documented deviation.                                                                               |
| S6-15  | Scientific conclusion classification | Apply staging-plan decision rules to the controlled evidence.                            | Outcome is explicitly classified as viable for further study, viable only at weak compression, inconclusive, or not viable under the tested design. |

## 12.4 Scientific interpretation rules

**12.4.1 **A low-Q failure with systematic improvement as Q increases
shall be recorded as capacity evidence, not immediate falsification.

**12.4.2 **A requirement for Q_d near the full row width shall be
treated as evidence against useful rowwise compression.

**12.4.3 **A major parameter or VRAM reduction with a modest quality
penalty may be recorded as promising, but shall not be called superior
without a stated trade-off criterion.

**12.4.4 **A negative scientific result shall not invalidate a
functionally correct implementation.

**12.4.5 **Learned coordinates, row permutations, Vermeer, auxiliary
losses, or three-dimensional fields shall not be introduced into the
Stage 6 test matrix without a successor specification.

# 13. Cross-stage regression and execution matrix

**13.1 **Each stage shall run its new tests and all earlier mandatory
CPU tests before merge.

**13.2 **GPU and DDP tests may be partitioned by environment, but the
proposed merge commit shall have complete required evidence before the
corresponding stage gate is accepted.

**13.3 **A failure in an earlier-stage test blocks acceptance of a later
stage unless the test has been validly superseded under Section 2.

|           |                                                         |                                          |                                                      |
|-----------|---------------------------------------------------------|------------------------------------------|------------------------------------------------------|
| **Stage** | **Mandatory regression scope**                          | **Primary environments**                 | **Gate evidence**                                    |
| 0         | Baseline dense and document tests.                      | CPU; GPU where available.                | Reproducible clean baseline.                         |
| 1         | Stage 0 plus all mathematical tests.                    | CPU float64/float32.                     | Basis validity through Q1024.                        |
| 2         | Stages 0-1 plus compact model and dense regressions.    | CPU reference.                           | Correct materialisation and trainable compact state. |
| 3         | Stages 0-2 plus training/checkpoint lifecycle.          | CPU and single GPU.                      | Shared trainer and compact resume.                   |
| 4         | Stages 0-3 plus checkpointed execution and inference.   | CPU and single GPU.                      | Equivalence, memory telemetry, direct inference.     |
| 5         | All functional tests through Stage 5.                   | Single GPU and multi-rank DDP.           | Target geometry and distributed acceptance.          |
| 6         | Full functional regression plus controlled pilot tests. | Pilot hardware and analysis environment. | Scientific evidence and as-built traceability.       |

# 14. Defect, failure, and regression-test rules

**14.1 **A test failure shall first be classified as implementation
defect, test defect, environment defect, nondeterministic infrastructure
fault, or unresolved.

**14.2 **An implementation defect shall be corrected without weakening
the original acceptance condition unless the governing design changes.

**14.3 **A test defect shall preserve the original test ID as superseded
and shall include a replacement or written rationale.

**14.4 **An environment fault shall be rerun only after the environment
difference is recorded and corrected or explicitly accepted.

**14.5 **An intermittent failure shall not be treated as passed because
a rerun succeeds; reproducibility and flakiness shall be investigated
before stage acceptance.

**14.6 **A fixed defect should receive the smallest deterministic
regression test that would have detected it before the fix.

**14.7 **Where the defect is visible only at GPU or DDP scale, the
regression may be a bounded hardware test rather than a CPU unit test.

**14.8 **Performance regressions shall be tested only after a stable
measurement protocol is established; noisy timing thresholds shall not
block merges without evidence of a material change.

# 15. Stage test completion reports

**15.1 **Each stage completion report shall identify the accepted commit
and pull request.

**15.2 **The report shall list every mandatory test ID, exact command,
environment, result, and runtime where material.

**15.3 **The report shall list tests added during implementation and
explain the risk that motivated each addition.

**15.4 **The report shall identify revised or superseded tests and
preserve before-and-after acceptance logic.

**15.5 **The report shall state all skips and why they do not invalidate
the stage gate.

**15.6 **The report shall identify failures encountered, their
classification, corrective commits, and regression tests.

**15.7 **The report shall state whether every staging-plan exit gate is
satisfied and whether any stop condition was triggered.

**15.8 **Large raw logs may remain external, but the report shall
preserve enough summarized evidence to audit the decision.

# 16. Overall test completion criteria

**16.1 **The initial Chebyshev Sheet implementation shall not be
test-complete until all mandatory functional tests through Stage 5 and
all Stage 6 control and as-built tests have passed or been validly
superseded.

**16.2 **Test completion shall require preserved dense behavior,
mathematically valid bases, correct compact materialisation, finite
gradients, stable initialization, deterministic optimizer grouping,
compact checkpoint/resume, checkpointed equivalence, direct inference,
bounded target execution, DDP evidence, and full traceability.

**16.3 **A scientifically negative or inconclusive pilot may still
satisfy functional test completion.

**16.4 **Any successor architecture shall receive its own requirements
and test-plan amendment rather than silently extending the accepted v0.1
matrix.

# 17. Immediate next actions

**17.1 **Complete the current Vermeer enhancement without interaction
from thog2 work.

**17.2 **Create the thog2 repository as a fresh nanoGPT clone and
provide access.

**17.3 **Add the requirements specification, implementation plan,
staging plan, and this Test Plan during Stage 0.

**17.4 **During Stage 0, inspect the actual baseline test facilities and
replace provisional path or command assumptions with repository-native
commands.

**17.5 **Before Stage 1 acceptance, run the basis tolerance and
construction-cost calibration study and record any Test Plan revisions.

**17.6 **Do not treat this v0.1 inventory as exhaustive; add tests when
implementation evidence shows that additional coverage is warranted.
