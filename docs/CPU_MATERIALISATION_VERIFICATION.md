# CPU materialisation v0.3 verification

The `cpu_materialisation` branch retains the recovered implementation and adds these completion fixes:

- CPU values use the existing `_PrematerializedDepthBundle` autograd node and consumer stream anchor. Each use receives a fresh binding, including fused QKV dependencies and checkpoint replay. GPU PREMAT matmul also uses its native operand views: this corrects an existing mixed-hit/fallback checkpoint metadata mismatch without changing the GPU scheduler or upload controls.
- Deferred uploads retain their individual previous-GEMM stream gates.
- Predicted upload submission requires qualified GEMM history and measured copy duration from the same bounded phase/layer/family/shape/numerical-policy context. Cold starts use ordinary fallback. Prediction counts include eligible matrix uses only.
- Unfinished CPU tasks retain nullable service time and explicit censoring. Full-update evidence retains shared-snapshot precursors; prediction, intended availability, observed availability and readiness have separate chart markers.
- Runner records execution-host affinity. CPU duration estimates require matching host and resolved native-thread budget; GPU history comparison retains its existing behavior.
- GitHub regression checks run on updates to this branch. Inline THOG markers in the changed integration files are aligned at column 156.

## Local checks

The execution environment uses Python 3.12, PyTorch 2.8.0 CPU, pytest 9.1 and Playwright 1.55.1. No CUDA device is available in this environment.

| Check | Result |
| --- | --- |
| CPU materialisation, checkpoint, lifecycle, evidence, controls, completion regressions, native GPU matmul binding, DEPTH/runtime/startup baselines, duration estimator and Runner | 367 passed; 177 hardware-dependent cases skipped |
| Prediction/readiness chart, censored intervals and legacy phase totals | Passed |
| Legacy settings, Runner, table retention, help and colour controls | Passed in Chromium and Firefox |
| CPU evidence, recap isolation, four downloads, zoom/reset, legends, pointer resizing and persistence | Passed in Chromium and Firefox |
| CPU evidence soak | 120 seconds per browser; Chromium 259 iterations, Firefox 258; no page errors; trace count stayed at 3 |
| Live twelve-run Workspace soak | 120 seconds, 119 health requests, 84 visible loss redraws; hidden accuracy card rendered once; no page errors |
| Workspace browser heap | 14.24 MiB initially, 15.43 MiB retained, 79.46 MiB peak; maximum measured main-thread pause 168 ms |
| Python compilation, changed JavaScript syntax, wrapper shell syntax and whitespace | Passed |

The CPU browser soak retained 3,688 DOM elements in Firefox. Chromium ended with 3,690 versus 3,688 initially; no unbounded growth was observed during these checks. These short automated soaks do not replace the ten-minute host/browser smoke session described in the testing guide.

## Exact-base regression comparison

The pristine comparison checkout is `e29e32db916fc5ceb87a089d057fb78456c1a98a`.

| Suite | Pristine base | CPU branch | New failures |
| --- | --- | --- | --- |
| Full Python suite | 1,900 passed, 27 skipped, 94 failed | 2,069 passed, 148 skipped, the same 94 failed | 0 |
| Final-code follow-up, excluding nine timeout cases already exercised above | 1,900 passed, 27 skipped, 85 failed | 2,117 passed, 196 skipped, the same 85 failed | 0 |
| Static Instra JavaScript | 40 scripts passed | 41 scripts passed | 0 |

Both Python comparisons also passed 484 subtests. Individual failure counts and exception kinds were compared, including multiple failing subtests under one test; neither increased. No baseline test was dropped. The CLI option assertion was explicitly renamed to check the existing options plus nine CPU controls.

The nine timeout cases comprise eight distributed-training checks and one existing Processing assertion. Other baseline failures include earlier wrapper/CLI, console, geometry and Processing contracts. These failures remain visible in the comparison reports; this verification does not claim that the entire repository suite passes.

Five legacy native-Firefox Python modules are excluded by the full comparison harness. Real-browser coverage is provided by the Chromium/Firefox settings and Runner suite, the CPU evidence suite and the live Workspace soak recorded above. The generated JUnit, logs, comparison JSON and coverage details are under `evidence/cpu_materialisation_ci` and `evidence/cpu_materialisation_final`; GitHub uploads its comparison reports as workflow artifacts.

## Hardware qualification

CUDA skips are not hardware passes. Real CUDA stream ordering, mixed precision, storage lifetime and numerical qualification must run on scruffy and dreedle using `tests/test_cpu_materialisation_cuda.py`, `tests/test_premat_matmul_binding.py` and the three-provider benchmark. C28 requires the resulting matched, unprofiled throughput and GPU/CPU memory measurements on both computers. No speedup or hardware-qualified completion is claimed by these CPU-only results.

Follow [CPU_MATERIALISATION_TESTING.md](CPU_MATERIALISATION_TESTING.md) for the download commands, smoke tests, tiny-cap fallback, replay checks, preliminary experiments and Instra/profiler verification. The hardware suite forces a real completed-upload success path outside the measured update; performance measurements retain immediate fallback and contain no diagnostic preparation waits.

## Field-harness RNG correction

Scruffy reported 386 passing CPU/CUDA correctness cases at `06ce6a1`, followed by a whole-trainer smoke parity failure with a maximum loss difference of 0.006978. The field harness seeded model construction and batch sampling, but omitted the training/dropout RNG reset. `build_training_model` uses `torch.random.fork_rng`, which restores the ambient RNG after initialisation; sequential provider runs therefore received different dropout masks.

Two actual ordinary DEPTH trainers reproduced this independently on CPU: initial models and sampled batches matched, while loss differences reached 0.012786. Resetting training RNG after construction produced identical losses, updated parameters and AdamW state. The corrected harness sets the same training seed for every provider in a repeat, records initial model and initial/final RNG/batch fingerprints, and retains the existing numerical tolerances. Failed runs retain JSON evidence and identify the provider and failed component; timing remains unqualified until all checks pass.

The affected local suite passed **375 tests**, with **183 CUDA-dependent cases skipped**, including six new whole-trainer hardware cases covering both backends and three dtypes. New CPU regressions exercise actual trainer dropout with and without warmup and activation checkpoints, provider-order rotation, RNG mismatches, missing gradients, numerical failures and evidence preservation. The prediction/readiness/censoring chart regression also passed. GitHub's preceding core and complete base-comparison jobs passed; its CPU browser job failed a Firefox maximise assertion. Captured pointer events reproduced an automated offscreen scroll moving the button during the click, which landed on the containing article. Explicitly settling that scroll corrected the test; the same maximise, resize, downloads and recap assertions then passed in Firefox. Maximise assertions also wait for the visible state.

The corrected three-provider smoke and preliminary experiments still require a rerun on the intended CUDA hosts. The earlier smoke timing numbers do not establish a speedup.

## Fused optimizer snapshot correction

At `9455fa6`, Scruffy's corrected smoke passed, but the 30-update, five-warmup preliminary experiment failed CPU+GPU loss, parameter and optimizer-state parity. GPU PREMAT matched ordinary DEPTH. PyTorch 2.8 fused AdamW reproduces a source mutation without a tensor version increment on CPU as well as on the CUDA optimizer path used by the trainer. Version-based fingerprints alone can therefore accept a stale CPU snapshot.

The trainer now passes its actual optimizer to the CPU runtime after the step, before gradients are cleared. If an optimizer group with a nonzero learning rate contains a selected coefficient or depth basis with a gradient, the runtime invalidates the snapshot, increments its worker generation, clears cached matrices and pending source jobs, and rejects late results from that snapshot. Zero-learning-rate groups, absent gradients and updates outside the selected families preserve reuse. Accumulation and checkpoint replay still share valid snapshots within an update. The existing source-copy mutation guard and immediate readiness fallback remain in force; the fix adds no CPU-preparation wait to Main.

New regressions exercise ordinary, foreach and fused AdamW; coefficient and learned-basis mutations; no-op boundaries; discarded late worker results and source copies. Four real-worker tests run 35 fused updates (the preliminary experiment's five warmup plus 30 updates), with two accumulation microsteps, dropout and non-reentrant checkpoint segments. Both materialisers and fixed/learned depth bases match native losses, gradients, updated parameters and optimizer state, with real cached matrix uses during original forward and replay. Fused coefficient version counters remain unchanged throughout these tests, so they cover the previously missed case directly.

The field benchmark also fixes a validation gap: SharedTrainer clears gradients at the end of each update. One extra, unmeasured update now captures actual gradients before the optimizer and records its loss and tensor count. Empty gradient sets fail parity. Parameters, optimizer state, RNG/batch fingerprints, timing and memory still refer to the requested timed endpoint, and all numerical tolerances remain unchanged. Two CUDA regressions reproduce the exact 30-update/five-warmup/three-repeat command with rotated providers, one per backend.

The targeted local suite passed **391 tests**, with **185 hardware-dependent cases skipped**. The six depth/PLASTIC failures introduced by the preceding benchmark tests were reproduced as test pollution: mocked benchmark-main tests installed global DEPTH methods and left backend settings changed. Those mocks now isolate installation, environment and TF32 settings; all six affected cases pass when run after the benchmark tests. Hardware parity and throughput still require rerunning Scruffy's smoke and preliminary commands on this fix.

The final broad comparison passed **2,141 tests** and **484 subtests**, with **204 skips** and the same **85 baseline failures**; nine timeout cases already exercised in the original comparison were deselected. No baseline test disappeared, no existing failure count increased and no new failure kind appeared. All **41 static Instra regression scripts** passed. The CPU evidence, recap, four downloads, zoom, maximize/restore, pointer resize/persistence and 20-second soak passed in Chromium 140 and Firefox 157 (Playwright 1.55.1 and 1.64.0 respectively), with 3,688 DOM elements and three traces retained and no page errors. Timeline maximize/restore, CPU summary restore and resize exercise real pointer handlers; CPU summary maximize also exercises keyboard activation. Resize assertions wait for persistence after the actual drag. These checks do not bypass UI handlers.

Browser checks also exposed CPU-card resizing losing pointer capture after a layout change. Its native move/up/cancel listeners now follow the active pointer on Window so the gesture continues and persists after capture loss. Other cards retain their existing handle listeners. Tests verify trusted native pointer events and stored CPU-card geometry using Playwright's drag action. Legacy settings and Runner browser regressions passed again in both browsers, and all 41 static Instra scripts passed after this UI fix.

## Mixed-precision admission and interrupted checkpoint correction

Scruffy's FP32 smoke and 30-update/five-warmup/three-repeat preliminary experiment passed at `a94dcd6`. Its subsequently supplied correctness log nevertheless recorded **four failures and 414 passes**. GPU PREMAT's FP16/BF16 matmul cases generated FP32 DOWN weights of 16,384 bytes while admitting only 8,192 bytes. Native DEPTH matmul disables autocast for materialisation; GPU PREMAT had incorrectly assumed that every generated weight followed activation precision.

Both model providers now resolve generated-weight element size from the actual DEPTH numerical policy at pass entry. Activation and unfused attention-intermediate charges continue to follow activation precision. This changes admission accounting, not the arithmetic or parity tolerances.

The later einsum checkpoint errors were reproduced as a cascade after an original-forward exception: retaining its traceback also retains PyTorch's checkpoint generator and saved-tensor hook context, which then captures unrelated later training tensors. All three logical checkpoint entry points now clear unwound traceback frame locals on exception so that generator closes. The original exception, message and traceback locations are retained; successful forwards and replay behavior are unchanged.

Eleven new CPU regressions exercise real FP16/BF16 weight arithmetic and exact retained/activation charges across both materialisers and both attention modes, plus interrupted-checkpoint recovery for dense, sparse and prefix execution. Before the production changes, seven cases failed and four passed; afterward all eleven passed. Simulated CUDA admission counters/events are not hardware qualification. The extended targeted suite passed **413 tests**, skipped **185 hardware cases**, and retained one existing baseline failure: the final-checkpoint-relay CLI spelling assertion (`SystemExit: 2` in both base and candidate). All **41 static Instra regression scripts** passed again.

The complete candidate comparison passed **2,153 tests and 484 subtests**, with **204 skips and 84 existing failures**; the same nine already-exercised timeout cases were deselected. Compared with the pristine base's 85 failures, there were **no new failure IDs, increased failure counts, new exception kinds or missing tests**. One existing fast-discard equivalence test no longer failed. Changed Python compilation and git whitespace checks passed. Detailed recovery/publication instructions are recorded in `CPU_MATERIALISATION_HANDOFF.md`.

The FP32 field passes do not establish mixed-precision qualification. Scruffy must rerun the full correctness command on the new commit; the existing actual-CUDA whole-trainer cases cover FP32, FP16 and BF16 with both materialisers. No new CUDA pass is claimed by these local CPU results.
