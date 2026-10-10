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
