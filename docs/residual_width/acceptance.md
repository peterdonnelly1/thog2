# Residual stream compression: acceptance record

Branch: `residuals_compression`. Baseline: `a88cafffa6055cbe75f937a1d1692b10a8e224ee`. Date: 6 October 2026. Source: the unchanged attached residual-stream spectral reparameterisation specification; see [README.md](README.md) for the filename/internal-version discrepancy and source hash. All 142 numbered requirements are retained in [requirements.csv](requirements.csv).

**CPU functional acceptance passes. The repository-wide suite remains red on inherited failures.** There are zero new failed test-case identities relative to the clean parent. The release implements the specified fixed representation and workflow; GPU performance, meaningful language-model quality and distributed/compiled width deployments are not qualified by these measurements.

## Numbered specification outcomes

| # | Area | Requirement count | Outcome |
| --- | --- | ---: | --- |
| 1 | Overview and fixed scope (OBJ) | 10 | Direct learned residual coefficients, independent fixed depth and bounded claims implemented. |
| 2 | Representation and numerical basis (REP) | 13 | D/r/A/F/L/P/V/T and operation N recorded; orthogonality, raw span, float64 rank and fingerprint checks pass. |
| 3 | Learning and spectral affordance (LRN) | 10 | Next-token gradients reach embeddings, compact operators, projected affines and depth banks; sensitivity is separated from quality. |
| 4 | Projected LayerNorm (LN) | 18 | All sites folded; float64 ≤1e−9 and float32 ≤1e−4 output/gradient fixtures pass, including general bases and mixed-precision checks. |
| 5 | Compact GPT architecture (ARC) | 16 | r-wide residuals, tied r-wide vocabulary head and direct P-contractions pass storage, dense mapping, projected-reference and causality checks. |
| 6 | Shared spectral registry (BAS) | 6 | Chebyshev, DCT, Haar, lapped cosine and a general orthonormal plug-in pass; unsupported/nonorthonormal/rank-deficient cases reject. |
| 7 | Configuration and selectors (CFG) | 14 | Required order, exact width preset, independent depth, inline/pinned versions, conflicts and all nine capture controls verified. |
| 8 | Memory and timing accounting (MEM) | 7 | Actual learned/optimizer storage reductions measured; all temporary/buffer categories and counter windows labelled. No total-memory reduction claim. |
| 9 | Bounded instrumentation and Instra (INS) | 23 | Off/basic/probes, deterministic fixed-token ablations, bounded curves/discarded energy and Runs/Multiview browser interaction pass. |
| 10 | Checkpoints and workflow (INT) | 6 | Public wrapper/Runner/Grid, complete CLI finish/resume, direct-checkpoint recovery and incompatible/corrupt checkpoint rejection pass. |
| 11 | Validation and acceptance (VAL) | 9 | 135 new acceptance cases pass; matched two-seed five-variant measurements and no-new-failure baseline comparison recorded. |
| 12 | Integration and source history (CODE) | 10 | Integration seams reconciled to the parent, coherent width modules added and original nanoGPT model source untouched. |

## Regression and acceptance evidence

| Check | Unchanged master | Release |
| --- | --- | --- |
| Full pytest ordinary passes | 1,672 | 1,813 |
| Failed pytest assertions (includes failing subtests) | 105 | 96 |
| Unique failed parent test cases | 99 | 90 |
| Skipped cases | 26 | 26 |
| Passed subtests (reported separately) | 483 | 484 |
| New width acceptance cases | — | 135 passed; 133 included in the full suite and 2 additional final fixtures |
| New failed parent cases against master | — | 0 |
| Baseline failed cases now passing | — | 9 |
| JavaScript regression programs | 38 pass, 2 inherited failures | 38 pass, same 2 inherited failures |
| Exact disabled-path fingerprints | Dense plus four depth families | All 5 identical |
| Real Chromium Runs/Multiview interaction | — | 10 checks, zero console errors |

The final focused numerical/integration/browser run passed all 135 cases in 34.92 seconds, with zero failures or skips. Its two additional fixtures independently verify tiled post-affine discarded energy against explicit feature-space affine normalization for constant-mode and general orthonormal bases, including sample and grid bounds. No runtime implementation changed after the full-suite run.

Final pytest output: `96 failed, 1813 passed, 26 skipped, 2 warnings, 484 subtests passed in 274.77s (0:04:34)`. Counts above distinguish parent cases from individual subtest assertions rather than presenting them as interchangeable. Evidence: [regression summary](../../evidence/residual_width_regression_summary.json), [JavaScript comparison](../../evidence/residual_width_javascript_regression.json), [disabled-path fingerprints](../../evidence/residual_width_disabled_regression.json), [browser record](../../evidence/residual_width_browser.json).

`train_OWT_core.sh` was incorrectly non-executable in master; publishing it as 100755 fixes nine canonical-launcher cases without changing their assertions. A remaining old help-text case initially stopped at permission failure; giving the unchanged baseline executable permission exposes the same stale `capital -C` assertion seen in the release. The regression summary records this additional baseline check. Other inherited failures include obsolete CLI/artifact/progress expectations, non-width option/optimizer checks, PREMAT/NCU fixtures, and DDP/full Instra launches blocked by this environment’s AF_UNIX restriction. Exact case identities and first messages are retained in the JSON report. These are not claimed as passing.

Numerical fixtures compare explicit C LN_D(Cᵀc) to the folded operation for every registered family plus a nonconstant-first orthonormal basis, bias both on and off, ordinary/constant/near-constant inputs, all c/γ/β gradients, float64 and float32, and CPU bfloat16 autocast. r=D fixtures consistently map embeddings, operators, affines and tied head to an ordinary dense model and compare outputs and gradients. r<D fixtures compare the specified explicitly projected architecture. Direct depth tests include higher nonzero P modes and all input/bank gradients. Causality, head tying, shared basis, compact residual/parameter boundaries and no normal materialization are exercised.

Workflow tests execute public shell resolution; a complete trainer pilot and its final diagnostics; public CLI fresh training/resume for width alone and independent Haar depth; lifecycle-less direct checkpoint recovery; exact continuation with optimizer/RNG/batch state; incompatible identity and corrupt-basis rejection. Capture tests check deterministic tokens, optimizer/gradient/RNG preservation, completed-update cadence and inclusive windows, history/sample bounds, disabled capture, bounded store reads and figure metadata. The real browser checks all three Plotly charts, step/site selection, curve bounds, axis labels, maximize/restore, paired Multiview steps/identities, rapid switching, legacy-run display and API metadata.

Acceptance regressions discovered and corrected during implementation included independent depth-version overwriting, legacy resume mock field access, lost shell capture arguments, end-of-run weight reconstruction, lifecycle-less width recovery, undefined browser labels/selectors, owner-install ordering, hidden selectors and axis-title controls. Each relevant path has a direct regression check or real-browser acceptance coverage.

## Matched next-token CPU comparison

Each variant uses the same embedded character corpus/tokenizer and exact batch/evaluation-token digests. D=32, L=4, A=32, F=128, context T=16, vocabulary V=20, batch=2, gradient accumulation=2, 40 optimizer updates (2,560 training tokens), 128 fixed held-out tokens, AdamW at constant LR=0.002, β=(0.9,0.95), weight decay=0.1, gradient clip=1.0, dropout=0, float32, two CPU threads, checkpoint segments of two, and fast discard. Model seeds are 77 and 1337; data seeds are model seed+5000. Width uses r=12/DCT; fixed depth uses P=2/Chebyshev. The ordinary same-r control replaces only projected normalization with conventional coefficient-space LayerNorm in an isolated benchmark worker; it is not a selectable product variant.

| Seed | Variant | Learned parameters | Optimizer state KiB | CPU lifetime peak MiB | Complete update ms | Tokens/s | Held-out CE | Perplexity |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 77 | dense | 52,032 | 406.70 | 374.25 | 23.175 | 2761.6 | 2.04401 | 7.7215 |
| 77 | depth_only | 27,456 | 214.57 | 379.84 | 39.189 | 1633.1 | 1.99943 | 7.3849 |
| 77 | width_only | 20,432 | 159.73 | 385.51 | 38.634 | 1656.6 | 2.34755 | 10.4599 |
| 77 | width_depth | 11,216 | 87.73 | 383.51 | 46.260 | 1383.5 | 2.37959 | 10.8005 |
| 77 | ordinary_same_r_norm_control | 20,072 | 156.92 | 384.02 | 23.791 | 2690.1 | 2.54724 | 12.7718 |
| 1337 | dense | 52,032 | 406.70 | 374.84 | 38.553 | 1660.1 | 2.00523 | 7.4278 |
| 1337 | depth_only | 27,456 | 214.57 | 378.77 | 39.227 | 1631.5 | 2.02773 | 7.5968 |
| 1337 | width_only | 20,432 | 159.73 | 385.86 | 38.196 | 1675.6 | 2.41545 | 11.1948 |
| 1337 | width_depth | 11,216 | 87.73 | 384.34 | 45.278 | 1413.5 | 2.45409 | 11.6359 |
| 1337 | ordinary_same_r_norm_control | 20,072 | 156.92 | 383.60 | 23.528 | 2720.1 | 2.58933 | 13.3209 |

Actual learned parameter bytes are 208,128 dense, 109,824 depth-only, 81,728 width-only and 44,864 joint. Width therefore reduces this measured category by 60.73%; joint reduces it by 78.44%. Actual AdamW tensor-state bytes also decrease (416,464 dense, 163,568 width, 89,840 joint). These reductions include embeddings, affines and biases; they do not substitute a core-matrix formula for measured total memory. For a hypothetical exact half-width/half-depth case the nominal core-matrix formula saves 75%, independently of this r=12 toy.

CPU lifetime RSS increases for width and joint. Across the two seeds, mean complete-update time is 30.864 ms dense, 38.415 ms width, and 45.769 ms joint. Mean held-out CE rises from 2.02462 dense to 2.38150 width and 2.41684 joint. Projected normalization beats the same-r conventional-norm control on held-out loss in both seeds in this small fixture, while adding normalization work. These observations do not establish a broad language-model quality advantage or a universal acceptable loss increase.

Timings measure complete forward/backward/optimizer updates, including normalization, direct depth contractions and checkpoint replay, with width capture off. They include the first two updates; the JSON also reports a warmed-update mean. The final benchmark was rerun after the full suite completed, using separate sequential child processes. This is a shared CPU host, so individual milliseconds remain noisy; no GPU speed claim is made. CPU peak RSS is a fresh-child process-lifetime maximum including imports, model construction, optimizer, training, validation and one profiler capture. GPU counters are explicitly unavailable.

Bounded single-sequence normalization profiling (nine sites, after training) attributes construction and application separately:

| Seed | Variant | G/b construction ms | Gain/statistics application ms |
| ---: | --- | ---: | ---: |
| 77 | width_only | 0.5520 | 1.1194 |
| 77 | width_depth | 0.6034 | 1.2720 |
| 1337 | width_only | 0.4939 | 1.1497 |
| 1337 | width_depth | 0.4471 | 1.2789 |

These are instrumented CPU profiler costs for one bounded validation forward, not synchronized per-normalization training timings or GPU estimates. Normal capture separately reports its own diagnostic time, retained history bytes, optimizer tensor state, nominal/actual gradients, logical saved tensor bytes, distinct nonparameter storage bytes and clearly labelled peak windows. G construction is untiled column scaling; post-affine energy diagnostics are tiled.

Full raw measurements and exact matched trace identities are in [residual_width_cpu_comparison.json](../../evidence/residual_width_cpu_comparison.json). Reproduce with `python tools/benchmark_residual_width.py --output evidence/residual_width_cpu_comparison.json`. Repeat seeds or budgets with `--seeds` and `--updates`; do not treat identical wall-clock timing as a reproducibility requirement.

## Production-size basis construction

Float64 construction verifies Chebyshev D=1024,r=128 at condition approximately 111 and full numerical rank. At r=512 its raw condition is approximately 4.5×10¹⁷ on this host and numerical rank 476, so the request correctly rejects; the attachment’s approximate 3.4×10¹⁷ is consistent with the same unresolved condition class, not an exact cross-platform value. DCT D=1024,r=512 is full rank. A pinned lapped-cosine D=64,r=20/window=8/overlap=0.5 construction also certifies its requested span. [Basis evidence](../../evidence/residual_width_basis_construction.json) contains fingerprints, coordinates/stabilization policy, rank tolerance, orthogonality and span errors. These are basis calculations, not training at D=1024.

## Reproduction and qualification limits

Install the repository training dependencies plus pytest, then run:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q tests/test_residual_width.py tests/test_residual_width_integration.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q --junitxml=regression.xml --tb=short
python tools/verify_width_disabled_regression.py --baseline /path/to/clean/master --output evidence/residual_width_disabled_regression.json
```

The final full suite additionally used `THOG_WIDTH_BROWSER_EXECUTABLE` and optional `THOG_WIDTH_CHROMIUM_PACKAGE` to execute the real width browser test. Without a configured browser that one case deliberately skips. The acceptance environment used Python 3.12.14, torch 2.14.1+cpu, pytest 9.1.1, Chromium 153.0.8010.0 and Playwright. JSON, Python AST, JavaScript syntax, shell syntax and `git diff --check` checks passed for the changed files.

CUDA mixed-precision training, actual GPU allocated/reserved peaks, GPU throughput, multi-rank width training, compiled width execution, large-vocabulary/long-context meaningful language-model evaluation and W&B cloud delivery remain untested here. CPU bfloat16 arithmetic and local/W&B-payload wiring are tested or exercised locally; they do not qualify those deployments. The framework supports the current learned-position GPT path; no positional, conversion, adaptive-width or element-geometry adapter is implied. THOGOPT’s reconstructed-depth optimizer is outside qualified width execution. Its existing compatibility check rejects this trajectory.

All release claims are limited to the passing numerical/workflow fixtures and measured storage categories above. The higher toy loss, mean CPU update-time regression, increased lifetime RSS and inherited repository test failures remain visible for the next model-size and hardware decision.

## Numbered outcomes for every specification item

The numbered CSV matrix additionally maps each item to its implementation, validation evidence and qualification scope. Numerical/workflow outcomes refer to CPU functional acceptance; documentary and reporting outcomes are confirmed by this release record.

| # | Requirement | Outcome |
| ---: | --- | --- |
| 1 | **CSA-OBJ-001** — The model shall store token embeddings directly as learned spectral coefficient vectors of the retained width. | Implemented and verified/documented; see numbered matrix for evidence. |
| 2 | **CSA-OBJ-002** — Width compression shall reduce the residual-facing dimensions of stored weight operators as well as the residual activations. | Implemented and verified/documented; see numbered matrix for evidence. |
| 3 | **CSA-OBJ-003** — Width and depth compression shall be independently selectable and jointly usable. | Implemented and verified/documented; see numbered matrix for evidence. |
| 4 | **CSA-OBJ-004** — Width compression shall preserve token count, causal sequence order, and the configured number of executed blocks. | Implemented and verified/documented; see numbered matrix for evidence. |
| 5 | **CSA-OBJ-005** — The preferred variant shall use compact projected LayerNorm and a direct coefficient-to-vocabulary head without expanded residuals in normal execution. | Implemented and verified/documented; see numbered matrix for evidence. |
| 6 | **CSA-OBJ-006** — Width basis selection shall use the shared THOG plug-in registry, with Chebyshev as the documented worked example. | Implemented and verified/documented; see numbered matrix for evidence. |
| 7 | **CSA-OBJ-007** — Version 0.1 shall preserve the existing attention internal width and MLP hidden width. | Implemented and verified/documented; see numbered matrix for evidence. |
| 8 | **CSA-OBJ-008** — Release claims shall distinguish mathematical equivalence of projected LayerNorm from unproven quality, memory, and timing results for the complete variant. | Implemented and verified/documented; see numbered matrix for evidence. |
| 9 | **CSA-OBJ-009** — Every material descriptive statement shall have an identified requirement, and every requirement shall have descriptive coverage. | Implemented and verified/documented; see numbered matrix for evidence. |
| 10 | **CSA-OBJ-010** — Version 0.1 shall use a fixed retained width and fixed basis, without automatic width adaptation or learned basis coordinates. | Implemented and verified/documented; see numbered matrix for evidence. |
| 11 | **CSA-REP-001** — Resolved configuration shall distinguish reference width D from retained coefficient count r and actual residual tensor width. | Implemented and verified/documented; see numbered matrix for evidence. |
| 12 | **CSA-REP-002** — Coefficient indices shall denote feature-basis modes without mixing token positions or layer indices. | Implemented and verified/documented; see numbered matrix for evidence. |
| 13 | **CSA-REP-003** — Shape and memory reports shall record L, P when selected, A, F, N, V, and T with the meanings in Section 2.1. | Implemented and verified/documented; see numbered matrix for evidence. |
| 14 | **CSA-REP-004** — The width representation shall use a fixed analysis matrix C of shape r × D and coefficient vectors of length r. | Implemented and verified/documented; see numbered matrix for evidence. |
| 15 | **CSA-REP-005** — Operator descriptions and exported shape metadata shall specify their matrix orientation. | Implemented and verified/documented; see numbered matrix for evidence. |
| 16 | **CSA-REP-006** — The Chebyshev example shall use the registered coordinate policy and first-kind polynomial recurrence for the requested D and r. | Implemented and verified/documented; see numbered matrix for evidence. |
| 17 | **CSA-REP-007** — Chebyshev construction shall use the registered deterministic QR policy and identify stored coordinates as orthonormalized Chebyshev-derived coefficients. | Implemented and verified/documented; see numbered matrix for evidence. |
| 18 | **CSA-REP-008** — For the Chebyshev constant-mode path, mode zero shall equal the positive constant 1/√D, and WIDTH.order shall mean coefficient count. | Implemented and verified/documented; see numbered matrix for evidence. |
| 19 | **CSA-REP-009** — Basis construction shall validate numerical rank and stabilization accuracy, record conditioning, and reject numerically unresolved requests before model allocation. | Implemented and verified/documented; see numbered matrix for evidence. |
| 20 | **CSA-REP-010** — Documentation and diagnostics shall distinguish analysis Cx, synthesis Cᵀc, and the lossy projection CᵀCx when r < D. | Implemented and verified/documented; see numbered matrix for evidence. |
| 21 | **CSA-REP-011** — Normal training shall not require a full-width learned input vector as the source of each coefficient embedding. | Implemented and verified/documented; see numbered matrix for evidence. |
| 22 | **CSA-REP-012** — Documentation shall identify C and its constrained operators as the spectral machinery and shall describe their linearity in activation values. | Implemented and verified/documented; see numbered matrix for evidence. |
| 23 | **CSA-REP-013** — Construction diagnostics shall use a declared float64 rank tolerance and distinguish conditioning limits of the existing Chebyshev grid from model-training performance. | Implemented and verified/documented; see numbered matrix for evidence. |
| 24 | **CSA-LRN-001** — Token coefficient rows shall be trainable parameters whose meaning is assigned by the fixed width basis and downstream operators. | Implemented and verified/documented; see numbered matrix for evidence. |
| 25 | **CSA-LRN-002** — Learned positional embeddings shall be parameterized and added in the same coefficient coordinates as token embeddings. | Implemented and verified/documented; see numbered matrix for evidence. |
| 26 | **CSA-LRN-003** — Training shall retain the next-token vocabulary cross-entropy objective and propagate its gradients through all selected width and depth parameters. | Implemented and verified/documented; see numbered matrix for evidence. |
| 27 | **CSA-LRN-004** — Quality claims shall recognize that retained-width and depth constraints can change the best achievable loss. | Implemented and verified/documented; see numbered matrix for evidence. |
| 28 | **CSA-LRN-005** — The learning explanation shall distinguish coefficient optimization from input analysis and identify the adjoint C used in synthesis-gradient transport. | Implemented and verified/documented; see numbered matrix for evidence. |
| 29 | **CSA-LRN-006** — The width-normalization explanation shall identify both mathematical analysis and synthesis and their folded implementation, without claiming that analysis is supplied by loss alone. | Implemented and verified/documented; see numbered matrix for evidence. |
| 30 | **CSA-LRN-007** — The depth analogy shall describe learned coefficient matrices, fixed basis evaluation, and gradient accumulation without requiring a separate full-weight analysis stage. | Implemented and verified/documented; see numbered matrix for evidence. |
| 31 | **CSA-LRN-008** — Joint width and depth execution shall allow independently selected basis families and preserve the meaning of direct depth contractions. | Implemented and verified/documented; see numbered matrix for evidence. |
| 32 | **CSA-LRN-009** — Projected LayerNorm shall derive G from C and a learned D-vector γ rather than learn a free r × r affine matrix. | Implemented and verified/documented; see numbered matrix for evidence. |
| 33 | **CSA-LRN-010** — Instrumentation shall report the strength of nonuniform normalization gain and resulting basis coupling without treating a named basis as evidence of quality advantage. | Implemented and verified/documented; see numbered matrix for evidence. |
| 34 | **CSA-LN-001** — LayerNorm statistics shall be computed per token over its D represented features using population variance. | Implemented and verified/documented; see numbered matrix for evidence. |
| 35 | **CSA-LN-002** — Projected LayerNorm shall preserve the existing epsilon and optional reference-space feature-wise affine gain and bias. | Implemented and verified/documented; see numbered matrix for evidence. |
| 36 | **CSA-LN-003** — Every projected normalization shall apply analysis to reference-space LayerNorm of the synthesized coefficient vector, as defined in Section 4.2. | Implemented and verified/documented; see numbered matrix for evidence. |
| 37 | **CSA-LN-004** — Normalization documentation and diagnostics shall identify the post-affine projection and its possible discarded components. | Implemented and verified/documented; see numbered matrix for evidence. |
| 38 | **CSA-LN-005** — The constant-mode shortcut shall use mean c₀/√D, zero-centered mode zero, and the sum of squared centered coefficients divided by D as its variance. | Implemented and verified/documented; see numbered matrix for evidence. |
| 39 | **CSA-LN-006** — Normalization gain and bias shall be derived as G = C diag(γ) Cᵀ and b = Cβ. | Implemented and verified/documented; see numbered matrix for evidence. |
| 40 | **CSA-LN-007** — The Chebyshev compact forward shall apply G to centered coefficients divided by √(variance + epsilon), then add b when enabled. | Implemented and verified/documented; see numbered matrix for evidence. |
| 41 | **CSA-LN-008** — Each normalization site shall own its affine parameters while sharing the fixed C; derived G and b shall not be independently trainable parameters. | Implemented and verified/documented; see numbered matrix for evidence. |
| 42 | **CSA-LN-009** — Normal normalization execution shall avoid an N × D represented-feature activation and any D × D diagonal allocation. | Implemented and verified/documented; see numbered matrix for evidence. |
| 43 | **CSA-LN-010** — G construction shall implement diagonal multiplication by column scaling, with β projected directly as a D-vector. | Implemented and verified/documented; see numbered matrix for evidence. |
| 44 | **CSA-LN-011** — Execution reporting shall account for gain-construction and gain-application work and the r × D construction temporary or its bounded tiled equivalent. | Implemented and verified/documented; see numbered matrix for evidence. |
| 45 | **CSA-LN-012** — Orthonormal plug-ins without a constant first mode shall use the general mean, variance, centering-gain, and projected-bias formula in Section 4.5. | Implemented and verified/documented; see numbered matrix for evidence. |
| 46 | **CSA-LN-013** — Version 0.1 width plug-ins shall supply an orthonormal basis; nonorthonormal representations shall be rejected with a capability explanation. | Implemented and verified/documented; see numbered matrix for evidence. |
| 47 | **CSA-LN-014** — Both per-block pre-attention and pre-MLP LayerNorms, and the final LayerNorm, shall use projected normalization whenever width compression is selected. | Implemented and verified/documented; see numbered matrix for evidence. |
| 48 | **CSA-LN-015** — The placement explanation shall identify the existing normalization boundaries and shall qualify optimality as unproven. | Implemented and verified/documented; see numbered matrix for evidence. |
| 49 | **CSA-LN-016** — Training shall preserve gradients through G and b and invalidate derived caches after any relevant parameter, basis, device, or precision change. | Implemented and verified/documented; see numbered matrix for evidence. |
| 50 | **CSA-LN-017** — Mixed-precision normalization statistics and gain accumulation shall use at least float32 precision and pass explicit-reference gradient tests. | Implemented and verified/documented; see numbered matrix for evidence. |
| 51 | **CSA-LN-018** — Fresh-run normalization gain shall initialize to one and enabled normalization bias to zero. | Implemented and verified/documented; see numbered matrix for evidence. |
| 52 | **CSA-ARC-001** — The model shall store Z as V × r and learned U as T × r and form the input residual by coefficient-space lookup and addition. | Implemented and verified/documented; see numbered matrix for evidence. |
| 53 | **CSA-ARC-002** — Version 0.1 shall support the current learned-position GPT path and reject positional mechanisms without a declared width-compatible adapter. | Implemented and verified/documented; see numbered matrix for evidence. |
| 54 | **CSA-ARC-003** — Each block shall preserve pre-norm attention and MLP order with coefficient-space residual additions. | Implemented and verified/documented; see numbered matrix for evidence. |
| 55 | **CSA-ARC-004** — Width transforms and normalization shall act independently per token, and attention shall retain its causal mask. | Implemented and verified/documented; see numbered matrix for evidence. |
| 56 | **CSA-ARC-005** — QKV and attention-output operators shall be stored directly with shapes 3A × r and r × A. | Implemented and verified/documented; see numbered matrix for evidence. |
| 57 | **CSA-ARC-006** — MLP up and down operators shall be stored directly with shapes F × r and r × F. | Implemented and verified/documented; see numbered matrix for evidence. |
| 58 | **CSA-ARC-007** — Attention head divisibility and KV-cache dimensions shall be determined by A rather than the retained residual width r. | Implemented and verified/documented; see numbered matrix for evidence. |
| 59 | **CSA-ARC-008** — The MLP shall apply regular GELU in its F-wide hidden channels without represented-feature synthesis or tiled synthesized-feature GELU. | Implemented and verified/documented; see numbered matrix for evidence. |
| 60 | **CSA-ARC-009** — Existing dropout probabilities and bias enablement shall apply to the produced channels, with residual-output biases of length r. | Implemented and verified/documented; see numbered matrix for evidence. |
| 61 | **CSA-ARC-010** — The final head shall map coefficients directly to vocabulary logits and preserve existing weight-tying semantics and vocabulary loss. | Implemented and verified/documented; see numbered matrix for evidence. |
| 62 | **CSA-ARC-011** — Normal head execution shall neither store nor construct a hypothetical D-wide head or reconstructed D-wide residual. | Implemented and verified/documented; see numbered matrix for evidence. |
| 63 | **CSA-ARC-012** — Joint depth execution shall store compact P-operator banks and support direct accumulation of their contractions without materializing layer weight matrices. | Implemented and verified/documented; see numbered matrix for evidence. |
| 64 | **CSA-ARC-013** — The existing depth participation control for normalization and bias vectors shall remain independent of width normalization; ln_f shall remain outside sampled block depth. | Implemented and verified/documented; see numbered matrix for evidence. |
| 65 | **CSA-ARC-014** — Version 0.1 shall reject PLASTIC depth scheduling and additional element geometries in width mode until explicit compatibility is implemented. | Implemented and verified/documented; see numbered matrix for evidence. |
| 66 | **CSA-ARC-015** — Compact linear operators shall use ordinary coefficient-to-channel and channel-to-coefficient matrix contractions without requiring polynomial-series multiplication or stored full-width precursor operators. | Implemented and verified/documented; see numbered matrix for evidence. |
| 67 | **CSA-ARC-016** — Fresh-run coefficient tables and compact operators shall use existing role-specific initialization at their stored shapes and preserve the existing depth-bank initialization policy. | Implemented and verified/documented; see numbered matrix for evidence. |
| 68 | **CSA-BAS-001** — Width basis selection shall reuse the shared registry, aliases, version resolution, coordinate policy, and stabilization implementation. | Implemented and verified/documented; see numbered matrix for evidence. |
| 69 | **CSA-BAS-002** — Implementation shall certify Chebyshev and DCT first and provide width adapters and tests for every existing registered spectral family before v0.1 release. | Implemented and verified/documented; see numbered matrix for evidence. |
| 70 | **CSA-BAS-003** — Width selection shall distinguish registered spectral bases from element compressors that do not satisfy the width-basis contract. | Implemented and verified/documented; see numbered matrix for evidence. |
| 71 | **CSA-BAS-004** — The width adapter shall expose orthonormal columns, mode ordering, and constant-mode capability and select the appropriate compact normalization formula. | Implemented and verified/documented; see numbered matrix for evidence. |
| 72 | **CSA-BAS-005** — Unsupported dimensions, unresolved numerical rank, and incompatible versions shall fail validation without substituting another basis. | Implemented and verified/documented; see numbered matrix for evidence. |
| 73 | **CSA-BAS-006** — Runs and checkpoints shall persist independently resolved width and depth basis metadata and fingerprints, with saved exact versions authoritative on resume. | Implemented and verified/documented; see numbered matrix for evidence. |
| 74 | **CSA-CFG-001** — Version 0.1 shall recognize --select-width with --geometry-preset width-type-I and use WIDTH as the scoped option target. | Implemented and verified/documented; see numbered matrix for evidence. |
| 75 | **CSA-CFG-002** — Width selection shall not enable depth implicitly; the same width preset shall compose with explicit fixed --select-depth. | Implemented and verified/documented; see numbered matrix for evidence. |
| 76 | **CSA-CFG-003** — Selection validation shall reject inactive WIDTH options, a width preset without its selector, and incompatible explicit presets. | Implemented and verified/documented; see numbered matrix for evidence. |
| 77 | **CSA-CFG-004** — WIDTH.order shall be required and shall be an integer coefficient count from 2 through D subject to basis-specific validity. | Implemented and verified/documented; see numbered matrix for evidence. |
| 78 | **CSA-CFG-005** — WIDTH.compressor shall default to chebyshev and resolve independently of the selected depth family. | Implemented and verified/documented; see numbered matrix for evidence. |
| 79 | **CSA-CFG-006** — WIDTH.compressor_version shall default to auto for new runs and record the resolved exact version. | Implemented and verified/documented; see numbered matrix for evidence. |
| 80 | **CSA-CFG-007** — The current GPT path shall retain D from --n-embd, A = D, F = 4D, and existing block, head, context, dropout, and bias meanings. | Implemented and verified/documented; see numbered matrix for evidence. |
| 81 | **CSA-CFG-008** — Joint width and depth shall preserve scoped DEPTH controls and existing depth-vector participation semantics, with 1 ≤ P ≤ L. | Implemented and verified/documented; see numbered matrix for evidence. |
| 82 | **CSA-CFG-009** — --explain-geometry shall report active axes, reference and runtime dimensions, basis metadata, and compact operator shapes. | Implemented and verified/documented; see numbered matrix for evidence. |
| 83 | **CSA-CFG-010** — WIDTH parsing shall preserve family@version syntax and reject conflicting versions, duplicate assignments, and unknown properties. | Implemented and verified/documented; see numbered matrix for evidence. |
| 84 | **CSA-CFG-011** — Legacy global basis flags shall retain their legacy behavior but explicitly supplied instances shall be rejected in the scoped width path. | Implemented and verified/documented; see numbered matrix for evidence. |
| 85 | **CSA-CFG-012** — The geometry resolver shall accept the specified width preset with width and optional depth selectors without changing unrelated legacy resolutions. | Implemented and verified/documented; see numbered matrix for evidence. |
| 86 | **CSA-CFG-013** — Command examples shall identify their reference dimensions and coefficient counts and shall not imply certified quality or numerical validity from selection alone. | Implemented and verified/documented; see numbered matrix for evidence. |
| 87 | **CSA-CFG-014** — The hyperparameter summary shall include the complete width-capture option set and identify its shared namespace and cross-reference to capture behavior. | Implemented and verified/documented; see numbered matrix for evidence. |
| 88 | **CSA-MEM-001** — Parameter estimates shall use L(4DA + 2DF) for dense core matrices and L(4rA + 2rF) or P(4rA + 2rF) for selected compact paths. | Implemented and verified/documented; see numbered matrix for evidence. |
| 89 | **CSA-MEM-002** — Half-width and half-depth savings shall be reported as 75% of core-matrix storage, with aggressive examples explicitly limited to that storage class. | Implemented and verified/documented; see numbered matrix for evidence. |
| 90 | **CSA-MEM-003** — Accounting shall report embeddings, tied or untied head, affine vectors, gradients, and optimizer states separately with their actual precision. | Implemented and verified/documented; see numbered matrix for evidence. |
| 91 | **CSA-MEM-004** — Normalization storage reports shall distinguish the fixed basis, affine vectors, construction temporary, derived G, projected bias, and token activations. | Implemented and verified/documented; see numbered matrix for evidence. |
| 92 | **CSA-MEM-005** — Memory measurements shall count actual shared copies, live derived operators, and backward storage and identify whether construction is tiled. | Implemented and verified/documented; see numbered matrix for evidence. |
| 93 | **CSA-MEM-006** — Memory reports shall identify attention scores, QKV, KV cache, MLP hidden activations, logits, and executed depth as outside residual-width reduction. | Implemented and verified/documented; see numbered matrix for evidence. |
| 94 | **CSA-MEM-007** — Performance claims shall use measured complete execution costs, including projected normalization, depth contractions, and separately reported instrumentation overhead. | Implemented and verified/documented; see numbered matrix for evidence. |
| 95 | **CSA-INS-001** — Instrumentation shall retain vocabulary cross-entropy and perplexity as the principal quality measurements and qualify any toy accuracy separately. | Implemented and verified/documented; see numbered matrix for evidence. |
| 96 | **CSA-INS-002** — Basic capture shall report mean-squared coefficient energies and reference mean and centered-energy contributions according to the selected basis’s constant-mode capability. | Implemented and verified/documented; see numbered matrix for evidence. |
| 97 | **CSA-INS-003** — Basic capture shall report normalized γ variation and G coupling relative to mean-gain identity without retaining full G histories. | Implemented and verified/documented; see numbered matrix for evidence. |
| 98 | **CSA-INS-004** — Accounting shall distinguish learned arrays, basis buffers, gradient and optimizer classes, and available actual peak-memory and saved-activation measurements. | Implemented and verified/documented; see numbered matrix for evidence. |
| 99 | **CSA-INS-005** — Timing shall report complete-update throughput and use bounded profiler facilities to attribute normalization construction and application cost. | Implemented and verified/documented; see numbered matrix for evidence. |
| 100 | **CSA-INS-006** — Loss probes shall apply retained-prefix masks at all coefficient boundaries listed in Section 9.2 and use the same evaluation tokens as the unmasked baseline. | Implemented and verified/documented; see numbered matrix for evidence. |
| 101 | **CSA-INS-007** — Ablation output shall state positive delta-loss meaning and identify the result as sensitivity of the trained model rather than performance of a retrained smaller architecture. | Implemented and verified/documented; see numbered matrix for evidence. |
| 102 | **CSA-INS-008** — Probes shall make no optimizer update, preserve training state, and use saved deterministic sample selection independent of training random state. | Implemented and verified/documented; see numbered matrix for evidence. |
| 103 | **CSA-INS-009** — Probe capture shall bound synthesized feature samples and measure post-affine projection-discarded energy using tiles or an equivalent analytic computation. | Implemented and verified/documented; see numbered matrix for evidence. |
| 104 | **CSA-INS-010** — Instrumentation shall not present the identity round trip CCᵀc as evidence of compression quality. | Implemented and verified/documented; see numbered matrix for evidence. |
| 105 | **CSA-INS-011** — Instra shall integrate width energy, retained-order loss probes, and sampled-curve inspection into existing run views and bounded history without a new telemetry service or adaptive-width controller. | Implemented and verified/documented; see numbered matrix for evidence. |
| 106 | **CSA-INS-012** — Width capture shall use the specified width_activation_curves prefix and off, basic, and probes modes, defaulting to off without suppressing existing run logging. | Implemented and verified/documented; see numbered matrix for evidence. |
| 107 | **CSA-INS-013** — Capture and probe cadences shall default to 100 and 1000 completed optimizer updates and be configurable positive integers. | Implemented and verified/documented; see numbered matrix for evidence. |
| 108 | **CSA-INS-014** — Capture bounds shall default to 20 records, 8 token samples per residual site, and min(256, D) feature points, with the domains in Section 7.3. | Implemented and verified/documented; see numbered matrix for evidence. |
| 109 | **CSA-INS-015** — Probe orders shall be distinct retained prefix counts below r; auto shall resolve to max(1, floor(r/2)) and the resolved values shall be recorded. | Implemented and verified/documented; see numbered matrix for evidence. |
| 110 | **CSA-INS-016** — Capture shall use an inclusive start and end update window with defaults 0 and unbounded, and cadence anchored at start_step. | Implemented and verified/documented; see numbered matrix for evidence. |
| 111 | **CSA-INS-017** — Startup validation shall reject invalid capture intervals, sample bounds, probe orders, and windows before training begins. | Implemented and verified/documented; see numbered matrix for evidence. |
| 112 | **CSA-INS-018** — Width metrics shall use existing chart and enabled export destinations and mark unavailable counters explicitly. | Implemented and verified/documented; see numbered matrix for evidence. |
| 113 | **CSA-INS-019** — Basic capture and probes shall report their own elapsed time and retained-buffer bytes separately from normal execution costs. | Implemented and verified/documented; see numbered matrix for evidence. |
| 114 | **CSA-INS-020** — Disabled width capture shall create no width hooks or rolling sample buffers. | Implemented and verified/documented; see numbered matrix for evidence. |
| 115 | **CSA-INS-021** — Enabled width capture shall obey its configured sample and history bounds and avoid full-model weight synthesis. | Implemented and verified/documented; see numbered matrix for evidence. |
| 116 | **CSA-INS-022** — Capture shall use the named residual and normalization sites in Section 9.1, label their location, and detach recorded samples from training graphs. | Implemented and verified/documented; see numbered matrix for evidence. |
| 117 | **CSA-INS-023** — Memory counters shall identify measurement windows, included execution classes, and logical-versus-distinct-storage accounting. | Implemented and verified/documented; see numbered matrix for evidence. |
| 118 | **CSA-INT-001** — Width checkpoints shall record representation identity, dimensions, basis metadata and fingerprint, operator shapes, and selected depth configuration. | Implemented and verified/documented; see numbered matrix for evidence. |
| 119 | **CSA-INT-002** — Checkpoints shall store compact learned arrays and reproducible basis data while treating G and b as rebuildable derived operators. | Implemented and verified/documented; see numbered matrix for evidence. |
| 120 | **CSA-INT-003** — Resume shall validate representation identity before tensor loading and restore compatible optimizer and deterministic instrumentation state. | Implemented and verified/documented; see numbered matrix for evidence. |
| 121 | **CSA-INT-004** — Version 0.1 shall preserve non-width behavior and reject legacy full-width checkpoint conversion rather than perform an implicit projection. | Implemented and verified/documented; see numbered matrix for evidence. |
| 122 | **CSA-INT-005** — Existing wrappers and Runner/Grid surfaces shall expose width controls without creating a separate run or checkpoint subsystem. | Implemented and verified/documented; see numbered matrix for evidence. |
| 123 | **CSA-INT-006** — Run comparison metadata shall distinguish reference width, retained width, and independent depth settings. | Implemented and verified/documented; see numbered matrix for evidence. |
| 124 | **CSA-VAL-001** — CPU folded-normalization output and gradient fixtures shall match explicit analysis and synthesis within 10⁻⁹ float64 and 10⁻⁴ float32 maximum absolute error. | Implemented and verified/documented; see numbered matrix for evidence. |
| 125 | **CSA-VAL-002** — Normalization fixtures shall cover constant-mode and general orthonormal bases, both bias modes, and constant or near-constant represented inputs. | Implemented and verified/documented; see numbered matrix for evidence. |
| 126 | **CSA-VAL-003** — A dropout-free r = D basis-mapped model shall match its dense reference, while truncated tests shall use the explicitly projected compact-architecture reference. | Implemented and verified/documented; see numbered matrix for evidence. |
| 127 | **CSA-VAL-004** — Direct depth contractions shall match small materialized-layer reference outputs and gradients within the fixture tolerances. | Implemented and verified/documented; see numbered matrix for evidence. |
| 128 | **CSA-VAL-005** — Causality tests shall demonstrate unchanged prefix logits after suffix changes, and allocation tests shall verify the specified compact storage and execution boundaries. | Implemented and verified/documented; see numbered matrix for evidence. |
| 129 | **CSA-VAL-006** — Acceptance coverage shall include each registered width basis, invalid configuration combinations, head tying, and shared-basis storage. | Implemented and verified/documented; see numbered matrix for evidence. |
| 130 | **CSA-VAL-007** — Comparative reporting shall cover dense, depth-only, width-only, and combined training with matched data and training conditions and record the quality, memory, and timing fields in Section 11.2. | Implemented and verified/documented; see numbered matrix for evidence. |
| 131 | **CSA-VAL-008** — A basis-benefit investigation shall use an ordinary same-r residual control and report trade-offs without treating toy accuracy or mode energy as proof of language-model quality. | Implemented and verified/documented; see numbered matrix for evidence. |
| 132 | **CSA-VAL-009** — Release shall retain a reproducible acceptance record, make only measured category-specific memory claims, and disclose regressions and untested scales. | Implemented and verified/documented; see numbered matrix for evidence. |
| 133 | **CSA-CODE-001** — Before implementation, the named integration seams shall be reconciled with the selected branch without representing this proposal as existing master behavior. | Implemented and verified/documented; see numbered matrix for evidence. |
| 134 | **CSA-CODE-002** — Width representation and projected normalization shall have a coherent module boundary using the shared registry and derived-operator lifecycle. | Implemented and verified/documented; see numbered matrix for evidence. |
| 135 | **CSA-CODE-003** — Implementation shall follow the staged sequence in Section 12.1 while delivering the full registered-basis and acceptance scope before v0.1 release. | Implemented and verified/documented; see numbered matrix for evidence. |
| 136 | **CSA-CODE-004** — The existing basis registry shall expose width capability and metadata without duplicating family definitions. | Implemented and verified/documented; see numbered matrix for evidence. |
| 137 | **CSA-CODE-005** — Existing geometry modules shall resolve and explain width selection and independent depth composition with the specified validation. | Implemented and verified/documented; see numbered matrix for evidence. |
| 138 | **CSA-CODE-006** — The SheetGPT model and configuration shall implement separated dimensions, compact arrays, projected norms, and direct head and depth execution. | Implemented and verified/documented; see numbered matrix for evidence. |
| 139 | **CSA-CODE-007** — Existing training entry points and wrappers shall expose width and capture arguments and preserve legacy and checkpoint validation behavior. | Implemented and verified/documented; see numbered matrix for evidence. |
| 140 | **CSA-CODE-008** — Existing chart storage and Instra readers shall persist and display width summaries, probes, curves, and comparison metadata. | Implemented and verified/documented; see numbered matrix for evidence. |
| 141 | **CSA-CODE-009** — Runner/Grid and tests shall expose and verify standalone width, joint depth, compatible resume, and instrumentation. | Implemented and verified/documented; see numbered matrix for evidence. |
| 142 | **CSA-CODE-010** — The width path shall integrate with existing GPT training and optimizer without requiring a new sequence compressor, custom GELU, or replacement dense baseline. | Implemented and verified/documented; see numbered matrix for evidence. |
