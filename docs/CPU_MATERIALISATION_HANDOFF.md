# CPU materialisation CUDA lifetime and diagnostics hand-off (10 October 2026)

## Scope and authorization

Repository `peterdonnelly1/thog2`, branch `cpu_materialisation`. The user authorizes branch publication through the GitHub connector without another confirmation. Give informative heartbeats with estimated progress, then the download and smoke/preliminary commands. Do not spawn agents without an explicit request. Use existing scratch inputs; do not retrieve them through Library.

Local checkout: `/workspace/scratch/75ac1893c477/thog2`.
Current publication parent: `63bd6bea793a9e74440de81f66c111c8fdfac5a1`.
Pristine base: `e29e32db916fc5ceb87a089d057fb78456c1a98a`, recreated at `/workspace/scratch/75ac1893c477/thog2_baseline`. Preserve older recovery checkouts/stashes. The published parent is also checked out read-only at `../thog2_published`.

## Field evidence and diagnosis

Input `/workspace/scratch/75ac1893c477/upload/Pasted text(20261010-052309).txt` has already been read. Scruffy used the correct parent and ran `tests/test_cpu_materialisation*.py tests/test_premat_matmul_binding.py`: **3 failed, 426 passed**. Failed whole-trainer cases were FP16 matmul (GPU, repeat 1), FP16 einsum (GPU, repeat 0), and BF16 einsum (CPU+GPU, repeat 0). Admission and retained-checkpoint errors from the preceding attempt were resolved. Assertion representations truncated tensor-level diagnostics.

Actual CPU trainers reproduced two independent causes:

1. PREMAT enables fast-discard, while the off reference retained detached operational matrices throughout the optimizer update. Different sum/project ordering produced tiny gradient differences amplified by AdamW near zero attention key-bias gradients. A scratch prototype that called the ordinary materializer also retained that update wrapper, initially obscuring this distinction. The corrected benchmark constructs all providers with fast-discard enabled and records that setting. The caller's environment is restored immediately after construction, including exceptions. Previous timings require rerunning with this changed off reference.
2. Legacy GPU einsum backward promoted operands to FP32, unlike native autocast bmm backward's low-precision result rounding before casting back to FP32. With the benchmark execution contract matched, four FP16/BF16 einsum trainer cases still failed on the published parent. Native autograd around the cached value fixes those local reproductions without adjusting tolerances.

## Implementation

- `sheet/premat_native_binding.py` substitutes one qualified mv/bmm (order-one einsum uses mul) and optional packed cat below autograd. Native forward graph construction, saved tensors and backward kernels remain intact; the physical contraction and packed copy do not run.
- Shared `_PrematerializedDepthBundle` dispatches both providers to this binding. `_PrematConsumerStreamAnchor` initializes the ordinary gradient edge on the consumer stream without an identity node. Upload storage leases live in native graph metadata. Learned rows retain required coefficient operands; fixed rows avoid full autocast coefficient copies using an unread scalar-backed shape placeholder, with private autocast caching disabled.
- Effective policy follows the tensor's actual device and treats order-one einsum as FP32 pointwise arithmetic. Worker materialisation disables ambient autocast to honor its explicit policy.
- `tools/benchmark_cpu_materialisation.py` matches fast-discard during trainer construction and records it in `initial_conditions`. Existing seeds, gradient capture, timed endpoint, no-wait fallback and tolerances remain.
- Hardware whole-trainer pytest failures print complete JSON rows for both cached providers rather than stopping at a truncated first-provider assertion. No dashboard production changes.

## Verification

New native-binding suite: **25 passed**, `../native_binding_final_tests.log`. Twelve trainer cases cover both backends, FP32/FP16/BF16, two seeds/orders, dropout, accumulation, checkpoint replay and AdamW. Both forced cached providers match native losses, gradients, parameters and optimizer state exactly (`atol=rtol=0`). Other cases cover graph/gradient rounding with learned rows, storage/lease lifetime, interception of physical contraction, order-one behavior and the consumer edge. The initial 23 cases against the published parent with matched fast-discard had **13 failures, ten passes**, including four numerical einsum trainer failures (`../native_binding_published_red.log`).

Affected suite before the two added order-one cases: **275 passed, 177 CUDA-dependent skips**, `../native_all_targeted.log`. The final complete suite includes all 25 new cases.

Fresh complete comparison: **2,178 passed, 204 skipped, 84 existing failures, nine deselected, 484 subtests passed**. Fresh pristine base has the same 84 failures. Compared test IDs, counts and exception kinds; no introduced failures or missing tests. The same five legacy browser exclusions and nine previously exercised timeout deselections apply. All **41 candidate static Instra regressions** and **40 baseline scripts** passed. Two initial script failures were missing NODE_PATH, corrected by using existing `browser_env/node_modules`; they were dependency failures in both trees. Changed Python compilation and whitespace checks passed.

Full evidence was generated in `evidence/cpu_materialisation_native/` and moved to `../native_regression_evidence/` before publication to keep scratch logs out of the commit. `comparison.json` records failure IDs/counts/kinds and missing-test results. `candidate.xml`, `base.xml` and `static.json` preserve detailed evidence. Baseline's stdout stopped updating early, but its complete XML contains every baseline case and the final test IDs; both suites completed with existing failures. Do not call the entire Python suite green.

## Publication and runtime

Python `/workspace/scratch/75ac1893c477/test_env/bin/python` is a symlink to `$CODEX_PRIMARY_RUNTIME_PYTHON`; Torch 2.8 is CPU-only. Use `PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. Static Node checks require `NODE_PATH=/workspace/scratch/75ac1893c477/browser_env/node_modules`.

Discover GitHub tools selectively. Fetch branch state before publishing, create an exact tree from reviewed files, verify it against the local git tree, create a commit and advance the branch non-forcibly with the expected parent. Do not shell-push. Identify this publication by `git log -1 --oneline -- docs/CPU_MATERIALISATION_HANDOFF.md` and inspect GitHub's head before any resumed mutation. GitHub branch was verified at `54c37c8` during implementation. Query Actions with `github_fetch` on `actions/runs?head_sha=...`; the workflow wrapper filters to PR events. Never assume prior CI results establish new CI success.

## User handoff and limits

The user asked why repeated attempts took so long. Already explained that independent implementation/harness faults accumulated, early validation was inadequate, and no local CUDA GPU makes hardware handoffs necessary. Continue authorized work without asking permission; do not present CPU tests as CUDA qualification.

After publication, provide fetch/switch/fast-forward commands and the new expected commit. First rerun the full correctness command on Scruffy with `set -o pipefail` and `tee`, then the FP32 smoke and 30-update/five-warmup/three-repeat preliminary commands. Testing documentation includes FP16/BF16 smoke commands for both backends. CUDA stream behavior, CPU/GPU value differences and performance remain unqualified until these actual hardware runs pass. Preserve JSONs even when parity fails. Link the branch testing instructions. No new speedup or memory-improvement claim.

## Latest Scruffy result and follow-up

Scruffy verified `63bd6bea` and supplied **two failures, 452 passes in 70 seconds**. FP16 matmul passed. The remaining whole-trainer failures were GPU-only FP16 einsum (repeat 0) and GPU-only BF16 einsum (repeat 1), both in updated parameters. FP16 affected attention input bias, attention value weight and MLP contraction weight; BF16 affected attention input bias. Every other comparison reached passed. CPU+GPU has no reported failed comparison, but the FP16 test stopped at repeat 0, so its second repeat was not qualified. The pass increase includes 25 new regressions and one formerly failing test; do not describe it as 26 corrected old failures.

Audit found a separate concrete lifetime gap: the GPU scheduler's release event covers the consuming forward GEMM, while native autograd can save the same low-precision cached storage for input gradients. Its final Python reference can disappear before asynchronous backward finishes. The shared binding now registers the consumer stream for unleased CUDA storage. CPU uploads already register that stream and have an explicit lease. This adds no copy, autograd node, CPU-preparation wait or device synchronization to training. Scheduler comments now distinguish forward ownership from backward protection.

Two new CUDA-only tests queue backward behind a device delay, drop the graph, churn the producer's allocation size and check that the original weight cannot be reused until Main finishes. They also check exact input gradients. These cases are skipped locally and have not been hardware verified. Crucially, the reported trainer failures use checkpoint replay, which normally recomputes weights on Main. The lifetime fix is **not a confirmed explanation or confirmed cure for those two failures**.

The six CUDA whole-trainer cases now finish both rotated-order repeats before reporting numerical failures. For each failed repeat they rerun ordinary DEPTH with the same settings and seed, and report `ordinary_control_passed` plus `ordinary_control_failed_comparisons`. This distinguishes variation already present without PREMAT from provider-specific drift. An ordinary control failure does not waive the original failure; tolerances are unchanged. The control is test-only and does not alter field timing.

Local affected suite: **354 passed, 187 skipped**, `../lifetime_targeted.log`. Full candidate: **2,177 passed, 206 skipped, 85 failed, nine deselected, 484 subtests passed**. Eighty-four failures match the prior complete pristine-base run. The extra fast-discard Stage4 equivalence assertion also fails in isolated runs on both the current candidate and pristine `e29e32d`, with the same MLP expansion weight assertion; it is an existing numerical variation, not a new CUDA-path regression. Preserve the raw comparison showing the difference and the paired isolated JUnit evidence. No baseline tests are missing and the 84 shared failure counts/kinds do not increase. All **41 static Instra scripts** pass again. Evidence is in `../lifetime_regression_evidence/`; the new comparison uses the complete earlier `../native_regression_evidence/base.xml` rather than repeating an unchanged full base suite.

Both Actions runs for parent `63bd6bea` succeeded (runs 38029721225 and 38029718394). Check the new commit's own runs after publication; parent success is not new-commit verification. Next step is the full correctness command on Scruffy, preserving its complete output. If failures remain, use both repeat diagnostics and ordinary controls before proposing another numerical change. Hardware qualification and performance remain open.

## CUDA einsum cast-cache follow-up

Scruffy's verified `c9c1b34a` run reports **two failed, 454 passed**. The same GPU FP16/BF16 einsum whole-trainer test remains. FP16 fails parameters in both repeats, BF16 in repeat one; every ordinary control passes. This isolates provider-specific drift rather than an independently varying ordinary baseline. The failure report alone does not establish its numerical cause.

Source research found a concrete omitted PyTorch 2.8 distinction: CUDA registers einsum itself for autocast, caching casts of the original FP32 coefficient leaves. CPU only registers the decomposed bmm, whose coefficient operands are non-leaf views. `bind_cached_depth` disabled its nested cache, so CUDA cache hits received separate cast edges and changed where low-precision gradients were summed. Existing exact CPU tests did not exercise that CUDA graph contract. Do not confuse this with generic cuBLAS stream nondeterminism or weaken the parameter tolerances.

`sheet/premat_native_binding.py` now preserves the caller's cache setting and permits full native values for cacheable coefficient casts, because later ordinary fallbacks can read the shared cast. Scalar-backed placeholders remain for uncacheable fixed-row casts. Side-stream no-grad materialisation still disables its own cache. `sheet/depth_numerical_policy.py` also follows CUDA's pre-decomposition autocast at order one; CPU order-one pointwise behavior is unchanged.

New `tests/test_cpu_materialisation_cuda_autocast_cache.py` emulates the exact CUDA cast boundary and native cache-clear lifetime on CPU, including full trainer AdamW/dropout/accumulation/checkpoint replay. The same final tests have **16 failures, six passes, 14 actual-CUDA skips** at unchanged `c9c1b34a`; the correction has **22 passes, 14 actual-CUDA skips**. Mixed cases start with either a hit or fallback, ensuring a shared placeholder cannot contaminate a later ordinary contraction. Learned row gradients and the cache-disabled no-copy path are checked. New real CUDA cases check exact shared-cast gradients and CUDA order-one dtype/gradients.

Final affected suite: **397 passed, 201 skipped in 33.73 seconds**. Additional checkpoint/layer-norm checks: **43 passed, 14 skipped, one existing relay CLI spelling failure**, confirmed on unchanged `c9c1b34a` in an isolated baseline run. All **41 static Instra scripts** pass. Compilation and whitespace pass. No Instra production file changed. Source and evidence are in the isolated worktree `/workspace/scratch/f6eec53704dc/thog2_einsum`, with candidate JUnit/log and baseline red output in its parent scratch directory.

The graph defect is reproduced and corrected locally, but no CUDA GPU is available here. Do not claim that the two field failures are cured until Scruffy reruns the complete hardware suite. After that passes, smoke the intended dtype/backend, then run the matched 30-update/five-warmup/three-repeat preliminary experiment. Native CUDA einsum may now retain genuine low-precision coefficient casts where the old code used unread placeholders; this follows ordinary behavior, and its time/memory impact needs fresh measurements. The tested matmul policy remains FP32 and is preferable for initial representative trials. Publication must preserve any concurrently advanced branch head and use the connector with an expected-head lease.
