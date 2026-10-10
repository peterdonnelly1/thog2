# CPU materialisation smoke and preliminary testing

Activate the existing THOG2 training environment. Keep its CUDA PyTorch build. From the repository root:

    git fetch origin
    git switch cpu_materialisation
    git pull --ff-only origin cpu_materialisation
    git rev-parse HEAD
    python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)'

Record the commit and environment for each host.

## Smoke testing

1. Run the CPU-capable correctness checks. These require PyTorch, pytest and psutil but no GPU:

       PYTHONPATH=. python -m pytest -q tests/test_cpu_materialisation*.py tests/test_premat_matmul_binding.py tests/test_instra_duration_workloads.py tests/test_instra_runner_unittest.py

   Expect no failures. This covers arithmetic, all preparation/batch scopes, actual spawned workers, snapshot reuse/invalidation, no-wait deadlines, late-copy isolation, stage caps, shared native PREMAT backward graphs, backward lifetime, real non-reentrant checkpoint source mixtures and synthetic actual-copy normalization. The native-binding suite also checks exact CPU whole-trainer parity in FP32/FP16/BF16 across both backends and cached providers. These CPU arithmetic checks do not qualify CUDA. Prediction requires qualified GEMM observations and measured upload cost from the same context; cold-start fallback remains normal. Unfinished CPU service remains censored, and duration estimates require matching CPU host and resolved thread budget.

2. On an actual CUDA host, run the hardware suite:

       PYTHONPATH=. python -m pytest -q tests/test_cpu_materialisation_cuda.py tests/test_cpu_materialisation_benchmark.py tests/test_premat_matmul_binding.py

   CUDA absence skips this suite; a skipped suite does not pass hardware qualification. Unsupported BF16 hardware skips only BF16 cases. Check that the real-worker success-path test runs and passes. It forces a completed upload before readiness outside any measured update, then checks input gradients and lifetime.

3. Run the tiny whole-trainer smoke test, with two accumulation microsteps, checkpoint segments of two layers and three updates for each provider:

       python tools/benchmark_cpu_materialisation.py --smoke --threads 1 --label scruffy --output evidence/scruffy_cpu_smoke.json

   Repeat on dreedle, changing the label and output filename. The final line must say PASS. Losses, coefficient/model gradients, updated parameters and optimizer states must match ordinary DEPTH within recorded dtype tolerances. GPU staging must stay at or below its cap; CPU workers must report no CUDA initialization. Tiny fast runs can legitimately record all CPU misses.

   The benchmark resets the training/dropout RNG after constructing every trainer. Its default `--training-seed 373` increases by one per repeat and is identical across providers within that repeat. Model seed 171 and data seed 272 remain fixed. JSON verifies identical initial model, training RNG and data RNG fingerprints, then identical final RNG and batch traces. This corrects the unmatched dropout masks in the original field harness; numerical tolerances remain unchanged. Evidence is saved after each provider even if parity fails, identifies the failing provider and component, and has `status: parity_failed` or `runtime_failed`. Timing remains unqualified until `status: passed`.

   Every provider now uses `fast_discard=true`, including ordinary DEPTH with PREMAT off. PREMAT already requires this setting. Earlier off runs used retained-update materialisations, which sum operational gradients before projection; comparing that accumulation order with fast-discard training amplified tiny differences through AdamW. The benchmark sets this only during trainer construction, restores the caller's environment and records it in `initial_conditions.fast_discard`. Rerun previous field timing comparisons: the corrected off reference has a different execution contract.

   Cached values now retain native PyTorch mv/bmm/cat backward nodes, including autocast rounding and checkpoint saved-tensor behavior. Physical materialisation and packed copies are intercepted at the binding; cached storage is reused and its upload lease follows the native graph. No Main-stream CPU-preparation wait is added. Order-one einsum retains native FP32 pointwise arithmetic under autocast. Hardware pytest failures print complete JSON diagnostics for both cached providers instead of truncated assertion representations.

   The trainer clears gradients after each optimizer update. The benchmark therefore captures real gradients immediately before the optimizer in one extra, unmeasured validation update. `gradient_validation` records its loss, update number and number of gradient tensors. Measured losses, parameters, optimizer state, RNG/batch traces, timing and memory remain those of the requested timed endpoint; gradient capture and its extra update are outside that measurement. Empty gradient sets fail qualification.

   The default smoke and preliminary commands use FP32 training. A pass does not qualify FP16 or BF16. After the mixed-precision admission fix, run the hardware tests above in a fresh process and smoke both backends explicitly:

       python tools/benchmark_cpu_materialisation.py --smoke --dtype float16 --backend matmul --threads 1 --label scruffy --output evidence/scruffy_cpu_float16_matmul_smoke.json
       python tools/benchmark_cpu_materialisation.py --smoke --dtype float16 --backend einsum --threads 1 --label scruffy --output evidence/scruffy_cpu_float16_einsum_smoke.json
       python tools/benchmark_cpu_materialisation.py --smoke --dtype bfloat16 --backend matmul --threads 1 --label scruffy --output evidence/scruffy_cpu_bfloat16_matmul_smoke.json
       python tools/benchmark_cpu_materialisation.py --smoke --dtype bfloat16 --backend einsum --threads 1 --label scruffy --output evidence/scruffy_cpu_bfloat16_einsum_smoke.json

   Repeat on dreedle where supported. Native DEPTH matmul retains FP32 generated weights under FP16/BF16 activation autocast; einsum follows the autocast output dtype. Both providers now price these weights using the actual materialiser policy, independently of activation and attention-intermediate precision. Checkpoint failures propagate their original exception and unwind forward hooks so later runs do not inherit the failed graph.

4. Smoke-test fallback and replay explicitly:

       python tools/benchmark_cpu_materialisation.py --smoke --threads 1 --staging-mb 0.00001 --label scruffy --output evidence/scruffy_cpu_tiny_cap.json
       python tools/benchmark_cpu_materialisation.py --smoke --threads 1 --replay enabled --transfer as_soon_as_ready --label scruffy --output evidence/scruffy_cpu_replay.json

   A cap too small for one matrix must still train correctly using COMPLETE MISS. Repeat on dreedle.

## Preliminary experiments

Use the same configuration on each provider; retain the same batch, context, physical layers, depth order, checkpoint segment, accumulation, dtype and seeds. Do not use Nsight or diagnostic delays for the performance decision.

A compact preliminary run:

    python tools/benchmark_cpu_materialisation.py --updates 30 --warmup 5 --repeats 3 --threads 1 --label scruffy --output evidence/scruffy_cpu_preliminary.json

This longer run is required even when the three-update smoke passes: CPU preparation may produce its first usable cached matrices only after warmup. Optimizer-boundary invalidation now cancels snapshots when an optimizer can update any selected coefficient or the depth basis, including fused AdamW updates that leave tensor version counters unchanged. Accumulation and checkpoint recomputation still reuse a snapshot before the optimizer boundary.

The actual-CUDA whole-trainer regression reproduces this exact warmup/update/repeat configuration for both DEPTH backends:

    PYTHONPATH=. python -m pytest -q tests/test_cpu_materialisation_optimizer.py tests/test_cpu_materialisation_benchmark.py

Then use your representative dimensions, for example:

    python tools/benchmark_cpu_materialisation.py \
      --layers 32 --width 1024 --heads 16 --depth-order 12 \
      --batch 16 --context 1024 --accumulation 6 --checkpoint 4 \
      --dtype bfloat16 --updates 30 --warmup 5 --repeats 3 \
      --threads 0 --label scruffy --output evidence/scruffy_cpu_representative.json

Use dimensions that fit that computer. Run the same command on dreedle with its label/output path. The benchmark rotates provider order, uses the same endpoint GPU drains for all providers and never waits for CPU preparation merely to improve a hit rate. The field harness explicitly installs the chosen DEPTH backend for all providers and disables TF32 for FP32 qualification. Initial fingerprints are collected before warmup; final fingerprints and comparisons are collected after the timed endpoint drain. JSON includes per-repeat losses, model/RNG/batch fingerprints, parity tolerances/differences, time, tokens/s, GPU allocated/reserved peaks, CPU RSS/PSS, worker health, storage states and control costs.

At a fixed staging cap, change one dimension at a time:

| Experiment | Changes |
| --- | --- |
| Arithmetic | --backend matmul, einsum (same effective backend for every provider) |
| CPU scope | --layer-batch single_layer, 4, 8, all_layers |
| Preparation | --preparation eager, scheduled, demand_driven |
| Upload timing | --transfer as_the_code_flies, previous_gemm_leading_edge, as_soon_as_ready, demand_driven |
| Replay | --replay disabled, enabled |
| CPU allocation | --workers 1/2 with --threads 1/2/0 within reported affinity budget |
| Prediction | --transfer predicted_gemm_start with --lead-ms 0, 1, 2, 4 |
| Instrumentation overhead | Matched --logging disabled versus enabled |

For prediction, retain cold-start/unavailable counts separately from qualified prediction misses, actual/target availability, prediction/arrival error, stage peak and control host time. A lead sweep can be neutral or slower; report the measured outcome. Compare medians and each repeat, GPU peak, CPU proportional memory and parity. Do not infer hidden cost from overlap alone.

## Instra and profiler checks

Restart Instra and the NodeAgent on each execution host after updating the branch, then refresh both browsers. This ensures Runner sees the new controls and actual CPU affinity evidence.

Append the canonical CPU fragment to an otherwise known working DEPTH wrapper command:

    --premat enabled \
    --premat_materialisation_device cpu_and_gpu \
    --premat_cpu_preparation eager \
    --premat_cpu_layer_batch_size all_layers \
    --premat_cpu_workers 1 \
    --premat_cpu_threads_per_worker 0 \
    --premat_cpu_transfer_timing as_the_code_flies \
    --premat_cpu_staging_limit_mb 0 \
    --premat_cpu_checkpoint_replay disabled \
    --premat_logging enabled \
    --premat_instra enabled

Check the PREMAT startup row and resolved JSON: requested/resolved thread counts, generated dtype, all L layers, stage cap and inactive GPU timing must agree. CPU flags are separate experiment dimensions in Runner. GPU Recipes must omit inactive CPU flags from generated commands; saved Recipes retain requested dimensions.

For complete update 50, append the existing controls:

    --premat_processing_logging enabled \
    --premat_processing_profiler nsys \
    --premat_processing_logging_capture_update 50 \
    --premat_instra__full_step_timing_capture_and_chart enable \
    --premat_instra__full_step_timing_capture_and_chart_capture step 50

Without the full-step enable, Processing remains the original-forward-only capture. Conflicting explicit selectors must fail. NCU uses its existing bounded --ncu-probe-layer command separately.

In Chromium and Firefox, verify:

- Original-forward recap retains all physical layers, arrows, colors and settled outcomes. CPU mode has FULL HIT / COMPLETE MISS / NOT TARGETED, with no PARTIAL HIT. Replay stays in the separate lifecycle/full-update evidence.
- Inspector and CSV show linked CPU precursors, readiness/fallback, upload and release IDs. Unknown clocks and censored intervals remain labelled.
- CPU worker lanes, D2H/H2D lanes and MAIN kernels are distinct. Resource metrics remain unsplit; kernel-idle bins with copies are labelled as copy-contaminated. Co-residency says N/A while Main NCU resource evidence remains usable.
- Legend toggles, colors, layer zoom/reset, linked pan, card resizing, maximize/restore, persisted geometry, per-run eyes and Workspace comparison work. CPU summary disappears on a legacy GPU run.
- Download the bundle, four CPU CSVs, metadata, raw trace, timing JSON/comparison and paired archive. All displayed inputs must be present under metadata-provided artifact-prefixed filenames.
- Run for at least ten minutes with repeated run/Workspace/Runner selection. Check responsiveness, bounded DOM/heap/request growth and no console errors.

Automated browser reproduction (in an environment with Node and Playwright):

    npm install --prefix /tmp/instra-browser playwright jsdom
    /tmp/instra-browser/node_modules/.bin/playwright install --with-deps chromium firefox
    NODE_PATH=/tmp/instra-browser/node_modules PYTHONPATH=. INSTRA_CPU_SOAK_SECONDS=120 \
      python tests/run_instra_oct07_browser.py tests/instra_cpu_materialisation_browser.js

Run `node tests/instra_cpu_evidence_regression.js` for the independent prediction/readiness markers, unknown timestamps and censored overlays. Run `python tools/check_cpu_materialisation_regressions.py /path/to/pristine/e29e32d-checkout` for the full Python/static Instra comparison; its evidence directory records baseline failures separately. GitHub runs these checks automatically when this branch is updated.

## Acceptance mapping and limits

| Criteria | Automated evidence |
| --- | --- |
| C01, C25 | Exact-base comparison, legacy Python/static/browser regressions, CPU controls/identity tests |
| C02–C07 | Snapshot, batch, deduplication, real spawned-worker and shared-buffer tests |
| C08–C13 | Target/order, scheduled preparation, immediate deadline fallback, late DMA isolation and cap tests |
| C14–C18 | Saved-tensor/gradient tests; all four actual checkpoint source combinations; actual-CUDA suite |
| C19–C23 | Copy/clock fixtures, independent kernel metrics, pairing guards, recap/CSV and both browser suites |
| C24, C26 | Shared public capture selectors, complete-update partition, normalized bundle reconstruction |
| C27, C29 | Event-driven scheduler, bounded histories/queues, predictor fixtures and matched logging/lead field runs |
| C28 | Unprofiled three-provider benchmark on each intended computer |

CPU CI cannot establish CUDA stream correctness or scruffy/dreedle throughput. Record the actual-CUDA result and field JSONs alongside the branch commit before considering hardware qualification complete. Existing baseline failures are recorded separately from introduced failures; the workflow fails on new failures or collection/infrastructure errors.
