# CPU materialisation

Implemented against specification v0.3 and baseline e29e32db916fc5ceb87a089d057fb78456c1a98a, on branch cpu_materialisation.

GPU PREMAT remains the default. Select CPU reconstruction with --premat enabled --premat_materialisation_device cpu_and_gpu. Coefficients, coefficient gradients, depth gradients and optimizer state stay on the training device. CPU workers use spawn, native arithmetic threads and shared immutable snapshots. Cache entries contain complete matrices, not autograd graphs. Every successful use gets a fresh gradient attachment on Main Stream.

A readiness check never waits for CPU preparation or H2D completion. A completed, current-snapshot upload is FULL HIT. Every targeted unavailable, late, blocked or failed CPU path immediately uses ordinary GPU materialisation and records COMPLETE MISS with its reason. Excluded matrices are NOT TARGETED. CPU mode produces no PARTIAL HIT.

## Controls

| Control | Default | Meaning |
| --- | --- | --- |
| premat_materialisation_device | gpu | gpu or cpu_and_gpu |
| premat_cpu_preparation | eager | eager, scheduled, demand_driven |
| premat_cpu_layer_batch_size | single_layer | single_layer, integer >=2, all_layers |
| premat_cpu_workers | 1 | Separate spawned processes |
| premat_cpu_threads_per_worker | 0 | Native threads; 0 resolves from affinity while reserving dispatch capacity |
| premat_cpu_transfer_timing | as_the_code_flies | as_the_code_flies, previous_gemm_leading_edge, as_soon_as_ready, demand_driven, predicted_gemm_start |
| premat_cpu_transfer_lead_ms | 0 | Target availability lead; explicit use requires predicted_gemm_start |
| premat_cpu_staging_limit_mb | 0 | MiB cap; 0 resolves to two selected layer sets at actual generated dtype |
| premat_cpu_checkpoint_replay | disabled | enabled permits CPU values during actual checkpoint recomputation |

Use canonical underscore names with two leading hyphens. Explicit CPU-only flags in GPU mode are rejected. CPU mode marks premat_timing inactive and rejects an explicit conflicting GPU trigger. Runner retains requested Recipe dimensions, removes inactive CPU fields from GPU commands and collapses duplicate concrete GPU runs.

Scheduled preparation uses the existing 0/1/2/10 target resolver independently of upload timing. Finite batches start at the requested layer and proceed through upcoming physical layers. all_layers covers all L layers for selected families in every preparation mode. Matrix order, matrix selector, reserve, allocator policy and Premat Stream priority retain their existing meanings.

## Storage and snapshots

Snapshot validity uses actual parameter identity, storage identity, tensor mutation version, shape, dtype, device, depth/basis values and numerical policy. Unchanged accumulation microsteps reuse the snapshot and CPU matrix cache. Parameter replacement, resume or a real mutation invalidates it. Each required source is downloaded once per snapshot in bounded chunks. CPU workers see it only after D2H completion. The next optimizer mutation waits for the last source-copy event when necessary; no duplicate GPU coefficient bank is created.

Pending reservations, copying destinations, available destinations and autograd-live weights all count against the GPU cap. Admission also leaves room for a separate fallback matrix and its workspace. A late copy retains its immutable pinned source and separate destination until DMA completion, then retires unused storage. Successful storage remains live through input-gradient computation and graph lifetime. Completion and safe-release notifications reconsider admission through the same synchronized scheduler.

The pageable CPU cache and shared source bank have calculated finite bounds. Pinned upload, pinned snapshot and pin-preparation bytes are reported separately. Worker workspace and pending-output bounds are explicit upper bounds, not measured RSS. Shared source views are counted once in the tensor ledger; per-process RSS is labelled separately, with summed PSS available in the field benchmark.

## Numerical and checkpoint contract

The effective backend and CUDA autocast state determine the policy. The effective DEPTH matmul path reconstructs in FP32 even during FP16/BF16 training. The einsum path converts both operands to the effective autocast dtype before CPU accumulation and output conversion. Casting an FP32 reconstruction afterwards is not substituted for this contract.

CPU capability is standalone DEPTH on one CUDA device in one training process, with FP32 coefficient storage and FP32/FP16/BF16 training. Unsupported geometry, sparse layer dropout, PLASTIC, HYPERBLOCK, compilation and unqualified precision policies fail clearly. FP32 einsum requires CUDA matmul TF32 disabled. Leading-edge timing and the matrix selector retain fused-attention restrictions.

Checkpoint segment size, non-reentrant execution, RNG preservation and early-stop remain unchanged. Replay is opt-in and begins at the actual segment-entry context. Forward and replay can independently use CPU hits or ordinary GPU fallback. Tests cover all four source combinations, coefficient/depth gradients, saved-tensor metadata and Adam updates.

## Instra and evidence

Original-forward recapitulation keeps its first sampled accumulation microstep. Replay events never replace that unit. CPU jobs are linked precursors; they do not create imaginary playback frames. The inspector and raw/detailed CSVs expose snapshot/task/matrix/use/upload IDs, readiness, fallback, predictions and release evidence.

Processing retains MAIN/PREMAT kernels and adds worker CPU lanes and distinct D2H/H2D lanes. Actual CUDA copy activity, including raw clocks, bytes, correlations, context/stream/device and censored boundaries, is a separate collection. CPU work and copies are never inserted into kernel intervals or numerically apportioned device counters. Kernel-IDLE can carry active copy coverage. Main NCU resource evidence stays available; CPU PREMAT co-residency is N/A.

CPU Processing schema is 5 and linked complete-update timing schema is 3. Existing GPU schema 4 and timing schema 2 remain readable. The metadata-driven bundle includes prefixed CPU task, transfer, memory and lifecycle CSVs, all existing normalized inputs and processing_data.json. Raw profiler reports and pair downloads retain established handling; CPU/GPU provider mismatches cannot silently pair.

Original-forward-only Nsight capture remains unchanged when full-step timing is disabled. To capture a complete update, use the existing full-step controls together with nsys. Explicit selectors must agree; a single explicit selector supplies the shared boundary. NCU remains a bounded kernel probe. CPU tasks and copy boundaries overlay the exclusive forward/backward/optimizer/other partition; parallel spans are not added to wall time.

Predicted arrival uses bounded matched-context CUDA timing histories without Nsight. Unqualified clocks or insufficient observations fall back to as_the_code_flies. Event completion remains mandatory. Intended availability, submission, actual qualified availability, GEMM error, arrival error and cold-start decisions are exported. Shape/backend changes reset histories; phases have separate contexts.

## Validation and field qualification

Automated CPU-capable, exact-base regression, static Instra and Chromium/Firefox suites run in the branch workflow. CUDA qualification is intentionally a separate actual-hardware suite; skipped CUDA tests are not qualification. Run the testing guide on both scruffy and dreedle before selecting a production configuration. No GPU throughput improvement, memory improvement or hardware qualification is claimed from CPU CI.

The implementation and acceptance mapping are in the testing guide. C15/C18 actual-CUDA execution and C27/C28 intended-host measurements require those host runs.
