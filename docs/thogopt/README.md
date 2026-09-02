# thogopt

thogopt applies AdamW in materialised layer coordinates while retaining compact polynomial histories. It addresses the information loss in coefficient-wise adaptive scaling: averaging a mixed coefficient gradient and then squaring it does not recover the histories of its constituent layer gradients.

Use `-y thogopt` with an existing fixed Chebyshev DEPTH training command. The Python lifecycle runner also accepts `--optimizer thogopt`. Existing SGD, AdamW and other optimizer selections retain their own paths and parameter groups.

| Option | Default | Meaning |
|---|---|---|
| `--thogopt__momentum_history_coefficients` | `auto` | `min(P,L)` coefficients, or an explicit integer from 1 through L |
| `--thogopt__scaling_history_coefficients` | `auto` | `min(2P-1,L)` coefficients, or an independent integer from 1 through L |
| `--instrumentation__optimizer_histories__full_matrix_every_n_steps` | `0` | Full-matrix snapshot cadence; zero disables it |
| `--reset-optimizer` | off | Fork-only explicit reset of optimizer histories and bias counters |

The existing learning-rate schedule, beta values, weight decay and parameter-group exclusions apply. Epsilon uses the existing AdamW default, 1e-8. The Python optimizer also accepts an explicit `eps` value. The automatic history sizes are approximation budgets; they do not guarantee that raw layer gradients lie in a low-order polynomial space. `H_m=H_v=L` is a lossless history mode. `P<L` still restricts the final weight update to the permitted weight space.

## Computation and memory

A hook captures each materialised matrix gradient before its reduction to coefficient gradients. Microbatch contributions are accumulated on the host. CUDA capture uses two reusable pinned buffers and a separate copy stream; the CPU reads each buffer only after its copy event completes. Raw storage is reused between updates, with the first contribution overwriting each layer and later contributions adding to it. Loss-scale removal, distributed averaging and physical-coordinate norm clipping occur before gradients are squared. Ordinary uncompressed tensors use AdamW updates with their existing decay policy.

For each family, the first moment follows the linear exponential recurrence in its own orthonormal Chebyshev basis. The second moment fits `beta2 * previous_layer_values + (1-beta2) * gradient²` by unweighted least squares subject to nonnegative values at actual layer nodes. Hildreth dual coordinate descent enforces those constraints. Scale-dependent roundoff corrections are counted; substantive negative or nonfinite values reject the update. At full history capacity, direct sample storage avoids an ill-conditioned high-order Vandermonde. Full-capacity histories use direct AdamW moment recurrences without identity matrix products or a nonnegative fit.

Bias-corrected layer moments determine the AdamW denominator and update. The actual production materializer's pseudoinverse then projects this update into the weight coefficients. Weight decay is decoupled and applied once. Candidate parameters and histories are prepared before commit. Weight candidates read their original device tiles directly. Candidate weights and histories stay on the GPU when the transient memory budget permits, otherwise they use the host transaction path. GPU candidate histories become committed state directly, avoiding a device-to-host-to-device round trip. Failed candidates leave parameters, moments and counters unchanged and use the trainer's existing nonfinite policy.

Persistent covered-family moments contain `(H_m+H_v)*N` values, compared with `2*L*N` for dense AdamW, where N is the number of coupling coordinates. Raw host staging still contains `L*N` values. Transaction candidates consume host or temporary device memory. Auto placement may increase the optimizer phase's peak GPU allocation; it does not add a permanent history replica. Its budget is the larger of released allocation-peak headroom and half of currently available memory (physical free plus unused allocator reserve), capped by availability, less reserves for missing persistent states and workspace. Candidate scratch is released before the next forward pass. Device work is tiled, with a default of at most 1,048,576 layer values per individual tile tensor; several such tensors coexist during fitting. Persistent bases, model parameters and ordinary parameter candidates are additional allocations. This is a device-history saving, not an elimination of host storage.

Startup estimates staging needs against available host memory, including a cgroup limit where present. `THOG2_THOGOPT_HOST_BUDGET_MIB` can impose a smaller explicit budget. The optimizer exposes component byte estimates, fit time, staging time, preparation time and CUDA peak allocation when running on CUDA. The benchmark records process peak RSS separately; that measure includes allocations outside the optimizer.

## Instra

Local optimizer-history capture follows the existing weight-chart destination, debug enablement, sampled coupling selection, step window, cadence and retention settings. The full-matrix cadence is independent and may also be changed on resume. New runs using dense AdamW can record actual dense moments; coefficient AdamW histories are never relabelled as dense moments. The System chart group remains visible if its W&B data are missing, still loading or unreadable, with an explanation when expanded. New captures record the SDK run directory so custom W&B roots and a different dashboard working directory do not prevent discovery.

**Momentum history** and **Scaling history** cover the existing six matrix families. They offer corrected and uncorrected moments, RMS, scaling factors, adaptive updates, latest or ranged captured steps, axis settings and standard/maximized charts. Reference samples use markers at actual layers; signed difference traces are initially hidden in the legend. Between-layer polynomial lines apply to moment quantities. RMS and scaling are shown at layer nodes, avoiding invented between-layer values.

A sampled same-gradient AdamW reference updates on every committed optimizer step. It is passive and is saved in checkpoints. Changing selected couplings starts a clearly labelled new reference origin. A late-start reference is not a complete reconstruction baseline. Separate-run differences include divergence of the training trajectories as well as history approximation.

The magnifying-glass action opens sampled values/differences or a full matrix when a snapshot exists. It supports step, layer, coefficient, reference-run and decimal controls, with four decimal places by default. Full matrices use bounded row/column windows with scrolling; hovering preserves the exact value and physical row/column. Full-matrix differences require matching captured steps, dimensions and coordinates. A sampled reference cannot supply a missing full matrix.

Captures retain FP32/FP64 source coefficients losslessly. Read-side polynomial reconstruction and comparison use FP64; these diagnostic evaluations can differ from a device's rounded FP32 arithmetic within numerical tolerance. Captures identify post-commit histories, source type, quantity conventions, steps, basis, coordinates and reference origin. Inspecting data never changes training state.

## Checkpoints and forks

An ordinary resume requires identical history dimensions, basis identity and history precision. It restores both moments, sampled reference state and the dynamic loss-scale growth tracker when enabled. Changing a history count or switching from coefficient AdamW requires an explicit optimizer reset on a fork. Keep the established fork learning-rate options:

```bash
./train_OWT.sh --fork <checkpoint-or-run-selector> \
  --reset-optimizer --optimizer thogopt \
  --thogopt__momentum_history_coefficients auto \
  --thogopt__scaling_history_coefficients auto \
  --fork-lr-mode restart_cosine \
  --fork-learning-rate 0.0006 --fork-min-lr 0.00006 \
  --fork-rewarm-iters 100 -n <absolute-child-update-target>
```

The fork preserves model weights, data position and global completed-update count. Moment histories and their bias-correction counters restart at zero; run metadata records that reset. Dense-snapshot initialization also starts fresh histories. No missing raw-gradient history is reconstructed from a coefficient AdamW checkpoint.

## Validation and limits

See [validation.md](validation.md), [cpu_benchmark.json](cpu_benchmark.json), [python_test_results.txt](python_test_results.txt) and [javascript_results.json](javascript_results.json). The original requirements are included in [requirements_v0.1.docx](requirements_v0.1.docx).

The initial implementation supports fixed public Chebyshev DEPTH geometry with all layers active. Mutable/plastic depth, chaos sampling, partial layer dropout and AMSGrad are rejected. CUDA/AMP execution, distributed execution, compilation and long OpenWebText convergence need validation on the training machines. The CPU benchmark demonstrates storage savings and quantifies fitting overhead; it does not establish improved training quality or practical GPU throughput.

Reproduce the CPU benchmark with `python -m tools.benchmark_thogopt --output results/thogopt_cpu.json`. The DDP worker is `python -m torch.distributed.run --standalone --nproc-per-node=2 -m tests.thogopt_ddp_worker`. The numerical tests require pytest and SciPy; the new JavaScript DOM regression uses jsdom 26.

## Runtime follow-up

The first GPU report showed about 10.0 s/update for thogopt versus 4.9 s/update for SGD, with L=P=H_m=H_v=16. The implementation performed unnecessary identity-history calculations and many small synchronous transfer tiles. The follow-up removes these calculations, increases bounded tile size eightfold, avoids a weight round trip, combines candidate finite checks into one host decision, and computes the FP64 norm reduction without materialising an entire FP64 squared-gradient tensor. Each FP32 layer tile is now at most 4 MiB; multiple work tensors coexist. Persistent history dimensions and checkpoint conventions are unchanged, so existing thogopt runs can resume normally.

The synthetic CPU full-history comparison (L=P=16, width128, 10 measured updates, one thread) reduced optimizer time from 0.7859 s to 0.6720 s across the measured updates (14.5%), and total training time from 1.6080 s to 1.4581 s (9.3%). These are CPU observations, not predicted GPU gains. Raw gradient capture and transactional moment staging remain transfer costs; no claim that all reported slowdown is removed. The user's screenshot baseline is SGD, not ordinary AdamW.

See [runtime_followup_results.json](runtime_followup_results.json) for before/after measurements and [runtime_followup_python_tests.txt](runtime_followup_python_tests.txt) and [runtime_followup_javascript_tests.json](runtime_followup_javascript_tests.json) for regression evidence. Restart Instra and refresh the browser after pulling. Training already in progress continues with its loaded code; stop at a saved checkpoint and resume to use the optimizer changes.

## Transfer optimization and timings

The first GPU follow-up reduced the reported time from about 10.0 to 7.9 s/update, still above the user's 4.8-4.9 s SGD baseline. The next change targets transfers and allocation rather than changing AdamW's mathematics:

- Two pinned host buffers overlap gradient transfers with GPU work. For D=1024 with a 4D MLP expansion these buffers total 32 MiB; no full additional layer-gradient replica is retained on the GPU.
- Gradient host storage is reused, removing a full allocation/zeroing pass per update.
- Candidates that fit the automatic device budget stay on the GPU through validation and commit. Candidates exceeding the budget use the established host path. Host-to-device raw gradients and any host candidate transactions remain transfer costs.

At the existing console log cadence, `THOGOPT last update:` now reports `stage`, `prepare`, `candidate`, `commit`, `gpu_candidates` in MiB, and `async_staging=yes|no`. These describe the last completed update, not an average across the logging interval. `stage` measures CPU time spent scheduling/draining gradient copies and accumulating values; it is not total DMA duration. `prepare` includes the final copy drain, unscale/reduction/norm/clipping. `candidate` and `commit` include CUDA completion waits. Forward/backward computation and instrumentation are outside these categories, so their sum is not whole-step time.

Execution controls for isolating performance are `THOG2_THOGOPT_ASYNC_STAGING=false` and `THOG2_THOGOPT_TRANSACTION_STORAGE=host`; defaults are `true` and `auto`. These do not change moment dimensions or require an optimizer reset. If pinned allocation is unavailable, capture falls back to synchronous staging and records the reason in optimizer configuration. A training process must restart/resume to load the code.

Validation: 108 CPU-executed tests passed, including old checkpoint compatibility, FIFO copy-consumer ordering, drain-before-clipping, discard semantics, reused-buffer overwrite behavior, AdamW/SGD trajectories, public CLI resume/fork, and console timing output. Four CUDA tests are included in `tests/test_thogopt_transfers.py` but were skipped locally because no CUDA device is available. Three pre-existing console-format expectation failures were reproduced on the untouched parent commit; they were not modified.

The same small CPU comparison was essentially unchanged: total 1.4778 -> 1.4865 s across 10 updates, optimizer 0.6753 -> 0.6756 s. It cannot exercise asynchronous CUDA staging or device candidates, and establishes no GPU speedup. See [transfer_optimization_results.json](transfer_optimization_results.json), [transfer_optimization_python_tests.txt](transfer_optimization_python_tests.txt), [transfer_optimization_budget_tests.txt](transfer_optimization_budget_tests.txt), and [transfer_baseline_console_tests.txt](transfer_baseline_console_tests.txt).
