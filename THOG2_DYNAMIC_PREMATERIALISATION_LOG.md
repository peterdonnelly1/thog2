# THOG2 Dynamic Prematerialisation - Work Log

## Scope and authority

- Branch: `THOG2_Dynamic_Prematerialisation_Enhancement`.
- Base: `master` commit `38763e41f31ae3f16f1f9a5609a0a7a63a686fe8`.
- Requirements: `THOG2_Dynamic_Prematerialisation_Enhancement_v0.2.docx`.
- Implementation plan: `THOG2_Dynamic_Prematerialisation_Enhancement_Implementation_Plans_v0.1.docx`.
- The requirements document governs any conflict with the implementation plan.

## 2026-09-08 - Baseline and design audit

- Recovered and read both authoritative documents in full.
- Audited the model, training wrapper, checkpoint path, OWT CLI, launch scripts, PLASTIC controller, Stage 6 telemetry, W&B integration, and local Instra dashboard.
- Recorded the unmodified master test baseline: 73 failed, 1,319 passed, 28 skipped, and 452 subtests passed in 202 seconds.
- The repository already has known failures; regression decisions therefore use exact failing node-ID comparison, not the failure total alone.

## 2026-09-08 - Runtime and policy implementation

- Added `sheet/premat.py` with validated configuration, admission decisions, lifecycle enforcement, a strict ordered candidate queue, and bounded structured telemetry.
- Added one lazily created persistent CUDA stream per active model device. Disabled mode creates no runtime or stream.
- Candidate state is strictly `UNAVAILABLE -> MATERIALISING -> AVAILABLE -> CONSUMING -> CONSUMED`; a critical-path miss is recorded independently from lifecycle state.
- At a deadline, AVAILABLE is consumed directly, MATERIALISING waits on the existing event without duplication, and UNAVAILABLE is atomically claimed by the main stream. Premat is never launched at the deadline.
- Queue order is QKV/O/UP/DOWN for fused attention and QK/V/O/UP/DOWN for unfused attention. A deferred head is not bypassed. Runtime candidates are limited to the current layer and l+1.
- Actual model materializers are authoritative. Fused QKV admission includes the separate Q/K/V products and concatenation transient. Unfused admission includes explicit `[B,H,T,T]` score/probability tensors and causal-mask pressure.
- Admission separates the ordinary process-peak guard from the global device-buffer guard and accounts for reusable allocator-reserved bytes without double counting them.
- CUDA tensors are associated with both producing and consuming streams through `record_stream`; event queries and stream waits are used without `torch.cuda.synchronize` on the fast path.
- Runtime failures include layer, family, state, policy, and memory-envelope context and fail closed.

## 2026-09-08 - Configuration, PLASTIC, and checkpoint integration

- Added exactly seven public controls: `--premat`, `--premat_attention_mode`, the two mutually exclusive headroom controls, `--premat_gpu_memory_buffer_gb`, `--premat_logging`, and `--premat_instra`.
- Enabled mode resolves to the current-peak policy when neither headroom selector is supplied and forces early discard. Explicit launcher `-E` is accepted and reported as redundant/ignored in enabled mode.
- Added unconditional Sheet startup reporting for supplied mode, attention mode, resolved headroom, buffer, logging, Instra, and effective fast-discard state.
- Retired `--plastic__layer_count__memory_budget_gib`. Supplying it fails with a diagnostic naming `--premat_gpu_memory_buffer_gb` as the replacement.
- PLASTIC `memory_budget` now derives its ceiling from current device free memory plus this process's reserved pool, minus the shared global buffer, so unrelated GPU users remain accounted for.
- Existing checkpoints drop the retired key during normalization. Live premat stream, event, queue, and candidate state are never serialized.
- Activation-checkpoint original execution and recomputation both start fresh segment-local premat state. Premat uses reentrant checkpointing so changing CUDA-headroom admission decisions cannot be mistaken for corresponding saved tensors; checkpointing without premat remains on the existing non-reentrant path.
- Enabled mode is rejected on CPU before model execution. The current host has a CUDA-enabled PyTorch build but no accessible CUDA device.
- Added resolved policy, telemetry schema, l+1 limit, and effective fast-discard provenance to the canonical run configuration.
- Preserved the exact premat underscore option spellings through the repository's process-wide CLI alias normalizer.

## 2026-09-08 - Telemetry and Instra

- Added schema-versioned aggregate telemetry for admissions, deferrals, direct hits, waits, main fallbacks, CUDA timings, memory peaks, minimum headroom, and estimated overlap.
- Aggregate telemetry is persisted for every enabled run even when detailed event logging is disabled.
- Detailed events include sequence/time, layer, family, state transition, policy, process/device memory, buffer/headroom, predicted envelope components, decision, outcome, reason, and wait/miss fields.
- Added bounded local storage for 128 detailed snapshots and a `/api/premat` endpoint.
- Added a dedicated Instra Premat tab with exactly the live l+1 layer above the current layer, separate fused/unfused topology, lifecycle colours, critical-miss overlays, live memory summary, and sortable/filterable forensic table.
- Added numeric premat aggregates to W&B training and final summaries.

## 2026-09-08 - Validation

- Focused runtime/Instra/startup/PLASTIC/checkpoint/compile set: 62 passed, 4 skipped.
- Complete premat/Instra/PLASTIC group after the CLI-normalizer fix: 498 passed, 6 skipped.
- Final full branch suite: 73 failed, 1,337 passed, 32 skipped, 452 subtests passed in 206 seconds.
- Exact pytest-cache comparison: master has 63 failing node IDs; branch has the same 63, with no branch-only or removed failures. The larger displayed failure total includes unittest subfailures.
- Static checks passed: Python `compileall`, `bash -n train_OWT.sh train_OWT_core.sh`, Node syntax for the Premat dashboard module, and `git diff --check`.
- Focused tests cover both admission guards, allocator-cache accounting, strict queue/no-bypass, the l+1 bound, legal transitions, no late deadline launch, reuse of in-flight event work, fused/unfused CPU math, exact CLI surface, retired-option diagnostics, resolved provenance, PLASTIC accounting for unrelated GPU users, telemetry persistence, bounded dashboard history, and live UI rendering.

## Remaining hardware acceptance

- `torch.cuda.is_available()` is false on this host, so four CUDA numerical/autograd tests are present but skipped: fused and unfused, each with checkpointing off and on.
- A19 profiler proof of real stream overlap and A21 representative steady-state speed benefit require a CUDA host and remain unexecuted.
- Peak-envelope stress under shared-device pressure also remains a CUDA-host task.
- The feature remains default-off. It must not be made default-on unless the required A19 and A21 evidence passes.

## 2026-09-08 - Publication

- Published implementation commit `32b94f59fec30688deca166f2623d7d5cc849e41` through the GitHub connector.
- Advanced the named branch with a non-forced fast-forward and verified the remote head before handoff.

## 2026-09-08 - Post-publication activation-checkpoint repair

- Peter's first S4 CUDA run failed on the initial backward pass with `torch.utils.checkpoint.CheckpointError`: original-forward and recompute saved tensors had different metadata and ordering.
- Root cause: the original checkpointed forward inherited one whole-forward premat queue, while recomputation created a new queue per segment. Adaptive admission also means CUDA memory headroom can legitimately change the order in which differentiable materialisation operations run.
- Repair: checkpoint closures now own fresh premat state in both original execution and recomputation, and premat checkpoint calls use the reentrant variant. Its original forward runs without recording an autograd graph and its full recomputation creates the gradient-bearing graph, removing the invalid saved-tensor pairing. Non-premat checkpoint calls still use the established non-reentrant variant.
- Added a CPU lifecycle/numerical/gradient regression that verifies two S2 forward passes run without gradients, the two reverse-order recomputations run with gradients, all pass state ends cleanly, and results match non-checkpointed execution.
- Added the missing unconditional `PREMAT:` diagnostics row to `run_thog2_owt.py`; the first real run proved the earlier row in `run_thog2_owt_core.py` was not the active presentation path.
- Static Python compilation and `git diff --check` pass in the resumed scratch environment. Its Python runtime does not currently contain PyTorch, so the executable test suite cannot be repeated here; CUDA validation remains explicit above.
- Published repair commit `2ee7d80b6e435484a457d0d436385acb144eb715` through the GitHub connector and verified it is exactly one fast-forward commit after the prior branch head.

## 2026-09-08 - Field-test memory and Instra repair

- Peter's scruffy field test removed the checkpoint metadata mismatch but exposed a first-backward CUDA OOM. The failing allocation was 3.07 GiB with 5.88 GiB allocator-reserved but unallocated.
- Root cause: the metadata repair selected reentrant checkpointing while retaining configured S4 segments. Reentrant backward reconstructs the complete segment graph, so four logical layers were simultaneously live and defeated the intended activation-memory bound.
- Premat checkpoint recomputation is now bounded to one logical layer. The original no-gradient forward again owns a whole-model Premat pass, retaining actual l+1 scheduling; each backward recompute gets fresh one-layer state. Non-Premat checkpoint segmentation is unchanged.
- Device admission now charges the complete candidate envelope against driver-reported free memory and the configured global buffer. It no longer assumes the allocator's aggregate unused reserve contains a reusable block of the required size, closing the fragmentation failure exposed on scruffy.
- Instra's visible run-detail navigation is generated from a separate fixed tab list. Premat is now registered in that list and controls the shared chart pane directly; the easy-to-miss toolbar toggle was removed.
- Added/updated focused regressions for gradient-equivalent one-layer Premat recomputation, conservative cache-fragmentation admission, and membership in the actual Instra tab strip.
- Static Python compilation, JavaScript syntax checks, and whitespace validation pass locally. This resumed container has no pytest executable, so executable unit/CUDA validation remains assigned to the scruffy acceptance run.

## 2026-09-08 - Live Instra and allocator-fragmentation follow-up

- Peter's post-repair field boundary improved to L10/P5. L10/P6 still OOMed even after `--premat_gpu_memory_buffer_gb` was reduced from 1.0 to 0.5 GiB; the matched non-Premat reference had previously run at L12/P7.
- The buffer is not a CUDA-stream reservation. Under the selected `stay_below_current_peak` policy it is the independent physical-device safety guard, so reducing it cannot repair the checkpoint allocation pattern and can only permit more aggressive admission under device pressure.
- The Instra screenshot showed `process_reserved=13.85 GiB`, `process_allocated=1.31 GiB`, and `premat_retained=0 KiB`. Together with the earlier OOM's 5.88 GiB reserved-but-unallocated pool, this identifies allocator fragmentation from changing one-layer reentrant recomputation allocations rather than an up-front Premat allocation.
- Premat wrapper runs now default `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` only when the user has not supplied an allocator policy. The effective allocator setting is recorded in run metadata and the unconditional `PREMAT:` startup row.
- The Premat graphic was not live: detailed snapshots were written only when the ordinary optimizer progress row was due, normally every ten updates. The apparent seven-event limit was the seven visible table rows; the retained payload was already capped at the newest 256 events, but the UI did not expose the monotonically increasing event sequence.
- Runtime transition snapshots are now emitted at pass boundaries and at a throttled 250 ms cadence. A latest-only, one-slot background writer replaces the active optimizer-update row in SQLite, preventing UI disk latency and host-memory backlog from entering scheduler decisions. Instra polls at 750 ms and displays both the retained window count and total event sequence.
- Added regressions for live pass-boundary reporting, total-versus-window event counts, human-visible polling cadence, and same-update SQLite replacement. The persistence writer smoke passed through a direct module load.
- Static Python compilation, JavaScript syntax, shell syntax, and `git diff --check` pass. The current container still has no PyTorch or pytest, so the L10/P6/L12/P7 CUDA capacity boundary and numerical/autograd tests require the scruffy field rerun.
- Published follow-up commit `84893008ef50312421be2b7f84cee83676c98611` through the GitHub connector and verified the named remote branch points to it.

## 2026-09-08 - Sampled Instra capture and client playback

- Detailed Premat visualisation does not issue a W&B call per transition. Its live path is the bounded local SQLite writer; W&B receives only aggregate numeric scalars on the ordinary optimizer-progress cadence.
- Live Instra publication is now limited to one complete optimizer update at the existing `log_interval` (`-l`). Update 1 and the final update are also captured. With `-l 10`, the detailed view captures updates 1, 10, 20, and so on while retaining the 250 ms intra-update feed.
- No new training hyperparameter was added. The existing `-l` setting remains the single capture-cadence control.
- Added a browser-only playback buffer and a 0.25×, 0.5×, 1×, 2×, 4× slider. Playback speed affects only how quickly already-polled frames are presented; it cannot affect training, CUDA scheduling, local capture, or W&B traffic.
- The client queue is bounded to 128 frames and the training-side writer remains latest-only, preventing either slow playback or disk latency from creating unbounded memory growth in the training process.
- Python compilation, JavaScript syntax, shell syntax, and whitespace validation pass in the current environment.

## 2026-09-08 - Backward-stream memory isolation

- The updated scruffy boundary is L9/P5 and L10/P5 successful, with L10/P6 failing. This confirms a Premat-path regression relative to the matched non-Premat L12/P7 reference.
- The active whole-model forward is the only phase with an l+1 scheduling opportunity. Reentrant checkpoint backward replays exactly one logical layer, so its fresh Premat pass could only put the current layer's materialisation on the auxiliary stream; it could never pre-materialise l+1.
- Those single-layer auxiliary-stream materialisations added no useful overlap while creating a second allocator-stream lifetime during the largest backward allocations. This is the remaining Premat-specific memory-path difference after one-layer checkpointing and expandable allocator segments.
- Reentrant backward now bypasses the Premat runtime and materialises its single replayed layer on the ordinary main stream. The original checkpointed forward retains whole-model l/l+1 scheduling, and Premat-disabled checkpoint behaviour is unchanged.
- Updated the CPU checkpoint regression to require a single outer Premat pass, gradient-equivalent direct recomputation, effective one-layer replay, and clean pass shutdown.

## 2026-09-08 - Autograd-transparent Premat correction

- Scruffy improved to L11/P6 but L12/P6 still OOMed, confirming that removing the recomputation-side CUDA stream reduced rather than eliminated the regression.
- The remaining cause was architectural: Premat had moved differentiable weight-materialisation operations to their physical prefetch time, sometimes across activation-checkpoint boundaries. The earlier checkpoint repairs changed checkpoint mode and segment size to accommodate that ordering.
- Corrected the ownership rule: physical weight generation now runs without autograd at admission time; the cached numerical tensor is bound to its DEPTH coefficients and depth row only at the ordinary consumption point.
- The binding uses the exact linear DEPTH derivatives for single families and fused QK/QKV bundles without copying the cached dense tensor. Admission timing can therefore vary between original execution and recomputation without changing checkpoint-visible operation order or gradients.
- Restored the established non-reentrant activation-checkpoint implementation and the user's configured segment size. Premat no longer selects a different checkpoint algorithm merely because it is enabled.
- Added CPU forward/gradient equivalence coverage for both a single materialised family and a fused QKV bundle, plus configured-segment checkpoint lifecycle coverage.
- The intermediate nested-reentrant checkpoint experiment was rejected before publication.

## 2026-09-08 - Strict release ordering and allocator launch repair

- Scruffy's L11/P6 run completed with only about 442 MiB driver-reported free memory; L12/P6 still OOMed. The configured 1.0 GiB value is a Premat admission buffer, not a reservation that can prevent ordinary training allocations from later crossing that floor.
- The startup row exposed a real launch defect: `cuda_allocator=default`. Premat controls placed beyond the core wrapper's pass-through separator reached Python but were invisible to the shell-only allocator setup. The wrapper now resolves the last forwarded `--premat` value before validation and allocator configuration, so the same runstring activates `expandable_segments:True` unless the user supplied an explicit policy.
- The same row's `checkpoint_recompute_segment=1` was stale diagnostic text. It now reports the configured segment size; the actual implementation already used the configured value after the autograd-transparent correction.
- Activation checkpointing itself is not treated as the residual memory cause. The remaining Premat-specific fault was a second auxiliary scheduler inside non-reentrant checkpoint recomputation. Recompute now reconstructs the consumption graph without starting Premat; the original forward remains the sole owner of l+1 scheduling.
- Early discard ordering is now explicit at every QK, V, QKV, O, UP, and DOWN call site: the caller deletes its final dense-weight reference before notifying the runtime of consumption.
- Consumption records a main-stream completion event. The scheduler retains its memory charge and cannot launch another auxiliary candidate until that event confirms the consuming kernel has completed. Actual CUDA free/allocated state remains authoritative if autograd or the allocator still owns storage.
- Instra's pipeline cards are taller and borderless, removing the grey separator. An always-visible key explains neutral, materialising, available, consuming, consumed, and critical-path-miss fills. The memory summary now labels the admission buffer precisely and reports its current met/breached margin.
- Python compilation, shell syntax, JavaScript syntax, and `git diff --check` pass. Pytest is not installed in this resumed container, so the focused CPU/CUDA tests remain unexecuted here.

## 2026-09-08 - Final fused-envelope and Instra correction

- Scruffy with a 0.5 GiB admission buffer progressed through L11/P7, L12/P7, L14/P7, L16/P7, L24/P7, L24/P8, L32/P9, and L32/P10, exceeding the previously reported non-Premat L12/P7 boundary.
- Fixed a fused-mode admission bug that charged full `[B,H,T,T]` score/probability tensors; those allocations remain charged only to explicit unfused attention. This was the reason real multi-GiB headroom produced no admissions.
- Main-stream fallback and checkpoint replay now use the ordinary differentiable materialiser. The custom auxiliary-result binding is used only after a genuine Premat launch, addressing the reported `AccumulateGrad` stream mismatch without suppressing the warning.
- Removed checkpoint-segment-local Premat ownership. The whole-model forward alone owns l/l+1 scheduling and telemetry.
- Instra now includes the optimizer step, exact layer numbers, arrows, larger boxes/key and spacing, aggregate admission/hit/fallback counts, `ATTN FUSED · QKV` / `ATTN UNFUSED · QK` / `ATTN UNFUSED · V`, and `MLP UP` / `MLP DN` labels.
- Python compilation, JavaScript syntax, launcher shell syntax, checkpoint-ownership search, and `git diff --check` all pass. Pytest/CUDA execution remains unavailable in this container.

## 2026-09-08 - Successful-admission checkpoint repair

- The first L32/P10 `-K flash2` run after fused-envelope correction reached backward without OOM, then reported 131 tensors saved in checkpoint forward versus 128 in recomputation.
- The exact excess of three came from fused QKV attachment saving each coefficient tensor even though fixed DEPTH rows do not require gradients. Ordinary Q/K/V einsums save only their three depth rows.
- The Premat autograd attachment now follows the ordinary einsum save contract: save a depth row only when its coefficient needs a gradient, and save a coefficient only when its depth row needs a gradient. Forward and recomputation therefore retain the same tensor count and metadata while preserving exact gradients.
- The next scruffy trace confirmed the count repair and exposed the remaining metadata detail: Premat saved three `[8]` FP32 depth rows while autocast einsum replay saved `[1,1,8]` BF16 working views. Attachment now stores the exact einsum working shape and generated-output dtype, while converting back to parameter dtype for its unchanged analytical gradient.
- The corrected L12/P8 run completed update 1 but emitted PyTorch's AccumulateGrad producer/consumer stream warning. Gradient accumulation can retain a coefficient leaf accumulator while different microbatches reach it through admitted Premat or ordinary replay. A zero-copy autograd anchor now sits between every trainable DEPTH input and the Premat binding; it is created at ordinary consumption and makes the final producer into the persistent leaf explicitly consumer-stream-owned.
- CUDA equivalence coverage now fails if this specific warning is emitted; the warning is not suppressed globally.

## 2026-09-08 - Exact-layer Instra replay

- Peter observed the first real `AVAILABLE` (orange) Premat candidate and its later `CONSUMED` state. This confirms admission, auxiliary materialisation, ownership hand-off, and consumption are active in the field run.
- The apparently permanent layer 11/31 display was a presentation defect: the live SQLite row deliberately retains only the newest snapshot for an optimizer update, so several layer transitions could be replaced before Instra's 750 ms poll.
- Every retained event now records the exact current and next logical layer indices. Instra reconstructs candidate state from unseen bounded events and replays each sampled `-l` update through its actual layer sequence instead of repeatedly presenting the terminal layer.
- Replay preserves separate `MATERIALISING`, `AVAILABLE`, `CONSUMING`, and `CONSUMED` frames. Each frame has a minimum visible duration of 250 ms at every playback-slider setting, making short consuming transitions perceptible.
- This is browser-only replay over the existing bounded event window. It adds no CUDA tensor retention, training-process queue, W&B call, or telemetry capture interval.
- JavaScript syntax, Python compilation, the direct Node replay harness, and `git diff --check` pass. Pytest/PyTorch remain unavailable in this container; the CUDA stream-warning regression requires the scruffy rerun.

## 2026-09-08 - Complete-microstep all-layer Instra view

- Replaced the polled l/l+1 tail with an atomic capture of the first original-forward accumulation microstep at each existing `-l` optimizer-update interval. Checkpoint recomputation remains excluded, pass events cannot be combined across microsteps, and only a completed pass is published.
- Instra now waits for the completed capture and replays every meaningful matrix transition across a permanent all-layer grid. Layer numbers are 1-based, layer 1 is at the bottom, row height contracts to a readable floor before the view scrolls, and terminal outcomes persist without adding artificial dwell time.
- The state-duration slider now controls the dwell for each timed state (default 0.25 s). Pause freezes the current frame; resume completes it. The final outcome grid holds for one second before playback moves directly to the newest completed capture, dropping older pending captures.
- The visual grammar distinguishes neutral work, full hits, waited/partial hits, and main-stream misses. `MATERIALISING`, `AVAILABLE`, `CONSUMING`, `WAITING FOR PREMAT`, `MAIN MATERIALISING`, `NO PREMAT`, and `TOO LATE` are all represented in the permanent bottom key.
- Candidate labels are `ATTN FUSED · QKV` or `ATTN UNFUSED · QK/V`, `ATTN O`, `MLP UP`, and `MLP DN`. An inspect control opens a vertically scrollable, complete one-row-per-opportunity table using the same nouns and verbs, including outcome, wait, materialisation timing, and admission reason.
- The dashboard API now returns only the newest snapshot after the client's last received update, avoiding repeated decoding and transfer of the retained history. SQLite history remains bounded and local.
- Python compilation, JavaScript syntax, DOM-ID/CSS structure checks, whitespace validation, a direct atomic-reporter harness, SQLite persistence checks, and a direct Node state/outcome reducer harness pass. PyTorch, Plotly, and pytest are not installed in this container, so the committed executable suites and scruffy CUDA/UI acceptance remain field checks.
- Field follow-up renamed the view `Premat Recapitulation - Step N` and reserved a fixed-width state/event field, preventing its changing text length from shifting the playback controls horizontally.

## 2026-09-09 - Pending-release starvation correction

- Scruffy field captures at L6/P3 and L4/P2 showed the same signature: a short initial run of full Premat hits followed by only main-path materialisations, despite respectively about 13.86 GiB and 659.8 MiB of reported admissible headroom. The latter still exceeds every individual fused candidate envelope (about 132--144 MiB).
- Root cause: every consumed matrix created a pending-release record, including ordinary main-path tensors with zero Premat-retained bytes, and `_advance()` treated any such record as a global launch prohibition. Once CPU dispatch outran the main CUDA stream, each fallback left an incomplete event and forced the following fallback, making starvation self-sustaining.
- Main-path consumption no longer creates a zero-byte pending release. An admitted Premat tensor remains strongly referenced and fully charged until its main-stream completion event resolves, but that pending allocation does not prevent another candidate being admitted against independently observed headroom.
- Added regressions for both sides of the invariant: an incomplete Premat release remains charged while the next affordable candidate launches, and an ordinary fallback cannot create a zero-byte release gate.

## 2026-09-09 - One-matrix-lead scheduling

- The first post-starvation field capture at L16 reported 9 fully hidden hits, 55 waited hits, and zero main-path materialisations with 13.18 GiB of headroom. Premat was therefore live for every matrix, but immediate-next targeting exhausted its initial lead and became a synchronous wait convoy.
- Default candidate selection now targets the matrix after the next matrix main will consume, then continues farther whenever prior results and the memory guards permit. This is a one-matrix pipeline lead: only an unprimed or drained pipeline uses main-path materialisation; it is not an alternating main/Premat policy.
- Added the opt-in `--premat_allow_premat_of_immediate_next_matrix` flag to restore zero-lead immediate-next targeting for comparison. The default is false.
- The selected matrix-lead policy is propagated through CLI, wrapper, run/training/model configuration, checkpoint compatibility, artifact identity, canonical metadata, runtime telemetry and the startup `PREMAT:` row.
- Added regression coverage showing the default starts with `O` while `QKV` supplies the single startup bubble, then advances to `UP` and `DOWN` as the preceding prepared matrix becomes consumable. A direct dependency-free scheduler harness passes for both lead policies.

## 2026-09-09 - Original next-layer-only scheduling restored

- Field runs at L32/P16, L12/P6, L6/P3 and L4/P2 showed that matrix-relative lead policies still spent most of their time waiting or falling back on the main path. Lighter runs were worse because their shorter foreground operations gave the auxiliary stream less time to get ahead.
- Restored the original layer pipeline: while logical layer `l` executes, every early-discard/reconsideration point may launch only the next ordered unmaterialised matrix of layer `l+1`. Current-layer opportunities are deadlines, not Premat targets.
- Removed the experimental immediate-next CLI/configuration option and matrix-lead metadata completely. Runtime and canonical telemetry now identify the fixed policy as `next_layer_only`.
- Instra's state-duration slider now spans 0.025--2.000 seconds, its label is lowercase, and the header states the active memory rule explicitly, including the configured global buffer when applicable.
- Python, shell and JavaScript syntax checks, whitespace validation, a dependency-free next-layer scheduler harness, and a direct memory-rule rendering harness pass. PyTorch/pytest remain unavailable in this container; scruffy remains the CUDA acceptance host.

## 2026-09-09 - Global-buffer allocator-cache correction

- The first next-layer-only L12/P8 capture reported about 10.8 GiB tensor allocation but a -242.2 MiB buffer margin against a 0.25 GiB buffer. All 48 candidates therefore fell back to the main path.
- The discrepancy was PyTorch's unused allocator cache: driver-reported free memory was almost exhausted even though several GiB were reserved but unallocated. Premat correctly refused to assume that fragmented, main-stream cache was reusable by its auxiliary stream, but the default allocator policy allowed that cache to consume all physical admission headroom.
- Premat's default native allocator configuration now combines `expandable_segments:True` with `garbage_collection_threshold:0.8`. PyTorch therefore actively reclaims old unused blocks above 80% capacity rather than hoarding them, while the existing admission guard continues charging the complete candidate envelope against actual driver-reported free memory. Explicit user allocator settings remain authoritative.

## 2026-09-09 - Instra outcome vocabulary and matrix sizes

- Replaced the mixed lifecycle key with separate `PROCESSING` and `OUTCOMES` rows. In-scope matrices awaiting playback are now visually distinct from genuinely out-of-scope compute stages.
- Normal terminal outcomes are exactly `FULL HIT`, `PARTIAL HIT`, and `COMPLETE MISS`. `TOO LATE` was removed from the normal graphic, key, and counters; a candidate released at pass end is retained only as an `INCOMPLETE PASS` diagnostic.
- Processing labels now match the inspector trace: `PRE-MATERIALISING`, `AVAILABLE`, `CONSUMING - NO WAITING`, `WAITING FOR PRE-MATERIALISATION`, `CONSUMING AFTER WAIT`, `PREMAT NOT STARTED - MAIN CODE MATERIALISING`, and `MAIN CODE CONSUMING`.
- Each materialisable column now displays its retained dense-matrix size in MiB above the layer grid. The same value is included in the inspector table; it is deliberately distinct from the larger admission envelope.
- JavaScript syntax, whitespace validation, the direct reducer harness, and a mocked-DOM size/alignment harness pass. PyTorch and pytest are unavailable in this container.

## 2026-09-09 - Reverse next-layer scheduling

- Replaced next-layer execution-order scheduling with the sole fixed reverse order. Fused attention now schedules `MLP DN`, `MLP UP`, `ATTN O`, then `QKV`; unfused schedules `MLP DN`, `MLP UP`, `ATTN O`, `V`, then `QK`.
- The policy still targets only layer `l+1` and still uses one auxiliary stream. It gives the largest, latest-deadline matrices the longest lead; the former left-to-right order is not retained as an option.
- Runtime, canonical, training-identity and startup diagnostics report `target_order=reverse_execution`.
- The Instra key is now one flowing sequence: `OUTCOMES` follows the last `PROCESSING` item directly, allowing the combined key to wrap into two compact lines. The inspector's wide-column rule now correctly applies to State trace rather than Size.
- Exact retained sizes are intentionally small: for D=1024 FP32 materialised weights they are 12 MiB QKV, 4 MiB ATTN O and 16 MiB for each MLP matrix. They exclude the separate transient admission envelope and activations.
- Python and JavaScript syntax, whitespace validation, and a dependency-free fused/unfused scheduler harness pass. PyTorch and pytest remain unavailable in this container.
