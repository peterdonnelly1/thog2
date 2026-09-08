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
