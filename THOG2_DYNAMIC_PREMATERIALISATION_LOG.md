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
