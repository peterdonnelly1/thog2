# THOG2 Dynamic Prematerialisation - Task List

Branch: `THOG2_Dynamic_Prematerialisation_Enhancement`

Base: `master` at `38763e41f31ae3f16f1f9a5609a0a7a63a686fe8`

Authoritative requirements: `THOG2_Dynamic_Prematerialisation_Enhancement_v0.2.docx`

Authoritative plan: `THOG2_Dynamic_Prematerialisation_Enhancement_Implementation_Plans_v0.1.docx`

## Implementation

- [x] Add and validate exactly the seven specified `--premat*` public options.
- [x] Preserve disabled-by-default behaviour and force fast discard only when premat is enabled.
- [x] Retire the PLASTIC-only memory threshold and use the shared global GPU buffer.
- [x] Implement lifecycle states and validate legal transitions.
- [x] Implement strict next-use order, no bypass, one in-flight candidate, and the l+1 limit.
- [x] Implement retained, transient, foreground, and combined candidate envelopes.
- [x] Keep process-peak and global-device-buffer admission guards separate.
- [x] Implement one persistent premat CUDA stream per active model device.
- [x] Route fused QKV, O, UP, and DOWN through authoritative candidate call points.
- [x] Route unfused QK, V, O, UP, and DOWN through authoritative candidate call points.
- [x] Include score, scale, mask, softmax, and probability/value pressure in unfused admission.
- [x] Implement completion events, deadline ownership, event waits, main fallback, and `record_stream` lifetime handling.
- [x] Give activation-checkpoint recomputation fresh local premat state.
- [x] Keep original-forward and checkpoint-recompute Premat ownership separate; recomputation starts with fresh pass-local state.
- [x] Isolate adaptive premat scheduling from non-reentrant saved-tensor matching.
- [x] Fail before the first forward when premat is enabled without CUDA.
- [x] Persist supplied/resolved configuration, aggregate counters, and CUDA timing fields.
- [x] Persist aggregate telemetry even when detailed logging and Instra are disabled.
- [x] Add the dedicated Instra Premat tab, live l/l+1 topology, memory summary, and forensic table.
- [x] Add lifecycle and critical-miss visuals without styling neutral compute nodes as weights.
- [x] Add focused scheduler, admission, CLI, provenance, telemetry, UI-render, PLASTIC, and checkpoint tests.

## Regression verification

- [x] Record master baseline: 73 failed, 1,319 passed, 28 skipped, 452 subtests passed.
- [x] Run final full branch suite: 73 failed, 1,337 passed, 32 skipped, 452 subtests passed.
- [x] Compare exact cached failing node IDs: 63 on master, 63 on branch, no additions or removals.
- [x] Run the combined premat/Instra/PLASTIC group: 498 passed, 6 CUDA-only skipped.
- [x] Run Python compilation, shell syntax, JavaScript syntax, and `git diff --check`.
- [x] Check the THOG source-marker convention in new and modified implementation areas.
- [x] Add a CPU checkpoint lifecycle/gradient regression test for fresh forward and recompute passes.
- [x] Add the missing `PREMAT:` startup row to the authoritative lifecycle runner.

## CUDA-host acceptance still required

- [ ] Execute fused enabled-versus-disabled forward, loss, and trajectory-gradient equivalence.
- [ ] Execute unfused enabled-versus-disabled forward, loss, and trajectory-gradient equivalence.
- [ ] Repeat both numerical checks with activation checkpointing enabled.
- [ ] Re-run Peter's reported S4 CUDA command and confirm the checkpoint metadata error is gone.
- [ ] Capture an Nsight or PyTorch-profiler trace proving real overlap and no forbidden global synchronization (A19).
- [ ] Measure representative steady-state step time and confirm a material benefit before considering default-on (A21).
- [ ] Stress both headroom modes on a shared GPU and confirm observed peaks stay inside the admitted envelope.

These CUDA items cannot run on the current host (`torch.cuda.is_available()` is false). The four numerical/checkpoint cases are committed as explicit skipped tests so they become executable on a CUDA host. The feature remains default-off, as required until A19 and A21 pass.

## Publication

- [x] Push the completed tree through the GitHub connector.
- [x] Verify the remote branch head and provide the download stanza.
- [x] Publish and verify the post-publication checkpoint repair commit.

## 2026-09-08 field-test repair

- [x] Diagnose the first-backward OOM as multi-layer reentrant graph reconstruction introduced by the checkpoint metadata repair.
- [x] Bound Premat reentrant recomputation to one logical layer while preserving whole-forward l+1 scheduling.
- [x] Stop treating allocator-reserved-but-unused bytes as guaranteed reusable physical capacity.
- [x] Put Premat in Instra's generated run-detail tab strip instead of a secondary toolbar toggle.
- [x] Add regressions for effective one-layer recomputation, conservative fragmented-cache admission, and real-tab registration.
- [ ] Re-run Peter's L8/P4 S4 command on scruffy and compare peak allocated/reserved memory with Premat disabled.
- [ ] Verify the visible Instra tab and live Premat snapshot on scruffy.

## 2026-09-08 live/fragmentation follow-up

- [x] Record the observed capacity boundary: Premat L10/P5 succeeds; L10/P6 OOMs with both 1.0 and 0.5 GiB buffers; matched non-Premat L12/P7 previously succeeds.
- [x] Confirm from Instra that the failing architecture is not retaining a large Premat tensor (`premat_retained=0 KiB`) while the allocator has a very large reserved/allocated gap.
- [x] Default Premat wrapper launches to expandable CUDA allocator segments without overriding an explicit user allocator configuration.
- [x] Record the effective CUDA allocator configuration in run metadata and startup diagnostics.
- [x] Replace optimizer-log-cadence-only Premat snapshots with bounded asynchronous live publication.
- [x] Display the total event sequence separately from the newest 256 retained table events.
- [x] Add live publication, bounded-writer, count-label, and polling-cadence regressions.
- [x] Re-run static Python, JavaScript, shell, whitespace, and direct persistence-writer checks.
- [x] Publish and verify the live/fragmentation follow-up through the GitHub connector.
- [ ] Re-run L10/P6 on scruffy with the new allocator default and compare peak allocated/reserved values.
- [ ] If L10/P6 passes, probe L12/P7 against the matched non-Premat reference before declaring DPE-SAFE-005 satisfied.
- [ ] Verify that the Instra event sequence increments during an update and that l/l+1 promotes at human-visible cadence.

## 2026-09-08 sampled Instra playback

- [x] Confirm detailed Premat Instra uses local SQLite rather than per-transition W&B calls.
- [x] Gate detailed live publication to update 1, every existing `-l` interval, and the final update.
- [x] Preserve the 250 ms transition cadence inside a selected update.
- [x] Add a bounded client-only playback queue and 0.25× through 4× speed slider.
- [x] Keep playback control isolated from training and telemetry capture.
- [x] Run static Python, JavaScript, shell, and whitespace checks.
- [ ] Confirm on scruffy that `-l 10` produces live captures at updates 1, 10, 20, ... and that all five playback speeds work.

## 2026-09-08 backward-stream memory isolation

- [x] Attribute the remaining L10/P6 regression to the Premat-specific checkpoint recomputation path rather than an up-front stream reservation or retained candidate tensor.
- [x] Confirm one-layer reentrant replay has no l+1 candidate and therefore no useful Premat overlap.
- [x] Bypass the auxiliary Premat stream during gradient-bearing checkpoint recomputation.
- [x] Preserve whole-model l/l+1 Premat scheduling during the original checkpointed forward.
- [x] Keep Premat-disabled checkpoint execution unchanged.
- [x] Update the checkpoint lifecycle/gradient regression for direct recomputation.
- [ ] Re-run L10/P6 on scruffy with this isolation repair.
- [ ] If L10/P6 passes, re-run L12/P7 and compare peak allocated/reserved memory against the matched Premat-disabled reference.

## 2026-09-08 autograd-transparent correction

- [x] Record the L11/P6 success and L12/P6 OOM boundary after recomputation-stream isolation.
- [x] Identify differentiable prematerialisation crossing checkpoint boundaries as the underlying incompatibility.
- [x] Move physical pre-materialisation outside autograd while binding coefficient/depth-row identity at ordinary consumption.
- [x] Preserve the cached dense allocation for fused QK/QKV binding without a second concatenation.
- [x] Restore non-reentrant checkpointing and the configured `-S` segment size for Premat runs.
- [x] Add single-family and fused-bundle forward/gradient equivalence regressions.
- [ ] Run the focused executable CPU tests in a PyTorch environment.
- [ ] Re-run L12/P6 and then L12/P7 on scruffy against the matched Premat-disabled reference.
