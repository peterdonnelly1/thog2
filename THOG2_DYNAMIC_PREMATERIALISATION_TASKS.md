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

## 2026-09-08 strict release and launch correction

- [x] Record L11/P6 success, L12/P6 OOM, and the approximately 442 MiB free-memory observation with a 1.0 GiB Premat admission buffer.
- [x] Fix Premat detection beyond the wrapper separator so the default expandable allocator policy is actually active.
- [x] Correct the startup checkpoint-segment diagnostic to report configured `-S`.
- [x] Prevent any auxiliary Premat scheduler from starting inside non-reentrant checkpoint recomputation.
- [x] Move dense-weight early discard before the runtime consumption notification at every Premat deadline.
- [x] Gate the next auxiliary candidate on a recorded main-stream completion event for the prior consumer.
- [x] Remove the Instra layer separator, enlarge the pipeline, and add a lifecycle key.
- [x] Show whether the current device-free value meets or breaches the Premat admission buffer.
- [x] Run Python compilation, shell syntax, JavaScript syntax, and whitespace validation.
- [ ] On scruffy, verify the startup row says `checkpoint_recompute_segment=4` and `cuda_allocator=expandable_segments:True`.
- [ ] Re-run L12/P6, then L12/P7, and compare against a current Premat-disabled L12/P7 control.
- [ ] Confirm the Instra key, borderless pipeline, and buffer-margin indicator.

## 2026-09-08 final field correction

- [x] Remove the erroneous quadratic fused-attention envelope charge.
- [x] Restore ordinary autograd for main fallback and checkpoint replay.
- [x] Make the whole-model forward the sole Premat scheduler owner.
- [x] Add step, exact candidate labels, arrows, larger boxes/key, and aggregate success counters to Instra.
- [ ] Confirm with `-K flash2` that headroom produces admissions/hits and no `AccumulateGrad` stream warning.
- [ ] Compare a matched Premat-disabled peak-memory control.
- [x] Fix the first successful-admission checkpoint mismatch (131 forward tensors versus 128 replay tensors).
- [ ] Re-run L32/P10 with `-K flash2` and confirm first backward completes.
- [x] Match the saved depth-row shape and autocast dtype (`[1,1,P]`, BF16 in the reported run) to ordinary einsum replay.
- [x] Add a zero-copy consumer-stream autograd anchor before persistent DEPTH leaf accumulation.
- [x] Make CUDA Premat equivalence tests reject the AccumulateGrad stream warning rather than suppress it.
- [x] Diagnose the fixed-terminal-layer Instra display as latest-row overwrite between browser polls.
- [x] Carry exact current/next logical layer indices on each bounded Premat event.
- [x] Reconstruct and pseudo-scroll sampled updates from unseen browser-side events.
- [x] Hold every rendered Premat lifecycle state for at least 250 ms at all playback speeds.
- [x] Confirm from Peter's field observation that a candidate reached `AVAILABLE` and then `CONSUMED`.
- [ ] On scruffy, confirm Instra pseudo-scrolls through exact layers and visibly holds the striped `CONSUMING` frame.
- [ ] On scruffy, confirm the consumer-stream anchor removes the AccumulateGrad stream warning.

## 2026-09-08 complete-microstep all-layer Instra view

- [x] Capture only the first original-forward accumulation microstep at each selected `-l` optimizer update.
- [x] Publish only complete passes and prevent adjacent microstep event histories from mixing.
- [x] Show every layer simultaneously, 1-based with layer 1 at the bottom and responsive row height plus scrolling.
- [x] Replay meaningful states at a user-controlled 0.25 s default; make terminal outcomes persistent but zero-duration.
- [x] Pause/resume the active capture, hold its completed grid for one second, then jump to the newest completed capture.
- [x] Distinguish full hits, waited partial hits, and main materialisation with a complete persistent key.
- [x] Rename `O` to `ATTN O` and use the agreed fused/unfused attention and MLP matrix nouns everywhere.
- [x] Move complete per-matrix data behind a magnifying-glass inspector with a vertically scrollable table and Outcome column.
- [x] Make the dashboard poll incrementally for only the newest unseen completed capture.
- [x] Add focused runtime, persistence, API, reducer, markup, layout, and visual-contract regressions.
- [x] Run all dependency-free static and direct harness checks available in this container.
- [ ] On scruffy, confirm the first sampled microstep replays coherently for all layers and the inspector matches its final grid.
- [ ] On scruffy, confirm every green `AVAILABLE` follows visible blue/white `MATERIALISING`, waited paths are orange, and main materialisation is red.
- [x] Rename the view `Premat Recapitulation - Step N` and eliminate state-text-driven control jitter.

## 2026-09-09 pending-release starvation correction

- [x] Correlate the L6/P3 and L4/P2 initial-green-prefix/all-red-tail captures with the scheduler release gate.
- [x] Remove zero-byte main-path pending releases.
- [x] Allow independent candidate admission while a prior Premat tensor remains pending release.
- [x] Keep each pending Premat tensor strongly referenced and memory-charged until its main-stream completion event resolves.
- [x] Add regressions for non-blocking charged Premat release and main-fallback non-gating.
- [ ] Confirm on scruffy that Premat continues attempting candidates beyond the first main-path fallback.

## 2026-09-09 one-matrix-lead scheduling

- [x] Make next-plus-one-and-beyond the default matrix candidate policy.
- [x] Ensure this primes one continuous pipeline rather than alternating main and Premat materialisations.
- [x] Add `--premat_allow_premat_of_immediate_next_matrix` as an opt-in restoration of the previous zero-lead policy.
- [x] Propagate the policy through wrapper, configuration, resume/artifact identity, startup reporting and telemetry.
- [x] Add and run a dependency-free regression harness for default and opt-in scheduling.
- [ ] On scruffy, compare tok/s, full hits, waited hits and main materialisations between the default and immediate-next option under matched steady-state headroom.

## 2026-09-09 original next-layer-only scheduling

- [x] Restrict Premat targets to the ordered matrices of logical layer `l+1`.
- [x] Continue reconsidering that next-layer queue at every current-layer early-discard event.
- [x] Remove the experimental immediate-next flag from the CLI, wrapper, runtime and persistent configuration identity.
- [x] Report `next_layer_only` consistently in runtime telemetry, canonical metadata and startup diagnostics.
- [x] Extend Instra state duration to 0.025--2.000 seconds and lowercase its label.
- [x] Replace the terse headroom label with the explicit active memory rule and configured buffer.
- [x] Pass all available static and dependency-free focused checks.
- [ ] On scruffy, confirm layer 1 is the startup bubble and later layers become green when auxiliary compute throughput permits.

## 2026-09-09 global-buffer allocator-cache correction

- [x] Diagnose the L12/P8 zero-admission capture as physical headroom consumed by unused PyTorch allocator cache.
- [x] Preserve conservative full-envelope admission rather than again assuming aggregate cached bytes are contiguous and auxiliary-stream-reusable.
- [x] Enable native allocator garbage collection at 80% capacity alongside expandable segments for default Premat launches.
- [x] Preserve explicit user `PYTORCH_CUDA_ALLOC_CONF` overrides.
- [ ] Confirm on scruffy that the startup row reports both allocator settings and later captured microsteps admit next-layer candidates.
