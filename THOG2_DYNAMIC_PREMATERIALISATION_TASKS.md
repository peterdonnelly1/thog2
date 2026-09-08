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

## CUDA-host acceptance still required

- [ ] Execute fused enabled-versus-disabled forward, loss, and trajectory-gradient equivalence.
- [ ] Execute unfused enabled-versus-disabled forward, loss, and trajectory-gradient equivalence.
- [ ] Repeat both numerical checks with activation checkpointing enabled.
- [ ] Capture an Nsight or PyTorch-profiler trace proving real overlap and no forbidden global synchronization (A19).
- [ ] Measure representative steady-state step time and confirm a material benefit before considering default-on (A21).
- [ ] Stress both headroom modes on a shared GPU and confirm observed peaks stay inside the admitted envelope.

These CUDA items cannot run on the current host (`torch.cuda.is_available()` is false). The four numerical/checkpoint cases are committed as explicit skipped tests so they become executable on a CUDA host. The feature remains default-off, as required until A19 and A21 pass.

## Publication

- [ ] Push the completed tree through the GitHub connector.
- [ ] Verify the remote branch head and provide the download stanza.
