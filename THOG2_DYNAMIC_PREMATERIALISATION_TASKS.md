# THOG2 Dynamic Prematerialisation - Task List

Branch: `THOG2_Dynamic_Prematerialisation_Enhancement`

Base: `master` at `38763e41f31ae3f16f1f9a5609a0a7a63a686fe8`

Authoritative requirements: `THOG2_Dynamic_Prematerialisation_Enhancement_v0.2.docx`

Authoritative plan: `THOG2_Dynamic_Prematerialisation_Enhancement_Implementation_Plans_v0.1.docx`

## Current status

- [x] Create branch from current master.
- [x] Create continuity task and log files.
- [x] Inspect all runtime, CLI, wrapper, checkpoint, telemetry, and Instra seams.
- [x] Attempt disabled-path regression baseline (blocked before collection: local runtime has no `pytest` or `torch`; see log).

## Stage 1 - scheduler, configuration, memory, telemetry

- [x] Add and validate the seven `--premat_*` public options.
- [x] Retire the PLASTIC-only memory threshold in favour of the global GPU buffer.
- [x] Implement lifecycle states and legal transitions.
- [x] Implement strict next-use queue order, no bypass, one-in-flight, and l+1 limit.
- [x] Implement predicted retained/transient memory envelopes.
- [x] Implement current-peak and global-buffer headroom policies without mixing process/device ceilings.
- [x] Implement structured event history and aggregate diagnostics.
- [x] Add Stage 1 deterministic policy/config tests (execution pending a torch-capable environment).

## Stage 2 - fused CUDA execution

- [x] Add exactly one persistent premat CUDA stream per active model device.
- [x] Route fused QKV, O, UP, and DOWN through authoritative candidate call points.
- [x] Implement completion events, deadline waits/fallbacks, ownership, and `record_stream` lifetime handling.
- [x] Give activation-checkpoint recomputation fresh segment-local premat state and preserve autograd tensor ownership.
- [ ] Add Stage 2 tests covering A07-A11 and A16-A19.

## Stage 3 - unfused attention

- [x] Add explicit fused-QK then later-V attention topology.
- [x] Include explicit attention intermediates in memory/admission decisions.
- [x] Add fused/unfused CPU equivalence test (execution pending a torch-capable environment).

## Stage 4 - Instra

- [ ] Expose stable runtime-to-Instra Premat telemetry.
- [ ] Add live l/l+1 Premat view for fused and unfused modes.
- [ ] Add lifecycle/miss visuals, event table, and memory summary.
- [ ] Add Stage 4 tests covering A22-A27.

## Final verification

- [ ] Run focused CPU suite.
- [ ] Run broad non-GPU regression suite.
- [ ] Run available CUDA tests or record the exact unavailable gates.
- [ ] Check THOG marker convention and disabled-path preservation.
- [ ] Update task/log files with exact tests and remaining limitations.
- [ ] Push all commits to the requested branch.
- [ ] Provide the user a download stanza.
