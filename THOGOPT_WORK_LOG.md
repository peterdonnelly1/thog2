# thogopt work log

## CUDA OOM follow-up: active diagnosis (35%)

Latest baseline 8e9beeb83826cb68db48297bf09956b001683568 verified through connector. Read user attachment Pasted text(20260902-233352).txt directly from uploaded scratch copy. Actual RTX4090 16GB run: first update 5.6s, stage3.139s prepare0.396s candidate0.245s commit0.005s, 2905MiB GPU candidates, async staging enabled. Next backward failed requesting2.30GiB; free1.58GiB, allocated7.81GiB, reserved/unallocated5.52GiB. One completed update is not a sustained throughput measurement.

The preceding change adopted GPU candidate moments as persistent histories. This can strand unused space in split allocator segments needed by the next backward; summed free/cached bytes do not establish contiguous allocation feasibility. This is an identified lifetime defect and a plausible explanation of this crash, not a GPU reproduction here. User asked about D=768 and rerunning A/B; advised keeping D=1024 while correcting the regression. Scope: stable histories, initialization placement, targeted regression coverage, connector delivery; no unrelated optimizer or Instra changes.

## CUDA OOM follow-up: correction verified (85%)

Any active parameter missing persistent state now forces the entire transaction onto the host. Nonparticipating parameters do not block future GPU candidates. Before allocating missing CUDA histories, synchronize completed candidates (existing boundary) and release unused cached blocks; this happens only on initialization, not routine updates. Permanent histories are separately allocated and reused; device candidates commit via device-to-device copies instead of becoming permanent state. Moved ordinary AdamW calculations unchanged into a method to release their scratch before state allocation. Asynchronous gradient transfer and existing automatic history dimensions remain unchanged.

New CPU regressions cover candidate-history lifetime and stable addresses for compressed plus ordinary state, and injected late-state allocation failure preserving weights, moments, storage and counters. Expanded existing CUDA test covers first-update host placement, subsequent GPU candidates, stable addresses, one cache cleanup and atomic rejection with interleaved64MiB temporary allocations; it remains unexecuted here. Consolidated CPU regression suite:110 passed,4 CUDA skipped in15.89s. Includes 500-update and full-model AdamW equivalence, SGD parity, real public-runner resume/reset fork, snapshots, telemetry and console output. Evidence: docs/thogopt/oom_followup_python_tests.txt. No UI code changed. No claim of GPU OOM reproduction or sustained speed measurement. Next: connector publication and exact remote/local hash verification.

Regression sensitivity confirmed against the exact parent optimizer module: the candidate-history release assertion fails because old candidate tensors remain alive as permanent histories. See docs/thogopt/oom_followup_parent_regression.txt. Final transfer/storage suite passes7 tests with4 CUDA skips after simplifying the test instrumentation to use a method present on both versions; production code unchanged since the consolidated110-test run. See docs/thogopt/oom_followup_storage_tests.txt. Ready to publish the reviewed correction.

## CUDA OOM follow-up: published and verified

Feature b743bebc7b942686f54e05c3444ee7b3da6b56cf, parent8e9beeb83826cb68db48297bf09956b001683568, tree6c7bb021bbc2d3da7f9b3791574bf4b996e342ff. All9 uploaded blob hashes and the full tree matched the local index. Non-forced GitHub connector update succeeded; a subsequent connector ref read verified the branch. Local commit object reconstructed and hash-verified against connector metadata; local and tracking refs aligned. This handoff-only follow-up records completion. No shell GitHub network operation was used.

User should retain D=1024 and existing A/B snapshots, pull this branch and restart/resume the same C run without resetting the optimizer. First fresh step uses host candidates and includes initialization; later steps retain budgeted GPU candidates and asynchronous gradient capture. Next required evidence is sustained GPU execution past the next backward, then timing after10-20 updates. Local result is110 passing CPU tests and4 CUDA skips, not proof that the actual4090 OOM is resolved. No unrelated optimizer/UI changes and no further code work planned before new GPU evidence.


## 2026-09-02: baseline and scope (5%)

User authorized implementation of the delivered v0.1 specification on the same branch, tagging the previous version, GitHub push and a final download stanza. Frequent informative percentage heartbeats and durable top-level handoff files are required.

GitHub connector verified instra_weight_inspector at e36fe2330d849ccee93514fb5349a21d6b3bea21, tree bb97f4d70832659f9308e7f067ccd012d1ed98d4. Previous sandpit's staged tree exactly matched. Created isolated thog2_sandpit here using local objects; reconstructed the unsigned remote commit from connector metadata and verified its exact Git object hash before checkout. Earlier sandpits remain untouched.

Created local tag pre_thogopt_20260902 at the starting remote commit. The exposed connector supports blob/tree/commit creation and branch updates but has no tag-creation action. Do not claim the tag is remote unless verified. Do not substitute a branch for a tag.

Initial integration seams: sheet/optimizer_factory.py; DepthTrajectory materialisation; sheet/trainer_step.py; Stage6Trainer and run_thog2_owt.py; depth-weight instrumentation and local dashboard assets. No AGENTS.md files found in available workspace repositories.

Spec contract: raw materialised layer gradients, two independent histories (auto P and min(2P-1,L)), layer-space AdamW division followed by production-basis pseudoinverse projection, nonnegative layer-sample least-squares second-moment fit, persistent compact state. Retain existing optimizer paths and existing SGD parity. Capture accumulated/reduced gradients before squaring; do not silently substitute projected coefficient gradients. Separate numerical correctness from empirical training performance.

## Current next actions

Read exact training-step, materialiser and capture interfaces; implement a small testable numerical core first. Keep new runtime work isolated to thogopt selection. Progress and verification commands/results follow below.

## Numerical core and training integration (30%)

Added sheet/thogopt_math.py and sheet/thogopt.py. History basis uses stable QR, with identity sample storage at H=L. Constrained second-moment fitting uses Hildreth dual coordinate descent and explicit convergence/roundoff checks. Gradients are staged on CPU; history evaluation and distributed reduction are tiled. Candidate parameters/histories are staged before commit; missing persistent buffers are allocated before parameter mutation.

Installed a dedicated CPU verification environment at ../thogopt_venv (PyTorch 2.8.0+cpu, pytest, scipy). Command: ../thogopt_venv/bin/python -m pytest -q tests/test_thogopt.py. Initial result: 9 passed in 8.70s, including 500-step FP64/FP32 AdamW parity, independent SciPy constrained-fit checks, raw microbatch accumulation, P<L projected oracle, restart, rejection without mutation and sampled reference state.

Factory, trainer, public CLI/wrapper, config persistence and lifecycle comparison now recognise thogopt and independent history budgets. Existing default settings are omitted from persisted config to avoid changing old optimizer identities. Public DEPTH does not use the direct-factorised MLP bypass; existing direct MLP configuration remains usable. Initial thogopt rejects geometry mutation and partial layer dropout.

Next: run tiny-model/checkpoint integration tests, add captured history storage/dashboard endpoints and Instra groups. GPU and real OpenWebText throughput have not been tested; CPU environment only.

## Training and history integration (55%)

Full Stage4 DEPTH model, two accumulated microbatches and exact checkpoint restart pass with THOG2_FAST_DISCARD=true and false. Confirmed five legacy tests/test_optimizer_wrapper.py failures independently on untouched baseline; current suite has exactly the same five failures. No unrelated test expectations altered.

Added passive same-gradient sampled references, actual DENSE AdamW state sampling, compact/full snapshot retention, read-only dashboard endpoints, Momentum and Scaling groups, quantity selectors, curve/reference/error inspection and bounded full-matrix windows. Full snapshot cadence is independent and opt-in via --instrumentation__optimizer_histories__full_matrix_every_n_steps; ordinary sampled capture follows existing weight settings. Need verify these new paths, UI compatibility, DDP and model parity before publish.

## Verification and delivery preparation (78%)

Passed 500-update FP64/FP32 synthetic AdamW parity and the 100-step FP32 complete model trajectory. Added FP64 model trajectory coverage, which exposed an existing hard-coded FP32 layer-normalization weight cast; its FP64-only path now preserves dtype while ordinary execution is unchanged. New capture/backend tests verify all six families, sampled and full matrices, signed differences, retention cleanup and passive training. Explicit --reset-optimizer is fork-only and preserves model/data position while resetting bias counters. New DOM harness checks history groups, run selection, latest queries, differences, decimal precision and standard/maximized inspection actions.

34 integration/recovery/local-chart tests passed. CPU Gloo could not initialize: Operation not permitted at gloo TCP device.cc:186, including loopback retry; no distributed execution claim. Browser binary download timed out; full rendered browser QA is unavailable here. A jsdom DOM harness is being used alongside the pre-existing Instra JavaScript suites. GPU/OpenWebText acceptance remains unmeasured.

Current work: finish FP64 model result, run the synthetic CPU benchmark and legacy JS suites, audit changed files, document practical limits, then publish through GitHub connector. Remote tag still unsupported by exposed connector; local baseline tag exists.

All 22 existing/new Instra JavaScript regression programs pass. The CPU microbenchmark (L=8,P=3,H_m=3,H_v=5,D=16,20 measured steps) reports 50% covered-family moment storage savings (98,304 vs 196,608 bytes). thogopt is slower on this tiny workload; fitting dominates its optimizer time. Exact numbers are in docs/thogopt/cpu_benchmark.json.

FP64 model comparison exposed differing existing embedding decay policies: THOG excludes embeddings, the generic DENSE optimizer decays them. The reference fixture now aligns physical parameter groups explicitly; production group policies are preserved. This is a required condition for optimizer-equivalence comparisons, not a numerical-tolerance relaxation.

## Final validation (92%)

FP64 and FP32 full-model AdamW comparisons now both pass for 100 updates with matched physical decay groups. The FP32 100-update SGD regression passes. The real public runner test passes fresh training, checkpoint resume, full/sample history capture and a fork with explicit optimizer reset and a changed momentum-history count. The shell wrapper dry-run forwards both exact thogopt option names and the full-matrix cadence correctly.

A combined targeted Python run passed 92 tests before adding the SGD and public lifecycle cases. All 22 JS regression programs passed; final consolidated rerun follows. Documentation and original requirements are in docs/thogopt. Remote branch was rechecked through the connector and remains at the preserved starting commit. No remote mutation yet.

## Ready to publish (96%)

Final combined targeted Python suite: 94 passed in 15.38s. Additional direct half-parameter test verifies persistent FP32 moments without small-square underflow. All 22 JS regression programs passed; the new UI harness was rerun after the final projected-error display change and passed. Shell syntax, exact argument forwarding and git diff whitespace checks pass.

Implementation, original v0.1 requirements, usage notes, numerical conditions, CPU benchmark and environment limitations are under docs/thogopt. GitHub publication will use immutable blobs/tree/commit then a non-forced update of instra_weight_inspector, retaining the verified starting commit as parent.

Final checkpoint review added thogopt-only GradScaler persistence, including its growth tracker. A CPU GradScaler fixture now reproduces the uninterrupted scale increase and exact model weights after restart. Existing optimizer checkpoint payloads remain unchanged. Captured missing UI values also remain unavailable rather than coercing null to zero. The final full targeted suite is rerun after these changes.

Final consolidated verification: 96 Python tests passed in 15.27s. All 22 JavaScript suites passed, including the final updated history DOM harness. Remote branch remained unchanged at e36fe2330d849ccee93514fb5349a21d6b3bea21 immediately before publication.

## Published and verified

Implementation commit 282ae4677bdbd611c62710a4a8bd29688d4af250 was created with parent e36fe2330d849ccee93514fb5349a21d6b3bea21 and tree f2c961ff8594cbea7f537ad39891bd971b121a2e. Every uploaded blob SHA and the complete remote tree SHA matched the local staged tree. GitHub connector updated instra_weight_inspector without force; a fresh connector read verified the branch at the implementation commit. Local branch/tracking refs were aligned by reproducing and hash-verifying the exact connector commit object.

This final handoff-only follow-up records delivery. The implementation is complete; no further code changes are planned. The preservation tag pre_thogopt_20260902 remains local because the connector has no tag-create operation. The final download stanza includes commands for the user to publish it. No shell GitHub write was used.

Verification limits remain explicit: no GPU; Gloo initialization blocked; browser binary download unavailable. Do not interpret CPU storage savings or curve parity as demonstrated long-run OpenWebText convergence or GPU speed.

## Follow-up investigation: runtime overhead and missing System tab

User reports ~2x slowdown in first actual GPU run and no System tab. Screenshot confirms 10.0s vs 4.9s/update, P=L=16, H_m=H_v=16, 6 microbatches. Remote branch still f1b67b12ec8aad2ae38e1eadd3e2b16e7ce5c796 and local working tree was clean. Initial code inspection: synchronous GPU-to-CPU raw-gradient copies on every layer/microbatch; per-tile CPU/GPU transaction staging; identity history bases still sent through general matrix products/constraint checks. These are implementation overhead, not evidence that compact-history math inherently doubles training time. Comparison screenshot is SGD, not conventional AdamW. Need inspect UI source before attributing disappearance.

Follow-up 45%: full H=L moments now use direct recurrences and bypass the identity matrix products and constrained solver; default tile is 1,048,576 layer values (previously 131,072), parameter candidates read device weights directly, finite checks use one host decision rather than three, norm reduction avoids a full FP64 square buffer. Persistent history/checkpoint conventions remain unchanged. Raw capture and transactional moment transfers remain synchronous: do not claim their elimination. System visibility issue is conditional W&B group discovery, not a removed top-level tab. New UI regression keeps System discoverable with precise missing/loading/read-error reasons and restores normal content when metrics arrive. W&B source discovery now honors recorded SDK run.dir and configured/offline roots; recorded directory persisted prospectively. Exact cause on user's machine remains unobserved. Old test interpreter disappeared across session runtime; restoring Python3.12/PyTorch2.8 environment through available package index after external Python/CPU-wheel endpoints timed out. New System DOM regression passed.

Follow-up 90%: restored test runtime (Python3.12.3, PyTorch2.8.0+cu128 executing on CPU). Consolidated suite passed 106 tests in22.31s; 23/23 JS regression suites passed. Independent history counts, tile boundaries, full-history negative-state rejection, full model AdamW/SGD parity, accumulation, checkpoint and real CLI resume/reset fork all covered. Custom/offline W&B source discovery tests passed. CPU full-history before/after measurement (L=P=16,width128,10 measured updates,one thread) gives optimizer 0.7859s ->0.6720s (-14.5%) and total1.6080s ->1.4581s (-9.3%). No GPU speedup claim. Evidence under docs/thogopt/runtime_followup_*. Existing thogopt checkpoint configuration is compatible; no optimizer reset needed. System data-file absence on user's machine cannot be established from Logs screenshots; UI now explains absence rather than hiding the group. Ready to upload validated changes via connector.

Follow-up published: feature commit 5d5356e9189577279ee310ea41c3bbddd84e7f3d, tree a02668277af7987c946e9167a5983cfe92f31bb3, parent f1b67b12ec8aad2ae38e1eadd3e2b16e7ce5c796. All15 uploaded blob hashes and the complete tree hash matched the validated local index. Non-forced connector branch update succeeded and a separate connector ref read confirmed the feature SHA. Local commit object reconstructed/hash-verified and refs aligned. This final handoff-only commit records completion. No further implementation planned until the user's GPU measurements identify remaining costs. Existing thogopt checkpoints can resume without reset; restart Instra and reload its browser page.

## Remaining overhead follow-up

User reports 7.9s/update after the first fix, still ~3s/update above preceding SGD run. No claim that all3s is optimizer-specific (SGD comparison). Remote/local baseline bf18073a76a9b3310fee328ff9365498f522740b. Focus now on synchronous raw-gradient D2H copies per layer/microbatch and candidate transaction transfers; preserve true raw gradients, compact persistent moments, checkpoint compatibility and rejection behavior. Need expose actual stage timing so subsequent GPU measurements identify costs instead of relying on CPU extrapolation.

55%: implemented HostGradientTransfers with two pinned buffers, dedicated CUDA copy stream, producer wait, source record_stream lifetime protection, FIFO event-gated CPU consumption, and explicit drain/discard at prepare/zero_grad. Raw gradients still accumulate on CPU and are reduced/unscaled/clipped before squaring. Device candidate budget=min(physical free+unused allocator reserve, historical peak-current allocation)-missing persistent states-workspace reserve. Budgeted candidates retain weights/moments on GPU through validation; moments become committed state directly. Host fallback retained. Execution-only THOG2_THOGOPT_ASYNC_STAGING=false and THOG2_THOGOPT_TRANSACTION_STORAGE=host allow isolation; defaults true/auto. New THOGOPT last update timing line at existing log cadence shows stage/prepare/candidate/commit, actual device candidate MiB and async state. Initial33 numerical/integration tests passed; expanded suite and queue/budget tests running. No CUDA execution available; GPU tests included and explicitly skipped locally.

Budget refinement during review: a strict historical-peak-only budget could retain the host bottleneck for small test models with abundant idle VRAM. Auto candidates may therefore use released peak headroom OR half of currently available (free+unused allocator reserve) memory, whichever is larger, bounded by actual availability, minus missing-state and workspace reserves. This may raise transient optimizer peak allocation, but creates no permanent additional history replica and releases candidate scratch before the next forward. Execution override host remains available. Raw CPU gradient buffers now reuse storage and overwrite first contributions; deferred copy/add order and discard semantics are tested.

85% verification:108 tests passed,4 CUDA cases skipped in15.25s. Queue/budget tests rerun after final budget refinement. Three console-format failures confirmed on untouched bf18073 parent (3failed,4passed); no unrelated expectation changes. CPU before/after10updates,L=P16,width128: total1.4778 ->1.4865s, optimizer0.6753 ->0.6756s (essentially unchanged; CUDA-specific paths not exercised). No GPU speedup claim. Documents/evidence now include transfer design, transient memory tradeoff, execution overrides and timing-field definitions. Default device candidates may use spare GPU memory beyond past allocation peak, with explicit memory reserves; persistent state remains compact. Ready for connector publication after final diff review.

Published transfer optimization: c516a3eb2cf7dc9595acccbd4dc18d2acccccb89, tree74c5739644f94e2e74ee773d64df0e39baf7a4cf, parentbf18073a76a9b3310fee328ff9365498f522740b. All12 blob hashes and full tree verified against local index; non-forced GitHub connector branch update verified through separate ref fetch. Local commit object reconstructed and hash-verified. Final handoff commit only updates these files. User should pull and resume without optimizer reset; next GPU report should include the new THOGOPT last update timing line. No measured GPU speedup claimed.
