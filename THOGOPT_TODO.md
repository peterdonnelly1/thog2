# thogopt implementation tasks

## Completed: CUDA OOM correction (published; GPU rerun outstanding)

Baseline: 8e9beeb83826cb68db48297bf09956b001683568. User log: first update 5.6s, GPU candidates 2905MiB, next backward requests 2.30GiB with 1.58GiB free and 5.52GiB reserved/unallocated. Keep D=1024 and the existing A/B baselines. Candidate adoption can pin split allocator segments; fix storage lifetime before reducing model size.

- [x] Read crash attachment and verify branch through GitHub connector.
- [x] Stabilize persistent moment storage and avoid candidate placement during state initialization.
- [x] Add storage-lifetime, initialization, rejection and resume regression coverage; 110 passed, 4 CUDA skipped.
- [x] Document actual GPU evidence and remaining hardware validation limits in docs/thogopt/README.md.
- [x] Publish through GitHub connector on the same branch; feature b743bebc7b942686f54e05c3444ee7b3da6b56cf verified remotely. Download stanza included in delivery.

Correction: any active missing state forces a host transaction; validated CUDA initialization releases unused cache before creating stable history buffers. Routine commits copy into the existing buffers; GPU candidates remain transient. Ordinary scratch is function-local. First fresh timing line should show gpu_candidates=0MiB; later lines can use GPU. Test evidence: docs/thogopt/oom_followup_python_tests.txt. No representative GPU rerun yet. Runtime: ../thogopt_fix_venv/bin/python.

Next external action: rerun the same C configuration at D=1024 on scruffy; preserve A/B. Collect progress/timing lines after 10-20 updates or any traceback. No additional implementation is planned until that evidence arrives. No optimizer reset required.


Progress estimate: Implementation complete and pushed: 282ae4677bdbd611c62710a4a8bd29688d4af250. Verified remotely through GitHub connector. 96 Python tests and 22 JavaScript suites passed.

Repository: peterdonnelly1/thog2. Branch: instra_weight_inspector.
Starting remote commit: e36fe2330d849ccee93514fb5349a21d6b3bea21.
Starting tree: bb97f4d70832659f9308e7f067ccd012d1ed98d4.
Local preservation tag: pre_thogopt_20260902.
Spec: ../THOG2_thogopt_Requirements_Specification_v0.1.docx.

- [x] Verify remote branch using GitHub connector and create isolated sandpit with exactly matching tree.
- [x] Preserve starting commit locally and establish this task list and THOGOPT_WORK_LOG.md.
- [x] Core optimizer: independent polynomial moments, constrained nonnegative second-moment fit, layer-space scaling, projection, bias correction, decay.
- [x] Raw-gradient capture, accumulation, reduction, unscale/clipping, bounded device workspaces and transactional steps (model verified; DDP runtime blocked by environment).
- [x] Configuration/help/wrappers/startup validation, metadata and checkpoint/resume support (explicit reset fork verified).
- [x] Sampled same-gradient reference and dense-AdamW instrumentation (verified).
- [x] Instra history groups, curves, quantities, matrix inspection and aligned differences (verified).
- [x] Numerical parity, compressed fit, accumulation, skip, restart and model integration tests.
- [x] Existing optimizer, SGD parity, instrumentation and JavaScript regression checks.
- [x] Benchmark/release evidence and explicit remaining limitations.
- [x] Publish exact reviewed tree through GitHub connector and verify remote branch. Final response includes download stanza.
- [ ] Publish preservation tag if connector supports tag creation; currently no tag-create action is exposed.

Handoff rule: update this file and THOGOPT_WORK_LOG.md after each meaningful milestone. Do not overwrite another sandbox. All remote repository interactions use the GitHub connector. No agents were requested.

Remaining external verification: CUDA/AMP/NCCL, representative OpenWebText convergence and throughput, and rendered browser QA. See docs/thogopt/validation.md for the exact environment blocks.


## Active follow-up: slowdown and missing System group

Starting commit: f1b67b12ec8aad2ae38e1eadd3e2b16e7ce5c796. Same branch, GitHub connector only. Progress: complete. Published feature commit 5d5356e9189577279ee310ea41c3bbddd84e7f3d, verified through GitHub connector.

- [x] Inspect reported 10.0s/update vs 4.9s/update screenshots and verify remote/local baseline.
- [x] Remove full-history identity fitting; increase bounded tile size; eliminate redundant weight transfers and FP64 norm temporary.
- [x] Keep System visible with missing/loading/error explanations and improve local W&B file discovery.
- [x] JavaScript regression suite: 23/23 passed, including new System visibility/recovery test.
- [x] Numerical/accumulation/checkpoint/rejection and backend regression tests: 106 passed.
- [x] CPU before/after measurement: optimizer time -14.5%, whole-step time -9.3%; GPU unmeasured.
- [x] Update documentation and publish via GitHub connector; remote SHA and complete tree verified. Download stanza supplied with delivery.

Verification interpreter: ../thogopt_fix_venv/bin/python (Python3.12.3, PyTorch2.8.0, CPU execution). Previous ../thogopt_venv interpreter missing in this runtime. Old package files remain intact. Baseline optimizer saved outside repo at ../thogopt_before.py for exact before/after comparison. No CUDA device accessible; do not claim measured GPU speedup or exact cause of System absence on user's machine.

## Active follow-up: remaining 7.9s/update overhead

Starting commit bf18073a76a9b3310fee328ff9365498f522740b. Progress complete. Feature c516a3eb2cf7dc9595acccbd4dc18d2acccccb89 published and verified. User GPU measured previous fix:10.0 ->7.9s/update, still vs4.8-4.9s SGD.
- [x] Inspect host staging, transaction copies and synchronization.
- [x] Implement two-buffer asynchronous staging, memory-budgeted device candidates and last-update timing output.
- [x] Numerical/restart/rejection/staging tests:108 passed,4 CUDA skipped; budget suite rerun after final refinement. CPU timings essentially unchanged; GPU-specific changes unmeasured.
- [x] Document limitations and push same branch via connector; remote tree and branch verified. Fetch stanza included in delivery.
Baseline optimizer: ../thogopt_before_transfer_fix.py. Runtime: ../thogopt_fix_venv/bin/python. CUDA unavailable locally. No agents requested.
