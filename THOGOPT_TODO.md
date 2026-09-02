# thogopt implementation tasks

Progress estimate: 98%. Implementation and verification complete (96 Python tests, 22 JS suites). Active work: GitHub publication.

Repository: peterdonnelly1/thog2. Branch: instra_weight_inspector.
Starting remote commit: e36fe2330d849ccee93514fb5349a21d6b3bea21.
Starting tree: bb97f4d70832659f9308e7f067ccd012d1ed98d4.
Local preservation tag: pre_thogopt_20260902.
Spec: ../THOG2_thogopt_Requirements_Specification_v0.1.docx.

- [x] Verify remote branch using GitHub connector and create isolated sandpit with exactly matching tree.
- [x] Preserve starting commit locally and establish this task list and THOGOPT_WORK_LOG.md.
- [x] Core optimizer: independent polynomial moments, constrained nonnegative second-moment fit, layer-space scaling, projection, bias correction, decay.
- [x] Raw-gradient capture, accumulation, reduction, unscale/clipping, bounded device workspaces and transactional steps (model/DDP verification pending).
- [x] Configuration/help/wrappers/startup validation, metadata and checkpoint/resume support (explicit reset fork verified).
- [x] Sampled same-gradient reference and dense-AdamW instrumentation (verified).
- [x] Instra history groups, curves, quantities, matrix inspection and aligned differences (verified).
- [x] Numerical parity, compressed fit, accumulation, skip, restart and model integration tests.
- [x] Existing optimizer, SGD parity, instrumentation and JavaScript regression checks.
- [x] Benchmark/release evidence and explicit remaining limitations.
- [ ] Publish exact reviewed tree through GitHub connector, verify remote branch, provide download stanza.
- [ ] Publish preservation tag if connector supports tag creation; currently no tag-create action is exposed.

Handoff rule: update this file and THOGOPT_WORK_LOG.md after each meaningful milestone. Do not overwrite another sandbox. All remote repository interactions use the GitHub connector. No agents were requested.
