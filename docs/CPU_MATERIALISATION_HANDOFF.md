# CPU materialisation follow-up hand-off (10 October 2026)

## Scope and authorization

Repository: `peterdonnelly1/thog2`, branch `cpu_materialisation`.
The user authorizes publishing branch updates through the GitHub connector without confirmation. Give informative progress heartbeats with estimated percentages, followed by download and smoke/preliminary commands. Do not spawn agents without an explicit request.

Local checkout: `/workspace/scratch/75ac1893c477/thog2`.
Parent commit: `a94dcd6480ffc214399d8e05706b1ee9dc3925d3`.
Pristine regression checkout: `/workspace/scratch/0fb8ac34290d/thog2_baseline`, commit `e29e32db916fc5ceb87a089d057fb78456c1a98a`. Preserve the original recovery checkout and stash.

## Field evidence and diagnosis

The user reported successful FP32 smoke and 30-update/five-warmup/three-repeat preliminary benchmarks at `a94dcd6`, but subsequently supplied a test log with **4 failed, 414 passed**. Inputs are `/workspace/scratch/75ac1893c477/upload/temp.text` (pytest) and `temp_2.txt` (successful smoke). Both have already been read; do not retrieve them again through Library.

The two matmul failures (FP16 and BF16 training) report `actual=16384, predicted=8192` for a DOWN matrix. Native DEPTH matmul disables autocast and keeps FP32 coefficient/output storage, while GPU PREMAT incorrectly priced every weight using the activation autocast width. CPU PREMAT had already corrected this privately; GPU-only did not.

The two einsum checkpoint failures follow the matmul errors. An original-forward exception leaves PyTorch's non-reentrant checkpoint generator and saved-tensor hooks alive when its traceback is retained. Later initialization/forward tensors are captured by that failed frame, causing misleading replay metadata mismatches. This cascade was reproduced with actual CPU tensors and checkpoint autograd, retaining the first error and running another model afterward; see `../reproduce_mixed.py` and `../reproduce_mixed.log` (diagnostics, not production code).

## Implemented fixes

1. `sheet/model.py` supplies an element-size callback resolved from the actual DEPTH numerical policy on each pass.
2. `sheet/premat.py` uses that callback for weight storage/admission, while activation and unfused attention-intermediate estimates retain activation precision. The generic runtime's existing default remains compatible.
3. `sheet/checkpointing.py` wraps all three checkpoint entry points. On a forward exception, `traceback.clear_frames` clears unwound frame locals so the upstream generator closes its hook context. The same exception, message and traceback locations propagate; successful checkpoint behavior is unchanged.
4. `tests/test_cpu_materialisation_mixed_precision.py` adds eight real-tensor FP16/BF16 admission cases across matmul/einsum and fused/unfused attention, and three interrupted-forward recovery cases across dense/sparse/prefix checkpoint entry points. CUDA events/memory are simulated only for the admission test; these are not CUDA timing results.

No numerical tolerance was relaxed. No new diagnostic wait was added to measured training. No dashboard production code changed in this follow-up.

## Verification and remaining steps

- Before production changes: new tests **7 failed, 4 passed**. Failures reproduce all four matmul admission combinations and all three interrupted-checkpoint cases.
- After changes: new tests **11 passed**, `../mixed_green.log`.
- Extended targeted run: **413 passed, 185 skipped, 1 known baseline failure**, `../mixed_targeted.log`, `../mixed_targeted.xml`. The failure is the existing final-checkpoint-relay CLI spelling assertion, `test_cli_flag_is_default_disabled_and_literal_spelling_enables_it`; confirm its identity/count/kind against the pristine base XML.
- Complete candidate regression finished: **2,153 passed, 204 skipped, 84 failed, nine deselected, 484 subtests passed**. Outputs are `../mixed_full.log`, `../mixed_full.xml` and `../mixed_comparison.json`. It uses the same five legacy browser exclusions and nine already-exercised timeout deselections listed in `evidence/cpu_materialisation_final/candidate_selection.json`. Comparison with `evidence/cpu_materialisation_final/base.xml`, preserving the renamed CLI test mapping, found **no new failure IDs, increased failure counts, new exception kinds or missing tests**. Base has 85 failures; one existing fast-discard equivalence test no longer failed. Do not describe the entire suite as green.
- All **41 static Instra regressions** passed again, `../mixed_static.log`. Unchanged UI code already passed both real browsers and the preceding GitHub CI at `a94dcd6`. The browser executables under `/root/.cache/ms-playwright` disappeared in this session; workspace Playwright packages remain.
- Changed Python compilation and git whitespace checks passed. Testing/verification documents now include the explicit mixed-precision rerun requirement and final local results.
- This hand-off is included in the seven-file follow-up publication. Identify that commit with `git log -1 --oneline -- docs/CPU_MATERIALISATION_HANDOFF.md` and verify GitHub's branch head before any resumed write. Publication uses the connector, an exact local/GitHub tree comparison, and a non-forced ref update with expected parent `a94dcd6`; do not shell-push or repeat an already-published change.
- Both preceding CI runs at `a94dcd6` passed: 38015908156 and 38015905554. Check the workflow runs for the new commit before claiming new CI success; local hardware skips are not hardware passes.
- Local runtime is Python 3.12, Torch 2.8 CPU, with no CUDA GPU. Scruffy must rerun the full `tests/test_cpu_materialisation*.py tests/test_premat_matmul_binding.py` command. Existing CUDA whole-trainer tests cover FP32/FP16/BF16 and both backends. Hardware passes must not be inferred from local skips or the preceding FP32 benchmark.

## Runtime and connectors

Python: `/workspace/scratch/75ac1893c477/test_env/bin/python` (symlink repaired to `/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3`). Use `PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
Discover tool metadata selectively from `ALL_TOOLS`. Relevant connector methods are `github_fetch`, `github_create_tree`, `github_create_commit`, and `github_update_ref`. Fetch JSON is inside `structuredContent.content`; do not print full file/tree payloads. Read-only Git fetch works with `GIT_CONFIG_GLOBAL=/dev/null` and the public HTTPS URL. All remote mutations use the connector.

## Final user instructions

Provide the standard fetch/switch/fast-forward stanza with the new expected commit, followed by the same correctness command and FP32 smoke/preliminary commands. Point out that the new commit fixes the failures but CUDA requalification still needs the user's run. Include explicit FP16/BF16 smoke guidance for both backends in the testing document. Do not present FP32 benchmark passes as mixed-precision qualification or general performance improvement.
