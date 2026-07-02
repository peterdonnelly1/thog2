# THOG2 Stage 0 Test Results

| Test | Result | Evidence |
|---|---|---|
| S0-01 Upstream identity | PASS | `thog2` baseline and selected upstream nanoGPT commit are both `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. |
| S0-02 Clean repository state | PASS | Verification cleanup removes all generated probes before the final commit. |
| S0-03 Dense CPU import and forward | PASS | `evidence/stage0_checkout_smoke.txt`. |
| S0-04 Dense backward and update | PASS | Finite gradients and at least one changed parameter tensor are recorded in `evidence/stage0_checkout_smoke.txt`. |
| S0-05 Dense single-GPU smoke | DEFERRED | The evidence container and GitHub runner expose no NVIDIA CUDA device; the deferral is explicit and remains open for the later GPU stage. |
| S0-06 Documentation-only guard | PASS | The final workflow rejects any changed path outside `.gitignore`, `docs/`, and `evidence/`. |
| S0-07 ODT structure validation | PASS | `evidence/stage0_odt_validation.txt` and `evidence/stage0_document_sha256.txt`. |
| S0-08 Artifact hygiene | PASS | `evidence/stage0_ignore_probe.txt`. |

Stage 0 is complete only after the pull request containing these results is merged to `master`.
