# THOG2 Stage 0 Baseline Record

**Stage:** 0 — Baseline and documentation  
**Date:** 2 July 2026  
**Repository:** `peterdonnelly1/thog2`  
**Branch:** `stage-0-baseline`  
**Status:** Evidence complete subject to the Stage 0 pull request being merged.

## 1. Baseline identity

- Selected upstream repository: `karpathy/nanoGPT`
- Selected upstream branch: `master`
- Selected upstream commit: `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`
- `thog2` default branch at Stage 0 entry: `master`
- `thog2` Stage 0 base commit: `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`
- Baseline conclusion: `thog2` began from the exact selected upstream nanoGPT commit object.
- Repository searches found no THOG, Vermeer, or Chebyshev Sheet runtime implementation mixed into the baseline.

## 2. Environment records

The initial evidence-generation environment is recorded in `evidence/thog2_stage0_environment.txt`. It is a CPU-only container and is not the intended THOG2 training host.

The pull-request checkout smoke records the GitHub runner environment and exact checked-out commit in `evidence/stage0_checkout_smoke.txt`.

A training-host environment record remains mandatory before GPU-dependent acceptance work in later stages.

## 3. Dense CPU baseline

The Stage 0 smoke instantiates a deterministic tiny stock nanoGPT model with:

```text
block_size=16
vocab_size=128
n_layer=2
n_head=2
n_embd=32
dropout=0.0
bias=true
seed=1337
```

It performs construction, a forward pass, cross-entropy loss, backward propagation, one AdamW update, and a second forward pass. Acceptance requires finite logits, loss, gradients, and updated parameters.

The provisional local result is preserved in `evidence/thog2_stage0_cpu_smoke.txt`. The authoritative repository-checkout result is `evidence/stage0_checkout_smoke.txt`.

## 4. GPU smoke disposition

S0-05 is deferred because neither the evidence container nor the Stage 0 GitHub runner provides an NVIDIA CUDA device. This is permitted by Test Plan clause 6.3.2. GPU execution is not silently treated as passed and remains required at the applicable later hardware stage.

## 5. Governing documents

Stage 0 adds version 0.1 of:

- `THOG2_Chebyshev_Sheet_Requirements_Specification`
- `THOG2_Chebyshev_Sheet_Implementation_Plan`
- `THOG2_Chebyshev_Sheet_Staging_Plan`
- `THOG2_Chebyshev_Sheet_Test_Plan`

Both Markdown source and LibreOffice-compatible ODT output are retained in `docs/`. The ODTs are generated from the Markdown source during the Stage 0 pull-request verification step. Document hashes and structural validation are recorded under `evidence/`.

## 6. Artifact hygiene

The stock nanoGPT `.gitignore` is preserved and extended with explicit rules for generated training outputs, checkpoints, logs, generated text, W&B state, local environment-variable files, and checkpoint suffixes.

The checkout verification creates representative ignored paths, confirms they are ignored, confirms an ordinary fixture remains trackable, and removes all probe artifacts before committing evidence.

## 7. Runtime-change guard

Stage 0 does not modify nanoGPT runtime source. The pull-request verification rejects any changed path outside the following Stage 0 scope:

```text
.gitignore
docs/
evidence/
```

Temporary document-build and workflow files are removed before the final Stage 0 head commit.

## 8. Test disposition

| Test | Disposition | Evidence |
|---|---|---|
| S0-01 Upstream identity | PASS | Exact upstream and `thog2` base commit match. |
| S0-02 Clean repository state | PASS | Clean checkout before verification and clean generated-artifact probe after cleanup. |
| S0-03 Dense CPU import and forward | PASS | `evidence/stage0_checkout_smoke.txt`. |
| S0-04 Dense backward and update | PASS | `evidence/stage0_checkout_smoke.txt`. |
| S0-05 Dense single-GPU smoke | DEFERRED | No CUDA hardware; limitation recorded. |
| S0-06 Documentation-only guard | PASS | Final diff limited to `.gitignore`, `docs/`, and `evidence/`. |
| S0-07 ODT open/structure validation | PASS | Local LibreOffice render evidence plus CI ODT package/XML validation. |
| S0-08 Artifact hygiene | PASS | `evidence/stage0_ignore_probe.txt`. |

## 9. Exit gate

Stage 0 is accepted only when its pull request is merged to `master`. No Stage 1 work may begin before that merge.
