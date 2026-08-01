# THOG2 Coupled Field Machine HYPERBLOCK

**Implementation Status and Continuation Record**  
**Version 0.1 — 1 August 2026**  
**Branch:** `COUPLED_FIELD_MACHINE_HYPERBLOCk`  
**Base:** `MODIFIED_FAST_DISCARD_FALSE_CASE` at `c38ef70c850a55a2f12556c72ee64b259bb1fbae`

## Purpose

This file is the hand-off record for continuing the implementation in another chat or checkout. It records what exists, what has actually passed, what remains provisional, and the next safe sequence of work. The requirements specification remains authoritative for concept and mathematics; this file is authoritative for branch status.

## Implemented

- New `sheet/hyperblock/` package:
  - `plan.py`
  - `basis_provider.py`
  - `materializer.py`
  - `trajectory.py`
- One direct-sum coefficient system:
  - `common`
  - `attention`
  - `mlp`
- Fixed non-breathing support with duplicate constant branch modes omitted.
- Independent retained orders for:
  - `WEIGHT_FAMILY_COMMON`
  - `WEIGHT_FAMILY_ATTENTION`
  - `WEIGHT_FAMILY_MLP`
  - `DEPTH`
  - `D_MODEL`
  - `MLP_HIDDEN`
  - `ATTENTION_HEAD`
  - `ATTENTION_HEAD_CHANNEL`
- Existing THOG2 registered basis implementation reused through `RegisteredAxisBasisProvider`.
- Reference `einsum` and staged `tensordot` materialisers.
- Exact nanoGPT routing for packed Q/K/V, attention output, MLP_UP and MLP_DOWN.
- `CoupledFieldTrajectory` integration into `SheetGPT` without removing legacy trajectory paths.
- Conventional per-layer LayerNorm and bias parameters for v0.
- Analytic orthogonal-mode variance-split initialization.
- Existing materialize-once/project-once retained-materialisation lifecycle integration.
- Public `--hyperblock` CLI and shell-wrapper controls.
- Resolved topology identity: `coupled_field_machine`.
- Artifact identity includes `HB_<compressor>` and `HFC/HFA/HFM/HL/HD/HM/HH/HC` orders.
- DCT provider train/save/resume test.
- Whole tiny model `torch.compile(..., backend="eager", fullgraph=True)` compatibility smoke.

## Local commits after exported remote baseline

```text
2461bbd Add HYPERBLOCK mathematical core
e2feb83 Add HYPERBLOCK trajectory
0d0b15c Integrate HYPERBLOCK model path
230c5cf Add HYPERBLOCK run configuration
b657af8 Add HYPERBLOCK wrapper and validation tests
c6d6902 Expose anisotropic HYPERBLOCK family orders
885da9e Update HYPERBLOCK as-built specification
```

Commit hashes above are local sandbox hashes. The remote branch must be checked after connector synchronization; do not assume these exact hashes survive a connector-created aggregate commit.

## Verified commands

```bash
cd ~/git/thog2

python -m compileall -q \
  sheet \
  run_thog2_owt_core.py \
  tests/test_hyperblock*.py

bash -n train_OWT.sh train_OWT_core.sh

PYTHONPATH=. pytest -q tests/test_hyperblock*.py
```

Latest dedicated result:

```text
35 passed
```

The 35 tests cover:

- coefficient shapes and exact accounting;
- canonical L32/D1024 count;
- orthonormal basis tables;
- reference/staged equality;
- full-region/per-family equality;
- omitted constant branch modes;
- common-gradient additivity;
- operational routing;
- DCT plug compatibility;
- materialiser compile compatibility;
- reduced family-order initialization/materialisation;
- family-order validation;
- nanoGPT family shapes;
- LayerNorm/bias initialization;
- initialization standard deviations;
- residual scaling;
- retained materialisation projection/release;
- basis regeneration on checkpoint load;
- model forward/backward;
- optimizer grouping;
- configuration/checkpoint identity;
- legacy-overlap rejection;
- retained/ephemeral numerical closeness;
- whole-model full-graph compile smoke;
- DCT train/save/resume;
- reduced-family-order whole-model forward/backward;
- CLI, artifact and wrapper propagation.

## Known numerical issue

`tests/test_modified_fast_discard_false.py::test_stage4_update_is_equivalent_and_releases_materializations` fails in the current sandbox on both:

- this HYPERBLOCK implementation checkout; and
- the untouched exported branch base.

The loss matches. Tiny gradient reduction-order differences are normalized by Adam and create a larger first-step state difference. HYPERBLOCK has a bounded numerical-closeness test rather than claiming bitwise equality. Do not classify the legacy failure as a HYPERBLOCK regression without reproducing it on the branch base in the same environment.

## Important design qualification

Full family orders `6/4/2` are invertible on their respective family domains and therefore do not force or earn compression across weight-family identity. They are the least presumptuous defaults. Reduced family orders create genuine cross-family compression/coupling. Any claim that the full-order run proves cross-family sharing would be false.

## Current limitations

- Production per-family materialisation recomputes some common contractions. Correctness is established; contraction reuse and peak-memory optimization are not.
- No meaningful GPU timing has been performed.
- Compile compatibility is tested, but no compile speed benefit is claimed.
- Haar and lapped-cosine are registry-reachable but do not yet have whole-model HYPERBLOCK train/checkpoint tests.
- Broad regression coverage is incomplete because the full suite is large and contains known base/environment failures.
- The temporary GitHub workflow source-snapshot artifact step must be removed before merge.
- Existing-file source-history comments must be audited against the project rule that replaced nanoGPT lines remain visible as comments.

## Next safe work sequence

1. Synchronize the complete local implementation and this status file to the remote branch.
2. Run the dedicated GitHub CPU workflow and inspect job logs.
3. Add whole-model Haar and lapped-cosine smoke/checkpoint tests if their basis constraints permit the chosen tiny shapes.
4. Run focused legacy regressions for model/config/checkpoint/wrapper/materialisation paths and compare any failures against the base branch.
5. Add a CPU test plan/completion record with exact pass/fail/skip counts.
6. Audit and preserve replaced existing source lines as comments where required.
7. Profile materialisation counts, contraction intermediates and peak memory on tiny and canonical shapes.
8. Only then run a small GPU smoke. Do not launch a long training run yet.
9. Remove the temporary source-snapshot workflow upload before proposing merge.
10. Update the requirements specification after every material as-built deviation.

## Smoke run shape

Use small orders and dimensions first:

```bash
./train_OWT.sh \
  --hyperblock \
  --hyperblock-compressor chebyshev \
  --hyperblock-common-family-order 3 \
  --hyperblock-attention-family-order 2 \
  --hyperblock-mlp-family-order 1 \
  --hyperblock-depth-order 2 \
  --hyperblock-d-model-order 4 \
  --hyperblock-mlp-hidden-order 4 \
  --hyperblock-attention-head-order 2 \
  --hyperblock-attention-head-channel-order 4 \
  -L 2 -H 2 -D 8 -C 8 \
  -b 1 -A 1 -n 2 -w 0 \
  -T float32 -K math -x true
```

Start with dry-run and CPU tests. The wrapper's normal dataset/training assumptions may still make a literal CPU CLI training run less convenient than the existing `Stage4Trainer` tests.

---

**HAND-OFF RULE:** Read this file and the v0.2 requirements specification before changing topology, coefficient support, family-coordinate policy, initialization or checkpoint identity.
