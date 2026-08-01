# THOG2 Coupled Field Machine HYPERBLOCK

**Implementation Status and Validation Record**  
**Version 0.2 — 1 August 2026**  
**Branch:** `COUPLED_FIELD_MACHINE_HYPERBLOCk`  
**Base:** `MODIFIED_FAST_DISCARD_FALSE_CASE` at `c38ef70c850a55a2f12556c72ee64b259bb1fbae`

## Outcome

The first non-breathing Coupled Field Machine HYPERBLOCK implementation is complete for CPU correctness and merge review. It directly owns and materialises Q, K, V, attention-output, MLP-up and MLP-down matrices. Existing DEPTH, SHEET, BLOCK, dense, LayerNorm, bias and embedding paths remain available and are not routed through HYPERBLOCK unless HYPERBLOCK is explicitly selected.

This status does **not** claim that HYPERBLOCK will train well, outperform DEPTH, save wall time or benefit from `torch.compile`. Those are GPU experiments, not correctness conclusions.

## As-built topology

The persistent coefficient object is the direct sum:

```text
common:
    WEIGHT_FAMILY × DEPTH × D_MODEL

attention extension:
    ATTENTION_WEIGHT_FAMILY × DEPTH × D_MODEL
    × (ATTENTION_HEAD × ATTENTION_HEAD_CHANNEL, excluding the doubly constant mode)

MLP extension:
    MLP_WEIGHT_FAMILY × DEPTH × D_MODEL
    × (MLP_HIDDEN, excluding the constant mode)
```

The common region is shared by all six matrix families. Attention and MLP extensions contain only variation that requires their physically unique axes. No `MLP_HIDDEN × ATTENTION_HEAD` coefficient exists because no operational scalar possesses both coordinates.

## Implemented modules

- `sheet/hyperblock/plan.py`
  - resolved topology and anisotropic orders;
  - exact coefficient shapes/counts;
  - arbitrary MLP hidden multiplier accounting;
  - checkpoint identity.
- `sheet/hyperblock/basis_provider.py`
  - adapter over the existing THOG2 basis registry;
  - no HYPERBLOCK-specific Chebyshev fork.
- `sheet/hyperblock/materializer.py`
  - clear `einsum` reference equations;
  - fixed-order production mode products;
  - exact Q/K/V/O/UP/DOWN routing.
- `sheet/hyperblock/trajectory.py`
  - persistent common/attention/MLP coefficients;
  - initialization;
  - conventional block vectors for v0;
  - retained-materialisation projection/release;
  - reports and checkpoint regeneration.
- Integration in:
  - `sheet/model.py`;
  - `sheet/training_config.py`;
  - `sheet/run_config.py`;
  - `sheet/training_model_factory.py`;
  - `run_thog2_owt_core.py`;
  - `train_OWT_core.sh`.

## Basis plug-in result

The same field machine has whole-model train/checkpoint/resume coverage for:

```text
chebyshev
dct
haar
lapped_cosine
```

A basis provider supplies one `[physical_length, retained_order]` table per axis. It does not own coefficient topology, branch support, routing or materialisation.

## Materialisation result

The production path is compared against two independent definitions:

1. a direct multi-axis `einsum` implementation;
2. a literal scalar summation loop on tiny domains.

Production expansion order is fixed rather than delegated to a large multi-operand contraction:

```text
attention: HEAD_CHANNEL → HEAD → D_MODEL
MLP:       MLP_HIDDEN → D_MODEL
```

This is a correctness-oriented first order, not a claim of optimal GPU contraction planning.

## Initialization correction

ATTENTION_OUTPUT and MLP_DOWN residual scaling is applied while constructing initial coefficients. A post-hoc materialise/scale/project operation is rejected for reduced family orders because it can leak the intended family-specific scaling into Q, K, V or MLP_UP through the reduced family basis.

## Fast-discard equivalence

The retained and ephemeral paths are tested at the correct mathematical boundary:

- identical loss within tight floating-point tolerance;
- persistent-parameter gradients agree before AdamW normalization;
- zero-momentum SGD produces matching parameter states;
- retained operational tensors are released after projection.

AdamW first-step state equality is not used as the primary invariant. Adam normalizes near-zero gradients and can magnify harmless sign noise caused by different reduction order.

## Dedicated test result

Latest local CPU result before final GitHub validation:

```text
49 passed
1 skipped
```

The skipped test is the CUDA bfloat16 one-update smoke when CUDA is unavailable. CPU bfloat16 runs with both retained and ephemeral materialisation.

Dedicated coverage includes:

- literal scalar-definition equivalence;
- reference-versus-production equality;
- finite-difference gradients for all three coefficient regions;
- common-gradient additivity;
- omitted duplicate constant modes;
- zero unique-order collapse;
- exact coefficient and dense-equivalent accounting;
- canonical L32/D1024 count;
- non-4× MLP width;
- all registered basis families;
- materialised shapes and routing;
- initialization statistics and residual scaling;
- reduced family orders;
- float32 and bfloat16 one-update training;
- retained/ephemeral gradient and SGD-update equivalence;
- checkpoint regeneration and resume;
- CLI, artifact identity and wrapper propagation;
- full-graph `torch.compile(..., backend="eager")` compatibility.

## Regression validation

The GitHub workflow checks each legacy `tests/test_*.py` file outside the HYPERBLOCK suite in an isolated process. A head failure is rerun against the exact PR base in the same runner. The workflow fails only when:

- the head fails a file that passes on the base;
- the head introduces a failing test node not present on the base; or
- the head times out where the base does not.

This avoids pretending that known stale or environment-sensitive base failures were introduced by HYPERBLOCK.

## Source-history validation

An automated audit compares modified Python and shell files with the exact base. A removed executable line must remain either:

- executable elsewhere in the same file; or
- present as a commented historical line.

All restored comments are inside THOG-marked blocks. The audit covers the existing files modified for HYPERBLOCK integration and enforces Peter's no-silent-deletion rule.

## Compile position

The pure numerical materialiser and a tiny complete model pass full-graph compile compatibility tests with the eager backend. Compilation remains optional. No performance benefit has been demonstrated, and no implementation complexity was added solely to chase compile speed.

## Remaining empirical work

Correctness completion does not remove the need for GPU experiments. Before any merge into `master`, the sensible empirical sequence is:

1. small CUDA bfloat16 smoke on scruffy;
2. inspect peak VRAM and materialisation time;
3. confirm no unexpected graph retention across accumulation;
4. compare a tiny HYPERBLOCK run against dense and DEPTH controls;
5. only then choose a credible longer-run coefficient budget.

A GPU is not available in GitHub's standard CPU runner. The CUDA bfloat16 smoke test is present and will execute automatically in any CUDA-enabled test environment.

## Pull command

```bash
cd ~/git/thog2

git fetch origin
git switch COUPLED_FIELD_MACHINE_HYPERBLOCk
git reset --hard origin/COUPLED_FIELD_MACHINE_HYPERBLOCk

chmod +x train_OWT.sh train_OWT_core.sh

python -m compileall -q sheet tests run_thog2_owt_core.py
bash -n train_OWT.sh train_OWT_core.sh
PYTHONPATH=. pytest -q tests/test_hyperblock*.py
```

The `reset --hard` is intentional here because this is the requested branch-stomp stanza. Do not run it on a checkout containing uncommitted work that must be retained.

---

**HAND-OFF RULE:** The v0.3 requirements specification and this v0.2 status record are the as-built references. Do not infer performance or training quality from correctness tests.
