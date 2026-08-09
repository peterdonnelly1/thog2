# PLASTIC same_batch_all_probes implementation log

Governing specification: `docs/THOG2_PLASTIC_Requirements_Specification_v0.53.txt`
Branch: `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION`
PR: #33
Started: 2026-08-09
Implementation code head validated: `da43638b2266ed7122618400299e768af3edb52d`
Corrected CUDA-smoke test head: `e5bd20485ac1bb463b2eca0b5e244932e5f6ae3f`

## Takeover status

Implementation is complete for CPU/DDP purposes. A first real CUDA smoke on scruffy reached COARSE, fresh FINE construction and a successful optimizer update, then exposed a stale smoke-test assertion caused by inherited warmup suppression. That smoke has been corrected to exercise a full same-batch window; the corrected real-CUDA rerun remains the final external gate.

## Implemented architecture

The existing FINE inline probe used the first training microbatch and therefore could not simply retain that batch across probe events without repeatedly training on the same data. Version 0.53 now separates the roles:

1. A same-batch window acquires one deterministic dedicated training-split probe batch.
2. Every probe event in that window reuses that batch and one deterministic sampled-token set.
3. Candidate decision losses are evaluated through a shared logical-layer chain under `torch.no_grad()`.
4. Candidate losses are handed to the already-installed v0.541 selector/audit stack; decision algorithms are not duplicated.
5. The authoritative optimizer update still uses its normal fresh training microbatch stream.
6. Before the configured W probes, framework gating forces STAY even if an underlying selector would attempt movement.
7. At W, one decision is accepted subject to existing decision rules and `plastic__layer_count__max_allowable_layer_change`.
8. The window is retired after CHANGE or STAY; histories are cleared and the next window receives a fresh batch.
9. Partial-window checkpoint/resume reconstructs the exact batch from deterministic global start indices and reconstructs sampled-token positions from the persisted window seed.
10. DDP stores one deterministic global start tuple; each rank reconstructs the same rank-local slice throughout the window.

## Main source changes

### `sheet/plastic_depth_same_batch_all_probes_patch.py`

New final runtime overlay. It provides:

- exact public flag `--plastic__layer_count__same_batch_all_probes` and explicit negative form;
- disabled default and false-mode metadata preservation;
- enabled-mode persistent config/compact identity;
- fixed non-overlapping window state and deterministic batch identity/digest;
- runtime batch and sampled-token caches;
- no-grad shared-chain candidate evidence;
- training-only sentinel path that prevents candidate evidence from being recomputed on the real training batch;
- complete-window gating and STAY/CHANGE retirement;
- structured window events and audit augmentation;
- checkpoint persistence and exact partial-window reconstruction.

Initial implementation commit: `6c74a6480a47a405682aa8bf8084b65c9f4aa38`.

### `sheet/plastic_depth_console_postfix_patch.py`

Imports the same-batch overlay after v0.531/v0.541 so the new evidence-acquisition framework wins last without replacing later selector/provenance behavior.

Integration commit: `6a7348215faba0346be8053e803ae8de32d42f1c`.

### `tests/test_plastic_depth_same_batch_all_probes.py`

Focused tests cover:

- exact public option spelling and enabled-mode persistence;
- false-mode metadata/checkpoint preservation;
- no-grad/no-optimizer/no-parameter-mutation evidence collection;
- ordinary training batch generator/trace not advanced by the dedicated evidence probe;
- same batch and sampled tokens throughout one window;
- retirement after STAY and fresh next window;
- window-local P provenance;
- exact partial-window checkpoint resume;
- malformed persisted provenance rejection.

Initial test commit: `71e8fc3f09fe1020b3ebdc1edd787dcbbd73a8f5`.

### `tests/plastic_depth_ddp_probe.py`

The existing two-rank COARSE/FINE lifecycle validation now explicitly enables same-batch mode and requires shared fixed-batch identity plus correct window ordinal/size/provenance.

Commit: `c9b855780b1e2abb84b430cf85168032930be20d`.

### `.github/workflows/plastic_coarse_fine_regression.yml`

The same-batch production module is explicitly included in `py_compile` validation.

Commit: `da43638b2266ed7122618400299e768af3edb52d`.

### `tests/test_plastic_depth_coarse_fine_gpu_smoke.py`

The external smoke was corrected after the first real run showed that the inherited `warmup_updates=1` prevented the first FINE probe while the stale test still demanded an audit row immediately. The corrected version:

- explicitly enables same-batch runtime mode only for this test;
- sets `warmup_updates=0` so the smoke probes immediately rather than testing warmup suppression;
- performs four real CUDA optimizer updates for W=4;
- checks one fixed batch digest across all four probes;
- checks ordinals 1..4 and exact window-local provenance growth;
- verifies no early layer-count change;
- verifies STAY retirement, cleared histories, full-radius candidate counts, audit replay and nonzero CUDA peak allocation.

Correction commit: `e5bd20485ac1bb463b2eca0b5e244932e5f6ae3f`.

## Chronological implementation notes

### Initialization

- Confirmed branch `PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION` and draft PR #33.
- Confirmed v0.53 spec and that later v0.531/v0.541 code already existed on the branch; implementation was therefore layered on top rather than rolling later work back.
- Local clone was unavailable because the execution environment could not resolve external DNS; repository reads/writes used the connected GitHub app and validation used GitHub Actions.
- Confirmed legacy resource timing/memory probes may deliberately execute backward without optimizer stepping; they remain separate from decision-loss evidence.

### Runtime design decisions

- Fixed batch is represented checkpoint-portably by deterministic global training-split start indices plus batch digest rather than persisting device tensors.
- Runtime cache may remain device resident; checkpoint cache is reconstructed on resume.
- Probe RNG is captured/restored around the no-grad evidence forward so the extra evidence computation does not perturb the authoritative training RNG trajectory.
- The ordinary training batch source generator and trace are not touched when constructing the dedicated probe batch.
- Count changes invalidate a partial window through active-count mismatch detection. Future global MLP-width integration should invoke the same invalidation hook on an F change.
- When later wall-time logic resets history inside an active window, a completed same-batch window whose directional history count is not aligned with its window ordinal is conservatively forced to STAY and retired rather than mixing incompatible evidence.

### CI regression found and corrected

The first broad CI run correctly found branch-only failures rather than a classifier false positive:

- the new public-flag test used `max_permitted_layers=5` while leaving `o_depth=16`, causing its own invalid fixture;
- the feature test left the process-global runtime mode selected, contaminating unrelated subsequent checkpoint tests in the same pytest process;
- an inherited v0.521 test asserted that its SE wrapper remained the final method owner, which was no longer true after the intentional final v0.53 wrapper.

Corrections:

- public-flag test now sets a valid `--o-depth 3`;
- feature-test fixture explicitly removes the runtime/explicit mode environment after every test;
- overlay ownership test now asserts that the v0.53 wrapper is final and that its captured underlying request remains the v0.521 paired-SE wrapper.

No classifier waiver or test deselection was added.

### First real CUDA smoke on scruffy

User ran the external smoke on 9 August 2026 at branch head `a23636f090c3cb0e21f9eb30f41dec10d29dbb49`.

Environment reported:

- NVIDIA GeForce RTX 4090 Laptop GPU;
- PyTorch 2.12.1+cu126;
- CUDA runtime 12.6.

Observed execution:

- COARSE trial completed successfully;
- fresh FINE state was constructed;
- first authoritative FINE optimizer update completed successfully and was not skipped;
- CUDA cleanup completed without an OOM report;
- the test then failed at `len(fine_state.trainer.plastic_depth_count_audit) == 1` because that attribute had not yet been created.

Root cause:

- `stage3_config()` defaults to `warmup_updates=1`;
- the final PLASTIC warmup guard intentionally returns no FINE probe context during a frozen warmup update;
- therefore update 1 correctly produced no probe evidence and no count audit;
- the smoke assertion was stale and incorrectly treated “one training update” as “one FINE probe.”

This failure did not establish a same-batch CUDA defect. In fact the smoke did not explicitly enable same-batch mode, so it was not a sufficient external gate for v0.53 even if the old assertion had passed.

The correction in `e5bd20485ac1bb463b2eca0b5e244932e5f6ae3f` makes the smoke a genuine v0.53 CUDA gate rather than merely repairing the failing assertion.

## Validation results

Code head: `da43638b2266ed7122618400299e768af3edb52d`.

- Regression workflow run `31289904430`: SUCCESS.
  - new module explicit `py_compile`: SUCCESS;
  - shell syntax: SUCCESS;
  - complete CPU pytest suite executed;
  - untouched-`PLASTIC_DEPTH` inherited-failure classifier: SUCCESS, no new branch-only failures;
  - regression evidence artifact retained.
- Disabled-equivalence workflow run `31289904395`: SUCCESS; exact disabled-path fingerprint equivalence.
- DDP workflow run `31289904420`: SUCCESS with same-batch mode enabled on two ranks; lifecycle/FINE probe and rank agreement passed.

PR #33 body has been updated with the v0.53 implementation and current validation evidence.

## Remaining external gate

Rerun the corrected real CUDA smoke from current branch head:

```bash
cd ~/git/thog2
git fetch origin PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git checkout PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
git reset --hard origin/PLASTIC_DEPTH_COARSE_FINE_IMPLEMENTATION
bash tools/run_plastic_coarse_fine_gpu_smoke.sh
```

A pass now means materially more than the original smoke: it exercises a complete four-probe same-batch window on real CUDA and verifies fixed-batch identity, window-local provenance, no early count movement, retirement, audit replay and allocator use.

## Do not regress these semantics

- Do not make the cached probe batch an ordinary training batch.
- Do not restore rolling overlap in same-batch mode.
- Do not carry evidence from a completed STAY window into the next window.
- Do not permit a decision before W valid probes.
- Do not replace the v0.541 selector stack with a duplicate v0.53 decision implementation.
- Do not treat resource-probe backward passes as decision-loss evidence.
- Do not lose P provenance or exact partial-window resume.
- Keep false mode on the established path.