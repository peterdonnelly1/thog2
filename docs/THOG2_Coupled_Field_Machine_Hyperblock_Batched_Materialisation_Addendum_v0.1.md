# THOG2 Coupled Field Machine HYPERBLOCK

**Batched Materialisation As-Built Addendum**  
**Version 0.1 — 1 August 2026**  
**Branch:** `COUPLED_FIELD_MACHINE_HYPERBLOCk`

## Status

This addendum supersedes the performance-path description in sections 10.4 and 13.3 of `THOG2_Coupled_Field_Machine_Hyperblock_Requirements_Specification_v0.3.md` where that document describes production materialisation as separate per-family/per-layer calls.

The mathematical coefficient object, basis definitions, initialization, checkpoint identity and operational Q/K/V/O/UP/DOWN orientations are unchanged. This is an execution optimization, not a new HYPERBLOCK topology or checkpoint format.

## Previous production execution

For each Transformer layer, the model requested the four operational matrix groups independently:

```text
packed QKV
attention output
MLP up
MLP down
```

The packed QKV request itself materialised Q, K and V independently. Consequently, one layer performed six family-specific reconstruction paths. Each path repeated some combination of:

- omitted-zero-mode reconstruction;
- family-basis contraction;
- DEPTH-row contraction;
- common-field contraction;
- small tensor-operation launches.

At 32 layers, one optimizer update's first retained-materialisation pass therefore entered 128 attention-family paths and 64 MLP-family paths before gradient-accumulation reuse.

## As-built batched execution

`materialize_layer_staged()` now materialises all six matrix families for one layer as one shared bundle:

```text
attention canonical field: [4, D_MODEL, HEAD, HEAD_CHANNEL]
MLP canonical field:       [2, D_MODEL, MLP_HIDDEN]
```

For each layer it performs:

1. one common-family expansion and one DEPTH-row contraction for all six families;
2. one attention-family expansion and one DEPTH-row contraction for Q/K/V/O;
3. one MLP-family expansion and one DEPTH-row contraction for UP/DOWN;
4. one omitted-zero-mode reconstruction per branch;
5. batched unique-axis and D_MODEL mode products;
6. routing into the existing packed QKV, attention-output, MLP-up and MLP-down layouts.

`CoupledFieldTrajectory.materialize_layer_matrices()` exposes the routed bundle. `SheetGPT._logical_block()` requests exactly one such bundle per HYPERBLOCK layer and passes it to both attention and MLP operations.

The individual `materialize(name, layer_index)` functions remain intact as independent correctness references and public compatibility paths.

## Retained-materialisation lifecycle

When retained materialisation is enabled, `UpdateRetainedMaterializations` now intercepts the layer-bundle API. On the first gradient-accumulation microstep it:

- constructs the shared bundle once per layer;
- detaches and retains the four routed operational matrices;
- reuses those tensors for later microsteps;
- projects all accumulated operational gradients through the shared graph once;
- releases the retained tensors after projection.

Legacy trajectories without a layer-bundle API continue through the existing individual-materialisation path unchanged.

When retained materialisation is disabled, the bundle is generated once per forward execution of each layer and released at the end of the logical block under fast-discard execution.

## Precision handling

The second DEPTH contraction converts its basis row to the dtype and device of the family-expanded intermediate, not merely to the persistent coefficient dtype. This is required because autocast may produce a bfloat16 intermediate from float32 persistent coefficients. CPU bfloat16 tests cover both retained and ephemeral execution.

## Validation

The dedicated tests require:

- batched attention and MLP fields to equal every existing per-family reference materialisation;
- gradients through common, attention and MLP coefficient regions to remain finite;
- routed layer bundles to equal the existing individual operational matrix API;
- retained bundles to be reused, projected once and released;
- `SheetGPT` to request exactly one bundle per HYPERBLOCK layer;
- float32 and CPU bfloat16 one-update training in retained and ephemeral modes;
- full existing HYPERBLOCK mathematical, checkpoint, compile and integration coverage.

Latest dedicated GitHub CPU result before the final legacy comparison:

```text
55 passed, 1 skipped
```

The skip is the existing CUDA-only bfloat16 test on a CPU runner.

## Performance claim

No speedup is claimed from CPU correctness tests. The optimization removes obvious redundant reconstruction and kernel-launch work, but the actual wall-time and peak-VRAM effect must be measured on scruffy with an otherwise identical HYPERBLOCK run. The current run should not be interrupted; compare the next run after fetching this branch.
