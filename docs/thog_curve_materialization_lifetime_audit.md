# THOG curve materialisation lifetime audit

This note audits the current curve-mode runtime path with the specific goal of keeping materialised tensors alive for no longer than needed, without breaking efficient fused operations such as the combined QKV projection.

Scope: `SheetGPT` / `TrainingSheetGPT` with `geometry_preset=curve`, especially the current KARITANE-style run shape `L144 H32 D2048 C256 P80`.

## Executive summary

The current implementation does not materialise a whole transformer layer before executing it. It already materialises weights mostly at point of use:

1. layernorm vectors;
2. combined QKV attention input weight and bias;
3. attention output weight and bias;
4. MLP expansion weight and bias;
5. MLP contraction weight and bias.

The big remaining issue is not gross full-layer materialisation. The issue is tensor lifetime hygiene: local Python references to large materialised tensors remain in scope longer than conceptually necessary, and autograd may additionally retain tensors required for backward. We can safely shorten Python lifetimes, but we must not assume that `del` forces CUDA memory to fall immediately or that it can remove tensors required by autograd.

The highest-value low-risk optimisation is a lifetime hygiene pass in `sheet/model.py` that keeps the existing algorithm and explicitly drops large temporaries immediately after their last forward use.

## Big-ticket tensors

For `D=2048`, curve mode materialises matrices in fp32 because the trajectory runtime dtype is currently `torch.float32`. Autocast may execute `F.linear` in lower precision, but the generated tensor itself is fp32 unless this path is changed.

Approximate materialised weight sizes:

| Tensor | Shape | fp32 size | Notes |
|---|---:|---:|---|
| combined QKV weight | `[3D, D] = [6144, 2048]` | ~50 MiB | Worth keeping as one fused projection unless profiling says otherwise. |
| attention output weight | `[D, D] = [2048, 2048]` | ~16 MiB | Materialised after SDPA. |
| MLP expansion weight | `[4D, D] = [8192, 2048]` | ~64 MiB | Large. Should not overlap with contraction weight by Python reference unless autograd needs it. |
| MLP contraction weight | `[D, 4D] = [2048, 8192]` | ~64 MiB | Large. |
| layernorm vectors | `[D]` | tiny | Not a big-ticket item. |

Activation sizes are also material:

| Tensor | Shape at `b=12, C=256, D=2048` | fp16-ish size | Notes |
|---|---:|---:|---|
| QKV activation | `[12, 256, 3D]` | ~38 MiB | Produced by fused QKV projection. |
| MLP hidden | `[12, 256, 4D]` | ~50 MiB | A major target for future streaming, but not a low-risk lifetime-only change. |
| attention score/probability tensors | backend dependent | can be large | SDPA backend choice matters, especially as `C` increases. |

## Current execution order

`SheetGPT._logical_block()` runs:

1. `ln_1` materialisation and layernorm;
2. `_attention()`;
3. residual add;
4. `ln_2` materialisation and layernorm;
5. `_mlp()`;
6. residual add.

`_attention()` currently runs:

1. materialise combined QKV weight;
2. materialise combined QKV bias if bias is enabled;
3. run one fused `F.linear(inputs, attention_weight, attention_bias)`;
4. split output into `query`, `key`, `value`;
5. reshape heads;
6. run `F.scaled_dot_product_attention(...)` when available;
7. materialise attention output weight;
8. materialise attention output bias if bias is enabled;
9. run output projection;
10. dropout.

`_mlp()` currently runs:

1. materialise MLP expansion weight;
2. materialise expansion bias if bias is enabled;
3. run expansion projection;
4. GELU;
5. materialise MLP contraction weight;
6. materialise contraction bias if bias is enabled;
7. run contraction projection;
8. dropout.

This is already mostly just-in-time. The code does not build all six major layer matrices up front.

## Backprop safety

Deleting a Python variable after its last forward use is safe if the tensor has already been passed into the operation that needs it.

However, if autograd needs the tensor for backward, PyTorch will keep an internal reference through the computation graph. This is not optional. In particular:

- `F.linear` backward may need the input activation and the weight to compute gradients;
- the materialised weight is itself a differentiable result of the depth-basis contraction, so the graph from generated weight back to the coefficient tensor must remain valid;
- activation checkpointing changes what is saved and what is recomputed, but it does not make needed backward data disappear.

Therefore:

- explicit `del` can shorten Python reference lifetimes;
- explicit `del` cannot and must not defeat autograd;
- a lack of immediate `nvidia-smi` improvement is not evidence that the change failed, because the CUDA caching allocator may retain freed blocks for reuse.

The right memory metric is `torch.cuda.max_memory_allocated()` or profiler memory events, not just `nvidia-smi`.

## Low-risk lifetime hygiene targets

### 1. `_attention()` combined QKV path

Keep the combined QKV projection. It is probably the right compute tradeoff.

Safe lifetime tightening:

1. after `qkv = F.linear(...)`, immediately drop `attention_weight` and `attention_bias` references;
2. after `query`, `key`, `value` have been consumed by SDPA, drop their references;
3. after output projection, drop `output_weight` and `output_bias` references;
4. avoid retaining intermediate names longer than needed.

Do not split QKV into three separate linears merely for ephemerality. That risks throwing away a well-optimised projection path.

### 2. `_mlp()` expansion/contraction path

This is likely the largest simple lifetime cleanup.

Safe lifetime tightening:

1. materialise `expansion_weight` and `expansion_bias`;
2. compute `hidden = F.linear(...)`;
3. immediately drop `expansion_weight` and `expansion_bias` references;
4. apply GELU;
5. materialise `contraction_weight` and `contraction_bias`;
6. compute output;
7. immediately drop `contraction_weight`, `contraction_bias`, and `hidden` references.

This keeps the current algorithm intact while reducing accidental overlap between the two 64 MiB MLP weights.

### 3. layernorm vectors

Low priority. They are small. Do not complicate code just to delete them earlier unless touching the code anyway.

### 4. materialiser internals

For curve mode, `_materialize_curve_matrix()` creates exactly the requested named matrix by contracting the selected depth basis row with that family’s coefficient tensor. This is fine. There is no evidence that it materialises unrelated families.

Possible later micro-optimisation: benchmark `einsum("p,rcp->rc", ...)` against an explicit flattened matmul form. This is performance tuning, not lifetime hygiene.

## Medium-risk optimisations to defer until profiling

### Combined QKV direct materialisation

Current `LEGACY_ATTENTION_INPUT_WEIGHT` materialisation builds Q, K, and V separately and concatenates them. A direct combined coefficient layout could avoid a concat and possibly one or more transient allocations, but it changes parameter layout or introduces a special materialiser path. Defer until profiling shows it matters.

### Scope-splitting helper functions

Breaking `_attention()` and `_mlp()` into smaller helpers may naturally shorten Python scopes. It may also add Python overhead across 144 logical layers. Do not do this before profiling the simple `del` cleanup.

### Autocast/materialised-weight dtype

Current generated weights are fp32 trajectory tensors. Running materialisation itself in lower precision, or explicitly casting generated weights before `F.linear`, could reduce memory but may affect training dynamics and coefficient gradients. Treat as a separate experiment.

## High-risk / high-reward future work

### Fused materialise-plus-linear

A custom kernel could avoid writing full generated matrices to global memory, or could fuse depth contraction with the linear projection. This is a real architecture/runtime project, not a cleanup.

For curve mode, the naive unfused formula is unattractive:

`y = sum_p a_p * (x @ C_p.T)`

With `P=80`, doing this as 80 dense matmuls is almost certainly worse than materialising `W_l` once and using one efficient GEMM. A useful fused version probably needs a custom CUDA kernel or a very careful compiler path.

### MLP hidden streaming

The MLP can theoretically be streamed over hidden-channel chunks:

`output = sum_j gelu(x @ W_up_j.T) @ W_down_j.T`

This could reduce peak MLP hidden activation and weight materialisation pressure. It changes execution order and backward behaviour, and needs proper tests/profiling.

### Head-wise attention streaming

Attention could theoretically be processed per head or head group. This may reduce transient activations but risks losing efficient SDPA kernels. Defer.

## Recommended implementation stage

Stage A: lifetime hygiene only.

Files:

- `sheet/model.py`

Rules:

- preserve combined QKV projection;
- preserve SDPA path;
- preserve `F.linear` calls;
- add explicit `del` only immediately after a tensor’s last forward use;
- do not call `torch.cuda.empty_cache()` in the hot path;
- do not change parameter layout;
- do not change numerical operation order except for harmless reference lifetime cleanup.

Tests:

1. CPU fixed-seed forward equivalence against pre-cleanup behaviour for a small curve model;
2. backward reaches curve coefficients after `_attention()` and `_mlp()`;
3. checkpointed training forward/backward still works;
4. CUDA diagnostic test records `max_memory_allocated` before and after the cleanup path, but does not require improvement as a hard assertion.

Acceptance criterion:

- exact or near-exact output/loss equivalence under fixed seed;
- no gradient regression;
- no speed regression large enough to matter;
- any memory improvement treated as a bonus unless profiling confirms stable reduction.

## Bottom line

The current code is already materially more JIT than a naive full-layer materialiser. The safe next improvement is a careful lifetime pass, especially around QKV and MLP weights, while preserving the existing efficient combined QKV projection and SDPA usage.
