# THOG2 W&B, Configuration, Runner, and Artifact Naming Record

## Status

This record describes the permanent post-Stage-6 THOG2 training interface for DENSE2 and SHEET.

The accepted Stage 6 pilot remains immutable historical evidence. Its original protocol manifests, hashes, and selector-compatible result handling remain supported. New ordinary training runs use the canonical interface described here.

## Architecture prefixes

THOG2 uses exactly:

- `DENSE2_` for the THOG2 dense control;
- `SHEET_` for the two-dimensional Chebyshev Sheet model.

`DENSE2` is intentionally distinct from THOG's `DENSE`, because DENSE, VERMEER, DENSE2, and SHEET runs may appear in the same W&B project and panels.

No padding underscore is added after `SHEET`.

## Canonical artifact grammar

Representative names are:

```text
DENSE2_scruffy__AKAROA__owt__l_72_h_12_d_768_ctx_256__b_12_ga_160__steps_100
SHEET_scruffy__AKAROA__owt__l_72_h_12_d_768_ctx_256_p_16_q_64__b_12_ga_160__steps_100
```

The grammar follows THOG's architecture-first convention:

1. architecture prefix and host;
2. named run;
3. dataset;
4. logical geometry;
5. architecture-specific orders;
6. local mini-batch and global gradient accumulation;
7. completed optimizer updates.

For SHEET, `p` is the depth order and `q` is the base row order. Wider tensor-family row orders remain deterministic consequences of `q` and are not duplicated in the filename.

Optimizer details, seeds, source identity, dtype, checkpointing controls, and complete configuration remain in checkpoints, result JSON, and W&B config rather than making every filesystem component unnecessarily long.

## Deterministic truncation

Artifact components are bounded before filesystem use. When a generated name exceeds the configured limit, THOG2 retains the beginning, the end, and a twelve-hex SHA-256 digest between them.

The marker is:

```text
__TRUNC_<digest>__
```

This follows BRAINIAC's principle of deterministic, identity-preserving truncation rather than silently clipping the tail. Log filenames are independently bounded to the normal 255-character filesystem-component limit while preserving their timestamp and `.log` suffix.

## Artifact layout

New runs use:

```text
checkpoints/<artifact_name>/ckpt.pt
logs/<artifact_name>/<bounded_artifact_name>_train_<timestamp>.log
results/<artifact_name>/result.json
wandb/...
```

The W&B run name equals the canonical artifact name.

## Public configuration vocabulary

DENSE2 and SHEET expose the same names for the same controls. Shared names also match THOG:

- `max_iters`;
- `warmup_iters`;
- `eval_iters`;
- `eval_interval`;
- `log_interval`;
- `batch_size`;
- `gradient_accumulation_steps`;
- `block_size`;
- `n_layer`;
- `n_head`;
- `n_embd`;
- `activation_checkpointing`;
- `checkpoint_segment_size`;
- `learning_rate`;
- `min_lr`;
- `weight_decay`;
- `beta1`;
- `beta2`;
- `grad_clip`;
- `model_seed`;
- `data_seed`;
- `dtype`;
- `device`.

SHEET additionally exposes `depth_order` and `base_row_order`.

The internal historical trainer still stores its original Stage 1-6 field names where required for checkpoint compatibility. The permanent runner converts the canonical public schema at one explicit boundary. New result JSON and W&B configurations carry the canonical names.

## Gradient accumulation semantics

`gradient_accumulation_steps` is global across all participating GPUs, matching THOG. For world size `G`, each rank executes:

```text
local_gradient_accumulation_steps = gradient_accumulation_steps / G
```

The global value must divide evenly by `G`. Tokens per optimizer update are:

```text
batch_size * gradient_accumulation_steps * block_size
```

## Activation checkpointing

DENSE2 and SHEET share the same segmented activation-checkpointing interface. The scruffy runners default to local mini-batch size 12, checkpointing enabled, and checkpoint segment size 12.

DENSE2 uses a training-only subclass of the unchanged nanoGPT model. It preserves parameter names and checkpoint state while routing the transformer block list through the same non-reentrant segmented checkpoint executor used by SHEET.

## User-facing runners

The top-level scripts are:

```text
current_scruffy_train_DENSE_OWT.sh
current_scruffy_train_SHEET_OWT.sh
```

They provide matching `getopts` letters for shared controls, resolved-setting output, fresh/resume protection, deterministic artifact identity, dry-run support, one- or multi-GPU launch, timestamped log capture, and direct dispatch to `python -m run_thog2_owt`.

There is no nearly empty intermediate shell wrapper.

## W&B ownership

THOG2 owns its ordinary W&B instrumentation directly. It does not require a runtime import or monkeypatch from the THOG repository.

One W&B run is created per architecture run under project `thog` by default. Job types are `dense2` and `sheet`.

The principal axes are `optimizer_update`, `iter`, `tokens_seen`, and `clean_training_seconds`. Training, evaluation, resource, and SHEET diagnostic metrics use the same vocabulary for DENSE2 and SHEET. Telemetry is emitted outside the synchronized clean training and evaluation timers. Online initialization falls back to offline logging on a W&B communication failure.

Local checkpoints and result JSON remain authoritative. W&B is visualization and operational telemetry only.

## Compatibility

The accepted Stage 6 protocol and evidence are not rewritten. Legacy Stage 6 manifests continue to use their locked field names and paths. The permanent runner is the supported interface for subsequent DENSE2 and SHEET experiments.
