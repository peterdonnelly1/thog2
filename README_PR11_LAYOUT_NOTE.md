# PR 11 run artifact layout note

This branch changes the two scruffy OpenWebText wrappers and the shared THOG2 run naming/path helpers.

## Durable run artifacts

New fresh runs still write program checkpoints under:

```text
checkpoints/<artifact_name>/ckpt.pt
```

Inspectable artifacts now live together under a timestamped log directory:

```text
logs/<YYMMDD_HHMM_artifact_name>/train.log
logs/<YYMMDD_HHMM_artifact_name>/result.json
```

The old `results/<artifact_name>/result.json` path is no longer used for new runs. `result_root` remains accepted for compatibility but is ignored by the new path builder.

## Artifact naming

The canonical artifact name now uses the wrapper option letters for the key included controls, in getopts-comment order:

```text
<PREFIX>_<host>__<RUN_NAME>__n_<steps>_b_<batch>_d_<dataset>_w_<warmup>_k_<checkpoint_interval>_A_<grad_accum>_L_<layers>_H_<heads>_D_<width>_C_<context>[_P_<depth_order>_Q_<row_order>]_S_<checkpoint_segment_size>
```

`RUN_MODE` remains part of the resolved configuration and manifest payload, but it is deliberately not included in the artifact name because `fresh` versus `resume` is an execution mode, not a model/run identity. Including it in the artifact name would make resume look for a different checkpoint path.

W&B mode, W&B enabled, dry-run, logging cadence, evaluation cadence, and other non-identity switches are not included in the artifact name.
