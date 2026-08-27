# vvv THOG
"""Append the canonical getopt/artifact descriptor registry to THOG2 CLI help."""

from __future__ import annotations

import argparse
from typing import Iterable, Sequence, Tuple


DescriptorRow = Tuple[str, str, str]
DescriptorSection = Tuple[str, Sequence[DescriptorRow]]


_DESCRIPTOR_SECTIONS: Tuple[DescriptorSection, ...] = (
    (
        "Model / run",
        (
            ("-p", "PRESET", "dense | legacy_sheet_col | depth | jpeg_like_v1 | head_aware_block | mlp_block | full_block"),
            ("-q", "RUN_MODE", "fresh | resume | fork"),
            ("-g", "RUN_NAME", "human run-name prefix"),
            ("-n", "STEPS", "total successful optimiser updates"),
            ("-b", "BATCH_SIZE", "microbatch size; scalar or wrapper grid"),
            ("-A", "GRADIENT_ACCUMULATION_STEPS", "global accumulation count"),
            ("-G", "NUM_GPUS", "local DDP process count"),
            ("-c", "LR_CODES", "maximum-LR code; 70 means 7.0e-04"),
            ("-f", "MIN_LR_CODE", "minimum-LR code; 06 means 6.0e-05"),
            ("-y", "OPTIMIZER", "adamw | sgd | sgd_nesterov | adafactor | rmsprop"),
            ("—", "--optimizer-momentum VALUE", "momentum for SGD, Nesterov and RMSprop"),
        ),
    ),
    (
        "Resume / fork lifecycle",
        (
            ("—", "--resume SELECTOR", "resume from a checkpoint selector"),
            ("—", "--fork SELECTOR", "fork from a checkpoint selector"),
            ("—", "--resume-from SELECTOR", "explicit selector for -q resume or -q fork"),
            ("—", "--fork-lr-mode MODE", "restart_cosine"),
            ("—", "--fork-learning-rate VALUE", "fork maximum learning rate"),
            ("—", "--fork-min-lr VALUE", "fork minimum learning rate"),
            ("—", "--fork-rewarm-iters N", "fork warmup updates"),
            ("—", "--wandb-continue-run | --no-wandb-continue-run", "retain or replace source W&B identity"),
        ),
    ),
    (
        "Schedule / logging",
        (
            ("-u", "EVAL_ITERS", "validation batches per evaluation"),
            ("-e", "EVAL_INTERVAL", "updates between evaluations"),
            ("-l", "LOG_INTERVAL", "updates between progress rows"),
            ("-w", "WARMUP_ITERS", "learning-rate warmup updates"),
            ("-k", "CHECKPOINT_INTERVAL", "periodic checkpoint interval; 0 disables"),
            ("-I", "INSTRUMENTATION", "tensorboard | wandb | both | wandb_offline | local | none"),
            ("-F", "DEPTH_CURVE_PLOTS", "none | final | eval"),
            ("-N", "DEPTH_CURVE_SAMPLE_ELEMENTS", "sample count for DEPTH diagnostics"),
            ("-U", "DEPTH_CURVE_RENDERER", "matplotlib | plotly | both"),
            ("-V", "DEPTH_CURVE_LOCAL_HTML", "true | false"),
            ("—", "--instrumentation__depth_weight_curves__scalar_weights_per_matrix N", "sampled scalar weights per matrix; default 3"),
            ("—", "--instrumentation__depth_weight_curves__depth_evaluation_points N", "continuous THOG curve resolution; DENSE remains discrete; default 256"),
            ("—", "--instrumentation__depth_weight_curves__time_mode latest|accumulate", "show the latest snapshot or accumulated history; default latest"),
            ("—", "--instrumentation__depth_weight_curves__history_length N", "open-ended accumulated-snapshot cap; finite capture windows auto-size; default 20"),
            ("—", "--instrumentation__depth_weight_curves__log_every_n_steps N", "successful updates between chart snapshots; default 100"),
            ("—", "--instrumentation__depth_weight_curves__same_coordinates_all_runs", "share deterministic scalar coordinates across runs"),
            ("—", "--no-instrumentation__depth_weight_curves__same_coordinates_all_runs", "select deterministic scalar coordinates per run; default"),
            ("—", "--instrumentation__depth_weight_curves__destination wandb|local|none", "six THOG curves or DENSE cross-marker charts; default local"),
            ("—", "--instrumentation__delta_loss_v_layer_heatmap log|linear", "absolute-layer observational Δloss heatmap; absent means disabled"),
            ("—", "--instrumentation__delta_loss_v_layer_heatmap__destination wandb|local|none", "heatmap destination; default local"),
            ("—", "--instrumentation__delta_loss_v_layer_heatmap_abs_limit VALUE", "fixed symmetric colour limit; default 0.05"),
            ("—", "--instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes N", "W&B-only sparse upload cadence; default 250 probes"),
            ("—", "--max-nonfinite-update-skips N", "maximum tolerated skipped non-finite updates"),
            ("—", "--initial-eval | --no-initial-eval", "enable or suppress update-zero validation"),
        ),
    ),
    (
        "Systematic geometry",
        (
            ("G0", "--select-depth", "select the universal registered DEPTH geometry"),
            ("G#", "--select-element SELECTOR", "select a registered element geometry; repeatable"),
            ("—", "--option TARGET.PROPERTY=VALUE", "assign a systematic geometry option; repeatable"),
            ("—", "--explain-geometry", "resolve and report geometry, then exit"),
        ),
    ),
    (
        "HYPERBLOCK",
        (
            ("HB", "--hyperblock | --no-hyperblock", "enable or disable coupled-field HYPERBLOCK"),
            ("HB", "--hyperblock-compressor NAME", "registered HYPERBLOCK compressor"),
            ("—", "--hyperblock-compressor-version VERSION", "auto or an exact registered version"),
            ("HFC", "--hyperblock-common-family-order N", "common-family coefficient order"),
            ("HFA", "--hyperblock-attention-family-order N", "attention-family coefficient order"),
            ("HFM", "--hyperblock-mlp-family-order N", "MLP-family coefficient order"),
            ("HL", "--hyperblock-depth-order N", "DEPTH-axis coefficient order"),
            ("HD", "--hyperblock-d-model-order N", "D_MODEL-axis coefficient order"),
            ("HM", "--hyperblock-mlp-hidden-order N", "MLP_HIDDEN-axis coefficient order"),
            ("HH", "--hyperblock-attention-head-order N", "attention-head coefficient order"),
            ("HC", "--hyperblock-attention-head-channel-order N", "head-channel coefficient order"),
            ("—", "--hyperblock-mlp-hidden-multiplier N", "physical MLP expansion multiplier"),
            ("HLC", "--hyperblock-loop-count N", "shared recurrence count"),
            ("HLD", "--hyperblock-loop-decay VALUE", "recurrent update decay in (0,1]"),
            ("—", "--direct-factorised-hyperblock-mlp | --no-direct-factorised-hyperblock-mlp", "direct HYPERBLOCK UP/DOWN execution"),
        ),
    ),
    (
        "PLASTIC DEPTH",
        (
            ("P", "--plastic__enabled | --no-plastic__enabled", "enable or disable the PLASTIC descriptor group"),
            ("—", "--plastic__coarse_phase enabled|disabled", "enable or bypass COARSE"),
            ("—", "--plastic__coarse_phase_roll_through | --no-plastic__coarse_phase_roll_through", "continue immediately or pause after COARSE"),
            ("—", "--plastic__log_interval_coarse N", "COARSE progress interval"),
            ("LCS", "--plastic__phase_1_n_steps N", "updates per COARSE trial"),
            ("LC", "--plastic__phase_1_starting_layer_count N", "first COARSE candidate count"),
            ("LCT", "--plastic__phase_1__number_of_trials N", "COARSE candidate count"),
            ("LCE", "--plastic__phase_1_evaluation_steps_count N", "final validation batches per trial"),
            ("LN", "--plastic__layers_to_sample N", "fixed active-layer count"),
            ("L_dyn", "--plastic__do_learn_layer_count", "enable dynamic active-layer count"),
            ("L", "--no-plastic__do_learn_layer_count", "retain fixed active-layer count"),
            ("LN", "--plastic__initial_layer_count N", "initial FINE active count"),
            ("LM", "--plastic__max_permitted_layers N", "maximum allocated layer capacity"),
            ("LI", "--plastic__layer_sampling_initialisation MODE", "equidistant | random"),
            ("LO", "--plastic__layer_count_objective OBJECTIVE", "lowest_loss | layer_efficiency | relative_training_wall_time | memory_budget"),
            ("LB", "--plastic__layer_count_update_brake N", "minimum updates between committed count changes"),
            ("LPI", "--plastic__layer_count_probe__probe_every_n_steps N", "updates between count probes"),
            ("LPT", "--plastic__layer_count_probe__number_of_sampled_valid_tokens N", "valid-token sample per probe microbatch; 0 means all valid tokens"),
            ("LPR", "--plastic__layer_count_probe_radius N", "integer candidate radius"),
            ("LMS", "--plastic__layer_count__max_allowable_layer_change N", "maximum committed count movement"),
            ("LSB", "--plastic__layer_count__same_batch_all_probes", "reuse one fixed evidence batch per strict non-overlapping probe window"),
            ("LSB", "--no-plastic__layer_count__same_batch_all_probes", "use rolling multi-batch probes; default"),
            ("LEW", "--plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence VALUE", "discount for extrapolative/right-side directional evidence"),
            ("LNW", "--plastic__layer_count_probe__window_size_as_number_of_probes N", "paired-score history window"),
            ("LNL", "--plastic__layer_count_probe_noise_lambda VALUE", "directional_coherence robust significance threshold; ignored by Sen/Kendall and jump modes"),
            ("LW", "--plastic__layer_count_cost_weight VALUE", "layer-cost penalty weight"),
            ("LMB", "--plastic__layer_count__memory_budget_gib VALUE", "memory_budget objective limit"),
            ("—", "--plastic__layer_count__cuda_allocator_reserve_gib VALUE", "upward-probe free-memory reserve"),
            ("LG", "--plastic__geometry_learning_rate_multiplier VALUE", "sampling-geometry LR multiplier"),
            ("LF", "--plastic__freeze_geometry_during_warmup | --no-plastic__freeze_geometry_during_warmup", "freeze or permit geometry changes during warmup"),
        ),
    ),
    (
        "PLASTIC DEPTH relative-wall-time objective",
        (
            ("WTD", "--plastic__wall_time_equivalent_time_gain_discount VALUE", "credited fraction of positive equivalent-time gain in [0,1]; default 0.9"),
            ("WTW", "--plastic__wall_time_equivalent_time_gain_loss_rate_window N", "rolling ordinary-training loss-rate window; default 64"),
            ("WTM", "--plastic__wall_time_equivalent_time_gain_loss_rate_min_observations N", "observations required before the loss-rate fit is usable; default 16"),
        ),
    ),
    (
        "Sampling-only chaos bump",
        (
            ("CB", "--chaos_bump__sampling__enabled", "enable sampling-coordinate chaos bumps"),
            ("CB", "--no-chaos_bump__sampling__enabled", "disable sampling-coordinate chaos bumps; default"),
            ("CBL", "--chaos_bump__sampling__initial_lockout__steps N", "initial successful updates during which bumps are prohibited; default 16"),
            ("CBN", "--chaos_bump__sampling__maximum_bumps N", "maximum bumps per run; default 1"),
            ("CBI0", "--chaos_bump__sampling__interlude__min_steps N", "minimum ordinary successful updates between bumps; default 128"),
            ("CBI1", "--chaos_bump__sampling__interlude__max_steps N", "maximum ordinary successful updates between bumps; default 256"),
            ("CBD0", "--chaos_bump__sampling__duration__min_steps N", "minimum bump duration in successful updates; default 16"),
            ("CBD1", "--chaos_bump__sampling__duration__max_steps N", "maximum bump duration in successful updates; default 256"),
            ("CBF", "--chaos_bump__sampling__duration__max_fraction_of_elapsed_steps VALUE", "duration ceiling as a fraction of elapsed successful updates; default 0.05"),
            ("CBM", "--chaos_bump__sampling__max_movement_fraction_of_local_gap VALUE", "maximum coordinate displacement as a fraction of the local gap; default 0.10"),
        ),
    ),
    (
        "Compact geometry",
        (
            ("-B", "BASIS_FAMILY", "chebyshev | dct | haar | lapped_cosine"),
            ("-v", "BASIS_VERSION", "auto or exact registered version"),
            ("-W", "LAPPED_COSINE_WINDOW_LENGTH", "even physical locality window"),
            ("-i", "LAPPED_COSINE_OVERLAP_FRACTION", "currently 0.5 only"),
            ("-a", "ATTENTION_GEOMETRY", "legacy attention geometry selector"),
            ("-m", "MLP_GEOMETRY", "legacy MLP geometry selector"),
            ("—", "--mlp-hidden-compressor NAME", "registered local MLP_HIDDEN compressor"),
            ("s", "--mlp-hidden-group-size N", "physical MLP_HIDDEN segment size"),
        ),
    ),
    (
        "Shape / runtime",
        (
            ("-L", "N_LAYER", "dense depth or maximum PLASTIC capacity"),
            ("-s", "STRATUM_SIZE", "layer-dropout nominal layers per stratum"),
            ("-M", "N_ACTIVE_PER_STRATUM", "active layers per stratum"),
            ("LI", "--layer-dropout-resample-steps N", "updates per sampled layer set"),
            ("-H", "N_HEAD", "attention-head count"),
            ("-D", "N_EMBD", "model width"),
            ("-C", "BLOCK_SIZE", "context length"),
            ("-P", "O_DEPTH", "DEPTH coefficient order"),
            ("-Q", "O_ATTN_D_MODEL", "attention D_MODEL order"),
            ("-J", "O_ATTN_QKV_PER_CHANNEL", "Q/K/V head-channel order"),
            ("-O", "O_ATTN_OUT_PER_CHANNEL", "attention-output head-channel order"),
            ("-X", "O_MLP_D_MODEL", "MLP D_MODEL order"),
            ("-Y", "O_MLP_HIDDEN", "MLP_HIDDEN order"),
            ("DLB", "--depth-compress-layer-norm-and-bias | --no-depth-compress-layer-norm-and-bias", "DEPTH LayerNorm/bias participation"),
            ("-S", "CHECKPOINT_SEGMENT_SIZE", "activation-checkpoint segment size"),
            ("-E", "FAST_DISCARD", "true | false"),
            ("-T", "DTYPE", "float32 | float16 | bfloat16"),
            ("-K", "ATTENTION_BACKEND", "auto | flash2 | sdpa | math"),
        ),
    ),
    (
        "Execution optimisation / debugging",
        (
            ("—", "--depth-materialisation-matmul true|false", "DEPTH matrix materialisation path"),
            ("—", "--materialisation-profiling true|false", "pure DEPTH materialisation timing"),
            ("—", "--torch-compile false|true|regional", "eager | whole-model | regional compile"),
            ("—", "THOG2_BYPASS_SEMANTIC_QKV_ADAPTER=true|false", "semantic Q/K/V adapter bypass"),
            ("—", "THOG2_DIRECT_FACTORISED_MLP=true|false", "direct factorised compact MLP execution"),
            ("—", "THOG2_VECTORISE_PER_HEAD_MATERIALISATION=true|false", "vectorised per-head materialisation"),
        ),
    ),
    (
        "Residual initialisation",
        (
            ("-r", "RESIDUAL_INIT_POLICY", "depth_scaled | unscaled"),
            ("-z", "RESIDUAL_INIT_DEPTH_SOURCE", "true_layer_depth | dof_implied_depth | user_forced_depth"),
            ("-Z", "RESIDUAL_INIT_DEPTH_VALUE", "explicit user-forced depth"),
        ),
    ),
    (
        "Paths",
        (
            ("-d", "DATASET_NAME", "dataset identity"),
            ("-t", "DATA_DIR", "directory containing train.bin and val.bin"),
            ("-o", "CHECKPOINT_ROOT", "checkpoint root"),
            ("-j", "LOG_ROOT", "log root"),
            ("-R", "RESULT_ROOT", "result root"),
            ("-x", "DRY_RUN", "true | false"),
        ),
    ),
    (
        "Advanced registered runner controls",
        (
            ("—", "--model-type TYPE", "dense | sheet; normally wrapper-derived"),
            ("—", "--host-label NAME", "machine label"),
            ("—", "--max-wall-minutes N", "soft wall-clock limit; 0 disables"),
            ("—", "--activation-checkpointing | --no-activation-checkpointing", "enable or disable activation checkpointing"),
            ("c", "--learning-rate VALUE", "direct floating-point maximum LR"),
            ("f", "--min-lr VALUE", "direct floating-point minimum LR"),
            ("—", "--weight-decay VALUE", "optimiser weight decay"),
            ("—", "--beta1 VALUE", "Adam first-moment coefficient"),
            ("—", "--beta2 VALUE", "Adam second-moment coefficient"),
            ("—", "--grad-clip VALUE", "gradient-norm clipping threshold"),
            ("—", "--nonfinite-update-policy POLICY", "raise | skip"),
            ("—", "--dropout VALUE", "model dropout probability"),
            ("—", "--bias | --no-bias", "enable or disable conventional bias parameters"),
            ("—", "--model-seed N", "model initialisation seed"),
            ("—", "--data-seed N", "batch-sampling seed"),
            ("—", "--device DEVICE", "PyTorch device string"),
            ("—", "--experiment-prefix TEXT", "low-level experiment-prefix override"),
            ("—", "--run-start-label LABEL", "explicit run-start label"),
            ("—", "--wandb | --no-wandb", "direct W&B enablement"),
            ("—", "--wandb-project NAME", "W&B project"),
            ("—", "--wandb-entity NAME", "W&B entity"),
            ("—", "--wandb-mode MODE", "online | offline | disabled"),
            ("—", "--wandb-root PATH", "local W&B root"),
            ("—", "--artifact-suffix TEXT", "additional artifact-name suffix"),
            ("—", "--artifact-name-limit N", "maximum artifact component length"),
            ("—", "--log-timestamp TIMESTAMP", "explicit log timestamp"),
            ("—", "--print-artifact-name", "print the resolved artifact name and exit"),
            ("—", "--print-resolved-json", "print the resolved configuration and exit"),
            ("—", "--print-geometry-registry", "print geometry and complete help"),
            ("-h", "--help", "show this help message and exit"),
        ),
    ),
)


_ORIGINAL_FORMAT_HELP = argparse.ArgumentParser.format_help


def descriptor_registry_rows() -> Iterable[DescriptorRow]:
    for _, rows in _DESCRIPTOR_SECTIONS:
        yield from rows


def format_descriptor_registry() -> str:
    rows = tuple(descriptor_registry_rows())
    abbreviation_width = max(len("abbrev"), *(len(row[0]) for row in rows))
    parameter_width = max(len("parameter"), *(len(row[1]) for row in rows))
    lines = [
        "getopt / artifact descriptor registry",
        "------------------------------------",
        f"{'abbrev':<{abbreviation_width}}  {'parameter':<{parameter_width}}  description",
    ]
    for section, section_rows in _DESCRIPTOR_SECTIONS:
        lines.append("")
        lines.append(f"[{section}]")
        lines.extend(
            f"{abbreviation:<{abbreviation_width}}  {parameter:<{parameter_width}}  {description}"
            for abbreviation, parameter, description in section_rows
        )
    return "\n".join(lines)


def _is_thog_training_parser(parser: argparse.ArgumentParser) -> bool:
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    return "--plastic__enabled" in options and "--n-layer" in options


def _format_help_with_descriptor_registry(parser: argparse.ArgumentParser) -> str:
    rendered = _ORIGINAL_FORMAT_HELP(parser)
    if not _is_thog_training_parser(parser):
        return rendered
    return f"{rendered.rstrip()}\n\n{format_descriptor_registry()}\n"


argparse.ArgumentParser.format_help = _format_help_with_descriptor_registry
# ^^^ THOG
