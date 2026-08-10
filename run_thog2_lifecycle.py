# vvv THOG
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

import run_thog2_owt_core as core
from sheet.checkpoint_resolver import ResolvedCheckpoint, resolve_checkpoint
from sheet.checkpoints import load_payload
from sheet.lr_schedule import (
    COSINE_SCHEDULE,
    RESTART_COSINE_SCHEDULE,
    learning_rate_for_lifecycle,
)
from sheet.optimizer_factory import (
    normalize_optimizer_name,
    optimizer_momentum_from_environment,
    optimizer_name_from_environment,
)
from sheet.run_config import OwtRunConfig
from sheet.run_lifecycle import (
    current_session_result_path,
    fork_lifecycle,
    fork_suffix,
    fresh_lifecycle,
    lifecycle_from_checkpoint,
    resume_lifecycle,
    start_label_from_artifact_name,
    update_wandb_identity,
    validate_start_label,
)
from sheet.run_manifest import write_run_manifest
# vvv THOG PLASTIC COARSE/FINE one-shot discovery, fresh FINE reconstruction and review-pause resume
from sheet.distributed import DistributedContext
from sheet.plastic_depth_coarse import (
    ResolvedPlasticCoarseConfig,
    resolve_plastic_coarse_config,
)
from sheet.plastic_depth_coarse_checkpoint import (
    PLASTIC_COARSE_TRIAL_CHECKPOINT_PHASE,
    PlasticCoarseTrialCheckpointState,
)
from sheet.plastic_depth_fresh_state import (
    PlasticFreshTrainingState,
    build_fresh_training_state,
)
from sheet.plastic_depth_lifecycle import run_plastic_coarse_fine_lifecycle
from sheet.plastic_depth_resume import (
    PLASTIC_RESUME_CHECKPOINT_EXIT,
    PLASTIC_RESUME_CONTINUE_FINE,
    resume_plastic_coarse_fine_boundary,
)
# ^^^ THOG
from sheet.training_config import TrainingConfig


# vvv THOG route persisted mid-COARSE state before ordinary FINE-boundary resume handling
def _plastic_coarse_trial_checkpoint_from_payload(
    payload: Mapping[str, Any],
) -> Optional[PlasticCoarseTrialCheckpointState]:
    raw_state = payload.get("plastic_coarse_fine_state")
    if not isinstance(raw_state, Mapping):
        return None
    if raw_state.get("phase") != PLASTIC_COARSE_TRIAL_CHECKPOINT_PHASE:
        return None
    return PlasticCoarseTrialCheckpointState.from_mapping(raw_state)
# ^^^ THOG


_WANDB_BACKENDS = {"wandb", "both"}
_OPERATIONAL_CONFIG_DESTINATIONS = {
    "max_wall_minutes",
    "eval_interval",
    "eval_iters",
    "log_interval",
    "checkpoint_interval",
    "attention_backend",
    "activation_checkpointing",
    "checkpoint_segment_size",
    "nonfinite_update_policy",
    "max_nonfinite_update_skips",
    "device",
    "dtype",
    "wandb_project",
    "wandb_entity",
    "wandb_mode",
    "wandb_root",
}
_ARGUMENT_TO_CONFIG = {
    "model_type": "model_type",
    "host_label": "host_label",
    "run_name": "run_name",
    "dataset": "dataset",
    "data_dir": "data_dir",
    "batch_size": "batch_size",
    "gradient_accumulation_steps": "gradient_accumulation_steps",
    "block_size": "block_size",
    "n_layer": "n_layer",
    "layer_dropout_stratum_size": "layer_dropout_stratum_size",
    "layer_dropout_active_per_stratum": "layer_dropout_active_per_stratum",
    "layer_dropout_resample_steps": "layer_dropout_resample_steps",
    "n_head": "n_head",
    "n_embd": "n_embd",
    "o_depth": "o_depth",
    "o_attn_d_model": "o_attn_d_model",
    "o_attn_qkv_per_channel": "o_attn_qkv_per_channel",
    "o_attn_out_per_channel": "o_attn_out_per_channel",
    "o_mlp_d_model": "o_mlp_d_model",
    "o_mlp_hidden": "o_mlp_hidden",
    "mlp_hidden_group_size": "mlp_hidden_group_size",
    "mlp_hidden_compressor": "mlp_hidden_compressor",
    "depth_compress_layer_norm_and_bias": "depth_compress_layer_norm_and_bias",
    "geometry_preset": "geometry_preset",
    "attention_geometry": "attention_geometry",
    "mlp_geometry": "mlp_geometry",
    "basis_family": "basis_family",
    "basis_version": "basis_version",
    "lapped_cosine_window_length": "lapped_cosine_window_length",
    "lapped_cosine_overlap_fraction": "lapped_cosine_overlap_fraction",
    "experiment_prefix": "experiment_prefix",
    "residual_init_policy": "residual_init_policy",
    "residual_init_depth_source": "residual_init_depth_source",
    "residual_init_depth_value": "residual_init_depth_value",
    "learning_rate": "learning_rate",
    "min_lr": "min_lr",
    "warmup_iters": "warmup_iters",
    "weight_decay": "weight_decay",
    "beta1": "beta1",
    "beta2": "beta2",
    "grad_clip": "grad_clip",
    "dropout": "dropout",
    "bias": "bias",
    "model_seed": "model_seed",
    "data_seed": "data_seed",
    "plastic__coarse_phase": "plastic__coarse_phase",
    "plastic__phase_1_n_steps": "plastic__phase_1_n_steps",
    "plastic__phase_1_starting_layer_count": "plastic__phase_1_starting_layer_count",
    "plastic__phase_1__number_of_trials": "plastic__phase_1__number_of_trials",
    "plastic__phase_1_evaluation_steps_count": "plastic__phase_1_evaluation_steps_count",
    "plastic__layer_count_probe__probe_every_n_steps": "plastic__layer_count_probe__probe_every_n_steps",
    "plastic__layer_count_probe_radius": "plastic__layer_count_probe_radius",
    "plastic__layer_count_max_step": "plastic__layer_count_max_step",
    "artifact_name_limit": "artifact_name_limit",
}


def _action_for_destination(parser: argparse.ArgumentParser, destination: str) -> argparse.Action:
    for action in parser._actions:
        if action.dest == destination:
            return action
    raise KeyError(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = core.build_parser()
    parser.description = "Train, resume, or fork one canonical THOG2 OpenWebText run"

    run_mode_action = _action_for_destination(parser, "run_mode")
    run_mode_action.choices = ("fresh", "resume", "fork")
    max_iters_action = _action_for_destination(parser, "max_iters")
    max_iters_action.default = None
    parser.set_defaults(max_iters=None)

    parser.add_argument("-q", dest="run_mode", choices=("fresh", "resume", "fork"))
    parser.add_argument("-n", dest="max_iters", type=int)
    parser.add_argument("-G", dest="requested_world_size", type=int)
    parser.add_argument(
        "-I",
        "--instrumentation",
        choices=("tensorboard", "wandb", "both", "wandb_offline", "none"),
    )
    parser.add_argument("--resume", dest="resume_selector")
    parser.add_argument("--fork", dest="fork_selector")
    parser.add_argument("--resume-from")
    parser.add_argument("--fork-lr-mode", choices=("restart_cosine",))
    parser.add_argument("--fork-learning-rate", type=float)
    parser.add_argument("--fork-min-lr", type=float)
    parser.add_argument("--fork-rewarm-iters", type=int)
    parser.add_argument(
        "--wandb-continue-run",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--optimizer")
    parser.add_argument("--optimizer-momentum", type=float)
    # vvv THOG keep the optional direct HYPERBLOCK MLP switch available through resume/fork lifecycle parsing
    parser.add_argument(
        "--direct-factorised-hyperblock-mlp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    # ^^^ THOG
    return parser


def explicit_destinations(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> Set[str]:
    option_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    explicit: Set[str] = set()
    for token in argv:
        if not token.startswith("-") or token == "-":
            continue
        option = token.split("=", 1)[0]
        destination = option_dest.get(option)
        if destination is not None:
            explicit.add(destination)
    return explicit


def _resolve_mode_and_selector(
    arguments: argparse.Namespace,
    explicit: Set[str],
) -> Tuple[str, Optional[str]]:
    values = vars(arguments)
    resume_selector = values["resume_selector"]
    fork_selector = values["fork_selector"]
    resume_from = values["resume_from"]
    requested_mode = values["run_mode"] or "fresh"

    if resume_selector is not None and fork_selector is not None:
        raise ValueError("--resume and --fork are mutually exclusive")

    if resume_selector is not None:
        if "run_mode" in explicit and requested_mode != "resume":
            raise ValueError("--resume conflicts with the explicitly selected run mode")
        if resume_from is not None and resume_from != resume_selector:
            raise ValueError("--resume and --resume-from select different checkpoints")
        return "resume", str(resume_selector)

    if fork_selector is not None:
        if "run_mode" in explicit and requested_mode != "fork":
            raise ValueError("--fork conflicts with the explicitly selected run mode")
        if resume_from is not None and resume_from != fork_selector:
            raise ValueError("--fork and --resume-from select different checkpoints")
        return "fork", str(fork_selector)

    if requested_mode == "fresh":
        if resume_from is not None:
            raise ValueError("fresh mode forbids --resume-from")
        return "fresh", None

    if resume_from is None:
        raise ValueError(f"{requested_mode} mode requires --resume-from or the convenience selector")
    return str(requested_mode), str(resume_from)


def _normalize_instrumentation(value: str, wandb_mode: str) -> Tuple[str, str]:
    normalized = value.strip().lower()
    if normalized == "wandb_offline":
        return "wandb", "offline"
    if normalized not in ("tensorboard", "wandb", "both", "none"):
        raise ValueError(f"unsupported instrumentation mode: {value!r}")
    return normalized, wandb_mode


def _selected_instrumentation(
    arguments: argparse.Namespace,
    explicit: Set[str],
    parent_lifecycle: Optional[Mapping[str, Any]],
    parent_config: Optional[OwtRunConfig],
) -> Tuple[str, str]:
    values = vars(arguments)
    requested_wandb_mode = str(values["wandb_mode"])

    if "instrumentation" in explicit:
        return _normalize_instrumentation(str(values["instrumentation"]), requested_wandb_mode)

    if parent_lifecycle is not None:
        backend = str(parent_lifecycle.get("instrumentation_backend", "tensorboard"))
        mode = parent_config.wandb_mode if parent_config is not None else requested_wandb_mode
        return _normalize_instrumentation(backend, mode)

    selected = os.environ.get("THOG2_INSTRUMENTATION", "tensorboard")
    return _normalize_instrumentation(selected, requested_wandb_mode)


def _selected_optimizer_for_fresh(arguments: argparse.Namespace, explicit: Set[str]) -> Tuple[str, float]:
    values = vars(arguments)
    if "optimizer" in explicit:
        optimizer_name = normalize_optimizer_name(str(values["optimizer"]))
    else:
        optimizer_name = optimizer_name_from_environment()
    if "optimizer_momentum" in explicit:
        optimizer_momentum = float(values["optimizer_momentum"])
    else:
        optimizer_momentum = optimizer_momentum_from_environment()
    if not 0.0 <= optimizer_momentum < 1.0:
        raise ValueError("optimizer momentum must be in [0, 1)")
    return optimizer_name, optimizer_momentum


def _optimizer_from_checkpoint(payload: Mapping[str, Any]) -> Tuple[str, float]:
    optimizer = payload.get("optimizer", {})
    groups = optimizer.get("param_groups", []) if isinstance(optimizer, Mapping) else []
    if not groups:
        return "adamw", 0.9
    first = groups[0]
    if not isinstance(first, Mapping):
        return "adamw", 0.9
    optimizer_name = normalize_optimizer_name(str(first.get("thog2_optimizer_name", "adamw")))
    optimizer_momentum = float(first.get("momentum", 0.9))
    return optimizer_name, optimizer_momentum


def _configure_optimizer_environment(optimizer_name: str, optimizer_momentum: float) -> None:
    os.environ["THOG2_OPTIMIZER"] = optimizer_name
    os.environ["THOG2_OPTIMIZER_MOMENTUM"] = str(optimizer_momentum)


def _run_config_field_names() -> Set[str]:
    return {field.name for field in fields(OwtRunConfig)}


def _filtered_run_config(values: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = _run_config_field_names()
    filtered = {name: value for name, value in values.items() if name in allowed}
    filtered["run_mode"] = "resume"
    return filtered


def _run_config_from_lifecycle(lifecycle: Mapping[str, Any]) -> OwtRunConfig:
    run_config = lifecycle.get("run_config")
    if not isinstance(run_config, Mapping):
        raise ValueError("lifecycle metadata does not contain a run_config mapping")
    return OwtRunConfig(**_filtered_run_config(run_config))


def _run_config_from_training_config(
    training_config: TrainingConfig,
    *,
    resolved: ResolvedCheckpoint,
    world_size: int,
    arguments: argparse.Namespace,
    explicit: Set[str],
) -> OwtRunConfig:
    values = vars(arguments)
    model_type = "dense" if training_config.model_type == "dense" else "sheet"
    checkpoint_segment_size = max(1, int(training_config.checkpoint_segment_size))
    run_start_label = start_label_from_artifact_name(resolved.artifact_name)
    data_dir = str(values["data_dir"]) if "data_dir" in explicit else "data/openwebtext"
    dataset = str(values["dataset"]) if "dataset" in explicit else "openwebtext"

    sheet_values: Dict[str, Any] = {}
    if model_type == "sheet":
        sheet_values = {
            "o_depth": training_config.depth_order,
            "o_attn_d_model": training_config.o_attn_d_model or training_config.base_row_order,
            "o_attn_qkv_per_channel": training_config.o_attn_qkv_per_channel or 1,
            "o_attn_out_per_channel": training_config.o_attn_out_per_channel or 1,
            "o_mlp_d_model": training_config.o_mlp_d_model or 1,
            "o_mlp_hidden": training_config.o_mlp_hidden or training_config.mlp_channel_order or 1,
            "mlp_hidden_group_size": training_config.mlp_hidden_group_size,
            "mlp_hidden_compressor": training_config.mlp_hidden_compressor,
            "depth_compress_layer_norm_and_bias": training_config.depth_compress_layer_norm_and_bias,
            "geometry_preset": training_config.geometry_preset,
            "attention_geometry": training_config.attention_geometry,
            "mlp_geometry": training_config.mlp_geometry,
            "basis_family": training_config.basis_family,
            "basis_version": training_config.basis_version,
            "resolved_geometry_plan": training_config.resolved_geometry_plan,
            "lapped_cosine_window_length": training_config.lapped_cosine_window_length,
            "lapped_cosine_overlap_fraction": training_config.lapped_cosine_overlap_fraction,
        }

    return OwtRunConfig(
        model_type=model_type,
        run_mode="resume",
        host_label=socket.gethostname().split(".")[0],
        run_name="RECOVERED",
        dataset=dataset,
        data_dir=data_dir,
        checkpoint_root=str(values["checkpoint_root"]),
        log_root=str(values["log_root"]),
        result_root=str(values["result_root"]),
        wandb_root=str(values["wandb_root"]),
        max_iters=training_config.max_updates,
        max_wall_minutes=training_config.max_wall_minutes,
        eval_interval=training_config.eval_interval,
        eval_iters=training_config.eval_batches,
        log_interval=training_config.log_interval,
        checkpoint_interval=training_config.checkpoint_interval,
        batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps * world_size,
        block_size=training_config.block_size,
        n_layer=training_config.n_layer,
        n_head=training_config.n_head,
        n_embd=training_config.n_embd,
        layer_dropout_stratum_size=training_config.layer_dropout_stratum_size,
        layer_dropout_active_per_stratum=training_config.layer_dropout_active_per_stratum,
        layer_dropout_resample_steps=training_config.layer_dropout_resample_steps,
        **sheet_values,
        attention_backend=str(values["attention_backend"]),
        experiment_prefix="RECOVERED",
        run_start_label=run_start_label,
        residual_init_policy=training_config.residual_init_policy,
        residual_init_depth_source=training_config.residual_init_depth_source,
        residual_init_depth_value=training_config.residual_init_depth_value,
        activation_checkpointing=training_config.checkpoint_segment_size > 0,
        checkpoint_segment_size=checkpoint_segment_size,
        learning_rate=training_config.learning_rate,
        min_lr=training_config.min_learning_rate,
        warmup_iters=training_config.warmup_updates,
        weight_decay=training_config.weight_decay,
        beta1=training_config.beta1,
        beta2=training_config.beta2,
        grad_clip=training_config.grad_clip,
        nonfinite_update_policy=training_config.nonfinite_update_policy,
        max_nonfinite_update_skips=training_config.max_nonfinite_update_skips,
        dropout=training_config.dropout,
        bias=training_config.bias,
        model_seed=training_config.model_seed,
        data_seed=training_config.data_seed,
        device=training_config.device,
        dtype=training_config.dtype,
        wandb_enabled=False,
        wandb_project=str(values["wandb_project"]),
        wandb_entity=values["wandb_entity"],
        wandb_mode=str(values["wandb_mode"]),
        artifact_name_limit=int(values["artifact_name_limit"]),
    )


def _original_lr_phase(training_config: TrainingConfig) -> Dict[str, Any]:
    return {
        "phase_type": COSINE_SCHEDULE,
        "phase_start_update": 0,
        "phase_end_update": int(training_config.decay_updates),
        "phase_peak_lr": float(training_config.learning_rate),
        "phase_min_lr": float(training_config.min_learning_rate),
        "phase_warmup_iters": int(training_config.warmup_updates),
    }


def _fork_lr_phase(
    *,
    completed_updates: int,
    phase_start_lr: float,
    phase_peak_lr: float,
    phase_rewarm_iters: int,
    phase_end_update: int,
    phase_min_lr: float,
) -> Dict[str, Any]:
    return {
        "phase_type": RESTART_COSINE_SCHEDULE,
        "phase_start_update": int(completed_updates),
        "phase_start_lr": float(phase_start_lr),
        "phase_peak_lr": float(phase_peak_lr),
        "phase_rewarm_iters": int(phase_rewarm_iters),
        "phase_end_update": int(phase_end_update),
        "phase_min_lr": float(phase_min_lr),
    }


def _paths_for_artifact(config: OwtRunConfig, artifact_name: str) -> Dict[str, Path]:
    paths = core.artifact_paths(
        artifact_name,
        checkpoint_root=Path(config.checkpoint_root),
        log_root=Path(config.log_root),
        result_root=Path(config.result_root),
        log_timestamp=None,
    )
    paths["manifest_path"] = paths["checkpoint_dir"] / "run_manifest.json"
    paths["tensorboard_dir"] = Path(os.environ.get("THOG2_CURVE_ROOT", "curves")) / artifact_name
    return paths


def _paths_from_lifecycle(
    lifecycle: Mapping[str, Any],
    resolved: ResolvedCheckpoint,
) -> Dict[str, Path]:
    log_path = Path(str(lifecycle["log_path"]))
    result_path = Path(str(lifecycle["result_path"]))
    tensorboard_path = Path(
        str(
            lifecycle.get(
                "tensorboard_dir",
                Path(os.environ.get("THOG2_CURVE_ROOT", "curves")) / resolved.artifact_name,
            )
        )
    )
    return {
        "checkpoint_dir": resolved.checkpoint_dir,
        "checkpoint_path": resolved.checkpoint_path,
        "manifest_path": resolved.checkpoint_dir / "run_manifest.json",
        "log_dir": log_path.parent,
        "log_path": log_path,
        "result_dir": result_path.parent,
        "result_path": result_path,
        "tensorboard_dir": tensorboard_path,
    }


def _distributed_world_size_from_payload(payload: Mapping[str, Any]) -> Optional[int]:
    distributed = payload.get("distributed_training")
    if isinstance(distributed, Mapping) and "world_size" in distributed:
        return int(distributed["world_size"])
    return None


def _synthetic_parent_lifecycle(
    *,
    payload: Mapping[str, Any],
    resolved: ResolvedCheckpoint,
    training_config: TrainingConfig,
    run_config: OwtRunConfig,
    world_size: int,
    instrumentation_backend: str,
    optimizer_name: str,
    optimizer_momentum: float,
) -> Dict[str, Any]:
    paths = _paths_for_artifact(run_config, resolved.artifact_name)
    paths["checkpoint_dir"] = resolved.checkpoint_dir
    paths["checkpoint_path"] = resolved.checkpoint_path
    paths["manifest_path"] = resolved.checkpoint_dir / "run_manifest.json"
    lifecycle = fresh_lifecycle(
        config=run_config,
        artifact_name=resolved.artifact_name,
        paths=paths,
        world_size=world_size,
        instrumentation_backend=instrumentation_backend,
        optimizer_name=optimizer_name,
        optimizer_momentum=optimizer_momentum,
        lr_phase=_original_lr_phase(training_config),
    )
    lifecycle["sessions"] = []
    lifecycle["synthetic_from_pre_enhancement_checkpoint"] = True
    lifecycle["target_updates"] = int(training_config.max_updates)
    lifecycle["checkpoint_path"] = str(resolved.checkpoint_path)
    lifecycle["run_start_label"] = start_label_from_artifact_name(resolved.artifact_name)
    lifecycle["root_start_label"] = lifecycle["run_start_label"]
    return lifecycle


def _assert_optimizer_identity(
    arguments: argparse.Namespace,
    explicit: Set[str],
    optimizer_name: str,
    optimizer_momentum: float,
    mode: str,
) -> None:
    values = vars(arguments)
    if "optimizer" in explicit:
        requested = normalize_optimizer_name(str(values["optimizer"]))
        if requested != optimizer_name:
            raise ValueError(
                f"{mode} material parameter mismatch: optimizer: "
                f"checkpoint={optimizer_name!r}, requested={requested!r}"
            )
    if "optimizer_momentum" in explicit:
        requested_momentum = float(values["optimizer_momentum"])
        if abs(requested_momentum - optimizer_momentum) > 1.0e-15:
            raise ValueError(
                f"{mode} material parameter mismatch: optimizer_momentum: "
                f"checkpoint={optimizer_momentum!r}, requested={requested_momentum!r}"
            )


def _assert_material_arguments(
    arguments: argparse.Namespace,
    explicit: Set[str],
    parent_config: OwtRunConfig,
    mode: str,
) -> None:
    values = vars(arguments)
    if any(name in explicit for name in ("select_depth", "select_element", "geometry_options", "explain_geometry")):
        raise ValueError(
            f"{mode} does not accept systematic geometry reconstruction; "
            "the checkpoint geometry is authoritative"
        )

    for destination, config_name in _ARGUMENT_TO_CONFIG.items():
        if destination not in explicit:
            continue
        if destination in _OPERATIONAL_CONFIG_DESTINATIONS:
            continue
        if mode == "fork" and destination in ("run_name", "experiment_prefix"):
            continue
        requested = values[destination]
        if destination == "basis_version" and requested == "auto":
            continue
        current = getattr(parent_config, config_name)
        if requested != current:
            raise ValueError(
                f"{mode} material parameter mismatch: {config_name}: "
                f"checkpoint={current!r}, requested={requested!r}"
            )

    if "artifact_suffix" in explicit:
        raise ValueError(f"{mode} lifecycle owns --artifact-suffix; user suffixes are not accepted")


def _apply_operational_config(
    parent_config: OwtRunConfig,
    arguments: argparse.Namespace,
    explicit: Set[str],
    *,
    target_updates: int,
    backend: str,
    wandb_mode: str,
) -> OwtRunConfig:
    values = vars(arguments)
    changes: Dict[str, Any] = {
        "run_mode": "resume",
        "max_iters": int(target_updates),
        "wandb_enabled": backend in _WANDB_BACKENDS,
        "wandb_mode": wandb_mode,
    }
    for destination in _OPERATIONAL_CONFIG_DESTINATIONS:
        if destination not in explicit:
            continue
        if destination == "eval_iters":
            changes["eval_iters"] = int(values[destination])
        else:
            changes[destination] = values[destination]
    return replace(parent_config, **changes)


def _target_updates_for_resume(
    arguments: argparse.Namespace,
    explicit: Set[str],
    lifecycle: Mapping[str, Any],
    completed_updates: int,
) -> int:
    values = vars(arguments)
    target = int(values["max_iters"]) if "max_iters" in explicit else int(lifecycle["target_updates"])
    if target <= completed_updates:
        relation = "equal to" if target == completed_updates else "less than"
        raise ValueError(
            f"resume target {target} is {relation} completed_updates {completed_updates}; "
            "-n is the absolute lifetime target, so specify a larger -n to extend the run"
        )
    return target


def _target_updates_for_fork(
    arguments: argparse.Namespace,
    explicit: Set[str],
    completed_updates: int,
) -> int:
    values = vars(arguments)
    if "max_iters" not in explicit:
        raise ValueError("fork requires explicit -n/--max-iters as the absolute child lifetime target")
    target = int(values["max_iters"])
    if target <= completed_updates:
        raise ValueError(
            f"fork target {target} must be greater than parent completed_updates {completed_updates}"
        )
    return target


def _wandb_continue_policy(
    arguments: argparse.Namespace,
    mode: str,
    backend: str,
) -> bool:
    values = vars(arguments)
    if backend not in _WANDB_BACKENDS:
        return False
    explicit_value = values["wandb_continue_run"]
    if explicit_value is not None:
        return bool(explicit_value)
    return mode == "resume"


def _training_config_for_lifecycle(
    checkpoint_config: TrainingConfig,
    config: OwtRunConfig,
    *,
    target_updates: int,
    out_dir: Path,
) -> TrainingConfig:
    values = asdict(checkpoint_config)
    values.update(
        {
            "max_updates": int(target_updates),
            "max_wall_minutes": int(config.max_wall_minutes),
            "eval_interval": int(config.eval_interval),
            "eval_batches": int(config.eval_iters),
            "checkpoint_interval": int(config.checkpoint_interval),
            "checkpoint_segment_size": int(config.checkpoint_segment_size) if config.activation_checkpointing else 0,
            "out_dir": str(out_dir),
            "log_interval": int(config.log_interval),
            "nonfinite_update_policy": config.nonfinite_update_policy,
            "max_nonfinite_update_skips": int(config.max_nonfinite_update_skips),
            "device": config.device,
            "dtype": config.dtype,
        }
    )
    return TrainingConfig(**values)


def _run_start_label(arguments: argparse.Namespace) -> str:
    values = vars(arguments)
    if values["run_start_label"]:
        return validate_start_label(str(values["run_start_label"]))
    if values["log_timestamp"]:
        return validate_start_label(core.compact_log_timestamp(str(values["log_timestamp"])).replace("_", "-"))
    return datetime.now().strftime("%y%m%d-%H%M")


def _protocol_digest(
    config: OwtRunConfig,
    lifecycle: Mapping[str, Any],
    dataset: Mapping[str, Any],
    world_size: int,
) -> str:
    payload = {
        "config": config.canonical_dict(world_size=world_size),
        "lr_phases": lifecycle.get("lr_phases", []),
        "dataset": dict(dataset),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_fresh(
    arguments: argparse.Namespace,
    explicit: Set[str],
    world_size: int,
) -> Dict[str, Any]:
    values = vars(arguments)
    if any(values[name] is not None for name in ("resume_selector", "fork_selector", "resume_from", "fork_lr_mode", "fork_learning_rate", "fork_min_lr", "fork_rewarm_iters")):
        raise ValueError("fresh mode forbids resume/fork lifecycle options")

    if values["max_iters"] is None:
        arguments.max_iters = 100
    if not values["run_start_label"] and not values["log_timestamp"]:
        arguments.run_start_label = _run_start_label(arguments)

    geometry_plan = core.geometry_plan_from_arguments(arguments)
    if values["explain_geometry"]:
        if geometry_plan is None:
            raise ValueError("--explain-geometry requires systematic geometry selections")
        print(core.format_geometry_plan(geometry_plan, detailed=True))
        return {"early_exit": True, "exit_code": 0}

    config = core.config_from_arguments(arguments, geometry_plan=geometry_plan)
    backend, wandb_mode = _selected_instrumentation(arguments, explicit, None, None)
    config = replace(
        config,
        wandb_enabled=backend in _WANDB_BACKENDS,
        wandb_mode=wandb_mode,
    )
    paths = config.paths(log_timestamp=values["log_timestamp"])
    paths["manifest_path"] = paths["checkpoint_dir"] / "run_manifest.json"
    paths["tensorboard_dir"] = Path(os.environ.get("THOG2_CURVE_ROOT", "curves")) / config.artifact_name
    optimizer_name, optimizer_momentum = _selected_optimizer_for_fresh(arguments, explicit)
    lifecycle = fresh_lifecycle(
        config=config,
        artifact_name=config.artifact_name,
        paths=paths,
        world_size=world_size,
        instrumentation_backend=backend,
        optimizer_name=optimizer_name,
        optimizer_momentum=optimizer_momentum,
        lr_phase={
            "phase_type": COSINE_SCHEDULE,
            "phase_start_update": 0,
            "phase_end_update": int(config.max_iters),
            "phase_peak_lr": float(config.learning_rate),
            "phase_min_lr": float(config.min_lr),
            "phase_warmup_iters": int(config.warmup_iters),
        },
    )
    return {
        "early_exit": False,
        "mode": "fresh",
        "config": config,
        "paths": paths,
        "lifecycle": lifecycle,
        "resolved": None,
        "checkpoint_payload": None,
        "checkpoint_training_config": None,
        "world_size": world_size,
        "backend": backend,
        "optimizer_name": optimizer_name,
        "optimizer_momentum": optimizer_momentum,
        "wandb_continue_run": False,
        "append_log": False,
        "geometry_plan": geometry_plan,
    }


def _prepare_parent_context(
    arguments: argparse.Namespace,
    explicit: Set[str],
    *,
    mode: str,
    selector: str,
) -> Dict[str, Any]:
    values = vars(arguments)
    resolved = resolve_checkpoint(selector, values["checkpoint_root"])
    payload = load_payload(resolved.checkpoint_path)
    trainer_config_values = payload.get("trainer_config")
    if not isinstance(trainer_config_values, Mapping):
        raise ValueError(
            "no-brainer resume/fork requires a checkpoint containing trainer_config; "
            "use the existing explicit legacy loader for older schema-less checkpoints"
        )
    checkpoint_training_config = TrainingConfig(**dict(trainer_config_values))
    checkpoint_completed_updates = int(payload.get("completed_updates", 0))
    plastic_coarse_trial_checkpoint = _plastic_coarse_trial_checkpoint_from_payload(payload)                                      # <<< THOG distinguish COARSE trial progress from FINE optimizer progress
    if plastic_coarse_trial_checkpoint is not None and mode == "fork":
        raise ValueError("fork is not permitted from a mid-COARSE-trial checkpoint")
    completed_updates = (                                                                                                         # <<< THOG FINE starts at optimizer step zero after resumed COARSE selection
        0
        if plastic_coarse_trial_checkpoint is not None
        else checkpoint_completed_updates
    )
    checkpoint_world_size = _distributed_world_size_from_payload(payload)
    requested_world_size = values["requested_world_size"]
    process_world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if requested_world_size is not None:
        world_size = int(requested_world_size)
    elif checkpoint_world_size is not None:
        world_size = int(checkpoint_world_size)
    else:
        world_size = process_world_size
    if world_size < 1:
        raise ValueError("world size must be positive")

    optimizer_name, optimizer_momentum = _optimizer_from_checkpoint(payload)

    raw_lifecycle = payload.get("lifecycle")
    if isinstance(raw_lifecycle, Mapping):
        parent_lifecycle = lifecycle_from_checkpoint(payload)
        parent_config = _run_config_from_lifecycle(parent_lifecycle)
    else:
        parent_config = _run_config_from_training_config(
            checkpoint_training_config,
            resolved=resolved,
            world_size=world_size,
            arguments=arguments,
            explicit=explicit,
        )
        provisional_backend, _ = _selected_instrumentation(arguments, explicit, None, parent_config)
        parent_lifecycle = _synthetic_parent_lifecycle(
            payload=payload,
            resolved=resolved,
            training_config=checkpoint_training_config,
            run_config=parent_config,
            world_size=world_size,
            instrumentation_backend=provisional_backend,
            optimizer_name=optimizer_name,
            optimizer_momentum=optimizer_momentum,
        )

    parent_world_size = int(parent_lifecycle.get("world_size", world_size))
    if world_size != parent_world_size:
        raise ValueError(
            f"{mode} world size mismatch: checkpoint={parent_world_size}, requested={world_size}"
        )

    _assert_optimizer_identity(
        arguments,
        explicit,
        optimizer_name,
        optimizer_momentum,
        mode,
    )
    _assert_material_arguments(arguments, explicit, parent_config, mode)

    backend, wandb_mode = _selected_instrumentation(
        arguments,
        explicit,
        parent_lifecycle,
        parent_config,
    )

    if mode == "resume":
        if any(values[name] is not None for name in ("fork_lr_mode", "fork_learning_rate", "fork_min_lr", "fork_rewarm_iters")):
            raise ValueError("resume rejects fork-only learning-rate options")
        target_updates = _target_updates_for_resume(
            arguments,
            explicit,
            parent_lifecycle,
            completed_updates,
        )
        config = _apply_operational_config(
            parent_config,
            arguments,
            explicit,
            target_updates=target_updates,
            backend=backend,
            wandb_mode=wandb_mode,
        )
        paths = _paths_from_lifecycle(parent_lifecycle, resolved)
        wandb_continue_run = _wandb_continue_policy(arguments, mode, backend)
        lifecycle = resume_lifecycle(
            parent_lifecycle,
            config=config,
            starting_completed_updates=completed_updates,
            target_updates=target_updates,
            instrumentation_backend=backend,
            wandb_continue_run=wandb_continue_run,
        )
        append_log = True
    else:
        if values["fork_lr_mode"] != "restart_cosine":
            raise ValueError("fork requires --fork-lr-mode restart_cosine")
        missing = [
            name
            for name in ("fork_learning_rate", "fork_min_lr", "fork_rewarm_iters")
            if values[name] is None
        ]
        if missing:
            raise ValueError(f"restart_cosine fork is missing required controls: {missing}")
        target_updates = _target_updates_for_fork(arguments, explicit, completed_updates)
        phase_peak_lr = float(values["fork_learning_rate"])
        phase_min_lr = float(values["fork_min_lr"])
        phase_rewarm_iters = int(values["fork_rewarm_iters"])
        if phase_peak_lr <= 0.0 or phase_min_lr < 0.0 or phase_min_lr > phase_peak_lr:
            raise ValueError("fork learning rates require peak > 0 and 0 <= min <= peak")
        if phase_rewarm_iters < 0:
            raise ValueError("fork rewarm iterations must be non-negative")
        phase_start_lr = learning_rate_for_lifecycle(
            checkpoint_training_config,
            parent_lifecycle,
            completed_updates,
        )
        root_start_label = parent_lifecycle.get("root_start_label") or parent_lifecycle.get("run_start_label")
        if not root_start_label:
            raise ValueError("fork requires a recoverable YYMMDD-HHMM root start label")
        generation = int(parent_lifecycle.get("fork_generation", 0)) + 1
        child_changes: Dict[str, Any] = {
            "run_mode": "resume",
            "max_iters": target_updates,
            "run_start_label": _run_start_label(arguments),
            "artifact_suffix": fork_suffix(generation, str(root_start_label)),
            "learning_rate": phase_peak_lr,
            "min_lr": phase_min_lr,
            "warmup_iters": phase_rewarm_iters,
            "wandb_enabled": backend in _WANDB_BACKENDS,
            "wandb_mode": wandb_mode,
        }
        if "run_name" in explicit:
            child_changes["run_name"] = str(values["run_name"])
        if "experiment_prefix" in explicit:
            child_changes["experiment_prefix"] = str(values["experiment_prefix"])
        config = replace(parent_config, **child_changes)
        config = _apply_operational_config(
            config,
            arguments,
            explicit,
            target_updates=target_updates,
            backend=backend,
            wandb_mode=wandb_mode,
        )
        artifact_name = config.artifact_name
        paths = _paths_for_artifact(config, artifact_name)
        wandb_continue_run = _wandb_continue_policy(arguments, mode, backend)
        lifecycle = fork_lifecycle(
            parent_lifecycle,
            config=config,
            artifact_name=artifact_name,
            paths=paths,
            parent_checkpoint=resolved.checkpoint_path,
            parent_completed_updates=completed_updates,
            target_updates=target_updates,
            world_size=world_size,
            instrumentation_backend=backend,
            wandb_continue_run=wandb_continue_run,
            child_lr_phase=_fork_lr_phase(
                completed_updates=completed_updates,
                phase_start_lr=phase_start_lr,
                phase_peak_lr=phase_peak_lr,
                phase_rewarm_iters=phase_rewarm_iters,
                phase_end_update=target_updates,
                phase_min_lr=phase_min_lr,
            ),
        )
        append_log = False

    return {
        "early_exit": False,
        "mode": mode,
        "config": config,
        "paths": paths,
        "lifecycle": lifecycle,
        "resolved": resolved,
        "checkpoint_payload": payload,
        "checkpoint_training_config": checkpoint_training_config,
        "completed_updates": completed_updates,
        "world_size": world_size,
        "backend": backend,
        "optimizer_name": optimizer_name,
        "optimizer_momentum": optimizer_momentum,
        "wandb_continue_run": wandb_continue_run,
        "append_log": append_log,
        "geometry_plan": None,
    }


def prepare_context(
    arguments: argparse.Namespace,
    explicit: Set[str],
) -> Dict[str, Any]:
    mode, selector = _resolve_mode_and_selector(arguments, explicit)
    if mode == "fresh":
        values = vars(arguments)
        world_size = int(values["requested_world_size"] or os.environ.get("WORLD_SIZE", "1"))
        return _prepare_fresh(arguments, explicit, world_size)
    assert selector is not None
    return _prepare_parent_context(
        arguments,
        explicit,
        mode=mode,
        selector=selector,
    )


def resolved_payload(context: Mapping[str, Any]) -> Dict[str, Any]:
    config: OwtRunConfig = context["config"]
    lifecycle: Mapping[str, Any] = context["lifecycle"]
    paths: Mapping[str, Path] = context["paths"]
    return {
        "artifact_name": str(lifecycle["artifact_name"]),
        "artifact_descriptor": str(lifecycle.get("artifact_descriptor", config.parameter_artifact_fragment())),
        "artifact_prefix": config.artifact_prefix,
        "model_type": config.model_type,
        "run_mode": context["mode"],
        "world_size": int(context["world_size"]),
        "tokens_per_iter": config.tokens_per_iter(),
        "append_log": bool(context["append_log"]),
        "session_id": lifecycle["session_id"],
        "target_updates": int(lifecycle["target_updates"]),
        "wandb_continue_run": bool(context["wandb_continue_run"]),
        "canonical_config": config.canonical_dict(world_size=int(context["world_size"])),
        "paths": {name: str(path) for name, path in paths.items()},
    }


def _configure_instrumentation_environment(context: Mapping[str, Any]) -> None:
    backend = str(context["backend"])
    lifecycle = context["lifecycle"]
    config: OwtRunConfig = context["config"]
    os.environ["THOG2_INSTRUMENTATION"] = backend
    tensorboard_dir = Path(str(lifecycle["tensorboard_dir"]))
    os.environ["THOG2_CURVE_ROOT"] = str(tensorboard_dir.parent)
    os.environ["WANDB_MODE"] = config.wandb_mode

    wandb_continue_run = bool(context["wandb_continue_run"])
    if backend in _WANDB_BACKENDS and wandb_continue_run:
        wandb_run_id = lifecycle.get("wandb_run_id")
        if not wandb_run_id:
            raise ValueError(
                "W&B continuation was requested but the checkpoint has no recoverable W&B run ID; "
                "use --no-wandb-continue-run to start a new telemetry run"
            )
        os.environ["WANDB_RUN_ID"] = str(wandb_run_id)
        os.environ["WANDB_RESUME"] = "must"
        if context["mode"] == "fork":
            print(
                "THOG2 WARNING: this fork is explicitly continuing the parent W&B run; "
                "continuing the parent or creating sibling forks can write conflicting values at the same optimizer/update steps.",
                flush=True,
            )
    else:
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)


def _print_schedule_startup(
    mode: str,
    lifecycle: Mapping[str, Any],
    training_config: TrainingConfig,
    completed_updates: int,
) -> None:
    first_lr = learning_rate_for_lifecycle(training_config, lifecycle, completed_updates)
    print(f"{mode} schedule", flush=True)
    print(f"  completed steps:             {completed_updates}", flush=True)
    print(f"  total steps:                 {int(lifecycle['target_updates'])}", flush=True)
    print(f"  remaining steps:             {int(lifecycle['target_updates']) - completed_updates}", flush=True)
    phases = lifecycle.get("lr_phases", [])
    active_index = int(lifecycle.get("active_lr_phase_index", max(0, len(phases) - 1)))
    active = phases[active_index] if phases else {"phase_type": COSINE_SCHEDULE}
    print(f"  schedule:                    {active.get('phase_type', COSINE_SCHEDULE)}", flush=True)
    if active.get("phase_type") == RESTART_COSINE_SCHEDULE:
        print(f"  active phase end:            {int(active['phase_end_update'])}", flush=True)
    else:
        print(f"  original decay end:          {training_config.decay_updates}", flush=True)
    print(f"  first {mode} step LR:         {first_lr:.3e}", flush=True)
    print(flush=True)


def _training_config_for_context(context: Mapping[str, Any], dataset: Mapping[str, Any]) -> TrainingConfig:
    config: OwtRunConfig = context["config"]
    paths: Mapping[str, Path] = context["paths"]
    if context["mode"] == "fresh":
        return config.to_training_config(
            vocab_size=int(dataset["vocab_size"]),
            world_size=int(context["world_size"]),
            out_dir=paths["checkpoint_dir"],
        )
    checkpoint_training_config: TrainingConfig = context["checkpoint_training_config"]
    if checkpoint_training_config.vocab_size != int(dataset["vocab_size"]):
        raise ValueError(
            "dataset vocab size differs from checkpoint: "
            f"checkpoint={checkpoint_training_config.vocab_size}, current={dataset['vocab_size']}"
        )
    return _training_config_for_lifecycle(
        checkpoint_training_config,
        config,
        target_updates=int(context["lifecycle"]["target_updates"]),
        out_dir=paths["checkpoint_dir"],
    )


def _resume_overrides(training_config: TrainingConfig) -> Dict[str, Any]:
    return {
        "device": training_config.device,
        "dtype": training_config.dtype,
        "max_updates": training_config.max_updates,
        "max_wall_minutes": training_config.max_wall_minutes,
        "eval_interval": training_config.eval_interval,
        "eval_batches": training_config.eval_batches,
        "checkpoint_interval": training_config.checkpoint_interval,
        "checkpoint_segment_size": training_config.checkpoint_segment_size,
        "out_dir": training_config.out_dir,
        "log_interval": training_config.log_interval,
        "nonfinite_update_policy": training_config.nonfinite_update_policy,
        "max_nonfinite_update_skips": training_config.max_nonfinite_update_skips,
    }


def _verify_actual_world_size(context: Mapping[str, Any]) -> None:
    actual_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    expected_world_size = int(context["world_size"])
    if actual_world_size != expected_world_size:
        raise ValueError(
            f"training process world size mismatch: expected {expected_world_size}, actual {actual_world_size}; "
            "invoke through train_OWT.sh or torch.distributed.run with the checkpoint world size"
        )


# vvv THOG lifecycle mode applies the wrapper-only direct HYPERBLOCK MLP option before model construction
def _configure_direct_factorised_hyperblock_mlp_environment(
    arguments: argparse.Namespace,
) -> None:
    requested = vars(arguments)["direct_factorised_hyperblock_mlp"]
    if requested is None:
        return
    os.environ["THOG2_DIRECT_FACTORISED_HYPERBLOCK_MLP"] = (
        "true" if bool(requested) else "false"
    )
# ^^^ THOG


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    arguments = parser.parse_args(actual_argv)
    explicit = explicit_destinations(parser, actual_argv)
    _configure_direct_factorised_hyperblock_mlp_environment(arguments)                                                                                      # <<< THOG apply lifecycle wrapper-only option before any context or model construction
    context = prepare_context(arguments, explicit)
    if context.get("early_exit"):
        return int(context["exit_code"])

    payload = resolved_payload(context)
    values = vars(arguments)
    if values["print_artifact_name"]:
        print(payload["artifact_name"])
        return 0
    if values["print_resolved_json"] or values["dry_run"]:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    _verify_actual_world_size(context)
    config: OwtRunConfig = context["config"]
    paths: Dict[str, Path] = context["paths"]
    lifecycle: Dict[str, Any] = context["lifecycle"]
    checkpoint_path = paths["checkpoint_path"]
    if context["mode"] in ("fresh", "fork") and checkpoint_path.exists():
        raise FileExistsError(f"{context['mode']} run refuses to overwrite {checkpoint_path}")

    _configure_optimizer_environment(
        str(context["optimizer_name"]),
        float(context["optimizer_momentum"]),
    )
    _configure_instrumentation_environment(context)
    core.configure_attention_backend(config.attention_backend)

    dataset_dir = Path(config.data_dir)
    dataset = core.validate_dataset(dataset_dir, config.block_size)
    training_config = _training_config_for_context(context, dataset)

    for key in ("checkpoint_dir", "log_dir", "result_dir", "tensorboard_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        write_run_manifest(paths["manifest_path"], lifecycle)

    train_tokens = core.load_tokens(dataset_dir / "train.bin")
    validation_tokens = core.load_tokens(dataset_dir / "val.bin")
    fresh_state = None
    coarse_fine_outcome = None

    # vvv THOG one callback persists both periodic COARSE progress and the fresh FINE review-pause boundary
    def checkpoint_plastic_coarse_fine(
        coarse_trainer: Any,
        state: Mapping[str, object],
    ) -> None:
        paused_lifecycle = {**lifecycle, "plastic_coarse_fine": dict(state)}
        coarse_trainer.lifecycle_metadata = paused_lifecycle
        coarse_trainer.plastic_coarse_fine_state = dict(state)
        coarse_trainer.save_checkpoint(checkpoint_path)
        if coarse_trainer.distributed.is_primary:
            write_run_manifest(paths["manifest_path"], paused_lifecycle)
    # ^^^ THOG

    if context["mode"] == "fresh":
        # vvv THOG COARSE is a one-shot pre-FINE lifecycle; disabled PLASTIC remains on the exact established constructor
        if training_config.plastic__enabled and training_config.plastic__coarse_phase == "enabled":
            coarse_config = resolve_plastic_coarse_config(
                coarse_phase=training_config.plastic__coarse_phase,
                plastic_enabled=training_config.plastic__enabled,
                do_learn_layer_count=training_config.plastic__do_learn_layer_count,
                n_steps=training_config.plastic__phase_1_n_steps,
                starting_layer_count=training_config.plastic__phase_1_starting_layer_count,
                number_of_trials=training_config.plastic__phase_1__number_of_trials,
                evaluation_steps_count=training_config.plastic__phase_1_evaluation_steps_count,
                max_permitted_layers=training_config.plastic__max_permitted_layers,
            )
            coarse_fine_outcome = run_plastic_coarse_fine_lifecycle(
                trainer_factory=core.OwtTrainer,
                resolved_config=training_config,
                train_tokens=train_tokens,
                validation_tokens=validation_tokens,
                coarse_config=coarse_config,
                objective=training_config.plastic__layer_count_objective,
                maximum_layers=int(training_config.plastic__max_permitted_layers),
                cost_weight=float(training_config.plastic__layer_count_cost_weight),
                memory_budget_gib=training_config.plastic__layer_count__memory_budget_gib,
                geometry_initialisation=training_config.plastic__layer_sampling_initialisation,
                checkpoint_callback=checkpoint_plastic_coarse_fine,
                coarse_checkpoint_interval=int(training_config.checkpoint_interval),
            )
            if coarse_fine_outcome.fine_state is None:
                coarse_fine_outcome.close_coordinator()
                return 0
            fresh_state = coarse_fine_outcome.fine_state
            trainer = fresh_state.trainer
            training_config = trainer.config
            lifecycle = {
                **lifecycle,
                "plastic_coarse_fine": dict(coarse_fine_outcome.provenance),
            }
            if trainer.distributed.is_primary:
                write_run_manifest(paths["manifest_path"], lifecycle)
        elif training_config.plastic__enabled:
            fresh_state = build_fresh_training_state(
                trainer_factory=core.OwtTrainer,
                resolved_config=training_config,
                train_tokens=train_tokens,
                validation_tokens=validation_tokens,
                phase="fine",
                active_layer_count=int(training_config.plastic__initial_active_layers),
                instrumentation_namespace="fine",
            )
            trainer = fresh_state.trainer
        else:
            trainer = core.OwtTrainer(training_config, train_tokens, validation_tokens)
        # ^^^ THOG
    else:
        resolved: ResolvedCheckpoint = context["resolved"]
        checkpoint_payload = context["checkpoint_payload"]
        plastic_coarse_resume = _plastic_coarse_trial_checkpoint_from_payload(                                                   # <<< THOG route mid-trial state before ordinary resume
            checkpoint_payload
        )
        coarse_resume_coordinator = (
            DistributedContext.from_environment(str(training_config.device))
            if plastic_coarse_resume is not None
            else None
        )
        try:
            trainer = core.OwtTrainer.from_checkpoint(
                resolved.checkpoint_path,
                train_tokens,
                validation_tokens,
                expected_config=training_config,
                overrides=_resume_overrides(training_config),
            )
        except BaseException:
            if coarse_resume_coordinator is not None:
                coarse_resume_coordinator.close()
            raise

        if plastic_coarse_resume is not None:
            # vvv THOG complete the interrupted COARSE trial and remaining candidates before constructing fresh FINE state
            resumed_coarse_state = PlasticFreshTrainingState(
                trainer=trainer,
                phase="coarse",
                active_layer_count=plastic_coarse_resume.current_trial_layers,
                instrumentation_namespace=(
                    f"coarse/trial_{plastic_coarse_resume.current_trial_index}"
                ),
                fingerprint={},
            )
            resumed_coarse_config = ResolvedPlasticCoarseConfig(
                enabled=True,
                candidate_layers=plastic_coarse_resume.candidate_layers,
                n_steps=plastic_coarse_resume.n_steps,
                evaluation_steps_count=(
                    plastic_coarse_resume.evaluation_steps_count
                ),
            )
            coarse_fine_outcome = run_plastic_coarse_fine_lifecycle(
                trainer_factory=core.OwtTrainer,
                resolved_config=training_config,
                train_tokens=train_tokens,
                validation_tokens=validation_tokens,
                coarse_config=resumed_coarse_config,
                objective=plastic_coarse_resume.objective,
                maximum_layers=plastic_coarse_resume.maximum_layers,
                cost_weight=plastic_coarse_resume.cost_weight,
                memory_budget_gib=plastic_coarse_resume.memory_budget_gib,
                geometry_initialisation=(
                    plastic_coarse_resume.geometry_initialisation
                ),
                checkpoint_callback=checkpoint_plastic_coarse_fine,
                coarse_checkpoint_interval=int(training_config.checkpoint_interval),
                resume_checkpoint_state=plastic_coarse_resume.structured(),
                resume_state=resumed_coarse_state,
                distributed_coordinator=coarse_resume_coordinator,
            )
            if coarse_fine_outcome.fine_state is None:
                coarse_fine_outcome.close_coordinator()
                return 0
            fresh_state = coarse_fine_outcome.fine_state
            trainer = fresh_state.trainer
            training_config = trainer.config
            lifecycle = {
                **lifecycle,
                "plastic_coarse_fine": dict(coarse_fine_outcome.provenance),
            }
            if trainer.distributed.is_primary:
                write_run_manifest(paths["manifest_path"], lifecycle)
            # ^^^ THOG
        else:
            resume_boundary = resume_plastic_coarse_fine_boundary(
                trainer,
                checkpoint_path,
            )
            if resume_boundary == PLASTIC_RESUME_CHECKPOINT_EXIT:
                trainer.close()
                return 0
            if resume_boundary == PLASTIC_RESUME_CONTINUE_FINE:
                lifecycle = {
                    **lifecycle,
                    "plastic_coarse_fine": dict(trainer.plastic_coarse_fine_state),
                }
                if trainer.distributed.is_primary:
                    write_run_manifest(paths["manifest_path"], lifecycle)
    trainer.lifecycle_metadata = dict(lifecycle)

    canonical = config.canonical_dict(world_size=int(context["world_size"]))
    source = core.source_identity()
    telemetry = core.WandbTelemetry(
        enabled=(str(context["backend"]) != "none" and trainer.distributed.is_primary),
        project=config.wandb_project,
        entity=config.wandb_entity,
        mode=config.wandb_mode,
        root=Path(config.wandb_root),
        name=str(lifecycle["artifact_name"]),
        group=config.experiment_prefix,
        job_type="dense2" if config.model_type == "dense" else "sheet",
        config={
            **canonical,
            "source_commit": source["commit"],
            "source_branch": source["branch"],
            "dataset_record": dataset,
            "parameter_report": trainer.parameter_report,
            "lifecycle": lifecycle,
        },
    )

    telemetry_exit_code: Optional[int] = None
    session_result_path = current_session_result_path(lifecycle)
    try:
        if trainer.distributed.is_primary:
            telemetry.start()
            expected_wandb_id = lifecycle.get("wandb_run_id") if context["wandb_continue_run"] else None
            actual_wandb_id = str(getattr(telemetry.run, "id", "")) if telemetry.run is not None else ""
            if expected_wandb_id:
                if actual_wandb_id != str(expected_wandb_id):
                    raise RuntimeError(
                        f"W&B strict continuation failed: expected id {expected_wandb_id!r}, got {actual_wandb_id!r}"
                    )
                if config.wandb_mode == "online" and os.environ.get("WANDB_MODE", "online").lower() == "offline":
                    raise RuntimeError("W&B strict online continuation fell back to offline mode")
            if actual_wandb_id:
                lifecycle = update_wandb_identity(lifecycle, actual_wandb_id)
                trainer.lifecycle_metadata = dict(lifecycle)
                write_run_manifest(paths["manifest_path"], lifecycle)
            telemetry.add_initial_summary(trainer.parameter_report)
            coarse_telemetry = lifecycle.get("plastic_coarse_fine")
            if isinstance(coarse_telemetry, Mapping):
                telemetry.log_plastic_coarse_fine(coarse_telemetry)

        gathered_lifecycle = trainer.distributed.all_gather_object(
            lifecycle if trainer.distributed.is_primary else None
        )
        lifecycle = gathered_lifecycle[0]
        trainer.lifecycle_metadata = dict(lifecycle)
        trainer.distributed.barrier()
        core.attach_telemetry(trainer, telemetry)

        if trainer.distributed.is_primary:
            if context["mode"] in ("resume", "fork"):
                _print_schedule_startup(
                    str(context["mode"]),
                    lifecycle,
                    training_config,
                    int(context["completed_updates"]),
                )
            geometry_plan = context.get("geometry_plan")
            if geometry_plan is not None:
                print(core.format_geometry_plan(geometry_plan), flush=True)
                print(flush=True)
            core.print_model_parameters_and_options(config, trainer)

        result = trainer.run_pilot(
            run_id=str(lifecycle["artifact_name"]),
            protocol_sha256=_protocol_digest(config, lifecycle, dataset, int(context["world_size"])),
            dataset=dataset,
            result_path=session_result_path,
        )
        result["artifact"] = {
            "name": str(lifecycle["artifact_name"]),
            "descriptor": lifecycle.get("artifact_descriptor"),
            "prefix": config.artifact_prefix,
            "paths": {name: str(path) for name, path in paths.items()},
        }
        result["lifecycle"] = lifecycle
        # vvv THOG keep COARSE trial evidence and selected-count provenance in the canonical result and telemetry payload
        if hasattr(trainer, "plastic_coarse_provenance"):
            result["plastic_coarse_fine"] = trainer.plastic_coarse_provenance
        # ^^^ THOG
        result["canonical_config"] = canonical
        result["source"] = source
        result["timing"]["session_training_seconds"] = result["timing"]["training_seconds"]
        if trainer.distributed.is_primary:
            session_result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            paths["result_path"].write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            write_run_manifest(paths["manifest_path"], lifecycle)
            telemetry.add_final_result(result)
            print(
                json.dumps(
                    {
                        "artifact_name": str(lifecycle["artifact_name"]),
                        "checkpoint": str(checkpoint_path),
                        "result": str(paths["result_path"]),
                        "session_result": str(session_result_path),
                        "completed_updates": result["budget"]["completed_updates"],
                        "consumed_tokens": result["budget"]["consumed_tokens"],
                        "final_validation_loss": result["evaluations"][-1]["val"] if result["evaluations"] else None,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return 0
    except KeyboardInterrupt:
        telemetry_exit_code = 0
        if trainer.distributed.is_primary:
            print("interrupted by Ctrl-C; finishing telemetry cleanly", flush=True)
        return 130
    finally:
        if rank == 0:
            telemetry.finish(exit_code=telemetry_exit_code)
        trainer.close()
        if fresh_state is not None:
            fresh_state.trainer = None
        if coarse_fine_outcome is not None:
            coarse_fine_outcome.close_coordinator()


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
