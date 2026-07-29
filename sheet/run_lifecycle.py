# vvv THOG
from __future__ import annotations

import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


LIFECYCLE_SCHEMA_VERSION = 2
_START_LABEL_PATTERN = re.compile(r"^\d{6}-\d{4}$")


def validate_start_label(value: str) -> str:
    label = value.strip().replace("_", "-")
    if not _START_LABEL_PATTERN.fullmatch(label):
        raise ValueError("run start label must have YYMMDD-HHMM form")
    return label


def start_label_from_artifact_name(artifact_name: str) -> Optional[str]:
    candidate = artifact_name[:11]
    try:
        return validate_start_label(candidate)
    except ValueError:
        return None


def fork_suffix(generation: int, root_start_label: str) -> str:
    if generation < 1:
        raise ValueError("fork generation must be positive")
    return f"FORK_{generation}_FROM_{validate_start_label(root_start_label)}"


def _session_record(
    *,
    session_id: str,
    creation_mode: str,
    starting_completed_updates: int,
    target_updates: int,
    wandb_continue_run: bool,
) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "creation_mode": creation_mode,
        "starting_completed_updates": int(starting_completed_updates),
        "target_updates": int(target_updates),
        "wandb_continue_run": bool(wandb_continue_run),
    }


def _base_lifecycle(
    *,
    mode: str,
    config: Any,
    artifact_name: str,
    paths: Mapping[str, Path],
    world_size: int,
    instrumentation_backend: str,
    optimizer_name: str,
    optimizer_momentum: float,
    target_updates: int,
    lr_phase: Mapping[str, Any],
) -> Dict[str, Any]:
    logical_run_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    run_start_label = config.run_start_label or start_label_from_artifact_name(artifact_name)
    lifecycle = {
        "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
        "creation_mode": mode,
        "logical_run_id": logical_run_id,
        "session_id": session_id,
        "artifact_name": artifact_name,
        "artifact_descriptor": config.parameter_artifact_fragment(),
        "artifact_prefix": config.artifact_prefix,
        "run_start_label": run_start_label,
        "root_run_id": logical_run_id,
        "root_start_label": run_start_label,
        "fork_generation": 0,
        "parent_logical_run_id": None,
        "parent_artifact_name": None,
        "parent_checkpoint": None,
        "parent_completed_updates": None,
        "parent_wandb_run_id": None,
        "lineage": [],
        "run_config": asdict(config),
        "checkpoint_path": str(paths["checkpoint_path"]),
        "log_path": str(paths["log_path"]),
        "result_path": str(paths["result_path"]),
        "tensorboard_dir": str(paths["tensorboard_dir"]),
        "world_size": int(world_size),
        "instrumentation_backend": instrumentation_backend,
        "wandb_run_id": None,
        "wandb_run_history": [],
        "optimizer_name": optimizer_name,
        "optimizer_momentum": float(optimizer_momentum),
        "target_updates": int(target_updates),
        "lr_phases": [dict(lr_phase)],
        "active_lr_phase_index": 0,
        "sessions": [],
    }
    lifecycle["sessions"].append(
        _session_record(
            session_id=session_id,
            creation_mode=mode,
            starting_completed_updates=0,
            target_updates=target_updates,
            wandb_continue_run=False,
        )
    )
    return validate_lifecycle(lifecycle)


def fresh_lifecycle(
    *,
    config: Any,
    artifact_name: str,
    paths: Mapping[str, Path],
    world_size: int,
    instrumentation_backend: str,
    optimizer_name: str,
    optimizer_momentum: float,
    lr_phase: Mapping[str, Any],
) -> Dict[str, Any]:
    return _base_lifecycle(
        mode="fresh",
        config=config,
        artifact_name=artifact_name,
        paths=paths,
        world_size=world_size,
        instrumentation_backend=instrumentation_backend,
        optimizer_name=optimizer_name,
        optimizer_momentum=optimizer_momentum,
        target_updates=int(config.max_iters),
        lr_phase=lr_phase,
    )


def resume_lifecycle(
    parent: Mapping[str, Any],
    *,
    config: Any,
    starting_completed_updates: int,
    target_updates: int,
    instrumentation_backend: str,
    wandb_continue_run: bool,
) -> Dict[str, Any]:
    lifecycle = dict(parent)
    lifecycle["creation_mode"] = "resume"
    lifecycle["session_id"] = str(uuid.uuid4())
    lifecycle["run_config"] = asdict(config)
    lifecycle["instrumentation_backend"] = instrumentation_backend
    lifecycle["resume_starting_completed_updates"] = int(starting_completed_updates)
    lifecycle["target_updates"] = int(target_updates)
    sessions = list(lifecycle.get("sessions", []))
    sessions.append(
        _session_record(
            session_id=lifecycle["session_id"],
            creation_mode="resume",
            starting_completed_updates=starting_completed_updates,
            target_updates=target_updates,
            wandb_continue_run=wandb_continue_run,
        )
    )
    lifecycle["sessions"] = sessions
    return validate_lifecycle(lifecycle)


def fork_lifecycle(
    parent: Mapping[str, Any],
    *,
    config: Any,
    artifact_name: str,
    paths: Mapping[str, Path],
    parent_checkpoint: str | Path,
    parent_completed_updates: int,
    target_updates: int,
    world_size: int,
    instrumentation_backend: str,
    wandb_continue_run: bool,
    child_lr_phase: Mapping[str, Any],
) -> Dict[str, Any]:
    parent_lineage = list(parent.get("lineage", []))
    parent_wandb_run_id = parent.get("wandb_run_id")                                                                                                     # <<< THOG retain immediate telemetry ancestry even when the child starts a new W&B run
    wandb_run_history = list(parent.get("wandb_run_history", []))
    if parent_wandb_run_id and parent_wandb_run_id not in wandb_run_history:
        wandb_run_history.append(parent_wandb_run_id)
    parent_record = {
        "logical_run_id": parent.get("logical_run_id"),
        "artifact_name": parent.get("artifact_name"),
        "checkpoint": str(parent_checkpoint),
        "completed_updates": int(parent_completed_updates),
        "fork_generation": int(parent.get("fork_generation", 0)),
        "wandb_run_id": parent_wandb_run_id,                                                                                                            # <<< THOG lineage records the parent's exact W&B identity when available
    }
    logical_run_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    root_start_label = parent.get("root_start_label") or parent.get("run_start_label")
    lifecycle = {
        "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
        "creation_mode": "fork",
        "logical_run_id": logical_run_id,
        "session_id": session_id,
        "artifact_name": artifact_name,
        "artifact_descriptor": config.parameter_artifact_fragment(),
        "artifact_prefix": config.artifact_prefix,
        "run_start_label": config.run_start_label or start_label_from_artifact_name(artifact_name),
        "root_run_id": parent.get("root_run_id") or parent.get("logical_run_id"),
        "root_start_label": root_start_label,
        "fork_generation": int(parent.get("fork_generation", 0)) + 1,
        "parent_logical_run_id": parent.get("logical_run_id"),
        "parent_artifact_name": parent.get("artifact_name"),
        "parent_checkpoint": str(parent_checkpoint),
        "parent_completed_updates": int(parent_completed_updates),
        "parent_wandb_run_id": parent_wandb_run_id,                                                                                                     # <<< THOG child metadata exposes immediate W&B ancestry separately from child identity
        "lineage": parent_lineage + [parent_record],
        "run_config": asdict(config),
        "checkpoint_path": str(paths["checkpoint_path"]),
        "log_path": str(paths["log_path"]),
        "result_path": str(paths["result_path"]),
        "tensorboard_dir": str(paths["tensorboard_dir"]),
        "world_size": int(world_size),
        "instrumentation_backend": instrumentation_backend,
        "wandb_run_id": parent_wandb_run_id if wandb_continue_run else None,
        "wandb_run_history": wandb_run_history,
        "optimizer_name": parent.get("optimizer_name", "adamw"),
        "optimizer_momentum": float(parent.get("optimizer_momentum", 0.9)),
        "target_updates": int(target_updates),
        "lr_phases": list(parent.get("lr_phases", [])) + [dict(child_lr_phase)],
        "active_lr_phase_index": len(list(parent.get("lr_phases", []))),
        "sessions": [],
    }
    lifecycle["sessions"].append(
        _session_record(
            session_id=session_id,
            creation_mode="fork",
            starting_completed_updates=parent_completed_updates,
            target_updates=target_updates,
            wandb_continue_run=wandb_continue_run,
        )
    )
    return validate_lifecycle(lifecycle)


def lifecycle_from_checkpoint(payload: Mapping[str, Any]) -> Dict[str, Any]:
    lifecycle = payload.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        raise ValueError("checkpoint does not contain enhanced lifecycle metadata")
    return validate_lifecycle(dict(lifecycle))


def update_wandb_identity(
    lifecycle: Mapping[str, Any],
    wandb_run_id: Optional[str],
) -> Dict[str, Any]:
    updated = dict(lifecycle)
    if not wandb_run_id:
        return validate_lifecycle(updated)
    previous = updated.get("wandb_run_id")
    history = list(updated.get("wandb_run_history", []))
    if previous and previous != wandb_run_id and previous not in history:
        history.append(previous)
    updated["wandb_run_id"] = wandb_run_id
    updated["wandb_run_history"] = history
    return validate_lifecycle(updated)


def current_session_result_path(lifecycle: Mapping[str, Any]) -> Path:
    result_path = Path(str(lifecycle["result_path"]))
    session_id = str(lifecycle["session_id"])
    return result_path.with_name(f"result_{session_id}.json")


def validate_lifecycle(lifecycle: Mapping[str, Any]) -> Dict[str, Any]:
    required = (
        "lifecycle_schema_version",
        "logical_run_id",
        "session_id",
        "artifact_name",
        "run_config",
        "world_size",
        "target_updates",
        "lr_phases",
    )
    missing = [name for name in required if name not in lifecycle]
    if missing:
        raise ValueError(f"lifecycle metadata missing fields: {missing}")
    version = int(lifecycle["lifecycle_schema_version"])
    if version not in (1, LIFECYCLE_SCHEMA_VERSION):
        raise ValueError(f"unsupported lifecycle schema version: {version}")
    normalized = dict(lifecycle)
    if version == 1:
        normalized["lifecycle_schema_version"] = LIFECYCLE_SCHEMA_VERSION
        normalized.setdefault("root_run_id", normalized.get("logical_run_id"))
        normalized.setdefault("wandb_run_history", [])
        normalized.setdefault("parent_wandb_run_id", None)
        normalized.setdefault("optimizer_name", "adamw")
        normalized.setdefault("optimizer_momentum", 0.9)
        run_config = normalized.get("run_config", {})
        normalized.setdefault("target_updates", int(run_config.get("max_iters", 0)))
    if int(normalized["target_updates"]) < 0:
        raise ValueError("target_updates must be non-negative")
    return normalized


__all__ = [
    "LIFECYCLE_SCHEMA_VERSION",
    "current_session_result_path",
    "fork_lifecycle",
    "fork_suffix",
    "fresh_lifecycle",
    "lifecycle_from_checkpoint",
    "resume_lifecycle",
    "start_label_from_artifact_name",
    "update_wandb_identity",
    "validate_lifecycle",
    "validate_start_label",
]
# ^^^ THOG
