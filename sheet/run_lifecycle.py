# vvv THOG
from __future__ import annotations

import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping


LIFECYCLE_SCHEMA_VERSION = 1
_START_LABEL_PATTERN = re.compile(r"^\d{6}-\d{4}$")


def validate_start_label(value: str) -> str:
    label = value.strip().replace("_", "-")
    if not _START_LABEL_PATTERN.fullmatch(label):
        raise ValueError("run start label must have YYMMDD-HHMM form")
    return label


def fork_suffix(generation: int, root_start_label: str) -> str:
    if generation < 1:
        raise ValueError("fork generation must be positive")
    return f"FORK_{generation}_FROM_{validate_start_label(root_start_label)}"


def _base_lifecycle(*, mode: str, config: Any, paths: Mapping[str, Path], world_size: int, instrumentation_backend: str, execution_options: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
        "creation_mode": mode,
        "logical_run_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "artifact_name": config.artifact_name,
        "artifact_descriptor": config.parameter_artifact_fragment(),
        "artifact_prefix": config.artifact_prefix,
        "run_start_label": config.run_start_label,
        "root_start_label": config.run_start_label,
        "fork_generation": 0,
        "parent_logical_run_id": None,
        "parent_artifact_name": None,
        "parent_checkpoint": None,
        "parent_completed_updates": None,
        "lineage": [],
        "run_config": asdict(config),
        "checkpoint_path": str(paths["checkpoint_path"]),
        "log_path": str(paths["log_path"]),
        "result_path": str(paths["result_path"]),
        "tensorboard_dir": str(paths["tensorboard_dir"]),
        "world_size": int(world_size),
        "instrumentation_backend": instrumentation_backend,
        "wandb_run_id": None,
        "execution_options": dict(execution_options),
        "lr_phases": [],
        "active_lr_phase_index": 0,
        "sessions": [],
    }


def _append_session(lifecycle: Dict[str, Any]) -> Dict[str, Any]:
    session = {"session_id": lifecycle["session_id"], "creation_mode": lifecycle["creation_mode"]}
    lifecycle.setdefault("sessions", []).append(session)
    return lifecycle


def fresh_lifecycle(*, config: Any, paths: Mapping[str, Path], world_size: int, instrumentation_backend: str, execution_options: Mapping[str, Any], lr_phase: Mapping[str, Any]) -> Dict[str, Any]:
    lifecycle = _base_lifecycle(mode="fresh", config=config, paths=paths, world_size=world_size, instrumentation_backend=instrumentation_backend, execution_options=execution_options)
    lifecycle["lr_phases"] = [dict(lr_phase)]
    return validate_lifecycle(_append_session(lifecycle))


def resume_lifecycle(parent: Mapping[str, Any], *, config: Any, starting_completed_updates: int, instrumentation_backend: str, execution_options: Mapping[str, Any]) -> Dict[str, Any]:
    lifecycle = dict(parent)
    lifecycle["creation_mode"] = "resume"
    lifecycle["session_id"] = str(uuid.uuid4())
    lifecycle["run_config"] = asdict(config)
    lifecycle["instrumentation_backend"] = instrumentation_backend
    lifecycle["execution_options"] = dict(execution_options)
    lifecycle["resume_starting_completed_updates"] = int(starting_completed_updates)
    return validate_lifecycle(_append_session(lifecycle))


def fork_lifecycle(parent: Mapping[str, Any], *, config: Any, paths: Mapping[str, Path], parent_checkpoint: str | Path, parent_completed_updates: int, world_size: int, instrumentation_backend: str, execution_options: Mapping[str, Any], child_lr_phase: Mapping[str, Any]) -> Dict[str, Any]:
    parent_lineage = list(parent.get("lineage", []))
    parent_record = {
        "logical_run_id": parent.get("logical_run_id"),
        "artifact_name": parent.get("artifact_name"),
        "checkpoint": str(parent_checkpoint),
        "completed_updates": int(parent_completed_updates),
        "fork_generation": int(parent.get("fork_generation", 0)),
    }
    lifecycle = _base_lifecycle(mode="fork", config=config, paths=paths, world_size=world_size, instrumentation_backend=instrumentation_backend, execution_options=execution_options)
    lifecycle["root_start_label"] = parent.get("root_start_label") or parent.get("run_start_label")
    lifecycle["fork_generation"] = int(parent.get("fork_generation", 0)) + 1
    lifecycle["parent_logical_run_id"] = parent.get("logical_run_id")
    lifecycle["parent_artifact_name"] = parent.get("artifact_name")
    lifecycle["parent_checkpoint"] = str(parent_checkpoint)
    lifecycle["parent_completed_updates"] = int(parent_completed_updates)
    lifecycle["lineage"] = parent_lineage + [parent_record]
    lifecycle["lr_phases"] = list(parent.get("lr_phases", [])) + [dict(child_lr_phase)]
    lifecycle["active_lr_phase_index"] = len(lifecycle["lr_phases"]) - 1
    return validate_lifecycle(_append_session(lifecycle))


def lifecycle_from_checkpoint(payload: Mapping[str, Any]) -> Dict[str, Any]:
    lifecycle = payload.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        raise ValueError("checkpoint does not contain enhanced lifecycle metadata")
    return validate_lifecycle(dict(lifecycle))


def update_wandb_identity(lifecycle: Mapping[str, Any], wandb_run_id: str | None) -> Dict[str, Any]:
    updated = dict(lifecycle)
    if wandb_run_id:
        updated["wandb_run_id"] = wandb_run_id
    return validate_lifecycle(updated)


def current_session_result_path(lifecycle: Mapping[str, Any]) -> Path:
    result_path = Path(str(lifecycle["result_path"]))
    session_id = str(lifecycle["session_id"])
    return result_path.with_name(f"result_{session_id}.json")


def validate_lifecycle(lifecycle: Mapping[str, Any]) -> Dict[str, Any]:
    required = ("lifecycle_schema_version", "logical_run_id", "session_id", "artifact_name", "run_config", "world_size", "lr_phases")
    missing = [name for name in required if name not in lifecycle]
    if missing:
        raise ValueError(f"lifecycle metadata missing fields: {missing}")
    if lifecycle["lifecycle_schema_version"] != LIFECYCLE_SCHEMA_VERSION:
        raise ValueError("unsupported lifecycle schema version")
    return dict(lifecycle)


__all__ = [
    "LIFECYCLE_SCHEMA_VERSION",
    "current_session_result_path",
    "fork_lifecycle",
    "fork_suffix",
    "fresh_lifecycle",
    "lifecycle_from_checkpoint",
    "resume_lifecycle",
    "update_wandb_identity",
    "validate_lifecycle",
    "validate_start_label",
]
# ^^^ THOG
