# vvv THOG
from __future__ import annotations

import copy
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


LIFECYCLE_SCHEMA_VERSION = 1
START_LABEL_PATTERN = re.compile(r"^\d{6}-\d{4}$")
FORK_SUFFIX_PATTERN = re.compile(r"^FORK_(?P<generation>[1-9][0-9]*)_FROM_(?P<root>\d{6}-\d{4})$")


def new_identifier() -> str:
    return uuid.uuid4().hex


def validate_start_label(value: str) -> str:
    if not START_LABEL_PATTERN.fullmatch(value):
        raise ValueError("run_start_label must use YYMMDD-HHMM")
    return value


def fork_suffix(generation: int, root_start_label: str) -> str:
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise ValueError("fork generation must be a positive integer")
    return f"FORK_{generation}_FROM_{validate_start_label(root_start_label)}"


def _session_record(
    *,
    session_id: str,
    mode: str,
    starting_completed_updates: int,
    target_steps: int,
    result_path: str,
) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "mode": mode,
        "starting_completed_updates": int(starting_completed_updates),
        "target_steps": int(target_steps),
        "result_path": result_path,
    }


def fresh_lifecycle(
    *,
    config: Any,
    paths: Mapping[str, Path],
    world_size: int,
    instrumentation_backend: str,
    execution_options: Mapping[str, Any],
    lr_phase: Mapping[str, Any],
) -> Dict[str, Any]:
    if config.run_start_label is None:
        raise ValueError("fresh lifecycle requires run_start_label")
    run_start_label = validate_start_label(config.run_start_label)
    logical_run_id = new_identifier()
    session_id = new_identifier()
    run_config = asdict(config)
    session_result_path = str(Path(paths["result_dir"]) / f"result_{session_id}.json")
    return {
        "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
        "logical_run_id": logical_run_id,
        "session_id": session_id,
        "session_mode": "fresh",
        "creation_mode": "fresh",
        "run_start_label": run_start_label,
        "artifact_name": config.artifact_name,
        "artifact_descriptor": config.artifact_descriptor,
        "root_logical_run_id": logical_run_id,
        "root_start_label": run_start_label,
        "fork_generation": 0,
        "parent_logical_run_id": None,
        "parent_artifact_name": None,
        "parent_checkpoint": None,
        "parent_completed_updates": None,
        "lineage": [],
        "run_config": run_config,
        "origin_run_config": copy.deepcopy(run_config),
        "world_size": int(world_size),
        "execution_options": dict(execution_options),
        "lr_phases": [dict(lr_phase)],
        "active_lr_phase_index": 0,
        "instrumentation_backend": instrumentation_backend,
        "wandb_run_id": None,
        "tensorboard_dir": str(paths["tensorboard_dir"]),
        "log_path": str(paths["log_path"]),
        "result_path": str(paths["result_path"]),
        "session_results": [session_result_path],
        "sessions": [
            _session_record(
                session_id=session_id,
                mode="fresh",
                starting_completed_updates=0,
                target_steps=config.max_iters,
                result_path=session_result_path,
            )
        ],
    }


def resume_lifecycle(
    parent: Mapping[str, Any],
    *,
    config: Any,
    starting_completed_updates: int,
    instrumentation_backend: str,
    execution_options: Mapping[str, Any],
) -> Dict[str, Any]:
    result = validate_lifecycle(parent)
    session_id = new_identifier()
    session_result_path = str(Path(result["result_path"]).parent / f"result_{session_id}.json")
    result["session_id"] = session_id
    result["session_mode"] = "resume"
    result["run_config"] = asdict(config)
    result["instrumentation_backend"] = instrumentation_backend
    result["execution_options"] = dict(execution_options)
    result.setdefault("session_results", []).append(session_result_path)
    result.setdefault("sessions", []).append(
        _session_record(
            session_id=session_id,
            mode="resume",
            starting_completed_updates=starting_completed_updates,
            target_steps=config.max_iters,
            result_path=session_result_path,
        )
    )
    return result


def fork_lifecycle(
    parent: Mapping[str, Any],
    *,
    config: Any,
    paths: Mapping[str, Path],
    parent_checkpoint: Path,
    parent_completed_updates: int,
    world_size: int,
    instrumentation_backend: str,
    execution_options: Mapping[str, Any],
    child_lr_phase: Mapping[str, Any],
) -> Dict[str, Any]:
    parent_lifecycle = validate_lifecycle(parent)
    generation = int(parent_lifecycle["fork_generation"]) + 1
    expected_suffix = fork_suffix(generation, str(parent_lifecycle["root_start_label"]))
    if config.artifact_suffix != expected_suffix:
        raise ValueError("fork artifact suffix does not match lifecycle generation")
    session_id = new_identifier()
    logical_run_id = new_identifier()
    run_config = asdict(config)
    lineage = list(parent_lifecycle.get("lineage", []))
    lineage.append(
        {
            "logical_run_id": parent_lifecycle["logical_run_id"],
            "artifact_name": parent_lifecycle["artifact_name"],
            "checkpoint": str(parent_checkpoint),
            "completed_updates": int(parent_completed_updates),
        }
    )
    phases = [dict(value) for value in parent_lifecycle.get("lr_phases", [])]
    phases.append(dict(child_lr_phase))
    session_result_path = str(Path(paths["result_dir"]) / f"result_{session_id}.json")
    return {
        "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
        "logical_run_id": logical_run_id,
        "session_id": session_id,
        "session_mode": "fork",
        "creation_mode": "fork",
        "run_start_label": config.run_start_label,
        "artifact_name": config.artifact_name,
        "artifact_descriptor": config.artifact_descriptor,
        "root_logical_run_id": parent_lifecycle["root_logical_run_id"],
        "root_start_label": parent_lifecycle["root_start_label"],
        "fork_generation": generation,
        "parent_logical_run_id": parent_lifecycle["logical_run_id"],
        "parent_artifact_name": parent_lifecycle["artifact_name"],
        "parent_checkpoint": str(parent_checkpoint),
        "parent_completed_updates": int(parent_completed_updates),
        "lineage": lineage,
        "run_config": run_config,
        "origin_run_config": copy.deepcopy(parent_lifecycle["origin_run_config"]),
        "world_size": int(world_size),
        "execution_options": dict(execution_options),
        "lr_phases": phases,
        "active_lr_phase_index": len(phases) - 1,
        "instrumentation_backend": instrumentation_backend,
        "wandb_run_id": None,
        "tensorboard_dir": str(paths["tensorboard_dir"]),
        "log_path": str(paths["log_path"]),
        "result_path": str(paths["result_path"]),
        "session_results": [session_result_path],
        "sessions": [
            _session_record(
                session_id=session_id,
                mode="fork",
                starting_completed_updates=parent_completed_updates,
                target_steps=config.max_iters,
                result_path=session_result_path,
            )
        ],
    }


def validate_lifecycle(value: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if result.get("lifecycle_schema_version") != LIFECYCLE_SCHEMA_VERSION:
        raise ValueError(
            "checkpoint does not contain the enhanced lifecycle schema: "
            f"expected {LIFECYCLE_SCHEMA_VERSION}, got {result.get('lifecycle_schema_version')!r}"
        )
    required = (
        "logical_run_id",
        "session_id",
        "run_start_label",
        "artifact_name",
        "artifact_descriptor",
        "root_logical_run_id",
        "root_start_label",
        "fork_generation",
        "run_config",
        "origin_run_config",
        "world_size",
        "execution_options",
        "lr_phases",
        "active_lr_phase_index",
        "instrumentation_backend",
        "tensorboard_dir",
        "log_path",
        "result_path",
        "session_results",
        "sessions",
    )
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(f"enhanced lifecycle metadata is missing {missing}")
    validate_start_label(str(result["run_start_label"]))
    validate_start_label(str(result["root_start_label"]))
    generation = int(result["fork_generation"])
    if generation < 0:
        raise ValueError("fork_generation must be non-negative")
    phases = result["lr_phases"]
    active_index = int(result["active_lr_phase_index"])
    if not isinstance(phases, list) or not phases:
        raise ValueError("lr_phases must be a non-empty list")
    if active_index < 0 or active_index >= len(phases):
        raise ValueError("active_lr_phase_index is out of range")
    return result


def lifecycle_from_checkpoint(payload: Mapping[str, Any]) -> Dict[str, Any]:
    lifecycle = payload.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        raise ValueError("checkpoint does not contain enhanced lifecycle metadata")
    return validate_lifecycle(lifecycle)


def update_wandb_identity(lifecycle: Mapping[str, Any], run_id: Optional[str]) -> Dict[str, Any]:
    result = validate_lifecycle(lifecycle)
    result["wandb_run_id"] = run_id
    return result


def current_session_result_path(lifecycle: Mapping[str, Any]) -> Path:
    result = validate_lifecycle(lifecycle)
    return Path(result["session_results"][-1])


__all__ = [
    "FORK_SUFFIX_PATTERN",
    "LIFECYCLE_SCHEMA_VERSION",
    "START_LABEL_PATTERN",
    "current_session_result_path",
    "fork_lifecycle",
    "fork_suffix",
    "fresh_lifecycle",
    "lifecycle_from_checkpoint",
    "new_identifier",
    "resume_lifecycle",
    "update_wandb_identity",
    "validate_lifecycle",
    "validate_start_label",
]
# ^^^ THOG
