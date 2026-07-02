# vvv THOG
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


TELEMETRY_FINISH_TIMEOUT_SECONDS = 120.0


def _git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def source_identity(repository_root: Path) -> Dict[str, Any]:
    root = repository_root.resolve()
    tracked_status = _git_output(
        root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_status:
        raise RuntimeError(
            "Stage 6 requires a clean tracked worktree before protocol lock; "
            f"tracked changes: {tracked_status!r}"
        )
    return {
        "commit": _git_output(root, "rev-parse", "HEAD"),
        "branch": _git_output(root, "branch", "--show-current"),
        "origin_url": _git_output(root, "config", "--get", "remote.origin.url"),
        "tracked_worktree_clean": True,
    }


def verify_source_identity(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    if not bool(actual.get("tracked_worktree_clean")):
        raise ValueError("Stage 6 source worktree is not clean")
    expected_commit = str(expected.get("commit", ""))
    actual_commit = str(actual.get("commit", ""))
    if not expected_commit:
        raise ValueError("Stage 6 protocol source commit is missing")
    if expected_commit != actual_commit:
        raise ValueError(
            "Stage 6 source commit differs from the locked protocol; "
            f"expected={expected_commit}, actual={actual_commit}"
        )
    expected_origin = str(expected.get("origin_url", ""))
    actual_origin = str(actual.get("origin_url", ""))
    if expected_origin and expected_origin != actual_origin:
        raise ValueError(
            "Stage 6 source repository differs from the locked protocol; "
            f"expected={expected_origin}, actual={actual_origin}"
        )


def verify_manifest_source(
    manifest: Mapping[str, Any],
    repository_root: Path,
) -> Dict[str, Any]:
    expected = manifest.get("source")
    if not isinstance(expected, Mapping):
        raise ValueError("Stage 6 protocol source identity is missing")
    actual = source_identity(repository_root)
    verify_source_identity(expected, actual)
    return actual


def init_resilient_telemetry(
    module: Any,
    *,
    project: str,
    entity: Optional[str],
    name: str,
    group: str,
    job_type: str,
    config: Dict[str, Any],
) -> Any:
    settings = module.Settings(
        finish_timeout=TELEMETRY_FINISH_TIMEOUT_SECONDS,
        finish_timeout_raises=False,
    )
    arguments: Dict[str, Any] = {
        "project": project,
        "name": name,
        "group": group,
        "job_type": job_type,
        "config": config,
        "settings": settings,
    }
    if entity:
        arguments["entity"] = entity
    if os.environ.get("WANDB_MODE", "").strip().lower() == "offline":
        return module.init(**arguments, mode="offline")
    try:
        return module.init(**arguments, mode="online")
    except Exception as error:
        communication_error = getattr(
            getattr(module, "errors", None),
            "CommError",
            None,
        )
        if communication_error is None or not isinstance(error, communication_error):
            raise
        print(
            "THOG2 telemetry online initialisation failed; "
            f"continuing with offline logging: {error}",
            flush=True,
        )
        teardown = getattr(module, "teardown", None)
        if callable(teardown):
            try:
                teardown(exit_code=1)
            except Exception as cleanup_error:
                print(
                    "THOG2 WARNING: telemetry cleanup failed; "
                    f"attempting offline logging anyway: {cleanup_error}",
                    flush=True,
                )
        os.environ["WANDB_MODE"] = "offline"
        return module.init(**arguments, mode="offline")


__all__ = [
    "TELEMETRY_FINISH_TIMEOUT_SECONDS",
    "init_resilient_telemetry",
    "source_identity",
    "verify_manifest_source",
    "verify_source_identity",
]
# ^^^ THOG
