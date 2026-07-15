# vvv THOG
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union

from .checkpoints import load_payload
from .run_lifecycle import lifecycle_from_checkpoint


_TIMESTAMP_SELECTOR = re.compile(r"^\d{6}-\d{4}$")


@dataclass(frozen=True)
class ResolvedCheckpoint:
    selector: str
    selector_kind: str
    checkpoint_path: Path
    checkpoint_dir: Path
    artifact_name: str


def _candidate_diagnostic(candidates: Iterable[Path]) -> str:
    values = sorted(str(path) for path in candidates)
    return "\n".join(f"  {value}" for value in values)


def _validate_checkpoint(path: Path, selector: str, selector_kind: str) -> ResolvedCheckpoint:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint is missing: {path}")
    payload = load_payload(path)
    lifecycle = lifecycle_from_checkpoint(payload)
    artifact_name = str(lifecycle["artifact_name"])
    if path.parent.name != artifact_name:
        raise ValueError(
            "checkpoint directory does not match lifecycle artifact_name: "
            f"directory={path.parent.name!r}, lifecycle={artifact_name!r}"
        )
    return ResolvedCheckpoint(
        selector=selector,
        selector_kind=selector_kind,
        checkpoint_path=path,
        checkpoint_dir=path.parent,
        artifact_name=artifact_name,
    )


def resolve_checkpoint(selector: str, checkpoint_root: Union[str, Path]) -> ResolvedCheckpoint:
    value = selector.strip()
    if not value:
        raise ValueError("--resume-from must not be empty")
    root = Path(checkpoint_root)
    if "/" in value or "\\" in value or value.endswith(".pt"):
        path = Path(value).expanduser()
        if path.is_dir():
            path = path / "ckpt.pt"
        return _validate_checkpoint(path, value, "path")

    if _TIMESTAMP_SELECTOR.fullmatch(value):
        candidates: List[Path] = sorted(
            path / "ckpt.pt"
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith(value + "_") and (path / "ckpt.pt").is_file()
        ) if root.is_dir() else []
        if not candidates:
            raise FileNotFoundError(
                f"no checkpoint matches leading start timestamp {value!r} under {root}"
            )
        if len(candidates) != 1:
            raise ValueError(
                f"multiple checkpoints match leading start timestamp {value!r}:\n"
                + _candidate_diagnostic(candidates)
            )
        return _validate_checkpoint(candidates[0], value, "start_timestamp")

    path = root / value / "ckpt.pt"
    return _validate_checkpoint(path, value, "artifact_name")


__all__ = ["ResolvedCheckpoint", "resolve_checkpoint"]
# ^^^ THOG
