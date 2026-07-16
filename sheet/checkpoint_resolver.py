# vvv THOG
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_TIMESTAMP_SELECTOR = re.compile(r"^\d{6}-\d{4}$")


@dataclass(frozen=True)
class ResolvedCheckpoint:
    selector: str
    checkpoint_dir: Path
    checkpoint_path: Path
    artifact_name: str


def _checkpoint_from_directory(selector: str, directory: Path) -> ResolvedCheckpoint:
    checkpoint_path = directory / "ckpt.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint directory does not contain ckpt.pt: {directory}")
    return ResolvedCheckpoint(selector=selector, checkpoint_dir=directory, checkpoint_path=checkpoint_path, artifact_name=directory.name)


def _matches_by_leading_timestamp(checkpoint_root: Path, selector: str) -> list[Path]:
    if not checkpoint_root.is_dir():
        return []
    prefix = selector + "_"
    return sorted(path for path in checkpoint_root.iterdir() if path.is_dir() and path.name.startswith(prefix))


def resolve_checkpoint(selector: str, checkpoint_root: str | Path = "checkpoints") -> ResolvedCheckpoint:
    if not selector:
        raise ValueError("--resume-from selector must be non-empty")
    root = Path(checkpoint_root)
    selector_path = Path(selector)
    if selector_path.suffix == ".pt" or "/" in selector or "\\" in selector:
        path = selector_path
        if path.is_dir():
            return _checkpoint_from_directory(selector, path)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint file not found: {path}")
        return ResolvedCheckpoint(selector=selector, checkpoint_dir=path.parent, checkpoint_path=path, artifact_name=path.parent.name)
    if _TIMESTAMP_SELECTOR.fullmatch(selector):
        matches = _matches_by_leading_timestamp(root, selector)
        if not matches:
            raise FileNotFoundError(f"no checkpoint artifact starts with timestamp {selector!r} under {root}")
        if len(matches) > 1:
            names = ", ".join(path.name for path in matches)
            raise ValueError(f"ambiguous checkpoint timestamp {selector!r}: {names}")
        return _checkpoint_from_directory(selector, matches[0])
    directory = root / selector
    if not directory.is_dir():
        raise FileNotFoundError(f"checkpoint artifact not found: {directory}")
    return _checkpoint_from_directory(selector, directory)


__all__ = ["ResolvedCheckpoint", "resolve_checkpoint"]
# ^^^ THOG
