# vvv THOG
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


__all__ = ["write_run_manifest"]
# ^^^ THOG
