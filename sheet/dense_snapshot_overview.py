# vvv THOG
"""Tensor-free Overview provenance, separate from the immutable snapshot schema."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def source_hyperparameters(config: Any) -> dict[str, Any]:
    values = asdict(config) if is_dataclass(config) else dict(vars(config))
    prefix = "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            suffix = key[len(prefix):].lower()
            suffix = {"scalar_weights_per_matrix": "coupling_pairs_per_matrix",
                      "same_coordinates_all_runs": "same_coupling_pairs_all_runs"}.get(suffix, suffix)
            values["instrumentation__depth_weight_curves__" + suffix] = value
    return json.loads(json.dumps(values, default=str))


def write_snapshot_overview(path: Path, payload: Mapping[str, Any], config: Any) -> None:
    """Best-effort diagnostics must never alter snapshot or training semantics."""
    sidecar = Path(str(path) + ".overview.json")
    temporary = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    try:
        details = {
            "compatibility_hash": payload["compatibility_hash"],
            "tensor_payload_hash": payload["tensor_payload_hash"],
            "source_hyperparameters": source_hyperparameters(config),
        }
        temporary.write_text(json.dumps(details, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(sidecar)
    except (OSError, TypeError, ValueError):
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def snapshot_overview_metadata(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    details = {"snapshot_hyperparameters": dict(payload["compatibility_payload"])}
    try:
        sidecar = json.loads(Path(str(path) + ".overview.json").read_text(encoding="utf-8"))
        if not isinstance(sidecar, dict):
            return details
        if any(sidecar.get(key) != payload[key] for key in ("compatibility_hash", "tensor_payload_hash")):
            return details
        parameters = sidecar.get("source_hyperparameters")
        if isinstance(parameters, dict):
            details["source_hyperparameters"] = parameters
    except (OSError, ValueError):
        pass
    return details
# ^^^ THOG
