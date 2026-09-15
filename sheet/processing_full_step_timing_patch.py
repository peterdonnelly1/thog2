# vvv THOG full-step timing public CLI, lightweight PREMAT outcomes, throughput retention, and dashboard overlay
from __future__ import annotations

import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from . import processing_update_timing_patch as _timing
from . import wandb_telemetry as _wandb
from .local_chart_store import ensure_local_chart_store
from .stage6_source import progress_tokens_per_second


_ENABLED_ENVIRONMENT_KEY = "THOG2_INTERNAL_PREMAT_INSTRA_FULL_STEP_TIMING_CAPTURE_AND_CHART"
_CAPTURE_ENVIRONMENT_KEY = "THOG2_INTERNAL_PREMAT_INSTRA_FULL_STEP_TIMING_CAPTURE_AND_CHART_CAPTURE"
_PUBLIC_ENABLED_KEY = "premat_instra__full_step_timing_capture_and_chart"
_PUBLIC_CAPTURE_KEY = "premat_instra__full_step_timing_capture_and_chart_capture"
_OUTCOME_ATTRIBUTE = "_thog_full_step_premat_microstep_outcomes"


def _full_step_timing_enabled() -> bool:
    return os.environ.get(_ENABLED_ENVIRONMENT_KEY, "disable").strip().lower() == "enable"


def _requested_capture_update(trainer: Any) -> int:
    raw = os.environ.get(_CAPTURE_ENVIRONMENT_KEY, "").strip()
    if raw:
        update = int(raw)
    else:
        update = int(trainer.config.max_updates)
    if update < 1:
        raise ValueError("full-step timing capture update must be a positive integer")
    if update > int(trainer.config.max_updates):
        raise ValueError(
            f"full-step timing capture update {update} exceeds max_updates={trainer.config.max_updates}"
        )
    return update


def _install_processing_throughput_retention(telemetry: Any) -> None:
    if str(telemetry.config.get("premat_processing_logging", "disabled")) == "enabled":
        return
    original_log_event = telemetry.log_event

    @wraps(original_log_event)
    def log_event_with_processing_throughput(event: str, payload: Mapping[str, Any]) -> None:
        if event == "optimizer_progress":
            tokens_per_second = progress_tokens_per_second(payload)
            if tokens_per_second is not None:
                ensure_local_chart_store(telemetry).append_processing_throughput(
                    int(payload.get("completed_updates", 0)),
                    tokens_per_second,
                )
        original_log_event(event, payload)

    telemetry.log_event = log_event_with_processing_throughput


def _premat_outcome_counts(snapshot: Mapping[str, Any]) -> Dict[str, int]:
    counts = {"full_hits": 0, "partial_hits": 0, "misses": 0}
    for candidate in snapshot.get("candidates", ()):
        if not isinstance(candidate, Mapping):
            continue
        outcome = str(candidate.get("final_outcome", "")).strip().upper()
        if outcome == "FULL HIT":
            counts["full_hits"] += 1
        elif outcome == "PARTIAL HIT":
            counts["partial_hits"] += 1
        elif outcome == "COMPLETE MISS":
            counts["misses"] += 1
    return counts


def _install_lightweight_premat_outcomes(trainer: Any, capture_update: int) -> None:
    if str(trainer.config.premat).strip().lower() != "enabled":
        return
    if str(trainer.config.premat_instra).strip().lower() == "enabled":
        return
    reporter_setter = getattr(
        getattr(trainer, "raw_model", None),
        "set_premat_live_reporter",
        None,
    )
    if not callable(reporter_setter) or not trainer.distributed.is_primary:
        return

    accumulation_steps = int(trainer.config.gradient_accumulation_steps)
    pass_to_microstep: Dict[int, int] = {}
    outcomes: Dict[int, Dict[str, int]] = {}
    setattr(trainer, _OUTCOME_ATTRIBUTE, outcomes)

    def capture_selected_update(pass_sequence: int) -> bool:
        prospective_update = int(trainer.state.completed_updates) + 1
        if prospective_update != int(capture_update):
            return False
        sequence = int(pass_sequence)
        if sequence not in pass_to_microstep:
            next_microstep = len(pass_to_microstep) + 1
            if next_microstep > accumulation_steps:
                return False
            pass_to_microstep[sequence] = next_microstep
        return True

    def publish_selected_outcomes(snapshot: Mapping[str, Any]) -> None:
        pass_sequence = int(snapshot.get("pass_sequence", 0))
        micro_step = pass_to_microstep.get(pass_sequence)
        if micro_step is None:
            return
        outcomes[micro_step] = _premat_outcome_counts(snapshot)

    reporter_setter(publish_selected_outcomes, capture_selected_update)


def _inject_premat_outcomes(trainer: Any, payload: Dict[str, Any]) -> None:
    if str(trainer.config.premat).strip().lower() != "enabled":
        return
    if str(trainer.config.premat_instra).strip().lower() == "enabled":
        return
    if not hasattr(trainer, _OUTCOME_ATTRIBUTE):
        return
    accumulation_steps = int(trainer.config.gradient_accumulation_steps)
    captured = getattr(trainer, _OUTCOME_ATTRIBUTE)
    payload["premat_microstep_outcomes"] = [
        {
            "micro_step": micro_step,
            **dict(captured.get(micro_step, {"full_hits": 0, "partial_hits": 0, "misses": 0})),
        }
        for micro_step in range(1, accumulation_steps + 1)
    ]
    payload["premat_microstep_outcomes_capture"] = "lightweight_completed_pass"


_ORIGINAL_PUBLISH_PAYLOAD = _timing._publish_payload


def _publish_payload_with_premat_outcomes(
    trainer: Any,
    store: Any,
    payload: Dict[str, Any],
    *,
    official_elapsed_seconds: float,
) -> None:
    _inject_premat_outcomes(trainer, payload)
    _ORIGINAL_PUBLISH_PAYLOAD(
        trainer,
        store,
        payload,
        official_elapsed_seconds=official_elapsed_seconds,
    )


_timing._publish_payload = _publish_payload_with_premat_outcomes


# The established timing patch remains the implementation substrate. This overlay
# supplies its requested update from the public CLI, so the retired shell variable
# cannot activate timing by itself and DENSE/THOG share exactly the same attachment.
def _attach_telemetry_with_public_full_step_timing(trainer: Any, telemetry: Any) -> None:
    if not _full_step_timing_enabled():
        _timing._ORIGINAL_ATTACH_TELEMETRY(trainer, telemetry)
        return

    capture_update = _requested_capture_update(trainer)
    telemetry.config[_PUBLIC_ENABLED_KEY] = "enable"
    telemetry.config[_PUBLIC_CAPTURE_KEY] = f"step {capture_update}"
    _install_processing_throughput_retention(telemetry)

    original_requested_update = _timing._requested_update
    _timing._requested_update = lambda: capture_update
    try:
        _timing._attach_telemetry_with_processing_update_timing(trainer, telemetry)
    finally:
        _timing._requested_update = original_requested_update

    _install_lightweight_premat_outcomes(trainer, capture_update)


_wandb.attach_telemetry = _attach_telemetry_with_public_full_step_timing


# vvv THOG dashboard keeps its established HTML/script surface; only this exact asset response receives the small full-step viewer overlay
_ORIGINAL_PATH_READ_BYTES = Path.read_bytes


def _dashboard_asset_read_bytes(path: Path) -> bytes:
    payload = _ORIGINAL_PATH_READ_BYTES(path)
    if (
        Path(sys.argv[0]).name == "run_thog2_local_dashboard.py"
        and path.name == "dashboard_processing.js"
        and path.parent.name == "local_dashboard_assets"
    ):
        patch = path.with_name("dashboard_processing_full_step_patch.js")
        if patch.is_file():
            payload += b"\n" + _ORIGINAL_PATH_READ_BYTES(patch)
    return payload


if Path(sys.argv[0]).name == "run_thog2_local_dashboard.py":
    Path.read_bytes = _dashboard_asset_read_bytes
# ^^^ THOG


__all__ = [
    "_attach_telemetry_with_public_full_step_timing",
    "_full_step_timing_enabled",
    "_premat_outcome_counts",
    "_requested_capture_update",
]
# ^^^ THOG
