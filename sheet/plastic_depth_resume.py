# vvv THOG
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, TextIO, Union

from .plastic_depth_pause import (
    PlasticCoarsePauseResult,
    run_distributed_plastic_coarse_review_pause,
    run_plastic_coarse_review_pause,
)


PLASTIC_RESUME_NOT_APPLICABLE = "not_applicable"
PLASTIC_RESUME_CONTINUE_FINE = "continue_fine"
PLASTIC_RESUME_CHECKPOINT_EXIT = "checkpoint_exit"


def _coarse_fine_state(trainer: Any) -> Optional[dict[str, object]]:
    value = getattr(trainer, "plastic_coarse_fine_state", None)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("plastic_coarse_fine_state must be a mapping")
    return copy.deepcopy(dict(value))


def resume_plastic_coarse_fine_boundary(
    trainer: Any,
    checkpoint_path: Union[str, Path],
    *,
    output: TextIO = sys.stdout,
    pause_runner=run_plastic_coarse_review_pause,
) -> str:
    """Resume an exact review-pause boundary or leave ordinary FINE resumes untouched."""

    state = _coarse_fine_state(trainer)
    if state is None or state.get("phase") != "review_pause":
        return PLASTIC_RESUME_NOT_APPLICABLE
    pause = state.get("pause")
    if not isinstance(pause, Mapping):
        raise ValueError("review-pause checkpoint lacks pause state")
    remaining_seconds = float(pause.get("remaining_seconds", 0.0))
    if remaining_seconds < 0.0:
        raise ValueError("review-pause remaining_seconds must be non-negative")

    result = run_distributed_plastic_coarse_review_pause(
        trainer.distributed,
        duration_seconds=remaining_seconds,
        output=output,
        pause_runner=pause_runner,
    )
    updated_pause = {
        "disposition": result.disposition,
        "elapsed_seconds": float(result.elapsed_seconds),
        "remaining_seconds": float(result.remaining_seconds),
    }
    state["pause"] = updated_pause

    if result.disposition == "checkpoint_exit":
        state["phase"] = "review_pause"
        trainer.plastic_coarse_fine_state = state
        trainer.plastic_coarse_provenance = state
        trainer.save_checkpoint(checkpoint_path)
        return PLASTIC_RESUME_CHECKPOINT_EXIT

    state["phase"] = "fine"
    trainer.plastic_coarse_fine_state = state
    trainer.plastic_coarse_provenance = state
    return PLASTIC_RESUME_CONTINUE_FINE


__all__ = [
    "PLASTIC_RESUME_CHECKPOINT_EXIT",
    "PLASTIC_RESUME_CONTINUE_FINE",
    "PLASTIC_RESUME_NOT_APPLICABLE",
    "resume_plastic_coarse_fine_boundary",
]
# ^^^ THOG
