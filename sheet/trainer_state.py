# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TrainerState:
    completed_updates: int = 0
    best_validation_loss: float = float("inf")
    latest_validation_loss: Optional[float] = None
    latest_training_loss: Optional[float] = None
    # vvv THOG
    skipped_nonfinite_updates: int = 0
    failed_update_attempts: int = 0
    # ^^^ THOG


@dataclass(frozen=True)
class TrainerEvent:
    name: str
    completed_updates: int
    payload: Dict[str, Any]
# ^^^ THOG
