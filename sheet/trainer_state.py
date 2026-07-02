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


@dataclass(frozen=True)
class TrainerEvent:
    name: str
    completed_updates: int
    payload: Dict[str, Any]
# ^^^ THOG
