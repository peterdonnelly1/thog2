# vvv THOG
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainerState:
    completed_updates: int = 0
    best_validation_loss: float = float("inf")
    latest_validation_loss: Optional[float] = None
    latest_training_loss: Optional[float] = None
    # vvv THOG persistent failed-attempt accounting for bounded non-finite recovery
    skipped_nonfinite_updates: int = 0
    failed_update_attempts: int = 0
    # ^^^ THOG
    # vvv THOG PLASTIC DEPTH robust paired-score evidence, count-change spacing and replayable decisions survive checkpoints
    plastic_depth_probe_histories: Dict[str, List[float]] = field(default_factory=dict)
    plastic_depth_last_count_change_update: int = -1
    plastic_depth_count_audit: List[Dict[str, Any]] = field(default_factory=list)
    # ^^^ THOG


@dataclass(frozen=True)
class TrainerEvent:
    name: str
    completed_updates: int
    payload: Dict[str, Any]
# ^^^ THOG

# vvv THOG retired PLASTIC DEPTH hold-controller source preserved for history audit
# from dataclasses import dataclass
# from typing import Any, Dict, Optional
# ^^^ THOG
