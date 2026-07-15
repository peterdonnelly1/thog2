# vvv THOG
from __future__ import annotations

import math
from typing import Any, Mapping


COSINE_SCHEDULE = "cosine"
RESTART_COSINE_SCHEDULE = "restart_cosine"
LR_SCHEDULE_KINDS = (COSINE_SCHEDULE, RESTART_COSINE_SCHEDULE)


def cosine_learning_rate(
    *,
    completed_updates: int,
    learning_rate: float,
    min_learning_rate: float,
    warmup_updates: int,
    decay_updates: int,
    decay_learning_rate: bool,
) -> float:
    if not decay_learning_rate:
        return learning_rate
    if completed_updates < warmup_updates:
        return learning_rate * (completed_updates + 1) / (warmup_updates + 1)
    if completed_updates > decay_updates:
        return min_learning_rate
    if decay_updates == warmup_updates:
        return min_learning_rate
    ratio = (completed_updates - warmup_updates) / (decay_updates - warmup_updates)
    ratio = min(1.0, max(0.0, ratio))
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_learning_rate + coefficient * (learning_rate - min_learning_rate)


def validate_restart_cosine_phase(phase: Mapping[str, Any]) -> None:
    required = (
        "phase_start_update",
        "phase_start_lr",
        "phase_peak_lr",
        "phase_rewarm_iters",
        "phase_end_update",
        "phase_min_lr",
    )
    missing = [name for name in required if phase.get(name) is None]
    if missing:
        raise ValueError(f"restart_cosine phase is missing {missing}")
    start_update = int(phase["phase_start_update"])
    end_update = int(phase["phase_end_update"])
    rewarm_iters = int(phase["phase_rewarm_iters"])
    start_lr = float(phase["phase_start_lr"])
    peak_lr = float(phase["phase_peak_lr"])
    min_lr = float(phase["phase_min_lr"])
    total_updates = end_update - start_update
    if start_update < 0:
        raise ValueError("phase_start_update must be non-negative")
    if total_updates < 2:
        raise ValueError("restart_cosine requires at least two child updates")
    if rewarm_iters < 0 or rewarm_iters > total_updates - 1:
        raise ValueError("phase_rewarm_iters must be in [0, child_updates - 1]")
    if start_lr < 0.0:
        raise ValueError("phase_start_lr must be non-negative")
    if peak_lr <= 0.0 or peak_lr < start_lr:
        raise ValueError("phase_peak_lr must be positive and at least phase_start_lr")
    if min_lr < 0.0 or min_lr > peak_lr:
        raise ValueError("phase_min_lr must be in [0, phase_peak_lr]")


def restart_cosine_learning_rate(*, completed_updates: int, phase: Mapping[str, Any]) -> float:
    validate_restart_cosine_phase(phase)
    start_update = int(phase["phase_start_update"])
    end_update = int(phase["phase_end_update"])
    start_lr = float(phase["phase_start_lr"])
    peak_lr = float(phase["phase_peak_lr"])
    rewarm_iters = int(phase["phase_rewarm_iters"])
    min_lr = float(phase["phase_min_lr"])
    if completed_updates < start_update:
        raise ValueError("completed_updates precedes restart_cosine phase_start_update")
    if completed_updates >= end_update:
        return min_lr

    relative_update = completed_updates - start_update
    child_updates = end_update - start_update
    if rewarm_iters == 0:
        ratio = relative_update / (child_updates - 1)
        coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return min_lr + coefficient * (peak_lr - min_lr)

    if relative_update < rewarm_iters:
        if rewarm_iters == 1:
            return peak_lr
        ratio = relative_update / (rewarm_iters - 1)
        return start_lr + ratio * (peak_lr - start_lr)

    decay_updates = child_updates - rewarm_iters
    decay_position = relative_update - rewarm_iters + 1
    ratio = decay_position / decay_updates
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coefficient * (peak_lr - min_lr)


def learning_rate_for_config(config: Any, completed_updates: int) -> float:
    kind = getattr(config, "lr_schedule_kind", COSINE_SCHEDULE)
    if kind == COSINE_SCHEDULE:
        return cosine_learning_rate(
            completed_updates=completed_updates,
            learning_rate=float(config.learning_rate),
            min_learning_rate=float(config.min_learning_rate),
            warmup_updates=int(config.warmup_updates),
            decay_updates=int(config.decay_updates),
            decay_learning_rate=bool(config.decay_learning_rate),
        )
    if kind == RESTART_COSINE_SCHEDULE:
        phase = {
            "phase_start_update": config.phase_start_update,
            "phase_start_lr": config.phase_start_lr,
            "phase_peak_lr": config.phase_peak_lr,
            "phase_rewarm_iters": config.phase_rewarm_iters,
            "phase_end_update": config.phase_end_update,
            "phase_min_lr": config.phase_min_lr,
        }
        return restart_cosine_learning_rate(completed_updates=completed_updates, phase=phase)
    raise ValueError(f"unsupported lr_schedule_kind: {kind!r}")


__all__ = [
    "COSINE_SCHEDULE",
    "LR_SCHEDULE_KINDS",
    "RESTART_COSINE_SCHEDULE",
    "cosine_learning_rate",
    "learning_rate_for_config",
    "restart_cosine_learning_rate",
    "validate_restart_cosine_phase",
]
# ^^^ THOG
