# vvv THOG
from __future__ import annotations

import math
from typing import Any


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
        return float(learning_rate)
    if completed_updates < warmup_updates:
        return float(learning_rate) * float(completed_updates + 1) / float(warmup_updates + 1)
    if completed_updates > decay_updates:
        return float(min_learning_rate)
    if decay_updates == warmup_updates:
        return float(min_learning_rate)
    ratio = (completed_updates - warmup_updates) / (decay_updates - warmup_updates)
    ratio = min(1.0, max(0.0, ratio))
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return float(min_learning_rate) + coefficient * (float(learning_rate) - float(min_learning_rate))


def restart_cosine_learning_rate(
    *,
    completed_updates: int,
    phase_start_update: int,
    phase_start_lr: float,
    phase_peak_lr: float,
    phase_rewarm_iters: int,
    phase_end_update: int,
    phase_min_lr: float,
) -> float:
    if completed_updates < phase_start_update:
        return float(phase_start_lr)
    if phase_end_update <= phase_start_update:
        return float(phase_min_lr)
    last_child_input = phase_end_update - 1
    if completed_updates >= last_child_input:
        return float(phase_min_lr)
    if phase_rewarm_iters <= 0:
        decay_start = phase_start_update
        if completed_updates <= decay_start:
            return float(phase_peak_lr)
    else:
        rewarm_end = phase_start_update + phase_rewarm_iters
        if completed_updates <= rewarm_end:
            fraction = (completed_updates - phase_start_update) / float(phase_rewarm_iters)
            fraction = min(1.0, max(0.0, fraction))
            return float(phase_start_lr) + fraction * (float(phase_peak_lr) - float(phase_start_lr))
        decay_start = rewarm_end
    denominator = max(1, last_child_input - decay_start)
    ratio = (completed_updates - decay_start) / float(denominator)
    ratio = min(1.0, max(0.0, ratio))
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return float(phase_min_lr) + coefficient * (float(phase_peak_lr) - float(phase_min_lr))


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
        return restart_cosine_learning_rate(
            completed_updates=completed_updates,
            phase_start_update=int(config.phase_start_update),
            phase_start_lr=float(config.phase_start_lr),
            phase_peak_lr=float(config.phase_peak_lr),
            phase_rewarm_iters=int(config.phase_rewarm_iters),
            phase_end_update=int(config.phase_end_update),
            phase_min_lr=float(config.phase_min_lr),
        )
    raise ValueError(f"unsupported learning-rate schedule: {kind!r}")


__all__ = [
    "COSINE_SCHEDULE",
    "RESTART_COSINE_SCHEDULE",
    "LR_SCHEDULE_KINDS",
    "cosine_learning_rate",
    "restart_cosine_learning_rate",
    "learning_rate_for_config",
]
# ^^^ THOG
