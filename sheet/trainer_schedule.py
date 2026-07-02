# vvv THOG
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import torch

from .trainer_state import TrainerEvent


class TrainerScheduleMixin:
    def _record(self, name: str, **payload: Any) -> None:
        distributed = getattr(self, "distributed", None)
        if distributed is not None and distributed.active:
            payload = {
                "rank": distributed.rank,
                "world_size": distributed.world_size,
                **payload,
            }
        self.events.append(
            TrainerEvent(name, self.state.completed_updates, dict(payload))
        )

    def autocast_context(self):
        if self.config.dtype == "float32":
            return nullcontext()
        dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def learning_rate_for_update(self, completed_updates: int) -> float:
        if not self.config.decay_learning_rate:
            return self.config.learning_rate
        if completed_updates < self.config.warmup_updates:
            return self.config.learning_rate * (completed_updates + 1) / (
                self.config.warmup_updates + 1
            )
        if completed_updates > self.config.decay_updates:
            return self.config.min_learning_rate
        if self.config.decay_updates == self.config.warmup_updates:
            return self.config.min_learning_rate
        ratio = (
            completed_updates - self.config.warmup_updates
        ) / (
            self.config.decay_updates - self.config.warmup_updates
        )
        ratio = min(1.0, max(0.0, ratio))
        coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.config.min_learning_rate + coefficient * (
            self.config.learning_rate - self.config.min_learning_rate
        )

    def _set_learning_rate(self) -> float:
        learning_rate = self.learning_rate_for_update(
            self.state.completed_updates
        )
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate
# ^^^ THOG
