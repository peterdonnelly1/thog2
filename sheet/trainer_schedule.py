# vvv THOG
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from .trainer_state import TrainerEvent
from .lr_schedule import learning_rate_for_config


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
        return learning_rate_for_config(self.config, completed_updates)

    def _set_learning_rate(self) -> float:
        learning_rate = self.learning_rate_for_update(
            self.state.completed_updates
        )
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate
# ^^^ THOG
