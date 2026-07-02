# vvv THOG
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

from torch import Tensor

from .memory import MemoryTelemetry
from .trainer import SharedTrainer
from .training_config import TrainingConfig


class Stage4Trainer(SharedTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        train_tokens: Tensor,
        validation_tokens: Tensor,
    ) -> None:
        super().__init__(config, train_tokens, validation_tokens)
        self.memory_telemetry = MemoryTelemetry(self.device)
        self.memory_telemetry.snapshot("model_and_optimizer_ready")

    @classmethod
    def resume_for_inference(
        cls,
        checkpoint_path: Union[str, Path],
        tokens: Tensor,
        *,
        device: str = "cpu",
        dtype: str = "float32",
    ):
        trainer = cls.from_checkpoint(
            checkpoint_path,
            tokens,
            tokens,
            overrides={
                "device": device,
                "dtype": dtype,
                "checkpoint_segment_size": 0,
            },
        )
        trainer.model.eval()
        return trainer

    def train_one_update(self) -> Dict[str, float]:
        before = self.state.completed_updates
        metrics = super().train_one_update()
        phase = "first_optimizer_state" if before == 0 else "steady_update"
        self.memory_telemetry.snapshot(phase)
        return metrics

    def evaluate(self) -> Dict[str, float]:
        metrics = super().evaluate()
        self.memory_telemetry.snapshot("evaluation")
        return metrics

    def save_checkpoint(self, path):
        target = super().save_checkpoint(path)
        self.memory_telemetry.snapshot("checkpoint")
        return target
# ^^^ THOG
