# vvv THOG
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import torch
from torch import Tensor

from .batch_source import DeterministicBatchSource
from .memory import MemoryTelemetry
from .trainer import SharedTrainer
from .trainer_state import TrainerEvent, TrainerState
from .training_config import TrainingConfig
from .training_model import TrainingSheetGPT
from .training_model_factory import (
    build_training_model,
    training_parameter_report,
)


class Stage4Trainer(SharedTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        train_tokens: Tensor,
        validation_tokens: Tensor,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA training requested but no CUDA device is available"
            )
        self.memory_telemetry = MemoryTelemetry(self.device)
        self.memory_telemetry.snapshot("trainer_start")
        self.model = build_training_model(config)
        if isinstance(self.model, TrainingSheetGPT):
            self.model.set_checkpoint_segment_size(
                config.checkpoint_segment_size
            )
        self.memory_telemetry.snapshot("model_construction")
        self.batch_source = DeterministicBatchSource(
            train_tokens,
            validation_tokens,
            block_size=config.block_size,
            batch_size=config.batch_size,
            data_seed=config.data_seed,
        )
        self.optimizer = self.model.configure_optimizers(
            config.weight_decay,
            config.learning_rate,
            (config.beta1, config.beta2),
            self.device.type,
        )
        self.memory_telemetry.snapshot("optimizer_allocation")
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.device.type == "cuda"
                and config.dtype == "float16"
            ),
        )
        self.state = TrainerState()
        self.events: List[TrainerEvent] = []
        self.parameter_report = training_parameter_report(
            self.model,
            config.model_type,
        )
        self._record(
            "model_constructed",
            parameter_report=self.parameter_report,
        )

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
