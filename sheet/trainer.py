# vvv THOG
from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from .batch_source import DeterministicBatchSource
from .model_factory import build_model, parameter_report
from .trainer_checkpoint_resume import TrainerCheckpointResumeMixin
from .trainer_checkpoint_save import TrainerCheckpointSaveMixin
from .trainer_run import TrainerRunMixin
from .trainer_schedule import TrainerScheduleMixin
from .trainer_state import TrainerEvent, TrainerState
from .trainer_step import TrainerStepMixin
from .training_config import TrainingConfig


class SharedTrainer(
    TrainerCheckpointResumeMixin,
    TrainerCheckpointSaveMixin,
    TrainerRunMixin,
    TrainerStepMixin,
    TrainerScheduleMixin,
):
    """One reference lifecycle for dense and SheetGPT training."""

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
        self.model = build_model(config)
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
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.device.type == "cuda"
                and config.dtype == "float16"
            ),
        )
        self.state = TrainerState()
        self.events: List[TrainerEvent] = []
        self.parameter_report = parameter_report(
            self.model,
            config.model_type,
        )
        self._record(
            "model_constructed",
            parameter_report=self.parameter_report,
        )


__all__ = [
    "SharedTrainer",
    "TrainerEvent",
    "TrainerState",
]
# ^^^ THOG
