# vvv THOG
from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

from .batch_source import DeterministicBatchSource
from .distributed import DistributedContext
from .memory import MemoryTelemetry
from .stage4_training_model_factory import (
    build_training_model,
    training_parameter_report,
)
from .trainer import SharedTrainer
from .trainer_state import TrainerEvent, TrainerState
from .training_config import TrainingConfig


class Stage4Trainer(SharedTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        train_tokens: Tensor,
        validation_tokens: Tensor,
    ) -> None:
        self.config = config
        self.distributed = DistributedContext.from_environment(config.device)
        self.device = self.distributed.device
        self.memory_telemetry = MemoryTelemetry(self.device)
        self.memory_telemetry.snapshot("trainer_start")
        self.state = TrainerState()
        self.events: List[TrainerEvent] = []

        self.raw_model = build_training_model(config, device=self.device)
        checkpoint_setter = getattr(
            self.raw_model,
            "set_checkpoint_segment_size",
            None,
        )
        if callable(checkpoint_setter):
            checkpoint_setter(config.checkpoint_segment_size)
        self.memory_telemetry.snapshot("model_construction")
        self.parameter_report = training_parameter_report(
            self.raw_model,
            config.model_type,
        )
        self.model = self.distributed.wrap_model(self.raw_model)
        self.optimizer = self.raw_model.configure_optimizers(
            config.weight_decay,
            config.learning_rate,
            (config.beta1, config.beta2),
            self.device.type,
        )
        self.memory_telemetry.snapshot("optimizer_allocation")
        self.batch_source = DeterministicBatchSource(
            train_tokens,
            validation_tokens,
            block_size=config.block_size,
            batch_size=config.batch_size,
            data_seed=config.data_seed,
            rank=self.distributed.rank,
            world_size=self.distributed.world_size,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.device.type == "cuda"
                and config.dtype == "float16"
            ),
        )

        structure_signature = self.distributed_structure_signature()
        self.distributed.assert_identical_object(
            structure_signature,
            "parameter registration and optimizer grouping",
        )
        self._record(
            "model_constructed",
            parameter_report=self.parameter_report,
            distributed=self.distributed.report(),
            structure_signature=structure_signature,
        )

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
