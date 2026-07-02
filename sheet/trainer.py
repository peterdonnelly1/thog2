# vvv THOG
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from .batch_source import DeterministicBatchSource
from .checkpoints import optimizer_group_names
from .distributed import DistributedContext
from .trainer_checkpoint_resume import TrainerCheckpointResumeMixin
from .trainer_checkpoint_save import TrainerCheckpointSaveMixin
from .trainer_run import TrainerRunMixin
from .trainer_schedule import TrainerScheduleMixin
from .trainer_state import TrainerEvent, TrainerState
from .trainer_step import TrainerStepMixin
from .training_config import TrainingConfig
from .training_model import TrainingSheetGPT
from .training_model_factory import build_training_model, training_parameter_report


class SharedTrainer(
    TrainerCheckpointResumeMixin,
    TrainerCheckpointSaveMixin,
    TrainerRunMixin,
    TrainerStepMixin,
    TrainerScheduleMixin,
):
    """One reference lifecycle for dense and SheetGPT, including DDP."""

    def __init__(
        self,
        config: TrainingConfig,
        train_tokens: Tensor,
        validation_tokens: Tensor,
    ) -> None:
        self.config = config
        self.distributed = DistributedContext.from_environment(config.device)
        self.device = self.distributed.device
        self.state = TrainerState()
        self.events: List[TrainerEvent] = []

        self.raw_model = build_training_model(config, device=self.device)
        if isinstance(self.raw_model, TrainingSheetGPT):
            self.raw_model.set_checkpoint_segment_size(config.checkpoint_segment_size)
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
            enabled=(self.device.type == "cuda" and config.dtype == "float16"),
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

    def distributed_structure_signature(self) -> Dict[str, Any]:
        parameters: Tuple[Tuple[str, Tuple[int, ...], str, bool], ...] = tuple(
            (
                name,
                tuple(parameter.shape),
                str(parameter.dtype),
                bool(parameter.requires_grad),
            )
            for name, parameter in self.raw_model.named_parameters()
        )
        return {
            "parameters": parameters,
            "optimizer_groups": optimizer_group_names(self.optimizer),
            "parameter_count": sum(
                parameter.numel() for parameter in self.raw_model.parameters()
            ),
        }

    def close(self) -> None:
        self.distributed.close()


__all__ = ["SharedTrainer", "TrainerEvent", "TrainerState"]
# ^^^ THOG
