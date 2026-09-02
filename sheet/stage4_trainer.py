# vvv THOG
from __future__ import annotations

import os
from typing import Dict, List

import torch
from torch import Tensor, nn

from .batch_source import DeterministicBatchSource
from .dense_snapshot import apply_dense_snapshot_startup
from .distributed import DistributedContext
from .memory import MemoryTelemetry
from .optimizer_factory import build_optimizer
from .trainer import SharedTrainer
from .trainer_state import TrainerEvent, TrainerState
from .training_config import TrainingConfig
from .training_model_factory import (
    build_training_model,
    training_parameter_report,
)


# vvv THOG optional torch.compile execution modes; raw_model remains the authoritative parameter/checkpoint/diagnostic module
def _torch_compile_mode() -> str:
    value = os.environ.get("THOG2_TORCH_COMPILE", "false").strip().lower()
    if value in {"false", "true", "regional"}:
        return value
    raise ValueError(
        "THOG2_TORCH_COMPILE must be false, true, or regional; "
        f"got {value!r}"
    )


def _execution_model(raw_model: nn.Module) -> nn.Module:
    compile_mode = _torch_compile_mode()
    if compile_mode == "false":
        return raw_model
    if compile_mode == "regional":
        compile_mode_setter = getattr(raw_model, "set_torch_compile_mode", None)
        if not callable(compile_mode_setter):
            raise RuntimeError(
                "THOG2_TORCH_COMPILE=regional requires a training model with "
                "set_torch_compile_mode support"
            )
        compile_function = getattr(torch, "compile", None)
        if compile_function is None:
            raise RuntimeError("THOG2_TORCH_COMPILE=regional requires torch.compile support")
        compile_mode_setter("regional")
        return raw_model
    compile_function = getattr(torch, "compile", None)
    if compile_function is None:
        raise RuntimeError("THOG2_TORCH_COMPILE=true requires torch.compile support")
    return compile_function(raw_model)
# ^^^ THOG


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
        # Stage4 owns the production construction path rather than delegating to
        # SharedTrainer.__init__, so it must invoke the same pre-optimizer hook.
        self.dense_snapshot_metadata = apply_dense_snapshot_startup(
            self.raw_model,
            config,
            self.distributed,
        )
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
        if self.dense_snapshot_metadata is not None:
            self.parameter_report = {
                **self.parameter_report,
                "dense_snapshot_baselining": dict(self.dense_snapshot_metadata),
            }
        # vvv THOG retained operational tensors disconnect compact parameters from the DDP forward graph until explicit gradient projection
        find_unused_parameters = False
        find_unused_requirement = getattr(
            self.raw_model,
            "requires_find_unused_parameters",
            None,
        )
        if callable(find_unused_requirement):
            find_unused_parameters = bool(find_unused_requirement())
        self.model = self.distributed.wrap_model(
            _execution_model(self.raw_model),
            find_unused_parameters=find_unused_parameters,
        )
        # ^^^ THOG
        self.optimizer = build_optimizer(
            self.raw_model,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            device_type=self.device.type,
            thogopt_config=config,  # <<< THOG independent history counts are checkpointed configuration
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
