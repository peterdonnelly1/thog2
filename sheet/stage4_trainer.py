# vvv THOG
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from .batch_source import DeterministicBatchSource
from .distributed import DistributedContext
from .memory import MemoryTelemetry
from .optimizer_factory import build_optimizer
from .recurrence_update_cache import RecurrenceUpdateCacheController
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
        # vvv THOG eager BQRG accumulation materialises once per optimizer update; compiled modes retain the established direct path until they gain an explicit cache contract
        trajectory = getattr(self.raw_model, "trajectory", None)
        attached_controller = getattr(
            trajectory,
            "_recurrence_update_cache_controller",
            None,
        )
        self._recurrence_update_cache_controller: Optional[RecurrenceUpdateCacheController] = (
            attached_controller
            if isinstance(attached_controller, RecurrenceUpdateCacheController)
            and config.gradient_accumulation_steps > 1
            and _torch_compile_mode() == "false"
            else None
        )
        # ^^^ THOG
        self.model = self.distributed.wrap_model(
            _execution_model(self.raw_model),
            find_unused_parameters=self._recurrence_update_cache_controller is not None,
        )
        self.optimizer = build_optimizer(
            self.raw_model,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            device_type=self.device.type,
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
            recurrence_update_cache=(
                self._recurrence_update_cache_controller is not None
            ),
        )

    # vvv THOG cached operational gradients are still GradScaler-scaled when the ordinary optimizer gradients have already been unscaled; materialise and unscale them before the shared finite check
    def _finalize_recurrence_update_cache(self) -> None:
        controller = self._recurrence_update_cache_controller
        if controller is None or not controller.active:
            return
        scale = float(self.scaler.get_scale()) if self.scaler.is_enabled() else 1.0
        finalized_parameters = controller.finalize(unscale_factor=1.0 / scale)
        self.distributed.mean_gradients_(finalized_parameters)
        self._record(
            "recurrence_update_cache_finalized",
            parameter_count=len(finalized_parameters),
        )

    def _local_gradients_are_finite(self) -> bool:
        self._finalize_recurrence_update_cache()
        return super()._local_gradients_are_finite()
    # ^^^ THOG

    def train_one_update(self) -> Dict[str, float]:
        before = self.state.completed_updates
        controller = self._recurrence_update_cache_controller
        if controller is not None:
            controller.begin()
            self._record(
                "recurrence_update_cache_started",
                accumulation_steps=self.config.gradient_accumulation_steps,
            )
        try:
            metrics = super().train_one_update()
        finally:
            if controller is not None:
                controller.discard()
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
