# vvv THOG
from __future__ import annotations

import gc
from typing import Dict

import torch

from .memory import MemoryTelemetry
from .stage4_trainer import Stage4Trainer
from .training_config import TrainingConfig


def runtime_check_config(
    segment_size: int,
    device: str,
    dtype: str,
) -> TrainingConfig:
    return TrainingConfig(
        model_type="thog2_sheet",
        block_size=64,
        vocab_size=256,
        n_layer=12,
        n_head=4,
        n_embd=128,
        depth_order=6,
        base_row_order=32,
        checkpoint_segment_size=segment_size,
        batch_size=2,
        max_updates=1,
        decay_updates=1,
        eval_interval=0,
        checkpoint_interval=0,
        model_seed=4201,
        data_seed=4202,
        device=device,
        dtype=dtype,
    )


def measure_runtime_case(
    segment_size: int,
    *,
    device: str,
    dtype: str,
) -> Dict[str, object]:
    config = runtime_check_config(segment_size, device, dtype)
    values = torch.arange(4096, dtype=torch.long) % config.vocab_size
    validation = torch.roll(values, shifts=17)
    telemetry = MemoryTelemetry(torch.device(device))
    telemetry.reset_peak()
    trainer = Stage4Trainer(config, values, validation)
    metrics = trainer.train_one_update()
    sample = telemetry.snapshot("completed_update")
    execution = trainer.model.last_execution_report
    result = {
        "segment_size": segment_size,
        "training_loss": metrics["training_loss"],
        "gradient_norm": metrics["gradient_norm"],
        "peak_allocated_bytes": sample.peak_allocated_bytes,
        "peak_reserved_bytes": sample.peak_reserved_bytes,
        "checkpointing_used": execution.checkpointing_used,
        "checkpoint_segments": execution.checkpoint_segments,
    }
    del trainer
    gc.collect()
    if telemetry.cuda_enabled:
        torch.cuda.empty_cache()
    return result
# ^^^ THOG
