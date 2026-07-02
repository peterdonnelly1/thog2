# vvv THOG
from __future__ import annotations

import gc
import math
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
        block_size=128,
        vocab_size=512,
        n_layer=24,
        n_head=8,
        n_embd=256,
        depth_order=8,
        base_row_order=64,
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
    values = torch.arange(8192, dtype=torch.long) % config.vocab_size
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


def compare_runtime_memory(
    *,
    device: str,
    dtype: str,
) -> Dict[str, object]:
    reference = measure_runtime_case(0, device=device, dtype=dtype)
    checkpointed = measure_runtime_case(2, device=device, dtype=dtype)
    reference_peak = int(reference["peak_allocated_bytes"])
    checkpointed_peak = int(checkpointed["peak_allocated_bytes"])
    reference_loss = float(reference["training_loss"])
    checkpointed_loss = float(checkpointed["training_loss"])
    loss_delta = abs(reference_loss - checkpointed_loss)
    satisfied = (
        math.isfinite(reference_loss)
        and math.isfinite(checkpointed_loss)
        and loss_delta <= 1.0e-4
        and checkpointed_peak < reference_peak
        and bool(checkpointed["checkpointing_used"])
        and int(checkpointed["checkpoint_segments"]) == 12
    )
    return {
        "test": "S4-07",
        "satisfied": satisfied,
        "reference": reference,
        "checkpointed": checkpointed,
        "peak_allocated_ratio": checkpointed_peak / reference_peak,
        "peak_allocated_reduction_bytes": reference_peak - checkpointed_peak,
        "loss_delta": loss_delta,
    }
# ^^^ THOG
