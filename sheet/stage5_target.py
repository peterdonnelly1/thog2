# vvv THOG
from __future__ import annotations

from typing import Any

from .training_config import TrainingConfig


def principal_stage5_config(
    *,
    device: str,
    dtype: str,
    max_updates: int = 4,
    checkpoint_segment_size: int = 4,
    out_dir: str = "out-stage5-principal",
) -> TrainingConfig:
    return TrainingConfig(
        model_type="thog2_sheet",
        block_size=256,
        vocab_size=50304,
        n_layer=144,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
        depth_order=16,
        base_row_order=128,
        checkpoint_segment_size=checkpoint_segment_size,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_updates=max_updates,
        learning_rate=6.0e-4,
        min_learning_rate=6.0e-5,
        warmup_updates=0,
        decay_updates=max_updates,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        eval_interval=0,
        eval_batches=1,
        checkpoint_interval=0,
        model_seed=5101,
        data_seed=5102,
        device=device,
        dtype=dtype,
        out_dir=out_dir,
    )


def reduced_stage5_config(
    *,
    device: str,
    dtype: str,
    checkpoint_segment_size: int = 2,
    **overrides: Any,
) -> TrainingConfig:
    values = dict(
        model_type="thog2_sheet",
        block_size=64,
        vocab_size=512,
        n_layer=12,
        n_head=8,
        n_embd=256,
        dropout=0.0,
        bias=True,
        depth_order=8,
        base_row_order=64,
        checkpoint_segment_size=checkpoint_segment_size,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_updates=1,
        learning_rate=6.0e-4,
        min_learning_rate=6.0e-5,
        warmup_updates=0,
        decay_updates=1,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        eval_interval=0,
        eval_batches=1,
        checkpoint_interval=0,
        model_seed=5201,
        data_seed=5202,
        device=device,
        dtype=dtype,
        out_dir="out-stage5-reduced",
    )
    values.update(overrides)
    return TrainingConfig(**values)


def synthetic_token_splits(vocab_size: int, *, length: int = 32768):
    import torch

    base = torch.arange(length, dtype=torch.long) % min(vocab_size, 257)
    validation = torch.roll(base, shifts=31)
    return base, validation


__all__ = [
    "principal_stage5_config",
    "reduced_stage5_config",
    "synthetic_token_splits",
]
# ^^^ THOG
