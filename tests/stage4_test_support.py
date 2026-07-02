# vvv THOG
from __future__ import annotations

from typing import Any

import torch

from sheet.model import SheetGPTConfig
from sheet.training_config import TrainingConfig
from sheet.training_model import TrainingSheetGPT


def stage4_model(
    *,
    dropout: float = 0.0,
    checkpoint_segment_size: int = 0,
) -> TrainingSheetGPT:
    torch.manual_seed(4101)
    model = TrainingSheetGPT(
        SheetGPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=4,
            n_head=2,
            n_embd=16,
            dropout=dropout,
            bias=True,
            depth_order=3,
            base_row_order=8,
        )
    )
    model.set_checkpoint_segment_size(checkpoint_segment_size)
    return model


def stage4_batch():
    inputs = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1, 0],
        ],
        dtype=torch.long,
    )
    targets = torch.roll(inputs, shifts=-1, dims=1)
    return inputs, targets


def stage4_training_config(**overrides: Any) -> TrainingConfig:
    values = dict(
        model_type="thog2_sheet",
        block_size=8,
        vocab_size=32,
        n_layer=4,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
        depth_order=3,
        base_row_order=8,
        checkpoint_segment_size=0,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_updates=2,
        learning_rate=1.0e-3,
        min_learning_rate=1.0e-4,
        decay_updates=2,
        weight_decay=0.01,
        grad_clip=1.0,
        eval_interval=0,
        eval_batches=1,
        checkpoint_interval=0,
        model_seed=4101,
        data_seed=4102,
        device="cpu",
        dtype="float32",
    )
    values.update(overrides)
    return TrainingConfig(**values)


def stage4_tokens(length: int = 256):
    values = torch.arange(length, dtype=torch.long) % 32
    return values.clone(), torch.roll(values, shifts=11)
# ^^^ THOG
