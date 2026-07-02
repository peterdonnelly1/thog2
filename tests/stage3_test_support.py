# vvv THOG
from __future__ import annotations

from typing import Any

import torch

from sheet.training_config import TrainingConfig


def token_splits(
    vocab_size: int = 32,
    length: int = 512,
):
    base = torch.arange(length, dtype=torch.long) % vocab_size
    return base.clone(), torch.roll(base, shifts=7)


def stage3_config(model_type: str, **overrides: Any) -> TrainingConfig:
    values = dict(
        model_type=model_type,
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
        depth_order=2,
        base_row_order=8,
        batch_size=2,
        gradient_accumulation_steps=2,
        max_updates=4,
        learning_rate=2.0e-3,
        min_learning_rate=2.0e-4,
        warmup_updates=1,
        decay_updates=4,
        weight_decay=0.01,
        grad_clip=1.0,
        eval_interval=2,
        eval_batches=1,
        checkpoint_interval=0,
        model_seed=101,
        data_seed=202,
        device="cpu",
        dtype="float32",
    )
    values.update(overrides)
    return TrainingConfig(**values)


def assert_nested_equal(test, left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            assert_nested_equal(test, left_item, right_item)
    else:
        test.assertEqual(left, right)
# ^^^ THOG
