# vvv THOG
from __future__ import annotations

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
# ^^^ THOG
