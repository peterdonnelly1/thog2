# vvv THOG
from __future__ import annotations

from .training_config import TrainingConfig


def architecture_run_name(config: TrainingConfig) -> str:
    if config.model_type == "thog2_sheet":
        return (
            f"THOG2_SHEET_L{config.n_layer}_H{config.n_head}_D{config.n_embd}"
            f"_C{config.block_size}_P{config.depth_order}_Q{config.base_row_order}"
        )
    return (
        f"DENSE_L{config.n_layer}_H{config.n_head}_D{config.n_embd}"
        f"_C{config.block_size}"
    )
# ^^^ THOG
