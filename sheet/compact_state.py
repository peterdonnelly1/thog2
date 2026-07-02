# vvv THOG
from __future__ import annotations

from typing import Any, Mapping, Tuple

import torch

from .checkpoints import strip_compiled_prefix
from .training_config import TrainingConfig
from .training_model_factory import build_training_model


def model_from_compact_state(
    payload: Mapping[str, Any],
    *,
    device: str = "cpu",
    dtype: str = "float32",
) -> Tuple[torch.nn.Module, TrainingConfig]:
    values = dict(payload["trainer_config"])
    values["device"] = device
    values["dtype"] = dtype
    values["checkpoint_segment_size"] = 0
    config = TrainingConfig(**values)
    model = build_training_model(config)
    model.load_state_dict(strip_compiled_prefix(payload["model"]))
    model.eval()
    return model, config
# ^^^ THOG
