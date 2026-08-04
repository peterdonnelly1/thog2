# vvv THOG
from __future__ import annotations

from typing import Any, Mapping, Tuple

import torch

# vvv THOG inference must reject unsafe PLASTIC geometry before model construction or state application
from .checkpoints import strip_compiled_prefix, validate_plastic_depth_checkpoint_format
# ^^^ THOG
from .training_config import TrainingConfig
from .training_model_factory import build_training_model


def model_from_compact_state(
    payload: Mapping[str, Any],
    *,
    device: str = "cpu",
    dtype: str = "float32",
) -> Tuple[torch.nn.Module, TrainingConfig]:
    # vvv THOG guard the compact inference path as strictly as trainer resume
    validate_plastic_depth_checkpoint_format(payload)
    # ^^^ THOG
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
