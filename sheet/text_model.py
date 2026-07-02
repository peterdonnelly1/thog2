# vvv THOG
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from .checkpoints import load_payload
from .compact_state import model_from_compact_state
from .training_config import TrainingConfig


def load_text_model(
    checkpoint_path: Path,
    *,
    device: str,
    dtype: str,
) -> Tuple[torch.nn.Module, TrainingConfig]:
    payload = load_payload(checkpoint_path, map_location="cpu")
    return model_from_compact_state(
        payload,
        device=device,
        dtype=dtype,
    )
# ^^^ THOG
