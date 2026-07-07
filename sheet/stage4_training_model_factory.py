# vvv THOG
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from model import GPTConfig
from .model import SheetGPTConfig
from .training_config import TrainingConfig
from .training_model import TrainingDenseGPT, TrainingSheetGPT
from .training_model_factory import _apply_sheet_residual_init_scaling, training_parameter_report


def _sheet_arguments_with_compact_selectors(config: TrainingConfig) -> Dict[str, Any]:
    arguments = dict(config.model_arguments())
    arguments.update(
        {
            "geometry_preset": config.geometry_preset,
            "attention_geometry": config.attention_geometry,
            "mlp_geometry": config.mlp_geometry,
            "basis_family": config.basis_family,
        }
    )
    return arguments


def build_training_model(
    config: TrainingConfig,
    *,
    device: Optional[torch.device] = None,
) -> nn.Module:
    target_device = torch.device(config.device) if device is None else device
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA model requested but no CUDA device is available")
    fork_devices: List[int] = []
    if target_device.type == "cuda":
        fork_devices = [target_device.index if target_device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(config.model_seed)
        if target_device.type == "cuda":
            torch.cuda.manual_seed_all(config.model_seed)
        if config.model_type == "dense":
            model: nn.Module = TrainingDenseGPT(GPTConfig(**config.model_arguments()))
        elif config.model_type == "thog2_sheet":
            model = TrainingSheetGPT(SheetGPTConfig(**_sheet_arguments_with_compact_selectors(config)))
            _apply_sheet_residual_init_scaling(model, config)
        else:
            raise ValueError(f"unsupported model_type: {config.model_type}")
    checkpoint_setter = getattr(model, "set_checkpoint_segment_size", None)
    if callable(checkpoint_setter):
        checkpoint_setter(config.checkpoint_segment_size)
    return model.to(target_device)
# ^^^ THOG
