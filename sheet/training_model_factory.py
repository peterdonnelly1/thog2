# vvv THOG
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn

from model import GPT, GPTConfig
from .model import SheetGPT, SheetGPTConfig
from .training_config import TrainingConfig
from .training_model import TrainingSheetGPT


def build_training_model(config: TrainingConfig) -> nn.Module:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA model requested but no CUDA device is available"
        )
    fork_devices: List[int] = []
    if device.type == "cuda":
        fork_devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(config.model_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.model_seed)
        if config.model_type == "dense":
            model: nn.Module = GPT(
                GPTConfig(**config.model_arguments())
            )
        elif config.model_type == "thog2_sheet":
            model = TrainingSheetGPT(
                SheetGPTConfig(**config.model_arguments())
            )
        else:
            raise ValueError(f"unsupported model_type: {config.model_type}")
    return model.to(device)


def training_parameter_report(
    model: nn.Module,
    model_type: str,
) -> Dict[str, Any]:
    if model_type == "thog2_sheet":
        if not isinstance(model, SheetGPT):
            raise TypeError("thog2_sheet report requires SheetGPT")
        return model.parameter_report()
    if model_type != "dense" or not isinstance(model, GPT):
        raise TypeError("dense report requires GPT")
    persistent = sum(parameter.numel() for parameter in model.parameters())
    return {
        "persistent_parameters": persistent,
        "sheet_coefficients": 0,
        "conventional_non_sheet_parameters": persistent,
        "dense_equivalent_repeated_parameters": persistent,
        "dense_equivalent_total_parameters": persistent,
        "matrix_sheet_coefficients": 0,
        "matrix_dense_equivalent_parameters": persistent,
        "families": (),
    }
# ^^^ THOG
