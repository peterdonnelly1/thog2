# vvv THOG
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from model import GPT, GPTConfig
from .model import SheetGPT, SheetGPTConfig
from .training_config import TrainingConfig
from .training_model import TrainingDenseGPT, TrainingSheetGPT
from .trajectory import MATRIX_RESIDUAL_FAMILIES


def _apply_sheet_residual_init_scaling(
    model: SheetGPT,
    config: TrainingConfig,
) -> None:
    residual_std = config.residual_init_config().residual_std(
        model_type=config.model_type,
        n_layer=config.n_layer,
        depth_order=config.depth_order,
    )
    with torch.no_grad():
        for item in model.trajectory.metadata:
            if item.name not in MATRIX_RESIDUAL_FAMILIES:
                continue
            previous_std = item.target_weight_std
            if previous_std <= 0.0:
                raise RuntimeError(
                    f"residual family {item.name} has non-positive init std {previous_std}"
                )
            model.trajectory.coefficients[item.name].mul_(residual_std / previous_std)
            object.__setattr__(item, "target_weight_std", residual_std)


def _sheet_model_arguments(config: TrainingConfig) -> Dict[str, Any]:
    arguments = dict(config.model_arguments())
    arguments["geometry_preset"] = config.geometry_preset
    arguments["attention_geometry"] = config.attention_geometry
    arguments["mlp_geometry"] = config.mlp_geometry
    arguments["basis_family"] = config.basis_family
    return arguments


def build_training_model(
    config: TrainingConfig,
    *,
    device: Optional[torch.device] = None,
) -> nn.Module:
    target_device = torch.device(config.device) if device is None else device
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA model requested but no CUDA device is available"
        )
    fork_devices: List[int] = []
    if target_device.type == "cuda":
        fork_devices = [
            target_device.index
            if target_device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(config.model_seed)
        if target_device.type == "cuda":
            torch.cuda.manual_seed_all(config.model_seed)
        if config.model_type == "dense":
            model: nn.Module = TrainingDenseGPT(
                GPTConfig(**config.model_arguments())
            )
        elif config.model_type == "thog2_sheet":
            model = TrainingSheetGPT(
                SheetGPTConfig(**_sheet_model_arguments(config))
            )
            _apply_sheet_residual_init_scaling(model, config)
        else:
            raise ValueError(f"unsupported model_type: {config.model_type}")
    checkpoint_setter = getattr(model, "set_checkpoint_segment_size", None)
    if callable(checkpoint_setter):
        checkpoint_setter(config.checkpoint_segment_size)
    return model.to(target_device)


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
