# vvv THOG
from __future__ import annotations

import inspect
import os
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn


OPTIMIZER_ADAMW = "adamw"
OPTIMIZER_SGD = "sgd"
OPTIMIZER_SGD_NESTEROV = "sgd_nesterov"
OPTIMIZER_ADAFACTOR = "adafactor"
OPTIMIZER_RMSPROP = "rmsprop"
OPTIMIZER_NAMES = (
    OPTIMIZER_ADAMW,
    OPTIMIZER_SGD,
    OPTIMIZER_SGD_NESTEROV,
    OPTIMIZER_ADAFACTOR,
    OPTIMIZER_RMSPROP,
)
DEFAULT_OPTIMIZER = OPTIMIZER_ADAMW
DEFAULT_OPTIMIZER_MOMENTUM = 0.9

_OPTIMIZER_ALIASES = {
    "adam": OPTIMIZER_ADAMW,
    "adamw": OPTIMIZER_ADAMW,
    "sgd": OPTIMIZER_SGD,
    "nesterov": OPTIMIZER_SGD_NESTEROV,
    "sgd_nesterov": OPTIMIZER_SGD_NESTEROV,
    "sgd-nesterov": OPTIMIZER_SGD_NESTEROV,
    "adafactor": OPTIMIZER_ADAFACTOR,
    "rmsprop": OPTIMIZER_RMSPROP,
}


def normalize_optimizer_name(value: str) -> str:
    normalized = value.strip().lower()
    try:
        return _OPTIMIZER_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(
            f"unsupported optimizer {value!r}; expected one of {OPTIMIZER_NAMES}"
        ) from error


def optimizer_name_from_environment() -> str:
    return normalize_optimizer_name(os.environ.get("THOG2_OPTIMIZER", DEFAULT_OPTIMIZER))


def optimizer_momentum_from_environment() -> float:
    raw_value = os.environ.get(
        "THOG2_OPTIMIZER_MOMENTUM",
        str(DEFAULT_OPTIMIZER_MOMENTUM),
    )
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(
            f"THOG2_OPTIMIZER_MOMENTUM must be numeric; got {raw_value!r}"
        ) from error
    if not 0.0 <= value < 1.0:
        raise ValueError(
            "THOG2_OPTIMIZER_MOMENTUM must be in [0, 1); "
            f"got {value!r}"
        )
    return value


def _generic_parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    decay_parameters: List[nn.Parameter] = []
    decay_names: List[str] = []
    no_decay_parameters: List[nn.Parameter] = []
    no_decay_names: List[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
            decay_names.append(name)
        else:
            no_decay_parameters.append(parameter)
            no_decay_names.append(name)
    return (
        {
            "params": decay_parameters,
            "parameter_names": tuple(decay_names),
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_parameters,
            "parameter_names": tuple(no_decay_names),
            "weight_decay": 0.0,
        },
    )


def optimizer_parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> Iterable[Dict[str, object]]:
    group_builder = getattr(model, "optimizer_parameter_groups", None)
    if callable(group_builder):
        return group_builder(weight_decay)
    return _generic_parameter_groups(model, weight_decay)


def build_optimizer(
    model: nn.Module,
    *,
    weight_decay: float,
    learning_rate: float,
    betas: Tuple[float, float],
    device_type: str,
) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name_from_environment()
    momentum = optimizer_momentum_from_environment()
    parameter_groups = optimizer_parameter_groups(model, weight_decay)

    if optimizer_name == OPTIMIZER_ADAMW:
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused,
        )
        details = f"fused={use_fused} betas={betas}"
    elif optimizer_name == OPTIMIZER_SGD:
        optimizer = torch.optim.SGD(
            parameter_groups,
            lr=learning_rate,
            momentum=momentum,
        )
        details = f"momentum={momentum:g} nesterov=False"
    elif optimizer_name == OPTIMIZER_SGD_NESTEROV:
        optimizer = torch.optim.SGD(
            parameter_groups,
            lr=learning_rate,
            momentum=momentum,
            nesterov=True,
        )
        details = f"momentum={momentum:g} nesterov=True"
    elif optimizer_name == OPTIMIZER_ADAFACTOR:
        adafactor = getattr(torch.optim, "Adafactor", None)
        if adafactor is None:
            raise RuntimeError(
                "optimizer 'adafactor' requires a PyTorch build exposing "
                "torch.optim.Adafactor"
            )
        optimizer = adafactor(parameter_groups, lr=learning_rate)
        details = "factored_second_moment=True"
    elif optimizer_name == OPTIMIZER_RMSPROP:
        optimizer = torch.optim.RMSprop(
            parameter_groups,
            lr=learning_rate,
            momentum=momentum,
        )
        details = f"momentum={momentum:g} alpha=0.99"
    else:
        raise AssertionError(f"unhandled optimizer: {optimizer_name}")

    for group in optimizer.param_groups:
        group["thog2_optimizer_name"] = optimizer_name
    print(
        f"using optimizer: {optimizer_name} lr={learning_rate:.3e} {details}",
        flush=True,
    )
    return optimizer


__all__ = [
    "DEFAULT_OPTIMIZER",
    "DEFAULT_OPTIMIZER_MOMENTUM",
    "OPTIMIZER_NAMES",
    "build_optimizer",
    "normalize_optimizer_name",
    "optimizer_momentum_from_environment",
    "optimizer_name_from_environment",
]
# ^^^ THOG
