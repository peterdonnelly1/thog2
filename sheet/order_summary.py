# vvv THOG
from __future__ import annotations

from typing import Dict

import torch

from .model import SheetGPT


def root_mean_square(values: torch.Tensor) -> float:
    detached = values.detach().float()
    return float(torch.sqrt(torch.mean(detached.square())))


def coefficient_order_summary(model: SheetGPT) -> Dict[str, object]:
    families: Dict[str, object] = {}
    for item in model.trajectory.metadata:
        values = model.trajectory.coefficients[item.name]
        families[item.name] = {
            "coefficient_rms": root_mean_square(values),
            "depth_order_rms": [
                root_mean_square(values[:, index, :])
                for index in range(values.shape[1])
            ],
            "row_order_rms": [
                root_mean_square(values[:, :, index])
                for index in range(values.shape[2])
            ],
        }
    return families
# ^^^ THOG
