# vvv THOG
from __future__ import annotations

import math
from typing import Dict, List

import torch

from .model import SheetGPT


def root_mean_square(values: torch.Tensor) -> float:
    detached = values.detach().float()
    return float(torch.sqrt(torch.mean(detached.square())))


def normalized_order_energy(values: torch.Tensor, axis: int) -> List[float]:
    detached = values.detach().float().square()
    reduce_axes = tuple(index for index in range(detached.ndim) if index != axis)
    energy = detached.sum(dim=reduce_axes)
    total = energy.sum()
    if float(total) == 0.0:
        return [0.0 for _ in range(energy.numel())]
    return [float(value) for value in energy / total]


def coefficient_order_summary(
    model: SheetGPT,
    high_order_fraction: float = 0.25,
) -> Dict[str, object]:
    if not 0.0 < high_order_fraction <= 1.0:
        raise ValueError("high_order_fraction must be in (0, 1]")
    families: Dict[str, object] = {}
    for item in model.trajectory.metadata:
        values = model.trajectory.coefficients[item.name]
        depth_energy = normalized_order_energy(values, 1)
        row_energy = normalized_order_energy(values, 2)
        depth_high_count = max(1, math.ceil(len(depth_energy) * high_order_fraction))
        row_high_count = max(1, math.ceil(len(row_energy) * high_order_fraction))
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
            "depth_order_energy_fraction": depth_energy,
            "row_order_energy_fraction": row_energy,
            "high_depth_order_energy_fraction": sum(depth_energy[-depth_high_count:]),
            "high_row_order_energy_fraction": sum(row_energy[-row_high_count:]),
        }
    return families
# ^^^ THOG
