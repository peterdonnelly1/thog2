# vvv THOG
from __future__ import annotations

from typing import Dict, Iterable

import torch

from .model import SheetGPT
from .order_summary import root_mean_square


def generated_weight_summary(
    model: SheetGPT,
    layer_indices: Iterable[int],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    selected = tuple(layer_indices)
    with torch.no_grad():
        for item in model.trajectory.metadata:
            samples = []
            for layer_index in selected:
                values = model.trajectory.materialize(
                    item.name,
                    layer_index,
                ).detach().float()
                endpoints = torch.cat(
                    (values[..., :1], values[..., -1:]),
                    dim=-1,
                )
                interior = values[..., 1:-1]
                interior_rms = (
                    root_mean_square(interior)
                    if interior.numel() > 0
                    else root_mean_square(endpoints)
                )
                endpoint_rms = root_mean_square(endpoints)
                samples.append(
                    {
                        "layer_index": layer_index,
                        "mean": float(values.mean()),
                        "rms": root_mean_square(values),
                        "standard_deviation": float(values.std(unbiased=False)),
                        "maximum_absolute": float(values.abs().max()),
                        "endpoint_to_interior_rms_ratio": (
                            endpoint_rms / interior_rms
                            if interior_rms > 0.0
                            else 0.0
                        ),
                    }
                )
            result[item.name] = samples
    return result
# ^^^ THOG
