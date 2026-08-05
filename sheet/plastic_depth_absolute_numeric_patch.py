# vvv THOG
"""Numerically stable display helpers for the PLASTIC absolute ruler."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import torch
from torch import Tensor

from . import plastic_depth_absolute_ruler_patch as _absolute


# vvv THOG keep the fixed capacity ruler endpoints exact in display/public-coordinate reports
def _capacity_public_coordinates_from_raw_stable(lattice: Any, raw_intervals: Tensor) -> Tensor:
    maximum_layers = int(lattice.maximum_layers)
    if maximum_layers == 1:
        return raw_intervals.new_tensor([50.5])
    intervals = lattice._positive_prefix(maximum_layers - 1, raw_intervals=raw_intervals)
    cumulative = torch.cat((intervals.new_zeros(1), torch.cumsum(intervals, dim=0)))
    coordinates = 1.0 + 99.0 * cumulative / intervals.sum()
    coordinates = coordinates.clone()
    coordinates[0] = 1.0
    coordinates[-1] = 100.0
    return coordinates


def _sample_layer_tuple_stable(public_values: Sequence[float], maximum_layers: int) -> Tuple[float, ...]:
    if int(maximum_layers) == 1:
        return tuple(1.0 for _ in public_values)
    scale = (float(maximum_layers) - 1.0) / 99.0
    return tuple(round(1.0 + scale * (float(value) - 1.0), 6) for value in public_values)


_absolute._capacity_public_coordinates_from_raw = _capacity_public_coordinates_from_raw_stable
_absolute._sample_layer_tuple = _sample_layer_tuple_stable
# ^^^ THOG
