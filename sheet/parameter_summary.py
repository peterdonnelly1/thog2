# vvv THOG
from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def derivative_norms(parameters: Dict[str, nn.Parameter]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for name, parameter in parameters.items():
        derivative = getattr(parameter, "grad", None)
        result[name] = (
            float(torch.linalg.vector_norm(derivative.detach().float()))
            if derivative is not None
            else 0.0
        )
    return result
# ^^^ THOG
