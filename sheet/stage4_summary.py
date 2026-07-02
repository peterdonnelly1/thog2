# vvv THOG
from __future__ import annotations

from typing import Dict, Iterable

from .model import SheetGPT
from .order_summary import coefficient_order_summary
from .parameter_summary import derivative_norms
from .weight_summary import generated_weight_summary


def summarize_sheet(
    model: SheetGPT,
    layer_indices: Iterable[int] = (0,),
) -> Dict[str, object]:
    parameters = {
        item.name: model.trajectory.coefficients[item.name]
        for item in model.trajectory.metadata
    }
    return {
        "orders": coefficient_order_summary(model),
        "derivatives": derivative_norms(parameters),
        "generated": generated_weight_summary(model, layer_indices),
    }
# ^^^ THOG
