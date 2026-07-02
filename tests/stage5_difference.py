# vvv THOG
from __future__ import annotations

import math
from typing import Any, Dict

import torch


def nested_tensor_difference(left: Any, right: Any) -> Dict[str, Any]:
    totals = {
        "delta_square_sum": 0.0,
        "reference_square_sum": 0.0,
        "max_absolute_delta": 0.0,
        "max_delta_path": "",
        "tensor_count": 0,
        "element_count": 0,
    }

    def visit(left_value: Any, right_value: Any, path: str) -> None:
        if isinstance(left_value, torch.Tensor):
            if not isinstance(right_value, torch.Tensor):
                raise TypeError(f"right value at {path} is not a tensor")
            if left_value.shape != right_value.shape:
                raise ValueError(
                    f"tensor shape mismatch at {path}: "
                    f"{left_value.shape} != {right_value.shape}"
                )
            left_double = left_value.detach().to(dtype=torch.float64, device="cpu")
            right_double = right_value.detach().to(dtype=torch.float64, device="cpu")
            delta = left_double - right_double
            maximum = float(delta.abs().max().item()) if delta.numel() else 0.0
            totals["delta_square_sum"] += float(torch.sum(delta * delta).item())
            totals["reference_square_sum"] += float(
                torch.sum(left_double * left_double).item()
            )
            totals["tensor_count"] += 1
            totals["element_count"] += int(delta.numel())
            if maximum > totals["max_absolute_delta"]:
                totals["max_absolute_delta"] = maximum
                totals["max_delta_path"] = path
            return
        if isinstance(left_value, dict):
            if not isinstance(right_value, dict) or set(left_value) != set(right_value):
                raise ValueError(f"mapping mismatch at {path}")
            for key in left_value:
                visit(left_value[key], right_value[key], f"{path}.{key}")
            return
        if isinstance(left_value, (list, tuple)):
            if not isinstance(right_value, (list, tuple)) or len(left_value) != len(right_value):
                raise ValueError(f"sequence mismatch at {path}")
            for index, (left_item, right_item) in enumerate(zip(left_value, right_value)):
                visit(left_item, right_item, f"{path}[{index}]")
            return
        if left_value != right_value:
            raise ValueError(
                f"non-tensor value mismatch at {path}: "
                f"{left_value!r} != {right_value!r}"
            )

    visit(left, right, "root")
    delta_norm = math.sqrt(float(totals.pop("delta_square_sum")))
    reference_norm = math.sqrt(float(totals.pop("reference_square_sum")))
    totals["delta_l2_norm"] = delta_norm
    totals["reference_l2_norm"] = reference_norm
    totals["relative_l2_error"] = delta_norm / max(reference_norm, 1.0e-30)
    return totals


__all__ = ["nested_tensor_difference"]
# ^^^ THOG
