# vvv THOG
from __future__ import annotations

from types import SimpleNamespace

import torch

from sheet.stage6_diagnostics import coefficient_utilization_report, normalized_energy


def _fake_model_for_coefficient(name: str, coefficient: torch.Tensor, *, depth_order: int = 3):
    metadata = SimpleNamespace(name=name, semantic_type="matrix")
    trajectory = SimpleNamespace(metadata=(metadata,), coefficients={name: coefficient})
    return SimpleNamespace(config=SimpleNamespace(depth_order=depth_order), trajectory=trajectory)


def test_stage6_diagnostics_flatten_nested_energy_tensors_before_serialising() -> None:
    fractions = normalized_energy(torch.ones(2, 3))
    assert len(fractions) == 6
    assert abs(sum(fractions) - 1.0) < 1.0e-12


def test_stage6_diagnostics_handle_head_aware_block_four_dimensional_coefficients() -> None:
    model = _fake_model_for_coefficient("attention_query_weight", torch.ones(2, 3, 4, 5), depth_order=3)
    report = coefficient_utilization_report(model)["attention_query_weight"]
    assert report["shape"] == [2, 3, 4, 5]
    assert len(report["depth_order_energy_fraction"]) == 3
    assert len(report["row_order_energy_fraction"]) == 20
    assert report["high_depth_order_energy_fraction"] > 0.0
    assert report["high_row_order_energy_fraction"] > 0.0


def test_stage6_diagnostics_handle_block_mlp_three_dimensional_coefficients() -> None:
    model = _fake_model_for_coefficient("mlp_expansion_weight", torch.ones(3, 4, 5), depth_order=3)
    report = coefficient_utilization_report(model)["mlp_expansion_weight"]
    assert report["shape"] == [3, 4, 5]
    assert len(report["depth_order_energy_fraction"]) == 3
    assert len(report["row_order_energy_fraction"]) == 20
# ^^^ THOG
