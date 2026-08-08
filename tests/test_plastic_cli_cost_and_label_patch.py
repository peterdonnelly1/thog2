# vvv THOG
from __future__ import annotations

import inspect

from run_thog2_owt_core import build_parser
from sheet.plastic_depth_cli_cost_and_label_patch import (
    _DEFAULT_LAYER_COUNT_COST_WEIGHT,
    _compact_offset_vector_labels,
)
from sheet.run_config import OwtRunConfig
from sheet.training_config import TrainingConfig


def test_cost_weight_default_is_consistent_across_cli_and_config_classes() -> None:
    expected = _DEFAULT_LAYER_COUNT_COST_WEIGHT
    assert expected == 0.02
    assert (
        inspect.signature(TrainingConfig).parameters[
            "plastic__layer_count_cost_weight"
        ].default
        == expected
    )
    assert (
        inspect.signature(OwtRunConfig).parameters[
            "plastic__layer_count_cost_weight"
        ].default
        == expected
    )
    assert build_parser().parse_args([]).plastic__layer_count_cost_weight == expected


def test_cost_weight_cli_explicit_override_is_preserved() -> None:
    arguments = build_parser().parse_args(
        ("--plastic__layer_count_cost_weight", "0")
    )
    assert arguments.plastic__layer_count_cost_weight == 0.0


def test_long_probe_and_score_offset_labels_use_ellipsis() -> None:
    line = (
        "probe_losses [L-5, L-4, L-3, L-2, L-1, L, L+1, L+2, L+3, L+4, L+5] = "
        "[1, 2, 3]  "
        "score_z [L-5, L-4, L-3, L-2, L-1, L+1, L+2, L+3, L+4, L+5] = "
        "[-1, 0, 1]"
    )

    rendered = _compact_offset_vector_labels(line)

    assert "probe_losses [L-5 ... L+5] =" in rendered
    assert "score_z [L-5 ... L+5] =" in rendered
    assert "L-4" not in rendered
    assert "L+4" not in rendered


def test_radius_one_labels_remain_explicit() -> None:
    line = (
        "probe_losses [L-1, L, L+1] = [1, 2, 3]  "
        "score_z [L-1, L+1] = [-1, 1]"
    )
    assert _compact_offset_vector_labels(line) == line
# ^^^ THOG
