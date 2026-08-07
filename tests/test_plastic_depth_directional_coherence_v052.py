# vvv THOG
from __future__ import annotations

import math
import re

import constants

from run_thog2_owt_core import build_parser
from sheet import plastic_depth_directional_coherence_patch as directional


def _score(count: int, value: float) -> dict[str, object]:
    return {
        "active_layers": count,
        "feasible": True,
        "score": value,
        "validation_loss": value,
    }


def _plain(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def test_right_support_uses_distance_decay_and_whole_side_extrapolation_discount() -> None:
    report = directional._directional_support(
        current_count=10,
        score_by_count={10: 10.0, 11: 9.0, 12: 11.0, 13: 9.0},
        extrapolation_weight=0.8,
    )
    expected = 0.8 * (1.0 + 0.0 * 0.8 + 1.0 * 0.8**2) / (1.0 + 0.8 + 0.8**2)
    assert math.isclose(report["right_support"], expected, rel_tol=0.0, abs_tol=1.0e-12)
    assert report["left_support"] is None


def test_both_sides_majority_is_ambiguous_even_if_one_support_is_larger() -> None:
    report = directional._directional_support(
        current_count=10,
        score_by_count={8: 8.5, 9: 9.0, 10: 10.0, 11: 9.0, 12: 9.0},
        extrapolation_weight=0.8,
    )
    assert report["left_support"] == 1.0
    assert report["right_support"] == 0.8
    assert report["vote"] == directional._DIRECTION_AMBIGUOUS


def test_boundary_with_only_right_side_can_vote_right() -> None:
    report = directional._directional_support(
        current_count=1,
        score_by_count={1: 10.0, 2: 9.0},
        extrapolation_weight=0.8,
    )
    assert report["left_support"] is None
    assert report["right_support"] > 0.5
    assert report["vote"] == directional._DIRECTION_RIGHT


def test_count_change_requires_the_complete_probe_history_window() -> None:
    score_report = (_score(9, 9.0), _score(10, 10.0), _score(11, 11.0))
    hold = directional.choose_plastic_depth_count_with_directional_coherence(
        current_count=10,
        score_report=score_report,
        histories={"10:-1": (-1.0,), "10:+1": (1.0,), "10:@LRA": (-1.0,)},
        noise_window=3,
        noise_lambda=0.0,
        update_number=20,
        last_count_change_update=0,
        update_brake=0,
        max_step=1,
        extrapolation_weight=0.8,
    )
    assert hold.selected_count == 10

    move = directional.choose_plastic_depth_count_with_directional_coherence(
        current_count=10,
        score_report=score_report,
        histories={"10:-1": (-1.0, -1.0), "10:+1": (1.0, 1.0), "10:@LRA": (-1.0, -1.0)},
        noise_window=3,
        noise_lambda=0.0,
        update_number=21,
        last_count_change_update=0,
        update_brake=0,
        max_step=1,
        extrapolation_weight=0.8,
    )
    assert move.selected_count == 9
    assert move.histories == {}


def test_directional_majority_vetoes_candidate_on_the_other_side() -> None:
    score_report = (_score(9, 11.0), _score(10, 10.0), _score(11, 9.0))
    decision = directional.choose_plastic_depth_count_with_directional_coherence(
        current_count=10,
        score_report=score_report,
        histories={
            "10:-1": (1.0, 1.0),
            "10:+1": (-1.0, -1.0),
            "10:@LRA": (-1.0, -1.0),
        },
        noise_window=3,
        noise_lambda=0.0,
        update_number=30,
        last_count_change_update=0,
        update_brake=0,
        max_step=1,
        extrapolation_weight=0.8,
    )
    assert decision.selected_count == 10
    assert all(not item.significant for item in decision.evidence)


def test_extrapolation_weight_one_restores_symmetric_side_weighting() -> None:
    report = directional._directional_support(
        current_count=10,
        score_by_count={8: 9.0, 9: 11.0, 10: 10.0, 11: 9.0, 12: 11.0},
        extrapolation_weight=1.0,
    )
    assert report["left_support"] == 0.5
    assert report["right_support"] == 0.5
    assert report["vote"] == directional._DIRECTION_AMBIGUOUS


def test_new_probe_names_are_canonical_and_min_probes_is_removed() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--plastic__layer_count_probe__probe_every_n_steps",
            "10",
            "--plastic__layer_count_probe__window_size_as_number_of_probes",
            "8",
            "--plastic__layer_count_extrapolation_weight",
            "0.8",
        ]
    )
    assert args.plastic__layer_count_probe__probe_every_n_steps == 10
    assert args.plastic__layer_count_probe__window_size_as_number_of_probes == 8
    assert args.plastic__layer_count_extrapolation_weight == 0.8
    assert not hasattr(args, "plastic__layer_count_min_probes")


def test_direction_console_text_is_exact() -> None:
    assert directional._format_lra_summary((1, 5, 2, 8, "R")) == "L/R/A=[1/5/2]/8=>R"
    assert directional._format_win_counts((2, 1, 3, 2, 1), (6, 7, 6, 5, 4), 8) == (
        "wins L[2,1,3,2,1]/8; R[6,7,6,5,4]/8"
    )


def test_sampled_moves_after_layers_and_changed_value_is_pink_for_one_row_only() -> None:
    directional._SAMPLED_BY_RUN_ID.pop("v052", None)
    base = "T 1 layers = 8\tprobe_losses [L-1, L, L+1] = [5.1, 5.0, 4.9]\tsampled = [1.0, 2.0, 3.0]"
    moved = directional._move_sampled_after_layers(base)
    assert _plain(moved).index("layers = 8") < _plain(moved).index("sampled =") < _plain(moved).index("probe_losses")
    first = directional._highlight_changed_sampled_values("v052", "optimizer_progress", moved)
    assert constants.PINK not in first

    changed = moved.replace("sampled = [1.0, 2.0, 3.0]", "sampled = [1.0, 1.9, 3.0]")
    second = directional._highlight_changed_sampled_values("v052", "optimizer_progress", changed)
    assert constants.PINK in second
    assert f"{constants.PINK}1.9{constants.R}" in second

    third = directional._highlight_changed_sampled_values("v052", "optimizer_progress", changed)
    assert constants.PINK not in third


def test_current_l_probe_loss_is_bold_white_using_offset_zero_not_vector_midpoint() -> None:
    line = "probe_losses [L-2 ... L+1] = [5.3, 5.2, 5.1, 5.0]"
    rendered = directional._bold_current_probe_loss(line, (-2, -1, 0, 1))
    assert f"{constants.BOLD_WHITE}5.1{constants.R}" in rendered
    assert f"{constants.BOLD_WHITE}5.2{constants.R}" not in rendered


def test_requested_loss_and_gradient_spacing_is_exact() -> None:
    line = "loss  =   4.7168  grad norm=   0.269"
    line = line.replace("loss  =", "loss=")
    line = re.sub(r"grad norm=\s+", "grad norm= ", line)
    assert line == "loss=   4.7168  grad norm= 0.269"
# ^^^ THOG
