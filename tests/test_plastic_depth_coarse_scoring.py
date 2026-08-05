from __future__ import annotations

import re

import pytest

from sheet.plastic_depth_coarse import (
    PlasticCoarseTrialResult,
    coarse_results_payload,
    render_plastic_coarse_report,
    score_plastic_coarse_trials,
)


def _trial(
    trial_index: int,
    layers: int,
    mean_loss: float,
    elapsed: float,
    peak: float,
) -> PlasticCoarseTrialResult:
    return PlasticCoarseTrialResult(
        trial_index=trial_index,
        layers=layers,
        status="success",
        validation_losses=(mean_loss - 0.01, mean_loss + 0.01),
        training_elapsed_seconds=elapsed,
        training_steps=10,
        tokens_per_update=100,
        peak_allocated_gib=peak,
        peak_reserved_gib=peak + 0.5,
    )


def test_lowest_loss_and_exact_tie_prefer_fewer_layers() -> None:
    rows, winner = score_plastic_coarse_trials(
        (_trial(1, 4, 4.0, 10.0, 2.0), _trial(2, 8, 4.0, 12.0, 3.0)),
        objective="lowest_loss",
        maximum_layers=16,
        cost_weight=0.0,
        memory_budget_gib=None,
    )

    assert rows[0].score == pytest.approx(4.0)
    assert winner.result.layers == 4


def test_layer_efficiency_uses_count_fraction() -> None:
    rows, winner = score_plastic_coarse_trials(
        (_trial(1, 4, 4.0, 10.0, 2.0), _trial(2, 8, 3.9, 10.0, 3.0)),
        objective="layer_efficiency",
        maximum_layers=16,
        cost_weight=0.4,
        memory_budget_gib=None,
    )

    assert rows[0].score == pytest.approx(4.1)
    assert rows[1].score == pytest.approx(4.1)
    assert winner.result.layers == 4


def test_relative_wall_time_uses_first_successful_trial_reference() -> None:
    failed = PlasticCoarseTrialResult(
        trial_index=1,
        layers=2,
        status="failed",
        error_class="RuntimeError",
        error_message="synthetic",
    )
    rows, winner = score_plastic_coarse_trials(
        (failed, _trial(2, 4, 4.0, 10.0, 2.0), _trial(3, 8, 3.7, 20.0, 3.0)),
        objective="relative_training_wall_time",
        maximum_layers=16,
        cost_weight=0.2,
        memory_budget_gib=None,
    )

    assert rows[1].reference_training_elapsed_seconds == pytest.approx(10.0)
    assert rows[1].score == pytest.approx(4.2)
    assert rows[2].score == pytest.approx(4.1)
    assert winner.result.layers == 8


def test_memory_budget_uses_peak_allocated_and_reports_within_budget() -> None:
    rows, winner = score_plastic_coarse_trials(
        (_trial(1, 4, 4.1, 10.0, 2.0), _trial(2, 8, 3.8, 12.0, 5.0)),
        objective="memory_budget",
        maximum_layers=16,
        cost_weight=0.0,
        memory_budget_gib=4.0,
    )

    assert rows[0].within_budget is True
    assert rows[1].within_budget is False
    assert not rows[1].selectable
    assert winner.result.layers == 4


def test_all_failed_or_outside_budget_stops() -> None:
    failed = PlasticCoarseTrialResult(
        trial_index=1,
        layers=2,
        status="failed",
        error_class="RuntimeError",
        error_message="synthetic",
    )
    with pytest.raises(RuntimeError, match="all PLASTIC COARSE trials failed"):
        score_plastic_coarse_trials(
            (failed,),
            objective="lowest_loss",
            maximum_layers=4,
            cost_weight=0.0,
            memory_budget_gib=None,
        )
    with pytest.raises(RuntimeError, match="outside the memory budget"):
        score_plastic_coarse_trials(
            (_trial(1, 4, 4.0, 10.0, 5.0),),
            objective="memory_budget",
            maximum_layers=4,
            cost_weight=0.0,
            memory_budget_gib=4.0,
        )


def test_report_wording_heading_alignment_and_green_winner() -> None:
    rows, winner = score_plastic_coarse_trials(
        (_trial(1, 4, 4.0, 10.0, 2.0), _trial(2, 8, 3.8, 12.0, 3.0)),
        objective="lowest_loss",
        maximum_layers=8,
        cost_weight=0.0,
        memory_budget_gib=None,
    )
    report = render_plastic_coarse_report(
        rows,
        winner,
        training_steps=10,
        evaluation_steps_count=2,
        ansi=True,
    )
    plain = re.sub(r"\x1b\[[0-9;]*m", "", report)

    assert "2 trials x 10 training steps" in plain
    assert "validation mean over final 2 batches" in plain
    assert "loss_score" in plain
    assert "<<< WINNER" in plain
    winner_line = next(line for line in report.splitlines() if "<<< WINNER" in line)
    assert winner_line.startswith("\033[1;92m")
    assert winner_line.endswith("\033[0m")


def test_structured_payload_agrees_with_selector() -> None:
    rows, winner = score_plastic_coarse_trials(
        (_trial(1, 4, 4.0, 10.0, 2.0), _trial(2, 8, 3.8, 12.0, 3.0)),
        objective="lowest_loss",
        maximum_layers=8,
        cost_weight=0.0,
        memory_budget_gib=None,
    )
    payload = coarse_results_payload(rows, winner)

    assert payload["selected_layers"] == 8
    assert payload["objective_heading"] == "loss_score"
    assert len(payload["trials"]) == 2
