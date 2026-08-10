from __future__ import annotations

from sheet.plastic_depth_lookahead_patch import (
    _lookahead_counts,
    choose_plastic_depth_count_with_exact_radius,
)


def _score(count: int, score: float):
    return {
        "active_layers": count,
        "feasible": True,
        "score": score,
    }


def test_full_radius_enumerates_every_integer() -> None:
    decision, execution = _lookahead_counts(31, 64, 2, 1)

    assert decision == (29, 30, 31, 32, 33)
    assert execution == decision


def test_full_radius_clips_at_capacity_boundaries() -> None:
    assert _lookahead_counts(1, 5, 3, 1)[0] == (1, 2, 3, 4)
    assert _lookahead_counts(5, 5, 3, 1)[0] == (2, 3, 4, 5)
    assert _lookahead_counts(3, 5, 99, 1)[0] == (1, 2, 3, 4, 5)


def test_distant_winner_is_committed_only_to_max_step() -> None:
    decision = choose_plastic_depth_count_with_exact_radius(
        current_count=31,
        score_report=(_score(29, 8.0), _score(30, 9.0), _score(31, 10.0), _score(32, 11.0), _score(33, 12.0)),
        histories={
            "31:-2": (-2.0, -2.0, -2.0, -2.0),
            "31:@LRA": (-1.0, -1.0, -1.0, -1.0),
        },
        noise_window=5,
        noise_lambda=3.0,
        update_number=20,
        last_count_change_update=0,
        update_brake=5,
        max_step=1,
    )

    assert decision.selected_count == 30
    assert decision.histories == {}
    winner = next(item for item in decision.evidence if item.candidate_count == 29)
    assert winner.significant


def test_latest_unfavourable_evidence_blocks_old_favourable_history() -> None:
    decision = choose_plastic_depth_count_with_exact_radius(
        current_count=10,
        score_report=(_score(8, 10.5), _score(9, 10.0), _score(10, 10.0), _score(11, 10.5), _score(12, 10.5)),
        histories={
            "10:-2": (-2.0, -2.0, -2.0, -2.0, -2.0),
            "10:@LRA": (-1.0, -1.0, -1.0, -1.0, -1.0),
        },
        noise_window=6,
        noise_lambda=0.1,
        update_number=20,
        last_count_change_update=0,
        update_brake=5,
        max_step=2,
    )

    assert decision.selected_count == 10
    evidence = next(item for item in decision.evidence if item.candidate_count == 8)
    assert evidence.paired_difference == 0.5
    assert not evidence.significant


def test_exact_z_tie_prefers_fewer_layers_within_permitted_side() -> None:
    decision = choose_plastic_depth_count_with_exact_radius(
        current_count=10,
        score_report=(_score(8, 9.0), _score(9, 9.0), _score(10, 10.0), _score(11, 11.0), _score(12, 11.0)),
        histories={
            "10:-2": (-1.0, -1.0, -1.0, -1.0),
            "10:-1": (-1.0, -1.0, -1.0, -1.0),
            "10:@LRA": (-1.0, -1.0, -1.0, -1.0),
        },
        noise_window=5,
        noise_lambda=0.1,
        update_number=20,
        last_count_change_update=0,
        update_brake=5,
        max_step=1,
    )

    assert decision.selected_count == 9
