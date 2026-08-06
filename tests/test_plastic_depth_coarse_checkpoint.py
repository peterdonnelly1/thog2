from __future__ import annotations

import copy

import pytest

from sheet.plastic_depth_coarse import PlasticCoarseTrialResult
from sheet.plastic_depth_coarse_checkpoint import (
    PLASTIC_COARSE_TRIAL_CHECKPOINT_PHASE,
    PlasticCoarseTrialCheckpointState,
    build_plastic_coarse_trial_checkpoint_state,
)


def _completed_result() -> PlasticCoarseTrialResult:
    return PlasticCoarseTrialResult(
        trial_index=1,
        layers=2,
        status="success",
        validation_losses=(3.1, 3.2),
        training_losses=(4.0, 3.8, 3.6, 3.4),
        training_elapsed_seconds=5.0,
        training_steps=4,
        tokens_per_update=128,
        peak_allocated_gib=1.0,
        peak_reserved_gib=1.5,
    )


def _state() -> PlasticCoarseTrialCheckpointState:
    return build_plastic_coarse_trial_checkpoint_state(
        candidate_layers=(2, 4, 8),
        n_steps=4,
        evaluation_steps_count=2,
        objective="lowest_loss",
        maximum_layers=8,
        cost_weight=0.0,
        memory_budget_gib=None,
        geometry_initialisation="equidistant",
        fine_max_updates=100,
        current_trial_index=2,
        current_trial_layers=4,
        completed_steps=2,
        training_losses=(4.2, 3.9),
        training_elapsed_seconds=2.5,
        completed_trial_results=(_completed_result(),),
    )


def test_mid_trial_checkpoint_round_trip_is_exact() -> None:
    state = _state()
    payload = state.structured()

    assert payload["phase"] == PLASTIC_COARSE_TRIAL_CHECKPOINT_PHASE
    assert PlasticCoarseTrialCheckpointState.from_mapping(payload) == state


def test_mid_trial_checkpoint_rejects_non_contiguous_completed_trials() -> None:
    result = _completed_result()
    wrong_index = PlasticCoarseTrialResult(
        trial_index=2,
        layers=result.layers,
        status=result.status,
        validation_losses=result.validation_losses,
        training_losses=result.training_losses,
        training_elapsed_seconds=result.training_elapsed_seconds,
        training_steps=result.training_steps,
        tokens_per_update=result.tokens_per_update,
    )

    with pytest.raises(ValueError, match="contiguous prefix"):
        build_plastic_coarse_trial_checkpoint_state(
            candidate_layers=(2, 4),
            n_steps=4,
            evaluation_steps_count=2,
            objective="lowest_loss",
            maximum_layers=4,
            cost_weight=0.0,
            memory_budget_gib=None,
            geometry_initialisation="equidistant",
            fine_max_updates=100,
            current_trial_index=2,
            current_trial_layers=4,
            completed_steps=1,
            training_losses=(4.0,),
            training_elapsed_seconds=1.0,
            completed_trial_results=(wrong_index,),
        )


def test_mid_trial_checkpoint_rejects_loss_history_length_mismatch() -> None:
    payload = copy.deepcopy(dict(_state().structured()))
    payload["current_trial"]["training_losses"] = [4.2]

    with pytest.raises(ValueError, match="loss history"):
        PlasticCoarseTrialCheckpointState.from_mapping(payload)


def test_mid_trial_checkpoint_rejects_candidate_schedule_mismatch() -> None:
    payload = copy.deepcopy(dict(_state().structured()))
    payload["current_trial"]["layers"] = 8

    with pytest.raises(ValueError, match="candidate schedule"):
        PlasticCoarseTrialCheckpointState.from_mapping(payload)


def test_non_coarse_checkpoint_phase_is_rejected() -> None:
    payload = dict(_state().structured())
    payload["phase"] = "fine_start"

    with pytest.raises(ValueError, match="not a PLASTIC mid-COARSE"):
        PlasticCoarseTrialCheckpointState.from_mapping(payload)
