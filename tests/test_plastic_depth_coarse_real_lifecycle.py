from __future__ import annotations

import io

from sheet.plastic_depth_coarse import resolve_plastic_coarse_config
from sheet.plastic_depth_fresh_state import (
    build_fresh_training_state,
    destroy_fresh_training_state,
)
from sheet.plastic_depth_lifecycle import run_plastic_coarse_fine_lifecycle
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import stage3_config, token_splits


def _config():
    return stage3_config(
        "thog2_sheet",
        geometry_preset="depth",
        basis_family="chebyshev",
        depth_order=3,
        n_layer=4,
        max_updates=2,
        eval_interval=0,
        plastic__enabled=True,
        plastic__runtime_phase="fine",
        plastic__coarse_phase="enabled",
        plastic__phase_1_n_steps=1,
        plastic__phase_1_starting_layer_count=2,
        plastic__phase_1__number_of_trials=2,
        plastic__phase_1_evaluation_steps_count=1,
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=None,
        plastic__max_permitted_layers=4,
        plastic__layer_count_update_brake=5,
        plastic__layer_count_probe_interval=5,
    )


def test_real_trials_finish_then_fine_reconstructs_at_step_zero() -> None:
    config = _config()
    train_tokens, validation_tokens = token_splits()
    coarse = resolve_plastic_coarse_config(
        coarse_phase=config.plastic__coarse_phase,
        plastic_enabled=config.plastic__enabled,
        do_learn_layer_count=config.plastic__do_learn_layer_count,
        n_steps=config.plastic__phase_1_n_steps,
        starting_layer_count=config.plastic__phase_1_starting_layer_count,
        number_of_trials=config.plastic__phase_1__number_of_trials,
        evaluation_steps_count=config.plastic__phase_1_evaluation_steps_count,
        max_permitted_layers=config.plastic__max_permitted_layers,
    )

    outcome = run_plastic_coarse_fine_lifecycle(
        trainer_factory=SharedTrainer,
        resolved_config=config,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        coarse_config=coarse,
        objective="lowest_loss",
        maximum_layers=4,
        cost_weight=0.0,
        memory_budget_gib=None,
        geometry_initialisation="equidistant",
        pause_duration_seconds=0.0,
        console_stream=io.StringIO(),
    )
    assert outcome.fine_state is not None
    fine_state = outcome.fine_state
    direct_state = build_fresh_training_state(
        trainer_factory=SharedTrainer,
        resolved_config=config,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        phase="fine",
        active_layer_count=outcome.selected_layers,
        instrumentation_namespace="fine/direct_reference",
    )
    try:
        assert len(outcome.trial_results) == 2
        assert all(result.status == "success" for result in outcome.trial_results)
        assert fine_state.trainer.state.completed_updates == 0
        assert fine_state.trainer.config.plastic__coarse_phase == "disabled"
        assert fine_state.trainer.raw_model.trajectory.plastic_sampling.current_active_layers == outcome.selected_layers
        assert fine_state.fingerprint == direct_state.fingerprint
        assert outcome.provenance["phase"] == "fine_start"
    finally:
        destroy_fresh_training_state(fine_state)
        destroy_fresh_training_state(direct_state)
        outcome.close_coordinator()
