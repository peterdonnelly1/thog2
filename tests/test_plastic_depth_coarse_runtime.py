from __future__ import annotations

import torch

from sheet.plastic_depth_fresh_state import (
    build_fresh_training_state,
    destroy_fresh_training_state,
)
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import stage3_config, token_splits


def _coarse_training_config():
    return stage3_config(
        "thog2_sheet",
        geometry_preset="depth",
        basis_family="chebyshev",
        depth_order=3,
        n_layer=4,
        max_updates=2,
        plastic__enabled=True,
        plastic__runtime_phase="coarse",
        plastic__coarse_phase="disabled",
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=2,
        plastic__max_permitted_layers=4,
        plastic__layer_count_update_brake=0,
        plastic__layer_count_probe_window_size=1,
        plastic__freeze_geometry_during_warmup=False,
    )


def test_coarse_update_keeps_count_geometry_and_controller_state_fixed() -> None:
    train_tokens, validation_tokens = token_splits()
    state = build_fresh_training_state(
        trainer_factory=SharedTrainer,
        resolved_config=_coarse_training_config(),
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        phase="coarse",
        active_layer_count=2,
        instrumentation_namespace="coarse/trial_1",
    )
    trainer = state.trainer
    lattice = trainer.raw_model.trajectory.plastic_sampling
    before_coordinates = lattice.public_coordinates().detach().clone()
    before_count_decisions = int(lattice.count_decision_number.item())
    before_timing_observations = lattice.training_time_observations.detach().clone()

    try:
        assert lattice.current_active_layers == 2
        assert all(not parameter.requires_grad for parameter in lattice.parameters())

        metrics = trainer.train_one_update()

        assert metrics["skipped_update"] == 0.0
        assert trainer.state.completed_updates == 1
        assert lattice.current_active_layers == 2
        torch.testing.assert_close(
            lattice.public_coordinates(),
            before_coordinates,
            rtol=0.0,
            atol=0.0,
        )
        assert int(lattice.count_decision_number.item()) == before_count_decisions
        torch.testing.assert_close(
            lattice.training_time_observations,
            before_timing_observations,
            rtol=0.0,
            atol=0.0,
        )
    finally:
        destroy_fresh_training_state(state)
