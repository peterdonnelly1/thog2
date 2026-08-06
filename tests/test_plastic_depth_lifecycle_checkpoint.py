from __future__ import annotations

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
        plastic__enabled=True,
        plastic__runtime_phase="fine",
        plastic__coarse_phase="disabled",
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=2,
        plastic__max_permitted_layers=4,
        plastic__layer_count_update_brake=5,
        plastic__layer_count_probe_window_size=5,
    )


def test_coarse_fine_state_round_trips_without_advancing_fine(tmp_path) -> None:
    train_tokens, validation_tokens = token_splits()
    config = _config()
    checkpoint_path = tmp_path / "review_pause.pt"
    state = {
        "phase": "review_pause",
        "selected_layers": 2,
        "candidate_layers": [2, 4],
        "trials": [
            {
                "trial_index": 1,
                "layers": 2,
                "status": "success",
                "mean_validation_loss": 3.0,
            }
        ],
        "pause": {
            "disposition": "checkpoint_exit",
            "elapsed_seconds": 120.0,
            "remaining_seconds": 780.0,
        },
    }

    trainer = SharedTrainer(config, train_tokens, validation_tokens)
    try:
        trainer.plastic_coarse_fine_state = state
        trainer.plastic_coarse_provenance = state
        trainer.save_checkpoint(checkpoint_path)
    finally:
        trainer.close()

    resumed = SharedTrainer.from_checkpoint(
        checkpoint_path,
        train_tokens,
        validation_tokens,
        expected_config=config,
    )
    try:
        assert resumed.state.completed_updates == 0
        assert resumed.plastic_coarse_fine_state == state
        assert resumed.plastic_coarse_provenance == state
        assert resumed.config.plastic__runtime_phase == "fine"
        assert resumed.config.plastic__coarse_phase == "disabled"
    finally:
        resumed.close()
