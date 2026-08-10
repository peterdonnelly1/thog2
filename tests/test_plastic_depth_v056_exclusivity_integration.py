from __future__ import annotations

import pytest

from sheet import plastic_depth_directional_coherence_patch as directional
from sheet import plastic_depth_theil_sen_kendall_patch as tsk
from sheet import plastic_depth_v056_objective_decision_patch as v056
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import token_splits
from tests.test_plastic_depth import plastic_training_config


def test_real_tsk_probe_never_executes_legacy_robust_z_machinery(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v056.STRATIFIED_ALGORITHM)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy robust-z/directional machinery executed under TSK")

    monkeypatch.setattr(directional, "_robust_scale", forbidden)
    monkeypatch.setattr(directional, "_directional_support", forbidden)

    train_tokens, validation_tokens = token_splits(length=1024)
    config = plastic_training_config(
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=3,
        plastic__max_permitted_layers=5,
        plastic__layer_count_objective="lowest_loss",
        plastic__layer_count_probe__probe_every_n_steps=1,
        plastic__layer_count_probe__window_size_as_number_of_probes=1,
        plastic__layer_count_probe_noise_lambda=0.0,
        plastic__layer_count_probe_radius=1,
        plastic__layer_count__max_allowable_layer_change=1,
        plastic__layer_count_update_brake=0,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=16,
        gradient_accumulation_steps=1,
        batch_size=2,
        block_size=8,
        max_updates=2,
        warmup_updates=0,
        checkpoint_segment_size=0,
        device="cpu",
        dtype="float32",
    )
    trainer = SharedTrainer(config, train_tokens, validation_tokens)
    try:
        trainer.train_one_update()
        count_events = [event for event in trainer.events if event.name == "plastic_depth_count_decision"]
        assert count_events
        decision_event = count_events[-1]
        assert all(
            item.get("standardized_improvement") is None
            and item.get("median") is None
            and item.get("mad") is None
            and item.get("sigma") is None
            for item in decision_event.payload.get("score_evidence", ())
        )
        assert not any(
            event.name == "plastic_depth_directional_decision"
            for event in trainer.events
        )
        tsk_events = [
            event for event in trainer.events
            if event.name == "plastic_depth_v055_sen_kendall_decision"
        ]
        assert tsk_events
        assert tsk_events[-1].payload["objective"] == "lowest_loss"
        assert tsk_events[-1].payload["score_units"] == "loss"
    finally:
        trainer.close()
