from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from sheet.plastic_depth_coarse_runner import run_fixed_plastic_coarse_trial
from sheet.plastic_depth_fresh_state import (
    PlasticFreshTrainingState,
    build_fresh_training_state,
    destroy_fresh_training_state,
)
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import stage3_config, token_splits


def _config():
    return stage3_config(
        "thog2_sheet",
        geometry_preset="depth",
        basis_family="chebyshev",
        depth_order=2,
        max_updates=4,
        eval_interval=0,
        checkpoint_interval=2,
        plastic__enabled=True,
        plastic__runtime_phase="coarse",
        plastic__coarse_phase="disabled",
        plastic__layers_to_sample=2,
        plastic__do_learn_layer_count=False,
        plastic__initial_layer_count=None,
        plastic__max_permitted_layers=None,
        plastic__layer_count_update_brake=5,
        plastic__layer_count_probe_interval=5,
    )


def _clock(*values: float):
    iterator = iter(values)
    return lambda: next(iterator)


def _assert_nested_equal(left: Any, right: Any) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right)
        assert left.dtype == right.dtype
        assert tuple(left.shape) == tuple(right.shape)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_equal(left_value, right_value)
        return
    assert left == right


def _snapshot(trainer) -> dict[str, Any]:
    return {
        "model": copy.deepcopy(trainer.raw_model.state_dict()),
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "batch_source": copy.deepcopy(trainer.batch_source.state_dict()),
        "trainer_state": copy.deepcopy(vars(trainer.state)),
    }


def test_interrupted_checkpoint_resume_matches_uninterrupted_trial_exactly(tmp_path) -> None:
    config = _config()
    train_tokens, validation_tokens = token_splits()

    uninterrupted_state = build_fresh_training_state(
        trainer_factory=SharedTrainer,
        resolved_config=config,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        phase="coarse",
        active_layer_count=2,
        instrumentation_namespace="coarse/uninterrupted",
    )
    try:
        uninterrupted_result = run_fixed_plastic_coarse_trial(
            uninterrupted_state,
            trial_index=1,
            n_steps=4,
            evaluation_steps_count=2,
            clock=_clock(0.0, 4.0),
        )
        uninterrupted_snapshot = _snapshot(uninterrupted_state.trainer)
    finally:
        destroy_fresh_training_state(uninterrupted_state)

    checkpoint_path = tmp_path / "coarse_step_2.pt"
    interrupted_state = build_fresh_training_state(
        trainer_factory=SharedTrainer,
        resolved_config=config,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        phase="coarse",
        active_layer_count=2,
        instrumentation_namespace="coarse/interrupted",
    )
    captured_progress = None

    def checkpoint_and_interrupt(progress, trainer) -> None:
        nonlocal captured_progress
        captured_progress = progress
        trainer.plastic_coarse_fine_state = {
            "phase": "coarse_trial",
            "completed_steps": progress.completed_steps,
        }
        trainer.save_checkpoint(checkpoint_path)
        raise KeyboardInterrupt

    try:
        with pytest.raises(KeyboardInterrupt):
            run_fixed_plastic_coarse_trial(
                interrupted_state,
                trial_index=1,
                n_steps=4,
                evaluation_steps_count=2,
                clock=_clock(0.0, 2.0),
                checkpoint_interval=2,
                checkpoint_callback=checkpoint_and_interrupt,
            )
    finally:
        destroy_fresh_training_state(interrupted_state)

    assert captured_progress is not None
    assert captured_progress.completed_steps == 2
    assert captured_progress.training_elapsed_seconds == 2.0
    assert len(captured_progress.training_losses) == 2

    resumed_trainer = SharedTrainer.from_checkpoint(
        checkpoint_path,
        train_tokens,
        validation_tokens,
        expected_config=config,
    )
    resumed_state = PlasticFreshTrainingState(
        trainer=resumed_trainer,
        phase="coarse",
        active_layer_count=2,
        instrumentation_namespace="coarse/resumed",
        fingerprint={},
    )
    try:
        resumed_result = run_fixed_plastic_coarse_trial(
            resumed_state,
            trial_index=1,
            n_steps=4,
            evaluation_steps_count=2,
            clock=_clock(0.0, 2.0),
            prior_training_losses=captured_progress.training_losses,
            prior_training_elapsed_seconds=(
                captured_progress.training_elapsed_seconds
            ),
        )
        resumed_snapshot = _snapshot(resumed_trainer)
    finally:
        destroy_fresh_training_state(resumed_state)

    assert resumed_result == uninterrupted_result
    _assert_nested_equal(resumed_snapshot, uninterrupted_snapshot)
