from __future__ import annotations

import io
from dataclasses import dataclass
from types import SimpleNamespace

from sheet.plastic_depth_coarse import (
    PlasticCoarseTrialResult,
    ResolvedPlasticCoarseConfig,
)
from sheet.plastic_depth_coarse_checkpoint import (
    PLASTIC_COARSE_TRIAL_CHECKPOINT_PHASE,
    build_plastic_coarse_trial_checkpoint_state,
)
from sheet.plastic_depth_coarse_runner import PlasticCoarseTrialProgress
from sheet.plastic_depth_fresh_state import PlasticFreshTrainingState
from sheet.plastic_depth_lifecycle import run_plastic_coarse_fine_lifecycle
from sheet.plastic_depth_pause import PlasticCoarsePauseResult


@dataclass(frozen=True)
class _Config:
    device: str = "cpu"
    plastic__coarse_phase: str = "enabled"
    plastic__runtime_phase: str = "fine"
    plastic__initial_layer_count: int = 2
    max_updates: int = 100


class _Coordinator:
    def __init__(self) -> None:
        self.is_primary = True
        self.closed = False
        self.barriers = 0

    def assert_identical_object(self, value, description: str) -> None:
        del value, description

    def all_gather_object(self, value):
        return [value]

    def barrier(self) -> None:
        self.barriers += 1

    def close(self) -> None:
        self.closed = True


def _result(
    trial_index: int,
    layers: int,
    validation_loss: float,
) -> PlasticCoarseTrialResult:
    return PlasticCoarseTrialResult(
        trial_index=trial_index,
        layers=layers,
        status="success",
        validation_losses=(validation_loss,),
        training_losses=(4.0, 3.8, 3.6),
        training_elapsed_seconds=float(trial_index),
        training_steps=3,
        tokens_per_update=128,
    )


def _state(config, *, phase: str, layers: int, completed_updates: int = 0):
    trainer = SimpleNamespace(
        config=config,
        state=SimpleNamespace(completed_updates=completed_updates),
    )
    return PlasticFreshTrainingState(
        trainer=trainer,
        phase=phase,
        active_layer_count=layers,
        instrumentation_namespace=("fine" if phase == "fine" else "coarse/resumed"),
        fingerprint={},
    )


def test_lifecycle_resumes_current_trial_skips_prefix_and_runs_suffix_fresh() -> None:
    coordinator = _Coordinator()
    completed = _result(1, 2, 3.4)
    resume_checkpoint = build_plastic_coarse_trial_checkpoint_state(
        candidate_layers=(2, 4, 8),
        n_steps=3,
        evaluation_steps_count=1,
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
        completed_trial_results=(completed,),
    )
    resumed_config = _Config(
        plastic__coarse_phase="disabled",
        plastic__runtime_phase="coarse",
        plastic__initial_layer_count=4,
        max_updates=3,
    )
    resumed_state = _state(
        resumed_config,
        phase="coarse",
        layers=4,
        completed_updates=2,
    )
    builds = []
    destroyed = []
    runner_calls = []

    def builder(**kwargs):
        builds.append((kwargs["phase"], kwargs["active_layer_count"]))
        return _state(
            kwargs["resolved_config"],
            phase=kwargs["phase"],
            layers=kwargs["active_layer_count"],
        )

    def runner(state, **kwargs):
        runner_calls.append((state, dict(kwargs)))
        if kwargs["trial_index"] == 2:
            assert state is resumed_state
            assert kwargs["prior_training_losses"] == (4.2, 3.9)
            assert kwargs["prior_training_elapsed_seconds"] == 2.5
            return PlasticCoarseTrialResult(
                trial_index=2,
                layers=4,
                status="success",
                validation_losses=(3.0,),
                training_losses=(4.2, 3.9, 3.6),
                training_elapsed_seconds=3.5,
                training_steps=3,
                tokens_per_update=128,
            )
        assert kwargs["trial_index"] == 3
        assert "prior_training_losses" not in kwargs
        return _result(3, 8, 3.2)

    def destroyer(state):
        destroyed.append((state.phase, state.active_layer_count))
        state.trainer = None

    outcome = run_plastic_coarse_fine_lifecycle(
        trainer_factory=lambda *_: None,
        resolved_config=_Config(),
        train_tokens=object(),
        validation_tokens=object(),
        coarse_config=ResolvedPlasticCoarseConfig(True, (2, 4, 8), 3, 1),
        objective="lowest_loss",
        maximum_layers=8,
        cost_weight=0.0,
        memory_budget_gib=None,
        geometry_initialisation="equidistant",
        console_stream=io.StringIO(),
        resume_checkpoint_state=resume_checkpoint.structured(),
        resume_state=resumed_state,
        distributed_coordinator=coordinator,
        fresh_state_builder=builder,
        trial_runner=runner,
        state_destroyer=destroyer,
        pause_runner=lambda **_: PlasticCoarsePauseResult("ctrl_f", 1.0, 899.0),
        coordinator_factory=lambda _: (_ for _ in ()).throw(
            AssertionError("provided coordinator must be reused")
        ),
    )

    assert [call[1]["trial_index"] for call in runner_calls] == [2, 3]
    assert builds == [("coarse", 8), ("fine", 4)]
    assert destroyed == [("coarse", 4), ("coarse", 8)]
    assert [result.trial_index for result in outcome.trial_results] == [1, 2, 3]
    assert outcome.selected_layers == 4
    assert outcome.fine_state is not None
    outcome.close_coordinator()
    assert coordinator.closed


def test_periodic_coarse_checkpoint_contains_replayable_lifecycle_state() -> None:
    coordinator = _Coordinator()
    checkpoints = []

    def builder(**kwargs):
        return _state(
            kwargs["resolved_config"],
            phase=kwargs["phase"],
            layers=kwargs["active_layer_count"],
        )

    def runner(state, **kwargs):
        progress = PlasticCoarseTrialProgress(
            trial_index=1,
            layers=2,
            completed_steps=1,
            n_steps=3,
            training_losses=(4.0,),
            training_elapsed_seconds=1.25,
        )
        kwargs["checkpoint_callback"](progress, state.trainer)
        return _result(1, 2, 3.0)

    def checkpoint_callback(trainer, payload):
        checkpoints.append((trainer, dict(payload)))

    outcome = run_plastic_coarse_fine_lifecycle(
        trainer_factory=lambda *_: None,
        resolved_config=_Config(),
        train_tokens=object(),
        validation_tokens=object(),
        coarse_config=ResolvedPlasticCoarseConfig(True, (2,), 3, 1),
        objective="lowest_loss",
        maximum_layers=2,
        cost_weight=0.0,
        memory_budget_gib=None,
        geometry_initialisation="equidistant",
        console_stream=io.StringIO(),
        checkpoint_callback=checkpoint_callback,
        coarse_checkpoint_interval=1,
        distributed_coordinator=coordinator,
        fresh_state_builder=builder,
        trial_runner=runner,
        state_destroyer=lambda state: setattr(state, "trainer", None),
        pause_runner=lambda **_: PlasticCoarsePauseResult("ctrl_f", 1.0, 899.0),
    )

    assert len(checkpoints) == 1
    trainer, payload = checkpoints[0]
    assert payload["phase"] == PLASTIC_COARSE_TRIAL_CHECKPOINT_PHASE
    assert payload["candidate_layers"] == [2]
    assert payload["current_trial"]["completed_steps"] == 1
    assert payload["current_trial"]["training_losses"] == [4.0]
    assert payload["fine_max_updates"] == 100
    assert trainer.plastic_coarse_fine_state == payload
    assert outcome.selected_layers == 2
    outcome.close_coordinator()
