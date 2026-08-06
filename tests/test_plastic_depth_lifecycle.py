from __future__ import annotations

import io
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from sheet.plastic_depth_coarse import (
    PlasticCoarseTrialResult,
    ResolvedPlasticCoarseConfig,
)
from sheet.plastic_depth_fresh_state import PlasticFreshTrainingState
from sheet.plastic_depth_lifecycle import run_plastic_coarse_fine_lifecycle
from sheet.plastic_depth_pause import PlasticCoarsePauseResult


@dataclass(frozen=True)
class _Config:
    device: str = "cpu"
    plastic__coarse_phase: str = "enabled"
    plastic__runtime_phase: str = "fine"
    plastic__initial_layer_count: int = 2
    plastic__coarse_phase_roll_through: bool = False
    plastic__coarse_phase_roll_through: bool = False
    max_updates: int = 100


class _Coordinator:
    def __init__(self) -> None:
        self.is_primary = True
        self.closed = False
        self.identities = []
        self.barriers = 0

    def all_gather_object(self, value):
        return [value]

    def assert_identical_object(self, value, description: str) -> None:
        self.identities.append((value, description))

    def barrier(self) -> None:
        self.barriers += 1

    def close(self) -> None:
        self.closed = True


def _coarse_config() -> ResolvedPlasticCoarseConfig:
    return ResolvedPlasticCoarseConfig(
        enabled=True,
        candidate_layers=(2, 4, 8),
        n_steps=3,
        evaluation_steps_count=2,
    )


def test_coarse_trials_are_destroyed_and_fine_is_fresh_at_measured_winner() -> None:
    coordinator = _Coordinator()
    builds = []
    destroyed = []
    losses = {2: 4.0, 4: 3.0, 8: 3.5}

    def builder(**kwargs):
        builds.append(kwargs)
        trainer = SimpleNamespace(config=kwargs["resolved_config"])
        return PlasticFreshTrainingState(
            trainer=trainer,
            phase=kwargs["phase"],
            active_layer_count=kwargs["active_layer_count"],
            instrumentation_namespace=kwargs["instrumentation_namespace"],
            fingerprint={},
        )

    def runner(state, *, trial_index: int, n_steps: int, evaluation_steps_count: int, progress_sink):
        assert state.phase == "coarse"
        assert n_steps == 3
        assert evaluation_steps_count == 2
        assert state.trainer.config.max_updates == 3
        if progress_sink is not None:
            progress_sink(f"C synthetic {trial_index}")
        loss = losses[state.active_layer_count]
        return PlasticCoarseTrialResult(
            trial_index=trial_index,
            layers=state.active_layer_count,
            status="success",
            validation_losses=(loss - 0.1, loss + 0.1),
            training_elapsed_seconds=float(trial_index),
            training_steps=3,
            tokens_per_update=100,
            peak_allocated_gib=1.0,
            peak_reserved_gib=1.5,
        )

    def destroyer(state):
        destroyed.append((state.phase, state.active_layer_count))
        state.trainer = None

    outcome = run_plastic_coarse_fine_lifecycle(
        trainer_factory=lambda *_: None,
        resolved_config=_Config(),
        train_tokens=object(),
        validation_tokens=object(),
        coarse_config=_coarse_config(),
        objective="lowest_loss",
        maximum_layers=8,
        cost_weight=0.0,
        memory_budget_gib=None,
        geometry_initialisation="equidistant",
        pause_duration_seconds=900.0,
        console_stream=io.StringIO(),
        fresh_state_builder=builder,
        trial_runner=runner,
        state_destroyer=destroyer,
        pause_runner=lambda **_: PlasticCoarsePauseResult("ctrl_f", 7.0, 893.0),
        coordinator_factory=lambda _: coordinator,
    )

    assert [item["active_layer_count"] for item in builds] == [2, 4, 8, 4]
    assert [item["phase"] for item in builds] == ["coarse", "coarse", "coarse", "fine"]
    assert [item["instrumentation_namespace"] for item in builds] == [
        "coarse/trial_1",
        "coarse/trial_2",
        "coarse/trial_3",
        "fine",
    ]
    assert destroyed == [("coarse", 2), ("coarse", 4), ("coarse", 8)]
    assert outcome.selected_layers == 4
    assert outcome.fine_state is not None
    assert outcome.fine_state.trainer.config.max_updates == 100
    assert outcome.fine_state.trainer.config.plastic__coarse_phase == "disabled"
    assert outcome.provenance["selected_layers"] == 4
    assert outcome.provenance["phase"] == "fine_start"
    assert outcome.provenance["pause"]["disposition"] == "ctrl_f"
    assert coordinator.identities
    assert coordinator.barriers == 1
    assert not coordinator.closed
    outcome.close_coordinator()
    assert coordinator.closed


def test_checkpoint_exit_constructs_fresh_fine_checkpoint_collectively_then_discards_it() -> None:
    coordinator = _Coordinator()
    builds = []
    destroyed = []
    checkpoints = []

    def builder(**kwargs):
        builds.append(kwargs)
        return PlasticFreshTrainingState(
            trainer=SimpleNamespace(config=kwargs["resolved_config"]),
            phase=kwargs["phase"],
            active_layer_count=kwargs["active_layer_count"],
            instrumentation_namespace=kwargs["instrumentation_namespace"],
            fingerprint={},
        )

    def destroyer(state):
        destroyed.append((state.phase, state.active_layer_count))
        state.trainer = None

    def checkpoint_callback(trainer, payload):
        checkpoints.append((trainer, dict(payload)))

    outcome = run_plastic_coarse_fine_lifecycle(
        trainer_factory=lambda *_: None,
        resolved_config=_Config(),
        train_tokens=object(),
        validation_tokens=object(),
        coarse_config=ResolvedPlasticCoarseConfig(True, (2,), 1, 1),
        objective="lowest_loss",
        maximum_layers=8,
        cost_weight=0.0,
        memory_budget_gib=None,
        geometry_initialisation="equidistant",
        console_stream=io.StringIO(),
        checkpoint_callback=checkpoint_callback,
        fresh_state_builder=builder,
        trial_runner=lambda state, **_: PlasticCoarseTrialResult(
            trial_index=1,
            layers=state.active_layer_count,
            status="success",
            validation_losses=(3.0,),
            training_elapsed_seconds=1.0,
            training_steps=1,
            tokens_per_update=100,
        ),
        state_destroyer=destroyer,
        pause_runner=lambda **_: PlasticCoarsePauseResult("checkpoint_exit", 2.0, 898.0),
        coordinator_factory=lambda _: coordinator,
    )

    assert outcome.fine_state is None
    assert [item["phase"] for item in builds] == ["coarse", "fine"]
    assert destroyed == [("coarse", 2), ("fine", 2)]
    assert len(checkpoints) == 1
    checkpoint_trainer, checkpoint_payload = checkpoints[0]
    assert checkpoint_trainer.config.plastic__coarse_phase == "disabled"
    assert checkpoint_payload["phase"] == "review_pause"
    assert checkpoint_payload["pause"]["remaining_seconds"] == 898.0
    assert outcome.provenance["pause"]["disposition"] == "checkpoint_exit"
    outcome.close_coordinator()


def test_checkpoint_exit_without_collective_callback_fails_closed() -> None:
    coordinator = _Coordinator()

    def builder(**kwargs):
        return PlasticFreshTrainingState(
            trainer=SimpleNamespace(config=kwargs["resolved_config"]),
            phase=kwargs["phase"],
            active_layer_count=kwargs["active_layer_count"],
            instrumentation_namespace=kwargs["instrumentation_namespace"],
            fingerprint={},
        )

    with pytest.raises(RuntimeError, match="collective checkpoint callback"):
        run_plastic_coarse_fine_lifecycle(
            trainer_factory=lambda *_: None,
            resolved_config=_Config(),
            train_tokens=object(),
            validation_tokens=object(),
            coarse_config=ResolvedPlasticCoarseConfig(True, (2,), 1, 1),
            objective="lowest_loss",
            maximum_layers=8,
            cost_weight=0.0,
            memory_budget_gib=None,
            geometry_initialisation="equidistant",
            console_stream=io.StringIO(),
            fresh_state_builder=builder,
            trial_runner=lambda state, **_: PlasticCoarseTrialResult(
                trial_index=1,
                layers=state.active_layer_count,
                status="success",
                validation_losses=(3.0,),
                training_elapsed_seconds=1.0,
                training_steps=1,
                tokens_per_update=100,
            ),
            state_destroyer=lambda state: setattr(state, "trainer", None),
            pause_runner=lambda **_: PlasticCoarsePauseResult("checkpoint_exit", 2.0, 898.0),
            coordinator_factory=lambda _: coordinator,
        )

    assert coordinator.closed


def test_unselectable_trials_close_coordinator_and_do_not_build_fine() -> None:
    coordinator = _Coordinator()

    def builder(**kwargs):
        return PlasticFreshTrainingState(
            trainer=SimpleNamespace(config=kwargs["resolved_config"]),
            phase=kwargs["phase"],
            active_layer_count=kwargs["active_layer_count"],
            instrumentation_namespace=kwargs["instrumentation_namespace"],
            fingerprint={},
        )

    with pytest.raises(RuntimeError, match="all PLASTIC COARSE trials failed"):
        run_plastic_coarse_fine_lifecycle(
            trainer_factory=lambda *_: None,
            resolved_config=_Config(),
            train_tokens=object(),
            validation_tokens=object(),
            coarse_config=ResolvedPlasticCoarseConfig(True, (2,), 1, 1),
            objective="lowest_loss",
            maximum_layers=8,
            cost_weight=0.0,
            memory_budget_gib=None,
            geometry_initialisation="equidistant",
            console_stream=io.StringIO(),
            fresh_state_builder=builder,
            trial_runner=lambda state, **_: PlasticCoarseTrialResult(
                trial_index=1,
                layers=state.active_layer_count,
                status="failed",
                error_class="RuntimeError",
                error_message="synthetic",
            ),
            state_destroyer=lambda state: setattr(state, "trainer", None),
            pause_runner=lambda **_: pytest.fail("pause must not run"),
            coordinator_factory=lambda _: coordinator,
        )

    assert coordinator.closed


def test_roll_through_skips_pause_and_builds_fine_immediately() -> None:
    coordinator = _Coordinator()
    builds = []
    output = io.StringIO()

    def builder(**kwargs):
        builds.append(kwargs["phase"])
        return PlasticFreshTrainingState(
            trainer=SimpleNamespace(config=kwargs["resolved_config"]),
            phase=kwargs["phase"],
            active_layer_count=kwargs["active_layer_count"],
            instrumentation_namespace=kwargs["instrumentation_namespace"],
            fingerprint={},
        )

    outcome = run_plastic_coarse_fine_lifecycle(
        trainer_factory=lambda *_: None,
        resolved_config=_Config(plastic__coarse_phase_roll_through=True),
        train_tokens=object(),
        validation_tokens=object(),
        coarse_config=ResolvedPlasticCoarseConfig(True, (2,), 1, 1),
        objective="lowest_loss",
        maximum_layers=8,
        cost_weight=0.0,
        memory_budget_gib=None,
        geometry_initialisation="equidistant",
        console_stream=output,
        fresh_state_builder=builder,
        trial_runner=lambda state, **_: PlasticCoarseTrialResult(
            trial_index=1,
            layers=state.active_layer_count,
            status="success",
            validation_losses=(3.0,),
            training_elapsed_seconds=1.0,
            training_steps=1,
            tokens_per_update=100,
        ),
        state_destroyer=lambda state: setattr(state, "trainer", None),
        pause_runner=lambda **_: pytest.fail("roll-through must not invoke the pause runner"),
        coordinator_factory=lambda _: coordinator,
    )

    assert builds == ["coarse", "fine"]
    assert outcome.pause_result.disposition == "roll_through"
    assert outcome.provenance["pause"]["disposition"] == "roll_through"
    assert "starting FINE immediately" in output.getvalue()
    assert coordinator.barriers == 1
    outcome.close_coordinator()
