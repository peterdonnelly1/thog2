from __future__ import annotations

import pytest

from sheet.plastic_depth_coarse import (
    resolve_plastic_coarse_config,
    resolve_plastic_probe_interval,
    validate_plastic_coarse_phase,
    validate_plastic_fine_count_controls,
)


def test_coarse_disabled_has_no_candidate_schedule() -> None:
    resolved = resolve_plastic_coarse_config(
        coarse_phase="disabled",
        plastic_enabled=True,
        do_learn_layer_count=True,
        n_steps=None,
        starting_layer_count=None,
        number_of_trials=None,
        evaluation_steps_count=None,
        max_permitted_layers=64,
    )

    assert not resolved.enabled
    assert resolved.candidate_layers == ()


def test_coarse_doubling_schedule_is_exact() -> None:
    resolved = resolve_plastic_coarse_config(
        coarse_phase="enabled",
        plastic_enabled=True,
        do_learn_layer_count=True,
        n_steps=500,
        starting_layer_count=4,
        number_of_trials=4,
        evaluation_steps_count=10,
        max_permitted_layers=64,
    )

    assert resolved.enabled
    assert resolved.candidate_layers == (4, 8, 16, 32)
    assert resolved.n_steps == 500
    assert resolved.evaluation_steps_count == 10


@pytest.mark.parametrize("value", ["", "true", "false", "coarse", None])
def test_coarse_phase_accepts_only_enabled_or_disabled(value: object) -> None:
    with pytest.raises(ValueError, match="plastic__coarse_phase"):
        validate_plastic_coarse_phase(value)  # type: ignore[arg-type]


def test_coarse_requires_plastic_and_learned_count() -> None:
    with pytest.raises(ValueError, match="plastic__enabled=true"):
        resolve_plastic_coarse_config(
            coarse_phase="enabled",
            plastic_enabled=False,
            do_learn_layer_count=True,
            n_steps=2,
            starting_layer_count=2,
            number_of_trials=2,
            evaluation_steps_count=2,
            max_permitted_layers=4,
        )

    with pytest.raises(ValueError, match="plastic__do_learn_layer_count=true"):
        resolve_plastic_coarse_config(
            coarse_phase="enabled",
            plastic_enabled=True,
            do_learn_layer_count=False,
            n_steps=2,
            starting_layer_count=2,
            number_of_trials=2,
            evaluation_steps_count=2,
            max_permitted_layers=4,
        )


def test_coarse_rejects_missing_and_non_positive_controls() -> None:
    with pytest.raises(ValueError, match="missing required controls"):
        resolve_plastic_coarse_config(
            coarse_phase="enabled",
            plastic_enabled=True,
            do_learn_layer_count=True,
            n_steps=None,
            starting_layer_count=2,
            number_of_trials=2,
            evaluation_steps_count=2,
            max_permitted_layers=4,
        )

    with pytest.raises(ValueError, match="plastic__phase_1__number_of_trials"):
        resolve_plastic_coarse_config(
            coarse_phase="disabled",
            plastic_enabled=True,
            do_learn_layer_count=True,
            n_steps=None,
            starting_layer_count=None,
            number_of_trials=0,
            evaluation_steps_count=None,
            max_permitted_layers=4,
        )


def test_coarse_rejects_candidate_above_capacity() -> None:
    with pytest.raises(ValueError, match="candidate layers exceed"):
        resolve_plastic_coarse_config(
            coarse_phase="enabled",
            plastic_enabled=True,
            do_learn_layer_count=True,
            n_steps=2,
            starting_layer_count=4,
            number_of_trials=3,
            evaluation_steps_count=2,
            max_permitted_layers=8,
        )


def test_probe_interval_defaults_to_update_brake_only_for_learned_count() -> None:
    assert resolve_plastic_probe_interval(
        probe_interval=None,
        update_brake=7,
        enabled=True,
        do_learn_layer_count=True,
    ) == 7
    assert resolve_plastic_probe_interval(
        probe_interval=None,
        update_brake=0,
        enabled=False,
        do_learn_layer_count=False,
    ) is None


def test_probe_interval_preserves_legacy_every_update_when_brake_is_zero() -> None:
    assert resolve_plastic_probe_interval(
        probe_interval=None,
        update_brake=0,
        enabled=True,
        do_learn_layer_count=True,
    ) == 1


def test_fine_radius_and_max_step_are_positive() -> None:
    validate_plastic_fine_count_controls(probe_radius=2, max_step=1)
    with pytest.raises(ValueError, match="probe_radius"):
        validate_plastic_fine_count_controls(probe_radius=0, max_step=1)
    with pytest.raises(ValueError, match="max_step"):
        validate_plastic_fine_count_controls(probe_radius=1, max_step=0)
