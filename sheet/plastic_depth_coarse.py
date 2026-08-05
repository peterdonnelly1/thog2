from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


PLASTIC_COARSE_PHASES = ("enabled", "disabled")


@dataclass(frozen=True)
class ResolvedPlasticCoarseConfig:
    enabled: bool
    candidate_layers: Tuple[int, ...]
    n_steps: Optional[int]
    evaluation_steps_count: Optional[int]


def _optional_positive_integer(name: str, value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None; got {value!r}")
    return value


def validate_plastic_coarse_phase(value: str) -> str:
    if value not in PLASTIC_COARSE_PHASES:
        raise ValueError(
            "plastic__coarse_phase must be one of "
            f"{PLASTIC_COARSE_PHASES}; got {value!r}"
        )
    return value


def resolve_plastic_coarse_config(
    *,
    coarse_phase: str,
    plastic_enabled: bool,
    do_learn_layer_count: bool,
    n_steps: Optional[int],
    starting_layer_count: Optional[int],
    number_of_trials: Optional[int],
    evaluation_steps_count: Optional[int],
    max_permitted_layers: Optional[int],
) -> ResolvedPlasticCoarseConfig:
    phase = validate_plastic_coarse_phase(coarse_phase)
    resolved_n_steps = _optional_positive_integer(
        "plastic__phase_1_n_steps",
        n_steps,
    )
    resolved_start = _optional_positive_integer(
        "plastic__phase_1_starting_layer_count",
        starting_layer_count,
    )
    resolved_trials = _optional_positive_integer(
        "plastic__phase_1__number_of_trials",
        number_of_trials,
    )
    resolved_evaluation_steps = _optional_positive_integer(
        "plastic__phase_1_evaluation_steps_count",
        evaluation_steps_count,
    )

    if phase == "disabled":
        return ResolvedPlasticCoarseConfig(
            enabled=False,
            candidate_layers=(),
            n_steps=resolved_n_steps,
            evaluation_steps_count=resolved_evaluation_steps,
        )

    if not plastic_enabled:
        raise ValueError("plastic__coarse_phase=enabled requires plastic__enabled=true")
    if not do_learn_layer_count:
        raise ValueError(
            "plastic__coarse_phase=enabled requires "
            "plastic__do_learn_layer_count=true"
        )

    missing = [
        name
        for name, value in (
            ("plastic__phase_1_n_steps", resolved_n_steps),
            ("plastic__phase_1_starting_layer_count", resolved_start),
            ("plastic__phase_1__number_of_trials", resolved_trials),
            (
                "plastic__phase_1_evaluation_steps_count",
                resolved_evaluation_steps,
            ),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "plastic__coarse_phase=enabled is missing required controls: "
            + ", ".join(missing)
        )
    if max_permitted_layers is None:
        raise ValueError(
            "plastic__coarse_phase=enabled requires "
            "plastic__max_permitted_layers"
        )

    assert resolved_start is not None
    assert resolved_trials is not None
    candidates = tuple(resolved_start * (2**index) for index in range(resolved_trials))
    outside = tuple(value for value in candidates if value > max_permitted_layers)
    if outside:
        raise ValueError(
            "PLASTIC COARSE candidate layers exceed "
            "plastic__max_permitted_layers: "
            f"candidates={candidates}, maximum={max_permitted_layers}"
        )

    return ResolvedPlasticCoarseConfig(
        enabled=True,
        candidate_layers=candidates,
        n_steps=resolved_n_steps,
        evaluation_steps_count=resolved_evaluation_steps,
    )


def resolve_plastic_probe_interval(
    *,
    probe_interval: Optional[int],
    update_brake: int,
    enabled: bool,
    do_learn_layer_count: bool,
) -> Optional[int]:
    if probe_interval is not None:
        return _optional_positive_integer(
            "plastic__layer_count_probe_interval",
            probe_interval,
        )
    if not enabled or not do_learn_layer_count:
        return None
    if update_brake < 1:
        raise ValueError(
            "plastic__layer_count_probe_interval must be supplied when "
            "plastic__layer_count_update_brake is zero"
        )
    return update_brake


def validate_plastic_fine_count_controls(
    *,
    probe_radius: int,
    max_step: int,
) -> None:
    _optional_positive_integer("plastic__layer_count_probe_radius", probe_radius)
    _optional_positive_integer("plastic__layer_count_max_step", max_step)
