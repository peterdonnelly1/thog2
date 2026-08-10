# vvv THOG
"""Final COARSE/FINE PLASTIC FINE probe and decision semantics."""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from . import plastic_depth_controller as _controller
from . import plastic_depth_lookahead_patch as _lookahead
from . import trainer_step as _trainer_step


def _positive_integer(value: Any, *, name: str) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return resolved


def _full_radius_counts(
    current: int,
    maximum: int,
    radius: int,
    max_step: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    resolved_current = int(current)
    resolved_maximum = int(maximum)
    resolved_radius = _positive_integer(
        radius,
        name="plastic__layer_count_probe_radius",
    )
    _positive_integer(max_step, name="plastic__layer_count__max_allowable_layer_change")
    if resolved_current < 1 or resolved_current > resolved_maximum:
        raise ValueError(
            "current PLASTIC layer count must lie within capacity; "
            f"current={resolved_current}, maximum={resolved_maximum}"
        )
    lower = max(1, resolved_current - resolved_radius)
    upper = min(resolved_maximum, resolved_current + resolved_radius)
    candidates = tuple(range(lower, upper + 1))
    return candidates, candidates


def _config_max_step(config: Any) -> int:
    return _positive_integer(
        getattr(config, "plastic__layer_count__max_allowable_layer_change", 1),
        name="plastic__layer_count__max_allowable_layer_change",
    )


def _history_key(current_count: int, offset: int) -> str:
    if int(offset) == 0:
        raise ValueError("PLASTIC DEPTH history offset must be non-zero")
    return f"{int(current_count)}:{int(offset):+d}"


def _finite_score_by_count(
    score_report: Sequence[Mapping[str, object]],
) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for item in score_report:
        count = int(item["active_layers"])
        feasible = bool(item.get("feasible", False))
        score = float(item.get("score", float("inf")))
        if feasible and math.isfinite(score):
            result[count] = score
    return result


def _candidate_offsets(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
) -> Tuple[int, ...]:
    values = []
    for item in score_report:
        try:
            offset = int(item["active_layers"]) - int(current_count)
        except (KeyError, TypeError, ValueError):
            continue
        if offset != 0:
            values.append(offset)
    return tuple(sorted(set(values)))


def _robust_scale(values: Sequence[float]) -> Tuple[float, float, float]:
    median = float(statistics.median(values))
    absolute_deviations = tuple(abs(float(value) - median) for value in values)
    mad = float(statistics.median(absolute_deviations))
    scale_floor = _controller.PLASTIC_DEPTH_MAD_SIGMA_FLOOR * max(
        1.0,
        abs(median),
        max(abs(float(value)) for value in values),
    )
    sigma = max(_controller.PLASTIC_DEPTH_MAD_SCALE * mad, scale_floor)
    return median, mad, sigma


def choose_plastic_depth_count_with_full_radius(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    histories: Mapping[str, Sequence[float]],
    noise_window: int,
    minimum_observations: int,
    noise_lambda: float,
    update_number: int,
    last_count_change_update: int,
    update_brake: int,
    max_step: int = 1,
) -> _controller.PlasticDepthRobustCountDecision:
    if noise_window < 1:
        raise ValueError("noise_window must be at least 1")
    if minimum_observations < 1 or minimum_observations > noise_window:
        raise ValueError("minimum_observations must lie in [1, noise_window]")
    if not math.isfinite(noise_lambda) or noise_lambda < 0.0:
        raise ValueError("noise_lambda must be finite and non-negative")
    if update_number < 1:
        raise ValueError("update_number must be positive")
    if update_brake < 0:
        raise ValueError("update_brake must be non-negative")
    resolved_max_step = _positive_integer(
        max_step,
        name="plastic__layer_count__max_allowable_layer_change",
    )

    resolved_current = int(current_count)
    score_by_count = _finite_score_by_count(score_report)
    current_score = score_by_count.get(resolved_current)
    updated_histories: Dict[str, Tuple[float, ...]] = {}
    for key, values in histories.items():
        resolved_values = tuple(float(value) for value in values[-noise_window:])
        if not all(math.isfinite(value) for value in resolved_values):
            raise ValueError(
                f"PLASTIC DEPTH paired-score history {key!r} contains a non-finite value"
            )
        updated_histories[str(key)] = resolved_values

    brake_active = (
        update_brake > 0
        and last_count_change_update >= 0
        and update_number - last_count_change_update < update_brake
    )
    evidence = []
    passing = []
    for offset in _candidate_offsets(
        current_count=resolved_current,
        score_report=score_report,
    ):
        candidate_count = resolved_current + offset
        candidate_score = score_by_count.get(candidate_count)
        feasible = current_score is not None and candidate_score is not None
        paired_difference: Optional[float] = None
        median: Optional[float] = None
        mad: Optional[float] = None
        sigma: Optional[float] = None
        standardized: Optional[float] = None
        significant = False
        key = _history_key(resolved_current, offset)
        values = list(updated_histories.get(key, ()))
        if feasible:
            paired_difference = float(candidate_score - current_score)
            values.append(paired_difference)
            values = values[-noise_window:]
            updated_histories[key] = tuple(values)
            median, mad, sigma = _robust_scale(values)
            ready = len(values) >= minimum_observations
            improving_observations = sum(value < 0.0 for value in values)
            improving_majority = improving_observations > len(values) / 2.0
            if ready:
                standardized = -median / sigma
            significant = (
                ready
                and median < -noise_lambda * sigma
                and paired_difference < 0.0
                and improving_majority
            )
            if significant and not brake_active and standardized is not None:
                passing.append((standardized, offset, candidate_count))
        evidence.append(
            _controller.PlasticDepthPairedDirectionEvidence(
                candidate_count=candidate_count,
                direction=offset,
                paired_difference=paired_difference,
                observation_count=len(values),
                median=median,
                mad=mad,
                sigma=sigma,
                standardized_improvement=standardized,
                significant=significant,
                feasible=feasible,
            )
        )

    selected_count = resolved_current
    if passing:
        _, winning_offset, _ = max(
            passing,
            key=lambda item: (item[0], -item[2]),
        )
        committed_offset = max(
            -resolved_max_step,
            min(resolved_max_step, int(winning_offset)),
        )
        selected_count = resolved_current + committed_offset
        updated_histories = {}

    return _controller.PlasticDepthRobustCountDecision(
        selected_count=selected_count,
        current_count=resolved_current,
        update_number=int(update_number),
        brake_active=brake_active,
        last_count_change_update=int(last_count_change_update),
        histories=updated_histories,
        evidence=tuple(evidence),
    )


_lookahead._lookahead_counts = _full_radius_counts
_lookahead._config_max_step = _config_max_step
_lookahead.choose_plastic_depth_count_with_exact_radius = (
    choose_plastic_depth_count_with_full_radius
)
_controller.choose_plastic_depth_count_with_mad = (
    choose_plastic_depth_count_with_full_radius
)
_trainer_step.choose_plastic_depth_count_with_mad = (
    choose_plastic_depth_count_with_full_radius
)
# ^^^ THOG
