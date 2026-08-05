# vvv THOG
"""Robust-history PLASTIC count decisions and clean wall-time estimation."""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from . import plastic_depth as _plastic_depth
from . import plastic_depth_absolute_ruler_patch as _absolute
from . import plastic_depth_controller as _controller
from . import plastic_depth_lookahead_patch as _lookahead
from . import plastic_depth_probe_interval_patch as _probe_interval
from . import trainer_step as _trainer_step


_ORIGINAL_RECORD_TRAINING_TIME = _plastic_depth.PlasticDepthSamplingLattice.record_training_time
_ORIGINAL_RECORD_OPTIMIZER_STEP_TIME = _plastic_depth.PlasticDepthSamplingLattice.record_optimizer_step_time
_ORIGINAL_TRAIN_ONE_UPDATE = _trainer_step.TrainerStepMixin.train_one_update
_ORIGINAL_IMPUTED_TRAINING_TIME = _absolute._imputed_training_time
_TIMING_SKIP_ATTRIBUTE = "_plastic_depth_skip_controller_timing_sample"


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


def _candidate_offsets_from_report(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
) -> Tuple[int, ...]:
    offsets = []
    for item in score_report:
        try:
            offset = int(item["active_layers"]) - int(current_count)
        except (KeyError, TypeError, ValueError):
            continue
        if offset != 0:
            offsets.append(offset)
    return tuple(sorted(set(offsets)))


def _history_key(current_count: int, offset: int) -> str:
    if int(offset) == 0:
        raise ValueError("PLASTIC DEPTH history offset must be non-zero")
    return f"{int(current_count)}:{int(offset):+d}"


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


def choose_plastic_depth_count_with_robust_history(
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
    """Require robust historical agreement rather than one favourable latest probe."""

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
    resolved_max_step = max(1, int(max_step))

    score_by_count = _finite_score_by_count(score_report)
    current_score = score_by_count.get(int(current_count))
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
    candidate_offsets = _candidate_offsets_from_report(
        current_count=int(current_count),
        score_report=score_report,
    )
    for offset in candidate_offsets:
        candidate_count = int(current_count) + int(offset)
        candidate_score = score_by_count.get(candidate_count)
        feasible = current_score is not None and candidate_score is not None
        paired_difference: Optional[float] = None
        median: Optional[float] = None
        mad: Optional[float] = None
        sigma: Optional[float] = None
        standardized: Optional[float] = None
        significant = False
        key = _history_key(int(current_count), int(offset))
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
                direction=int(offset),
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

    selected_count = int(current_count)
    if passing:
        _, selected_offset, _ = max(
            passing,
            key=lambda item: (item[0], -item[2]),
        )
        step = max(-resolved_max_step, min(resolved_max_step, int(selected_offset)))
        selected_count = int(current_count) + step
        for offset in candidate_offsets:
            updated_histories.pop(_history_key(selected_count, int(offset)), None)

    return _controller.PlasticDepthRobustCountDecision(
        selected_count=selected_count,
        current_count=int(current_count),
        update_number=int(update_number),
        brake_active=brake_active,
        last_count_change_update=int(last_count_change_update),
        histories=updated_histories,
        evidence=tuple(evidence),
    )


_controller.choose_plastic_depth_count_with_mad = choose_plastic_depth_count_with_robust_history
_trainer_step.choose_plastic_depth_count_with_mad = choose_plastic_depth_count_with_robust_history
_lookahead.choose_plastic_depth_count_with_exact_radius = choose_plastic_depth_count_with_robust_history
# ^^^ THOG


# vvv THOG controller timing learns only from ordinary updates; probe and transition work must not be charged to the selected depth
def _record_training_time_without_probe_contamination(
    self: Any,
    active_layers: int,
    elapsed_seconds: float,
    *,
    smoothing: float = 0.9,
    update_reference: bool = False,
) -> None:
    if bool(getattr(self, _TIMING_SKIP_ATTRIBUTE, False)):
        return
    _ORIGINAL_RECORD_TRAINING_TIME(
        self,
        active_layers,
        elapsed_seconds,
        smoothing=smoothing,
        update_reference=update_reference,
    )


def _record_optimizer_step_time_without_probe_contamination(
    self: Any,
    elapsed_seconds: float,
    *,
    smoothing: float = 0.9,
) -> None:
    if bool(getattr(self, _TIMING_SKIP_ATTRIBUTE, False)):
        return
    _ORIGINAL_RECORD_OPTIMIZER_STEP_TIME(
        self,
        elapsed_seconds,
        smoothing=smoothing,
    )


def _train_one_update_with_clean_controller_timing(self: Any) -> Dict[str, Any]:
    lattice = self._plastic_depth_lattice() if bool(getattr(self.config, "plastic__enabled", False)) else None
    next_update = int(self.state.completed_updates) + 1
    probe_update = False
    if lattice is not None and bool(getattr(self.config, "plastic__do_learn_layer_count", False)):
        interval = _probe_interval._plastic_depth_probe_interval(self)
        probe_update = next_update == 1 or next_update % interval == 0
        setattr(lattice, _TIMING_SKIP_ATTRIBUTE, probe_update)
    try:
        return _ORIGINAL_TRAIN_ONE_UPDATE(self)
    finally:
        if lattice is not None:
            setattr(lattice, _TIMING_SKIP_ATTRIBUTE, False)


_plastic_depth.PlasticDepthSamplingLattice.record_training_time = _record_training_time_without_probe_contamination
_plastic_depth.PlasticDepthSamplingLattice.record_optimizer_step_time = _record_optimizer_step_time_without_probe_contamination
_trainer_step.TrainerStepMixin.train_one_update = _train_one_update_with_clean_controller_timing
# ^^^ THOG


# vvv THOG use observed local timing slopes when available; retain the existing fixed-fraction model only as a one-anchor fallback
def _imputed_training_time_from_observed_slope(
    measurements: Sequence[Any],
    measurement: Any,
) -> Optional[float]:
    anchors = [
        candidate
        for candidate in measurements
        if candidate.training_time is not None
        and math.isfinite(float(candidate.training_time))
        and float(candidate.training_time) > 0.0
    ]
    distinct_counts = sorted({int(candidate.active_layers) for candidate in anchors})
    if len(distinct_counts) < 2:
        return _ORIGINAL_IMPUTED_TRAINING_TIME(measurements, measurement)

    time_by_count = {
        count: float(
            statistics.median(
                float(candidate.training_time)
                for candidate in anchors
                if int(candidate.active_layers) == count
            )
        )
        for count in distinct_counts
    }
    slopes = []
    for left_index, left_count in enumerate(distinct_counts):
        for right_count in distinct_counts[left_index + 1 :]:
            slopes.append(
                (time_by_count[right_count] - time_by_count[left_count])
                / float(right_count - left_count)
            )
    slope = max(0.0, float(statistics.median(slopes)))
    intercept = max(
        0.0,
        float(
            statistics.median(
                time_by_count[count] - slope * count
                for count in distinct_counts
            )
        ),
    )
    estimate = intercept + slope * int(measurement.active_layers)
    if not math.isfinite(estimate) or estimate <= 0.0:
        return _ORIGINAL_IMPUTED_TRAINING_TIME(measurements, measurement)
    return estimate


_absolute._imputed_training_time = _imputed_training_time_from_observed_slope
# ^^^ THOG
