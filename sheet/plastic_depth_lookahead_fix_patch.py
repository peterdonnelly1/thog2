# vvv THOG
"""Compatibility fixes for exact-radius PLASTIC DEPTH lookahead."""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from . import plastic_depth_controller as _controller
from . import plastic_depth_lookahead_patch as _lookahead
from . import trainer_step as _trainer_step


# vvv THOG keep existing tests and downstream patches effective by resolving trainer_step symbols at call time
def _candidate_measurement_proxy(*args: Any, **kwargs: Any) -> Any:
    return _trainer_step.PlasticDepthCandidateMeasurement(*args, **kwargs)


def _candidate_selector_proxy(*args: Any, **kwargs: Any) -> Any:
    return _trainer_step.choose_plastic_depth_candidate(*args, **kwargs)


def _cuda_reserve_proxy(*args: Any, **kwargs: Any) -> Any:
    return _trainer_step.PlasticDepthCudaAllocatorReserve(*args, **kwargs)


_lookahead.PlasticDepthCandidateMeasurement = _candidate_measurement_proxy
_lookahead.choose_plastic_depth_candidate = _candidate_selector_proxy
_lookahead.PlasticDepthCudaAllocatorReserve = _cuda_reserve_proxy
# ^^^ THOG


# vvv THOG learned-count growth has exactly one pre-made dormant probe; wider radius is a shrink-lookahead only
def _single_probe_growth_counts(
    current: int,
    maximum: int,
    radius: int,
    max_step: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    if int(max_step) != 1:
        raise ValueError("PLASTIC DEPTH count transitions currently require --plastic-layer-count-max-step 1")
    decision_counts = {int(current)}
    execution_counts = {int(current)}
    if int(current) - int(radius) >= 1:
        decision_counts.add(int(current) - int(radius))
        execution_counts.add(int(current) - 1)
    if int(current) < int(maximum):
        decision_counts.add(int(current) + 1)
        execution_counts.add(int(current) + 1)
    return tuple(sorted(decision_counts)), tuple(sorted(execution_counts))


def _config_max_step(config: Any) -> int:
    resolved = int(getattr(config, "plastic__layer_count__max_allowable_layer_change", 1))
    if resolved != 1:
        raise ValueError("PLASTIC DEPTH count transitions currently require plastic__layer_count__max_allowable_layer_change=1")
    return 1


_lookahead._lookahead_counts = _single_probe_growth_counts
_lookahead._config_max_step = _config_max_step
# ^^^ THOG


def _history_key(current_count: int, offset: int) -> str:
    if offset == 0:
        raise ValueError("PLASTIC DEPTH history offset must be non-zero")
    return f"{current_count}:{offset:+d}"


def _finite_score_by_count(score_report: Sequence[Mapping[str, object]]) -> Dict[int, float]:
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
            offset = int(item["active_layers"]) - current_count
        except (KeyError, TypeError, ValueError):
            continue
        if offset != 0:
            offsets.append(offset)
    return tuple(sorted(set(offsets)))


def _robust_scale(values: Sequence[float], current_difference: float) -> Tuple[float, float, float]:
    median = float(statistics.median(values))
    absolute_deviations = tuple(abs(value - median) for value in values)
    mad = float(statistics.median(absolute_deviations))
    scale_floor = _controller.PLASTIC_DEPTH_MAD_SIGMA_FLOOR * max(
        1.0,
        abs(median),
        abs(current_difference),
    )
    sigma = max(_controller.PLASTIC_DEPTH_MAD_SCALE * mad, scale_floor)
    return median, mad, sigma


def choose_plastic_depth_count_with_exact_radius(
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
    """Choose one bounded step while preserving evidence for infeasible probed offsets."""

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
    if int(max_step) != 1:
        raise ValueError("PLASTIC DEPTH count transitions currently require max_step=1")
    resolved_max_step = 1

    score_by_count = _finite_score_by_count(score_report)
    current_score = score_by_count.get(current_count)
    updated_histories: Dict[str, Tuple[float, ...]] = {}
    for key, values in histories.items():
        resolved_values = tuple(float(value) for value in values[-noise_window:])
        if not all(math.isfinite(value) for value in resolved_values):
            raise ValueError(f"PLASTIC DEPTH paired-score history {key!r} contains a non-finite value")
        updated_histories[str(key)] = resolved_values

    brake_active = (
        update_brake > 0
        and last_count_change_update >= 0
        and update_number - last_count_change_update < update_brake
    )
    candidate_offsets = _candidate_offsets_from_report(
        current_count=current_count,
        score_report=score_report,
    )
    evidence = []
    passing = []
    for offset in candidate_offsets:
        candidate_count = current_count + offset
        candidate_score = score_by_count.get(candidate_count)
        feasible = current_score is not None and candidate_score is not None
        paired_difference: Optional[float] = None
        median: Optional[float] = None
        mad: Optional[float] = None
        sigma: Optional[float] = None
        standardized: Optional[float] = None
        significant = False
        key = _history_key(current_count, offset)
        values = list(updated_histories.get(key, ()))
        if feasible:
            paired_difference = float(candidate_score - current_score)
            values.append(paired_difference)
            values = values[-noise_window:]
            updated_histories[key] = tuple(values)
            median, mad, sigma = _robust_scale(values, paired_difference)
            standardized = -paired_difference / sigma
            significant = (
                len(values) >= minimum_observations
                and paired_difference < -noise_lambda * sigma
            )
            if significant and not brake_active:
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

    selected_count = current_count
    if passing:
        _, selected_offset, _ = max(passing, key=lambda item: (item[0], -item[2]))
        step = max(-resolved_max_step, min(resolved_max_step, selected_offset))
        selected_count = current_count + step
        for offset in candidate_offsets:
            updated_histories.pop(_history_key(selected_count, offset), None)

    return _controller.PlasticDepthRobustCountDecision(
        selected_count=selected_count,
        current_count=current_count,
        update_number=update_number,
        brake_active=brake_active,
        last_count_change_update=last_count_change_update,
        histories=updated_histories,
        evidence=tuple(evidence),
    )


_lookahead.choose_plastic_depth_count_with_exact_radius = choose_plastic_depth_count_with_exact_radius
_controller.choose_plastic_depth_count_with_mad = choose_plastic_depth_count_with_exact_radius
_trainer_step.choose_plastic_depth_count_with_mad = choose_plastic_depth_count_with_exact_radius
# ^^^ THOG
