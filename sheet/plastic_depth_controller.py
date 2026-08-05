# vvv THOG
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple


PLASTIC_DEPTH_MAD_SCALE = 1.4826
PLASTIC_DEPTH_MAD_SIGMA_FLOOR = 1.0e-12


@dataclass(frozen=True)
class PlasticDepthPairedDirectionEvidence:
    candidate_count: int
    direction: int
    paired_difference: Optional[float]
    observation_count: int
    median: Optional[float]
    mad: Optional[float]
    sigma: Optional[float]
    standardized_improvement: Optional[float]
    significant: bool
    feasible: bool


@dataclass(frozen=True)
class PlasticDepthRobustCountDecision:
    selected_count: int
    current_count: int
    update_number: int
    brake_active: bool
    last_count_change_update: int
    histories: Dict[str, Tuple[float, ...]]
    evidence: Tuple[PlasticDepthPairedDirectionEvidence, ...]

    def report(self) -> Tuple[Dict[str, object], ...]:
        return tuple(
            {
                "candidate_count": item.candidate_count,
                "direction": item.direction,
                "paired_difference": item.paired_difference,
                "observation_count": item.observation_count,
                "median": item.median,
                "mad": item.mad,
                "sigma": item.sigma,
                "standardized_improvement": item.standardized_improvement,
                "significant": item.significant,
                "feasible": item.feasible,
                "brake_active": self.brake_active,
                "selected": item.candidate_count == self.selected_count,
            }
            for item in self.evidence
        )


def _history_key(current_count: int, direction: int) -> str:
    if direction not in (-1, 1):
        raise ValueError(f"direction must be -1 or +1; got {direction}")
    return f"{current_count}:{direction:+d}"


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


def _robust_scale(values: Sequence[float], current_difference: float) -> Tuple[float, float, float]:
    median = float(statistics.median(values))
    absolute_deviations = tuple(abs(value - median) for value in values)
    mad = float(statistics.median(absolute_deviations))
    scale_floor = PLASTIC_DEPTH_MAD_SIGMA_FLOOR * max(
        1.0,
        abs(median),
        abs(current_difference),
    )
    sigma = max(PLASTIC_DEPTH_MAD_SCALE * mad, scale_floor)
    return median, mad, sigma


def choose_plastic_depth_count_with_mad(
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
) -> PlasticDepthRobustCountDecision:
    """Choose N-1/N/N+1 using paired score differences and a robust MAD gate."""

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
    evidence = []
    passing = []
    for direction in (-1, 1):
        candidate_count = current_count + direction
        candidate_score = score_by_count.get(candidate_count)
        feasible = current_score is not None and candidate_score is not None
        paired_difference: Optional[float] = None
        median: Optional[float] = None
        mad: Optional[float] = None
        sigma: Optional[float] = None
        standardized: Optional[float] = None
        significant = False
        key = _history_key(current_count, direction)
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
                passing.append((standardized, candidate_count))
        evidence.append(
            PlasticDepthPairedDirectionEvidence(
                candidate_count=candidate_count,
                direction=direction,
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
        # vvv THOG larger standardized improvement wins; exact ties prefer the lower count for deterministic compute conservation
        selected_count = max(passing, key=lambda item: (item[0], -item[1]))[1]
        # ^^^ THOG
        for direction in (-1, 1):
            updated_histories.pop(_history_key(selected_count, direction), None)

    return PlasticDepthRobustCountDecision(
        selected_count=selected_count,
        current_count=current_count,
        update_number=update_number,
        brake_active=brake_active,
        last_count_change_update=last_count_change_update,
        histories=updated_histories,
        evidence=tuple(evidence),
    )


__all__ = [
    "PLASTIC_DEPTH_MAD_SCALE",
    "PLASTIC_DEPTH_MAD_SIGMA_FLOOR",
    "PlasticDepthPairedDirectionEvidence",
    "PlasticDepthRobustCountDecision",
    "choose_plastic_depth_count_with_mad",
]
# ^^^ THOG
