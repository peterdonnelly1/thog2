# vvv THOG
"""Fixed-capacity absolute-ruler semantics for learned-count PLASTIC DEPTH.

Learned-count PLASTIC now owns the complete max-permitted capacity lattice over
Chebyshev's native [-1,+1] interval. The active model executes the prefix of
that fixed lattice. Wider lookahead radii are therefore real future samples,
not invented coordinates produced by rescaling the current active prefix.
"""

from __future__ import annotations

import math
import os
import statistics
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from . import depth_trajectory as _depth_trajectory
from . import plastic_depth as _plastic_depth
from . import plastic_depth_controller as _controller
from . import plastic_depth_lookahead_patch as _lookahead
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step
from .basis import chebyshev_first_kind_basis, deterministic_reduced_qr, normalized_coordinates


_FIXED_TIMING_FRACTION = 0.20
_ORIGINAL_CHOOSE_CANDIDATE = _plastic_depth.choose_plastic_depth_candidate
_ORIGINAL_DEPTH_TRAJECTORY_INIT = _depth_trajectory.DepthTrajectory.__init__
_ORIGINAL_LATTICE_INIT = _plastic_depth.PlasticDepthSamplingLattice.__init__
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload


def _validate_active_count(lattice: Any, active_layers: int) -> int:
    resolved = int(active_layers)
    if resolved < 1 or resolved > int(lattice.maximum_layers):
        raise ValueError(
            "active_layers must lie in [1, maximum_layers]; "
            f"got active_layers={resolved}, maximum_layers={lattice.maximum_layers}"
        )
    return resolved


def _capacity_public_coordinates_from_raw(lattice: Any, raw_intervals: Tensor) -> Tensor:
    maximum_layers = int(lattice.maximum_layers)
    if maximum_layers == 1:
        return raw_intervals.new_tensor([50.5])
    intervals = lattice._positive_prefix(maximum_layers - 1, raw_intervals=raw_intervals)
    cumulative = torch.cat((intervals.new_zeros(1), torch.cumsum(intervals, dim=0)))
    return 1.0 + 99.0 * cumulative / intervals.sum()


def _public_coordinates_from_raw_absolute(
    self: Any,
    active_layers: int,
    *,
    include_probe: bool,
    raw_intervals: Tensor,
) -> Tensor:
    resolved_count = _validate_active_count(self, active_layers)
    capacity_coordinates = _capacity_public_coordinates_from_raw(self, raw_intervals)
    if self.learn_layer_count and bool(include_probe):
        return capacity_coordinates
    return capacity_coordinates[:resolved_count]


def _public_coordinates_absolute(
    self: Any,
    active_layers: Optional[int] = None,
    *,
    include_probe: Optional[bool] = None,
) -> Tensor:
    resolved_count = self.current_active_layers if active_layers is None else int(active_layers)
    resolved_count = _validate_active_count(self, resolved_count)
    resolved_include_probe = self.learn_layer_count if include_probe is None else bool(include_probe)
    return self._public_coordinates_from_raw(
        resolved_count,
        include_probe=resolved_include_probe,
        raw_intervals=self.raw_intervals,
    )


def _active_public_coordinates_absolute(self: Any, active_layers: Optional[int] = None) -> Tensor:
    resolved_count = self.current_active_layers if active_layers is None else int(active_layers)
    resolved_count = _validate_active_count(self, resolved_count)
    return self.public_coordinates(resolved_count, include_probe=False)


def _probe_public_coordinate_absolute(self: Any) -> Tensor:
    if not self.learn_layer_count:
        raise RuntimeError("fixed-count PLASTIC DEPTH has no future probe coordinate")
    if self.current_active_layers >= self.maximum_layers:
        raise RuntimeError("PLASTIC DEPTH is already at maximum active layers")
    capacity_coordinates = self.public_coordinates(self.current_active_layers, include_probe=True)
    return capacity_coordinates[self.current_active_layers : self.current_active_layers + 1]


def _chart_span_absolute(self: Any, active_layers: int, *, raw_intervals: Tensor) -> Tensor:
    if int(self.maximum_layers) == 1:
        return raw_intervals.new_tensor(1.0)
    return self._positive_prefix(int(self.maximum_layers) - 1, raw_intervals=raw_intervals).sum()


def _prepare_count_transition_absolute(self: Any, new_active_layers: int) -> Any:
    if not self.learn_layer_count:
        raise RuntimeError("fixed-count PLASTIC DEPTH has no layer-count controller transition")
    previous_count = int(self.current_active_layers)
    resolved_count = _validate_active_count(self, int(new_active_layers))
    expected = self.raw_intervals.detach().clone()
    proposed = expected.clone()
    return _plastic_depth.PlasticDepthGeometryTransition(
        previous_active_layers=previous_count,
        new_active_layers=resolved_count,
        old_from_new_scale=1.0,
        old_from_new_shift=0.0,
        expected_raw_intervals=expected,
        proposed_raw_intervals=proposed,
    )


def _set_active_layer_count_absolute(self: Any, active_layers: int) -> None:
    if not self.learn_layer_count:
        raise RuntimeError("fixed-count PLASTIC DEPTH has no layer-count controller")
    resolved_count = _validate_active_count(self, int(active_layers))
    self.active_layer_count.fill_(resolved_count)


def _sample_layer_tuple(public_values: Sequence[float], maximum_layers: int) -> Tuple[float, ...]:
    if int(maximum_layers) == 1:
        return tuple(1.0 for _ in public_values)
    scale = (float(maximum_layers) - 1.0) / 99.0
    return tuple(1.0 + scale * (float(value) - 1.0) for value in public_values)


def _interval_report_absolute(self: Any) -> Dict[str, object]:
    coordinates = self.public_coordinates(include_probe=True)
    active_coordinates = coordinates[: self.current_active_layers]
    if active_coordinates.numel() <= 1:
        intervals = active_coordinates.new_empty(0)
    else:
        intervals = active_coordinates[1:] - active_coordinates[:-1]
    initial = self.initial_public_coordinates.to(coordinates.device)
    comparable = min(int(initial.numel()), int(coordinates.numel()))
    movement = coordinates[:comparable].to(dtype=torch.float64) - initial[:comparable]
    capacity_public = tuple(float(value) for value in coordinates.detach().cpu().tolist())
    active_public = tuple(float(value) for value in active_coordinates.detach().cpu().tolist())
    capacity_sample_layer = _sample_layer_tuple(capacity_public, int(self.maximum_layers))
    active_sample_layer = _sample_layer_tuple(active_public, int(self.maximum_layers))
    return {
        "maximum_layers": self.maximum_layers,
        "active_layers": self.current_active_layers,
        "public_coordinates": capacity_public,
        "active_ranks": self.active_ranks(),
        "active_public_coordinates": active_public,
        "sample_layer_coordinates": capacity_sample_layer,
        "active_sample_layer_coordinates": active_sample_layer,
        "probe_public_coordinate": (
            capacity_public[self.current_active_layers]
            if self.learn_layer_count and len(capacity_public) > self.current_active_layers
            else None
        ),
        "probe_sample_layer_coordinate": (
            capacity_sample_layer[self.current_active_layers]
            if self.learn_layer_count and len(capacity_sample_layer) > self.current_active_layers
            else None
        ),
        "minimum_interval": float(intervals.min().item()) if intervals.numel() else None,
        "maximum_interval": float(intervals.max().item()) if intervals.numel() else None,
        "mean_absolute_movement": float(movement.abs().mean().item()) if movement.numel() else 0.0,
        "initialisation": self.initialisation,
        "learn_layer_count": self.learn_layer_count,
        "version": _plastic_depth.PLASTIC_DEPTH_VERSION,
        "ruler": "absolute_max_capacity",
    }


# Install method replacements before wrapping __init__, so the original constructor's
# initial_public_coordinates buffer is built on the absolute capacity ruler.
_plastic_depth.PlasticDepthSamplingLattice._public_coordinates_from_raw = _public_coordinates_from_raw_absolute
_plastic_depth.PlasticDepthSamplingLattice.public_coordinates = _public_coordinates_absolute
_plastic_depth.PlasticDepthSamplingLattice.active_public_coordinates = _active_public_coordinates_absolute
_plastic_depth.PlasticDepthSamplingLattice.probe_public_coordinate = _probe_public_coordinate_absolute
_plastic_depth.PlasticDepthSamplingLattice._chart_span = _chart_span_absolute
_plastic_depth.PlasticDepthSamplingLattice.prepare_count_transition = _prepare_count_transition_absolute
_plastic_depth.PlasticDepthSamplingLattice.set_active_layer_count = _set_active_layer_count_absolute
_plastic_depth.PlasticDepthSamplingLattice.interval_report = _interval_report_absolute


def _init_absolute_ruler(self: Any, *args: Any, **kwargs: Any) -> None:
    seed = int(kwargs.get("seed", 1337))
    if len(args) >= 4:
        seed = int(args[3])
    _ORIGINAL_LATTICE_INIT(self, *args, **kwargs)
    if self.initialisation == "random" and self.raw_intervals.numel() > 0:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        positive_intervals = torch.rand(
            self.raw_intervals.numel(),
            generator=generator,
            dtype=torch.float64,
        ) + 0.25
        with torch.no_grad():
            self.raw_intervals.copy_(
                self._inverse_softplus(positive_intervals).to(self.raw_intervals)
            )
    initial_active_layers = self.current_active_layers if self.learn_layer_count else self.maximum_layers
    initial_coordinates = self.public_coordinates(
        initial_active_layers,
        include_probe=self.learn_layer_count,
    ).detach().to(dtype=torch.float64)
    self._buffers["initial_public_coordinates"] = initial_coordinates


_plastic_depth.PlasticDepthSamplingLattice.__init__ = _init_absolute_ruler


def _init_depth_trajectory_absolute(self: Any, *args: Any, **kwargs: Any) -> None:
    _ORIGINAL_DEPTH_TRAJECTORY_INIT(self, *args, **kwargs)
    if not bool(getattr(self, "plastic_enabled", False)) or self.plastic_sampling is None:
        return
    reference_sample_count = max(int(self.config.depth_order), int(self.config.n_layer))
    reference_coordinates = normalized_coordinates(
        reference_sample_count,
        dtype=torch.float64,
        device="cpu",
    )
    reference_raw = chebyshev_first_kind_basis(reference_coordinates, int(self.config.depth_order))
    _, reference_r = deterministic_reduced_qr(reference_raw)
    self._buffers["plastic_depth_inverse_r"] = torch.linalg.inv(reference_r)
    self._buffers["plastic_depth_reference_sample_count"] = torch.tensor(
        reference_sample_count,
        dtype=torch.long,
    )


_depth_trajectory.DepthTrajectory.__init__ = _init_depth_trajectory_absolute


# vvv THOG restore full exact-radius candidate semantics now that the future samples are real fixed-capacity coordinates
def _lookahead_counts_absolute(current: int, maximum: int, radius: int, max_step: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    resolved_current = int(current)
    resolved_maximum = int(maximum)
    resolved_radius = int(radius)
    resolved_step = int(max_step)
    if resolved_radius < 1:
        raise ValueError("plastic__layer_count_probe_radius must be positive")
    if resolved_step < 1:
        raise ValueError("plastic__layer_count__max_allowable_layer_change must be positive")
    decision_counts = {resolved_current}
    execution_counts = {resolved_current}
    lower_decision = max(1, resolved_current - resolved_radius)
    upper_decision = min(resolved_maximum, resolved_current + resolved_radius)
    if lower_decision != resolved_current:
        decision_counts.add(lower_decision)
        execution_counts.add(max(1, resolved_current - resolved_step))
    if upper_decision != resolved_current:
        decision_counts.add(upper_decision)
        execution_counts.add(min(resolved_maximum, resolved_current + resolved_step))
    execution_counts.update(decision_counts)
    return tuple(sorted(decision_counts)), tuple(sorted(execution_counts))


def _config_max_step_absolute(config: Any) -> int:
    value = getattr(config, "plastic__layer_count__max_allowable_layer_change", os.environ.get("THOG2_PLASTIC_LAYER_COUNT_MAX_STEP", 1))
    resolved = int(value)
    if resolved < 1:
        raise ValueError("plastic__layer_count__max_allowable_layer_change must be positive")
    return resolved


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


def _candidate_offsets_from_report(*, current_count: int, score_report: Sequence[Mapping[str, object]]) -> Tuple[int, ...]:
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


def choose_plastic_depth_count_with_absolute_radius(
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
) -> Any:
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


_lookahead._lookahead_counts = _lookahead_counts_absolute
_lookahead._config_max_step = _config_max_step_absolute
_lookahead.choose_plastic_depth_count_with_exact_radius = choose_plastic_depth_count_with_absolute_radius
_controller.choose_plastic_depth_count_with_mad = choose_plastic_depth_count_with_absolute_radius
_trainer_step.choose_plastic_depth_count_with_mad = choose_plastic_depth_count_with_absolute_radius
# ^^^ THOG


# vvv THOG give relative wall-time candidates an imputed timing estimate instead of deadlocking on unvisited counts
def _imputed_training_time(measurements: Sequence[Any], measurement: Any) -> Optional[float]:
    anchors = [
        candidate
        for candidate in measurements
        if candidate.training_time is not None
        and math.isfinite(float(candidate.training_time))
        and float(candidate.training_time) > 0.0
    ]
    if not anchors:
        return None
    anchor = min(anchors, key=lambda candidate: abs(int(candidate.active_layers) - int(measurement.active_layers)))
    if int(anchor.active_layers) <= 0:
        return None
    ratio = _FIXED_TIMING_FRACTION + (1.0 - _FIXED_TIMING_FRACTION) * int(measurement.active_layers) / int(anchor.active_layers)
    return float(anchor.training_time) * max(0.05, ratio)


def choose_plastic_depth_candidate_with_timing_imputation(
    measurements: Sequence[Any],
    *,
    objective: str,
    maximum_layers: int,
    cost_weight: float,
    reference_training_time: Optional[float],
    memory_budget_gib: Optional[float],
) -> Tuple[Any, Tuple[Dict[str, object], ...]]:
    resolved_measurements = tuple(measurements)
    imputed_measurements = []
    timing_sources = []
    for measurement in resolved_measurements:
        timing = measurement.training_time
        source = "observed" if timing is not None and math.isfinite(float(timing)) else None
        if objective == "relative_training_wall_time" and source is None:
            timing = _imputed_training_time(resolved_measurements, measurement)
            source = "imputed" if timing is not None else None
        imputed_measurements.append(
            _plastic_depth.PlasticDepthCandidateMeasurement(
                active_layers=int(measurement.active_layers),
                validation_loss=float(measurement.validation_loss),
                training_time=None if timing is None else float(timing),
                peak_allocated_gib=measurement.peak_allocated_gib,
                peak_reserved_gib=measurement.peak_reserved_gib,
            )
        )
        timing_sources.append(source)
    selected, report = _ORIGINAL_CHOOSE_CANDIDATE(
        tuple(imputed_measurements),
        objective=objective,
        maximum_layers=maximum_layers,
        cost_weight=cost_weight,
        reference_training_time=reference_training_time,
        memory_budget_gib=memory_budget_gib,
    )
    enriched = []
    for item, source in zip(report, timing_sources):
        row = dict(item)
        row["training_time_source"] = source
        enriched.append(row)
    return selected, tuple(enriched)


_plastic_depth.choose_plastic_depth_candidate = choose_plastic_depth_candidate_with_timing_imputation
_trainer_step.choose_plastic_depth_candidate = choose_plastic_depth_candidate_with_timing_imputation
_lookahead.choose_plastic_depth_candidate = choose_plastic_depth_candidate_with_timing_imputation
# ^^^ THOG


# vvv THOG show active coordinates on the absolute layer ruler rather than the old 1-100 UI ruler
def _prepare_console_progress_payload_with_sample_layer(self: Any, event: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    if event in {"optimizer_progress", "evaluation_completed"} and "depth_sample_points" in values:
        lattice = self._plastic_depth_lattice() if bool(getattr(getattr(self, "config", None), "plastic__enabled", False)) else None
        if lattice is not None:
            values["depth_sample_points"] = _sample_layer_tuple(
                tuple(float(value) for value in values["depth_sample_points"]),
                int(lattice.maximum_layers),
            )
            values["plastic_depth_coordinate_label"] = "sample_layer"
    return values


_stage6.Stage6Trainer._prepare_console_progress_payload = _prepare_console_progress_payload_with_sample_layer
# ^^^ THOG

__all__ = ["_sample_layer_tuple"]
# ^^^ THOG
