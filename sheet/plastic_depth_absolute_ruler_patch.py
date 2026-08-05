# vvv THOG
"""Fixed-capacity absolute-ruler semantics for learned-count PLASTIC DEPTH.

Learned-count PLASTIC now owns the complete max-permitted capacity lattice over
Chebyshev's native [-1,+1] interval. The active model executes the prefix of
that fixed lattice. Wider lookahead radii are therefore real future samples,
not invented coordinates produced by rescaling the current active prefix.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

from . import plastic_depth as _plastic_depth
from . import plastic_depth_controller as _controller
from . import plastic_depth_lookahead_patch as _lookahead
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step


_FIXED_TIMING_FRACTION = 0.20


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


def _sample_layer_coordinates_from_public(public_coordinates: Tensor, maximum_layers: int) -> Tensor:
    if int(maximum_layers) == 1:
        return torch.ones_like(public_coordinates, dtype=public_coordinates.dtype)
    return 1.0 + (float(maximum_layers) - 1.0) * (public_coordinates - 1.0) / 99.0


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
            active_public[-1] if self.learn_layer_count and len(capacity_public) > self.current_active_layers else None
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


_ORIGINAL_LATTICE_INIT = _plastic_depth.PlasticDepthSamplingLattice.__init__


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
    initialisation = kwargs.get("initialisation")
    seed = int(kwargs.get("seed", 1337))
    if len(args) >= 3 and initialisation is None:
        initialisation = args[2]
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


# vvv THOG restore full exact-radius candidate semantics now that the future samples are real fixed-capacity coordinates
def _lookahead_counts_absolute(current: int, maximum: int, radius: int, max_step: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    resolved_current = int(current)
    resolved_maximum = int(maximum)
    resolved_radius = int(radius)
    resolved_step = int(max_step)
    if resolved_radius < 1:
        raise ValueError("plastic__layer_count_probe_radius must be positive")
    if resolved_step < 1:
        raise ValueError("plastic__layer_count_max_step must be positive")
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
    value = getattr(config, "plastic__layer_count_max_step", getattr(_lookahead, "os").environ.get("THOG2_PLASTIC_LAYER_COUNT_MAX_STEP", 1))
    resolved = int(value)
    if resolved < 1:
        raise ValueError("plastic__layer_count_max_step must be positive")
    return resolved


def choose_plastic_depth_count_with_absolute_radius(
    *,
    current_count: int,
    score_report: Sequence[Dict[str, object]],
    histories: Dict[str, Sequence[float]],
    noise_window: int,
    minimum_observations: int,
    noise_lambda: float,
    update_number: int,
    last_count_change_update: int,
    update_brake: int,
    max_step: int = 1,
) -> Any:
    return _lookahead.choose_plastic_depth_count_with_exact_radius(
        current_count=current_count,
        score_report=score_report,
        histories=histories,
        noise_window=noise_window,
        minimum_observations=minimum_observations,
        noise_lambda=noise_lambda,
        update_number=update_number,
        last_count_change_update=last_count_change_update,
        update_brake=update_brake,
        max_step=max(1, int(max_step)),
    )


_lookahead._lookahead_counts = _lookahead_counts_absolute
_lookahead._config_max_step = _config_max_step_absolute
_controller.choose_plastic_depth_count_with_mad = choose_plastic_depth_count_with_absolute_radius
_trainer_step.choose_plastic_depth_count_with_mad = choose_plastic_depth_count_with_absolute_radius
# ^^^ THOG


# vvv THOG give relative wall-time candidates an imputed timing estimate instead of deadlocking on unvisited counts
_ORIGINAL_CANDIDATE_SCORE = _plastic_depth.plastic_depth_candidate_score


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
    selected, report = _plastic_depth.choose_plastic_depth_candidate.__wrapped__(  # type: ignore[attr-defined]
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


if not hasattr(_plastic_depth.choose_plastic_depth_candidate, "__wrapped__"):
    _plastic_depth.choose_plastic_depth_candidate.__wrapped__ = _plastic_depth.choose_plastic_depth_candidate  # type: ignore[attr-defined]
_plastic_depth.choose_plastic_depth_candidate = choose_plastic_depth_candidate_with_timing_imputation
_trainer_step.choose_plastic_depth_candidate = choose_plastic_depth_candidate_with_timing_imputation
_lookahead.choose_plastic_depth_candidate = choose_plastic_depth_candidate_with_timing_imputation
# ^^^ THOG


# vvv THOG show active coordinates on the absolute layer ruler rather than the old 1-100 UI ruler
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload


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
