# vvv THOG
"""PLASTIC v0.531 equivalent-time wall-time objective, cheap timing telemetry, and final console polish."""

from __future__ import annotations

import contextvars
import math
import re
from collections import deque
from dataclasses import replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import constants as _constants

from . import plastic_depth as _plastic_depth
from . import plastic_depth_lookahead_patch as _lookahead
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step
from . import wandb_telemetry as _telemetry


WALL_TIME_ALGORITHM = "wall_time_equivalent_time_gain"
WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT = 0.9
WALL_TIME_LOSS_RATE_WINDOW = 64
WALL_TIME_LOSS_RATE_MIN_OBSERVATIONS = 16

_STATE_ATTRIBUTE = "_plastic_wall_time_equivalent_time_gain_state"
_PROBE_ATTRIBUTE = "_plastic_wall_time_probe_this_update"
_COUNT_ATTRIBUTE = "_plastic_wall_time_layer_count_this_update"
_ACTIVE_TRAINER: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "plastic_wall_time_active_trainer",
    default=None,
)

_DIRECTION_SUMMARY = re.compile(
    r"L/R/A=\[(?P<left>\d+)/(?P<right>\d+)/(?P<ambiguous>\d+)\]/"
    r"(?P<total>\d+)=>(?P<outcome>stet|L|R|-)"
)
_LAYERS_SAMPLED = re.compile(
    r"layers = (?P<count>\d+)[ \t]+sampled ="
)
_SCORE_Z_VECTOR = re.compile(
    r"(?P<prefix>score_z \[[^\]]+\] = \[)(?P<body>[^\]]*)(?P<close>\])"
)
_SIGNED_ONE_DIGIT_FIXED = re.compile(
    r"(?P<sign>[+-])(?P<integer>\d)(?P<fraction>\.\d+)(?![eE\d])"
)

_ORIGINAL_CHOOSE_PLASTIC_DEPTH_CANDIDATE = _plastic_depth.choose_plastic_depth_candidate
_ORIGINAL_BEGIN_INLINE_UPDATE = _trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update
_ORIGINAL_INLINE_PROBE_REQUEST = _trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request
_ORIGINAL_TIMED = _stage6.Stage6Trainer._timed
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line
_ORIGINAL_ATTACH_TELEMETRY = _telemetry.attach_telemetry
_ORIGINAL_TRAINING_METRICS = _telemetry._training_metrics
_ORIGINAL_START_WANDB = _telemetry.WandbTelemetry._start_wandb
_ORIGINAL_PLASTIC_DEVICE_SYNCHRONIZE = _trainer_step.TrainerStepMixin._plastic_depth_device_synchronize


def _runtime_state(trainer: Any) -> Dict[str, Any]:
    state = getattr(trainer, _STATE_ATTRIBUTE, None)
    if state is not None:
        return state
    state = {
        "timing_n": 0,
        "timing_sum_x": 0.0,
        "timing_sum_y": 0.0,
        "timing_sum_x2": 0.0,
        "timing_sum_xy": 0.0,
        "timing_sum_y2": 0.0,
        "timing_distinct_counts": set(),
        "last_timing_sample_seconds": None,
        "last_timing_sample_update": None,
        "last_timing_residual_seconds": None,
        "ordinary_elapsed_seconds": 0.0,
        "loss_history": deque(maxlen=WALL_TIME_LOSS_RATE_WINDOW),
        "activated": False,
        "activation_update": None,
    }
    setattr(trainer, _STATE_ATTRIBUTE, state)
    return state


def _linear_fit_from_sums(
    *,
    n: int,
    sum_x: float,
    sum_y: float,
    sum_x2: float,
    sum_xy: float,
    sum_y2: float,
) -> Optional[Dict[str, float]]:
    if n < 2:
        return None
    resolved_n = float(n)
    sxx = float(sum_x2) - float(sum_x) * float(sum_x) / resolved_n
    if not math.isfinite(sxx) or sxx <= 0.0:
        return None
    sxy = float(sum_xy) - float(sum_x) * float(sum_y) / resolved_n
    slope = sxy / sxx
    intercept = (float(sum_y) - slope * float(sum_x)) / resolved_n
    sst = max(
        0.0,
        float(sum_y2) - float(sum_y) * float(sum_y) / resolved_n,
    )
    sse = max(0.0, sst - slope * sxy)
    r_squared = None if sst <= 0.0 else max(-1.0, min(1.0, 1.0 - sse / sst))
    slope_standard_error = None
    if n > 2:
        residual_variance = sse / float(n - 2)
        slope_standard_error = math.sqrt(max(0.0, residual_variance / sxx))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared) if r_squared is not None else float("nan"),
        "slope_standard_error": (
            float(slope_standard_error)
            if slope_standard_error is not None
            else float("nan")
        ),
        "sse": float(sse),
        "sst": float(sst),
    }


def _linear_fit_points(points: Sequence[Tuple[float, float]]) -> Optional[Dict[str, float]]:
    if len(points) < 2:
        return None
    return _linear_fit_from_sums(
        n=len(points),
        sum_x=sum(float(x) for x, _ in points),
        sum_y=sum(float(y) for _, y in points),
        sum_x2=sum(float(x) * float(x) for x, _ in points),
        sum_xy=sum(float(x) * float(y) for x, y in points),
        sum_y2=sum(float(y) * float(y) for _, y in points),
    )


def _timing_fit(trainer: Any) -> Optional[Dict[str, float]]:
    state = _runtime_state(trainer)
    if len(state["timing_distinct_counts"]) < 2:
        return None
    fit = _linear_fit_from_sums(
        n=int(state["timing_n"]),
        sum_x=float(state["timing_sum_x"]),
        sum_y=float(state["timing_sum_y"]),
        sum_x2=float(state["timing_sum_x2"]),
        sum_xy=float(state["timing_sum_xy"]),
        sum_y2=float(state["timing_sum_y2"]),
    )
    if fit is None:
        return None
    if not math.isfinite(float(fit["slope"])) or float(fit["slope"]) <= 0.0:
        return None
    return fit


def _loss_rate_fit(trainer: Any) -> Optional[Dict[str, float]]:
    state = _runtime_state(trainer)
    points = tuple(state["loss_history"])
    if len(points) < WALL_TIME_LOSS_RATE_MIN_OBSERVATIONS:
        return None
    fit = _linear_fit_points(points)
    if fit is None:
        return None
    loss_improvement_rate = -float(fit["slope"])
    if not math.isfinite(loss_improvement_rate) or loss_improvement_rate <= 0.0:
        return None
    return {
        **fit,
        "loss_improvement_rate": loss_improvement_rate,
        "observations": float(len(points)),
    }


def _wall_time_horizon_updates(config: Any) -> int:
    probe_interval = max(
        1,
        int(
            getattr(
                config,
                "plastic__layer_count_probe__probe_every_n_steps",
                1,
            )
            or 1
        ),
    )
    history_window = max(
        1,
        int(
            getattr(
                config,
                "plastic__layer_count_probe__window_size_as_number_of_probes",
                1,
            )
        ),
    )
    update_brake = max(
        0,
        int(getattr(config, "plastic__layer_count_update_brake", 0)),
    )
    brake_probe_gaps = (
        max(1, math.ceil(update_brake / probe_interval))
        if update_brake > 0
        else 1
    )
    probe_gaps = max(1, history_window, brake_probe_gaps)
    return probe_gaps * probe_interval - 1


def _equivalent_time_score(
    *,
    current_probe_loss: float,
    candidate_probe_loss: float,
    loss_improvement_rate: float,
    predicted_current_update_seconds: float,
    predicted_candidate_update_seconds: float,
    horizon_updates: int,
    discount: float = WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT,
) -> Dict[str, float]:
    if not 0.0 <= float(discount) <= 1.0:
        raise ValueError("equivalent-time gain discount must lie in [0, 1]")
    if not math.isfinite(loss_improvement_rate) or loss_improvement_rate <= 0.0:
        raise ValueError("loss_improvement_rate must be finite and positive")
    if horizon_updates < 0:
        raise ValueError("horizon_updates must be non-negative")
    probe_loss_gain = float(current_probe_loss) - float(candidate_probe_loss)
    raw_equivalent_time_gain = probe_loss_gain / float(loss_improvement_rate)
    credited_equivalent_time_gain = (
        float(discount) * raw_equivalent_time_gain
        if raw_equivalent_time_gain > 0.0
        else raw_equivalent_time_gain
    )
    extra_wall_time = (
        float(predicted_candidate_update_seconds)
        - float(predicted_current_update_seconds)
    ) * int(horizon_updates)
    score = extra_wall_time - credited_equivalent_time_gain
    return {
        "probe_loss_gain": probe_loss_gain,
        "raw_equivalent_time_gain_seconds": raw_equivalent_time_gain,
        "credited_equivalent_time_gain_seconds": credited_equivalent_time_gain,
        "extra_wall_time_seconds": extra_wall_time,
        "score_seconds": score,
    }


def _bootstrap_score_report(
    measurements: Sequence[Any],
) -> Tuple[Any, Tuple[Dict[str, object], ...]]:
    scored = []
    for measurement in measurements:
        feasible = math.isfinite(float(measurement.validation_loss))
        score = float(measurement.validation_loss) if feasible else float("inf")
        scored.append((measurement, feasible, score))
    feasible_scored = [item for item in scored if item[1]]
    if not feasible_scored:
        raise RuntimeError("no feasible PLASTIC DEPTH layer-count candidate")
    selected = min(
        feasible_scored,
        key=lambda item: (item[2], int(item[0].active_layers)),
    )[0]
    report = tuple(
        {
            "active_layers": int(measurement.active_layers),
            "validation_loss": float(measurement.validation_loss),
            "training_time": measurement.training_time,
            "peak_allocated_gib": measurement.peak_allocated_gib,
            "peak_reserved_gib": measurement.peak_reserved_gib,
            "feasible": bool(feasible),
            "score": float(score),
            "wall_time_algorithm": WALL_TIME_ALGORITHM,
            "wall_time_bootstrap": True,
            "wall_time_discount": WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT,
        }
        for measurement, feasible, score in scored
    )
    return selected, report


def _hold_score_report(
    measurements: Sequence[Any],
    *,
    current_count: int,
    reason: str,
) -> Tuple[Any, Tuple[Dict[str, object], ...]]:
    current = None
    report = []
    for measurement in measurements:
        is_current = int(measurement.active_layers) == int(current_count)
        if is_current:
            current = measurement
        report.append(
            {
                "active_layers": int(measurement.active_layers),
                "validation_loss": float(measurement.validation_loss),
                "training_time": measurement.training_time,
                "peak_allocated_gib": measurement.peak_allocated_gib,
                "peak_reserved_gib": measurement.peak_reserved_gib,
                "feasible": bool(is_current and math.isfinite(float(measurement.validation_loss))),
                "score": 0.0 if is_current else float("inf"),
                "wall_time_algorithm": WALL_TIME_ALGORITHM,
                "wall_time_bootstrap": False,
                "wall_time_discount": WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT,
                "wall_time_hold_reason": reason,
            }
        )
    if current is None or not math.isfinite(float(current.validation_loss)):
        raise RuntimeError("relative wall-time HOLD has no finite current-L candidate")
    return current, tuple(report)


def _choose_wall_time_equivalent_time_gain(
    trainer: Any,
    measurements: Sequence[Any],
) -> Tuple[Any, Tuple[Dict[str, object], ...]]:
    context = getattr(trainer, "_plastic_depth_inline_update_context", None)
    lattice = trainer._plastic_depth_lattice()
    if lattice is None:
        raise RuntimeError("PLASTIC DEPTH lattice absent during wall-time scoring")
    current_count = int(
        context["current_count"]
        if isinstance(context, Mapping) and "current_count" in context
        else lattice.current_active_layers
    )
    by_count = {int(item.active_layers): item for item in measurements}
    current = by_count.get(current_count)
    if current is None or not math.isfinite(float(current.validation_loss)):
        raise RuntimeError("relative wall-time scoring has no finite current-L probe loss")

    state = _runtime_state(trainer)
    timing_fit = _timing_fit(trainer)
    loss_fit = _loss_rate_fit(trainer)
    ready = timing_fit is not None and loss_fit is not None

    if not bool(state["activated"]):
        if not ready:
            return _bootstrap_score_report(measurements)
        state["activated"] = True
        state["activation_update"] = int(trainer.state.completed_updates) + 1
        trainer.state.plastic_depth_probe_histories = {}

    if timing_fit is None:
        return _hold_score_report(
            measurements,
            current_count=current_count,
            reason="timing_model_unavailable",
        )
    if loss_fit is None:
        return _hold_score_report(
            measurements,
            current_count=current_count,
            reason="loss_rate_unavailable",
        )

    slope = float(timing_fit["slope"])
    intercept = float(timing_fit["intercept"])
    loss_improvement_rate = float(loss_fit["loss_improvement_rate"])
    horizon_updates = _wall_time_horizon_updates(trainer.config)
    predicted_current = intercept + slope * current_count
    if not math.isfinite(predicted_current) or predicted_current <= 0.0:
        return _hold_score_report(
            measurements,
            current_count=current_count,
            reason="invalid_predicted_current_time",
        )

    scored = []
    report = []
    for measurement in measurements:
        count = int(measurement.active_layers)
        validation_loss = float(measurement.validation_loss)
        predicted_candidate = intercept + slope * count
        feasible = (
            math.isfinite(validation_loss)
            and math.isfinite(predicted_candidate)
            and predicted_candidate > 0.0
        )
        details: Dict[str, float] = {}
        score = float("inf")
        if feasible:
            details = _equivalent_time_score(
                current_probe_loss=float(current.validation_loss),
                candidate_probe_loss=validation_loss,
                loss_improvement_rate=loss_improvement_rate,
                predicted_current_update_seconds=predicted_current,
                predicted_candidate_update_seconds=predicted_candidate,
                horizon_updates=horizon_updates,
            )
            score = float(details["score_seconds"])
            scored.append((measurement, score))
        report.append(
            {
                "active_layers": count,
                "validation_loss": validation_loss,
                "training_time": (
                    predicted_candidate if feasible else None
                ),
                "peak_allocated_gib": measurement.peak_allocated_gib,
                "peak_reserved_gib": measurement.peak_reserved_gib,
                "feasible": feasible,
                "score": score,
                "wall_time_algorithm": WALL_TIME_ALGORITHM,
                "wall_time_bootstrap": False,
                "wall_time_discount": WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT,
                "wall_time_horizon_updates": horizon_updates,
                "loss_improvement_rate_per_second": loss_improvement_rate,
                "predicted_current_update_seconds": predicted_current,
                "predicted_candidate_update_seconds": (
                    predicted_candidate if feasible else None
                ),
                **details,
            }
        )
    if not scored:
        raise RuntimeError("no feasible PLASTIC DEPTH equivalent-time candidate")
    selected = min(
        scored,
        key=lambda item: (item[1], int(item[0].active_layers)),
    )[0]
    return selected, tuple(report)


def _choose_plastic_depth_candidate_v0531(
    measurements: Iterable[Any],
    *,
    objective: str,
    maximum_layers: int,
    cost_weight: float,
    reference_training_time: Optional[float],
    memory_budget_gib: Optional[float],
):
    resolved_measurements = tuple(measurements)
    if objective != "relative_training_wall_time":
        return _ORIGINAL_CHOOSE_PLASTIC_DEPTH_CANDIDATE(
            resolved_measurements,
            objective=objective,
            maximum_layers=maximum_layers,
            cost_weight=cost_weight,
            reference_training_time=reference_training_time,
            memory_budget_gib=memory_budget_gib,
        )
    trainer = _ACTIVE_TRAINER.get()
    if trainer is None:
        return _ORIGINAL_CHOOSE_PLASTIC_DEPTH_CANDIDATE(
            resolved_measurements,
            objective=objective,
            maximum_layers=maximum_layers,
            cost_weight=cost_weight,
            reference_training_time=reference_training_time,
            memory_budget_gib=memory_budget_gib,
        )
    return _choose_wall_time_equivalent_time_gain(trainer, resolved_measurements)


_plastic_depth.choose_plastic_depth_candidate = _choose_plastic_depth_candidate_v0531
_lookahead.choose_plastic_depth_candidate = _choose_plastic_depth_candidate_v0531
_trainer_step.choose_plastic_depth_candidate = _choose_plastic_depth_candidate_v0531


def _begin_plastic_depth_inline_update_v0531(self: Any):
    context = _ORIGINAL_BEGIN_INLINE_UPDATE(self)
    setattr(self, _PROBE_ATTRIBUTE, context is not None)
    lattice = self._plastic_depth_lattice() if bool(getattr(self.config, "plastic__enabled", False)) else None
    setattr(
        self,
        _COUNT_ATTRIBUTE,
        None if lattice is None else int(lattice.current_active_layers),
    )
    return context


def _plastic_depth_inline_probe_request_v0531(
    self: Any,
    targets: Any,
    context: Dict[str, Any],
):
    request = _ORIGINAL_INLINE_PROBE_REQUEST(self, targets, context)
    original_selector = request.selector

    def selector(candidates: Any) -> int:
        token = _ACTIVE_TRAINER.set(self)
        try:
            return int(original_selector(candidates))
        finally:
            _ACTIVE_TRAINER.reset(token)

    return replace(request, selector=selector)


_trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update = (
    _begin_plastic_depth_inline_update_v0531
)
_trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request = (
    _plastic_depth_inline_probe_request_v0531
)


def _plastic_depth_device_synchronize_v0531(self: Any) -> None:
    if str(getattr(getattr(self, "config", None), "plastic__layer_count_objective", "")) == "memory_budget":
        _ORIGINAL_PLASTIC_DEVICE_SYNCHRONIZE(self)
    return None


_trainer_step.TrainerStepMixin._plastic_depth_device_synchronize = (
    _plastic_depth_device_synchronize_v0531
)


def _record_ordinary_update_sample(
    trainer: Any,
    *,
    update_number: int,
    active_layers: int,
    elapsed_seconds: float,
    training_loss: float,
) -> None:
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        return
    state = _runtime_state(trainer)
    x = float(active_layers)
    y = float(elapsed_seconds)
    state["timing_n"] += 1
    state["timing_sum_x"] += x
    state["timing_sum_y"] += y
    state["timing_sum_x2"] += x * x
    state["timing_sum_xy"] += x * y
    state["timing_sum_y2"] += y * y
    state["timing_distinct_counts"].add(int(active_layers))
    state["last_timing_sample_seconds"] = y
    state["last_timing_sample_update"] = int(update_number)
    state["ordinary_elapsed_seconds"] += y

    fit = _timing_fit(trainer)
    state["last_timing_residual_seconds"] = (
        None
        if fit is None
        else y - (float(fit["intercept"]) + float(fit["slope"]) * x)
    )

    if (
        int(update_number) > int(getattr(trainer.config, "warmup_updates", 0))
        and math.isfinite(float(training_loss))
    ):
        state["loss_history"].append(
            (
                float(state["ordinary_elapsed_seconds"]),
                float(training_loss),
            )
        )


def _timed_with_wall_time_statistics(self: Any, function: Any):
    result, elapsed = _ORIGINAL_TIMED(self, function)
    function_name = str(getattr(function, "__name__", ""))
    if "train_one_update" not in function_name:
        return result, elapsed
    config = getattr(self, "config", None)
    if not bool(getattr(config, "plastic__enabled", False)):
        return result, elapsed
    if not bool(getattr(config, "plastic__do_learn_layer_count", False)):
        return result, elapsed
    if str(getattr(config, "plastic__runtime_phase", "fine")) != "fine":
        return result, elapsed
    if bool(getattr(self, _PROBE_ATTRIBUTE, False)):
        return result, elapsed
    if not isinstance(result, Mapping):
        return result, elapsed
    if float(result.get("skipped_update", 0.0)) != 0.0:
        return result, elapsed
    active_layers = getattr(self, _COUNT_ATTRIBUTE, None)
    if active_layers is None:
        lattice = self._plastic_depth_lattice()
        active_layers = None if lattice is None else int(lattice.current_active_layers)
    if active_layers is None:
        return result, elapsed
    training_loss = float(result.get("training_loss", float("nan")))
    update_number = int(result.get("completed_updates", self.state.completed_updates))
    _record_ordinary_update_sample(
        self,
        update_number=update_number,
        active_layers=int(active_layers),
        elapsed_seconds=float(elapsed),
        training_loss=training_loss,
    )
    return result, elapsed


_stage6.Stage6Trainer._timed = _timed_with_wall_time_statistics


def _wall_time_telemetry_values(
    trainer: Any,
    *,
    completed_update: Optional[int] = None,
) -> Dict[str, float | int]:
    config = getattr(trainer, "config", None)
    if not bool(getattr(config, "plastic__enabled", False)):
        return {}
    if not bool(getattr(config, "plastic__do_learn_layer_count", False)):
        return {}
    state = _runtime_state(trainer)
    timing_fit = _timing_fit(trainer)
    loss_fit = _loss_rate_fit(trainer)
    lattice = trainer._plastic_depth_lattice()
    current_layers = (
        0 if lattice is None else int(lattice.current_active_layers)
    )
    values: Dict[str, float | int] = {
        "plastic_wall_time_layer_count": current_layers,
        "plastic_wall_time_timing_n_observations": int(state["timing_n"]),
        "plastic_wall_time_timing_distinct_layer_counts": len(
            state["timing_distinct_counts"]
        ),
        "plastic_wall_time_timing_model_ready": int(timing_fit is not None),
        "plastic_wall_time_loss_rate_ready": int(loss_fit is not None),
        "plastic_wall_time_algorithm_ready": int(
            timing_fit is not None and loss_fit is not None
        ),
        "plastic_wall_time_algorithm_activated": int(bool(state["activated"])),
        "plastic_wall_time_discount": WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT,
        "plastic_wall_time_horizon_updates": _wall_time_horizon_updates(
            trainer.config
        ),
    }
    if timing_fit is not None:
        values.update(
            {
                "plastic_wall_time_fitted_seconds_per_layer": float(
                    timing_fit["slope"]
                ),
                "plastic_wall_time_fitted_intercept_seconds": float(
                    timing_fit["intercept"]
                ),
                "plastic_wall_time_timing_r_squared": float(
                    timing_fit["r_squared"]
                ),
                "plastic_wall_time_slope_standard_error_seconds": float(
                    timing_fit["slope_standard_error"]
                ),
            }
        )
    if loss_fit is not None:
        values.update(
            {
                "plastic_wall_time_loss_improvement_per_second": float(
                    loss_fit["loss_improvement_rate"]
                ),
                "plastic_wall_time_loss_rate_r_squared": float(
                    loss_fit["r_squared"]
                ),
                "plastic_wall_time_loss_rate_observations": int(
                    loss_fit["observations"]
                ),
            }
        )
    last_sample_update = state.get("last_timing_sample_update")
    if (
        completed_update is not None
        and last_sample_update is not None
        and int(last_sample_update) == int(completed_update)
    ):
        last_seconds = state.get("last_timing_sample_seconds")
        residual_seconds = state.get("last_timing_residual_seconds")
        if last_seconds is not None and math.isfinite(float(last_seconds)):
            values["plastic_wall_time_update_seconds"] = float(last_seconds)
        if residual_seconds is not None and math.isfinite(float(residual_seconds)):
            values["plastic_wall_time_residual_seconds"] = float(residual_seconds)
    return values


_TELEMETRY_NAMES = {
    "plastic_wall_time_layer_count": "plastic/timing/layer_count",
    "plastic_wall_time_update_seconds": "plastic/timing/update_ms",
    "plastic_wall_time_fitted_seconds_per_layer": "plastic/timing/fitted_ms_per_layer",
    "plastic_wall_time_fitted_intercept_seconds": "plastic/timing/fitted_intercept_ms",
    "plastic_wall_time_timing_r_squared": "plastic/timing/r_squared",
    "plastic_wall_time_residual_seconds": "plastic/timing/residual_ms",
    "plastic_wall_time_timing_n_observations": "plastic/timing/n_observations",
    "plastic_wall_time_timing_distinct_layer_counts": "plastic/timing/distinct_layer_counts",
    "plastic_wall_time_slope_standard_error_seconds": "plastic/timing/slope_standard_error_ms",
    "plastic_wall_time_loss_improvement_per_second": "plastic/wall_time/loss_improvement_per_second",
    "plastic_wall_time_loss_rate_r_squared": "plastic/wall_time/loss_rate_r_squared",
    "plastic_wall_time_loss_rate_observations": "plastic/wall_time/loss_rate_observations",
    "plastic_wall_time_timing_model_ready": "plastic/wall_time/timing_model_ready",
    "plastic_wall_time_loss_rate_ready": "plastic/wall_time/loss_rate_ready",
    "plastic_wall_time_algorithm_ready": "plastic/wall_time/algorithm_ready",
    "plastic_wall_time_algorithm_activated": "plastic/wall_time/algorithm_activated",
    "plastic_wall_time_discount": "plastic/wall_time/equivalent_time_gain_discount",
    "plastic_wall_time_horizon_updates": "plastic/wall_time/horizon_updates",
}
_MILLISECOND_SOURCE_KEYS = {
    "plastic_wall_time_update_seconds",
    "plastic_wall_time_fitted_seconds_per_layer",
    "plastic_wall_time_fitted_intercept_seconds",
    "plastic_wall_time_residual_seconds",
    "plastic_wall_time_slope_standard_error_seconds",
}


def _training_metrics_with_wall_time(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = _ORIGINAL_TRAINING_METRICS(payload)
    for source_name, destination_name in _TELEMETRY_NAMES.items():
        value = payload.get(source_name)
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        if source_name in _MILLISECOND_SOURCE_KEYS:
            numeric *= 1000.0
        if isinstance(value, int) and source_name not in _MILLISECOND_SOURCE_KEYS:
            metrics[destination_name] = int(value)
        else:
            metrics[destination_name] = numeric
    return metrics


_telemetry._training_metrics = _training_metrics_with_wall_time


def _start_wandb_with_plastic_wall_time_metrics(self: Any) -> None:
    _ORIGINAL_START_WANDB(self)
    if self.run is None:
        return
    define_metric = (
        self.run.define_metric
        if hasattr(self.run, "define_metric")
        else self.module.define_metric
    )
    define_metric("plastic/*", step_metric="optimizer/update")


_telemetry.WandbTelemetry._start_wandb = _start_wandb_with_plastic_wall_time_metrics


def _attach_telemetry_with_wall_time(trainer: Any, telemetry: Any) -> None:
    _ORIGINAL_ATTACH_TELEMETRY(trainer, telemetry)
    attached_progress = trainer._print_progress

    def progress(run_id: str, event: str, **payload: Any) -> None:
        if event == "optimizer_progress":
            completed_update = int(payload.get("completed_updates", 0))
            payload.update(
                _wall_time_telemetry_values(
                    trainer,
                    completed_update=completed_update,
                )
            )
        attached_progress(run_id, event, **payload)

    trainer._print_progress = progress


_telemetry.attach_telemetry = _attach_telemetry_with_wall_time


def _zero_pad_score_z_fixed_values(line: str) -> str:
    def replace_vector(match: re.Match[str]) -> str:
        body = _SIGNED_ONE_DIGIT_FIXED.sub(
            lambda value: (
                f"{value.group('sign')}0"
                f"{value.group('integer')}{value.group('fraction')}"
            ),
            match.group("body"),
        )
        return f"{match.group('prefix')}{body}{match.group('close')}"

    return _SCORE_Z_VECTOR.sub(replace_vector, line)


def _finalize_console_v0531(line: str) -> str:
    line = line.replace("grad norm=", "g nrm=")

    def replace_direction(match: re.Match[str]) -> str:
        outcome = match.group("outcome")
        rendered_outcome = {
            "stet": "●",
            "-": "●",
            "L": "↓",
            "R": "↑",
        }[outcome]
        if outcome in {"L", "R"}:
            rendered_outcome = (
                f"{_constants.BOLD}{_constants.YELLOW}"
                f"{rendered_outcome}{_constants.R}"
            )
        return (
            f"↓|↑|? =[{match.group('left')}/{match.group('right')}/"
            f"{match.group('ambiguous')}]/{match.group('total')}"
            f"=>{rendered_outcome}"
        )

    line = _DIRECTION_SUMMARY.sub(replace_direction, line)

    def replace_layers_sampled(match: re.Match[str]) -> str:
        return f"layers = {match.group('count'):<4}\tsampled ="

    line = _LAYERS_SAMPLED.sub(replace_layers_sampled, line)
    return _zero_pad_score_z_fixed_values(line)


def _format_progress_line_v0531(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    return _finalize_console_v0531(
        _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    )


_stage6.format_progress_line = _format_progress_line_v0531


__all__ = [
    "WALL_TIME_ALGORITHM",
    "WALL_TIME_EQUIVALENT_TIME_GAIN_DISCOUNT",
    "_equivalent_time_score",
    "_finalize_console_v0531",
    "_linear_fit_points",
    "_wall_time_horizon_updates",
    "_wall_time_telemetry_values",
]
# ^^^ THOG
