# vvv THOG
"""PLASTIC v0.54 Theil-Sen/Kendall wall-time-gradient layer-count decision algorithm."""

from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import sys
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from . import plastic_depth_controller as _controller
from . import plastic_depth_directional_coherence_patch as _directional
from . import plastic_depth_lookahead_patch as _lookahead
from . import plastic_depth_wall_time_equivalent_time_gain_patch as _wall_time
from . import run_config as _run_config
from . import stage6_trainer as _stage6
from . import trainer_checkpoint_resume as _checkpoint_resume
from . import trainer_step as _trainer_step
from . import training_config as _training_config


GRADIENT_ALGORITHM = "wall_time__gradient__theil_sen_kendall_slope_tau"
LEGACY_DIRECTIONAL_ALGORITHM = "directional_coherence"
DECISION_ALGORITHMS = (LEGACY_DIRECTIONAL_ALGORITHM, GRADIENT_ALGORITHM)
DEFAULT_MINIMUM_ABSOLUTE_KENDALL_TAU = 0.5

_ALGORITHM_KEY = "plastic__layer_count_decision_algorithm"
_TAU_KEY = "plastic__layer_count_gradient__minimum_absolute_kendall_tau"
_ALGORITHM_OPTION = "--plastic__layer_count_decision_algorithm"
_TAU_OPTION = "--plastic__layer_count_gradient__minimum_absolute_kendall_tau"
_ALGORITHM_ENV = "THOG2_PLASTIC_LAYER_COUNT_DECISION_ALGORITHM"
_TAU_ENV = "THOG2_PLASTIC_LAYER_COUNT_GRADIENT__MINIMUM_ABSOLUTE_KENDALL_TAU"
_GRADIENT_HISTORY_SUFFIX = "@TSK"
_DIRECTION_LEFT = -1.0
_DIRECTION_AMBIGUOUS = 0.0
_DIRECTION_RIGHT = 1.0
_DIRECTION_MARKER = re.compile(r"(?P<spacing>\s+)(?P<marker>↓\|↑\|\?)")


def _runtime_algorithm() -> str:
    value = os.environ.get(_ALGORITHM_ENV, LEGACY_DIRECTIONAL_ALGORITHM).strip()
    if value not in DECISION_ALGORITHMS:
        raise ValueError(
            f"{_ALGORITHM_KEY} must be one of {DECISION_ALGORITHMS}; got {value!r}"
        )
    return value


def _runtime_minimum_absolute_kendall_tau() -> float:
    raw = os.environ.get(_TAU_ENV, str(DEFAULT_MINIMUM_ABSOLUTE_KENDALL_TAU))
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{_TAU_KEY} must be finite and lie in [0, 1]; got {raw!r}")
    return value


def _set_runtime_algorithm(value: str) -> None:
    resolved = str(value).strip()
    if resolved not in DECISION_ALGORITHMS:
        raise ValueError(
            f"{_ALGORITHM_KEY} must be one of {DECISION_ALGORITHMS}; got {value!r}"
        )
    os.environ[_ALGORITHM_ENV] = resolved


def _set_runtime_tau(value: Any) -> None:
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{_TAU_KEY} must be finite and lie in [0, 1]; got {value!r}")
    os.environ[_TAU_ENV] = repr(resolved)


# vvv THOG add the selectable algorithm and Kendall threshold without perturbing the established parser surface
_ORIGINAL_PARSE_ARGS = argparse.ArgumentParser.parse_args
_ORIGINAL_PARSE_KNOWN_ARGS = argparse.ArgumentParser.parse_known_args
_ORIGINAL_FORMAT_HELP = argparse.ArgumentParser.format_help


def _strip_gradient_options(arguments: Optional[Sequence[str]]) -> Tuple[list[str], Optional[str], Optional[float]]:
    source = list(sys.argv[1:] if arguments is None else arguments)
    remaining: list[str] = []
    algorithm: Optional[str] = None
    tau: Optional[float] = None
    index = 0
    while index < len(source):
        argument = source[index]
        if argument == _ALGORITHM_OPTION:
            if index + 1 >= len(source):
                raise SystemExit(f"{_ALGORITHM_OPTION} requires a value")
            algorithm = source[index + 1]
            index += 2
            continue
        if argument.startswith(_ALGORITHM_OPTION + "="):
            algorithm = argument.split("=", 1)[1]
            index += 1
            continue
        if argument == _TAU_OPTION:
            if index + 1 >= len(source):
                raise SystemExit(f"{_TAU_OPTION} requires a value")
            try:
                tau = float(source[index + 1])
            except ValueError as error:
                raise SystemExit(f"{_TAU_OPTION} requires a finite number in [0, 1]") from error
            index += 2
            continue
        if argument.startswith(_TAU_OPTION + "="):
            try:
                tau = float(argument.split("=", 1)[1])
            except ValueError as error:
                raise SystemExit(f"{_TAU_OPTION} requires a finite number in [0, 1]") from error
            index += 1
            continue
        remaining.append(argument)
        index += 1
    return remaining, algorithm, tau


def _attach_gradient_options(namespace: Any, algorithm: Optional[str], tau: Optional[float]) -> Any:
    try:
        if algorithm is not None:
            _set_runtime_algorithm(algorithm)
        if tau is not None:
            _set_runtime_tau(tau)
        setattr(namespace, _ALGORITHM_KEY, _runtime_algorithm())
        setattr(namespace, _TAU_KEY, _runtime_minimum_absolute_kendall_tau())
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return namespace


def _parse_args_with_gradient(self: argparse.ArgumentParser, args=None, namespace=None):
    remaining, algorithm, tau = _strip_gradient_options(args)
    parsed = _ORIGINAL_PARSE_ARGS(self, remaining, namespace)
    return _attach_gradient_options(parsed, algorithm, tau)


def _parse_known_args_with_gradient(self: argparse.ArgumentParser, args=None, namespace=None):
    remaining, algorithm, tau = _strip_gradient_options(args)
    parsed, extras = _ORIGINAL_PARSE_KNOWN_ARGS(self, remaining, namespace)
    return _attach_gradient_options(parsed, algorithm, tau), extras


def _format_help_with_gradient(self: argparse.ArgumentParser) -> str:
    rendered = _ORIGINAL_FORMAT_HELP(self)
    if _ALGORITHM_OPTION in rendered:
        return rendered
    if not any(action.dest == "plastic__enabled" for action in self._actions):
        return rendered
    return (
        rendered.rstrip()
        + f"\n  {_ALGORITHM_OPTION} ALGORITHM\n"
        + f"                        {LEGACY_DIRECTIONAL_ALGORITHM} (default) or {GRADIENT_ALGORITHM}\n"
        + f"  {_TAU_OPTION} X\n"
        + f"                        minimum absolute Kendall tau-b for the gradient algorithm; default {DEFAULT_MINIMUM_ABSOLUTE_KENDALL_TAU}\n"
    )


argparse.ArgumentParser.parse_args = _parse_args_with_gradient
argparse.ArgumentParser.parse_known_args = _parse_known_args_with_gradient
argparse.ArgumentParser.format_help = _format_help_with_gradient
# ^^^ THOG


# vvv THOG expose/persist the new material controls only when the gradient algorithm is selected, preserving legacy identity

def _algorithm_property(_self: Any) -> str:
    return _runtime_algorithm()


def _tau_property(_self: Any) -> float:
    return _runtime_minimum_absolute_kendall_tau()


if not hasattr(_training_config.TrainingConfig, _ALGORITHM_KEY):
    setattr(_training_config.TrainingConfig, _ALGORITHM_KEY, property(_algorithm_property))
if not hasattr(_training_config.TrainingConfig, _TAU_KEY):
    setattr(_training_config.TrainingConfig, _TAU_KEY, property(_tau_property))
if not hasattr(_run_config.OwtRunConfig, _ALGORITHM_KEY):
    setattr(_run_config.OwtRunConfig, _ALGORITHM_KEY, property(_algorithm_property))
if not hasattr(_run_config.OwtRunConfig, _TAU_KEY):
    setattr(_run_config.OwtRunConfig, _TAU_KEY, property(_tau_property))


_ORIGINAL_TRAINING_POST_INIT = _training_config.TrainingConfig.__post_init__
_ORIGINAL_TRAINING_PERSISTENT_DICT = _training_config.TrainingConfig.persistent_dict
_ORIGINAL_TRAINING_COMPACT_IDENTITY = _training_config.TrainingConfig.compact_identity_metadata
_ORIGINAL_RUN_PERSISTENT_DICT = _run_config.OwtRunConfig.persistent_dict
_ORIGINAL_RUN_COMPACT_IDENTITY = _run_config.OwtRunConfig.compact_identity
_ORIGINAL_NORMALIZE_PLASTIC_CONFIG = _training_config.normalize_plastic_v0541_config_fields


def _training_post_init_with_gradient(self: Any) -> None:
    _ORIGINAL_TRAINING_POST_INIT(self)
    algorithm = _runtime_algorithm()
    tau = _runtime_minimum_absolute_kendall_tau()
    if algorithm == GRADIENT_ALGORITHM:
        if not bool(self.plastic__enabled) or not bool(self.plastic__do_learn_layer_count):
            raise ValueError(f"{GRADIENT_ALGORITHM} requires learned-count PLASTIC DEPTH")
        if str(self.plastic__layer_count_objective) != "relative_training_wall_time":
            raise ValueError(
                f"{GRADIENT_ALGORITHM} requires plastic__layer_count_objective=relative_training_wall_time"
            )
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"{_TAU_KEY} must lie in [0, 1]")


def _persistent_with_gradient(original: Any, config: Any) -> Dict[str, Any]:
    values = original(config)
    if bool(getattr(config, "plastic__enabled", False)) and _runtime_algorithm() == GRADIENT_ALGORITHM:
        values[_ALGORITHM_KEY] = GRADIENT_ALGORITHM
        values[_TAU_KEY] = _runtime_minimum_absolute_kendall_tau()
    return values


def _training_persistent_dict_with_gradient(self: Any) -> Dict[str, Any]:
    return _persistent_with_gradient(_ORIGINAL_TRAINING_PERSISTENT_DICT, self)


def _run_persistent_dict_with_gradient(self: Any) -> Dict[str, Any]:
    return _persistent_with_gradient(_ORIGINAL_RUN_PERSISTENT_DICT, self)


def _identity_with_gradient(identity: Dict[str, Any], *, plastic_enabled: bool) -> Dict[str, Any]:
    if not plastic_enabled or _runtime_algorithm() != GRADIENT_ALGORITHM:
        return identity
    plastic_identity = identity.get("plastic_depth")
    if not isinstance(plastic_identity, Mapping):
        return identity
    updated = dict(identity)
    updated["plastic_depth"] = {
        **dict(plastic_identity),
        _ALGORITHM_KEY: GRADIENT_ALGORITHM,
        _TAU_KEY: _runtime_minimum_absolute_kendall_tau(),
    }
    return updated


def _training_compact_identity_with_gradient(self: Any) -> Dict[str, Any]:
    return _identity_with_gradient(
        _ORIGINAL_TRAINING_COMPACT_IDENTITY(self),
        plastic_enabled=bool(self.plastic__enabled),
    )


def _run_compact_identity_with_gradient(self: Any) -> Dict[str, Any]:
    return _identity_with_gradient(
        _ORIGINAL_RUN_COMPACT_IDENTITY(self),
        plastic_enabled=bool(self.plastic__enabled),
    )


def _normalize_plastic_config_with_gradient(values: Mapping[str, Any]) -> Dict[str, Any]:
    source = dict(values)
    algorithm = source.pop(_ALGORITHM_KEY, None)
    tau = source.pop(_TAU_KEY, None)
    if algorithm is not None:
        _set_runtime_algorithm(str(algorithm))
    if tau is not None:
        _set_runtime_tau(tau)
    return _ORIGINAL_NORMALIZE_PLASTIC_CONFIG(source)


_training_config.TrainingConfig.__post_init__ = _training_post_init_with_gradient
_training_config.TrainingConfig.persistent_dict = _training_persistent_dict_with_gradient
_training_config.TrainingConfig.compact_identity_metadata = _training_compact_identity_with_gradient
_training_config.normalize_plastic_v0541_config_fields = _normalize_plastic_config_with_gradient
_run_config.OwtRunConfig.persistent_dict = _run_persistent_dict_with_gradient
_run_config.OwtRunConfig.compact_identity = _run_compact_identity_with_gradient
_checkpoint_resume.normalize_plastic_v0541_config_fields = _normalize_plastic_config_with_gradient
# ^^^ THOG


# vvv THOG dependency-free robust local-gradient statistics over tiny candidate sets
def theil_sen_slope(points: Sequence[Tuple[float, float]]) -> float:
    finite = tuple((float(x), float(y)) for x, y in points if math.isfinite(float(x)) and math.isfinite(float(y)))
    slopes = []
    for index, (x_i, y_i) in enumerate(finite):
        for x_j, y_j in finite[index + 1 :]:
            dx = x_j - x_i
            if dx == 0.0:
                continue
            slopes.append((y_j - y_i) / dx)
    if not slopes:
        return float("nan")
    return float(statistics.median(slopes))


def kendall_tau_b(points: Sequence[Tuple[float, float]]) -> Tuple[float, int, int, int, int]:
    finite = tuple((float(x), float(y)) for x, y in points if math.isfinite(float(x)) and math.isfinite(float(y)))
    concordant = 0
    discordant = 0
    x_ties = 0
    y_ties = 0
    for index, (x_i, y_i) in enumerate(finite):
        for x_j, y_j in finite[index + 1 :]:
            dx = x_j - x_i
            dy = y_j - y_i
            if dx == 0.0 and dy == 0.0:
                x_ties += 1
                y_ties += 1
            elif dx == 0.0:
                x_ties += 1
            elif dy == 0.0:
                y_ties += 1
            elif dx * dy > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        float(concordant + discordant + x_ties)
        * float(concordant + discordant + y_ties)
    )
    tau = (
        float(concordant - discordant) / denominator
        if denominator > 0.0
        else float("nan")
    )
    return tau, concordant, discordant, x_ties, y_ties
# ^^^ THOG


# vvv THOG one probe becomes L/R/A only from equivalent-time scores, robust gradient coherence and the adjacent economic test
def _gradient_direction_history_key(current_count: int) -> str:
    return f"{int(current_count)}:{_GRADIENT_HISTORY_SUFFIX}"


def _equivalent_time_scores(score_report: Sequence[Mapping[str, object]]) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for item in score_report:
        if str(item.get("wall_time_algorithm", "")) != _wall_time.WALL_TIME_ALGORITHM:
            continue
        if bool(item.get("wall_time_bootstrap", False)):
            continue
        if not bool(item.get("feasible", False)):
            continue
        try:
            count = int(item["active_layers"])
            score = float(item.get("score", float("inf")))
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(score):
            result[count] = score
    return result


def _gradient_probe_classification(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    minimum_absolute_kendall_tau: float,
) -> Dict[str, Any]:
    threshold = float(minimum_absolute_kendall_tau)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{_TAU_KEY} must be finite and lie in [0, 1]")
    current = int(current_count)
    score_by_count = _equivalent_time_scores(score_report)
    decision_counts = tuple(sorted(count for count in score_by_count if count <= current + 1))
    points = tuple((float(count - current), float(score_by_count[count])) for count in decision_counts)
    slope = theil_sen_slope(points) if len(points) >= 2 else float("nan")
    tau, concordant, discordant, x_ties, y_ties = kendall_tau_b(points) if len(points) >= 2 else (float("nan"), 0, 0, 0, 0)
    current_score = score_by_count.get(current)
    left_score = score_by_count.get(current - 1)
    right_score = score_by_count.get(current + 1)
    vote = _DIRECTION_AMBIGUOUS
    if current_score is not None and math.isfinite(slope) and math.isfinite(tau):
        if (
            left_score is not None
            and left_score < current_score
            and slope > 0.0
            and tau >= threshold
        ):
            vote = _DIRECTION_LEFT
        elif (
            right_score is not None
            and right_score < current_score
            and slope < 0.0
            and tau <= -threshold
        ):
            vote = _DIRECTION_RIGHT
    all_offsets = _directional._candidate_offsets(
        current_count=current,
        score_report=score_report,
    )
    return {
        "algorithm": GRADIENT_ALGORITHM,
        "decision_fit_offsets": tuple(count - current for count in decision_counts),
        "candidate_offsets": all_offsets,
        "theil_sen_slope_seconds_per_layer": slope,
        "kendall_tau": tau,
        "minimum_absolute_kendall_tau": threshold,
        "left_adjacent_score_seconds": left_score,
        "current_score_seconds": current_score,
        "right_adjacent_score_seconds": right_score,
        "per_probe_vote": vote,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "x_tied_pairs": x_ties,
        "y_tied_pairs": y_ties,
        "fit_point_count": len(points),
    }


def _updated_histories_and_gradient_direction(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    histories: Mapping[str, Sequence[float]],
    noise_window: int,
    minimum_absolute_kendall_tau: float,
) -> Tuple[Dict[str, Tuple[float, ...]], Dict[str, Any]]:
    resolved_window = _directional._positive_integer(
        noise_window,
        name="plastic__layer_count_probe__window_size_as_number_of_probes",
    )
    updated_histories: Dict[str, Tuple[float, ...]] = {}
    for key, values in histories.items():
        resolved = tuple(float(value) for value in values[-resolved_window:])
        if not all(math.isfinite(value) for value in resolved):
            raise ValueError(f"PLASTIC DEPTH paired-score history {key!r} contains a non-finite value")
        updated_histories[str(key)] = resolved

    score_by_count = _equivalent_time_scores(score_report)
    current = int(current_count)
    current_score = score_by_count.get(current)
    candidate_offsets = _directional._candidate_offsets(current_count=current, score_report=score_report)
    if current_score is not None:
        for offset in candidate_offsets:
            candidate_score = score_by_count.get(current + offset)
            if candidate_score is None:
                continue
            key = _directional._history_key(current, offset)
            values = list(updated_histories.get(key, ()))
            values.append(float(candidate_score - current_score))
            updated_histories[key] = tuple(values[-resolved_window:])

    probe = _gradient_probe_classification(
        current_count=current,
        score_report=score_report,
        minimum_absolute_kendall_tau=minimum_absolute_kendall_tau,
    )
    direction_key = _gradient_direction_history_key(current)
    direction_values = list(updated_histories.get(direction_key, ()))
    direction_values.append(float(probe["per_probe_vote"]))
    updated_histories[direction_key] = tuple(direction_values[-resolved_window:])
    direction_values = list(updated_histories[direction_key])
    left_votes, right_votes, ambiguous_votes = _directional._direction_vote_counts(direction_values)
    conclusion = _directional._direction_conclusion(direction_values, resolved_window)

    left_offsets = tuple(offset for offset in candidate_offsets if offset < 0)
    right_offsets = tuple(offset for offset in candidate_offsets if offset > 0)
    left_win_counts = tuple(
        sum(value < 0.0 for value in updated_histories.get(_directional._history_key(current, offset), ()))
        for offset in left_offsets
    )
    right_win_counts = tuple(
        sum(value < 0.0 for value in updated_histories.get(_directional._history_key(current, offset), ()))
        for offset in right_offsets
    )
    report = {
        **probe,
        "direction_history": tuple(direction_values),
        "direction_window_votes": tuple(
            "L" if value == _DIRECTION_LEFT else "R" if value == _DIRECTION_RIGHT else "A"
            for value in direction_values
        ),
        "left_votes": left_votes,
        "right_votes": right_votes,
        "ambiguous_votes": ambiguous_votes,
        "vote_total": len(direction_values),
        "conclusion": conclusion,
        "direction_conclusion": conclusion,
        "left_offsets": left_offsets,
        "right_offsets": right_offsets,
        "left_win_counts": left_win_counts,
        "right_win_counts": right_win_counts,
    }
    return updated_histories, report


def _updated_histories_and_direction_dispatch(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    histories: Mapping[str, Sequence[float]],
    noise_window: int,
    extrapolation_weight: float,
) -> Tuple[Dict[str, Tuple[float, ...]], Dict[str, Any]]:
    trainer = _wall_time._ACTIVE_TRAINER.get()
    algorithm = _runtime_algorithm() if trainer is not None else LEGACY_DIRECTIONAL_ALGORITHM
    if algorithm != GRADIENT_ALGORITHM:
        return _ORIGINAL_UPDATED_HISTORIES_AND_DIRECTION(
            current_count=current_count,
            score_report=score_report,
            histories=histories,
            noise_window=noise_window,
            extrapolation_weight=extrapolation_weight,
        )
    return _updated_histories_and_gradient_direction(
        current_count=current_count,
        score_report=score_report,
        histories=histories,
        noise_window=noise_window,
        minimum_absolute_kendall_tau=_runtime_minimum_absolute_kendall_tau(),
    )
# ^^^ THOG


# vvv THOG preserve all existing final safety gates while replacing only per-probe direction classification in the selected mode
def choose_plastic_depth_count_with_theil_sen_kendall(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    histories: Mapping[str, Sequence[float]],
    noise_window: int,
    noise_lambda: float,
    update_number: int,
    last_count_change_update: int,
    update_brake: int,
    max_step: int = 1,
    minimum_observations: Optional[int] = None,
    minimum_absolute_kendall_tau: float = DEFAULT_MINIMUM_ABSOLUTE_KENDALL_TAU,
) -> _controller.PlasticDepthRobustCountDecision:
    del minimum_observations
    resolved_window = _directional._positive_integer(
        noise_window,
        name="plastic__layer_count_probe__window_size_as_number_of_probes",
    )
    resolved_max_step = _directional._positive_integer(
        max_step,
        name="plastic__layer_count__max_allowable_layer_change",
    )
    if not math.isfinite(noise_lambda) or noise_lambda < 0.0:
        raise ValueError("noise_lambda must be finite and non-negative")
    if update_number < 1:
        raise ValueError("update_number must be positive")
    if update_brake < 0:
        raise ValueError("update_brake must be non-negative")

    current = int(current_count)
    score_by_count = _equivalent_time_scores(score_report)
    current_score = score_by_count.get(current)
    updated_histories, direction_report = _updated_histories_and_gradient_direction(
        current_count=current,
        score_report=score_report,
        histories=histories,
        noise_window=resolved_window,
        minimum_absolute_kendall_tau=minimum_absolute_kendall_tau,
    )
    brake_active = (
        update_brake > 0
        and last_count_change_update >= 0
        and update_number - last_count_change_update < update_brake
    )
    conclusion = str(direction_report["conclusion"])
    evidence = []
    passing = []
    for offset in _directional._candidate_offsets(current_count=current, score_report=score_report):
        candidate_count = current + offset
        candidate_score = score_by_count.get(candidate_count)
        feasible = current_score is not None and candidate_score is not None
        paired_difference: Optional[float] = None
        median: Optional[float] = None
        mad: Optional[float] = None
        sigma: Optional[float] = None
        standardized: Optional[float] = None
        significant = False
        key = _directional._history_key(current, offset)
        values = tuple(updated_histories.get(key, ()))
        if feasible and values:
            paired_difference = float(candidate_score - current_score)
            median, mad, sigma = _directional._robust_scale(values)
            ready = len(values) >= resolved_window
            improving_observations = sum(value < 0.0 for value in values)
            improving_majority = improving_observations * 2 > len(values)
            if ready:
                standardized = -median / sigma
            local_significant = (
                ready
                and median < -noise_lambda * sigma
                and paired_difference < 0.0
                and improving_majority
            )
            directional_permitted = (
                (offset < 0 and conclusion == "L")
                or (offset == 1 and conclusion == "R")
            )
            significant = local_significant and directional_permitted
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

    selected_count = current
    decision_histories = updated_histories
    if passing:
        _, winning_offset, _ = max(passing, key=lambda item: (item[0], -item[2]))
        committed_offset = max(-resolved_max_step, min(resolved_max_step, int(winning_offset)))
        selected_count = current + committed_offset
        decision_histories = {}

    return _controller.PlasticDepthRobustCountDecision(
        selected_count=selected_count,
        current_count=current,
        update_number=int(update_number),
        brake_active=brake_active,
        last_count_change_update=int(last_count_change_update),
        histories=decision_histories,
        evidence=tuple(evidence),
    )


def _choose_count_dispatch(*, max_step: int = 1, extrapolation_weight: float = 0.8, **kwargs: Any):
    trainer = _wall_time._ACTIVE_TRAINER.get()
    algorithm = _runtime_algorithm() if trainer is not None else LEGACY_DIRECTIONAL_ALGORITHM
    if algorithm != GRADIENT_ALGORITHM:
        return _ORIGINAL_DIRECTIONAL_SELECTOR(
            max_step=max_step,
            extrapolation_weight=extrapolation_weight,
            **kwargs,
        )
    resolved_max_step = int(
        getattr(
            trainer.config,
            "plastic__layer_count__max_allowable_layer_change",
            max_step,
        )
    )
    return choose_plastic_depth_count_with_theil_sen_kendall(
        max_step=resolved_max_step,
        minimum_absolute_kendall_tau=_runtime_minimum_absolute_kendall_tau(),
        **kwargs,
    )


_ORIGINAL_DIRECTIONAL_SELECTOR = _directional.choose_plastic_depth_count_with_directional_coherence
_ORIGINAL_UPDATED_HISTORIES_AND_DIRECTION = _directional._updated_histories_and_direction
_directional._updated_histories_and_direction = _updated_histories_and_direction_dispatch
_lookahead.choose_plastic_depth_count_with_exact_radius = _choose_count_dispatch
_controller.choose_plastic_depth_count_with_mad = _choose_count_dispatch
_trainer_step.choose_plastic_depth_count_with_mad = _choose_count_dispatch
# ^^^ THOG


# vvv THOG persist reconstructable gradient decision telemetry and augment the existing count audit
_ORIGINAL_COMMIT_INLINE_UPDATE = _trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update


def _commit_plastic_depth_inline_update_with_gradient(
    self: Any,
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    transition = _ORIGINAL_COMMIT_INLINE_UPDATE(self, context)
    if context is None or _runtime_algorithm() != GRADIENT_ALGORITHM:
        return transition
    report = context.get("plastic_directional_report")
    decision = context.get("decision")
    if not isinstance(report, Mapping) or decision is None or report.get("algorithm") != GRADIENT_ALGORITHM:
        return transition
    payload = dict(report)
    payload.update(
        {
            "probe_id": context.get("plastic_probe_sequence"),
            "update_number": int(decision.update_number),
            "current_count": int(context["current_count"]),
            "selected_count": int(decision.selected_count),
            "brake_active": bool(decision.brake_active),
            "direction_window_probe_ids": tuple(int(value) for value in context.get("plastic_probe_provenance", ())),
        }
    )
    self._record("plastic_depth_gradient_decision", **payload)
    rows = getattr(self, "plastic_depth_count_audit", None)
    if isinstance(rows, list) and rows:
        row = rows[-1]
        if int(row.get("update_number", -1)) == int(decision.update_number):
            row["gradient_decision"] = payload
    return transition


_trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update = (
    _commit_plastic_depth_inline_update_with_gradient
)
# ^^^ THOG


# vvv THOG compact console diagnostics identify economic Theil-Sen units and Kendall coherence on the actual probe row
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line


def _latest_gradient_report(trainer: Any) -> Optional[Dict[str, Any]]:
    for event in reversed(getattr(trainer, "events", ())):
        if event.name == "plastic_depth_gradient_decision":
            return dict(event.payload)
    return None


def _prepare_console_progress_payload_with_gradient(
    self: Any,
    event: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    if _runtime_algorithm() != GRADIENT_ALGORITHM or event not in {"optimizer_progress", "evaluation_completed"}:
        return values
    report = _latest_gradient_report(self)
    if report is None:
        return values
    try:
        completed_updates = int(values.get("completed_updates", payload.get("completed_updates")))
    except (TypeError, ValueError):
        return values
    if completed_updates != int(report.get("update_number", -1)):
        return values
    values["plastic_gradient_ts"] = report.get("theil_sen_slope_seconds_per_layer")
    values["plastic_gradient_tau"] = report.get("kendall_tau")
    return values


def _format_gradient_value(value: Any, *, digits: int) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(numeric):
        return "-"
    return f"{numeric:+.{digits}f}"


def _format_progress_line_with_gradient(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    local_payload = dict(payload)
    ts = local_payload.pop("plastic_gradient_ts", None)
    tau = local_payload.pop("plastic_gradient_tau", None)
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, local_payload)
    if _runtime_algorithm() != GRADIENT_ALGORITHM or (ts is None and tau is None):
        return line
    diagnostic = (
        f"ts={_format_gradient_value(ts, digits=3)}s/layer "
        f"tau={_format_gradient_value(tau, digits=2)}"
    )
    match = _DIRECTION_MARKER.search(line)
    if match is None:
        return f"{line}  {diagnostic}"
    return line[: match.start()] + f"  {diagnostic}" + line[match.start() :]


_stage6.Stage6Trainer._prepare_console_progress_payload = (
    _prepare_console_progress_payload_with_gradient
)
_stage6.format_progress_line = _format_progress_line_with_gradient
# ^^^ THOG


__all__ = [
    "DEFAULT_MINIMUM_ABSOLUTE_KENDALL_TAU",
    "DECISION_ALGORITHMS",
    "GRADIENT_ALGORITHM",
    "LEGACY_DIRECTIONAL_ALGORITHM",
    "choose_plastic_depth_count_with_theil_sen_kendall",
    "kendall_tau_b",
    "theil_sen_slope",
]
# ^^^ THOG
