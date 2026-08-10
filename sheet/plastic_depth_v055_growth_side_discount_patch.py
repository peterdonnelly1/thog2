# vvv THOG
"""Full-radius v0.55 Sen/Kendall evidence with an explicit asymmetric growth-side credibility discount."""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from . import plastic_depth_directional_coherence_patch as _directional
from . import plastic_depth_sen_kendall_v055_patch as _v055
from . import plastic_depth_theil_sen_kendall_patch as _tsk
from . import run_config as _run_config
from . import trainer_checkpoint_resume as _checkpoint_resume
from . import training_config as _training_config


_CONFIG_KEY = "plastic__layer_count_decision_algorithm__growth_side_discount"
_PUBLIC_OPTION = "--plastic__layer_count_decision_algorithm__growth_side_discount"
_RUNTIME_ENV = "THOG2_PLASTIC_LAYER_COUNT_DECISION_ALGORITHM__GROWTH_SIDE_DISCOUNT"
_EXPLICIT_ENV = "THOG2_PLASTIC_LAYER_COUNT_DECISION_ALGORITHM__GROWTH_SIDE_DISCOUNT_EXPLICIT"
_DEFAULT_GROWTH_SIDE_DISCOUNT = 1.0
_RAW_ADJACENT_HISTORY_PREFIX = "@SK_STRAT_V055_RAW_ADJ"


def _validate_growth_side_discount(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{_CONFIG_KEY} must be a finite number in [0, 1]; got {value!r}")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{_CONFIG_KEY} must be a finite number in [0, 1]; got {value!r}")
    return resolved


def _runtime_growth_side_discount() -> float:
    raw = os.environ.get(_RUNTIME_ENV, repr(_DEFAULT_GROWTH_SIDE_DISCOUNT))
    return _validate_growth_side_discount(raw)


def _set_runtime_growth_side_discount(value: Any, *, explicit: bool = False) -> None:
    resolved = _validate_growth_side_discount(value)
    os.environ[_RUNTIME_ENV] = repr(resolved)
    if explicit:
        os.environ[_EXPLICIT_ENV] = repr(resolved)


def _explicit_growth_side_discount() -> Optional[float]:
    raw = os.environ.get(_EXPLICIT_ENV)
    if raw is None:
        return None
    return _validate_growth_side_discount(raw)


# vvv THOG expose the exact public knob without perturbing the established dataclass constructor surfaces
_ORIGINAL_PARSE_ARGS = argparse.ArgumentParser.parse_args
_ORIGINAL_PARSE_KNOWN_ARGS = argparse.ArgumentParser.parse_known_args
_ORIGINAL_FORMAT_HELP = argparse.ArgumentParser.format_help


def _strip_growth_side_discount(arguments: Optional[Sequence[str]]) -> Tuple[list[str], Optional[float]]:
    source = list(sys.argv[1:] if arguments is None else arguments)
    remaining: list[str] = []
    requested: Optional[float] = None
    index = 0
    while index < len(source):
        argument = source[index]
        if argument == _PUBLIC_OPTION:
            if index + 1 >= len(source):
                raise SystemExit(f"{_PUBLIC_OPTION} requires a value in [0, 1]")
            try:
                requested = _validate_growth_side_discount(source[index + 1])
            except ValueError as error:
                raise SystemExit(str(error)) from error
            index += 2
            continue
        if argument.startswith(_PUBLIC_OPTION + "="):
            try:
                requested = _validate_growth_side_discount(argument.split("=", 1)[1])
            except ValueError as error:
                raise SystemExit(str(error)) from error
            index += 1
            continue
        remaining.append(argument)
        index += 1
    return remaining, requested


def _attach_growth_side_discount(namespace: Any, requested: Optional[float]) -> Any:
    if requested is not None:
        _set_runtime_growth_side_discount(requested, explicit=True)
    setattr(namespace, _CONFIG_KEY, _runtime_growth_side_discount())
    return namespace


def _parse_args_with_growth_side_discount(self: argparse.ArgumentParser, args=None, namespace=None):
    remaining, requested = _strip_growth_side_discount(args)
    parsed = _ORIGINAL_PARSE_ARGS(self, remaining, namespace)
    return _attach_growth_side_discount(parsed, requested)


def _parse_known_args_with_growth_side_discount(self: argparse.ArgumentParser, args=None, namespace=None):
    remaining, requested = _strip_growth_side_discount(args)
    parsed, extras = _ORIGINAL_PARSE_KNOWN_ARGS(self, remaining, namespace)
    return _attach_growth_side_discount(parsed, requested), extras


def _format_help_with_growth_side_discount(self: argparse.ArgumentParser) -> str:
    rendered = _ORIGINAL_FORMAT_HELP(self)
    if _PUBLIC_OPTION in rendered:
        return rendered
    if not any(action.dest == "plastic__enabled" for action in self._actions):
        return rendered
    return (
        rendered.rstrip()
        + f"\n  {_PUBLIC_OPTION} X\n"
        + "                        v0.55 Sen/Kendall only: credit X of beneficial growth-side economic evidence; adverse growth evidence is undiscounted; default 1.0\n"
    )


argparse.ArgumentParser.parse_args = _parse_args_with_growth_side_discount
argparse.ArgumentParser.parse_known_args = _parse_known_args_with_growth_side_discount
argparse.ArgumentParser.format_help = _format_help_with_growth_side_discount
# ^^^ THOG


# vvv THOG expose and persist the material TSK control while leaving legacy directional-coherence identity untouched
def _growth_side_discount_property(_self: Any) -> float:
    return _runtime_growth_side_discount()


if not hasattr(_training_config.TrainingConfig, _CONFIG_KEY):
    setattr(_training_config.TrainingConfig, _CONFIG_KEY, property(_growth_side_discount_property))
if not hasattr(_run_config.OwtRunConfig, _CONFIG_KEY):
    setattr(_run_config.OwtRunConfig, _CONFIG_KEY, property(_growth_side_discount_property))


_ORIGINAL_TRAINING_POST_INIT = _training_config.TrainingConfig.__post_init__
_ORIGINAL_TRAINING_PERSISTENT_DICT = _training_config.TrainingConfig.persistent_dict
_ORIGINAL_TRAINING_COMPACT_IDENTITY = _training_config.TrainingConfig.compact_identity_metadata
_ORIGINAL_RUN_PERSISTENT_DICT = _run_config.OwtRunConfig.persistent_dict
_ORIGINAL_RUN_COMPACT_IDENTITY = _run_config.OwtRunConfig.compact_identity
_ORIGINAL_NORMALIZE_PLASTIC_CONFIG = _training_config.normalize_plastic_v0541_config_fields


def _training_post_init_with_growth_side_discount(self: Any) -> None:
    _ORIGINAL_TRAINING_POST_INIT(self)
    _validate_growth_side_discount(_runtime_growth_side_discount())


def _persistent_with_growth_side_discount(original: Any, config: Any) -> Dict[str, Any]:
    values = original(config)
    if (
        bool(getattr(config, "plastic__enabled", False))
        and _v055._runtime_algorithm() in _v055.SEN_KENDALL_ALGORITHMS
    ):
        values[_CONFIG_KEY] = _runtime_growth_side_discount()
    return values


def _training_persistent_dict_with_growth_side_discount(self: Any) -> Dict[str, Any]:
    return _persistent_with_growth_side_discount(_ORIGINAL_TRAINING_PERSISTENT_DICT, self)


def _run_persistent_dict_with_growth_side_discount(self: Any) -> Dict[str, Any]:
    return _persistent_with_growth_side_discount(_ORIGINAL_RUN_PERSISTENT_DICT, self)


def _identity_with_growth_side_discount(identity: Dict[str, Any], *, plastic_enabled: bool) -> Dict[str, Any]:
    if not plastic_enabled or _v055._runtime_algorithm() not in _v055.SEN_KENDALL_ALGORITHMS:
        return identity
    plastic_identity = identity.get("plastic_depth")
    if not isinstance(plastic_identity, Mapping):
        return identity
    updated = dict(identity)
    updated["plastic_depth"] = {
        **dict(plastic_identity),
        _CONFIG_KEY: _runtime_growth_side_discount(),
    }
    return updated


def _training_compact_identity_with_growth_side_discount(self: Any) -> Dict[str, Any]:
    return _identity_with_growth_side_discount(
        _ORIGINAL_TRAINING_COMPACT_IDENTITY(self),
        plastic_enabled=bool(self.plastic__enabled),
    )


def _run_compact_identity_with_growth_side_discount(self: Any) -> Dict[str, Any]:
    return _identity_with_growth_side_discount(
        _ORIGINAL_RUN_COMPACT_IDENTITY(self),
        plastic_enabled=bool(self.plastic__enabled),
    )


def _normalize_plastic_config_with_growth_side_discount(values: Mapping[str, Any]) -> Dict[str, Any]:
    source = dict(values)
    checkpoint_value = source.pop(_CONFIG_KEY, None)
    if checkpoint_value is not None:
        checkpoint_discount = _validate_growth_side_discount(checkpoint_value)
        explicit = _explicit_growth_side_discount()
        if explicit is not None and not math.isclose(explicit, checkpoint_discount, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "resume material parameter mismatch: "
                f"{_CONFIG_KEY}: checkpoint={checkpoint_discount!r}, requested={explicit!r}"
            )
        _set_runtime_growth_side_discount(checkpoint_discount)
    return _ORIGINAL_NORMALIZE_PLASTIC_CONFIG(source)


_training_config.TrainingConfig.__post_init__ = _training_post_init_with_growth_side_discount
_training_config.TrainingConfig.persistent_dict = _training_persistent_dict_with_growth_side_discount
_training_config.TrainingConfig.compact_identity_metadata = _training_compact_identity_with_growth_side_discount
_training_config.normalize_plastic_v0541_config_fields = _normalize_plastic_config_with_growth_side_discount
_run_config.OwtRunConfig.persistent_dict = _run_persistent_dict_with_growth_side_discount
_run_config.OwtRunConfig.compact_identity = _run_compact_identity_with_growth_side_discount
if hasattr(_run_config, "normalize_plastic_v0541_config_fields"):
    _run_config.normalize_plastic_v0541_config_fields = _normalize_plastic_config_with_growth_side_discount
_checkpoint_resume.normalize_plastic_v0541_config_fields = _normalize_plastic_config_with_growth_side_discount
# ^^^ THOG


def _discount_growth_delta(offset: int, delta: float, discount: float) -> float:
    resolved = float(delta)
    if int(offset) > 0 and resolved < 0.0:
        return resolved * float(discount)
    return resolved


def growth_discounted_score_map(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    growth_side_discount: Optional[float] = None,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Return raw and TSK-fit economic scores; only beneficial right-side deltas are attenuated."""

    raw_scores = _tsk._equivalent_time_scores(score_report)
    current = int(current_count)
    current_score = raw_scores.get(current)
    fit_scores = dict(raw_scores)
    if current_score is None:
        return raw_scores, fit_scores
    discount = (
        _runtime_growth_side_discount()
        if growth_side_discount is None
        else _validate_growth_side_discount(growth_side_discount)
    )
    for count, score in tuple(raw_scores.items()):
        offset = int(count) - current
        delta = float(score) - float(current_score)
        adjusted_delta = _discount_growth_delta(offset, delta, discount)
        if adjusted_delta != delta:
            fit_scores[int(count)] = float(current_score) + adjusted_delta
    return raw_scores, fit_scores


# vvv THOG LRA keeps the exact raw adjacent action gate but lets every feasible radius point inform discounted Sen/Kendall direction evidence
_ORIGINAL_GRADIENT_PROBE_CLASSIFICATION = _tsk._gradient_probe_classification


def _gradient_probe_classification_with_full_radius_growth_discount(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    minimum_absolute_kendall_tau: float,
) -> Dict[str, Any]:
    if _v055._runtime_algorithm() not in _v055.SEN_KENDALL_ALGORITHMS:
        return _ORIGINAL_GRADIENT_PROBE_CLASSIFICATION(
            current_count=current_count,
            score_report=score_report,
            minimum_absolute_kendall_tau=minimum_absolute_kendall_tau,
        )
    threshold = float(minimum_absolute_kendall_tau)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum absolute Kendall tau must be finite and lie in [0, 1]")
    current = int(current_count)
    raw_scores, fit_scores = growth_discounted_score_map(
        current_count=current,
        score_report=score_report,
    )
    decision_counts = tuple(sorted(fit_scores))
    points = tuple(
        (float(count - current), float(fit_scores[count]))
        for count in decision_counts
    )
    slope = _tsk.theil_sen_slope(points) if len(points) >= 2 else float("nan")
    tau, concordant, discordant, x_ties, y_ties = (
        _tsk.kendall_tau_b(points)
        if len(points) >= 2
        else (float("nan"), 0, 0, 0, 0)
    )
    current_score = raw_scores.get(current)
    left_score = raw_scores.get(current - 1)
    right_score = raw_scores.get(current + 1)
    vote = _tsk._DIRECTION_AMBIGUOUS
    if current_score is not None and math.isfinite(slope) and math.isfinite(tau):
        if (
            left_score is not None
            and left_score < current_score
            and slope > 0.0
            and tau >= threshold
        ):
            vote = _tsk._DIRECTION_LEFT
        elif (
            right_score is not None
            and right_score < current_score
            and slope < 0.0
            and tau <= -threshold
        ):
            vote = _tsk._DIRECTION_RIGHT
    all_offsets = _directional._candidate_offsets(
        current_count=current,
        score_report=score_report,
    )
    return {
        "algorithm": _tsk.GRADIENT_ALGORITHM,
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
        "growth_side_discount": _runtime_growth_side_discount(),
    }


_tsk._gradient_probe_classification = _gradient_probe_classification_with_full_radius_growth_discount
# ^^^ THOG


def _raw_adjacent_history_key(current: int, direction: int) -> str:
    return f"{int(current)}:{_RAW_ADJACENT_HISTORY_PREFIX}:{int(direction):+d}"


def _append_window_value(
    updated: Dict[str, Tuple[float, ...]],
    *,
    key: str,
    value: float,
    window: int,
) -> None:
    values = list(updated.get(key, ()))
    values.append(float(value))
    updated[key] = tuple(values[-int(window):])


def _partial_raw_adjacent_median(
    updated: Mapping[str, Sequence[float]],
    *,
    current: int,
    direction: int,
    available_now: bool,
) -> Optional[float]:
    if not available_now:
        return None
    values = tuple(float(value) for value in updated.get(_raw_adjacent_history_key(current, direction), ()))
    return float(statistics.median(values)) if values else None


# vvv THOG stratified TSK pools every feasible radius point while preserving the exact raw adjacent economic veto across the same probe window
def choose_plastic_depth_count_with_stratified_sen_kendall_growth_discount(
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
    **_ignored: Any,
) -> Any:
    del noise_lambda, minimum_observations, max_step
    window = _directional._positive_integer(
        noise_window,
        name="plastic__layer_count_probe__window_size_as_number_of_probes",
    )
    current = int(current_count)
    updated = _v055._copy_histories(histories, window)
    raw_scores, fit_scores = growth_discounted_score_map(
        current_count=current,
        score_report=score_report,
    )
    decision_offsets = tuple(sorted(count - current for count in fit_scores))
    complete_probe = 0 in decision_offsets and len(decision_offsets) >= 2
    if complete_probe:
        current_raw_score = float(raw_scores[current])
        current_fit_score = float(fit_scores[current])
        for offset in decision_offsets:
            _append_window_value(
                updated,
                key=_v055._stratified_history_key(current, offset),
                value=float(fit_scores[current + offset] - current_fit_score),
                window=window,
            )
        for direction in (-1, 1):
            candidate_score = raw_scores.get(current + direction)
            if candidate_score is not None:
                _append_window_value(
                    updated,
                    key=_raw_adjacent_history_key(current, direction),
                    value=float(candidate_score - current_raw_score),
                    window=window,
                )

    lengths = [
        len(updated.get(_v055._stratified_history_key(current, offset), ()))
        for offset in decision_offsets
    ]
    row_count = min(lengths) if lengths else 0
    rows = []
    if row_count > 0:
        for row_index in range(-row_count, 0):
            points = []
            for offset in decision_offsets:
                values = updated.get(_v055._stratified_history_key(current, offset), ())
                points.append((float(offset), float(values[row_index])))
            rows.append(tuple(points))

    sen = _v055.stratified_sen_slope(rows) if rows else float("nan")
    ken, concordant, discordant, y_ties = (
        _v055.stratified_kendall_tau_b(rows)
        if rows
        else (float("nan"), 0, 0, 0)
    )
    left_adjacent = _partial_raw_adjacent_median(
        updated,
        current=current,
        direction=-1,
        available_now=(current - 1) in raw_scores,
    )
    right_adjacent = _partial_raw_adjacent_median(
        updated,
        current=current,
        direction=1,
        available_now=(current + 1) in raw_scores,
    )
    proposed_direction = (
        -1 if math.isfinite(sen) and sen > 0.0
        else 1 if math.isfinite(sen) and sen < 0.0
        else 0
    )
    adjacent = (
        left_adjacent if proposed_direction < 0
        else right_adjacent if proposed_direction > 0
        else None
    )
    brake = _v055._brake_active(
        update_number=update_number,
        last_count_change_update=last_count_change_update,
        update_brake=update_brake,
    )
    ready = row_count >= window
    selected = current
    conclusion = "-"
    if ready and proposed_direction < 0:
        if (
            math.isfinite(ken)
            and ken >= _v055.FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU
            and adjacent is not None
            and adjacent < 0.0
        ):
            conclusion = "L"
            if not brake:
                selected = current - 1
    elif ready and proposed_direction > 0:
        if (
            math.isfinite(ken)
            and ken <= -_v055.FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU
            and adjacent is not None
            and adjacent < 0.0
        ):
            conclusion = "R"
            if not brake:
                selected = current + 1
    if selected != current:
        updated = {}

    all_offsets = _directional._candidate_offsets(
        current_count=current,
        score_report=score_report,
    )
    report = {
        "algorithm": _v055.STRATIFIED_ALGORITHM,
        "decision_fit_offsets": decision_offsets,
        "candidate_offsets": all_offsets,
        "sen_slope_seconds_per_layer": sen,
        "theil_sen_slope_seconds_per_layer": sen,
        "kendall_tau": ken,
        "adjacent_score_seconds": adjacent,
        "left_adjacent_score_seconds": left_adjacent,
        "right_adjacent_score_seconds": right_adjacent,
        "minimum_absolute_kendall_tau": _v055.FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU,
        "growth_side_discount": _runtime_growth_side_discount(),
        "strata_count": row_count,
        "vote_total": row_count,
        "window_ready": ready,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "y_tied_pairs": y_ties,
        "direction_conclusion": conclusion,
        "conclusion": conclusion,
        "selected_count": selected,
        "brake_active": brake,
    }
    trainer = _v055._wall_time._ACTIVE_TRAINER.get()
    if trainer is not None:
        _v055._store_report(trainer, report)
    return _directional._controller.PlasticDepthRobustCountDecision(
        selected_count=selected,
        current_count=current,
        update_number=int(update_number),
        brake_active=brake,
        last_count_change_update=int(last_count_change_update),
        histories=updated,
        evidence=_v055._evidence(
            current=current,
            score_by_count=raw_scores,
            observation_count=row_count,
            selected_count=selected,
        ),
    )


_v055.choose_plastic_depth_count_with_stratified_sen_kendall_v055 = (
    choose_plastic_depth_count_with_stratified_sen_kendall_growth_discount
)
# ^^^ THOG


__all__ = [
    "growth_discounted_score_map",
    "choose_plastic_depth_count_with_stratified_sen_kendall_growth_discount",
]
# ^^^ THOG
