# vvv THOG
"""PLASTIC v0.57 standalone trend decisions and direct raw-loss jumping."""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from functools import wraps
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from . import help_registry_descriptor_patch as _registry
from . import plastic_depth_audit_patch as _audit
from . import plastic_depth_console_compact_layout_patch as _compact
from . import plastic_depth_directional_coherence_patch as _directional
from . import plastic_depth_lookahead_patch as _lookahead
from . import plastic_depth_sen_kendall_v055_audit_fix_patch as _audit_v055
from . import plastic_depth_sen_kendall_v055_patch as _v055
from . import plastic_depth_theil_sen_kendall_patch as _tsk
from . import plastic_depth_v055_growth_side_discount_patch as _growth
from . import plastic_depth_v056_objective_decision_patch as _v056
from . import run_config as _run_config
from . import trainer_checkpoint_resume as _checkpoint_resume
from . import trainer_step as _trainer_step
from . import training_config as _training_config


SEN_ALGORITHM = "sen"
KENDALL_ALGORITHM = "kendall"
JUMP_TO_LOWEST_LOSS_ALGORITHM = "jump_to_lowest_loss"
STANDALONE_STATISTICAL_ALGORITHMS = (SEN_ALGORITHM, KENDALL_ALGORITHM)
STATISTICAL_ALGORITHMS = (*_v056.SEN_KENDALL_ALGORITHMS, *STANDALONE_STATISTICAL_ALGORITHMS)
OWNED_ALGORITHMS = (*STATISTICAL_ALGORITHMS, JUMP_TO_LOWEST_LOSS_ALGORITHM)
DECISION_ALGORITHMS = (_tsk.LEGACY_DIRECTIONAL_ALGORITHM, *OWNED_ALGORITHMS)

SEN_THRESHOLD_KEY = "plastic__layer_count_decision_algorithm__sen__minimum_absolute_slope"
KENDALL_THRESHOLD_KEY = "plastic__layer_count_decision_algorithm__kendall__minimum_absolute_tau"
JUMP_THRESHOLD_KEY = (
    "plastic__layer_count_decision_algorithm__jump_to_lowest_loss__minimum_improvement_percent"
)

_SEN_THRESHOLD_OPTION = f"--{SEN_THRESHOLD_KEY}"
_KENDALL_THRESHOLD_OPTION = f"--{KENDALL_THRESHOLD_KEY}"
_JUMP_THRESHOLD_OPTION = f"--{JUMP_THRESHOLD_KEY}"
_SEN_THRESHOLD_ENV = "THOG2_PLASTIC_LAYER_COUNT_DECISION_ALGORITHM__SEN__MINIMUM_ABSOLUTE_SLOPE"
_KENDALL_THRESHOLD_ENV = "THOG2_PLASTIC_LAYER_COUNT_DECISION_ALGORITHM__KENDALL__MINIMUM_ABSOLUTE_TAU"
_JUMP_THRESHOLD_ENV = (
    "THOG2_PLASTIC_LAYER_COUNT_DECISION_ALGORITHM__JUMP_TO_LOWEST_LOSS__MINIMUM_IMPROVEMENT_PERCENT"
)
_SEN_THRESHOLD_EXPLICIT_ENV = _SEN_THRESHOLD_ENV + "_EXPLICIT"
_KENDALL_THRESHOLD_EXPLICIT_ENV = _KENDALL_THRESHOLD_ENV + "_EXPLICIT"
_JUMP_THRESHOLD_EXPLICIT_ENV = _JUMP_THRESHOLD_ENV + "_EXPLICIT"
_DEFAULT_SEN_THRESHOLD = 0.0
_DEFAULT_KENDALL_THRESHOLD = 0.5
_DEFAULT_JUMP_THRESHOLD = 0.0
_SECTION = "PLASTIC DEPTH decision algorithms"
_EVENT_NAME = "plastic_depth_v057_decision"


def _validate_nonnegative_threshold(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number; got {value!r}")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number; got {value!r}")
    return resolved


def _validate_tau_threshold(value: Any) -> float:
    resolved = _validate_nonnegative_threshold(value, name=KENDALL_THRESHOLD_KEY)
    if resolved > 1.0:
        raise ValueError(f"{KENDALL_THRESHOLD_KEY} must lie in [0, 1]; got {value!r}")
    return resolved


def _runtime_threshold(key: str) -> float:
    if key == SEN_THRESHOLD_KEY:
        return _validate_nonnegative_threshold(
            os.environ.get(_SEN_THRESHOLD_ENV, repr(_DEFAULT_SEN_THRESHOLD)),
            name=key,
        )
    if key == KENDALL_THRESHOLD_KEY:
        return _validate_tau_threshold(
            os.environ.get(_KENDALL_THRESHOLD_ENV, repr(_DEFAULT_KENDALL_THRESHOLD))
        )
    if key == JUMP_THRESHOLD_KEY:
        return _validate_nonnegative_threshold(
            os.environ.get(_JUMP_THRESHOLD_ENV, repr(_DEFAULT_JUMP_THRESHOLD)),
            name=key,
        )
    raise KeyError(key)


def _set_runtime_threshold(key: str, value: Any, *, explicit: bool = False) -> None:
    if key == KENDALL_THRESHOLD_KEY:
        resolved = _validate_tau_threshold(value)
        runtime_env = _KENDALL_THRESHOLD_ENV
        explicit_env = _KENDALL_THRESHOLD_EXPLICIT_ENV
    elif key == SEN_THRESHOLD_KEY:
        resolved = _validate_nonnegative_threshold(value, name=key)
        runtime_env = _SEN_THRESHOLD_ENV
        explicit_env = _SEN_THRESHOLD_EXPLICIT_ENV
    elif key == JUMP_THRESHOLD_KEY:
        resolved = _validate_nonnegative_threshold(value, name=key)
        runtime_env = _JUMP_THRESHOLD_ENV
        explicit_env = _JUMP_THRESHOLD_EXPLICIT_ENV
    else:
        raise KeyError(key)
    os.environ[runtime_env] = repr(resolved)
    if explicit:
        os.environ[explicit_env] = repr(resolved)


def _explicit_threshold(key: str) -> Optional[float]:
    env = {
        SEN_THRESHOLD_KEY: _SEN_THRESHOLD_EXPLICIT_ENV,
        KENDALL_THRESHOLD_KEY: _KENDALL_THRESHOLD_EXPLICIT_ENV,
        JUMP_THRESHOLD_KEY: _JUMP_THRESHOLD_EXPLICIT_ENV,
    }[key]
    raw = os.environ.get(env)
    return None if raw is None else _runtime_threshold_from_value(key, raw)


def _runtime_threshold_from_value(key: str, value: Any) -> float:
    if key == KENDALL_THRESHOLD_KEY:
        return _validate_tau_threshold(value)
    return _validate_nonnegative_threshold(value, name=key)


def _runtime_algorithm() -> str:
    value = os.environ.get(_tsk._ALGORITHM_ENV, _tsk.LEGACY_DIRECTIONAL_ALGORITHM).strip()
    if value in _v056._RETIRED_ALGORITHMS:
        replacements = (
            f"{_v056.LRA_ALGORITHM} or {_v056.STRATIFIED_ALGORITHM}"
            if value != "wall_time__gradient__theil_sen_kendall_slope_tau"
            else _v056.LRA_ALGORITHM
        )
        raise ValueError(
            f"{value} is retired in PLASTIC v0.56; select the objective separately and use {replacements}"
        )
    if value not in DECISION_ALGORITHMS:
        raise ValueError(
            f"{_tsk._ALGORITHM_KEY} must be one of {DECISION_ALGORITHMS}; got {value!r}"
        )
    return value


def _set_runtime_algorithm(value: str) -> None:
    resolved = str(value).strip()
    if resolved in _v056._RETIRED_ALGORITHMS:
        previous = os.environ.get(_tsk._ALGORITHM_ENV)
        try:
            os.environ[_tsk._ALGORITHM_ENV] = resolved
            _runtime_algorithm()
        finally:
            if previous is None:
                os.environ.pop(_tsk._ALGORITHM_ENV, None)
            else:
                os.environ[_tsk._ALGORITHM_ENV] = previous
    if resolved not in DECISION_ALGORITHMS:
        raise ValueError(
            f"{_tsk._ALGORITHM_KEY} must be one of {DECISION_ALGORITHMS}; got {value!r}"
        )
    os.environ[_tsk._ALGORITHM_ENV] = resolved


# vvv THOG give every retained layer access to the expanded selector vocabulary while
# keeping growth-side persistence limited to algorithms that actually consume it.
_v056.DECISION_ALGORITHMS = DECISION_ALGORITHMS
_v056.SEN_KENDALL_ALGORITHMS = OWNED_ALGORITHMS
_v056._runtime_algorithm = _runtime_algorithm
_v056._set_runtime_algorithm = _set_runtime_algorithm
_v055.DECISION_ALGORITHMS = DECISION_ALGORITHMS
_v055.SEN_KENDALL_ALGORITHMS = STATISTICAL_ALGORITHMS
_v055._runtime_algorithm = _runtime_algorithm
_v055._set_runtime_algorithm = _set_runtime_algorithm
_tsk.DECISION_ALGORITHMS = DECISION_ALGORITHMS
_tsk._runtime_algorithm = _runtime_algorithm
_tsk._set_runtime_algorithm = _set_runtime_algorithm
# ^^^ THOG


def _validate_v057_config(config: Any) -> None:
    algorithm = _runtime_algorithm()
    if algorithm not in OWNED_ALGORITHMS:
        return
    if not bool(getattr(config, "plastic__enabled", False)) or not bool(
        getattr(config, "plastic__do_learn_layer_count", False)
    ):
        raise ValueError(f"{algorithm} requires learned-count PLASTIC DEPTH")
    if algorithm in STATISTICAL_ALGORITHMS and int(
        getattr(config, "plastic__layer_count__max_allowable_layer_change", 1)
    ) != 1:
        raise ValueError(
            f"{algorithm} requires plastic__layer_count__max_allowable_layer_change=1"
        )
    _runtime_threshold(SEN_THRESHOLD_KEY)
    _runtime_threshold(KENDALL_THRESHOLD_KEY)
    _runtime_threshold(JUMP_THRESHOLD_KEY)


_v056._validate_v056_config = _validate_v057_config
# ^^^ THOG


# vvv THOG parse the three algorithm-specific thresholds without widening the core
# dataclass constructors.
_ORIGINAL_PARSE_ARGS = argparse.ArgumentParser.parse_args
_ORIGINAL_PARSE_KNOWN_ARGS = argparse.ArgumentParser.parse_known_args
_ORIGINAL_FORMAT_HELP = argparse.ArgumentParser.format_help
_OPTION_TO_KEY = {
    _SEN_THRESHOLD_OPTION: SEN_THRESHOLD_KEY,
    _KENDALL_THRESHOLD_OPTION: KENDALL_THRESHOLD_KEY,
    _JUMP_THRESHOLD_OPTION: JUMP_THRESHOLD_KEY,
}


def _strip_threshold_options(
    arguments: Optional[Sequence[str]],
) -> Tuple[list[str], Dict[str, float]]:
    source = list(sys.argv[1:] if arguments is None else arguments)
    remaining: list[str] = []
    requested: Dict[str, float] = {}
    index = 0
    while index < len(source):
        argument = source[index]
        option = next(
            (
                candidate
                for candidate in _OPTION_TO_KEY
                if argument == candidate or argument.startswith(candidate + "=")
            ),
            None,
        )
        if option is None:
            remaining.append(argument)
            index += 1
            continue
        if argument == option:
            if index + 1 >= len(source):
                raise SystemExit(f"{option} requires a value")
            raw_value = source[index + 1]
            index += 2
        else:
            raw_value = argument.split("=", 1)[1]
            index += 1
        key = _OPTION_TO_KEY[option]
        try:
            requested[key] = _runtime_threshold_from_value(key, raw_value)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    return remaining, requested


def _attach_thresholds(namespace: Any, requested: Mapping[str, float]) -> Any:
    for key, value in requested.items():
        _set_runtime_threshold(key, value, explicit=True)
    for key in (SEN_THRESHOLD_KEY, KENDALL_THRESHOLD_KEY, JUMP_THRESHOLD_KEY):
        setattr(namespace, key, _runtime_threshold(key))
    return namespace


def _parse_args_v057(self: argparse.ArgumentParser, args=None, namespace=None):
    remaining, requested = _strip_threshold_options(args)
    return _attach_thresholds(_ORIGINAL_PARSE_ARGS(self, remaining, namespace), requested)


def _parse_known_args_v057(self: argparse.ArgumentParser, args=None, namespace=None):
    remaining, requested = _strip_threshold_options(args)
    parsed, extras = _ORIGINAL_PARSE_KNOWN_ARGS(self, remaining, namespace)
    return _attach_thresholds(parsed, requested), extras


def _format_help_v057(self: argparse.ArgumentParser) -> str:
    rendered = _ORIGINAL_FORMAT_HELP(self)
    if not any(action.dest == "plastic__enabled" for action in self._actions):
        return rendered
    rendered = rendered.replace(
        "directional_coherence (default), theil_sen_kendall_LRA, or sen_kendall__tau__stratified",
        "directional_coherence (default), theil_sen_kendall_LRA, sen_kendall__tau__stratified, sen, kendall, or jump_to_lowest_loss",
    )
    additions = []
    if _SEN_THRESHOLD_OPTION not in rendered:
        additions.extend(
            (
                f"  {_SEN_THRESHOLD_OPTION} X",
                "                        standalone sen: minimum absolute objective-score slope per layer; default 0",
                f"  {_KENDALL_THRESHOLD_OPTION} X",
                "                        standalone kendall: minimum absolute tau-b in [0,1]; default 0.5",
                f"  {_JUMP_THRESHOLD_OPTION} PERCENT",
                "                        jump_to_lowest_loss: minimum raw validation-loss improvement percentage; default 0",
            )
        )
    return rendered.rstrip() + ("\n" + "\n".join(additions) if additions else "") + "\n"


argparse.ArgumentParser.parse_args = _parse_args_v057
argparse.ArgumentParser.parse_known_args = _parse_known_args_v057
argparse.ArgumentParser.format_help = _format_help_v057
# ^^^ THOG


def _threshold_property(key: str):
    return property(lambda _self: _runtime_threshold(key))


for _config_type in (_training_config.TrainingConfig, _run_config.OwtRunConfig):
    for _key in (SEN_THRESHOLD_KEY, KENDALL_THRESHOLD_KEY, JUMP_THRESHOLD_KEY):
        if not hasattr(_config_type, _key):
            setattr(_config_type, _key, _threshold_property(_key))


_ORIGINAL_TRAINING_INIT = _training_config.TrainingConfig.__init__


@wraps(_ORIGINAL_TRAINING_INIT)
def _training_config_init_v057(self: Any, *args: Any, **kwargs: Any) -> None:
    source = dict(kwargs)
    for key in (SEN_THRESHOLD_KEY, KENDALL_THRESHOLD_KEY, JUMP_THRESHOLD_KEY):
        value = source.pop(key, None)
        if value is not None:
            _set_runtime_threshold(key, value)
    _ORIGINAL_TRAINING_INIT(self, *args, **source)


_training_config.TrainingConfig.__init__ = _training_config_init_v057


_ORIGINAL_TRAINING_PERSISTENT_DICT = _training_config.TrainingConfig.persistent_dict
_ORIGINAL_TRAINING_COMPACT_IDENTITY = _training_config.TrainingConfig.compact_identity_metadata
_ORIGINAL_RUN_PERSISTENT_DICT = _run_config.OwtRunConfig.persistent_dict
_ORIGINAL_RUN_COMPACT_IDENTITY = _run_config.OwtRunConfig.compact_identity
_ORIGINAL_NORMALIZE_PLASTIC_CONFIG = _training_config.normalize_plastic_v0541_config_fields


def _active_threshold_key(algorithm: str) -> Optional[str]:
    return {
        SEN_ALGORITHM: SEN_THRESHOLD_KEY,
        KENDALL_ALGORITHM: KENDALL_THRESHOLD_KEY,
        JUMP_TO_LOWEST_LOSS_ALGORITHM: JUMP_THRESHOLD_KEY,
    }.get(algorithm)


def _persistent_v057(original: Any, config: Any) -> Dict[str, Any]:
    values = original(config)
    algorithm = _runtime_algorithm()
    if bool(getattr(config, "plastic__enabled", False)) and algorithm in OWNED_ALGORITHMS:
        values[_tsk._ALGORITHM_KEY] = algorithm
        threshold_key = _active_threshold_key(algorithm)
        if threshold_key is not None:
            values[threshold_key] = _runtime_threshold(threshold_key)
    return values


def _training_persistent_dict_v057(self: Any) -> Dict[str, Any]:
    return _persistent_v057(_ORIGINAL_TRAINING_PERSISTENT_DICT, self)


def _run_persistent_dict_v057(self: Any) -> Dict[str, Any]:
    return _persistent_v057(_ORIGINAL_RUN_PERSISTENT_DICT, self)


def _identity_v057(identity: Dict[str, Any], *, plastic_enabled: bool) -> Dict[str, Any]:
    algorithm = _runtime_algorithm()
    if not plastic_enabled or algorithm not in OWNED_ALGORITHMS:
        return identity
    plastic = identity.get("plastic_depth")
    if not isinstance(plastic, Mapping):
        return identity
    plastic_updated = {**dict(plastic), _tsk._ALGORITHM_KEY: algorithm}
    threshold_key = _active_threshold_key(algorithm)
    if threshold_key is not None:
        plastic_updated[threshold_key] = _runtime_threshold(threshold_key)
    return {**dict(identity), "plastic_depth": plastic_updated}


def _training_compact_identity_v057(self: Any) -> Dict[str, Any]:
    return _identity_v057(
        _ORIGINAL_TRAINING_COMPACT_IDENTITY(self),
        plastic_enabled=bool(self.plastic__enabled),
    )


def _run_compact_identity_v057(self: Any) -> Dict[str, Any]:
    return _identity_v057(
        _ORIGINAL_RUN_COMPACT_IDENTITY(self),
        plastic_enabled=bool(self.plastic__enabled),
    )


def _normalize_plastic_config_v057(values: Mapping[str, Any]) -> Dict[str, Any]:
    source = dict(values)
    for key in (SEN_THRESHOLD_KEY, KENDALL_THRESHOLD_KEY, JUMP_THRESHOLD_KEY):
        checkpoint_value = source.pop(key, None)
        if checkpoint_value is None:
            continue
        resolved = _runtime_threshold_from_value(key, checkpoint_value)
        explicit = _explicit_threshold(key)
        if explicit is not None and not math.isclose(
            explicit,
            resolved,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "resume material parameter mismatch: "
                f"{key}: checkpoint={resolved!r}, requested={explicit!r}"
            )
        _set_runtime_threshold(key, resolved)
    return _ORIGINAL_NORMALIZE_PLASTIC_CONFIG(source)


_training_config.TrainingConfig.persistent_dict = _training_persistent_dict_v057
_training_config.TrainingConfig.compact_identity_metadata = _training_compact_identity_v057
_training_config.normalize_plastic_v0541_config_fields = _normalize_plastic_config_v057
_run_config.OwtRunConfig.persistent_dict = _run_persistent_dict_v057
_run_config.OwtRunConfig.compact_identity = _run_compact_identity_v057
if hasattr(_run_config, "normalize_plastic_v0541_config_fields"):
    _run_config.normalize_plastic_v0541_config_fields = _normalize_plastic_config_v057
_checkpoint_resume.normalize_plastic_v0541_config_fields = _normalize_plastic_config_v057
# ^^^ THOG


def _append_history(
    histories: Dict[str, Tuple[float, ...]],
    *,
    key: str,
    value: float,
    window: int,
) -> None:
    values = list(histories.get(key, ()))
    values.append(float(value))
    histories[key] = tuple(values[-window:])


def _standalone_statistical_decision(
    *,
    algorithm: str,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    histories: Mapping[str, Sequence[float]],
    noise_window: int,
    update_number: int,
    last_count_change_update: int,
    update_brake: int,
    **_ignored: Any,
) -> Any:
    window = _directional._positive_integer(
        noise_window,
        name="plastic__layer_count_probe__window_size_as_number_of_probes",
    )
    current = int(current_count)
    updated = _v055._copy_histories(histories, window)
    raw_scores, fit_scores = _growth.growth_discounted_score_map(
        current_count=current,
        score_report=score_report,
    )
    decision_offsets = tuple(sorted(count - current for count in fit_scores))
    if 0 in decision_offsets and len(decision_offsets) >= 2:
        current_fit = float(fit_scores[current])
        current_raw = float(raw_scores[current])
        for offset in decision_offsets:
            _append_history(
                updated,
                key=_v055._stratified_history_key(current, offset),
                value=float(fit_scores[current + offset] - current_fit),
                window=window,
            )
        for direction in (-1, 1):
            candidate = raw_scores.get(current + direction)
            if candidate is not None:
                _append_history(
                    updated,
                    key=_growth._raw_adjacent_history_key(current, direction),
                    value=float(candidate - current_raw),
                    window=window,
                )

    lengths = [
        len(updated.get(_v055._stratified_history_key(current, offset), ()))
        for offset in decision_offsets
    ]
    row_count = min(lengths) if lengths else 0
    rows = []
    for row_index in range(-row_count, 0):
        rows.append(
            tuple(
                (
                    float(offset),
                    float(updated[_v055._stratified_history_key(current, offset)][row_index]),
                )
                for offset in decision_offsets
            )
        )
    sen = _v055.stratified_sen_slope(rows) if rows else float("nan")
    kendall, concordant, discordant, y_ties = (
        _v055.stratified_kendall_tau_b(rows)
        if rows
        else (float("nan"), 0, 0, 0)
    )
    left_adjacent = _growth._partial_raw_adjacent_median(
        updated,
        current=current,
        direction=-1,
        available_now=(current - 1) in raw_scores,
    )
    right_adjacent = _growth._partial_raw_adjacent_median(
        updated,
        current=current,
        direction=1,
        available_now=(current + 1) in raw_scores,
    )
    threshold = _runtime_threshold(
        SEN_THRESHOLD_KEY if algorithm == SEN_ALGORITHM else KENDALL_THRESHOLD_KEY
    )
    direction = 0
    if row_count >= window:
        if algorithm == SEN_ALGORITHM and math.isfinite(sen):
            direction = -1 if sen > threshold else 1 if sen < -threshold else 0
        elif algorithm == KENDALL_ALGORITHM and math.isfinite(kendall):
            direction = -1 if kendall > 0.0 and kendall >= threshold else 1 if kendall < 0.0 and kendall <= -threshold else 0
    adjacent = left_adjacent if direction < 0 else right_adjacent if direction > 0 else None
    brake = _v055._brake_active(
        update_number=update_number,
        last_count_change_update=last_count_change_update,
        update_brake=update_brake,
    )
    accepted = direction != 0 and adjacent is not None and adjacent < 0.0
    selected = current if brake or not accepted else current + direction
    if selected != current:
        updated = {}
    conclusion = "L" if accepted and direction < 0 else "R" if accepted else "-"
    report = {
        "algorithm": algorithm,
        "decision_fit_offsets": decision_offsets,
        "candidate_offsets": _directional._candidate_offsets(
            current_count=current,
            score_report=score_report,
        ),
        "sen_slope_seconds_per_layer": sen,
        "theil_sen_slope_seconds_per_layer": sen,
        "kendall_tau": kendall,
        "minimum_absolute_sen_slope": (
            threshold if algorithm == SEN_ALGORITHM else None
        ),
        "minimum_absolute_kendall_tau": (
            threshold if algorithm == KENDALL_ALGORITHM else None
        ),
        "adjacent_score_seconds": adjacent,
        "left_adjacent_score_seconds": left_adjacent,
        "right_adjacent_score_seconds": right_adjacent,
        "growth_side_discount": _growth._runtime_growth_side_discount(),
        "strata_count": row_count,
        "vote_total": row_count,
        "window_ready": row_count >= window,
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


def _raw_loss_map(score_report: Sequence[Mapping[str, object]]) -> Dict[int, float]:
    losses: Dict[int, float] = {}
    for item in score_report:
        try:
            count = int(item["active_layers"])
            loss = float(item["validation_loss"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(loss):
            losses[count] = loss
    return losses


def _improvement_percent(current_loss: float, candidate_loss: float) -> float:
    improvement = float(current_loss) - float(candidate_loss)
    denominator = abs(float(current_loss))
    if denominator == 0.0:
        return float("inf") if improvement > 0.0 else 0.0
    return 100.0 * improvement / denominator


def _jump_winner(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    threshold: float,
) -> Tuple[int, Optional[float], Optional[float]]:
    current = int(current_count)
    losses = _raw_loss_map(score_report)
    current_loss = losses.get(current)
    if current_loss is None:
        return current, None, None
    candidates = []
    for count, loss in losses.items():
        if count == current or loss >= current_loss:
            continue
        improvement = _improvement_percent(current_loss, loss)
        if improvement >= threshold:
            candidates.append((loss, abs(count - current), count, improvement))
    if not candidates:
        return current, current_loss, 0.0
    loss, _distance, count, improvement = min(candidates)
    return int(count), float(loss), float(improvement)


def _jump_to_lowest_loss_decision(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    histories: Mapping[str, Sequence[float]],
    update_number: int,
    last_count_change_update: int,
    update_brake: int,
    **_ignored: Any,
) -> Any:
    del histories
    current = int(current_count)
    threshold = _runtime_threshold(JUMP_THRESHOLD_KEY)
    raw_selected, selected_loss, improvement = _jump_winner(
        current_count=current,
        score_report=score_report,
        threshold=threshold,
    )
    brake = _v055._brake_active(
        update_number=update_number,
        last_count_change_update=last_count_change_update,
        update_brake=update_brake,
    )
    selected = current if brake else raw_selected
    losses = _raw_loss_map(score_report)
    evidence = tuple(
        _directional._controller.PlasticDepthPairedDirectionEvidence(
            candidate_count=count,
            direction=count - current,
            paired_difference=(
                None if current not in losses else float(loss - losses[current])
            ),
            observation_count=1,
            median=None,
            mad=None,
            sigma=None,
            standardized_improvement=(
                None
                if current not in losses
                else _improvement_percent(losses[current], loss)
            ),
            significant=count == selected and selected != current,
            feasible=True,
        )
        for count, loss in sorted(losses.items())
        if count != current
    )
    report = {
        "algorithm": JUMP_TO_LOWEST_LOSS_ALGORITHM,
        # The v0.56 report bridge records the configured objective in ``objective``;
        # keep the bulldozer's actual ranking quantity explicit and unambiguous.
        "decision_objective": "raw_validation_loss",
        "configured_objective_ignored": True,
        "statistical_window_ignored": True,
        "max_allowable_layer_change_ignored": True,
        "minimum_improvement_percent": threshold,
        "current_loss": losses.get(current),
        "winning_loss": selected_loss,
        "winning_improvement_percent": improvement,
        "raw_selected_count": raw_selected,
        "selected_count": selected,
        "window_ready": True,
        "vote_total": 1,
        "direction_conclusion": (
            "L" if raw_selected < current else "R" if raw_selected > current else "-"
        ),
        "conclusion": (
            "L" if raw_selected < current else "R" if raw_selected > current else "-"
        ),
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
        histories={},
        evidence=evidence,
    )


_ORIGINAL_SELECTOR = _lookahead.choose_plastic_depth_count_with_exact_radius


def _choose_count_v057(*, max_step: int = 1, **kwargs: Any):
    algorithm = _runtime_algorithm()
    if algorithm in STANDALONE_STATISTICAL_ALGORITHMS:
        return _standalone_statistical_decision(algorithm=algorithm, **kwargs)
    if algorithm == JUMP_TO_LOWEST_LOSS_ALGORITHM:
        return _jump_to_lowest_loss_decision(max_step=max_step, **kwargs)
    return _ORIGINAL_SELECTOR(max_step=max_step, **kwargs)


_lookahead.choose_plastic_depth_count_with_exact_radius = _choose_count_v057
_directional._controller.choose_plastic_depth_count_with_mad = _choose_count_v057
_trainer_step.choose_plastic_depth_count_with_mad = _choose_count_v057
# ^^^ THOG


# vvv THOG replay standalone adjacent decisions through the established v0.55 audit,
# and replay direct jumps without applying max-step limiting.
_ORIGINAL_AUDIT_WINNING_COUNT = _audit._winning_probe_count
_ORIGINAL_AUDIT_REPLAY = _audit.replay_plastic_depth_count_audit
_ORIGINAL_V055_WINNING_FROM_AUDIT = _audit_v055._sen_kendall_winning_count_from_audit


def _report_from_audit(audit: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    report = audit.get("sen_kendall_report")
    if isinstance(report, Mapping):
        return report
    report = audit.get("directional_report")
    return report if isinstance(report, Mapping) else None


def _framework_held(report: Mapping[str, Any]) -> bool:
    return bool(str(report.get("framework_hold_reason", "")).strip())


def _winning_probe_count_v057(decision: Any, current_count: int) -> int:
    if _runtime_algorithm() != JUMP_TO_LOWEST_LOSS_ALGORITHM:
        return _ORIGINAL_AUDIT_WINNING_COUNT(decision, current_count)
    if bool(decision.brake_active):
        return int(current_count)
    significant = tuple(
        int(item.candidate_count)
        for item in decision.evidence
        if bool(item.feasible) and bool(item.significant)
    )
    if len(significant) > 1:
        raise ValueError(f"jump_to_lowest_loss produced multiple winners: {significant}")
    return int(current_count) if not significant else int(significant[0])


def _standalone_winning_count_from_audit(audit: Mapping[str, Any]) -> int:
    current = int(audit["previous_count"])
    report = _report_from_audit(audit)
    if not isinstance(report, Mapping):
        raise ValueError("standalone Sen/Kendall audit lacks its decision report")
    if bool(audit["brake_active"]) or not bool(report.get("window_ready", False)):
        return current
    algorithm = str(report.get("algorithm", ""))
    adjacent = _audit_v055._finite_float(report.get("adjacent_score_seconds"))
    accepted = False
    direction = 0
    if algorithm == SEN_ALGORITHM:
        value = _audit_v055._finite_float(report.get("sen_slope_seconds_per_layer"))
        threshold = _audit_v055._finite_float(report.get("minimum_absolute_sen_slope"))
        if value is not None and threshold is not None:
            direction = -1 if value > threshold else 1 if value < -threshold else 0
            accepted = direction != 0
    elif algorithm == KENDALL_ALGORITHM:
        value = _audit_v055._finite_float(report.get("kendall_tau"))
        threshold = _audit_v055._finite_float(report.get("minimum_absolute_kendall_tau"))
        if value is not None and threshold is not None:
            direction = -1 if value > 0.0 and value >= threshold else 1 if value < 0.0 and value <= -threshold else 0
            accepted = direction != 0
    else:
        return int(_ORIGINAL_V055_WINNING_FROM_AUDIT(audit))
    raw_winner = current + direction if accepted and adjacent is not None and adjacent < 0.0 else current
    if not _framework_held(report):
        return raw_winner
    recorded_raw = report.get("framework_raw_selected_count")
    if recorded_raw is not None and int(recorded_raw) != raw_winner:
        raise ValueError(
            "standalone Sen/Kendall framework-hold raw decision mismatch: "
            f"recorded={recorded_raw}, replayed={raw_winner}"
        )
    return current


def _jump_replay(audit: Mapping[str, Any]) -> Dict[str, object]:
    current = int(audit["previous_count"])
    report = _report_from_audit(audit)
    if not isinstance(report, Mapping):
        raise ValueError("jump_to_lowest_loss audit lacks its decision report")
    threshold = float(report["minimum_improvement_percent"])
    raw_winner, _loss, _improvement = _jump_winner(
        current_count=current,
        score_report=tuple(audit.get("score_table", ())),
        threshold=threshold,
    )
    brake = bool(audit["brake_active"])
    warmup_brake = bool(audit.get("warmup_brake_active", False))
    winning = current if brake or _framework_held(report) else raw_winner
    committed = current if warmup_brake else winning
    reason = _audit._decision_reason(
        current_count=current,
        winning_probe_count=winning,
        committed_count=committed,
        brake_active=brake,
        warmup_brake_active=warmup_brake,
    )
    replay = {
        "winning_probe_count": winning,
        "committed_count": committed,
        "decision_reason": reason,
    }
    expected = {
        key: audit[key]
        for key in ("winning_probe_count", "committed_count", "decision_reason")
    }
    if replay != expected:
        raise ValueError(
            "PLASTIC jump_to_lowest_loss audit replay mismatch: "
            f"recorded={expected}, replayed={replay}"
        )
    return replay


def _replay_v057(audit: Mapping[str, Any]) -> Dict[str, object]:
    report = _report_from_audit(audit)
    algorithm = "" if report is None else str(report.get("algorithm", ""))
    if algorithm == JUMP_TO_LOWEST_LOSS_ALGORITHM:
        return _jump_replay(audit)
    return _ORIGINAL_AUDIT_REPLAY(audit)


_audit_v055._sen_kendall_winning_count_from_audit = _standalone_winning_count_from_audit
_audit._winning_probe_count = _winning_probe_count_v057
_audit.replay_plastic_depth_count_audit = _replay_v057
# ^^^ THOG


# vvv THOG persist a mode-neutral public decision event for the three new modes.
_ORIGINAL_COMMIT_INLINE_UPDATE = _trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update


def _commit_inline_update_v057(self: Any, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    transition = _ORIGINAL_COMMIT_INLINE_UPDATE(self, context)
    if context is None or _runtime_algorithm() not in (
        *STANDALONE_STATISTICAL_ALGORITHMS,
        JUMP_TO_LOWEST_LOSS_ALGORITHM,
    ):
        return transition
    report = context.get("plastic_v055_sen_kendall_report")
    decision = context.get("decision")
    if not isinstance(report, Mapping) or decision is None:
        return transition
    payload = {
        **dict(report),
        "probe_id": context.get("plastic_probe_sequence"),
        "update_number": int(decision.update_number),
        "current_count": int(context["current_count"]),
        "selected_count": int(decision.selected_count),
        "probe_ids": tuple(int(value) for value in context.get("plastic_probe_provenance", ())),
    }
    self._record(_EVENT_NAME, **payload)
    return transition


_trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update = _commit_inline_update_v057
# ^^^ THOG


def _gradient_header_rows_v057() -> tuple[tuple[str, str], ...]:
    algorithm = _runtime_algorithm()
    rows = [(_compact._ALGORITHM_LABEL, algorithm)]
    if algorithm == SEN_ALGORITHM:
        rows.append((f"{SEN_THRESHOLD_KEY}:", f"{_runtime_threshold(SEN_THRESHOLD_KEY):g}"))
    elif algorithm == KENDALL_ALGORITHM:
        rows.append((f"{KENDALL_THRESHOLD_KEY}:", f"{_runtime_threshold(KENDALL_THRESHOLD_KEY):g}"))
    elif algorithm == JUMP_TO_LOWEST_LOSS_ALGORITHM:
        rows.extend(
            (
                (f"{JUMP_THRESHOLD_KEY}:", f"{_runtime_threshold(JUMP_THRESHOLD_KEY):g}%"),
                (
                    "decision semantics:",
                    "direct raw validation-loss minimum; configured objective, statistics, window and max-step are ignored",
                ),
            )
        )
    elif algorithm in _v056.SEN_KENDALL_ALGORITHMS:
        rows.append(
            (
                "plastic__layer_count_decision_algorithm__growth_side_discount:",
                f"{_growth._runtime_growth_side_discount():g}",
            )
        )
    return tuple(rows)


_compact._gradient_header_rows = _gradient_header_rows_v057


def _descriptor_sections_v057():
    sections = []
    for section, rows in _registry._DESCRIPTOR_SECTIONS:
        if section != _SECTION:
            sections.append((section, rows))
            continue
        retained = tuple(
            row
            for row in rows
            if row[1]
            not in {
                "--plastic__layer_count_decision_algorithm ALGORITHM",
                "objective × decision compatibility",
            }
        )
        additions = (
            (
                "LDA",
                "--plastic__layer_count_decision_algorithm ALGORITHM",
                "directional_coherence | theil_sen_kendall_LRA | sen_kendall__tau__stratified | sen | kendall | jump_to_lowest_loss",
            ),
            (
                "—",
                "objective × statistical decision compatibility",
                "all 4 objectives are permitted with directional_coherence, both combined modes, sen and kendall",
            ),
            (
                "—",
                _SEN_THRESHOLD_OPTION + " X",
                "standalone sen commits adjacent ±1 only when |slope| exceeds X and the raw adjacent objective score improves",
            ),
            (
                "—",
                _KENDALL_THRESHOLD_OPTION + " X",
                "standalone kendall commits adjacent ±1 only when |tau-b| reaches X and the raw adjacent objective score improves",
            ),
            (
                "—",
                JUMP_TO_LOWEST_LOSS_ALGORITHM,
                "bulldozer mode: jump directly to any finite probed count with the lowest raw validation loss; configured goals are ignored",
            ),
            (
                "—",
                _JUMP_THRESHOLD_OPTION + " PERCENT",
                "minimum 100 × (loss_L - loss_candidate) / |loss_L|; ignores the configured objective, statistical window and max-step",
            ),
        )
        sections.append((section, (*additions, *retained)))
    return tuple(sections)


_registry._DESCRIPTOR_SECTIONS = _descriptor_sections_v057()
# ^^^ THOG


__all__ = [
    "DECISION_ALGORITHMS",
    "JUMP_THRESHOLD_KEY",
    "JUMP_TO_LOWEST_LOSS_ALGORITHM",
    "KENDALL_ALGORITHM",
    "KENDALL_THRESHOLD_KEY",
    "OWNED_ALGORITHMS",
    "SEN_ALGORITHM",
    "SEN_THRESHOLD_KEY",
    "STANDALONE_STATISTICAL_ALGORITHMS",
    "_jump_winner",
    "_standalone_statistical_decision",
]
# ^^^ THOG
