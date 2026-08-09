# vvv THOG
"""PLASTIC v0.55 Sen/Kendall decision modes: LRA compatibility and stratified-window control."""

from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from . import plastic_depth_controller as _controller
from . import plastic_depth_console_compact_layout_patch as _compact
from . import plastic_depth_directional_coherence_patch as _directional
from . import plastic_depth_lookahead_patch as _lookahead
from . import plastic_depth_theil_sen_kendall_patch as _tsk
from . import plastic_depth_wall_time_equivalent_time_gain_patch as _wall_time
from . import run_config as _run_config
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step
from . import training_config as _training_config


LRA_ALGORITHM = "wall_time__theil_sen_kendall_LRA"
STRATIFIED_ALGORITHM = "wall_time__sen_kendall__tau__stratified"
SEN_KENDALL_ALGORITHMS = (LRA_ALGORITHM, STRATIFIED_ALGORITHM)
DECISION_ALGORITHMS = (_tsk.LEGACY_DIRECTIONAL_ALGORITHM, *SEN_KENDALL_ALGORITHMS)
FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU = 0.5
_RETIRED_ALGORITHM = "wall_time__gradient__theil_sen_kendall_slope_tau"
_RETIRED_TAU_OPTION = "--plastic__layer_count_gradient__minimum_absolute_kendall_tau"
_RETIRED_TAU_KEY = "plastic__layer_count_gradient__minimum_absolute_kendall_tau"
_LRA_HISTORY_SUFFIX = "@TSK_LRA_V055"
_STRATIFIED_HISTORY_PREFIX = "@SK_STRAT_V055"
_EVENT_NAME = "plastic_depth_v055_sen_kendall_decision"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_SCORE_Z = re.compile(r"[ \t]+score_z \[[^\]]+\] = \[[^\]]*\]")
_OLD_TSK = re.compile(r"[ \t]+ts=[^ \t]+s/layer[ \t]+tau=[^ \t]+")
_LRA_SUMMARY = re.compile(
    r"(?P<marker>(?:L/R/A|⇩\|⇧\|\?|↓\|↑\|\?|▼\|▲\|\?))\s*=\s*"
    r"\[(?P<left>\d+)/(?P<right>\d+)/(?P<ambiguous>\d+)\]/(?P<total>\d+)"
    r"(?:=>(?P<outcome>stet|L|R|-|⇩|⇧|↓|↑|▼|▲|●))?"
)
_PROVENANCE = re.compile(r"[ \t]+\(P(?P<body>[0-9,P ]+)\)")


def _runtime_algorithm() -> str:
    value = os.environ.get(_tsk._ALGORITHM_ENV, _tsk.LEGACY_DIRECTIONAL_ALGORITHM).strip()
    if value == _RETIRED_ALGORITHM:
        raise ValueError(
            f"{_RETIRED_ALGORITHM} was renamed in PLASTIC v0.55; use {LRA_ALGORITHM}"
        )
    if value not in DECISION_ALGORITHMS:
        raise ValueError(
            f"{_tsk._ALGORITHM_KEY} must be one of {DECISION_ALGORITHMS}; got {value!r}"
        )
    return value


def _set_runtime_algorithm(value: str) -> None:
    resolved = str(value).strip()
    if resolved == _RETIRED_ALGORITHM:
        raise ValueError(
            f"{_RETIRED_ALGORITHM} was renamed in PLASTIC v0.55; use {LRA_ALGORITHM}"
        )
    if resolved not in DECISION_ALGORITHMS:
        raise ValueError(
            f"{_tsk._ALGORITHM_KEY} must be one of {DECISION_ALGORITHMS}; got {value!r}"
        )
    os.environ[_tsk._ALGORITHM_ENV] = resolved


def _fixed_tau() -> float:
    return FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU


# Retarget the v0.54 compatibility layer to the renamed LRA mode; v0.55 dispatch below owns both Sen/Kendall modes.
_tsk.GRADIENT_ALGORITHM = LRA_ALGORITHM
_tsk.DECISION_ALGORITHMS = DECISION_ALGORITHMS
_tsk._runtime_algorithm = _runtime_algorithm
_tsk._set_runtime_algorithm = _set_runtime_algorithm
_tsk._runtime_minimum_absolute_kendall_tau = _fixed_tau
_tsk.DEFAULT_MINIMUM_ABSOLUTE_KENDALL_TAU = FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU
os.environ.pop(_tsk._TAU_ENV, None)


# vvv THOG remove the retired public tau control while extending the existing algorithm selector with both v0.55 names
_ORIGINAL_PARSE_ARGS = argparse.ArgumentParser.parse_args
_ORIGINAL_PARSE_KNOWN_ARGS = argparse.ArgumentParser.parse_known_args
_ORIGINAL_FORMAT_HELP = argparse.ArgumentParser.format_help


def _contains_retired_tau(arguments: Optional[Sequence[str]]) -> bool:
    source = list(sys.argv[1:] if arguments is None else arguments)
    return any(
        argument == _RETIRED_TAU_OPTION or argument.startswith(_RETIRED_TAU_OPTION + "=")
        for argument in source
    )


def _strip_retired_namespace_field(namespace: Any) -> Any:
    if hasattr(namespace, _RETIRED_TAU_KEY):
        try:
            delattr(namespace, _RETIRED_TAU_KEY)
        except AttributeError:
            pass
    return namespace


def _parse_args_v055(self: argparse.ArgumentParser, args=None, namespace=None):
    if _contains_retired_tau(args):
        raise SystemExit(
            f"{_RETIRED_TAU_OPTION} was removed in PLASTIC v0.55; Kendall coherence is fixed at "
            f"|tau| >= {FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU:g} for the Sen/Kendall algorithms"
        )
    return _strip_retired_namespace_field(_ORIGINAL_PARSE_ARGS(self, args, namespace))


def _parse_known_args_v055(self: argparse.ArgumentParser, args=None, namespace=None):
    if _contains_retired_tau(args):
        raise SystemExit(
            f"{_RETIRED_TAU_OPTION} was removed in PLASTIC v0.55; Kendall coherence is fixed at "
            f"|tau| >= {FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU:g} for the Sen/Kendall algorithms"
        )
    parsed, extras = _ORIGINAL_PARSE_KNOWN_ARGS(self, args, namespace)
    return _strip_retired_namespace_field(parsed), extras


def _format_help_v055(self: argparse.ArgumentParser) -> str:
    rendered = _ORIGINAL_FORMAT_HELP(self)
    lines = rendered.splitlines()
    output = []
    skip_description = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(_RETIRED_TAU_OPTION):
            skip_description = True
            continue
        if skip_description and line.startswith("                        "):
            skip_description = False
            continue
        skip_description = False
        if "directional_coherence (default) or " in line and "wall_time__" in line:
            output.append(
                "                        directional_coherence (default), "
                f"{LRA_ALGORITHM}, or {STRATIFIED_ALGORITHM}"
            )
            continue
        output.append(line)
    return "\n".join(output) + ("\n" if rendered.endswith("\n") else "")


argparse.ArgumentParser.parse_args = _parse_args_v055
argparse.ArgumentParser.parse_known_args = _parse_known_args_v055
argparse.ArgumentParser.format_help = _format_help_v055
# ^^^ THOG


# vvv THOG persist only the selected v0.55 algorithm name; the retired configurable Kendall threshold is no longer identity
_ORIGINAL_TRAINING_POST_INIT = _training_config.TrainingConfig.__post_init__
_ORIGINAL_TRAINING_PERSISTENT_DICT = _training_config.TrainingConfig.persistent_dict
_ORIGINAL_TRAINING_COMPACT_IDENTITY = _training_config.TrainingConfig.compact_identity_metadata
_ORIGINAL_RUN_PERSISTENT_DICT = _run_config.OwtRunConfig.persistent_dict
_ORIGINAL_RUN_COMPACT_IDENTITY = _run_config.OwtRunConfig.compact_identity


def _validate_v055_config(config: Any) -> None:
    algorithm = _runtime_algorithm()
    if algorithm not in SEN_KENDALL_ALGORITHMS:
        return
    if not bool(getattr(config, "plastic__enabled", False)) or not bool(
        getattr(config, "plastic__do_learn_layer_count", False)
    ):
        raise ValueError(f"{algorithm} requires learned-count PLASTIC DEPTH")
    if str(getattr(config, "plastic__layer_count_objective", "")) != "relative_training_wall_time":
        raise ValueError(
            f"{algorithm} requires plastic__layer_count_objective=relative_training_wall_time"
        )
    if int(getattr(config, "plastic__layer_count__max_allowable_layer_change", 1)) != 1:
        raise ValueError(
            f"{algorithm} requires plastic__layer_count__max_allowable_layer_change=1"
        )


def _training_post_init_v055(self: Any) -> None:
    _ORIGINAL_TRAINING_POST_INIT(self)
    _validate_v055_config(self)


def _persistent_v055(values: Dict[str, Any], *, plastic_enabled: bool) -> Dict[str, Any]:
    updated = dict(values)
    updated.pop(_RETIRED_TAU_KEY, None)
    algorithm = _runtime_algorithm()
    if plastic_enabled and algorithm in SEN_KENDALL_ALGORITHMS:
        updated[_tsk._ALGORITHM_KEY] = algorithm
    return updated


def _training_persistent_dict_v055(self: Any) -> Dict[str, Any]:
    return _persistent_v055(
        _ORIGINAL_TRAINING_PERSISTENT_DICT(self),
        plastic_enabled=bool(getattr(self, "plastic__enabled", False)),
    )


def _run_persistent_dict_v055(self: Any) -> Dict[str, Any]:
    return _persistent_v055(
        _ORIGINAL_RUN_PERSISTENT_DICT(self),
        plastic_enabled=bool(getattr(self, "plastic__enabled", False)),
    )


def _compact_identity_v055(values: Dict[str, Any], *, plastic_enabled: bool) -> Dict[str, Any]:
    updated = dict(values)
    plastic = updated.get("plastic_depth")
    if isinstance(plastic, Mapping):
        plastic_updated = dict(plastic)
        plastic_updated.pop(_RETIRED_TAU_KEY, None)
        algorithm = _runtime_algorithm()
        if plastic_enabled and algorithm in SEN_KENDALL_ALGORITHMS:
            plastic_updated[_tsk._ALGORITHM_KEY] = algorithm
        updated["plastic_depth"] = plastic_updated
    return updated


def _training_compact_identity_v055(self: Any) -> Dict[str, Any]:
    return _compact_identity_v055(
        _ORIGINAL_TRAINING_COMPACT_IDENTITY(self),
        plastic_enabled=bool(getattr(self, "plastic__enabled", False)),
    )


def _run_compact_identity_v055(self: Any) -> Dict[str, Any]:
    return _compact_identity_v055(
        _ORIGINAL_RUN_COMPACT_IDENTITY(self),
        plastic_enabled=bool(getattr(self, "plastic__enabled", False)),
    )


_training_config.TrainingConfig.__post_init__ = _training_post_init_v055
_training_config.TrainingConfig.persistent_dict = _training_persistent_dict_v055
_training_config.TrainingConfig.compact_identity_metadata = _training_compact_identity_v055
_run_config.OwtRunConfig.persistent_dict = _run_persistent_dict_v055
_run_config.OwtRunConfig.compact_identity = _run_compact_identity_v055
# ^^^ THOG


# vvv THOG v0.55 startup shows only the decision-algorithm selector; the configurable Kendall threshold was retired
def _sen_kendall_header_rows_v055() -> tuple[tuple[str, str], ...]:
    return ((_compact._ALGORITHM_LABEL, _runtime_algorithm()),)


_compact._gradient_header_rows = _sen_kendall_header_rows_v055
# ^^^ THOG


# vvv THOG replace raw-loss decision fallback with a deterministic provisional timing statistic until two-count ordinary timing exists
_ORIGINAL_TIMING_FIT = _wall_time._timing_fit


def _timing_fit_v055(trainer: Any) -> Optional[Dict[str, float]]:
    empirical = _ORIGINAL_TIMING_FIT(trainer)
    if empirical is not None or _runtime_algorithm() not in SEN_KENDALL_ALGORITHMS:
        return empirical
    state = _wall_time._runtime_state(trainer)
    timing_n = int(state.get("timing_n", 0))
    distinct = tuple(sorted(int(value) for value in state.get("timing_distinct_counts", ())))
    if timing_n < 1 or len(distinct) != 1:
        return None
    current_count = distinct[0]
    if current_count < 1:
        return None
    mean_update_seconds = float(state.get("timing_sum_y", 0.0)) / float(timing_n)
    if not math.isfinite(mean_update_seconds) or mean_update_seconds <= 0.0:
        return None
    lattice = trainer._plastic_depth_lattice()
    optimizer_step_seconds = 0.0
    if lattice is not None:
        candidate = float(lattice.optimizer_step_time_ema.item())
        if math.isfinite(candidate) and 0.0 <= candidate < mean_update_seconds:
            optimizer_step_seconds = candidate
    slope = (mean_update_seconds - optimizer_step_seconds) / float(current_count)
    if not math.isfinite(slope) or slope <= 0.0:
        return None
    return {
        "slope": float(slope),
        "intercept": float(optimizer_step_seconds),
        "r_squared": float("nan"),
        "slope_standard_error": float("nan"),
        "sse": float("nan"),
        "sst": float("nan"),
        "provisional": 1.0,
    }


_wall_time._timing_fit = _timing_fit_v055
# ^^^ THOG


def _copy_histories(
    histories: Mapping[str, Sequence[float]],
    window: int,
) -> Dict[str, Tuple[float, ...]]:
    copied: Dict[str, Tuple[float, ...]] = {}
    for key, values in histories.items():
        resolved = tuple(float(value) for value in values[-window:])
        if all(math.isfinite(value) for value in resolved):
            copied[str(key)] = resolved
    return copied


def _brake_active(
    *,
    update_number: int,
    last_count_change_update: int,
    update_brake: int,
) -> bool:
    return (
        update_brake > 0
        and last_count_change_update >= 0
        and update_number - last_count_change_update < update_brake
    )


def _adjacent_difference(
    score_by_count: Mapping[int, float],
    current: int,
    direction: int,
) -> Optional[float]:
    current_score = score_by_count.get(current)
    candidate_score = score_by_count.get(current + direction)
    if current_score is None or candidate_score is None:
        return None
    value = float(candidate_score - current_score)
    return value if math.isfinite(value) else None


def _evidence(
    *,
    current: int,
    score_by_count: Mapping[int, float],
    observation_count: int,
    selected_count: int,
) -> Tuple[_controller.PlasticDepthPairedDirectionEvidence, ...]:
    result = []
    for direction in (-1, 1):
        paired = _adjacent_difference(score_by_count, current, direction)
        candidate = current + direction
        result.append(
            _controller.PlasticDepthPairedDirectionEvidence(
                candidate_count=candidate,
                direction=direction,
                paired_difference=paired,
                observation_count=observation_count,
                median=None,
                mad=None,
                sigma=None,
                standardized_improvement=None,
                significant=candidate == selected_count and selected_count != current,
                feasible=paired is not None,
            )
        )
    return tuple(result)


def _current_probe_adjacent_from_sen(
    score_by_count: Mapping[int, float],
    current: int,
    sen: float,
) -> Optional[float]:
    if not math.isfinite(sen) or sen == 0.0:
        return None
    return _adjacent_difference(score_by_count, current, -1 if sen > 0.0 else 1)


def _store_report(trainer: Any, report: Dict[str, Any]) -> None:
    context = getattr(trainer, "_plastic_depth_inline_update_context", None)
    if isinstance(context, dict):
        context["plastic_v055_sen_kendall_report"] = dict(report)
        context["plastic_directional_report"] = dict(report)


# vvv THOG renamed LRA mode keeps per-probe TSK plus strict-majority window but removes MAD/score_z candidate gating entirely
def choose_plastic_depth_count_with_tsk_lra_v055(
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
) -> _controller.PlasticDepthRobustCountDecision:
    del noise_lambda, minimum_observations, max_step
    window = _directional._positive_integer(
        noise_window,
        name="plastic__layer_count_probe__window_size_as_number_of_probes",
    )
    current = int(current_count)
    updated = _copy_histories(histories, window)
    score_by_count = _tsk._equivalent_time_scores(score_report)
    brake = _brake_active(
        update_number=update_number,
        last_count_change_update=last_count_change_update,
        update_brake=update_brake,
    )
    probe = _tsk._gradient_probe_classification(
        current_count=current,
        score_report=score_report,
        minimum_absolute_kendall_tau=FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU,
    )
    history_key = f"{current}:{_LRA_HISTORY_SUFFIX}"
    votes = list(updated.get(history_key, ()))
    if score_by_count:
        votes.append(float(probe["per_probe_vote"]))
        votes = votes[-window:]
        updated[history_key] = tuple(votes)
    left_votes, right_votes, ambiguous_votes = _directional._direction_vote_counts(votes)
    conclusion = _directional._direction_conclusion(votes, window)
    sen = float(probe["theil_sen_slope_seconds_per_layer"])
    ken = float(probe["kendall_tau"])
    adjacent = _current_probe_adjacent_from_sen(score_by_count, current, sen)
    selected = current
    if len(votes) >= window and not brake:
        if conclusion == "L":
            latest_adjacent = _adjacent_difference(score_by_count, current, -1)
            if latest_adjacent is not None and latest_adjacent < 0.0:
                selected = current - 1
        elif conclusion == "R":
            latest_adjacent = _adjacent_difference(score_by_count, current, 1)
            if latest_adjacent is not None and latest_adjacent < 0.0:
                selected = current + 1
    if selected != current:
        updated = {}
    report = {
        **probe,
        "algorithm": LRA_ALGORITHM,
        "sen_slope_seconds_per_layer": sen,
        "kendall_tau": ken,
        "adjacent_score_seconds": adjacent,
        "left_votes": left_votes,
        "right_votes": right_votes,
        "ambiguous_votes": ambiguous_votes,
        "vote_total": len(votes),
        "direction_window_votes": tuple(
            "L" if value < 0.0 else "R" if value > 0.0 else "A" for value in votes
        ),
        "direction_conclusion": conclusion,
        "conclusion": conclusion,
        "selected_count": selected,
        "brake_active": brake,
        "window_ready": len(votes) >= window,
    }
    trainer = _wall_time._ACTIVE_TRAINER.get()
    if trainer is not None:
        _store_report(trainer, report)
    return _controller.PlasticDepthRobustCountDecision(
        selected_count=selected,
        current_count=current,
        update_number=int(update_number),
        brake_active=brake,
        last_count_change_update=int(last_count_change_update),
        histories=updated,
        evidence=_evidence(
            current=current,
            score_by_count=score_by_count,
            observation_count=len(votes),
            selected_count=selected,
        ),
    )
# ^^^ THOG


def _stratified_history_key(current: int, offset: int) -> str:
    return f"{current}:{_STRATIFIED_HISTORY_PREFIX}:{offset:+d}"


def stratified_sen_slope(
    strata: Sequence[Sequence[Tuple[float, float]]],
) -> float:
    slopes = []
    for points in strata:
        finite = tuple(
            (float(x), float(y))
            for x, y in points
            if math.isfinite(float(x)) and math.isfinite(float(y))
        )
        for index, (x_i, y_i) in enumerate(finite):
            for x_j, y_j in finite[index + 1 :]:
                dx = x_j - x_i
                if dx != 0.0:
                    slopes.append((y_j - y_i) / dx)
    return float(statistics.median(slopes)) if slopes else float("nan")


def stratified_kendall_tau_b(
    strata: Sequence[Sequence[Tuple[float, float]]],
) -> Tuple[float, int, int, int]:
    signed_pairs = 0
    denominator_sum = 0.0
    concordant_total = 0
    discordant_total = 0
    y_ties_total = 0
    for points in strata:
        _tau, concordant, discordant, x_ties, y_ties = _tsk.kendall_tau_b(points)
        if x_ties:
            raise ValueError("stratified Kendall requires unique layer offsets within each probe")
        denominator = math.sqrt(
            float(concordant + discordant)
            * float(concordant + discordant + y_ties)
        )
        signed_pairs += int(concordant - discordant)
        denominator_sum += denominator
        concordant_total += int(concordant)
        discordant_total += int(discordant)
        y_ties_total += int(y_ties)
    tau = (
        float(signed_pairs) / denominator_sum
        if denominator_sum > 0.0
        else float("nan")
    )
    return tau, concordant_total, discordant_total, y_ties_total


def _median_adjacent_from_strata(
    strata: Sequence[Mapping[int, float]],
    direction: int,
) -> Optional[float]:
    values = []
    for row in strata:
        current = row.get(0)
        candidate = row.get(direction)
        if current is None or candidate is None:
            continue
        difference = float(candidate - current)
        if math.isfinite(difference):
            values.append(difference)
    return float(statistics.median(values)) if values else None


# vvv THOG new stratified mode pools only within-probe pairwise Sen/Kendall evidence across the configured window and commits one adjacent layer
def choose_plastic_depth_count_with_stratified_sen_kendall_v055(
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
) -> _controller.PlasticDepthRobustCountDecision:
    del noise_lambda, minimum_observations, max_step
    window = _directional._positive_integer(
        noise_window,
        name="plastic__layer_count_probe__window_size_as_number_of_probes",
    )
    current = int(current_count)
    updated = _copy_histories(histories, window)
    score_by_count = _tsk._equivalent_time_scores(score_report)
    all_offsets = _directional._candidate_offsets(
        current_count=current,
        score_report=score_report,
    )
    decision_offsets = tuple(sorted({0, *(offset for offset in all_offsets if offset <= 1)}))
    complete_probe = (
        len(decision_offsets) >= 2
        and all((current + offset) in score_by_count for offset in decision_offsets)
    )
    if complete_probe:
        current_score = float(score_by_count[current])
        for offset in decision_offsets:
            key = _stratified_history_key(current, offset)
            values = list(updated.get(key, ()))
            values.append(float(score_by_count[current + offset] - current_score))
            updated[key] = tuple(values[-window:])

    lengths = [
        len(updated.get(_stratified_history_key(current, offset), ()))
        for offset in decision_offsets
    ]
    row_count = min(lengths) if lengths else 0
    rows = []
    row_maps = []
    if row_count > 0:
        for row_index in range(-row_count, 0):
            points = []
            row_map: Dict[int, float] = {}
            for offset in decision_offsets:
                values = updated.get(_stratified_history_key(current, offset), ())
                value = float(values[row_index])
                points.append((float(offset), value))
                row_map[int(offset)] = value
            rows.append(tuple(points))
            row_maps.append(row_map)

    sen = stratified_sen_slope(rows) if rows else float("nan")
    ken, concordant, discordant, y_ties = (
        stratified_kendall_tau_b(rows) if rows else (float("nan"), 0, 0, 0)
    )
    left_adjacent = _median_adjacent_from_strata(row_maps, -1)
    right_adjacent = _median_adjacent_from_strata(row_maps, 1)
    proposed_direction = -1 if math.isfinite(sen) and sen > 0.0 else 1 if math.isfinite(sen) and sen < 0.0 else 0
    adjacent = left_adjacent if proposed_direction < 0 else right_adjacent if proposed_direction > 0 else None
    brake = _brake_active(
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
            and ken >= FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU
            and adjacent is not None
            and adjacent < 0.0
        ):
            conclusion = "L"
            if not brake:
                selected = current - 1
    elif ready and proposed_direction > 0:
        if (
            math.isfinite(ken)
            and ken <= -FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU
            and adjacent is not None
            and adjacent < 0.0
        ):
            conclusion = "R"
            if not brake:
                selected = current + 1
    if selected != current:
        updated = {}
    report = {
        "algorithm": STRATIFIED_ALGORITHM,
        "decision_fit_offsets": decision_offsets,
        "candidate_offsets": all_offsets,
        "sen_slope_seconds_per_layer": sen,
        "theil_sen_slope_seconds_per_layer": sen,
        "kendall_tau": ken,
        "adjacent_score_seconds": adjacent,
        "left_adjacent_score_seconds": left_adjacent,
        "right_adjacent_score_seconds": right_adjacent,
        "minimum_absolute_kendall_tau": FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU,
        "strata_count": row_count,
        "vote_total": row_count,  # framework window-alignment count; stratified mode has no L/R/A votes
        "window_ready": ready,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "y_tied_pairs": y_ties,
        "direction_conclusion": conclusion,
        "conclusion": conclusion,
        "selected_count": selected,
        "brake_active": brake,
    }
    trainer = _wall_time._ACTIVE_TRAINER.get()
    if trainer is not None:
        _store_report(trainer, report)
    return _controller.PlasticDepthRobustCountDecision(
        selected_count=selected,
        current_count=current,
        update_number=int(update_number),
        brake_active=brake,
        last_count_change_update=int(last_count_change_update),
        histories=updated,
        evidence=_evidence(
            current=current,
            score_by_count=score_by_count,
            observation_count=row_count,
            selected_count=selected,
        ),
    )
# ^^^ THOG


# vvv THOG final selector dispatch bypasses the v0.54 raw-loss bootstrap controller for both v0.55 Sen/Kendall modes
_ORIGINAL_SELECTOR = _lookahead.choose_plastic_depth_count_with_exact_radius


def _choose_count_v055(*, max_step: int = 1, extrapolation_weight: float = 0.8, **kwargs: Any):
    algorithm = _runtime_algorithm()
    if algorithm == LRA_ALGORITHM:
        return choose_plastic_depth_count_with_tsk_lra_v055(max_step=1, **kwargs)
    if algorithm == STRATIFIED_ALGORITHM:
        return choose_plastic_depth_count_with_stratified_sen_kendall_v055(max_step=1, **kwargs)
    return _ORIGINAL_SELECTOR(
        max_step=max_step,
        extrapolation_weight=extrapolation_weight,
        **kwargs,
    )


_lookahead.choose_plastic_depth_count_with_exact_radius = _choose_count_v055
_controller.choose_plastic_depth_count_with_mad = _choose_count_v055
_trainer_step.choose_plastic_depth_count_with_mad = _choose_count_v055
# ^^^ THOG


# vvv THOG persist window-level Sen/Kendall diagnostics and provenance independently of the retired score_z machinery
_ORIGINAL_COMMIT_INLINE_UPDATE = _trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update


def _commit_inline_update_v055(
    self: Any,
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    transition = _ORIGINAL_COMMIT_INLINE_UPDATE(self, context)
    if context is None or _runtime_algorithm() not in SEN_KENDALL_ALGORITHMS:
        return transition
    report = context.get("plastic_v055_sen_kendall_report")
    decision = context.get("decision")
    if not isinstance(report, Mapping) or decision is None:
        return transition
    payload = dict(report)
    payload.update(
        {
            "probe_id": context.get("plastic_probe_sequence"),
            "update_number": int(decision.update_number),
            "current_count": int(context["current_count"]),
            "selected_count": int(decision.selected_count),
            "direction_window_probe_ids": tuple(
                int(value) for value in context.get("plastic_probe_provenance", ())
            ),
        }
    )
    self._record(_EVENT_NAME, **payload)
    rows = getattr(self, "plastic_depth_count_audit", None)
    if isinstance(rows, list) and rows:
        row = rows[-1]
        if int(row.get("update_number", -1)) == int(decision.update_number):
            row["sen_kendall_decision"] = payload
    return transition


_trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update = _commit_inline_update_v055
# ^^^ THOG


# vvv THOG final operator console always keeps probe_Δloss, hides score_z for Sen/Kendall modes, and uses explicit therefore glyphs
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line


def _latest_v055_report(trainer: Any) -> Optional[Dict[str, Any]]:
    for event in reversed(getattr(trainer, "events", ())):
        if event.name == _EVENT_NAME:
            return dict(event.payload)
    return None


def _prepare_console_progress_payload_v055(
    self: Any,
    event: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    if _runtime_algorithm() not in SEN_KENDALL_ALGORITHMS or event not in {
        "optimizer_progress",
        "evaluation_completed",
    }:
        return values
    report = _latest_v055_report(self)
    if report is None:
        return values
    try:
        completed_updates = int(values.get("completed_updates", payload.get("completed_updates")))
    except (TypeError, ValueError):
        return values
    if completed_updates != int(report.get("update_number", -1)):
        return values
    values["plastic_v055_algorithm"] = report.get("algorithm")
    values["plastic_v055_sen"] = report.get("sen_slope_seconds_per_layer")
    values["plastic_v055_ken"] = report.get("kendall_tau")
    values["plastic_v055_adj"] = report.get("adjacent_score_seconds")
    values["plastic_v055_conclusion"] = report.get("direction_conclusion", "-")
    values["plastic_v055_selected_count"] = report.get("selected_count")
    values["plastic_v055_current_count"] = report.get("current_count")
    values["plastic_v055_probe_ids"] = report.get("direction_window_probe_ids", ())
    values["plastic_v055_left_votes"] = report.get("left_votes")
    values["plastic_v055_right_votes"] = report.get("right_votes")
    values["plastic_v055_ambiguous_votes"] = report.get("ambiguous_votes")
    values["plastic_v055_vote_total"] = report.get("vote_total")
    return values


def _number(value: Any, digits: int) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(numeric):
        return "-"
    return f"{numeric:+.{digits}f}"


def _compact_probe_ids(values: Sequence[Any]) -> str:
    ids = []
    for value in values:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    if not ids:
        return ""
    return "(P" + ",".join(str(value) for value in ids) + ")"


def _outcome_symbol(conclusion: str, selected: Any, current: Any) -> str:
    del conclusion
    try:
        selected_count = int(selected)
        current_count = int(current)
    except (TypeError, ValueError):
        return "●"
    if selected_count < current_count:
        return "▼"
    if selected_count > current_count:
        return "▲"
    return "●"


def _format_progress_line_v055(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    local = dict(payload)
    algorithm = local.pop("plastic_v055_algorithm", None)
    sen = local.pop("plastic_v055_sen", None)
    ken = local.pop("plastic_v055_ken", None)
    adj = local.pop("plastic_v055_adj", None)
    conclusion = str(local.pop("plastic_v055_conclusion", "-"))
    selected = local.pop("plastic_v055_selected_count", None)
    current = local.pop("plastic_v055_current_count", None)
    probe_ids = tuple(local.pop("plastic_v055_probe_ids", ()) or ())
    left_votes = local.pop("plastic_v055_left_votes", None)
    right_votes = local.pop("plastic_v055_right_votes", None)
    ambiguous_votes = local.pop("plastic_v055_ambiguous_votes", None)
    vote_total = local.pop("plastic_v055_vote_total", None)
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, local)
    if algorithm not in SEN_KENDALL_ALGORITHMS:
        return line
    line = _SCORE_Z.sub("", line)
    line = _OLD_TSK.sub("", line)
    line = re.sub(r"[ \t]{2,}", "  ", line)
    diagnostic = f"sen={_number(sen, 3)} ken={_number(ken, 2)} adj={_number(adj, 3)}"
    provenance = _compact_probe_ids(probe_ids)
    outcome = _outcome_symbol(conclusion, selected, current)

    if algorithm == LRA_ALGORITHM:
        match = _LRA_SUMMARY.search(line)
        if match is not None:
            total = int(match.group("total"))
            counts = (
                int(match.group("left")),
                int(match.group("right")),
                int(match.group("ambiguous")),
            )
            rendered = (
                f"{diagnostic} ∴ ▼|▲|? =[{counts[0]}/{counts[1]}/{counts[2]}]/{total} "
                f"∴ {outcome}"
            )
            line = line[: match.start()] + rendered + line[match.end() :]
        else:
            line = f"{line}  {diagnostic} ∴ {outcome}"
    else:
        match = _LRA_SUMMARY.search(line)
        if match is not None:
            line = line[: match.start()].rstrip() + line[match.end() :]
        line = f"{line.rstrip()}  {diagnostic} ∴ {outcome}"

    line = _PROVENANCE.sub("", line)
    if provenance:
        line = f"{line.rstrip()} {provenance}"
    return line


_stage6.Stage6Trainer._prepare_console_progress_payload = _prepare_console_progress_payload_v055
_stage6.format_progress_line = _format_progress_line_v055
# ^^^ THOG


__all__ = [
    "DECISION_ALGORITHMS",
    "FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU",
    "LRA_ALGORITHM",
    "SEN_KENDALL_ALGORITHMS",
    "STRATIFIED_ALGORITHM",
    "choose_plastic_depth_count_with_stratified_sen_kendall_v055",
    "choose_plastic_depth_count_with_tsk_lra_v055",
    "stratified_kendall_tau_b",
    "stratified_sen_slope",
]
# ^^^ THOG
