# vvv THOG
"""Final PLASTIC DEPTH directional-coherence gate and compact probe console diagnostics."""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import constants as _constants

from . import plastic_depth_console_minor_patch as _console_minor
from . import plastic_depth_controller as _controller
from . import plastic_depth_lookahead_patch as _lookahead
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step


_DEFAULT_EXTRAPOLATION_WEIGHT = 0.8
_DIRECTION_HISTORY_SUFFIX = "@LRA"
_DIRECTION_LEFT = -1.0
_DIRECTION_AMBIGUOUS = 0.0
_DIRECTION_RIGHT = 1.0
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_SAMPLED_ARRAY = re.compile(r"sampled = \[(?P<body>[^\]]*)\]")
_PROBE_VECTOR = re.compile(
    r"(?P<prefix>probe_losses \[[^\]]+\] = \[)(?P<body>[^\]]*)(?P<close>\])"
)
_LAYER_FIELD = re.compile(r"layers = (?P<count>\d+)")
_SAMPLED_BY_RUN_ID: Dict[str, Tuple[str, ...]] = {}
_MIN_FINAL_SAMPLED_COLUMN = 328


def _positive_integer(value: Any, *, name: str) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return resolved


def _validate_extrapolation_weight(value: Any) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or not (0.5 < resolved <= 1.0):
        raise ValueError(
            "plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence must be finite and lie in (0.5, 1.0]; "
            f"got {value!r}"
        )
    return resolved


def _history_key(current_count: int, offset: int) -> str:
    if int(offset) == 0:
        raise ValueError("PLASTIC DEPTH history offset must be non-zero")
    return f"{int(current_count)}:{int(offset):+d}"


def _direction_history_key(current_count: int) -> str:
    return f"{int(current_count)}:{_DIRECTION_HISTORY_SUFFIX}"


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


def _candidate_offsets(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
) -> Tuple[int, ...]:
    values = []
    for item in score_report:
        try:
            offset = int(item["active_layers"]) - int(current_count)
        except (KeyError, TypeError, ValueError):
            continue
        if offset != 0:
            values.append(offset)
    return tuple(sorted(set(values)))


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


def _directional_support(
    *,
    current_count: int,
    score_by_count: Mapping[int, float],
    extrapolation_weight: float,
) -> Dict[str, Any]:
    current_score = score_by_count.get(int(current_count))
    negative_offsets = tuple(
        sorted(
            count - int(current_count)
            for count in score_by_count
            if count < int(current_count)
        )
    )
    positive_offsets = tuple(
        sorted(
            count - int(current_count)
            for count in score_by_count
            if count > int(current_count)
        )
    )

    left_support: Optional[float] = None
    if current_score is not None and negative_offsets:
        left_wins = tuple(
            1.0 if score_by_count[int(current_count) + offset] < current_score else 0.0
            for offset in negative_offsets
        )
        left_support = sum(left_wins) / len(left_wins)

    right_support: Optional[float] = None
    if current_score is not None and positive_offsets:
        weights = tuple(
            extrapolation_weight ** (abs(offset) - 1)
            for offset in positive_offsets
        )
        weighted_wins = sum(
            weight
            * (
                1.0
                if score_by_count[int(current_count) + offset] < current_score
                else 0.0
            )
            for offset, weight in zip(positive_offsets, weights)
        )
        right_support = extrapolation_weight * weighted_wins / sum(weights)

    left_majority = left_support is not None and left_support > 0.5
    right_majority = right_support is not None and right_support > 0.5
    if left_majority and not right_majority:
        vote = _DIRECTION_LEFT
    elif right_majority and not left_majority:
        vote = _DIRECTION_RIGHT
    else:
        vote = _DIRECTION_AMBIGUOUS

    return {
        "left_support": left_support,
        "right_support": right_support,
        "vote": vote,
        "negative_offsets": negative_offsets,
        "positive_offsets": positive_offsets,
    }


def _direction_vote_counts(values: Sequence[float]) -> Tuple[int, int, int]:
    left = sum(value == _DIRECTION_LEFT for value in values)
    right = sum(value == _DIRECTION_RIGHT for value in values)
    ambiguous = sum(value == _DIRECTION_AMBIGUOUS for value in values)
    return left, right, ambiguous


def _direction_conclusion(values: Sequence[float], noise_window: int) -> str:
    if len(values) < int(noise_window):
        return "-"
    left, right, _ambiguous = _direction_vote_counts(values)
    if left * 2 > len(values):
        return "L"
    if right * 2 > len(values):
        return "R"
    return "-"


def _updated_histories_and_direction(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    histories: Mapping[str, Sequence[float]],
    noise_window: int,
    extrapolation_weight: float,
) -> Tuple[Dict[str, Tuple[float, ...]], Dict[str, Any]]:
    updated_histories: Dict[str, Tuple[float, ...]] = {}
    for key, values in histories.items():
        resolved_values = tuple(float(value) for value in values[-noise_window:])
        if not all(math.isfinite(value) for value in resolved_values):
            raise ValueError(
                f"PLASTIC DEPTH paired-score history {key!r} contains a non-finite value"
            )
        updated_histories[str(key)] = resolved_values

    score_by_count = _finite_score_by_count(score_report)
    current_score = score_by_count.get(int(current_count))
    candidate_offsets = _candidate_offsets(
        current_count=int(current_count),
        score_report=score_report,
    )
    for offset in candidate_offsets:
        candidate_score = score_by_count.get(int(current_count) + offset)
        if current_score is None or candidate_score is None:
            continue
        key = _history_key(int(current_count), offset)
        values = list(updated_histories.get(key, ()))
        values.append(float(candidate_score - current_score))
        updated_histories[key] = tuple(values[-noise_window:])

    side = _directional_support(
        current_count=int(current_count),
        score_by_count=score_by_count,
        extrapolation_weight=extrapolation_weight,
    )
    direction_key = _direction_history_key(int(current_count))
    direction_values = list(updated_histories.get(direction_key, ()))
    direction_values.append(float(side["vote"]))
    updated_histories[direction_key] = tuple(direction_values[-noise_window:])
    direction_values = list(updated_histories[direction_key])
    left_votes, right_votes, ambiguous_votes = _direction_vote_counts(direction_values)

    left_offsets = tuple(offset for offset in candidate_offsets if offset < 0)
    right_offsets = tuple(offset for offset in candidate_offsets if offset > 0)
    left_win_counts = tuple(
        sum(value < 0.0 for value in updated_histories.get(_history_key(int(current_count), offset), ()))
        for offset in left_offsets
    )
    right_win_counts = tuple(
        sum(value < 0.0 for value in updated_histories.get(_history_key(int(current_count), offset), ()))
        for offset in right_offsets
    )

    report = {
        **side,
        "direction_history": tuple(direction_values),
        "left_votes": left_votes,
        "right_votes": right_votes,
        "ambiguous_votes": ambiguous_votes,
        "vote_total": len(direction_values),
        "conclusion": _direction_conclusion(direction_values, noise_window),
        "left_offsets": left_offsets,
        "right_offsets": right_offsets,
        "left_win_counts": left_win_counts,
        "right_win_counts": right_win_counts,
    }
    return updated_histories, report


def choose_plastic_depth_count_with_directional_coherence(
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
    extrapolation_weight: float = _DEFAULT_EXTRAPOLATION_WEIGHT,
    minimum_observations: Optional[int] = None,
) -> _controller.PlasticDepthRobustCountDecision:
    """Apply full-window candidate evidence plus goal-agnostic L/R/A coherence."""

    del minimum_observations  # compatibility only; readiness is now exactly the full configured history window
    resolved_noise_window = _positive_integer(
        noise_window,
        name="plastic__layer_count_probe__window_size_as_number_of_probes",
    )
    if not math.isfinite(noise_lambda) or noise_lambda < 0.0:
        raise ValueError("noise_lambda must be finite and non-negative")
    if update_number < 1:
        raise ValueError("update_number must be positive")
    if update_brake < 0:
        raise ValueError("update_brake must be non-negative")
    resolved_max_step = _positive_integer(
        max_step,
        name="plastic__layer_count__max_allowable_layer_change",
    )
    resolved_weight = _validate_extrapolation_weight(extrapolation_weight)
    resolved_current = int(current_count)
    score_by_count = _finite_score_by_count(score_report)
    current_score = score_by_count.get(resolved_current)
    updated_histories, direction_report = _updated_histories_and_direction(
        current_count=resolved_current,
        score_report=score_report,
        histories=histories,
        noise_window=resolved_noise_window,
        extrapolation_weight=resolved_weight,
    )

    brake_active = (
        update_brake > 0
        and last_count_change_update >= 0
        and update_number - last_count_change_update < update_brake
    )
    direction_conclusion = str(direction_report["conclusion"])
    evidence = []
    passing = []
    for offset in _candidate_offsets(
        current_count=resolved_current,
        score_report=score_report,
    ):
        candidate_count = resolved_current + offset
        candidate_score = score_by_count.get(candidate_count)
        feasible = current_score is not None and candidate_score is not None
        paired_difference: Optional[float] = None
        median: Optional[float] = None
        mad: Optional[float] = None
        sigma: Optional[float] = None
        standardized: Optional[float] = None
        significant = False
        key = _history_key(resolved_current, offset)
        values = tuple(updated_histories.get(key, ()))
        if feasible:
            paired_difference = float(candidate_score - current_score)
            median, mad, sigma = _robust_scale(values)
            ready = len(values) >= resolved_noise_window
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
                (offset < 0 and direction_conclusion == "L")
                or (offset > 0 and direction_conclusion == "R")
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

    selected_count = resolved_current
    decision_histories = updated_histories
    if passing:
        _, winning_offset, _ = max(
            passing,
            key=lambda item: (item[0], -item[2]),
        )
        committed_offset = max(
            -resolved_max_step,
            min(resolved_max_step, int(winning_offset)),
        )
        selected_count = resolved_current + committed_offset
        decision_histories = {}

    return _controller.PlasticDepthRobustCountDecision(
        selected_count=selected_count,
        current_count=resolved_current,
        update_number=int(update_number),
        brake_active=brake_active,
        last_count_change_update=int(last_count_change_update),
        histories=decision_histories,
        evidence=tuple(evidence),
    )


_lookahead.choose_plastic_depth_count_with_exact_radius = (
    choose_plastic_depth_count_with_directional_coherence
)
_controller.choose_plastic_depth_count_with_mad = (
    choose_plastic_depth_count_with_directional_coherence
)
_trainer_step.choose_plastic_depth_count_with_mad = (
    choose_plastic_depth_count_with_directional_coherence
)
# ^^^ THOG


# vvv THOG capture one per-probe directional snapshot and preserve its history through warmup holds
_ORIGINAL_INLINE_PROBE_REQUEST = _trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request
_ORIGINAL_COMMIT_INLINE_UPDATE = _trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update


def _plastic_depth_inline_probe_request_with_directional_snapshot(
    self: Any,
    targets: Any,
    context: Dict[str, Any],
):
    request = _ORIGINAL_INLINE_PROBE_REQUEST(self, targets, context)
    original_selector = request.selector

    def selector(candidates: Tuple[Tuple[int, Any], ...]) -> int:
        selected = int(original_selector(candidates))
        score_report = context.get("score_report")
        decision = context.get("decision")
        if score_report is None or decision is None:
            return selected
        weight = _validate_extrapolation_weight(
            getattr(
                self.config,
                "plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence",
                _DEFAULT_EXTRAPOLATION_WEIGHT,
            )
        )
        updated_histories, report = _updated_histories_and_direction(
            current_count=int(context["current_count"]),
            score_report=score_report,
            histories=self.state.plastic_depth_probe_histories,
            noise_window=int(
                self.config.plastic__layer_count_probe__window_size_as_number_of_probes
            ),
            extrapolation_weight=weight,
        )
        report["update_number"] = int(decision.update_number)
        report["current_count"] = int(context["current_count"])
        report["selected_count"] = int(decision.selected_count)
        report["extrapolation_weight"] = weight
        context["plastic_directional_report"] = report
        if int(decision.selected_count) == int(context["current_count"]):
            direction_key = _direction_history_key(int(context["current_count"]))
            if direction_key not in decision.histories:
                decision = replace(decision, histories=updated_histories)
                context["decision"] = decision
                context["score_evidence"] = decision.report()
        return int(context.get("selected_count", selected))

    return replace(request, selector=selector)


def _commit_plastic_depth_inline_update_with_directional_snapshot(
    self: Any,
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    transition = _ORIGINAL_COMMIT_INLINE_UPDATE(self, context)
    if context is None:
        return transition
    report = context.get("plastic_directional_report")
    if report is not None:
        self._record("plastic_depth_directional_decision", **dict(report))
    return transition


_trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request = (
    _plastic_depth_inline_probe_request_with_directional_snapshot
)
_trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update = (
    _commit_plastic_depth_inline_update_with_directional_snapshot
)
# ^^^ THOG


# vvv THOG add the compact L/R/A summary and optional DEBUG>9 win counts only on the actual probe row
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line


def _latest_directional_report(trainer: Any) -> Optional[Dict[str, Any]]:
    for event in reversed(getattr(trainer, "events", ())):
        if event.name != "plastic_depth_directional_decision":
            continue
        return dict(event.payload)
    return None


def _prepare_console_progress_payload_with_directional_summary(
    self: Any,
    event: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    if event not in {"optimizer_progress", "evaluation_completed"}:
        return values
    report = _latest_directional_report(self)
    if report is None:
        return values
    try:
        completed_updates = int(values.get("completed_updates", payload.get("completed_updates")))
    except (TypeError, ValueError):
        return values
    if completed_updates != int(report["update_number"]):
        return values
    values["plastic_lra_summary"] = (
        int(report["left_votes"]),
        int(report["right_votes"]),
        int(report["ambiguous_votes"]),
        int(report["vote_total"]),
        str(report["conclusion"]),
    )
    values["plastic_lra_left_wins"] = tuple(int(value) for value in report["left_win_counts"])
    values["plastic_lra_right_wins"] = tuple(int(value) for value in report["right_win_counts"])
    return values


def _format_lra_summary(value: Sequence[Any]) -> str:
    left, right, ambiguous, total, conclusion = value
    return f"L/R/A=[{int(left)}/{int(right)}/{int(ambiguous)}]/{int(total)}=>{conclusion}"


def _format_win_counts(left: Sequence[Any], right: Sequence[Any], denominator: int) -> str:
    left_text = ",".join(str(int(value)) for value in left)
    right_text = ",".join(str(int(value)) for value in right)
    return f"wins L[{left_text}]/{denominator}; R[{right_text}]/{denominator}"


def _move_sampled_after_layers(line: str) -> str:
    sampled = _SAMPLED_ARRAY.search(line)
    if sampled is None:
        return line
    sampled_text = sampled.group(0)
    line_without_sampled = line[: sampled.start()].rstrip(" \t") + line[sampled.end() :]
    layer = _LAYER_FIELD.search(line_without_sampled)
    if layer is None:
        return line
    return (
        line_without_sampled[: layer.end()]
        + "\t"
        + sampled_text
        + line_without_sampled[layer.end() :]
    )


# vvv THOG retain one real tab before sampled while forcing T/V sampled to the same visible terminal column
def _align_sampled_to_minimum_tab_column(line: str) -> str:
    sampled = _SAMPLED_ARRAY.search(line)
    if sampled is None:
        return line
    prefix = line[: sampled.start()].rstrip(" \t")
    suffix = line[sampled.start() :]
    prefix += "\t"
    while len(_ANSI_ESCAPE.sub("", prefix).expandtabs(8)) < _MIN_FINAL_SAMPLED_COLUMN:
        prefix += "\t"
    return prefix + suffix
# ^^^ THOG


def _highlight_changed_sampled_values(run_id: str, event: str, line: str) -> str:
    if event != "optimizer_progress":
        return line
    sampled = _SAMPLED_ARRAY.search(line)
    if sampled is None:
        return line
    current = tuple(item.strip() for item in sampled.group("body").split(",") if item.strip())
    previous = _SAMPLED_BY_RUN_ID.get(run_id)
    _SAMPLED_BY_RUN_ID[run_id] = current
    if previous is None or len(previous) != len(current):
        return line
    rendered = tuple(
        (
            f"{_constants.PINK}{value}{_constants.R}"
            if value != previous[index]
            else value
        )
        for index, value in enumerate(current)
    )
    replacement = f"sampled = [{', '.join(rendered)}]"
    return line[: sampled.start()] + replacement + line[sampled.end() :]


def _bold_current_probe_loss(line: str, offsets: Optional[Sequence[Any]]) -> str:
    if offsets is None:
        return line
    resolved_offsets = tuple(int(value) for value in offsets)
    if 0 not in resolved_offsets:
        return line
    target_index = resolved_offsets.index(0)

    def replace_vector(match: re.Match[str]) -> str:
        items = [item.strip() for item in match.group("body").split(",")]
        if target_index >= len(items):
            return match.group(0)
        items[target_index] = f"{_constants.BOLD_WHITE}{items[target_index]}{_constants.R}"
        return f"{match.group('prefix')}{', '.join(items)}{match.group('close')}"

    return _PROBE_VECTOR.sub(replace_vector, line, count=1)


def _format_progress_line_with_directional_summary(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    local_payload = dict(payload)
    lra_summary = local_payload.pop("plastic_lra_summary", None)
    left_wins = tuple(local_payload.pop("plastic_lra_left_wins", ()))
    right_wins = tuple(local_payload.pop("plastic_lra_right_wins", ()))
    offsets = local_payload.get("plastic_probe_offsets")
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, local_payload)
    line = line.replace("loss  =", "loss=")
    line = re.sub(r"grad norm=\s+", "grad norm= ", line)
    line = _move_sampled_after_layers(line)
    line = _align_sampled_to_minimum_tab_column(line)
    line = _highlight_changed_sampled_values(run_id, event, line)
    line = _bold_current_probe_loss(line, offsets)
    if lra_summary is not None:
        summary_text = _format_lra_summary(lra_summary)
        if int(getattr(_constants, "DEBUG", 0)) > 9:
            denominator = int(lra_summary[3])
            line = f"{line}  {_format_win_counts(left_wins, right_wins, denominator)}  {summary_text}"
        else:
            line = f"{line}  {summary_text}"
    return _console_minor._align_final_progress_line(run_id, event, line)


_stage6.Stage6Trainer._prepare_console_progress_payload = (
    _prepare_console_progress_payload_with_directional_summary
)
_stage6.format_progress_line = _format_progress_line_with_directional_summary
# ^^^ THOG


__all__ = [
    "_DEFAULT_EXTRAPOLATION_WEIGHT",
    "_directional_support",
    "_updated_histories_and_direction",
    "choose_plastic_depth_count_with_directional_coherence",
]
# ^^^ THOG
