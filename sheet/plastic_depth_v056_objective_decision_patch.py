# vvv THOG
"""PLASTIC v0.56 objective-neutral Sen/Kendall and strict decision-algorithm ownership."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from . import plastic_depth_directional_coherence_patch as _directional
from . import plastic_depth_lookahead_patch as _lookahead
from . import plastic_depth_sen_kendall_v055_patch as _v055
from . import plastic_depth_sen_kendall_v055_runtime_fix_patch as _v055_runtime
from . import plastic_depth_theil_sen_kendall_patch as _tsk
from . import plastic_depth_v055_growth_side_discount_patch as _growth
from . import plastic_depth_wall_time_equivalent_time_gain_patch as _wall_time
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step
from . import training_config as _training_config


LRA_ALGORITHM = "theil_sen_kendall_LRA"
STRATIFIED_ALGORITHM = "sen_kendall__tau__stratified"
SEN_KENDALL_ALGORITHMS = (LRA_ALGORITHM, STRATIFIED_ALGORITHM)
DECISION_ALGORITHMS = (_tsk.LEGACY_DIRECTIONAL_ALGORITHM, *SEN_KENDALL_ALGORITHMS)
_RETIRED_ALGORITHMS = {
    "wall_time__gradient__theil_sen_kendall_slope_tau",
    "wall_time__theil_sen_kendall_LRA",
    "wall_time__sen_kendall__tau__stratified",
}
_OLD_GRADIENT_SENTINEL = "__retired_v054_gradient_internal__"


def _runtime_algorithm() -> str:
    value = os.environ.get(_tsk._ALGORITHM_ENV, _tsk.LEGACY_DIRECTIONAL_ALGORITHM).strip()
    if value in _RETIRED_ALGORITHMS:
        replacements = (
            f"{LRA_ALGORITHM} or {STRATIFIED_ALGORITHM}"
            if value != "wall_time__gradient__theil_sen_kendall_slope_tau"
            else LRA_ALGORITHM
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
    if resolved in _RETIRED_ALGORITHMS:
        _runtime_before = os.environ.get(_tsk._ALGORITHM_ENV)
        try:
            os.environ[_tsk._ALGORITHM_ENV] = resolved
            _runtime_algorithm()
        finally:
            if _runtime_before is None:
                os.environ.pop(_tsk._ALGORITHM_ENV, None)
            else:
                os.environ[_tsk._ALGORITHM_ENV] = _runtime_before
    if resolved not in DECISION_ALGORITHMS:
        raise ValueError(
            f"{_tsk._ALGORITHM_KEY} must be one of {DECISION_ALGORITHMS}; got {value!r}"
        )
    os.environ[_tsk._ALGORITHM_ENV] = resolved


# vvv THOG rename TSK modes without coupling either name to the selected objective
_v055.LRA_ALGORITHM = LRA_ALGORITHM
_v055.STRATIFIED_ALGORITHM = STRATIFIED_ALGORITHM
_v055.SEN_KENDALL_ALGORITHMS = SEN_KENDALL_ALGORITHMS
_v055.DECISION_ALGORITHMS = DECISION_ALGORITHMS
_v055._runtime_algorithm = _runtime_algorithm
_v055._set_runtime_algorithm = _set_runtime_algorithm
_tsk.GRADIENT_ALGORITHM = LRA_ALGORITHM
_tsk.DECISION_ALGORITHMS = DECISION_ALGORITHMS
_tsk._runtime_algorithm = _runtime_algorithm
_tsk._set_runtime_algorithm = _set_runtime_algorithm
# ^^^ THOG


def _validate_v056_config(config: Any) -> None:
    algorithm = _runtime_algorithm()
    if algorithm not in SEN_KENDALL_ALGORITHMS:
        return
    if not bool(getattr(config, "plastic__enabled", False)) or not bool(
        getattr(config, "plastic__do_learn_layer_count", False)
    ):
        raise ValueError(f"{algorithm} requires learned-count PLASTIC DEPTH")
    if int(getattr(config, "plastic__layer_count__max_allowable_layer_change", 1)) != 1:
        raise ValueError(
            f"{algorithm} requires plastic__layer_count__max_allowable_layer_change=1"
        )


# vvv THOG remove the obsolete v0.55 wall-time-only validation while preserving the older layered config checks
_v055._validate_v055_config = _validate_v056_config
_ORIGINAL_TRAINING_POST_INIT = _training_config.TrainingConfig.__post_init__


def _training_post_init_v056(self: Any) -> None:
    # The retained v0.54 post-init compares the runtime name against _tsk.GRADIENT_ALGORITHM
    # and would otherwise re-impose relative_training_wall_time for the renamed LRA mode.
    previous_gradient_name = _tsk.GRADIENT_ALGORITHM
    _tsk.GRADIENT_ALGORITHM = _OLD_GRADIENT_SENTINEL
    try:
        _ORIGINAL_TRAINING_POST_INIT(self)
    finally:
        _tsk.GRADIENT_ALGORITHM = previous_gradient_name
    _validate_v056_config(self)


_training_config.TrainingConfig.__post_init__ = _training_post_init_v056
# ^^^ THOG


def objective_score_map(score_report: Sequence[Mapping[str, object]]) -> Dict[int, float]:
    """Return the selected objective's canonical finite lower-is-better score by feasible count."""

    result: Dict[int, float] = {}
    for item in score_report:
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


# vvv THOG TSK consumes the already-selected objective score; only relative wall time happens to be measured in equivalent seconds
_tsk._equivalent_time_scores = objective_score_map
# ^^^ THOG


def _active_objective(trainer: Any = None) -> Optional[str]:
    resolved_trainer = trainer if trainer is not None else _wall_time._ACTIVE_TRAINER.get()
    config = getattr(resolved_trainer, "config", None)
    value = getattr(config, "plastic__layer_count_objective", None)
    return None if value is None else str(value)


def _score_units(objective: Optional[str]) -> str:
    if objective == "relative_training_wall_time":
        return "equivalent_seconds"
    if objective == "lowest_loss" or objective == "memory_budget":
        return "loss"
    if objective == "layer_efficiency":
        return "layer_efficiency_score"
    return "objective_score"


# vvv THOG add objective-neutral audit names while retaining v0.55 field aliases for compatibility with established console/audit code
_ORIGINAL_STORE_REPORT = _v055._store_report


def _store_report_v056(trainer: Any, report: Dict[str, Any]) -> None:
    updated = dict(report)
    objective = _active_objective(trainer)
    updated["objective"] = objective
    updated["score_units"] = _score_units(objective)
    if "sen_slope_score_per_layer" not in updated:
        updated["sen_slope_score_per_layer"] = updated.get(
            "sen_slope_seconds_per_layer",
            updated.get("theil_sen_slope_seconds_per_layer"),
        )
    if "adjacent_score_delta" not in updated:
        updated["adjacent_score_delta"] = updated.get("adjacent_score_seconds")
    if "left_adjacent_score_delta" not in updated:
        updated["left_adjacent_score_delta"] = updated.get("left_adjacent_score_seconds")
    if "right_adjacent_score_delta" not in updated:
        updated["right_adjacent_score_delta"] = updated.get("right_adjacent_score_seconds")
    _ORIGINAL_STORE_REPORT(trainer, updated)


_v055._store_report = _store_report_v056
# ^^^ THOG


def _copy_decision_histories(decision: Any, fallback: Mapping[str, Sequence[float]]) -> Dict[str, Tuple[float, ...]]:
    source = fallback if decision is None else getattr(decision, "histories", fallback)
    return {
        str(key): tuple(float(value) for value in values)
        for key, values in source.items()
    }


def _neutral_legacy_report(current_count: int, decision: Any, report: Any) -> Dict[str, Any]:
    if isinstance(report, Mapping):
        return _v055_runtime._legacy_compatible_v055_report(report)
    selected = int(getattr(decision, "selected_count", current_count)) if decision is not None else int(current_count)
    return {
        "algorithm": _runtime_algorithm(),
        "left_votes": 0,
        "right_votes": 0,
        "ambiguous_votes": 0,
        "vote_total": 0,
        "conclusion": "-",
        "direction_conclusion": "-",
        "left_win_counts": (),
        "right_win_counts": (),
        "left_offsets": (),
        "right_offsets": (),
        "selected_count": selected,
    }


# vvv THOG under TSK the retained directional snapshot shim may carry compatibility state but must never execute legacy direction/z calculations
_ORIGINAL_UPDATED_HISTORIES_AND_DIRECTION = _directional._updated_histories_and_direction


def _updated_histories_and_direction_v056(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    histories: Mapping[str, Sequence[float]],
    noise_window: int,
    extrapolation_weight: float,
):
    if _runtime_algorithm() not in SEN_KENDALL_ALGORITHMS:
        return _ORIGINAL_UPDATED_HISTORIES_AND_DIRECTION(
            current_count=current_count,
            score_report=score_report,
            histories=histories,
            noise_window=noise_window,
            extrapolation_weight=extrapolation_weight,
        )
    trainer = _wall_time._ACTIVE_TRAINER.get()
    context = getattr(trainer, "_plastic_depth_inline_update_context", None) if trainer is not None else None
    decision = context.get("decision") if isinstance(context, dict) else None
    report = context.get("plastic_v055_sen_kendall_report") if isinstance(context, dict) else None
    return (
        _copy_decision_histories(decision, histories),
        _neutral_legacy_report(int(current_count), decision, report),
    )


_directional._updated_histories_and_direction = _updated_histories_and_direction_v056
# ^^^ THOG


# vvv THOG remove the compatibility directional snapshot before commit so TSK emits no current legacy directional-decision event
_ORIGINAL_INLINE_PROBE_REQUEST = _trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request


def _plastic_depth_inline_probe_request_v056(self: Any, targets: Any, context: Dict[str, Any]):
    request = _ORIGINAL_INLINE_PROBE_REQUEST(self, targets, context)
    if _runtime_algorithm() not in SEN_KENDALL_ALGORITHMS:
        return request
    original_selector = request.selector

    def selector(candidates: Any) -> int:
        selected = int(original_selector(candidates))
        context.pop("plastic_directional_report", None)
        return selected

    return replace(request, selector=selector)


_trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request = _plastic_depth_inline_probe_request_v056
# ^^^ THOG


# vvv THOG reconstruct raw probe diagnostics without ever traversing standardized_improvement under TSK, including on later non-probe rows
_ORIGINAL_LATEST_PROBE_REPORT = _lookahead._latest_probe_report


def _latest_probe_report_v056(trainer: Any) -> Optional[Dict[str, Any]]:
    if _runtime_algorithm() not in SEN_KENDALL_ALGORITHMS:
        return _ORIGINAL_LATEST_PROBE_REPORT(trainer)
    if not bool(getattr(getattr(trainer, "config", None), "plastic__do_learn_layer_count", False)):
        return None
    for event in reversed(getattr(trainer, "events", ())):
        if event.name != "plastic_depth_count_decision":
            continue
        payload = event.payload
        if "previous_active_layers" not in payload:
            return None
        current_count = int(payload["previous_active_layers"])
        counts = tuple(int(value) for value in payload.get("decision_candidate_counts", ()))
        if not counts:
            counts = tuple(int(item["active_layers"]) for item in payload.get("candidates", ()))
        losses_by_count: Dict[int, Optional[float]] = {}
        for item in payload.get("candidates", ()):
            try:
                count = int(item["active_layers"])
            except (KeyError, TypeError, ValueError):
                continue
            loss = item.get("validation_loss")
            losses_by_count[count] = None if loss is None else float(loss)
        current_loss = losses_by_count.get(current_count)
        offsets = tuple(count - current_count for count in counts)
        losses = tuple(losses_by_count.get(count) for count in counts)
        edge_offsets = tuple(offset for offset in offsets if offset != 0)
        loss_gain = tuple(
            None
            if current_loss is None or losses_by_count.get(current_count + offset) is None
            else float(current_loss - losses_by_count[current_count + offset])
            for offset in edge_offsets
        )
        return {
            "offsets": offsets,
            "edge_offsets": edge_offsets,
            "losses": losses,
            "loss_gain": loss_gain,
            "score_z": None,
        }
    return None


_lookahead._latest_probe_report = _latest_probe_report_v056
# ^^^ THOG


# vvv THOG final defensive payload ownership: TSK never exposes legacy z fields even if an old compatibility layer supplied stale data
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload


def _prepare_console_progress_payload_v056(self: Any, event: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    if _runtime_algorithm() in SEN_KENDALL_ALGORITHMS:
        for key in (
            "plastic_score_z",
            "plastic_change_z",
            "score_z",
            "change_z",
        ):
            values.pop(key, None)
    return values


_stage6.Stage6Trainer._prepare_console_progress_payload = _prepare_console_progress_payload_v056
# ^^^ THOG


__all__ = [
    "DECISION_ALGORITHMS",
    "LRA_ALGORITHM",
    "SEN_KENDALL_ALGORITHMS",
    "STRATIFIED_ALGORITHM",
    "objective_score_map",
]
# ^^^ THOG
