# vvv THOG
"""Restore pre-v0.531 PLASTIC integration surfaces around the equivalent-time wall-time overlay."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, Optional

from . import plastic_depth as _plastic_depth
from . import plastic_depth_console_minor_patch as _console_minor
from . import plastic_depth_lookahead_patch as _lookahead
from . import plastic_depth_probe_se_v0521_patch as _probe_se
from . import plastic_depth_wall_time_equivalent_time_gain_patch as _wall_time
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step


_BASE_CANDIDATE_SELECTOR = _wall_time._ORIGINAL_CHOOSE_PLASTIC_DEPTH_CANDIDATE
_PAIRED_SE_INNER_PROBE_REQUEST = _probe_se._ORIGINAL_INLINE_PROBE_REQUEST
_FINAL_V0531_FORMATTER = _stage6.format_progress_line


def _candidate_selector_with_v0531_wall_time(
    measurements: Iterable[Any],
    *,
    objective: str,
    maximum_layers: int,
    cost_weight: float,
    reference_training_time: Optional[float],
    memory_budget_gib: Optional[float],
):
    resolved_measurements = tuple(measurements)
    trainer = _wall_time._ACTIVE_TRAINER.get()
    if objective == "relative_training_wall_time" and trainer is not None:
        return _wall_time._choose_wall_time_equivalent_time_gain(
            trainer,
            resolved_measurements,
        )
    return _trainer_step.choose_plastic_depth_candidate(
        resolved_measurements,
        objective=objective,
        maximum_layers=maximum_layers,
        cost_weight=cost_weight,
        reference_training_time=reference_training_time,
        memory_budget_gib=memory_budget_gib,
    )


# vvv THOG restore the existing live-selector proxy contract: v0.531 owns only active FINE relative-wall-time scoring
_plastic_depth.choose_plastic_depth_candidate = _BASE_CANDIDATE_SELECTOR
_trainer_step.choose_plastic_depth_candidate = _BASE_CANDIDATE_SELECTOR
_lookahead.choose_plastic_depth_candidate = _candidate_selector_with_v0531_wall_time
# ^^^ THOG


def _paired_se_inner_probe_request_with_v0531_context(
    self: Any,
    targets: Any,
    context: Dict[str, Any],
):
    request = _PAIRED_SE_INNER_PROBE_REQUEST(self, targets, context)
    original_selector = request.selector

    def selector(candidates: Any) -> int:
        token = _wall_time._ACTIVE_TRAINER.set(self)
        try:
            return int(original_selector(candidates))
        finally:
            _wall_time._ACTIVE_TRAINER.reset(token)

    return replace(request, selector=selector)


# vvv THOG keep the v0.521 paired-SE wrapper as the public probe owner while injecting v0.531 context underneath it
_probe_se._ORIGINAL_INLINE_PROBE_REQUEST = _paired_se_inner_probe_request_with_v0531_context
_trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request = (
    _probe_se._inline_probe_request_with_paired_token_se
)
# ^^^ THOG


def _format_progress_line_with_post_v0531_alignment(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    line = _FINAL_V0531_FORMATTER(run_id, event, payload)
    if event == "optimizer_progress":
        _console_minor._record_alignment(run_id, line)
    elif event == "evaluation_completed":
        line = _console_minor._align_validation_row(run_id, line)
    return line


# vvv THOG re-run visible-column alignment after v0.531 changes labels, glyphs and fixed-width score rendering
_stage6.format_progress_line = _format_progress_line_with_post_v0531_alignment
# ^^^ THOG


__all__ = ["_candidate_selector_with_v0531_wall_time"]
# ^^^ THOG
