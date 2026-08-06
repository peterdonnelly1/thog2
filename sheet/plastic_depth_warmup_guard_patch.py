# vvv THOG
"""Final PLASTIC warmup count guard and compact sampled-array console rendering."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import torch

from . import plastic_depth_console_minor_patch as _console_minor
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step
from .plastic_depth_controller import PlasticDepthRobustCountDecision
from .plastic_depth_inline import PlasticDepthInlineProbeRequest


_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line
_ORIGINAL_INLINE_PROBE_REQUEST = (
    _trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request
)
_ORIGINAL_COMMIT_INLINE_UPDATE = (
    _trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update
)
_SAMPLED_ARRAY = re.compile(r"sampled = \[(?P<body>[^\]]*)\]")


# vvv THOG compact every displayed sample index to one decimal place with exactly one separator space
def _compact_sampled_array(line: str) -> str:
    def replace_array(match: re.Match[str]) -> str:
        body = match.group("body")
        rendered = []
        for item in body.split(","):
            text = item.strip()
            if not text:
                continue
            try:
                numeric = float(text)
            except ValueError:
                rendered.append(text)
                continue
            rendered.append(f"{numeric:.1f}" if math.isfinite(numeric) else str(numeric))
        return f"sampled = [{', '.join(rendered)}]"

    return _SAMPLED_ARRAY.sub(replace_array, line)


def _format_progress_line_with_compact_sampled_array(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    line = _compact_sampled_array(line)
    return line.replace("<<< warmup braked enabled", "<<< warmup brake enabled")


_stage6.format_progress_line = _format_progress_line_with_compact_sampled_array
# ^^^ THOG


# vvv THOG a completed row belongs to warmup when that just-finished update started below warmup_updates
def _row_has_warmup_brake_at_actual_schedule_boundary(
    trainer: Any,
    completed_updates: int,
) -> bool:
    config = getattr(trainer, "config", None)
    return (
        bool(getattr(config, "plastic__enabled", False))
        and bool(getattr(config, "plastic__do_learn_layer_count", False))
        and bool(getattr(config, "plastic__freeze_geometry_during_warmup", False))
        and 0 < int(completed_updates) <= int(getattr(config, "warmup_updates", 0))
    )


_console_minor._row_has_warmup_brake = (
    _row_has_warmup_brake_at_actual_schedule_boundary
)
# ^^^ THOG


# vvv THOG enforce the warmup brake in the FINE selector and again at commit as a state-safety backstop
def _count_warmup_brake_active(
    trainer: Any,
    *,
    update_number: Optional[int] = None,
) -> bool:
    config = getattr(trainer, "config", None)
    resolved_update = (
        int(getattr(getattr(trainer, "state", None), "completed_updates", 0)) + 1
        if update_number is None
        else int(update_number)
    )
    return (
        bool(getattr(config, "plastic__enabled", False))
        and bool(getattr(config, "plastic__do_learn_layer_count", False))
        and bool(getattr(config, "plastic__freeze_geometry_during_warmup", False))
        and resolved_update > 0
        and resolved_update <= int(getattr(config, "warmup_updates", 0))
    )


def _warmup_histories(
    trainer: Any,
    decision: PlasticDepthRobustCountDecision,
) -> Dict[str, Tuple[float, ...]]:
    noise_window = int(trainer.config.plastic__layer_count_probe_noise_window)
    histories = {
        str(key): tuple(float(value) for value in values[-noise_window:])
        for key, values in trainer.state.plastic_depth_probe_histories.items()
    }
    for evidence in decision.evidence:
        if not evidence.feasible or evidence.paired_difference is None:
            continue
        key = f"{int(decision.current_count)}:{int(evidence.direction):+d}"
        values = list(histories.get(key, ()))
        values.append(float(evidence.paired_difference))
        histories[key] = tuple(values[-noise_window:])
    return histories


def _apply_count_warmup_brake(
    trainer: Any,
    context: Dict[str, Any],
) -> int:
    current_count = int(context["current_count"])
    decision = context.get("decision")
    if decision is None:
        raise RuntimeError("PLASTIC warmup guard found no count decision")
    if not _count_warmup_brake_active(
        trainer,
        update_number=int(decision.update_number),
    ):
        return int(context["selected_count"])
    if int(decision.selected_count) != current_count:
        decision = replace(
            decision,
            selected_count=current_count,
            histories=_warmup_histories(trainer, decision),
        )
        context["decision"] = decision
        context["paired_evidence"] = decision.report()
    context["selected_count"] = current_count
    context["warmup_brake_active"] = True
    return current_count


def _plastic_depth_inline_probe_request_with_count_warmup_brake(
    self: Any,
    targets: torch.Tensor,
    context: Dict[str, Any],
) -> PlasticDepthInlineProbeRequest:
    request = _ORIGINAL_INLINE_PROBE_REQUEST(self, targets, context)
    if not _count_warmup_brake_active(self):
        return request
    original_selector = request.selector

    def selector(candidates: Tuple[Tuple[int, torch.Tensor], ...]) -> int:
        original_selector(candidates)
        return _apply_count_warmup_brake(self, context)

    return replace(request, selector=selector)


def _commit_plastic_depth_inline_update_with_count_warmup_backstop(
    self: Any,
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if context is not None and _count_warmup_brake_active(self):
        _apply_count_warmup_brake(self, context)
    return _ORIGINAL_COMMIT_INLINE_UPDATE(self, context)


_trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request = (
    _plastic_depth_inline_probe_request_with_count_warmup_brake
)
_trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update = (
    _commit_plastic_depth_inline_update_with_count_warmup_backstop
)
# ^^^ THOG
