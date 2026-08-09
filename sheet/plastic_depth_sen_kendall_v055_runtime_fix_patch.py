# vvv THOG
"""Final PLASTIC v0.55 Sen/Kendall runtime ownership and operator-console fixes."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Sequence, Tuple

from . import plastic_depth_directional_coherence_patch as _directional
from . import plastic_depth_sen_kendall_v055_patch as _v055
from . import stage6_trainer as _stage6


# vvv THOG Sen/Kendall owns its own window histories; the retained pre-v0.55 directional snapshot must never replace them after a STAY
_ORIGINAL_UPDATED_HISTORIES_AND_DIRECTION = _directional._updated_histories_and_direction


def _decision_histories(decision: Any) -> Dict[str, Tuple[float, ...]]:
    return {
        str(key): tuple(float(value) for value in values)
        for key, values in decision.histories.items()
    }


def _legacy_compatible_v055_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    guarded = dict(report)
    guarded.setdefault("left_votes", 0)
    guarded.setdefault("right_votes", 0)
    guarded.setdefault("ambiguous_votes", 0)
    guarded.setdefault("vote_total", int(guarded.get("strata_count", 0)))
    guarded.setdefault("conclusion", guarded.get("direction_conclusion", "-"))
    guarded.setdefault("left_win_counts", ())
    guarded.setdefault("right_win_counts", ())
    guarded.setdefault("left_offsets", ())
    guarded.setdefault("right_offsets", ())
    return guarded


def _updated_histories_and_direction_without_legacy_sen_kendall_ownership(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    histories: Mapping[str, Sequence[float]],
    noise_window: int,
    extrapolation_weight: float,
):
    if _v055._runtime_algorithm() not in _v055.SEN_KENDALL_ALGORITHMS:
        return _ORIGINAL_UPDATED_HISTORIES_AND_DIRECTION(
            current_count=current_count,
            score_report=score_report,
            histories=histories,
            noise_window=noise_window,
            extrapolation_weight=extrapolation_weight,
        )
    trainer = _v055._wall_time._ACTIVE_TRAINER.get()
    context = getattr(trainer, "_plastic_depth_inline_update_context", None) if trainer is not None else None
    decision = context.get("decision") if isinstance(context, dict) else None
    report = context.get("plastic_v055_sen_kendall_report") if isinstance(context, dict) else None
    if decision is None or not isinstance(report, Mapping):
        return _ORIGINAL_UPDATED_HISTORIES_AND_DIRECTION(
            current_count=current_count,
            score_report=score_report,
            histories=histories,
            noise_window=noise_window,
            extrapolation_weight=extrapolation_weight,
        )
    return _decision_histories(decision), _legacy_compatible_v055_report(report)


_directional._updated_histories_and_direction = (
    _updated_histories_and_direction_without_legacy_sen_kendall_ownership
)
# ^^^ THOG


# vvv THOG same-batch console provenance is window-local for v0.55 diagnostics even though durable probe sequence numbers remain global internally
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload


def _prepare_console_progress_payload_with_window_local_v055_provenance(
    self: Any,
    event: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    if _v055._runtime_algorithm() not in _v055.SEN_KENDALL_ALGORITHMS:
        return values
    algorithm = values.get("plastic_v055_algorithm")
    if algorithm not in _v055.SEN_KENDALL_ALGORITHMS:
        return values
    provenance = tuple(values.get("plastic_probe_provenance", ()) or ())
    if provenance:
        values["plastic_v055_probe_ids"] = tuple(int(value) for value in provenance)
    return values


_stage6.Stage6Trainer._prepare_console_progress_payload = (
    _prepare_console_progress_payload_with_window_local_v055_provenance
)
# ^^^ THOG


# vvv THOG align sampled identically and suppress all retired z-score plus bare-direction fields before the authoritative v0.55 result
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line
_LAYERS_SAMPLED = re.compile(r"(?P<layers>layers\s+\d+)[ \t]+sampled[ \t]+")
_LEGACY_BARE_PROBE_OUTCOME = re.compile(
    r"(?P<close>\])=>(?:\x1b\[[0-9;]*m)*(?:▼|▲|⇩|⇧|↓|↑)(?:\x1b\[[0-9;]*m)*"
)
_RETIRED_SCORE_VECTOR = re.compile(
    r"[ \t]+(?:score_z|change_z)\s+\[[^\]]+\]\s*=\s*\[[^\]]*\]"
)


def _align_sampled_field(line: str) -> str:
    return _LAYERS_SAMPLED.sub(
        lambda match: f"{match.group('layers')}  sampled ",
        line,
        count=1,
    )


def _remove_legacy_bare_probe_outcome(line: str) -> str:
    return _LEGACY_BARE_PROBE_OUTCOME.sub(r"\g<close>", line, count=1)


def _remove_retired_score_vectors(line: str) -> str:
    return _RETIRED_SCORE_VECTOR.sub("", line)


def _format_progress_line_with_v055_runtime_fixes(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    if event not in {"optimizer_progress", "evaluation_completed"}:
        return line
    line = _align_sampled_field(line)
    if _v055._runtime_algorithm() in _v055.SEN_KENDALL_ALGORITHMS:
        line = _remove_legacy_bare_probe_outcome(line)
        line = _remove_retired_score_vectors(line)
    return line.rstrip(" \t")


_stage6.format_progress_line = _format_progress_line_with_v055_runtime_fixes


__all__ = [
    "_align_sampled_field",
    "_legacy_compatible_v055_report",
    "_prepare_console_progress_payload_with_window_local_v055_provenance",
    "_remove_legacy_bare_probe_outcome",
    "_remove_retired_score_vectors",
    "_updated_histories_and_direction_without_legacy_sen_kendall_ownership",
]
# ^^^ THOG
