# vvv THOG
"""PLASTIC v0.541 public-control, probe-provenance and final console refinement."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Dict, Optional, Sequence

from . import plastic_depth_console_minor_patch as _console_minor
from . import plastic_depth_directional_coherence_patch as _directional
from . import plastic_depth_probe_se_v0521_patch as _probe_se
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step


_ORIGINAL_BEGIN_INLINE_UPDATE = _trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update
_ORIGINAL_INLINE_PROBE_REQUEST = _probe_se._ORIGINAL_INLINE_PROBE_REQUEST
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line
_POSTFIX_START = re.compile(r"(?P<postfix>[ \t]+(?:\x1b\[[0-9;]*m)*<<<)")


def _begin_plastic_depth_inline_update_v0541(self: Any) -> Optional[Dict[str, Any]]:
    return _ORIGINAL_BEGIN_INLINE_UPDATE(self)


def _advance_probe_provenance(
    previous: Sequence[int],
    *,
    probe_sequence: int,
    vote_total: int,
) -> tuple[int, ...]:
    needed_prior = max(0, int(vote_total) - 1)
    prior = tuple(int(value) for value in previous)
    if len(prior) != needed_prior:
        prior = prior[-needed_prior:] if needed_prior else ()
    return (*prior, int(probe_sequence))


def _probe_provenance_from_sequence(
    *,
    probe_sequence: int,
    vote_total: int,
) -> tuple[int, ...]:
    resolved_sequence = int(probe_sequence)
    resolved_total = max(1, int(vote_total))
    first = resolved_sequence - resolved_total + 1
    if first < 1:
        # vvv THOG never fabricate provenance for inconsistent injected/legacy evidence; normal persistent PLASTIC state cannot take this path
        return ()
        # ^^^ THOG
    return tuple(range(first, resolved_sequence + 1))


def _plastic_depth_inline_probe_request_v0541(
    self: Any,
    targets: Any,
    context: Dict[str, Any],
):
    lattice = self._plastic_depth_lattice()
    if lattice is None:
        raise RuntimeError("PLASTIC FINE probe provenance lacks its sampling lattice")
    # count_decision_number is already PLASTIC-only, persistent and incremented exactly once
    # after each successful FINE probe decision; the in-flight probe is therefore +1.
    probe_sequence = int(lattice.count_decision_number.item()) + 1
    context["plastic_probe_sequence"] = probe_sequence
    request = _ORIGINAL_INLINE_PROBE_REQUEST(self, targets, context)
    original_selector = request.selector

    def selector(candidates: Any) -> int:
        selected = int(original_selector(candidates))
        report = context.get("plastic_directional_report")
        decision = context.get("decision")
        if report is None or decision is None:
            context["plastic_probe_provenance"] = (probe_sequence,)
            return selected
        provenance = _probe_provenance_from_sequence(
            probe_sequence=probe_sequence,
            vote_total=int(report.get("vote_total", 1)),
        )
        report["probe_sequence"] = probe_sequence
        report["probe_provenance"] = provenance
        context["plastic_probe_provenance"] = provenance
        return selected

    return replace(request, selector=selector)


_trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update = _begin_plastic_depth_inline_update_v0541
_probe_se._ORIGINAL_INLINE_PROBE_REQUEST = _plastic_depth_inline_probe_request_v0541
_trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request = _probe_se._inline_probe_request_with_paired_token_se


def _prepare_console_progress_payload_v0541(
    self: Any,
    event: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    if event not in {"optimizer_progress", "evaluation_completed"}:
        return values
    report = _directional._latest_directional_report(self)
    if report is None or "probe_sequence" not in report:
        return values
    try:
        completed_updates = int(values.get("completed_updates", payload.get("completed_updates")))
    except (TypeError, ValueError):
        return values
    if completed_updates != int(report.get("update_number", -1)):
        return values
    values["plastic_probe_sequence"] = int(report["probe_sequence"])
    values["plastic_probe_provenance"] = tuple(int(value) for value in report.get("probe_provenance", ()))
    return values


_stage6.Stage6Trainer._prepare_console_progress_payload = _prepare_console_progress_payload_v0541


def _provenance_text(values: Sequence[Any]) -> str:
    resolved = tuple(int(value) for value in values)
    if not resolved:
        return ""
    return " (P" + ",".join(str(value) for value in resolved) + ")"


def _finalize_console_v0541(
    line: str,
    *,
    probe_sequence: Optional[int],
    probe_provenance: Sequence[Any],
) -> str:
    line = re.sub(
        r"(probe_Δloss \[[^\]]*?) \.\.\. ([^\]]*?\])",
        r"\1 .. \2",
        line,
        count=1,
    )
    if probe_sequence is not None and "probe_Δloss" in line:
        line = line.replace("probe_Δloss", f"P{int(probe_sequence):4d}  probe_Δloss", 1)

    # ANSI has no portable font-size control. ⇩/⇧ are larger text glyphs than ↓/↑
    # without introducing emoji-width instability. Existing ANSI colour spans are preserved.
    line = line.replace("↓", "⇩").replace("↑", "⇧")

    provenance = _provenance_text(probe_provenance)
    if provenance and "⇩|⇧|? =" in line:
        summary_start = line.find("⇩|⇧|? =")
        postfix = _POSTFIX_START.search(line, summary_start)
        if postfix is None:
            line = f"{line}{provenance}"
        else:
            line = f"{line[:postfix.start()]}{provenance}{line[postfix.start():]}"
    return line


def _format_progress_line_v0541(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    local_payload = dict(payload)
    probe_sequence = local_payload.pop("plastic_probe_sequence", None)
    probe_provenance = tuple(local_payload.pop("plastic_probe_provenance", ()))
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, local_payload)
    line = _finalize_console_v0541(
        line,
        probe_sequence=None if probe_sequence is None else int(probe_sequence),
        probe_provenance=probe_provenance,
    )
    if event == "optimizer_progress":
        _console_minor._record_alignment(run_id, line)
    elif event == "evaluation_completed":
        line = _console_minor._align_validation_row(run_id, line)
    return line


_stage6.format_progress_line = _format_progress_line_v0541


__all__ = [
    "_advance_probe_provenance",
    "_finalize_console_v0541",
    "_probe_provenance_from_sequence",
    "_provenance_text",
]
# ^^^ THOG
