# vvv THOG
"""Final human-facing PLASTIC progress-row postfix cleanup."""

from __future__ import annotations

import re
from typing import Any, Dict

from . import stage6_trainer as _stage6


_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line
_POSTFIX_ANNOTATION = re.compile(
    r"(?P<spacing>[ \t]*)(?P<colour>\x1b\[[0-9;]*m)?"
    r"(?P<text><<< (?:update brake on|warmup braked? enabled|stopped by memory limit))"
    r"(?P<reset>\x1b\[[0-9;]*m)?"
)
_NEUTRAL_LRA = re.compile(r"(?P<prefix>L/R/A=\[[^\]]+\]/\d+)=>-(?=$|[ \t])")


def _finalize_plastic_postfixes(line: str) -> str:
    rendered = _NEUTRAL_LRA.sub(r"\g<prefix>=>stet", line)
    annotations = []

    def collect_annotation(match: re.Match[str]) -> str:
        text = match.group("text").replace("warmup braked enabled", "warmup brake enabled")
        annotations.append(
            f"{match.group('colour') or ''}{text}{match.group('reset') or ''}"
        )
        return ""

    rendered = _POSTFIX_ANNOTATION.sub(collect_annotation, rendered).rstrip(" \t")
    if annotations:
        rendered = f"{rendered}  {'  '.join(annotations)}"
    return rendered


def _format_progress_line_with_final_postfixes(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    return _finalize_plastic_postfixes(
        _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    )


_stage6.format_progress_line = _format_progress_line_with_final_postfixes


__all__ = ["_finalize_plastic_postfixes"]
# ^^^ THOG

# vvv THOG install the v0.531 equivalent-time wall-time objective and final operator-console overlay after every earlier PLASTIC formatter
from . import plastic_depth_wall_time_equivalent_time_gain_patch as _plastic_depth_wall_time_equivalent_time_gain_patch
# ^^^ THOG

# vvv THOG restore pre-v0.531 selector patchability, paired-SE ownership and final visible-column alignment after the new overlay
from . import plastic_depth_wall_time_v0531_integration_patch as _plastic_depth_wall_time_v0531_integration_patch
# ^^^ THOG

# vvv THOG install v0.541 canonical controls, durable probe provenance and final console glyph refinement
from . import plastic_depth_v0541_patch as _plastic_depth_v0541_patch
# ^^^ THOG

# vvv THOG install v0.53 fixed-batch non-overlapping FINE probe windows after every later selector/provenance overlay
from . import plastic_depth_same_batch_all_probes_patch as _plastic_depth_same_batch_all_probes_patch
# ^^^ THOG

# vvv THOG let lifecycle resume reconstruct TrainingConfig from the v0.53 persisted same-batch field without changing ordinary constructor semantics
from . import plastic_depth_same_batch_resume_config_patch as _plastic_depth_same_batch_resume_config_patch
# ^^^ THOG

# vvv THOG make same-batch mode explicit in startup/probe console and reset visible P provenance for each fresh evidence batch
from . import plastic_depth_same_batch_visibility_patch as _plastic_depth_same_batch_visibility_patch
_plastic_depth_same_batch_visibility_patch._PROBE_COLUMN = 300                                                                                             # <<< THOG anchor PLASTIC probe block at terminal column 300
_ORIGINAL_ALIGN_PROBE_SECTION = _plastic_depth_same_batch_visibility_patch._align_probe_section


def _align_probe_section_with_probe_priority(line: str) -> str:
    match = _plastic_depth_same_batch_visibility_patch._P_SECTION.search(line)
    if match is None:
        return line
    probe = line[match.start() :]
    probe = _plastic_depth_same_batch_visibility_patch._P_SECTION.sub(
        lambda value: f"P{value.group('number')}  probe_Δloss",
        probe,
        count=1,
    )
    prefix = line[: match.start()].rstrip(" \t")
    prefix_width = _plastic_depth_same_batch_visibility_patch._PROBE_COLUMN - 3
    prefix = _plastic_depth_same_batch_visibility_patch._truncate_visible(prefix, prefix_width)
    visible = _plastic_depth_same_batch_visibility_patch._visible_length(prefix)
    if visible < prefix_width:
        prefix = f"{prefix}{' ' * (prefix_width - visible)}"
    return f"{prefix}  {probe}"


_plastic_depth_same_batch_visibility_patch._align_probe_section = _align_probe_section_with_probe_priority
# ^^^ THOG

# vvv THOG install continuous DEPTH scalar-weight charts, DEBUG-gated legacy coefficient charts, darker RHS gains and fixed-run observational probes
from . import depth_weight_curves_and_observational_probes_patch as _depth_weight_curves_and_observational_probes_patch
# ^^^ THOG
