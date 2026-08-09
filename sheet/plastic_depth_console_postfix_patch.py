# vvv THOG
"""Final human-facing PLASTIC progress-row postfix cleanup."""

from __future__ import annotations

import re
from typing import Any, Dict

from . import stage6_trainer as _stage6


_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line
_POSTFIX_ANNOTATION = re.compile(
    r"(?P<spacing>[ \t]*)(?P<colour>\x1b\[[0-9;]*m)?"
    r"(?P<text><<< (?:update brake on|warmup braked? enabled))"
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
_plastic_depth_same_batch_visibility_patch._PROBE_COLUMN = 300                                                                                             # <<< THOG shift PLASTIC probe block to terminal column 300
# ^^^ THOG
