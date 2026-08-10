# vvv THOG
"""Expose the v0.55 growth-side discount in the existing PLASTIC startup header."""

from __future__ import annotations

from . import plastic_depth_console_compact_layout_patch as _compact
from . import plastic_depth_sen_kendall_v055_patch as _v055
from . import plastic_depth_v055_growth_side_discount_patch as _growth
from . import plastic_depth_v055_growth_side_help_patch as _growth_help


_LABEL = "plastic__layer_count_decision_algorithm__growth_side_discount:"
_ORIGINAL_GRADIENT_HEADER_ROWS = _compact._gradient_header_rows


def _gradient_header_rows_with_growth_side_discount() -> tuple[tuple[str, str], ...]:
    rows = tuple(_ORIGINAL_GRADIENT_HEADER_ROWS())
    if _v055._runtime_algorithm() not in _v055.SEN_KENDALL_ALGORITHMS:
        return rows
    return (
        *rows,
        (_LABEL, f"{_growth._runtime_growth_side_discount():g}"),
    )


_compact._gradient_header_rows = _gradient_header_rows_with_growth_side_discount


__all__ = ["_gradient_header_rows_with_growth_side_discount"]
# ^^^ THOG
