# vvv THOG
"""Bound accumulated DEPTH diagnostic tables without splitting normal scalar curves."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from . import depth_weight_curves_and_observational_probes_patch as _depth
from . import plastic_depth_wandb_probe_curves_patch as _probe_wandb


_ORIGINAL_DEPTH_CHART_ROWS = _depth._depth_chart_rows


def _bounded_depth_chart_rows(
    snapshots: Iterable[Mapping[str, Any]],
    chart_name: str,
) -> list[list[Any]]:
    retained = tuple(snapshots)
    maximum_rows = int(_probe_wandb._MAX_TABLE_ROWS)
    while retained:
        rows = _ORIGINAL_DEPTH_CHART_ROWS(retained, chart_name)
        if len(rows) <= maximum_rows:
            return rows
        if len(retained) > 1:
            retained = retained[1:]
            continue

        family = retained[0]["families"][chart_name]
        points_per_curve = len(tuple(family["depth_coordinates"]))
        if points_per_curve < 1:
            return []
        complete_curve_count = maximum_rows // points_per_curve
        if complete_curve_count < 1:
            return rows[-maximum_rows:]
        keep_rows = complete_curve_count * points_per_curve
        return rows[-keep_rows:]
    return []


_depth._depth_chart_rows = _bounded_depth_chart_rows
# ^^^ THOG


__all__ = ["_bounded_depth_chart_rows"]
# ^^^ THOG
