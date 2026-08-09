# vvv THOG
"""Keep infeasible lower probe points out of v0.55 stratified decision strata without hiding informational far-right probes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import plastic_depth_sen_kendall_v055_patch as _v055


_ORIGINAL_STRATIFIED_SELECTOR = (
    _v055.choose_plastic_depth_count_with_stratified_sen_kendall_v055
)


def _stratified_selector_with_feasible_decision_points(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    **kwargs: Any,
):
    current = int(current_count)
    finite_economic_scores = _v055._tsk._equivalent_time_scores(score_report)
    filtered_report = tuple(
        item
        for item in score_report
        if int(item.get("active_layers", current)) > current + 1
        or int(item.get("active_layers", current)) in finite_economic_scores
    )
    return _ORIGINAL_STRATIFIED_SELECTOR(
        current_count=current,
        score_report=filtered_report,
        **kwargs,
    )


_v055.choose_plastic_depth_count_with_stratified_sen_kendall_v055 = (
    _stratified_selector_with_feasible_decision_points
)
# ^^^ THOG
