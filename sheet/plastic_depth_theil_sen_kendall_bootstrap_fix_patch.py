# vvv THOG
"""Preserve v0.531 raw-loss bootstrap exploration until v0.54 TSK has real economic scores."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from . import plastic_depth_controller as _controller
from . import plastic_depth_directional_coherence_patch as _directional
from . import plastic_depth_lookahead_patch as _lookahead
from . import plastic_depth_theil_sen_kendall_patch as _gradient
from . import plastic_depth_wall_time_equivalent_time_gain_patch as _wall_time
from . import trainer_step as _trainer_step


_ORIGINAL_GRADIENT_UPDATED_HISTORIES_AND_DIRECTION = _directional._updated_histories_and_direction
_ORIGINAL_GRADIENT_CHOOSE_COUNT = _lookahead.choose_plastic_depth_count_with_exact_radius


def _wall_time_score_report_is_bootstrap(
    score_report: Sequence[Mapping[str, object]],
) -> bool:
    wall_time_rows = tuple(
        item
        for item in score_report
        if str(item.get("wall_time_algorithm", "")) == _wall_time.WALL_TIME_ALGORITHM
    )
    return bool(wall_time_rows) and all(
        bool(item.get("wall_time_bootstrap", False))
        for item in wall_time_rows
    )


def _gradient_runtime_selected() -> bool:
    trainer = _wall_time._ACTIVE_TRAINER.get()
    return trainer is not None and _gradient._runtime_algorithm() == _gradient.GRADIENT_ALGORITHM


def _updated_histories_and_direction_with_v0531_bootstrap(
    *,
    current_count: int,
    score_report: Sequence[Mapping[str, object]],
    histories: Mapping[str, Sequence[float]],
    noise_window: int,
    extrapolation_weight: float,
) -> Tuple[Dict[str, Tuple[float, ...]], Dict[str, Any]]:
    if _gradient_runtime_selected() and _wall_time_score_report_is_bootstrap(score_report):
        return _gradient._ORIGINAL_UPDATED_HISTORIES_AND_DIRECTION(
            current_count=current_count,
            score_report=score_report,
            histories=histories,
            noise_window=noise_window,
            extrapolation_weight=extrapolation_weight,
        )
    return _ORIGINAL_GRADIENT_UPDATED_HISTORIES_AND_DIRECTION(
        current_count=current_count,
        score_report=score_report,
        histories=histories,
        noise_window=noise_window,
        extrapolation_weight=extrapolation_weight,
    )


def _choose_count_with_v0531_bootstrap(
    *,
    max_step: int = 1,
    extrapolation_weight: float = 0.8,
    **kwargs: Any,
):
    score_report = kwargs.get("score_report", ())
    if _gradient_runtime_selected() and _wall_time_score_report_is_bootstrap(score_report):
        return _gradient._ORIGINAL_DIRECTIONAL_SELECTOR(
            max_step=max_step,
            extrapolation_weight=extrapolation_weight,
            **kwargs,
        )
    return _ORIGINAL_GRADIENT_CHOOSE_COUNT(
        max_step=max_step,
        extrapolation_weight=extrapolation_weight,
        **kwargs,
    )


_directional._updated_histories_and_direction = _updated_histories_and_direction_with_v0531_bootstrap
_lookahead.choose_plastic_depth_count_with_exact_radius = _choose_count_with_v0531_bootstrap
_controller.choose_plastic_depth_count_with_mad = _choose_count_with_v0531_bootstrap
_trainer_step.choose_plastic_depth_count_with_mad = _choose_count_with_v0531_bootstrap
# ^^^ THOG
