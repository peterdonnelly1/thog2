# vvv THOG
"""Final operator-facing PLASTIC header and compact progress-row layout."""

from __future__ import annotations

import re
import sys
from typing import Any, Optional

from . import plastic_depth_directional_coherence_patch as _directional
from . import plastic_depth_same_batch_visibility_patch as _visibility
from . import plastic_depth_theil_sen_kendall_patch as _gradient
from . import stage6_trainer as _stage6


_ORIGINAL_FIX_GRADIENT_NORM_WIDTH = _visibility._fix_gradient_norm_width
_ORIGINAL_STAGE6_INIT = _stage6.Stage6Trainer.__init__
_ORIGINAL_FINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line
_STARTUP_GRADIENT_VISIBILITY_INSTALLED = False
_WINDOW_LABEL = "plastic__layer_count_probe__window_size_as_number_of_probes:"
_ALGORITHM_LABEL = "plastic__layer_count_decision_algorithm:"
_TAU_LABEL = "plastic__layer_count_gradient__minimum_absolute_kendall_tau:"
_FINAL_SAMPLED_ARRAY = re.compile(r"sampled \[(?P<body>[^\]]*)\]")


def _progress_elapsed_decimal_hours(value: Any, completed_updates: Any) -> str:
    del completed_updates
    elapsed_seconds = max(0, int(round(float(str(value).strip()))))
    return f"{elapsed_seconds:7d}s {elapsed_seconds / 3600.0:8.3f}"


_stage6._progress_elapsed = _progress_elapsed_decimal_hours


def _compact_progress_fields(line: str) -> str:
    rendered = _ORIGINAL_FIX_GRADIENT_NORM_WIDTH(line)
    rendered = rendered.replace("Δstep=", "Δ=", 1)
    rendered = re.sub(
        r"Δ=\s*(?P<seconds>[+-]?(?:\d+(?:\.\d*)?|\.\d+))s",
        lambda match: f"Δ={float(match.group('seconds')):5.1f}s",
        rendered,
        count=1,
    )
    rendered = rendered.replace("tokens=", "toks=", 1)
    rendered = rendered.replace("g nrm=", "g/n=", 1)
    return rendered


_visibility._fix_gradient_norm_width = _compact_progress_fields


def _visible_width(text: str) -> int:
    return len(_visibility._ANSI_ESCAPE.sub("", text).expandtabs(8))


def _pull_sampled_left_three_columns(line: str) -> str:
    sampled = _directional._SAMPLED_ARRAY.search(line)
    if sampled is None:
        return line
    current_prefix = line[: sampled.start()]
    current_column = _visible_width(current_prefix)
    compact_prefix = current_prefix.rstrip(" \t")
    compact_column = _visible_width(compact_prefix)
    target_column = max(compact_column + 1, current_column - 3)
    gap_width = max(1, target_column - compact_column)
    return compact_prefix + (" " * gap_width) + line[sampled.start() :]


_directional._align_sampled_to_minimum_tab_column = _pull_sampled_left_three_columns


def _gradient_header_rows() -> tuple[tuple[str, str], tuple[str, str]]:
    return (
        (_ALGORITHM_LABEL, _gradient._runtime_algorithm()),
        (
            _TAU_LABEL,
            f"{_gradient._runtime_minimum_absolute_kendall_tau():g}",
        ),
    )


def _startup_runner_module() -> Optional[Any]:
    for module_name in ("run_thog2_owt", "__main__"):
        runner = sys.modules.get(module_name)
        if runner is not None and hasattr(runner, "_print_plastic_option"):
            return runner
    return None


def _install_gradient_startup_visibility() -> None:
    global _STARTUP_GRADIENT_VISIBILITY_INSTALLED
    if _STARTUP_GRADIENT_VISIBILITY_INSTALLED:
        return
    runner = _startup_runner_module()
    if runner is None:
        return
    original = runner._print_plastic_option

    def print_plastic_option_with_gradient_controls(label: str, value: str) -> None:
        original(label, value)
        if label == _WINDOW_LABEL:
            for extra_label, extra_value in _gradient_header_rows():
                original(extra_label, extra_value)

    runner._print_plastic_option = print_plastic_option_with_gradient_controls
    _STARTUP_GRADIENT_VISIBILITY_INSTALLED = True


def _stage6_init_with_gradient_startup_visibility(self: Any, *args: Any, **kwargs: Any) -> None:
    _ORIGINAL_STAGE6_INIT(self, *args, **kwargs)
    _install_gradient_startup_visibility()


_stage6.Stage6Trainer.__init__ = _stage6_init_with_gradient_startup_visibility


def _finalize_compact_progress_line(line: str) -> str:
    current_sampled = _directional._SAMPLED_ARRAY.search(line)
    target_sampled_column = None
    if current_sampled is not None:
        target_sampled_column = max(0, _visible_width(line[: current_sampled.start()]) - 4)

    rendered = line.replace("layers = ", "layers ", 1)
    rendered = rendered.replace("sampled = ", "sampled ", 1)

    sampled = _FINAL_SAMPLED_ARRAY.search(rendered)
    if sampled is None or target_sampled_column is None:
        return rendered
    compact_prefix = rendered[: sampled.start()].rstrip(" \t")
    compact_column = _visible_width(compact_prefix)
    gap_width = max(1, target_sampled_column - compact_column)
    return compact_prefix + (" " * gap_width) + rendered[sampled.start() :]


def _format_progress_line_with_final_compaction(
    run_id: str,
    event: str,
    payload: dict[str, Any],
) -> str:
    return _finalize_compact_progress_line(
        _ORIGINAL_FINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    )


_stage6.format_progress_line = _format_progress_line_with_final_compaction


__all__ = [
    "_compact_progress_fields",
    "_finalize_compact_progress_line",
    "_gradient_header_rows",
    "_install_gradient_startup_visibility",
    "_progress_elapsed_decimal_hours",
    "_pull_sampled_left_three_columns",
]
# ^^^ THOG
