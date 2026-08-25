# vvv THOG
"""Retain the absolute centre/L probe loss needed by local heatmap relative display modes."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from . import plastic_depth_wandb_probe_curves_patch as _probe_curves


_ORIGINAL_PROBE_RECORD_FROM_EVENT = _probe_curves._probe_record_from_event
_ORIGINAL_HEATMAP_RENDER_DATA = _probe_curves._delta_loss_heatmap_render_data
_ORIGINAL_HEATMAP_FIGURE = _probe_curves._delta_loss_heatmap_figure


def _probe_record_from_event_with_current_loss(event: Any) -> Optional[Dict[str, Any]]:
    record = _ORIGINAL_PROBE_RECORD_FROM_EVENT(event)
    if record is None:
        return None
    payload = getattr(event, "payload", None)
    if not isinstance(payload, Mapping):
        return record
    current = int(record["active_layers"])
    for candidate in payload.get("candidates", ()):
        if not isinstance(candidate, Mapping):
            continue
        try:
            if int(candidate["active_layers"]) != current:
                continue
            current_loss = float(candidate["validation_loss"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(current_loss):
            return {**record, "current_loss": current_loss}
    return record


def _heatmap_render_data_with_current_loss(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    rendered = _ORIGINAL_HEATMAP_RENDER_DATA(*args, **kwargs)
    history = args[0] if args else kwargs["history"]
    rendered_row_limit = int(
        kwargs.get(
            "rendered_row_limit",
            _probe_curves._DELTA_LOSS_HEATMAP_MAX_RENDERED_ROWS,
        )
    )
    indices = _probe_curves._evenly_spaced_record_indices(
        len(history),
        rendered_row_limit,
    )
    rendered["current_losses"] = tuple(
        None
        if history[index].get("current_loss") is None
        else float(history[index]["current_loss"])
        for index in indices
    )
    for field, default in (
        ("selected_layers", None),
        ("brake_active", False),
        ("decision_committed", False),
        ("chaos_bump", None),
    ):
        rendered[field] = tuple(
            (
                int(history[index].get(field, history[index]["active_layers"]))
                if field == "selected_layers"
                else history[index].get(field, default)
            )
            for index in indices
        )
    return rendered


def _heatmap_figure_with_current_loss(
    history: Any,
    *,
    maximum_layers: int,
    abs_limit: float,
    go_module: Any = None,
) -> Any:
    figure = _ORIGINAL_HEATMAP_FIGURE(
        history,
        maximum_layers=maximum_layers,
        abs_limit=abs_limit,
        go_module=go_module,
    )
    rendered = _heatmap_render_data_with_current_loss(
        history,
        maximum_layers=maximum_layers,
    )
    figure.update_layout(
        meta={
            **(dict(figure.layout.meta) if isinstance(figure.layout.meta, Mapping) else {}),
            "thog2_current_losses": list(rendered["current_losses"]),
            "thog2_active_layers": list(rendered["active_layers"]),
            "thog2_selected_layers": list(rendered["selected_layers"]),
            "thog2_brake_active": list(rendered["brake_active"]),
            "thog2_decision_committed": list(rendered["decision_committed"]),
            "thog2_chaos_bump": list(rendered["chaos_bump"]),
            "thog2_optimizer_updates": list(rendered["optimizer_updates"]),
        }
    )
    return figure


_probe_curves._probe_record_from_event = _probe_record_from_event_with_current_loss
_probe_curves._delta_loss_heatmap_render_data = _heatmap_render_data_with_current_loss
_probe_curves._delta_loss_heatmap_figure = _heatmap_figure_with_current_loss
# ^^^ THOG
