# vvv THOG
"""Keep the established depth-weight snapshot seam patchable while retaining v2 Plotly rendering."""

from __future__ import annotations

from typing import Any, Dict

import constants as _constants

from . import depth_weight_curves_and_observational_probes_patch as _depth
from . import depth_weight_curves_v2_patch as _v2
# vvv THOG local destination persists exact snapshots without constructing or versioning Plotly media in the training process
from .local_chart_store import ensure_local_chart_store
# ^^^ THOG


# vvv THOG runtime dispatch intentionally resolves _depth._depth_weight_snapshot at call time so tests and later overlays can replace snapshot generation without replacing telemetry wiring
def _log_depth_weight_snapshot_with_patchable_snapshot(
    trainer: Any,
    telemetry: Any,
    *,
    optimizer_update: int,
) -> None:
    if int(getattr(_constants, "DEBUG", 0)) <= 2:
        return
    destination = _depth._destination()
    if destination == "none":
        return
    if destination == "local" and bool(
        getattr(telemetry, "_thog_local_depth_weight_capture_disabled", False)
    ):
        return

    snapshot = _depth._depth_weight_snapshot(
        trainer,
        telemetry,
        optimizer_update=optimizer_update,
    )
    if not snapshot:
        return
    if destination == "local":
        try:
            ensure_local_chart_store(telemetry).append_depth_weight_snapshot(
                snapshot,
                history_length=(
                    1 if _depth._time_mode() == "latest" else _depth._history_length()
                ),
            )
        except Exception as error:
            telemetry._thog_local_depth_weight_capture_disabled = True
            print(
                "THOG2 WARNING: local DEPTH weight logging failed; "
                f"continuing training without further weight capture: {error}",
                flush=True,
            )
        return
    if telemetry.run is None:
        raise RuntimeError(
            "DEPTH weight-curve destination wandb requires active W&B instrumentation"
        )
    history = _depth._depth_weight_history(telemetry)
    history.append(snapshot)
    snapshots = (snapshot,) if _depth._time_mode() == "latest" else tuple(history)

    payload: Dict[str, Any] = {}
    for chart_name in _v2._CHART_FAMILIES:
        payload[f"depth/{chart_name}"] = _v2._build_depth_plotly_figure(snapshots, chart_name)
    try:
        telemetry.run.log(payload, step=int(optimizer_update))
    except TypeError:
        telemetry.run.log(payload)

    # vvv THOG W&B receives only the six requested depth figures; selection metadata remains in run configuration and must not create scalar panels
    # if not bool(getattr(telemetry, "_thog_depth_weight_curve_selection_logged", False)):
    #     selection = _v2._selected_scalar_coordinates_v2(trainer, telemetry)
    #     metadata = {
    #         "depth/selection_seed": int(selection["seed"]),
    #         "depth/attention_head": int(selection["attention_head"]),
    #         "depth/same_coordinates_all_runs": bool(_depth._same_coordinates_all_runs()),
    #         "depth/scalar_weights_per_matrix": int(_depth._scalar_weights_per_matrix()),
    #         "depth/evaluation_points": int(_depth._depth_evaluation_points()),
    #         "depth/x_axis": "executed_layer_index",
    #         "depth/chart_renderer": "plotly",
    #     }
    #     try:
    #         telemetry.run.log(metadata, step=int(optimizer_update))
    #     except TypeError:
    #         telemetry.run.log(metadata)
    #     telemetry._thog_depth_weight_curve_selection_logged = True
    # ^^^ THOG
# ^^^ THOG


_depth._log_depth_weight_snapshot = _log_depth_weight_snapshot_with_patchable_snapshot


__all__ = ["_log_depth_weight_snapshot_with_patchable_snapshot"]
# ^^^ THOG
