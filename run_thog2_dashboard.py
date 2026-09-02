# vvv THOG
"""Launch the local THOG2 dashboard with an obvious Linux process name."""

from __future__ import annotations

import ctypes
import shutil
import sys
import tempfile
from pathlib import Path

from sheet import local_heatmap_loss_metadata_patch as _local_heatmap_loss_metadata_patch
from sheet import local_dashboard_logs_patch as _local_dashboard_logs_patch
from sheet import local_dashboard_wandb_charts_patch as _local_dashboard_wandb_charts_patch
from sheet import local_dashboard_wandb_catchup_patch as _local_dashboard_wandb_catchup_patch
from sheet import local_dashboard_heatmap_window_patch as _local_dashboard_heatmap_window_patch
from sheet import local_dashboard_performance_patch as _local_dashboard_performance_patch
from sheet import local_dashboard_current_weights_performance_patch as _local_dashboard_current_weights_performance_patch
from sheet import local_dashboard_weight_step_range_patch as _local_dashboard_weight_step_range_patch
from sheet import local_dashboard_notes_patch as _local_dashboard_notes_patch                                                                                                             # <<< THOG add durable editable Overview notes without altering training telemetry
from sheet import matched_weight_selection_patch as _matched_weight_selection_patch
import run_thog2_local_dashboard as _dashboard


_local_dashboard_logs_patch.install(_dashboard)
_local_dashboard_wandb_charts_patch.install(_dashboard)
_local_dashboard_wandb_catchup_patch.install(_local_dashboard_wandb_charts_patch)
_local_dashboard_heatmap_window_patch.install()
_local_dashboard_performance_patch.install(_dashboard)
_matched_weight_selection_patch.install_dashboard(_dashboard)
_local_dashboard_current_weights_performance_patch.install(_dashboard)
_local_dashboard_weight_step_range_patch.install(_dashboard)
_local_dashboard_notes_patch.install(_dashboard)                                                                                                                                          # <<< THOG install the notes API/status seam before constructing the dashboard handler

from sheet.thogopt_dashboard import install as _install_thogopt_dashboard
_install_thogopt_dashboard(_dashboard)

_PROCESS_NAME = b"thog2-dashboard"
_PR_SET_NAME = 15
_EXTRA_ASSET_NAMES = (
    "dashboard_heatmap_loss_patch.js",
    "dashboard_heatmap_centre_format_patch.js",
    "dashboard_heatmap_geometry_final_patch.js",
    "dashboard_logs_modes_patch.js",
    "dashboard_overview_font_patch.js",
    "dashboard_synthetic_groups_patch.js",
    "dashboard_processing_copy_patch.js",
    "dashboard_navigation_polish_patch.js",
    "dashboard_wandb_groups_patch.js",
    "dashboard_group_stability_patch.js",
    "dashboard_heatmap_flip_log_reset_patch.js",
    "dashboard_heatmap_top_anchor_pencil_patch.js",
    "dashboard_maximize_lband_patch.js",
    "dashboard_final_presentation_settings_patch.js",
    "dashboard_heatmap_dom_alignment_patch.js",
    "dashboard_weight_request_router_patch.js",                                                                                                            # <<< THOG route Weights requests by final per-chart current/range semantics before the performance layer captures fetch_json
    "dashboard_performance_patch.js",
    "dashboard_heatmap_zoom_geometry_patch.js",
    "dashboard_heatmap_y_axis_refinement_patch.js",
    "dashboard_heatmap_v057_patch.js",
    "dashboard_v058_repair_workspace_patch.js",
    "dashboard_weights_group_settings_patch.js",
    "dashboard_preparing_workspace_train_patch.js",
    "dashboard_matched_weight_selection_patch.js",
    "dashboard_matched_weight_validation_patch.js",
    "dashboard_matched_weight_workspace_repair_patch.js",
    "dashboard_workspace_stability_performance_patch.js",
    "dashboard_weight_controls_run_table_patch.js",
    "dashboard_run_table_s_checkpoint_patch.js",
    "dashboard_weight_coupling_presentation_patch.js",
    "dashboard_workspace_depth_cache_patch.js",
    "dashboard_render_visibility_performance_patch.js",
    "dashboard_weight_step_controls_patch.js",
    "dashboard_legacy_heatmap_repair_patch.js",                                                                                                            # <<< THOG retain the legacy absolute-delta heatmap fallback without reintroducing global Weights state
    "dashboard_weight_stability_final_patch.js",                                                                                                           # <<< THOG install one dependency-gated final owner for run-scoped ranges, chart/group settings, loading state, and time filtering
    "dashboard_weight_coupling_reliability_patch.js",                                                                                                     # <<< THOG install deterministic final checkbox/render reliability only after the consolidated Weights owner
    "dashboard_weight_regression_final_patch.js",                                                                                                          # <<< THOG guard editable ranges, stacked layout, y headroom, Overview density, and retained-data RND last
    "dashboard_weight_range_interaction_final_patch.js",                                                                                                   # <<< THOG make explicit ranges authoritative, expose step hover, fit four-digit coupling indices, and restore functional RND
    "dashboard_consistency_final_patch.js",                                                                                                                # <<< THOG reconcile first-mount heatmaps, separate the colour key, and finalize the Overview allocation
    "dashboard_aug30_enhancements_patch.js",                                                                                                               # <<< THOG preserve Overview state, stack/collapse panels, resize NAME, and place STEPS after preset p
    "dashboard_instra_further_enhancements_patch.js",                                                                                                      # <<< THOG finalize the August 31 Overview, table, fullscreen, live-weight, palette, and view-state requirements
    "dashboard_weight_inspector.js",                                                                                                                       # <<< THOG inspect exact retained weights in a virtual grid and finalize latest-step deduplication
    "dashboard_thogopt.js",
)


def _set_process_name() -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None)
        prctl = libc.prctl
        prctl.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        )
        prctl.restype = ctypes.c_int
        prctl(_PR_SET_NAME, _PROCESS_NAME, 0, 0, 0)
    except (AttributeError, OSError):
        return


def _prepare_runtime_assets() -> tempfile.TemporaryDirectory[str]:
    source_root = Path(_dashboard._ASSET_ROOT)
    temporary = tempfile.TemporaryDirectory(prefix="thog2-dashboard-assets-")
    runtime_root = Path(temporary.name)
    for source in source_root.iterdir():
        if source.is_file():
            shutil.copy2(source, runtime_root / source.name)

    index_path = runtime_root / "index.html"
    index_html = index_path.read_text(encoding="utf-8")
    for asset_name in _EXTRA_ASSET_NAMES:
        script_tag = f'  <script src="/assets/{asset_name}" defer></script>\n'
        if script_tag not in index_html:
            index_html = index_html.replace("</head>", f"{script_tag}</head>", 1)
    index_path.write_text(index_html, encoding="utf-8")

    _dashboard._ASSET_ROOT = runtime_root
    _dashboard._ASSET_NAMES = frozenset(
        (*_dashboard._ASSET_NAMES, *_EXTRA_ASSET_NAMES)
    )
    return temporary


if __name__ == "__main__":
    _set_process_name()
    runtime_assets = _prepare_runtime_assets()
    try:
        raise SystemExit(_dashboard.main())
    finally:
        runtime_assets.cleanup()
# ^^^ THOG
