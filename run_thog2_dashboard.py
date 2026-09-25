# vvv THOG
"""Launch the local THOG2 dashboard with an obvious Linux process name."""

from __future__ import annotations

import ctypes
import errno                                                                                                                                                  # <<< THOG distinguish an occupied Instra port from other socket failures
import os
import signal
import shutil
import socket                                                                                                                                                 # <<< THOG check the listener address before registering a new Instra backend PID
import subprocess
import sys
import tempfile
import time
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

# vvv THOG Networks POST must precede the matched-weight handler, which returns plain-text 404 for every other POST path
_handler_for_before_network_post = _dashboard._handler_for


def _handler_for_with_network_post(catalog):
    from urllib.parse import urlparse

    handler = _handler_for_before_network_post(catalog)

    class NetworkPostHandler(handler):
        def do_POST(self):
            if urlparse(self.path).path == "/api/network/action":
                return _dashboard._network_do_post(self)
            return super().do_POST()

    return NetworkPostHandler


_dashboard._handler_for = _handler_for_with_network_post
# ^^^ THOG

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
    "dashboard_sep07_fixes_and_enhancements.js",                                                                                                          # <<< THOG final owner for September functionality, memory, table and interaction fixes
    "dashboard_sep20_integrated_repairs.js",                                                                                                              # <<< THOG final owner for paired/ordinary chart coexistence and September UI repairs
    "dashboard_sep21_instra_repairs.js",                                                                                                                  # <<< THOG final owner for long-session quiescence, pairing, columns, colours and timing-chart refinements
)

# Most entries above are retained source assets for old focused repairs and
# their regression fixtures.  Loading the entire historical stack gives many
# obsolete wrappers simultaneous ownership of render_runs(), select_run() and
# Plotly resize.  Keep the runtime deliberately small and ordered.
_ACTIVE_EXTRA_ASSET_NAMES = (
    "dashboard_wandb_groups_patch.js",
    "dashboard_group_stability_patch.js",
    "dashboard_sep20_integrated_repairs.js",
    "dashboard_sep21_instra_repairs.js",
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
    canonical_asset_root = Path(__file__).resolve().parent / "sheet" / "local_dashboard_assets"
    temporary = tempfile.TemporaryDirectory(prefix="thog2-dashboard-assets-")
    runtime_root = Path(temporary.name)
    for source in source_root.iterdir():
        if source.is_file():
            shutil.copy2(source, runtime_root / source.name)
    # The lower-level dashboard wrapper contains only its assembled core asset
    # set.  Late feature assets live in the canonical source directory and must
    # be copied as well as named in the HTML/allow-list; otherwise every script
    # tag below resolves to a 404 and regular Train/Val/System/Memory charts
    # silently disappear.
    for asset_name in _EXTRA_ASSET_NAMES:
        shutil.copy2(canonical_asset_root / asset_name, runtime_root / asset_name)
    for asset_name in ("dashboard_networks.js", "dashboard_networks.css"):
        shutil.copy2(canonical_asset_root / asset_name, runtime_root / asset_name)

    index_path = runtime_root / "index.html"
    index_html = index_path.read_text(encoding="utf-8")
    for asset_name in _ACTIVE_EXTRA_ASSET_NAMES:
        script_tag = f'  <script src="/assets/{asset_name}" defer></script>\n'
        if script_tag not in index_html:
            index_html = index_html.replace("</head>", f"{script_tag}</head>", 1)
    index_html = index_html.replace("</head>", '  <script src="/assets/dashboard_networks.js" defer></script>\n</head>', 1)                     # <<< THOG load Networks after the established dashboard owners
    index_path.write_text(index_html, encoding="utf-8")

    _dashboard._ASSET_ROOT = runtime_root
    _dashboard._ASSET_NAMES = frozenset(
        (*_dashboard._ASSET_NAMES, *_EXTRA_ASSET_NAMES, "dashboard_networks.js", "dashboard_networks.css")
    )
    return temporary


# vvv THOG one launch action starts the independent local node agent and records this backend's restart command
def _start_node_agent() -> None:
    import instra_node_agent

    try:
        instra_node_agent.request("state", timeout=1)
    except (OSError, RuntimeError):
        instra_node_agent.STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (instra_node_agent.STATE_DIR / "agent.log").open("ab") as output:
            subprocess.Popen([sys.executable, str(Path(instra_node_agent.__file__).resolve()), "serve"],
                             cwd=Path(__file__).resolve().parent, stdin=subprocess.DEVNULL,
                             stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(40):
            try:
                instra_node_agent.request("state", timeout=1)
                break
            except (OSError, RuntimeError):
                time.sleep(0.05)
        else:
            raise RuntimeError("Instra node agent did not start")
    instra_node_agent._install_agent_entry()                                                                                                                  # <<< THOG refresh the SSH entry even when an older node agent is already serving
    arguments = _dashboard._base.build_parser().parse_args()
    # vvv THOG a duplicate dashboard must not displace the running backend PID in the node agent
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((str(arguments.host), int(arguments.port)))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                raise RuntimeError(f"Instra is already listening on {arguments.host}:{arguments.port}; use the existing instance") from error
            raise
    # ^^^ THOG
    instra_node_agent.request("configure", {"launch_command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                                            "logs_root": str(arguments.root.resolve()), "backend_pid": os.getpid()})
# ^^^ THOG


def _run_backend_with_agent():
    _set_process_name()
    _start_node_agent()                                                                                                                                     # <<< THOG keep local discovery and recovery available after backend exit
    intentional_exit = False

    def _intentional_sigterm(_number, _frame):
        nonlocal intentional_exit
        intentional_exit = True
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _intentional_sigterm)                                                                                                      # <<< THOG record a deliberate user stop before the independent agent evaluates restart
    runtime_assets = _prepare_runtime_assets()
    try:
        exit_code = _dashboard.main()
        intentional_exit = exit_code == 0
        return exit_code
    finally:
        # vvv THOG a backend exception must leave the agent free to restart it
        if intentional_exit:
            try:
                import instra_node_agent
                instra_node_agent.request("backend_exited", {"backend_pid": os.getpid()}, timeout=1)
            except (OSError, RuntimeError):
                pass
        # ^^^ THOG
        runtime_assets.cleanup()


if __name__ == "__main__":
    raise SystemExit(_run_backend_with_agent())
# ^^^ THOG
