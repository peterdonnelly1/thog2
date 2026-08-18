# vvv THOG
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import run_thog2_local_dashboard as dashboard
from sheet.local_chart_store import (
    LocalChartStore,
    LocalChartReader,
    close_local_chart_store,
    ensure_local_chart_store,
)


def _telemetry(*, artifact: str, run_id: str, url: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        name=artifact,
        config={"host_label": "scruffy", "model_type": "sheet"},
        run=SimpleNamespace(id=run_id, url=url),
    )


def _heatmap_record(update: int) -> dict:
    return {
        "optimizer_update": update,
        "probe_id": f"P{update}",
        "active_layers": 4,
        "selected_layers": 4,
        "shrink": ((1, -0.01, 3, -1), (0, 0.0, 4, 0)),
        "growth": ((0, 0.0, 4, 0), (1, 0.01, 5, 1)),
    }


def test_viewer_catalog_waits_when_started_before_training(tmp_path: Path) -> None:
    catalog = dashboard.DashboardCatalog(root=tmp_path)

    before = catalog.runs()

    assert before["runs"] == []
    assert before["waiting"] is True
    assert before["recommended_run_id"] is None


def test_wandb_run_id_separates_repeated_artifact_names(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("THOG2_INSTRUMENTATION_LOCAL_ROOT", str(tmp_path))
    first = _telemetry(
        artifact="same_artifact",
        run_id="wandb_a1",
        url="https://wandb.ai/example/project/runs/wandb_a1",
    )
    second = _telemetry(
        artifact="same_artifact",
        run_id="wandb_b2",
        url="https://wandb.ai/example/project/runs/wandb_b2",
    )
    first_store = ensure_local_chart_store(first)
    first_store.append_heatmap_records((_heatmap_record(10),))
    close_local_chart_store(first)
    second_store = ensure_local_chart_store(second)
    second_store.append_heatmap_records((_heatmap_record(20),))
    close_local_chart_store(second)

    assert first_store.path != second_store.path
    assert first_store.path == tmp_path / "same_artifact" / "wandb_a1" / "charts.sqlite3"
    assert second_store.path == tmp_path / "same_artifact" / "wandb_b2" / "charts.sqlite3"
    assert LocalChartReader(first_store.path).status()["heatmap_maximum_update"] == 10
    assert LocalChartReader(second_store.path).status()["heatmap_maximum_update"] == 20

    runs = dashboard.DashboardCatalog(root=tmp_path).runs()["runs"]
    assert runs[0]["local_run_id"] == "wandb_b2"
    assert {run["local_run_id"] for run in runs} == {"wandb_a1", "wandb_b2"}
    assert {run["artifact_name"] for run in runs} == {"same_artifact"}
    assert {run["run_state"] for run in runs} == {"finished"}
    assert {run["host_label"] for run in runs} == {"scruffy"}
    assert all(Path(run["run_directory"]).is_dir() for run in runs)


def test_same_wandb_run_id_intentionally_continues_local_history(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("THOG2_INSTRUMENTATION_LOCAL_ROOT", str(tmp_path))
    initial = _telemetry(artifact="resumed_artifact", run_id="resume_me")
    initial_store = ensure_local_chart_store(initial)
    initial_store.append_heatmap_records((_heatmap_record(10),))
    close_local_chart_store(initial)

    resumed = _telemetry(artifact="resumed_artifact", run_id="resume_me")
    resumed_store = ensure_local_chart_store(resumed)
    resumed_store.append_heatmap_records((_heatmap_record(20),))
    close_local_chart_store(resumed)

    assert resumed_store.path == initial_store.path
    assert LocalChartReader(resumed_store.path).status()["heatmap_count"] == 2


def test_requested_run_can_appear_after_catalog_creation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("THOG2_INSTRUMENTATION_LOCAL_ROOT", str(tmp_path))
    catalog = dashboard.DashboardCatalog(root=tmp_path, requested_run="later_id")
    assert catalog.runs()["waiting"] is True

    telemetry = _telemetry(artifact="later_artifact", run_id="later_id")
    store = ensure_local_chart_store(telemetry)
    store.append_heatmap_records((_heatmap_record(30),))

    after = catalog.runs()
    assert after["waiting"] is False
    assert after["runs"][0]["local_run_id"] == "later_id"
    assert catalog.state_for_run("later_id").status()["maximum_update"] == 30
    close_local_chart_store(telemetry)


def test_artifact_request_recommends_active_wandb_run_over_legacy_data(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("THOG2_INSTRUMENTATION_LOCAL_ROOT", str(tmp_path))
    legacy = LocalChartStore(
        tmp_path / "reused_artifact" / "charts.sqlite3",
        run_name="reused_artifact",
        config={},
    )
    legacy.append_heatmap_records((_heatmap_record(38),))
    legacy.close()

    active = _telemetry(artifact="reused_artifact", run_id="fresh_wandb_id")
    active_store = ensure_local_chart_store(active)
    active_store.append_heatmap_records((_heatmap_record(3),))

    catalog = dashboard.DashboardCatalog(
        root=tmp_path,
        requested_run="reused_artifact",
    ).runs()

    assert len(catalog["runs"]) == 2
    assert catalog["recommended_run_id"] == "fresh_wandb_id"
    by_id = {run["local_run_id"]: run for run in catalog["runs"]}
    assert by_id["reused_artifact"]["is_legacy_layout"] is True
    assert by_id["reused_artifact"]["dashboard_run_id"] == "legacy:reused_artifact"
    assert by_id["fresh_wandb_id"]["is_legacy_layout"] is False
    assert (
        dashboard.DashboardCatalog(root=tmp_path)
        .state_for_run("reused_artifact")
        .status()["maximum_update"]
        == 3
    )
    assert (
        dashboard.DashboardCatalog(root=tmp_path)
        .state_for_run("legacy:reused_artifact")
        .status()["maximum_update"]
        == 38
    )
    assert (
        dashboard.DashboardCatalog(root=tmp_path)
        .state_for_run("fresh_wandb_id")
        .status()["maximum_update"]
        == 3
    )
    close_local_chart_store(active)


def test_delete_run_removes_only_local_chart_database_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("THOG2_INSTRUMENTATION_LOCAL_ROOT", str(tmp_path))
    telemetry = _telemetry(artifact="delete_artifact", run_id="delete_me")
    store = ensure_local_chart_store(telemetry)
    store.append_heatmap_records((_heatmap_record(7),))
    close_local_chart_store(telemetry)
    sibling = store.path.parent / "keep_this_file.txt"
    sibling.write_text("not dashboard data", encoding="utf-8")

    catalog = dashboard.DashboardCatalog(root=tmp_path)
    result = catalog.delete_run("delete_me")

    assert result["deleted_run_id"] == "delete_me"
    assert not store.path.exists()
    assert sibling.read_text(encoding="utf-8") == "not dashboard data"
    assert result["removed_directory"] is False
    assert catalog.runs()["runs"] == []


def test_instra_files_browse_only_the_selected_local_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("THOG2_INSTRUMENTATION_LOCAL_ROOT", str(tmp_path))
    telemetry = _telemetry(artifact="files_artifact", run_id="files_run")
    store = ensure_local_chart_store(telemetry)
    local_root = store.path.parent
    notes = local_root / "notes"
    notes.mkdir()
    local_file = notes / "summary.txt"
    local_file.write_text("instra file browser", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("must remain private", encoding="utf-8")
    blocked_link = local_root / "outside-link"
    blocked_link.symlink_to(outside)

    catalog = dashboard.DashboardCatalog(root=tmp_path)
    root_listing = catalog.local_files("files_run")
    nested_listing = catalog.local_files("files_run", "notes")

    assert root_listing["source"] == "instra"
    assert root_listing["current_path"] == ""
    assert {entry["name"]: entry["kind"] for entry in root_listing["entries"]}[
        "notes"
    ] == "folder"
    assert {entry["name"]: entry["kind"] for entry in root_listing["entries"]}[
        "outside-link"
    ] == "symlink"
    assert nested_listing["parent_path"] == ""
    assert nested_listing["entries"][0]["path"] == "notes/summary.txt"
    assert catalog.local_file("files_run", "notes/summary.txt") == local_file
    with pytest.raises(ValueError):
        catalog.local_file("files_run", "../outside.txt")
    with pytest.raises(PermissionError):
        catalog.local_file("files_run", "outside-link")
    close_local_chart_store(telemetry)


def test_wandb_files_are_exposed_as_a_folder_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("THOG2_INSTRUMENTATION_LOCAL_ROOT", str(tmp_path))
    telemetry = _telemetry(
        artifact="wandb_files_artifact",
        run_id="wandb_files_run",
        url="https://wandb.ai/example/project/runs/wandb_files_run",
    )
    ensure_local_chart_store(telemetry)
    requested_references = []
    remote_files = (
        SimpleNamespace(
            name="config.yaml",
            size=120,
            mimetype="application/yaml",
            md5="config-digest",
            updated_at="2026-08-16T10:30:00Z",
            direct_url="https://storage.wandb.ai/files/config.yaml",
            url="https://api.wandb.ai/files/config.yaml",
        ),
        SimpleNamespace(
            name="media/table/data.json",
            size=640,
            mimetype="application/json",
            md5="data-digest",
            updated_at="2026-08-16T10:31:00Z",
            direct_url="https://storage.wandb.ai/files/data.json",
            url="https://api.wandb.ai/files/data.json",
        ),
    )

    class FakeApi:
        def run(self, reference):
            requested_references.append(reference)
            return SimpleNamespace(files=lambda: remote_files)

    catalog = dashboard.DashboardCatalog(
        root=tmp_path,
        wandb_api_factory=FakeApi,
    )
    root_listing = catalog.wandb_files("wandb_files_run")
    media_listing = catalog.wandb_files("wandb_files_run", "media")

    assert requested_references == ["example/project/wandb_files_run"]
    assert root_listing["available"] is True
    assert root_listing["manifest_count"] == 2
    assert [(entry["name"], entry["kind"]) for entry in root_listing["entries"]] == [
        ("media", "folder"),
        ("config.yaml", "file"),
    ]
    assert media_listing["entries"][0]["path"] == "media/table"
    assert root_listing["entries"][1]["download_url"].endswith("/config.yaml")
    assert root_listing["entries"][1]["modified_at"] == "2026-08-16T10:30:00Z"
    assert root_listing["wandb_files_url"].endswith("/wandb_files_run/files")
    with pytest.raises(ValueError):
        catalog.wandb_files("wandb_files_run", "../outside")
    close_local_chart_store(telemetry)


def test_dashboard_uses_persistent_split_workspace_and_clean_plot_nodes() -> None:
    html = (dashboard._ASSET_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (dashboard._ASSET_ROOT / "dashboard.js").read_text(encoding="utf-8")
    heatmap_patch = (
        dashboard._ASSET_ROOT / "dashboard_heatmap_patch.js"
    ).read_text(encoding="utf-8")
    heatmap_loss_patch = (
        dashboard._ASSET_ROOT / "dashboard_heatmap_loss_patch.js"
    ).read_text(encoding="utf-8")
    heatmap_centre_format_patch = (
        dashboard._ASSET_ROOT / "dashboard_heatmap_centre_format_patch.js"
    ).read_text(encoding="utf-8")
    heatmap_zoom_geometry_patch = (
        dashboard._ASSET_ROOT / "dashboard_heatmap_zoom_geometry_patch.js"
    ).read_text(encoding="utf-8")
    heatmap_top_anchor_patch = (
        dashboard._ASSET_ROOT / "dashboard_heatmap_top_anchor_pencil_patch.js"
    ).read_text(encoding="utf-8")
    heatmap_final_presentation_patch = (
        dashboard._ASSET_ROOT / "dashboard_final_presentation_settings_patch.js"
    ).read_text(encoding="utf-8")
    heatmap_dom_alignment_patch = (
        dashboard._ASSET_ROOT / "dashboard_heatmap_dom_alignment_patch.js"
    ).read_text(encoding="utf-8")
    wandb_groups_patch = (
        dashboard._ASSET_ROOT / "dashboard_wandb_groups_patch.js"
    ).read_text(encoding="utf-8")
    stylesheet = (dashboard._ASSET_ROOT / "dashboard.css").read_text(encoding="utf-8")

    assert 'id="heatmap_placeholder"' in html
    assert 'id="heatmap_plot"' in html
    assert 'id="runs_pane"' in html
    assert 'id="workspace_divider"' in html
    assert 'class="icon-rail"' in html
    assert 'id="settings_nav"' in html
    assert 'id="page_size"' in html
    assert 'id="sort_direction"' in html
    assert 'id="run_menu"' in html
    assert 'id="chart_settings_overlay"' in html
    assert 'id="chart_x_min"' in html
    assert 'id="chart_x_max"' in html
    assert 'id="chart_y_min"' in html
    assert 'id="chart_y_max"' in html
    assert 'id="chart_settings_preview"' in html
    assert 'id="chart_title_value"' in html
    assert 'id="chart_x_label"' in html
    assert 'id="chart_x_axis_mode"' in html
    assert '>Relative Time (Wall)</option>' in html
    assert '>Relative Time (Process)</option>' in html
    assert '>Wall Time</option>' in html
    assert 'id="chart_y_label"' in html
    assert 'id="chart_max_snapshots"' in html
    assert 'id="chart_exclude_outliers"' in html
    assert 'id="chart_smoothing"' in html
    assert 'id="chart_line_width"' in html
    assert 'id="chart_heatmap_row_height"' in html
    assert 'id="chart_show_grid"' in html
    assert 'id="depth_chart_group"' in html
    assert 'id="depth_group_toggle"' in html
    assert "instra · THOG2 instrumentation" in html
    assert 'class="run-tabs"' not in html
    assert 'id="files_workspace"' in html
    assert 'id="instra_files_tab"' in html
    assert 'id="wandb_files_tab"' in html
    assert 'id="file_breadcrumbs"' in html
    assert 'id="files_body"' in html
    assert "<th>File name</th>" in html
    assert ">Last modified</th>" in html
    assert ">Download</th>" in html
    assert ">Probes</th>" in html
    assert ">Curves</th>" in html
    assert ">Logged</th>" in html
    assert "mount.replaceChildren();" in javascript
    assert "Plotly.newPlot" in javascript
    assert "transpose_heatmap" in javascript
    assert "signed_layer_offset(offset),\n        Number(cell)," in heatmap_patch
    assert 'scaleanchor = "x"' in javascript
    assert 'constraintoward = "bottom"' in javascript
    assert 'color: "white"' in javascript
    assert "migrate_panel_layout" in javascript
    assert "toggle_chart_group" in javascript
    assert "eye_closed:" in javascript
    assert "crashed:" in javascript
    assert "toggle_maximized_chart" in javascript
    assert "position_restore_button" in javascript
    assert "chart_axis_settings" in javascript
    assert "apply_chart_axis_settings" in javascript
    assert "save_chart_settings" in javascript
    assert "reset_chart_settings" in javascript
    assert "render_chart_settings_preview" in javascript
    assert "apply_chart_display_settings" in javascript
    assert "function apply_chart_x_axis_mode" in javascript
    assert "value / 3600.0" in javascript
    assert "value * 1000.0" in javascript
    assert "dynamic_chart_figures" in javascript
    assert "dynamic_chart_metadata" in javascript
    assert "limit_curve_snapshots" in javascript
    assert "apply_outlier_resistant_y_range" in javascript
    assert "smoothed_values" in javascript
    assert "function chart_settings_icon()" in javascript
    assert 'button.appendChild(chart_settings_icon())' in javascript
    assert "function install_universal_chart_settings()" in javascript
    assert 'root.querySelectorAll(".chart-card[data-chart]").forEach(ensure_chart_settings_button);' in javascript
    assert "app.chart_settings_observer.observe(root, {childList: true, subtree: true});" in javascript
    assert "install_universal_chart_settings();" in javascript
    assert '<svg viewBox="0 0 24 24" aria-hidden="true">' in html
    assert "start_chart_resize" in javascript
    assert "should_follow_recommendation" in javascript
    assert "set_workspace_view" not in javascript
    assert "set_file_source" in javascript
    assert 'const local_detail_tabs = Object.freeze(["charts", "overview", "logs", "files", "artifacts"]);' in heatmap_patch
    assert 'by_id("files_workspace").hidden = !has_run || !files;' in heatmap_patch
    assert 'if (tab_name === "files") refresh_files();' in heatmap_patch
    assert "actions.insertBefore(control, maximize);" in heatmap_patch
    assert "header.insertBefore(control, maximize);" not in heatmap_patch
    assert 'const actions = maximize?.parentElement;' in heatmap_loss_patch
    assert "actions.insertBefore(button, vertical_control || maximize);" in heatmap_loss_patch
    assert "heatmap_header.insertBefore(button" not in heatmap_loss_patch
    assert "...centre_annotations(prepared, heatmap_trace, current_losses)" not in heatmap_loss_patch
    assert "annotations.push(...centre_annotations" not in heatmap_centre_format_patch
    assert "prepared.layout.shapes = existing_shapes;" in heatmap_centre_format_patch
    assert "annotations.push(...dynamic_centre_annotations" not in heatmap_zoom_geometry_patch
    assert '.filter(shape => shape?.name !== "thog2-centre-datum-background")' in heatmap_zoom_geometry_patch
    assert "const heatmap_chrome_height_px = 152;" in heatmap_top_anchor_patch
    assert "best_better_loss_annotations" in heatmap_top_anchor_patch
    assert "candidate_loss = current_loss + candidate_delta" in heatmap_top_anchor_patch
    assert 'name: "thog2-best-better-loss"' in heatmap_top_anchor_patch
    assert 'color: "#000000"' in heatmap_top_anchor_patch
    assert "prepared.layout.xaxis2 =" in heatmap_top_anchor_patch
    assert 'overlaying: "x"' in heatmap_top_anchor_patch
    assert 'matches: "x"' in heatmap_top_anchor_patch
    assert "b: 76" in heatmap_top_anchor_patch
    assert "const heatmap_chrome_height_px = 152;" in heatmap_final_presentation_patch
    assert 'for (const axis_name of ["xaxis", "xaxis2"])' in heatmap_final_presentation_patch
    assert "t: 76" in heatmap_final_presentation_patch
    assert "b: 76" in heatmap_final_presentation_patch
    assert "yshift: 50" not in heatmap_final_presentation_patch
    assert "heatmap_x_title_node" not in heatmap_dom_alignment_patch
    assert 'update["margin.t"]' not in heatmap_zoom_geometry_patch
    assert 'chart: stored_chart_settings("heatmap")' in heatmap_patch
    assert "app.dynamic_chart_figures?.[chart_name]" in heatmap_patch
    assert "trajectory_chart_names_fast.map(chart_name => [chart_name, stored_chart_settings(chart_name)])" in heatmap_patch
    assert 'name_button.addEventListener("click", () => set_file_path(entry.path));' in javascript
    assert 'window.open(local_file_url(entry.path), "_blank", "noopener")' in javascript
    assert '"/api/local-files"' in javascript
    assert '"/api/wandb-files"' in javascript
    assert '.chart-card[data-chart="heatmap"] { flex: 1 1 100%; }' in stylesheet
    assert '.chart-card:not([data-chart="heatmap"]) { min-width: 180px; flex: 1 1 calc(33.333% - 10px); }' in stylesheet
    assert ".plot-shell { position: absolute; inset: 52px 0 0 0; overflow: auto;" in stylesheet
    assert '.chart-card:not([data-chart="heatmap"]) { min-width: 180px;' in stylesheet
    assert ".plot-mount { position: relative; z-index: 2; width: 100%; min-width: 0;" in stylesheet
    assert '.chart-card[data-chart="heatmap"] .plot-mount { width: max(100%, 720px);' in stylesheet
    assert ".chart-card.maximized .maximize-button, .chart-card.maximized .chart-settings-button { position: fixed;" in stylesheet
    assert ".files-workspace { flex: 1 1 auto; min-height: 0;" in stylesheet
    assert ".file-source-tabs { display: flex;" in stylesheet
    assert ".file-list-panel { flex: 1 1 auto; min-height: 0;" in stylesheet
    assert ".files-table-wrap { flex: 1 1 auto; position: relative; min-height: 0;" in stylesheet
    assert ".chart-settings-workspace { min-height: 0; display: grid;" in stylesheet
    assert ".chart-settings-preview-pane { min-width: 0; min-height: 0;" in stylesheet
    assert ".chart-settings-controls { min-width: 0; min-height: 0;" in stylesheet
    assert ".chart-settings-button svg { display: block; width: 17px; height: 17px;" in stylesheet
    assert "actions.append(chart_settings_button(key, title.textContent), maximize);" in wandb_groups_patch
    assert "app.dynamic_chart_figures[key] = figure;" in wandb_groups_patch
    assert "thog2_x_variants: series.x_variants || {}" in wandb_groups_patch
    assert "await render_plot(mount, figure, key);" in wandb_groups_patch


def test_dashboard_html_is_read_for_each_page_request() -> None:
    server_source = Path(dashboard.__file__).read_text(encoding="utf-8")

    handler_body = server_source.split("def _handler_for", 1)[1]
    assert 'index_html = (_ASSET_ROOT / "index.html").read_bytes()' not in handler_body
    assert '(_ASSET_ROOT / "index.html").read_bytes(),' in handler_body
# ^^^ THOG
