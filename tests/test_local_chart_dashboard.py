# vvv THOG
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_dashboard_uses_persistent_split_workspace_and_clean_plot_nodes() -> None:
    html = (dashboard._ASSET_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (dashboard._ASSET_ROOT / "dashboard.js").read_text(encoding="utf-8")

    assert 'id="heatmap_placeholder"' in html
    assert 'id="heatmap_plot"' in html
    assert 'id="runs_pane"' in html
    assert 'id="workspace_divider"' in html
    assert 'class="icon-rail"' in html
    assert 'id="settings_nav"' in html
    assert 'id="page_size"' in html
    assert 'id="sort_direction"' in html
    assert 'id="run_menu"' in html
    assert "Heatmap probes" in html
    assert "Latest logged step" in html
    assert "mount.replaceChildren();" in javascript
    assert "Plotly.newPlot" in javascript
    assert "transpose_heatmap" in javascript
    assert 'scaleanchor = "x"' in javascript
    assert "toggle_maximized_chart" in javascript
    assert "start_chart_resize" in javascript
    assert "should_follow_recommendation" in javascript
# ^^^ THOG
