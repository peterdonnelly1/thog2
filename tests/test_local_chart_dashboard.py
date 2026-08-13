# vvv THOG
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import run_thog2_local_dashboard as dashboard
from sheet.local_chart_store import (
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
    assert {run["local_run_id"] for run in runs} == {"wandb_a1", "wandb_b2"}
    assert {run["artifact_name"] for run in runs} == {"same_artifact"}
    assert {run["run_state"] for run in runs} == {"finished"}
    assert {run["host_label"] for run in runs} == {"scruffy"}


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


def test_dashboard_mounts_placeholders_outside_clean_plot_nodes() -> None:
    html = (dashboard._ASSET_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (dashboard._ASSET_ROOT / "dashboard.js").read_text(encoding="utf-8")

    assert 'id="heatmap_placeholder"' in html
    assert 'id="heatmap_plot"' in html
    assert "mount.replaceChildren();" in javascript
    assert "Plotly.newPlot" in javascript
    assert "open_modal" in javascript
    assert "modal_square_heatmap" in javascript
# ^^^ THOG
