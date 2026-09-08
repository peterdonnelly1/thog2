# vvv THOG Premat Instra persistence and dedicated-view regression tests
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from run_thog2_local_dashboard import RunDashboardState
from sheet.local_chart_store import LocalChartReader, LocalChartStore


def _snapshot(update: int) -> dict:
    return {
        "version": 1,
        "attention_mode": "fused",
        "headroom_mode": "stay_below_current_peak",
        "current_layer_index": 3,
        "next_layer_index": 4,
        "candidates": [
            {
                "layer_index": 3,
                "family": "QKV",
                "state": "CONSUMED",
                "owner": "premat",
                "critical_path_miss": False,
            }
        ],
        "events": [{"sequence": update, "event": "consumed", "layer_index": 3}],
        "memory": {"global_buffer_bytes": 1024 ** 3},
        "aggregate": {"available_hits": update},
    }


def test_premat_snapshot_history_is_bounded_and_served(tmp_path: Path) -> None:
    database = tmp_path / "charts.sqlite3"
    store = LocalChartStore(database, run_name="premat-test", config={"max_updates": 140})
    try:
        for update in range(1, 141):
            store.append_premat_snapshot(update, _snapshot(update))
        status = LocalChartReader(database).status()
        assert status["premat_snapshot_count"] == 128
        assert status["premat_minimum_update"] == 13
        assert status["premat_maximum_update"] == 140
        payload = RunDashboardState(database).premat()
        assert payload["snapshot_count"] == 128
        assert payload["latest"]["optimizer_update"] == 140
        assert payload["latest"]["next_layer_index"] == 4
    finally:
        store.close(final_state="finished")


def test_premat_view_declares_every_required_lifecycle_visual() -> None:
    asset_root = Path(__file__).resolve().parents[1] / "sheet" / "local_dashboard_assets"
    css = (asset_root / "dashboard_premat.css").read_text(encoding="utf-8")
    javascript = (asset_root / "dashboard_premat.js").read_text(encoding="utf-8")
    for state in ("materialising", "available", "consuming", "consumed", "critical-path-miss"):
        assert f".{state}" in css
    for stage in (
        "QKV",
        "SDPA",
        "QK",
        "score",
        "scale / mask",
        "softmax",
        "V",
        "probabilities × V",
        "O",
        "UP",
        "GELU",
        "DOWN",
    ):
        assert stage in javascript
    assert "premat-neutral" in javascript
    assert "premat_event_filter" in javascript
    for field in (
        "Process reserved",
        "Device free",
        "Active ceiling",
        "Premat headroom",
        "Next candidate",
        "predicted_foreground_overlap_bytes",
    ):
        assert field in javascript


def test_premat_is_in_the_actual_instra_run_detail_tab_strip() -> None:
    asset_root = Path(__file__).resolve().parents[1] / "sheet" / "local_dashboard_assets"
    navigation = (asset_root / "dashboard_heatmap_patch.js").read_text(encoding="utf-8")
    index = (asset_root / "index.html").read_text(encoding="utf-8")
    assert '["charts", "premat", "overview", "logs", "files", "artifacts"]' in navigation
    assert 'local_active_detail_tab === "premat"' in navigation
    assert 'id="premat_chart_group"' in index


def test_premat_aggregate_persists_without_detailed_snapshot(tmp_path: Path) -> None:
    database = tmp_path / "charts.sqlite3"
    store = LocalChartStore(database, run_name="premat-aggregate", config={})
    try:
        snapshot = _snapshot(7)
        snapshot["aggregate"]["main_stream_wait_ms_total"] = 1.25
        store.update_premat_aggregate(7, snapshot)
        metadata = LocalChartReader(database).metadata()
        assert metadata["premat_last_update"] == "7"
        assert json.loads(metadata["premat_aggregate_json"])["main_stream_wait_ms_total"] == 1.25
        assert LocalChartReader(database).status()["premat_snapshot_count"] == 0
    finally:
        store.close(final_state="finished")


def test_premat_javascript_renders_live_pair_and_forensic_fields() -> None:
    asset = Path(__file__).resolve().parents[1] / "sheet" / "local_dashboard_assets" / "dashboard_premat.js"
    snapshot = _snapshot(9)
    snapshot.update(
        {
            "attention_mode": "unfused",
            "queue_head": {
                "layer_index": 4,
                "family": "V",
                "deferred": True,
                "admission_reason": "global_device_buffer",
            },
            "memory": {
                "process_allocated_bytes": 10,
                "process_reserved_bytes": 12,
                "process_ordinary_peak_bytes": 14,
                "device_free_bytes": 20,
                "device_used_bytes": 30,
                "active_ceiling_bytes": 40,
                "active_ceiling_scope": "process_ordinary_peak",
                "global_buffer_bytes": 5,
                "premat_headroom_bytes": 4,
                "premat_retained_bytes": 3,
            },
            "events": [
                {
                    "sequence": 9,
                    "elapsed_ms": 1.5,
                    "layer_index": 4,
                    "family": "V",
                    "old_state": "MATERIALISING",
                    "new_state": "CONSUMING",
                    "headroom_policy": "stay_below_current_peak",
                    "process_allocated_bytes": 10,
                    "process_ordinary_peak_bytes": 14,
                    "device_free_bytes": 20,
                    "device_used_bytes": 30,
                    "global_buffer_bytes": 5,
                    "premat_headroom_bytes": 4,
                    "predicted_retained_bytes": 3,
                    "predicted_materialisation_peak_bytes": 3,
                    "predicted_foreground_overlap_bytes": 7,
                    "predicted_envelope_bytes": 10,
                    "outcome": "waited_for_premat",
                    "reason": "materialising_at_deadline",
                    "wait_ms": 0.25,
                    "critical_path_miss": True,
                }
            ],
        }
    )
    harness = f"""
const elements = new Map();
function element(id) {{
  if (!elements.has(id)) elements.set(id, {{
    id, hidden: false, innerHTML: "", textContent: "", value: "",
    classList: {{toggle() {{}}, add() {{}}, remove() {{}}}},
    setAttribute() {{}}, addEventListener() {{}}
  }});
  return elements.get(id);
}}
global.by_id = element;
global.document = {{getElementById: element, querySelector: () => null, querySelectorAll: () => []}};
global.window = {{addEventListener() {{}}}};
global.app = {{current_run_id: null}};
global.fetch_json = async () => ({{}});
{asset.read_text(encoding="utf-8")}
render_premat({json.dumps({"latest": snapshot})});
console.log(JSON.stringify({{
  layers: element("premat_layers").innerHTML,
  memory: element("premat_memory").innerHTML,
  events: element("premat_events_body").innerHTML,
}}));
"""
    completed = subprocess.run(
        ("node", "-e", harness),
        text=True,
        capture_output=True,
        check=True,
    )
    rendered = json.loads(completed.stdout)
    assert rendered["layers"].index("lookahead · l+1") < rendered["layers"].index("current · l")
    assert "premat-neutral" in rendered["layers"]
    assert "probabilities × V" in rendered["layers"]
    assert "Process reserved" in rendered["memory"]
    assert "global_device_buffer" in rendered["memory"]
    assert "MATERIALISING → CONSUMING" in rendered["events"]
    assert "0.250 ms · MISS" in rendered["events"]
# ^^^ THOG
