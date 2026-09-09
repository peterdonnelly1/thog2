# vvv THOG Premat Instra completed-microstep persistence and playback regression tests
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from run_thog2_local_dashboard import RunDashboardState
from sheet.local_chart_store import LocalChartReader, LocalChartStore, LocalPrematLiveWriter


ASSET_ROOT = Path(__file__).resolve().parents[1] / "sheet" / "local_dashboard_assets"


def _snapshot(update: int) -> dict:
    return {
        "version": 1,
        "optimizer_update": update,
        "pass_sequence": update,
        "pass_complete": True,
        "attention_mode": "fused",
        "headroom_mode": "stay_below_current_peak",
        "layer_indices": [0, 1],
        "events": [
            {"sequence": update, "event": "pass_end", "pass_sequence": update}
        ],
        "memory": {"global_buffer_bytes": 1024**3},
        "aggregate": {"available_hits": update},
    }


def _playback_snapshot() -> dict:
    snapshot = _snapshot(10)
    snapshot["latest_event_sequence"] = 19
    snapshot["events"] = [
        {"sequence": 1, "event": "pass_begin"},
        {
            "sequence": 2, "event": "materialising", "layer_index": 0,
            "current_layer_index": 0, "family": "QKV", "new_state": "MATERIALISING",
            "owner": "premat", "decision": "admit", "admission_reason": "admitted",
            "materialisation_ms": 0.4, "predicted_retained_bytes": 12 * 1024**2,
        },
        {
            "sequence": 3, "event": "available", "layer_index": 0,
            "current_layer_index": 0, "family": "QKV", "new_state": "AVAILABLE",
            "owner": "premat", "admission_reason": "admitted",
        },
        {
            "sequence": 4, "event": "deadline_qkv", "layer_index": 0,
            "current_layer_index": 0, "family": "QKV", "new_state": "AVAILABLE",
            "owner": "premat", "outcome": "deadline",
        },
        {
            "sequence": 5, "event": "consuming", "layer_index": 0,
            "current_layer_index": 0, "family": "QKV", "new_state": "CONSUMING",
            "owner": "premat", "outcome": "fully_hidden",
        },
        {
            "sequence": 6, "event": "consumed", "layer_index": 0,
            "current_layer_index": 0, "family": "QKV", "new_state": "CONSUMED",
            "owner": "premat",
        },
        {
            "sequence": 7, "event": "materialising", "layer_index": 0,
            "current_layer_index": 0, "family": "O", "new_state": "MATERIALISING",
            "owner": "premat", "decision": "admit", "admission_reason": "admitted",
            "predicted_retained_bytes": 4 * 1024**2,
        },
        {
            "sequence": 8, "event": "deadline_o", "layer_index": 0,
            "current_layer_index": 0, "family": "O", "new_state": "MATERIALISING",
            "owner": "premat", "outcome": "deadline",
        },
        {
            "sequence": 9, "event": "critical_path_wait", "layer_index": 0,
            "current_layer_index": 0, "family": "O", "new_state": "CONSUMING",
            "owner": "premat", "outcome": "waited_for_premat",
            "reason": "materialising_at_deadline", "critical_path_miss": True,
            "wait_ms": 0.25,
        },
        {
            "sequence": 10, "event": "consumed", "layer_index": 0,
            "current_layer_index": 0, "family": "O", "new_state": "CONSUMED",
            "owner": "premat", "critical_path_miss": True,
        },
        {
            "sequence": 11, "event": "admission_deferred", "layer_index": 0,
            "current_layer_index": 0, "family": "UP", "new_state": "UNAVAILABLE",
            "owner": "none", "decision": "defer", "admission_reason": "global_device_buffer",
        },
        {
            "sequence": 12, "event": "deadline_up", "layer_index": 0,
            "current_layer_index": 0, "family": "UP", "new_state": "UNAVAILABLE",
            "owner": "none", "outcome": "deadline",
        },
        {
            "sequence": 13, "event": "materialising_on_critical_path", "layer_index": 0,
            "current_layer_index": 0, "family": "UP", "new_state": "MATERIALISING",
            "owner": "main", "decision": "main_claim", "admission_reason": "global_device_buffer",
            "main_stream_materialisation_ms": 0.55,
            "predicted_retained_bytes": 16 * 1024**2,
        },
        {
            "sequence": 14, "event": "available_on_critical_path", "layer_index": 0,
            "current_layer_index": 0, "family": "UP", "new_state": "AVAILABLE",
            "owner": "main",
        },
        {
            "sequence": 15, "event": "consuming", "layer_index": 0,
            "current_layer_index": 0, "family": "UP", "new_state": "CONSUMING",
            "owner": "main", "outcome": "main_stream_fallback",
        },
        {
            "sequence": 16, "event": "consumed", "layer_index": 0,
            "current_layer_index": 0, "family": "UP", "new_state": "CONSUMED",
            "owner": "main",
        },
        {
            "sequence": 17, "event": "materialising", "layer_index": 1,
            "current_layer_index": 1, "family": "DOWN", "new_state": "MATERIALISING",
            "owner": "premat", "decision": "admit", "admission_reason": "admitted",
        },
        {
            "sequence": 18, "event": "pass_end_release", "layer_index": 1,
            "current_layer_index": 1, "family": "DOWN", "new_state": "MATERIALISING",
            "owner": "premat", "outcome": "released_at_pass_end",
        },
        {"sequence": 19, "event": "pass_end"},
    ]
    return snapshot


def _javascript_model(snapshot: dict) -> dict:
    asset = ASSET_ROOT / "dashboard_premat.js"
    harness = f"""
global.window = {{addEventListener() {{}}, innerHeight: 900}};
const view = require({json.dumps(str(asset))});
const model = view.premat_build_model({json.dumps(snapshot)});
console.log(JSON.stringify({{
  attention_mode: model.attention_mode,
  layers: model.layers,
  families: model.families,
  records: [...model.records.values()],
  frames: model.frames,
}}));
"""
    completed = subprocess.run(
        ("node", "-e", harness),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def test_premat_snapshot_history_is_bounded_and_served_incrementally(tmp_path: Path) -> None:
    database = tmp_path / "charts.sqlite3"
    store = LocalChartStore(database, run_name="premat-test", config={"max_updates": 140})
    try:
        for update in range(1, 141):
            store.append_premat_snapshot(update, _snapshot(update))
        reader = LocalChartReader(database)
        status = reader.status()
        assert status["premat_snapshot_count"] == 128
        assert status["premat_minimum_update"] == 13
        assert status["premat_maximum_update"] == 140
        state = RunDashboardState(database)
        payload = state.premat()
        assert payload["snapshot_count"] == 128
        assert payload["latest"]["optimizer_update"] == 140
        unchanged = state.premat(after_update=140)
        assert unchanged["latest"] is None
        assert unchanged["latest_update"] == 140
        assert unchanged["unchanged"] is True
    finally:
        store.close(final_state="finished")


def test_live_writer_replaces_the_active_update_without_growing_history(tmp_path: Path) -> None:
    database = tmp_path / "charts.sqlite3"
    store = LocalChartStore(database, run_name="premat-live", config={})
    writer = LocalPrematLiveWriter(database)
    try:
        writer.append(3, _snapshot(1))
        writer.append(3, _snapshot(2))
        snapshots = LocalChartReader(database).premat_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0]["optimizer_update"] == 3
        assert snapshots[0]["pass_sequence"] == 2
    finally:
        writer.close()
        store.close(final_state="finished")


def test_playback_reducer_preserves_processing_and_outcome_paths() -> None:
    rendered = _javascript_model(_playback_snapshot())
    records = {(record["layer_index"], record["family"]): record for record in rendered["records"]}
    assert rendered["layers"] == [0, 1]
    assert len(records) == 8
    assert records[(0, "QKV")]["trace"] == ["PRE-MATERIALISING", "AVAILABLE", "CONSUMING - NO WAITING", "FULL HIT"]
    assert records[(0, "QKV")]["outcome"] == "FULL HIT"
    assert records[(0, "QKV")]["retained_bytes"] == 12 * 1024**2
    assert records[(0, "O")]["trace"] == ["PRE-MATERIALISING", "WAITING FOR PRE-MATERIALISATION", "CONSUMING AFTER WAIT", "PARTIAL HIT"]
    assert records[(0, "O")]["outcome"] == "PARTIAL HIT"
    assert records[(0, "UP")]["trace"] == ["PREMAT NOT STARTED - MAIN CODE MATERIALISING", "MAIN CODE CONSUMING", "COMPLETE MISS"]
    assert records[(0, "UP")]["outcome"] == "COMPLETE MISS"
    assert records[(1, "DOWN")]["trace"] == ["PRE-MATERIALISING", "INCOMPLETE PASS"]
    assert records[(1, "DOWN")]["outcome"] == "INCOMPLETE PASS"
    frame_states = [frame["frame_state"] for frame in rendered["frames"]]
    assert frame_states == [
        "PRE-MATERIALISING", "AVAILABLE", "CONSUMING - NO WAITING", "PRE-MATERIALISING",
        "WAITING FOR PRE-MATERIALISATION", "CONSUMING AFTER WAIT",
        "PREMAT NOT STARTED - MAIN CODE MATERIALISING", "MAIN CODE CONSUMING",
        "PRE-MATERIALISING", "COMPLETE",
    ]
    assert rendered["frames"][-1]["final"] is True


def test_premat_view_has_all_layer_playback_controls_complete_key_and_inspector() -> None:
    index = (ASSET_ROOT / "index.html").read_text(encoding="utf-8")
    css = (ASSET_ROOT / "dashboard_premat.css").read_text(encoding="utf-8")
    javascript = (ASSET_ROOT / "dashboard_premat.js").read_text(encoding="utf-8")
    for element_id in (
        "premat_step", "premat_play_toggle", "premat_state_duration",
        "premat_inspect_button", "premat_layers", "premat_inspector",
        "premat_inspector_body",
    ):
        assert f'id="{element_id}"' in index
    assert "Premat Recapitulation - Step" in index
    assert "flex: 0 0 210px" in css
    for state_class in (
        "premat-neutral", "premat-pending", "premat-state-materialising", "premat-state-available",
        "premat-state-consuming-full", "premat-state-consumed-full",
        "premat-state-waiting", "premat-state-consuming-waited",
        "premat-state-consumed-waited", "premat-state-main-materialising",
        "premat-state-main-consuming", "premat-state-consumed-main",
    ):
        assert f".{state_class}" in css
        assert state_class in index
    for label in (
        "PROCESSING", "OUTCOMES", "OUT OF SCOPE", "NOT YET REACHED",
        "PRE-MATERIALISING", "CONSUMING - NO WAITING",
        "WAITING FOR PRE-MATERIALISATION", "CONSUMING AFTER WAIT",
        "PREMAT NOT STARTED - MAIN CODE MATERIALISING", "MAIN CODE CONSUMING",
        "FULL HIT", "PARTIAL HIT", "COMPLETE MISS",
    ):
        assert label in index
    assert "TOO LATE" not in index
    assert "premat-matrix-size-row" in javascript
    assert "predicted_retained_bytes" in javascript
    for label in (
        "ATTN FUSED · QKV", "ATTN UNFUSED · QK", "ATTN UNFUSED · V",
        "ATTN O", "MLP UP", "MLP DN",
    ):
        assert label in javascript
    assert 'min="25" max="2000" step="25" value="250"' in index
    assert "state duration" in index
    assert "rule: do not cross global buffer - currently" in javascript
    assert "rule: stay below current peak memory" in javascript
    assert "PREMAT_FINAL_HOLD_MS = 1000" in javascript
    assert "setInterval(refresh_premat, 750)" in javascript
    assert "&after=${after}" in javascript


def test_premat_layer_rows_are_one_indexed_descending_and_shrink_with_a_floor() -> None:
    javascript = (ASSET_ROOT / "dashboard_premat.js").read_text(encoding="utf-8")
    css = (ASSET_ROOT / "dashboard_premat.css").read_text(encoding="utf-8")
    assert ".sort((left, right) => right - left)" in javascript
    assert "${layer_index + 1}" in javascript
    assert "Math.max(20, Math.min(42" in javascript
    assert "gap: 3px" in css
    assert "overflow: auto" in css


def test_premat_is_in_the_actual_instra_run_detail_tab_strip() -> None:
    navigation = (ASSET_ROOT / "dashboard_heatmap_patch.js").read_text(encoding="utf-8")
    index = (ASSET_ROOT / "index.html").read_text(encoding="utf-8")
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
# ^^^ THOG
