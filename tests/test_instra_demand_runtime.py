# vvv THOG regress more-than-sixteen histories, replaced mirror files, aggregate retention and one-chart weight construction
from collections import defaultdict
from http import HTTPStatus
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sheet.local_dashboard_wandb_charts_patch import _ScannerCatalog, _WandbRunScanner, _MAX_RETAINED_POINTS_PER_SCANNER
from sheet.local_chart_store import LocalChartStore, LocalChartReader
from sheet.local_dashboard_demand import install


def write_wandb(path, steps=100):
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2
    store = DataStore()
    store.open_for_write(str(path))
    for step in range(steps):
        record = wandb_internal_pb2.Record()
        record.history.step.num = step
        record.history.item.add(key="train/loss", value_json=str(5 - step / 1000))
        store.write(record)
    store.close()


def finish_scan(scanner):
    for _ in range(100):
        scanner.refresh()
        if not scanner.catching_up:
            return
    pytest.fail("scanner never reached EOF")


def test_twenty_four_warm_histories_resume_without_replaying_records(tmp_path):
    catalog = _ScannerCatalog(SimpleNamespace(root=tmp_path))
    catalog._find_path = lambda run_id, _status: tmp_path / (run_id + ".wandb")
    states = []
    for index in range(24):
        run_id = str(index)
        write_wandb(tmp_path / (run_id + ".wandb"), steps=200)
        states.append(SimpleNamespace(status=lambda run_id=run_id: {"wandb_run_id": run_id}))
    counts = []
    consume = _WandbRunScanner._consume_record
    def counted(scanner, record):
        counts.append(1)
        return consume(scanner, record)
    with patch.object(_WandbRunScanner, "_consume_record", counted):
        for state in states:
            finish_scan(catalog.scanner_for(state))
        assert len(counts) == 4800
        for _ in range(3):
            for state in states:
                scanner = catalog.scanner_for(state)
                finish_scan(scanner)
                assert scanner.record_count == 200
                assert scanner.series["train"]["train/loss\0train/loss"][-1][0] == 199
        assert len(counts) == 4800, "evicted histories were replayed from zero"
    assert len(catalog.scanners) <= 16
    assert len(catalog.checkpoints) <= 256


def test_replaced_remote_wandb_file_reopens_the_new_inode(tmp_path):
    path = tmp_path / "run.wandb"
    write_wandb(path, steps=200)
    scanner = _WandbRunScanner(path)
    finish_scan(scanner)
    replacement = tmp_path / "replacement.wandb"
    write_wandb(replacement, steps=250)
    replacement.replace(path)
    finish_scan(scanner)
    assert scanner.record_count == 250
    assert scanner.file_identity == (path.stat().st_dev, path.stat().st_ino)
    assert scanner.series["train"]["train/loss\0train/loss"][-1][0] == 249


def test_aggregate_samples_are_bounded_across_many_metrics(tmp_path):
    scanner = _WandbRunScanner(tmp_path / "unused.wandb")
    for step in range(1200):
        for metric in range(100):
            scanner._append("system", str(metric), str(metric), str(metric), step, 10000 if step == 731 else step % 17,
                            "step", step=step, relative_wall_seconds=step,
                            relative_process_seconds=step, wall_time_epoch_seconds=step, default_x_axis_mode="step")
    retained = sum(len(points) for group in scanner.series.values() for points in group.values())
    assert retained <= _MAX_RETAINED_POINTS_PER_SCANNER
    for points in scanner.series["system"].values():
        assert points[0][0] == 0 and points[-1][0] == 1199
        assert any(point[0] == 731 and point[1] == 10000 for point in points)


def test_unchanged_group_payload_survives_scanner_eviction(tmp_path):
    path = tmp_path / "run.wandb"
    path.write_bytes(b"first")
    state = SimpleNamespace(database_path=tmp_path / "charts.sqlite3", status=lambda: {"wandb_run_id": "a"})
    catalog = _ScannerCatalog(SimpleNamespace(root=tmp_path))
    catalog._find_path = lambda *_: path
    common = {"available": True, "catching_up": False, "error": ""}
    payload = {"name": "train", "charts": [{"series": [{"x": [1, 2], "y": [3, 4]}]}], "revision": 2}
    key = catalog._payload_key(state, "train")
    catalog.remember_payload(state, "train", common, payload, key)
    assert catalog.cached_payload(state, "train") == (common, payload)
    path.write_bytes(b"changed")
    assert catalog.cached_payload(state, "train") is None
    catalog.remember_payload(state, "train", {**common, "catching_up": True}, payload, key)
    assert catalog.cached_payload(state, "train") is None


def test_weight_endpoint_constructs_one_selected_chart_and_limits_snapshots(tmp_path):
    path = tmp_path / "charts.sqlite3"
    store = LocalChartStore(path, run_name="weights", run_id="weights", config={})
    for step in (10, 20, 30, 40):
        store.append_depth_weight_snapshot({"optimizer_update": step, "families": {
            name: {"curves": [{"scalar_id": "r0_c0", "x": [1], "y": [step]}]}
            for name in ("chart_a", "chart_b")}}, history_length=8)
    store.close()
    reader = LocalChartReader(path)
    state = SimpleNamespace(reader=reader, lock=threading.Lock(), database_path=path, status=reader.status)
    calls = []
    class Handler:
        def _send_json(self, value, status=HTTPStatus.OK):
            self.sent = (status, value)
        def do_GET(self):
            raise AssertionError("unexpected legacy request")
    def build(snapshots, chart_name):
        values = [snapshot["optimizer_update"] for snapshot in snapshots]
        calls.append((chart_name, values))
        return SimpleNamespace(to_plotly_json=lambda: {"steps": values})
    dashboard = SimpleNamespace(_handler_for=lambda _: Handler, HTTPStatus=HTTPStatus,
                                depth_curves=SimpleNamespace(_CHART_FAMILIES=("chart_a", "chart_b"), _build_depth_plotly_figure=build))
    install(dashboard)
    handler = dashboard._handler_for(SimpleNamespace(state_for_run=lambda _: state))()
    handler.path = "/api/weight-figure?run=weights&chart=chart_b&snapshots=2"
    handler.do_GET()
    assert handler.sent[0] == HTTPStatus.OK
    assert handler.sent[1]["depth"] == {"chart_b": {"steps": [30, 40]}}
    handler.do_GET()
    assert calls == [("chart_b", [30, 40])]
    handler.path = "/api/weight-figure?run=weights&chart=chart_a&current_only=1"
    handler.do_GET()
    assert calls[-1] == ("chart_a", [40])
    handler.path = "/api/weight-figure?run=weights&chart=chart_b&snapshots=2&window=from_zero"
    handler.do_GET()
    assert calls[-1] == ("chart_b", [10, 20])


def test_processing_discovery_is_stat_only_and_recognises_timing_only_capture(tmp_path):
    class Handler:
        def _send_json(self, value, status=HTTPStatus.OK):
            self.sent = (status, value)
    state = SimpleNamespace(database_path=tmp_path / "charts.sqlite3")
    dashboard = SimpleNamespace(_handler_for=lambda _: Handler, HTTPStatus=HTTPStatus)
    install(dashboard)
    handler = dashboard._handler_for(SimpleNamespace(state_for_run=lambda _: state))()
    handler.path = "/api/processing-status?run=timing"
    directory = tmp_path / "processing"
    directory.mkdir()
    (directory / "update_timing.json").write_text("No JSON parsing is required for discovery")
    handler.do_GET()
    assert handler.sent[0] == HTTPStatus.OK
    assert handler.sent[1]["timing_available"] is True
    assert handler.sent[1]["trace_available"] is False
    (directory / "processing_data.json").write_text("Not loaded here")
    handler.do_GET()
    assert handler.sent[1]["trace_available"] is True


def test_empty_sqlite_wal_does_not_invalidate_data_cache(tmp_path):
    from sheet.local_dashboard_responsiveness import file_signature
    path = tmp_path / "charts.sqlite3-wal"
    path.write_bytes(b"")
    assert file_signature(path) is None
    path.touch()
    assert file_signature(path) is None
    path.write_bytes(b"new records")
    signature = file_signature(path)
    assert signature is not None
    path.write_bytes(b"more records")
    assert file_signature(path) != signature
# ^^^ THOG
