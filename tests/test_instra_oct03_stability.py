# vvv THOG regress reader lifetime, immutable discovery, busy databases, Recipes and LibreOffice-compatible BIFF8 exports
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sheet.local_chart_store import LocalChartReader
from sheet.local_dashboard_chart_exports import workbook_bytes, validate_export
from sheet.local_dashboard_responsiveness import install as install_read_cache
from sheet.local_dashboard_wandb_charts_patch import _ScannerCatalog, _WandbRunScanner


def test_status_cache_is_single_flight_isolated_and_invalidated(tmp_path, monkeypatch):
    path = tmp_path / "charts.sqlite3"
    path.write_bytes(b"first")
    calls = []

    class State:
        def __init__(self):
            self.database_path = path

        def status(self):
            calls.append(1)
            time.sleep(.01)
            return {"configuration": {"value": path.read_bytes().decode()}}

    monkeypatch.setattr("sheet.local_dashboard_logs_patch._runner_attempt_log", lambda state: None)
    install_read_cache(SimpleNamespace(RunDashboardState=State))
    state = State()
    with ThreadPoolExecutor(max_workers=16) as workers:
        results = list(workers.map(lambda _: state.status(), range(200)))
    assert len(calls) == 1
    results[0]["configuration"]["value"] = "caller edit"
    assert state.status()["configuration"]["value"] == "first"
    path.write_bytes(b"second")
    assert state.status()["configuration"]["value"] == "second"
    (tmp_path / "train.log").write_text("new console status")
    state.status()
    assert len(calls) == 3


def test_busy_chart_reader_releases_http_worker_within_one_second(tmp_path):
    path = tmp_path / "locked.sqlite3"
    writer = sqlite3.connect(path)
    writer.execute("CREATE TABLE data (value INTEGER)")
    writer.commit()
    writer.execute("BEGIN EXCLUSIVE")
    reader = LocalChartReader(path)._connection()
    started = time.monotonic()
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            reader.execute("SELECT * FROM data").fetchall()
        assert time.monotonic() - started < 1
    finally:
        reader.close()
        writer.rollback()
        writer.close()


def test_retirement_closes_file_and_cannot_restart_background_reading(tmp_path):
    scanner = _WandbRunScanner(tmp_path / "unused.wandb")
    handle = io.BytesIO(b"history")
    scanner.store = SimpleNamespace(_fp=handle)
    scanner._open = Mock(side_effect=AssertionError("retired reader reopened"))
    scanner.close()
    scanner.refresh()
    assert handle.closed
    assert scanner.retired and not scanner.catching_up
    scanner._open.assert_not_called()


def test_evictions_and_deletions_retire_scanners(tmp_path):
    catalogue = _ScannerCatalog(SimpleNamespace(root=tmp_path))
    catalogue._find_path = lambda run_id, status: tmp_path / (run_id + ".wandb")
    readers = []
    for index in range(80):
        readers.append(catalogue.scanner_for(SimpleNamespace(status=lambda i=index: {"wandb_run_id": str(i)})))
    assert len(catalogue.scanners) == 16
    assert all(scanner.retired for scanner in readers[:-16])
    assert all(not scanner.retired for scanner in readers[-16:])
    catalogue.paths["remote:79"] = readers[-1].path
    catalogue.forget_runs(["remote:79"])
    assert readers[-1].retired


def test_background_workers_and_pending_histories_stay_bounded():
    from sheet.local_dashboard_wandb_catchup_patch import install

    class Scanner:
        def __init__(self):
            self.catching_up = True
            self.retired = False
            self.bursts = 0

        def refresh(self):
            time.sleep(.001)
            self.bursts += 1
            self.catching_up = self.bursts < 3

    install(SimpleNamespace(_WandbRunScanner=Scanner))
    workers = Scanner._thog2_background_workers
    for _ in range(500):
        Scanner().retired = True
    pending = Scanner._thog2_background_pending
    assert len(workers) == 2 and all(worker.is_alive() for worker in workers)
    assert pending.qsize() <= 16
    deadline = time.monotonic() + 2
    while pending.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(.01)
    assert pending.unfinished_tasks == 0


def test_discovery_cache_invalidates_new_records_and_preserves_live_revision(tmp_path):
    catalogue = _ScannerCatalog(SimpleNamespace(root=tmp_path))
    wandb = tmp_path / "run.wandb"
    wandb.write_bytes(b"first")
    db = tmp_path / "charts.sqlite3"
    db.touch()
    state = SimpleNamespace(database_path=db, status=lambda: {"wandb_run_id": "fixture"})
    catalogue._find_path = lambda run_id, status: wandb
    common = {"available": True, "catching_up": False, "error": "", "record_count": 10}
    groups = [{"name": "train", "chart_count": 1, "revision": 10}]
    before = catalogue._discovery_key(state, 3)
    catalogue.remember_discovery(state, 3, common, groups, before)
    assert catalogue.cached_discovery(state, 3) == (common, groups)
    assert catalogue.cached_discovery(state, 4) is None
    wandb.write_bytes(b"new record appended")
    assert catalogue.cached_discovery(state, 3) is None
    catalogue.remember_discovery(state, 3, common, groups, before)
    assert catalogue.cached_discovery(state, 3) is None  # changed during a read
    catalogue.remember_discovery(state, 3, {**common, "catching_up": True}, groups)
    assert catalogue.cached_discovery(state, 3) is None


def test_partial_wandb_record_waits_for_a_writer_instead_of_spinning():
    from sheet.local_dashboard_wandb_catchup_patch import install

    class PartialScanner:
        def __init__(self):
            self.good_offset = 0
            self.catching_up = True
            self.retired = False
            self.bursts = 0

        def refresh(self):
            self.bursts += 1

    install(SimpleNamespace(_WandbRunScanner=PartialScanner))
    scanners = [PartialScanner(), PartialScanner()]
    time.sleep(.6)
    for scanner in scanners:
        scanner.retired = True
        assert 1 <= scanner.bursts <= 4


def test_edited_recipes_create_numbered_copies_and_preserve_launched_history(tmp_path, monkeypatch):
    import instra_runner as runner
    import instra_network as network
    from tests.test_instra_runner_unittest import FakeNetwork

    for attribute, value in {"STATE_DIR": tmp_path, "STATE_PATH": tmp_path / "runner.json",
                             "LEASE_PATH": tmp_path / "lease", "GRID_SCRIPTS": tmp_path / "scripts"}.items():
        monkeypatch.setattr(runner, attribute, value)
    monkeypatch.setattr(network, "log_event", lambda *args: None)
    service = runner.RunnerService(FakeNetwork(), start_worker=False)
    try:
        recipe = {"label": "Original", "parameters": {"--max-iters": 2, "--warmup-iters": 0,
                  "--n-layer": 2, "--n-embd": 64, "--n-head": 4, "--batch-size": 1, "--block-size": 32}}
        original = service.save_recipe(None, recipe)
        grid = service.launch(original["recipe_id"])
        edited = json.loads(json.dumps(recipe))
        edited["parameters"]["--max-iters"] = 3
        first = service.save_recipe(original["recipe_id"], edited)
        second = service.save_recipe(original["recipe_id"], edited)
        assert first["recipe"]["label"] == "Copy (1) of Original"
        assert second["recipe"]["label"] == "Copy (2) of Original"
        assert first["recipe_id"] != original["recipe_id"]
        state = runner._read()
        assert state["recipes"][original["recipe_id"]]["recipe"] == recipe
        assert state["grids"][0]["recipe"] == recipe
        assert state["grids"][0]["grid_id"] == grid["grid_id"]
        assert service.save_recipe(original["recipe_id"], recipe)["recipe_id"] == original["recipe_id"]
        long = {**recipe, "label": "L" * 120}
        saved = service.save_recipe(None, long)
        copy_one = service.save_recipe(saved["recipe_id"], {**long, "parameters": edited["parameters"]})
        copy_two = service.save_recipe(saved["recipe_id"], {**long, "parameters": edited["parameters"]})
        assert len(copy_one["recipe"]["label"]) == 120
        assert copy_two["recipe"]["label"].startswith("Copy (2) of ")
    finally:
        service.close()


def export_payload(count=3):
    return {"schema": "instra.visible_curves.v1", "metric": "training_loss", "selected_run_name": "α_run",
            "series": [{"run_id": "a", "run_name": "α_run", "name": "Loss", "colour": "#123456",
                        "x": list(range(count)), "y": [index / 10 for index in range(count)],
                        "displayed_x": list(range(count)), "x_variants": {"step": list(range(count))}}]}


def test_genuine_xls_roundtrip_preserves_numbers_unicode_and_splits_large_curves():
    import xlrd
    data = workbook_bytes(export_payload(65540))
    assert data[:8] == bytes.fromhex("d0cf11e0a1b11ae1")
    book = xlrd.open_workbook(file_contents=data)
    first, second = book.sheet_by_name("curve_001_01"), book.sheet_by_name("curve_001_02")
    assert first.nrows == 65536 and second.nrows == 6
    assert first.cell_value(1, 2) == "α_run"
    assert first.cell_value(2, 5) == .1
    assert second.cell_value(1, 0) == 65535
    assert second.cell_value(5, 4) == 65539
    assert first.row_values(0)[-1] == "step"


def test_export_rejects_mismatched_axes_and_overlarge_requests():
    payload = export_payload()
    payload["series"][0]["x"].pop()
    with pytest.raises(ValueError, match="lengths"):
        validate_export(payload)
    with pytest.raises(ValueError, match="schema"):
        validate_export({})
# ^^^ THOG
