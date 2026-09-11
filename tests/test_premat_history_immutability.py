from __future__ import annotations

from sheet.local_chart_store import LocalChartReader, LocalChartStore, LocalPrematLiveWriter


def _snapshot(pass_sequence: int, marker: str, *, complete: bool) -> dict[str, object]:
    return {
        "pass_sequence": pass_sequence,
        "pass_complete": complete,
        "events": [{"event": "pass_end"}] if complete else [{"event": "consumed"}],
        "marker": marker,
        "aggregate": {},
    }


def test_first_complete_premat_pass_for_update_is_immutable(tmp_path) -> None:
    path = tmp_path / "charts.sqlite3"
    store = LocalChartStore(path, run_name="test-run", config={})
    store.append_premat_snapshot(47, _snapshot(1, "first", complete=True))
    store.append_premat_snapshot(47, _snapshot(2, "later", complete=True))
    rows = LocalChartReader(path).premat_snapshots()
    assert len(rows) == 1
    assert rows[0]["marker"] == "first"
    assert rows[0]["pass_sequence"] == 1
    store.close()


def test_incomplete_row_can_progress_until_first_complete_snapshot(tmp_path) -> None:
    path = tmp_path / "charts.sqlite3"
    store = LocalChartStore(path, run_name="test-run", config={})
    store.append_premat_snapshot(50, _snapshot(1, "partial", complete=False))
    store.append_premat_snapshot(50, _snapshot(1, "complete", complete=True))
    store.append_premat_snapshot(50, _snapshot(2, "must-not-replace", complete=True))
    row = LocalChartReader(path).latest_premat_snapshot()
    assert row is not None
    assert row["marker"] == "complete"
    assert row["pass_sequence"] == 1
    store.close()


def test_live_writer_cannot_replace_completed_history_row(tmp_path) -> None:
    path = tmp_path / "charts.sqlite3"
    store = LocalChartStore(path, run_name="test-run", config={})
    store.append_premat_snapshot(51, _snapshot(1, "first", complete=True))
    writer = LocalPrematLiveWriter(path)
    writer.append(51, _snapshot(2, "later", complete=True))
    writer.close()
    row = LocalChartReader(path).latest_premat_snapshot()
    assert row is not None
    assert row["marker"] == "first"
    store.close()
