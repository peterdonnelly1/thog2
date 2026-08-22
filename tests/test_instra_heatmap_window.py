# vvv THOG
from __future__ import annotations

from pathlib import Path

from sheet import local_chart_store as store
from sheet import local_dashboard_heatmap_window_patch as heatmap_window_patch


def _probe(step: int) -> dict[str, object]:
    active = 8
    return {
        "optimizer_update": step,
        "probe_id": f"p{step}",
        "active_layers": active,
        "selected_layers": active,
        "shrink": ((1, -0.001 * (step + 1), active - 1, -1),),
        "growth": ((1, 0.001 * (step + 1), active + 1, 1),),
    }


def test_heatmap_history_window_decodes_only_requested_rows(tmp_path: Path, monkeypatch) -> None:
    heatmap_window_patch.install()
    database = tmp_path / "charts.sqlite3"
    writer = store.LocalChartStore(
        database,
        run_name="heatmap-window-test",
        run_id="heatmap-window-test",
        config={},
    )
    writer.append_heatmap_records((_probe(step) for step in range(2400)))
    writer.close()

    decode_calls = 0
    original_decode = store._decode_payload

    def counted_decode(payload: bytes) -> dict[str, object]:
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(payload)

    monkeypatch.setattr(store, "_decode_payload", counted_decode)
    reader = store.LocalChartReader(database)

    rolling = reader.heatmap_history_window(probe_count=100, window_mode="rolling")
    assert decode_calls == 100
    assert len(rolling) == 100
    assert rolling[0]["optimizer_update"] == 2300
    assert rolling[-1]["optimizer_update"] == 2399
    assert all(len(record["values"]) == 9 for record in rolling)

    decode_calls = 0
    from_zero = reader.heatmap_history_window(probe_count=37, window_mode="from_zero")
    assert decode_calls == 37
    assert len(from_zero) == 37
    assert from_zero[0]["optimizer_update"] == 0
    assert from_zero[-1]["optimizer_update"] == 36


def test_heatmap_history_window_validates_mode(tmp_path: Path) -> None:
    heatmap_window_patch.install()
    database = tmp_path / "charts.sqlite3"
    writer = store.LocalChartStore(
        database,
        run_name="heatmap-window-mode-test",
        run_id="heatmap-window-mode-test",
        config={},
    )
    writer.append_heatmap_records((_probe(step) for step in range(3)))
    writer.close()
    reader = store.LocalChartReader(database)

    try:
        reader.heatmap_history_window(probe_count=10, window_mode="sideways")
    except ValueError as error:
        assert "rolling" in str(error)
    else:
        raise AssertionError("invalid heatmap window mode was accepted")
# ^^^ THOG
