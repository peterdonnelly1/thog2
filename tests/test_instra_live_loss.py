# vvv THOG
from types import SimpleNamespace
from sheet.local_dashboard_live_loss import LiveLossReader


def reader_for(tmp_path, text):
    path = tmp_path / "artifact" / "train.log"
    path.parent.mkdir()
    path.write_text(text)
    state = SimpleNamespace(status=lambda: {"artifact_name": "artifact"}, database_path=tmp_path / "charts.sqlite3")
    catalog = SimpleNamespace(root=tmp_path)
    dashboard = SimpleNamespace(_modified_time=lambda path: 0)
    reader = LiveLossReader()
    reader.refresh(catalog, state, dashboard)
    return reader, path, catalog, state, dashboard


def test_first_point_partial_append_and_no_validation_contamination(tmp_path):
    reader, path, catalog, state, dashboard = reader_for(tmp_path,
        "\x1b[32mT 10 00:00:04 loss  = 7.1234 lr=1e-3\x1b[0m\n"
        "V 10 00:00:04 training loss=6.1111 validation loss=7.2345\nT 20 loss=6.")
    assert reader.values == {"train": {10: 7.1234}, "val": {10: 7.2345}}
    revision = reader.revision
    reader.refresh(catalog, state, dashboard)
    assert reader.revision == revision
    with path.open("a") as handle:
        handle.write("9876\nT 30 loss=nan\nT 31 loss=1e309\n")
    reader.refresh(catalog, state, dashboard)
    assert reader.values["train"] == {10: 7.1234, 20: 6.9876}


def test_exact_wandb_replaces_printed_values_and_anchors_live_time_axes(tmp_path):
    reader, *_ = reader_for(tmp_path, "T 10 loss=7.1234\nT 20 loss=6.9876\nT 30 loss=6.5432\n")
    # vvv THOG make the provisional console-tail clock deterministic relative to the exact W&B anchor
    reader.wall_times["train"][30] = 103.0
    # ^^^ THOG
    exact = {"name": "train", "revision": 4, "charts": [{
        "id": "train/loss",
        "default_x_axis_mode": "step",
        "available_x_axis_modes": ["step", "relative_wall", "relative_process", "wall_time"],
        "series": [{
            "name": "train/loss",
            "x": [10, 20],
            "y": [7.12345678, 6.98765432],
            "x_variants": {
                "step": [10, 20],
                "relative_wall": [1, 2],
                "relative_process": [0.5, 1.5],
                "wall_time": [100, 101],
            },
        }],
    }]}
    chart = reader.merge(exact)["charts"][0]
    merged = chart["series"][0]
    assert merged["x"] == [10, 20, 30]
    assert merged["y"] == [7.12345678, 6.98765432, 6.5432]
    assert merged["point_sources"][-1] == "train.log (printed precision)"
    assert merged["x_variants"]["relative_wall"] == [1, 2, 4.0]
    assert merged["x_variants"]["relative_process"] == [0.5, 1.5, 3.5]
    assert merged["x_variants"]["wall_time"] == [100, 101, 103.0]
    assert chart["available_x_axis_modes"] == [
        "step", "relative_wall", "relative_process", "wall_time",
    ]
    assert chart["default_x_axis_mode"] == "step"
    assert exact["charts"][0]["series"][0]["x"] == [10, 20]
    exact["charts"][0]["series"][0].update(
        x=[10, 20, 30],
        y=[7.12345678, 6.98765432, 6.54321987],
    )
    assert reader.merge(exact)["charts"][0]["series"][0]["y"][-1] == 6.54321987


def test_live_only_chart_exposes_all_three_time_modes(tmp_path):
    reader, *_ = reader_for(tmp_path, "T 10 loss=7.1\nT 20 loss=6.9\n")
    # vvv THOG deterministic provisional timestamps prove relative, relative-process and wall-time live tails all advance
    reader.wall_times["train"] = {10: 100.0, 20: 105.0}
    reader.first_wall_time = 100.0
    # ^^^ THOG
    chart = reader.merge({"name": "train", "charts": [], "revision": 0})["charts"][0]
    series = chart["series"][0]
    assert chart["available_x_axis_modes"] == [
        "step", "relative_wall", "relative_process", "wall_time",
    ]
    assert series["x_variants"]["step"] == [10, 20]
    assert series["x_variants"]["relative_wall"] == [0.0, 5.0]
    assert series["x_variants"]["relative_process"] == [0.0, 5.0]
    assert series["x_variants"]["wall_time"] == [100.0, 105.0]


def test_refresh_backfills_batch_wall_times_from_step_duration(tmp_path):
    reader, *_ = reader_for(
        tmp_path,
        "T 10 120926-1310 0010 Δstep=4.0s loss=7.1\n"
        "V 10 120926-1310 0010 Δstep=4.0s training loss=7.1 validation loss=7.2\n"
        "T 20 120926-1310 0020 Δstep=6.0s loss=6.9\n",
    )
    # vvv THOG newest file mtime is the anchor; only exact intra-batch Δstep differences are asserted
    newest = reader.wall_times["train"][20]
    assert reader.wall_times["val"][10] == newest - 6.0
    assert reader.wall_times["train"][10] == newest - 6.0
    # ^^^ THOG



def test_restart_large_log_drains_to_newest_progress_rows_in_one_refresh(tmp_path):
    # vvv THOG a restarted Instra must not timestamp an old 1-MiB prefix with the current train.log mtime
    noise = "not a progress row\n" * 70000
    reader, *_ = reader_for(
        tmp_path,
        noise
        + "T 10 120926-1400 0010 Δstep=4.0s loss=7.1\n"
        + "T 20 120926-1400 0020 Δstep=6.0s loss=6.9\n",
    )
    assert reader.values["train"] == {10: 7.1, 20: 6.9}
    assert reader.wall_times["train"][10] == reader.wall_times["train"][20] - 6.0
    # ^^^ THOG

def test_rotation_summary_and_bounded_memory(tmp_path):
    reader, path, catalog, state, dashboard = reader_for(tmp_path, "T 10 loss=7\n")
    assert reader.summaries([], None)[0]["name"] == "train"
    path.write_text("T 1 loss=5\n")
    reader.refresh(catalog, state, dashboard)
    assert reader.values["train"] == {1: 5}
    with path.open("a") as handle:
        for step in range(2, 4000):
            handle.write(f"T {step} loss=5\n")
    reader.refresh(catalog, state, dashboard)
    assert len(reader.values["train"]) == 3200
    assert len(reader.wall_times["train"]) == 3200
    assert max(reader.values["train"]) == 3999


def test_absent_log_is_optional(tmp_path):
    reader = LiveLossReader()
    state = SimpleNamespace(status=lambda: {"artifact_name": "missing"})
    reader.refresh(SimpleNamespace(root=tmp_path), state, SimpleNamespace())
    assert reader.summaries([], None) == []
    assert reader.merge({"name": "system", "charts": [], "revision": 0})["charts"] == []


def test_loss_chart_works_with_only_local_optimizer_metrics(tmp_path):
    from sheet.local_chart_store import LocalChartStore, LocalChartReader

    store = LocalChartStore(tmp_path / "artifact" / "charts.sqlite3", run_name="artifact", config={})
    store.append_training_loss(1, 7.12)
    store.append_training_loss(2, 6.34)
    store.append_processing_throughput(2, 1200)
    state = SimpleNamespace(status=lambda: {"artifact_name": "artifact"}, reader=LocalChartReader(store.path))
    reader = LiveLossReader()
    catalog = SimpleNamespace(root=tmp_path)
    reader.refresh(catalog, state, SimpleNamespace())
    assert state.reader.latest_recorded_loss() == 6.34
    assert reader.summaries([], None)[0]["chart_count"] == 1
    chart = reader.merge({"name": "train", "charts": [], "revision": 0})["charts"][0]
    assert chart["id"] == "train/loss"
    assert chart["series"][0]["x"] == [1, 2]
    assert chart["series"][0]["y"] == [7.12, 6.34]
    revision = reader.revision
    reader.refresh(catalog, state, SimpleNamespace())
    assert reader.revision == revision
    store.close()


def test_runner_attempt_log_restores_existing_run_loss_chart(tmp_path, monkeypatch):
    from sheet.local_chart_store import LocalChartStore, LocalChartReader
    from sheet.local_dashboard_logs_patch import _resolve_train_log

    attempt_id = "a" * 32
    state_directory = tmp_path / "instra-state"
    state_directory.mkdir()
    monkeypatch.setenv("INSTRA_STATE_DIR", str(state_directory))
    attempt_log = state_directory / f"attempt-{attempt_id}.log"
    attempt_log.write_text("Runner launching\nT 10 loss=7.1234\nT 20 loss=6.9876\n")
    store = LocalChartStore(tmp_path / "artifact" / "run-id" / "charts.sqlite3",
                            run_name="artifact", run_id="run-id",
                            config={"runner": {"attempt_id": attempt_id}})
    state = SimpleNamespace(status=lambda: {"artifact_name": "artifact"},
                            database_path=store.path, reader=LocalChartReader(store.path))
    catalog = SimpleNamespace(root=tmp_path)
    dashboard = SimpleNamespace(_modified_time=lambda path: 0)
    assert _resolve_train_log(catalog, state, dashboard) == attempt_log
    reader = LiveLossReader()
    reader.refresh(catalog, state, dashboard)
    chart = reader.merge({"name": "train", "charts": [], "revision": 0})["charts"][0]
    assert chart["id"] == "train/loss"
    assert chart["series"][0]["x"] == [10, 20]
    assert chart["series"][0]["y"] == [7.1234, 6.9876]
    with attempt_log.open("a") as handle:
        handle.write("T 30 loss=6.5432\n")
    reader.refresh(catalog, state, dashboard)
    assert reader.merge({"name": "train", "charts": [], "revision": 0})["charts"][0]["series"][0]["y"][-1] == 6.5432
    store.close()
# ^^^ THOG
