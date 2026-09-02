# vvv THOG
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from sheet.local_dashboard_wandb_charts_patch import _WandbRunScanner, _downsample_points


def _item(key: str, value: object) -> SimpleNamespace:
    return SimpleNamespace(key=key, nested_key=(), value_json=json.dumps(value))


def _history(step: int, timestamp: float, runtime: float, loss: float) -> SimpleNamespace:
    return SimpleNamespace(
        step=SimpleNamespace(num=step),
        item=(
            _item("_step", step),
            _item("_timestamp", timestamp),
            _item("_runtime", runtime),
            _item("train/loss", loss),
        ),
    )


def _stats(timestamp: float, gpu_usage: float) -> SimpleNamespace:
    seconds = int(timestamp)
    nanos = int(round((timestamp - seconds) * 1_000_000_000))
    return SimpleNamespace(
        timestamp=SimpleNamespace(seconds=seconds, nanos=nanos),
        item=(_item("gpu.0.gpu", gpu_usage),),
    )


def test_history_payload_exposes_step_and_three_real_time_axes() -> None:
    scanner = _WandbRunScanner(Path("unused.wandb"))
    scanner._consume_history(_history(10, 1_000.0, 20.0, 5.0))
    scanner._consume_history(_history(20, 1_060.0, 80.0, 4.0))

    chart = scanner.group_payload("train")["charts"][0]
    series = chart["series"][0]

    assert chart["default_x_axis_mode"] == "step"
    assert chart["available_x_axis_modes"] == [
        "step",
        "relative_wall",
        "relative_process",
        "wall_time",
    ]
    assert series["x_variants"] == {
        "step": [10.0, 20.0],
        "relative_wall": [0.0, 60.0],
        "relative_process": [20.0, 80.0],
        "wall_time": [1_000.0, 1_060.0],
    }


def test_system_payload_uses_real_timestamps_and_process_anchor() -> None:
    scanner = _WandbRunScanner(Path("unused.wandb"))
    scanner._consume_history(_history(10, 1_000.0, 20.0, 5.0))
    scanner._consume_stats(_stats(1_010.0, 60.0))
    scanner._consume_stats(_stats(1_070.0, 70.0))

    chart = scanner.group_payload("system")["charts"][0]
    series = chart["series"][0]

    assert chart["default_x_axis_mode"] == "relative_wall"
    assert chart["available_x_axis_modes"] == [
        "relative_wall",
        "relative_process",
        "wall_time",
    ]
    assert series["x_variants"]["relative_wall"] == [0.0, 60.0]
    assert series["x_variants"]["relative_process"] == [30.0, 90.0]
    assert series["x_variants"]["wall_time"] == [1_010.0, 1_070.0]


def test_downsampling_keeps_every_x_axis_aligned_with_y() -> None:
    points = [
        (float(index), float(index * 10), float(index), float(index + 100), float(index + 200), float(index + 300))
        for index in range(20)
    ]

    selected = _downsample_points(points, limit=7)

    assert 2 <= len(selected) <= 7
    assert selected[0] == points[0]
    assert selected[-1] == points[-1]
    for x_value, y_value, step, relative_wall, relative_process, wall_time in selected:
        assert y_value == x_value * 10
        assert step == x_value
        assert relative_wall == x_value + 100
        assert relative_process == x_value + 200
        assert wall_time == x_value + 300
# ^^^ THOG



def test_system_source_discovery_uses_recorded_sdk_directory(tmp_path, monkeypatch):
    from sheet.local_dashboard_wandb_charts_patch import _ScannerCatalog
    unrelated = tmp_path / "viewer"; unrelated.mkdir(); monkeypatch.chdir(unrelated)
    directory = tmp_path / "elsewhere" / "run-260903-test123"; (directory / "files").mkdir(parents=True)
    source = directory / "run-test123.wandb"; source.write_bytes(b"placeholder")
    catalog = _ScannerCatalog(SimpleNamespace(root=unrelated / "logs"))
    assert catalog._find_path("test123", {"wandb_run_directory":str(directory / "files")}) == source


def test_system_source_discovery_custom_root_offline_and_late_file(tmp_path, monkeypatch):
    from sheet.local_dashboard_wandb_charts_patch import _ScannerCatalog
    unrelated = tmp_path / "viewer"; unrelated.mkdir(); monkeypatch.chdir(unrelated)
    project = tmp_path / "project"; (project / "logs").mkdir(parents=True)
    catalog = _ScannerCatalog(SimpleNamespace(root=project / "logs"))
    status = {"configuration":{"wandb_root":"monitoring"}}
    assert catalog._find_path("test123",status) is None
    directory = project / "monitoring" / "wandb" / "offline-run-260903-test123"; directory.mkdir(parents=True)
    source = directory / "run-test123.wandb"; source.write_bytes(b"placeholder")
    assert catalog._find_path("test123",status) == source
