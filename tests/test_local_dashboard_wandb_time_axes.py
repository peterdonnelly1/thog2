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
        "relative_process": [0.0, 60.0],                                                                                                                   # <<< THOG process-relative axes begin at the first recorded process sample
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
        "step",
        "relative_wall",
        "relative_process",
        "wall_time",
    ]
    assert series["x_variants"]["step"] == [10.0, 10.0]                                                                                                  # <<< THOG system samples expose the latest optimizer step
    assert series["x_variants"]["relative_wall"] == [10.0, 70.0]
    assert series["x_variants"]["relative_process"] == [10.0, 70.0]
    assert series["x_variants"]["wall_time"] == [1_010.0, 1_070.0]


# vvv THOG W&B may emit system samples before the first loss/history record
def test_system_process_axis_remains_complete_when_stats_arrive_first() -> None:
    scanner = _WandbRunScanner(Path("unused.wandb"))
    scanner._consume_stats(_stats(990.0, 50.0))
    scanner._consume_stats(_stats(995.0, 55.0))
    scanner._consume_history(_history(10, 1_000.0, 20.0, 5.0))
    scanner._consume_stats(_stats(1_010.0, 60.0))

    chart = scanner.group_payload("system")["charts"][0]
    assert "relative_process" in chart["available_x_axis_modes"]
    assert chart["series"][0]["x_variants"]["relative_process"] == [0.0, 5.0, 20.0]
# ^^^ THOG


# vvv THOG memory grouping keeps only the assigned process and builds a monotonic peak
def test_memory_group_is_process_specific_gpu_filtered_and_monotonic() -> None:
    scanner = _WandbRunScanner(Path("unused.wandb"), gpu_index=1)
    scanner._consume_history(_history(10, 1_000.0, 20.0, 5.0))

    def stats(timestamp: float, allocated: float) -> SimpleNamespace:
        seconds = int(timestamp)
        return SimpleNamespace(
            timestamp=SimpleNamespace(seconds=seconds, nanos=0),
            item=(
                _item("gpu.0.gpu", 10.0),
                _item("gpu.1.gpu", 20.0),
                _item("gpu.1.memoryClock", 3000.0),
                _item("gpu.1.memoryAllocatedBytes", allocated * 2),
                _item("gpu.process.1.memoryAllocatedBytes", allocated),
            ),
        )

    scanner._consume_stats(stats(1_010.0, float(4 * 1024 ** 3)))
    scanner._consume_stats(stats(1_020.0, float(3 * 1024 ** 3)))

    system = scanner.group_payload("system")
    assert len(system["charts"]) == 2
    assert {chart["series"][0]["name"] for chart in system["charts"]} == {"GPU 1"}
    assert {chart["id"] for chart in system["charts"]} == {"gpu.gpu", "gpu.memoryClock"}
    memory = scanner.group_payload("memory")
    peak = next(chart for chart in memory["charts"] if chart["id"] == "gpu.process.peak_memory_allocated_gb")
    assert peak["series"][0]["y"] == [4.0, 4.0]
# ^^^ THOG


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


# vvv THOG older scruffy runs predate explicit CUDA_VISIBLE_DEVICES metadata but have an unambiguous single physical GPU
def test_scanner_recovers_scruffy_gpu_zero_from_host_label(tmp_path):
    from sheet.local_dashboard_wandb_charts_patch import _ScannerCatalog
    source = tmp_path / "run-old.wandb"
    source.write_bytes(b"placeholder")
    catalog = _ScannerCatalog(SimpleNamespace(root=tmp_path / "logs"))
    catalog.paths["old"] = source
    state = SimpleNamespace(status=lambda: {
        "wandb_run_id": "old",
        "host_label": "scruffy",
        "configuration": {},
    })
    assert catalog.scanner_for(state).gpu_index == 0
# ^^^ THOG
