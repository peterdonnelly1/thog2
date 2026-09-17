# vvv THOG
import json
import sys
from types import SimpleNamespace

import sheet.premat_processing as processing


def test_capture_elapsed_time_uses_active_capture_origin(monkeypatch):
    monkeypatch.setattr(processing, "_capture_active", True)
    monkeypatch.setattr(processing, "_capture_start_ns", 1_000_000_000)
    monkeypatch.setattr(processing.time, "perf_counter_ns", lambda: 1_002_500_000)

    assert processing.processing_capture_elapsed_ms() == 2.5


def test_capture_elapsed_time_is_absent_outside_capture(monkeypatch):
    monkeypatch.setattr(processing, "_capture_active", False)
    monkeypatch.setattr(processing, "_capture_start_ns", None)

    assert processing.processing_capture_elapsed_ms() is None


def test_capture_scope_persists_the_host_clock_origin(monkeypatch, tmp_path):
    metadata_path = tmp_path / "capture.json"
    clock = iter((1_000_000_000, 1_002_500_000))
    nvtx = SimpleNamespace(range_push=lambda _label: None, range_pop=lambda: None)
    cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda _device: None,
        nvtx=nvtx,
    )
    fake_torch = SimpleNamespace(
        cuda=cuda,
        device=lambda _device: SimpleNamespace(type="cuda"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv(processing._PROCESSING_CAPTURE_METADATA_ENV, str(metadata_path))
    monkeypatch.setattr(processing, "_capture_done", False)
    monkeypatch.setattr(processing.time, "perf_counter_ns", lambda: next(clock))

    with processing.processing_capture_scope(
        enabled=True,
        completed_updates=0,
        max_updates=1,
        log_interval=1,
        capture_update=1,
        micro_step=0,
        device="cuda:0",
    ):
        assert processing.processing_capture_elapsed_ms() == 2.5

    metadata = json.loads(metadata_path.read_text())
    assert metadata["host_start_ns"] == 1_000_000_000
    assert metadata["optimizer_update"] == 1
# ^^^ THOG
