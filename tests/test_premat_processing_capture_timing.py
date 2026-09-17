# vvv THOG
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
# ^^^ THOG
