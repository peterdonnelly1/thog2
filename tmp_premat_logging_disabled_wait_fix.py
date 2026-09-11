from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


premat_path = Path("sheet/premat.py")
premat = premat_path.read_text(encoding="utf-8")
premat = replace_once(
    premat,
    '''        payload = timing.event_payload\n        if payload is None:\n            raise RuntimeError("main-stream wait timing lost its event payload")\n        candidate = timing.candidate\n''',
    '''        payload = timing.event_payload\n        # vvv THOG keep GPU hit classification operational when PREMAT telemetry is disabled\n        # if payload is None:\n        #     raise RuntimeError("main-stream wait timing lost its event payload")\n        if payload is None:\n            payload = {}\n        # ^^^ THOG\n        candidate = timing.candidate\n''',
    "optional wait-classification telemetry payload",
)
premat_path.write_text(premat, encoding="utf-8")


test_path = Path("tests/test_premat.py")
tests = test_path.read_text(encoding="utf-8")
anchor = '''def test_sampled_live_report_waits_for_gpu_classification_without_sync(monkeypatch) -> None:\n'''
new_test = '''# vvv THOG regression: CUDA wait classification must not depend on event-history logging\ndef test_gpu_wait_classification_survives_disabled_premat_logging(monkeypatch) -> None:\n    runtime, _fake_cuda, _calls = _runtime(\n        monkeypatch,\n        stay_below_current_peak=False,\n    )\n    runtime._logging_enabled = False\n    runtime.layer_start(3)\n    runtime.layer_start(5)\n    runtime.acquire("DOWN", 5)\n    timing = next(item for item in runtime._pending_timings if item.kind == "main_stream_wait")\n    assert timing.event_payload is None\n    timing.start_event.elapsed_time_override = (\n        lambda other: -0.125 if other is timing.dependency_event else 0.010\n    )\n\n    runtime._resolve_pending_timings()\n\n    candidate = timing.candidate\n    assert candidate is not None\n    assert candidate.final_outcome == "FULL HIT"\n    assert candidate.critical_path_miss is False\n    assert runtime._aggregate["fully_hidden_hits"] == 1\n    assert runtime._aggregate["waited_hits"] == 0\n    assert runtime._aggregate["main_stream_wait_ms_total"] == pytest.approx(0.0)\n# ^^^ THOG\n\n\n'''
tests = replace_once(
    tests,
    anchor,
    new_test + anchor,
    "logging-disabled GPU wait regression insertion",
)
test_path.write_text(tests, encoding="utf-8")
