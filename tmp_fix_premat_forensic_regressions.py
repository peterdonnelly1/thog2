from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing anchor: {label}")
    return text.replace(old, new, 1)


model_path = Path("sheet/model.py")
model = model_path.read_text()
model = replace_once(
    model,
    '''    # vvv THOG bracket identical foreground matrix-use GEMMs so REAL/SHADOW comparison isolates side-stream contention
    def _premat_forensic_main_work_start(self, family: str, layer_index: int) -> None:
''',
    '''    # vvv THOG optional forensic markers tolerate checkpoint/runtime doubles while the real PREMAT runtime exposes every marker
    def _premat_forensic_layer_start(self, layer_index: int) -> None:
        runtime = self._premat_runtime
        marker = getattr(runtime, "forensic_layer_start", None)
        if runtime is not None and runtime.active and callable(marker):
            marker(layer_index)

    def _premat_forensic_layer_end(self, layer_index: int) -> None:
        runtime = self._premat_runtime
        marker = getattr(runtime, "forensic_layer_end", None)
        if runtime is not None and runtime.active and callable(marker):
            marker(layer_index)

    # ^^^ THOG
    # vvv THOG bracket identical foreground matrix-use GEMMs so REAL/SHADOW comparison isolates side-stream contention
    def _premat_forensic_main_work_start(self, family: str, layer_index: int) -> None:
''',
    "forensic layer helper methods",
)
model = replace_once(
    model,
    '''            self._premat_runtime.layer_start(layer_index)
            self._premat_runtime.forensic_layer_start(layer_index)                                                      # <<< THOG time Main Stream layer body only when explicit GPU diagnostic is enabled
''',
    '''            self._premat_runtime.layer_start(layer_index)
            self._premat_forensic_layer_start(layer_index)                                                             # <<< THOG time Main Stream layer body only when explicit GPU diagnostic is enabled
''',
    "logical block layer start helper",
)
model = replace_once(
    model,
    '''            self._premat_runtime.forensic_layer_end(layer_index)                                                        # <<< THOG close Main Stream layer interval before optional host-only delay
            self._premat_runtime.layer_complete(layer_index)
''',
    '''            self._premat_forensic_layer_end(layer_index)                                                               # <<< THOG close Main Stream layer interval before optional host-only delay
            self._premat_runtime.layer_complete(layer_index)
''',
    "logical block layer end helper",
)
model_path.write_text(model)


test_path = Path("tests/test_premat.py")
tests = test_path.read_text()
tests = replace_once(
    tests,
    '''    for timing in runtime._pending_timings:
        timing.end_event.complete = True
    runtime.begin((3, 5, 7), reference=_FakeTensor())
''',
    '''    for timing in runtime._pending_timings:
        timing.end_event.complete = True
    # vvv THOG the sampled pass now waits for both legacy hit classification and the non-blocking overlap-forensic event set
    for pending in runtime._pending_forensic_passes:
        pending.origin_event.complete = True
        for interval in pending.layer_intervals:
            interval.start_event.complete = True
            if interval.end_event is not None:
                interval.end_event.complete = True
        for interval in pending.main_work_intervals:
            interval.start_event.complete = True
            if interval.end_event is not None:
                interval.end_event.complete = True
        for interval in pending.candidate_intervals:
            interval.start_event.complete = True
            interval.end_event.complete = True
            if interval.dependency_event is not None:
                interval.dependency_event.complete = True
            if interval.wait_end_event is not None:
                interval.wait_end_event.complete = True
    # ^^^ THOG
    runtime.begin((3, 5, 7), reference=_FakeTensor())
''',
    "sampled live forensic completion",
)
tests = replace_once(
    tests,
    '''    assert aggregate["real_premat_materialisations_launched"] == 4
    assert aggregate["real_premat_materialisations_consumed"] == 1
    assert aggregate["real_premat_materialisations_unused"] == 3
''',
    '''    # layer_start(5) correctly schedules the next target layer before DOWN(5) is consumed.
    assert aggregate["real_premat_materialisations_launched"] == 8
    assert aggregate["real_premat_materialisations_consumed"] == 1
    assert aggregate["real_premat_materialisations_unused"] == 7
''',
    "real launch lifecycle expectations",
)
test_path.write_text(tests)
