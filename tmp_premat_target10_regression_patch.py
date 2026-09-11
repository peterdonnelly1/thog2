from __future__ import annotations

from pathlib import Path
import re


# Refine target-10 event provenance after the main patch: candidate events carry
# their actual +1 or +0 target, while non-candidate events retain the configured
# mnemonic target value.
path = Path("sheet/premat.py")
text = path.read_text()
marker = "    def _record(\n"
if text.count(marker) != 1:
    raise RuntimeError("could not uniquely locate Premat _record")
prefix, record = text.split(marker, 1)
old = "        now_ns = time.perf_counter_ns()\n        payload: Dict[str, object] = {\n"
new = (
    "        now_ns = time.perf_counter_ns()\n"
    "        candidate_target_offset = self._target_offset_for_candidate(candidate)\n"
    "        payload: Dict[str, object] = {\n"
)
if record.count(old) != 1:
    raise RuntimeError("could not add candidate target-offset provenance")
record = record.replace(old, new, 1)
old = '            "target_offset": self._target_layer,\n'
new = (
    '            "target_offset": (\n'
    '                self._target_layer\n'
    '                if candidate_target_offset is None\n'
    '                else candidate_target_offset\n'
    '            ),\n'
)
if record.count(old) < 1:
    raise RuntimeError("could not update Premat event target offset")
record = record.replace(old, new, 1)
path.write_text(prefix + marker + record)


# Align PREMAT scheduler regressions with the already-adopted behaviour:
# blocked matrices do not head-of-line block later matrices, and Premat-owned
# tensors stay associated with the Premat stream until explicit safe release.
path = Path("tests/test_premat.py")
text = path.read_text()
text = text.replace(
    "def test_public_cli_exposes_exactly_the_twelve_premat_options() -> None:",
    "def test_public_cli_exposes_exactly_the_thirteen_premat_options() -> None:",
    1,
)
old = '        "--premat",\n        "--premat_attention_mode",'
new = '        "--premat",\n        "--premat_allocator_aware_admission",\n        "--premat_attention_mode",'
if text.count(old) < 1:
    raise RuntimeError("could not update PREMAT public CLI option set")
text = text.replace(old, new, 1)
old = "    assert weight.recorded_streams[-1] is fake_cuda.main_stream\n"
new = "    assert weight.recorded_streams[-1] is runtime._stream\n"
if text.count(old) != 1:
    raise RuntimeError("could not update Premat stream-lifetime assertion")
text = text.replace(old, new, 1)
pattern = r"def test_cumulative_charge_blocks_second_candidate_without_bypass\(monkeypatch\) -> None:\n.*?\n\ndef test_lifecycle_rejects_illegal_transition"
replacement = '''def test_cumulative_charge_blocks_unaffordable_candidates_without_head_of_line_blocking(monkeypatch) -> None:
    runtime, fake_cuda, calls = _runtime(
        monkeypatch,
        stay_below_current_peak=False,
    )
    fake_cuda.free = 1_300
    fake_cuda.total = 1_400
    runtime.layer_start(3)

    # DOWN fits, UP is blocked, O still gets its own admission test and fits,
    # while QKV remains blocked. A blocked matrix must not head-of-line block
    # a later independently affordable candidate.
    assert calls == [("DOWN", 5), ("O", 5)]
    report = runtime.report()
    deferred_families = {
        event.get("family")
        for event in report["events"]
        if event.get("event") == "admission_deferred"
    }
    assert {"UP", "QKV"} <= deferred_families
    assert report["memory"]["premat_cumulative_charged_bytes"] > 512


def test_lifecycle_rejects_illegal_transition'''
text, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
if count != 1:
    raise RuntimeError(f"failed to update cumulative-charge regression: {count}")
old = '    assert report["target_scope"] == "relative_layer_1_then_0"\n'
new = '''    assert report["target_scope"] == "relative_layer_1_then_0"
    materialising_offsets = [
        event["target_offset"]
        for event in report["events"]
        if event.get("event") == "materialising"
    ]
    assert materialising_offsets[:4] == [1, 1, 1, 1]
    assert materialising_offsets[4:8] == [0, 0, 0, 0]
'''
if text.count(old) != 1:
    raise RuntimeError("could not extend target10 provenance regression")
text = text.replace(old, new, 1)
path.write_text(text)


# Align Instra regressions with immutable completed history and the requested
# renamed/streamlined key and faster playback controls.
path = Path("tests/test_premat_instra.py")
text = path.read_text()
text = text.replace(
    "def test_live_writer_replaces_the_active_update_without_growing_history(tmp_path: Path) -> None:",
    "def test_live_writer_freezes_first_completed_update_without_growing_history(tmp_path: Path) -> None:",
    1,
)
old = '        assert snapshots[0]["pass_sequence"] == 2\n'
new = '        assert snapshots[0]["pass_sequence"] == 1\n'
if text.count(old) != 1:
    raise RuntimeError("could not update immutable live-writer regression")
text = text.replace(old, new, 1)
text = text.replace("CONSUMING - NO WAITING", "CONSUMING - NO WAIT")
text = text.replace(
    "PREMAT NOT STARTED - MAIN CODE MATERIALISING",
    "PREMAT NOT STARTED - MAIN STREAM MATERIALISING",
)
text = text.replace("MAIN CODE CONSUMING", "MAIN STREAM CONSUMING")
old = '''    for state_class in (
        "premat-neutral", "premat-pending", "premat-state-materialising", "premat-state-available",
        "premat-state-consuming-full", "premat-state-consumed-full",
        "premat-state-waiting", "premat-state-consuming-waited",
        "premat-state-consumed-waited", "premat-state-main-materialising",
        "premat-state-main-consuming", "premat-state-consumed-main",
    ):
        assert f".{state_class}" in css
        assert state_class in index
    for label in (
        "PROCESSING", "OUTCOMES", "OUT OF SCOPE", "NOT YET REACHED",
        "PRE-MATERIALISING", "CONSUMING - NO WAIT",
        "WAITING FOR PRE-MATERIALISATION", "CONSUMING AFTER WAIT",
        "PREMAT NOT STARTED - MAIN STREAM MATERIALISING", "MAIN STREAM CONSUMING",
        "FULL HIT", "PARTIAL HIT", "COMPLETE MISS",
    ):
        assert label in normalized_index
'''
new = '''    for state_class in (
        "premat-neutral", "premat-pending", "premat-state-materialising", "premat-state-available",
        "premat-state-consuming-full", "premat-state-consumed-full",
        "premat-state-waiting", "premat-state-consuming-waited",
        "premat-state-consumed-waited", "premat-state-main-materialising",
        "premat-state-main-consuming", "premat-state-consumed-main",
    ):
        assert f".{state_class}" in css
        if state_class not in {"premat-neutral", "premat-pending"}:
            assert state_class in index
    for label in (
        "FULL SUCCESS", "PART SUCCESS", "MISS",
        "PRE-MATERIALISING", "AVAILABLE", "CONSUMING - NO WAIT",
        "WAITING FOR PRE-MATERIALISATION", "CONSUMING AFTER WAIT",
        "PREMAT NOT STARTED - MAIN STREAM MATERIALISING", "MAIN STREAM CONSUMING",
        "FULL HIT", "PARTIAL HIT", "COMPLETE MISS",
    ):
        assert label in normalized_index
    assert "OUT OF SCOPE" not in normalized_index
    assert "NOT YET REACHED" not in normalized_index
'''
if text.count(old) != 1:
    raise RuntimeError("could not locate old Instra key regression block")
text = text.replace(old, new, 1)
old = '    assert \'min="10" max="2000" step="10" value="250"\' in index\n'
new = '    assert \'min="1" max="1000" step="1" value="100"\' in index\n'
if text.count(old) != 1:
    raise RuntimeError("could not update slider regression")
text = text.replace(old, new, 1)
old = '    assert "PREMAT_FINAL_HOLD_MS = 1000" in javascript\n'
new = '    assert "PREMAT_FINAL_HOLD_MS = 250" in javascript\n'
if text.count(old) != 1:
    raise RuntimeError("could not update final-hold regression")
text = text.replace(old, new, 1)
old = '    assert "grid-template-columns: repeat(9, minmax(100px, 1fr))" in css\n'
new = '    assert ".premat-key-flow" in css\n    assert ".premat-key-arrow" in css\n'
if text.count(old) != 1:
    raise RuntimeError("could not update key-layout regression")
text = text.replace(old, new, 1)
path.write_text(text)

print("aligned PREMAT target10 provenance and regression expectations")
