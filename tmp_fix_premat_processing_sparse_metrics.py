# vvv THOG
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}")
    target.write_text(text.replace(old, new, 1))


replace_once(
    "sheet/premat_processing.py",
    '''def _mean_metric(samples: Sequence[Mapping[str, Any]], key: str, start_us: float, end_us: float) -> Optional[float]:\n    values = [\n        float(row[key])\n        for row in samples\n        if key in row and start_us <= float(row["time_us"]) <= end_us and math.isfinite(float(row[key]))\n    ]\n    return sum(values) / len(values) if values else None\n''',
    '''# vvv THOG Nsight GPU_METRICS rows are sparse by timestamp; ignore absent/non-numeric metric cells rather than coercing empty placeholders\ndef _mean_metric(samples: Sequence[Mapping[str, Any]], key: str, start_us: float, end_us: float) -> Optional[float]:\n    values: list[float] = []\n    for row in samples:\n        if not start_us <= float(row["time_us"]) <= end_us:\n            continue\n        raw_value = row.get(key)\n        if raw_value in (None, ""):\n            continue\n        try:\n            value = float(raw_value)\n        except (TypeError, ValueError):\n            continue\n        if math.isfinite(value):\n            values.append(value)\n    return sum(values) / len(values) if values else None\n# ^^^ THOG\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''    environment[_PROCESSING_CHILD_ENV] = "1"\n    environment[_PROCESSING_HANDOFF_ENV] = str(handoff_path)\n    environment[_PROCESSING_CAPTURE_METADATA_ENV] = str(capture_metadata_path)\n''',
    '''    environment[_PROCESSING_CHILD_ENV] = "1"\n    environment[_PROCESSING_HANDOFF_ENV] = str(handoff_path)\n    environment[_PROCESSING_CAPTURE_METADATA_ENV] = str(capture_metadata_path)\n    environment["NSYS_NVTX_PROFILER_REGISTER_ONLY"] = "0"  # <<< THOG PyTorch range_push uses unregistered NVTX strings; Nsight capture trigger must accept them\n''',
)

replace_once(
    "tests/test_premat_processing.py",
    '''    _nsys_profile_command,\n    normalize_nsys_sqlite,\n''',
    '''    _mean_metric,\n    _nsys_profile_command,\n    normalize_nsys_sqlite,\n''',
)

replace_once(
    "tests/test_premat_processing.py",
    '''def test_processing_configuration_is_cuda_but_not_premat_dependent() -> None:\n''',
    '''def test_processing_sparse_metric_samples_ignore_empty_cells() -> None:\n    samples = [\n        {"time_us": 1.0, "sm_active_pct": 80.0, "tensor_active_pct": ""},\n        {"time_us": 2.0, "sm_active_pct": "", "tensor_active_pct": 90.0},\n        {"time_us": 3.0, "sm_active_pct": None, "tensor_active_pct": "not-a-number"},\n    ]\n    assert _mean_metric(samples, "sm_active_pct", 0.0, 4.0) == 80.0\n    assert _mean_metric(samples, "tensor_active_pct", 0.0, 4.0) == 90.0\n    assert _mean_metric(samples, "sm_issue_pct", 0.0, 4.0) is None\n\n\ndef test_processing_configuration_is_cuda_but_not_premat_dependent() -> None:\n''',
)

log_path = ROOT / "THOG2_DYNAMIC_PREMATERIALISATION_LOG.md"
log = log_path.read_text()
entry = "\n- Scruffy's first successful Nsight capture produced a valid `.nsys-rep` but normalization failed on sparse per-timestamp GPU metric rows because missing metrics were represented as empty cells and `_mean_metric` attempted `float(\"\")`. Normalization now ignores absent/non-numeric cells. The processing launcher also sets `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` automatically because PyTorch `torch.cuda.nvtx.range_push` emits unregistered NVTX strings.\n"
if entry.strip() not in log:
    log_path.write_text(log.rstrip() + entry)

print("PREMAT processing sparse-metric/NVTX repair applied")
# ^^^ THOG
