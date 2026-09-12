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
    '''def _find_nsys() -> Optional[str]:\n    resolved = shutil.which("nsys")\n    if resolved:\n        return resolved\n    candidates = sorted(Path("/opt/nvidia/nsight-systems").glob("*/bin/nsys"), reverse=True)\n    candidates.extend(Path("/usr/local/cuda/bin").glob("nsys"))\n    return str(candidates[0]) if candidates else None\n\n\ndef register_processing_handoff(\n''',
    '''def _find_nsys() -> Optional[str]:\n    resolved = shutil.which("nsys")\n    if resolved:\n        return resolved\n    candidates = sorted(Path("/opt/nvidia/nsight-systems").glob("*/bin/nsys"), reverse=True)\n    candidates.extend(Path("/usr/local/cuda/bin").glob("nsys"))\n    return str(candidates[0]) if candidates else None\n\n\n# vvv THOG make the Nsight launch contract explicit: preserve THOG environment, show child output, wait for the child, and collect GPU-only diagnostics\ndef _nsys_profile_command(\n    nsys: str,\n    *,\n    report_base: Path,\n    frequency: int,\n    entrypoint: Path,\n    arguments: Sequence[str],\n) -> list[str]:\n    return [\n        nsys,\n        "profile",\n        "--trace=cuda,nvtx",\n        "--sample=none",\n        "--cpuctxsw=none",\n        "--show-output=true",\n        "--inherit-environment=true",\n        "--wait=primary",\n        "--capture-range=nvtx",\n        "--capture-range-end=stop",\n        f"--nvtx-capture={PROCESSING_CAPTURE_RANGE}",\n        "--gpu-metrics-devices=cuda-visible",\n        f"--gpu-metrics-frequency={frequency}",\n        "--force-overwrite=true",\n        f"--output={report_base}",\n        sys.executable,\n        str(Path(entrypoint).resolve()),\n        *arguments,\n    ]\n# ^^^ THOG\n\n\ndef register_processing_handoff(\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''    command = [\n        nsys,\n        "profile",\n        "--trace=cuda,nvtx",\n        "--capture-range=nvtx",\n        "--capture-range-end=stop",\n        f"--nvtx-capture={PROCESSING_CAPTURE_RANGE}",\n        "--gpu-metrics-devices=cuda-visible",\n        f"--gpu-metrics-frequency={frequency}",\n        "--force-overwrite=true",\n        f"--output={report_base}",\n        sys.executable,\n        str(Path(entrypoint).resolve()),\n        *rewritten_arguments,\n    ]\n''',
    '''    command = _nsys_profile_command(\n        nsys,\n        report_base=report_base,\n        frequency=frequency,\n        entrypoint=entrypoint,\n        arguments=rewritten_arguments,\n    )\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''    if not handoff_path.exists():\n        raise RuntimeError("PREMAT processing capture completed but no INSTRA run handoff was written")\n''',
    '''    if not handoff_path.exists():\n        raise RuntimeError(\n            "PREMAT processing capture completed but no INSTRA run handoff was written; "\n            "the profiled child did not reach telemetry attachment or did not inherit the THOG processing environment"\n        )\n''',
)

replace_once(
    "tests/test_premat_processing.py",
    '''from sheet.premat_processing import (\n    normalize_nsys_sqlite,\n    processing_requested_from_argv,\n    rewrite_processing_cli_for_core,\n    validate_processing_configuration,\n)\n''',
    '''from sheet.premat_processing import (\n    _nsys_profile_command,\n    normalize_nsys_sqlite,\n    processing_requested_from_argv,\n    rewrite_processing_cli_for_core,\n    validate_processing_configuration,\n)\n''',
)

replace_once(
    "tests/test_premat_processing.py",
    '''def test_processing_configuration_is_cuda_but_not_premat_dependent() -> None:\n''',
    '''def test_nsys_profile_command_preserves_child_environment_and_waits(tmp_path: Path) -> None:\n    command = _nsys_profile_command(\n        "/usr/bin/nsys",\n        report_base=tmp_path / "trace",\n        frequency=10000,\n        entrypoint=tmp_path / "runner.py",\n        arguments=["--processing_logging_internal", "enabled"],\n    )\n    assert "--inherit-environment=true" in command\n    assert "--show-output=true" in command\n    assert "--wait=primary" in command\n    assert "--sample=none" in command\n    assert "--cpuctxsw=none" in command\n    assert "--capture-range-end=stop" in command\n    assert "--gpu-metrics-frequency=10000" in command\n\n\ndef test_processing_configuration_is_cuda_but_not_premat_dependent() -> None:\n''',
)

log_path = ROOT / "THOG2_DYNAMIC_PREMATERIALISATION_LOG.md"
log = log_path.read_text()
entry = "\n- Added INSTRA Processing diagnostics backed by Nsight Systems GPU metrics, with downloadable normalized CSV/JSON/ZIP and raw `.nsys-rep`. The Nsight launcher now explicitly inherits the THOG environment, mirrors target stdout/stderr, waits for the primary profiled process after the bounded NVTX capture, and disables unrelated CPU sampling/context-switch collection.\n"
if entry.strip() not in log:
    log_path.write_text(log.rstrip() + entry)

print("PREMAT Nsight launch repair applied")
# ^^^ THOG
