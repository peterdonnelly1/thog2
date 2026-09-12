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
    '''_PROCESSING_CAPTURE_METADATA_ENV = "THOG2_PREMAT_PROCESSING_CAPTURE_METADATA"\n''',
    '''_PROCESSING_CAPTURE_METADATA_ENV = "THOG2_PREMAT_PROCESSING_CAPTURE_METADATA"\n_NSYS_NVTX_REGISTER_ONLY_ENV = "NSYS_NVTX_PROFILER_REGISTER_ONLY"\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''def _find_nsys() -> Optional[str]:\n    resolved = shutil.which("nsys")\n    if resolved:\n        return resolved\n    candidates = sorted(Path("/opt/nvidia/nsight-systems").glob("*/bin/nsys"), reverse=True)\n    candidates.extend(Path("/usr/local/cuda/bin").glob("nsys"))\n    return str(candidates[0]) if candidates else None\n\n\n# vvv THOG make the Nsight launch contract explicit''',
    '''def _find_nsys() -> Optional[str]:\n    resolved = shutil.which("nsys")\n    if resolved:\n        return resolved\n    candidates = sorted(Path("/opt/nvidia/nsight-systems").glob("*/bin/nsys"), reverse=True)\n    candidates.extend(Path("/usr/local/cuda/bin").glob("nsys"))\n    return str(candidates[0]) if candidates else None\n\n\n# vvv THOG PyTorch nvtx.range_push uses unregistered ASCII strings, so Nsight must permit full NVTX trigger matching\ndef _processing_nsys_environment(\n    base_environment: Mapping[str, str],\n    *,\n    handoff_path: Path,\n    capture_metadata_path: Path,\n) -> Dict[str, str]:\n    environment = dict(base_environment)\n    environment[_PROCESSING_CHILD_ENV] = "1"\n    environment[_PROCESSING_HANDOFF_ENV] = str(handoff_path)\n    environment[_PROCESSING_CAPTURE_METADATA_ENV] = str(capture_metadata_path)\n    environment[_NSYS_NVTX_REGISTER_ONLY_ENV] = "0"\n    return environment\n# ^^^ THOG\n\n\n# vvv THOG make the Nsight launch contract explicit''',
)

replace_once(
    "sheet/premat_processing.py",
    '''    environment = dict(os.environ)\n    environment[_PROCESSING_CHILD_ENV] = "1"\n    environment[_PROCESSING_HANDOFF_ENV] = str(handoff_path)\n    environment[_PROCESSING_CAPTURE_METADATA_ENV] = str(capture_metadata_path)\n''',
    '''    # vvv THOG build the profiler child environment centrally so PyTorch ASCII NVTX ranges can trigger capture\n    environment = _processing_nsys_environment(\n        os.environ,\n        handoff_path=handoff_path,\n        capture_metadata_path=capture_metadata_path,\n    )\n    # ^^^ THOG\n''',
)

replace_once(
    "tests/test_premat_processing.py",
    '''from sheet.premat_processing import (\n    _nsys_profile_command,\n''',
    '''from sheet.premat_processing import (\n    _nsys_profile_command,\n    _processing_nsys_environment,\n''',
)

replace_once(
    "tests/test_premat_processing.py",
    '''def test_nsys_profile_command_preserves_child_environment_and_waits(tmp_path: Path) -> None:\n''',
    '''def test_nsys_environment_allows_unregistered_pytorch_nvtx_trigger(tmp_path: Path) -> None:\n    handoff = tmp_path / "handoff.json"\n    capture = tmp_path / "capture.json"\n    environment = _processing_nsys_environment(\n        {"KEEP": "yes", "NSYS_NVTX_PROFILER_REGISTER_ONLY": "1"},\n        handoff_path=handoff,\n        capture_metadata_path=capture,\n    )\n    assert environment["KEEP"] == "yes"\n    assert environment["THOG2_PREMAT_PROCESSING_UNDER_NSYS"] == "1"\n    assert environment["THOG2_PREMAT_PROCESSING_HANDOFF"] == str(handoff)\n    assert environment["THOG2_PREMAT_PROCESSING_CAPTURE_METADATA"] == str(capture)\n    assert environment["NSYS_NVTX_PROFILER_REGISTER_ONLY"] == "0"\n\n\ndef test_nsys_profile_command_preserves_child_environment_and_waits(tmp_path: Path) -> None:\n''',
)

log_path = ROOT / "THOG2_DYNAMIC_PREMATERIALISATION_LOG.md"
log = log_path.read_text()
entry = "\n- Scruffy Nsight 2026.5 completed the target run but generated no report because the NVTX capture trigger never matched. PyTorch `torch.cuda.nvtx.range_push` uses unregistered ASCII `rangePushA` messages, while Nsight defaults to registered-string-only trigger matching. Processing runs now force `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` in the profiler child environment so the bounded `THOG2_PREMAT_PROCESSING_CAPTURE` range starts collection.\n"
if entry.strip() not in log:
    log_path.write_text(log.rstrip() + entry)

print("PREMAT processing NVTX trigger repair applied")
# ^^^ THOG
