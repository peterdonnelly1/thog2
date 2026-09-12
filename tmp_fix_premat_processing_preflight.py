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
    '''    return rewritten\n\n\ndef _find_nsys() -> Optional[str]:\n''',
    '''    return rewritten\n\n\n# vvv THOG wrapper metadata/help probes must never start Nsight; only the actual training invocation is profiled\n_PROCESSING_NON_TRAINING_FLAGS = frozenset({\n    "-h",\n    "--help",\n    "--dry-run",\n    "--explain-geometry",\n    "--print-artifact-name",\n    "--print-geometry-registry",\n    "--print-resolved-json",\n})\n\n\ndef processing_invocation_is_non_training(arguments: Sequence[str]) -> bool:\n    for raw_argument in arguments:\n        argument = str(raw_argument)\n        option_name = argument.split("=", 1)[0]\n        if option_name in _PROCESSING_NON_TRAINING_FLAGS:\n            return True\n    return False\n# ^^^ THOG\n\n\ndef _find_nsys() -> Optional[str]:\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''    requested, frequency = processing_requested_from_argv(arguments)\n    rewritten_arguments = rewrite_processing_cli_for_core(arguments)\n    if not requested:\n        sys.argv[:] = [sys.argv[0], *rewritten_arguments]\n        return None\n''',
    '''    requested, frequency = processing_requested_from_argv(arguments)\n    rewritten_arguments = rewrite_processing_cli_for_core(arguments)\n    # vvv THOG train_OWT_core.sh first calls --print-resolved-json with the full processing CLI; keep that metadata probe outside Nsight\n    if not requested or processing_invocation_is_non_training(arguments):\n        sys.argv[:] = [sys.argv[0], *rewritten_arguments]\n        return None\n    # ^^^ THOG\n''',
)

replace_once(
    "sheet/premat_processing.py",
    '''    "processing_requested_from_argv",\n    "rewrite_processing_cli_for_core",\n''',
    '''    "processing_invocation_is_non_training",\n    "processing_requested_from_argv",\n    "rewrite_processing_cli_for_core",\n''',
)

replace_once(
    "tests/test_premat_processing.py",
    '''    normalize_nsys_sqlite,\n    processing_requested_from_argv,\n''',
    '''    normalize_nsys_sqlite,\n    processing_invocation_is_non_training,\n    processing_requested_from_argv,\n''',
)

replace_once(
    "tests/test_premat_processing.py",
    '''def test_nsys_profile_command_preserves_child_environment_and_waits(tmp_path: Path) -> None:\n''',
    '''def test_processing_metadata_probe_does_not_require_nsys() -> None:\n    assert processing_invocation_is_non_training([\n        "--premat_processing_logging", "enabled",\n        "--print-resolved-json",\n    ])\n    assert processing_invocation_is_non_training(["--dry-run"])\n    assert processing_invocation_is_non_training(["--print-artifact-name=true"])\n    assert processing_invocation_is_non_training(["--help"])\n    assert not processing_invocation_is_non_training([\n        "--premat_processing_logging", "enabled",\n        "--model-type", "sheet",\n    ])\n\n\ndef test_nsys_profile_command_preserves_child_environment_and_waits(tmp_path: Path) -> None:\n''',
)

log_path = ROOT / "THOG2_DYNAMIC_PREMATERIALISATION_LOG.md"
log = log_path.read_text()
entry = "\n- Corrected the Nsight processing wrapper so `train_OWT_core.sh` metadata/help probes, especially its mandatory `--print-resolved-json` preflight, never launch Nsight or demand an INSTRA handoff. Only the subsequent genuine training invocation is profiled.\n"
if entry.strip() not in log:
    log_path.write_text(log.rstrip() + entry)

print("PREMAT processing preflight repair applied")
# ^^^ THOG
