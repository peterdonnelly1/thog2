# vvv THOG
"""Apply the final PLASTIC probe naming, radius parsing, and console-tab fixes."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

# Keep the public/configuration vocabulary exact across runtime code, wrappers,
# generators, tests, workflows, and text documentation. Artifact abbreviations
# are intentionally preserved to avoid an unrelated identity migration.
_NAME_REPLACEMENTS = (
    (
        "THOG2_PLASTIC_LAYER_COUNT_PROBE_NOISE_MIN_OBSERVATIONS",
        "THOG2_PLASTIC_LAYER_COUNT_MIN_PROBES",
    ),
    (
        "THOG2_PLASTIC_LAYER_COUNT_PROBE_INTERVAL",
        "THOG2_PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE",
    ),
    (
        "layer_count_probe_noise_min_observations",
        "layer_count_min_probes",
    ),
    (
        "layer_count_probe_interval",
        "layer_count_probe_window_size",
    ),
)

_RADIUS_PARSE_OLD = '''    parsed, extras = _ORIGINAL_ARGPARSE_PARSE_KNOWN_ARGS(self, stripped, namespace)
    setattr(parsed, "plastic__layer_count_probe_radius", _positive_int(os.environ.get(_RADIUS_ENV, 1), name="plastic__layer_count_probe_radius"))
    setattr(parsed, "plastic__layer_count_max_step", _positive_int(os.environ.get(_MAX_STEP_ENV, 1), name="plastic__layer_count_max_step"))
    return parsed, extras
'''

_RADIUS_PARSE_NEW = '''    parsed, extras = _ORIGINAL_ARGPARSE_PARSE_KNOWN_ARGS(self, stripped, namespace)
    parsed_probe_radius = getattr(parsed, "plastic__layer_count_probe_radius", None)
    parsed_max_step = getattr(parsed, "plastic__layer_count_max_step", None)
    resolved_probe_radius = probe_radius if probe_radius is not None else parsed_probe_radius
    resolved_max_step = max_step if max_step is not None else parsed_max_step
    if resolved_probe_radius is None:
        resolved_probe_radius = os.environ.get(_RADIUS_ENV, 1)
    if resolved_max_step is None:
        resolved_max_step = os.environ.get(_MAX_STEP_ENV, 1)
    setattr(parsed, "plastic__layer_count_probe_radius", _positive_int(resolved_probe_radius, name="plastic__layer_count_probe_radius"))
    setattr(parsed, "plastic__layer_count_max_step", _positive_int(resolved_max_step, name="plastic__layer_count_max_step"))
    return parsed, extras
'''

_PROBE_FIELD_OLD = '        fields.append(f"\\tprobe_losses [{probe_label}] = [{formatted_losses}]")'
_PROBE_FIELD_NEW = '        fields.append(f"probe_losses [{probe_label}] = [{formatted_losses}]")'

_INSERT_OLD = '''    if marker in line:
        return line.replace(marker, f"  {inserted}{marker}", 1)
    return f"{line}  {inserted}"
'''

_INSERT_NEW = '''    if marker in line:
        return line.replace(marker, f"\\t{inserted}\\tlayer indices = ", 1)
    return f"{line}\\t{inserted}"
'''


def _tracked_paths() -> tuple[Path, ...]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
    )
    return tuple(
        ROOT / value.decode("utf-8")
        for value in output.split(b"\0")
        if value
    )


def _replace_once(text: str, old: str, new: str, *, path: Path, label: str) -> str:
    count = text.count(old)
    if count == 0:
        if new in text:
            return text
        raise RuntimeError(f"{path}: cannot find {label} source or replacement")
    if count != 1:
        raise RuntimeError(f"{path}: expected one {label} source, found {count}")
    return text.replace(old, new, 1)


def _rewrite_public_names() -> None:
    for path in _tracked_paths():
        if path == SELF or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in _NAME_REPLACEMENTS:
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _repair_lookahead_patch() -> None:
    path = ROOT / "sheet" / "plastic_depth_lookahead_patch.py"
    text = path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        _RADIUS_PARSE_OLD,
        _RADIUS_PARSE_NEW,
        path=path,
        label="radius parser override",
    )
    text = _replace_once(
        text,
        _PROBE_FIELD_OLD,
        _PROBE_FIELD_NEW,
        path=path,
        label="probe field tab",
    )
    text = _replace_once(
        text,
        _INSERT_OLD,
        _INSERT_NEW,
        path=path,
        label="probe/sampled insertion tabs",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    _rewrite_public_names()
    _repair_lookahead_patch()


if __name__ == "__main__":
    main()
# ^^^ THOG
