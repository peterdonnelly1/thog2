#!/usr/bin/env python3
# vvv THOG
"""Apply PLASTIC v0.52 directional-coherence controls and canonical console/config names."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

_NAME_REPLACEMENTS = (
    ("THOG2_PLASTIC_LAYER_COUNT_PROBE_NOISE_WINDOW", "THOG2_PLASTIC_LAYER_COUNT_PROBE__WINDOW_SIZE_AS_NUMBER_OF_PROBES"),
    ("THOG2_PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE", "THOG2_PLASTIC_LAYER_COUNT_PROBE__PROBE_EVERY_N_STEPS"),
    ("PLASTIC_LAYER_COUNT_PROBE_NOISE_WINDOW", "PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE_AS_NUMBER_OF_PROBES"),
    ("PLASTIC_LAYER_COUNT_PROBE_INTERVAL", "PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS"),
    ("plastic__layer_count_probe_noise_window", "plastic__layer_count_probe__window_size_as_number_of_probes"),
    ("plastic__layer_count_probe_window_size", "plastic__layer_count_probe__probe_every_n_steps"),
    ("layer_count_probe_noise_window", "layer_count_probe__window_size_as_number_of_probes"),
    ("layer_count_probe_window_size", "layer_count_probe__probe_every_n_steps"),
)

_ACTIVE_MIN_PROBE_FILES = (
    "run_thog2_owt.py",
    "run_thog2_owt_core.py",
    "sheet/checkpoints.py",
    "sheet/help_registry_descriptor_patch.py",
    "sheet/model.py",
    "sheet/plastic_depth.py",
    "sheet/plastic_depth_lookahead_patch.py",
    "sheet/run_config.py",
    "sheet/training_config.py",
    "sheet/trainer_step.py",
)


def _tracked_paths() -> tuple[Path, ...]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return tuple(ROOT / item.decode("utf-8") for item in output.split(b"\0") if item)


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def _replace_once(text: str, old: str, new: str, *, path: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def _rename_controls_everywhere() -> None:
    for path in _tracked_paths():
        if path == SELF or not path.is_file() or ".github/workflows" in path.as_posix():
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


def _remove_min_probe_validation(text: str) -> str:
    pattern = re.compile(
        r"\n        if \(\n"
        r"            isinstance\(self\.plastic__layer_count_min_probes, bool\)"
        r".*?"
        r"\n            \)\n",
        flags=re.DOTALL,
    )
    return pattern.sub("\n", text)


def _remove_min_probe_runtime_fields() -> None:
    for relative in _ACTIVE_MIN_PROBE_FILES:
        text = _read(relative)
        if relative in {"sheet/model.py", "sheet/run_config.py", "sheet/training_config.py"}:
            text = _remove_min_probe_validation(text)
        lines = []
        for line in text.splitlines(keepends=True):
            if "plastic__layer_count_min_probes" in line:
                continue
            if relative == "sheet/plastic_depth.py" and "layer_count_min_probes" in line:
                continue
            lines.append(line)
        _write(relative, "".join(lines))


def _remove_min_probe_shell() -> None:
    path = "train_OWT_core.sh"
    text = _read(path)
    text = text.replace("|--plastic__layer_count_min_probes=*", "")
    text = text.replace("|--plastic__layer_count_min_probes", "")
    lines = []
    for line in text.splitlines(keepends=True):
        if "PLASTIC_LAYER_COUNT_PROBE_NOISE_MIN_OBSERVATIONS" in line:
            continue
        if "--plastic__layer_count_min_probes" in line:
            continue
        lines.append(line)
    _write(path, "".join(lines))


def _retire_old_min_probe_migrations() -> None:
    paths = (
        "tools/apply_plastic_probe_naming_and_console_fix.py",
        "tools/apply_plastic_core_canonicalisation_final.py",
    )
    for path in paths:
        text = _read(path)
        text = text.replace("|--plastic__layer_count_min_probes=*", "")
        text = text.replace("|--plastic__layer_count_min_probes", "")
        text = text.replace("LNM.*--plastic__layer_count_min_probes N", "")
        if path.endswith("apply_plastic_probe_naming_and_console_fix.py"):
            text = re.sub(
                r"\n    \(\n        \"THOG2_PLASTIC_LAYER_COUNT_PROBE_NOISE_MIN_OBSERVATIONS\",\n        \"THOG2_PLASTIC_LAYER_COUNT_MIN_PROBES\",\n    \),",
                "",
                text,
            )
            text = re.sub(
                r"\n    \(\n        \"layer_count_probe_noise_min_observations\",\n        \"layer_count_min_probes\",\n    \),",
                "",
                text,
            )
        lines = []
        for line in text.splitlines(keepends=True):
            if "--plastic__layer_count_min_probes" in line:
                continue
            lines.append(line)
        _write(path, "".join(lines))


def _add_extrapolation_weight_to_run_config() -> None:
    path = "sheet/run_config.py"
    text = _read(path)
    text = _replace_once(text, '    "plastic__layer_count_max_step",\n', '    "plastic__layer_count_max_step",\n    "plastic__layer_count_extrapolation_weight",\n', path=path, label="PLASTIC run-config field tuple")
    text = _replace_once(text, "    plastic__layer_count_max_step: int = 1\n", "    plastic__layer_count_max_step: int = 1\n    plastic__layer_count_extrapolation_weight: float = 0.8\n", path=path, label="run-config extrapolation field")
    anchor = "        validate_plastic_fine_count_controls(\n            probe_radius=self.plastic__layer_count_probe_radius,\n            max_step=self.plastic__layer_count_max_step,\n        )\n"
    addition = anchor + (
        "        if (\n"
        "            isinstance(self.plastic__layer_count_extrapolation_weight, bool)\n"
        "            or not isinstance(self.plastic__layer_count_extrapolation_weight, (int, float))\n"
        "            or not math.isfinite(float(self.plastic__layer_count_extrapolation_weight))\n"
        "            or not (0.5 < float(self.plastic__layer_count_extrapolation_weight) <= 1.0)\n"
        "        ):\n"
        "            raise ValueError(\n"
        "                \"plastic__layer_count_extrapolation_weight must lie in (0.5, 1.0]\"\n"
        "            )\n"
    )
    text = _replace_once(text, anchor, addition, path=path, label="run-config weight validation")
    text = _replace_once(text, "                layer_count_max_step=self.plastic__layer_count_max_step,\n", "                layer_count_max_step=self.plastic__layer_count_max_step,\n                layer_count_extrapolation_weight=float(self.plastic__layer_count_extrapolation_weight),\n", path=path, label="run identity weight")
    text = _replace_once(text, "            plastic__layer_count_max_step=self.plastic__layer_count_max_step,\n", "            plastic__layer_count_max_step=self.plastic__layer_count_max_step,\n            plastic__layer_count_extrapolation_weight=float(self.plastic__layer_count_extrapolation_weight),\n", path=path, label="training-config weight pass")
    _write(path, text)


def _add_extrapolation_weight_to_training_config() -> None:
    path = "sheet/training_config.py"
    text = _read(path)
    text = _replace_once(text, '    "plastic__layer_count_max_step",\n', '    "plastic__layer_count_max_step",\n    "plastic__layer_count_extrapolation_weight",\n', path=path, label="PLASTIC training-config field tuple")
    text = _replace_once(text, "    plastic__layer_count_max_step: int = 1\n", "    plastic__layer_count_max_step: int = 1\n    plastic__layer_count_extrapolation_weight: float = 0.8\n", path=path, label="training-config extrapolation field")
    anchor = "        validate_plastic_fine_count_controls(\n            probe_radius=self.plastic__layer_count_probe_radius,\n            max_step=self.plastic__layer_count_max_step,\n        )\n"
    addition = anchor + (
        "        if (\n"
        "            isinstance(self.plastic__layer_count_extrapolation_weight, bool)\n"
        "            or not isinstance(self.plastic__layer_count_extrapolation_weight, (int, float))\n"
        "            or not math.isfinite(float(self.plastic__layer_count_extrapolation_weight))\n"
        "            or not (0.5 < float(self.plastic__layer_count_extrapolation_weight) <= 1.0)\n"
        "        ):\n"
        "            raise ValueError(\n"
        "                \"plastic__layer_count_extrapolation_weight must lie in (0.5, 1.0]; \"\n"
        "                f\"got {self.plastic__layer_count_extrapolation_weight!r}\"\n"
        "            )\n"
    )
    text = _replace_once(text, anchor, addition, path=path, label="training-config weight validation")
    text = _replace_once(text, "                layer_count_max_step=self.plastic__layer_count_max_step,\n", "                layer_count_max_step=self.plastic__layer_count_max_step,\n                layer_count_extrapolation_weight=float(self.plastic__layer_count_extrapolation_weight),\n", path=path, label="training identity weight")
    _write(path, text)


def _add_extrapolation_weight_to_identity() -> None:
    path = "sheet/plastic_depth.py"
    text = _read(path)
    text = _replace_once(text, "    layer_count_max_step: int = 1,\n", "    layer_count_max_step: int = 1,\n    layer_count_extrapolation_weight: float = 0.8,\n", path=path, label="identity weight parameter")
    text = _replace_once(text, '        "plastic__layer_count_max_step": int(layer_count_max_step),\n', '        "plastic__layer_count_max_step": int(layer_count_max_step),\n        "plastic__layer_count_extrapolation_weight": float(layer_count_extrapolation_weight),\n', path=path, label="identity weight value")
    _write(path, text)


def _add_extrapolation_weight_to_parser() -> None:
    path = "run_thog2_owt_core.py"
    text = _read(path)
    text = _replace_once(text, '    parser.add_argument("--plastic__layer_count_max_step", dest="plastic__layer_count_max_step", type=int, default=1)\n', '    parser.add_argument("--plastic__layer_count_max_step", dest="plastic__layer_count_max_step", type=int, default=1)\n    parser.add_argument("--plastic__layer_count_extrapolation_weight", dest="plastic__layer_count_extrapolation_weight", type=float, default=0.8)\n', path=path, label="parser extrapolation weight")
    text = _replace_once(text, "        plastic__layer_count_max_step=arguments.plastic__layer_count_max_step,\n", "        plastic__layer_count_max_step=arguments.plastic__layer_count_max_step,\n        plastic__layer_count_extrapolation_weight=arguments.plastic__layer_count_extrapolation_weight,\n", path=path, label="argument weight pass")
    _write(path, text)


def _wire_directional_selector() -> None:
    path = "sheet/plastic_depth_lookahead_patch.py"
    text = _read(path)
    anchor = "            noise_window=self.config.plastic__layer_count_probe__window_size_as_number_of_probes,\n"
    if "extrapolation_weight=float(self.config.plastic__layer_count_extrapolation_weight)" not in text:
        if anchor not in text:
            raise RuntimeError(f"{path}: missing selector history-window anchor")
        text = text.replace(anchor, anchor + "            extrapolation_weight=float(self.config.plastic__layer_count_extrapolation_weight),\n", 1)
    _write(path, text)


def _add_directional_overlay_import() -> None:
    path = "sheet/__init__.py"
    text = _read(path)
    block = (
        "\n# vvv THOG install v0.52 goal-agnostic directional coherence and final PLASTIC progress formatting after every earlier overlay\n"
        "from . import plastic_depth_directional_coherence_patch as _plastic_depth_directional_coherence_patch\n"
        "# ^^^ THOG\n"
    )
    if "plastic_depth_directional_coherence_patch" not in text:
        text = text.rstrip() + "\n" + block
    _write(path, text)


def _add_startup_weight() -> None:
    path = "run_thog2_owt.py"
    text = _read(path)
    text = _replace_once(text, '    "plastic__layer_count_max_step:",\n', '    "plastic__layer_count_max_step:",\n    "plastic__layer_count_extrapolation_weight:",\n', path=path, label="startup weight label")
    text = _replace_once(text, '    _print_plastic_option("plastic__layer_count_max_step:", str(max_step))\n', '    _print_plastic_option("plastic__layer_count_max_step:", str(max_step))\n    _print_plastic_option("plastic__layer_count_extrapolation_weight:", _startup_float(config.plastic__layer_count_extrapolation_weight))\n', path=path, label="startup weight value")
    _write(path, text)


def _add_checkpoint_weight() -> None:
    path = "sheet/checkpoints.py"
    text = _read(path)
    text = _replace_once(text, '        "count_update_brake": value.get("plastic__layer_count_update_brake"),\n', '        "count_update_brake": value.get("plastic__layer_count_update_brake"),\n        "extrapolation_weight": value.get("plastic__layer_count_extrapolation_weight"),\n', path=path, label="checkpoint extrapolation identity")
    _write(path, text)


def _add_registry_weight() -> None:
    path = "sheet/help_registry_descriptor_patch.py"
    text = _read(path)
    text = _replace_once(text, '            ("LMS", "--plastic__layer_count_max_step N", "maximum committed count movement"),\n', '            ("LMS", "--plastic__layer_count_max_step N", "maximum committed count movement"),\n            ("LEW", "--plastic__layer_count_extrapolation_weight VALUE", "discount for extrapolative/right-side directional evidence"),\n', path=path, label="registry extrapolation weight")
    _write(path, text)


def _add_shell_weight() -> None:
    path = "train_OWT_core.sh"
    text = _read(path)
    text = _replace_once(text, 'PLASTIC_LAYER_COUNT_MAX_STEP="${THOG2_PLASTIC_LAYER_COUNT_MAX_STEP:-1}"\n', 'PLASTIC_LAYER_COUNT_MAX_STEP="${THOG2_PLASTIC_LAYER_COUNT_MAX_STEP:-1}"\nPLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT="0.8"\n', path=path, label="shell extrapolation default")
    text = _replace_once(text, '  --plastic__layer_count_max_step N=${PLASTIC_LAYER_COUNT_MAX_STEP}\n', '  --plastic__layer_count_max_step N=${PLASTIC_LAYER_COUNT_MAX_STEP}\n  --plastic__layer_count_extrapolation_weight VALUE=${PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT}\n', path=path, label="shell extrapolation help")
    text = text.replace("|--plastic__layer_count_probe_noise_lambda|", "|--plastic__layer_count_probe_noise_lambda|--plastic__layer_count_extrapolation_weight|")
    text = text.replace("|--plastic__layer_count_probe_noise_lambda=*|", "|--plastic__layer_count_probe_noise_lambda=*|--plastic__layer_count_extrapolation_weight=*|")
    text = text.replace('        --plastic__layer_count_probe_noise_lambda) PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA="$2" ;;\n', '        --plastic__layer_count_probe_noise_lambda) PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA="$2" ;;\n        --plastic__layer_count_extrapolation_weight) PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT="$2" ;;\n')
    text = text.replace('        --plastic__layer_count_probe_noise_lambda) PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA="$plastic_value" ;;\n', '        --plastic__layer_count_probe_noise_lambda) PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA="$plastic_value" ;;\n        --plastic__layer_count_extrapolation_weight) PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT="$plastic_value" ;;\n')
    validation_anchor = 'validate_nonnegative_number "$PLASTIC_CUDA_ALLOCATOR_RESERVE_GIB" "PLASTIC_CUDA_ALLOCATOR_RESERVE_GIB"\n'
    validation = (
        'python - "$PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT" <<\'PY\'\n'
        'import math, sys\n'
        'value = float(sys.argv[1])\n'
        'if not math.isfinite(value) or not (0.5 < value <= 1.0):\n'
        '    raise SystemExit("PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT must lie in (0.5, 1.0]")\n'
        'PY\n'
    )
    if validation not in text:
        if validation_anchor not in text:
            raise RuntimeError(f"{path}: missing allocator validation anchor")
        text = text.replace(validation_anchor, validation + validation_anchor, 1)
    text = text.replace('    optional_args+=(--plastic__layer_count_probe_noise_lambda "$PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA")\n', '    optional_args+=(--plastic__layer_count_probe_noise_lambda "$PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA")\n    optional_args+=(--plastic__layer_count_extrapolation_weight "$PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT")\n')
    text = text.replace('  plastic fine:       probe_interval=${PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS:-update_brake} radius=$PLASTIC_LAYER_COUNT_PROBE_RADIUS max_step=$PLASTIC_LAYER_COUNT_MAX_STEP brake=$PLASTIC_LAYER_COUNT_UPDATE_BRAKE\n', '  plastic fine:       probe_every=${PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS:-update_brake} window=$PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE_AS_NUMBER_OF_PROBES radius=$PLASTIC_LAYER_COUNT_PROBE_RADIUS max_step=$PLASTIC_LAYER_COUNT_MAX_STEP brake=$PLASTIC_LAYER_COUNT_UPDATE_BRAKE extrap_w=$PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT\n')
    _write(path, text)


def _add_audit_direction_fields() -> None:
    path = "sheet/plastic_depth_audit_patch.py"
    text = _read(path)
    text = text.replace('"probe_interval": int(self.config.plastic__layer_count_probe__probe_every_n_steps),', '"probe_every_n_steps": int(self.config.plastic__layer_count_probe__probe_every_n_steps),')
    anchor = '        "objective_cost_weight": float(self.config.plastic__layer_count_cost_weight),\n'
    if '"extrapolation_weight"' not in text:
        if anchor not in text:
            raise RuntimeError(f"{path}: missing objective-cost audit anchor")
        text = text.replace(anchor, anchor + '        "extrapolation_weight": float(self.config.plastic__layer_count_extrapolation_weight),\n', 1)
    if '"directional_report"' not in text:
        text = text.replace('        "robust_evidence": _evidence_payload(decision),\n', '        "robust_evidence": _evidence_payload(decision),\n        "directional_report": copy.deepcopy(context.get("plastic_directional_report")),\n', 1)
    _write(path, text)


def main() -> None:
    _rename_controls_everywhere()
    _remove_min_probe_runtime_fields()
    _remove_min_probe_shell()
    _retire_old_min_probe_migrations()
    _add_extrapolation_weight_to_run_config()
    _add_extrapolation_weight_to_training_config()
    _add_extrapolation_weight_to_identity()
    _add_extrapolation_weight_to_parser()
    _wire_directional_selector()
    _add_directional_overlay_import()
    _add_startup_weight()
    _add_checkpoint_weight()
    _add_registry_weight()
    _add_shell_weight()
    _add_audit_direction_fields()


if __name__ == "__main__":
    main()
# ^^^ THOG
