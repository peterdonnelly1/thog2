#!/usr/bin/env python3
# vvv THOG
"""Keep PLASTIC v0.521 deliberately free of pre-v0.521 checkpoint compatibility."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "tools/apply_plastic_v0521_probe_sampling_and_console.py"


def remove_once(content: str, snippet: str, *, label: str) -> str:
    count = content.count(snippet)
    if count == 0:
        return content
    if count != 1:
        raise RuntimeError(f"{label}: expected at most one match, found {count}")
    return content.replace(snippet, "", 1)


def replace_once(content: str, old: str, new: str, *, label: str) -> str:
    if new in content:
        return content
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return content.replace(old, new, 1)


# First make the primary idempotent migration tool itself obey the decision: no old-checkpoint special case.
primary = PRIMARY.read_text(encoding="utf-8")
primary = primary.replace(
    "# Canonical runner CLI, resolved config, startup report, and exact old-checkpoint migration.\n",
    "# Canonical runner CLI, resolved config and startup report.\n",
)
primary = remove_once(
    primary,
    '''_replace_once(\n    "run_thog2_owt_core.py",\n    '    stored = TrainingConfig(**payload["trainer_config"])\\n',\n    '    # vvv THOG v0.521 checkpoints written before the public probe-token knob retain their historical hard-coded 256-token semantics\\n    stored_values = dict(payload["trainer_config"])\\n    if stored_values.get("plastic__enabled", False) and "plastic__layer_count_probe__number_of_sampled_valid_tokens" not in stored_values:\\n        stored_values["plastic__layer_count_probe__number_of_sampled_valid_tokens"] = 256\\n    stored = TrainingConfig(**stored_values)\\n    # ^^^ THOG\\n',\n)\n''',
    label="primary run_thog2 legacy migration",
)
primary = remove_once(
    primary,
    '''# Shared checkpoint resume path gets the same legacy-256 migration before TrainingConfig construction.\n_replace_once(\n    "sheet/trainer_checkpoint_resume.py",\n    '        checkpoint_config = TrainingConfig(**payload["trainer_config"])\\n',\n    '        # vvv THOG v0.521 preserve exact pre-knob PLASTIC resume semantics: missing probe-token field means the historical fixed 256-token sample\\n        checkpoint_config_values = dict(payload["trainer_config"])\\n        if checkpoint_config_values.get("plastic__enabled", False) and "plastic__layer_count_probe__number_of_sampled_valid_tokens" not in checkpoint_config_values:\\n            checkpoint_config_values["plastic__layer_count_probe__number_of_sampled_valid_tokens"] = 256\\n        checkpoint_config = TrainingConfig(**checkpoint_config_values)\\n        # ^^^ THOG\\n',\n)\n\n''',
    label="primary trainer-resume legacy migration",
)
primary = primary.replace(
    '    "Checkpoint migration is semantic: a pre-v0.521 PLASTIC checkpoint that lacks the new field is interpreted as 256 sampled valid tokens, matching the previously hard-coded implementation.\\n"\n',
    "",
)
primary = primary.replace(
    '    "• Pre-v0.521 PLASTIC checkpoints lacking the field retain the historical 256-token meaning.\\n"\n',
    "",
)
PRIMARY.write_text(primary, encoding="utf-8")

# Apply the now-clean v0.521 implementation.
subprocess.run(
    ["python", str(PRIMARY.relative_to(ROOT))],
    cwd=ROOT,
    check=True,
)

# If an earlier queued writeback raced us and already installed compatibility code, remove it from runtime source too.
runner_path = ROOT / "run_thog2_owt_core.py"
runner = runner_path.read_text(encoding="utf-8")
runner = replace_once(
    runner,
    '''    # vvv THOG v0.521 checkpoints written before the public probe-token knob retain their historical hard-coded 256-token semantics\n    stored_values = dict(payload["trainer_config"])\n    if stored_values.get("plastic__enabled", False) and "plastic__layer_count_probe__number_of_sampled_valid_tokens" not in stored_values:\n        stored_values["plastic__layer_count_probe__number_of_sampled_valid_tokens"] = 256\n    stored = TrainingConfig(**stored_values)\n    # ^^^ THOG\n''',
    '    stored = TrainingConfig(**payload["trainer_config"])\n',
    label="runtime run_thog2 legacy migration",
) if "pre-v0.521" in runner or "hard-coded 256-token" in runner else runner
runner_path.write_text(runner, encoding="utf-8")

resume_path = ROOT / "sheet/trainer_checkpoint_resume.py"
resume = resume_path.read_text(encoding="utf-8")
resume = replace_once(
    resume,
    '''        # vvv THOG v0.521 preserve exact pre-knob PLASTIC resume semantics: missing probe-token field means the historical fixed 256-token sample\n        checkpoint_config_values = dict(payload["trainer_config"])\n        if checkpoint_config_values.get("plastic__enabled", False) and "plastic__layer_count_probe__number_of_sampled_valid_tokens" not in checkpoint_config_values:\n            checkpoint_config_values["plastic__layer_count_probe__number_of_sampled_valid_tokens"] = 256\n        checkpoint_config = TrainingConfig(**checkpoint_config_values)\n        # ^^^ THOG\n''',
    '        checkpoint_config = TrainingConfig(**payload["trainer_config"])\n',
    label="runtime trainer-resume legacy migration",
) if "pre-knob PLASTIC resume semantics" in resume else resume
resume_path.write_text(resume, encoding="utf-8")

# Generated spec must state only current v0.521 semantics.
spec_path = ROOT / "docs/THOG2_PLASTIC_Requirements_Specification_v0.521.txt"
if spec_path.exists():
    spec = spec_path.read_text(encoding="utf-8")
    spec = spec.replace(
        "Checkpoint migration is semantic: a pre-v0.521 PLASTIC checkpoint that lacks the new field is interpreted as 256 sampled valid tokens, matching the previously hard-coded implementation.\n",
        "",
    )
    spec = spec.replace(
        "• Pre-v0.521 PLASTIC checkpoints lacking the field retain the historical 256-token meaning.\n",
        "",
    )
    spec_path.write_text(spec, encoding="utf-8")

print("PLASTIC v0.521 now has no pre-v0.521 checkpoint compatibility path.")
# ^^^ THOG
