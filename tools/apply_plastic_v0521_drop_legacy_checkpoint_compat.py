#!/usr/bin/env python3
# vvv THOG
"""Finalize PLASTIC v0.521 as a current-only probe-sampling and console contract."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "tools/apply_plastic_v0521_probe_sampling_and_console.py"


def remove_between(content: str, start: str, end: str, *, label: str) -> str:
    start_index = content.find(start)
    if start_index < 0:
        return content
    second_start = content.find(start, start_index + len(start))
    if second_start >= 0:
        raise RuntimeError(f"{label}: start marker is not unique")
    end_index = content.find(end, start_index)
    if end_index < 0:
        raise RuntimeError(f"{label}: end marker not found")
    end_index += len(end)
    return content[:start_index] + content[end_index:]


def replace_exact_once(content: str, old: str, new: str, *, label: str) -> str:
    count = content.count(old)
    if count == 0:
        if new in content:
            return content
        raise RuntimeError(f"{label}: replacement target not found")
    if count != 1:
        raise RuntimeError(f"{label}: expected one target, found {count}")
    return content.replace(old, new, 1)


# Make the primary migration itself current-only before applying it again.
primary = PRIMARY.read_text(encoding="utf-8")
primary = primary.replace(
    "# Canonical runner CLI, resolved config, startup report, and exact old-checkpoint migration.\n",
    "# Canonical runner CLI, resolved config and startup report.\n",
)
primary = remove_between(
    primary,
    '_replace_once(\n    "run_thog2_owt_core.py",\n    \'    stored = TrainingConfig(**payload["trainer_config"])\\n\',\n',
    ')\n_replace_once(\n    "run_thog2_owt_core.py",\n    \'        "layer_dropout_stratum_size",',
    label="primary runner compatibility block",
)
primary = primary.replace(
    ')\n_replace_once(\n    "run_thog2_owt_core.py",\n    \'        "layer_dropout_stratum_size",',
    '_replace_once(\n    "run_thog2_owt_core.py",\n    \'        "layer_dropout_stratum_size",',
    1,
)
primary = remove_between(
    primary,
    "# Shared checkpoint resume path gets the same legacy-256 migration before TrainingConfig construction.\n",
    "# Wrapper: public default/usage, both long-option forms, validation and forwarding.\n",
    label="primary shared checkpoint compatibility block",
)
primary += "" if primary.endswith("\n") else "\n"
if "# Wrapper: public default/usage, both long-option forms, validation and forwarding.\n" not in primary:
    # The range removal deliberately consumed the wrapper marker; restore it immediately before its first operation.
    marker = '_insert_after(\n    "train_OWT_core.sh",\n    \'PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS=""\\n\','
    index = primary.find(marker)
    if index < 0:
        raise RuntimeError("primary wrapper insertion anchor not found after compatibility removal")
    primary = primary[:index] + "# Wrapper: public default/usage, both long-option forms, validation and forwarding.\n" + primary[index:]

# Capacity is a learned-count probe constraint, not a dormant global batch-size constraint.
primary = primary.replace(
    "        if self.plastic__layer_count_probe__number_of_sampled_valid_tokens > probe_token_capacity:\\n",
    "        if (\\n"
    "            self.plastic__enabled\\n"
    "            and self.plastic__do_learn_layer_count\\n"
    "            and self.plastic__layer_count_probe__number_of_sampled_valid_tokens > probe_token_capacity\\n"
    "        ):\\n",
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

# Apply the clean primary implementation. Existing generated pieces are idempotent.
subprocess.run(
    ["python", str(PRIMARY.relative_to(ROOT))],
    cwd=ROOT,
    check=True,
)

# Remove any compatibility block emitted by an earlier invocation in this same job.
runner_path = ROOT / "run_thog2_owt_core.py"
runner = runner_path.read_text(encoding="utf-8")
runner = remove_between(
    runner,
    "    # vvv THOG v0.521 checkpoints written before the public probe-token knob retain their historical hard-coded 256-token semantics\n",
    "    # ^^^ THOG\n",
    label="runtime runner compatibility block",
)
if '    stored = TrainingConfig(**payload["trainer_config"])\n' not in runner:
    anchor = '    if "trainer_config" not in payload:\n        return\n'
    if anchor not in runner:
        raise RuntimeError("runtime validate_resume_controls anchor not found")
    runner = runner.replace(anchor, anchor + '    stored = TrainingConfig(**payload["trainer_config"])\n', 1)
runner_path.write_text(runner, encoding="utf-8")

resume_path = ROOT / "sheet/trainer_checkpoint_resume.py"
resume = resume_path.read_text(encoding="utf-8")
resume = remove_between(
    resume,
    "        # vvv THOG v0.521 preserve exact pre-knob PLASTIC resume semantics: missing probe-token field means the historical fixed 256-token sample\n",
    "        # ^^^ THOG\n",
    label="runtime trainer-resume compatibility block",
)
if '        checkpoint_config = TrainingConfig(**payload["trainer_config"])\n' not in resume:
    anchor = "        validate_plastic_depth_checkpoint_format(payload)\n        # ^^^ THOG\n"
    if anchor not in resume:
        raise RuntimeError("runtime trainer checkpoint-config anchor not found")
    resume = resume.replace(anchor, anchor + '        checkpoint_config = TrainingConfig(**payload["trainer_config"])\n', 1)
resume_path.write_text(resume, encoding="utf-8")

# Gate installed static-capacity checks to the mode that can actually execute count probes.
for relative_path in ("sheet/training_config.py", "sheet/run_config.py"):
    path = ROOT / relative_path
    content = path.read_text(encoding="utf-8")
    old = "        if self.plastic__layer_count_probe__number_of_sampled_valid_tokens > probe_token_capacity:\n"
    new = (
        "        if (\n"
        "            self.plastic__enabled\n"
        "            and self.plastic__do_learn_layer_count\n"
        "            and self.plastic__layer_count_probe__number_of_sampled_valid_tokens > probe_token_capacity\n"
        "        ):\n"
    )
    content = replace_exact_once(content, old, new, label=f"{relative_path} active-probe capacity gate")
    path.write_text(content, encoding="utf-8")

# Make capacity tests exercise active learned-count probing, while explicitly proving a dormant tiny config is unaffected.
test_path = ROOT / "tests/test_plastic_depth_probe_sampling_v0521.py"
tests = test_path.read_text(encoding="utf-8")
tests = tests.replace(
    '''    OwtRunConfig(\n        model_type="dense",\n        batch_size=2,\n        block_size=8,\n        plastic__layer_count_probe__number_of_sampled_valid_tokens=0,\n    )\n    OwtRunConfig(\n        model_type="dense",\n        batch_size=2,\n        block_size=8,\n        plastic__layer_count_probe__number_of_sampled_valid_tokens=16,\n    )\n''',
    '''    OwtRunConfig(\n        model_type="sheet",\n        n_layer=8,\n        o_depth=4,\n        batch_size=2,\n        block_size=8,\n        plastic__enabled=True,\n        plastic__do_learn_layer_count=True,\n        plastic__initial_layer_count=4,\n        plastic__max_permitted_layers=8,\n        plastic__layer_count_probe__number_of_sampled_valid_tokens=0,\n    )\n    OwtRunConfig(\n        model_type="sheet",\n        n_layer=8,\n        o_depth=4,\n        batch_size=2,\n        block_size=8,\n        plastic__enabled=True,\n        plastic__do_learn_layer_count=True,\n        plastic__initial_layer_count=4,\n        plastic__max_permitted_layers=8,\n        plastic__layer_count_probe__number_of_sampled_valid_tokens=16,\n    )\n    OwtRunConfig(\n        model_type="dense",\n        batch_size=1,\n        block_size=8,\n    )\n''',
)
tests = tests.replace(
    '''        OwtRunConfig(\n            model_type="dense",\n            batch_size=2,\n            block_size=8,\n            plastic__layer_count_probe__number_of_sampled_valid_tokens=value,\n        )\n''',
    '''        OwtRunConfig(\n            model_type="sheet",\n            n_layer=8,\n            o_depth=4,\n            batch_size=2,\n            block_size=8,\n            plastic__enabled=True,\n            plastic__do_learn_layer_count=True,\n            plastic__initial_layer_count=4,\n            plastic__max_permitted_layers=8,\n            plastic__layer_count_probe__number_of_sampled_valid_tokens=value,\n        )\n''',
)
test_path.write_text(tests, encoding="utf-8")

# Generated v0.521 specification contains only current semantics.
spec_path = ROOT / "docs/THOG2_PLASTIC_Requirements_Specification_v0.521.txt"
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

print("PLASTIC v0.521 finalized with current-only semantics and active-probe capacity validation.")
# ^^^ THOG
