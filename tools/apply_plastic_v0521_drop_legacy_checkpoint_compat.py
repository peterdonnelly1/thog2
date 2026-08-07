#!/usr/bin/env python3
# vvv THOG
"""Finalize PLASTIC v0.521 as a current-only probe-sampling and console contract."""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "tools/apply_plastic_v0521_probe_sampling_and_console.py"


def remove_top_level_nodes_containing(source: str, needles: tuple[str, ...]) -> str:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    ranges: list[tuple[int, int]] = []
    for node in tree.body:
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue
        segment = "".join(lines[node.lineno - 1 : node.end_lineno])
        if any(needle in segment for needle in needles):
            ranges.append((node.lineno - 1, node.end_lineno))
    for start, end in reversed(ranges):
        del lines[start:end]
    return "".join(lines)


def replace_exact_once(content: str, old: str, new: str, *, label: str) -> str:
    count = content.count(old)
    if count == 0:
        if new in content:
            return content
        raise RuntimeError(f"{label}: replacement target not found")
    if count != 1:
        raise RuntimeError(f"{label}: expected one target, found {count}")
    return content.replace(old, new, 1)


def remove_runtime_marker_block(
    content: str,
    marker: str,
    replacement: str,
    *,
    label: str,
) -> str:
    start = content.find(marker)
    if start < 0:
        return content
    if content.find(marker, start + len(marker)) >= 0:
        raise RuntimeError(f"{label}: marker is not unique")
    end_marker = "# ^^^ THOG\n"
    end = content.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"{label}: closing THOG marker not found")
    end += len(end_marker)
    return content[:start] + replacement + content[end:]


def collapse_duplicate_capacity_guards(content: str) -> str:
    pattern = re.compile(
        r"(?P<block>"
        r"        # vvv THOG v0\.521 reject a requested sample larger than the physical first-microbatch token capacity\n"
        r".*?"
        r"        # \^\^\^ THOG\n"
        r")(?P=block)",
        re.DOTALL,
    )
    while True:
        content, count = pattern.subn(r"\g<block>", content, count=1)
        if count == 0:
            return content


# Remove obsolete checkpoint-compatibility generator calls structurally. Old PLASTIC checkpoints are intentionally out of scope.
primary = PRIMARY.read_text(encoding="utf-8")
primary = remove_top_level_nodes_containing(
    primary,
    (
        "historical hard-coded 256-token semantics",
        "historical fixed 256-token sample",
    ),
)
primary = primary.replace(
    "# Canonical runner CLI, resolved config, startup report, and exact old-checkpoint migration.\n",
    "# Canonical runner CLI, resolved config and startup report.\n",
)
primary = primary.replace(
    "# Shared checkpoint resume path gets the same legacy-256 migration before TrainingConfig construction.\n",
    "",
)
primary = primary.replace(
    '    "Checkpoint migration is semantic: a pre-v0.521 PLASTIC checkpoint that lacks the new field is interpreted as 256 sampled valid tokens, matching the previously hard-coded implementation.\\n"\n',
    "",
)
primary = primary.replace(
    '    "• Pre-v0.521 PLASTIC checkpoints lacking the field retain the historical 256-token meaning.\\n"\n',
    "",
)

# Keep the primary migration itself idempotent with the final active-probe capacity semantics.
primary = primary.replace(
    "        if self.plastic__layer_count_probe__number_of_sampled_valid_tokens > probe_token_capacity:\\n",
    "        if (\\n"
    "            self.plastic__enabled\\n"
    "            and self.plastic__do_learn_layer_count\\n"
    "            and self.plastic__layer_count_probe__number_of_sampled_valid_tokens > probe_token_capacity\\n"
    "        ):\\n",
)
ast.parse(primary)
PRIMARY.write_text(primary, encoding="utf-8")

# The preceding fixture migration already applied the primary migration to this workspace. Do not apply it a second time here.
# vvv THOG current-only finalization therefore cleans the generated workspace in place and remains idempotent across later runs
runner_path = ROOT / "run_thog2_owt_core.py"
runner = runner_path.read_text(encoding="utf-8")
runner = remove_runtime_marker_block(
    runner,
    "    # vvv THOG v0.521 checkpoints written before the public probe-token knob retain their historical hard-coded 256-token semantics\n",
    '    stored = TrainingConfig(**payload["trainer_config"])\n',
    label="runner compatibility block",
)
runner_path.write_text(runner, encoding="utf-8")

resume_path = ROOT / "sheet/trainer_checkpoint_resume.py"
resume = resume_path.read_text(encoding="utf-8")
resume = remove_runtime_marker_block(
    resume,
    "        # vvv THOG v0.521 preserve exact pre-knob PLASTIC resume semantics: missing probe-token field means the historical fixed 256-token sample\n",
    '        checkpoint_config = TrainingConfig(**payload["trainer_config"])\n',
    label="trainer-resume compatibility block",
)
resume_path.write_text(resume, encoding="utf-8")
# ^^^ THOG

# Gate installed static-capacity checks to active learned-count probing and collapse any duplicate emitted by an older two-apply workflow.
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
    content = collapse_duplicate_capacity_guards(content)
    path.write_text(content, encoding="utf-8")

# Capacity tests exercise active learned-count probing; dormant tiny configs remain unaffected.
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

# Generated specification contains only the current v0.521 contract.
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

print("PLASTIC v0.521 finalized idempotently with current-only semantics and one active-probe capacity guard.")
# ^^^ THOG
