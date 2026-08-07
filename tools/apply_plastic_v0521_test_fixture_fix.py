#!/usr/bin/env python3
# vvv THOG
"""Apply PLASTIC v0.521 and migrate pre-existing tiny test fixtures to explicit legal probe-token samples."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

subprocess.run(
    ["python", "tools/apply_plastic_v0521_probe_sampling_and_console.py"],
    cwd=ROOT,
    check=True,
)

# vvv THOG the artifact-name fixture has 2 x 32 = 64 token positions and should not inherit the production 1024-token default
path = ROOT / "tests/test_plastic_cli_console_refinements.py"
content = path.read_text(encoding="utf-8")
addition = "        plastic__layer_count_probe__number_of_sampled_valid_tokens=64,\n"
if addition not in content:
    anchor = "        plastic__max_permitted_layers=32,\n"
    if content.count(anchor) != 1:
        raise RuntimeError(
            "expected exactly one tiny artifact-fixture max-layer anchor; "
            f"found {content.count(anchor)}"
        )
    content = content.replace(anchor, anchor + addition, 1)
    path.write_text(content, encoding="utf-8")
# ^^^ THOG

# vvv THOG shared PLASTIC unit-test training fixtures use the stage3 2 x 8 = 16 token microbatch unless a test overrides it
path = ROOT / "tests/test_plastic_depth.py"
content = path.read_text(encoding="utf-8")
addition = "        plastic__layer_count_probe__number_of_sampled_valid_tokens=16,\n"
if addition not in content:
    anchor = "        plastic__freeze_geometry_during_warmup=False,\n        depth_order=3,\n"
    if content.count(anchor) != 1:
        raise RuntimeError(
            "expected exactly one shared plastic_training_config fixture anchor; "
            f"found {content.count(anchor)}"
        )
    content = content.replace(
        anchor,
        "        plastic__freeze_geometry_during_warmup=False,\n" + addition + "        depth_order=3,\n",
        1,
    )
    path.write_text(content, encoding="utf-8")
# ^^^ THOG

# vvv THOG stage3 PLASTIC CPU fixtures explicitly sample their complete tiny microbatch; production fresh-run default remains 1024
path = ROOT / "tests/stage3_test_support.py"
content = path.read_text(encoding="utf-8")
old = "    values.update(overrides)\n    return TrainingConfig(**values)\n"
new = """    values.update(overrides)
    if (
        bool(values.get(\"plastic__enabled\", False))
        and \"plastic__layer_count_probe__number_of_sampled_valid_tokens\" not in overrides
    ):
        values[\"plastic__layer_count_probe__number_of_sampled_valid_tokens\"] = (
            int(values[\"batch_size\"]) * int(values[\"block_size\"])
        )
    return TrainingConfig(**values)
"""
if old in content:
    if content.count(old) != 1:
        raise RuntimeError(f"expected one stage3 fixture return anchor; found {content.count(old)}")
    content = content.replace(old, new, 1)
elif new not in content:
    raise RuntimeError("stage3 PLASTIC probe-token fixture correction was not found")
path.write_text(content, encoding="utf-8")
# ^^^ THOG

# vvv THOG the durable FINE audit must retain the v0.521 paired-token SE diagnostic for every candidate and zero for current L
path = ROOT / "tests/test_plastic_depth_audit.py"
content = path.read_text(encoding="utf-8")
anchor = "        assert required <= set(audit)\n"
addition = """        # vvv THOG v0.521 paired-token SE is durable candidate diagnostic state and must not disappear before audit emission
        assert all(
            \"paired_delta_standard_error\" in item
            for item in audit[\"score_table\"]
        )
        current_rows = [
            item
            for item in audit[\"score_table\"]
            if int(item[\"active_layers\"]) == int(audit[\"previous_count\"])
        ]
        assert len(current_rows) == 1
        assert current_rows[0][\"paired_delta_standard_error\"] == pytest.approx(0.0)
        # ^^^ THOG
"""
if addition not in content:
    if content.count(anchor) != 1:
        raise RuntimeError(f"expected one audit required-fields anchor; found {content.count(anchor)}")
    content = content.replace(anchor, anchor + addition, 1)
    path.write_text(content, encoding="utf-8")
# ^^^ THOG

# vvv THOG the deterministic-sampling regression now asserts the configured cardinality rather than the retired hard-coded 256
path = ROOT / "tests/test_plastic_depth_inline_probe.py"
content = path.read_text(encoding="utf-8")
old = "        assert first.numel() == 256\n"
new = "        assert first.numel() == trainer.config.plastic__layer_count_probe__number_of_sampled_valid_tokens\n"
if old in content:
    if content.count(old) != 1:
        raise RuntimeError(f"expected one retired 256-token assertion; found {content.count(old)}")
    content = content.replace(old, new, 1)
elif new not in content:
    raise RuntimeError("deterministic probe sample-count assertion was not found")
path.write_text(content, encoding="utf-8")
# ^^^ THOG

# vvv THOG v0.521 supersedes raw probe_losses in the compact operator vector and adds paired-token SE without changing controller semantics
path = ROOT / "docs/THOG2_PLASTIC_Requirements_Specification_v0.521.txt"
content = path.read_text(encoding="utf-8")
old = "Console probe vectors retain the absolute current-L probe loss in the centre, rendered bold white. Every L-k/L+k entry is rendered as signed candidate_loss minus L_loss; negative deltas are bold bright green and positive deltas are uncoloured. The public label is probe_Δloss while the positional L-r ... L+r label remains unchanged.\n"
new = old + "For each non-current candidate the implementation also records paired-token delta standard error SE = sample_stddev(candidate_token_loss - L_token_loss) / sqrt(n), combining sufficient statistics across DDP ranks. Current L records SE=0. This precision diagnostic is retained in candidate diagnostics/audit and does not alter median/MAD/z-score, directional coherence, brakes or movement.\n"
if new not in content:
    if content.count(old) != 1:
        raise RuntimeError(f"expected one v0.521 console-delta paragraph; found {content.count(old)}")
    content = content.replace(old, new, 1)

old = "• Arbitrary-radius console vectors render signed candidate-loss deltas around an absolute bold-white L; every negative delta is bold bright green.\n"
new = old + "• Every candidate diagnostic records paired-token delta standard error; current L records SE=0, and this diagnostic has no controller-decision role in Version 0.521.\n"
if new not in content:
    if content.count(old) != 1:
        raise RuntimeError(f"expected one v0.521 console acceptance bullet; found {content.count(old)}")
    content = content.replace(old, new, 1)

old = "WHAT DOES NOT CHANGE The selected PLASTIC objective still defines what “better” means. Full-radius probe losses/scores remain naked and inspectable. Existing per-offset paired histories, robust median/MAD scale, z-score ranking, latest-win gate, strict-majority gate, update brake, max_step and history reset remain conceptually intact except that readiness now means a complete configured history window rather than a separate min-probes threshold.\n"
new = "WHAT DOES NOT CHANGE The selected PLASTIC objective still defines what “better” means. Full-radius objective scores and raw candidate losses remain retained and auditable; Version 0.521 changes only the compact operator-facing loss vector to the hybrid probe_Δloss display. Existing per-offset paired histories, robust median/MAD scale, z-score ranking, latest-win gate, strict-majority gate, update brake, max_step and history reset remain conceptually intact except that readiness now means a complete configured history window rather than a separate min-probes threshold.\n"
if old in content:
    if content.count(old) != 1:
        raise RuntimeError(f"expected one stale WHAT DOES NOT CHANGE paragraph; found {content.count(old)}")
    content = content.replace(old, new, 1)
elif new not in content:
    raise RuntimeError("v0.521 auditable-raw-loss clarification was not found")

old = "OPERATOR VISIBILITY Normal probe rows show L/R/A=[L/R/A]/N=>D. With DEBUG>9 they additionally show per-offset win counts as wins L[…]/N; R[…]/N. sampled moves immediately after layers; changed sampled coordinate values are pale pink for one optimizer row; the current-L probe loss is bold white. The existing raw probe_losses and score_z remain visible.\n"
new = "OPERATOR VISIBILITY Normal probe rows show L/R/A=[L/R/A]/N=>D. With DEBUG>9 they additionally show per-offset win counts as wins L[…]/N; R[…]/N. sampled moves immediately after layers; changed sampled coordinate values are pale pink for one optimizer row. The probe_Δloss vector keeps current-L as absolute bold-white loss, renders every non-current entry as signed candidate_loss - L_loss, and renders negative deltas bold bright green. score_z remains visible; raw candidate losses remain retained in structured diagnostics/audit rather than duplicated in the compact operator vector.\n"
if old in content:
    if content.count(old) != 1:
        raise RuntimeError(f"expected one stale OPERATOR VISIBILITY paragraph; found {content.count(old)}")
    content = content.replace(old, new, 1)
elif new not in content:
    raise RuntimeError("v0.521 operator-visibility clarification was not found")
path.write_text(content, encoding="utf-8")
# ^^^ THOG

print("Applied PLASTIC v0.521, migrated tiny fixtures, locked SE audit retention, and reconciled the v0.521 spec.")
# ^^^ THOG