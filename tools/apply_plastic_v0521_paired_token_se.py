#!/usr/bin/env python3
# vvv THOG
"""Install the v0.521 diagnostic paired-token standard-error overlay and acceptance tests."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def insert_after(path: Path, anchor: str, addition: str) -> None:
    content = path.read_text(encoding="utf-8")
    if addition.strip() in content:
        return
    if content.count(anchor) != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {content.count(anchor)}")
    path.write_text(content.replace(anchor, anchor + addition, 1), encoding="utf-8")


# Install after the v0.521 sampler/display overlay so it observes the final sampled-token set while leaving controller logic untouched.
init_path = ROOT / "sheet/__init__.py"
anchor = (
    "# vvv THOG v0.521 makes probe-token sampling configurable and renders arbitrary-radius probe losses as signed deltas around bold-white L\n"
    "from . import plastic_depth_probe_sampling_v0521_patch as _plastic_depth_probe_sampling_v0521_patch\n"
    "# ^^^ THOG\n"
)
addition = (
    "\n# vvv THOG v0.521 report paired per-token delta standard errors as diagnostics only; robust MAD/z-score decisions are unchanged\n"
    "from . import plastic_depth_probe_se_v0521_patch as _plastic_depth_probe_se_v0521_patch\n"
    "# ^^^ THOG\n"
)
insert_after(init_path, anchor, addition)

# Add focused mathematical/installation coverage to the existing v0.521 test surface.
test_path = ROOT / "tests/test_plastic_depth_probe_sampling_v0521.py"
test_addition = r'''

# vvv THOG v0.521 paired-token SE is a diagnostic precision estimate and never participates in count selection
def test_paired_token_standard_error_uses_paired_deltas_and_sample_standard_deviation() -> None:
    from sheet import plastic_depth_probe_se_v0521_patch as probe_se

    counts = (6, 7, 8)
    current = torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float64)
    left = current + torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    right = current + torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
    local = probe_se._local_paired_delta_stats(
        counts=counts,
        current_count=7,
        token_losses=(left, current, right),
    )
    standard_errors = probe_se._combine_paired_delta_standard_errors(
        counts=counts,
        current_count=7,
        gathered_stats=(local,),
    )

    expected_left = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64).std(unbiased=True).item() / 2.0
    assert standard_errors[6] == pytest.approx(expected_left)
    assert standard_errors[7] == 0.0
    assert standard_errors[8] == pytest.approx(0.0)


def test_paired_token_se_overlay_is_installed_after_v0521_sampler() -> None:
    from sheet import plastic_depth_probe_se_v0521_patch as probe_se
    from sheet.training_model import TrainingSheetGPT

    assert TrainingSheetGPT._plastic_depth_candidate_head_loss.__module__ == probe_se.__name__
    assert TrainerStepMixin._plastic_depth_inline_probe_request.__module__ == probe_se.__name__
# ^^^ THOG
'''
content = test_path.read_text(encoding="utf-8")
if "test_paired_token_standard_error_uses_paired_deltas_and_sample_standard_deviation" not in content:
    test_path.write_text(content.rstrip() + test_addition.rstrip() + "\n", encoding="utf-8")

# Extend the v0.521 delta without changing the controller decision rule.
spec_path = ROOT / "docs/THOG2_PLASTIC_Requirements_Specification_v0.521.txt"
spec = spec_path.read_text(encoding="utf-8")
paragraph = (
    "Paired-token standard-error diagnostic. For each non-current candidate on a probe event, define per-token paired deltas d_i = candidate_token_loss_i − L_token_loss_i on the exact shared sampled positions. Report the diagnostic standard error SE = sample_stddev(d_i) / sqrt(n), combining sufficient statistics across DDP ranks. The current-L entry has SE=0. This is diagnostic only in Version 0.521: it must be recorded with candidate diagnostics but must not alter the established median/MAD/z-score, L/R/A, brake or movement rules. Because token losses within sequences are correlated, this SE is a useful precision diagnostic rather than a claim that token observations are independent.\n"
)
if paragraph not in spec:
    marker = (
        "All candidate layer counts in one probe event must continue to use exactly the same selected token positions. "
        "The sampler remains confined to the first probe microbatch; representativeness across training data is provided temporally by the multi-probe evidence window rather than by probing every gradient-accumulation microbatch.\n"
    )
    if spec.count(marker) != 1:
        raise RuntimeError(f"spec token-sampling marker count={spec.count(marker)}")
    spec = spec.replace(marker, marker + paragraph, 1)
    spec_path.write_text(spec, encoding="utf-8")

print("Installed PLASTIC v0.521 paired-token standard-error diagnostics.")
# ^^^ THOG
