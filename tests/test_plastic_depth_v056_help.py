from __future__ import annotations

import argparse

from sheet import help_registry_descriptor_patch as registry
from sheet import plastic_depth_v056_help_patch as help_patch
from sheet import plastic_depth_v056_objective_decision_patch as v056


def test_help_registry_contains_v056_decision_section() -> None:
    section = next(
        rows for name, rows in registry._DESCRIPTOR_SECTIONS
        if name == help_patch._SECTION
    )
    text = "\n".join(" | ".join(row) for row in section)
    assert "all 4 objectives are permitted with all 3 decision algorithms" in text
    assert "directional_coherence" in text
    assert v056.LRA_ALGORITHM in text
    assert v056.STRATIFIED_ALGORITHM in text
    assert "Sen slope: sen < 0" in text
    assert "adding layers tends to improve the selected objective score" in text
    assert "Kendall tau ≈ -0.5" in text
    assert "minimum accepted indication toward adding layers" in text
    assert "Kendall tau ≈ +0.5" in text
    assert "minimum accepted indication toward removing layers" in text
    assert "fixed |tau| >= 0.5" in text
    assert "undiscounted objective-score delta adj < 0" in text
    assert "change_z/score_z" in text


def test_argparse_help_uses_objective_neutral_tsk_names() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plastic__enabled", action="store_true")
    rendered = parser.format_help()
    assert "wall_time__theil_sen_kendall_LRA" not in rendered
    assert "wall_time__sen_kendall__tau__stratified" not in rendered
    assert v056.LRA_ALGORITHM in rendered
    assert v056.STRATIFIED_ALGORITHM in rendered
