# vvv THOG
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_final_dashboard_asset_is_registered_and_javascript_is_valid() -> None:
    launcher = (ROOT / "run_thog2_dashboard.py").read_text(encoding="utf-8")
    asset = ROOT / "sheet/local_dashboard_assets/dashboard_sep07_fixes_and_enhancements.js"
    assert '"dashboard_sep07_fixes_and_enhancements.js"' in launcher
    subprocess.run(("node", "--check", str(asset)), check=True)


def test_startup_table_colour_and_hover_contracts_are_present() -> None:
    source = (ROOT / "sheet/local_dashboard_assets/dashboard_sep07_fixes_and_enhancements.js").read_text(encoding="utf-8")
    for required in (
        'workspace_nav")?.click()',
        'train: false',
        'optimizer_momentum: true',
        'optimizer_scaling: true',
        'metric-chart-id="train/loss"',
        'input.readOnly = false',
        'label: "GB"',
        'label: "PARMS"',
        'label: "EQUIV"',
        'label: "C_p"',
        'label: "GPU"',
        'instra-curve-hover',
        'line.dash',
        'ordinary_text_size',
        'actions.lastElementChild !== maximize',
    ):
        assert required in source
    palette = (ROOT / "sheet/local_dashboard_assets/dashboard_instra_further_enhancements_patch.js").read_text(encoding="utf-8")
    assert '"#FF0000", "#00FF00", "#0000FF", "#00FFFF", "#FF00FF", "#FFFF00", "#000000", "#FFFFFF"' in palette


def test_group_grips_are_removed_from_all_group_markup() -> None:
    for relative_path in (
        "sheet/local_dashboard_assets/index.html",
        "sheet/local_dashboard_assets/dashboard_wandb_groups_patch.js",
        "sheet/local_dashboard_assets/dashboard_synthetic_groups_patch.js",
        "sheet/local_dashboard_assets/dashboard_group_stability_patch.js",
    ):
        assert "⠿" not in (ROOT / relative_path).read_text(encoding="utf-8")


def test_resume_commands_are_literal_final_wrapper_lines() -> None:
    fresh = (ROOT / "train_OWT_core.sh").read_text(encoding="utf-8")
    lifecycle = (ROOT / "resume_and_fork_OWT.sh").read_text(encoding="utf-8")
    expected = './train_OWT.sh --resume %q -n 40000 --host-label "$THOG2_HOST_LABEL"'
    assert expected in fresh
    assert expected in lifecycle
# ^^^ THOG
