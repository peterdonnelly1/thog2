# vvv THOG
from __future__ import annotations

import json
import re
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


def test_feedback_table_layout_and_lockup_guards_are_present() -> None:
    source = (ROOT / "sheet/local_dashboard_assets/dashboard_sep07_fixes_and_enhancements.js").read_text(encoding="utf-8")
    legacy_source = (ROOT / "sheet/local_dashboard_assets/dashboard_aug30_enhancements_patch.js").read_text(encoding="utf-8")
    order_source = re.search(r"const table_column_order = Object\.freeze\(\[([\s\S]*?)\]\);", source)
    assert order_source is not None
    order = json.loads(f"[{order_source.group(1).rstrip().rstrip(',')}]")
    assert order == [
        "select", "visibility", "state", "duration", "name", "wandb", "host", "gpu",
        "preset", "optimizer", "steps", "gb", "warmup", "layers", "depth_order", "parms", "equiv",
        "context", "d_model", "heads", "grad_accum", "activation_checkpointing", "learning_rate",
        "min_learning_rate", "probe_start", "probe_end", "curve_start", "curve_end", "capture_period",
        "updated", "menu",
    ]
    assert "run-name-column-resizer" in source
    assert 'set_runs_pane_width((workspace_width - divider_width) / 2)' in source
    assert 'grid-template-columns: repeat(2, minmax(0, 1fr))' in source
    assert "instra-just-now-flash" not in source
    assert "updated_text_by_run" not in source
    assert '.observe(runs_table, {childList: true, subtree: true})' not in source
    assert '.observe(chart_root, {childList: true, subtree: true})' not in source
    assert 'run-name-column-resizer[data-instra-owner="sep07-feedback"]' in legacy_source
# ^^^ THOG
