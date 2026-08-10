# vvv THOG
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = "./plastic_depth_lookahead_wrapper_options.sh"
CORE = "./train_OWT_core.sh"


def _bash(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script, "bash", *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_wall_time_controls_are_routed_after_separator() -> None:
    result = _bash(
        'set -- "$@"; source "$1"; shift; printf "%s\\n" "$@"',
        HELPER,
        "-g",
        "RUN",
        "--plastic__wall_time_equivalent_time_gain_discount",
        "0.9",
        "--plastic__layer_count_probe_radius",
        "1",
        "--plastic__wall_time_equivalent_time_gain_loss_rate_window",
        "64",
        "--plastic__wall_time_equivalent_time_gain_loss_rate_min_observations",
        "16",
        "--plastic__layer_count__max_allowable_layer_change",
        "1",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "-g",
        "RUN",
        "--plastic__layer_count_probe_radius",
        "1",
        "--plastic__layer_count__max_allowable_layer_change",
        "1",
        "--",
        "--plastic__wall_time_equivalent_time_gain_discount",
        "0.9",
        "--plastic__wall_time_equivalent_time_gain_loss_rate_window",
        "64",
        "--plastic__wall_time_equivalent_time_gain_loss_rate_min_observations",
        "16",
    ]


def test_wall_time_controls_join_existing_extra_args_separator() -> None:
    result = _bash(
        'set -- "$@"; source "$1"; shift; printf "%s\\n" "$@"',
        HELPER,
        "-g",
        "RUN",
        "--plastic__wall_time_equivalent_time_gain_discount=0.9",
        "--",
        "--artifact-suffix",
        "X",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "-g",
        "RUN",
        "--",
        "--plastic__wall_time_equivalent_time_gain_discount=0.9",
        "--artifact-suffix",
        "X",
    ]


def test_routed_wall_time_controls_do_not_reach_getopts_as_long_options() -> None:
    result = _bash(
        'set -- "$@"; helper="$1"; core="$2"; shift 2; source "$helper"; source "$core" "$@"',
        HELPER,
        CORE,
        "--plastic__wall_time_equivalent_time_gain_discount",
        "0.9",
        "--plastic__wall_time_equivalent_time_gain_loss_rate_window",
        "64",
        "--plastic__wall_time_equivalent_time_gain_loss_rate_min_observations",
        "16",
        "-h",
    )
    assert result.returncode == 0, result.stderr
    assert "Unknown option: --" not in result.stderr
# ^^^ THOG
