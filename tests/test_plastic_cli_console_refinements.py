# vvv THOG
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import run_thog2_owt_core as core
from sheet.plastic_depth_coarse import PlasticCoarseTrialResult, ScoredPlasticCoarseTrial, render_plastic_coarse_report
from sheet.plastic_depth_coarse_runner import render_plastic_coarse_trial_header
from sheet.run_config import OwtRunConfig


ROOT = Path(__file__).resolve().parents[1]


def test_core_parser_accepts_only_double_underscore_plastic_names() -> None:
    parser = core.build_parser()
    parsed = parser.parse_args(["--plastic__enabled", "--plastic__coarse_phase_roll_through"])
    assert parsed.plastic__enabled is True
    assert parsed.plastic__coarse_phase_roll_through is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--plastic-enabled"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--plastic_enabled"])


def test_wrapper_rejects_noncanonical_plastic_aliases() -> None:
    for alias in ("--plastic-enabled", "--plastic_enabled"):
        completed = subprocess.run(
            ("bash", "./train_OWT.sh", alias, "-h"),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 2
        assert "Non-canonical PLASTIC option rejected" in completed.stderr


def test_artifact_uses_dataset_first_and_one_plastic_group_prefix() -> None:
    config = OwtRunConfig(
        model_type="sheet",
        n_layer=32,
        n_head=4,
        n_embd=64,
        block_size=32,
        batch_size=2,
        gradient_accumulation_steps=3,
        max_iters=20,
        warmup_iters=2,
        plastic__enabled=True,
        plastic__coarse_phase="enabled",
        plastic__phase_1_starting_layer_count=4,
        plastic__phase_1_n_steps=20,
        plastic__phase_1__number_of_trials=4,
        plastic__phase_1_evaluation_steps_count=10,
        plastic__do_learn_layer_count=True,
        plastic__max_permitted_layers=32,
    )
    fragment = config.parameter_artifact_fragment()
    assert fragment.startswith("d_owt_A_3_b_2_c_60_f_6_w_2")
    assert "__P__LN_4_LM_32_LI_equ_LO_loss_LC_4_LCS_20_LCT_4_LCE_10" in fragment
    assert "PLN_" not in fragment
    assert "PLI_" not in fragment


def test_coarse_header_and_winner_marker_contract() -> None:
    header = render_plastic_coarse_trial_header(
        trial_index=1,
        trial_count=4,
        layers=4,
        n_steps=20,
        evaluation_steps_count=10,
        objective="lowest_loss",
        geometry_initialisation="equidistant",
    )
    assert header.startswith("TRIAL 1/4")
    assert "steps:       20" in header
    assert "starting at step" not in header

    result = PlasticCoarseTrialResult(
        trial_index=1,
        layers=4,
        status="success",
        validation_losses=(3.0,),
        training_losses=(4.0,),
        training_elapsed_seconds=1.0,
        training_steps=1,
        tokens_per_update=10,
    )
    winner = ScoredPlasticCoarseTrial(result, "lowest_loss", "loss_score", 3.0, True, None, None)
    report = render_plastic_coarse_report((winner,), winner, training_steps=1, evaluation_steps_count=1, ansi=True)
    winner_line = report.splitlines()[-1]
    assert winner_line.startswith("    1")
    assert not winner_line.startswith("\x1b[")
    assert "\x1b[1;92m<<< WINNER\x1b[0m" in winner_line


def test_defaults() -> None:
    parser = core.build_parser()
    values = parser.parse_args([])
    assert values.max_nonfinite_update_skips == 99999
    assert values.plastic__log_interval_coarse == 10
    assert values.plastic__coarse_phase_roll_through is False
# ^^^ THOG


def test_wrapper_help_covers_every_registered_plastic_option() -> None:
    parser = core.build_parser()
    completed = subprocess.run(
        ("bash", "./train_OWT.sh", "-h"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    registered = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--plastic__") or option.startswith("--no-plastic__")
    }
    missing = sorted(option for option in registered if option not in completed.stdout)
    assert missing == []
    assert "--max-nonfinite-update-skips" in completed.stdout


def test_warmup_brake_ends_at_the_actual_schedule_boundary() -> None:
    from types import SimpleNamespace
    from sheet import plastic_depth_console_minor_patch as console

    trainer = SimpleNamespace(
        config=SimpleNamespace(
            plastic__enabled=True,
            plastic__do_learn_layer_count=True,
            plastic__freeze_geometry_during_warmup=True,
            warmup_updates=100,
        )
    )
    # vvv THOG preserve the superseded post-update boundary assertions for source history
    # assert console._row_has_warmup_brake(trainer, 99)
    # assert not console._row_has_warmup_brake(trainer, 100)
    assert console._row_has_warmup_brake(trainer, 100)
    assert not console._row_has_warmup_brake(trainer, 101)
    # ^^^ THOG


def test_registered_help_is_generated_from_the_complete_parser() -> None:
    source = (ROOT / "train_OWT.sh").read_text(encoding="utf-8")
    assert "registered runner hyperparameters" in source
    assert "build_parser().format_help()" in source


def test_shell_core_owns_every_coarse_and_fine_control() -> None:
    source = (ROOT / "train_OWT_core.sh").read_text(encoding="utf-8")
    for option in (
        "--plastic__coarse_phase",
        "--plastic__phase_1_n_steps",
        "--plastic__phase_1_starting_layer_count",
        "--plastic__phase_1__number_of_trials",
        "--plastic__phase_1_evaluation_steps_count",
        "--plastic__layer_count_probe_window_size",
        "--plastic__layer_count_probe_radius",
        "--plastic__layer_count_max_step",
        "--plastic__log_interval_coarse",
        "--plastic__coarse_phase_roll_through",
    ):
        assert option in source
    helper = (ROOT / "plastic_depth_lookahead_wrapper_options.sh").read_text(encoding="utf-8")
    assert "LOOKAHEAD_EXTRA_ARGS" not in helper
    assert "Python-extra boundary" not in helper
