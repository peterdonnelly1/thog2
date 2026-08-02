# vvv THOG
from __future__ import annotations

import re
from types import SimpleNamespace
from unittest import mock

from sheet.stage6_trainer import Stage6Trainer, format_progress_line


def test_training_console_row_uses_hh_mm_ss_after_step_one_and_colors_negative_delta_bright_green() -> None:
    line = format_progress_line(
        "OPTIMO",
        "optimizer_progress",
        {
            "completed_updates": "     2",
            "consumed_tokens": "         24576",
            "training_loss": "  10.8831",
            "training_loss_delta": "  -0.123",
            "gradient_norm": "   3.120",
            "learning_rate": " 5.714e-05",
            "cumulative_training_seconds": "   196",
            "tok/s": "          63",
        },
    )

    assert line.startswith("T       2  00:03:16  tok/s=          63")
    assert "\033[1;38;2;0;255;0mΔloss=  -0.123\033[0m" in line
    assert "\033[1;92m" not in line
    assert "updates=" not in line
    assert "cum time" not in line
    assert "run_id=" not in line
    assert "{" not in line
    assert "}" not in line
    assert '"' not in line
    assert line.index("learning rate=") < line.index("gradient norm=")
    assert "tokens=         24576" in line


def test_step_one_console_row_keeps_elapsed_seconds() -> None:
    line = format_progress_line(
        "OPTIMO",
        "optimizer_progress",
        {
            "completed_updates": "     1",
            "cumulative_training_seconds": "     9",
            "training_loss": "  10.8831",
            "training_loss_delta": "     n/a",
        },
    )
    assert line.startswith("T       1        9s")
    assert "00:00:09" not in line


def test_step_one_seconds_field_aligns_following_columns_with_hh_mm_ss_rows() -> None:
    common = {
        "mean_step_seconds": " 4.7500",
        "training_loss": "   6.0000",
        "training_loss_delta": "  -0.125",
    }
    step_one = format_progress_line(
        "OPTIMO",
        "optimizer_progress",
        {
            "completed_updates": "     1",
            "cumulative_training_seconds": "     5",
            **common,
        },
    )
    later = format_progress_line(
        "OPTIMO",
        "optimizer_progress",
        {
            "completed_updates": "     2",
            "cumulative_training_seconds": "    65",
            **common,
        },
    )
    plain_step_one = re.sub(r"\x1b\[[0-9;]*m", "", step_one)
    plain_later = re.sub(r"\x1b\[[0-9;]*m", "", later)
    assert plain_step_one.index("Δstep=") == plain_later.index("Δstep=")


def test_positive_delta_is_red_and_signed() -> None:
    line = format_progress_line(
        "OPTIMO",
        "optimizer_progress",
        {
            "completed_updates": "    10",
            "cumulative_training_seconds": "    87",
            "training_loss": "   6.0000",
            "training_loss_delta": "  +0.125",
        },
    )
    assert "\033[1;31mΔloss=  +0.125\033[0m" in line


def test_validation_console_row_uses_bold_explicit_rgb_yellow_for_validation_loss() -> None:
    line = format_progress_line(
        "OPTIMO",
        "evaluation_completed",
        {
            "completed_updates": "     2",
            "consumed_tokens": "         24576",
            "cumulative_training_seconds": "   196",
            "tok/s": "          63",
            "training_loss": "  10.8831",
            "validation_loss": "  10.7777",
        },
    )

    assert line.startswith("\033[33mV       2  00:03:16  tok/s=          63")
    assert line.endswith("\033[0m")
    assert line.count("\033[1;38;2;255;255;0m") == 1
    assert "\033[1;38;2;255;255;0mvalidation loss=  10.7777\033[33m" in line
    assert "\033[1;93m" not in line
    assert "updates=" not in line
    assert "cum time" not in line
    assert "tokens=         24576" in line


def test_training_and_validation_loss_numerals_start_in_the_same_column() -> None:
    line = format_progress_line(
        "OPTIMO",
        "evaluation_completed",
        {
            "completed_updates": "     2",
            "consumed_tokens": "         24576",
            "cumulative_training_seconds": "   196",
            "training_loss": "  10.8831",
            "validation_loss": "  10.7777",
        },
    )
    plain = re.sub(r"\x1b\[[0-9;]*m", "", line)
    training_value_column = plain.index("10.8831")
    validation_field_start = plain.index("validation loss")
    validation_value_column = plain.index("10.7777")
    training_field_start = plain.index("training loss")
    assert training_value_column - training_field_start == validation_value_column - validation_field_start


def test_run_started_console_output_is_followed_by_one_blank_line() -> None:
    fake_trainer = SimpleNamespace(
        distributed=SimpleNamespace(is_primary=True),
        state=SimpleNamespace(completed_updates=0),
    )
    with mock.patch("builtins.print") as print_spy:
        Stage6Trainer._print_progress(
            fake_trainer,
            "OPTIMO",
            "run_started",
            max_updates="   100",
            tokens_per_update="       12288",
        )
    assert print_spy.call_count == 2
    assert print_spy.call_args_list[0].kwargs == {"flush": True}
    assert print_spy.call_args_list[1].args == ()
    assert print_spy.call_args_list[1].kwargs == {"flush": True}
# ^^^ THOG
