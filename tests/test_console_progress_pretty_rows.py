# vvv THOG
from __future__ import annotations

import re
from types import SimpleNamespace
from unittest import mock

from sheet.stage6_trainer import Stage6Trainer, format_progress_line


def test_training_console_row_puts_bare_update_first_and_bare_seconds_second() -> None:
    line = format_progress_line(
        "OPTIMO",
        "optimizer_progress",
        {
            "completed_updates": "     2",
            "consumed_tokens": "         24576",
            "training_loss": "  10.8831",
            "gradient_norm": "   3.120",
            "learning_rate": " 5.714e-05",
            "cumulative_training_seconds": "   196",
            "tok/s": "          63",
        },
    )

    assert line.startswith("T       2     196s  tok/s=          63")
    assert "updates=" not in line
    assert "cum time" not in line
    assert line.endswith("run_id=OPTIMO")
    assert "{" not in line
    assert "}" not in line
    assert '"' not in line
    assert line.index("learning rate=") < line.index("gradient norm=")
    assert "cum tokens=         24576" in line


def test_validation_console_row_is_entirely_bold_yellow_and_keeps_run_id_last() -> None:
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

    assert line.startswith("\033[1;33mV       2     196s  tok/s=          63")
    assert line.endswith("run_id=OPTIMO\033[0m")
    assert line.count("\033[1;33m") == 1
    assert line.count("\033[0m") == 1
    assert "updates=" not in line
    assert "cum time" not in line
    assert "cum tokens=         24576" in line
    assert "validation loss=  10.7777" in line


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
    fake_trainer = SimpleNamespace(distributed=SimpleNamespace(is_primary=True))
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
