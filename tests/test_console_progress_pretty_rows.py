# vvv THOG
from __future__ import annotations

import re

from sheet.stage6_trainer import format_progress_line


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


def test_validation_console_row_uses_bold_yellow_v_and_keeps_run_id_last() -> None:
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

    assert line.startswith("\033[1;33mV\033[0m       2     196s  tok/s=          63")
    assert line.endswith("run_id=OPTIMO")
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
# ^^^ THOG
