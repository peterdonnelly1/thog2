# vvv THOG
from __future__ import annotations

from sheet.stage6_trainer import format_progress_line


def test_training_console_row_is_brace_free_aligned_and_orders_requested_fields() -> None:
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

    assert line.startswith("T  cum time=   196  tok/s=          63")
    assert line.endswith("run_id=OPTIMO")
    assert "{" not in line
    assert "}" not in line
    assert '"' not in line
    assert line.index("learning rate=") < line.index("gradient norm=")
    assert "cum tokens=         24576" in line


def test_validation_console_row_starts_v_and_keeps_run_id_last() -> None:
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

    assert line.startswith("V  cum time=   196  tok/s=          63")
    assert line.endswith("run_id=OPTIMO")
    assert "cum tokens=         24576" in line
    assert "validation loss=  10.7777" in line
# ^^^ THOG
