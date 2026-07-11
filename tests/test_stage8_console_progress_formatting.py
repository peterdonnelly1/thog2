# vvv THOG
from __future__ import annotations

from run_thog2_owt import format_console_progress_payload


def test_stage8_console_progress_formats_optimizer_numbers_for_vertical_alignment() -> None:
    payload = format_console_progress_payload(
        {
            "completed_updates": 2,
            "consumed_tokens": 24576,
            "cumulative_training_seconds": 33.45872121897992,
            "gradient_norm": 3.1200876235961914,
            "learning_rate": 5.7142857142857135e-05,
            "training_loss": 10.883082071940104,
        }
    )
    assert payload["completed_updates"] == "     2"
    assert payload["consumed_tokens"] == "       24576"
    assert payload["cumulative_training_seconds"] == "    33"
    assert payload["gradient_norm"] == "   3.120"
    assert payload["learning_rate"] == " 5.714e-05"
    assert payload["training_loss"] == "  10.8831"


def test_stage8_console_progress_leaves_non_numeric_identity_fields_unmodified() -> None:
    payload = format_console_progress_payload(
        {
            "event": "optimizer_progress",
            "run_id": "RUN",
            "stage": 6,
        }
    )
    assert payload == {"event": "optimizer_progress", "run_id": "RUN", "stage": 6}
# ^^^ THOG
