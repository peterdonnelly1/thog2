# vvv THOG
from __future__ import annotations

from run_thog2_owt import add_console_tokens_per_second, format_console_progress_payload


def test_stage8_console_progress_adds_right_aligned_tokens_per_second() -> None:
    payload = add_console_tokens_per_second(
        {
            "completed_updates": 2,
            "consumed_tokens": 24576,
            "cumulative_training_seconds": 33.45872121897992,
            "gradient_norm": 3.1200876235961914,
            "learning_rate": 5.7142857142857135e-05,
            "training_loss": 10.883082071940104,
        }
    )
    formatted = format_console_progress_payload(payload)

    assert formatted["tok/s"] == "         735"
    assert formatted["completed_updates"] == "     2"
    assert formatted["consumed_tokens"] == "         24576"
    assert formatted["cumulative_training_seconds"] == "    33"
    assert formatted["gradient_norm"] == "   3.120"
    assert formatted["learning_rate"] == " 5.714e-05"
    assert formatted["training_loss"] == "  10.8831"


def test_stage8_console_progress_does_not_add_token_rate_without_elapsed_training_time() -> None:
    payload = add_console_tokens_per_second(
        {
            "event": "evaluation_completed",
            "consumed_tokens": 0,
            "training_loss": 10.0,
        }
    )

    assert "tok/s" not in payload
# ^^^ THOG
