# vvv THOG
from __future__ import annotations

from sheet import plastic_depth_console_postfix_patch as postfix


def test_neutral_lra_renders_stet_without_changing_directional_state_text_elsewhere() -> None:
    line = "T 10 layers = 32\tsampled = [1.0, 2.0]  L/R/A=[1/0/1]/2=>-"
    assert postfix._finalize_plastic_postfixes(line) == (
        "T 10 layers = 32\tsampled = [1.0, 2.0]  L/R/A=[1/0/1]/2=>stet"
    )


def test_brake_annotation_moves_after_lra_summary() -> None:
    colour = "\033[38;2;150;220;255m"
    reset = "\033[0m"
    line = (
        "T 10 layers = 32\tsampled = [1.0, 2.0]  "
        f"{colour}<<< warmup braked enabled{reset}  "
        "L/R/A=[1/0/1]/2=>-"
    )
    rendered = postfix._finalize_plastic_postfixes(line)
    assert "L/R/A=[1/0/1]/2=>stet" in rendered
    assert rendered.endswith(f"{colour}<<< warmup brake enabled{reset}")
    assert rendered.index("L/R/A=") < rendered.index("<<< warmup brake enabled")


def test_update_brake_annotation_is_also_kept_at_physical_line_end() -> None:
    colour = "\033[38;2;255;150;150m"
    reset = "\033[0m"
    line = (
        "T 20 layers = 31\tsampled = [1.0, 2.0]  "
        f"{colour}<<< update brake on{reset}  "
        "L/R/A=[0/2/0]/2=>R"
    )
    rendered = postfix._finalize_plastic_postfixes(line)
    assert rendered.endswith(f"{colour}<<< update brake on{reset}")
    assert rendered.index("L/R/A=") < rendered.index("<<< update brake on")
# ^^^ THOG
