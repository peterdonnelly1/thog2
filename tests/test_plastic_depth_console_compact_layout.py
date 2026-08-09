# vvv THOG
from __future__ import annotations

from sheet import plastic_depth_console_compact_layout_patch as compact
from sheet import plastic_depth_theil_sen_kendall_patch as gradient


def test_elapsed_shows_seconds_then_fixed_width_decimal_hours() -> None:
    rendered = compact._progress_elapsed_decimal_hours("4455", 12)
    assert rendered == "   4455s    1.238"


def test_compact_progress_labels_and_one_decimal_step_seconds() -> None:
    rendered = compact._compact_progress_fields(
        "Δstep=  5.4972s  tokens=    73728  g nrm=   5.921"
    )
    assert "Δ=  5.5s" in rendered
    assert "toks=    73728" in rendered
    assert "g/n=  5.921" in rendered
    assert "Δstep=" not in rendered
    assert "5.4972s" not in rendered
    assert "tokens=" not in rendered
    assert "g nrm=" not in rendered


def test_final_sampled_column_is_four_left_of_current_layout() -> None:
    natural = "layers = 22\tsampled = [1.0, 2.0]"
    current = compact._pull_sampled_left_three_columns(natural)
    current_column = current.expandtabs(8).index("sampled =")
    rendered = compact._finalize_compact_progress_line(current)
    new_column = rendered.expandtabs(8).index("sampled [")
    assert new_column == current_column - 4


def test_final_progress_labels_drop_equals_signs() -> None:
    rendered = compact._finalize_compact_progress_line(
        "layers = 22    sampled = [1.0, 2.0]"
    )
    assert "layers 22" in rendered
    assert "sampled [1.0, 2.0]" in rendered
    assert "layers = " not in rendered
    assert "sampled = " not in rendered


def test_gradient_header_rows_show_both_new_controls(monkeypatch) -> None:
    monkeypatch.setenv(gradient._ALGORITHM_ENV, gradient.GRADIENT_ALGORITHM)
    monkeypatch.setenv(gradient._TAU_ENV, "0.5")
    rows = dict(compact._gradient_header_rows())
    assert rows["plastic__layer_count_decision_algorithm:"] == gradient.GRADIENT_ALGORITHM
    assert rows["plastic__layer_count_gradient__minimum_absolute_kendall_tau:"] == "0.5"
# ^^^ THOG
