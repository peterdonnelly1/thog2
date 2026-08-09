# vvv THOG
from __future__ import annotations

from sheet import plastic_depth_console_compact_layout_patch as compact
from sheet import plastic_depth_theil_sen_kendall_patch as gradient


def test_elapsed_is_fixed_width_decimal_hours() -> None:
    rendered = compact._progress_elapsed_decimal_hours("72", 12)
    assert rendered == "   0.020"
    assert len(rendered) == 8


def test_compact_progress_labels() -> None:
    rendered = compact._compact_progress_fields(
        "Δstep=  5.4972s  tokens=    73728  g nrm=   5.921"
    )
    assert "Δ=  5.4972s" in rendered
    assert "toks=    73728" in rendered
    assert "g/n=  5.921" in rendered
    assert "Δstep=" not in rendered
    assert "tokens=" not in rendered
    assert "g nrm=" not in rendered


def test_sampled_moves_left_exactly_three_visible_columns() -> None:
    original = "layers = 22\tsampled = [1.0, 2.0]"
    current_column = original.expandtabs(8).index("sampled =")
    rendered = compact._pull_sampled_left_three_columns(original)
    new_column = rendered.expandtabs(8).index("sampled =")
    assert new_column == current_column - 3


def test_gradient_header_rows_show_both_new_controls(monkeypatch) -> None:
    monkeypatch.setenv(gradient._ALGORITHM_ENV, gradient.GRADIENT_ALGORITHM)
    monkeypatch.setenv(gradient._TAU_ENV, "0.5")
    rows = dict(compact._gradient_header_rows())
    assert rows["plastic__layer_count_decision_algorithm:"] == gradient.GRADIENT_ALGORITHM
    assert rows["plastic__layer_count_gradient__minimum_absolute_kendall_tau:"] == "0.5"
# ^^^ THOG
