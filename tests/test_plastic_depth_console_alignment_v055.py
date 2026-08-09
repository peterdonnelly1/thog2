# vvv THOG
from __future__ import annotations

from sheet import plastic_depth_console_alignment_v055_patch as alignment


def test_elapsed_runtime_field_keeps_seconds_and_decimal_hours_with_fixed_width() -> None:
    assert alignment._progress_elapsed_seconds_and_decimal_hours("4455", 12) == "   4455s    1.238h"
    assert len(alignment._progress_elapsed_seconds_and_decimal_hours("6", 1)) == len(
        alignment._progress_elapsed_seconds_and_decimal_hours("4455", 12)
    )


def test_prefix_uses_time_only_and_fixed_step_elapsed_columns() -> None:
    short = alignment._format_progress_prefix(
        "T  7  260809:2236  34s  0.009  Δ=5.8s"
    )
    long = alignment._format_progress_prefix(
        "T  123  260809:2236  4455s  1.238  Δ=5.8s"
    )
    assert short.startswith("T      7  2236       34s    0.009h  Δ=5.8s")
    assert long.startswith("T    123  2236     4455s    1.238h  Δ=5.8s")
    assert short.index("2236") == long.index("2236")
    assert short.index("Δ=") == long.index("Δ=")
    assert "260809" not in short


def test_numeric_fields_restore_stable_visual_widths() -> None:
    one = alignment._restore_fixed_numeric_fields(
        "Δ=5.8s  tok/s=12776  toks=73,728  loss=11.4140  Δ=n/a  lr=9.000e-04  g/n=10.746  layers 22  sampled [1.0]"
    )
    two = alignment._restore_fixed_numeric_fields(
        "Δ=15.3s  tok/s=999  toks=1,474,560  loss=9.5728  \x1b[1;32mΔ=-0.597\x1b[0m  lr=4.200e-05  g/n=0.242  layers 7  sampled [1.0]"
    )
    assert "Δ=  5.8s" in one
    assert "tok/s= 12776" in one
    assert "toks=     73,728" in one
    assert "loss= 11.4140" in one
    assert "Δ=     n/a" in one
    assert "lr= 9.000e-04" in one
    assert "g/n=  10.746" in one
    assert "layers  22" in one
    assert "Δ= 15.3s" in two
    assert "tok/s=   999" in two
    assert "toks=  1,474,560" in two
    assert "\x1b[1;32mΔ=  -0.597\x1b[0m" in two
    assert "g/n=   0.242" in two
    assert "layers   7" in two


def test_probe_block_moves_exactly_thirty_columns_right() -> None:
    original = (
        "T      7  2236       34s    0.009h  layers  22  sampled [1.0, 2.0]  "
        "P2  probe_Δloss [L-5 .. L+5] = [-0.035, 9.025]  sen=- ken=- adj=- ∴ ● (P1,2)"
    )
    shifted = alignment._shift_probe_block_right(original)
    assert shifted.index("P2  probe_Δloss") - original.index("P2  probe_Δloss") == 30
    assert shifted.endswith("sen=- ken=- adj=- ∴ ● (P1,2)")
# ^^^ THOG
