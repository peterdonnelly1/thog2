# vvv THOG
from __future__ import annotations

from sheet import plastic_depth_console_postfix_patch as _postfix  # noqa: F401
from sheet import plastic_depth_same_batch_visibility_patch as visibility


def test_probe_section_overwrites_sampled_vector_and_preserves_probe_tail() -> None:
    sampled = ", ".join(f"{value / 10.0:.1f}" for value in range(10, 700))
    probe_tail = (
        "P1  probe_Δloss [L-1, L, L+1] = [+0.002, 8.947, -0.002]  "
        "score_z [L-1, L+1] = [-, -]"
    )
    line = f"T     5  sampled = [{sampled}]  {probe_tail}"

    rendered = visibility._align_probe_section(line)
    plain = visibility._ANSI_ESCAPE.sub("", rendered).expandtabs(8)

    assert plain.index("P1  probe_Δloss") == 299
    assert plain.endswith(probe_tail)
    assert "sampled = [" in plain[:299]
    assert plain[297:299] == "  "
# ^^^ THOG
