# vvv THOG
from __future__ import annotations

from sheet import plastic_depth_directional_coherence_patch as directional


def test_sampled_follows_layers_with_exactly_one_tab_and_no_fixed_column_padding() -> None:
    source = (
        "T 4 070826-1540 00:00:31 loss= 10.2769 grad norm= 5.475 "
        "layers = 32\tprobe_losses [L-5 ... L+5] = [10.5, 10.4, 10.3] "
        "sampled = [1.0, 2.0, 3.0]"
    )

    moved = directional._move_sampled_after_layers(source)
    rendered = directional._align_sampled_to_minimum_tab_column(moved)

    layer_text = "layers = 32"
    sampled_text = "sampled = ["
    layer_end = rendered.index(layer_text) + len(layer_text)
    sampled_start = rendered.index(sampled_text)

    assert rendered[layer_end:sampled_start] == "\t"
    assert sampled_start < rendered.index("probe_losses")


# ^^^ THOG
