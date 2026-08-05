# vvv THOG
from __future__ import annotations

import re
import unittest

from sheet import plastic_depth_console_cleanup_patch as cleanup


_PROBE_PATTERN = re.compile(
    r"probe_losses \[(?P<label>[^\]]+)\] = \[(?P<body>[^\]]+)\]"
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _render_probe_losses(body: str) -> str:
    source = f"probe_losses [L-1, L, L+1] = [{body}]"
    return _PROBE_PATTERN.sub(cleanup._colour_probe_losses, source)


class PlasticDepthProbeDeltaConsoleTests(unittest.TestCase):
    def test_edge_losses_are_replaced_by_candidate_minus_current_deltas(self) -> None:
        rendered = _render_probe_losses("  6.331,   6.329,   6.331")
        visible = _ANSI_ESCAPE.sub("", rendered)
        self.assertEqual(
            visible,
            "probe_losses [L-1, L, L+1] = [ +0.002,   6.329, +0.002]",
        )
        self.assertEqual(rendered.count(cleanup._GREEN), 2)

    def test_nonpositive_edge_deltas_are_not_green(self) -> None:
        rendered = _render_probe_losses("  6.327,   6.329,   6.328")
        visible = _ANSI_ESCAPE.sub("", rendered)
        self.assertEqual(
            visible,
            "probe_losses [L-1, L, L+1] = [ -0.002,   6.329, -0.001]",
        )
        self.assertNotIn(cleanup._GREEN, rendered)

    def test_zero_edge_delta_keeps_an_explicit_sign_without_green(self) -> None:
        rendered = _render_probe_losses("  6.329,   6.329,   6.329")
        visible = _ANSI_ESCAPE.sub("", rendered)
        self.assertEqual(
            visible,
            "probe_losses [L-1, L, L+1] = [ +0.000,   6.329, +0.000]",
        )
        self.assertNotIn(cleanup._GREEN, rendered)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
