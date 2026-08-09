# vvv THOG
"""Align v0.54 gradient diagnostics with the final v0.541 fat-arrow console glyphs."""

from __future__ import annotations

import re

from . import plastic_depth_theil_sen_kendall_patch as _gradient


_gradient._DIRECTION_MARKER = re.compile(
    r"(?P<spacing>\s+)(?P<marker>(?:⇩\|⇧|↓\|↑)\|\?)"
)
# ^^^ THOG
