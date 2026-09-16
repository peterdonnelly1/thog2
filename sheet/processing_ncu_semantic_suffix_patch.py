# vvv THOG
"""Accept Nsight Compute 2024.x suffixes appended to THOG2 semantic NVTX labels."""

from __future__ import annotations

import re
from typing import Any, Dict

from . import processing_ncu_compatibility as _compat


_original_parse_processing_nvtx_label = _compat.parse_processing_nvtx_label
_semantic_label_pattern = re.compile(
    r"THOG2_PROCESSING\|"
    r"owner=(?P<owner>MAIN|PREMAT)\|"
    r"operation=(?P<operation>[^|/:\s]+)\|"
    r"family=(?P<family>[^|/:\s]+)\|"
    r"layer=(?P<layer>-?\d+)"
)


def parse_processing_nvtx_label_with_ncu_suffix(value: Any) -> Dict[str, Any] | None:
    """Parse THOG2 labels even when NCU appends ``/kernel`` or ``:none...``."""

    text = str(value or "")
    match = _semantic_label_pattern.search(text)
    if match is None:
        return _original_parse_processing_nvtx_label(value)
    return {
        "role": match.group("owner").upper(),
        "operation": match.group("operation").lower(),
        "family": match.group("family").upper(),
        "layer": int(match.group("layer")),
    }


_compat.parse_processing_nvtx_label = parse_processing_nvtx_label_with_ncu_suffix

# Install the final full-MAIN-headroom compatibility classes after NCU 2024
# normalization and semantic parsing are authoritative.
from . import processing_ncu_classification_patch as _processing_ncu_classification_patch  # noqa: E402,F401
# ^^^ THOG
