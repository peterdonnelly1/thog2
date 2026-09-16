# vvv THOG
"""Align NCU compatibility classes with the documented full-MAIN-headroom semantics."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from . import processing_ncu_compatibility as _compat


_original_compatibility_row = _compat.compatibility_row


def compatibility_row_with_full_main_semantics(
    main_row: Mapping[str, Any],
    premat_row: Mapping[str, Any],
) -> Dict[str, Any]:
    row = dict(_original_compatibility_row(main_row, premat_row))
    pair = bool(row.get("pair_can_co_reside"))
    one_main = int(row.get("premat_blocks_with_one_main_block", 0) or 0)
    full_main = int(row.get("premat_blocks_with_full_main_residency", 0) or 0)

    if not pair:
        klass = "RED"
    elif full_main >= 1:
        # The defining GREEN condition in the design: at least one PREMAT block
        # still fits while MAIN retains its full theoretical block residency.
        klass = "GREEN"
    elif one_main <= 1:
        # Pair co-residency exists, but even after MAIN falls to a single block
        # only one PREMAT block can fit: this is the constrained ORANGE case.
        klass = "ORANGE"
    else:
        # Co-residency is available only after MAIN residency falls, but it is
        # less constrained than the one-plus-one-only case.
        klass = "YELLOW"

    row["compatibility_class"] = klass
    return row


_compat.compatibility_row = compatibility_row_with_full_main_semantics
# ^^^ THOG
