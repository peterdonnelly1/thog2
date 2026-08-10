# vvv THOG
"""Register the v0.55 growth-side discount in the canonical PLASTIC help registry."""

from __future__ import annotations

from . import help_registry_descriptor_patch as _registry


_OPTION = "--plastic__layer_count_decision_algorithm__growth_side_discount VALUE"
_ROW = (
    "—",
    _OPTION,
    "v0.55 Sen/Kendall: credit fraction [0,1] of beneficial growth-side economic evidence; default 1.0",
)


def _descriptor_sections_with_growth_side_discount():
    sections = []
    for section, rows in _registry._DESCRIPTOR_SECTIONS:
        if section != "PLASTIC DEPTH":
            sections.append((section, tuple(rows)))
            continue
        updated_rows = list(rows)
        if not any(parameter == _OPTION for _abbreviation, parameter, _description in updated_rows):
            insertion_index = next(
                (
                    index + 1
                    for index, (_abbreviation, parameter, _description) in enumerate(updated_rows)
                    if parameter.startswith("--plastic__layer_count_objective")
                ),
                len(updated_rows),
            )
            updated_rows.insert(insertion_index, _ROW)
        sections.append((section, tuple(updated_rows)))
    return tuple(sections)


_registry._DESCRIPTOR_SECTIONS = _descriptor_sections_with_growth_side_discount()
# ^^^ THOG


__all__ = ["_descriptor_sections_with_growth_side_discount"]
