from __future__ import annotations

from sheet import help_registry_descriptor_patch as registry
from sheet import plastic_depth_v055_growth_side_help_patch as growth_help


def test_growth_side_discount_is_in_canonical_descriptor_registry() -> None:
    rendered = registry.format_descriptor_registry()

    assert growth_help._OPTION in rendered
    assert "beneficial growth-side objective evidence" in rendered
    assert "beneficial growth-side economic evidence" not in rendered
