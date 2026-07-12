# vvv THOG
from __future__ import annotations

import pytest

from sheet.compact_identity import (
    ATTENTION_GEOMETRY_DEPTH,
    ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
    ATTENTION_GEOMETRY_LEGACY_SHEET_COL,
    DEPTH_MATERIALIZATION_VERSION,
    FULL_BLOCK_MATERIALIZATION_VERSION,
    GEOMETRY_PRESET_CONVENTIONAL,
    GEOMETRY_PRESET_DEPTH,
    GEOMETRY_PRESET_FULL_BLOCK,
    GEOMETRY_PRESET_HEAD_AWARE_BLOCK,
    GEOMETRY_PRESET_LEGACY_SHEET_COL,
    GEOMETRY_PRESET_MLP_BLOCK,
    GEOMETRY_PRESETS,
    HEAD_AWARE_BLOCK_MATERIALIZATION_VERSION,
    LEGACY_SHEET_COL_MATERIALIZATION_VERSION,
    MLP_BLOCK_MATERIALIZATION_VERSION,
    MLP_GEOMETRY_DEPTH,
    MLP_GEOMETRY_LEGACY_SHEET_COL,
    MLP_GEOMETRY_MLP_BLOCK,
    compact_materialization_version,
    resolve_compact_selectors,
)


@pytest.mark.parametrize(
    ("preset", "attention_geometry", "mlp_geometry", "materialization_version"),
    (
        (
            GEOMETRY_PRESET_LEGACY_SHEET_COL,
            ATTENTION_GEOMETRY_LEGACY_SHEET_COL,
            MLP_GEOMETRY_LEGACY_SHEET_COL,
            LEGACY_SHEET_COL_MATERIALIZATION_VERSION,
        ),
        (
            GEOMETRY_PRESET_DEPTH,
            ATTENTION_GEOMETRY_DEPTH,
            MLP_GEOMETRY_DEPTH,
            DEPTH_MATERIALIZATION_VERSION,
        ),
        (
            GEOMETRY_PRESET_MLP_BLOCK,
            ATTENTION_GEOMETRY_DEPTH,
            MLP_GEOMETRY_MLP_BLOCK,
            MLP_BLOCK_MATERIALIZATION_VERSION,
        ),
        (
            GEOMETRY_PRESET_HEAD_AWARE_BLOCK,
            ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
            MLP_GEOMETRY_DEPTH,
            HEAD_AWARE_BLOCK_MATERIALIZATION_VERSION,
        ),
        (
            GEOMETRY_PRESET_FULL_BLOCK,
            ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
            MLP_GEOMETRY_MLP_BLOCK,
            FULL_BLOCK_MATERIALIZATION_VERSION,
        ),
    ),
)
def test_picton_final_preset_maps_to_exact_subsystem_geometries(
    preset: str,
    attention_geometry: str,
    mlp_geometry: str,
    materialization_version: str,
) -> None:
    selectors = resolve_compact_selectors(geometry_preset=preset)
    assert selectors.geometry_preset == preset
    assert selectors.attention_geometry == attention_geometry
    assert selectors.mlp_geometry == mlp_geometry
    assert compact_materialization_version(selectors) == materialization_version


def test_picton_preset_registry_contains_only_final_names_plus_internal_conventional() -> None:
    assert GEOMETRY_PRESETS == (
        GEOMETRY_PRESET_LEGACY_SHEET_COL,
        GEOMETRY_PRESET_DEPTH,
        GEOMETRY_PRESET_MLP_BLOCK,
        GEOMETRY_PRESET_HEAD_AWARE_BLOCK,
        GEOMETRY_PRESET_FULL_BLOCK,
        GEOMETRY_PRESET_CONVENTIONAL,
    )
    assert "curve" not in GEOMETRY_PRESETS
    assert "block" not in GEOMETRY_PRESETS


@pytest.mark.parametrize("retired_name", ("curve", "block"))
def test_picton_retired_preset_names_are_rejected_not_silently_aliased(retired_name: str) -> None:
    with pytest.raises(ValueError):
        resolve_compact_selectors(geometry_preset=retired_name)


def test_picton_selector_infers_named_preset_from_explicit_subsystem_pair() -> None:
    depth = resolve_compact_selectors(
        attention_geometry=ATTENTION_GEOMETRY_DEPTH,
        mlp_geometry=MLP_GEOMETRY_DEPTH,
    )
    assert depth.geometry_preset == GEOMETRY_PRESET_DEPTH

    mlp_block = resolve_compact_selectors(
        attention_geometry=ATTENTION_GEOMETRY_DEPTH,
        mlp_geometry=MLP_GEOMETRY_MLP_BLOCK,
    )
    assert mlp_block.geometry_preset == GEOMETRY_PRESET_MLP_BLOCK

    head_aware = resolve_compact_selectors(
        attention_geometry=ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
        mlp_geometry=MLP_GEOMETRY_DEPTH,
    )
    assert head_aware.geometry_preset == GEOMETRY_PRESET_HEAD_AWARE_BLOCK

    full_block = resolve_compact_selectors(
        attention_geometry=ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
        mlp_geometry=MLP_GEOMETRY_MLP_BLOCK,
    )
    assert full_block.geometry_preset == GEOMETRY_PRESET_FULL_BLOCK
# ^^^ THOG
