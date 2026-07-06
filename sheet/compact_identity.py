# vvv THOG
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from .basis import BASIS_VERSION


GEOMETRY_PRESET_LEGACY_SHEET_COL = "legacy_sheet_col"
GEOMETRY_PRESET_CURVE = "curve"
GEOMETRY_PRESET_MLP_BLOCK = "mlp_block"
GEOMETRY_PRESET_BLOCK = "block"
GEOMETRY_PRESET_CONVENTIONAL = "conventional"

ATTENTION_GEOMETRY_LEGACY_SHEET_COL = "legacy_sheet_col"
ATTENTION_GEOMETRY_CURVE = "curve"
ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK = "head_aware_block"
ATTENTION_GEOMETRY_CONVENTIONAL = "conventional"

MLP_GEOMETRY_LEGACY_SHEET_COL = "legacy_sheet_col"
MLP_GEOMETRY_CURVE = "curve"
MLP_GEOMETRY_MLP_BLOCK = "mlp_block"
MLP_GEOMETRY_CONVENTIONAL = "conventional"

BASIS_FAMILY_CHEBYSHEV = "chebyshev"
BASIS_FAMILY_DCT = "dct"
BASIS_FAMILY_CONVENTIONAL = "conventional"

COMPACT_MATERIALIZATION_VERSION = "legacy_sheet_col_v1"
CONVENTIONAL_MATERIALIZATION_VERSION = "conventional_dense_v1"

GEOMETRY_PRESETS = (
    GEOMETRY_PRESET_LEGACY_SHEET_COL,
    GEOMETRY_PRESET_CURVE,
    GEOMETRY_PRESET_MLP_BLOCK,
    GEOMETRY_PRESET_BLOCK,
    GEOMETRY_PRESET_CONVENTIONAL,
)
ATTENTION_GEOMETRIES = (
    ATTENTION_GEOMETRY_LEGACY_SHEET_COL,
    ATTENTION_GEOMETRY_CURVE,
    ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
    ATTENTION_GEOMETRY_CONVENTIONAL,
)
MLP_GEOMETRIES = (
    MLP_GEOMETRY_LEGACY_SHEET_COL,
    MLP_GEOMETRY_CURVE,
    MLP_GEOMETRY_MLP_BLOCK,
    MLP_GEOMETRY_CONVENTIONAL,
)
BASIS_FAMILIES = (
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    BASIS_FAMILY_CONVENTIONAL,
)

_PRESET_DEFAULTS: Mapping[str, Tuple[str, str]] = {
    GEOMETRY_PRESET_LEGACY_SHEET_COL: (
        ATTENTION_GEOMETRY_LEGACY_SHEET_COL,
        MLP_GEOMETRY_LEGACY_SHEET_COL,
    ),
    GEOMETRY_PRESET_CURVE: (
        ATTENTION_GEOMETRY_CURVE,
        MLP_GEOMETRY_CURVE,
    ),
    GEOMETRY_PRESET_MLP_BLOCK: (
        ATTENTION_GEOMETRY_CURVE,
        MLP_GEOMETRY_MLP_BLOCK,
    ),
    GEOMETRY_PRESET_BLOCK: (
        ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
        MLP_GEOMETRY_MLP_BLOCK,
    ),
    GEOMETRY_PRESET_CONVENTIONAL: (
        ATTENTION_GEOMETRY_CONVENTIONAL,
        MLP_GEOMETRY_CONVENTIONAL,
    ),
}


@dataclass(frozen=True)
class ResolvedCompactSelectors:
    requested_geometry_preset: Optional[str]
    requested_attention_geometry: Optional[str]
    requested_mlp_geometry: Optional[str]
    requested_basis_family: Optional[str]
    geometry_preset: str
    attention_geometry: str
    mlp_geometry: str
    basis_family: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompactIdentity:
    requested_geometry_preset: Optional[str]
    requested_attention_geometry: Optional[str]
    requested_mlp_geometry: Optional[str]
    requested_basis_family: Optional[str]
    geometry_preset: str
    attention_geometry: str
    mlp_geometry: str
    basis_family: str
    basis_version: str
    materialization_version: str
    row_order_scaling_rule: str
    n_layer: int
    n_embd: int
    n_head: int
    head_dim: int
    depth_order: int
    base_row_order: int
    qkv_role_ranges: Dict[str, Tuple[int, int]]
    qkv_role_head_ranges: Dict[str, Tuple[Tuple[int, int], ...]]
    attention_output_input_head_column_ranges: Tuple[Tuple[int, int], ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _canonical_optional_string(name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None; got {value!r}")
    normalized = value.strip().lower()
    if not normalized:
        return None
    return normalized


def _require_member(name: str, value: Optional[str], allowed: Tuple[str, ...]) -> Optional[str]:
    normalized = _canonical_optional_string(name, value)
    if normalized is not None and normalized not in allowed:
        raise ValueError(f"{name} must be one of {allowed} or None; got {value!r}")
    return normalized


def resolve_compact_selectors(
    *,
    geometry_preset: Optional[str] = None,
    attention_geometry: Optional[str] = None,
    mlp_geometry: Optional[str] = None,
    basis_family: Optional[str] = None,
) -> ResolvedCompactSelectors:
    requested_preset = _require_member("geometry_preset", geometry_preset, GEOMETRY_PRESETS)
    requested_attention = _require_member("attention_geometry", attention_geometry, ATTENTION_GEOMETRIES)
    requested_mlp = _require_member("mlp_geometry", mlp_geometry, MLP_GEOMETRIES)
    requested_basis = _require_member("basis_family", basis_family, BASIS_FAMILIES)

    preset = requested_preset or GEOMETRY_PRESET_LEGACY_SHEET_COL
    default_attention, default_mlp = _PRESET_DEFAULTS[preset]
    resolved_attention = requested_attention or default_attention
    resolved_mlp = requested_mlp or default_mlp
    resolved_basis = requested_basis or (
        BASIS_FAMILY_CONVENTIONAL if preset == GEOMETRY_PRESET_CONVENTIONAL else BASIS_FAMILY_CHEBYSHEV
    )
    if resolved_attention == ATTENTION_GEOMETRY_CONVENTIONAL or resolved_mlp == MLP_GEOMETRY_CONVENTIONAL:
        if preset != GEOMETRY_PRESET_CONVENTIONAL:
            raise ValueError("conventional module geometry requires geometry_preset='conventional'")
    if preset == GEOMETRY_PRESET_CONVENTIONAL and resolved_basis != BASIS_FAMILY_CONVENTIONAL:
        raise ValueError("conventional geometry must use basis_family='conventional' or None")
    return ResolvedCompactSelectors(
        requested_geometry_preset=requested_preset,
        requested_attention_geometry=requested_attention,
        requested_mlp_geometry=requested_mlp,
        requested_basis_family=requested_basis,
        geometry_preset=preset,
        attention_geometry=resolved_attention,
        mlp_geometry=resolved_mlp,
        basis_family=resolved_basis,
    )


def validate_stage1_sheet_support(selectors: ResolvedCompactSelectors) -> None:
    if selectors.geometry_preset != GEOMETRY_PRESET_LEGACY_SHEET_COL:
        raise ValueError(
            "Stage 1 supports only legacy_sheet_col materialization; "
            f"got geometry_preset={selectors.geometry_preset!r}"
        )
    if selectors.attention_geometry != ATTENTION_GEOMETRY_LEGACY_SHEET_COL:
        raise ValueError(
            "Stage 1 supports only legacy_sheet_col attention geometry; "
            f"got attention_geometry={selectors.attention_geometry!r}"
        )
    if selectors.mlp_geometry != MLP_GEOMETRY_LEGACY_SHEET_COL:
        raise ValueError(
            "Stage 1 supports only legacy_sheet_col MLP geometry; "
            f"got mlp_geometry={selectors.mlp_geometry!r}"
        )
    if selectors.basis_family != BASIS_FAMILY_CHEBYSHEV:
        raise ValueError(
            "Stage 1 supports only chebyshev basis family; "
            f"got basis_family={selectors.basis_family!r}"
        )


def validate_dense_compact_fields(
    *,
    geometry_preset: Optional[str] = None,
    attention_geometry: Optional[str] = None,
    mlp_geometry: Optional[str] = None,
    basis_family: Optional[str] = None,
) -> None:
    selectors = resolve_compact_selectors(
        geometry_preset=geometry_preset or GEOMETRY_PRESET_CONVENTIONAL,
        attention_geometry=attention_geometry,
        mlp_geometry=mlp_geometry,
        basis_family=basis_family,
    )
    if selectors.geometry_preset != GEOMETRY_PRESET_CONVENTIONAL:
        raise ValueError("dense model_type rejects compact geometry_preset fields")
    if selectors.attention_geometry != ATTENTION_GEOMETRY_CONVENTIONAL:
        raise ValueError("dense model_type rejects compact attention_geometry fields")
    if selectors.mlp_geometry != MLP_GEOMETRY_CONVENTIONAL:
        raise ValueError("dense model_type rejects compact mlp_geometry fields")
    if selectors.basis_family != BASIS_FAMILY_CONVENTIONAL:
        raise ValueError("dense model_type rejects compact basis_family fields")


def head_metadata(n_embd: int, n_head: int) -> Dict[str, Any]:
    if n_embd % n_head != 0:
        raise ValueError(f"n_embd must be divisible by n_head; got n_embd={n_embd}, n_head={n_head}")
    head_dim = n_embd // n_head
    role_ranges = {
        "query": (0, n_embd),
        "key": (n_embd, 2 * n_embd),
        "value": (2 * n_embd, 3 * n_embd),
    }
    role_head_ranges = {
        role_name: tuple(
            (role_start + head_index * head_dim, role_start + (head_index + 1) * head_dim)
            for head_index in range(n_head)
        )
        for role_name, (role_start, _) in role_ranges.items()
    }
    output_columns = tuple(
        (head_index * head_dim, (head_index + 1) * head_dim)
        for head_index in range(n_head)
    )
    return {
        "head_dim": head_dim,
        "qkv_role_ranges": role_ranges,
        "qkv_role_head_ranges": role_head_ranges,
        "attention_output_input_head_column_ranges": output_columns,
    }


def compact_identity_metadata(
    *,
    n_layer: int,
    n_embd: int,
    n_head: int,
    depth_order: int,
    base_row_order: int,
    basis_version: str = BASIS_VERSION,
    row_order_scaling_rule: str,
    geometry_preset: Optional[str] = None,
    attention_geometry: Optional[str] = None,
    mlp_geometry: Optional[str] = None,
    basis_family: Optional[str] = None,
    require_stage1_sheet_support: bool = True,
) -> Dict[str, Any]:
    selectors = resolve_compact_selectors(
        geometry_preset=geometry_preset,
        attention_geometry=attention_geometry,
        mlp_geometry=mlp_geometry,
        basis_family=basis_family,
    )
    if require_stage1_sheet_support:
        validate_stage1_sheet_support(selectors)
    if basis_version != BASIS_VERSION:
        raise ValueError(f"unsupported basis_version: {basis_version}")
    heads = head_metadata(n_embd, n_head)
    return CompactIdentity(
        requested_geometry_preset=selectors.requested_geometry_preset,
        requested_attention_geometry=selectors.requested_attention_geometry,
        requested_mlp_geometry=selectors.requested_mlp_geometry,
        requested_basis_family=selectors.requested_basis_family,
        geometry_preset=selectors.geometry_preset,
        attention_geometry=selectors.attention_geometry,
        mlp_geometry=selectors.mlp_geometry,
        basis_family=selectors.basis_family,
        basis_version=basis_version,
        materialization_version=COMPACT_MATERIALIZATION_VERSION,
        row_order_scaling_rule=row_order_scaling_rule,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        head_dim=int(heads["head_dim"]),
        depth_order=depth_order,
        base_row_order=base_row_order,
        qkv_role_ranges=heads["qkv_role_ranges"],
        qkv_role_head_ranges=heads["qkv_role_head_ranges"],
        attention_output_input_head_column_ranges=heads["attention_output_input_head_column_ranges"],
    ).to_dict()


def conventional_identity_metadata(*, n_layer: int, n_embd: int, n_head: int) -> Dict[str, Any]:
    heads = head_metadata(n_embd, n_head)
    return {
        "geometry_preset": GEOMETRY_PRESET_CONVENTIONAL,
        "attention_geometry": ATTENTION_GEOMETRY_CONVENTIONAL,
        "mlp_geometry": MLP_GEOMETRY_CONVENTIONAL,
        "basis_family": BASIS_FAMILY_CONVENTIONAL,
        "materialization_version": CONVENTIONAL_MATERIALIZATION_VERSION,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": n_head,
        "head_dim": int(heads["head_dim"]),
    }
# ^^^ THOG
