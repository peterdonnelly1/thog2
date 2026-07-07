# vvv THOG
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from .basis import BASIS_VERSION
from .basis_kernel import BASIS_FAMILY_DCT as KERNEL_BASIS_FAMILY_DCT, DCT_BASIS_VERSION, basis_version_for_family


GEOMETRY_PRESET_LEGACY_SHEET_COL = "legacy_sheet_col"
GEOMETRY_PRESET_CURVE = "curve"
GEOMETRY_PRESET_MLP_BLOCK = "mlp_block"
GEOMETRY_PRESET_HEAD_AWARE_BLOCK = "head_aware_block"
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
BASIS_FAMILY_DCT = KERNEL_BASIS_FAMILY_DCT
BASIS_FAMILY_CONVENTIONAL = "conventional"

LEGACY_SHEET_COL_MATERIALIZATION_VERSION = "legacy_sheet_col_v1"
CURVE_MATERIALIZATION_VERSION = "curve_v1"
MLP_BLOCK_MATERIALIZATION_VERSION = "mlp_block_v1"
HEAD_AWARE_BLOCK_MATERIALIZATION_VERSION = "head_aware_block_v1"
BLOCK_MATERIALIZATION_VERSION = "block_v1"
COMPACT_MATERIALIZATION_VERSION = LEGACY_SHEET_COL_MATERIALIZATION_VERSION
CONVENTIONAL_MATERIALIZATION_VERSION = "conventional_dense_v1"

GEOMETRY_PRESETS = (GEOMETRY_PRESET_LEGACY_SHEET_COL, GEOMETRY_PRESET_CURVE, GEOMETRY_PRESET_MLP_BLOCK, GEOMETRY_PRESET_HEAD_AWARE_BLOCK, GEOMETRY_PRESET_BLOCK, GEOMETRY_PRESET_CONVENTIONAL)
ATTENTION_GEOMETRIES = (ATTENTION_GEOMETRY_LEGACY_SHEET_COL, ATTENTION_GEOMETRY_CURVE, ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK, ATTENTION_GEOMETRY_CONVENTIONAL)
MLP_GEOMETRIES = (MLP_GEOMETRY_LEGACY_SHEET_COL, MLP_GEOMETRY_CURVE, MLP_GEOMETRY_MLP_BLOCK, MLP_GEOMETRY_CONVENTIONAL)
BASIS_FAMILIES = (BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT, BASIS_FAMILY_CONVENTIONAL)

_PRESET_DEFAULTS: Mapping[str, Tuple[str, str]] = {
    GEOMETRY_PRESET_LEGACY_SHEET_COL: (ATTENTION_GEOMETRY_LEGACY_SHEET_COL, MLP_GEOMETRY_LEGACY_SHEET_COL),
    GEOMETRY_PRESET_CURVE: (ATTENTION_GEOMETRY_CURVE, MLP_GEOMETRY_CURVE),
    GEOMETRY_PRESET_MLP_BLOCK: (ATTENTION_GEOMETRY_CURVE, MLP_GEOMETRY_MLP_BLOCK),
    GEOMETRY_PRESET_HEAD_AWARE_BLOCK: (ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK, MLP_GEOMETRY_CURVE),
    GEOMETRY_PRESET_BLOCK: (ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK, MLP_GEOMETRY_MLP_BLOCK),
    GEOMETRY_PRESET_CONVENTIONAL: (ATTENTION_GEOMETRY_CONVENTIONAL, MLP_GEOMETRY_CONVENTIONAL),
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
    return normalized or None


def _require_member(name: str, value: Optional[str], allowed: Tuple[str, ...]) -> Optional[str]:
    normalized = _canonical_optional_string(name, value)
    if normalized is not None and normalized not in allowed:
        raise ValueError(f"{name} must be one of {allowed} or None; got {value!r}")
    return normalized


def resolve_compact_selectors(*, geometry_preset: Optional[str] = None, attention_geometry: Optional[str] = None, mlp_geometry: Optional[str] = None, basis_family: Optional[str] = None) -> ResolvedCompactSelectors:
    requested_preset = _require_member("geometry_preset", geometry_preset, GEOMETRY_PRESETS)
    requested_attention = _require_member("attention_geometry", attention_geometry, ATTENTION_GEOMETRIES)
    requested_mlp = _require_member("mlp_geometry", mlp_geometry, MLP_GEOMETRIES)
    requested_basis = _require_member("basis_family", basis_family, BASIS_FAMILIES)
    preset = requested_preset or GEOMETRY_PRESET_LEGACY_SHEET_COL
    default_attention, default_mlp = _PRESET_DEFAULTS[preset]
    resolved_attention = requested_attention or default_attention
    resolved_mlp = requested_mlp or default_mlp
    if requested_preset is None and resolved_attention == ATTENTION_GEOMETRY_CURVE and resolved_mlp == MLP_GEOMETRY_CURVE:
        preset = GEOMETRY_PRESET_CURVE
    if requested_preset is None and resolved_attention == ATTENTION_GEOMETRY_CURVE and resolved_mlp == MLP_GEOMETRY_MLP_BLOCK:
        preset = GEOMETRY_PRESET_MLP_BLOCK
    if requested_preset is None and resolved_attention == ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK and resolved_mlp == MLP_GEOMETRY_CURVE:
        preset = GEOMETRY_PRESET_HEAD_AWARE_BLOCK
    if requested_preset is None and resolved_attention == ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK and resolved_mlp == MLP_GEOMETRY_MLP_BLOCK:
        preset = GEOMETRY_PRESET_BLOCK
    resolved_basis = requested_basis or (BASIS_FAMILY_CONVENTIONAL if preset == GEOMETRY_PRESET_CONVENTIONAL else BASIS_FAMILY_CHEBYSHEV)
    if resolved_attention == ATTENTION_GEOMETRY_CONVENTIONAL or resolved_mlp == MLP_GEOMETRY_CONVENTIONAL:
        if preset != GEOMETRY_PRESET_CONVENTIONAL:
            raise ValueError("conventional module geometry requires geometry_preset='conventional'")
    if preset == GEOMETRY_PRESET_CONVENTIONAL and resolved_basis != BASIS_FAMILY_CONVENTIONAL:
        raise ValueError("conventional geometry must use basis_family='conventional' or None")
    return ResolvedCompactSelectors(requested_preset, requested_attention, requested_mlp, requested_basis, preset, resolved_attention, resolved_mlp, resolved_basis)


def compact_materialization_version(selectors: ResolvedCompactSelectors) -> str:
    if selectors.geometry_preset == GEOMETRY_PRESET_LEGACY_SHEET_COL:
        return LEGACY_SHEET_COL_MATERIALIZATION_VERSION
    if selectors.geometry_preset == GEOMETRY_PRESET_CURVE:
        return CURVE_MATERIALIZATION_VERSION
    if selectors.geometry_preset == GEOMETRY_PRESET_MLP_BLOCK:
        return MLP_BLOCK_MATERIALIZATION_VERSION
    if selectors.geometry_preset == GEOMETRY_PRESET_HEAD_AWARE_BLOCK:
        return HEAD_AWARE_BLOCK_MATERIALIZATION_VERSION
    if selectors.geometry_preset == GEOMETRY_PRESET_BLOCK:
        return BLOCK_MATERIALIZATION_VERSION
    raise ValueError(f"unsupported compact materialization preset: {selectors.geometry_preset!r}")


def normalize_compact_basis_version(selectors: ResolvedCompactSelectors, basis_version: str) -> str:
    if selectors.basis_family == BASIS_FAMILY_CONVENTIONAL:
        return basis_version
    expected_version = basis_version_for_family(selectors.basis_family)
    if selectors.basis_family == BASIS_FAMILY_DCT and basis_version == BASIS_VERSION:
        return DCT_BASIS_VERSION
    if basis_version != expected_version:
        raise ValueError(f"basis_version mismatch for basis_family={selectors.basis_family!r}: expected {expected_version!r}, got {basis_version!r}")
    return basis_version


def validate_current_sheet_support(selectors: ResolvedCompactSelectors) -> None:
    supported_basis = selectors.basis_family in (BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT)
    legacy = selectors.geometry_preset == GEOMETRY_PRESET_LEGACY_SHEET_COL and selectors.attention_geometry == ATTENTION_GEOMETRY_LEGACY_SHEET_COL and selectors.mlp_geometry == MLP_GEOMETRY_LEGACY_SHEET_COL and supported_basis
    curve = selectors.geometry_preset == GEOMETRY_PRESET_CURVE and selectors.attention_geometry == ATTENTION_GEOMETRY_CURVE and selectors.mlp_geometry == MLP_GEOMETRY_CURVE and supported_basis
    mlp_block = selectors.geometry_preset == GEOMETRY_PRESET_MLP_BLOCK and selectors.attention_geometry == ATTENTION_GEOMETRY_CURVE and selectors.mlp_geometry == MLP_GEOMETRY_MLP_BLOCK and supported_basis
    head_aware = selectors.geometry_preset == GEOMETRY_PRESET_HEAD_AWARE_BLOCK and selectors.attention_geometry == ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK and selectors.mlp_geometry == MLP_GEOMETRY_CURVE and supported_basis
    block = selectors.geometry_preset == GEOMETRY_PRESET_BLOCK and selectors.attention_geometry == ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK and selectors.mlp_geometry == MLP_GEOMETRY_MLP_BLOCK and supported_basis
    if legacy or curve or mlp_block or head_aware or block:
        return
    raise ValueError(
        "Stage 7 supports only legacy_sheet_col, curve, mlp_block, head_aware_block, or block materialization with chebyshev or dct basis; "
        f"got geometry_preset={selectors.geometry_preset!r}, attention_geometry={selectors.attention_geometry!r}, mlp_geometry={selectors.mlp_geometry!r}, basis_family={selectors.basis_family!r}"
    )


def validate_stage1_sheet_support(selectors: ResolvedCompactSelectors) -> None:
    validate_current_sheet_support(selectors)


def validate_dense_compact_fields(*, geometry_preset: Optional[str] = None, attention_geometry: Optional[str] = None, mlp_geometry: Optional[str] = None, basis_family: Optional[str] = None) -> None:
    selectors = resolve_compact_selectors(geometry_preset=geometry_preset or GEOMETRY_PRESET_CONVENTIONAL, attention_geometry=attention_geometry, mlp_geometry=mlp_geometry, basis_family=basis_family)
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
    role_ranges = {"query": (0, n_embd), "key": (n_embd, 2 * n_embd), "value": (2 * n_embd, 3 * n_embd)}
    role_head_ranges = {role_name: tuple((role_start + head_index * head_dim, role_start + (head_index + 1) * head_dim) for head_index in range(n_head)) for role_name, (role_start, _) in role_ranges.items()}
    output_columns = tuple((head_index * head_dim, (head_index + 1) * head_dim) for head_index in range(n_head))
    return {"head_dim": head_dim, "qkv_role_ranges": role_ranges, "qkv_role_head_ranges": role_head_ranges, "attention_output_input_head_column_ranges": output_columns}


def compact_identity_metadata(*, n_layer: int, n_embd: int, n_head: int, depth_order: int, base_row_order: int, basis_version: str = BASIS_VERSION, row_order_scaling_rule: str, geometry_preset: Optional[str] = None, attention_geometry: Optional[str] = None, mlp_geometry: Optional[str] = None, basis_family: Optional[str] = None, require_stage1_sheet_support: bool = True) -> Dict[str, Any]:
    selectors = resolve_compact_selectors(geometry_preset=geometry_preset, attention_geometry=attention_geometry, mlp_geometry=mlp_geometry, basis_family=basis_family)
    if require_stage1_sheet_support:
        validate_current_sheet_support(selectors)
    basis_version = normalize_compact_basis_version(selectors, basis_version)
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
        materialization_version=compact_materialization_version(selectors),
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
    return {"geometry_preset": GEOMETRY_PRESET_CONVENTIONAL, "attention_geometry": ATTENTION_GEOMETRY_CONVENTIONAL, "mlp_geometry": MLP_GEOMETRY_CONVENTIONAL, "basis_family": BASIS_FAMILY_CONVENTIONAL, "materialization_version": CONVENTIONAL_MATERIALIZATION_VERSION, "n_layer": n_layer, "n_embd": n_embd, "n_head": n_head, "head_dim": int(heads["head_dim"])}
# ^^^ THOG
