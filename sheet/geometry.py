# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


MATRIX_FAMILY_NAMES = (
    "attention_input_weight",
    "attention_output_weight",
    "mlp_expansion_weight",
    "mlp_contraction_weight",
)


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def derive_row_order(row_width: int, model_width: int, base_row_order: int) -> int:
    """Apply Q_f = min(C_f, ceil(C_f * Q_d / d)) using integer arithmetic."""

    _require_positive_integer("row_width", row_width)
    _require_positive_integer("model_width", model_width)
    _require_positive_integer("base_row_order", base_row_order)
    if base_row_order > model_width:
        raise ValueError(
            f"base_row_order must not exceed model_width; "
            f"got base_row_order={base_row_order}, model_width={model_width}"
        )
    scaled_order = (row_width * base_row_order + model_width - 1) // model_width
    derived_order = min(row_width, scaled_order)
    if derived_order <= 0 or derived_order > row_width:
        raise RuntimeError(
            f"derived row order is invalid: row_width={row_width}, derived_order={derived_order}"
        )
    return derived_order


@dataclass(frozen=True)
class SheetGeometryConfig:
    n_layer: int
    n_embd: int
    n_head: int
    depth_order: int
    base_row_order: int
    bias: bool = True

    def __post_init__(self) -> None:
        _require_positive_integer("n_layer", self.n_layer)
        _require_positive_integer("n_embd", self.n_embd)
        _require_positive_integer("n_head", self.n_head)
        _require_positive_integer("depth_order", self.depth_order)
        _require_positive_integer("base_row_order", self.base_row_order)
        if self.depth_order > self.n_layer:
            raise ValueError(
                f"depth_order must not exceed n_layer; "
                f"got depth_order={self.depth_order}, n_layer={self.n_layer}"
            )
        if self.base_row_order > self.n_embd:
            raise ValueError(
                f"base_row_order must not exceed n_embd; "
                f"got base_row_order={self.base_row_order}, n_embd={self.n_embd}"
            )
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd must be divisible by n_head; "
                f"got n_embd={self.n_embd}, n_head={self.n_head}"
            )
        if not isinstance(self.bias, bool):
            raise ValueError(f"bias must be bool; got {self.bias!r}")


@dataclass(frozen=True)
class FamilyGeometry:
    name: str
    output_rows: int
    row_width: int
    row_order: int
    family_kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("family name must be a non-empty string")
        _require_positive_integer("output_rows", self.output_rows)
        _require_positive_integer("row_width", self.row_width)
        _require_positive_integer("row_order", self.row_order)
        if self.row_order > self.row_width:
            raise ValueError(
                f"row_order must not exceed row_width for {self.name}; "
                f"got row_order={self.row_order}, row_width={self.row_width}"
            )
        if self.family_kind not in ("matrix", "vector"):
            raise ValueError(
                f"family_kind must be 'matrix' or 'vector'; got {self.family_kind!r}"
            )
        if self.family_kind == "vector" and self.output_rows != 1:
            raise ValueError(
                f"vector family {self.name} must have exactly one output row; "
                f"got {self.output_rows}"
            )

    def coefficient_shape(self, depth_order: int) -> Tuple[int, int, int]:
        _require_positive_integer("depth_order", depth_order)
        return (self.output_rows, depth_order, self.row_order)

    def sheet_parameter_count(self, depth_order: int) -> int:
        rows, depth_terms, row_terms = self.coefficient_shape(depth_order)
        return rows * depth_terms * row_terms

    def dense_equivalent_count(self, n_layer: int) -> int:
        _require_positive_integer("n_layer", n_layer)
        return n_layer * self.output_rows * self.row_width


def _family(
    name: str,
    output_rows: int,
    row_width: int,
    family_kind: str,
    config: SheetGeometryConfig,
) -> FamilyGeometry:
    return FamilyGeometry(
        name=name,
        output_rows=output_rows,
        row_width=row_width,
        row_order=derive_row_order(row_width, config.n_embd, config.base_row_order),
        family_kind=family_kind,
    )


def transformer_family_geometries(
    config: SheetGeometryConfig,
    *,
    include_vectors: bool = True,
) -> Tuple[FamilyGeometry, ...]:
    """Return all repeated transformer-block tensor families in stable order."""

    width = config.n_embd
    families = [
        _family("attention_input_weight", 3 * width, width, "matrix", config),
        _family("attention_output_weight", width, width, "matrix", config),
        _family("mlp_expansion_weight", 4 * width, width, "matrix", config),
        _family("mlp_contraction_weight", width, 4 * width, "matrix", config),
    ]
    if not include_vectors:
        return tuple(families)

    families.extend(
        (
            _family("ln_1_weight", 1, width, "vector", config),
            _family("ln_2_weight", 1, width, "vector", config),
        )
    )
    if config.bias:
        families.extend(
            (
                _family("ln_1_bias", 1, width, "vector", config),
                _family("ln_2_bias", 1, width, "vector", config),
                _family("attention_input_bias", 1, 3 * width, "vector", config),
                _family("attention_output_bias", 1, width, "vector", config),
                _family("mlp_expansion_bias", 1, 4 * width, "vector", config),
                _family("mlp_contraction_bias", 1, width, "vector", config),
            )
        )
    return tuple(families)


def family_geometry_map(
    config: SheetGeometryConfig,
    *,
    include_vectors: bool = True,
) -> Dict[str, FamilyGeometry]:
    families = transformer_family_geometries(config, include_vectors=include_vectors)
    return {family.name: family for family in families}


def total_sheet_parameter_count(
    families: Iterable[FamilyGeometry],
    depth_order: int,
) -> int:
    _require_positive_integer("depth_order", depth_order)
    return sum(family.sheet_parameter_count(depth_order) for family in families)


def total_dense_equivalent_count(
    families: Iterable[FamilyGeometry],
    n_layer: int,
) -> int:
    _require_positive_integer("n_layer", n_layer)
    return sum(family.dense_equivalent_count(n_layer) for family in families)


def parameter_count_rows(
    config: SheetGeometryConfig,
    *,
    include_vectors: bool = True,
) -> Tuple[Dict[str, object], ...]:
    rows = []
    for family in transformer_family_geometries(config, include_vectors=include_vectors):
        rows.append(
            {
                "name": family.name,
                "family_kind": family.family_kind,
                "output_rows": family.output_rows,
                "row_width": family.row_width,
                "row_order": family.row_order,
                "coefficient_shape": family.coefficient_shape(config.depth_order),
                "sheet_parameters": family.sheet_parameter_count(config.depth_order),
                "dense_equivalent_parameters": family.dense_equivalent_count(config.n_layer),
            }
        )
    return tuple(rows)
# ^^^ THOG
