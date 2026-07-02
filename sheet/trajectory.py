# vvv THOG
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional, Tuple

import torch
from torch import Tensor, nn

from .basis import BASIS_VERSION, BasisCache, BasisOwner
from .geometry import (
    FamilyGeometry,
    SheetGeometryConfig,
    total_dense_equivalent_count,
    total_sheet_parameter_count,
    transformer_family_geometries,
)


MATRIX_STANDARD_FAMILIES = {
    "attention_input_weight",
    "mlp_expansion_weight",
}
MATRIX_RESIDUAL_FAMILIES = {
    "attention_output_weight",
    "mlp_contraction_weight",
}
LAYERNORM_WEIGHT_FAMILIES = {
    "ln_1_weight",
    "ln_2_weight",
}


@dataclass(frozen=True)
class FamilyMetadata:
    geometry: FamilyGeometry
    semantic_type: str
    initialization: str
    target_weight_std: float
    weight_decay: bool

    @property
    def name(self) -> str:
        return self.geometry.name

    def coefficient_shape(self, depth_order: int) -> Tuple[int, int, int]:
        return self.geometry.coefficient_shape(depth_order)


def build_family_metadata(config: SheetGeometryConfig) -> Tuple[FamilyMetadata, ...]:
    metadata = []
    residual_std = 0.02 / math.sqrt(2.0 * config.n_layer)
    for geometry in transformer_family_geometries(config, include_vectors=True):
        if geometry.name in MATRIX_STANDARD_FAMILIES:
            metadata.append(
                FamilyMetadata(geometry, "matrix", "matrix_normal", 0.02, True)
            )
        elif geometry.name in MATRIX_RESIDUAL_FAMILIES:
            metadata.append(
                FamilyMetadata(geometry, "matrix", "matrix_normal", residual_std, True)
            )
        elif geometry.name in LAYERNORM_WEIGHT_FAMILIES:
            metadata.append(
                FamilyMetadata(geometry, "layernorm", "layernorm_one", 0.0, False)
            )
        elif geometry.name.endswith("_bias"):
            metadata.append(
                FamilyMetadata(geometry, "bias", "zero", 0.0, False)
            )
        else:
            raise ValueError(f"no semantic policy for sheet family {geometry.name}")
    return tuple(metadata)


class SheetTrajectory(nn.Module):
    """Compact coefficients and reproducible fixed bases for all block families."""

    def __init__(
        self,
        config: SheetGeometryConfig,
        *,
        runtime_dtype: torch.dtype = torch.float32,
        basis_version: str = BASIS_VERSION,
        basis_cache: Optional[BasisCache] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.runtime_dtype = runtime_dtype
        self.basis_version = basis_version
        self.metadata = build_family_metadata(config)
        self._metadata_by_name: Dict[str, FamilyMetadata] = {
            item.name: item for item in self.metadata
        }

        self.bases = BasisOwner(basis_cache)
        self.bases.add_basis(
            "depth_basis",
            config.n_layer,
            config.depth_order,
            runtime_dtype=runtime_dtype,
            version=basis_version,
        )

        self._row_basis_name_by_family: Dict[str, str] = {}
        distinct_row_bases: Dict[Tuple[int, int], str] = {}
        for item in self.metadata:
            key = (item.geometry.row_width, item.geometry.row_order)
            basis_name = distinct_row_bases.get(key)
            if basis_name is None:
                basis_name = f"row_basis_c{key[0]}_q{key[1]}"
                self.bases.add_basis(
                    basis_name,
                    key[0],
                    key[1],
                    runtime_dtype=runtime_dtype,
                    version=basis_version,
                )
                distinct_row_bases[key] = basis_name
            self._row_basis_name_by_family[item.name] = basis_name

        self.coefficients = nn.ParameterDict()
        for item in self.metadata:
            self.coefficients[item.name] = nn.Parameter(
                torch.empty(
                    item.coefficient_shape(config.depth_order),
                    dtype=runtime_dtype,
                )
            )
        self.reset_parameters()

    def family_metadata(self, name: str) -> FamilyMetadata:
        try:
            return self._metadata_by_name[name]
        except KeyError as error:
            raise KeyError(f"unknown sheet family: {name}") from error

    def row_basis(self, name: str) -> Tensor:
        self.family_metadata(name)
        return getattr(self.bases, self._row_basis_name_by_family[name])

    @property
    def depth_basis(self) -> Tensor:
        return self.bases.depth_basis

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for item in self.metadata:
                coefficient = self.coefficients[item.name]
                coefficient.zero_()
                if item.initialization == "matrix_normal":
                    coefficient_std = item.target_weight_std * math.sqrt(
                        self.config.n_layer
                        * item.geometry.row_width
                        / item.geometry.row_order
                    )
                    torch.nn.init.normal_(
                        coefficient[:, 0, :],
                        mean=0.0,
                        std=coefficient_std,
                    )
                elif item.initialization == "layernorm_one":
                    coefficient[0, 0, 0] = math.sqrt(
                        self.config.n_layer * item.geometry.row_width
                    )
                elif item.initialization == "zero":
                    continue
                else:
                    raise RuntimeError(
                        f"unsupported initialization policy {item.initialization} "
                        f"for {item.name}"
                    )

    def materialize(self, name: str, layer_index: int) -> Tensor:
        item = self.family_metadata(name)
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise ValueError(f"layer_index must be an integer; got {layer_index!r}")
        if layer_index < 0 or layer_index >= self.config.n_layer:
            raise IndexError(
                f"layer_index out of range: {layer_index}; n_layer={self.config.n_layer}"
            )
        coefficient = self.coefficients[name]
        depth_row = self.depth_basis[layer_index].to(
            device=coefficient.device,
            dtype=coefficient.dtype,
        )
        row_basis = self.row_basis(name).to(
            device=coefficient.device,
            dtype=coefficient.dtype,
        )
        mixed = torch.einsum("p,rpq->rq", depth_row, coefficient)
        generated = mixed @ row_basis.transpose(0, 1)
        expected_shape = (item.geometry.output_rows, item.geometry.row_width)
        if generated.shape != expected_shape:
            raise RuntimeError(
                f"materialized {name} has shape {tuple(generated.shape)}; "
                f"expected {expected_shape}"
            )
        return generated

    def materialize_vector(self, name: str, layer_index: int) -> Tensor:
        generated = self.materialize(name, layer_index)
        if generated.shape[0] != 1:
            raise ValueError(f"family {name} is not a vector family")
        return generated[0]

    def direct_value(
        self,
        name: str,
        layer_index: int,
        output_row: int,
        row_index: int,
    ) -> Tensor:
        item = self.family_metadata(name)
        if output_row < 0 or output_row >= item.geometry.output_rows:
            raise IndexError(f"output_row out of range for {name}: {output_row}")
        if row_index < 0 or row_index >= item.geometry.row_width:
            raise IndexError(f"row_index out of range for {name}: {row_index}")
        coefficient = self.coefficients[name][output_row]
        depth_row = self.depth_basis[layer_index].to(coefficient)
        row_value = self.row_basis(name)[row_index].to(coefficient)
        return depth_row @ coefficient @ row_value

    def named_semantic_parameters(
        self,
    ) -> Iterator[Tuple[str, nn.Parameter, FamilyMetadata]]:
        for item in self.metadata:
            yield item.name, self.coefficients[item.name], item

    def sheet_parameter_count(self) -> int:
        return total_sheet_parameter_count(
            (item.geometry for item in self.metadata),
            self.config.depth_order,
        )

    def dense_equivalent_count(self) -> int:
        return total_dense_equivalent_count(
            (item.geometry for item in self.metadata),
            self.config.n_layer,
        )

    def matrix_sheet_parameter_count(self) -> int:
        return total_sheet_parameter_count(
            (item.geometry for item in self.metadata if item.semantic_type == "matrix"),
            self.config.depth_order,
        )

    def matrix_dense_equivalent_count(self) -> int:
        return total_dense_equivalent_count(
            (item.geometry for item in self.metadata if item.semantic_type == "matrix"),
            self.config.n_layer,
        )

    def family_report(self) -> Tuple[Dict[str, object], ...]:
        rows = []
        for item in self.metadata:
            geometry = item.geometry
            rows.append(
                {
                    "name": item.name,
                    "semantic_type": item.semantic_type,
                    "initialization": item.initialization,
                    "target_weight_std": item.target_weight_std,
                    "weight_decay": item.weight_decay,
                    "output_rows": geometry.output_rows,
                    "row_width": geometry.row_width,
                    "row_order": geometry.row_order,
                    "coefficient_shape": item.coefficient_shape(self.config.depth_order),
                    "sheet_parameters": geometry.sheet_parameter_count(
                        self.config.depth_order
                    ),
                    "dense_equivalent_parameters": geometry.dense_equivalent_count(
                        self.config.n_layer
                    ),
                }
            )
        return tuple(rows)

    def persistent_basis_keys(self) -> Tuple[str, ...]:
        state_keys = set(self.state_dict().keys())
        return tuple(sorted(key for key in state_keys if key.startswith("bases.")))
# ^^^ THOG
