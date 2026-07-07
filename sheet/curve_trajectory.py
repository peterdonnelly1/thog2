# vvv THOG
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch import Tensor, nn

from .basis import BASIS_FAMILY_CHEBYSHEV, BASIS_VERSION, BasisCache, BasisOwner
from .geometry import SheetGeometryConfig
from .semantic_materializer import (
    ATTENTION_KEY_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    LEGACY_ATTENTION_INPUT_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)
from .trajectory import build_family_metadata


CURVE_MATRIX_FAMILIES = (
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_KEY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    MLP_EXPANSION_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
)


@dataclass(frozen=True)
class CurveFamilyMetadata:
    name: str
    semantic_type: str
    initialization: str
    target_weight_std: float
    weight_decay: bool
    output_rows: int
    row_width: int
    row_order: int

    def coefficient_shape(self, depth_order: int) -> Tuple[int, ...]:
        if self.semantic_type == "matrix":
            return (self.output_rows, self.row_width, depth_order)
        return (self.output_rows, depth_order, self.row_order)

    def sheet_parameter_count(self, depth_order: int) -> int:
        count = 1
        for value in self.coefficient_shape(depth_order):
            count *= value
        return count

    def dense_equivalent_count(self, n_layer: int) -> int:
        return n_layer * self.output_rows * self.row_width


class CurveTrajectory(nn.Module):
    """CHEBY_CURVE repeated-matrix coefficients plus legacy sheet vectors."""

    def __init__(
        self,
        config: SheetGeometryConfig,
        *,
        runtime_dtype: torch.dtype = torch.float32,
        basis_version: str = BASIS_VERSION,
        basis_cache: Optional[BasisCache] = None,
        basis_family: str = BASIS_FAMILY_CHEBYSHEV,
    ) -> None:
        super().__init__()
        if basis_family != BASIS_FAMILY_CHEBYSHEV:
            raise ValueError(f"CurveTrajectory supports only chebyshev basis; got {basis_family!r}")
        self.config = config
        self.runtime_dtype = runtime_dtype
        self.basis_version = basis_version
        self.basis_family = basis_family
        self.metadata = self._build_metadata()
        self._metadata_by_name: Dict[str, CurveFamilyMetadata] = {item.name: item for item in self.metadata}

        self.bases = BasisOwner(basis_cache)
        self.bases.add_basis(
            "depth_basis",
            config.n_layer,
            config.depth_order,
            runtime_dtype=runtime_dtype,
            version=basis_version,
            basis_family=basis_family,
        )
        self._row_basis_name_by_family: Dict[str, str] = {}
        distinct_row_bases: Dict[Tuple[int, int], str] = {}
        for item in self.metadata:
            if item.semantic_type == "matrix":
                continue
            key = (item.row_width, item.row_order)
            basis_name = distinct_row_bases.get(key)
            if basis_name is None:
                basis_name = f"row_basis_c{key[0]}_q{key[1]}"
                self.bases.add_basis(
                    basis_name,
                    key[0],
                    key[1],
                    runtime_dtype=runtime_dtype,
                    version=basis_version,
                    basis_family=basis_family,
                )
                distinct_row_bases[key] = basis_name
            self._row_basis_name_by_family[item.name] = basis_name

        self.coefficients = nn.ParameterDict()
        for item in self.metadata:
            self.coefficients[item.name] = nn.Parameter(torch.empty(item.coefficient_shape(config.depth_order), dtype=runtime_dtype))
        self.reset_parameters()

    def _build_metadata(self) -> Tuple[CurveFamilyMetadata, ...]:
        width = self.config.n_embd
        residual_std = 0.02 / math.sqrt(2.0 * self.config.n_layer)
        rows = [
            CurveFamilyMetadata(ATTENTION_QUERY_WEIGHT, "matrix", "curve_matrix_normal", 0.02, True, width, width, width),
            CurveFamilyMetadata(ATTENTION_KEY_WEIGHT, "matrix", "curve_matrix_normal", 0.02, True, width, width, width),
            CurveFamilyMetadata(ATTENTION_VALUE_WEIGHT, "matrix", "curve_matrix_normal", 0.02, True, width, width, width),
            CurveFamilyMetadata(ATTENTION_OUTPUT_WEIGHT, "matrix", "curve_matrix_normal", residual_std, True, width, width, width),
            CurveFamilyMetadata(MLP_EXPANSION_WEIGHT, "matrix", "curve_matrix_normal", 0.02, True, 4 * width, width, width),
            CurveFamilyMetadata(MLP_CONTRACTION_WEIGHT, "matrix", "curve_matrix_normal", residual_std, True, width, 4 * width, 4 * width),
        ]
        for item in build_family_metadata(self.config):
            if item.semantic_type == "matrix":
                continue
            geometry = item.geometry
            rows.append(
                CurveFamilyMetadata(
                    item.name,
                    item.semantic_type,
                    item.initialization,
                    item.target_weight_std,
                    item.weight_decay,
                    geometry.output_rows,
                    geometry.row_width,
                    geometry.row_order,
                )
            )
        return tuple(rows)

    def family_metadata(self, name: str) -> CurveFamilyMetadata:
        try:
            return self._metadata_by_name[name]
        except KeyError as error:
            raise KeyError(f"unknown curve family: {name}") from error

    @property
    def depth_basis(self) -> Tensor:
        return self.bases.depth_basis

    def row_basis(self, name: str) -> Tensor:
        self.family_metadata(name)
        return getattr(self.bases, self._row_basis_name_by_family[name])

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for item in self.metadata:
                coefficient = self.coefficients[item.name]
                coefficient.zero_()
                if item.initialization == "curve_matrix_normal":
                    coefficient_std = item.target_weight_std * math.sqrt(self.config.n_layer)
                    torch.nn.init.normal_(coefficient[:, :, 0], mean=0.0, std=coefficient_std)
                elif item.initialization == "layernorm_one":
                    coefficient[0, 0, 0] = math.sqrt(self.config.n_layer * item.row_width)
                elif item.initialization == "zero":
                    continue
                else:
                    raise RuntimeError(f"unsupported initialization policy {item.initialization} for {item.name}")

    def _materialize_curve_matrix(self, name: str, layer_index: int) -> Tensor:
        coefficient = self.coefficients[name]
        depth_row = self.depth_basis[layer_index].to(device=coefficient.device, dtype=coefficient.dtype)
        generated = torch.einsum("p,rcp->rc", depth_row, coefficient)
        item = self.family_metadata(name)
        expected_shape = (item.output_rows, item.row_width)
        if tuple(generated.shape) != expected_shape:
            raise RuntimeError(f"curve matrix {name} has shape {tuple(generated.shape)}; expected {expected_shape}")
        return generated

    def _materialize_vector_sheet(self, name: str, layer_index: int) -> Tensor:
        item = self.family_metadata(name)
        coefficient = self.coefficients[name]
        depth_row = self.depth_basis[layer_index].to(device=coefficient.device, dtype=coefficient.dtype)
        row_basis = self.row_basis(name).to(device=coefficient.device, dtype=coefficient.dtype)
        mixed = torch.einsum("p,rpq->rq", depth_row, coefficient)
        generated = mixed @ row_basis.transpose(0, 1)
        expected_shape = (item.output_rows, item.row_width)
        if tuple(generated.shape) != expected_shape:
            raise RuntimeError(f"vector sheet {name} has shape {tuple(generated.shape)}; expected {expected_shape}")
        return generated

    def materialize(self, name: str, layer_index: int) -> Tensor:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise ValueError(f"layer_index must be an integer; got {layer_index!r}")
        if layer_index < 0 or layer_index >= self.config.n_layer:
            raise IndexError(f"layer_index out of range: {layer_index}; n_layer={self.config.n_layer}")
        if name == LEGACY_ATTENTION_INPUT_WEIGHT:
            return torch.cat(
                (
                    self._materialize_curve_matrix(ATTENTION_QUERY_WEIGHT, layer_index),
                    self._materialize_curve_matrix(ATTENTION_KEY_WEIGHT, layer_index),
                    self._materialize_curve_matrix(ATTENTION_VALUE_WEIGHT, layer_index),
                ),
                dim=0,
            )
        item = self.family_metadata(name)
        if item.semantic_type == "matrix":
            return self._materialize_curve_matrix(name, layer_index)
        return self._materialize_vector_sheet(name, layer_index)

    def materialize_vector(self, name: str, layer_index: int) -> Tensor:
        generated = self.materialize(name, layer_index)
        if generated.shape[0] != 1:
            raise ValueError(f"family {name} is not a vector family")
        return generated[0]

    def direct_value(self, name: str, layer_index: int, output_row: int, row_index: int) -> Tensor:
        if name == LEGACY_ATTENTION_INPUT_WEIGHT:
            width = self.config.n_embd
            if output_row < width:
                return self.direct_value(ATTENTION_QUERY_WEIGHT, layer_index, output_row, row_index)
            if output_row < 2 * width:
                return self.direct_value(ATTENTION_KEY_WEIGHT, layer_index, output_row - width, row_index)
            if output_row < 3 * width:
                return self.direct_value(ATTENTION_VALUE_WEIGHT, layer_index, output_row - 2 * width, row_index)
            raise IndexError(f"output_row out of range for {name}: {output_row}")
        item = self.family_metadata(name)
        if output_row < 0 or output_row >= item.output_rows:
            raise IndexError(f"output_row out of range for {name}: {output_row}")
        if row_index < 0 or row_index >= item.row_width:
            raise IndexError(f"row_index out of range for {name}: {row_index}")
        if item.semantic_type == "matrix":
            coefficient = self.coefficients[name][output_row, row_index]
            depth_row = self.depth_basis[layer_index].to(coefficient)
            return depth_row @ coefficient
        coefficient = self.coefficients[name][output_row]
        depth_row = self.depth_basis[layer_index].to(coefficient)
        row_value = self.row_basis(name)[row_index].to(coefficient)
        return depth_row @ coefficient @ row_value

    def named_semantic_parameters(self) -> Iterator[Tuple[str, nn.Parameter, CurveFamilyMetadata]]:
        for item in self.metadata:
            yield item.name, self.coefficients[item.name], item

    def sheet_parameter_count(self) -> int:
        return sum(item.sheet_parameter_count(self.config.depth_order) for item in self.metadata)

    def dense_equivalent_count(self) -> int:
        return sum(item.dense_equivalent_count(self.config.n_layer) for item in self.metadata)

    def matrix_sheet_parameter_count(self) -> int:
        return sum(item.sheet_parameter_count(self.config.depth_order) for item in self.metadata if item.semantic_type == "matrix")

    def matrix_dense_equivalent_count(self) -> int:
        return sum(item.dense_equivalent_count(self.config.n_layer) for item in self.metadata if item.semantic_type == "matrix")

    def family_report(self) -> Tuple[Dict[str, object], ...]:
        rows = []
        for item in self.metadata:
            rows.append(
                {
                    "name": item.name,
                    "semantic_type": item.semantic_type,
                    "initialization": item.initialization,
                    "target_weight_std": item.target_weight_std,
                    "weight_decay": item.weight_decay,
                    "output_rows": item.output_rows,
                    "row_width": item.row_width,
                    "row_order": item.row_order,
                    "coefficient_shape": item.coefficient_shape(self.config.depth_order),
                    "sheet_parameters": item.sheet_parameter_count(self.config.depth_order),
                    "dense_equivalent_parameters": item.dense_equivalent_count(self.config.n_layer),
                }
            )
        return tuple(rows)

    def persistent_basis_keys(self) -> Tuple[str, ...]:
        state_keys = set(self.state_dict().keys())
        return tuple(sorted(key for key in state_keys if key.startswith("bases.")))
# ^^^ THOG
