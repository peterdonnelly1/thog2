# vvv THOG
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch
from torch import Tensor, nn

from .basis import BASIS_FAMILY_CHEBYSHEV, BASIS_VERSION, BasisOwner
from .curve_trajectory import CurveTrajectory
from .geometry import SheetGeometryConfig, derive_row_order
from .semantic_materializer import (
    ATTENTION_KEY_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    LEGACY_ATTENTION_INPUT_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)


BLOCK_MATERIALIZATION_VERSION = "block_v1"
BLOCK_MATRIX_FAMILIES = (
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_KEY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    MLP_EXPANSION_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
)


@dataclass(frozen=True)
class BlockMetadata:
    name: str
    semantic_type: str
    initialization: str
    target_weight_std: float
    weight_decay: bool
    output_rows: int
    row_width: int
    row_order: int
    output_order: int
    input_order: int

    def coefficient_shape(self, depth_order: int) -> Tuple[int, int, int]:
        return (depth_order, self.output_order, self.input_order)

    def sheet_parameter_count(self, depth_order: int) -> int:
        return depth_order * self.output_order * self.input_order

    def dense_equivalent_count(self, n_layer: int) -> int:
        return n_layer * self.output_rows * self.row_width


class BlockTrajectory(nn.Module):
    """CHEBY_BLOCK repeated matrix coefficients plus legacy sheet vectors."""

    def __init__(
        self,
        config: SheetGeometryConfig,
        *,
        runtime_dtype: torch.dtype = torch.float32,
        basis_version: str = BASIS_VERSION,
        basis_family: str = BASIS_FAMILY_CHEBYSHEV,
    ) -> None:
        super().__init__()
        if basis_family != BASIS_FAMILY_CHEBYSHEV:
            raise ValueError(f"BlockTrajectory supports only chebyshev basis; got {basis_family!r}")
        self.config = config
        self.runtime_dtype = runtime_dtype
        self.basis_version = basis_version
        self.basis_family = basis_family
        self.curve = CurveTrajectory(
            config,
            runtime_dtype=runtime_dtype,
            basis_version=basis_version,
            basis_family=basis_family,
        )
        for name in BLOCK_MATRIX_FAMILIES:
            del self.curve.coefficients[name]
        self.block_metadata = self._build_block_metadata()
        self._block_metadata_by_name = {item.name: item for item in self.block_metadata}
        self.bases = BasisOwner()
        self.bases.add_basis(
            "depth_basis",
            config.n_layer,
            config.depth_order,
            runtime_dtype=runtime_dtype,
            version=basis_version,
            basis_family=basis_family,
        )
        self._output_basis_names: Dict[str, str] = {}
        self._input_basis_names: Dict[str, str] = {}
        for item in self.block_metadata:
            output_name = f"block_output_basis_c{item.output_rows}_q{item.output_order}"
            input_name = f"block_input_basis_c{item.row_width}_q{item.input_order}"
            if not hasattr(self.bases, output_name):
                self.bases.add_basis(
                    output_name,
                    item.output_rows,
                    item.output_order,
                    runtime_dtype=runtime_dtype,
                    version=basis_version,
                    basis_family=basis_family,
                )
            if not hasattr(self.bases, input_name):
                self.bases.add_basis(
                    input_name,
                    item.row_width,
                    item.input_order,
                    runtime_dtype=runtime_dtype,
                    version=basis_version,
                    basis_family=basis_family,
                )
            self._output_basis_names[item.name] = output_name
            self._input_basis_names[item.name] = input_name
        self.coefficients = nn.ParameterDict({name: parameter for name, parameter in self.curve.coefficients.items()})
        for item in self.block_metadata:
            self.coefficients[item.name] = nn.Parameter(
                torch.empty(item.coefficient_shape(config.depth_order), dtype=runtime_dtype)
            )
        self.metadata = tuple(item for item in self.curve.metadata if item.name not in set(BLOCK_MATRIX_FAMILIES)) + self.block_metadata
        self._reset_block_parameters()

    def _build_block_metadata(self) -> Tuple[BlockMetadata, ...]:
        width = self.config.n_embd
        residual_std = 0.02 / math.sqrt(2.0 * self.config.n_layer)
        specs = (
            (ATTENTION_QUERY_WEIGHT, 0.02, width, width),
            (ATTENTION_KEY_WEIGHT, 0.02, width, width),
            (ATTENTION_VALUE_WEIGHT, 0.02, width, width),
            (ATTENTION_OUTPUT_WEIGHT, residual_std, width, width),
            (MLP_EXPANSION_WEIGHT, 0.02, 4 * width, width),
            (MLP_CONTRACTION_WEIGHT, residual_std, width, 4 * width),
        )
        metadata = []
        for name, target_std, output_rows, row_width in specs:
            output_order = derive_row_order(output_rows, width, self.config.base_row_order)
            input_order = derive_row_order(row_width, width, self.config.base_row_order)
            metadata.append(
                BlockMetadata(
                    name=name,
                    semantic_type="matrix",
                    initialization="block_matrix_normal",
                    target_weight_std=target_std,
                    weight_decay=True,
                    output_rows=output_rows,
                    row_width=row_width,
                    row_order=input_order,
                    output_order=output_order,
                    input_order=input_order,
                )
            )
        return tuple(metadata)

    @property
    def depth_basis(self) -> Tensor:
        return self.bases.depth_basis

    def family_metadata(self, name: str):
        if name in self._block_metadata_by_name:
            return self._block_metadata_by_name[name]
        return self.curve.family_metadata(name)

    def output_basis(self, name: str) -> Tensor:
        return getattr(self.bases, self._output_basis_names[name])

    def input_basis(self, name: str) -> Tensor:
        return getattr(self.bases, self._input_basis_names[name])

    def row_basis(self, name: str) -> Tensor:
        return self.curve.row_basis(name)

    def reset_parameters(self) -> None:
        self._reset_block_parameters()

    def _reset_block_parameters(self) -> None:
        with torch.no_grad():
            for item in self.block_metadata:
                coefficient = self.coefficients[item.name]
                coefficient.zero_()
                coefficient_std = item.target_weight_std * math.sqrt(
                    self.config.n_layer
                    * item.output_rows
                    * item.row_width
                    / (item.output_order * item.input_order)
                )
                torch.nn.init.normal_(coefficient[0], mean=0.0, std=coefficient_std)

    def _materialize_block(self, name: str, layer_index: int) -> Tensor:
        coefficient = self.coefficients[name]
        depth_row = self.depth_basis[layer_index].to(device=coefficient.device, dtype=coefficient.dtype)
        mixed = torch.einsum("p,pab->ab", depth_row, coefficient)
        generated = (
            self.output_basis(name).to(coefficient)
            @ mixed
            @ self.input_basis(name).to(coefficient).transpose(0, 1)
        )
        item = self.family_metadata(name)
        expected_shape = (item.output_rows, item.row_width)
        if tuple(generated.shape) != expected_shape:
            raise RuntimeError(f"block matrix {name} has shape {tuple(generated.shape)}; expected {expected_shape}")
        return generated

    def materialize(self, name: str, layer_index: int) -> Tensor:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise ValueError(f"layer_index must be an integer; got {layer_index!r}")
        if layer_index < 0 or layer_index >= self.config.n_layer:
            raise IndexError(f"layer_index out of range: {layer_index}; n_layer={self.config.n_layer}")
        if name == LEGACY_ATTENTION_INPUT_WEIGHT:
            return torch.cat(
                (
                    self._materialize_block(ATTENTION_QUERY_WEIGHT, layer_index),
                    self._materialize_block(ATTENTION_KEY_WEIGHT, layer_index),
                    self._materialize_block(ATTENTION_VALUE_WEIGHT, layer_index),
                ),
                dim=0,
            )
        if name in self._block_metadata_by_name:
            return self._materialize_block(name, layer_index)
        return self.curve.materialize(name, layer_index)

    def materialize_vector(self, name: str, layer_index: int) -> Tensor:
        return self.curve.materialize_vector(name, layer_index)

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
        if name not in self._block_metadata_by_name:
            return self.curve.direct_value(name, layer_index, output_row, row_index)
        item = self.family_metadata(name)
        if output_row < 0 or output_row >= item.output_rows:
            raise IndexError(f"output_row out of range for {name}: {output_row}")
        if row_index < 0 or row_index >= item.row_width:
            raise IndexError(f"row_index out of range for {name}: {row_index}")
        coefficient = self.coefficients[name]
        mixed = torch.einsum("p,pab->ab", self.depth_basis[layer_index].to(coefficient), coefficient)
        return self.output_basis(name)[output_row].to(coefficient) @ mixed @ self.input_basis(name)[row_index].to(coefficient)

    def named_semantic_parameters(self) -> Iterator[Tuple[str, nn.Parameter, object]]:
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
            row = {
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
            if isinstance(item, BlockMetadata):
                row["output_order"] = item.output_order
                row["input_order"] = item.input_order
            rows.append(row)
        return tuple(rows)

    def persistent_basis_keys(self) -> Tuple[str, ...]:
        return tuple(sorted(key for key in self.state_dict() if key.startswith("bases.") or key.startswith("curve.bases.")))
# ^^^ THOG
