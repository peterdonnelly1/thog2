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
from .semantic_materializer import MLP_CONTRACTION_WEIGHT, MLP_EXPANSION_WEIGHT

MLP_BLOCK_MATERIALIZATION_VERSION = "mlp_block_v1"
MLP_BLOCK_MATRIX_FAMILIES = ("attention_query_weight", "attention_key_weight", "attention_value_weight", "attention_output_weight", MLP_EXPANSION_WEIGHT, MLP_CONTRACTION_WEIGHT)

@dataclass
class MlpBlockMetadata:
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

class MlpBlockTrajectory(nn.Module):
    def __init__(self, config: SheetGeometryConfig, *, runtime_dtype: torch.dtype = torch.float32, basis_version: str = BASIS_VERSION, basis_family: str = BASIS_FAMILY_CHEBYSHEV) -> None:
        super().__init__()
        self.config, self.runtime_dtype, self.basis_version, self.basis_family = config, runtime_dtype, basis_version, basis_family
        self.curve = CurveTrajectory(config, runtime_dtype=runtime_dtype, basis_version=basis_version, basis_family=basis_family)
        del self.curve.coefficients[MLP_EXPANSION_WEIGHT]
        del self.curve.coefficients[MLP_CONTRACTION_WEIGHT]
        width = config.n_embd
        self.mlp_metadata = (
            MlpBlockMetadata(MLP_EXPANSION_WEIGHT, "matrix", "mlp_block_matrix_normal", 0.02, True, 4 * width, width, width, derive_row_order(4 * width, width, config.base_row_order), derive_row_order(width, width, config.base_row_order)),
            MlpBlockMetadata(MLP_CONTRACTION_WEIGHT, "matrix", "mlp_block_matrix_normal", 0.02 / math.sqrt(2.0 * config.n_layer), True, width, 4 * width, 4 * width, derive_row_order(width, width, config.base_row_order), derive_row_order(4 * width, width, config.base_row_order)),
        )
        self._mlp_metadata_by_name = {item.name: item for item in self.mlp_metadata}
        self.bases = BasisOwner()
        self.bases.add_basis("depth_basis", config.n_layer, config.depth_order, runtime_dtype=runtime_dtype, version=basis_version, basis_family=basis_family)
        self._output_basis_names: Dict[str, str] = {}
        self._input_basis_names: Dict[str, str] = {}
        for item in self.mlp_metadata:
            output_name = f"mlp_output_basis_c{item.output_rows}_q{item.output_order}"
            input_name = f"mlp_input_basis_c{item.row_width}_q{item.input_order}"
            if not hasattr(self.bases, output_name):
                self.bases.add_basis(output_name, item.output_rows, item.output_order, runtime_dtype=runtime_dtype, version=basis_version, basis_family=basis_family)
            if not hasattr(self.bases, input_name):
                self.bases.add_basis(input_name, item.row_width, item.input_order, runtime_dtype=runtime_dtype, version=basis_version, basis_family=basis_family)
            self._output_basis_names[item.name] = output_name
            self._input_basis_names[item.name] = input_name
        self.coefficients = nn.ParameterDict({name: parameter for name, parameter in self.curve.coefficients.items()})
        for item in self.mlp_metadata:
            self.coefficients[item.name] = nn.Parameter(torch.empty(item.coefficient_shape(config.depth_order), dtype=runtime_dtype))
        self.metadata = tuple(item for item in self.curve.metadata if item.name not in {MLP_EXPANSION_WEIGHT, MLP_CONTRACTION_WEIGHT}) + self.mlp_metadata
        self._reset_mlp_parameters()

    @property
    def depth_basis(self) -> Tensor:
        return self.bases.depth_basis
    def family_metadata(self, name: str):
        return self._mlp_metadata_by_name[name] if name in self._mlp_metadata_by_name else self.curve.family_metadata(name)
    def output_basis(self, name: str) -> Tensor:
        return getattr(self.bases, self._output_basis_names[name])
    def input_basis(self, name: str) -> Tensor:
        return getattr(self.bases, self._input_basis_names[name])
    def row_basis(self, name: str) -> Tensor:
        return self.curve.row_basis(name)
    def reset_parameters(self) -> None:
        self._reset_mlp_parameters()
    def _reset_mlp_parameters(self) -> None:
        with torch.no_grad():
            for item in self.mlp_metadata:
                coefficient = self.coefficients[item.name]
                coefficient.zero_()
                std = item.target_weight_std * math.sqrt(self.config.n_layer * item.output_rows * item.row_width / (item.output_order * item.input_order))
                torch.nn.init.normal_(coefficient[0], mean=0.0, std=std)
    def _materialize_mlp_block(self, name: str, layer_index: int) -> Tensor:
        coefficient = self.coefficients[name]
        depth_row = self.depth_basis[layer_index].to(device=coefficient.device, dtype=coefficient.dtype)
        mixed = torch.einsum("p,pab->ab", depth_row, coefficient)
        return self.output_basis(name).to(coefficient) @ mixed @ self.input_basis(name).to(coefficient).transpose(0, 1)
    def materialize(self, name: str, layer_index: int) -> Tensor:
        return self._materialize_mlp_block(name, layer_index) if name in self._mlp_metadata_by_name else self.curve.materialize(name, layer_index)
    def materialize_vector(self, name: str, layer_index: int) -> Tensor:
        return self.curve.materialize_vector(name, layer_index)
    def direct_value(self, name: str, layer_index: int, output_row: int, row_index: int) -> Tensor:
        if name not in self._mlp_metadata_by_name:
            return self.curve.direct_value(name, layer_index, output_row, row_index)
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
        return tuple({"name": item.name, "semantic_type": item.semantic_type, "initialization": item.initialization, "target_weight_std": item.target_weight_std, "weight_decay": item.weight_decay, "output_rows": item.output_rows, "row_width": item.row_width, "row_order": item.row_order, "coefficient_shape": item.coefficient_shape(self.config.depth_order), "sheet_parameters": item.sheet_parameter_count(self.config.depth_order), "dense_equivalent_parameters": item.dense_equivalent_count(self.config.n_layer)} for item in self.metadata)
    def persistent_basis_keys(self) -> Tuple[str, ...]:
        return tuple(sorted(key for key in self.state_dict() if key.startswith("bases.") or key.startswith("curve.bases.")))
# ^^^ THOG
