# vvv THOG
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch
from torch import Tensor, nn

from .basis import BASIS_FAMILY_CHEBYSHEV, BASIS_VERSION, BasisOwner
from .depth_trajectory import DepthTrajectory
from .geometry import SheetGeometryConfig
from .semantic_materializer import ATTENTION_KEY_WEIGHT, ATTENTION_OUTPUT_WEIGHT, ATTENTION_QUERY_WEIGHT, ATTENTION_VALUE_WEIGHT, LEGACY_ATTENTION_INPUT_WEIGHT, MLP_CONTRACTION_WEIGHT, MLP_EXPANSION_WEIGHT


HEAD_AWARE_BLOCK_MATERIALIZATION_VERSION = "head_aware_block_v2"
FULL_BLOCK_MATERIALIZATION_VERSION = "full_block_v1"
HEAD_AWARE_BLOCK_MATRIX_FAMILIES = (
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_KEY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
)
MLP_BLOCK_MATRIX_FAMILIES = (MLP_EXPANSION_WEIGHT, MLP_CONTRACTION_WEIGHT)
FULL_BLOCK_MATRIX_FAMILIES = HEAD_AWARE_BLOCK_MATRIX_FAMILIES + MLP_BLOCK_MATRIX_FAMILIES


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
    attention_head_axis: str = "none"
    head_count: int = 1
    head_dim: int = 0

    def coefficient_shape(self, depth_order: int) -> Tuple[int, ...]:
        if self.attention_head_axis == "none":
            return (depth_order, self.output_order, self.input_order)
        return (self.head_count, depth_order, self.output_order, self.input_order)

    def sheet_parameter_count(self, depth_order: int) -> int:
        count = depth_order * self.output_order * self.input_order
        if self.attention_head_axis != "none":
            count *= self.head_count
        return count

    def dense_equivalent_count(self, n_layer: int) -> int:
        return n_layer * self.output_rows * self.row_width

    @property
    def basis_crosses_attention_head_boundary(self) -> bool:
        return False


class BlockTrajectory(nn.Module):
    """Head-aware attention blocks plus optional MLP blocks over DEPTH fallback families."""

    def __init__(
        self,
        config: SheetGeometryConfig,
        *,
        runtime_dtype: torch.dtype = torch.float32,
        basis_version: str = BASIS_VERSION,
        basis_family: str = BASIS_FAMILY_CHEBYSHEV,
        compact_attention: bool = True,
        compact_mlp: bool = True,
    ) -> None:
        super().__init__()
        if not compact_attention and not compact_mlp:
            raise ValueError("BlockTrajectory requires at least one compact subsystem")
        self.config = config
        self.runtime_dtype = runtime_dtype
        self.basis_version = basis_version
        self.basis_family = basis_family
        self.compact_attention = compact_attention
        self.compact_mlp = compact_mlp
        self.depth = DepthTrajectory(
            config,
            runtime_dtype=runtime_dtype,
            basis_version=basis_version,
            basis_family=basis_family,
        )
        self.block_matrix_families = self._selected_block_matrix_families()
        for name in self.block_matrix_families:
            del self.depth.coefficients[name]
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
            output_name = f"block_output_basis_c{self._basis_output_width(item)}_q{item.output_order}"
            input_name = f"block_input_basis_c{self._basis_input_width(item)}_q{item.input_order}"
            if not hasattr(self.bases, output_name):
                self.bases.add_basis(output_name, self._basis_output_width(item), item.output_order, runtime_dtype=runtime_dtype, version=basis_version, basis_family=basis_family)
            if not hasattr(self.bases, input_name):
                self.bases.add_basis(input_name, self._basis_input_width(item), item.input_order, runtime_dtype=runtime_dtype, version=basis_version, basis_family=basis_family)
            self._output_basis_names[item.name] = output_name
            self._input_basis_names[item.name] = input_name
        self.coefficients = nn.ParameterDict({name: parameter for name, parameter in self.depth.coefficients.items()})
        for item in self.block_metadata:
            self.coefficients[item.name] = nn.Parameter(torch.empty(item.coefficient_shape(config.depth_order), dtype=runtime_dtype))
        self.metadata = tuple(item for item in self.depth.metadata if item.name not in set(self.block_matrix_families)) + self.block_metadata
        self._reset_block_parameters()

    def _selected_block_matrix_families(self) -> Tuple[str, ...]:
        families = []
        if self.compact_attention:
            families.extend(HEAD_AWARE_BLOCK_MATRIX_FAMILIES)
        if self.compact_mlp:
            families.extend(MLP_BLOCK_MATRIX_FAMILIES)
        return tuple(families)

    def _build_block_metadata(self) -> Tuple[BlockMetadata, ...]:
        width = self.config.n_embd
        head_count = self.config.n_head
        head_dim = width // head_count
        residual_std = 0.02 / math.sqrt(2.0 * self.config.n_layer)
        rows = []
        if self.compact_attention:
            rows.extend([
                BlockMetadata(
                    ATTENTION_QUERY_WEIGHT,
                    "matrix",
                    "head_aware_block_matrix_normal",
                    0.02,
                    True,
                    width,
                    width,
                    self.config.resolved_o_attn_d_model,
                    self.config.resolved_o_attn_qkv_per_channel,
                    self.config.resolved_o_attn_d_model,
                    "output",
                    head_count,
                    head_dim,
                ),
                BlockMetadata(
                    ATTENTION_KEY_WEIGHT,
                    "matrix",
                    "head_aware_block_matrix_normal",
                    0.02,
                    True,
                    width,
                    width,
                    self.config.resolved_o_attn_d_model,
                    self.config.resolved_o_attn_qkv_per_channel,
                    self.config.resolved_o_attn_d_model,
                    "output",
                    head_count,
                    head_dim,
                ),
                BlockMetadata(
                    ATTENTION_VALUE_WEIGHT,
                    "matrix",
                    "head_aware_block_matrix_normal",
                    0.02,
                    True,
                    width,
                    width,
                    self.config.resolved_o_attn_d_model,
                    self.config.resolved_o_attn_qkv_per_channel,
                    self.config.resolved_o_attn_d_model,
                    "output",
                    head_count,
                    head_dim,
                ),
                BlockMetadata(
                    ATTENTION_OUTPUT_WEIGHT,
                    "matrix",
                    "head_aware_block_matrix_normal",
                    residual_std,
                    True,
                    width,
                    width,
                    self.config.resolved_o_attn_out_per_channel,
                    self.config.resolved_o_attn_d_model,
                    self.config.resolved_o_attn_out_per_channel,
                    "input",
                    head_count,
                    head_dim,
                ),
            ])
        if self.compact_mlp:
            rows.extend([
                BlockMetadata(
                    MLP_EXPANSION_WEIGHT,
                    "matrix",
                    "block_matrix_normal",
                    0.02,
                    True,
                    4 * width,
                    width,
                    self.config.resolved_o_mlp_d_model,
                    self.config.resolved_o_mlp_hidden,
                    self.config.resolved_o_mlp_d_model,
                ),
                BlockMetadata(
                    MLP_CONTRACTION_WEIGHT,
                    "matrix",
                    "block_matrix_normal",
                    residual_std,
                    True,
                    width,
                    4 * width,
                    self.config.resolved_o_mlp_hidden,
                    self.config.resolved_o_mlp_d_model,
                    self.config.resolved_o_mlp_hidden,
                ),
            ])
        return tuple(rows)

    @staticmethod
    def _basis_output_width(item: BlockMetadata) -> int:
        if item.attention_head_axis == "output":
            return item.head_dim
        return item.output_rows

    @staticmethod
    def _basis_input_width(item: BlockMetadata) -> int:
        if item.attention_head_axis == "input":
            return item.head_dim
        return item.row_width

    @property
    def depth_basis(self) -> Tensor:
        return self.bases.depth_basis

    def family_metadata(self, name: str):
        if name in self._block_metadata_by_name:
            return self._block_metadata_by_name[name]
        return self.depth.family_metadata(name)

    def output_basis(self, name: str) -> Tensor:
        return getattr(self.bases, self._output_basis_names[name])

    def input_basis(self, name: str) -> Tensor:
        return getattr(self.bases, self._input_basis_names[name])

    def row_basis(self, name: str) -> Tensor:
        return self.depth.row_basis(name)

    def reset_parameters(self) -> None:
        self._reset_block_parameters()

    def _reset_block_parameters(self) -> None:
        with torch.no_grad():
            for item in self.block_metadata:
                coefficient = self.coefficients[item.name]
                coefficient.zero_()
                represented_rows = self._basis_output_width(item)
                represented_columns = self._basis_input_width(item)
                coefficient_std = item.target_weight_std * math.sqrt(
                    self.config.n_layer * represented_rows * represented_columns / (item.output_order * item.input_order)
                )
                if item.attention_head_axis == "none":
                    torch.nn.init.normal_(coefficient[0], mean=0.0, std=coefficient_std)
                else:
                    torch.nn.init.normal_(coefficient[:, 0], mean=0.0, std=coefficient_std)

    def _materialize_generic_block(self, name: str, layer_index: int) -> Tensor:
        coefficient = self.coefficients[name]
        depth_row = self.depth_basis[layer_index].to(device=coefficient.device, dtype=coefficient.dtype)
        mixed = torch.einsum("p,pab->ab", depth_row, coefficient)
        return self.output_basis(name).to(coefficient) @ mixed @ self.input_basis(name).to(coefficient).transpose(0, 1)

    def _materialize_head_output_block(self, name: str, layer_index: int) -> Tensor:
        item = self.family_metadata(name)
        coefficient = self.coefficients[name]
        depth_row = self.depth_basis[layer_index].to(device=coefficient.device, dtype=coefficient.dtype)
        output_basis = self.output_basis(name).to(coefficient)
        input_basis = self.input_basis(name).to(coefficient)
        pieces = []
        for head_index in range(item.head_count):
            mixed = torch.einsum("p,pab->ab", depth_row, coefficient[head_index])
            pieces.append(output_basis @ mixed @ input_basis.transpose(0, 1))
        return torch.cat(pieces, dim=0)

    def _materialize_head_input_block(self, name: str, layer_index: int) -> Tensor:
        item = self.family_metadata(name)
        coefficient = self.coefficients[name]
        depth_row = self.depth_basis[layer_index].to(device=coefficient.device, dtype=coefficient.dtype)
        output_basis = self.output_basis(name).to(coefficient)
        input_basis = self.input_basis(name).to(coefficient)
        pieces = []
        for head_index in range(item.head_count):
            mixed = torch.einsum("p,pab->ab", depth_row, coefficient[head_index])
            pieces.append(output_basis @ mixed @ input_basis.transpose(0, 1))
        return torch.cat(pieces, dim=1)

    def _materialize_block(self, name: str, layer_index: int) -> Tensor:
        item = self.family_metadata(name)
        if item.attention_head_axis == "output":
            generated = self._materialize_head_output_block(name, layer_index)
        elif item.attention_head_axis == "input":
            generated = self._materialize_head_input_block(name, layer_index)
        else:
            generated = self._materialize_generic_block(name, layer_index)
        expected_shape = (item.output_rows, item.row_width)
        if tuple(generated.shape) != expected_shape:
            raise RuntimeError(f"block matrix {name} has shape {tuple(generated.shape)}; expected {expected_shape}")
        return generated

    def materialize(self, name: str, layer_index: int) -> Tensor:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise ValueError(f"layer_index must be an integer; got {layer_index!r}")
        if layer_index < 0 or layer_index >= self.config.n_layer:
            raise IndexError(f"layer_index out of range: {layer_index}; n_layer={self.config.n_layer}")
        if name == LEGACY_ATTENTION_INPUT_WEIGHT and self.compact_attention:
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
        return self.depth.materialize(name, layer_index)

    def materialize_vector(self, name: str, layer_index: int) -> Tensor:
        return self.depth.materialize_vector(name, layer_index)

    def _direct_head_output_value(self, item: BlockMetadata, layer_index: int, output_row: int, row_index: int) -> Tensor:
        coefficient = self.coefficients[item.name]
        head_index = output_row // item.head_dim
        local_output_row = output_row % item.head_dim
        mixed = torch.einsum("p,pab->ab", self.depth_basis[layer_index].to(coefficient), coefficient[head_index])
        return self.output_basis(item.name)[local_output_row].to(coefficient) @ mixed @ self.input_basis(item.name)[row_index].to(coefficient)

    def _direct_head_input_value(self, item: BlockMetadata, layer_index: int, output_row: int, row_index: int) -> Tensor:
        coefficient = self.coefficients[item.name]
        head_index = row_index // item.head_dim
        local_input_index = row_index % item.head_dim
        mixed = torch.einsum("p,pab->ab", self.depth_basis[layer_index].to(coefficient), coefficient[head_index])
        return self.output_basis(item.name)[output_row].to(coefficient) @ mixed @ self.input_basis(item.name)[local_input_index].to(coefficient)

    def direct_value(self, name: str, layer_index: int, output_row: int, row_index: int) -> Tensor:
        if name == LEGACY_ATTENTION_INPUT_WEIGHT and self.compact_attention:
            width = self.config.n_embd
            if output_row < width:
                return self.direct_value(ATTENTION_QUERY_WEIGHT, layer_index, output_row, row_index)
            if output_row < 2 * width:
                return self.direct_value(ATTENTION_KEY_WEIGHT, layer_index, output_row - width, row_index)
            if output_row < 3 * width:
                return self.direct_value(ATTENTION_VALUE_WEIGHT, layer_index, output_row - 2 * width, row_index)
            raise IndexError(f"output_row out of range for {name}: {output_row}")
        if name not in self._block_metadata_by_name:
            return self.depth.direct_value(name, layer_index, output_row, row_index)
        item = self.family_metadata(name)
        if item.attention_head_axis == "output":
            return self._direct_head_output_value(item, layer_index, output_row, row_index)
        if item.attention_head_axis == "input":
            return self._direct_head_input_value(item, layer_index, output_row, row_index)
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
        return tuple(
            {
                "name": item.name,
                "semantic_type": item.semantic_type,
                "initialization": item.initialization,
                "target_weight_std": item.target_weight_std,
                "weight_decay": item.weight_decay,
                "output_rows": item.output_rows,
                "row_width": item.row_width,
                "row_order": item.row_order,
                "output_order": item.output_order,
                "input_order": item.input_order,
                "attention_head_axis": item.attention_head_axis,
                "head_count": item.head_count,
                "head_dim": item.head_dim,
                "coefficient_shape": item.coefficient_shape(self.config.depth_order),
                "sheet_parameters": item.sheet_parameter_count(self.config.depth_order),
                "dense_equivalent_parameters": item.dense_equivalent_count(self.config.n_layer),
            }
            for item in self.metadata
        )

    def persistent_basis_keys(self) -> Tuple[str, ...]:
        return tuple(sorted(key for key in self.state_dict() if key.startswith("bases.") or key.startswith("depth.bases.")))
# ^^^ THOG
