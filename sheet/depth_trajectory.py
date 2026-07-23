# vvv THOG
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch import Tensor, nn

from .basis import BASIS_FAMILY_CHEBYSHEV, BASIS_VERSION, BasisCache, BasisOwner
from .geometry import SheetGeometryConfig
from .semantic_materializer import ATTENTION_KEY_WEIGHT, ATTENTION_OUTPUT_WEIGHT, ATTENTION_QUERY_WEIGHT, ATTENTION_VALUE_WEIGHT, LEGACY_ATTENTION_INPUT_WEIGHT, MLP_CONTRACTION_WEIGHT, MLP_EXPANSION_WEIGHT
from .trajectory import build_family_metadata


DEPTH_MATRIX_FAMILIES = (
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_KEY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    MLP_EXPANSION_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
)


@dataclass(frozen=True)
class DepthFamilyMetadata:
    name: str
    semantic_type: str
    initialization: str
    target_weight_std: float
    weight_decay: bool
    output_rows: int
    row_width: int
    row_order: int

    def depth_coefficient_shape(self, depth_order: int) -> Tuple[int, int, int]:
        return (self.output_rows, self.row_width, depth_order)

    def conventional_parameter_shape(self, n_layer: int) -> Tuple[int, int, int]:
        return (n_layer, self.output_rows, self.row_width)

    def depth_coefficient_count(self, depth_order: int) -> int:
        rows, width, depth_terms = self.depth_coefficient_shape(depth_order)
        return rows * width * depth_terms

    def dense_equivalent_count(self, n_layer: int) -> int:
        return n_layer * self.output_rows * self.row_width


class DepthTrajectory(nn.Module):
    """Pure depth trajectories for block weights with selectable LayerNorm/bias participation."""

    def __init__(
        self,
        config: SheetGeometryConfig,
        *,
        runtime_dtype: torch.dtype = torch.float32,
        basis_version: str = BASIS_VERSION,
        basis_cache: Optional[BasisCache] = None,
        basis_family: str = BASIS_FAMILY_CHEBYSHEV,
        depth_compress_layer_norm_and_bias: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(depth_compress_layer_norm_and_bias, bool):
            raise ValueError(
                "depth_compress_layer_norm_and_bias must be bool; "
                f"got {depth_compress_layer_norm_and_bias!r}"
            )
        self.config = config
        self.runtime_dtype = runtime_dtype
        self.basis_version = basis_version
        self.basis_family = basis_family
        self.depth_compress_layer_norm_and_bias = depth_compress_layer_norm_and_bias
        self.metadata = self._build_metadata()
        self._metadata_by_name: Dict[str, DepthFamilyMetadata] = {
            item.name: item for item in self.metadata
        }
        self.bases = BasisOwner(basis_cache)
        self.bases.add_basis(
            "depth_basis",
            config.n_layer,
            config.depth_order,
            runtime_dtype=runtime_dtype,
            version=basis_version,
            basis_family=basis_family,
        )
        # vvv THOG DEPTH owns no within-tensor basis; vectors are either direct per-layer parameters or pure depth trajectories.
        self.coefficients = nn.ParameterDict()
        for item in self.metadata:
            shape = (
                item.depth_coefficient_shape(config.depth_order)
                if self._is_depth_compressed(item)
                else item.conventional_parameter_shape(config.n_layer)
            )
            self.coefficients[item.name] = nn.Parameter(
                torch.empty(shape, dtype=runtime_dtype)
            )
        # ^^^ THOG
        self.reset_parameters()

    def _build_metadata(self) -> Tuple[DepthFamilyMetadata, ...]:
        width = self.config.n_embd
        residual_std = 0.02 / math.sqrt(2.0 * self.config.n_layer)
        rows = [
            DepthFamilyMetadata(ATTENTION_QUERY_WEIGHT, "matrix", "depth_matrix_normal", 0.02, True, width, width, width),
            DepthFamilyMetadata(ATTENTION_KEY_WEIGHT, "matrix", "depth_matrix_normal", 0.02, True, width, width, width),
            DepthFamilyMetadata(ATTENTION_VALUE_WEIGHT, "matrix", "depth_matrix_normal", 0.02, True, width, width, width),
            DepthFamilyMetadata(ATTENTION_OUTPUT_WEIGHT, "matrix", "depth_matrix_normal", residual_std, True, width, width, width),
            DepthFamilyMetadata(MLP_EXPANSION_WEIGHT, "matrix", "depth_matrix_normal", 0.02, True, 4 * width, width, width),
            DepthFamilyMetadata(MLP_CONTRACTION_WEIGHT, "matrix", "depth_matrix_normal", residual_std, True, width, 4 * width, 4 * width),
        ]
        for item in build_family_metadata(self.config):
            if item.semantic_type == "matrix":
                continue
            geometry = item.geometry
            rows.append(
                DepthFamilyMetadata(
                    item.name,
                    item.semantic_type,
                    item.initialization,
                    item.target_weight_std,
                    item.weight_decay,
                    geometry.output_rows,
                    geometry.row_width,
                    geometry.row_width,
                )
            )
        return tuple(rows)

    def _is_depth_compressed(self, item: DepthFamilyMetadata) -> bool:
        return item.semantic_type == "matrix" or self.depth_compress_layer_norm_and_bias

    def family_metadata(self, name: str) -> DepthFamilyMetadata:
        try:
            return self._metadata_by_name[name]
        except KeyError as error:
            raise KeyError(f"unknown depth family: {name}") from error

    @property
    def depth_basis(self) -> Tensor:
        return self.bases.depth_basis

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for item in self.metadata:
                parameter = self.coefficients[item.name]
                parameter.zero_()
                if self._is_depth_compressed(item):
                    if item.initialization == "depth_matrix_normal":
                        coefficient_std = item.target_weight_std * math.sqrt(self.config.n_layer)
                        torch.nn.init.normal_(parameter[:, :, 0], mean=0.0, std=coefficient_std)
                    elif item.initialization == "layernorm_one":
                        parameter[:, :, 0].fill_(math.sqrt(self.config.n_layer))
                    elif item.initialization == "zero":
                        continue
                    else:
                        raise RuntimeError(
                            f"unsupported initialization policy {item.initialization} for {item.name}"
                        )
                    continue
                if item.initialization == "layernorm_one":
                    parameter.fill_(1.0)
                elif item.initialization == "zero":
                    continue
                else:
                    raise RuntimeError(
                        f"unsupported conventional initialization policy {item.initialization} for {item.name}"
                    )

    def _materialize_depth_parameter(self, name: str, layer_index: int) -> Tensor:
        coefficient = self.coefficients[name]
        depth_row = self.depth_basis[layer_index].to(
            device=coefficient.device,
            dtype=coefficient.dtype,
        )
        generated = torch.einsum("p,rcp->rc", depth_row, coefficient)
        item = self.family_metadata(name)
        expected_shape = (item.output_rows, item.row_width)
        if tuple(generated.shape) != expected_shape:
            raise RuntimeError(
                f"depth parameter {name} has shape {tuple(generated.shape)}; expected {expected_shape}"
            )
        return generated

    def _materialize_conventional_parameter(self, name: str, layer_index: int) -> Tensor:
        item = self.family_metadata(name)
        generated = self.coefficients[name][layer_index]
        expected_shape = (item.output_rows, item.row_width)
        if tuple(generated.shape) != expected_shape:
            raise RuntimeError(
                f"conventional parameter {name} has shape {tuple(generated.shape)}; expected {expected_shape}"
            )
        return generated

    def materialize(self, name: str, layer_index: int) -> Tensor:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise ValueError(f"layer_index must be an integer; got {layer_index!r}")
        if layer_index < 0 or layer_index >= self.config.n_layer:
            raise IndexError(
                f"layer_index out of range: {layer_index}; n_layer={self.config.n_layer}"
            )
        if name == LEGACY_ATTENTION_INPUT_WEIGHT:
            return torch.cat(
                (
                    self._materialize_depth_parameter(ATTENTION_QUERY_WEIGHT, layer_index),
                    self._materialize_depth_parameter(ATTENTION_KEY_WEIGHT, layer_index),
                    self._materialize_depth_parameter(ATTENTION_VALUE_WEIGHT, layer_index),
                ),
                dim=0,
            )
        item = self.family_metadata(name)
        if self._is_depth_compressed(item):
            return self._materialize_depth_parameter(name, layer_index)
        return self._materialize_conventional_parameter(name, layer_index)

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
        if name == LEGACY_ATTENTION_INPUT_WEIGHT:
            width = self.config.n_embd
            if output_row < width:
                return self.direct_value(
                    ATTENTION_QUERY_WEIGHT,
                    layer_index,
                    output_row,
                    row_index,
                )
            if output_row < 2 * width:
                return self.direct_value(
                    ATTENTION_KEY_WEIGHT,
                    layer_index,
                    output_row - width,
                    row_index,
                )
            if output_row < 3 * width:
                return self.direct_value(
                    ATTENTION_VALUE_WEIGHT,
                    layer_index,
                    output_row - 2 * width,
                    row_index,
                )
            raise IndexError(f"output_row out of range for {name}: {output_row}")
        item = self.family_metadata(name)
        if output_row < 0 or output_row >= item.output_rows:
            raise IndexError(f"output_row out of range for {name}: {output_row}")
        if row_index < 0 or row_index >= item.row_width:
            raise IndexError(f"row_index out of range for {name}: {row_index}")
        parameter = self.coefficients[name]
        if self._is_depth_compressed(item):
            coefficient = parameter[output_row, row_index]
            depth_row = self.depth_basis[layer_index].to(coefficient)
            return depth_row @ coefficient
        return parameter[layer_index, output_row, row_index]

    def named_semantic_parameters(
        self,
    ) -> Iterator[Tuple[str, nn.Parameter, DepthFamilyMetadata]]:
        for item in self.metadata:
            yield item.name, self.coefficients[item.name], item

    def sheet_parameter_count(self) -> int:
        return sum(
            item.depth_coefficient_count(self.config.depth_order)
            for item in self.metadata
            if self._is_depth_compressed(item)
        )

    def dense_equivalent_count(self) -> int:
        return sum(
            item.dense_equivalent_count(self.config.n_layer)
            for item in self.metadata
        )

    def matrix_sheet_parameter_count(self) -> int:
        return sum(
            item.depth_coefficient_count(self.config.depth_order)
            for item in self.metadata
            if item.semantic_type == "matrix"
        )

    def matrix_dense_equivalent_count(self) -> int:
        return sum(
            item.dense_equivalent_count(self.config.n_layer)
            for item in self.metadata
            if item.semantic_type == "matrix"
        )

    def family_report(self) -> Tuple[Dict[str, object], ...]:
        rows = []
        for item in self.metadata:
            depth_compressed = self._is_depth_compressed(item)
            parameter_shape = (
                item.depth_coefficient_shape(self.config.depth_order)
                if depth_compressed
                else item.conventional_parameter_shape(self.config.n_layer)
            )
            persistent_parameters = math.prod(parameter_shape)
            rows.append(
                {
                    "name": item.name,
                    "semantic_type": item.semantic_type,
                    "initialization": item.initialization,
                    "target_weight_std": item.target_weight_std,
                    "weight_decay": item.weight_decay,
                    "output_rows": item.output_rows,
                    "row_width": item.row_width,
                    "row_order": item.row_width,
                    "representation": (
                        "depth_coefficients"
                        if depth_compressed
                        else "conventional_per_layer"
                    ),
                    "coefficient_shape": parameter_shape if depth_compressed else None,
                    "parameter_shape": parameter_shape,
                    "sheet_parameters": persistent_parameters if depth_compressed else 0,
                    "persistent_parameters": persistent_parameters,
                    "dense_equivalent_parameters": item.dense_equivalent_count(
                        self.config.n_layer
                    ),
                }
            )
        return tuple(rows)

    def persistent_basis_keys(self) -> Tuple[str, ...]:
        return tuple(
            sorted(key for key in self.state_dict() if key.startswith("bases."))
        )
# ^^^ THOG
