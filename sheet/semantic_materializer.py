# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .trajectory import SheetTrajectory


ATTENTION_QUERY_WEIGHT = "attention_query_weight"
ATTENTION_KEY_WEIGHT = "attention_key_weight"
ATTENTION_VALUE_WEIGHT = "attention_value_weight"
ATTENTION_OUTPUT_WEIGHT = "attention_output_weight"
MLP_EXPANSION_WEIGHT = "mlp_expansion_weight"
MLP_CONTRACTION_WEIGHT = "mlp_contraction_weight"
ATTENTION_QUERY_BIAS = "attention_query_bias"
ATTENTION_KEY_BIAS = "attention_key_bias"
ATTENTION_VALUE_BIAS = "attention_value_bias"
LEGACY_ATTENTION_INPUT_WEIGHT = "attention_input_weight"
LEGACY_ATTENTION_INPUT_BIAS = "attention_input_bias"

SEMANTIC_MATRIX_FAMILIES = (
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_KEY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    MLP_EXPANSION_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
)
SEMANTIC_VECTOR_FAMILIES = (
    ATTENTION_QUERY_BIAS,
    ATTENTION_KEY_BIAS,
    ATTENTION_VALUE_BIAS,
)


@dataclass(frozen=True)
class SemanticFamilySpec:
    name: str
    legacy_family: str
    family_kind: str
    output_rows: int
    row_width: int
    legacy_row_start: Optional[int] = None
    legacy_row_stop: Optional[int] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.family_kind == "matrix":
            return (self.output_rows, self.row_width)
        if self.family_kind == "vector":
            return (self.row_width,)
        raise ValueError(f"unknown family_kind: {self.family_kind!r}")


class LegacySheetColMaterializer:
    def __init__(self, trajectory: SheetTrajectory) -> None:
        self.trajectory = trajectory
        self.config = trajectory.config
        if self.config.n_embd % self.config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.head_dim = self.config.n_embd // self.config.n_head
        self._matrix_specs = self._build_matrix_specs()
        self._vector_specs = self._build_vector_specs()

    def _qkv_row_range(self, role_index: int) -> Tuple[int, int]:
        start = role_index * self.config.n_embd
        return start, start + self.config.n_embd

    def _build_matrix_specs(self) -> Dict[str, SemanticFamilySpec]:
        q0, q1 = self._qkv_row_range(0)
        k0, k1 = self._qkv_row_range(1)
        v0, v1 = self._qkv_row_range(2)
        width = self.config.n_embd
        return {
            ATTENTION_QUERY_WEIGHT: SemanticFamilySpec(ATTENTION_QUERY_WEIGHT, LEGACY_ATTENTION_INPUT_WEIGHT, "matrix", width, width, q0, q1),
            ATTENTION_KEY_WEIGHT: SemanticFamilySpec(ATTENTION_KEY_WEIGHT, LEGACY_ATTENTION_INPUT_WEIGHT, "matrix", width, width, k0, k1),
            ATTENTION_VALUE_WEIGHT: SemanticFamilySpec(ATTENTION_VALUE_WEIGHT, LEGACY_ATTENTION_INPUT_WEIGHT, "matrix", width, width, v0, v1),
            ATTENTION_OUTPUT_WEIGHT: SemanticFamilySpec(ATTENTION_OUTPUT_WEIGHT, ATTENTION_OUTPUT_WEIGHT, "matrix", width, width),
            MLP_EXPANSION_WEIGHT: SemanticFamilySpec(MLP_EXPANSION_WEIGHT, MLP_EXPANSION_WEIGHT, "matrix", 4 * width, width),
            MLP_CONTRACTION_WEIGHT: SemanticFamilySpec(MLP_CONTRACTION_WEIGHT, MLP_CONTRACTION_WEIGHT, "matrix", width, 4 * width),
        }

    def _build_vector_specs(self) -> Dict[str, SemanticFamilySpec]:
        q0, q1 = self._qkv_row_range(0)
        k0, k1 = self._qkv_row_range(1)
        v0, v1 = self._qkv_row_range(2)
        width = self.config.n_embd
        return {
            ATTENTION_QUERY_BIAS: SemanticFamilySpec(ATTENTION_QUERY_BIAS, LEGACY_ATTENTION_INPUT_BIAS, "vector", 1, width, q0, q1),
            ATTENTION_KEY_BIAS: SemanticFamilySpec(ATTENTION_KEY_BIAS, LEGACY_ATTENTION_INPUT_BIAS, "vector", 1, width, k0, k1),
            ATTENTION_VALUE_BIAS: SemanticFamilySpec(ATTENTION_VALUE_BIAS, LEGACY_ATTENTION_INPUT_BIAS, "vector", 1, width, v0, v1),
        }

    def matrix_spec(self, name: str) -> SemanticFamilySpec:
        if name not in self._matrix_specs:
            raise KeyError(f"unknown semantic matrix family: {name}")
        return self._matrix_specs[name]

    def vector_spec(self, name: str) -> SemanticFamilySpec:
        if name not in self._vector_specs:
            raise KeyError(f"unknown semantic vector family: {name}")
        return self._vector_specs[name]

    def materialize_matrix(self, name: str, layer_index: int) -> Tensor:
        spec = self.matrix_spec(name)
        tensor = self.trajectory.materialize(spec.legacy_family, layer_index)
        if spec.legacy_row_start is not None and spec.legacy_row_stop is not None:
            tensor = tensor[spec.legacy_row_start:spec.legacy_row_stop, :]
        if tuple(tensor.shape) != spec.shape:
            raise RuntimeError(f"semantic matrix {name} has shape {tuple(tensor.shape)}; expected {spec.shape}")
        return tensor

    def materialize_vector(self, name: str, layer_index: int) -> Tensor:
        spec = self.vector_spec(name)
        tensor = self.trajectory.materialize_vector(spec.legacy_family, layer_index)
        if spec.legacy_row_start is not None and spec.legacy_row_stop is not None:
            tensor = tensor[spec.legacy_row_start:spec.legacy_row_stop]
        if tuple(tensor.shape) != spec.shape:
            raise RuntimeError(f"semantic vector {name} has shape {tuple(tensor.shape)}; expected {spec.shape}")
        return tensor

    def materialize(self, name: str, layer_index: int) -> Tensor:
        if name in self._matrix_specs:
            return self.materialize_matrix(name, layer_index)
        if name in self._vector_specs:
            return self.materialize_vector(name, layer_index)
        raise KeyError(f"unknown semantic family: {name}")

    def packed_attention_input_weight(self, layer_index: int) -> Tensor:
        return self.trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, layer_index)

    def packed_attention_input_bias(self, layer_index: int) -> Tensor:
        return self.trajectory.materialize_vector(LEGACY_ATTENTION_INPUT_BIAS, layer_index)

    def reconstructed_attention_input_weight(self, layer_index: int) -> Tensor:
        return torch.cat((
            self.materialize_matrix(ATTENTION_QUERY_WEIGHT, layer_index),
            self.materialize_matrix(ATTENTION_KEY_WEIGHT, layer_index),
            self.materialize_matrix(ATTENTION_VALUE_WEIGHT, layer_index),
        ), dim=0)

    def reconstructed_attention_input_bias(self, layer_index: int) -> Tensor:
        return torch.cat((
            self.materialize_vector(ATTENTION_QUERY_BIAS, layer_index),
            self.materialize_vector(ATTENTION_KEY_BIAS, layer_index),
            self.materialize_vector(ATTENTION_VALUE_BIAS, layer_index),
        ), dim=0)

    def direct_matrix_value(self, name: str, layer_index: int, output_row: int, row_index: int) -> Tensor:
        spec = self.matrix_spec(name)
        legacy_output_row = output_row if spec.legacy_row_start is None else spec.legacy_row_start + output_row
        return self.trajectory.direct_value(spec.legacy_family, layer_index, legacy_output_row, row_index)

    def direct_vector_value(self, name: str, layer_index: int, row_index: int) -> Tensor:
        spec = self.vector_spec(name)
        legacy_index = row_index if spec.legacy_row_start is None else spec.legacy_row_start + row_index
        return self.trajectory.materialize_vector(spec.legacy_family, layer_index)[legacy_index]

    def head_metadata(self) -> Dict[str, object]:
        role_ranges = {
            "query": self._qkv_row_range(0),
            "key": self._qkv_row_range(1),
            "value": self._qkv_row_range(2),
        }
        role_head_ranges = {
            role_name: tuple((start + head * self.head_dim, start + (head + 1) * self.head_dim) for head in range(self.config.n_head))
            for role_name, (start, _) in role_ranges.items()
        }
        output_columns = tuple((head * self.head_dim, (head + 1) * self.head_dim) for head in range(self.config.n_head))
        return {
            "head_dim": self.head_dim,
            "attention_input_role_row_ranges": role_ranges,
            "attention_input_role_head_row_ranges": role_head_ranges,
            "attention_output_input_head_column_ranges": output_columns,
        }

    def semantic_family_report(self) -> Tuple[Dict[str, object], ...]:
        rows = []
        for name in SEMANTIC_MATRIX_FAMILIES:
            spec = self.matrix_spec(name)
            rows.append({"name": spec.name, "legacy_family": spec.legacy_family, "family_kind": spec.family_kind, "shape": spec.shape, "legacy_row_range": (spec.legacy_row_start, spec.legacy_row_stop)})
        for name in SEMANTIC_VECTOR_FAMILIES:
            spec = self.vector_spec(name)
            rows.append({"name": spec.name, "legacy_family": spec.legacy_family, "family_kind": spec.family_kind, "shape": spec.shape, "legacy_row_range": (spec.legacy_row_start, spec.legacy_row_stop)})
        return tuple(rows)
# ^^^ THOG
