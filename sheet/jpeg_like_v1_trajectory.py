# vvv THOG
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch
from torch import Tensor, nn

from .basis import BASIS_VERSION, BasisOwner
from .bases import basis_version_for_family, normalize_registered_basis_family
from .depth_trajectory import DepthTrajectory
from .geometry import SheetGeometryConfig
from .semantic_materializer import MLP_EXPANSION_WEIGHT


JPEG_LIKE_V1_MATERIALIZATION_VERSION = "jpeg_like_v1"


@dataclass(frozen=True)
class JpegLikeV1Metadata:
    name: str
    semantic_type: str
    initialization: str
    target_weight_std: float
    weight_decay: bool
    output_rows: int
    row_width: int
    group_size: int
    retained_modes: int

    @property
    def group_count(self) -> int:
        return self.output_rows // self.group_size

    def coefficient_shape(self, depth_order: int) -> Tuple[int, int, int, int]:
        return (self.group_count, self.retained_modes, self.row_width, depth_order)

    def sheet_parameter_count(self, depth_order: int) -> int:
        groups, modes, width, depth = self.coefficient_shape(depth_order)
        return groups * modes * width * depth

    def dense_equivalent_count(self, n_layer: int) -> int:
        return n_layer * self.output_rows * self.row_width


class JpegLikeV1Trajectory(nn.Module):
    """DEPTH geometry with a registered local compressor on MLP_UP/MLP_HIDDEN.

    JPEG_LIKE_V1 partitions the MLP_HIDDEN output-row axis of MLP_UP into
    contiguous non-overlapping segments.  The selected registered basis is
    applied only within each segment, independently at every MLP_D_MODEL
    coordinate.  DEPTH remains an independent transform.
    """

    def __init__(
        self,
        config: SheetGeometryConfig,
        *,
        mlp_hidden_group_size: int,
        mlp_hidden_compressor: str,
        runtime_dtype: torch.dtype = torch.float32,
        basis_version: str = BASIS_VERSION,
        basis_family: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.runtime_dtype = runtime_dtype
        self.basis_version = basis_version
        self.basis_family = basis_family
        self.mlp_hidden_group_size = self._validate_positive_integer(
            "mlp_hidden_group_size", mlp_hidden_group_size
        )
        self.mlp_hidden_compressor = normalize_registered_basis_family(
            mlp_hidden_compressor
        )
        self.mlp_hidden_compressor_version = basis_version_for_family(
            self.mlp_hidden_compressor
        )

        output_rows = 4 * config.n_embd
        if output_rows % self.mlp_hidden_group_size != 0:
            raise ValueError(
                "4*d_model must be divisible by mlp_hidden_group_size; "
                f"got 4*d_model={output_rows}, group_size={self.mlp_hidden_group_size}"
            )
        retained_modes = config.resolved_o_mlp_hidden
        if retained_modes > self.mlp_hidden_group_size:
            raise ValueError(
                "o_mlp_hidden/Y must not exceed mlp_hidden_group_size for "
                f"JPEG_LIKE_V1; got Y={retained_modes}, group_size={self.mlp_hidden_group_size}"
            )

        self.depth = DepthTrajectory(
            config,
            runtime_dtype=runtime_dtype,
            basis_version=basis_version,
            basis_family=basis_family,
        )
        del self.depth.coefficients[MLP_EXPANSION_WEIGHT]

        self.jpeg_metadata = JpegLikeV1Metadata(
            name=MLP_EXPANSION_WEIGHT,
            semantic_type="matrix",
            initialization="jpeg_like_v1_matrix_normal",
            target_weight_std=0.02,
            weight_decay=True,
            output_rows=output_rows,
            row_width=config.n_embd,
            group_size=self.mlp_hidden_group_size,
            retained_modes=retained_modes,
        )
        self._jpeg_metadata_by_name = {MLP_EXPANSION_WEIGHT: self.jpeg_metadata}
        self.metadata = tuple(
            item for item in self.depth.metadata if item.name != MLP_EXPANSION_WEIGHT
        ) + (self.jpeg_metadata,)

        self.bases = BasisOwner()
        self.bases.add_basis(
            "mlp_hidden_group_basis",
            self.mlp_hidden_group_size,
            retained_modes,
            runtime_dtype=runtime_dtype,
            version=self.mlp_hidden_compressor_version,
            basis_family=self.mlp_hidden_compressor,
        )

        self.coefficients = nn.ParameterDict(
            {name: parameter for name, parameter in self.depth.coefficients.items()}
        )
        self.coefficients[MLP_EXPANSION_WEIGHT] = nn.Parameter(
            torch.empty(
                self.jpeg_metadata.coefficient_shape(config.depth_order),
                dtype=runtime_dtype,
            )
        )
        self._reset_mlp_expansion_parameters()

    @staticmethod
    def _validate_positive_integer(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")
        return value

    @property
    def depth_basis(self) -> Tensor:
        return self.depth.depth_basis

    @property
    def mlp_hidden_group_basis(self) -> Tensor:
        return self.bases.mlp_hidden_group_basis

    def family_metadata(self, name: str):
        if name == MLP_EXPANSION_WEIGHT:
            return self.jpeg_metadata
        return self.depth.family_metadata(name)

    def row_basis(self, name: str) -> Tensor:
        return self.depth.row_basis(name)

    def reset_parameters(self) -> None:
        # DepthTrajectory initialized its surviving families before MLP_UP was
        # removed.  Reinitializing it afterward would try to access the removed
        # parameter, so this trajectory owns only the replacement reset here.
        self._reset_mlp_expansion_parameters()

    def _reset_mlp_expansion_parameters(self) -> None:
        with torch.no_grad():
            coefficient = self.coefficients[MLP_EXPANSION_WEIGHT]
            coefficient.zero_()
            # The average squared row norm of a BxY orthonormal prefix is Y/B.
            # Compensate so the materialized MLP_UP weights retain the existing
            # target variance when Y < B.  The DEPTH first mode contributes the
            # existing sqrt(n_layer) scaling.
            coefficient_std = self.jpeg_metadata.target_weight_std * math.sqrt(
                self.config.n_layer
                * self.jpeg_metadata.group_size
                / self.jpeg_metadata.retained_modes
            )
            torch.nn.init.normal_(
                coefficient[..., 0], mean=0.0, std=coefficient_std
            )

    def encode_depth_coefficients(self, depth_coefficients: Tensor) -> Tensor:
        """Project [4*d_model, d_model, P] depth coefficients into local modes."""
        expected = (
            self.jpeg_metadata.output_rows,
            self.jpeg_metadata.row_width,
            self.config.depth_order,
        )
        if tuple(depth_coefficients.shape) != expected:
            raise ValueError(
                f"depth_coefficients has shape {tuple(depth_coefficients.shape)}; "
                f"expected {expected}"
            )
        blocks = depth_coefficients.reshape(
            self.jpeg_metadata.group_count,
            self.jpeg_metadata.group_size,
            self.jpeg_metadata.row_width,
            self.config.depth_order,
        )
        basis = self.mlp_hidden_group_basis.to(depth_coefficients)
        return torch.einsum("by,gbdp->gydp", basis, blocks)

    def reconstruct_depth_coefficients(self, local_coefficients: Tensor) -> Tensor:
        """Reconstruct [4*d_model, d_model, P] coefficients from local modes."""
        expected = self.jpeg_metadata.coefficient_shape(self.config.depth_order)
        if tuple(local_coefficients.shape) != expected:
            raise ValueError(
                f"local_coefficients has shape {tuple(local_coefficients.shape)}; "
                f"expected {expected}"
            )
        basis = self.mlp_hidden_group_basis.to(local_coefficients)
        blocks = torch.einsum("by,gydp->gbdp", basis, local_coefficients)
        return blocks.reshape(
            self.jpeg_metadata.output_rows,
            self.jpeg_metadata.row_width,
            self.config.depth_order,
        )

    def _materialize_mlp_expansion(self, layer_index: int) -> Tensor:
        coefficient = self.coefficients[MLP_EXPANSION_WEIGHT]
        depth_row = self.depth_basis[layer_index].to(coefficient)
        mixed = torch.einsum("p,gydp->gyd", depth_row, coefficient)
        basis = self.mlp_hidden_group_basis.to(coefficient)
        blocks = torch.einsum("by,gyd->gbd", basis, mixed)
        return blocks.reshape(
            self.jpeg_metadata.output_rows,
            self.jpeg_metadata.row_width,
        )

    def materialize(self, name: str, layer_index: int) -> Tensor:
        if name == MLP_EXPANSION_WEIGHT:
            if isinstance(layer_index, bool) or not isinstance(layer_index, int):
                raise ValueError(f"layer_index must be an integer; got {layer_index!r}")
            if layer_index < 0 or layer_index >= self.config.n_layer:
                raise IndexError(
                    f"layer_index out of range: {layer_index}; n_layer={self.config.n_layer}"
                )
            return self._materialize_mlp_expansion(layer_index)
        return self.depth.materialize(name, layer_index)

    def materialize_vector(self, name: str, layer_index: int) -> Tensor:
        return self.depth.materialize_vector(name, layer_index)

    def direct_value(
        self,
        name: str,
        layer_index: int,
        output_row: int,
        row_index: int,
    ) -> Tensor:
        if name != MLP_EXPANSION_WEIGHT:
            return self.depth.direct_value(name, layer_index, output_row, row_index)
        if output_row < 0 or output_row >= self.jpeg_metadata.output_rows:
            raise IndexError(f"output_row out of range for {name}: {output_row}")
        if row_index < 0 or row_index >= self.jpeg_metadata.row_width:
            raise IndexError(f"row_index out of range for {name}: {row_index}")
        if layer_index < 0 or layer_index >= self.config.n_layer:
            raise IndexError(
                f"layer_index out of range: {layer_index}; n_layer={self.config.n_layer}"
            )
        group_index, local_row = divmod(
            output_row, self.jpeg_metadata.group_size
        )
        coefficient = self.coefficients[name][group_index, :, row_index, :]
        depth_row = self.depth_basis[layer_index].to(coefficient)
        mode_values = coefficient @ depth_row
        basis_row = self.mlp_hidden_group_basis[local_row].to(coefficient)
        return basis_row @ mode_values

    def named_semantic_parameters(self) -> Iterator[Tuple[str, nn.Parameter, object]]:
        for item in self.metadata:
            yield item.name, self.coefficients[item.name], item

    def sheet_parameter_count(self) -> int:
        return sum(
            item.sheet_parameter_count(self.config.depth_order) for item in self.metadata
        )

    def dense_equivalent_count(self) -> int:
        return sum(
            item.dense_equivalent_count(self.config.n_layer) for item in self.metadata
        )

    def matrix_sheet_parameter_count(self) -> int:
        return sum(
            item.sheet_parameter_count(self.config.depth_order)
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
            row: Dict[str, object] = {
                "name": item.name,
                "semantic_type": item.semantic_type,
                "initialization": item.initialization,
                "target_weight_std": item.target_weight_std,
                "weight_decay": item.weight_decay,
                "output_rows": item.output_rows,
                "row_width": item.row_width,
                "coefficient_shape": item.coefficient_shape(self.config.depth_order),
                "sheet_parameters": item.sheet_parameter_count(self.config.depth_order),
                "dense_equivalent_parameters": item.dense_equivalent_count(
                    self.config.n_layer
                ),
            }
            if item.name == MLP_EXPANSION_WEIGHT:
                row.update(
                    {
                        "axis": "MLP_HIDDEN",
                        "group_size": self.jpeg_metadata.group_size,
                        "group_count": self.jpeg_metadata.group_count,
                        "retained_modes": self.jpeg_metadata.retained_modes,
                        "compressor": self.mlp_hidden_compressor,
                        "compressor_version": self.mlp_hidden_compressor_version,
                    }
                )
            rows.append(row)
        return tuple(rows)

    def persistent_basis_keys(self) -> Tuple[str, ...]:
        return tuple(
            sorted(
                key
                for key in self.state_dict()
                if key.startswith("bases.") or key.startswith("depth.bases.")
            )
        )


__all__ = [
    "JPEG_LIKE_V1_MATERIALIZATION_VERSION",
    "JpegLikeV1Metadata",
    "JpegLikeV1Trajectory",
]
# ^^^ THOG
