# vvv THOG
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from .basis_provider import AxisBasisProvider, HyperblockBasisTables
from .direct_mlp import FactorisedHyperblockMlpLayer, factorise_hyperblock_mlp_layer                                                                    # <<< THOG optional direct HYPERBLOCK MLP factors
from .materializer import (
    materialize_attention_family_layer,
    materialize_attention_layer_staged,                                                                                                                  # <<< THOG attention-only layer bundle when MLP stays factorised
    materialize_layer_staged,
    materialize_mlp_family_layer,
    route_attention_input_matrices,
    route_attention_matrix,
    route_mlp_matrix,
)
from .plan import (
    ATTENTION_FAMILIES,
    MLP_FAMILIES,
    WEIGHT_FAMILIES,
    ResolvedHyperblockPlan,
)


ATTENTION_INPUT_WEIGHT = "attention_input_weight"
ATTENTION_OUTPUT_WEIGHT = "attention_output_weight"
MLP_EXPANSION_WEIGHT = "mlp_expansion_weight"
MLP_CONTRACTION_WEIGHT = "mlp_contraction_weight"

MATRIX_FAMILY_NAMES = (
    ATTENTION_INPUT_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    MLP_EXPANSION_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
)

VECTOR_FAMILY_SHAPES = {
    "ln_1_weight": "d_model",
    "ln_1_bias": "d_model",
    "ln_2_weight": "d_model",
    "ln_2_bias": "d_model",
    "attention_input_bias": "three_d_model",
    "attention_output_bias": "d_model",
    "mlp_expansion_bias": "mlp_hidden",
    "mlp_contraction_bias": "d_model",
}


@dataclass(frozen=True)
class HyperblockParameterMetadata:
    name: str
    semantic_type: str
    initialization: str
    target_weight_std: float
    weight_decay: bool


class CoupledFieldTrajectory(nn.Module):
    """One coupled coefficient system generating the six large block matrices."""

    materialized_matrix_family_names = MATRIX_FAMILY_NAMES
    attention_materialized_matrix_family_names = (ATTENTION_INPUT_WEIGHT, ATTENTION_OUTPUT_WEIGHT)                                                       # <<< THOG partial retained bundle identity

    def __init__(
        self,
        plan: ResolvedHyperblockPlan,
        *,
        bias: bool,
        runtime_dtype: torch.dtype = torch.float32,
        basis_provider: Optional[AxisBasisProvider] = None,
        residual_weight_std: Optional[float] = None,
    ) -> None:
        super().__init__()
        if not isinstance(bias, bool):
            raise ValueError(f"bias must be bool; got {bias!r}")
        self.plan = plan
        self.config = plan
        self.bias = bias
        self.runtime_dtype = runtime_dtype
        self.bases = HyperblockBasisTables(
            plan,
            runtime_dtype=runtime_dtype,
            provider=basis_provider,
        )
        self.coefficients = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(shape, dtype=runtime_dtype))
                for name, shape in plan.coefficient_shapes.items()
            }
        )
        self.vector_parameters = nn.ParameterDict()
        for name, width in self._vector_widths().items():
            self.vector_parameters[name] = nn.Parameter(
                torch.empty(plan.n_layer, width, dtype=runtime_dtype)
            )
        default_residual_std = 0.02 / math.sqrt(2.0 * plan.n_layer)
        self._target_weight_stds: Dict[str, float] = {
            "Q": 0.02,
            "K": 0.02,
            "V": 0.02,
            "ATTENTION_OUTPUT": default_residual_std if residual_weight_std is None else residual_weight_std,
            "MLP_UP": 0.02,
            "MLP_DOWN": default_residual_std if residual_weight_std is None else residual_weight_std,
        }
        self.metadata = self._build_metadata()
        self._metadata_by_name = {item.name: item for item in self.metadata}
        self.reset_parameters()

    def _vector_widths(self) -> Dict[str, int]:
        d_model = self.plan.n_embd
        widths = {
            "ln_1_weight": d_model,
            "ln_2_weight": d_model,
        }
        if self.bias:
            widths.update(
                {
                    "ln_1_bias": d_model,
                    "ln_2_bias": d_model,
                    "attention_input_bias": 3 * d_model,
                    "attention_output_bias": d_model,
                    "mlp_expansion_bias": self.plan.mlp_hidden,
                    "mlp_contraction_bias": d_model,
                }
            )
        return widths

    def _build_metadata(self) -> Tuple[HyperblockParameterMetadata, ...]:
        residual_std = self._target_weight_stds["ATTENTION_OUTPUT"]
        rows = [
            HyperblockParameterMetadata(ATTENTION_INPUT_WEIGHT, "matrix", "coupled_field", 0.02, True),
            HyperblockParameterMetadata(ATTENTION_OUTPUT_WEIGHT, "matrix", "coupled_field", residual_std, True),
            HyperblockParameterMetadata(MLP_EXPANSION_WEIGHT, "matrix", "coupled_field", 0.02, True),
            HyperblockParameterMetadata(MLP_CONTRACTION_WEIGHT, "matrix", "coupled_field", residual_std, True),
            HyperblockParameterMetadata("ln_1_weight", "layernorm", "layernorm_one", 0.0, False),
            HyperblockParameterMetadata("ln_2_weight", "layernorm", "layernorm_one", 0.0, False),
        ]
        if self.bias:
            rows.extend(
                HyperblockParameterMetadata(name, "bias", "zero", 0.0, False)
                for name in (
                    "ln_1_bias",
                    "ln_2_bias",
                    "attention_input_bias",
                    "attention_output_bias",
                    "mlp_expansion_bias",
                    "mlp_contraction_bias",
                )
            )
        return tuple(rows)

    def family_metadata(self, name: str) -> HyperblockParameterMetadata:
        try:
            return self._metadata_by_name[name]
        except KeyError as error:
            raise KeyError(f"unknown HYPERBLOCK family: {name}") from error

    @staticmethod
    def _family_basis_project(physical_coefficients: Tensor, family_basis: Tensor) -> Tensor:
        return torch.tensordot(
            family_basis.to(physical_coefficients).transpose(0, 1),
            physical_coefficients,
            dims=([1], [0]),
        )

    @staticmethod
    def _family_basis_expand(coefficients: Tensor, family_basis: Tensor) -> Tensor:
        return torch.tensordot(
            family_basis.to(coefficients),
            coefficients,
            dims=([1], [0]),
        )

    @staticmethod
    def _normal_with_family_std(
        shape: Tuple[int, ...],
        family_stds: Tuple[float, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        values = torch.randn(shape, dtype=dtype, device=device)
        std = torch.tensor(family_stds, dtype=dtype, device=device)
        view_shape = (shape[0],) + (1,) * (len(shape) - 1)
        return values * std.reshape(view_shape)

    def reset_parameters(self) -> None:
        plan = self.plan
        base_energy = (
            plan.orders.depth / plan.n_layer
            * plan.orders.d_model / plan.n_embd
        )
        attention_mode_count = (
            plan.orders.attention_head
            * plan.orders.attention_head_channel
        )
        mlp_mode_count = plan.orders.mlp_hidden
        attention_physical_count = plan.n_head * plan.head_dim
        mlp_physical_count = plan.mlp_hidden

        common_stds = []
        for family_name in WEIGHT_FAMILIES:
            unique_mode_count = (
                attention_mode_count
                if family_name in ATTENTION_FAMILIES
                else mlp_mode_count
            )
            common_stds.append(
                self._target_weight_stds[family_name]
                * math.sqrt(1.0 / unique_mode_count / base_energy)
            )

        attention_stds = tuple(
            self._target_weight_stds[family_name]
            * math.sqrt(
                attention_physical_count
                / attention_mode_count
                / base_energy
            )
            for family_name in ATTENTION_FAMILIES
        )
        mlp_stds = tuple(
            self._target_weight_stds[family_name]
            * math.sqrt(
                mlp_physical_count
                / mlp_mode_count
                / base_energy
            )
            for family_name in MLP_FAMILIES
        )

        with torch.no_grad():
            common_parameter = self.coefficients["common"]
            attention_parameter = self.coefficients["attention"]
            mlp_parameter = self.coefficients["mlp"]

            common_physical = self._normal_with_family_std(
                (len(WEIGHT_FAMILIES), *tuple(common_parameter.shape[1:])),
                tuple(common_stds),
                dtype=common_parameter.dtype,
                device=common_parameter.device,
            )
            attention_physical = self._normal_with_family_std(
                (len(ATTENTION_FAMILIES), *tuple(attention_parameter.shape[1:])),
                attention_stds,
                dtype=attention_parameter.dtype,
                device=attention_parameter.device,
            )
            mlp_physical = self._normal_with_family_std(
                (len(MLP_FAMILIES), *tuple(mlp_parameter.shape[1:])),
                mlp_stds,
                dtype=mlp_parameter.dtype,
                device=mlp_parameter.device,
            )
            common_parameter.copy_(
                self._family_basis_project(
                    common_physical,
                    self.bases.weight_family_common,
                )
            )
            attention_parameter.copy_(
                self._family_basis_project(
                    attention_physical,
                    self.bases.weight_family_attention,
                )
            )
            mlp_parameter.copy_(
                self._family_basis_project(
                    mlp_physical,
                    self.bases.weight_family_mlp,
                )
            )

            self.vector_parameters["ln_1_weight"].fill_(1.0)
            self.vector_parameters["ln_2_weight"].fill_(1.0)
            for name, parameter in self.vector_parameters.items():
                if name not in {"ln_1_weight", "ln_2_weight"}:
                    parameter.zero_()

    def _validate_layer_index(self, layer_index: int) -> None:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise ValueError(f"layer_index must be an integer; got {layer_index!r}")
        if layer_index < 0 or layer_index >= self.plan.n_layer:
            raise IndexError(
                f"layer_index out of range: {layer_index}; n_layer={self.plan.n_layer}"
            )

    def _basis_mapping(self) -> Mapping[str, Tensor]:
        return self.bases.as_mapping()

    def _attention_family(self, family_index: int, layer_index: int) -> Tensor:
        return materialize_attention_family_layer(
            self.coefficients["common"],
            self.coefficients["attention"],
            self._basis_mapping(),
            family_index=family_index,
            layer_index=layer_index,
        )

    def _mlp_family(
        self,
        *,
        common_family_index: int,
        mlp_family_index: int,
        layer_index: int,
    ) -> Tensor:
        return materialize_mlp_family_layer(
            self.coefficients["common"],
            self.coefficients["mlp"],
            self._basis_mapping(),
            common_family_index=common_family_index,
            mlp_family_index=mlp_family_index,
            layer_index=layer_index,
        )

    # vvv THOG batch the six matrix families for one layer and route them to the existing nanoGPT operational layouts
    # def materialize_layer_matrices(self, layer_index: int) -> Dict[str, Tensor]:                                                                 # <<< THOG preserved pre-option signature
    def materialize_layer_matrices(
        self,
        layer_index: int,
        *,
        include_mlp: bool = True,
    ) -> Dict[str, Tensor]:
        self._validate_layer_index(layer_index)
        if not isinstance(include_mlp, bool):
            raise TypeError(f"include_mlp must be bool; got {include_mlp!r}")
        # vvv THOG direct HYPERBLOCK MLP requests only Q/K/V/O; the established full bundle remains the default
        if not include_mlp:
            attention_layer = materialize_attention_layer_staged(
                self.coefficients["common"],
                self.coefficients["attention"],
                self._basis_mapping(),
                layer_index=layer_index,
            )
            return {
                ATTENTION_INPUT_WEIGHT: route_attention_input_matrices(
                    attention_layer.attention[:3]
                ),
                ATTENTION_OUTPUT_WEIGHT: route_attention_matrix(
                    attention_layer.attention[3],
                    output_projection=True,
                ),
            }
        # ^^^ THOG
        layer = materialize_layer_staged(
            self.coefficients["common"],
            self.coefficients["attention"],
            self.coefficients["mlp"],
            self._basis_mapping(),
            layer_index=layer_index,
        )
        return {
            ATTENTION_INPUT_WEIGHT: route_attention_input_matrices(layer.attention[:3]),
            ATTENTION_OUTPUT_WEIGHT: route_attention_matrix(
                layer.attention[3],
                output_projection=True,
            ),
            MLP_EXPANSION_WEIGHT: route_mlp_matrix(layer.mlp[0], expansion=True),
            MLP_CONTRACTION_WEIGHT: route_mlp_matrix(layer.mlp[1], expansion=False),
        }
    # ^^^ THOG

    # vvv THOG low-dimensional UP/DOWN factors share one family/depth contraction per logical layer
    def factorised_mlp_layer(
        self,
        layer_index: int,
    ) -> FactorisedHyperblockMlpLayer:
        self._validate_layer_index(layer_index)
        return factorise_hyperblock_mlp_layer(
            self.coefficients["common"],
            self.coefficients["mlp"],
            self._basis_mapping(),
            layer_index=layer_index,
        )
    # ^^^ THOG

    def materialize(self, name: str, layer_index: int) -> Tensor:
        self._validate_layer_index(layer_index)
        if name == ATTENTION_INPUT_WEIGHT:
            qkv = tuple(
                route_attention_matrix(
                    self._attention_family(family_index, layer_index),
                    output_projection=False,
                )
                for family_index in range(3)
            )
            return torch.cat(qkv, dim=0)
        if name == ATTENTION_OUTPUT_WEIGHT:
            return route_attention_matrix(
                self._attention_family(3, layer_index),
                output_projection=True,
            )
        if name == MLP_EXPANSION_WEIGHT:
            return route_mlp_matrix(
                self._mlp_family(
                    common_family_index=4,
                    mlp_family_index=0,
                    layer_index=layer_index,
                ),
                expansion=True,
            )
        if name == MLP_CONTRACTION_WEIGHT:
            return route_mlp_matrix(
                self._mlp_family(
                    common_family_index=5,
                    mlp_family_index=1,
                    layer_index=layer_index,
                ),
                expansion=False,
            )
        if name in self.vector_parameters:
            return self.vector_parameters[name][layer_index]
        raise KeyError(f"unknown HYPERBLOCK family: {name}")

    def materialize_vector(self, name: str, layer_index: int) -> Tensor:
        self._validate_layer_index(layer_index)
        if name not in self.vector_parameters:
            raise KeyError(f"unknown HYPERBLOCK vector family: {name}")
        return self.vector_parameters[name][layer_index]

    def direct_value(
        self,
        name: str,
        layer_index: int,
        output_row: int,
        row_index: int,
    ) -> Tensor:
        materialized = self.materialize(name, layer_index)
        if materialized.ndim != 2:
            raise ValueError(f"family {name} is not a matrix family")
        return materialized[output_row, row_index]

    def named_semantic_parameters(
        self,
    ) -> Iterator[Tuple[str, nn.Parameter, HyperblockParameterMetadata]]:
        coefficient_metadata = {
            "common": HyperblockParameterMetadata(
                "common",
                "hyperblock_coefficients",
                "orthogonal_mode_variance_split",
                0.0,
                True,
            ),
            "attention": HyperblockParameterMetadata(
                "attention",
                "hyperblock_coefficients",
                "orthogonal_mode_variance_split",
                0.0,
                True,
            ),
            "mlp": HyperblockParameterMetadata(
                "mlp",
                "hyperblock_coefficients",
                "orthogonal_mode_variance_split",
                0.0,
                True,
            ),
        }
        for name, parameter in self.coefficients.items():
            yield name, parameter, coefficient_metadata[name]
        for name, parameter in self.vector_parameters.items():
            yield name, parameter, self.family_metadata(name)

    def _scale_family_coefficients(
        self,
        parameter_name: str,
        family_basis: Tensor,
        family_index: int,
        ratio: float,
    ) -> None:
        parameter = self.coefficients[parameter_name]
        physical = self._family_basis_expand(parameter, family_basis)
        physical[family_index].mul_(ratio)
        parameter.copy_(self._family_basis_project(physical, family_basis))

    def apply_residual_init_scaling(self, residual_weight_std: float) -> None:
        if residual_weight_std <= 0.0:
            raise ValueError(
                f"residual_weight_std must be positive; got {residual_weight_std!r}"
            )
        previous = self._target_weight_stds["ATTENTION_OUTPUT"]
        if previous <= 0.0:
            raise RuntimeError(f"stored residual std must be positive; got {previous}")
        ratio = residual_weight_std / previous
        family_orders_are_full = (
            self.plan.orders.common_family == len(WEIGHT_FAMILIES)
            and self.plan.orders.attention_family == len(ATTENTION_FAMILIES)
            and self.plan.orders.mlp_family == len(MLP_FAMILIES)
        )
        if not math.isclose(ratio, 1.0) and not family_orders_are_full:
            raise ValueError(
                "post-hoc residual-family scaling is not isolated when a WEIGHT_FAMILY "
                "axis is compressed; supply residual_weight_std during initialization"
            )
        with torch.no_grad():
            self._scale_family_coefficients(
                "common",
                self.bases.weight_family_common,
                3,
                ratio,
            )
            self._scale_family_coefficients(
                "common",
                self.bases.weight_family_common,
                5,
                ratio,
            )
            self._scale_family_coefficients(
                "attention",
                self.bases.weight_family_attention,
                3,
                ratio,
            )
            self._scale_family_coefficients(
                "mlp",
                self.bases.weight_family_mlp,
                1,
                ratio,
            )
        self._target_weight_stds["ATTENTION_OUTPUT"] = residual_weight_std
        self._target_weight_stds["MLP_DOWN"] = residual_weight_std

    def sheet_parameter_count(self) -> int:
        return self.plan.coefficient_counts["total"]

    def matrix_sheet_parameter_count(self) -> int:
        return self.sheet_parameter_count()

    def conventional_vector_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.vector_parameters.values())

    def matrix_dense_equivalent_count(self) -> int:
        return self.plan.dense_equivalent_matrix_count

    def dense_equivalent_count(self) -> int:
        return self.matrix_dense_equivalent_count() + self.conventional_vector_parameter_count()

    def family_report(self) -> Tuple[Dict[str, object], ...]:
        rows = []
        matrix_shapes = {
            "Q": (self.plan.n_embd, self.plan.n_embd),
            "K": (self.plan.n_embd, self.plan.n_embd),
            "V": (self.plan.n_embd, self.plan.n_embd),
            "ATTENTION_OUTPUT": (self.plan.n_embd, self.plan.n_embd),
            "MLP_UP": (self.plan.mlp_hidden, self.plan.n_embd),
            "MLP_DOWN": (self.plan.n_embd, self.plan.mlp_hidden),
        }
        for family_name in WEIGHT_FAMILIES:
            rows.append(
                {
                    "name": family_name,
                    "semantic_type": "matrix",
                    "target_weight_std": self._target_weight_stds[family_name],
                    "weight_decay": True,
                    "operational_shape_per_layer": matrix_shapes[family_name],
                    "dense_equivalent_parameters": self.plan.n_layer
                    * math.prod(matrix_shapes[family_name]),
                    "coefficient_ownership": (
                        "common+attention"
                        if family_name in ATTENTION_FAMILIES
                        else "common+mlp"
                    ),
                }
            )
        for name, parameter in self.vector_parameters.items():
            rows.append(
                {
                    "name": name,
                    "semantic_type": self.family_metadata(name).semantic_type,
                    "weight_decay": False,
                    "operational_shape": tuple(parameter.shape),
                    "persistent_parameters": parameter.numel(),
                    "coefficient_ownership": "conventional_dense_vector",
                }
            )
        return tuple(rows)

    def persistent_basis_keys(self) -> Tuple[str, ...]:
        return tuple(
            sorted(
                key
                for key in self.state_dict().keys()
                if key.startswith("bases.")
            )
        )

    def hyperblock_report(self) -> Dict[str, object]:
        return {
            "plan": self.plan.identity(),
            "basis": self.bases.diagnostics(),
            "coefficient_counts": self.plan.coefficient_counts,
            "dense_equivalent_matrix_count": self.plan.dense_equivalent_matrix_count,
            "compression_ratio": self.plan.compression_ratio,
            "conventional_vector_parameters": self.conventional_vector_parameter_count(),
            "materialization_execution": "batched_all_matrix_families_per_layer_v1",
        }
# ^^^ THOG
