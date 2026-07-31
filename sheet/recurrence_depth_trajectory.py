# vvv THOG
from __future__ import annotations

import torch
from torch import Tensor

from .basis import BASIS_FAMILY_CHEBYSHEV, BASIS_VERSION, BasisCache
from .depth_trajectory import DepthTrajectory
from .geometry import SheetGeometryConfig
from .recurrence_generators import get_recurrence_generator_definition, validate_recurrence_generator_width
from .semantic_materializer import ATTENTION_KEY_WEIGHT, ATTENTION_QUERY_WEIGHT, ATTENTION_VALUE_WEIGHT, LEGACY_ATTENTION_INPUT_WEIGHT


class RecurrenceDepthTrajectory(DepthTrajectory):
    """Public DEPTH trajectory whose compact scalars parameterise a registered recurrence generator."""

    def __init__(
        self,
        config: SheetGeometryConfig,
        *,
        generator_family: str,
        runtime_dtype: torch.dtype = torch.float32,
        basis_cache: BasisCache = None,
        depth_compress_layer_norm_and_bias: bool = False,
    ) -> None:
        self.generator_definition = get_recurrence_generator_definition(generator_family)
        if "DEPTH" not in self.generator_definition.supported_targets:
            raise ValueError(
                f"recurrence generator {self.generator_definition.family}@{self.generator_definition.version} does not support DEPTH"
            )
        validate_recurrence_generator_width(self.generator_definition.family, config.depth_order)
        super().__init__(
            config,
            runtime_dtype=runtime_dtype,
            basis_version=BASIS_VERSION,
            basis_cache=basis_cache,
            basis_family=BASIS_FAMILY_CHEBYSHEV,
            depth_compress_layer_norm_and_bias=depth_compress_layer_norm_and_bias,
        )
        self.basis_family = self.generator_definition.family
        self.basis_version = self.generator_definition.version

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for item in self.metadata:
                parameter = self.coefficients[item.name]
                representation = self._representation(item)
                if representation == "depth_coefficients":
                    self.generator_definition.initialize_parameters(
                        parameter,
                        item.initialization,
                        item.target_weight_std,
                        self.config.n_layer,
                    )
                elif representation == "conventional_per_layer":
                    parameter.zero_()
                    if item.initialization == "layernorm_one":
                        parameter.fill_(1.0)
                    elif item.initialization == "zero":
                        continue
                    else:
                        raise RuntimeError(
                            f"unsupported conventional initialization policy {item.initialization} for {item.name}"
                        )
                else:
                    raise RuntimeError("recurrence generators are restricted to public DEPTH and cannot use legacy SHEET_COL vectors")

    def _materialize_depth_parameter(self, name: str, layer_index: int) -> Tensor:
        coefficient = self.coefficients[name]
        generated = self.generator_definition.materialize_at(coefficient, layer_index)
        item = self.family_metadata(name)
        expected_shape = (item.output_rows, item.row_width)
        if tuple(generated.shape) != expected_shape:
            raise RuntimeError(
                f"recurrence depth parameter {name} has shape {tuple(generated.shape)}; expected {expected_shape}"
            )
        return generated

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
        representation = self._representation(item)
        parameter = self.coefficients[name]
        if representation == "depth_coefficients":
            return self.generator_definition.materialize_at(parameter[output_row, row_index], layer_index)
        if representation == "conventional_per_layer":
            return parameter[layer_index, output_row, row_index]
        raise RuntimeError("recurrence generators are restricted to public DEPTH")
# ^^^ THOG
