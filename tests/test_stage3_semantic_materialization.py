# vvv THOG
from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Dict

import torch
from torch.nn import functional as F

from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import (
    ATTENTION_KEY_BIAS,
    ATTENTION_KEY_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_BIAS,
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_VALUE_BIAS,
    ATTENTION_VALUE_WEIGHT,
    LEGACY_ATTENTION_INPUT_BIAS,
    LEGACY_ATTENTION_INPUT_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
    SEMANTIC_MATRIX_FAMILIES,
    SEMANTIC_VECTOR_FAMILIES,
    LegacySheetColMaterializer,
)
from sheet.trajectory import SheetTrajectory


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "stage0_legacy_sheet_col_fixture.json"


class Stage3SemanticMaterializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
            cls.fixture: Dict[str, Any] = json.load(handle)

    def config(self) -> SheetGPTConfig:
        return SheetGPTConfig(**self.fixture["config"])

    def trajectory(self) -> SheetTrajectory:
        torch.manual_seed(self.fixture["seed"])
        config = self.config()
        return SheetTrajectory(config.sheet_geometry(), runtime_dtype=torch.float32, basis_version=config.basis_version)

    def materializer(self) -> LegacySheetColMaterializer:
        return LegacySheetColMaterializer(self.trajectory())

    def test_semantic_family_registry_and_legacy_mapping(self) -> None:
        materializer = self.materializer()
        self.assertEqual(
            SEMANTIC_MATRIX_FAMILIES,
            (
                ATTENTION_QUERY_WEIGHT,
                ATTENTION_KEY_WEIGHT,
                ATTENTION_VALUE_WEIGHT,
                ATTENTION_OUTPUT_WEIGHT,
                MLP_EXPANSION_WEIGHT,
                MLP_CONTRACTION_WEIGHT,
            ),
        )
        self.assertEqual(
            SEMANTIC_VECTOR_FAMILIES,
            (ATTENTION_QUERY_BIAS, ATTENTION_KEY_BIAS, ATTENTION_VALUE_BIAS),
        )
        self.assertEqual(materializer.matrix_spec(ATTENTION_QUERY_WEIGHT).legacy_family, LEGACY_ATTENTION_INPUT_WEIGHT)
        self.assertEqual(materializer.matrix_spec(ATTENTION_KEY_WEIGHT).legacy_family, LEGACY_ATTENTION_INPUT_WEIGHT)
        self.assertEqual(materializer.matrix_spec(ATTENTION_VALUE_WEIGHT).legacy_family, LEGACY_ATTENTION_INPUT_WEIGHT)
        self.assertEqual(materializer.vector_spec(ATTENTION_QUERY_BIAS).legacy_family, LEGACY_ATTENTION_INPUT_BIAS)
        self.assertEqual(materializer.matrix_spec(ATTENTION_OUTPUT_WEIGHT).legacy_family, ATTENTION_OUTPUT_WEIGHT)
        self.assertEqual(materializer.matrix_spec(MLP_EXPANSION_WEIGHT).legacy_family, MLP_EXPANSION_WEIGHT)
        self.assertEqual(materializer.matrix_spec(MLP_CONTRACTION_WEIGHT).legacy_family, MLP_CONTRACTION_WEIGHT)
        with self.assertRaisesRegex(KeyError, "unknown semantic"):
            materializer.materialize("not_a_family", 0)

    def test_semantic_shapes_and_head_metadata_match_stage0_audit(self) -> None:
        config = self.config()
        materializer = self.materializer()
        self.assertEqual(tuple(materializer.materialize_matrix(ATTENTION_QUERY_WEIGHT, 2).shape), (config.n_embd, config.n_embd))
        self.assertEqual(tuple(materializer.materialize_matrix(ATTENTION_KEY_WEIGHT, 2).shape), (config.n_embd, config.n_embd))
        self.assertEqual(tuple(materializer.materialize_matrix(ATTENTION_VALUE_WEIGHT, 2).shape), (config.n_embd, config.n_embd))
        self.assertEqual(tuple(materializer.materialize_matrix(ATTENTION_OUTPUT_WEIGHT, 2).shape), (config.n_embd, config.n_embd))
        self.assertEqual(tuple(materializer.materialize_matrix(MLP_EXPANSION_WEIGHT, 2).shape), (4 * config.n_embd, config.n_embd))
        self.assertEqual(tuple(materializer.materialize_matrix(MLP_CONTRACTION_WEIGHT, 2).shape), (config.n_embd, 4 * config.n_embd))
        self.assertEqual(tuple(materializer.materialize_vector(ATTENTION_QUERY_BIAS, 2).shape), (config.n_embd,))
        self.assertEqual(tuple(materializer.materialize_vector(ATTENTION_KEY_BIAS, 2).shape), (config.n_embd,))
        self.assertEqual(tuple(materializer.materialize_vector(ATTENTION_VALUE_BIAS, 2).shape), (config.n_embd,))

        metadata = materializer.head_metadata()
        audit = self.fixture["head_packing_audit"]
        self.assertEqual(metadata["head_dim"], audit["head_dim"])
        self.assertEqual(
            {name: list(bounds) for name, bounds in metadata["attention_input_role_row_ranges"].items()},
            audit["attention_input_role_row_ranges"],
        )
        self.assertEqual(
            {name: [list(bounds) for bounds in ranges] for name, ranges in metadata["attention_input_role_head_row_ranges"].items()},
            audit["attention_input_role_head_row_ranges"],
        )
        self.assertEqual(
            [list(bounds) for bounds in metadata["attention_output_input_head_column_ranges"]],
            audit["attention_output_input_head_column_ranges"],
        )

    def test_semantic_materializations_reconstruct_legacy_packed_tensors_exactly(self) -> None:
        trajectory = self.trajectory()
        materializer = LegacySheetColMaterializer(trajectory)
        legacy_weight = trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, 2)
        legacy_bias = trajectory.materialize_vector(LEGACY_ATTENTION_INPUT_BIAS, 2)
        torch.testing.assert_close(materializer.reconstructed_attention_input_weight(2), legacy_weight, rtol=0.0, atol=0.0)
        torch.testing.assert_close(materializer.reconstructed_attention_input_bias(2), legacy_bias, rtol=0.0, atol=0.0)
        torch.testing.assert_close(materializer.packed_attention_input_weight(2), legacy_weight, rtol=0.0, atol=0.0)
        torch.testing.assert_close(materializer.packed_attention_input_bias(2), legacy_bias, rtol=0.0, atol=0.0)
        torch.testing.assert_close(materializer.materialize_matrix(ATTENTION_OUTPUT_WEIGHT, 2), trajectory.materialize(ATTENTION_OUTPUT_WEIGHT, 2), rtol=0.0, atol=0.0)
        torch.testing.assert_close(materializer.materialize_matrix(MLP_EXPANSION_WEIGHT, 2), trajectory.materialize(MLP_EXPANSION_WEIGHT, 2), rtol=0.0, atol=0.0)
        torch.testing.assert_close(materializer.materialize_matrix(MLP_CONTRACTION_WEIGHT, 2), trajectory.materialize(MLP_CONTRACTION_WEIGHT, 2), rtol=0.0, atol=0.0)

    def test_stage0_selected_materializations_still_match_through_semantic_seam(self) -> None:
        materializer = self.materializer()
        selected = self.fixture["selected_materializations"]
        semantic_names = {
            ATTENTION_OUTPUT_WEIGHT: ATTENTION_OUTPUT_WEIGHT,
            MLP_EXPANSION_WEIGHT: MLP_EXPANSION_WEIGHT,
            MLP_CONTRACTION_WEIGHT: MLP_CONTRACTION_WEIGHT,
        }
        for semantic_name, fixture_name in semantic_names.items():
            expected = selected[fixture_name]
            materialized = materializer.materialize_matrix(semantic_name, expected["layer_index"])
            expected_values = torch.tensor(expected["first_values"], dtype=materialized.dtype)
            torch.testing.assert_close(materialized.flatten()[: expected_values.numel()], expected_values, rtol=5.0e-6, atol=5.0e-7)
            torch.testing.assert_close(materialized.sum(), torch.tensor(expected["sum"], dtype=materialized.dtype), rtol=5.0e-6, atol=5.0e-7)
            torch.testing.assert_close(materialized.norm(), torch.tensor(expected["norm"], dtype=materialized.dtype), rtol=5.0e-6, atol=5.0e-7)

    def test_direct_values_and_boundary_linear_projection_are_preserved(self) -> None:
        trajectory = self.trajectory()
        materializer = LegacySheetColMaterializer(trajectory)
        semantic_direct = materializer.direct_matrix_value(ATTENTION_KEY_WEIGHT, 2, 1, 9)
        legacy_direct = trajectory.direct_value(LEGACY_ATTENTION_INPUT_WEIGHT, 2, self.config().n_embd + 1, 9)
        torch.testing.assert_close(semantic_direct, legacy_direct, rtol=0.0, atol=0.0)
        semantic_bias = materializer.direct_vector_value(ATTENTION_VALUE_BIAS, 2, 3)
        legacy_bias = trajectory.materialize_vector(LEGACY_ATTENTION_INPUT_BIAS, 2)[2 * self.config().n_embd + 3]
        torch.testing.assert_close(semantic_bias, legacy_bias, rtol=0.0, atol=0.0)

        inputs = torch.randn(2, 3, self.config().n_embd)
        legacy_projection = F.linear(inputs, trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, 2), trajectory.materialize_vector(LEGACY_ATTENTION_INPUT_BIAS, 2))
        semantic_projection = F.linear(inputs, materializer.reconstructed_attention_input_weight(2), materializer.reconstructed_attention_input_bias(2))
        torch.testing.assert_close(semantic_projection, legacy_projection, rtol=0.0, atol=0.0)

    def test_existing_model_forward_backward_and_optimizer_coverage_still_work(self) -> None:
        torch.manual_seed(self.fixture["seed"])
        model = SheetGPT(self.config())
        idx = torch.randint(0, self.config().vocab_size, (2, 4))
        targets = torch.randint(0, self.config().vocab_size, (2, 4))
        logits, loss = model(idx, targets)
        self.assertEqual(tuple(logits.shape), (2, 4, self.config().vocab_size))
        self.assertIsNotNone(loss)
        loss.backward()
        groups = model.optimizer_parameter_groups(0.1)
        grouped = sum(parameter.numel() for group in groups for parameter in group["params"])
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        self.assertEqual(grouped, trainable)
        self.assertEqual(model.parameter_report()["persistent_parameters"], self.fixture["parameter_report"]["persistent_parameters"])

    def test_negative_validation_paths(self) -> None:
        materializer = self.materializer()
        with self.assertRaises(KeyError):
            materializer.materialize_matrix(ATTENTION_QUERY_BIAS, 0)
        with self.assertRaises(KeyError):
            materializer.materialize_vector(ATTENTION_QUERY_WEIGHT, 0)
        with self.assertRaises(IndexError):
            materializer.materialize_matrix(ATTENTION_QUERY_WEIGHT, self.config().n_layer)
        with self.assertRaises(IndexError):
            materializer.direct_matrix_value(ATTENTION_QUERY_WEIGHT, 0, self.config().n_embd, 0)
        with self.assertRaises(IndexError):
            materializer.direct_matrix_value(ATTENTION_QUERY_WEIGHT, 0, 0, self.config().n_embd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
