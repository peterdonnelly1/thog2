# vvv THOG
from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Dict

import torch

from sheet.basis import BASIS_VERSION, basis_sha256
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.trajectory import SheetTrajectory


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "stage0_legacy_sheet_col_fixture.json"


class Stage0LegacySheetColBaselineTests(unittest.TestCase):
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

    def test_family_geometry_matches_fixture(self) -> None:
        trajectory = self.trajectory()
        report = trajectory.family_report()
        expected_families = self.fixture["families"]
        self.assertEqual([row["name"] for row in report], [row["name"] for row in expected_families])
        actual_by_name = {row["name"]: row for row in report}
        for expected in expected_families:
            actual = actual_by_name[expected["name"]]
            for key, expected_value in expected.items():
                if key == "family_kind":
                    actual_value = trajectory.family_metadata(expected["name"]).geometry.family_kind
                else:
                    actual_value = actual[key]
                if key == "coefficient_shape":
                    actual_value = list(actual_value)
                self.assertEqual(actual_value, expected_value, msg=f"legacy SHEET_COL family audit mismatch for {expected['name']} {key}")

    def test_public_parameter_report_matches_fixture(self) -> None:
        torch.manual_seed(self.fixture["seed"])
        report = SheetGPT(self.config()).parameter_report()
        for key, expected_value in self.fixture["parameter_report"].items():
            self.assertEqual(report[key], expected_value, msg=f"parameter report mismatch for {key}")

    def test_basis_hashes_and_non_persistence_match_fixture(self) -> None:
        trajectory = self.trajectory()
        self.assertEqual(BASIS_VERSION, self.fixture["config"]["basis_version"])
        expected_hashes = self.fixture["basis_hashes"]
        self.assertEqual(basis_sha256(trajectory.depth_basis), expected_hashes["depth_4_order_3_float32"])
        self.assertEqual(basis_sha256(trajectory.row_basis("attention_input_weight")), expected_hashes["row_16_order_8_float32"])
        self.assertEqual(basis_sha256(trajectory.row_basis("attention_input_bias")), expected_hashes["row_48_order_24_float32"])
        self.assertEqual(basis_sha256(trajectory.row_basis("mlp_contraction_weight")), expected_hashes["row_64_order_32_float32"])
        self.assertEqual(trajectory.persistent_basis_keys(), ())

    def test_selected_materializations_match_fixture(self) -> None:
        trajectory = self.trajectory()
        for name, expected in self.fixture["selected_materializations"].items():
            materialized = trajectory.materialize(name, expected["layer_index"])
            self.assertEqual(list(materialized.shape), expected["shape"], msg=name)
            expected_values = torch.tensor(expected["first_values"], dtype=materialized.dtype, device=materialized.device)
            torch.testing.assert_close(materialized.flatten()[: expected_values.numel()], expected_values, rtol=5.0e-6, atol=5.0e-7, msg=f"first materialized values changed for {name}")
            torch.testing.assert_close(materialized.sum(), torch.tensor(expected["sum"], dtype=materialized.dtype), rtol=5.0e-6, atol=5.0e-7, msg=f"materialized sum changed for {name}")
            torch.testing.assert_close(materialized.norm(), torch.tensor(expected["norm"], dtype=materialized.dtype), rtol=5.0e-6, atol=5.0e-7, msg=f"materialized norm changed for {name}")

    def test_direct_value_matches_materialize_for_sampled_matrix_entries(self) -> None:
        trajectory = self.trajectory()
        samples = (("attention_input_weight", 2, 0, 0), ("attention_input_weight", 2, 17, 9), ("attention_output_weight", 2, 4, 12), ("mlp_expansion_weight", 2, 63, 15), ("mlp_contraction_weight", 2, 15, 63))
        for name, layer_index, output_row, row_index in samples:
            materialized = trajectory.materialize(name, layer_index)
            direct = trajectory.direct_value(name, layer_index, output_row, row_index)
            torch.testing.assert_close(direct, materialized[output_row, row_index], rtol=1.0e-6, atol=1.0e-7, msg=f"direct materialization path changed for {name}")

    def test_current_attention_head_packing_audit_matches_fixture(self) -> None:
        trajectory = self.trajectory()
        config = self.config()
        audit = self.fixture["head_packing_audit"]
        head_dim = config.n_embd // config.n_head
        self.assertEqual(head_dim, audit["head_dim"])
        attention_input = trajectory.materialize("attention_input_weight", 2)
        self.assertEqual(list(attention_input.shape), audit["attention_input_weight_shape"])
        role_ranges = {"query": [0, config.n_embd], "key": [config.n_embd, 2 * config.n_embd], "value": [2 * config.n_embd, 3 * config.n_embd]}
        self.assertEqual(role_ranges, audit["attention_input_role_row_ranges"])
        role_head_ranges = {role_name: [[role_start + head_index * head_dim, role_start + (head_index + 1) * head_dim] for head_index in range(config.n_head)] for role_name, (role_start, _) in role_ranges.items()}
        self.assertEqual(role_head_ranges, audit["attention_input_role_head_row_ranges"])
        attention_output = trajectory.materialize("attention_output_weight", 2)
        self.assertEqual(list(attention_output.shape), audit["attention_output_weight_shape"])
        output_head_columns = [[head_index * head_dim, (head_index + 1) * head_dim] for head_index in range(config.n_head)]
        self.assertEqual(output_head_columns, audit["attention_output_input_head_column_ranges"])
        attention_input_shape = trajectory.family_metadata("attention_input_weight").coefficient_shape(config.depth_order)
        attention_output_geometry = trajectory.family_metadata("attention_output_weight").geometry
        self.assertEqual(attention_input_shape[0], 3 * config.n_embd)
        self.assertEqual(attention_output_geometry.row_width, config.n_embd)
        self.assertEqual(trajectory.row_basis("attention_output_weight").shape[0], config.n_embd)
        self.assertTrue(audit["legacy_sheet_col_compacts_attention_input_columns"])
        self.assertTrue(audit["legacy_sheet_col_compacts_attention_output_input_head_boundary"])
        self.assertFalse(audit["legacy_sheet_col_compacts_qkv_output_rows"])


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
