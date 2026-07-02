# vvv THOG
from __future__ import annotations

import unittest

from sheet.geometry import (
    SheetGeometryConfig,
    total_dense_equivalent_count,
    total_sheet_parameter_count,
    transformer_family_geometries,
)
from sheet.stage5_target import principal_stage5_config


class Stage5TargetGeometryTests(unittest.TestCase):
    def test_s5_principal_geometry_and_matrix_counts_are_locked(self) -> None:
        training = principal_stage5_config(
            device="cpu",
            dtype="float32",
        )
        self.assertEqual(training.n_layer, 144)
        self.assertEqual(training.n_head, 12)
        self.assertEqual(training.n_embd, 768)
        self.assertEqual(training.block_size, 256)
        self.assertEqual(training.depth_order, 16)
        self.assertEqual(training.base_row_order, 128)

        geometry = SheetGeometryConfig(
            n_layer=training.n_layer,
            n_embd=training.n_embd,
            n_head=training.n_head,
            depth_order=training.depth_order,
            base_row_order=training.base_row_order,
            bias=training.bias,
        )
        matrices = transformer_family_geometries(
            geometry,
            include_vectors=False,
        )
        rows = {family.name: family for family in matrices}
        self.assertEqual(rows["attention_input_weight"].row_order, 128)
        self.assertEqual(rows["attention_output_weight"].row_order, 128)
        self.assertEqual(rows["mlp_expansion_weight"].row_order, 128)
        self.assertEqual(rows["mlp_contraction_weight"].row_order, 512)
        self.assertEqual(
            total_sheet_parameter_count(matrices, training.depth_order),
            18_874_368,
        )
        self.assertEqual(
            total_dense_equivalent_count(matrices, training.n_layer),
            1_019_215_872,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
