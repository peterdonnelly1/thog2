from __future__ import annotations

import unittest

import torch

from sheet.bases import BASIS_FAMILY_CHEBYSHEV
from sheet.geometry import SheetGeometryConfig
from sheet.jpeg_like_v1_trajectory import JpegLikeV1Trajectory
from sheet.semantic_materializer import MLP_CONTRACTION_WEIGHT, MLP_EXPANSION_WEIGHT


class JpegLikeV1TrajectoryTests(unittest.TestCase):
    def config(self, *, retained_modes: int = 4) -> SheetGeometryConfig:
        return SheetGeometryConfig(
            n_layer=4,
            n_embd=8,
            n_head=2,
            depth_order=3,
            base_row_order=4,
            o_attn_d_model=4,
            o_attn_qkv_per_channel=2,
            o_attn_out_per_channel=2,
            o_mlp_d_model=4,
            o_mlp_hidden=retained_modes,
            bias=True,
        )

    def trajectory(self, *, retained_modes: int = 4, compressor: str = "dct") -> JpegLikeV1Trajectory:
        return JpegLikeV1Trajectory(
            self.config(retained_modes=retained_modes),
            mlp_hidden_group_size=4,
            mlp_hidden_compressor=compressor,
            runtime_dtype=torch.float64,
            basis_family=BASIS_FAMILY_CHEBYSHEV,
        )

    def test_coefficient_shape_targets_only_mlp_up_hidden_axis(self) -> None:
        trajectory = self.trajectory()
        self.assertEqual(tuple(trajectory.coefficients[MLP_EXPANSION_WEIGHT].shape), (8, 4, 8, 3))
        self.assertEqual(tuple(trajectory.coefficients[MLP_CONTRACTION_WEIGHT].shape), (8, 32, 3))
        report = {row["name"]: row for row in trajectory.family_report()}
        self.assertEqual(report[MLP_EXPANSION_WEIGHT]["axis"], "MLP_HIDDEN")
        self.assertEqual(report[MLP_EXPANSION_WEIGHT]["group_size"], 4)
        self.assertEqual(report[MLP_EXPANSION_WEIGHT]["retained_modes"], 4)

    def test_full_mode_encode_decode_is_exact_for_dct(self) -> None:
        trajectory = self.trajectory()
        source = torch.randn(32, 8, 3, dtype=torch.float64)
        encoded = trajectory.encode_depth_coefficients(source)
        reconstructed = trajectory.reconstruct_depth_coefficients(encoded)
        torch.testing.assert_close(reconstructed, source, rtol=1.0e-12, atol=1.0e-12)

    def test_full_mode_encode_decode_is_exact_for_another_registered_compressor(self) -> None:
        trajectory = self.trajectory(compressor="haar")
        source = torch.randn(32, 8, 3, dtype=torch.float64)
        encoded = trajectory.encode_depth_coefficients(source)
        reconstructed = trajectory.reconstruct_depth_coefficients(encoded)
        torch.testing.assert_close(reconstructed, source, rtol=1.0e-12, atol=1.0e-12)
        self.assertEqual(trajectory.mlp_hidden_compressor, "haar")

    def test_materialization_matches_depth_then_local_reconstruction(self) -> None:
        trajectory = self.trajectory()
        source = torch.randn(32, 8, 3, dtype=torch.float64)
        with torch.no_grad():
            trajectory.coefficients[MLP_EXPANSION_WEIGHT].copy_(trajectory.encode_depth_coefficients(source))
        for layer_index in range(4):
            expected = torch.einsum("p,rdp->rd", trajectory.depth_basis[layer_index], source)
            actual = trajectory.materialize(MLP_EXPANSION_WEIGHT, layer_index)
            torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_direct_value_matches_materialized_weight(self) -> None:
        trajectory = self.trajectory()
        materialized = trajectory.materialize(MLP_EXPANSION_WEIGHT, 2)
        for output_row, row_index in ((0, 0), (3, 7), (4, 2), (31, 6)):
            direct = trajectory.direct_value(MLP_EXPANSION_WEIGHT, 2, output_row, row_index)
            torch.testing.assert_close(direct, materialized[output_row, row_index])

    def test_backward_reaches_native_local_coefficients(self) -> None:
        trajectory = self.trajectory(retained_modes=2)
        weight = trajectory.materialize(MLP_EXPANSION_WEIGHT, 1)
        loss = weight.square().mean()
        loss.backward()
        gradient = trajectory.coefficients[MLP_EXPANSION_WEIGHT].grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_truncated_mode_parameter_count_is_proportional(self) -> None:
        full = self.trajectory(retained_modes=4)
        half = self.trajectory(retained_modes=2)
        full_count = full.jpeg_metadata.sheet_parameter_count(full.config.depth_order)
        half_count = half.jpeg_metadata.sheet_parameter_count(half.config.depth_order)
        self.assertEqual(half_count * 2, full_count)

    def test_rejects_group_size_that_does_not_divide_mlp_hidden(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            JpegLikeV1Trajectory(
                self.config(),
                mlp_hidden_group_size=6,
                mlp_hidden_compressor="dct",
                runtime_dtype=torch.float64,
                basis_family="chebyshev",
            )

    def test_rejects_more_modes_than_group_positions(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            JpegLikeV1Trajectory(
                self.config(retained_modes=5),
                mlp_hidden_group_size=4,
                mlp_hidden_compressor="dct",
                runtime_dtype=torch.float64,
                basis_family="chebyshev",
            )

    def test_shape_validation_on_encode_and_decode(self) -> None:
        trajectory = self.trajectory()
        with self.assertRaisesRegex(ValueError, "expected"):
            trajectory.encode_depth_coefficients(torch.zeros(31, 8, 3))
        with self.assertRaisesRegex(ValueError, "expected"):
            trajectory.reconstruct_depth_coefficients(torch.zeros(7, 4, 8, 3))


if __name__ == "__main__":
    unittest.main()
