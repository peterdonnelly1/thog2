from __future__ import annotations

import dataclasses
import unittest

import torch

from sheet.compact_identity import (
    ATTENTION_GEOMETRY_DEPTH,
    GEOMETRY_PRESET_JPEG_LIKE_V1,
    JPEG_LIKE_V1_MATERIALIZATION_VERSION,
    MLP_GEOMETRY_JPEG_LIKE_V1,
    compact_identity_metadata,
    resolve_compact_selectors,
)
from sheet.jpeg_like_v1_trajectory import JpegLikeV1Trajectory
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.run_config import OwtRunConfig
from sheet.training_config import ROW_ORDER_SCALING_RULE, TrainingConfig


class JpegLikeV1IntegrationTests(unittest.TestCase):
    def test_preset_resolves_depth_plus_jpeg_like_mlp(self) -> None:
        selectors = resolve_compact_selectors(
            geometry_preset=GEOMETRY_PRESET_JPEG_LIKE_V1,
            basis_family="chebyshev",
        )
        self.assertEqual(selectors.attention_geometry, ATTENTION_GEOMETRY_DEPTH)
        self.assertEqual(selectors.mlp_geometry, MLP_GEOMETRY_JPEG_LIKE_V1)

    def test_selector_infers_preset_from_explicit_module_geometries(self) -> None:
        selectors = resolve_compact_selectors(
            attention_geometry="depth",
            mlp_geometry="jpeg_like_v1",
            basis_family="dct",
        )
        self.assertEqual(selectors.geometry_preset, GEOMETRY_PRESET_JPEG_LIKE_V1)

    def test_identity_records_geometry_and_independent_compressor(self) -> None:
        identity = compact_identity_metadata(
            n_layer=4,
            n_embd=8,
            n_head=2,
            o_depth=3,
            o_attn_d_model=4,
            o_attn_qkv_per_channel=2,
            o_attn_out_per_channel=2,
            o_mlp_d_model=4,
            o_mlp_hidden=2,
            mlp_hidden_group_size=4,
            mlp_hidden_compressor="haar",
            row_order_scaling_rule=ROW_ORDER_SCALING_RULE,
            geometry_preset="jpeg_like_v1",
            basis_family="chebyshev",
        )
        self.assertEqual(identity["materialization_version"], JPEG_LIKE_V1_MATERIALIZATION_VERSION)
        self.assertEqual(identity["basis_family"], "chebyshev")
        self.assertEqual(identity["mlp_hidden_compressor"], "haar")
        self.assertEqual(identity["mlp_hidden_group_size"], 4)
        self.assertEqual(identity["o_mlp_hidden"], 2)

    def test_run_config_artifact_identity_includes_compressor_group_and_y(self) -> None:
        config = OwtRunConfig(
            model_type="sheet",
            max_iters=2,
            warmup_iters=0,
            n_layer=4,
            n_head=2,
            n_embd=8,
            o_depth=3,
            o_attn_d_model=4,
            o_attn_qkv_per_channel=2,
            o_attn_out_per_channel=2,
            o_mlp_d_model=4,
            o_mlp_hidden=2,
            mlp_hidden_group_size=4,
            mlp_hidden_compressor="dct",
            geometry_preset="jpeg_like_v1",
            basis_family="chebyshev",
            device="cpu",
            dtype="float32",
        )
        self.assertIn("CHEBY_JPEG_LIKE_V1_DCT", config.artifact_name)
        self.assertIn("Y_2", config.artifact_name)
        self.assertIn("MHG_4", config.artifact_name)
        identity = config.compact_identity()
        self.assertEqual(identity["basis_family"], "chebyshev")
        self.assertEqual(identity["mlp_hidden_compressor"], "dct")

    def test_training_config_round_trip_preserves_new_checkpoint_fields(self) -> None:
        config = TrainingConfig(
            model_type="thog2_sheet",
            block_size=8,
            vocab_size=32,
            n_layer=4,
            n_head=2,
            n_embd=8,
            depth_order=3,
            base_row_order=4,
            o_attn_d_model=4,
            o_attn_qkv_per_channel=2,
            o_attn_out_per_channel=2,
            o_mlp_d_model=4,
            o_mlp_hidden=2,
            mlp_hidden_group_size=4,
            mlp_hidden_compressor="haar",
            geometry_preset="jpeg_like_v1",
            basis_family="dct",
            max_updates=2,
            decay_updates=2,
        )
        restored = TrainingConfig(**dataclasses.asdict(config))
        self.assertEqual(restored.compatibility_signature(), config.compatibility_signature())
        arguments = restored.model_arguments()
        self.assertEqual(arguments["mlp_hidden_group_size"], 4)
        self.assertEqual(arguments["mlp_hidden_compressor"], "haar")

    def test_model_uses_jpeg_like_trajectory_and_runs_forward_backward(self) -> None:
        torch.manual_seed(7)
        model = SheetGPT(
            SheetGPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=4,
                n_head=2,
                n_embd=8,
                depth_order=3,
                base_row_order=4,
                o_attn_d_model=4,
                o_attn_qkv_per_channel=2,
                o_attn_out_per_channel=2,
                o_mlp_d_model=4,
                o_mlp_hidden=2,
                mlp_hidden_group_size=4,
                mlp_hidden_compressor="dct",
                geometry_preset="jpeg_like_v1",
                basis_family="chebyshev",
                direct_factorised_mlp=False,
            )
        )
        self.assertIsInstance(model.trajectory, JpegLikeV1Trajectory)
        indices = torch.randint(0, 32, (2, 8))
        logits, loss = model(indices, indices)
        self.assertEqual(tuple(logits.shape), (2, 8, 32))
        self.assertIsNotNone(loss)
        loss.backward()
        gradient = model.trajectory.coefficients["mlp_expansion_weight"].grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_existing_depth_preset_still_selects_depth_trajectory(self) -> None:
        model = SheetGPT(
            SheetGPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=4,
                n_head=2,
                n_embd=8,
                depth_order=3,
                base_row_order=4,
                o_attn_d_model=4,
                o_attn_qkv_per_channel=2,
                o_attn_out_per_channel=2,
                o_mlp_d_model=4,
                o_mlp_hidden=4,
                geometry_preset="depth",
                basis_family="chebyshev",
            )
        )
        self.assertEqual(type(model.trajectory).__name__, "DepthTrajectory")

    def test_identity_rejects_invalid_group_or_y(self) -> None:
        common = dict(
            n_layer=4,
            n_embd=8,
            n_head=2,
            o_depth=3,
            o_attn_d_model=4,
            o_attn_qkv_per_channel=2,
            o_attn_out_per_channel=2,
            o_mlp_d_model=4,
            row_order_scaling_rule=ROW_ORDER_SCALING_RULE,
            geometry_preset="jpeg_like_v1",
            basis_family="dct",
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            compact_identity_metadata(
                **common,
                o_mlp_hidden=2,
                mlp_hidden_group_size=6,
                mlp_hidden_compressor="dct",
            )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            compact_identity_metadata(
                **common,
                o_mlp_hidden=5,
                mlp_hidden_group_size=4,
                mlp_hidden_compressor="dct",
            )


if __name__ == "__main__":
    unittest.main()
