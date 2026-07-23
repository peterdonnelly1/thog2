# vvv THOG
from __future__ import annotations

import unittest
from pathlib import Path

import torch

from run_thog2_owt import build_parser, config_from_arguments
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.run_config import OwtRunConfig
from sheet.training_config import TrainingConfig


class DepthLayerNormBiasModesTest(unittest.TestCase):
    def _model_config(self, *, compress: bool = False, **overrides) -> SheetGPTConfig:
        values = {
            "block_size": 4,
            "vocab_size": 32,
            "n_layer": 4,
            "n_head": 2,
            "n_embd": 8,
            "depth_order": 2,
            "base_row_order": 7,
            "mlp_channel_order": 31,
            "o_attn_d_model": 7,
            "o_attn_qkv_per_channel": 3,
            "o_attn_out_per_channel": 4,
            "o_mlp_d_model": 6,
            "o_mlp_hidden": 29,
            "geometry_preset": "depth",
            "basis_family": "chebyshev",
            "depth_compress_layer_norm_and_bias": compress,
        }
        values.update(overrides)
        return SheetGPTConfig(**values)

    def _run_config(self, *, compress: bool = False, **overrides) -> OwtRunConfig:
        values = {
            "model_type": "sheet",
            "run_name": "DEPTH_VECTOR_TEST",
            "max_iters": 4,
            "warmup_iters": 1,
            "block_size": 4,
            "n_layer": 4,
            "n_head": 2,
            "n_embd": 8,
            "o_depth": 2,
            "o_attn_d_model": 7,
            "o_attn_qkv_per_channel": 3,
            "o_attn_out_per_channel": 4,
            "o_mlp_d_model": 6,
            "o_mlp_hidden": 29,
            "geometry_preset": "depth",
            "basis_family": "chebyshev",
            "depth_compress_layer_norm_and_bias": compress,
            "device": "cpu",
            "dtype": "float32",
            "wandb_enabled": False,
            "wandb_mode": "disabled",
        }
        values.update(overrides)
        return OwtRunConfig(**values)

    def test_default_depth_keeps_layer_norm_and_bias_conventional_per_layer(self) -> None:
        config = self._model_config()
        self.assertEqual(config.base_row_order, 1)
        self.assertEqual(config.mlp_channel_order, 1)
        self.assertEqual(config.o_attn_d_model, 1)
        self.assertEqual(config.o_attn_qkv_per_channel, 1)
        self.assertEqual(config.o_attn_out_per_channel, 1)
        self.assertEqual(config.o_mlp_d_model, 1)
        self.assertEqual(config.o_mlp_hidden, 1)

        model = SheetGPT(config)
        trajectory = model.trajectory
        self.assertEqual(tuple(trajectory.coefficients["attention_query_weight"].shape), (8, 8, 2))
        self.assertEqual(tuple(trajectory.coefficients["ln_1_weight"].shape), (4, 1, 8))
        self.assertEqual(tuple(trajectory.coefficients["attention_input_bias"].shape), (4, 1, 24))
        self.assertEqual(set(dict(trajectory.bases.named_buffers())), {"depth_basis"})

        for layer_index in range(config.n_layer):
            torch.testing.assert_close(
                trajectory.materialize_vector("ln_1_weight", layer_index),
                torch.ones(config.n_embd),
            )
            torch.testing.assert_close(
                trajectory.materialize_vector("attention_input_bias", layer_index),
                torch.zeros(3 * config.n_embd),
            )

        with torch.no_grad():
            trajectory.coefficients["ln_1_weight"][2, 0, 3] = 7.0
        self.assertEqual(float(trajectory.materialize_vector("ln_1_weight", 2)[3]), 7.0)
        self.assertEqual(float(trajectory.materialize_vector("ln_1_weight", 1)[3]), 1.0)

    def test_optional_mode_purely_depth_compresses_layer_norm_and_bias(self) -> None:
        config = self._model_config(compress=True)
        model = SheetGPT(config)
        trajectory = model.trajectory

        self.assertEqual(tuple(trajectory.coefficients["ln_1_weight"].shape), (1, 8, 2))
        self.assertEqual(tuple(trajectory.coefficients["attention_input_bias"].shape), (1, 24, 2))
        self.assertEqual(set(dict(trajectory.bases.named_buffers())), {"depth_basis"})
        for parameter in trajectory.coefficients.values():
            self.assertEqual(parameter.shape[-1], config.depth_order)

        for layer_index in range(config.n_layer):
            torch.testing.assert_close(
                trajectory.materialize_vector("ln_1_weight", layer_index),
                torch.ones(config.n_embd),
                atol=1.0e-6,
                rtol=1.0e-6,
            )
            torch.testing.assert_close(
                trajectory.materialize_vector("attention_input_bias", layer_index),
                torch.zeros(3 * config.n_embd),
            )

    def test_family_report_distinguishes_the_two_representations(self) -> None:
        default_rows = {
            row["name"]: row for row in SheetGPT(self._model_config()).trajectory.family_report()
        }
        compressed_rows = {
            row["name"]: row for row in SheetGPT(self._model_config(compress=True)).trajectory.family_report()
        }
        self.assertEqual(default_rows["attention_query_weight"]["representation"], "depth_coefficients")
        self.assertEqual(default_rows["ln_1_weight"]["representation"], "conventional_per_layer")
        self.assertIsNone(default_rows["ln_1_weight"]["coefficient_shape"])
        self.assertEqual(compressed_rows["ln_1_weight"]["representation"], "depth_coefficients")
        self.assertEqual(compressed_rows["ln_1_weight"]["coefficient_shape"], (1, 8, 2))

    def test_all_within_tensor_orders_are_inert_for_depth(self) -> None:
        first = self._run_config(
            o_attn_d_model=1,
            o_attn_qkv_per_channel=1,
            o_attn_out_per_channel=1,
            o_mlp_d_model=1,
            o_mlp_hidden=1,
        )
        second = self._run_config(
            o_attn_d_model=999,
            o_attn_qkv_per_channel=999,
            o_attn_out_per_channel=999,
            o_mlp_d_model=999,
            o_mlp_hidden=999,
        )
        self.assertEqual(first.artifact_name, second.artifact_name)
        self.assertEqual(first.compact_identity(), second.compact_identity())
        self.assertEqual(first.o_attn_d_model, 1)
        self.assertEqual(second.o_mlp_hidden, 1)
        for dead_fragment in ("_Q_", "_J_", "_O_", "_X_", "_Y_"):
            self.assertNotIn(dead_fragment, first.artifact_name)
        self.assertIn("_P_2_DLB_0_", first.artifact_name)

        torch.manual_seed(1234)
        first_model = SheetGPT(self._model_config(
            base_row_order=1,
            mlp_channel_order=1,
            o_attn_d_model=1,
            o_attn_qkv_per_channel=1,
            o_attn_out_per_channel=1,
            o_mlp_d_model=1,
            o_mlp_hidden=1,
        ))
        torch.manual_seed(1234)
        second_model = SheetGPT(self._model_config(
            base_row_order=8,
            mlp_channel_order=32,
            o_attn_d_model=8,
            o_attn_qkv_per_channel=4,
            o_attn_out_per_channel=4,
            o_mlp_d_model=8,
            o_mlp_hidden=32,
        ))
        self.assertEqual(first_model.state_dict().keys(), second_model.state_dict().keys())
        for name, value in first_model.state_dict().items():
            torch.testing.assert_close(value, second_model.state_dict()[name])

    def test_parameter_accounting_does_not_double_count_conventional_vectors(self) -> None:
        model = SheetGPT(self._model_config(compress=False))
        trajectory = model.trajectory
        report = model.parameter_report()
        total_persistent = sum(parameter.numel() for parameter in model.parameters())
        external_conventional = (
            total_persistent
            - trajectory.sheet_parameter_count()
            - trajectory.conventional_repeated_parameter_count()
        )
        expected_total = external_conventional + trajectory.all_repeated_dense_equivalent_count()
        self.assertEqual(report["dense_equivalent_total_parameters"], expected_total)
        self.assertEqual(
            trajectory.conventional_repeated_parameter_count()
            + trajectory.dense_equivalent_count(),
            trajectory.all_repeated_dense_equivalent_count(),
        )

    def test_depth_mode_is_checkpoint_and_artifact_identity(self) -> None:
        conventional = self._run_config(compress=False)
        compressed = self._run_config(compress=True)
        self.assertIn("_DLB_0_", conventional.artifact_name)
        self.assertIn("_DLB_1_", compressed.artifact_name)
        self.assertNotEqual(conventional.artifact_name, compressed.artifact_name)

        conventional_training = conventional.to_training_config(
            vocab_size=32,
            world_size=1,
            out_dir=Path("out-conventional"),
        )
        compressed_training = compressed.to_training_config(
            vocab_size=32,
            world_size=1,
            out_dir=Path("out-compressed"),
        )
        self.assertFalse(conventional_training.depth_compress_layer_norm_and_bias)
        self.assertTrue(compressed_training.depth_compress_layer_norm_and_bias)
        self.assertNotEqual(
            conventional_training.compatibility_signature(),
            compressed_training.compatibility_signature(),
        )

    def test_option_is_rejected_outside_depth(self) -> None:
        with self.assertRaisesRegex(ValueError, "only for geometry_preset='depth'"):
            self._model_config(
                compress=True,
                geometry_preset="legacy_sheet_col",
                base_row_order=8,
                mlp_channel_order=32,
                o_attn_d_model=8,
                o_attn_qkv_per_channel=4,
                o_attn_out_per_channel=4,
                o_mlp_d_model=8,
                o_mlp_hidden=32,
            )
        with self.assertRaisesRegex(ValueError, "only for geometry_preset='depth'"):
            self._run_config(
                compress=True,
                geometry_preset="legacy_sheet_col",
                o_attn_d_model=8,
                o_attn_qkv_per_channel=4,
                o_attn_out_per_channel=4,
                o_mlp_d_model=8,
                o_mlp_hidden=32,
            )
        with self.assertRaisesRegex(ValueError, "only for geometry_preset='depth'"):
            TrainingConfig(
                model_type="dense",
                n_layer=2,
                n_head=2,
                n_embd=8,
                depth_order=1,
                base_row_order=1,
                max_updates=2,
                decay_updates=2,
                depth_compress_layer_norm_and_bias=True,
            )

    def test_optimizer_groups_keep_layer_norm_and_bias_out_of_weight_decay(self) -> None:
        for compress in (False, True):
            model = SheetGPT(self._model_config(compress=compress))
            decay, no_decay = model.optimizer_parameter_groups(weight_decay=0.1)
            decay_names = set(decay["parameter_names"])
            no_decay_names = set(no_decay["parameter_names"])
            self.assertFalse(decay_names & no_decay_names)
            self.assertIn("trajectory.coefficients.ln_1_weight", no_decay_names)
            self.assertIn("trajectory.coefficients.attention_input_bias", no_decay_names)
            self.assertIn("trajectory.coefficients.attention_query_weight", decay_names)
            all_parameters = list(decay["params"]) + list(no_decay["params"])
            self.assertEqual(len(all_parameters), len({id(parameter) for parameter in all_parameters}))

    def test_forward_backward_is_finite_in_both_modes(self) -> None:
        for compress in (False, True):
            torch.manual_seed(42)
            model = SheetGPT(self._model_config(compress=compress))
            indices = torch.randint(0, 32, (2, 4))
            targets = torch.randint(0, 32, (2, 4))
            logits, loss = model(indices, targets)
            self.assertEqual(tuple(logits.shape), (2, 4, 32))
            self.assertIsNotNone(loss)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_cli_boolean_reaches_the_canonical_run_config(self) -> None:
        arguments = build_parser().parse_args([
            "--model-type",
            "sheet",
            "--depth-compress-layer-norm-and-bias",
        ])
        config = config_from_arguments(arguments)
        self.assertTrue(config.depth_compress_layer_norm_and_bias)
        self.assertIn("_DLB_1_", config.artifact_name)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
