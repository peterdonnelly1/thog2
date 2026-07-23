# vvv THOG
from __future__ import annotations

import unittest
from pathlib import Path

from sheet.run_config import OwtRunConfig


class OwtRunConfigTests(unittest.TestCase):
    def test_s6_28_shared_public_names_map_to_internal_config(self) -> None:
        config = OwtRunConfig(
            model_type="sheet",
            max_iters=20,
            warmup_iters=2,
            eval_iters=7,
            min_lr=1.0e-5,
            gradient_accumulation_steps=8,
            checkpoint_segment_size=3,
        )
        internal = config.to_training_config(
            vocab_size=128,
            world_size=2,
            out_dir=Path("test-output"),
        )
        self.assertEqual(internal.max_updates, 20)
        self.assertEqual(internal.warmup_updates, 2)
        self.assertEqual(internal.eval_batches, 7)
        self.assertEqual(internal.min_learning_rate, 1.0e-5)
        self.assertEqual(internal.gradient_accumulation_steps, 4)
        self.assertEqual(internal.checkpoint_segment_size, 3)
        canonical = config.canonical_dict(world_size=2)
        self.assertIn("max_iters", canonical)
        self.assertIn("warmup_iters", canonical)
        self.assertIn("eval_iters", canonical)
        self.assertIn("min_lr", canonical)
        self.assertNotIn("max_updates", canonical)
        self.assertNotIn("warmup_updates", canonical)
        self.assertNotIn("eval_batches", canonical)
        self.assertNotIn("min_learning_rate", canonical)

    def test_s6_29_global_accumulation_and_tokens_match_thog_semantics(self) -> None:
        config = OwtRunConfig(
            model_type="dense",
            batch_size=12,
            gradient_accumulation_steps=160,
            block_size=256,
        )
        self.assertEqual(config.local_gradient_accumulation_steps(2), 80)
        self.assertEqual(config.tokens_per_iter(), 12 * 160 * 256)
        with self.assertRaisesRegex(ValueError, "divisible"):
            config.local_gradient_accumulation_steps(3)

    def test_s6_30_checkpointing_is_explicit_and_shared(self) -> None:
        for model_type in ("dense", "sheet"):
            enabled = OwtRunConfig(
                model_type=model_type,
                activation_checkpointing=True,
                checkpoint_segment_size=12,
            ).to_training_config(
                vocab_size=128,
                world_size=1,
                out_dir=Path("test-output"),
            )
            disabled = OwtRunConfig(
                model_type=model_type,
                activation_checkpointing=False,
                checkpoint_segment_size=12,
            ).to_training_config(
                vocab_size=128,
                world_size=1,
                out_dir=Path("test-output"),
            )
            self.assertEqual(enabled.checkpoint_segment_size, 12)
            self.assertEqual(disabled.checkpoint_segment_size, 0)

    def test_s6_31_dense_omits_sheet_fields_and_depth_canonicalizes_dead_orders(self) -> None:
        dense = OwtRunConfig(model_type="dense").canonical_dict(world_size=1)
        sheet = OwtRunConfig(
            model_type="sheet",
            o_attn_d_model=64,
            o_attn_qkv_per_channel=6,
            o_attn_out_per_channel=6,
            o_mlp_d_model=64,
            o_mlp_hidden=256,
        ).canonical_dict(world_size=1)
        semantic_orders = {
            "o_depth": 16,
            "o_attn_d_model": 1,
            "o_attn_qkv_per_channel": 1,
            "o_attn_out_per_channel": 1,
            "o_mlp_d_model": 1,
            "o_mlp_hidden": 1,
        }
        for name, expected in semantic_orders.items():
            self.assertNotIn(name, dense)
            self.assertEqual(sheet[name], expected)
        self.assertNotIn("depth_compress_layer_norm_and_bias", dense)
        self.assertFalse(sheet["depth_compress_layer_norm_and_bias"])
        self.assertNotIn("depth_order", dense)
        self.assertNotIn("base_row_order", dense)
        self.assertNotIn("depth_order", sheet)
        self.assertNotIn("base_row_order", sheet)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
