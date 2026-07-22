# vvv THOG
from __future__ import annotations

import unittest
from pathlib import Path

import torch

from sheet.bases import BASIS_FAMILY_LAPPED_COSINE, lapped_cosine_basis_version
from sheet.run_config import OwtRunConfig
from sheet.training_config import TrainingConfig
from sheet.training_model_factory import build_training_model


class LappedCosineTrainingAndCheckpointTests(unittest.TestCase):
    def training_config(self, *, window_length: int = 8) -> TrainingConfig:
        return TrainingConfig(
            model_type="thog2_sheet",
            block_size=8,
            vocab_size=64,
            n_layer=8,
            n_head=2,
            n_embd=16,
            depth_order=4,
            base_row_order=8,
            mlp_channel_order=32,
            o_attn_d_model=8,
            o_attn_qkv_per_channel=4,
            o_attn_out_per_channel=4,
            o_mlp_d_model=8,
            o_mlp_hidden=32,
            basis_family=BASIS_FAMILY_LAPPED_COSINE,
            basis_version=lapped_cosine_basis_version(window_length, 0.5),
            lapped_cosine_window_length=window_length,
            lapped_cosine_overlap_fraction=0.5,
            geometry_preset="depth",
            batch_size=2,
            gradient_accumulation_steps=1,
            max_updates=2,
            warmup_updates=0,
            decay_updates=2,
            eval_interval=0,
            checkpoint_interval=0,
            checkpoint_segment_size=0,
            device="cpu",
            dtype="float32",
        )

    def test_01_tiny_depth_model_forward_backward_and_update_are_finite(self) -> None:
        config = self.training_config()
        model = build_training_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        generator = torch.Generator().manual_seed(2026)
        inputs = torch.randint(0, config.vocab_size, (2, 8), generator=generator)
        targets = torch.randint(0, config.vocab_size, (2, 8), generator=generator)
        logits, loss = model(inputs, targets)
        self.assertEqual(logits.shape, (2, 8, config.vocab_size))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        optimizer.step()

    def test_02_controls_are_checkpoint_compatibility_fields(self) -> None:
        first = self.training_config(window_length=8)
        second = self.training_config(window_length=12)
        first_signature = first.compatibility_signature()
        second_signature = second.compatibility_signature()
        self.assertEqual(first_signature["lapped_cosine_window_length"], 8)
        self.assertEqual(first_signature["lapped_cosine_overlap_fraction"], 0.5)
        self.assertNotEqual(first_signature, second_signature)
        self.assertNotEqual(first.basis_version, second.basis_version)

    def test_03_run_identity_and_training_config_preserve_controls(self) -> None:
        run = OwtRunConfig(
            model_type="sheet",
            run_name="LAPPED_TEST",
            max_iters=2,
            warmup_iters=1,
            n_layer=8,
            n_head=2,
            n_embd=16,
            block_size=8,
            o_depth=4,
            o_attn_d_model=8,
            o_attn_qkv_per_channel=4,
            o_attn_out_per_channel=4,
            o_mlp_d_model=8,
            o_mlp_hidden=32,
            basis_family=BASIS_FAMILY_LAPPED_COSINE,
            basis_version="auto",
            lapped_cosine_window_length=8,
            lapped_cosine_overlap_fraction=0.5,
            device="cpu",
            dtype="float32",
            wandb_enabled=False,
            wandb_mode="disabled",
        )
        self.assertIn("LAPPED_COSINE_DEPTH", run.artifact_name)
        self.assertIn("LCW_8", run.artifact_name)
        self.assertIn("LCO_50", run.artifact_name)
        training = run.to_training_config(
            vocab_size=64,
            world_size=1,
            out_dir=Path("."),
        )
        self.assertEqual(training.lapped_cosine_window_length, 8)
        self.assertEqual(training.lapped_cosine_overlap_fraction, 0.5)
        self.assertEqual(
            training.basis_version,
            lapped_cosine_basis_version(8, 0.5),
        )

    def test_04_non_lapped_basis_rejects_nondefault_lapped_controls(self) -> None:
        with self.assertRaisesRegex(ValueError, "only when"):
            OwtRunConfig(
                model_type="sheet",
                max_iters=2,
                warmup_iters=1,
                basis_family="haar",
                basis_version="auto",
                lapped_cosine_window_length=24,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
