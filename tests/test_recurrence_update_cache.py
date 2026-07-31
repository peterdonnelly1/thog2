# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.recurrence_generators import BQRG_FAMILY, BQRG_VERSION, materialize_bqrg_at
from sheet.semantic_materializer import ATTENTION_QUERY_WEIGHT
from sheet.stage4_trainer import Stage4Trainer
from sheet.training_config import TrainingConfig
from sheet.training_model_factory import build_training_model


def tiny_training_config(*, accumulation_steps: int = 2) -> TrainingConfig:
    return TrainingConfig(
        model_type="thog2_sheet",
        block_size=4,
        vocab_size=64,
        n_layer=16,
        n_head=2,
        n_embd=8,
        depth_order=16,
        base_row_order=1,
        mlp_channel_order=1,
        o_attn_d_model=1,
        o_attn_qkv_per_channel=1,
        o_attn_out_per_channel=1,
        o_mlp_d_model=1,
        o_mlp_hidden=1,
        geometry_preset="depth",
        basis_family=BQRG_FAMILY,
        basis_version=BQRG_VERSION,
        checkpoint_segment_size=4,
        batch_size=1,
        gradient_accumulation_steps=accumulation_steps,
        max_updates=2,
        decay_updates=2,
        eval_batches=1,
        log_interval=1,
        device="cpu",
        dtype="float32",
    )


class RecurrenceUpdateCacheTests(unittest.TestCase):
    def test_cached_microbatch_gradients_match_direct_bqrg_gradient(self) -> None:
        torch.manual_seed(41)
        model = build_training_model(
            tiny_training_config(),
            device=torch.device("cpu"),
        )
        trajectory = model.trajectory
        controller = trajectory._recurrence_update_cache_controller
        parameter = trajectory.coefficients[ATTENTION_QUERY_WEIGHT]
        reference_parameter = parameter.detach().clone().requires_grad_()
        upstream_1 = torch.randn(8, 8)
        upstream_2 = torch.randn(8, 8)

        controller.begin()
        cached_1 = trajectory.materialize(ATTENTION_QUERY_WEIGHT, 5)
        cached_2 = trajectory.materialize(ATTENTION_QUERY_WEIGHT, 11)
        ((cached_1 * upstream_1).sum() + (cached_2 * upstream_2).sum()).backward()
        finalized = controller.finalize(unscale_factor=0.25)

        reference_1 = materialize_bqrg_at(reference_parameter, 5)
        reference_2 = materialize_bqrg_at(reference_parameter, 11)
        0.25 * ((reference_1 * upstream_1).sum() + (reference_2 * upstream_2).sum()).backward()

        self.assertIn(parameter, finalized)
        self.assertFalse(controller.active)
        self.assertIsNotNone(parameter.grad)
        self.assertIsNotNone(reference_parameter.grad)
        self.assertTrue(
            torch.allclose(
                parameter.grad,
                reference_parameter.grad,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        )

    def test_trainer_finalizes_and_discards_cache_once_per_update(self) -> None:
        tokens = torch.arange(256, dtype=torch.long) % 64
        trainer = Stage4Trainer(
            tiny_training_config(accumulation_steps=2),
            tokens,
            tokens.clone(),
        )
        try:
            controller = trainer._recurrence_update_cache_controller
            self.assertIsNotNone(controller)
            metrics = trainer.train_one_update()
            self.assertEqual(metrics["completed_updates"], 1.0)
            self.assertFalse(controller.active)
            event_names = tuple(event.name for event in trainer.events)
            self.assertIn("recurrence_update_cache_started", event_names)
            self.assertIn("recurrence_update_cache_finalized", event_names)
        finally:
            trainer.close()


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
