# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.training_config import TrainingConfig
from sheet.training_model import TrainingDenseGPT
from sheet.training_model_factory import build_training_model


class DenseCheckpointingTests(unittest.TestCase):
    def config(self, segment_size: int) -> TrainingConfig:
        return TrainingConfig(
            model_type="dense",
            block_size=8,
            vocab_size=32,
            n_layer=4,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=True,
            depth_order=1,
            base_row_order=1,
            checkpoint_segment_size=segment_size,
            batch_size=2,
            gradient_accumulation_steps=1,
            max_updates=1,
            decay_updates=1,
            eval_interval=0,
            eval_batches=1,
            device="cpu",
            dtype="float32",
            model_seed=101,
            data_seed=202,
        )

    def test_s6_32_dense_checkpointed_forward_and_gradients_match_reference(self) -> None:
        reference = build_training_model(self.config(0))
        checkpointed = build_training_model(self.config(2))
        self.assertIsInstance(reference, TrainingDenseGPT)
        self.assertIsInstance(checkpointed, TrainingDenseGPT)
        checkpointed.load_state_dict(reference.state_dict())
        inputs = torch.arange(16, dtype=torch.long).view(2, 8) % 32
        targets = torch.roll(inputs, shifts=-1, dims=1)

        reference.train()
        checkpointed.train()
        reference_logits, reference_loss = reference(inputs, targets)
        checkpointed_logits, checkpointed_loss = checkpointed(inputs, targets)
        self.assertIsNotNone(reference_loss)
        self.assertIsNotNone(checkpointed_loss)
        self.assertTrue(torch.equal(reference_logits, checkpointed_logits))
        self.assertEqual(float(reference_loss), float(checkpointed_loss))
        reference_loss.backward()
        checkpointed_loss.backward()
        for (left_name, left), (right_name, right) in zip(
            reference.named_parameters(), checkpointed.named_parameters()
        ):
            self.assertEqual(left_name, right_name)
            self.assertIsNotNone(left.grad)
            self.assertIsNotNone(right.grad)
            self.assertTrue(
                torch.allclose(left.grad, right.grad, atol=1.0e-6, rtol=1.0e-5),
                msg=left_name,
            )
        self.assertFalse(reference.last_execution_report.checkpointing_used)
        self.assertTrue(checkpointed.last_execution_report.checkpointing_used)
        self.assertEqual(checkpointed.last_execution_report.checkpoint_segments, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
