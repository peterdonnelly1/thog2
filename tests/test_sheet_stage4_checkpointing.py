# vvv THOG
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from sheet.checkpointing import validate_checkpoint_segment_size
from sheet.trainer import SharedTrainer
from tests.stage4_test_support import (
    stage4_batch,
    stage4_model,
    stage4_tokens,
    stage4_training_config,
)


def run_backward(model, rng_state):
    torch.set_rng_state(rng_state.clone())
    model.zero_grad(set_to_none=True)
    inputs, targets = stage4_batch()
    logits, loss = model(inputs, targets)
    if loss is None:
        raise AssertionError("expected training loss")
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return (
        logits.detach().clone(),
        loss.detach().clone(),
        gradients,
        torch.get_rng_state().clone(),
    )


class Stage4CheckpointingTests(unittest.TestCase):
    def assert_gradient_maps_close(self, left, right) -> None:
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertTrue(
                torch.allclose(left[name], right[name], rtol=1.0e-5, atol=1.0e-6),
                msg=f"gradient mismatch for {name}",
            )

    def test_s4_01_and_s4_02_forward_and_gradient_equivalence(self) -> None:
        reference = stage4_model(checkpoint_segment_size=0)
        checkpointed = stage4_model(checkpoint_segment_size=2)
        checkpointed.load_state_dict(reference.state_dict())
        reference.train()
        checkpointed.train()
        rng_state = torch.get_rng_state().clone()

        reference_result = run_backward(reference, rng_state)
        checkpointed_result = run_backward(checkpointed, rng_state)

        self.assertTrue(
            torch.allclose(
                reference_result[0], checkpointed_result[0], rtol=1.0e-5, atol=1.0e-6
            )
        )
        self.assertTrue(
            torch.allclose(
                reference_result[1], checkpointed_result[1], rtol=1.0e-6, atol=1.0e-7
            )
        )
        self.assert_gradient_maps_close(reference_result[2], checkpointed_result[2])
        self.assertFalse(reference.last_execution_report.checkpointing_used)
        self.assertTrue(checkpointed.last_execution_report.checkpointing_used)
        self.assertEqual(checkpointed.last_execution_report.checkpoint_segments, 2)

    def test_s4_03_dropout_rng_preservation(self) -> None:
        reference = stage4_model(dropout=0.25, checkpoint_segment_size=0)
        checkpointed = stage4_model(dropout=0.25, checkpoint_segment_size=2)
        checkpointed.load_state_dict(reference.state_dict())
        reference.train()
        checkpointed.train()
        rng_state = torch.get_rng_state().clone()

        reference_result = run_backward(reference, rng_state)
        checkpointed_result = run_backward(checkpointed, rng_state)

        self.assertTrue(
            torch.allclose(
                reference_result[0], checkpointed_result[0], rtol=1.0e-5, atol=1.0e-6
            )
        )
        self.assertTrue(
            torch.allclose(
                reference_result[1], checkpointed_result[1], rtol=1.0e-6, atol=1.0e-7
            )
        )
        self.assert_gradient_maps_close(reference_result[2], checkpointed_result[2])
        self.assertTrue(torch.equal(reference_result[3], checkpointed_result[3]))

    def test_s4_04_segment_size_coverage(self) -> None:
        inputs, targets = stage4_batch()
        for segment_size in (0, 1, 2, 3, 4, 7):
            model = stage4_model(checkpoint_segment_size=segment_size)
            model.train()
            _, loss = model(inputs, targets)
            self.assertIsNotNone(loss)
            assert loss is not None
            loss.backward()
            expected_segments = (
                0 if segment_size == 0 else math.ceil(model.config.n_layer / segment_size)
            )
            self.assertEqual(
                model.last_execution_report.checkpoint_segments,
                expected_segments,
            )

    def test_s4_04_invalid_segment_sizes_fail_directly(self) -> None:
        for value in (-1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "checkpoint_segment_size"):
                validate_checkpoint_segment_size(value)  # type: ignore[arg-type]

    def test_s4_05_evaluation_bypasses_checkpointing(self) -> None:
        model = stage4_model(checkpoint_segment_size=2)
        model.eval()
        inputs, _ = stage4_batch()
        with torch.no_grad():
            model(inputs)
        self.assertFalse(model.last_execution_report.checkpointing_used)
        self.assertEqual(model.last_execution_report.checkpoint_segments, 0)

    def test_execution_control_can_change_on_resume(self) -> None:
        train_tokens, validation_tokens = stage4_tokens()
        trainer = SharedTrainer(
            stage4_training_config(checkpoint_segment_size=0),
            train_tokens,
            validation_tokens,
        )
        trainer.train_one_update()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "stage4_resume.pt"
            trainer.save_checkpoint(checkpoint_path)
            resumed = SharedTrainer.from_checkpoint(
                checkpoint_path,
                train_tokens,
                validation_tokens,
                overrides={"checkpoint_segment_size": 2},
            )
        self.assertEqual(resumed.config.checkpoint_segment_size, 2)
        self.assertEqual(resumed.model.checkpoint_segment_size, 2)
        resumed.train_one_update()


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
