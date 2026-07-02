# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sheet.compact_state import model_from_compact_state
from sheet.generation import generate_tokens
from sheet.stage4_runtime_check import compare_runtime_memory
from sheet.stage4_trainer import Stage4Trainer
from sheet.training_model import TrainingSheetGPT
from tests.stage4_test_support import (
    stage4_batch,
    stage4_model,
    stage4_tokens,
    stage4_training_config,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Stage4CudaAcceptanceTests(unittest.TestCase):
    @staticmethod
    def run_backward(model: TrainingSheetGPT, cpu_state, cuda_state):
        torch.set_rng_state(cpu_state.clone())
        torch.cuda.set_rng_state(cuda_state.clone())
        model.zero_grad(set_to_none=True)
        inputs, targets = stage4_batch()
        logits, loss = model(inputs.cuda(), targets.cuda())
        if loss is None:
            raise AssertionError("expected training loss")
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return logits.detach(), loss.detach(), gradients

    def assert_gradient_maps_close(self, left, right) -> None:
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertTrue(
                torch.allclose(left[name], right[name], rtol=1.0e-5, atol=1.0e-6),
                msg=f"CUDA gradient mismatch for {name}",
            )

    def test_cuda_segment_size_equivalence(self) -> None:
        reference = stage4_model(checkpoint_segment_size=0).cuda()
        reference.train()
        cpu_state = torch.get_rng_state().clone()
        cuda_state = torch.cuda.get_rng_state().clone()
        reference_result = self.run_backward(reference, cpu_state, cuda_state)
        for segment_size in (1, 2, 3, 7):
            checkpointed = stage4_model(
                checkpoint_segment_size=segment_size
            ).cuda()
            checkpointed.load_state_dict(reference.state_dict())
            checkpointed.train()
            result = self.run_backward(checkpointed, cpu_state, cuda_state)
            self.assertTrue(
                torch.allclose(reference_result[0], result[0], rtol=1.0e-5, atol=1.0e-6)
            )
            self.assertTrue(
                torch.allclose(reference_result[1], result[1], rtol=1.0e-6, atol=1.0e-7)
            )
            self.assert_gradient_maps_close(reference_result[2], result[2])

    def test_cuda_dropout_rng_equivalence(self) -> None:
        reference = stage4_model(dropout=0.25, checkpoint_segment_size=0).cuda()
        checkpointed = stage4_model(dropout=0.25, checkpoint_segment_size=2).cuda()
        checkpointed.load_state_dict(reference.state_dict())
        reference.train()
        checkpointed.train()
        cpu_state = torch.get_rng_state().clone()
        cuda_state = torch.cuda.get_rng_state().clone()
        reference_result = self.run_backward(reference, cpu_state, cuda_state)
        checkpointed_result = self.run_backward(checkpointed, cpu_state, cuda_state)
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

    def test_cuda_bfloat16_multi_update_equivalence(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(512)
        reference = Stage4Trainer(
            stage4_training_config(
                checkpoint_segment_size=0,
                max_updates=3,
                decay_updates=3,
                device="cuda",
                dtype="bfloat16",
            ),
            train_tokens,
            validation_tokens,
        )
        checkpointed = Stage4Trainer(
            stage4_training_config(
                checkpoint_segment_size=2,
                max_updates=3,
                decay_updates=3,
                device="cuda",
                dtype="bfloat16",
            ),
            train_tokens,
            validation_tokens,
        )
        checkpointed.model.load_state_dict(reference.model.state_dict())
        for _ in range(3):
            reference_metrics = reference.train_one_update()
            checkpointed_metrics = checkpointed.train_one_update()
            self.assertAlmostEqual(
                reference_metrics["training_loss"],
                checkpointed_metrics["training_loss"],
                places=6,
            )
            self.assertAlmostEqual(
                reference_metrics["gradient_norm"],
                checkpointed_metrics["gradient_norm"],
                places=5,
            )
        for name, reference_value in reference.model.state_dict().items():
            self.assertTrue(
                torch.allclose(
                    reference_value,
                    checkpointed.model.state_dict()[name],
                    rtol=1.0e-5,
                    atol=1.0e-6,
                ),
                msg=f"multi-update state mismatch for {name}",
            )

    def test_cuda_float16_training_smoke(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(512)
        trainer = Stage4Trainer(
            stage4_training_config(
                checkpoint_segment_size=2,
                max_updates=2,
                decay_updates=2,
                device="cuda",
                dtype="float16",
            ),
            train_tokens,
            validation_tokens,
        )
        for _ in range(2):
            metrics = trainer.train_one_update()
            self.assertTrue(torch.isfinite(torch.tensor(metrics["training_loss"])))
            self.assertTrue(torch.isfinite(torch.tensor(metrics["gradient_norm"])))
        self.assertTrue(trainer.model.last_execution_report.checkpointing_used)

    def test_cuda_resume_with_new_segment_size(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(512)
        trainer = Stage4Trainer(
            stage4_training_config(
                checkpoint_segment_size=1,
                max_updates=2,
                decay_updates=2,
                device="cuda",
                dtype="bfloat16",
            ),
            train_tokens,
            validation_tokens,
        )
        trainer.train_one_update()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            trainer.save_checkpoint(path)
            resumed = Stage4Trainer.from_checkpoint(
                path,
                train_tokens,
                validation_tokens,
                overrides={
                    "checkpoint_segment_size": 3,
                    "device": "cuda",
                    "dtype": "bfloat16",
                    "max_updates": 2,
                },
            )
        self.assertEqual(resumed.state.completed_updates, 1)
        self.assertEqual(resumed.config.checkpoint_segment_size, 3)
        resumed.train_one_update()
        self.assertEqual(resumed.state.completed_updates, 2)
        self.assertEqual(resumed.model.last_execution_report.checkpoint_segments, 2)

    def test_cuda_compact_model_inference(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(512)
        trainer = Stage4Trainer(
            stage4_training_config(
                checkpoint_segment_size=2,
                device="cuda",
                dtype="bfloat16",
            ),
            train_tokens,
            validation_tokens,
        )
        trainer.train_one_update()
        model, config = model_from_compact_state(
            trainer.checkpoint_payload(),
            device="cuda",
            dtype="bfloat16",
        )
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        first = generate_tokens(
            model,
            prompt,
            device=torch.device("cuda"),
            dtype=config.dtype,
            max_new_tokens=3,
            top_k=8,
            seed=55,
        )
        second = generate_tokens(
            model,
            prompt,
            device=torch.device("cuda"),
            dtype=config.dtype,
            max_new_tokens=3,
            top_k=8,
            seed=55,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (1, 6))
        self.assertFalse(model.training)
        self.assertFalse(hasattr(model, "optimizer"))

    def test_cuda_peak_memory_reduction(self) -> None:
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        evidence = compare_runtime_memory(device="cuda", dtype=dtype)
        self.assertTrue(evidence["satisfied"], evidence)
        self.assertLess(evidence["peak_allocated_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
