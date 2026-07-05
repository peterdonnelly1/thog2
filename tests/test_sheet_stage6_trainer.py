# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sheet.stage6_trainer import Stage6Trainer
from sheet.training_config import TrainingConfig


class Stage6TrainerTests(unittest.TestCase):
    def token_splits(self):
        train = torch.arange(1024, dtype=torch.long) % 32
        validation = torch.roll(train, shifts=11)
        return train, validation

    def config(self, model_type: str, out_dir: Path) -> TrainingConfig:
        return TrainingConfig(
            model_type=model_type,
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=True,
            depth_order=2,
            base_row_order=4,
            checkpoint_segment_size=(1 if model_type == "thog2_sheet" else 0),
            batch_size=2,
            gradient_accumulation_steps=1,
            max_updates=2,
            learning_rate=1.0e-3,
            min_learning_rate=1.0e-4,
            warmup_updates=0,
            decay_updates=2,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            grad_clip=1.0,
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=0,
            log_interval=1,
            model_seed=6101,
            data_seed=6102,
            device="cpu",
            dtype="float32",
            out_dir=str(out_dir),
        )

    def run_case(self, model_type: str):
        train, validation = self.token_splits()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / model_type
            run_dir.mkdir()
            result_path = run_dir / "result.json"
            trainer = Stage6Trainer(
                self.config(model_type, run_dir),
                train,
                validation,
            )
            try:
                result = trainer.run_pilot(
                    run_id=model_type,
                    protocol_sha256="protocol",
                    dataset={"fixture": True},
                    result_path=result_path,
                )
            finally:
                trainer.close()
            self.assertTrue(result_path.exists())
            self.assertEqual(result["budget"]["completed_updates"], 2)
            self.assertEqual(result["budget"]["consumed_tokens"], 32)
            self.assertEqual(
                [row["completed_updates"] for row in result["evaluations"]],
                [0, 1, 2],
            )
            self.assertEqual(len(result["trace"]["training_starts"]), 2)
            self.assertGreater(result["checkpoint"]["bytes"], 0)
            return result

    def test_s6_12_sheet_pilot_records_diagnostics(self) -> None:
        result = self.run_case("thog2_sheet")
        self.assertIsNotNone(result["sheet_diagnostics"])
        self.assertEqual(len(result["gradient_diagnostics"]), 2)
        self.assertEqual(
            result["sheet_diagnostics"]["compact_state_violations"],
            [],
        )

    def test_s6_13_dense_pilot_uses_same_evidence_schema(self) -> None:
        result = self.run_case("dense")
        self.assertIsNone(result["sheet_diagnostics"])
        self.assertEqual(result["gradient_diagnostics"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
