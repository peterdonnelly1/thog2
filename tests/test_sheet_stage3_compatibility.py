# vvv THOG
from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import torch

from sheet.checkpoints import load_payload
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import stage3_config, token_splits


class Stage3CompatibilityTests(unittest.TestCase):
    def test_s3_09_incompatible_checkpoint_rejection(self) -> None:
        train_tokens, validation_tokens = token_splits()
        with tempfile.TemporaryDirectory() as directory:
            base = stage3_config("thog2_sheet", max_updates=1)
            trainer = SharedTrainer(base, train_tokens, validation_tokens)
            path = trainer.save_checkpoint(Path(directory) / "ckpt.pt")
            incompatible = (
                dataclasses.replace(base, model_type="dense"),
                dataclasses.replace(base, n_layer=3),
                dataclasses.replace(
                    base,
                    n_embd=24,
                    n_head=3,
                    base_row_order=8,
                ),
                dataclasses.replace(base, depth_order=1),
                dataclasses.replace(base, base_row_order=4),
            )
            for candidate in incompatible:
                with self.subTest(
                    candidate=candidate.compatibility_signature()
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "incompatible checkpoint",
                    ):
                        SharedTrainer.from_checkpoint(
                            path,
                            train_tokens,
                            validation_tokens,
                            expected_config=candidate,
                        )
            payload = load_payload(path)
            payload["trainer_config"]["basis_version"] = "invalid_basis"
            bad_path = Path(directory) / "bad_basis.pt"
            torch.save(payload, bad_path)
            with self.assertRaisesRegex(ValueError, "basis_version"):
                SharedTrainer.from_checkpoint(
                    bad_path,
                    train_tokens,
                    validation_tokens,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
