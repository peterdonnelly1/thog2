# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sheet.checkpoints import load_payload
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import stage3_config, token_splits


class Stage3SchemaTests(unittest.TestCase):
    def test_s3_05_checkpoint_schema(self) -> None:
        train_tokens, validation_tokens = token_splits()
        with tempfile.TemporaryDirectory() as directory:
            trainer = SharedTrainer(
                stage3_config("thog2_sheet", max_updates=1),
                train_tokens,
                validation_tokens,
            )
            trainer.run()
            payload = load_payload(
                trainer.save_checkpoint(Path(directory) / "ckpt.pt")
            )
            required = {
                "schema_version", "model_type", "model_args",
                "compatibility_signature", "basis_version",
                "row_order_scaling_rule", "model", "optimizer",
                "optimizer_group_parameter_names", "trainer_state",
                "completed_updates", "trainer_config", "batch_source",
                "rng_state", "parameter_report",
            }
            self.assertTrue(required.issubset(payload))
            self.assertFalse(any(
                key.startswith("trajectory.bases.")
                for key in payload["model"]
            ))
            self.assertFalse(any(
                "transformer.h." in key for key in payload["model"]
            ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
