# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from sheet.checkpoints import load_payload, validate_compatibility
from sheet.stage4_trainer import Stage4Trainer
from tests.stage4_test_support import stage4_tokens, stage4_training_config


class Stage1CompactIdentityCheckpointSchemaTests(unittest.TestCase):
    def test_checkpoint_payload_records_compact_identity_and_parameter_report_uses_the_same_identity(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(128)
        trainer = Stage4Trainer(stage4_training_config(max_updates=1), train_tokens, validation_tokens)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "compact_identity.pt"
            trainer.save_checkpoint(checkpoint_path)
            payload = load_payload(checkpoint_path)
            expected_identity = trainer.config.compact_identity_metadata()
            self.assertEqual(payload["compact_identity"], expected_identity)
            self.assertEqual(payload["parameter_report"]["compact_identity"], expected_identity)
            resumed = Stage4Trainer.from_checkpoint(checkpoint_path, train_tokens, validation_tokens)
            self.assertEqual(resumed.config.compact_identity_metadata(), expected_identity)

    def test_checkpoint_compatibility_rejects_missing_or_mismatched_compact_identity_fields(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(128)
        trainer = Stage4Trainer(stage4_training_config(max_updates=1), train_tokens, validation_tokens)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "compact_identity.pt"
            trainer.save_checkpoint(checkpoint_path)
            payload = load_payload(checkpoint_path)
        validate_compatibility(payload, trainer.config)
        missing_identity = deepcopy(payload)
        missing_identity.pop("compact_identity")
        with self.assertRaisesRegex(ValueError, "compact_identity"):
            validate_compatibility(missing_identity, trainer.config)
        for key, value in (
            ("geometry_preset", "curve"),
            ("attention_geometry", "curve"),
            ("mlp_geometry", "mlp_block"),
            ("basis_family", "dct"),
            ("basis_version", "chebyshev_first_kind_qr_v999"),
            ("n_head", 999),
            ("n_embd", 999),
            ("depth_order", 999),
            ("base_row_order", 999),
        ):
            mutated = deepcopy(payload)
            mutated["compact_identity"][key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "compact_identity"):
                    validate_compatibility(mutated, trainer.config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
