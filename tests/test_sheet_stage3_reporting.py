# vvv THOG
from __future__ import annotations

import math
import unittest

from sheet.trainer import SharedTrainer
from tests.stage3_test_support import stage3_config, token_splits


class Stage3ReportingTests(unittest.TestCase):
    def test_s3_13_parameter_reporting(self) -> None:
        train_tokens, validation_tokens = token_splits()
        sheet = SharedTrainer(
            stage3_config("thog2_sheet"),
            train_tokens,
            validation_tokens,
        )
        report = sheet.parameter_report
        explicit = sum(
            parameter.numel()
            for parameter in sheet.model.parameters()
        )
        self.assertEqual(report["persistent_parameters"], explicit)
        self.assertEqual(
            report["persistent_parameters"],
            report["sheet_coefficients"]
            + report["conventional_non_sheet_parameters"],
        )
        dense = SharedTrainer(
            stage3_config("dense"),
            train_tokens,
            validation_tokens,
        )
        self.assertEqual(dense.parameter_report["sheet_coefficients"], 0)
        self.assertEqual(
            dense.parameter_report["persistent_parameters"],
            sum(
                parameter.numel()
                for parameter in dense.model.parameters()
            ),
        )

    def test_s3_14_dense_trainer_regression(self) -> None:
        train_tokens, validation_tokens = token_splits()
        trainer = SharedTrainer(
            stage3_config("dense", max_updates=2),
            train_tokens,
            validation_tokens,
        )
        history = trainer.run()
        self.assertEqual(trainer.state.completed_updates, 2)
        self.assertTrue(
            all(
                math.isfinite(item["training_loss"])
                for item in history
                if "training_loss" in item
            )
        )
        self.assertFalse(
            any(
                name.startswith("trajectory.")
                for name, _ in trainer.model.named_parameters()
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
