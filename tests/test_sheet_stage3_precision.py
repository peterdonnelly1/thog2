# vvv THOG
from __future__ import annotations

import math
import unittest

from sheet.trainer import SharedTrainer
from tests.stage3_test_support import stage3_config, token_splits


class Stage3PrecisionTests(unittest.TestCase):
    def test_s3_12_mixed_precision_reference(self) -> None:
        train_tokens, validation_tokens = token_splits()
        for dtype in ("float32", "bfloat16"):
            with self.subTest(dtype=dtype):
                trainer = SharedTrainer(
                    stage3_config(
                        "thog2_sheet",
                        max_updates=1,
                        dtype=dtype,
                    ),
                    train_tokens,
                    validation_tokens,
                )
                metrics = trainer.train_one_update()
                self.assertTrue(
                    math.isfinite(metrics["training_loss"])
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
