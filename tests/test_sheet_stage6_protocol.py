# vvv THOG
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from sheet.stage6_protocol import (
    PilotBudget,
    adamw_fp32_minimum_training_bytes,
    dense_parameter_count,
    principal_dense_feasibility,
    protocol_manifest,
    verify_protocol_manifest,
)


class Stage6ProtocolTests(unittest.TestCase):
    def make_manifest(self, *, vocab_size: int = 50304):
        return protocol_manifest(
            budget=PilotBudget(),
            vocab_size=vocab_size,
            device="cpu",
            dtype="float32",
            output_root=Path("/tmp/stage6-fixture"),
            dataset={
                "path": "/fixture",
                "vocab_size": vocab_size,
                "train_tokens": 10000,
                "validation_tokens": 2000,
            },
            device_total_bytes=None,
        )

    def test_s6_01_principal_dense_memory_is_not_credible_on_16_gib(self) -> None:
        parameter_count = dense_parameter_count(
            n_layer=144,
            n_embd=768,
            block_size=256,
            vocab_size=50304,
        )
        self.assertEqual(parameter_count, 1_059_485_184)
        self.assertEqual(
            adamw_fp32_minimum_training_bytes(parameter_count),
            16_951_762_944,
        )
        report = principal_dense_feasibility(
            vocab_size=50304,
            device_total_bytes=16 * 1024 ** 3,
        )
        self.assertFalse(report["feasible_under_current_fp32_adamw_path"])
        self.assertGreater(report["minimum_training_state_gib"], 15.7)

    def test_s6_02_budget_is_locked_to_equal_updates_and_tokens(self) -> None:
        budget = PilotBudget()
        self.assertEqual(budget.n_layer, 72)
        self.assertEqual(budget.depth_order, 16)
        self.assertEqual(budget.tokens_per_update, 4096)
        self.assertEqual(budget.max_updates, 250)
        self.assertEqual(budget.consumed_tokens, 1_024_000)

    def test_s6_03_dense_and_sheet_controls_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = protocol_manifest(
                budget=PilotBudget(),
                vocab_size=50304,
                device="cpu",
                dtype="float32",
                output_root=Path(directory),
                dataset={
                    "path": "/fixture",
                    "vocab_size": 50304,
                    "train_tokens": 10000,
                    "validation_tokens": 2000,
                },
                device_total_bytes=None,
            )
        signatures = [
            row["logical_control_signature"]
            for row in manifest["runs"]
        ]
        self.assertTrue(all(signature == signatures[0] for signature in signatures))
        self.assertEqual(
            [row["base_row_order"] for row in manifest["runs"]],
            [1, 64, 128, 256],
        )
        self.assertEqual(
            [row["row_order_4d"] for row in manifest["runs"]],
            [None, 256, 512, 1024],
        )
        self.assertEqual(len({row["run_id"] for row in manifest["runs"]}), 4)
        self.assertEqual(len({row["out_dir"] for row in manifest["runs"]}), 4)

    def test_s6_04_architecture_only_fields_are_excluded_from_control_signature(self) -> None:
        manifest = self.make_manifest(vocab_size=512)
        dense_config = manifest["runs"][0]["training_config"]
        q128_config = manifest["runs"][2]["training_config"]
        self.assertNotEqual(dense_config["model_type"], q128_config["model_type"])
        self.assertNotEqual(dense_config["base_row_order"], q128_config["base_row_order"])
        self.assertEqual(
            manifest["runs"][0]["logical_control_signature"],
            manifest["runs"][2]["logical_control_signature"],
        )

    def test_s6_05_protocol_digest_detects_post_lock_changes(self) -> None:
        manifest = self.make_manifest()
        verify_protocol_manifest(manifest)
        changed = copy.deepcopy(manifest)
        changed["budget"]["max_updates"] += 1
        with self.assertRaisesRegex(ValueError, "protocol digest mismatch"):
            verify_protocol_manifest(changed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
