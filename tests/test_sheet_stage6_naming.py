# vvv THOG
from __future__ import annotations

import unittest
from pathlib import Path

from run_thog2_stage6_pilot import apply_artifact_naming
from sheet.stage6_protocol import PilotBudget
from sheet.stage6_protocol import protocol_manifest


class Stage6NamingTests(unittest.TestCase):
    def make_manifest(self):
        return protocol_manifest(
            budget=PilotBudget(),
            vocab_size=50304,
            device="cpu",
            dtype="float32",
            output_root=Path("/tmp/stage6-naming-fixture"),
            dataset={
                "path": "/fixture",
                "vocab_size": 50304,
                "train_tokens": 10000,
                "validation_tokens": 2000,
            },
            device_total_bytes=None,
        )

    def named_manifest(self):
        manifest = self.make_manifest()
        apply_artifact_naming(
            manifest,
            host_label="scruffy",
            run_name="stage6",
            suffix="pilot_test",
            source_commit="a" * 40,
        )
        return manifest

    def test_s6_23_prefixes_and_optimizer_group_match_thog(self) -> None:
        manifest = self.named_manifest()
        names = [row["artifact_name"] for row in manifest["runs"]]
        self.assertTrue(names[0].startswith("DENSE2_scruffy_STAGE6__OWT__"))
        self.assertTrue(names[1].startswith("SHEET__scruffy_STAGE6__OWT__"))
        self.assertEqual(
            [row["artifact_prefix"] for row in manifest["runs"]],
            ["DENSE2", "SHEET", "SHEET", "SHEET"],
        )
        for name in names:
            self.assertIn(
                "__LR0.0006_MLR6e-05_WD0.1_B10.9_B20.95_GC1.0__",
                name,
            )
            self.assertLessEqual(len(name), 240)

    def test_s6_24_sheet_names_carry_orders(self) -> None:
        manifest = self.named_manifest()
        self.assertIn("_P16_Q64_Q4D256__", manifest["runs"][1]["artifact_name"])
        self.assertIn("_P16_Q128_Q4D512__", manifest["runs"][2]["artifact_name"])
        self.assertIn("_P16_Q256_Q4D1024__", manifest["runs"][3]["artifact_name"])
        self.assertEqual(
            {row["wandb"]["group"] for row in manifest["runs"]},
            {"STAGE6__PILOT_TEST"},
        )

    def test_s6_25_paths_are_derived_from_full_name(self) -> None:
        manifest = self.named_manifest()
        for row in manifest["runs"]:
            artifact_name = row["artifact_name"]
            self.assertEqual(Path(row["out_dir"]).name, artifact_name)
            self.assertEqual(Path(row["checkpoint_path"]).name, "ckpt.pt")
            self.assertEqual(Path(row["worker_result_path"]).name, "result.json")
            self.assertEqual(Path(row["result_path"]).stem, artifact_name)
            self.assertEqual(Path(row["log_path"]).stem, artifact_name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
