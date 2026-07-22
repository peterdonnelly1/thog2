from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from sheet.stage6_trainer import _max_wall_seconds
from sheet.training_config import TrainingConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GRID_WRAPPER = REPOSITORY_ROOT / "scruffy_model_geometry_max_wall_minutes_grid.sh"


class MaxWallMinutesTests(unittest.TestCase):
    def test_training_config_accepts_zero_disabled_and_positive_budget(self):
        self.assertEqual(TrainingConfig(max_wall_minutes=0).max_wall_minutes, 0)
        self.assertEqual(TrainingConfig(max_wall_minutes=6).max_wall_minutes, 6)
        with self.assertRaisesRegex(ValueError, "max_wall_minutes"):
            TrainingConfig(max_wall_minutes=-1)

    def test_stage6_converts_minutes_to_optional_seconds(self):
        self.assertIsNone(_max_wall_seconds(0))
        self.assertEqual(_max_wall_seconds(6), 360.0)

    def test_run_config_surfaces_wall_budget_in_resolved_json_and_artifact_name(self):
        command = [
            sys.executable,
            "-m",
            "run_thog2_owt",
            "--model-type",
            "sheet",
            "--geometry-preset",
            "full_block",
            "--max-iters",
            "100",
            "--max-wall-minutes",
            "7",
            "--eval-interval",
            "10",
            "--eval-iters",
            "1",
            "--log-interval",
            "1",
            "--batch-size",
            "2",
            "--gradient-accumulation-steps",
            "1",
            "--block-size",
            "8",
            "--n-layer",
            "4",
            "--n-head",
            "2",
            "--n-embd",
            "8",
            "--o-depth",
            "2",
            "--o-attn-d-model",
            "4",
            "--o-attn-qkv-per-channel",
            "2",
            "--o-attn-out-per-channel",
            "2",
            "--o-mlp-d-model",
            "4",
            "--o-mlp-hidden",
            "8",
            "--checkpoint-segment-size",
            "1",
            "--warmup-iters",
            "1",
            "--print-resolved-json",
        ]
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        resolved = json.loads(completed.stdout)
        self.assertEqual(resolved["canonical_config"]["max_wall_minutes"], 7)
        self.assertEqual(resolved["canonical_config"]["tokens_per_iter"], 16)
        self.assertIn("_M_7_", resolved["artifact_name"])

    def test_geometry_wall_time_grid_wrapper_has_valid_shell_syntax_and_passes_budget(self):
        completed = subprocess.run(
            ["bash", "-n", str(GRID_WRAPPER)],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        wrapper = GRID_WRAPPER.read_text(encoding="utf-8")
        self.assertIn("legacy_sheet_col depth head_aware_block mlp_block full_block", wrapper)
        self.assertIn("--max-wall-minutes", wrapper)
        self.assertIn("THOG2_GEOM_GRID_MAX_WALL_MINUTES", wrapper)


if __name__ == "__main__":
    unittest.main()
