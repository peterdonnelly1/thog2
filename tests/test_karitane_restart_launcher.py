from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "current_DREEDLE_runstring.sh"
RUNNER = REPOSITORY_ROOT / "run_karitane_long_resume.py"


class KaritaneRestartLauncherTests(unittest.TestCase):
    def test_launcher_has_valid_shell_syntax_and_calls_dedicated_resume_module(self):
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stderr,
        )
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("-m run_karitane_long_resume", launcher)
        self.assertIn('THOG2_INSTRUMENTATION="wandb"', launcher)
        self.assertIn("THOG2_FAST_DISCARD", launcher)
        self.assertIn(
            'export CUDA_VISIBLE_DEVICES="0,1"',
            launcher,
            msg=(
                "the original Dreedle checkpoint saved CUDA RNG states for both "
                "visible TITAN RTX devices"
            ),
        )

    def test_runner_locks_original_karitane_geometry_schedule_and_checkpoint_path(self):
        runner = RUNNER.read_text(encoding="utf-8")
        expected_fragments = (
            "KARITANE_LONG_260706_145723",
            "n_99999_b_12",
            "L_144_H_32_D_2048_C_256_P_80_Q_256",
            "block_size=256",
            "n_layer=144",
            "n_head=32",
            "n_embd=2048",
            "depth_order=80",
            "base_row_order=256",
            "batch_size=12",
            "gradient_accumulation_steps=4",
            "learning_rate=6.0e-4",
            "min_learning_rate=6.0e-5",
            "warmup_updates=20",
            "decay_updates=99999",
            'nonfinite_update_policy="skip"',
            "max_nonfinite_update_skips=10",
            "default=10",
        )
        for fragment in expected_fragments:
            self.assertIn(fragment, runner)


if __name__ == "__main__":
    unittest.main()
