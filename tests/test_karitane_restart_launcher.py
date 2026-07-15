from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from sheet.residual_run_config import OwtRunConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "current_DREEDLE_runstring.sh"
EXPECTED_ARTIFACT = (
    "SHEET_dreedle__KARITANE_LONG_260706_145723__n_99999_b_12_d_owt_"
    "w_20_k_500_A_4_L_144_H_32_D_2048_C_256_P_80_Q_256_"
    "r_depth_scaled_z_dof_implied_depth_S_12"
)


class KaritaneRestartLauncherTests(unittest.TestCase):
    def test_launcher_is_valid_bash(self):
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_legacy_resolver_reconstructs_exact_checkpoint_identity(self):
        config = OwtRunConfig(
            model_type="sheet",
            run_mode="resume",
            host_label="dreedle",
            run_name="KARITANE_LONG_260706_145723",
            dataset="openwebtext",
            max_iters=99999,
            batch_size=12,
            gradient_accumulation_steps=4,
            block_size=256,
            n_layer=144,
            n_head=32,
            n_embd=2048,
            depth_order=80,
            base_row_order=256,
            learning_rate=6.0e-4,
            min_lr=6.0e-5,
            warmup_iters=20,
            checkpoint_interval=500,
            residual_init_policy="depth_scaled",
            residual_init_depth_source="dof_implied_depth",
            checkpoint_segment_size=12,
            dtype="float16",
            eval_interval=100,
            eval_iters=10,
            log_interval=10,
        )
        self.assertEqual(config.artifact_name, EXPECTED_ARTIFACT)
        self.assertEqual(
            str(config.paths()["checkpoint_path"]),
            f"checkpoints/{EXPECTED_ARTIFACT}/ckpt.pt",
        )
        self.assertEqual(config.nonfinite_update_policy, "skip")
        self.assertEqual(config.max_nonfinite_update_skips, 10)


if __name__ == "__main__":
    unittest.main()
