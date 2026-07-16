# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sheet.checkpoint_resolver import resolve_checkpoint
from sheet.lr_schedule import restart_cosine_learning_rate
from sheet.run_config import OwtRunConfig


class EnhancedResumeLifecycleMinimalTests(unittest.TestCase):
    def test_restart_cosine_zero_rewarm_two_update_phase_uses_peak_then_minimum(self) -> None:
        values = [
            restart_cosine_learning_rate(
                completed_updates=step,
                phase_start_update=20,
                phase_start_lr=6.0e-5,
                phase_peak_lr=1.0e-4,
                phase_rewarm_iters=0,
                phase_end_update=22,
                phase_min_lr=1.0e-5,
            )
            for step in (20, 21)
        ]
        self.assertEqual(values, [1.0e-4, 1.0e-5])

    def test_restart_cosine_rewarm_phase_starts_at_parent_lr_reaches_peak_and_finishes_at_minimum(self) -> None:
        values = [
            restart_cosine_learning_rate(
                completed_updates=step,
                phase_start_update=20,
                phase_start_lr=6.0e-5,
                phase_peak_lr=1.0e-4,
                phase_rewarm_iters=1,
                phase_end_update=24,
                phase_min_lr=1.0e-5,
            )
            for step in (20, 21, 22, 23)
        ]
        self.assertAlmostEqual(values[0], 6.0e-5)
        self.assertAlmostEqual(values[1], 1.0e-4)
        self.assertAlmostEqual(values[-1], 1.0e-5)
        self.assertGreater(values[2], values[-1])
        self.assertLess(values[2], values[1])

    def test_timestamp_selector_ignores_root_timestamp_embedded_in_fork_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "260101-0000_PARENT"
            fork = root / "260102-0000_CHILD__FORK_1_FROM_260101-0000"
            original.mkdir()
            fork.mkdir()
            (original / "ckpt.pt").write_bytes(b"original")
            (fork / "ckpt.pt").write_bytes(b"fork")
            self.assertEqual(resolve_checkpoint("260101-0000", root).artifact_name, original.name)
            self.assertEqual(resolve_checkpoint("260102-0000", root).artifact_name, fork.name)

    def test_artifact_descriptor_excludes_mutable_runtime_cadence_fields(self) -> None:
        base = OwtRunConfig(model_type="dense", max_iters=100, warmup_iters=10, checkpoint_interval=20, checkpoint_segment_size=7)
        changed = OwtRunConfig(model_type="dense", max_iters=200, warmup_iters=30, checkpoint_interval=40, checkpoint_segment_size=11)
        self.assertEqual(base.artifact_descriptor, changed.artifact_descriptor)
        for forbidden in ("n_", "w_", "k_", "S_"):
            self.assertNotIn(forbidden, base.artifact_descriptor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
