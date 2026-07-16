# vvv THOG
from __future__ import annotations

import math
import unittest

from sheet.lr_schedule import restart_cosine_learning_rate, validate_restart_cosine_phase


class ForkLearningRateScheduleTests(unittest.TestCase):
    def phase(self, **changes):
        value = {
            "phase_start_update": 10,
            "phase_start_lr": 1.0e-4,
            "phase_peak_lr": 4.0e-4,
            "phase_rewarm_iters": 3,
            "phase_end_update": 20,
            "phase_min_lr": 5.0e-5,
        }
        value.update(changes)
        return value

    def test_nonzero_rewarm_starts_at_parent_lr_and_reaches_peak(self) -> None:
        phase = self.phase()
        self.assertEqual(restart_cosine_learning_rate(completed_updates=10, phase=phase), 1.0e-4)
        self.assertEqual(restart_cosine_learning_rate(completed_updates=12, phase=phase), 4.0e-4)

    def test_zero_rewarm_uses_peak_on_first_child_update(self) -> None:
        phase = self.phase(phase_rewarm_iters=0)
        self.assertEqual(restart_cosine_learning_rate(completed_updates=10, phase=phase), 4.0e-4)

    def test_one_step_rewarm_uses_peak_on_first_child_update(self) -> None:
        phase = self.phase(phase_rewarm_iters=1)
        self.assertEqual(restart_cosine_learning_rate(completed_updates=10, phase=phase), 4.0e-4)

    def test_final_child_update_uses_requested_minimum_lr(self) -> None:
        phase = self.phase()
        self.assertTrue(math.isclose(restart_cosine_learning_rate(completed_updates=19, phase=phase), 5.0e-5))
        self.assertEqual(restart_cosine_learning_rate(completed_updates=20, phase=phase), 5.0e-5)

    def test_phase_before_start_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "precedes"):
            restart_cosine_learning_rate(completed_updates=9, phase=self.phase())

    def test_restart_cosine_requires_at_least_two_child_updates(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            validate_restart_cosine_phase(self.phase(phase_end_update=11))

    def test_rewarm_cannot_consume_final_minimum_lr_update(self) -> None:
        with self.assertRaisesRegex(ValueError, "child_updates - 1"):
            validate_restart_cosine_phase(self.phase(phase_rewarm_iters=10))

    def test_peak_must_not_be_below_parent_boundary_lr(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least phase_start_lr"):
            validate_restart_cosine_phase(self.phase(phase_peak_lr=5.0e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
