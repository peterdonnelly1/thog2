# vvv THOG
from __future__ import annotations

import unittest

from sheet.plastic_depth import (
    PlasticDepthCandidateMeasurement,
    PlasticDepthSamplingLattice,
)
from sheet.plastic_depth_absolute_ruler_patch import _imputed_training_time
from sheet.plastic_depth_controller_stability_patch import (
    _TIMING_SKIP_ATTRIBUTE,
    choose_plastic_depth_count_with_robust_history,
)


def _score_report(current_count: int, candidate_count: int, paired_difference: float):
    return (
        {
            "active_layers": current_count,
            "feasible": True,
            "score": 1.0,
        },
        {
            "active_layers": candidate_count,
            "feasible": True,
            "score": 1.0 + paired_difference,
        },
    )


class PlasticDepthControllerStabilityTests(unittest.TestCase):
    def test_one_favourable_outlier_cannot_reverse_opposing_history(self) -> None:
        decision = choose_plastic_depth_count_with_robust_history(
            current_count=15,
            score_report=_score_report(15, 16, -0.004),
            histories={"15:+1": [0.010, 0.008, 0.006, 0.005]},
            noise_window=16,
            minimum_observations=4,
            noise_lambda=0.5,
            update_number=200,
            last_count_change_update=150,
            update_brake=5,
            max_step=1,
        )
        self.assertEqual(decision.selected_count, 15)
        evidence = decision.evidence[0]
        self.assertFalse(evidence.significant)
        self.assertLess(float(evidence.standardized_improvement), 0.0)

    def test_consistent_robust_improvement_changes_count(self) -> None:
        decision = choose_plastic_depth_count_with_robust_history(
            current_count=16,
            score_report=_score_report(16, 15, -0.005),
            histories={"16:-1": [-0.010, -0.008, -0.006]},
            noise_window=16,
            minimum_observations=4,
            noise_lambda=0.5,
            update_number=150,
            last_count_change_update=-1,
            update_brake=5,
            max_step=1,
        )
        self.assertEqual(decision.selected_count, 15)
        evidence = decision.evidence[0]
        self.assertTrue(evidence.significant)
        self.assertGreater(float(evidence.standardized_improvement), 0.5)

    def test_latest_probe_must_agree_with_favourable_history(self) -> None:
        decision = choose_plastic_depth_count_with_robust_history(
            current_count=16,
            score_report=_score_report(16, 15, 0.001),
            histories={"16:-1": [-0.010, -0.008, -0.006, -0.005]},
            noise_window=16,
            minimum_observations=4,
            noise_lambda=0.5,
            update_number=150,
            last_count_change_update=-1,
            update_brake=5,
            max_step=1,
        )
        self.assertEqual(decision.selected_count, 16)
        self.assertFalse(decision.evidence[0].significant)

    def test_score_z_is_hidden_until_minimum_observations(self) -> None:
        decision = choose_plastic_depth_count_with_robust_history(
            current_count=16,
            score_report=_score_report(16, 15, -0.005),
            histories={},
            noise_window=16,
            minimum_observations=4,
            noise_lambda=0.5,
            update_number=110,
            last_count_change_update=-1,
            update_brake=5,
            max_step=1,
        )
        self.assertIsNone(decision.evidence[0].standardized_improvement)
        self.assertFalse(decision.evidence[0].significant)

    def test_probe_update_does_not_contaminate_timing_ema(self) -> None:
        lattice = PlasticDepthSamplingLattice(
            16,
            initial_active_layers=8,
            initialisation="equidistant",
            seed=7,
            learn_layer_count=True,
        )
        setattr(lattice, _TIMING_SKIP_ATTRIBUTE, True)
        lattice.record_training_time(8, 9.0, update_reference=True)
        self.assertEqual(int(lattice.training_time_observations[8].item()), 0)
        self.assertTrue(bool(lattice.reference_training_time.isnan().item()))
        setattr(lattice, _TIMING_SKIP_ATTRIBUTE, False)
        lattice.record_training_time(8, 4.0, update_reference=True)
        self.assertEqual(int(lattice.training_time_observations[8].item()), 1)
        self.assertEqual(float(lattice.training_time_ema[8].item()), 4.0)
        self.assertEqual(float(lattice.reference_training_time.item()), 4.0)

    def test_imputation_uses_observed_local_slope(self) -> None:
        measurements = (
            PlasticDepthCandidateMeasurement(14, 1.0, training_time=3.93),
            PlasticDepthCandidateMeasurement(15, 1.0, training_time=None),
            PlasticDepthCandidateMeasurement(16, 1.0, training_time=4.38),
        )
        estimate = _imputed_training_time(measurements, measurements[1])
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(float(estimate), 4.155, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
