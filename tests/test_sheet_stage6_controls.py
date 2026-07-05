# vvv THOG
from __future__ import annotations

import unittest

from sheet.stage6_analysis import equal_time_rows
from sheet.stage6_analysis import equal_update_rows
from sheet.stage6_analysis import resource_rows
from sheet.stage6_analysis import validate_controls


class Stage6ControlAnalysisTests(unittest.TestCase):
    def manifest(self):
        return {
            "protocol_sha256": "protocol",
            "budget": {"max_updates": 1},
            "consumed_tokens_per_run": 128,
            "runs": [
                {"model_type": "dense", "base_row_order": 1},
                {"model_type": "thog2_sheet", "base_row_order": 64},
                {"model_type": "thog2_sheet", "base_row_order": 128},
                {"model_type": "thog2_sheet", "base_row_order": 256},
            ],
        }

    def result(self, persistent: int, seconds: float, final_loss: float):
        return {
            "protocol_sha256": "protocol",
            "budget": {
                "completed_updates": 1,
                "consumed_tokens": 128,
            },
            "trace": {
                "training_sha256": "train",
                "validation_sha256": "validation",
            },
            "evaluations": [
                {
                    "completed_updates": 0,
                    "consumed_tokens": 0,
                    "training_seconds": 0.0,
                    "val": final_loss + 1.0,
                },
                {
                    "completed_updates": 1,
                    "consumed_tokens": 128,
                    "training_seconds": seconds,
                    "val": final_loss,
                },
            ],
            "parameter_report": {
                "persistent_parameters": persistent,
                "dense_equivalent_total_parameters": 1000,
            },
            "memory": {
                "samples": [{
                    "peak_allocated_bytes": persistent * 10,
                    "peak_reserved_bytes": persistent * 12,
                }]
            },
            "checkpoint": {"bytes": persistent * 4},
            "timing": {
                "training_seconds": seconds,
                "evaluation_seconds": 0.1,
                "checkpoint_seconds": 0.1,
                "wall_seconds": seconds + 0.2,
                "tokens_per_training_second": 128 / seconds,
            },
            "sheet_diagnostics": None,
        }

    def results(self):
        return {
            "dense": self.result(1000, 2.0, 3.0),
            "q64": self.result(300, 3.0, 3.5),
            "q128": self.result(400, 3.5, 3.3),
            "q256": self.result(600, 4.0, 3.1),
        }

    def test_s6_09_matched_controls_are_accepted(self) -> None:
        checks = validate_controls(self.manifest(), self.results())
        self.assertEqual(
            set(checks),
            {"dense", "q64", "q128", "q256"},
        )

    def test_s6_10_trace_mismatch_is_rejected(self) -> None:
        results = self.results()
        results["q128"]["trace"]["training_sha256"] = "different"
        with self.assertRaisesRegex(
            ValueError,
            "training batch trace differs",
        ):
            validate_controls(self.manifest(), results)

    def test_s6_11_comparison_tables_are_aligned(self) -> None:
        results = self.results()
        update_rows = equal_update_rows(results)
        time_rows = equal_time_rows(results)
        resources = resource_rows(self.manifest(), results)
        self.assertEqual(
            [row["completed_updates"] for row in update_rows],
            [0, 1],
        )
        self.assertEqual(len(time_rows), 2)
        self.assertEqual(len(resources), 4)
        self.assertAlmostEqual(
            resources[1]["persistent_parameter_reduction_fraction"],
            0.7,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
