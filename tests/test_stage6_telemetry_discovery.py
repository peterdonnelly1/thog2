# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sheet.wandb_telemetry import WandbTelemetry


class FakeRun:
    def __init__(self) -> None:
        self.logged = []
        self.defined = []
        self.summary = {}
        self.finished = False

    def define_metric(self, *arguments, **keywords) -> None:
        self.defined.append((arguments, keywords))

    def log(self, payload, **keywords) -> None:
        self.logged.append(dict(payload))

    def finish(self) -> None:
        self.finished = True


class FakeTelemetryModule:
    class Settings:
        def __init__(self, **keywords) -> None:
            self.keywords = keywords

    class errors:
        class CommError(Exception):
            pass

    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_arguments = None

    def init(self, **arguments):
        self.init_arguments = arguments
        return self.run

    def define_metric(self, *arguments, **keywords) -> None:
        self.run.define_metric(*arguments, **keywords)


class Stage6TelemetryDiscoveryTests(unittest.TestCase):
    def test_direct_backend_logs_training_evaluation_resource_and_sheet_metrics(self) -> None:
        module = FakeTelemetryModule()
        with tempfile.TemporaryDirectory() as directory:
            telemetry = WandbTelemetry(
                enabled=True,
                project="thog",
                entity=None,
                mode="offline",
                root=Path(directory),
                name="SHEET_scruffy__TEST",
                group="TEST",
                job_type="sheet",
                config={"artifact_prefix": "SHEET", "model_type": "sheet"},
            )
            telemetry.backend = "wandb"
            with mock.patch("sheet.wandb_telemetry.importlib.import_module", return_value=module):
                telemetry.start()
            telemetry.add_initial_summary({
                "persistent_parameters": 100,
                "dense_equivalent_total_parameters": 1000,
            })
            telemetry.log_event("optimizer_progress", {
                "completed_updates": 1,
                "consumed_tokens": 128,
                "cumulative_training_seconds": 2.0,
                "training_loss": 3.0,
                "learning_rate": 1.0e-3,
                "gradient_norm": 1.5,
            })
            telemetry.log_event("evaluation_completed", {
                "completed_updates": 1,
                "consumed_tokens": 128,
                "validation_loss": 2.5,
                "training_loss": 2.4,
            })
            telemetry.add_final_result({
                "budget": {"completed_updates": 1, "consumed_tokens": 128},
                "parameter_report": {
                    "persistent_parameters": 100,
                    "dense_equivalent_total_parameters": 1000,
                },
                "checkpoint": {"bytes": 4096},
                "timing": {
                    "training_seconds": 2.0,
                    "tokens_per_training_second": 64.0,
                },
                "memory": {"samples": [{"peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048}]},
                "evaluations": [{"val": 2.5}],
                "sheet_diagnostics": {
                    "coefficient_utilization": {
                        "attention_input_weight": {
                            "coefficient_rms": 0.1,
                            "high_depth_order_energy_fraction": 0.01,
                            "high_row_order_energy_fraction": 0.25,
                        }
                    },
                    "compact_state_violations": [],
                },
            })
            telemetry.finish()

        self.assertEqual(module.init_arguments["project"], "thog")
        self.assertEqual(module.init_arguments["name"], "SHEET_scruffy__TEST")
        self.assertTrue(any("train/step_loss" in row for row in module.run.logged))
        self.assertTrue(any("test/loss" in row for row in module.run.logged))
        self.assertTrue(any("resource/checkpoint_bytes" in row for row in module.run.logged))
        self.assertTrue(any("sheet/attention_input_weight/high_depth_order_energy_fraction" in row for row in module.run.logged))
        self.assertTrue(module.run.finished)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
