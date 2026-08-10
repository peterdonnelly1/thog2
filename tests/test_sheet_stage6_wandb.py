# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import constants

# from sheet.wandb_telemetry import WandbTelemetry
from sheet.wandb_telemetry import WandbTelemetry, _final_metrics


class FakeRun:
    def __init__(self) -> None:
        self.logged = []
        self.defined = []
        self.summary = {}
        self.finished = False

    def define_metric(self, *arguments, **keywords) -> None:
        self.defined.append((arguments, keywords))

    def log(self, payload) -> None:
        self.logged.append(dict(payload))

    def finish(self) -> None:
        self.finished = True


class FakeWandb:
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


class WandbTelemetryTests(unittest.TestCase):
    # vvv THOG default W&B setup defines only the two application loss series; raw GPU memory remains W&B-owned system telemetry
    def test_default_wandb_surface_is_minimal(self) -> None:
        module = FakeWandb()
        with mock.patch.object(constants, "DEBUG", 9), tempfile.TemporaryDirectory() as directory:
            telemetry = WandbTelemetry(
                enabled=True,
                project="thog",
                entity=None,
                mode="offline",
                root=Path(directory),
                name="SHEET_scruffy__MINIMAL",
                group="TEST",
                job_type="sheet",
                config={
                    "artifact_prefix": "SHEET",
                    "model_type": "sheet",
                    "plastic__enabled": True,
                },
            )
            with mock.patch(
                "sheet.wandb_telemetry.importlib.import_module",
                return_value=module,
            ):
                telemetry.start()

        defined_names = {
            arguments[0]
            for arguments, _keywords in module.run.defined
        }
        self.assertEqual(defined_names, {"train/loss", "val/val_loss"})
    # ^^^ THOG

    def test_s6_36_direct_telemetry_logs_events_and_resources(self) -> None:
        module = FakeWandb()
        with mock.patch.object(constants, "DEBUG", 10), tempfile.TemporaryDirectory() as directory:
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
            with mock.patch(
                "sheet.wandb_telemetry.importlib.import_module",
                return_value=module,
            ):
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
                "memory": {"samples": [{
                    "peak_allocated_bytes": 1024,
                    "peak_reserved_bytes": 2048,
                }]},
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
        self.assertEqual(module.init_arguments["mode"], "offline")
        self.assertTrue(any("train/loss" in row for row in module.run.logged))
        # vvv THOG evaluation telemetry names distinguish train-split evaluation from held-out validation
        self.assertTrue(any("val/train_loss" in row for row in module.run.logged))
        self.assertTrue(any("val/val_loss" in row for row in module.run.logged))
        # ^^^ THOG
        self.assertTrue(any("resource/checkpoint_bytes" in row for row in module.run.logged))
        self.assertTrue(
            any(
                "sheet/attention_input_weight/high_depth_order_energy_fraction" in row
                for row in module.run.logged
            )
        )
        self.assertTrue(module.run.finished)

    # vvv THOG unsupported coefficient order-axis diagnostics remain absent from final telemetry rather than crashing completed runs
    def test_s6_37_final_metrics_omit_unsupported_order_axis_diagnostics(self) -> None:
        metrics = _final_metrics({
            "budget": {"completed_updates": 8, "consumed_tokens": 16384},
            "parameter_report": {
                "persistent_parameters": 100,
                "dense_equivalent_total_parameters": 1000,
            },
            "checkpoint": {"bytes": 4096},
            "timing": {
                "training_seconds": 7.0,
                "tokens_per_training_second": 2260.0,
            },
            "sheet_diagnostics": {
                "coefficient_utilization": {
                    "depth_vector_example": {
                        "coefficient_rms": 0.1,
                        "high_depth_order_energy_fraction": None,
                        "high_row_order_energy_fraction": None,
                    },
                    "spectral_example": {
                        "coefficient_rms": 0.2,
                        "high_depth_order_energy_fraction": 0.01,
                        "high_row_order_energy_fraction": 0.25,
                    },
                },
                "compact_state_violations": [],
            },
        })

        self.assertNotIn(
            "sheet/depth_vector_example/high_depth_order_energy_fraction",
            metrics,
        )
        self.assertNotIn(
            "sheet/depth_vector_example/high_row_order_energy_fraction",
            metrics,
        )
        self.assertEqual(
            metrics["sheet/spectral_example/high_depth_order_energy_fraction"],
            0.01,
        )
        self.assertEqual(
            metrics["sheet/spectral_example/high_row_order_energy_fraction"],
            0.25,
        )
    # ^^^ THOG

    def test_s6_38_disabled_telemetry_is_a_noop(self) -> None:
        telemetry = WandbTelemetry(
            enabled=False,
            project="thog",
            entity=None,
            mode="online",
            root=Path("wandb"),
            name="DENSE2_scruffy__TEST",
            group="TEST",
            job_type="dense2",
            config={},
        )
        telemetry.start()
        telemetry.log_event("optimizer_progress", {})
        telemetry.finish()
        self.assertIsNone(telemetry.run)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
