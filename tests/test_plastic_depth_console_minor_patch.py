# vvv THOG
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sheet import plastic_depth_console_minor_patch as console_patch
from sheet import stage6_trainer


class _DummyTrainer:
    # def __init__(self, *, last_decision_update: int, brake_active: bool) -> None:                                                                           # <<< THOG preserve the event-only brake fixture
    def __init__(
        self,
        *,
        last_decision_update: int,
        brake_active: bool,
        last_change_update: int = -1,
        update_brake: int = 5,
    ) -> None:
        self.config = SimpleNamespace(
            plastic__enabled=True,
            plastic__do_learn_layer_count=True,
            plastic__layer_count_update_brake=update_brake,
        )
        # self.state = SimpleNamespace(completed_updates=last_decision_update)                                                                                # <<< THOG preserve fixture state before direct brake-window derivation
        self.state = SimpleNamespace(
            completed_updates=last_decision_update,
            plastic_depth_last_count_change_update=last_change_update,
        )
        self.events = [
            SimpleNamespace(
                name="plastic_depth_count_decision",
                payload={"brake_active": brake_active},
            )
        ]
        self._lattice = SimpleNamespace(
            last_count_decision_update=torch.tensor(last_decision_update),
        )

    def _plastic_depth_lattice(self):
        return self._lattice


class PlasticDepthConsoleMinorPatchTests(unittest.TestCase):
    def setUp(self) -> None:
        console_patch._ALIGNMENT_BY_RUN_ID.clear()

    def test_non_probe_rows_drop_stale_probe_fields(self) -> None:
        trainer = _DummyTrainer(
            last_decision_update=20,
            brake_active=True,
            last_change_update=-1,
        )
        stale = {
            "completed_updates": 19,
            "plastic_probe_losses": (1.0, 1.1, 1.2),
            "plastic_probe_offsets": (-1, 0, 1),
            "plastic_probe_edge_offsets": (-1, 1),
            "plastic_score_z": (0.7, -0.4),
        }
        with patch.object(
            console_patch,
            "_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD",
            return_value=dict(stale),
        ):
            values = console_patch._prepare_console_progress_payload_with_probe_visibility(
                trainer,
                "optimizer_progress",
                {"completed_updates": 19},
            )
        self.assertNotIn("plastic_probe_losses", values)
        self.assertNotIn("plastic_probe_offsets", values)
        self.assertNotIn("plastic_score_z", values)
        self.assertNotIn("plastic_update_brake_active", values)

    def test_probe_row_preserves_fields_and_marks_brake(self) -> None:
        trainer = _DummyTrainer(
            last_decision_update=20,
            brake_active=True,
            last_change_update=18,
        )
        current = {
            "completed_updates": 20,
            "plastic_probe_losses": (1.0, 1.1, 1.2),
            "plastic_probe_offsets": (-1, 0, 1),
            "plastic_probe_edge_offsets": (-1, 1),
            "plastic_score_z": (0.7, -0.4),
        }
        with patch.object(
            console_patch,
            "_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD",
            return_value=dict(current),
        ):
            values = console_patch._prepare_console_progress_payload_with_probe_visibility(
                trainer,
                "optimizer_progress",
                {"completed_updates": 20},
            )
        self.assertEqual(values["plastic_probe_losses"], current["plastic_probe_losses"])
        self.assertTrue(values["plastic_update_brake_active"])

    # vvv THOG logged non-probe rows inside the committed brake window retain the brake suffix while stale probe fields remain hidden
    def test_non_probe_row_inside_brake_window_is_marked(self) -> None:
        trainer = _DummyTrainer(
            last_decision_update=20,
            brake_active=False,
            last_change_update=16,
            update_brake=5,
        )
        stale = {
            "completed_updates": 19,
            "plastic_probe_losses": (1.0, 1.1, 1.2),
            "plastic_score_z": (0.7, -0.4),
        }
        with patch.object(
            console_patch,
            "_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD",
            return_value=dict(stale),
        ):
            values = console_patch._prepare_console_progress_payload_with_probe_visibility(
                trainer,
                "optimizer_progress",
                {"completed_updates": 19},
            )
        self.assertNotIn("plastic_probe_losses", values)
        self.assertNotIn("plastic_score_z", values)
        self.assertTrue(values["plastic_update_brake_active"])

    def test_transition_row_does_not_claim_brake_was_enforced(self) -> None:
        trainer = _DummyTrainer(
            last_decision_update=20,
            brake_active=False,
            last_change_update=20,
            update_brake=5,
        )
        self.assertFalse(console_patch._row_has_active_update_brake(trainer, 20))
        self.assertTrue(console_patch._row_has_active_update_brake(trainer, 21))
        self.assertFalse(console_patch._row_has_active_update_brake(trainer, 25))
    # ^^^ THOG

    def test_tiny_score_uses_zero_sign_marker(self) -> None:
        self.assertEqual(console_patch._format_score_z_with_zero_sign(0.004).strip(), "0+")
        self.assertEqual(console_patch._format_score_z_with_zero_sign(-0.004).strip(), "0-")
        self.assertEqual(console_patch._format_score_z_with_zero_sign(0.006).strip(), "+0.01")

    def test_validation_plastic_suffix_matches_training_columns(self) -> None:
        common = {
            "completed_updates": 250,
            "timestamp": "050826-2210",
            "cumulative_training_seconds": 1076,
            "mean_step_seconds": "  4.39",
            "tok/s": " 17136",
            "consumed_tokens": " 18,432,000",
            "training_loss": "  5.8004",
            "current_layer_count": 15,
            "plastic_probe_offsets": (-1, 0, 1),
            "plastic_probe_edge_offsets": (-1, 1),
            "plastic_probe_losses": (6.306, 6.306, 6.311),
            "plastic_score_z": (0.64, -1.47),
            "depth_sample_points": tuple(float(value) for value in range(1, 16)),
        }
        training_payload = {
            **common,
            "training_loss_delta": "  -0.120",
            "learning_rate": "9.000e-04",
            "gradient_norm": "  0.314",
        }
        validation_payload = {
            **common,
            "training_loss": "  5.9158",
            "validation_loss": "  5.8171",
        }
        training_line = stage6_trainer.format_progress_line(
            "column_test",
            "optimizer_progress",
            training_payload,
        )
        training_line = console_patch._align_final_progress_line(
            "column_test",
            "optimizer_progress",
            training_line,
        )
        validation_line = stage6_trainer.format_progress_line(
            "column_test",
            "evaluation_completed",
            validation_payload,
        )
        validation_line = console_patch._align_final_progress_line(
            "column_test",
            "evaluation_completed",
            validation_line,
        )
        self.assertEqual(
            console_patch._field_start(training_line, "score_z"),
            console_patch._field_start(validation_line, "score_z"),
        )

    # vvv THOG the runtime print path performs one final alignment after public formatters have introduced tabs and ANSI styling
    def test_final_alignment_runs_after_late_formatter_tabs(self) -> None:
        training_line = (
            "T      20  260805:2300  00:01:20      80s  training loss= 6.0"
            "\tprobe_losses [L-1, L, L+1] = [ 5.9, 6.0, 6.1]"
            "  score_z [L-1, L+1] = [ +0.80, -0.20]"
            "\tsampled = [1.00, 2.00]"
        )
        validation_line = (
            "\033[33mV      20  260805:2300  00:01:20      80s  training loss= 6.0  validation loss= 6.1"
            "\tprobe_losses [L-1, L, L+1] = [ 5.9, 6.0, 6.1]"
            "  score_z [L-1, L+1] = [ +0.80, -0.20]"
            "\tsampled = [1.00, 2.00]\033[0m"
        )
        final_training = console_patch._align_final_progress_line(
            "late_formatter",
            "optimizer_progress",
            training_line,
        )
        final_validation = console_patch._align_final_progress_line(
            "late_formatter",
            "evaluation_completed",
            validation_line,
        )
        for label in ("probe_losses", "score_z", "sampled ="):
            self.assertEqual(
                console_patch._field_start(final_training, label),
                console_patch._field_start(final_validation, label),
            )
    # ^^^ THOG

    def test_brake_annotation_is_appended_in_pale_red(self) -> None:
        line = stage6_trainer.format_progress_line(
            "brake_test",
            "optimizer_progress",
            {
                "completed_updates": 10,
                "timestamp": "050826-2200",
                "cumulative_training_seconds": 40,
                "mean_step_seconds": "  4.00",
                "tok/s": " 18000",
                "consumed_tokens": "    737,280",
                "training_loss": "  6.0000",
                "training_loss_delta": "  -0.100",
                "learning_rate": "9.000e-04",
                "gradient_norm": "  0.300",
                "plastic_update_brake_active": True,
            },
        )
        self.assertIn("<<< update brake on", line)
        self.assertIn(console_patch._PALE_RED, line)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
