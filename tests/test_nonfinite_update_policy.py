# vvv THOG
from __future__ import annotations

import json
import unittest

import torch

from sheet.trainer_state import TrainerState
from sheet.trainer_step import TrainerStepMixin
from sheet.training_config import TrainingConfig


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # vvv THOG
        # Match the synthetic four-value diagnostic gradient installed by the tests.
        self.bad_weight = torch.nn.Parameter(torch.ones(4))
        # ^^^ THOG


class _FakeDistributed:
    rank = 0

    def all_gather_object(self, value):
        return [value]


class _FakeOptimizer:
    def __init__(self, model: _TinyModel) -> None:
        self.model = model
        self.zero_grad_calls = 0

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.zero_grad_calls += 1
        if set_to_none:
            for parameter in self.model.parameters():
                parameter.grad = None


# vvv THOG
# Minimal GradScaler stand-in for asserting that post-unscale skips reset scaler state.
class _FakeScaler:
    def __init__(self) -> None:
        self.update_calls = 0

    def update(self) -> None:
        self.update_calls += 1
# ^^^ THOG


class _NonfiniteHarness(TrainerStepMixin):
    def __init__(self, *, policy: str = "raise", max_skips: int = 10) -> None:
        self.config = TrainingConfig(
            nonfinite_update_policy=policy,
            max_nonfinite_update_skips=max_skips,
        )
        self.state = TrainerState()
        self.distributed = _FakeDistributed()
        self.raw_model = _TinyModel()
        self.optimizer = _FakeOptimizer(self.raw_model)
        # vvv THOG
        self.scaler = _FakeScaler()
        # ^^^ THOG
        self.events = []

    def _record(self, name, **payload):
        self.events.append((name, payload))


class NonfiniteUpdatePolicyTests(unittest.TestCase):
    def _install_bad_gradient(self, harness: _NonfiniteHarness) -> None:
        harness.raw_model.bad_weight.grad = torch.tensor([
            float("nan"),
            float("inf"),
            -float("inf"),
            2.0,
        ])

    def test_default_raise_policy_reports_the_named_nonfinite_gradient_parameter(self) -> None:
        harness = _NonfiniteHarness(policy="raise")
        self._install_bad_gradient(harness)

        with self.assertRaises(FloatingPointError) as context:
            harness._handle_nonfinite_update(
                reason="gradient",
                learning_rate=1.0e-4,
                training_loss=4.5,
                gradient_norm=None,
                micro_step=None,
                microbatch_starts=[(11, 22)],
            )

        message = str(context.exception)
        self.assertIn("bad_weight", message)
        self.assertIn("nan_count", message)
        self.assertEqual(harness.state.completed_updates, 0)
        self.assertEqual(harness.state.failed_update_attempts, 1)
        self.assertIsNone(harness.raw_model.bad_weight.grad)
        self.assertEqual(harness.scaler.update_calls, 0)

    def test_skip_policy_zeros_gradients_counts_the_skip_and_does_not_complete_an_update(self) -> None:
        harness = _NonfiniteHarness(policy="skip")
        self._install_bad_gradient(harness)

        metrics = harness._handle_nonfinite_update(
            reason="gradient",
            learning_rate=1.0e-4,
            training_loss=4.5,
            gradient_norm=None,
            micro_step=None,
            microbatch_starts=[(11, 22)],
        )

        self.assertEqual(metrics["skipped_update"], 1.0)
        self.assertEqual(harness.state.completed_updates, 0)
        self.assertEqual(harness.state.skipped_nonfinite_updates, 1)
        self.assertEqual(harness.state.failed_update_attempts, 1)
        self.assertIsNone(harness.raw_model.bad_weight.grad)
        self.assertEqual(harness.optimizer.zero_grad_calls, 1)
        self.assertEqual(harness.scaler.update_calls, 0)
        self.assertEqual(harness.events[0][0], "nonfinite_update_skipped")
        diagnostic_json = json.dumps(metrics["nonfinite_diagnostics"], sort_keys=True)
        self.assertIn("bad_weight", diagnostic_json)

    def test_skip_policy_after_unscale_updates_grad_scaler_before_the_next_attempt(self) -> None:
        harness = _NonfiniteHarness(policy="skip")
        self._install_bad_gradient(harness)

        metrics = harness._handle_nonfinite_update(
            reason="gradient",
            learning_rate=1.0e-4,
            training_loss=4.5,
            gradient_norm=None,
            micro_step=None,
            microbatch_starts=[(11, 22)],
            scaler_unscaled=True,
        )

        self.assertEqual(metrics["skipped_update"], 1.0)
        self.assertEqual(harness.scaler.update_calls, 1)
        self.assertEqual(harness.optimizer.zero_grad_calls, 1)
        self.assertIsNone(harness.raw_model.bad_weight.grad)

    def test_skip_policy_raises_after_the_configured_maximum_skips_is_reached(self) -> None:
        harness = _NonfiniteHarness(policy="skip", max_skips=1)
        self._install_bad_gradient(harness)

        harness._handle_nonfinite_update(
            reason="gradient",
            learning_rate=1.0e-4,
            training_loss=4.5,
            gradient_norm=None,
            micro_step=None,
            microbatch_starts=[(11, 22)],
        )
        self._install_bad_gradient(harness)

        with self.assertRaises(FloatingPointError) as context:
            harness._handle_nonfinite_update(
                reason="gradient",
                learning_rate=1.0e-4,
                training_loss=4.5,
                gradient_norm=None,
                micro_step=None,
                microbatch_starts=[(33, 44)],
            )

        self.assertIn("skip limit exceeded", str(context.exception))
        self.assertEqual(harness.state.skipped_nonfinite_updates, 1)
        self.assertEqual(harness.state.failed_update_attempts, 2)

    def test_trainer_state_loads_old_checkpoint_payload_without_skip_counters(self) -> None:
        state = TrainerState(
            completed_updates=7,
            best_validation_loss=4.0,
            latest_validation_loss=4.1,
            latest_training_loss=4.2,
        )

        self.assertEqual(state.completed_updates, 7)
        self.assertEqual(state.skipped_nonfinite_updates, 0)
        self.assertEqual(state.failed_update_attempts, 0)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
