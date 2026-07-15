from __future__ import annotations

import unittest

import torch

from sheet.trainer_state import TrainerState
from sheet.trainer_step import TrainerStepMixin
from sheet.training_config import TrainingConfig


class _FakeDistributed:
    rank = 0

    def all_true(self, value):
        return bool(value)

    def all_gather_object(self, value):
        return [value]


class _FakeOptimizer:
    def __init__(self, model):
        self.model = model
        self.zero_grad_calls = 0

    def zero_grad(self, *, set_to_none):
        self.zero_grad_calls += 1
        for parameter in self.model.parameters():
            parameter.grad = None


class _FakeScaler:
    def __init__(self):
        self.update_calls = 0

    def update(self):
        self.update_calls += 1


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))


class _NonfiniteHarness(TrainerStepMixin):
    def __init__(self, *, policy="skip", max_skips=2):
        self.config = TrainingConfig(
            nonfinite_update_policy=policy,
            max_nonfinite_update_skips=max_skips,
        )
        self.state = TrainerState()
        self.distributed = _FakeDistributed()
        self.raw_model = _TinyModel()
        self.optimizer = _FakeOptimizer(self.raw_model)
        self.scaler = _FakeScaler()
        self.events = []

    def _record(self, name, **payload):
        self.events.append((name, payload))


class KaritaneNonfiniteUpdateRecoveryTests(unittest.TestCase):
    def test_default_policy_skips_at_most_ten_nonfinite_updates(self):
        config = TrainingConfig()
        self.assertEqual(config.nonfinite_update_policy, "skip")
        self.assertEqual(config.max_nonfinite_update_skips, 10)

    def test_old_checkpoint_config_without_new_fields_gets_safe_defaults(self):
        legacy_values = TrainingConfig().__dict__.copy()
        legacy_values.pop("nonfinite_update_policy")
        legacy_values.pop("max_nonfinite_update_skips")
        restored = TrainingConfig(**legacy_values)
        self.assertEqual(restored.nonfinite_update_policy, "skip")
        self.assertEqual(restored.max_nonfinite_update_skips, 10)

    def test_skip_discards_attempt_without_completing_update_and_reports_named_bad_gradient(self):
        harness = _NonfiniteHarness()
        harness.raw_model.weight.grad = torch.tensor([float("nan"), 2.0])

        metrics = harness._handle_nonfinite_update(
            reason="gradient",
            learning_rate=1.0e-4,
            training_loss=4.5,
            gradient_norm=None,
            micro_step=None,
            microbatch_starts=[(11, 22)],
            scaler_unscaled=True,
        )

        self.assertEqual(harness.state.completed_updates, 0)
        self.assertEqual(harness.state.skipped_nonfinite_updates, 1)
        self.assertEqual(harness.state.failed_update_attempts, 1)
        self.assertEqual(metrics["skipped_update"], 1.0)
        self.assertIsNone(harness.raw_model.weight.grad)
        self.assertEqual(harness.scaler.update_calls, 1)
        report = metrics["nonfinite_diagnostics"]["rank_reports"][0]
        self.assertEqual(report["gradient_reports"][0]["parameter_name"], "weight")

    def test_pre_unscale_loss_skip_does_not_advance_grad_scaler(self):
        harness = _NonfiniteHarness()

        metrics = harness._handle_nonfinite_update(
            reason="loss",
            learning_rate=1.0e-4,
            training_loss=float("nan"),
            gradient_norm=None,
            micro_step=0,
            microbatch_starts=[(3, 4)],
            scaler_unscaled=False,
        )

        self.assertEqual(metrics["skipped_update"], 1.0)
        self.assertEqual(harness.scaler.update_calls, 0)

    def test_post_unscale_skip_advances_grad_scaler_before_next_attempt(self):
        harness = _NonfiniteHarness()
        harness.raw_model.weight.grad = torch.tensor([float("nan"), 1.0])
        harness._handle_nonfinite_update(
            reason="gradient",
            learning_rate=1.0e-4,
            training_loss=4.5,
            gradient_norm=None,
            micro_step=None,
            microbatch_starts=[],
            scaler_unscaled=True,
        )
        harness.raw_model.weight.grad = torch.tensor([float("inf"), 1.0])
        harness._handle_nonfinite_update(
            reason="gradient",
            learning_rate=1.0e-4,
            training_loss=4.4,
            gradient_norm=None,
            micro_step=None,
            microbatch_starts=[],
            scaler_unscaled=True,
        )

        self.assertEqual(harness.scaler.update_calls, 2)
        self.assertEqual(harness.state.skipped_nonfinite_updates, 2)

    def test_raise_policy_cleans_post_unscale_state_before_raising(self):
        harness = _NonfiniteHarness(policy="raise")
        harness.raw_model.weight.grad = torch.tensor([float("inf"), 1.0])

        with self.assertRaises(FloatingPointError):
            harness._handle_nonfinite_update(
                reason="gradient",
                learning_rate=1.0e-4,
                training_loss=4.5,
                gradient_norm=None,
                micro_step=None,
                microbatch_starts=[],
                scaler_unscaled=True,
            )

        self.assertEqual(harness.scaler.update_calls, 1)
        self.assertIsNone(harness.raw_model.weight.grad)
        self.assertEqual(harness.state.failed_update_attempts, 1)

    def test_skip_limit_exhaustion_preserves_completed_and_skip_counts(self):
        harness = _NonfiniteHarness(max_skips=0)
        harness.raw_model.weight.grad = torch.tensor([float("nan"), 1.0])

        with self.assertRaises(FloatingPointError):
            harness._handle_nonfinite_update(
                reason="gradient",
                learning_rate=1.0e-4,
                training_loss=4.5,
                gradient_norm=None,
                micro_step=None,
                microbatch_starts=[],
                scaler_unscaled=True,
            )

        self.assertEqual(harness.state.completed_updates, 0)
        self.assertEqual(harness.state.skipped_nonfinite_updates, 0)
        self.assertEqual(harness.state.failed_update_attempts, 1)

    def test_recovery_counters_round_trip_through_trainer_state(self):
        original = TrainerState(
            completed_updates=7,
            skipped_nonfinite_updates=2,
            failed_update_attempts=3,
        )
        restored = TrainerState(**original.__dict__)
        self.assertEqual(restored, original)


if __name__ == "__main__":
    unittest.main()
