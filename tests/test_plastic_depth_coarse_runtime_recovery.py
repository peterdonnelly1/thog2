# vvv THOG
from __future__ import annotations

import weakref
from pathlib import Path
from types import SimpleNamespace

import constants

from sheet import plastic_depth_coarse_runtime_recovery_patch as recovery
from sheet import plastic_depth_fresh_state
from sheet import plastic_depth_lifecycle
from sheet.plastic_depth_coarse import (
    PlasticCoarseTrialResult,
    ScoredPlasticCoarseTrial,
    render_plastic_coarse_report,
)
from sheet.wandb_telemetry import WandbTelemetry


class _CleanupTarget:
    def __init__(self) -> None:
        self.calls = []

    def end_optimizer_update(self) -> None:
        self.calls.append("end_optimizer_update")

    def clear_plastic_depth_update_layer_count(self) -> None:
        self.calls.append("clear_plastic_depth_update_layer_count")

    def clear_plastic_depth_basis_cache(self) -> None:
        self.calls.append("clear_plastic_depth_basis_cache")

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.calls.append(("zero_grad", set_to_none))


class _CleanupTrainer:
    def __init__(self) -> None:
        self.device = "cpu"
        self.optimizer = _CleanupTarget()
        self.raw_model = _CleanupTarget()
        self.raw_model.trajectory = _CleanupTarget()
        self.model = object()
        self.scaler = object()
        self.batch_source = object()
        self.memory_telemetry = object()
        self.events = []
        self.parameter_report = {}
        self.distributed = object()
        self.clear_inline_calls = 0
        self.close_calls = 0

    def _clear_plastic_depth_inline_update(self) -> None:
        self.clear_inline_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_destroy_fresh_state_breaks_all_trainer_ownership_edges() -> None:
    trainer = _CleanupTrainer()
    optimizer = trainer.optimizer
    raw_model = trainer.raw_model
    trajectory = raw_model.trajectory
    state = SimpleNamespace(
        trainer=trainer,
        phase="coarse",
        active_layer_count=8,
    )

    plastic_depth_fresh_state.destroy_fresh_training_state(state)

    assert state.trainer is None
    assert trainer.close_calls == 1
    assert trainer.clear_inline_calls == 1
    assert ("zero_grad", True) in optimizer.calls
    assert "end_optimizer_update" in raw_model.calls
    assert "clear_plastic_depth_update_layer_count" in raw_model.calls
    assert ("zero_grad", True) in raw_model.calls
    assert "clear_plastic_depth_basis_cache" in trajectory.calls
    for attribute in (
        "model",
        "raw_model",
        "optimizer",
        "scaler",
        "batch_source",
        "memory_telemetry",
        "events",
        "parameter_report",
        "distributed",
    ):
        assert getattr(trainer, attribute) is None


def test_destroy_releases_trajectory_before_final_cuda_snapshot(monkeypatch) -> None:
    trainer = _CleanupTrainer()
    trajectory_reference = weakref.ref(trainer.raw_model.trajectory)
    state = SimpleNamespace(
        trainer=trainer,
        phase="coarse",
        active_layer_count=8,
    )
    trajectory_liveness_at_snapshots = []

    def fake_cuda_snapshot(device):
        trajectory_liveness_at_snapshots.append(trajectory_reference() is not None)
        return (0, 0)

    monkeypatch.setattr(recovery, "_cuda_snapshot", fake_cuda_snapshot)
    monkeypatch.setattr(recovery, "_cuda_key", lambda device: "cpu")
    monkeypatch.setattr(recovery, "_cuda_device", lambda device: None)

    plastic_depth_fresh_state.destroy_fresh_training_state(state)

    assert trajectory_liveness_at_snapshots == [True, False]


def test_lifecycle_defaults_use_hard_builder_and_destroyer() -> None:
    defaults = plastic_depth_lifecycle.run_plastic_coarse_fine_lifecycle.__kwdefaults__
    assert defaults is not None
    assert defaults["fresh_state_builder"] is plastic_depth_fresh_state.build_fresh_training_state
    assert defaults["state_destroyer"] is plastic_depth_fresh_state.destroy_fresh_training_state


def test_coarse_report_prints_failed_trial_exception() -> None:
    success = PlasticCoarseTrialResult(
        trial_index=1,
        layers=4,
        status="success",
        validation_losses=(3.0, 3.1),
        training_losses=(4.0, 3.5),
        training_elapsed_seconds=2.0,
        training_steps=2,
        tokens_per_update=100,
    )
    failed = PlasticCoarseTrialResult(
        trial_index=2,
        layers=8,
        status="failed",
        training_losses=(4.1,),
        training_elapsed_seconds=1.0,
        training_steps=1,
        tokens_per_update=100,
        error_class="OutOfMemoryError",
        error_message="CUDA out of memory while allocating the second update",
    )
    winner = ScoredPlasticCoarseTrial(
        result=success,
        objective="lowest_loss",
        objective_heading="loss_score",
        score=success.mean_validation_loss,
        selectable=True,
        within_budget=None,
        reference_training_elapsed_seconds=None,
    )
    failed_row = ScoredPlasticCoarseTrial(
        result=failed,
        objective="lowest_loss",
        objective_heading="loss_score",
        score=None,
        selectable=False,
        within_budget=None,
        reference_training_elapsed_seconds=None,
    )

    report = render_plastic_coarse_report(
        (winner, failed_row),
        winner,
        training_steps=2,
        evaluation_steps_count=2,
        ansi=False,
    )

    assert "PLASTIC COARSE FAILURES" not in report
    assert "failed - because OutOfMemoryError: CUDA out of memory" in report


class _Run:
    def __init__(self) -> None:
        self.defined = []
        self.logged = []
        self.summary = {}

    def define_metric(self, name, **kwargs) -> None:
        self.defined.append((name, dict(kwargs)))

    def log(self, payload, **kwargs) -> None:
        self.logged.append((dict(payload), dict(kwargs)))


class _Writer:
    def __init__(self) -> None:
        self.scalars = []
        self.text = []

    def add_scalar(self, name, value, step) -> None:
        self.scalars.append((name, value, step))

    def add_text(self, name, value, step) -> None:
        self.text.append((name, value, step))


def _telemetry(tmp_path: Path) -> WandbTelemetry:
    telemetry = WandbTelemetry(
        enabled=True,
        project="test",
        entity=None,
        mode="offline",
        root=tmp_path,
        name="plastic",
        group="test",
        job_type="sheet",
        config={"device": "cpu", "plastic__enabled": True},
    )
    telemetry.backend = "both"
    telemetry.run = _Run()
    telemetry.writer = _Writer()
    return telemetry


def test_fine_wandb_logging_does_not_rewind_after_coarse(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(constants, "DEBUG", 10)
    telemetry = _telemetry(tmp_path)
    telemetry.log_plastic_coarse_fine(
        {
            "selected_layers": 4,
            "trials": [
                {
                    "trial_index": 1,
                    "layers": 4,
                    "status": "success",
                    "training_steps": 2,
                    "training_losses": [4.0, 3.5],
                    "mean_validation_loss": 3.4,
                    "validation_loss_std": 0.1,
                    "seconds_per_step": 1.2,
                    "tokens_per_second": 100.0,
                    "score": 3.4,
                    "error_class": None,
                    "error_message": None,
                }
            ],
        }
    )
    telemetry.log_event(
        "optimizer_progress",
        {
            "completed_updates": 1,
            "consumed_tokens": 100,
            "cumulative_training_seconds": 1.0,
            "training_loss": 3.3,
            "learning_rate": 1.0e-3,
            "gradient_norm": 0.5,
        },
    )

    payload, kwargs = telemetry.run.logged[-1]
    assert kwargs == {}
    assert payload["optimizer/update"] == 1
    assert payload["fine/update"] == 1
    assert payload["fine/train_loss"] == 3.3
# ^^^ THOG
