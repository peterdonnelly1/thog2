from __future__ import annotations

from pathlib import Path

import constants

from sheet.wandb_telemetry import WandbTelemetry


class _Run:
    def __init__(self) -> None:
        self.defined = []
        self.logged = []

    def define_metric(self, name, **kwargs) -> None:
        self.defined.append((name, dict(kwargs)))

    def log(self, payload, **kwargs) -> None:
        self.logged.append((dict(payload), dict(kwargs)))


class _Writer:
    def __init__(self) -> None:
        self.scalars = []

    def add_scalar(self, name, value, step) -> None:
        self.scalars.append((name, value, step))


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


def test_coarse_trials_have_independent_local_step_axes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(constants, "DEBUG", 10)
    telemetry = _telemetry(tmp_path)
    provenance = {
        "selected_layers": 8,
        "trials": [
            {
                "trial_index": 1,
                "layers": 4,
                "training_steps": 2,
                "training_losses": [4.0, 3.5],
                "mean_validation_loss": 3.4,
                "validation_loss_std": 0.1,
                "seconds_per_step": 1.2,
                "tokens_per_second": 100.0,
                "score": 3.4,
            },
            {
                "trial_index": 2,
                "layers": 8,
                "training_steps": 2,
                "training_losses": [3.8, 3.2],
                "mean_validation_loss": 3.1,
                "validation_loss_std": 0.1,
                "seconds_per_step": 1.8,
                "tokens_per_second": 80.0,
                "score": 3.1,
            },
        ],
    }

    telemetry.log_plastic_coarse_fine(provenance)

    defined_names = {name for name, _ in telemetry.run.defined}
    assert "coarse/trial_1/step" in defined_names
    assert "coarse/trial_2/step" in defined_names
    logged_payloads = [payload for payload, _ in telemetry.run.logged]
    assert any(
        payload.get("coarse/trial_1/step") == 1
        and payload.get("coarse/trial_1/training_loss") == 4.0
        for payload in logged_payloads
    )
    assert any(
        payload.get("coarse/trial_2/step") == 1
        and payload.get("coarse/trial_2/training_loss") == 3.8
        for payload in logged_payloads
    )
    assert any(
        name == "coarse/trial_1/training_loss" and step == 1
        for name, _, step in telemetry.writer.scalars
    )
    assert any(
        name == "coarse/trial_2/training_loss" and step == 1
        for name, _, step in telemetry.writer.scalars
    )


def test_fine_metrics_are_mirrored_under_fine_update_axis(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(constants, "DEBUG", 10)
    telemetry = _telemetry(tmp_path)

    telemetry.log_event(
        "optimizer_progress",
        {
            "completed_updates": 7,
            "consumed_tokens": 700,
            "cumulative_training_seconds": 12.0,
            "training_loss": 3.0,
            "learning_rate": 1.0e-3,
            "gradient_norm": 0.5,
        },
    )

    payload, kwargs = telemetry.run.logged[-1]
    assert kwargs == {"step": 7}
    assert payload["optimizer/update"] == 7
    assert payload["fine/update"] == 7
    assert payload["fine/train_loss"] == 3.0
    assert payload["fine/tokens_seen"] == 700
    assert any(
        name == "fine/train_loss" and step == 7
        for name, _, step in telemetry.writer.scalars
    )


# vvv THOG normal W&B owns only the two loss scalars while TensorBoard retains complete diagnostics
def test_normal_wandb_logs_only_train_and_validation_loss(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(constants, "DEBUG", 9)
    telemetry = _telemetry(tmp_path)

    telemetry.log_event(
        "optimizer_progress",
        {
            "completed_updates": 7,
            "consumed_tokens": 700,
            "cumulative_training_seconds": 12.0,
            "training_loss": 3.0,
            "learning_rate": 1.0e-3,
            "gradient_norm": 0.5,
        },
    )
    telemetry.log_event(
        "evaluation_completed",
        {
            "completed_updates": 7,
            "consumed_tokens": 700,
            "validation_loss": 2.7,
            "training_loss": 2.8,
        },
    )

    assert telemetry.run.logged == [
        ({"train/loss": 3.0}, {"step": 7}),
        ({"val/val_loss": 2.7}, {"step": 7}),
    ]
    writer_names = {name for name, _, _ in telemetry.writer.scalars}
    assert "optim/lr" in writer_names
    assert "val/train_loss" in writer_names
    assert not any(name.startswith("fine/") for name in telemetry.run.logged[0][0])
# ^^^ THOG
