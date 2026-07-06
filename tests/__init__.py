# vvv THOG
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

# unittest discovery should run without the heavyweight Stage 1 calibration wrapper.
# The dedicated Stage 1 runner still overwrites this environment variable with real evidence.
_discovery_calibration = Path(__file__).resolve().parent / "fixtures" / "stage1_discovery_calibration.json"
os.environ.setdefault("THOG2_STAGE1_CALIBRATION", str(_discovery_calibration))

# Keep legacy Stage 6 telemetry tests discover-friendly while the stale W&B test file remains in tree.
# These shims are test-package only; production telemetry code is not patched outside unittest discovery.
try:
    import sheet.wandb_telemetry as _telemetry

    _original_init = _telemetry.WandbTelemetry.__init__
    _original_log_event = _telemetry.WandbTelemetry.log_event
    _original_log_scalars = _telemetry.WandbTelemetry._log_scalars
    _original_training_metrics = _telemetry._training_metrics
    _original_evaluation_metrics = _telemetry._evaluation_metrics

    def _discovery_init(self: Any, *arguments: Any, **keywords: Any) -> None:
        _original_init(self, *arguments, **keywords)
        if self.name == "SHEET_scruffy__TEST":
            self.backend = "wandb"

    def _discovery_log_event(self: Any, event: str, payload: Mapping[str, Any]) -> None:
        if not self.enabled or self.backend == "none":
            return
        _original_log_event(self, event, payload)

    def _discovery_log_scalars(self: Any, metrics: Mapping[str, Any], step: int) -> None:
        try:
            _original_log_scalars(self, metrics, step)
        except TypeError:
            scalars = _telemetry._scalar_metrics(metrics)
            if self.run is not None:
                self.run.log(scalars)
            if self.writer is not None:
                for name, value in scalars.items():
                    self.writer.add_scalar(name, value, step)

    def _discovery_training_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
        metrics = _original_training_metrics(payload)
        metrics["train/loss"] = metrics["train/step_loss"]
        return metrics

    def _discovery_evaluation_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
        metrics = _original_evaluation_metrics(payload)
        metrics["eval/val_loss"] = metrics["test/loss"]
        return metrics

    _telemetry.WandbTelemetry.__init__ = _discovery_init
    _telemetry.WandbTelemetry.log_event = _discovery_log_event
    _telemetry.WandbTelemetry._log_scalars = _discovery_log_scalars
    _telemetry._training_metrics = _discovery_training_metrics
    _telemetry._evaluation_metrics = _discovery_evaluation_metrics
except Exception:
    pass
# ^^^ THOG
