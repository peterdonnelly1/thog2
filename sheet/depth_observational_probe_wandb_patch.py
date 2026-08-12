# vvv THOG
"""Let fixed-run observational DEPTH probes use the established W&B probe charts."""

from __future__ import annotations

from typing import Any

from . import depth_weight_curves_and_observational_probes_patch as _depth
from . import plastic_depth_wandb_probe_curves_patch as _probe_wandb
from . import wandb_telemetry as _wandb


_RUNTIME_GATE_ATTRIBUTE = "_thog_runtime_legacy_coefficient_debug_gate"


# vvv THOG keep private helper semantics intact for established tests while attached runs skip all old coefficient sampling/refresh work below DEBUG>9
def _capture_coefficient_record_runtime_gated(
    trainer: Any,
    telemetry: Any,
    *,
    optimizer_update: int,
):
    if (
        bool(getattr(telemetry, _RUNTIME_GATE_ATTRIBUTE, False))
        and not _depth._legacy_coefficient_chart_enabled()
    ):
        return {}
    return _depth._ORIGINAL_CAPTURE_COEFFICIENT_RECORD(
        trainer,
        telemetry,
        optimizer_update=optimizer_update,
    )


def _should_refresh_coefficient_chart_runtime_gated(
    telemetry: Any,
    *,
    evaluation: bool,
) -> bool:
    if (
        bool(getattr(telemetry, _RUNTIME_GATE_ATTRIBUTE, False))
        and not _depth._legacy_coefficient_chart_enabled()
    ):
        return False
    return bool(
        _depth._ORIGINAL_SHOULD_REFRESH_COEFFICIENT_CHART(
            telemetry,
            evaluation=evaluation,
        )
    )


_probe_wandb._capture_coefficient_record = _capture_coefficient_record_runtime_gated
_probe_wandb._should_refresh_coefficient_chart = _should_refresh_coefficient_chart_runtime_gated
# ^^^ THOG


_ORIGINAL_ATTACH_TELEMETRY = _wandb.attach_telemetry


def _attach_telemetry_with_observational_probe_charts(trainer: Any, telemetry: Any) -> None:
    _ORIGINAL_ATTACH_TELEMETRY(trainer, telemetry)
    setattr(telemetry, _RUNTIME_GATE_ATTRIBUTE, True)
    if not _depth._observational_probe_enabled(trainer):
        return
    original_progress = trainer._print_progress

    def progress(run_id: str, event: str, **payload: Any) -> None:
        original_progress(run_id, event, **payload)
        if not trainer.distributed.is_primary:
            return
        if telemetry.run is None or telemetry.module is None:
            return
        if event not in {"optimizer_progress", "evaluation_completed"}:
            return
        records = _probe_wandb._consume_new_probe_records(trainer, telemetry)
        evaluation = event == "evaluation_completed"
        if not _probe_wandb._should_refresh_charts(
            telemetry,
            records,
            evaluation=evaluation,
        ):
            return
        try:
            step = int(
                str(
                    payload.get(
                        "completed_updates",
                        trainer.state.completed_updates,
                    )
                )
                .strip()
                .replace(",", "")
            )
            _probe_wandb._log_rolling_probe_charts(
                telemetry,
                step=step,
                include_probe_charts=True,
                include_coefficient_chart=False,
            )
        except Exception as error:
            print(
                "THOG2 WARNING: W&B observational DEPTH probe chart logging failed; "
                f"continuing without this refresh: {error}",
                flush=True,
            )

    trainer._print_progress = progress


_wandb.attach_telemetry = _attach_telemetry_with_observational_probe_charts
# ^^^ THOG


__all__ = ["_attach_telemetry_with_observational_probe_charts"]
# ^^^ THOG
