# vvv THOG
"""Let fixed-run observational DEPTH probes use the established W&B probe charts."""

from __future__ import annotations

from typing import Any

from . import depth_weight_curves_and_observational_probes_patch as _depth
from . import plastic_depth_wandb_probe_curves_patch as _probe_wandb
from . import wandb_telemetry as _wandb
# vvv THOG local heatmaps persist raw probe records through a concurrent compact store instead of versioned Plotly media
from .local_chart_store import ensure_local_chart_store
# ^^^ THOG


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
    observational_enabled = _depth._observational_probe_enabled(trainer)
    heatmap_enabled = _probe_wandb._delta_loss_heatmap_enabled(telemetry)
    heatmap_destination = _probe_wandb._delta_loss_heatmap_destination(telemetry)
    wandb_charts_available = telemetry.run is not None and telemetry.module is not None
    if not observational_enabled and not heatmap_enabled:
        return
    original_progress = trainer._print_progress

    def progress(run_id: str, event: str, **payload: Any) -> None:
        original_progress(run_id, event, **payload)
        if not trainer.distributed.is_primary:
            return
        if event not in {"optimizer_progress", "evaluation_completed", "run_completed"}:
            return
        records = (
            _probe_wandb._consume_new_probe_records(trainer, telemetry)
            if (
                observational_enabled
                and wandb_charts_available
                and event != "run_completed"
            )
            else ()
        )
        heatmap_records = (
            _probe_wandb._consume_new_delta_loss_heatmap_records(trainer, telemetry)
            if heatmap_enabled
            else ()
        )
        if heatmap_records and heatmap_destination == "local":
            try:
                maximum_step = (
                    telemetry.config.get(
                        "instrumentation__delta_loss_v_layer_heatmap_linear"
                    )
                    if telemetry.config.get(
                        "instrumentation__delta_loss_v_layer_heatmap"
                    ) == "linear"
                    else None
                )
                ensure_local_chart_store(telemetry).append_heatmap_records(
                    heatmap_records,
                    maximum_step=maximum_step,
                )
            except Exception as error:
                telemetry._delta_loss_heatmap_disabled = True
                print(
                    "THOG2 WARNING: local observational DEPTH heatmap logging failed; "
                    f"continuing without this chart: {error}",
                    flush=True,
                )
                return
        if heatmap_records and heatmap_destination == "wandb":
            maximum_layers = _probe_wandb._maximum_candidate_layer(
                heatmap_records,
                minimum=int(trainer.config.n_layer),
            )
            _probe_wandb._append_delta_loss_heatmap_records(
                telemetry,
                heatmap_records,
                maximum_layers=maximum_layers,
            )
        evaluation = event == "evaluation_completed"
        probe_charts_due = bool(
            observational_enabled
            and wandb_charts_available
            and _probe_wandb._plastic_wandb_charts_enabled()
            and event != "run_completed"
            and _probe_wandb._should_refresh_charts(
                telemetry,
                records,
                evaluation=evaluation,
            )
        )
        heatmap_due = bool(
            heatmap_enabled
            and heatmap_destination == "wandb"
            and telemetry.run is not None
            and telemetry.module is not None
            and _probe_wandb._should_refresh_delta_loss_heatmap(
                telemetry,
                force=event == "run_completed",
            )
        )
        if not probe_charts_due and not heatmap_due:
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
                include_probe_charts=probe_charts_due,
                include_coefficient_chart=False,
                include_delta_loss_heatmap=heatmap_due,
            )
        except Exception as error:
            if heatmap_due:
                telemetry._delta_loss_heatmap_disabled = True
            print(
                "THOG2 WARNING: W&B observational DEPTH chart logging failed; "
                f"continuing without this refresh: {error}",
                flush=True,
            )

    trainer._print_progress = progress


_wandb.attach_telemetry = _attach_telemetry_with_observational_probe_charts
# ^^^ THOG


__all__ = ["_attach_telemetry_with_observational_probe_charts"]
# ^^^ THOG
