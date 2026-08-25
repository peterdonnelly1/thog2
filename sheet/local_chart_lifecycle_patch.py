# vvv THOG
"""Give local instrumentation an explicit lifecycle independent of chart writes."""

from __future__ import annotations

import os
from typing import Any, Optional

import constants as _constants

from . import dense_weight_curves_patch as _dense
from . import depth_weight_curves_and_observational_probes_patch as _depth
from . import wandb_telemetry as _wandb
from .local_chart_store import LocalChartStore, ensure_local_chart_store


def _capture_bounds() -> tuple[Optional[int], Optional[int]]:
    start = _depth._optional_capture_step("START_STEP")
    end = _depth._optional_capture_step("END_STEP")
    if start is not None and end is not None and end < start:
        raise ValueError(
            f"{_depth._environment_name('END_STEP')} must be greater than or equal to "
            f"{_depth._environment_name('START_STEP')}"
        )
    return start, end


def _finite_capture_count(
    *,
    start_step: Optional[int],
    end_step: Optional[int],
    cadence: int,
) -> Optional[int]:
    if end_step is None:
        return None
    end = int(end_step)
    if end < 1:
        return 0
    step = max(1, int(cadence))
    if start_step is not None and int(start_step) > 1:
        start = int(start_step)
        if end < start:
            return 0
        aligned_count = 1 + (end - start) // step
        last_aligned = start + (aligned_count - 1) * step
        return aligned_count + (0 if last_aligned == end else 1)

    multiples = end // step
    count = multiples
    if step != 1:
        count += 1
    if end != 1 and end % step != 0:
        count += 1
    return count


def _ensure_capture_retention() -> int:
    requested = _depth._history_length()
    if _depth._time_mode() != "accumulate":
        return requested
    start, end = _capture_bounds()
    required = _finite_capture_count(
        start_step=start,
        end_step=end,
        cadence=_depth._log_every_n_steps(),
    )
    if required is None or required <= requested:
        return requested

    effective = int(required)
    os.environ[_depth._environment_name("HISTORY_LENGTH")] = str(effective)
    print(
        "THOG2 WARNING: increasing effective "
        "instrumentation__depth_weight_curves__history_length "
        f"from {requested} to {effective} so the configured inclusive capture "
        "window remains authoritative.",
        flush=True,
    )
    return effective


def _validate_capture_retention() -> None:
    """Compatibility seam: validation now promotes retention instead of aborting."""

    _ensure_capture_retention()


def _weight_curves_supported(trainer: Any) -> bool:
    if int(_constants.DEBUG) <= 2:
        return False
    raw_model = trainer.raw_model
    return bool(
        _dense._dense_model(raw_model)
        or _depth._depth_trajectory_from_model(raw_model) is not None
    )


def _heatmap_local(telemetry: Any) -> bool:
    config = telemetry.config
    mode = config.get("instrumentation__delta_loss_v_layer_heatmap")
    enabled = mode not in {None, False, "", "false", "none", "off"}
    destination = str(
        config.get("instrumentation__delta_loss_v_layer_heatmap__destination", "local")
    ).strip().lower()
    return enabled and destination == "local"


def _weight_phase(
    optimizer_update: int,
    *,
    start_step: Optional[int],
    end_step: Optional[int],
) -> str:
    update = max(0, int(optimizer_update))
    first_step = max(1, int(start_step)) if start_step is not None else 1
    if end_step is not None and int(end_step) < first_step:
        return "monitoring"
    if update < first_step:
        return "preparing"
    if end_step is not None and update > int(end_step):
        return "monitoring"
    return "recording"


def _active_phase(
    store: LocalChartStore,
    optimizer_update: int,
    *,
    weight_local: bool,
    heatmap_local: bool,
    start_step: Optional[int],
    end_step: Optional[int],
) -> str:
    if heatmap_local and store._has_heatmap_records:
        return "recording"
    if weight_local:
        return _weight_phase(
            optimizer_update,
            start_step=start_step,
            end_step=end_step,
        )
    return "preparing"


def _heartbeat(
    telemetry: Any,
    optimizer_update: int,
    *,
    weight_local: bool,
    heatmap_local: bool,
    start_step: Optional[int],
    end_step: Optional[int],
    force: bool = False,
) -> None:
    store = telemetry._thog_local_chart_store
    state = _active_phase(
        store,
        optimizer_update,
        weight_local=weight_local,
        heatmap_local=heatmap_local,
        start_step=start_step,
        end_step=end_step,
    )
    try:
        store.heartbeat(optimizer_update, run_state=state, force=force)
    except Exception as error:
        if not bool(getattr(telemetry, "_thog_local_chart_heartbeat_warning", False)):
            print(
                "THOG2 WARNING: local instrumentation heartbeat failed; "
                f"continuing training: {error}",
                flush=True,
            )
            telemetry._thog_local_chart_heartbeat_warning = True


_ORIGINAL_ATTACH_TELEMETRY = _wandb.attach_telemetry


def _attach_telemetry_with_local_lifecycle(trainer: Any, telemetry: Any) -> None:
    weight_supported = _weight_curves_supported(trainer)
    weight_destination = _depth._destination() if weight_supported else "none"
    weight_enabled = weight_destination != "none"
    effective_history_length = _depth._history_length()
    if weight_enabled:
        effective_history_length = _ensure_capture_retention()
    _ORIGINAL_ATTACH_TELEMETRY(trainer, telemetry)
    weight_local = weight_destination == "local"
    heatmap_local = _heatmap_local(telemetry)
    if not weight_local and not heatmap_local:
        return

    store = ensure_local_chart_store(telemetry)
    start_step, end_step = _capture_bounds()
    if weight_local:
        store.configure_weight_capture(
            start_step=start_step,
            end_step=end_step,
            cadence=_depth._log_every_n_steps(),
            history_length=effective_history_length,
        )

    initial_update = int(trainer.state.completed_updates)
    _heartbeat(
        telemetry,
        initial_update,
        weight_local=weight_local,
        heatmap_local=heatmap_local,
        start_step=start_step,
        end_step=end_step,
        force=True,
    )

    original_timed = trainer._timed
    train_one_update = trainer.train_one_update

    def timed(function: Any):
        metrics, elapsed = original_timed(function)
        if (
            function == train_one_update
            and trainer.distributed.is_primary
            and not bool(float(metrics.get("skipped_update", 0.0)))
        ):
            _heartbeat(
                telemetry,
                int(trainer.state.completed_updates),
                weight_local=weight_local,
                heatmap_local=heatmap_local,
                start_step=start_step,
                end_step=end_step,
            )
        return metrics, elapsed

    trainer._timed = timed


_wandb.attach_telemetry = _attach_telemetry_with_local_lifecycle


__all__ = [
    "_active_phase",
    "_attach_telemetry_with_local_lifecycle",
    "_ensure_capture_retention",
    "_finite_capture_count",
    "_validate_capture_retention",
    "_weight_phase",
]
# ^^^ THOG
