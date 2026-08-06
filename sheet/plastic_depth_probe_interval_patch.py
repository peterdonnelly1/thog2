# vvv THOG
"""Gate PLASTIC DEPTH inline count probes by a configurable interval."""

from __future__ import annotations

import os
from typing import Any, Optional

from . import trainer_step as _trainer_step


_ENVIRONMENT_KEY = "THOG2_PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE"
_ORIGINAL_BEGIN_PLASTIC_DEPTH_INLINE_UPDATE = _trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update


def _positive_interval_from_value(value: Any, *, label: str) -> int:
    try:
        interval = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive integer; got {value!r}") from error
    if interval < 1:
        raise ValueError(f"{label} must be a positive integer; got {value!r}")
    return interval


def _plastic_depth_probe_interval(trainer: Any) -> int:
    configured = os.environ.get(_ENVIRONMENT_KEY)
    if configured is not None and configured.strip() != "":
        return _positive_interval_from_value(configured, label=_ENVIRONMENT_KEY)
    config = getattr(trainer, "config", None)
    direct_value: Optional[Any] = getattr(config, "plastic__layer_count_probe_window_size", None)
    if direct_value is not None:
        return _positive_interval_from_value(direct_value, label="plastic__layer_count_probe_window_size")
    brake_value = getattr(config, "plastic__layer_count_update_brake", 1)
    try:
        brake_interval = int(brake_value)
    except (TypeError, ValueError):
        brake_interval = 1
    return max(1, brake_interval)


def _begin_plastic_depth_inline_update_with_interval(self: Any):
    config = getattr(self, "config", None)
    if not bool(getattr(config, "plastic__enabled", False)) or not bool(getattr(config, "plastic__do_learn_layer_count", False)):
        return _ORIGINAL_BEGIN_PLASTIC_DEPTH_INLINE_UPDATE(self)
    interval = _plastic_depth_probe_interval(self)
    next_update = int(getattr(self.state, "completed_updates")) + 1
    if next_update == 1 or next_update % interval == 0:
        return _ORIGINAL_BEGIN_PLASTIC_DEPTH_INLINE_UPDATE(self)
    return None


_trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update = _begin_plastic_depth_inline_update_with_interval
# ^^^ THOG
