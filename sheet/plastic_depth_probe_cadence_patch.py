# vvv THOG
"""Throttle PLASTIC learned-count inline probes without changing ordinary PLASTIC training."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from . import trainer_step as _trainer_step


_ORIGINAL_BEGIN_PLASTIC_DEPTH_INLINE_UPDATE = _trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update


def _plastic_depth_probe_interval(trainer: Any) -> int:
    value = getattr(
        getattr(trainer, "config", None),
        "plastic__layer_count_probe_interval",
        None,
    )
    if value is None:
        value = os.environ.get("THOG2_PLASTIC_LAYER_COUNT_PROBE_INTERVAL", "1")
    try:
        interval = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(
            "THOG2_PLASTIC_LAYER_COUNT_PROBE_INTERVAL must be a positive integer; "
            f"got {value!r}"
        ) from error
    if interval < 1:
        raise ValueError(
            "THOG2_PLASTIC_LAYER_COUNT_PROBE_INTERVAL must be a positive integer; "
            f"got {value!r}"
        )
    return interval


def _begin_plastic_depth_inline_update_with_cadence(self: Any) -> Optional[Dict[str, Any]]:
    config = getattr(self, "config", None)
    if not bool(getattr(config, "plastic__enabled", False)):
        return _ORIGINAL_BEGIN_PLASTIC_DEPTH_INLINE_UPDATE(self)
    if not bool(getattr(config, "plastic__do_learn_layer_count", False)):
        return _ORIGINAL_BEGIN_PLASTIC_DEPTH_INLINE_UPDATE(self)
    next_update = int(self.state.completed_updates) + 1
    interval = _plastic_depth_probe_interval(self)
    should_probe = (
        next_update == 1
        or next_update == int(getattr(config, "max_updates", next_update))
        or next_update % interval == 0
    )
    if should_probe:
        return _ORIGINAL_BEGIN_PLASTIC_DEPTH_INLINE_UPDATE(self)
    return None


_trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update = _begin_plastic_depth_inline_update_with_cadence
# ^^^ THOG
