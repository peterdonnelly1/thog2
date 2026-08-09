# vvv THOG
"""Allow lifecycle resume to reconstruct TrainingConfig from v0.53 same-batch checkpoints."""

from __future__ import annotations

from functools import wraps
from typing import Any

from . import plastic_depth_same_batch_all_probes_patch as _same_batch
from . import training_config as _training_config


_ORIGINAL_TRAINING_CONFIG_INIT = _training_config.TrainingConfig.__init__


@wraps(_ORIGINAL_TRAINING_CONFIG_INIT)
def _training_config_init_with_same_batch_resume(
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    if _same_batch._CONFIG_KEY not in kwargs:
        _ORIGINAL_TRAINING_CONFIG_INIT(self, *args, **kwargs)
        return
    normalized = _same_batch._normalize_plastic_config_with_same_batch(kwargs)
    _ORIGINAL_TRAINING_CONFIG_INIT(self, *args, **normalized)


_training_config.TrainingConfig.__init__ = _training_config_init_with_same_batch_resume


__all__ = ["_training_config_init_with_same_batch_resume"]
# ^^^ THOG
