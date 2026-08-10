# vvv THOG
"""Allow checkpoint reconstruction to consume v0.54 synthetic gradient-control fields."""

from __future__ import annotations

from functools import wraps
from typing import Any

from . import plastic_depth_theil_sen_kendall_patch as _gradient
from . import training_config as _training_config


_ORIGINAL_TRAINING_CONFIG_INIT = _training_config.TrainingConfig.__init__


@wraps(_ORIGINAL_TRAINING_CONFIG_INIT)
def _training_config_init_with_gradient_resume(
    self: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    source = dict(kwargs)
    algorithm = source.pop(_gradient._ALGORITHM_KEY, None)
    tau = source.pop(_gradient._TAU_KEY, None)
    if algorithm is not None:
        _gradient._set_runtime_algorithm(str(algorithm))
    if tau is not None:
        _gradient._set_runtime_tau(tau)
    _ORIGINAL_TRAINING_CONFIG_INIT(self, *args, **source)


_training_config.TrainingConfig.__init__ = _training_config_init_with_gradient_resume


__all__ = ["_training_config_init_with_gradient_resume"]
# ^^^ THOG
