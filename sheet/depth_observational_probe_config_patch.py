# vvv THOG
"""Make read-only DEPTH probe controls effective and resumable without PLASTIC."""

from __future__ import annotations

from typing import Any, Dict

from . import run_config as _run_config
from . import training_config as _training_config


_OBSERVATIONAL_PROBE_FIELDS = (
    "plastic__layer_count_probe__probe_every_n_steps",
    "plastic__layer_count_probe__number_of_sampled_valid_tokens",
    "plastic__layer_count_probe_radius",
)


def _observational_probe_requested(config: Any) -> bool:
    return (
        not bool(getattr(config, "plastic__do_learn_layer_count", False))
        and getattr(config, "plastic__layer_count_probe__probe_every_n_steps", None) is not None
    )


def _validate_observational_probe_token_capacity(config: Any) -> None:
    if not _observational_probe_requested(config):
        return
    requested = int(
        getattr(
            config,
            "plastic__layer_count_probe__number_of_sampled_valid_tokens",
            0,
        )
    )
    if requested == 0:
        return
    capacity = int(config.batch_size) * int(config.block_size)
    if requested > capacity:
        raise ValueError(
            "plastic__layer_count_probe__number_of_sampled_valid_tokens must not exceed "
            f"batch_size * block_size ({capacity}) when observational DEPTH probing is enabled"
        )


_ORIGINAL_RUN_POST_INIT = _run_config.OwtRunConfig.__post_init__
_ORIGINAL_TRAINING_POST_INIT = _training_config.TrainingConfig.__post_init__
_ORIGINAL_RUN_PERSISTENT_DICT = _run_config.OwtRunConfig.persistent_dict
_ORIGINAL_TRAINING_PERSISTENT_DICT = _training_config.TrainingConfig.persistent_dict


def _run_post_init_with_observational_probe_controls(self: Any) -> None:
    _ORIGINAL_RUN_POST_INIT(self)
    _validate_observational_probe_token_capacity(self)


def _training_post_init_with_observational_probe_controls(self: Any) -> None:
    _ORIGINAL_TRAINING_POST_INIT(self)
    _validate_observational_probe_token_capacity(self)


def _persistent_with_observational_probe_controls(
    original: Any,
    config: Any,
) -> Dict[str, Any]:
    values = original(config)
    if bool(getattr(config, "plastic__enabled", False)):
        return values
    if not _observational_probe_requested(config):
        return values
    for name in _OBSERVATIONAL_PROBE_FIELDS:
        values[name] = getattr(config, name)
    return values


def _run_persistent_dict_with_observational_probe_controls(self: Any) -> Dict[str, Any]:
    return _persistent_with_observational_probe_controls(
        _ORIGINAL_RUN_PERSISTENT_DICT,
        self,
    )


def _training_persistent_dict_with_observational_probe_controls(self: Any) -> Dict[str, Any]:
    return _persistent_with_observational_probe_controls(
        _ORIGINAL_TRAINING_PERSISTENT_DICT,
        self,
    )


_run_config.OwtRunConfig.__post_init__ = _run_post_init_with_observational_probe_controls
_training_config.TrainingConfig.__post_init__ = _training_post_init_with_observational_probe_controls
_run_config.OwtRunConfig.persistent_dict = _run_persistent_dict_with_observational_probe_controls
_training_config.TrainingConfig.persistent_dict = _training_persistent_dict_with_observational_probe_controls
# ^^^ THOG


__all__ = [
    "_OBSERVATIONAL_PROBE_FIELDS",
    "_observational_probe_requested",
    "_validate_observational_probe_token_capacity",
]
# ^^^ THOG
