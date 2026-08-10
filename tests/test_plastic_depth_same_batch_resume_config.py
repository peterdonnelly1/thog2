# vvv THOG
from __future__ import annotations

import os

import pytest

from sheet import plastic_depth_same_batch_all_probes_patch as same_batch
from sheet.training_config import TrainingConfig
from tests.test_plastic_depth import plastic_training_config


@pytest.fixture(autouse=True)
def _isolated_same_batch_environment(monkeypatch):
    monkeypatch.delenv(same_batch._RUNTIME_ENV, raising=False)
    monkeypatch.delenv(same_batch._EXPLICIT_ENV, raising=False)
    try:
        yield
    finally:
        os.environ.pop(same_batch._RUNTIME_ENV, None)
        os.environ.pop(same_batch._EXPLICIT_ENV, None)


def _persisted_same_batch_training_config() -> dict:
    same_batch._set_runtime_enabled(True)
    config = plastic_training_config(
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=3,
        plastic__max_permitted_layers=5,
    )
    values = config.persistent_dict()
    assert values[same_batch._CONFIG_KEY] is True
    return values


def test_training_config_constructor_restores_persisted_same_batch_mode() -> None:
    values = _persisted_same_batch_training_config()
    same_batch._set_runtime_enabled(False)

    restored = TrainingConfig(**dict(values))

    assert restored.plastic__layer_count__same_batch_all_probes is True
    assert same_batch._runtime_enabled() is True


def test_training_config_constructor_rejects_explicit_same_batch_resume_mismatch() -> None:
    values = _persisted_same_batch_training_config()
    same_batch._set_runtime_enabled(False, explicit=True)

    with pytest.raises(
        ValueError,
        match=r"resume material parameter mismatch: plastic__layer_count__same_batch_all_probes",
    ):
        TrainingConfig(**dict(values))
# ^^^ THOG
