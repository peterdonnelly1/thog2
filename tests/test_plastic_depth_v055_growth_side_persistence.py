from __future__ import annotations

from types import SimpleNamespace

import pytest

from sheet import plastic_depth_sen_kendall_v055_patch as v055
from sheet import plastic_depth_theil_sen_kendall_patch as tsk
from sheet import plastic_depth_v055_growth_side_discount_patch as growth


def test_sen_kendall_persistent_dict_contains_growth_side_discount(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v055.STRATIFIED_ALGORITHM)
    monkeypatch.setenv(growth._RUNTIME_ENV, "0.65")
    config = SimpleNamespace(plastic__enabled=True)

    values = growth._persistent_with_growth_side_discount(lambda _config: {"existing": 1}, config)

    assert values["existing"] == 1
    assert values[growth._CONFIG_KEY] == pytest.approx(0.65)


def test_legacy_persistent_dict_does_not_add_growth_side_discount(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, tsk.LEGACY_DIRECTIONAL_ALGORITHM)
    monkeypatch.setenv(growth._RUNTIME_ENV, "0.65")
    config = SimpleNamespace(plastic__enabled=True)

    values = growth._persistent_with_growth_side_discount(lambda _config: {"existing": 1}, config)

    assert growth._CONFIG_KEY not in values


def test_checkpoint_value_conflicting_with_explicit_cli_value_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv(growth._RUNTIME_ENV, "0.7")
    monkeypatch.setenv(growth._EXPLICIT_ENV, "0.7")

    with pytest.raises(ValueError, match="resume material parameter mismatch"):
        growth._normalize_plastic_config_with_growth_side_discount(
            {growth._CONFIG_KEY: 0.5}
        )
